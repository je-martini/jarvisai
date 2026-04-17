/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState, useRef, useEffect } from 'react';
import { GoogleGenAI, Modality, type LiveServerMessage } from "@google/genai";
import { Mic, MicOff, Volume2, VolumeX, MessageCircle, Power } from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';

// --- Constants & Types ---
const SAMPLING_RATE = 16000;
const MODEL_NAME = "gemini-3.1-flash-live-preview";

export default function App() {
  // --- Refs ---
  const aiRef = useRef<any>(null);
  const sessionRef = useRef<any>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const audioStreamRef = useRef<MediaStream | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const audioOutputQueueRef = useRef<Int16Array[]>([]);
  const isPlayingRef = useRef(false);

  // --- State ---
  const [isConnected, setIsConnected] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);
  const [isMuted, setIsMuted] = useState(false);
  const [transcript, setTranscript] = useState<string>("");
  const [userSpeech, setUserSpeech] = useState<string>("");
  const [audioLevel, setAudioLevel] = useState(0);

  // --- Initialization ---
  useEffect(() => {
    // Initialize GoogleGenAI with the API key from environment
    aiRef.current = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
    
    return () => {
      stopService();
    };
  }, []);

  // --- Audio Utilities ---
  const floatTo16BitPCM = (float32Array: Float32Array) => {
    const buffer = new Int16Array(float32Array.length);
    for (let i = 0; i < float32Array.length; i++) {
      const s = Math.max(-1, Math.min(1, float32Array[i]));
      buffer[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
    }
    return buffer.buffer;
  };

  const base64EncodeArrayBuffer = (buffer: ArrayBuffer) => {
    let binary = '';
    const bytes = new Uint8Array(buffer);
    const len = bytes.byteLength;
    for (let i = 0; i < len; i++) {
      binary += String.fromCharCode(bytes[i]);
    }
    return btoa(binary);
  };

  const decodeBase64ToArrayBuffer = (base64: string) => {
    const binaryString = atob(base64);
    const len = binaryString.length;
    const bytes = new Uint8Array(len);
    for (let i = 0; i < len; i++) {
        bytes[i] = binaryString.charCodeAt(i);
    }
    return bytes.buffer;
  };

  // --- Playback Logic ---
  const playNextChunk = async () => {
    if (!audioContextRef.current || audioOutputQueueRef.current.length === 0) {
      isPlayingRef.current = false;
      return;
    }

    isPlayingRef.current = true;
    const pcmData = audioOutputQueueRef.current.shift()!;
    
    // Convert Int16 PCM to Float32 for Web Audio
    const float32Data = new Float32Array(pcmData.length);
    for (let i = 0; i < pcmData.length; i++) {
      float32Data[i] = pcmData[i] / 32768.0;
    }

    const audioBuffer = audioContextRef.current.createBuffer(1, float32Data.length, 24000); // Live API output is 24kHz
    audioBuffer.getChannelData(0).set(float32Data);

    const source = audioContextRef.current.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(audioContextRef.current.destination);
    
    source.onended = () => {
      playNextChunk();
    };
    source.start();
  };

  // --- Core Lifecycle ---
  const startService = async () => {
    if (isConnecting || isConnected) return;
    
    setIsConnecting(true);
    setTranscript("");
    setUserSpeech("");

    try {
      // Initialize Audio Context on user gesture
      if (!audioContextRef.current) {
        audioContextRef.current = new AudioContext({ sampleRate: 16000 });
      } else if (audioContextRef.current.state === 'suspended') {
        await audioContextRef.current.resume();
      }

      // 1. Connect to Live API
      const sessionPromise = aiRef.current.live.connect({
        model: MODEL_NAME,
        config: {
          responseModalities: [Modality.AUDIO],
          speechConfig: {
            voiceConfig: { prebuiltVoiceConfig: { voiceName: "Zephyr" } },
          },
          systemInstruction: "Eres un asistente de voz amigable y eficiente. Responde de manera concisa y natural. Habla siempre en español a menos que se te pida lo contrario. Intenta ser empático y útil.",
          inputAudioTranscription: {},
          outputAudioTranscription: {},
        },
        callbacks: {
          onopen: () => {
            console.log("Conexión abierta");
            setIsConnected(true);
            setIsConnecting(false);
            setupMicrophone(sessionPromise);
          },
          onmessage: async (message: LiveServerMessage) => {
            // Handle transcriptions
            if (message.serverContent?.modelTurn?.parts) {
                const textPart = message.serverContent.modelTurn.parts.find(p => p.text);
                if (textPart?.text) {
                    setTranscript(prev => prev + " " + textPart.text);
                }
            }

            // Handle Audio output
            const audioPart = message.serverContent?.modelTurn?.parts?.find(p => p.inlineData);
            if (audioPart?.inlineData?.data) {
              const audioBuffer = decodeBase64ToArrayBuffer(audioPart.inlineData.data);
              audioOutputQueueRef.current.push(new Int16Array(audioBuffer));
              if (!isPlayingRef.current) {
                playNextChunk();
              }
            }

            // Handle transcription messages (both user and model)
            if (message.serverContent?.interrupted) {
              // Stop playback on interruption
              audioOutputQueueRef.current = [];
              isPlayingRef.current = false;
            }
          },
          onclose: (event: any) => {
            console.log("Conexión cerrada", event);
            stopService();
          },
          onerror: (error: any) => {
            console.error("Error en Live API:", error);
            stopService();
          }
        }
      });

      sessionRef.current = await sessionPromise;
    } catch (error) {
      console.error("Error al iniciar el servicio:", error);
      setIsConnecting(false);
    }
  };

  const setupMicrophone = async (sessionPromise: Promise<any>) => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      audioStreamRef.current = stream;

      const source = audioContextRef.current!.createMediaStreamSource(stream);
      const processor = audioContextRef.current!.createScriptProcessor(4096, 1, 1);
      processorRef.current = processor;

      const analyser = audioContextRef.current!.createAnalyser();
      analyser.fftSize = 256;
      const dataArray = new Uint8Array(analyser.frequencyBinCount);

      source.connect(analyser);
      analyser.connect(processor);
      processor.connect(audioContextRef.current!.destination);

      processor.onaudioprocess = (e) => {
        if (isMuted) return;

        const inputData = e.inputBuffer.getChannelData(0);
        
        // Calculate audio level for visualization
        analyser.getByteFrequencyData(dataArray);
        const sum = dataArray.reduce((acc, v) => acc + v, 0);
        setAudioLevel(sum / dataArray.length / 255);

        // Send to model
        const pcmBuffer = floatTo16BitPCM(inputData);
        const base64Data = base64EncodeArrayBuffer(pcmBuffer);

        sessionPromise.then(session => {
          session.sendRealtimeInput({
            audio: { data: base64Data, mimeType: 'audio/pcm;rate=16000' }
          });
        });
      };
    } catch (error) {
      console.error("Error al acceder al micrófono:", error);
    }
  };

  const stopService = () => {
    setIsConnected(false);
    setIsConnecting(false);
    
    if (sessionRef.current) {
      sessionRef.current.close();
      sessionRef.current = null;
    }

    if (processorRef.current) {
      processorRef.current.disconnect();
      processorRef.current = null;
    }

    if (audioStreamRef.current) {
      audioStreamRef.current.getTracks().forEach(track => track.stop());
      audioStreamRef.current = null;
    }

    audioOutputQueueRef.current = [];
    isPlayingRef.current = false;
    setAudioLevel(0);
  };

  const toggleMute = () => {
    setIsMuted(!isMuted);
  };

  return (
    <div className="min-h-screen bg-[#0a0502] text-white flex flex-col font-sans selection:bg-[#ff4e00]/30 selection:text-white">
      {/* Background Atmosphere (Recipe 7) */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none z-0">
        <motion.div 
          className="absolute inset-0 opacity-40 blur-[100px]"
          animate={{
            background: [
              "radial-gradient(circle at 50% 30%, #3a1510 0%, transparent 60%), radial-gradient(circle at 10% 80%, #ff4e00 0%, transparent 50%)",
              "radial-gradient(circle at 30% 50%, #1a1530 0%, transparent 60%), radial-gradient(circle at 80% 20%, #4e00ff 0%, transparent 50%)",
              "radial-gradient(circle at 50% 30%, #3a1510 0%, transparent 60%), radial-gradient(circle at 10% 80%, #ff4e00 0%, transparent 50%)",
            ]
          }}
          transition={{ duration: 15, repeat: Infinity, ease: "linear" }}
        />
      </div>

      {/* Header */}
      <header className="relative z-10 px-8 py-6 flex justify-between items-center bg-black/20 backdrop-blur-md">
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-[#ff4e00] animate-pulse" />
          <h1 className="text-xs font-mono uppercase tracking-[0.2em] text-[#ff4e00]">Gemini Live</h1>
        </div>
        <div className="flex gap-4">
          <button 
            onClick={toggleMute}
            disabled={!isConnected}
            className={`p-2 rounded-full transition-all duration-300 ${isMuted ? 'bg-red-500/20 text-red-400' : 'bg-white/5 text-white/60 hover:text-white'}`}
          >
            {isMuted ? <MicOff size={18} /> : <Mic size={18} />}
          </button>
        </div>
      </header>

      {/* Main Experience */}
      <main className="relative z-10 flex-1 flex flex-col items-center justify-center p-8">
        <AnimatePresence mode="wait">
          {!isConnected ? (
            <motion.div 
              key="landing"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.9 }}
              className="text-center max-w-lg"
            >
              <h2 className="text-5xl md:text-7xl font-light tracking-tighter mb-8 leading-tight">
                Habla con tu <br />
                <span className="italic font-serif">Asistente Live</span>
              </h2>
              <p className="text-white/40 text-sm mb-12 font-mono uppercase tracking-widest">
                Experiencia de voz en tiempo real con Gemini
              </p>
              
              <button
                onClick={startService}
                disabled={isConnecting}
                className="group relative px-12 py-5 bg-white text-black font-medium text-sm tracking-widest overflow-hidden transition-all hover:scale-105 active:scale-95 disabled:opacity-50"
              >
                <div className="relative z-10 flex items-center gap-3">
                  {isConnecting ? (
                    <div className="w-5 h-5 border-2 border-black/30 border-t-black rounded-full animate-spin" />
                  ) : (
                    <Power size={18} />
                  )}
                  <span>{isConnecting ? "CONECTANDO..." : "INICIAR CONVERSACIÓN"}</span>
                </div>
                <div className="absolute inset-0 bg-[#ff4e00] translate-y-full group-hover:translate-y-0 transition-transform duration-300 ease-out z-0" />
              </button>
            </motion.div>
          ) : (
            <motion.div 
              key="active"
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.8 }}
              className="flex flex-col items-center"
            >
              {/* Interaction Orb */}
              <div className="relative w-64 h-64 md:w-96 md:h-96 flex items-center justify-center mb-16">
                {/* Background Ring */}
                <motion.div 
                  className="absolute inset-0 rounded-full border border-white/10"
                  animate={{ scale: [1, 1.1, 1] }}
                  transition={{ duration: 4, repeat: Infinity }}
                />
                
                {/* Interaction Glow */}
                <motion.div 
                  className="absolute inset-4 rounded-full bg-gradient-to-br from-[#ff4e00] to-[#ff9e00] blur-xl opacity-20"
                  animate={{ 
                    scale: 1 + audioLevel * 1.5,
                    opacity: 0.1 + audioLevel * 0.4
                  }}
                />

                {/* Main Orb */}
                <motion.div 
                  className="w-48 h-48 md:w-64 md:h-64 rounded-full bg-black border border-white/20 shadow-[0_0_50px_rgba(255,78,0,0.1)] flex items-center justify-center overflow-hidden"
                  animate={{ 
                    scale: 0.95 + audioLevel * 0.2
                  }}
                >
                   {/* Waveform Visualization (simplified) */}
                   <div className="flex gap-1 h-32 items-center">
                    {[...Array(12)].map((_, i) => (
                      <motion.div 
                        key={i}
                        className="w-1 md:w-1.5 bg-gradient-to-t from-[#ff4e00] to-[#ff9e00] rounded-full"
                        animate={{ 
                          height: 10 + Math.random() * (audioLevel * 100 + 40)
                        }}
                        transition={{ duration: 0.1, repeat: Infinity }}
                      />
                    ))}
                  </div>
                </motion.div>

                {/* Status Indicator */}
                <div className="absolute -bottom-8 left-1/2 -translate-x-1/2 px-4 py-1 rounded-full bg-white/5 border border-white/10 backdrop-blur-sm">
                  <span className="text-[10px] font-mono uppercase tracking-[0.2em] text-white/60">
                    {isMuted ? "SILENCIADO" : "ESCUCHANDO"}
                  </span>
                </div>
              </div>

              {/* Transcript Display */}
              <div className="max-w-2xl w-full text-center space-y-6">
                <AnimatePresence>
                  {transcript && (
                    <motion.div 
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      className="px-6 py-4 rounded-2xl bg-white/5 backdrop-blur-sm border border-white/10"
                    >
                      <p className="text-white/80 text-lg leading-relaxed font-serif italic">
                        "{transcript.split('.').pop() || transcript}"
                      </p>
                    </motion.div>
                  )}
                  
                  {!transcript && (
                    <motion.p 
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      className="text-white/40 font-mono text-xs uppercase tracking-widest"
                    >
                      Esperando respuesta...
                    </motion.p>
                  )}
                </AnimatePresence>
              </div>

              {/* End Button */}
              <button 
                onClick={stopService}
                className="mt-20 p-4 rounded-full bg-red-500/20 text-red-500 hover:bg-red-500 hover:text-white transition-all duration-300"
              >
                <Power size={24} />
              </button>
            </motion.div>
          )}
        </AnimatePresence>
      </main>

      {/* Footer Meta */}
      <footer className="relative z-10 p-8 flex justify-between items-end border-t border-white/5">
        <div className="space-y-1">
          <p className="text-[10px] font-mono text-white/30 uppercase tracking-[0.2em]">Tecnología</p>
          <p className="text-xs text-white/60 font-medium tracking-tight">Multimodal Live API</p>
        </div>
        <div className="text-right space-y-1">
          <p className="text-[10px] font-mono text-white/30 uppercase tracking-[0.2em]">Status</p>
          <p className="text-xs text-white/60 font-medium tracking-tight">{isConnected ? "ONLINE" : "STANDBY"}</p>
        </div>
      </footer>
    </div>
  );
}
