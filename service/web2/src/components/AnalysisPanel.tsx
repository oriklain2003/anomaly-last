import React, { useState, useRef, useEffect, useCallback } from 'react';
import { X, Bot, Send, User, Trash2, FileText, Loader2, MonitorUp, MonitorOff, Eye, Camera, Clock, ScanEye } from 'lucide-react';
import type { AnomalyReport, TrackPoint } from '../types';
import { analyzeWithAI } from '../api';
import { stripDataUrlPrefix } from '../utils/screenshot';
import { parseActionsFromResponse, processActions, stripActionsFromText, type AIAction } from '../utils/aiActions';
import type { ProcessedActions } from '../utils/aiActions';
import clsx from 'clsx';
import { TypewriterMarkdown } from '../utils/markdown';

// Import the original ReportPanel content component
import { ReportPanel } from './ReportPanel';

// ============================================================
// Types
// ============================================================

interface ChatMessage {
    role: 'user' | 'assistant';
    content: string;
    screenshot?: string; // base64 thumbnail
    actions?: AIAction[];
}

interface AnalysisPanelProps {
    anomaly: AnomalyReport | null;
    flightPoints: TrackPoint[];
    onClose: () => void;
    onAIActions: (actions: ProcessedActions) => void;
    onFlyTo?: (lat: number, lon: number, zoom?: number) => void;
    className?: string;
    mode?: 'historical' | 'realtime' | 'research' | 'rules' | 'feedback' | 'ai-results';
}

// TypewriterText removed - now using TypewriterMarkdown from utils/markdown

// ============================================================
// Main Component
// ============================================================

export const AnalysisPanel: React.FC<AnalysisPanelProps> = ({ 
    anomaly, 
    flightPoints,
    onClose,
    onAIActions,
    onFlyTo,
    className,
    mode = 'historical' 
}) => {
    const [activeTab, setActiveTab] = useState<'report' | 'ai'>('report');
    const [messages, setMessages] = useState<ChatMessage[]>([
        { role: 'assistant', content: "👁️ **Visual Analyst**\n\nI analyze what you see on screen for this flight. Share your screen to enable visual analysis, then ask me questions about what you're looking at." }
    ]);
    
    // Countdown state for next capture
    const [captureCountdown, setCaptureCountdown] = useState(5);
    const countdownIntervalRef = useRef<NodeJS.Timeout | null>(null);
    const [input, setInput] = useState('');
    const [loading, setLoading] = useState(false);
    const messagesEndRef = useRef<HTMLDivElement>(null);
    
    // Screen sharing state
    const [isScreenSharing, setIsScreenSharing] = useState(false);
    const [latestScreenshot, setLatestScreenshot] = useState<string | null>(null);
    const [lastCaptureTime, setLastCaptureTime] = useState<Date | null>(null);
    const screenStreamRef = useRef<MediaStream | null>(null);
    const captureIntervalRef = useRef<NodeJS.Timeout | null>(null);
    const videoRef = useRef<HTMLVideoElement | null>(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    // Reset messages when anomaly changes
    useEffect(() => {
        setMessages([
            { role: 'assistant', content: "👁️ **Visual Analyst**\n\nI analyze what you see on screen for this flight. Share your screen to enable visual analysis, then ask me questions about what you're looking at." }
        ]);
    }, [anomaly?.flight_id]);

    // Cleanup on unmount
    useEffect(() => {
        return () => {
            stopScreenSharing();
            if (countdownIntervalRef.current) {
                clearInterval(countdownIntervalRef.current);
            }
        };
    }, []);

    // ============================================================
    // Screen Sharing with Continuous Capture
    // ============================================================

    const captureFrame = useCallback(() => {
        if (!videoRef.current || !screenStreamRef.current) return;
        
        const video = videoRef.current;
        if (video.readyState < 2) return; // Video not ready
        
        try {
            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            const ctx = canvas.getContext('2d');
            
            if (ctx) {
                ctx.drawImage(video, 0, 0);
                const screenshot = canvas.toDataURL('image/png');
                setLatestScreenshot(screenshot);
                setLastCaptureTime(new Date());
            }
        } catch (e) {
            console.error('Failed to capture frame:', e);
        }
    }, []);

    const startScreenSharing = async () => {
        try {
            // Request screen capture permission
            const stream = await navigator.mediaDevices.getDisplayMedia({
                video: {
                    displaySurface: 'browser'
                } as MediaTrackConstraints,
                audio: false
            });
            
            screenStreamRef.current = stream;
            
            // Create video element to capture from
            const video = document.createElement('video');
            video.srcObject = stream;
            video.muted = true;
            await video.play();
            videoRef.current = video;
            
            // Wait for video to be ready
            await new Promise(resolve => setTimeout(resolve, 500));
            
            // Capture initial frame
            captureFrame();
            setCaptureCountdown(5);
            
            // Start countdown interval (every second)
            countdownIntervalRef.current = setInterval(() => {
                setCaptureCountdown(prev => {
                    if (prev <= 1) {
                        captureFrame();
                        return 5;
                    }
                    return prev - 1;
                });
            }, 1000);
            
            setIsScreenSharing(true);
            
            // Handle stream end (user stops sharing)
            stream.getVideoTracks()[0].addEventListener('ended', () => {
                stopScreenSharing();
            });
            
            setMessages(prev => [...prev, {
                role: 'assistant',
                content: "✅ Screen sharing active! I'm capturing every 5 seconds. Ask me about anything you see on the map."
            }]);
            
        } catch (e) {
            console.error('Screen sharing failed:', e);
            setMessages(prev => [...prev, {
                role: 'assistant',
                content: "Screen sharing was cancelled. Click 'Share Screen' when you're ready."
            }]);
        }
    };

    const stopScreenSharing = () => {
        // Stop the capture interval (now handled by countdown)
        if (captureIntervalRef.current) {
            clearInterval(captureIntervalRef.current);
            captureIntervalRef.current = null;
        }
        
        // Stop countdown interval
        if (countdownIntervalRef.current) {
            clearInterval(countdownIntervalRef.current);
            countdownIntervalRef.current = null;
        }
        
        // Stop the stream
        if (screenStreamRef.current) {
            screenStreamRef.current.getTracks().forEach(track => track.stop());
            screenStreamRef.current = null;
        }
        
        // Clean up video
        if (videoRef.current) {
            videoRef.current.pause();
            videoRef.current.srcObject = null;
            videoRef.current = null;
        }
        
        setIsScreenSharing(false);
        setLatestScreenshot(null);
        setLastCaptureTime(null);
        setCaptureCountdown(5);
    };
    
    // Manual capture trigger
    const captureNow = useCallback(() => {
        if (isScreenSharing) {
            captureFrame();
            setCaptureCountdown(5);
            // Reset the countdown interval
            if (countdownIntervalRef.current) {
                clearInterval(countdownIntervalRef.current);
            }
            countdownIntervalRef.current = setInterval(() => {
                setCaptureCountdown(prev => {
                    if (prev <= 1) {
                        captureFrame();
                        return 5;
                    }
                    return prev - 1;
                });
            }, 1000);
        }
    }, [isScreenSharing, captureFrame]);

    const toggleScreenSharing = () => {
        if (isScreenSharing) {
            stopScreenSharing();
            setMessages(prev => [...prev, {
                role: 'assistant',
                content: "Screen sharing stopped. Click 'Start Sharing' to resume."
            }]);
        } else {
            startScreenSharing();
        }
    };

    // ============================================================
    // Send Message
    // ============================================================

    const handleSend = async (e?: React.FormEvent) => {
        if (e) e.preventDefault();
        if (!input.trim() || loading || !anomaly) return;

        // Always use the latest screenshot if available
        const screenshotToSend = latestScreenshot;

        const userMsg: ChatMessage = { 
            role: 'user', 
            content: input,
            screenshot: screenshotToSend || undefined
        };
        
        setMessages(prev => [...prev, userMsg]);
        setInput('');
        setLoading(true);

        try {
            // Send full conversation history (excluding the message we just added)
            const historyToSend = messages
                .map(m => ({ role: m.role, content: m.content }));

            const response = await analyzeWithAI({
                screenshot: screenshotToSend ? stripDataUrlPrefix(screenshotToSend) : '',
                question: input,
                flight_id: anomaly.flight_id,
                flight_data: flightPoints,
                anomaly_report: anomaly.full_report,
                history: historyToSend
            });

            // Parse actions from response
            const actions = response.actions || parseActionsFromResponse(response.response);
            const cleanedText = actions.length > 0 
                ? stripActionsFromText(response.response) 
                : response.response;

            // Process and apply actions
            if (actions.length > 0) {
                const processedActions = processActions(actions, flightPoints);
                onAIActions(processedActions);
            }

            setMessages(prev => [...prev, {
                role: 'assistant',
                content: cleanedText,
                actions: actions.length > 0 ? actions : undefined
            }]);

        } catch (err: any) {
            setMessages(prev => [...prev, {
                role: 'assistant',
                content: `Sorry, I encountered an error: ${err.message}`
            }]);
        } finally {
            setLoading(false);
        }
    };

    // ============================================================
    // Clear Chat
    // ============================================================

    const handleClearChat = () => {
        setMessages([
            { role: 'assistant', content: isScreenSharing 
                ? "Chat cleared! Still watching your screen - ask me anything." 
                : "👁️ **Visual Analyst**\n\nI analyze what you see on screen for this flight. Share your screen to enable visual analysis, then ask me questions about what you're looking at."
            }
        ]);
        onAIActions({ highlightedPoint: null, highlightedSegment: null, zoomBounds: null });
    };

    if (!anomaly) return null;

    // ============================================================
    // Render
    // ============================================================

    return (
        <aside className={clsx(
            "bg-surface rounded-xl flex flex-col h-full overflow-hidden border border-white/5 animate-in slide-in-from-right-4",
            className || "col-span-3"
        )}>
            {/* Header with Tabs */}
            <div className="border-b border-white/10 bg-surface-highlight/50">
                {/* Title Row */}
                <div className="p-4 pb-2 flex items-center justify-between">
                    <div>
                        <h3 className="text-white font-bold">Flight Analysis</h3>
                        <p className="text-xs text-white/60">{anomaly.callsign || anomaly.flight_id}</p>
                    </div>
                    <button 
                        onClick={onClose} 
                        className="text-white/60 hover:text-white p-1 rounded hover:bg-white/10"
                    >
                        <X className="size-5" />
                    </button>
                </div>
                
                {/* Tab Buttons */}
                <div className="flex px-4 gap-1">
                    <button
                        onClick={() => setActiveTab('report')}
                        className={clsx(
                            "flex items-center gap-2 px-4 py-2 text-sm font-medium rounded-t-lg transition-all border-b-2",
                            activeTab === 'report'
                                ? "bg-surface text-white border-primary"
                                : "text-white/60 hover:text-white border-transparent hover:bg-white/5"
                        )}
                    >
                        <FileText className="size-4" />
                        Report
                    </button>
                    <button
                        onClick={() => setActiveTab('ai')}
                        className={clsx(
                            "flex items-center gap-2 px-4 py-2 text-sm font-medium rounded-t-lg transition-all border-b-2",
                            activeTab === 'ai'
                                ? "bg-surface text-white border-amber-500"
                                : "text-white/60 hover:text-white border-transparent hover:bg-white/5"
                        )}
                    >
                        <ScanEye className="size-4" />
                        Visual Analyst
                        {isScreenSharing && (
                            <span className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
                        )}
                    </button>
                </div>
            </div>

            {/* Tab Content */}
            <div className="flex-1 overflow-hidden">
                {activeTab === 'report' ? (
                    // Report Tab - Use existing ReportPanel content
                    <ReportPanelContent anomaly={anomaly} onClose={onClose} mode={mode} onFlyTo={onFlyTo} />
                ) : (
                    // AI Assistant Tab
                    <div className="flex flex-col h-full">
                        {/* Screen Sharing Status Bar - Enhanced */}
                        {isScreenSharing && (
                            <div className="px-4 py-3 bg-gradient-to-r from-green-500/10 to-amber-500/10 border-b border-green-500/20">
                                <div className="flex items-center justify-between mb-2">
                                    <div className="flex items-center gap-2">
                                        <div className="relative">
                                            <Eye className="size-4 text-green-400" />
                                            <span className="absolute -top-0.5 -right-0.5 w-2 h-2 bg-green-500 rounded-full animate-ping" />
                                            <span className="absolute -top-0.5 -right-0.5 w-2 h-2 bg-green-500 rounded-full" />
                                        </div>
                                        <span className="text-xs text-green-300 font-bold">LIVE</span>
                                    </div>
                                    <div className="flex items-center gap-2">
                                        <button
                                            onClick={captureNow}
                                            className="flex items-center gap-1 px-2 py-1 rounded bg-amber-500/20 text-amber-300 text-[10px] font-medium hover:bg-amber-500/30 transition-colors border border-amber-500/30"
                                            title="Capture now"
                                        >
                                            <Camera className="size-3" />
                                            Capture
                                        </button>
                                        <div className="flex items-center gap-1 px-2 py-1 rounded bg-white/5 text-white/60 text-[10px]">
                                            <Clock className="size-3" />
                                            <span className="font-mono w-3 text-center">{captureCountdown}</span>s
                                        </div>
                                    </div>
                                </div>
                                {/* Screenshot preview */}
                                {latestScreenshot && (
                                    <div className="relative group">
                                        <img 
                                            src={latestScreenshot} 
                                            alt="Latest capture" 
                                            className="w-full h-20 object-cover rounded-lg border border-white/10 opacity-90 group-hover:opacity-100 transition-opacity"
                                        />
                                        <div className="absolute bottom-1 left-1 px-1.5 py-0.5 rounded bg-black/60 text-[9px] text-white/70">
                                            {lastCaptureTime?.toLocaleTimeString()}
                                        </div>
                                    </div>
                                )}
                            </div>
                        )}

                        {/* Messages */}
                        <div className="flex-1 overflow-y-auto p-4 space-y-4">
                            {messages.map((msg, i) => (
                                <div
                                    key={i}
                                    className={clsx(
                                        "flex gap-3 items-start max-w-[90%]",
                                        msg.role === 'user' ? "self-end flex-row-reverse" : "self-start"
                                    )}
                                >
                                    <div
                                        className={clsx(
                                            "h-8 w-8 rounded-full flex items-center justify-center shrink-0",
                                            msg.role === 'user'
                                                ? "bg-primary/20 text-primary"
                                                : "bg-amber-500/20 text-amber-400"
                                        )}
                                    >
                                        {msg.role === 'user' ? <User className="h-4 w-4" /> : <Bot className="h-4 w-4" />}
                                    </div>

                                    <div className="flex flex-col gap-2">
                                        {/* Screenshot thumbnail for user messages */}
                                        {msg.screenshot && (
                                            <img 
                                                src={msg.screenshot} 
                                                alt="Screen capture" 
                                                className="rounded-lg border border-white/10 max-w-[200px] opacity-80"
                                            />
                                        )}
                                        
                                        <div
                                            className={clsx(
                                                "px-3 py-2 rounded-xl text-sm leading-relaxed",
                                                msg.role === 'user'
                                                    ? "bg-primary text-white rounded-tr-none whitespace-pre-wrap"
                                                    : "bg-white/5 border border-white/10 rounded-tl-none text-white/90"
                                            )}
                                        >
                                            {msg.role === 'assistant' ? (
                                                <TypewriterMarkdown 
                                                    text={msg.content} 
                                                    shouldAnimate={i === messages.length - 1} 
                                                />
                                            ) : (
                                                msg.content
                                            )}
                                        </div>

                                        {/* Action indicator */}
                                        {msg.actions && msg.actions.length > 0 && (
                                            <div className="flex items-center gap-1 text-[10px] text-amber-400">
                                                <ScanEye className="size-3" />
                                                <span>Map highlight applied</span>
                                            </div>
                                        )}
                                    </div>
                                </div>
                            ))}

                            {loading && (
                                <div className="flex gap-3 items-start">
                                    <div className="h-8 w-8 rounded-full bg-amber-500/20 flex items-center justify-center">
                                        <Bot className="h-4 w-4 text-amber-400" />
                                    </div>
                                    <div className="px-3 py-2 rounded-xl rounded-tl-none bg-white/5 border border-white/10">
                                        <div className="flex gap-1">
                                            <span className="w-2 h-2 bg-amber-400/60 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></span>
                                            <span className="w-2 h-2 bg-amber-400/60 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></span>
                                            <span className="w-2 h-2 bg-amber-400/60 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></span>
                                        </div>
                                    </div>
                                </div>
                            )}

                            <div ref={messagesEndRef} />
                        </div>

                        {/* Input Area */}
                        <div className="p-3 border-t border-white/10 bg-surface-highlight/30">
                            {/* Screen Sharing Toggle - Enhanced */}
                            <div className="flex gap-2 mb-3">
                                <button
                                    onClick={toggleScreenSharing}
                                    className={clsx(
                                        "flex-1 flex items-center justify-center gap-2 py-2.5 rounded-lg text-sm font-medium transition-all",
                                        isScreenSharing
                                            ? "bg-gradient-to-r from-green-500/20 to-green-600/20 text-green-300 border border-green-500/40 hover:from-red-500/20 hover:to-red-600/20 hover:text-red-300 hover:border-red-500/40"
                                            : "bg-gradient-to-r from-amber-500/20 to-orange-500/20 text-amber-300 border border-amber-500/40 hover:from-amber-500/30 hover:to-orange-500/30"
                                    )}
                                >
                                    {isScreenSharing ? (
                                        <>
                                            <MonitorOff className="size-4" />
                                            Stop Sharing
                                        </>
                                    ) : (
                                        <>
                                            <MonitorUp className="size-4" />
                                            Share Screen
                                        </>
                                    )}
                                </button>
                                <button
                                    onClick={handleClearChat}
                                    className="p-2.5 rounded-lg text-white/40 hover:text-white hover:bg-white/10 transition-colors border border-white/10"
                                    title="Clear chat"
                                >
                                    <Trash2 className="size-4" />
                                </button>
                            </div>

                            {/* Hint when not sharing */}
                            {!isScreenSharing && (
                                <div className="flex items-center gap-2 mb-3 p-2 rounded-lg bg-amber-500/10 border border-amber-500/20">
                                    <ScanEye className="size-4 text-amber-400 shrink-0" />
                                    <p className="text-[10px] text-amber-300/80">
                                        Share your screen so I can analyze what you're looking at
                                    </p>
                                </div>
                            )}

                            {/* Text Input */}
                            <form onSubmit={handleSend} className="flex gap-2">
                                <input
                                    type="text"
                                    value={input}
                                    onChange={e => setInput(e.target.value)}
                                    placeholder={isScreenSharing ? "What do you see on the map?" : "Share screen first, or type a question..."}
                                    className="flex-1 px-3 py-2.5 rounded-lg bg-black/20 border border-white/10 text-white text-sm placeholder:text-white/30 focus:outline-none focus:border-amber-500/50 focus:ring-1 focus:ring-amber-500/20"
                                    disabled={loading}
                                />
                                <button
                                    type="submit"
                                    disabled={loading || !input.trim()}
                                    className="p-2.5 rounded-lg bg-amber-500 text-white hover:bg-amber-600 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
                                >
                                    {loading ? (
                                        <Loader2 className="size-4 animate-spin" />
                                    ) : (
                                        <Send className="size-4" />
                                    )}
                                </button>
                            </form>
                        </div>
                    </div>
                )}
            </div>
        </aside>
    );
};

// ============================================================
// Report Panel Content (extracted from ReportPanel for embedding)
// ============================================================

const ReportPanelContent: React.FC<{ anomaly: AnomalyReport; onClose: () => void; mode?: 'historical' | 'realtime' | 'research' | 'rules' | 'feedback' | 'ai-results'; onFlyTo?: (lat: number, lon: number, zoom?: number) => void }> = ({ anomaly, onClose, mode, onFlyTo }) => {
    // This wraps the ReportPanel but removes its outer container for embedding
    return (
        <div className="h-full overflow-y-auto">
            <ReportPanel 
                anomaly={anomaly} 
                onClose={onClose} 
                className="!col-span-full !rounded-none !border-0 !animate-none"
                mode={mode}
                onFlyTo={onFlyTo}
            />
        </div>
    );
};
