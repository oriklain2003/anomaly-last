import React, { useState, useRef, useEffect } from 'react';
import { Send, User, ChevronRight, ChevronLeft, Trash2, Plane, Database, Search } from 'lucide-react';
import clsx from 'clsx';
import type { ChatMessage } from '../chatTypes';
import type { AnomalyReport, TrackPoint, AIReasoningResponse } from '../types';
import { sendReasoningQuery } from '../api';
import { TypewriterMarkdown } from '../utils/markdown';

interface ReasoningChatProps {
    isOpen: boolean;
    onToggle: () => void;
    onFlightsReceived: (flights: AnomalyReport[]) => void;
    selectedFlight?: {
        flightId: string;
        callsign?: string;
        points: TrackPoint[];
        report?: any;
    } | null;
    className?: string;
}

// TypewriterText removed - now using TypewriterMarkdown from utils/markdown

export const ReasoningChat: React.FC<ReasoningChatProps> = ({
    isOpen,
    onToggle,
    onFlightsReceived,
    selectedFlight,
    className
}) => {
    const [messages, setMessages] = useState<ChatMessage[]>([
        { role: 'assistant', content: '✈️ **AI Flight Assistant**\n\n**Search flights:**\n• "Turn anomalies from last week"\n• "Go-arounds yesterday"\n\n**Ask about selected flight:**\n• "Why was this flagged?"\n• "Explain this anomaly"\n\n**General questions:**\n• "What airport is LLBG?"' }
    ]);
    const [input, setInput] = useState('');
    const [loading, setLoading] = useState(false);
    const messagesEndRef = useRef<HTMLDivElement>(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages, isOpen]);

    const handleSend = async (e?: React.FormEvent) => {
        if (e) e.preventDefault();
        if (!input.trim() || loading) return;

        const userMsg: ChatMessage = { role: "user", content: input };
        setMessages(prev => [...prev, userMsg]);
        setInput("");
        setLoading(true);

        try {
            // Build flight context if a flight is selected (for map image generation)
            const flightContext = selectedFlight && selectedFlight.points.length >= 2 ? {
                flightId: selectedFlight.flightId,
                points: selectedFlight.points,
                anomalyReport: selectedFlight.report
            } : undefined;

            const response: AIReasoningResponse = await sendReasoningQuery(
                input,
                messages.filter(m => m.role !== 'system'),
                flightContext
            );

            // Handle response based on type
            if (response.type === 'flights' && response.flights && response.flights.length > 0) {
                // Notify parent about flights
                onFlightsReceived(response.flights);
                
                setMessages(prev => [...prev, {
                    role: "assistant",
                    content: `${response.response}\n\n✨ Found ${response.flights?.length} flight(s) - check the **AI Results** tab in the sidebar to view them.`
                }]);
            } else {
                setMessages(prev => [...prev, {
                    role: "assistant",
                    content: response.response
                }]);
            }

        } catch (err: any) {
            setMessages(prev => [...prev, {
                role: "assistant",
                content: `Sorry, I encountered an error: ${err.message}`
            }]);
        } finally {
            setLoading(false);
        }
    };

    const clearChat = () => {
        setMessages([
            { role: 'assistant', content: '✈️ **AI Flight Assistant**\n\n**Search flights:**\n• "Turn anomalies from last week"\n• "Go-arounds yesterday"\n\n**Ask about selected flight:**\n• "Why was this flagged?"\n• "Explain this anomaly"\n\n**General questions:**\n• "What airport is LLBG?"' }
        ]);
    };

    // Collapsed state - show expand tab
    if (!isOpen) {
        return (
            <button
                onClick={onToggle}
                className={clsx(
                    "fixed right-0 top-1/2 -translate-y-1/2 z-40",
                    "flex items-center gap-2 px-2 py-4",
                    "bg-gradient-to-r from-violet-600 to-fuchsia-600 text-white",
                    "rounded-l-xl shadow-lg hover:shadow-xl",
                    "transition-all hover:px-3",
                    className
                )}
            >
                <ChevronLeft className="h-5 w-5" />
                <div className="flex flex-col items-center gap-1">
                    <Database className="h-5 w-5" />
                    <span className="text-xs font-bold writing-mode-vertical" style={{ writingMode: 'vertical-rl', textOrientation: 'mixed' }}>
                        AI Assistant
                    </span>
                </div>
            </button>
        );
    }

    return (
        <div className={clsx(
            "flex flex-col h-full bg-surface rounded-xl border border-white/10 overflow-hidden",
            "animate-in slide-in-from-right-4 duration-300",
            className
        )}>
            {/* Header */}
            <div className="flex items-center justify-between p-4 bg-gradient-to-r from-violet-600 to-fuchsia-600 text-white">
                <div className="flex flex-col">
                    <div className="flex items-center gap-2">
                        <Database className="h-5 w-5" />
                        <h3 className="font-semibold">AI Assistant</h3>
                    </div>
                    <span className="text-[10px] text-white/70 ml-7">Search, analyze & ask questions</span>
                </div>
                <div className="flex items-center gap-1">
                    <button 
                        onClick={clearChat}
                        className="hover:bg-white/20 p-2 rounded-lg transition-colors group"
                        title="Clear Chat"
                    >
                        <Trash2 className="h-4 w-4 text-white/70 group-hover:text-white" />
                    </button>
                    <button 
                        onClick={onToggle}
                        className="hover:bg-white/20 p-2 rounded-lg transition-colors"
                        title="Collapse Panel"
                    >
                        <ChevronRight className="h-5 w-5" />
                    </button>
                </div>
            </div>

            {/* Flight Context Banner */}
            {selectedFlight && (
                <div className="px-4 py-2 bg-gradient-to-r from-violet-500/10 to-fuchsia-500/10 border-b border-violet-500/20 flex items-center justify-between">
                    <div className="flex items-center gap-2">
                        <Plane className="h-4 w-4 text-violet-400" />
                        <span className="text-xs text-white/90">
                            <span className="text-white font-bold">{selectedFlight.callsign || selectedFlight.flightId}</span>
                        </span>
                    </div>
                    <span className="text-[10px] text-violet-300/70">Ask me about this flight</span>
                </div>
            )}

            {/* Messages */}
            <div className="flex-1 overflow-y-auto p-4 flex flex-col gap-4">
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
                                "h-8 w-8 rounded-full flex items-center justify-center shadow-md shrink-0",
                                msg.role === 'user'
                                    ? "bg-blue-500/20 text-blue-400"
                                    : "bg-gradient-to-br from-violet-500/20 to-fuchsia-500/20 text-violet-400"
                            )}
                        >
                            {msg.role === 'user' ? <User className="h-4 w-4" /> : <Search className="h-4 w-4" />}
                        </div>

                        <div
                            className={clsx(
                                "px-4 py-3 rounded-2xl text-sm leading-relaxed shadow-sm",
                                msg.role === 'user'
                                    ? "bg-blue-500 text-white rounded-tr-none whitespace-pre-wrap"
                                    : "bg-white/10 border border-white/10 rounded-tl-none text-white/90"
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
                    </div>
                ))}

                {loading && (
                    <div className="flex gap-3 items-start">
                        <div className="h-8 w-8 rounded-full bg-gradient-to-br from-violet-500/20 to-fuchsia-500/20 flex items-center justify-center">
                            <Search className="h-4 w-4 text-violet-400 animate-pulse" />
                        </div>

                        <div className="px-4 py-3 rounded-2xl rounded-tl-none bg-white/10 border border-white/10 shadow-sm">
                            <div className="flex items-center gap-2">
                                <div className="flex gap-1">
                                    <span className="w-2 h-2 bg-violet-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></span>
                                    <span className="w-2 h-2 bg-violet-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></span>
                                    <span className="w-2 h-2 bg-violet-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></span>
                                </div>
                                <span className="text-xs text-white/50">Thinking...</span>
                            </div>
                        </div>
                    </div>
                )}

                <div ref={messagesEndRef} />
            </div>

            {/* Input */}
            <form onSubmit={handleSend} className="p-4 bg-background-dark/50 border-t border-white/10 flex gap-3">
                <input
                    type="text"
                    value={input}
                    onChange={e => setInput(e.target.value)}
                    placeholder={selectedFlight ? "Ask about this flight or search..." : "Search flights or ask a question..."}
                    className="flex-1 px-4 py-2.5 rounded-xl bg-white/10 border border-white/10 text-white placeholder:text-white/40 focus:ring-2 focus:ring-violet-500/30 focus:border-violet-500/50 outline-none transition-all"
                />
                <button
                    type="submit"
                    disabled={loading || !input.trim()}
                    className="p-3 rounded-xl bg-gradient-to-r from-violet-600 to-fuchsia-600 text-white hover:opacity-90 transition-opacity disabled:opacity-40 disabled:cursor-not-allowed"
                >
                    <Send className="h-5 w-5" />
                </button>
            </form>
        </div>
    );
};

