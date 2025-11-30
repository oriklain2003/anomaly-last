import React, { useState, useRef, useEffect } from 'react';
import { Send, Bot, User, X, MessageSquare } from 'lucide-react';
import clsx from 'clsx';
import type { AnalysisResult } from '../types';
import type { ChatMessage } from '../chatTypes';

interface ChatInterfaceProps {
  data: AnalysisResult | null;
  flightId: string;
}

const TypewriterText: React.FC<{ text: string; shouldAnimate: boolean }> = ({ text, shouldAnimate }) => {
  const [displayedText, setDisplayedText] = useState(shouldAnimate ? '' : text);
  const hasAnimatedRef = useRef(false);

  useEffect(() => {
    if (!shouldAnimate) {
        setDisplayedText(text);
        return;
    }
    
    if (hasAnimatedRef.current) return;

    setDisplayedText('');
    let i = 0;
    const timer = setInterval(() => {
      setDisplayedText((_prev) => text.slice(0, i + 1));
      i++;
      if (i >= text.length) {
        clearInterval(timer);
        hasAnimatedRef.current = true;
      }
    }, 15);

    return () => clearInterval(timer);
  }, [text, shouldAnimate]);

  return <span>{displayedText}</span>;
};

export const ChatInterface: React.FC<ChatInterfaceProps> = ({ data, flightId }) => {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>([
    { role: 'assistant', content: '👋 Hi! I can help you analyze this flight. Ask me anything about anomalies, rules, or flight events.' }
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

    const userMsg: ChatMessage = { role: 'user', content: input };
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setLoading(true);

    try {
        // In a real app, you should proxy this through your backend to hide the key.
        // For this demo, we'll do it client-side as requested, but THIS IS NOT SECURE FOR PRODUCTION.
        const OPENAI_KEY = import.meta.env.VITE_OPENAI_API_KEY;
        
        const systemPrompt = `You are an expert aviation anomaly detection assistant. 
        You are provided with JSON analysis data for a flight (ID: ${flightId}).
        
        Data Context:
        ${JSON.stringify(data, null, 2)}
        
        Explain technical details simply. If there are anomalies, explain why they might be dangerous.
        Focus on the triggered rules and model scores.
        If the user asks about something not in the data, say you don't know based on the current analysis.`;
  
        const response = await fetch('https://api.openai.com/v1/chat/completions', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${OPENAI_KEY}`
          },
          body: JSON.stringify({
            model: "gpt-5", 
            messages: [
              { role: 'system', content: systemPrompt },
              ...messages.map(m => ({ role: m.role, content: m.content })),
              { role: 'user', content: userMsg.content }
            ]
          })
        });
  
        if (!response.ok) {
          throw new Error(`OpenAI API Error: ${response.statusText}`);
        }
  
        const json = await response.json();
        const aiMsg = json.choices[0].message;
        
        setMessages(prev => [...prev, { role: 'assistant', content: aiMsg.content }]);
  
      } catch (error: any) {
        setMessages(prev => [...prev, { role: 'assistant', content: `Sorry, I encountered an error: ${error.message}` }]);
      } finally {
        setLoading(false);
      }
    };
  

  // ---------------------------
  // Floating Button
  // ---------------------------

  if (!isOpen) {
    return (
      <button
        onClick={() => setIsOpen(true)}
        className="fixed bottom-6 right-6 h-16 w-16 bg-primary text-white rounded-full shadow-2xl flex items-center justify-center hover:bg-blue-700 transition-all z-50 hover:scale-110 backdrop-blur-sm"
      >
        <MessageSquare className="h-8 w-8" />
      </button>
    );
  }

  // ---------------------------
  // CHAT WINDOW
  // ---------------------------

  return (
    <div className="
      fixed bottom-6 right-6 
      w-[420px] h-[620px]
      max-h-[80vh] max-w-[90vw]
      rounded-3xl border border-white/20 shadow-2xl z-50 overflow-hidden
      bg-white/80 dark:bg-gray-900/80 
      backdrop-blur-xl
      animate-in slide-in-from-bottom-4 fade-in duration-300
      flex flex-col
    ">
      
      {/* Header */}
      <div className="flex items-center justify-between p-4 bg-primary text-white shadow-lg">
        <div className="flex items-center gap-2">
          <Bot className="h-5 w-5" />
          <h3 className="font-semibold">Flight Analyst AI</h3>
        </div>
        <button 
          onClick={() => setIsOpen(false)} 
          className="hover:bg-white/20 p-1 rounded-lg transition-colors"
        >
          <X className="h-5 w-5" />
        </button>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-5 flex flex-col gap-5">
        {messages.map((msg, i) => (
          <div
            key={i}
            className={clsx(
              "flex gap-3 items-start max-w-[85%]",
              msg.role === 'user' ? "self-end flex-row-reverse" : "self-start"
            )}
          >
            <div
              className={clsx(
                "h-9 w-9 rounded-full flex items-center justify-center shadow-md",
                msg.role === 'user'
                  ? "bg-blue-100 dark:bg-blue-800/50 text-blue-600"
                  : "bg-primary/10 text-primary"
              )}
            >
              {msg.role === 'user' ? <User className="h-4 w-4" /> : <Bot className="h-4 w-4" />}
            </div>

            <div
              className={clsx(
                "px-4 py-3 rounded-2xl text-sm leading-relaxed shadow-sm whitespace-pre-wrap",
                msg.role === 'user'
                  ? "bg-primary text-white rounded-tr-none"
                  : "bg-white/90 dark:bg-gray-800/90 border border-gray-200/40 dark:border-gray-700/40 rounded-tl-none"
              )}
            >
              {msg.role === 'assistant' ? (
                <TypewriterText 
                    text={msg.content} 
                    shouldAnimate={i === messages.length - 1} 
                />
              ) : (
                msg.content
              )}
            </div>
          </div>
        ))}

        {/* Typing Indicator */}
        {loading && (
          <div className="flex gap-3 items-start">
            <div className="h-9 w-9 rounded-full bg-primary/10 flex items-center justify-center">
              <Bot className="h-4 w-4 text-primary" />
            </div>

            <div className="px-4 py-3 rounded-2xl rounded-tl-none bg-white/90 dark:bg-gray-800/90 border border-gray-200 dark:border-gray-700 shadow-sm">
              <div className="flex gap-1">
                <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></span>
                <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></span>
                <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></span>
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <form onSubmit={handleSend} className="p-4 bg-white/50 dark:bg-gray-900/50 border-t border-gray-200/40 dark:border-gray-700/40 backdrop-blur-xl flex gap-3">
        <input
          type="text"
          value={input}
          onChange={e => setInput(e.target.value)}
          placeholder="Ask about the flight..."
          className="flex-1 px-4 py-2 rounded-xl bg-gray-100/80 dark:bg-gray-800/80 border-transparent focus:ring-2 focus:ring-primary/30 outline-none"
        />
        <button
          type="submit"
          disabled={loading || !input.trim()}
          className="p-3 rounded-xl bg-primary text-white hover:bg-blue-700 transition-colors disabled:opacity-40"
        >
          <Send className="h-5 w-5" />
        </button>
      </form>
    </div>
  );
};
