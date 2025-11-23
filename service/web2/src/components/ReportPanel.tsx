import React, { useState } from 'react';
import { X, AlertTriangle, CheckCircle, ThumbsUp } from 'lucide-react';
import type { AnomalyReport } from '../types';
import { submitFeedback } from '../api';
import clsx from 'clsx';

interface ReportPanelProps {
    anomaly: AnomalyReport | null;
    onClose: () => void;
}

const LayerCard: React.FC<{ title: string; data: any; type: 'rules' | 'model' }> = ({ title, data, type }) => {
    if (!data) return null;

    const isAnomaly = type === 'rules' ? data.status === 'ANOMALY' : data.is_anomaly;
    const statusColor = isAnomaly ? 'text-red-400' : 'text-green-400';
    const Icon = isAnomaly ? AlertTriangle : CheckCircle;

    // Distinct styling for anomalies
    const cardStyle = isAnomaly 
        ? "bg-red-500/10 border-red-500/30" 
        : "bg-white/5 border-white/10";

    return (
        <div className={clsx("rounded-lg p-3 border transition-all", cardStyle)}>
            <div className="flex items-center justify-between mb-2">
                <h4 className="text-sm font-bold text-white">{title}</h4>
                <Icon className={clsx("size-4", statusColor)} />
            </div>
            
            <div className="space-y-1">
                {type === 'rules' && (
                    <>
                        <p className="text-xs text-white/60">Status: <span className={statusColor}>{data.status}</span></p>
                        {data.triggers && data.triggers.length > 0 && (
                            <div className="mt-1">
                                <p className="text-xs text-white/60 mb-1">Triggers:</p>
                                <ul className="list-disc list-inside text-xs text-white/80">
                                    {data.triggers.map((t: string, i: number) => (
                                        <li key={i}>{t}</li>
                                    ))}
                                </ul>
                            </div>
                        )}
                    </>
                )}

                {type === 'model' && (
                    <>
                         <p className="text-xs text-white/60">
                            Prediction: <span className={statusColor}>{isAnomaly ? 'Anomaly' : 'Normal'}</span>
                         </p>
                         {data.severity !== undefined && (
                            <div className="mt-1">
                                <p className="text-xs text-white/60">Severity Score</p>
                                <div className="h-1.5 w-full bg-white/10 rounded-full mt-1 overflow-hidden">
                                    <div 
                                        className={clsx("h-full rounded-full", isAnomaly ? "bg-red-500" : "bg-green-500")} 
                                        style={{ width: `${Math.min(100, data.severity * 100)}%` }}
                                    />
                                </div>
                                <p className="text-[10px] text-right text-white/40 mt-0.5">{data.severity.toFixed(3)}</p>
                            </div>
                         )}
                         {data.score !== undefined && (
                             <p className="text-xs text-white/60">Score: {data.score.toFixed(3)}</p>
                         )}
                    </>
                )}
            </div>
        </div>
    );
};

export const ReportPanel: React.FC<ReportPanelProps> = ({ anomaly, onClose }) => {
    const [feedbackStatus, setFeedbackStatus] = useState<'idle' | 'submitting' | 'success' | 'error'>('idle');
    const [comment, setComment] = useState('');
    const [copied, setCopied] = useState(false);

    if (!anomaly) return null;

    const handleCopy = (text: string) => {
        navigator.clipboard.writeText(text);
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
    };

    const handleFeedback = async (isAnomaly: boolean) => {
        setFeedbackStatus('submitting');
        try {
            await submitFeedback(anomaly.flight_id, isAnomaly, comment);
            setFeedbackStatus('success');
            setComment('');
            
            if (!isAnomaly) {
                // If confirmed normal, wait a moment then close/remove
                setTimeout(() => {
                    onClose();
                    // Optionally trigger a refresh of the list if parent passed a callback
                    // but onClose should handle removing selection from UI
                }, 1000);
            } else {
                // Reset status after 3 seconds for anomalies (keep panel open)
                setTimeout(() => setFeedbackStatus('idle'), 3000);
            }
        } catch (e) {
            setFeedbackStatus('error');
        }
    };

    const report = anomaly.full_report || {};
    const summary = report.summary || {};

    const getConfidenceColor = (score: number) => {
        if (score > 85) return "text-red-500";
        if (score > 70) return "text-purple-500";
        if (score > 20) return "text-yellow-500";
        return "text-pink-500";
    };

    const confidenceColor = getConfidenceColor(summary.confidence_score || 0);

    return (
        <aside className="col-span-3 bg-[#2C2F33] rounded-xl flex flex-col h-full overflow-hidden border border-white/5 animate-in slide-in-from-right-4">
            {/* Header */}
            <div className="p-4 border-b border-white/10 flex items-center justify-between bg-white/5">
                <div>
                    <h3 className="text-white font-bold">Analysis Report</h3>
                    <p className="text-xs text-white/60">{anomaly.flight_id}</p>
                    
                    {anomaly.callsign && (
                        <div className="mt-3">
                            <p className="text-[10px] text-pink-300 mb-0.5 animate-pulse font-medium">
                                ✨ click me to copy
                            </p>
                            <button 
                                onClick={() => handleCopy(anomaly.callsign!)}
                                className={clsx(
                                    "text-sm font-mono font-bold px-3 py-1 rounded border transition-all duration-200",
                                    copied 
                                        ? "bg-green-500/20 text-green-300 border-green-500/30" 
                                        : "bg-white/10 text-white hover:bg-white/20 border-white/10 hover:border-white/30"
                                )}
                            >
                                {copied ? "Copied!" : anomaly.callsign}
                            </button>
                        </div>
                    )}
                </div>
                <button onClick={onClose} className="text-white/60 hover:text-white p-1 rounded hover:bg-white/10 self-start">
                    <X className="size-5" />
                </button>
            </div>

            {/* Content */}
            <div className="flex-1 overflow-y-auto p-4 space-y-4">
                
                {/* Overall Summary */}
                <div className="bg-primary/10 rounded-lg p-3 border border-primary/20">
                    <p className="text-xs text-primary font-bold uppercase mb-1">System Verdict</p>
                    <div className="flex items-center gap-2 mb-2">
                        <span className={clsx("text-lg font-bold", summary.is_anomaly ? "text-red-400" : "text-green-400")}>
                            {summary.is_anomaly ? "ANOMALY DETECTED" : "NORMAL FLIGHT"}
                        </span>
                    </div>
                    <div className="flex flex-col gap-1">
                        <p className="text-xs text-white/80">
                            Confidence Score: <span className={clsx("font-mono font-bold", confidenceColor)}>{summary.confidence_score}%</span>
                        </p>
                        <p className="text-xs text-white/60">
                            Detected At: <span className="font-mono">{new Date(anomaly.timestamp * 1000).toLocaleString()}</span>
                        </p>
                    </div>
                </div>

                {/* Layers */}
                <div className="space-y-3">
                    <p className="text-xs text-white/40 font-bold uppercase tracking-wider">Layer Analysis</p>
                    
                    <LayerCard 
                        title="Layer 1: Rule Engine" 
                        data={report.layer_1_rules} 
                        type="rules" 
                    />
                    
                    <LayerCard 
                        title="Layer 2: XGBoost" 
                        data={report.layer_2_xgboost} 
                        type="model" 
                    />

                    <LayerCard 
                        title="Layer 3: Deep Dense Autoencoder" 
                        data={report.layer_3_deep_dense} 
                        type="model" 
                    />

                    <LayerCard 
                        title="Layer 4: Deep CNN" 
                        data={report.layer_4_deep_cnn} 
                        type="model" 
                    />

                    <LayerCard 
                        title="Layer 5: Transformer" 
                        data={report.layer_5_transformer} 
                        type="model" 
                    />
                </div>

                {/* Feedback Section */}
                <div className="bg-white/5 rounded-lg p-3 border border-white/10 mt-4">
                    <p className="text-xs text-white/40 font-bold uppercase tracking-wider mb-3">Human Feedback</p>
                    
                    {feedbackStatus === 'success' ? (
                        <div className="text-green-400 text-sm flex items-center gap-2 p-2 bg-green-500/10 rounded border border-green-500/20 animate-in fade-in">
                            <CheckCircle className="size-4" />
                            <span>Feedback submitted successfully</span>
                        </div>
                    ) : (
                        <div className="space-y-3">
                            <p className="text-xs text-white/60">Is this actually an anomaly?</p>
                            
                            <div className="flex gap-2">
                                <button
                                    onClick={() => handleFeedback(true)}
                                    disabled={feedbackStatus === 'submitting'}
                                    className="flex-1 flex items-center justify-center gap-2 p-2 rounded bg-red-500/20 hover:bg-red-500/30 border border-red-500/30 text-red-200 text-sm transition-colors disabled:opacity-50"
                                >
                                    <AlertTriangle className="size-4" />
                                    Yes, Anomaly
                                </button>
                                <button
                                    onClick={() => handleFeedback(false)}
                                    disabled={feedbackStatus === 'submitting'}
                                    className="flex-1 flex items-center justify-center gap-2 p-2 rounded bg-green-500/20 hover:bg-green-500/30 border border-green-500/30 text-green-200 text-sm transition-colors disabled:opacity-50"
                                >
                                    <ThumbsUp className="size-4" />
                                    No, Normal
                                </button>
                            </div>

                            <div className="relative">
                                <input
                                    type="text"
                                    value={comment}
                                    onChange={(e) => setComment(e.target.value)}
                                    placeholder="Optional comments..."
                                    className="w-full bg-black/20 border border-white/10 rounded p-2 text-xs text-white placeholder:text-white/20 focus:outline-none focus:border-primary/50"
                                />
                            </div>
                            
                            {feedbackStatus === 'error' && (
                                <p className="text-xs text-red-400">Failed to submit feedback. Try again.</p>
                            )}
                        </div>
                    )}
                </div>
            </div>
        </aside>
    );
};

