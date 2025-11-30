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

    // Special handling for "Dangerous Proximity" (ID 4)
    const renderProximityEvents = (rule: any) => {
        if (!rule.details?.events?.length) return null;
        return (
            <div className="mt-2 space-y-2">
                <p className="text-xs font-bold text-red-300">Conflict Details:</p>
                {rule.details.events.map((ev: any, idx: number) => (
                    <div key={idx} className="bg-red-500/10 p-2 rounded border border-red-500/20 text-[10px] text-red-200">
                        <div className="flex justify-between">
                            <span>Other Flight: <span className="font-bold font-mono">{ev.other_flight}</span></span>
                            <span className="font-mono text-white/60">
                                {new Date(ev.timestamp * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
                            </span>
                        </div>
                        <div className="flex justify-between mt-1 text-white/40">
                             <span>Dist: {ev.distance_nm} NM</span>
                             <span>Alt Diff: {ev.altitude_diff_ft} ft</span>
                        </div>
                    </div>
                ))}
            </div>
        );
    };

    // Special handling for "Path Deviation" (ID 11)
    const renderPathDeviation = (rule: any) => {
        if (!rule.details?.segments) return null;
        return (
            <div className="mt-2 space-y-2">
                <p className="text-xs font-bold text-white/60">Segment Analysis:</p>
                {Object.entries(rule.details.segments).map(([phase, data]: [string, any]) => {
                    // Check if phase was skipped
                    const isSkipped = data === "skipped_phase" || data === "skipped_short" || data === "skipped_resample";
                    
                    if (isSkipped) {
                        return (
                             <div key={phase} className="bg-white/5 p-2 rounded border border-white/10 text-[10px] text-white/60 opacity-50">
                                <div className="flex justify-between mb-1">
                                    <span className="font-bold uppercase text-white/40">{phase}</span>
                                    <span className="text-white/40">SKIPPED</span>
                                </div>
                                <span className="italic text-white/30">Not analyzed (not cruise or too short)</span>
                             </div>
                        );
                    }

                    return (
                        <div key={phase} className="bg-white/5 p-2 rounded border border-white/10 text-[10px] text-white/60">
                            <div className="flex justify-between mb-1">
                                <span className="font-bold uppercase text-white/40">{phase}</span>
                                <span className={data.match_found ? "text-green-400" : "text-red-400"}>
                                    {data.match_found ? "MATCH" : "NO MATCH"}
                                </span>
                            </div>
                            {data.match_found ? (
                                <div className="flex flex-col gap-0.5">
                                    <span>Flow: {data.flow_id} ({data.layer})</span>
                                    <span>Dist: {data.dist_nm} NM</span>
                                </div>
                            ) : (
                                <span className="italic">No matching path</span>
                            )}
                            {data.closest_loose_dist_nm !== undefined && (
                                 <div className="mt-1 pt-1 border-t border-white/10 text-blue-300">
                                     Loose Dist: {data.closest_loose_dist_nm} NM
                                 </div>
                            )}
                        </div>
                    );
                })}
            </div>
        );
    };

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
                                
                                {/* Show details for specific rules if present */}
                                {data.report?.matched_rules?.map((rule: any) => {
                                    if (rule.id === 4) return <div key={rule.id}>{renderProximityEvents(rule)}</div>;
                                    if (rule.id === 11) return <div key={rule.id}>{renderPathDeviation(rule)}</div>;
                                    return null;
                                })}
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

                    <LayerCard 
                        title="Layer 6: Hybrid CNN-Transformer" 
                        data={report.layer_6_hybrid} 
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

