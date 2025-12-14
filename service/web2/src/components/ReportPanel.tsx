import React, { useState, useEffect, useRef } from 'react';
import { X, AlertTriangle, CheckCircle, ThumbsUp, PlayCircle, Radio, Plane, Navigation, MapPin, TrendingDown, RotateCcw, Compass, ShieldAlert, Wifi, RefreshCw, Loader2, ExternalLink, ChevronDown } from 'lucide-react';
import type { AnomalyReport } from '../types';
import { submitFeedback, fetchCallsignFromResearch, fetchRules, reanalyzeFeedbackFlight } from '../api';
import clsx from 'clsx';
import { ReplayModal, ReplayEvent } from './ReplayModal';

// Available rules type
interface Rule {
    id: number;
    name: string;
    description: string;
}

// Rule icon mapping
const getRuleIcon = (ruleId: number) => {
    const iconMap: Record<number, React.ComponentType<any>> = {
        1: Radio,           // Emergency Squawk
        2: TrendingDown,    // Altitude Change
        3: RotateCcw,       // Abrupt Turn
        4: ShieldAlert,     // Proximity Alert
        6: Plane,           // Go-Around
        7: RotateCcw,       // Return to Field
        8: MapPin,          // Diversion
        9: TrendingDown,    // Low Altitude
        10: Wifi,           // Signal Loss
        11: Compass,        // Off Course
        12: MapPin,         // Unplanned Landing
    };
    return iconMap[ruleId] || AlertTriangle;
};

interface ReportPanelProps {
    anomaly: AnomalyReport | null;
    onClose: () => void;
    className?: string;
    mode?: 'historical' | 'realtime' | 'research' | 'rules' | 'feedback' | 'ai-results';
    onFlyTo?: (lat: number, lon: number, zoom?: number) => void;
}

// Layer color mapping for ML models
const LAYER_COLORS: Record<string, { bg: string; border: string; text: string; accent: string }> = {
    'Layer 3: Deep Dense Autoencoder': { bg: 'bg-purple-500/15', border: 'border-purple-400', text: 'text-purple-300', accent: 'bg-purple-500/30' },
    'Layer 4: Deep CNN': { bg: 'bg-orange-500/15', border: 'border-orange-400', text: 'text-orange-300', accent: 'bg-orange-500/30' },
    'Layer 5: Transformer': { bg: 'bg-cyan-500/15', border: 'border-cyan-400', text: 'text-cyan-300', accent: 'bg-cyan-500/30' },
    'Layer 6: Hybrid CNN-Transformer': { bg: 'bg-pink-500/15', border: 'border-pink-400', text: 'text-pink-300', accent: 'bg-pink-500/30' },
};

const LayerCard: React.FC<{ title: string; data: any; type: 'rules' | 'model'; resolvedCallsigns?: Record<string, string>; onFlyTo?: (lat: number, lon: number, zoom?: number) => void }> = ({ title, data, type, resolvedCallsigns, onFlyTo }) => {
    const [anomalyLocationsCollapsed, setAnomalyLocationsCollapsed] = useState(true);
    
    if (!data) return null;

    const isAnomaly = type === 'rules' ? data.status === 'ANOMALY' : data.is_anomaly;
    const statusColor = isAnomaly ? 'text-red-400' : 'text-green-400';
    const Icon = isAnomaly ? AlertTriangle : CheckCircle;
    
    // Get layer-specific colors for ML models
    const layerColor = LAYER_COLORS[title] || { bg: 'bg-orange-500/15', border: 'border-orange-400', text: 'text-orange-300', accent: 'bg-orange-500/30' };

    // Distinct styling for anomalies
    const cardStyle = isAnomaly 
        ? "bg-red-500/10 border-red-500/30" 
        : "bg-surface border-white/10";

    // Special handling for "Dangerous Proximity" (ID 4)
    const renderProximityEvents = (rule: any) => {
        if (!rule.details?.events?.length) return null;
        return (
            <div className="mt-2 space-y-2">
                <p className="text-xs font-bold text-red-300">Conflict Details:</p>
                {rule.details.events.map((ev: any, idx: number) => {
                    const callsign = ev.other_callsign || (resolvedCallsigns && resolvedCallsigns[ev.other_flight]);
                    return (
                        <div key={idx} className="bg-red-500/10 p-2 rounded border border-red-500/20 text-[10px] text-red-200">
                            <div className="flex justify-between">
                                <span>Other Flight: <span className="font-bold font-mono">{callsign ? `${callsign} (${ev.other_flight})` : ev.other_flight}</span></span>
                                <span className="font-mono text-white/60">
                                    {new Date(ev.timestamp * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
                                </span>
                            </div>
                            <div className="flex justify-between mt-1 text-white/40">
                                <span>Dist: {ev.distance_nm} NM</span>
                                <span>Alt Diff: {ev.altitude_diff_ft} ft</span>
                            </div>
                        </div>
                    );
                })}
            </div>
        );
    };

    // Special handling for "Path Deviation" (ID 11)
    const renderPathDeviation = (rule: any) => {
        const details = rule.details || {};
        
        return (
            <div className="mt-2 space-y-2">
                {details.deviations && details.deviations.length > 0 && (
                    <div className="space-y-1">
                        <p className="text-xs font-bold text-white/60">Deviations ({details.deviations.length}):</p>
                        <div className="max-h-40 overflow-y-auto space-y-1 pr-1">
                             {details.deviations.slice(0, 50).map((dev: any, idx: number) => (
                                <div key={idx} className="bg-red-500/10 p-2 rounded border border-red-500/20 text-[10px] text-red-200">
                                    <div className="flex justify-between items-center">
                                        <span className="font-mono opacity-70">
                                            {new Date(dev.timestamp * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
                                        </span>
                                        <span className="font-bold">{dev.dist_nm} NM off</span>
                                    </div>
                                    <div className="flex justify-between items-center mt-1 opacity-60">
                                        <span>Alt: {dev.alt} ft</span>
                                        <span>{dev.lat.toFixed(2)}, {dev.lon.toFixed(2)}</span>
                                    </div>
                                </div>
                            ))}
                            {details.deviations.length > 50 && (
                                <p className="text-[10px] text-center text-white/40 italic">
                                    + {details.deviations.length - 50} more points...
                                </p>
                            )}
                        </div>
                    </div>
                )}

                {details.segments && (
                    <div className="space-y-2">
                        <p className="text-xs font-bold text-white/60">Segment Analysis:</p>
                        {Object.entries(details.segments).map(([phase, data]: [string, any]) => {
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
                )}
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
                         
                         {/* ML Anomaly Points - Always show section with layer-specific colors */}
                         <div className="mt-3 pt-2 border-t border-white/10">
                            <div 
                                className="flex items-center justify-between mb-2 cursor-pointer hover:bg-white/5 -mx-1 px-1 py-1 rounded transition-colors"
                                onClick={() => setAnomalyLocationsCollapsed(!anomalyLocationsCollapsed)}
                            >
                                <p className="text-xs font-bold flex items-center gap-1.5">
                                    <ChevronDown className={clsx("size-3 transition-transform", isAnomaly ? layerColor.text : "text-white/40", anomalyLocationsCollapsed && "-rotate-90")} />
                                    <MapPin className={clsx("size-3", isAnomaly ? layerColor.text : "text-white/40")} />
                                    <span className={isAnomaly ? layerColor.text : "text-white/40"}>
                                        Detected Anomaly Locations
                                    </span>
                                </p>
                                {data.anomaly_points && data.anomaly_points.length > 0 && (
                                    <span className={clsx("text-[10px] px-1.5 py-0.5 rounded font-bold", layerColor.accent, layerColor.text)}>
                                        {data.anomaly_points.length} points
                                    </span>
                                )}
                            </div>
                            
                            {!anomalyLocationsCollapsed && (
                                data.anomaly_points && data.anomaly_points.length > 0 ? (
                                    <div className="max-h-40 overflow-y-auto space-y-1.5 pr-1">
                                        {data.anomaly_points.map((pt: any, idx: number) => (
                                            <div 
                                                key={idx} 
                                                className={clsx(
                                                    "p-2.5 rounded-lg border-l-2 text-[10px] transition-all",
                                                    layerColor.bg, layerColor.border,
                                                    onFlyTo && "cursor-pointer hover:brightness-125 hover:scale-[1.02]"
                                                )}
                                                onClick={() => onFlyTo?.(pt.lat, pt.lon, 14)}
                                            >
                                                <div className="flex items-center justify-between">
                                                    <div className="flex items-center gap-2">
                                                        <span className={clsx("w-5 h-5 rounded-full flex items-center justify-center font-bold text-[9px]", layerColor.accent, layerColor.text)}>
                                                            {idx + 1}
                                                        </span>
                                                        <span className="font-mono text-white/80">
                                                            {new Date(pt.timestamp * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
                                                        </span>
                                                    </div>
                                                    <div className="flex items-center gap-1">
                                                        <span className="text-white/40">Score:</span>
                                                        <span className={clsx("font-mono font-bold", layerColor.text)}>{pt.point_score.toFixed(4)}</span>
                                                    </div>
                                                </div>
                                                <div className="flex items-center justify-between mt-1.5 pl-7">
                                                    <span className="text-white/50 font-mono text-[9px]">
                                                        {pt.lat.toFixed(4)}°, {pt.lon.toFixed(4)}°
                                                    </span>
                                                    <span className="text-white/30 text-[9px]">
                                                        reconstruction error
                                                    </span>
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                ) : (
                                    <div className="text-[10px] text-white/30 italic text-center py-2 bg-white/5 rounded">
                                        {isAnomaly ? "No specific points identified" : "No anomalies detected"}
                                    </div>
                                )
                            )}
                         </div>
                    </>
                )}
            </div>
        </div>
    );
};

export const ReportPanel: React.FC<ReportPanelProps> = ({ anomaly, onClose, className, mode, onFlyTo }) => {
    const [feedbackStatus, setFeedbackStatus] = useState<'idle' | 'submitting' | 'success' | 'error'>('idle');
    const [comment, setComment] = useState('');
    const [copied, setCopied] = useState(false);
    const [frStatus, _setFrStatus] = useState<'checking' | 'valid' | 'invalid'>('checking');
    const [showReplay, setShowReplay] = useState(false);
    const [resolvedCallsigns, setResolvedCallsigns] = useState<Record<string, string>>({});
    
    // Rule selection state
    const [rules, setRules] = useState<Rule[]>([]);
    const [selectedRuleId, setSelectedRuleId] = useState<number | null | 'other'>(null);
    const [otherDetails, setOtherDetails] = useState('');
    const [ruleError, setRuleError] = useState(false);
    const [showRuleSelector, setShowRuleSelector] = useState(false);
    
    // Actions dropdown state
    const [showActions, setShowActions] = useState(false);
    const actionsRef = useRef<HTMLDivElement>(null);

    // Close actions menu when clicking outside
    useEffect(() => {
        const handleClickOutside = (event: MouseEvent) => {
            if (actionsRef.current && !actionsRef.current.contains(event.target as Node)) {
                setShowActions(false);
            }
        };
        document.addEventListener('mousedown', handleClickOutside);
        return () => {
            document.removeEventListener('mousedown', handleClickOutside);
        };
    }, []);
    
    // Re-analysis state
    const [isReanalyzing, setIsReanalyzing] = useState(false);
    const [localAnomaly, setLocalAnomaly] = useState<AnomalyReport | null>(anomaly);

    // Sync prop to local state when prop changes
    useEffect(() => {
        setLocalAnomaly(anomaly);
    }, [anomaly]);
    
    // Fetch available rules on mount
    useEffect(() => {
        fetchRules().then(setRules).catch(console.error);
    }, []);

    useEffect(() => {
        if (!localAnomaly) return;

        // Find missing callsigns for proximity rules
        const checkAndFetchCallsigns = async () => {
            const rules = localAnomaly.full_report?.matched_rules || 
                          localAnomaly.full_report?.layer_1_rules?.report?.matched_rules || [];
            
            const missingIds = new Set<string>();
            
            rules.forEach((rule: any) => {
                if (rule.id === 4 && rule.details?.events) {
                     rule.details.events.forEach((ev: any) => {
                         if (!ev.other_callsign && ev.other_flight && !resolvedCallsigns[ev.other_flight]) {
                             missingIds.add(ev.other_flight);
                         }
                     });
                }
            });

            if (missingIds.size > 0) {
                const newResolved: Record<string, string> = {};
                await Promise.all(Array.from(missingIds).map(async (fid) => {
                    const callsign = await fetchCallsignFromResearch(fid);
                    if (callsign) {
                        newResolved[fid] = callsign;
                    }
                }));
                
                if (Object.keys(newResolved).length > 0) {
                    setResolvedCallsigns(prev => ({ ...prev, ...newResolved }));
                }
            }
        };

        checkAndFetchCallsigns();
    }, [localAnomaly]);

    const frUrl = React.useMemo(() => {
        if (!localAnomaly?.callsign || !localAnomaly?.flight_id) return '';
        
        let callsignForUrl = localAnomaly.callsign;
        const upperCallsign = callsignForUrl.toUpperCase();

        if (upperCallsign.startsWith('RJA')) {
            callsignForUrl = 'RJ' + callsignForUrl.substring(3);
        } else if (upperCallsign.startsWith('ELY')) {
            callsignForUrl = 'LY' + callsignForUrl.substring(3);
        }else if (upperCallsign.startsWith('ISR')) {
            callsignForUrl = '6H' + callsignForUrl.substring(4);
        }
        
        return `https://www.flightradar24.com/data/flights/${callsignForUrl}#${localAnomaly.flight_id}`;
    }, [localAnomaly?.callsign, localAnomaly?.flight_id]);



    if (!localAnomaly) return null;

    const handleCopy = (text: string) => {
        navigator.clipboard.writeText(text);
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
    };

    const handleReanalyze = async () => {
        if (!localAnomaly) return;
        setIsReanalyzing(true);
        try {
            const updatedReport = await reanalyzeFeedbackFlight(localAnomaly.flight_id);
            // Merge the updated report with existing local state (preserving feedback IDs etc if not returned)
            setLocalAnomaly(prev => prev ? ({
                ...prev,
                ...updatedReport,
                // Ensure we keep the feedback context if the API didn't return it fully populated
                feedback_id: prev.feedback_id,
                user_label: prev.user_label
            }) : updatedReport);
        } catch (error) {
            console.error("Re-analysis failed:", error);
        } finally {
            setIsReanalyzing(false);
        }
    };


    const handleFeedback = async (isAnomaly: boolean) => {
        // If marking as anomaly, validate rule selection
        if (isAnomaly) {
            if (selectedRuleId === null) {
                setRuleError(true);
                return;
            }
            if (selectedRuleId === 'other' && !otherDetails.trim()) {
                setRuleError(true);
                return;
            }
        }
        
        setRuleError(false);
        setFeedbackStatus('submitting');
        try {
            await submitFeedback({
                flightId: localAnomaly.flight_id,
                isAnomaly,
                comments: comment,
                ruleId: isAnomaly ? (selectedRuleId === 'other' ? null : selectedRuleId as number) : undefined,
                otherDetails: isAnomaly && selectedRuleId === 'other' ? otherDetails : undefined
            });
            setFeedbackStatus('success');
            setComment('');
            setSelectedRuleId(null);
            setOtherDetails('');
            
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

    const report = localAnomaly.full_report || {};
    const summary = report.summary || {};

    const getConfidenceColor = (score: number) => {
        if (score > 85) return "text-red-500";
        if (score > 70) return "text-purple-500";
        if (score > 20) return "text-yellow-500";
        return "text-pink-500";
    };

    const confidenceColor = getConfidenceColor(summary.confidence_score || 0);

    // Extract secondary flight IDs for proximity rule (ID 4)
    const getSecondaryFlightIds = () => {
        // Look in both the main report and layer 1 rules
        const rules = localAnomaly.full_report?.matched_rules || 
                     localAnomaly.full_report?.layer_1_rules?.report?.matched_rules || [];
                     
        const proximityRule = rules.find((r: any) => r.id === 4);
        
        if (!proximityRule?.details?.events) return [];
        return proximityRule.details.events
            .map((e: any) => e.other_flight)
            .filter((id: string) => id && id !== localAnomaly.flight_id);
    };

    const getReplayEvents = (): ReplayEvent[] => {
        const events: ReplayEvent[] = [];
        const rules = localAnomaly.full_report?.matched_rules || 
                     localAnomaly.full_report?.layer_1_rules?.report?.matched_rules || [];

        rules.forEach((rule: any) => {
            if (rule.id === 4 && rule.details?.events) {
                // Dangerous Proximity
                rule.details.events.forEach((ev: any) => {
                    events.push({
                        timestamp: ev.timestamp,
                        type: 'proximity',
                        description: `Conflict with ${ev.other_callsign || ev.other_flight}. Dist: ${ev.distance_nm} NM, Alt Diff: ${ev.altitude_diff_ft} ft`,
                        // Lat/Lon might not be in the event record for proximity, but timestamp is key
                    });
                });
            } else if (rule.id === 11 && rule.details?.deviations) {
                // Path Deviation
                rule.details.deviations.forEach((dev: any) => {
                    events.push({
                        timestamp: dev.timestamp,
                        type: 'deviation',
                        description: `Path deviation: ${dev.dist_nm} NM off course.`,
                        lat: dev.lat,
                        lon: dev.lon
                    });
                });
            }
        });

        // Add ML model anomaly points
        const mlLayers = [
            { key: 'layer_3_deep_dense', name: 'Deep Dense' },
            { key: 'layer_4_deep_cnn', name: 'Deep CNN' },
            { key: 'layer_5_transformer', name: 'Transformer' },
            { key: 'layer_6_hybrid', name: 'Hybrid' }
        ];

        mlLayers.forEach(({ key, name }) => {
            const layerData = localAnomaly.full_report?.[key];
            if (layerData?.anomaly_points && layerData.is_anomaly) {
                // Only add top 2 points from each model to avoid clutter
                layerData.anomaly_points.slice(0, 2).forEach((pt: any, idx: number) => {
                    events.push({
                        timestamp: pt.timestamp,
                        type: 'ml_anomaly',
                        description: `${name} detected anomaly #${idx + 1} (score: ${pt.point_score.toFixed(4)})`,
                        lat: pt.lat,
                        lon: pt.lon
                    });
                });
            }
        });

        // Also add the main anomaly timestamp as a generic event if not covered
        if (!events.find(e => Math.abs(e.timestamp - localAnomaly.timestamp) < 5)) {
             events.push({
                timestamp: localAnomaly.timestamp,
                type: 'other',
                description: 'Anomaly Detection Time',
            });
        }

        return events.sort((a, b) => a.timestamp - b.timestamp);
    };

    return (
        <>
        <aside className={clsx("bg-surface rounded-xl flex flex-col h-full overflow-hidden border border-white/5 animate-in slide-in-from-right-4", className || "col-span-3")}>
            {/* Header */}
            <div className="p-4 border-b border-white/10 flex items-center justify-between bg-surface-highlight/50">
                <div>
                    <h3 className="text-white font-bold">Analysis Report</h3>
                    <p className="text-xs text-white/60">{localAnomaly.flight_id}</p>
                    
                    <div className="mt-3 flex flex-col gap-2">
                        {localAnomaly.callsign && (
                            <p className="text-[10px] text-pink-300 mb-0.5 animate-pulse font-medium">
                                ✨ click me to copy
                            </p>
                        )}
                        <div className="flex items-center gap-2 flex-wrap">
                            {localAnomaly.callsign && (
                                <button 
                                    onClick={() => handleCopy(localAnomaly.callsign!)}
                                    className={clsx(
                                        "text-sm font-mono font-bold px-3 py-1 rounded border transition-all duration-200",
                                        copied 
                                            ? "bg-green-500/20 text-green-300 border-green-500/30" 
                                            : "bg-white/10 text-white hover:bg-white/20 border-white/10 hover:border-white/30"
                                    )}
                                >
                                    {copied ? "Copied!" : localAnomaly.callsign}
                                </button>
                            )}

                            {/* Actions Dropdown */}
                            <div className="relative" ref={actionsRef}>
                                <button
                                    onClick={() => setShowActions(!showActions)}
                                    className="text-sm font-mono font-bold px-3 py-1 rounded border bg-white/10 text-white hover:bg-white/20 border-white/10 hover:border-white/30 transition-all duration-200 flex items-center gap-1"
                                >
                                    <span>Follow Up</span>
                                    <ChevronDown className="size-3" />
                                </button>

                                {showActions && (
                                    <div className="absolute top-full left-0 mt-2 w-52 bg-surface/95 border border-white/10 rounded-lg shadow-xl z-50 overflow-hidden backdrop-blur-md animate-in fade-in slide-in-from-top-2 duration-200">
                                        <div className="p-1.5 space-y-1">
                                            {localAnomaly.callsign && (
                                                 <a 
                                                    href={frUrl}
                                                    target="_blank"
                                                    rel="noopener noreferrer"
                                                    className={clsx(
                                                        "flex items-center gap-3 w-full px-3 py-2.5 text-sm text-left rounded-md transition-all duration-200 no-underline group",
                                                        "hover:bg-green-500/10 hover:text-green-300",
                                                        frStatus === 'valid' ? "text-green-400" : "text-white/80"
                                                    )}
                                                >
                                                    <div className="p-1.5 rounded bg-white/5 group-hover:bg-green-500/20 transition-colors">
                                                        <ExternalLink className="size-3.5 group-hover:scale-110 transition-transform" />
                                                    </div>
                                                    <span className="font-medium">Open in FR24</span>
                                                </a>
                                            )}
                                            
                                            <button
                                                onClick={() => {
                                                    setShowReplay(true);
                                                    setShowActions(false);
                                                }}
                                                className={clsx(
                                                    "flex items-center gap-3 w-full px-3 py-2.5 text-sm text-left rounded-md transition-all duration-200 group",
                                                    "text-white/80 hover:text-blue-300 hover:bg-blue-500/10"
                                                )}
                                            >
                                                <div className="p-1.5 rounded bg-white/5 group-hover:bg-blue-500/20 transition-colors">
                                                    <PlayCircle className="size-3.5 group-hover:scale-110 transition-transform" />
                                                </div>
                                                <span className="font-medium">Replay Flight</span>
                                            </button>

                                            {mode === 'feedback' && (
                                                <button
                                                    onClick={() => {
                                                        handleReanalyze();
                                                        setShowActions(false);
                                                    }}
                                                    disabled={isReanalyzing}
                                                    className={clsx(
                                                        "flex items-center gap-3 w-full px-3 py-2.5 text-sm text-left rounded-md transition-all duration-200 group disabled:opacity-50",
                                                        "text-white/80 hover:text-purple-300 hover:bg-purple-500/10"
                                                    )}
                                                >
                                                    <div className="p-1.5 rounded bg-white/5 group-hover:bg-purple-500/20 transition-colors">
                                                        {isReanalyzing ? (
                                                            <Loader2 className="size-3.5 animate-spin" /> 
                                                        ) : (
                                                            <RefreshCw className="size-3.5 group-hover:rotate-180 transition-transform duration-500" />
                                                        )}
                                                    </div>
                                                    <span className="font-medium">
                                                        {isReanalyzing ? "Analyzing..." : "Re-Analyze"}
                                                    </span>
                                                </button>
                                            )}
                                        </div>
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>
                </div>
                <button onClick={onClose} className="text-white/60 hover:text-white p-1 rounded hover:bg-white/10 self-start">
                    <X className="size-5" />
                </button>
            </div>

            {/* Content */}
            <div className="flex-1 overflow-y-auto p-4 space-y-4 relative">
                
                {/* Re-analyzing Overlay */}
                {isReanalyzing && (
                    <div className="absolute inset-0 bg-surface/50 backdrop-blur-sm z-10 flex flex-col items-center justify-center animate-in fade-in duration-200">
                        <div className="bg-surface border border-white/10 p-6 rounded-xl shadow-2xl flex flex-col items-center gap-4">
                            <div className="relative">
                                <div className="absolute inset-0 bg-purple-500/30 blur-xl rounded-full animate-pulse"></div>
                                <Loader2 className="size-10 text-purple-400 animate-spin relative z-10" />
                            </div>
                            <div className="text-center">
                                <p className="font-bold text-white text-lg">Re-analyzing Flight</p>
                                <p className="text-sm text-white/60">Running anomaly detection pipeline...</p>
                            </div>
                        </div>
                    </div>
                )}

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
                            Detected At: <span className="font-mono">{new Date(localAnomaly.timestamp * 1000).toLocaleString()}</span>
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
                        resolvedCallsigns={resolvedCallsigns}
                    />
                    
                    <LayerCard 
                        title="Layer 2: XGBoost" 
                        data={report.layer_2_xgboost} 
                        type="model"
                        onFlyTo={onFlyTo}
                    />

                    <LayerCard 
                        title="Layer 3: Deep Dense Autoencoder" 
                        data={report.layer_3_deep_dense} 
                        type="model"
                        onFlyTo={onFlyTo}
                    />

                    <LayerCard 
                        title="Layer 4: Deep CNN" 
                        data={report.layer_4_deep_cnn} 
                        type="model"
                        onFlyTo={onFlyTo}
                    />

                    <LayerCard 
                        title="Layer 5: Transformer" 
                        data={report.layer_5_transformer} 
                        type="model"
                        onFlyTo={onFlyTo}
                    />

                    <LayerCard 
                        title="Layer 6: Hybrid CNN-Transformer" 
                        data={report.layer_6_hybrid} 
                        type="model"
                        onFlyTo={onFlyTo} 
                    />
                </div>

                {/* Feedback Section */}
                <div className="rounded-xl p-4 border shadow-lg mt-4" style={{
                    background: 'rgb(var(--color-surface) / 0.8)',
                    borderColor: 'rgb(var(--color-primary) / 0.2)'
                }}>
                    <div className="flex items-center gap-2 mb-4">
                        <div className="h-8 w-1 rounded-full" style={{
                            background: 'rgb(var(--color-primary))'
                        }}></div>
                        <p className="text-sm font-bold uppercase tracking-wide" style={{
                            color: 'rgb(var(--color-text))'
                        }}>Human Feedback</p>
                    </div>
                    
                    {feedbackStatus === 'success' ? (
                        <div className="rounded-xl p-4 border-2 animate-in fade-in slide-in-from-top-2 shadow-lg" style={{
                            background: 'rgb(34 197 94 / 0.15)',
                            borderColor: 'rgb(34 197 94 / 0.4)'
                        }}>
                            <div className="flex items-center gap-3">
                                <div className="rounded-full p-2" style={{
                                    background: 'rgb(34 197 94 / 0.3)'
                                }}>
                                    <CheckCircle className="size-6" style={{
                                        color: 'rgb(134 239 172)'
                                    }} />
                                </div>
                                <div>
                                    <p className="font-bold" style={{
                                        color: 'rgb(134 239 172)'
                                    }}>Feedback Submitted!</p>
                                    <p className="text-xs mt-0.5" style={{
                                        color: 'rgb(var(--color-text-muted))'
                                    }}>Thank you for helping improve our system</p>
                                </div>
                            </div>
                        </div>
                    ) : (
                        <div className="space-y-4">
                            <div className="rounded-lg p-3 border-l-4" style={{
                                background: 'rgb(var(--color-primary) / 0.1)',
                                borderColor: 'rgb(var(--color-primary) / 0.5)'
                            }}>
                                <p className="text-sm font-medium" style={{
                                    color: 'rgb(var(--color-text))'
                                }}>Is this actually an anomaly?</p>
                                <p className="text-xs mt-1" style={{
                                    color: 'rgb(var(--color-text-muted))'
                                }}>Your feedback helps train our AI models</p>
                            </div>
                            
                            {/* Rule Selection Button - Required for anomaly */}
                            <div className="space-y-2">
                                <label className="text-xs font-medium flex items-center gap-1" style={{
                                    color: 'rgb(var(--color-text) / 0.9)'
                                }}>
                                    Which rule triggered this anomaly?
                                    <span className="text-red-400 text-sm">*</span>
                                </label>
                                
                                <button
                                    type="button"
                                    onClick={() => setShowRuleSelector(true)}
                                    className="w-full border-2 rounded-lg p-4 transition-all flex items-center justify-between group"
                                    style={{
                                        background: selectedRuleId !== null 
                                            ? 'rgb(var(--color-primary) / 0.1)' 
                                            : 'rgb(var(--color-background) / 0.5)',
                                        borderColor: ruleError && selectedRuleId === null 
                                            ? 'rgb(239 68 68 / 0.5)' 
                                            : selectedRuleId !== null 
                                                ? 'rgb(var(--color-primary) / 0.5)'
                                                : 'rgb(var(--color-border) / 0.3)',
                                    }}
                                >
                                    <div className="flex items-center gap-3">
                                        {selectedRuleId !== null && selectedRuleId !== 'other' ? (
                                            <>
                                                {React.createElement(getRuleIcon(selectedRuleId as number), { 
                                                    className: "size-5",
                                                    style: { color: 'rgb(var(--color-primary))' }
                                                })}
                                                <div className="text-left">
                                                    <p className="font-medium" style={{ color: 'rgb(var(--color-text))' }}>
                                                        {rules.find(r => r.id === selectedRuleId)?.name}
                                                    </p>
                                                    <p className="text-xs" style={{ color: 'rgb(var(--color-text-muted))' }}>
                                                        {rules.find(r => r.id === selectedRuleId)?.description}
                                                    </p>
                                                </div>
                                            </>
                                        ) : selectedRuleId === 'other' ? (
                                            <>
                                                <AlertTriangle className="size-5 text-yellow-400" />
                                                <div className="text-left">
                                                    <p className="font-medium" style={{ color: 'rgb(var(--color-text))' }}>
                                                        Other / Custom Anomaly
                                                    </p>
                                                    <p className="text-xs" style={{ color: 'rgb(var(--color-text-muted))' }}>
                                                        Custom anomaly type
                                                    </p>
                                                </div>
                                            </>
                                        ) : (
                                            <div className="text-left">
                                                <p className="font-medium" style={{ color: 'rgb(var(--color-text-muted))' }}>
                                                    Select a rule...
                                                </p>
                                                <p className="text-xs" style={{ color: 'rgb(var(--color-text-muted) / 0.6)' }}>
                                                    Click to choose from available rules
                                                </p>
                                            </div>
                                        )}
                                    </div>
                                    <Navigation className="size-5 group-hover:translate-x-1 transition-transform" style={{
                                        color: 'rgb(var(--color-text-muted))'
                                    }} />
                                </button>
                                
                                {ruleError && selectedRuleId === null && (
                                    <div className="flex items-center gap-1.5 text-xs text-red-400 animate-in slide-in-from-top-1">
                                        <AlertTriangle className="size-3" />
                                        <span>Please select which rule applies to submit feedback</span>
                                    </div>
                                )}
                            </div>
                            
                            {/* Other Details Textbox - Shows when "Other" is selected */}
                            {selectedRuleId === 'other' && (
                                <div className="space-y-2 animate-in slide-in-from-top-2 duration-300">
                                    <div className="rounded-lg p-3 border border-yellow-500/30" style={{
                                        background: 'rgb(234 179 8 / 0.1)'
                                    }}>
                                        <label className="text-xs font-medium flex items-center gap-1.5 mb-2" style={{
                                            color: 'rgb(250 204 21)'
                                        }}>
                                            <AlertTriangle className="size-3.5" />
                                            Describe the custom anomaly type
                                            <span className="text-red-400 text-sm">*</span>
                                        </label>
                                        <textarea
                                            value={otherDetails}
                                            onChange={(e) => {
                                                setOtherDetails(e.target.value);
                                                setRuleError(false);
                                            }}
                                            placeholder="E.g., Unusual communication pattern, unexpected holding pattern, suspicious route deviation, etc..."
                                            rows={4}
                                            style={{
                                                background: 'rgb(var(--color-background) / 0.5)',
                                                borderColor: ruleError && !otherDetails.trim() 
                                                    ? 'rgb(239 68 68 / 0.5)' 
                                                    : 'rgb(234 179 8 / 0.3)',
                                                color: 'rgb(var(--color-text))'
                                            }}
                                            className={clsx(
                                                "w-full border rounded-lg p-3 text-sm placeholder:opacity-50 focus:outline-none focus:ring-2 focus:ring-yellow-500/50 resize-none transition-all",
                                                ruleError && !otherDetails.trim() && "ring-2 ring-red-500/20"
                                            )}
                                        />
                                        <p className="text-[10px] mt-2 italic" style={{
                                            color: 'rgb(var(--color-text-muted))'
                                        }}>
                                            💡 Be specific about what makes this flight anomalous
                                        </p>
                                    </div>
                                    {ruleError && !otherDetails.trim() && (
                                        <div className="flex items-center gap-1.5 text-xs text-red-400 animate-in slide-in-from-top-1">
                                            <AlertTriangle className="size-3" />
                                            <span>Please describe the anomaly type to continue</span>
                                        </div>
                                    )}
                                </div>
                            )}
                            
                            <div className="flex gap-3">
                                <button
                                    onClick={() => handleFeedback(true)}
                                    disabled={feedbackStatus === 'submitting'}
                                    className="flex-1 flex flex-col items-center justify-center gap-2 p-4 rounded-lg border-2 font-medium transition-all disabled:opacity-50 disabled:cursor-not-allowed shadow-lg active:scale-95 hover:shadow-xl"
                                    style={{
                                        background: 'rgb(239 68 68 / 0.15)',
                                        borderColor: 'rgb(239 68 68 / 0.5)',
                                        color: 'rgb(252 165 165)'
                                    }}
                                    onMouseEnter={(e) => {
                                        e.currentTarget.style.background = 'rgb(239 68 68 / 0.25)';
                                        e.currentTarget.style.borderColor = 'rgb(239 68 68 / 0.7)';
                                    }}
                                    onMouseLeave={(e) => {
                                        e.currentTarget.style.background = 'rgb(239 68 68 / 0.15)';
                                        e.currentTarget.style.borderColor = 'rgb(239 68 68 / 0.5)';
                                    }}
                                >
                                    <AlertTriangle className="size-5" />
                                    <span className="text-sm">Yes, Anomaly</span>
                                </button>
                                <button
                                    onClick={() => handleFeedback(false)}
                                    disabled={feedbackStatus === 'submitting'}
                                    className="flex-1 flex flex-col items-center justify-center gap-2 p-4 rounded-lg border-2 font-medium transition-all disabled:opacity-50 disabled:cursor-not-allowed shadow-lg active:scale-95 hover:shadow-xl"
                                    style={{
                                        background: 'rgb(34 197 94 / 0.15)',
                                        borderColor: 'rgb(34 197 94 / 0.5)',
                                        color: 'rgb(134 239 172)'
                                    }}
                                    onMouseEnter={(e) => {
                                        e.currentTarget.style.background = 'rgb(34 197 94 / 0.25)';
                                        e.currentTarget.style.borderColor = 'rgb(34 197 94 / 0.7)';
                                    }}
                                    onMouseLeave={(e) => {
                                        e.currentTarget.style.background = 'rgb(34 197 94 / 0.15)';
                                        e.currentTarget.style.borderColor = 'rgb(34 197 94 / 0.5)';
                                    }}
                                >
                                    <ThumbsUp className="size-5" />
                                    <span className="text-sm">No, Normal</span>
                                </button>
                            </div>

                            <div className="relative">
                                <label className="text-xs mb-1.5 block" style={{
                                    color: 'rgb(var(--color-text-muted))'
                                }}>Additional Comments (Optional)</label>
                                <input
                                    type="text"
                                    value={comment}
                                    onChange={(e) => setComment(e.target.value)}
                                    placeholder="Add any additional notes or context..."
                                    style={{
                                        background: 'rgb(var(--color-background) / 0.5)',
                                        borderColor: 'rgb(var(--color-border) / 0.3)',
                                        color: 'rgb(var(--color-text))'
                                    }}
                                    className="w-full border rounded-lg p-3 text-sm placeholder:opacity-40 focus:outline-none focus:ring-2 transition-all"
                                    onFocus={(e) => {
                                        e.currentTarget.style.borderColor = 'rgb(var(--color-primary) / 0.5)';
                                    }}
                                    onBlur={(e) => {
                                        e.currentTarget.style.borderColor = 'rgb(var(--color-border) / 0.3)';
                                    }}
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

        {showReplay && (
            <ReplayModal 
                mainFlightId={localAnomaly.flight_id}
                secondaryFlightIds={getSecondaryFlightIds()}
                events={getReplayEvents()}
                onClose={() => setShowReplay(false)} 
            />
        )}
        
        {/* Floating Rule Circles */}
        {showRuleSelector && (
            <div 
                className="fixed inset-0 z-50 flex items-end justify-center pb-12"
                onClick={() => setShowRuleSelector(false)}
            >
                {/* Backdrop */}
                <div className="absolute inset-0 bg-black/70  animate-in fade-in" />
                
                {/* Floating Circles Container */}
                <div 
                    className="relative w-full max-w-7xl px-8 animate-in slide-in-from-bottom-12 duration-500"
                    onClick={(e) => e.stopPropagation()}
                >
                    {/* Close Button */}
                    <button
                        onClick={() => setShowRuleSelector(false)}
                        className="absolute -top-16 right-8 p-3 rounded-full transition-all hover:scale-110 hover:rotate-90"
                        style={{
                            background: 'rgb(var(--color-surface) / 0.9)',
                            color: 'rgb(var(--color-text))',
                            backdropFilter: 'blur(10px)'
                        }}
                    >
                        <X className="size-6" />
                    </button>
                    
                    {/* Title */}
                    <div className="text-center mb-8">
                        <h3 className="text-2xl font-bold mb-2" style={{ 
                            color: 'rgb(var(--color-text))',
                            textShadow: '0 2px 10px rgba(0,0,0,0.5)'
                        }}>
                            Select Rule Type
                        </h3>
                        <p className="text-sm" style={{ 
                            color: 'rgb(var(--color-text-muted))',
                            textShadow: '0 1px 5px rgba(0,0,0,0.5)'
                        }}>
                            Click on the rule that best describes this anomaly
                        </p>
                    </div>
                    
                    {/* Floating Rules */}
                    <div className="space-y-6">
                        {/* Emergency & Safety */}
                        <div className="flex flex-wrap gap-6 justify-center items-center">
                                {rules.filter(r => 
                                    r.name.toLowerCase().includes('squawk') || 
                                    r.name.toLowerCase().includes('proximity') ||
                                    (r.name.toLowerCase().includes('altitude') && r.name.toLowerCase().includes('low'))
                                ).map((rule, idx) => {
                                    const Icon = getRuleIcon(rule.id);
                                    const isSelected = selectedRuleId === rule.id;
                                    return (
                                        <button
                                            key={rule.id}
                                            onClick={() => {
                                                setSelectedRuleId(rule.id);
                                                setRuleError(false);
                                                setShowRuleSelector(false);
                                            }}
                                            className="flex flex-col items-center gap-2 transition-all hover:scale-110 group relative opacity-0"
                                            style={{
                                                animation: 'jumpIn 0.5s cubic-bezier(0.68, -0.55, 0.265, 1.55) forwards',
                                                animationDelay: `${idx * 0.1}s`
                                            }}
                                        >
                                            <div 
                                                className="w-20 h-20 rounded-full flex items-center justify-center transition-all group-hover:shadow-2xl"
                                                style={{
                                                    background: isSelected 
                                                        ? 'linear-gradient(135deg, rgb(239 68 68 / 0.4), rgb(220 38 38 / 0.6))' 
                                                        : 'linear-gradient(135deg, rgb(239 68 68 / 0.3), rgb(220 38 38 / 0.2))',
                                                    border: isSelected ? '4px solid rgb(239 68 68)' : '3px solid rgb(239 68 68 / 0.5)',
                                                    boxShadow: isSelected 
                                                        ? '0 10px 40px rgb(239 68 68 / 0.6), inset 0 2px 10px rgb(255 255 255 / 0.1)' 
                                                        : '0 5px 20px rgb(239 68 68 / 0.3)',
                                                    backdropFilter: 'blur(10px)'
                                                }}
                                            >
                                                <Icon className="size-8 text-red-300 group-hover:scale-110 transition-transform" />
                                                {isSelected && (
                                                    <CheckCircle className="size-5 text-red-300 absolute -top-1 -right-1 animate-in zoom-in" />
                                                )}
                                            </div>
                                            <div className="text-center max-w-[100px]">
                                                <p className="text-xs font-bold" style={{ 
                                                    color: 'rgb(var(--color-text))',
                                                    textShadow: '0 2px 5px rgba(0,0,0,0.5)'
                                                }}>
                                                    {rule.name}
                                                </p>
                                            </div>
                                        </button>
                                    );
                                })}
                        
                        {/* Flight Path & Navigation */}
                                {rules.filter(r => 
                                    (r.name.toLowerCase().includes('altitude') && !r.name.toLowerCase().includes('low')) ||
                                    r.name.toLowerCase().includes('turn') ||
                                    r.name.toLowerCase().includes('course') ||
                                    r.name.toLowerCase().includes('off')
                                ).map((rule, idx) => {
                                    const Icon = getRuleIcon(rule.id);
                                    const isSelected = selectedRuleId === rule.id;
                                    const baseDelay = 3; // Start after red circles
                                    return (
                                        <button
                                            key={rule.id}
                                            onClick={() => {
                                                setSelectedRuleId(rule.id);
                                                setRuleError(false);
                                                setShowRuleSelector(false);
                                            }}
                                            className="flex flex-col items-center gap-2 transition-all hover:scale-110 group relative opacity-0"
                                            style={{
                                                animation: 'jumpIn 0.5s cubic-bezier(0.68, -0.55, 0.265, 1.55) forwards',
                                                animationDelay: `${(baseDelay + idx) * 0.1}s`
                                            }}
                                        >
                                            <div 
                                                className="w-20 h-20 rounded-full flex items-center justify-center transition-all group-hover:shadow-2xl"
                                                style={{
                                                    background: isSelected 
                                                        ? 'linear-gradient(135deg, rgb(59 130 246 / 0.4), rgb(37 99 235 / 0.6))' 
                                                        : 'linear-gradient(135deg, rgb(59 130 246 / 0.3), rgb(37 99 235 / 0.2))',
                                                    border: isSelected ? '4px solid rgb(59 130 246)' : '3px solid rgb(59 130 246 / 0.5)',
                                                    boxShadow: isSelected 
                                                        ? '0 10px 40px rgb(59 130 246 / 0.6), inset 0 2px 10px rgb(255 255 255 / 0.1)' 
                                                        : '0 5px 20px rgb(59 130 246 / 0.3)',
                                                    backdropFilter: 'blur(10px)'
                                                }}
                                            >
                                                <Icon className="size-8 text-blue-300 group-hover:scale-110 transition-transform" />
                                                {isSelected && (
                                                    <CheckCircle className="size-5 text-blue-300 absolute -top-1 -right-1 animate-in zoom-in" />
                                                )}
                                            </div>
                                            <div className="text-center max-w-[100px]">
                                                <p className="text-xs font-bold" style={{ 
                                                    color: 'rgb(var(--color-text))',
                                                    textShadow: '0 2px 5px rgba(0,0,0,0.5)'
                                                }}>
                                                    {rule.name}
                                                </p>
                                            </div>
                                        </button>
                                    );
                                })}
                        
                        {/* Landing & Departure */}
                                {rules.filter(r => 
                                    r.name.toLowerCase().includes('around') ||
                                    r.name.toLowerCase().includes('return') ||
                                    r.name.toLowerCase().includes('diversion') ||
                                    r.name.toLowerCase().includes('landing')
                                ).map((rule, idx) => {
                                    const Icon = getRuleIcon(rule.id);
                                    const isSelected = selectedRuleId === rule.id;
                                    const baseDelay = 7; // Start after blue circles
                                    return (
                                        <button
                                            key={rule.id}
                                            onClick={() => {
                                                setSelectedRuleId(rule.id);
                                                setRuleError(false);
                                                setShowRuleSelector(false);
                                            }}
                                            className="flex flex-col items-center gap-2 transition-all hover:scale-110 group relative opacity-0"
                                            style={{
                                                animation: 'jumpIn 0.5s cubic-bezier(0.68, -0.55, 0.265, 1.55) forwards',
                                                animationDelay: `${(baseDelay + idx) * 0.1}s`
                                            }}
                                        >
                                            <div 
                                                className="w-20 h-20 rounded-full flex items-center justify-center transition-all group-hover:shadow-2xl"
                                                style={{
                                                    background: isSelected 
                                                        ? 'linear-gradient(135deg, rgb(34 197 94 / 0.4), rgb(22 163 74 / 0.6))' 
                                                        : 'linear-gradient(135deg, rgb(34 197 94 / 0.3), rgb(22 163 74 / 0.2))',
                                                    border: isSelected ? '4px solid rgb(34 197 94)' : '3px solid rgb(34 197 94 / 0.5)',
                                                    boxShadow: isSelected 
                                                        ? '0 10px 40px rgb(34 197 94 / 0.6), inset 0 2px 10px rgb(255 255 255 / 0.1)' 
                                                        : '0 5px 20px rgb(34 197 94 / 0.3)',
                                                    backdropFilter: 'blur(10px)'
                                                }}
                                            >
                                                <Icon className="size-8 text-green-300 group-hover:scale-110 transition-transform" />
                                                {isSelected && (
                                                    <CheckCircle className="size-5 text-green-300 absolute -top-1 -right-1 animate-in zoom-in" />
                                                )}
                                            </div>
                                            <div className="text-center max-w-[100px]">
                                                <p className="text-xs font-bold" style={{ 
                                                    color: 'rgb(var(--color-text))',
                                                    textShadow: '0 2px 5px rgba(0,0,0,0.5)'
                                                }}>
                                                    {rule.name}
                                                </p>
                                            </div>
                                        </button>
                                    );
                                })}
                        
                        {/* Technical */}
                                {rules.filter(r => 
                                    r.name.toLowerCase().includes('signal') ||
                                    r.name.toLowerCase().includes('loss') ||
                                    r.name.toLowerCase().includes('communication')
                                ).map((rule, idx) => {
                                    const Icon = getRuleIcon(rule.id);
                                    const isSelected = selectedRuleId === rule.id;
                                    const baseDelay = 11; // Start after green circles
                                    return (
                                        <button
                                            key={rule.id}
                                            onClick={() => {
                                                setSelectedRuleId(rule.id);
                                                setRuleError(false);
                                                setShowRuleSelector(false);
                                            }}
                                            className="flex flex-col items-center gap-2 transition-all hover:scale-110 group relative opacity-0"
                                            style={{
                                                animation: 'jumpIn 0.5s cubic-bezier(0.68, -0.55, 0.265, 1.55) forwards',
                                                animationDelay: `${(baseDelay + idx) * 0.1}s`
                                            }}
                                        >
                                            <div 
                                                className="w-20 h-20 rounded-full flex items-center justify-center transition-all group-hover:shadow-2xl"
                                                style={{
                                                    background: isSelected 
                                                        ? 'linear-gradient(135deg, rgb(168 85 247 / 0.4), rgb(147 51 234 / 0.6))' 
                                                        : 'linear-gradient(135deg, rgb(168 85 247 / 0.3), rgb(147 51 234 / 0.2))',
                                                    border: isSelected ? '4px solid rgb(168 85 247)' : '3px solid rgb(168 85 247 / 0.5)',
                                                    boxShadow: isSelected 
                                                        ? '0 10px 40px rgb(168 85 247 / 0.6), inset 0 2px 10px rgb(255 255 255 / 0.1)' 
                                                        : '0 5px 20px rgb(168 85 247 / 0.3)',
                                                    backdropFilter: 'blur(10px)'
                                                }}
                                            >
                                                <Icon className="size-8 text-purple-300 group-hover:scale-110 transition-transform" />
                                                {isSelected && (
                                                    <CheckCircle className="size-5 text-purple-300 absolute -top-1 -right-1 animate-in zoom-in" />
                                                )}
                                            </div>
                                            <div className="text-center max-w-[100px]">
                                                <p className="text-xs font-bold" style={{ 
                                                    color: 'rgb(var(--color-text))',
                                                    textShadow: '0 2px 5px rgba(0,0,0,0.5)'
                                                }}>
                                                    {rule.name}
                                                </p>
                                            </div>
                                        </button>
                                    );
                                })}
                        
                        {/* Other / Custom */}
                            <button
                                onClick={() => {
                                    setSelectedRuleId('other');
                                    setRuleError(false);
                                    setShowRuleSelector(false);
                                }}
                                className="flex flex-col items-center gap-2 transition-all hover:scale-110 group relative opacity-0"
                                style={{
                                    animation: 'jumpIn 0.5s cubic-bezier(0.68, -0.55, 0.265, 1.55) forwards',
                                    animationDelay: '1.2s'
                                }}
                            >
                                <div 
                                    className="w-20 h-20 rounded-full flex items-center justify-center transition-all group-hover:shadow-2xl"
                                    style={{
                                        background: selectedRuleId === 'other' 
                                            ? 'linear-gradient(135deg, rgb(234 179 8 / 0.4), rgb(202 138 4 / 0.6))' 
                                            : 'linear-gradient(135deg, rgb(234 179 8 / 0.3), rgb(202 138 4 / 0.2))',
                                        border: selectedRuleId === 'other' ? '4px solid rgb(234 179 8)' : '3px solid rgb(234 179 8 / 0.5)',
                                        boxShadow: selectedRuleId === 'other' 
                                            ? '0 10px 40px rgb(234 179 8 / 0.6), inset 0 2px 10px rgb(255 255 255 / 0.1)' 
                                            : '0 5px 20px rgb(234 179 8 / 0.3)',
                                        backdropFilter: 'blur(10px)'
                                    }}
                                >
                                    <AlertTriangle className="size-8 text-yellow-300 group-hover:scale-110 transition-transform" />
                                    {selectedRuleId === 'other' && (
                                        <CheckCircle className="size-5 text-yellow-300 absolute -top-1 -right-1 animate-in zoom-in" />
                                    )}
                                </div>
                                <div className="text-center max-w-[100px]">
                                    <p className="text-xs font-bold" style={{ 
                                        color: 'rgb(var(--color-text))',
                                        textShadow: '0 2px 5px rgba(0,0,0,0.5)'
                                    }}>
                                        Other
                                    </p>
                                </div>
                            </button>
                    </div>
                </div>
            </div>
            </div>
        )}
        </>
    );
};

