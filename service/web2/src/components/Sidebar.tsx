import React, { useState, useEffect, useRef } from 'react';
import { ChevronLeft, ChevronRight, Search, Radio, Filter } from 'lucide-react';
import { fetchLiveAnomalies } from '../api';
import type { AnomalyReport } from '../types';
import clsx from 'clsx';
import { ALERT_AUDIO_SRC, SOUND_COOLDOWN_MS } from '../constants';

interface SidebarProps {
    onSelectAnomaly: (anomaly: AnomalyReport) => void;
    selectedAnomalyId?: string;
}

export const Sidebar: React.FC<SidebarProps> = ({ onSelectAnomaly, selectedAnomalyId }) => {
    const [mode, setMode] = useState<'historical' | 'realtime'>('historical');
    const [selectedDate, setSelectedDate] = useState(new Date());
    const [anomalies, setAnomalies] = useState<AnomalyReport[]>([]);
    const [loading, setLoading] = useState(false);
    const [filter, setFilter] = useState('');
    
    // Filters
    const [minScore, setMinScore] = useState(0);
    const [selectedTrigger, setSelectedTrigger] = useState('All');
    const [showFilters, setShowFilters] = useState(false);

    const audioRef = useRef<HTMLAudioElement | null>(null);
    const lastSoundTimeRef = useRef(0);

    const triggerOptions = ['All', 'Rules', 'XGBoost', 'DeepDense', 'DeepCNN', 'Transformer'];

    // Realtime tracking
    const lastFetchTimeRef = useRef<number>(0);
    const intervalRef = useRef<any>(null);

    // Effect for fetching anomalies based on mode
    useEffect(() => {
        if (intervalRef.current) {
            clearInterval(intervalRef.current);
            intervalRef.current = null;
        }

        if (mode === 'historical') {
            fetchHistorical();
        } else {
            fetchRealtimeInitial();
            intervalRef.current = setInterval(fetchRealtimeUpdate, 5000);
        }

        return () => {
            if (intervalRef.current) clearInterval(intervalRef.current);
        };
    }, [mode, selectedDate]);

    const fetchHistorical = async () => {
        setLoading(true);
        try {
            const start = new Date(selectedDate);
            start.setHours(0, 0, 0, 0);
            const end = new Date(selectedDate);
            end.setHours(23, 59, 59, 999);

            const data = await fetchLiveAnomalies(
                Math.floor(start.getTime() / 1000),
                Math.floor(end.getTime() / 1000)
            );
            setAnomalies(data);
        } catch (error) {
            console.error("Error fetching historical anomalies:", error);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        const audio = new Audio(ALERT_AUDIO_SRC);
        audio.preload = 'auto';
        audioRef.current = audio;

        return () => {
            audio.pause();
            audioRef.current = null;
        };
    }, []);

    const triggerRealtimeAlert = () => {
        const now = Date.now();
        if (now - lastSoundTimeRef.current < SOUND_COOLDOWN_MS) {
            return;
        }

        lastSoundTimeRef.current = now;
        const audio = audioRef.current;
        if (!audio) return;

        try {
            audio.pause();
            audio.currentTime = 0;
            void audio.play();
        } catch (error) {
            console.warn('Unable to play alert sound', error);
        }
    };

    const fetchRealtimeInitial = async () => {
        setLoading(true);
        setAnomalies([]);
        try {
            const now = Math.floor(Date.now() / 1000);
            const start = now - 3600; 
            
            const data = await fetchLiveAnomalies(start, now);
            setAnomalies(data);
            lastFetchTimeRef.current = now;
        } catch (error) {
            console.error("Error fetching initial realtime data:", error);
        } finally {
            setLoading(false);
        }
    };

    const fetchRealtimeUpdate = async () => {
        try {
            const now = Math.floor(Date.now() / 1000);
            const start = lastFetchTimeRef.current;
            
            if (now <= start) return;

            const newData = await fetchLiveAnomalies(start, now);
            
            if (newData.length > 0) {
                let shouldPlayAlert = false;
                setAnomalies(prev => {
                    const existingIds = new Set(prev.map(a => `${a.flight_id}-${a.timestamp}`));
                    const uniqueNew = newData.filter(a => !existingIds.has(`${a.flight_id}-${a.timestamp}`));
                    if (uniqueNew.length > 0) {
                        shouldPlayAlert = true;
                    }
                    return [...uniqueNew, ...prev].sort((a, b) => b.timestamp - a.timestamp);
                });

                if (shouldPlayAlert) {
                    triggerRealtimeAlert();
                }
            }
            lastFetchTimeRef.current = now;
        } catch (error) {
            console.error("Error polling realtime data:", error);
        }
    };

    const filteredAnomalies = Array.from(
        anomalies.reduce((map, a) => {
            if (!map.has(a.flight_id)) map.set(a.flight_id, a);
            return map;
        }, new Map<string, AnomalyReport>()).values()
    ).filter(a => {
        const score = a.full_report?.summary?.confidence_score || 0;
        const matchesSearch = a.flight_id.toLowerCase().includes(filter.toLowerCase()) ||
            (a.callsign || '').toLowerCase().includes(filter.toLowerCase()) ||
            (a.full_report?.summary?.triggers?.join(' ') || '').toLowerCase().includes(filter.toLowerCase());
        
        const triggers = a.full_report?.summary?.triggers || [];
        const matchesTrigger = selectedTrigger === 'All' || triggers.includes(selectedTrigger);

        const matchesScore = score >= minScore;
        
        return matchesSearch && matchesScore && matchesTrigger;
    });

    const changeDate = (days: number) => {
        const newDate = new Date(selectedDate);
        newDate.setDate(newDate.getDate() + days);
        setSelectedDate(newDate);
    };

    const getConfidenceColor = (score: number) => {
        if (score > 85) return "bg-red-500";
        if (score > 70) return "bg-purple-500";
        if (score > 20) return "bg-yellow-500";
        return "bg-pink-500";
    };

    const getScoreLabel = (score: number) => {
        if (score > 85) return "Critical (>85)";
        if (score > 70) return "High (>70)";
        if (score > 20) return "Medium (>20)";
        return "Low (0-20)";
    };

    return (
        <aside className="col-span-3 flex flex-col gap-6 overflow-y-auto h-full pr-2">
            
            {/* Mode Switcher */}
            <div className="bg-[#2C2F33] rounded-xl p-1 flex">
                <button 
                    onClick={() => setMode('historical')}
                    className={clsx(
                        "flex-1 py-2 rounded-lg text-sm font-bold transition-all",
                        mode === 'historical' ? "bg-primary text-background-dark" : "text-white/60 hover:text-white"
                    )}
                >
                    Historical
                </button>
                <button 
                    onClick={() => setMode('realtime')}
                    className={clsx(
                        "flex-1 py-2 rounded-lg text-sm font-bold transition-all flex items-center justify-center gap-2",
                        mode === 'realtime' ? "bg-red-500 text-white" : "text-white/60 hover:text-white"
                    )}
                >
                    <Radio className={clsx("size-4", mode === 'realtime' && "animate-pulse")} />
                    Realtime
                </button>
            </div>

            {/* Date Filter (Only visible in Historical Mode) */}
            {mode === 'historical' && (
                <div className="bg-[#2C2F33] rounded-xl p-4 flex flex-col gap-4 shrink-0 animate-in fade-in slide-in-from-top-2">
                    <p className="text-white text-base font-bold leading-tight">Filter by Date</p>
                    <div className="flex items-center p-1 justify-between">
                        <button onClick={() => changeDate(-1)} className="text-white/80 hover:text-white p-1 rounded hover:bg-white/10">
                            <ChevronLeft className="size-6" />
                        </button>
                        <p className="text-white text-sm font-bold leading-tight flex-1 text-center">
                            {selectedDate.toLocaleDateString(undefined, { weekday: 'short', year: 'numeric', month: 'long', day: 'numeric' })}
                        </p>
                        <button onClick={() => changeDate(1)} className="text-white/80 hover:text-white p-1 rounded hover:bg-white/10">
                            <ChevronRight className="size-6" />
                        </button>
                    </div>
                </div>
            )}

            {/* Search & List */}
            <div className="bg-[#2C2F33] rounded-xl p-4 flex flex-col gap-4 flex-1 min-h-0">
                {/* Search Bar with Filter Toggle */}
                <div className="flex items-center gap-2">
                    <label className="flex flex-col w-full h-12 flex-1">
                        <div className="flex w-full flex-1 items-stretch rounded-lg h-full bg-background-dark">
                            <div className="text-white/60 flex items-center justify-center pl-4">
                                <Search className="size-5" />
                            </div>
                            <input 
                                className="flex w-full flex-1 bg-transparent text-white focus:outline-none px-4 placeholder:text-white/60 text-sm"
                                placeholder="Search Call sign or Flight ID" 
                                value={filter}
                                onChange={e => setFilter(e.target.value)}
                            />
                        </div>
                    </label>
                    <button 
                        onClick={() => setShowFilters(!showFilters)}
                        className={clsx(
                            "h-12 w-12 flex items-center justify-center rounded-lg transition-colors",
                            showFilters ? "bg-primary text-background-dark" : "bg-background-dark text-white/60 hover:text-white"
                        )}
                    >
                        <Filter className="size-5" />
                    </button>
                </div>

                {/* Advanced Filters */}
                {showFilters && (
                    <div className="bg-background-dark rounded-lg p-3 animate-in slide-in-from-top-2 space-y-4">
                        {/* Confidence Score Filter */}
                        <div>
                            <p className="text-xs text-white/60 font-bold uppercase mb-2">Minimum Confidence Score: {minScore}%</p>
                            <input 
                                type="range" 
                                min="0" 
                                max="100" 
                                value={minScore} 
                                onChange={(e) => setMinScore(Number(e.target.value))}
                                className="w-full h-2 bg-white/10 rounded-lg appearance-none cursor-pointer accent-primary"
                            />
                            <div className="flex justify-between text-[10px] text-white/40 mt-1">
                                <span>0%</span>
                                <span>{getScoreLabel(minScore)}</span>
                                <span>100%</span>
                            </div>
                        </div>

                        {/* Trigger Reason Filter */}
                        <div>
                            <p className="text-xs text-white/60 font-bold uppercase mb-2">Filter by Layer</p>
                            <div className="grid grid-cols-2 gap-2">
                                {triggerOptions.map((option) => (
                                    <button
                                        key={option}
                                        onClick={() => setSelectedTrigger(option)}
                                        className={clsx(
                                            "px-2 py-1.5 rounded-md text-xs font-medium transition-colors",
                                            selectedTrigger === option
                                                ? "bg-primary text-background-dark"
                                                : "bg-white/5 text-white/60 hover:bg-white/10 hover:text-white"
                                        )}
                                    >
                                        {option}
                                    </button>
                                ))}
                            </div>
                        </div>
                    </div>
                )}

                <div className="flex flex-col gap-2 overflow-y-auto pr-2 -mr-2 flex-1">
                    {loading && anomalies.length === 0 ? (
                        <p className="text-white/60 text-center py-4">Loading anomalies...</p>
                    ) : filteredAnomalies.length === 0 ? (
                        <p className="text-white/60 text-center py-4">
                            {mode === 'realtime' ? "No anomalies detected recently." : "No anomalies match criteria."}
                        </p>
                    ) : (
                        filteredAnomalies.map((anomaly) => {
                             // Determine severity color based on confidence score
                             const score = anomaly.full_report?.summary?.confidence_score || 0;
                             const severityColor = getConfidenceColor(score);
                             
                             // Triggers list or type
                             const triggers = anomaly.full_report?.summary?.triggers || [];
                             const type = triggers.length > 0 ? triggers.join(', ') : 'Unknown Anomaly';
                             
                             // Display Title: Callsign > Flight ID
                             const displayTitle = anomaly.callsign || anomaly.flight_id;
                             const subTitle = anomaly.callsign ? `ID: ${anomaly.flight_id}` : '';

                             return (
                                <div 
                                    key={`${anomaly.flight_id}-${anomaly.timestamp}`}
                                    onClick={() => onSelectAnomaly(anomaly)}
                                    className={clsx(
                                        "flex flex-col gap-2 p-3 rounded-lg cursor-pointer transition-colors border",
                                        selectedAnomalyId === anomaly.flight_id 
                                            ? "bg-primary/20 border-primary" 
                                            : "hover:bg-white/5 border-transparent"
                                    )}
                                >
                                    <div className="flex justify-between items-center">
                                        <p className="text-sm font-bold text-white">{displayTitle}</p>
                                        <span className={`h-2.5 w-2.5 rounded-full ${severityColor}`}></span>
                                    </div>
                                    <p className="text-xs text-white/80 truncate">{type}</p>
                                    <div className="flex justify-between items-center text-xs text-white/60">
                                        <span>{subTitle}</span>
                                        <span>{new Date(anomaly.timestamp * 1000).toLocaleTimeString()}</span>
                                    </div>
                                </div>
                             );
                        })
                    )}
                </div>
            </div>
        </aside>
    );
};
