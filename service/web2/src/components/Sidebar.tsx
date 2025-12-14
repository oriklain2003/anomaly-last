import React, { useState, useEffect, useRef } from 'react';
import { ChevronLeft, ChevronRight, Search, Radio, Filter, Beaker, Calendar, List, ArrowLeft, Plane, History, Sparkles } from 'lucide-react';
import { fetchLiveAnomalies, fetchResearchAnomalies, fetchRules, fetchFlightsByRule, fetchFeedbackHistory as apiFetchFeedbackHistory } from '../api';
import type { AnomalyReport } from '../types';
import clsx from 'clsx';
import { ALERT_AUDIO_SRC, SOUND_COOLDOWN_MS } from '../constants';

export type SidebarMode = 'historical' | 'realtime' | 'research' | 'rules' | 'feedback' | 'ai-results';

interface SidebarProps {
    onSelectAnomaly: (anomaly: AnomalyReport) => void;
    selectedAnomalyId?: string;
    mode: SidebarMode;
    setMode: (mode: SidebarMode) => void;
    selectedDate: Date;
    setSelectedDate: (date: Date) => void;
    className?: string;
    aiResultFlights?: AnomalyReport[];
}

const LoadingPlane: React.FC<{ message?: string }> = ({ message = 'Searching anomalies...' }) => (
    <div className="flex flex-col items-center justify-center py-6 text-white/70 gap-3">
        <div className="relative w-16 h-16">
            <div className="absolute inset-0 rounded-full border-4 border-white/10 border-t-primary animate-spin" />
            <div className="absolute inset-2 rounded-full border-2 border-dashed border-primary/60 animate-spin" />
            <div className="absolute inset-0 flex items-center justify-center text-primary">
                <Plane className="w-6 h-6" />
            </div>
        </div>
        <p className="text-sm font-semibold text-white/70">{message}</p>
    </div>
);

export const Sidebar: React.FC<SidebarProps> = ({ 
    onSelectAnomaly, 
    selectedAnomalyId, 
    mode, 
    setMode, 
    selectedDate, 
    setSelectedDate, 
    className,
    aiResultFlights = []
}) => {
    const [anomalies, setAnomalies] = useState<AnomalyReport[]>([]);
    const [loading, setLoading] = useState(false);
    const [filter, setFilter] = useState('');
    
    // Rules State
    const [rules, setRules] = useState<{ id: number; name: string; description: string }[]>([]);
    const [selectedRuleId, setSelectedRuleId] = useState<number | null>(null);
    
    // Filters
    const [minScore, setMinScore] = useState(0);
    const [selectedTrigger, setSelectedTrigger] = useState('All');
    const [selectedLayerCombo, setSelectedLayerCombo] = useState<string[]>([]);
    const [selectedVersion, setSelectedVersion] = useState('All');
    const [showFilters, setShowFilters] = useState(false);
    
    // Feedback Mode Specific Filters
    const [showNormalFeedback, setShowNormalFeedback] = useState(false);

    const audioRef = useRef<HTMLAudioElement | null>(null);
    const lastSoundTimeRef = useRef(0);

    const triggerOptions = ['All', 'Combination', 'Rules', 'XGBoost', 'DeepDense', 'DeepCNN', 'Transformer', 'Hybrid'];
    const versionOptions = ['All', 'v1', 'v2', 'v3', 'v4'];

    // Realtime tracking
    const lastFetchTimeRef = useRef<number>(0);
    const intervalRef = useRef<any>(null);
    const searchAbortRef = useRef<AbortController | null>(null);

    const startNewSearch = () => {
        if (searchAbortRef.current) {
            searchAbortRef.current.abort();
        }
        const controller = new AbortController();
        searchAbortRef.current = controller;
        return controller;
    };

    const finishSearch = (controller: AbortController) => {
        if (searchAbortRef.current === controller) {
            setLoading(false);
            searchAbortRef.current = null;
        }
    };

    // Effect for fetching anomalies based on mode
    useEffect(() => {
        if (intervalRef.current) {
            clearInterval(intervalRef.current);
            intervalRef.current = null;
        }

        if (searchAbortRef.current) {
            searchAbortRef.current.abort();
            searchAbortRef.current = null;
        }

        setAnomalies([]);

        if (mode === 'rules') {
            fetchRulesList();
        } else if (mode === 'feedback') {
            fetchFeedbackHistory();
        } else if (mode === 'ai-results') {
            // AI results are managed externally via props
            setLoading(false);
        } else if (mode === 'historical' || mode === 'research') {
            fetchHistoricalOrResearch();
        } else {
            fetchRealtimeInitial();
            intervalRef.current = setInterval(fetchRealtimeUpdate, 5000);
        }

        return () => {
            if (intervalRef.current) clearInterval(intervalRef.current);
            if (searchAbortRef.current) {
                searchAbortRef.current.abort();
                searchAbortRef.current = null;
            }
        };
    }, [mode, selectedDate]);

    // Effect for fetching flights when a rule is selected
    useEffect(() => {
        if (mode === 'rules' && selectedRuleId !== null) {
            fetchRuleFlights(selectedRuleId);
        }
    }, [mode, selectedRuleId]);

    const fetchRulesList = async () => {
        setLoading(true);
        try {
            const data = await fetchRules();
            setRules(data);
        } catch (error) {
            console.error("Error fetching rules:", error);
        } finally {
            setLoading(false);
        }
    };

    const fetchRuleFlights = async (ruleId: number) => {
        const controller = startNewSearch();
        setLoading(true);
        setAnomalies([]);
        try {
            const data = await fetchFlightsByRule(ruleId, controller.signal);
            if (controller.signal.aborted) return;
            setAnomalies(data);
        } catch (error: any) {
            if (error?.name === 'AbortError') return;
            console.error("Error fetching flights for rule:", error);
            setAnomalies([]);
        } finally {
            finishSearch(controller);
        }
    };

    const fetchHistoricalOrResearch = async () => {
        const controller = startNewSearch();
        setLoading(true);
        setAnomalies([]);
        try {
            const start = new Date(selectedDate);
            start.setHours(0, 0, 0, 0);
            const end = new Date(selectedDate);
            end.setHours(23, 59, 59, 999);

            const apiFunc = mode === 'research' ? fetchResearchAnomalies : fetchLiveAnomalies;

            const data = await apiFunc(
                Math.floor(start.getTime() / 1000),
                Math.floor(end.getTime() / 1000)
            );
            if (controller.signal.aborted) return;
            setAnomalies(data);
        } catch (error: any) {
            if (error?.name === 'AbortError') return;
            console.error("Error fetching data:", error);
            setAnomalies([]);
        } finally {
            finishSearch(controller);
        }
    };

    const fetchFeedbackHistory = async () => {
        const controller = startNewSearch();
        setLoading(true);
        setAnomalies([]);
        try {
            const start = new Date(selectedDate);
            start.setHours(0, 0, 0, 0);
            const end = new Date(selectedDate);
            end.setHours(23, 59, 59, 999);

            const data = await apiFetchFeedbackHistory(
                Math.floor(start.getTime() / 1000),
                Math.floor(end.getTime() / 1000),
                100
            );
            if (controller.signal.aborted) return;
            
            // Data is already in AnomalyReport format from the API
            setAnomalies(data);
        } catch (error: any) {
            if (error?.name === 'AbortError') return;
            console.error("Error fetching feedback history:", error);
            setAnomalies([]);
        } finally {
            finishSearch(controller);
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
        const controller = startNewSearch();
        setLoading(true);
        setAnomalies([]);
        try {
            const now = Math.floor(Date.now() / 1000);
            const start = now - 3600; 
            
            const data = await fetchLiveAnomalies(start, now);
            if (controller.signal.aborted) return;
            setAnomalies(data);
            lastFetchTimeRef.current = now;
        } catch (error: any) {
            if (error?.name === 'AbortError') return;
            console.error("Error fetching initial realtime data:", error);
            setAnomalies([]);
        } finally {
            finishSearch(controller);
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

    // Use AI results when in ai-results mode, otherwise use fetched anomalies
    const sourceAnomalies = mode === 'ai-results' ? aiResultFlights : anomalies;
    
    const filteredAnomalies = Array.from(
        sourceAnomalies.reduce((map, a) => {
            if (!map.has(a.flight_id)) map.set(a.flight_id, a);
            return map;
        }, new Map<string, AnomalyReport>()).values()
    ).filter(a => {
        const score = a.full_report?.summary?.confidence_score || 0;
        const matchesSearch = a.flight_id.toLowerCase().includes(filter.toLowerCase()) ||
            (a.callsign || '').toLowerCase().includes(filter.toLowerCase()) ||
            (a.full_report?.summary?.triggers?.join(' ') || '').toLowerCase().includes(filter.toLowerCase());
        
        const triggers = a.full_report?.summary?.triggers || [];
        const matchesTrigger = selectedTrigger === 'All' 
            ? true 
            : selectedTrigger === 'Combination'
                ? (selectedLayerCombo.length === 0 
                    ? triggers.length > 1 
                    : selectedLayerCombo.every(layer => triggers.includes(layer)))
                : triggers.includes(selectedTrigger);

        const cutoffTimestampV2 = new Date('2025-07-08T20:00:00Z').getTime() / 1000;
        const cutoffTimestampV3 = new Date('2025-07-17T00:00:00Z').getTime() / 1000;
        const cutoffTimestampV4 = new Date('2025-11-01T00:00:00Z').getTime() / 1000;
        
        let version = 'v1';
        if (a.timestamp >= cutoffTimestampV4) {
            version = 'v4';
        } else if (a.timestamp >= cutoffTimestampV3) {
            version = 'v3';
        } else if (a.timestamp >= cutoffTimestampV2) {
            version = 'v2';
        }

        const matchesVersion = selectedVersion === 'All' || version === selectedVersion;

        const matchesScore = score >= minScore;
        
        // In feedback mode, only show confirmed anomalies (user_label = 1) unless showNormalFeedback is true
        const matchesFeedback = mode === 'feedback' 
            ? (showNormalFeedback ? true : (a.user_label === 1 || a.user_label === undefined)) 
            : true;

        return matchesSearch && matchesScore && matchesTrigger && matchesVersion && matchesFeedback;
    });

    // Count hidden normal flights for feedback mode
    const feedbackHiddenCount = mode === 'feedback' && !showNormalFeedback
        ? sourceAnomalies.filter(a => a.user_label === 0).length
        : 0;

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
        <aside className={clsx("flex flex-col gap-6 overflow-y-auto h-full pr-2", className || "col-span-3")}>
            
            {/* Mode Switcher - Top Row */}
            <div className="bg-surface rounded-xl p-1 flex gap-1">
                <button 
                    onClick={() => setMode('historical')}
                    className={clsx(
                        "flex-1 py-2 rounded-lg text-sm font-bold transition-all",
                        mode === 'historical' ? "bg-primary text-background-dark" : "text-white/60 hover:text-white"
                    )}
                >
                    History
                </button>
                <button 
                    onClick={() => setMode('research')}
                    className={clsx(
                        "flex-1 py-2 rounded-lg text-sm font-bold transition-all flex items-center justify-center gap-2",
                        mode === 'research' ? "bg-primary text-background-dark" : "text-white/60 hover:text-white"
                    )}
                >
                    <Beaker className="size-4" />
                    Research
                </button>
                <button 
                    onClick={() => { setMode('rules'); setSelectedRuleId(null); }}
                    className={clsx(
                        "flex-1 py-2 rounded-lg text-sm font-bold transition-all flex items-center justify-center gap-2",
                        mode === 'rules' ? "bg-primary text-background-dark" : "text-white/60 hover:text-white"
                    )}
                >
                    <List className="size-4" />
                    Rules
                </button>
                <button 
                    onClick={() => setMode('realtime')}
                    className={clsx(
                        "flex-1 py-2 rounded-lg text-sm font-bold transition-all flex items-center justify-center gap-2",
                        mode === 'realtime' ? "bg-primary text-background-dark" : "text-white/60 hover:text-white"
                    )}
                >
                    <Radio className={clsx("size-4", mode === 'realtime' && "animate-pulse")} />
                    Live
                </button>
            </div>

            {/* Mode Switcher - Bottom Row */}
            <div className="bg-surface rounded-xl p-1 flex gap-1 -mt-4">
                <button 
                    onClick={() => setMode('feedback')}
                    className={clsx(
                        "flex-1 py-2 rounded-lg text-sm font-bold transition-all flex items-center justify-center gap-2",
                        mode === 'feedback' ? "bg-primary text-background-dark" : "text-white/60 hover:text-white"
                    )}
                >
                    <History className="size-4" />
                    Feedback
                </button>
                {/* AI Results tab - only shown when there are results */}
                {aiResultFlights.length > 0 && (
                    <button 
                        onClick={() => setMode('ai-results')}
                        className={clsx(
                            "flex-1 py-2 rounded-lg text-sm font-bold transition-all flex items-center justify-center gap-2 animate-in slide-in-from-right-2 duration-300",
                            mode === 'ai-results' 
                                ? "bg-gradient-to-r from-violet-500 to-fuchsia-500 text-white" 
                                : "text-white/60 hover:text-white bg-gradient-to-r from-violet-500/20 to-fuchsia-500/20 border border-violet-500/30"
                        )}
                    >
                        <Sparkles className={clsx("size-4", mode === 'ai-results' && "animate-pulse")} />
                        AI Results
                        <span className="ml-1 px-1.5 py-0.5 text-[10px] bg-white/20 rounded-full font-bold">
                            {aiResultFlights.length}
                        </span>
                    </button>
                )}
            </div>

            {/* Date Filter (Only visible in Historical/Research/Feedback Mode) */}
            {(mode === 'historical' || mode === 'research' || mode === 'feedback') && (
                <div className="bg-surface rounded-xl p-4 flex flex-col gap-4 shrink-0 animate-in fade-in slide-in-from-top-2">
                    <p className="text-white text-base font-bold leading-tight">Filter by Date</p>
                    <div className="flex items-center p-1 justify-between">
                        <button onClick={() => changeDate(-1)} className="text-white/80 hover:text-white p-1 rounded hover:bg-white/10 transition-colors">
                            <ChevronLeft className="size-6" />
                        </button>
                        
                        <div className="flex-1 flex items-center justify-center gap-2 relative group cursor-pointer py-1 rounded hover:bg-white/5 transition-colors" onClick={() => {
                            // Programmatically trigger the date input
                            const input = document.getElementById('date-picker');
                            if (input && 'showPicker' in input) {
                                (input as any).showPicker();
                            }
                        }}>
                            <Calendar className="size-4 text-white/60 group-hover:text-white transition-colors" />
                            <p className="text-white text-sm font-bold leading-tight text-center select-none">
                                {selectedDate.toLocaleDateString(undefined, { weekday: 'short', year: 'numeric', month: 'long', day: 'numeric' })}
                            </p>
                            <input 
                                id="date-picker"
                                type="date"
                                className="absolute inset-0 opacity-0 cursor-pointer w-full h-full z-10 pointer-events-auto"
                                value={selectedDate.toISOString().split('T')[0]}
                                onChange={(e) => {
                                    if (e.target.value) {
                                        const parts = e.target.value.split('-');
                                        // Create date using local time components to avoid timezone shifts
                                        const newDate = new Date(
                                            parseInt(parts[0]), 
                                            parseInt(parts[1]) - 1, 
                                            parseInt(parts[2])
                                        );
                                        setSelectedDate(newDate);
                                    }
                                }}
                            />
                        </div>

                        <button onClick={() => changeDate(1)} className="text-white/80 hover:text-white p-1 rounded hover:bg-white/10 transition-colors">
                            <ChevronRight className="size-6" />
                        </button>
                    </div>
                </div>
            )}

            {/* Search & List */}
            <div className="bg-surface rounded-xl p-4 flex flex-col gap-4 flex-1 min-h-0">
                
                {/* Rule List View */}
                {mode === 'rules' && selectedRuleId === null ? (
                    <div className="flex flex-col gap-2 overflow-y-auto pr-2 -mr-2 flex-1">
                        {loading ? (
                            <p className="text-white/60 text-center py-4">Loading rules...</p>
                        ) : (
                            rules.map(rule => (
                                <div 
                                    key={rule.id}
                                    onClick={() => setSelectedRuleId(rule.id)}
                                    className="flex flex-col gap-1 p-3 rounded-lg cursor-pointer hover:bg-white/5 border border-transparent transition-colors"
                                >
                                    <div className="flex justify-between items-center">
                                        <p className="text-sm font-bold text-white">{rule.name}</p>
                                        <span className="text-xs text-white/40">ID: {rule.id}</span>
                                    </div>
                                    <p className="text-xs text-white/60">{rule.description}</p>
                                </div>
                            ))
                        )}
                    </div>
                ) : (
                    <>
                        {/* Header for Rule Details */}
                        {mode === 'rules' && selectedRuleId !== null && (
                            <div className="flex items-center gap-2 border-b border-white/5 pb-2">
                                <button 
                                    onClick={() => { setSelectedRuleId(null); setAnomalies([]); }}
                                    className="p-1 hover:bg-white/10 rounded transition-colors"
                                >
                                    <ArrowLeft className="size-5 text-white" />
                                </button>
                                <div>
                                    <p className="text-sm font-bold text-white">
                                        {rules.find(r => r.id === selectedRuleId)?.name || 'Rule Details'}
                                    </p>
                                    <p className="text-[10px] text-white/40">
                                        {anomalies.length} flights found
                                    </p>
                                </div>
                            </div>
                        )}

                        {/* Header for AI Results */}
                        {mode === 'ai-results' && (
                            <div className="flex items-center gap-2 border-b border-white/5 pb-2">
                                <div className="p-1.5 rounded-lg bg-gradient-to-r from-violet-500/20 to-fuchsia-500/20">
                                    <Sparkles className="size-4 text-violet-400" />
                                </div>
                                <div>
                                    <p className="text-sm font-bold text-white">AI Search Results</p>
                                    <p className="text-[10px] text-white/40">
                                        {aiResultFlights.length} flight{aiResultFlights.length !== 1 ? 's' : ''} found by AI
                                    </p>
                                </div>
                            </div>
                        )}

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
                        {/* Feedback Mode Filters */}
                        {mode === 'feedback' && (
                            <div>
                                <p className="text-xs text-white/60 font-bold uppercase mb-2">Feedback View</p>
                                <label className="flex items-center gap-2 cursor-pointer group">
                                    <input 
                                        type="checkbox"
                                        checked={showNormalFeedback}
                                        onChange={(e) => setShowNormalFeedback(e.target.checked)}
                                        className="w-4 h-4 rounded border-white/20 bg-white/5 text-primary focus:ring-primary focus:ring-offset-background-dark"
                                    />
                                    <span className="text-sm text-white/80 group-hover:text-white transition-colors">
                                        Show flights marked as Normal
                                    </span>
                                </label>
                            </div>
                        )}

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
                            <p className="text-[10px] text-white/40 mt-1">
                                Select Combination to choose multiple layers that must all appear on a flight.
                            </p>
                            {selectedTrigger === 'Combination' && (
                                <div className="mt-2 space-y-2">
                                    <p className="text-[11px] text-white/60 font-semibold">Choose layers to combine</p>
                                    <div className="flex flex-wrap gap-2">
                                        {triggerOptions
                                            .filter(option => option !== 'All' && option !== 'Combination')
                                            .map(layer => {
                                                const isActive = selectedLayerCombo.includes(layer);
                                                return (
                                                    <button
                                                        key={layer}
                                                        onClick={() => {
                                                            setSelectedLayerCombo(prev => {
                                                                if (prev.includes(layer)) {
                                                                    return prev.filter(l => l !== layer);
                                                                }
                                                                return [...prev, layer];
                                                            });
                                                        }}
                                                        className={clsx(
                                                            "px-2 py-1 rounded-md text-xs font-medium border transition-colors",
                                                            isActive 
                                                                ? "bg-primary/20 border-primary text-white" 
                                                                : "bg-white/5 border-white/10 text-white/70 hover:bg-white/10"
                                                        )}
                                                    >
                                                        {layer}
                                                    </button>
                                                );
                                            })}
                                    </div>
                                    <p className="text-[10px] text-white/40">
                                        No selection = any flight triggered by 2+ layers.
                                    </p>
                                </div>
                            )}
                        </div>

                        {/* Version Filter */}
                        <div>
                            <p className="text-xs text-white/60 font-bold uppercase mb-2">Filter by Version</p>
                            <div className="grid grid-cols-4 gap-2">
                                {versionOptions.map((option) => (
                                    <button
                                        key={option}
                                        onClick={() => setSelectedVersion(option)}
                                        className={clsx(
                                            "px-2 py-1.5 rounded-md text-xs font-medium transition-colors",
                                            selectedVersion === option
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
                    {loading && sourceAnomalies.length === 0 ? (
                        <LoadingPlane message={mode === 'realtime' ? "Scanning for live anomalies..." : "Searching anomalies..."} />
                    ) : filteredAnomalies.length === 0 ? (
                        <div className="flex flex-col items-center justify-center text-center py-8 gap-2">
                            <p className="text-white/60">
                                {mode === 'ai-results' 
                                    ? "Ask the AI to find flights. Try: \"Show me turn anomalies from yesterday\"" 
                                    : mode === 'realtime' 
                                        ? "No anomalies detected recently." 
                                        : "No flights match criteria."}
                            </p>
                            {feedbackHiddenCount > 0 && (
                                <button 
                                    onClick={() => setShowNormalFeedback(true)}
                                    className="text-xs text-primary hover:underline"
                                >
                                    Show {feedbackHiddenCount} hidden normal flights
                                </button>
                            )}
                        </div>
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

                             // Version Badge Logic
                             const cutoffTimestampV2 = new Date('2025-07-08T20:00:00Z').getTime() / 1000;
                             const cutoffTimestampV3 = new Date('2025-07-21T00:00:00Z').getTime() / 1000;
                             const cutoffTimestampV4 = new Date('2025-11-01T00:00:00Z').getTime() / 1000;
                             
                             let versionLabel = 'v1 OLD';
                             let versionStyle = "bg-zinc-800 text-zinc-500 border-zinc-700";

                             if (anomaly.timestamp >= cutoffTimestampV4) {
                                 versionLabel = 'v4 NEW';
                                 versionStyle = "badge-v4 animate-gradient-x";
                             } else if (anomaly.timestamp >= cutoffTimestampV3) {
                                 versionLabel = 'v3 OLD';
                                 versionStyle = "bg-zinc-800 text-zinc-500 border-zinc-700";
                             } else if (anomaly.timestamp >= cutoffTimestampV2) {
                                 versionLabel = 'v2 OLD';
                                 versionStyle = "bg-zinc-800 text-zinc-500 border-zinc-700";
                             }

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
                                        <div className="flex items-center gap-2">
                                            <p className="text-sm font-bold text-white">{displayTitle}</p>
                                            <span className={clsx(
                                                "text-[10px] font-bold px-1.5 py-0.5 rounded border select-none",
                                                versionStyle
                                            )}>
                                                {versionLabel}
                                            </span>
                                        </div>
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
                </>
            )}
            </div>
        </aside>
    );
};
