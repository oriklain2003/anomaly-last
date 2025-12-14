import { useState, useEffect, useMemo, useRef, useCallback } from 'react';
import { Link } from 'react-router-dom';
import { Sidebar, type SidebarMode } from './components/Sidebar';
import { MapComponent, type MapComponentHandle, type AIHighlightedPoint, type AIHighlightedSegment, type MLAnomalyPoint } from './components/MapComponent';
import { AnalysisPanel } from './components/AnalysisPanel';
import { SettingsModal } from './components/SettingsModal';
import { ReasoningChat } from './components/ReasoningChat';
import { fetchLiveTrack, fetchResearchTrack, fetchUnifiedTrack, fetchFeedbackTrack } from './api';
import type { AnomalyReport, FlightTrack } from './types';
import type { ProcessedActions } from './utils/aiActions';
import { Settings, Bell } from 'lucide-react';
import clsx from 'clsx';
import { ALERT_AUDIO_SRC } from './constants';

export function DesktopApp() {
  // Parse URL parameters for initial state
  const getInitialState = () => {
    const params = new URLSearchParams(window.location.search);
    const modeParam = params.get('mode');
    const dateParam = params.get('date');
    
    const validModes = ['historical', 'realtime', 'research', 'rules', 'feedback', 'ai-results'];
    const initialMode = validModes.includes(modeParam || '') ? (modeParam as SidebarMode) : 'historical';
    
    // Parse date safely
    let initialDate = new Date();
    if (dateParam) {
        const parts = dateParam.split('-');
        if (parts.length === 3) {
            initialDate = new Date(
                parseInt(parts[0]),
                parseInt(parts[1]) - 1,
                parseInt(parts[2])
            );
        }
    }
    
    return { mode: initialMode, date: initialDate };
  };

  const initialState = getInitialState();

  const [mode, setMode] = useState<SidebarMode>(initialState.mode);
  const [selectedDate, setSelectedDate] = useState<Date>(initialState.date);
  const [selectedAnomaly, setSelectedAnomaly] = useState<AnomalyReport | null>(null);
  const [flightData, setFlightData] = useState<FlightTrack | null>(null);
  const [secondaryFlightData, setSecondaryFlightData] = useState<FlightTrack | null>(null);
  const [, setLoadingTrack] = useState(false);
  const [showReport, setShowReport] = useState(false);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const bellAudioRef = useRef<HTMLAudioElement | null>(null);
  
  // AI Highlight State
  const [aiHighlightedPoint, setAiHighlightedPoint] = useState<AIHighlightedPoint | null>(null);
  const [aiHighlightedSegment, setAiHighlightedSegment] = useState<AIHighlightedSegment | null>(null);
  
  // AI Reasoning Panel State
  const [isAIPanelOpen, setIsAIPanelOpen] = useState(true);
  const [aiResultFlights, setAiResultFlights] = useState<AnomalyReport[]>([]);
  
  // Map refs
  const mapRef = useRef<MapComponentHandle>(null);
  const mapContainerRef = useRef<HTMLDivElement>(null);

  // Sync state to URL
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    params.set('mode', mode);
    
    // Format date as YYYY-MM-DD
    const year = selectedDate.getFullYear();
    const month = String(selectedDate.getMonth() + 1).padStart(2, '0');
    const day = String(selectedDate.getDate()).padStart(2, '0');
    const dateStr = `${year}-${month}-${day}`;
    
    params.set('date', dateStr);
    
    const newUrl = `${window.location.pathname}?${params.toString()}`;
    window.history.replaceState({}, '', newUrl);
  }, [mode, selectedDate]);

  // Auto-switch away from ai-results mode when results are cleared
  useEffect(() => {
    if (mode === 'ai-results' && aiResultFlights.length === 0) {
      setMode('historical');
    }
  }, [mode, aiResultFlights.length]);

  useEffect(() => {
    if (selectedAnomaly) {
        setLoadingTrack(true);
        setShowReport(true); // Open report when anomaly selected
        
        // Clear AI highlights when switching flights
        setAiHighlightedPoint(null);
        setAiHighlightedSegment(null);
        
        let fetcher: (id: string) => Promise<FlightTrack>;
        if (mode === 'feedback') {
            fetcher = fetchFeedbackTrack;
        } else if (mode === 'rules' || mode === 'ai-results') {
            fetcher = fetchUnifiedTrack;
        } else if (mode === 'research') {
            fetcher = fetchResearchTrack;
        } else {
            fetcher = fetchLiveTrack;
        }
        
        // Fetch Main Flight
        fetcher(selectedAnomaly.flight_id)
            .then(track => {
                setFlightData(track);
            })
            .catch(err => {
                console.error("Failed to load track", err);
                setFlightData(null);
            })
            .finally(() => setLoadingTrack(false));

        // Check for Proximity Alert (Rule ID 4) and fetch secondary flight
        const layer1 = selectedAnomaly.full_report?.layer_1_rules;
        const proximityRule = layer1?.report?.matched_rules?.find((r: any) => r.id === 4);
        
        if (proximityRule && proximityRule.details?.events?.length > 0) {
            const otherFlightId = proximityRule.details.events[0].other_flight;
            if (otherFlightId) {
                console.log("Found Proximity Alert, fetching secondary flight:", otherFlightId);
                fetcher(otherFlightId)
                    .then(track => {
                        setSecondaryFlightData(track);
                    })
                    .catch(err => {
                        console.error("Failed to load secondary track", err);
                        setSecondaryFlightData(null);
                    });
            } else {
                setSecondaryFlightData(null);
            }
        } else {
            setSecondaryFlightData(null);
        }

    } else {
        setFlightData(null);
        setSecondaryFlightData(null);
        setShowReport(false);
        setAiHighlightedPoint(null);
        setAiHighlightedSegment(null);
    }
  }, [selectedAnomaly, mode]);

  useEffect(() => {
    const audio = new Audio(ALERT_AUDIO_SRC);
    audio.preload = 'auto';
    bellAudioRef.current = audio;

    return () => {
        audio.pause();
        bellAudioRef.current = null;
    };
  }, []);

  const handleBellClick = () => {
    const audio = bellAudioRef.current;
    if (!audio) return;

    try {
        audio.pause();
        audio.currentTime = 0;
        void audio.play();
    } catch (error) {
        console.warn('Unable to play bell sound', error);
    }
  };

    const handleCloseReport = () => {
        setShowReport(false);
        setSelectedAnomaly(null); // Deselect
        setAiHighlightedPoint(null);
        setAiHighlightedSegment(null);
    };

    // Handle AI actions from the AnalysisPanel
    const handleAIActions = useCallback((actions: ProcessedActions) => {
        setAiHighlightedPoint(actions.highlightedPoint);
        setAiHighlightedSegment(actions.highlightedSegment);
        
        // Handle zoom bounds if specified
        if (actions.zoomBounds && mapRef.current) {
            mapRef.current.fitBounds(
                actions.zoomBounds.north,
                actions.zoomBounds.south,
                actions.zoomBounds.east,
                actions.zoomBounds.west
            );
        }
    }, []);

    // Clear AI highlights
    const handleClearAIHighlights = useCallback(() => {
        setAiHighlightedPoint(null);
        setAiHighlightedSegment(null);
    }, []);

    // Handle flights received from AI reasoning
    const handleAIFlightsReceived = useCallback((flights: AnomalyReport[]) => {
        setAiResultFlights(flights);
        setMode('ai-results'); // Switch to AI Results tab
    }, []);

    // Extract anomaly timestamps for visualization
    const anomalyTimestamps = useMemo(() => {
        if (!selectedAnomaly || !selectedAnomaly.full_report || !flightData) return [];

        const timestamps = new Set<number>();
        const report = selectedAnomaly.full_report;
        const points = flightData.points;

        // Check Layer 1: Rule Engine
        const layer1 = report.layer_1_rules;
        if (layer1?.report?.matched_rules) {
            layer1.report.matched_rules.forEach((rule: any) => {
                // 1. Events Array
                if (rule.details?.events && Array.isArray(rule.details.events)) {
                    rule.details.events.forEach((event: any) => {
                        if (event.timestamp) timestamps.add(event.timestamp);
                        
                        // Ranges (e.g. holding pattern)
                        if (event.start_ts && event.end_ts) {
                            points.forEach(p => {
                                if (p.timestamp >= event.start_ts && p.timestamp <= event.end_ts) {
                                    timestamps.add(p.timestamp);
                                }
                            });
                        }
                    });
                }

                // 2. Gaps (Signal Loss)
                if (rule.details?.gaps && Array.isArray(rule.details.gaps)) {
                    rule.details.gaps.forEach((gap: any) => {
                        if (gap.start_ts) timestamps.add(gap.start_ts);
                        if (gap.end_ts) timestamps.add(gap.end_ts);
                    });
                }

                // 3. Special Rule Details (e.g. Return to Field)
                if (rule.id === 7 && rule.details?.takeoff_ts && rule.details?.landing_ts) {
                    points.forEach(p => {
                        if (p.timestamp >= rule.details.takeoff_ts && p.timestamp <= rule.details.landing_ts) {
                            timestamps.add(p.timestamp);
                        }
                    });
                }

                // 4. Path Learning (Rule 11) - Off Course Points
                if (rule.id === 11 && rule.details?.off_course_timestamps && Array.isArray(rule.details.off_course_timestamps)) {
                    rule.details.off_course_timestamps.forEach((ts: number) => timestamps.add(ts));
                }
            });
        }

        return Array.from(timestamps);
    }, [selectedAnomaly, flightData]);

    // Extract ML anomaly points for map visualization
    const mlAnomalyPoints = useMemo((): MLAnomalyPoint[] => {
        if (!selectedAnomaly || !selectedAnomaly.full_report) return [];

        const points: MLAnomalyPoint[] = [];
        const report = selectedAnomaly.full_report;

        const layerMap: Record<string, string> = {
            'layer_3_deep_dense': 'Deep Dense',
            'layer_4_deep_cnn': 'Deep CNN',
            'layer_5_transformer': 'Transformer',
            'layer_6_hybrid': 'Hybrid'
        };

        Object.entries(layerMap).forEach(([key, layerName]) => {
            const layerData = report[key];
            if (layerData?.anomaly_points && layerData.is_anomaly) {
                layerData.anomaly_points.forEach((pt: any) => {
                    points.push({
                        lat: pt.lat,
                        lon: pt.lon,
                        timestamp: pt.timestamp,
                        point_score: pt.point_score,
                        layer: layerName
                    });
                });
            }
        });

        return points;
    }, [selectedAnomaly]);

    return (
    <div className="flex h-screen w-full flex-col bg-background-light dark:bg-background-dark text-white overflow-hidden">
      {/* Header */}
      <header className="flex items-center justify-between whitespace-nowrap border-b border-solid border-white/10 px-6 py-3 shrink-0 bg-surface">
        <div className="flex items-center gap-4 text-white">
            <div className="size-6 text-primary">
                <svg fill="none" viewBox="0 0 48 48" xmlns="http://www.w3.org/2000/svg">
                    <path d="M13.8261 17.4264C16.7203 18.1174 20.2244 18.5217 24 18.5217C27.7756 18.5217 31.2797 18.1174 34.1739 17.4264C36.9144 16.7722 39.9967 15.2331 41.3563 14.1648L24.8486 40.6391C24.4571 41.267 23.5429 41.267 23.1514 40.6391L6.64374 14.1648C8.00331 15.2331 11.0856 16.7722 13.8261 17.4264Z" fill="currentColor"></path>
                    <path clipRule="evenodd" d="M39.998 12.236C39.9944 12.2537 39.9875 12.2845 39.9748 12.3294C39.9436 12.4399 39.8949 12.5741 39.8346 12.7175C39.8168 12.7597 39.7989 12.8007 39.7813 12.8398C38.5103 13.7113 35.9788 14.9393 33.7095 15.4811C30.9875 16.131 27.6413 16.5217 24 16.5217C20.3587 16.5217 17.0125 16.131 14.2905 15.4811C12.0012 14.9346 9.44505 13.6897 8.18538 12.8168C8.17384 12.7925 8.16216 12.767 8.15052 12.7408C8.09919 12.6249 8.05721 12.5114 8.02977 12.411C8.00356 12.3152 8.00039 12.2667 8.00004 12.2612C8.00004 12.261 8 12.2607 8.00004 12.2612C8.00004 12.2359 8.0104 11.9233 8.68485 11.3686C9.34546 10.8254 10.4222 10.2469 11.9291 9.72276C14.9242 8.68098 19.1919 8 24 8C28.8081 8 33.0758 8.68098 36.0709 9.72276C37.5778 10.2469 38.6545 10.8254 39.3151 11.3686C39.9006 11.8501 39.9857 12.1489 39.998 12.236ZM4.95178 15.2312L21.4543 41.6973C22.6288 43.5809 25.3712 43.5809 26.5457 41.6973L43.0534 15.223C43.0709 15.1948 43.0878 15.1662 43.104 15.1371L41.3563 14.1648C43.104 15.1371 43.1038 15.1374 43.104 15.1371L43.1051 15.135L43.1065 15.1325L43.1101 15.1261L43.1199 15.1082C43.1276 15.094 43.1377 15.0754 43.1497 15.0527C43.1738 15.0075 43.2062 14.9455 43.244 14.8701C43.319 14.7208 43.4196 14.511 43.5217 14.2683C43.6901 13.8679 44 13.0689 44 12.2609C44 10.5573 43.003 9.22254 41.8558 8.2791C40.6947 7.32427 39.1354 6.55361 37.385 5.94477C33.8654 4.72057 29.133 4 24 4C18.867 4 14.1346 4.72057 10.615 5.94478C8.86463 6.55361 7.30529 7.32428 6.14419 8.27911C4.99695 9.22255 3.99999 10.5573 3.99999 12.2609C3.99999 13.1275 4.29264 13.9078 4.49321 14.3607C4.60375 14.6102 4.71348 14.8196 4.79687 14.9689C4.83898 15.0444 4.87547 15.1065 4.9035 15.1529C4.91754 15.1762 4.92954 15.1957 4.93916 15.2111L4.94662 15.223L4.95178 15.2312ZM35.9868 18.996L24 38.22L12.0131 18.996C12.4661 19.1391 12.9179 19.2658 13.3617 19.3718C16.4281 20.1039 20.0901 20.5217 24 20.5217C27.9099 20.5217 31.5719 20.1039 34.6383 19.3718C35.082 19.2658 35.5339 19.1391 35.9868 18.996Z" fill="currentColor" fillRule="evenodd"></path>
                </svg>
            </div>
            <h2 className="text-white text-xl font-bold leading-tight tracking-[-0.015em]">Onyx Anomaly Explorer</h2>
        </div>
        <div className="flex flex-1 justify-end gap-2">
            <Link
                to="/explorer"
                className="flex h-10 px-3 items-center justify-center rounded-lg bg-surface-highlight text-white/80 hover:text-white transition-colors border border-white/10 text-sm font-bold no-underline"
            >
                Explorer
            </Link>
            <button
                onClick={() => setIsSettingsOpen(true)}
                className="flex h-10 w-10 items-center justify-center rounded-lg bg-surface-highlight text-white/80 hover:text-white transition-colors border border-white/10"
            >
                <Settings className="h-5 w-5" />
            </button>
            <button 
                className="flex h-10 w-10 items-center justify-center rounded-lg bg-surface-highlight text-white/80 hover:text-white transition-colors border border-white/10"
                onClick={handleBellClick}
            >
                <Bell className="h-5 w-5" />
            </button>
            <div className="ml-2 size-10 rounded-full bg-gray-600" />
        </div>
      </header>

      <main className="flex-1 flex gap-6 p-6 overflow-hidden h-[calc(100vh-65px)]">
        
        {/* Left Section: Sidebar + Map + Report */}
        <div className={clsx(
            "flex-1 grid grid-cols-12 gap-6 transition-all duration-300",
            isAIPanelOpen ? "mr-0" : "mr-0"
        )}>
            {/* Sidebar */}
            <Sidebar 
                onSelectAnomaly={setSelectedAnomaly} 
                selectedAnomalyId={selectedAnomaly?.flight_id} 
                mode={mode}
                setMode={setMode}
                selectedDate={selectedDate}
                setSelectedDate={setSelectedDate}
                aiResultFlights={aiResultFlights}
            />

            {/* Map Area */}
            <section 
                ref={mapContainerRef}
                className={clsx(
                    "bg-surface rounded-xl relative overflow-hidden border border-white/5 transition-all duration-300",
                    showReport ? "col-span-6" : "col-span-9"
                )}
            >
                <MapComponent 
                    ref={mapRef}
                    points={flightData?.points || []} 
                    secondaryPoints={secondaryFlightData?.points}
                    anomalyTimestamps={anomalyTimestamps}
                    mlAnomalyPoints={mlAnomalyPoints}
                    aiHighlightedPoint={aiHighlightedPoint}
                    aiHighlightedSegment={aiHighlightedSegment}
                    onClearAIHighlights={handleClearAIHighlights}
                />
                
                {/* Legend Overlay */}
                <div className="absolute bottom-4 right-4 bg-background-dark/80 backdrop-blur-sm p-3 rounded-lg border border-white/10 text-white z-10">
                    <p className="text-sm font-bold mb-2">Legend</p>
                    <div className="flex flex-col gap-2 text-xs">
                        <div className="flex items-center gap-2">
                            <span className="h-2.5 w-2.5 rounded-full bg-red-500"></span>
                            <span>Anomaly Event</span>
                        </div>
                        <div className="flex items-center gap-2">
                            <span className="h-2.5 w-2.5 rounded-full bg-blue-500"></span>
                            <span>Normal Flight Path</span>
                        </div>
                        <div className="flex items-center gap-2">
                            <span className="h-2.5 w-2.5 rounded-full bg-green-500"></span>
                            <span>Start</span>
                        </div>
                         <div className="flex items-center gap-2">
                            <span className="h-2.5 w-2.5 rounded-full bg-red-500"></span>
                            <span>End</span>
                        </div>
                        {secondaryFlightData && (
                             <div className="flex items-center gap-2">
                                <span className="h-2.5 w-2.5 rounded-full bg-orange-400"></span>
                                <span>Conflicting Flight</span>
                            </div>
                        )}
                        {(aiHighlightedPoint || aiHighlightedSegment) && (
                            <div className="flex items-center gap-2">
                                <span className="h-2.5 w-2.5 rounded-full bg-amber-500 animate-pulse"></span>
                                <span>AI Highlight</span>
                            </div>
                        )}
                        {/* ML Anomaly Points Legend */}
                        {mlAnomalyPoints.length > 0 && (
                            <>
                                <div className="border-t border-white/10 pt-2 mt-1">
                                    <p className="text-[10px] text-white/50 font-bold mb-1">ML Anomaly Points</p>
                                </div>
                                <div className="flex items-center gap-2">
                                    <span className="h-2.5 w-2.5 rounded-full bg-purple-500"></span>
                                    <span>Deep Dense</span>
                                </div>
                                <div className="flex items-center gap-2">
                                    <span className="h-2.5 w-2.5 rounded-full bg-orange-500"></span>
                                    <span>Deep CNN</span>
                                </div>
                                <div className="flex items-center gap-2">
                                    <span className="h-2.5 w-2.5 rounded-full bg-cyan-500"></span>
                                    <span>Transformer</span>
                                </div>
                                <div className="flex items-center gap-2">
                                    <span className="h-2.5 w-2.5 rounded-full bg-pink-500"></span>
                                    <span>Hybrid</span>
                                </div>
                            </>
                        )}
                    </div>
                </div>
            </section>

            {/* Analysis Panel (Report + AI Assistant) */}
            {showReport && selectedAnomaly && (
                <AnalysisPanel 
                    anomaly={selectedAnomaly}
                    flightPoints={flightData?.points || []}
                    onClose={handleCloseReport}
                    onAIActions={handleAIActions}
                    onFlyTo={(lat, lon, zoom) => mapRef.current?.flyTo(lat, lon, zoom)}
                    mode={mode}
                />
            )}
        </div>

        {/* AI Reasoning Panel - Always visible, collapsible */}
        <div className={clsx(
            "transition-all duration-300 shrink-0",
            isAIPanelOpen ? "w-[380px]" : "w-0"
        )}>
            {isAIPanelOpen && (
                <ReasoningChat
                    isOpen={isAIPanelOpen}
                    onToggle={() => setIsAIPanelOpen(!isAIPanelOpen)}
                    onFlightsReceived={handleAIFlightsReceived}
                    selectedFlight={selectedAnomaly && flightData ? {
                        flightId: selectedAnomaly.flight_id,
                        callsign: selectedAnomaly.callsign,
                        points: flightData.points,
                        report: selectedAnomaly.full_report
                    } : null}
                />
            )}
        </div>

        {/* Collapsed AI Panel Toggle */}
        {!isAIPanelOpen && (
            <ReasoningChat
                isOpen={false}
                onToggle={() => setIsAIPanelOpen(true)}
                onFlightsReceived={handleAIFlightsReceived}
                selectedFlight={null}
            />
        )}

      </main>

      <SettingsModal 
        isOpen={isSettingsOpen} 
        onClose={() => setIsSettingsOpen(false)} 
      />
    </div>
  )
}
