import { useState, useEffect, useMemo, useRef } from 'react';
import { Sidebar } from './components/Sidebar';
import { MapComponent } from './components/MapComponent';
import { ChatInterface } from './components/ChatInterface';
import { ReportPanel } from './components/ReportPanel';
import { fetchLiveTrack, fetchResearchTrack } from './api';
import type { AnomalyReport, FlightTrack } from './types';
import { Settings, Bell } from 'lucide-react';
import clsx from 'clsx';
import { ALERT_AUDIO_SRC } from './constants';

function App() {
  const [mode, setMode] = useState<'historical' | 'realtime' | 'research'>('historical');
  const [selectedAnomaly, setSelectedAnomaly] = useState<AnomalyReport | null>(null);
  const [flightData, setFlightData] = useState<FlightTrack | null>(null);
  const [, setLoadingTrack] = useState(false);
  const [showReport, setShowReport] = useState(false);
  const bellAudioRef = useRef<HTMLAudioElement | null>(null);

  useEffect(() => {
    if (selectedAnomaly) {
        setLoadingTrack(true);
        setShowReport(true); // Open report when anomaly selected
        
        const fetcher = mode === 'research' ? fetchResearchTrack : fetchLiveTrack;
        
        fetcher(selectedAnomaly.flight_id)
            .then(track => {
                setFlightData(track);
            })
            .catch(err => {
                console.error("Failed to load track", err);
                setFlightData(null);
            })
            .finally(() => setLoadingTrack(false));
    } else {
        setFlightData(null);
        setShowReport(false);
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
    };

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
            });
        }

        return Array.from(timestamps);
    }, [selectedAnomaly, flightData]);

    return (
    <div className="flex h-screen w-full flex-col bg-background-light dark:bg-background-dark text-white overflow-hidden">
      {/* Header */}
      <header className="flex items-center justify-between whitespace-nowrap border-b border-solid border-white/10 px-6 py-3 shrink-0 bg-[#1A1A1D]">
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
            <button className="flex h-10 w-10 items-center justify-center rounded-lg bg-[#2C2F33] text-white/80 hover:text-white transition-colors">
                <Settings className="h-5 w-5" />
            </button>
            <button 
                className="flex h-10 w-10 items-center justify-center rounded-lg bg-[#2C2F33] text-white/80 hover:text-white transition-colors"
                onClick={handleBellClick}
            >
                <Bell className="h-5 w-5" />
            </button>
            <div className="ml-2 size-10 rounded-full bg-gray-600" />
        </div>
      </header>

      <main className="flex-1 grid grid-cols-12 gap-6 p-6 overflow-hidden h-[calc(100vh-65px)]">
        
        {/* Sidebar */}
        <Sidebar 
            onSelectAnomaly={setSelectedAnomaly} 
            selectedAnomalyId={selectedAnomaly?.flight_id} 
            mode={mode}
            setMode={setMode}
        />

        {/* Map Area */}
        <section className={clsx(
            "bg-[#2C2F33] rounded-xl relative overflow-hidden border border-white/5 transition-all duration-300",
            showReport ? "col-span-6" : "col-span-9"
        )}>
            <MapComponent 
                points={flightData?.points || []} 
                anomalyTimestamps={anomalyTimestamps}
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
                </div>
            </div>
        </section>

        {/* Report Panel */}
        {showReport && selectedAnomaly && (
            <ReportPanel 
                anomaly={selectedAnomaly} 
                onClose={handleCloseReport} 
            />
        )}

      </main>

      {/* Chat - Always present, passed selected flight data */}
      <ChatInterface 
        data={selectedAnomaly?.full_report} 
        flightId={selectedAnomaly?.flight_id || "No Flight Selected"} 
        flightPoints={flightData?.points || []}
      />
    </div>
  )
}

export default App
