import React, { useEffect, useRef, useState } from 'react';
import maplibregl from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';
import { X, Play, Pause, SkipBack, SkipForward, AlertTriangle, MapPin, Wrench, Check } from 'lucide-react';
import { fetchUnifiedTrack } from '../api';
import type { TrackPoint } from '../types';
import clsx from 'clsx';

export interface ReplayEvent {
    timestamp: number;
    description: string;
    type: 'proximity' | 'deviation' | 'ml_anomaly' | 'other';
    lat?: number;
    lon?: number;
}

interface ReplayModalProps {
    mainFlightId: string;
    secondaryFlightIds?: string[];
    events?: ReplayEvent[];
    onClose: () => void;
}

interface FlightData {
    id: string;
    points: TrackPoint[];
    color: string;
}

type TelemetryData = 
    | { status: 'waiting' }
    | ({ status: 'active' | 'ended' } & TrackPoint);

const COLORS = ['#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6'];

// Airports list from rule_config.json
const AIRPORTS = [
    { code: "LLBG", name: "Ben Gurion Intl", lat: 32.011389, lon: 34.886667, elevation_ft: 135 },
    { code: "LLER", name: "Ramon Intl", lat: 29.723704, lon: 35.01145, elevation_ft: 648 },
    { code: "LLHA", name: "Haifa", lat: 32.809444, lon: 35.043056, elevation_ft: 28 },
    { code: "LLBS", name: "Beersheba", lat: 31.287, lon: 34.723, elevation_ft: 886 },
    { code: "LLOV", name: "Ovda", lat: 29.940, lon: 34.935, elevation_ft: 1492 },
    { code: "LLNV", name: "Nevatim AFB", lat: 31.207, lon: 35.012, elevation_ft: 1312 },
    { code: "LLMG", name: "Megiddo", lat: 32.597, lon: 35.228, elevation_ft: 217 },
    { code: "LLHZ", name: "Herzliya", lat: 32.186, lon: 34.835, elevation_ft: 121 },
    { code: "LCRA", name: "RAF Akrotiri", lat: 34.5900, lon: 32.9870, elevation_ft: 76 },
    { code: "OLBA", name: "Beirut Rafic Hariri Intl", lat: 33.820889, lon: 35.488389, elevation_ft: 89 },
    { code: "OLKA", name: "Rayak Air Base", lat: 33.850, lon: 35.987, elevation_ft: 2953 },
    { code: "OJAI", name: "Queen Alia Intl (Amman)", lat: 31.722556, lon: 35.993214, elevation_ft: 2395 },
    { code: "OJAM", name: "Amman Civil Airport (Marka)", lat: 31.9697, lon: 35.9917, elevation_ft: 2555 },
    { code: "OJAQ", name: "King Hussein Intl (Aqaba)", lat: 29.611, lon: 35.018, elevation_ft: 170 },
    { code: "OJMF", name: "Mafraq", lat: 32.356, lon: 36.259, elevation_ft: 2240 },
    { code: "HEGR", name: "El Gora Airport", lat: 31.0686, lon: 34.1296, elevation_ft: 295 },
    { code: "OSDI", name: "Damascus Intl", lat: 33.411, lon: 36.516, elevation_ft: 2020 }
];

export const ReplayModal: React.FC<ReplayModalProps> = ({ mainFlightId, secondaryFlightIds = [], events = [], onClose }) => {
    const mapContainer = useRef<HTMLDivElement>(null);
    const map = useRef<maplibregl.Map | null>(null);
    const popupRef = useRef<maplibregl.Popup | null>(null);
    
    const [flights, setFlights] = useState<FlightData[]>([]);
    const [loading, setLoading] = useState(true);
    const [isPlaying, setIsPlaying] = useState(false);
    const [currentTime, setCurrentTime] = useState<number>(0);
    const [speed, setSpeed] = useState<number>(1);
    const [minTime, setMinTime] = useState<number>(0);
    const [maxTime, setMaxTime] = useState<number>(0);
    
    // Tools State
    const [showTools, setShowTools] = useState(false);
    const [distanceTool, setDistanceTool] = useState(true);
    const [airportDistanceTool, setAirportDistanceTool] = useState(true);

    const animationRef = useRef<number | undefined>(undefined);
    const lastFrameTime = useRef<number>(0);

    // Helper: Haversine Distance in NM
    const getDistanceNM = (lat1: number, lon1: number, lat2: number, lon2: number) => {
        const R = 3440.065; // Earth radius in NM
        const dLat = (lat2 - lat1) * Math.PI / 180;
        const dLon = (lon2 - lon1) * Math.PI / 180;
        const a = 
            Math.sin(dLat/2) * Math.sin(dLat/2) +
            Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) * 
            Math.sin(dLon/2) * Math.sin(dLon/2);
        const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
        return R * c;
    };

    // Zoom to event handler
    const handleEventClick = (event: ReplayEvent) => {
        setCurrentTime(event.timestamp);
        
        // Find the flight position at this time if lat/lon not provided in event
        let targetLat = event.lat;
        let targetLon = event.lon;

        if ((!targetLat || !targetLon) && flights.length > 0) {
            // Try to find position of main flight at this time
            const mainFlight = flights.find(f => f.id === mainFlightId);
            if (mainFlight) {
                // Find closest point
                const point = mainFlight.points.reduce((prev, curr) => 
                    Math.abs(curr.timestamp - event.timestamp) < Math.abs(prev.timestamp - event.timestamp) ? curr : prev
                );
                if (point) {
                    targetLat = point.lat;
                    targetLon = point.lon;
                }
            }
        }

        if (targetLat && targetLon && map.current) {
            map.current.flyTo({
                center: [targetLon, targetLat],
                zoom: 14,
                speed: 1.5
            });
        }
    };

    // Fetch data
    useEffect(() => {
        const loadData = async () => {
            setLoading(true);
            try {
                // Ensure unique IDs and remove any empty strings
                const ids = [mainFlightId, ...new Set(secondaryFlightIds)].filter(Boolean);
                console.log("[Replay] Loading flights:", ids);
                
                const results = await Promise.all(
                    ids.map(async (id, index) => {
                        try {
                            const track = await fetchUnifiedTrack(id);
                            return {
                                id,
                                points: track.points.sort((a, b) => a.timestamp - b.timestamp),
                                color: COLORS[index % COLORS.length]
                            };
                        } catch (e) {
                            console.error(`Failed to load track for ${id}`, e);
                            return null;
                        }
                    })
                );

                const validFlights = results.filter((f): f is FlightData => f !== null);
                setFlights(validFlights);

                if (validFlights.length > 0) {
                    const allPoints = validFlights.flatMap(f => f.points);
                    const min = Math.min(...allPoints.map(p => p.timestamp));
                    const max = Math.max(...allPoints.map(p => p.timestamp));
                    setMinTime(min);
                    setMaxTime(max);
                    setCurrentTime(min);
                }
            } finally {
                setLoading(false);
            }
        };

        loadData();
    }, [mainFlightId, secondaryFlightIds]);

    // Initialize Map
    useEffect(() => {
        if (!mapContainer.current || loading || flights.length === 0) return;

        map.current = new maplibregl.Map({
            container: mapContainer.current,
            style: 'https://api.maptiler.com/maps/darkmatter/style.json?key=r7kaQpfNDVZdaVp23F1r',
            center: [0, 0],
            zoom: 2,
        });

        popupRef.current = new maplibregl.Popup({
            closeButton: false,
            closeOnClick: false,
            maxWidth: '200px'
        });

        map.current.on('load', () => {
            if (!map.current) return;

            // Fit bounds
            const bounds = new maplibregl.LngLatBounds();
            flights.forEach(f => {
                f.points.forEach(p => bounds.extend([p.lon, p.lat]));
            });
            map.current.fitBounds(bounds, { padding: 50 });

            // Add sources and layers for each flight
            flights.forEach(flight => {
                // Ghost Line source (Full path)
                map.current!.addSource(`source-${flight.id}-ghost`, {
                    type: 'geojson',
                    data: {
                        type: 'FeatureCollection',
                        features: [{
                            type: 'Feature',
                            properties: {},
                            geometry: {
                                type: 'LineString',
                                coordinates: flight.points.map(p => [p.lon, p.lat])
                            }
                        }]
                    }
                });

                // Ghost Line layer
                map.current!.addLayer({
                    id: `layer-${flight.id}-ghost`,
                    type: 'line',
                    source: `source-${flight.id}-ghost`,
                    layout: { 'line-join': 'round', 'line-cap': 'round' },
                    paint: {
                        'line-color': flight.color,
                        'line-width': 2,
                        'line-opacity': 0.2, // Faint
                        'line-dasharray': [2, 2] // Dashed
                    }
                });

                // Line source
                map.current!.addSource(`source-${flight.id}`, {
                    type: 'geojson',
                    data: { type: 'FeatureCollection', features: [] }
                });

                // Line layer
                map.current!.addLayer({
                    id: `layer-${flight.id}-line`,
                    type: 'line',
                    source: `source-${flight.id}`,
                    layout: { 'line-join': 'round', 'line-cap': 'round' },
                    paint: {
                        'line-color': flight.color,
                        'line-width': 4,
                        'line-opacity': 0.8
                    }
                });
                
                // Add click handler for the line (to jump time?) or just info?
                // For now, let's keep it simple.

                // Current position marker source
                map.current!.addSource(`source-${flight.id}-pos`, {
                    type: 'geojson',
                    data: { type: 'FeatureCollection', features: [] }
                });

                // Current position marker layer (circle/dot)
                map.current!.addLayer({
                    id: `layer-${flight.id}-pos`,
                    type: 'circle',
                    source: `source-${flight.id}-pos`,
                    paint: {
                        'circle-radius': 8,
                        'circle-color': flight.color,
                        'circle-stroke-width': 2,
                        'circle-stroke-color': '#fff'
                    }
                });
                
                // Add flight label
                map.current!.addLayer({
                    id: `layer-${flight.id}-label`,
                    type: 'symbol',
                    source: `source-${flight.id}-pos`,
                    layout: {
                        'text-field': flight.id,
                        'text-offset': [0, -2],
                        'text-size': 12,
                        'text-anchor': 'bottom',
                        'text-allow-overlap': true
                    },
                    paint: {
                        'text-color': '#ffffff',
                        'text-halo-color': '#000000',
                        'text-halo-width': 2
                    }
                });

                // Interaction for popup
                const showPopup = (e: any) => {
                    if (!map.current || !popupRef.current) return;
                    map.current.getCanvas().style.cursor = 'pointer';

                    const feature = e.features[0];
                    const props = feature.properties;
                    const coords = (feature.geometry as any).coordinates.slice();

                    while (Math.abs(e.lngLat.lng - coords[0]) > 180) {
                        coords[0] += e.lngLat.lng > coords[0] ? 360 : -360;
                    }

                    const timeStr = new Date(props.timestamp * 1000).toLocaleTimeString();
                    
                    popupRef.current
                        .setLngLat(coords)
                        .setHTML(`
                            <div class="text-gray-900 p-2 text-xs font-sans min-w-[120px]">
                                <div class="font-bold border-b border-gray-300 pb-1 mb-1 text-sm">${flight.id}</div>
                                <div class="grid grid-cols-2 gap-x-2 gap-y-1">
                                    <div class="text-gray-500">Time:</div>
                                    <div class="font-mono font-bold text-right">${timeStr}</div>
                                    <div class="text-gray-500">Alt:</div>
                                    <div class="font-mono font-bold text-right">${props.alt} ft</div>
                                    <div class="text-gray-500">Hdg:</div>
                                    <div class="font-mono font-bold text-right">${props.track}°</div>
                                    <div class="text-gray-500">GS:</div>
                                    <div class="font-mono font-bold text-right">${props.gspeed || 0} kts</div>
                                </div>
                            </div>
                        `)
                        .addTo(map.current);
                };

                const hidePopup = () => {
                    if (!map.current || !popupRef.current) return;
                    map.current.getCanvas().style.cursor = '';
                    popupRef.current.remove();
                };

                map.current?.on('mouseenter', `layer-${flight.id}-pos`, showPopup);
                map.current?.on('mouseleave', `layer-${flight.id}-pos`, hidePopup);
                map.current?.on('click', `layer-${flight.id}-pos`, showPopup); // Also work on click
            });
        });

        return () => {
            map.current?.remove();
        };
    }, [loading, flights]);

    // Animation Loop
    useEffect(() => {
        if (!isPlaying) {
            lastFrameTime.current = 0;
            if (animationRef.current) cancelAnimationFrame(animationRef.current);
            return;
        }

        const animate = (time: number) => {
            if (lastFrameTime.current === 0) {
                lastFrameTime.current = time;
            }

            const delta = (time - lastFrameTime.current) / 1000; // seconds
            lastFrameTime.current = time;

            setCurrentTime(prev => {
                const next = prev + (delta * speed);
                if (next >= maxTime) {
                    setIsPlaying(false);
                    return maxTime;
                }
                return next;
            });

            animationRef.current = requestAnimationFrame(animate);
        };

        animationRef.current = requestAnimationFrame(animate);

        return () => {
            if (animationRef.current) cancelAnimationFrame(animationRef.current);
        };
    }, [isPlaying, speed, maxTime]);

    // Update Map Data based on Current Time
    useEffect(() => {
        if (!map.current || !map.current.isStyleLoaded()) return;

        flights.forEach(flight => {
            const activePoints = flight.points.filter(p => p.timestamp <= currentTime);
            
            if (activePoints.length === 0) {
                // Clear
                (map.current!.getSource(`source-${flight.id}`) as maplibregl.GeoJSONSource)?.setData({
                    type: 'FeatureCollection',
                    features: []
                });
                (map.current!.getSource(`source-${flight.id}-pos`) as maplibregl.GeoJSONSource)?.setData({
                    type: 'FeatureCollection',
                    features: []
                });
                return;
            }

            // Update Line
            (map.current!.getSource(`source-${flight.id}`) as maplibregl.GeoJSONSource)?.setData({
                type: 'FeatureCollection',
                features: [{
                    type: 'Feature',
                    properties: {},
                    geometry: {
                        type: 'LineString',
                        coordinates: activePoints.map(p => [p.lon, p.lat])
                    }
                }] as any
            });

            // Update Head Marker
            const lastPoint = activePoints[activePoints.length - 1];
            (map.current!.getSource(`source-${flight.id}-pos`) as maplibregl.GeoJSONSource)?.setData({
                type: 'FeatureCollection',
                features: [{
                    type: 'Feature',
                    properties: {
                        timestamp: lastPoint.timestamp,
                        alt: lastPoint.alt,
                        track: lastPoint.track || 0,
                        gspeed: lastPoint.gspeed || 0
                    },
                    geometry: {
                        type: 'Point',
                        coordinates: [lastPoint.lon, lastPoint.lat]
                    }
                }] as any
            });
        });

    }, [currentTime, flights]);

    const formatTime = (ts: number) => {
        return new Date(ts * 1000).toLocaleTimeString();
    };

    const handleSeek = (e: React.ChangeEvent<HTMLInputElement>) => {
        setCurrentTime(Number(e.target.value));
    };

    // Helper to get current telemetry for display in UI overlay
    const getCurrentTelemetry = (flight: FlightData): TelemetryData => {
        const started = flight.points[0].timestamp <= currentTime;
        // Check if we are past the last point
        const ended = currentTime > flight.points[flight.points.length - 1].timestamp;

        if (!started) return { status: 'waiting' };
        
        const activePoints = flight.points.filter(p => p.timestamp <= currentTime);
        if (activePoints.length === 0) return { status: 'waiting' };
        
        const p = activePoints[activePoints.length - 1];
        
        return { 
            status: ended ? 'ended' : 'active',
            ...p
        };
    };

    // Calculate distances to all other flights if tool enabled
    const getDistancesToOthers = (currentFlightId: string, currentLat: number, currentLon: number) => {
        if (!distanceTool) return [];

        return flights
            .filter(f => f.id !== currentFlightId)
            .map(f => {
                const tel = getCurrentTelemetry(f);
                if (tel.status === 'waiting' || tel.status === 'ended' || !('lat' in tel)) return null;
                
                return {
                    id: f.id,
                    dist: getDistanceNM(currentLat, currentLon, tel.lat, tel.lon),
                    color: f.color
                };
            })
            .filter((d): d is { id: string; dist: number; color: string } => d !== null)
            .sort((a, b) => a.dist - b.dist);
    };

    // Calculate distance to nearest airport
    const getNearestAirport = (lat: number, lon: number) => {
        if (!airportDistanceTool) return null;

        let nearest: { code: string; name: string; dist: number } | null = null;
        
        for (const airport of AIRPORTS) {
            const dist = getDistanceNM(lat, lon, airport.lat, airport.lon);
            if (!nearest || dist < nearest.dist) {
                nearest = { code: airport.code, name: airport.name, dist };
            }
        }
        
        return nearest;
    };

    if (loading) {
        return (
            <div className="fixed inset-0 z-50 bg-black/90 flex items-center justify-center backdrop-blur-sm">
                <style>{`
                    .u-loading {
                        width: 128px;
                        height: 128px;
                        display: block;
                    }
                    .u-loading__symbol {
                        background-color: var(--color-background); /* primary: background */
                        padding: 8px;
                        animation: loading 3s infinite;
                        border-radius: 5px;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        width: 100%;
                        height: 100%;
                    }
                    .u-loading__content {
                         display: block;
                         width: 100%;
                         animation: loading-icon 3s infinite;
                    }
                    @keyframes loading {
                        0% { transform: perspective(250px) rotateX(0deg) rotateY(0deg); }
                        15% { background-color: var(--color-background); }
                        16% { background-color: rgb(var(--color-primary)); } /* secondary: accent */
                        50% { transform: perspective(250px) rotateX(180deg) rotateY(0deg); background-color: rgb(var(--color-primary)); }
                        65% { background-color: rgb(var(--color-primary)); }
                        66% { background-color: var(--color-background); }
                        100% { transform: perspective(250px) rotateX(180deg) rotateY(-180deg); }
                    }
                    @keyframes loading-icon {
                        0% { transform: perspective(250px) rotateX(0deg) rotateY(0deg); }
                        15% { transform: perspective(250px) rotateX(0deg) rotateY(0deg); }
                        16% { transform: perspective(250px) rotateX(180deg) rotateY(0deg); }
                        50% { transform: perspective(250px) rotateX(180deg) rotateY(0deg); }
                        65% { transform: perspective(250px) rotateX(180deg) rotateY(0deg); }
                        66% { transform: perspective(250px) rotateX(180deg) rotateY(180deg); }
                        100% { transform: perspective(250px) rotateX(180deg) rotateY(180deg); }
                    }
                `}</style>
                
                <div className="u-loading">
                    <div className="u-loading__symbol shadow-[0_0_30px_rgba(59,130,246,0.6)]">
                        <div className="u-loading__content">
                             <svg viewBox="0 0 100 40" className="w-full h-full" fill="none" xmlns="http://www.w3.org/2000/svg">
                                <text x="50" y="28" textAnchor="middle" fontFamily="sans-serif" fontWeight="900" fontSize="28" fill="white" letterSpacing="2">ONYX</text>
                             </svg>
                        </div>
                    </div>
                </div>
            </div>
        );
    }

    return (
        <div className="fixed inset-0 z-50 bg-black/80 flex items-center justify-center p-8 backdrop-blur-sm animate-in fade-in">
            <div className="bg-surface w-full h-full rounded-2xl overflow-hidden flex flex-col border border-white/10 shadow-2xl relative">
                
                {/* Header / Close */}
                <div className="absolute top-4 right-4 z-50 flex gap-2">
                    <div className="relative">
                        <button 
                            onClick={() => setShowTools(!showTools)}
                            className={clsx(
                                "p-2 rounded-full transition-colors border",
                                showTools ? "bg-primary text-white border-primary" : "bg-black/50 hover:bg-black/70 text-white border-transparent"
                            )}
                            title="Tools"
                        >
                            <Wrench className="size-6 p-1" />
                        </button>
                        
                        {showTools && (
                            <div className="absolute right-0 top-full mt-2 w-48 bg-surface border border-white/10 rounded-lg shadow-xl p-2 animate-in fade-in slide-in-from-top-2">
                                <div className="text-xs font-bold text-white/40 uppercase mb-2 px-2">Replay Tools</div>
                                <button 
                                    onClick={() => setDistanceTool(!distanceTool)}
                                    className="w-full flex items-center justify-between p-2 rounded hover:bg-white/5 text-sm text-white transition-colors"
                                >
                                    <span>Distance Calc</span>
                                    {distanceTool && <Check className="size-4 text-primary" />}
                                </button>
                                <button 
                                    onClick={() => setAirportDistanceTool(!airportDistanceTool)}
                                    className="w-full flex items-center justify-between p-2 rounded hover:bg-white/5 text-sm text-white transition-colors"
                                >
                                    <span>Nearest Airport</span>
                                    {airportDistanceTool && <Check className="size-4 text-primary" />}
                                </button>
                            </div>
                        )}
                    </div>

                    <button 
                        onClick={onClose}
                        className="bg-black/50 hover:bg-black/70 text-white p-2 rounded-full transition-colors"
                    >
                        <X className="size-6" />
                    </button>
                </div>

                <div className="flex-1 flex overflow-hidden relative">
                    {/* Events Sidebar */}
                    <div className="w-72 bg-surface border-r border-white/10 flex flex-col z-30 shrink-0">
                        <div className="p-4 border-b border-white/10 bg-white/5">
                            <h3 className="font-bold text-white text-sm flex items-center gap-2">
                                <AlertTriangle className="size-4 text-primary" />
                                Event Log
                            </h3>
                        </div>
                        <div className="flex-1 overflow-y-auto p-2 space-y-2">
                            {events.length === 0 ? (
                                <div className="text-white/40 text-xs text-center mt-10 p-4 border border-dashed border-white/10 rounded-lg mx-2">
                                    No anomaly events logged for this replay.
                                </div>
                            ) : (
                                events.map((ev, i) => (
                                    <button 
                                        key={i}
                                        onClick={() => handleEventClick(ev)}
                                        className="w-full text-left bg-black/20 hover:bg-white/5 border border-white/5 hover:border-white/10 p-3 rounded-lg transition-all group"
                                    >
                                        <div className="flex items-center justify-between mb-1">
                                            <span className={clsx(
                                                "text-[10px] font-bold uppercase px-1.5 py-0.5 rounded",
                                                ev.type === 'proximity' ? "bg-red-500/20 text-red-300" :
                                                ev.type === 'deviation' ? "bg-orange-500/20 text-orange-300" :
                                                ev.type === 'ml_anomaly' ? "bg-purple-500/20 text-purple-300" :
                                                "bg-blue-500/20 text-blue-300"
                                            )}>
                                                {ev.type === 'ml_anomaly' ? 'ML' : ev.type}
                                            </span>
                                            <span className="font-mono text-[10px] text-white/40">
                                                {formatTime(ev.timestamp)}
                                            </span>
                                        </div>
                                        <p className="text-xs text-white/80 line-clamp-2 mb-2">{ev.description}</p>
                                        <div className="flex items-center gap-1 text-[10px] text-primary opacity-0 group-hover:opacity-100 transition-opacity">
                                            <MapPin className="size-3" />
                                            Jump to event
                                        </div>
                                    </button>
                                ))
                            )}
                        </div>
                    </div>

                    {/* Map Area */}
                    <div className="flex-1 relative">
                        {/* Live Telemetry Overlay */}
                        <div className="absolute top-4 left-4 z-20 flex flex-col gap-2 pointer-events-none">
                            {flights.map(f => {
                                const tel = getCurrentTelemetry(f);
                                if (!tel) return null;

                                                const distances = (tel.status !== 'waiting' && 'lat' in tel)
                                                    ? getDistancesToOthers(f.id, tel.lat, tel.lon)
                                                    : [];
                                                
                                                const nearestAirport = (tel.status !== 'waiting' && 'lat' in tel)
                                                    ? getNearestAirport(tel.lat, tel.lon)
                                                    : null;

                                return (
                                    <div key={f.id} className="bg-black/60 backdrop-blur border border-white/10 p-3 rounded-lg text-xs w-48 shadow-lg transition-all pointer-events-auto">
                                        <div className="flex items-center gap-2 mb-2 border-b border-white/10 pb-1">
                                            <div className="w-2 h-2 rounded-full" style={{ backgroundColor: f.color }} />
                                            <span className="font-bold text-white">{f.id}</span>
                                            {tel.status === 'waiting' && <span className="text-[10px] text-yellow-500 ml-auto italic">Waiting...</span>}
                                            {tel.status === 'ended' && <span className="text-[10px] text-red-400 ml-auto italic">Ended</span>}
                                        </div>
                                        {tel.status !== 'waiting' && (
                                            <div className={clsx("grid grid-cols-2 gap-1 text-white/80", tel.status === 'ended' && "opacity-50")}>
                                                <span>Alt:</span>
                                                <span className="font-mono text-right">{tel.alt} ft</span>
                                                <span>Hdg:</span>
                                                <span className="font-mono text-right">{tel.track}°</span>
                                                <span>Speed:</span>
                                                <span className="font-mono text-right">{tel.gspeed || 0} kts</span>
                                                
                                                <span className="col-span-2 border-t border-white/5 mt-1 pt-1 flex justify-between opacity-60">
                                                     <span>Lat/Lon:</span>
                                                     <span className="font-mono text-[10px]">{tel.lat.toFixed(3)}, {tel.lon.toFixed(3)}</span>
                                                </span>

                                                {distances.length > 0 && (
                                                    <div className="col-span-2 mt-2">
                                                        <div className="text-[10px] font-bold text-white/40 uppercase mb-1">Distances</div>
                                                        <div className="space-y-0.5">
                                                            {distances.map(d => (
                                                                <div key={d.id} className="flex justify-between items-center bg-white/5 px-1.5 py-0.5 rounded">
                                                                    <div className="flex items-center gap-1.5">
                                                                        <div className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: d.color }} />
                                                                        <span className="text-white/70">{d.id}</span>
                                                                    </div>
                                                                    <span className="font-mono font-bold text-primary">{d.dist.toFixed(1)} NM</span>
                                                                </div>
                                                            ))}
                                                        </div>
                                                    </div>
                                                )}

                                                {nearestAirport && (
                                                    <div className="col-span-2 mt-2">
                                                        <div className="text-[10px] font-bold text-white/40 uppercase mb-1">Nearest Airport</div>
                                                        <div className="flex justify-between items-center bg-emerald-500/10 px-1.5 py-1 rounded border border-emerald-500/20">
                                                            <div className="flex flex-col">
                                                                <span className="font-mono font-bold text-emerald-400">{nearestAirport.code}</span>
                                                                <span className="text-[9px] text-white/50 truncate max-w-[100px]">{nearestAirport.name}</span>
                                                            </div>
                                                            <span className="font-mono font-bold text-emerald-300">{nearestAirport.dist.toFixed(1)} NM</span>
                                                        </div>
                                                    </div>
                                                )}
                                            </div>
                                        )}
                                    </div>
                                );
                            })}
                        </div>

                        {/* Map */}
                        <div ref={mapContainer} className="w-full h-full bg-gray-900" />
                    </div>
                </div>

                {/* Controls */}
                <div className="bg-surface border-t border-white/10 p-4 z-40">
                    
                    {/* Time Slider */}
                    <div className="flex items-center gap-4 mb-4">
                        <span className="text-xs font-mono text-white/60 min-w-[80px]">{formatTime(currentTime)}</span>
                        <input 
                            type="range" 
                            min={minTime} 
                            max={maxTime} 
                            step={1}
                            value={currentTime}
                            onChange={handleSeek}
                            className="flex-1 h-2 bg-white/10 rounded-lg appearance-none cursor-pointer accent-primary hover:accent-primary/80"
                        />
                         <span className="text-xs font-mono text-white/60 min-w-[80px] text-right">{formatTime(maxTime)}</span>
                    </div>

                    {/* Buttons */}
                    <div className="flex items-center justify-between">
                        
                        <div className="flex items-center gap-2">
                             <div className="flex items-center gap-1 bg-white/5 rounded-lg p-1 border border-white/10">
                                <button 
                                    onClick={() => setSpeed(1)} 
                                    className={clsx("px-2 py-1 text-xs font-bold rounded", speed === 1 ? "bg-primary text-white" : "text-white/40 hover:text-white")}
                                >1x</button>
                                <button 
                                    onClick={() => setSpeed(5)} 
                                    className={clsx("px-2 py-1 text-xs font-bold rounded", speed === 5 ? "bg-primary text-white" : "text-white/40 hover:text-white")}
                                >5x</button>
                                <button 
                                    onClick={() => setSpeed(10)} 
                                    className={clsx("px-2 py-1 text-xs font-bold rounded", speed === 10 ? "bg-primary text-white" : "text-white/40 hover:text-white")}
                                >10x</button>
                                <button 
                                    onClick={() => setSpeed(20)} 
                                    className={clsx("px-2 py-1 text-xs font-bold rounded", speed === 20 ? "bg-primary text-white" : "text-white/40 hover:text-white")}
                                >20x</button>
                                <button 
                                    onClick={() => setSpeed(60)} 
                                    className={clsx("px-2 py-1 text-xs font-bold rounded", speed === 60 ? "bg-primary text-white" : "text-white/40 hover:text-white")}
                                >60x</button>
                             </div>
                        </div>

                        <div className="flex items-center gap-4">
                            <button 
                                onClick={() => setCurrentTime(minTime)}
                                className="text-white/60 hover:text-white transition-colors"
                            >
                                <SkipBack className="size-5" />
                            </button>
                            
                            <button 
                                onClick={() => setIsPlaying(!isPlaying)}
                                className="bg-primary hover:bg-primary/80 text-white p-3 rounded-full transition-transform active:scale-95 shadow-lg shadow-primary/20"
                            >
                                {isPlaying ? <Pause className="size-6 fill-current" /> : <Play className="size-6 fill-current pl-1" />}
                            </button>

                             <button 
                                onClick={() => setCurrentTime(maxTime)}
                                className="text-white/60 hover:text-white transition-colors"
                            >
                                <SkipForward className="size-5" />
                            </button>
                        </div>

                        <div className="flex flex-col items-end gap-1 min-w-[200px]">
                            {flights.map(f => (
                                <div key={f.id} className="flex items-center gap-2 text-xs">
                                    <div className="w-2 h-2 rounded-full" style={{ backgroundColor: f.color }} />
                                    <span className="text-white/80 font-mono">{f.id}</span>
                                </div>
                            ))}
                        </div>

                    </div>

                </div>

            </div>
        </div>
    );
};
