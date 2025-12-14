import { useEffect, useRef, useState, useImperativeHandle, forwardRef } from 'react';
import maplibregl from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';
import type { TrackPoint } from '../types';
import { fetchLearnedLayers, type LearnedLayers } from '../api';

// Fix for Hebrew text rendering (RTL)
try {
  if (maplibregl.getRTLTextPluginStatus() === 'unavailable') {
    maplibregl.setRTLTextPlugin(
      'https://unpkg.com/@mapbox/mapbox-gl-rtl-text@0.2.3/mapbox-gl-rtl-text.min.js',
      true // Lazy load
    );
  }
} catch (err) {
  console.error('Failed to set RTL text plugin', err);
}

// ============================================================
// AI Highlight Types
// ============================================================

export interface AIHighlightedPoint {
    lat: number;
    lon: number;
    label?: string;
}

export interface AIHighlightedSegment {
    startIndex: number;
    endIndex: number;
}

// ============================================================
// Component Props and Ref Handle
// ============================================================

// ML Anomaly Point for map display
export interface MLAnomalyPoint {
    lat: number;
    lon: number;
    timestamp: number;
    point_score: number;
    layer: string;  // e.g., 'Deep Dense', 'CNN', 'Transformer', 'Hybrid'
}

interface MapComponentProps {
  points: TrackPoint[];
  secondaryPoints?: TrackPoint[];
  anomalyTimestamps?: number[];
  mlAnomalyPoints?: MLAnomalyPoint[];
  aiHighlightedPoint?: AIHighlightedPoint | null;
  aiHighlightedSegment?: AIHighlightedSegment | null;
  onClearAIHighlights?: () => void;
}

export interface MapComponentHandle {
    getContainer: () => HTMLDivElement | null;
    flyTo: (lat: number, lon: number, zoom?: number) => void;
    fitBounds: (north: number, south: number, east: number, west: number) => void;
    highlightPoint: (lat: number, lon: number, label?: string) => void;
    highlightSegment: (startIndex: number, endIndex: number) => void;
    clearHighlights: () => void;
    captureScreenshot: () => Promise<string | null>;
}

// ============================================================
// Component Implementation
// ============================================================

export const MapComponent = forwardRef<MapComponentHandle, MapComponentProps>(({ 
    points, 
    secondaryPoints = [], 
    anomalyTimestamps = [],
    mlAnomalyPoints = [],
    aiHighlightedPoint,
    aiHighlightedSegment,
    onClearAIHighlights
}, ref) => {
  const mapContainer = useRef<HTMLDivElement>(null);
  const map = useRef<maplibregl.Map | null>(null);
  const aiMarkerRef = useRef<maplibregl.Marker | null>(null);
  const mlMarkersRef = useRef<maplibregl.Marker[]>([]);
  const apiKey = 'r7kaQpfNDVZdaVp23F1r';

  const [learnedLayers, setLearnedLayers] = useState<LearnedLayers | null>(null);
  const [showPaths, setShowPaths] = useState(false);
  const [showTurns, setShowTurns] = useState(false);
  const [showSids, setShowSids] = useState(false);
  const [showStars, setShowStars] = useState(false);
  const [showMLPoints, setShowMLPoints] = useState(true);

  // Expose imperative handle for parent components
  useImperativeHandle(ref, () => ({
    getContainer: () => mapContainer.current,
    
    captureScreenshot: async (): Promise<string | null> => {
        if (!map.current || !mapContainer.current) return null;
        
        try {
            // Wait for any pending renders to complete
            await new Promise<void>(resolve => {
                if (map.current!.loaded()) {
                    resolve();
                } else {
                    map.current!.once('idle', () => resolve());
                }
            });
            
            // Force a render
            map.current.triggerRepaint();
            
            // Wait a frame for the repaint
            await new Promise(resolve => requestAnimationFrame(resolve));
            
            // Get the WebGL canvas
            const mapCanvas = map.current.getCanvas();
            
            // Create a combined canvas with the map and overlays
            const rect = mapContainer.current.getBoundingClientRect();
            const combinedCanvas = document.createElement('canvas');
            combinedCanvas.width = rect.width;
            combinedCanvas.height = rect.height;
            const ctx = combinedCanvas.getContext('2d');
            
            if (!ctx) return null;
            
            // Draw the map canvas
            try {
                ctx.drawImage(mapCanvas, 0, 0, rect.width, rect.height);
            } catch (e) {
                console.error('Failed to draw map canvas (CORS issue?):', e);
                // Fill with dark background as fallback
                ctx.fillStyle = '#1a1a2e';
                ctx.fillRect(0, 0, rect.width, rect.height);
            }
            
            // Draw markers manually since they're DOM elements
            // Get all marker elements and draw them
            const markers = mapContainer.current.querySelectorAll('.maplibregl-marker');
            markers.forEach(marker => {
                const markerEl = marker as HTMLElement;
                const markerRect = markerEl.getBoundingClientRect();
                const containerRect = mapContainer.current!.getBoundingClientRect();
                
                // Calculate position relative to container
                const x = markerRect.left - containerRect.left + markerRect.width / 2;
                const y = markerRect.top - containerRect.top + markerRect.height;
                
                // Draw a simple marker representation
                ctx.beginPath();
                ctx.arc(x, y - 15, 10, 0, Math.PI * 2);
                
                // Check marker color (green for start, red for end)
                const svg = markerEl.querySelector('svg');
                if (svg) {
                    const fill = svg.getAttribute('fill') || '#ef4444';
                    ctx.fillStyle = fill;
                } else {
                    ctx.fillStyle = '#ef4444';
                }
                ctx.fill();
                ctx.strokeStyle = 'white';
                ctx.lineWidth = 2;
                ctx.stroke();
            });
            
            return combinedCanvas.toDataURL('image/png');
        } catch (e) {
            console.error('Screenshot capture failed:', e);
            return null;
        }
    },
    
    flyTo: (lat: number, lon: number, zoom?: number) => {
        if (map.current) {
            map.current.flyTo({
                center: [lon, lat],
                zoom: zoom ?? map.current.getZoom(),
                duration: 1000
            });
        }
    },
    
    fitBounds: (north: number, south: number, east: number, west: number) => {
        if (map.current) {
            map.current.fitBounds(
                [[west, south], [east, north]],
                { padding: 50, duration: 1000 }
            );
        }
    },
    
    highlightPoint: (lat: number, lon: number, label?: string) => {
        if (!map.current) return;
        
        // Remove existing AI marker
        if (aiMarkerRef.current) {
            aiMarkerRef.current.remove();
        }
        
        // Create pulsing marker element
        const el = document.createElement('div');
        el.className = 'ai-highlight-marker';
        el.innerHTML = `
            <div class="ai-marker-pulse"></div>
            <div class="ai-marker-center"></div>
        `;
        
        const marker = new maplibregl.Marker({ element: el })
            .setLngLat([lon, lat]);
        
        if (label) {
            marker.setPopup(new maplibregl.Popup({ offset: 25 }).setHTML(
                `<div class="text-gray-900 p-2 text-sm font-sans font-bold">${label}</div>`
            ));
        }
        
        marker.addTo(map.current);
        aiMarkerRef.current = marker;
        
        // Fly to the point
        map.current.flyTo({
            center: [lon, lat],
            zoom: Math.max(map.current.getZoom(), 10),
            duration: 1000
        });
    },
    
    highlightSegment: (startIndex: number, endIndex: number) => {
        if (!map.current || !points || points.length === 0) return;
        
        const source = map.current.getSource('ai-highlight-segment') as maplibregl.GeoJSONSource;
        if (!source) return;
        
        const start = Math.max(0, startIndex);
        const end = Math.min(points.length - 1, endIndex);
        
        const segmentPoints = points.slice(start, end + 1);
        
        source.setData({
            type: 'FeatureCollection',
            features: [{
                type: 'Feature',
                properties: {},
                geometry: {
                    type: 'LineString',
                    coordinates: segmentPoints.map(p => [p.lon, p.lat])
                }
            }]
        });
        
        // Fit to segment bounds
        if (segmentPoints.length > 0) {
            const bounds = new maplibregl.LngLatBounds();
            segmentPoints.forEach(p => bounds.extend([p.lon, p.lat]));
            map.current.fitBounds(bounds, { padding: 100, duration: 1000 });
        }
    },
    
    clearHighlights: () => {
        // Remove AI marker
        if (aiMarkerRef.current) {
            aiMarkerRef.current.remove();
            aiMarkerRef.current = null;
        }
        
        // Clear AI segment highlight
        if (map.current) {
            const source = map.current.getSource('ai-highlight-segment') as maplibregl.GeoJSONSource;
            if (source) {
                source.setData({ type: 'FeatureCollection', features: [] });
            }
        }
    }
  }), [points]);

  useEffect(() => {
    fetchLearnedLayers()
        .then(data => {
            setLearnedLayers(data);
            console.log("[MapComponent] Loaded learned layers:", {
                paths: data.paths?.length || 0,
                turns: data.turns?.length || 0,
                sids: data.sids?.length || 0,
                stars: data.stars?.length || 0
            });
        })
        .catch(err => console.error("[MapComponent] Failed to load learned layers", err));
  }, []);

  useEffect(() => {
    if (map.current || !mapContainer.current) return;

    map.current = new maplibregl.Map({
      container: mapContainer.current,
      style: `https://api.maptiler.com/maps/darkmatter/style.json?key=${apiKey}`,
      center: [34.8516, 31.0461], // Israel center
      zoom: 6,
      // @ts-expect-error - preserveDrawingBuffer is a valid WebGL option for screenshot capture
      preserveDrawingBuffer: true
    });

    map.current.on('load', () => {
      if (!map.current) return;

      map.current.addSource('route', {
        type: 'geojson',
        data: {
          type: 'FeatureCollection',
          features: []
        },
      });

      map.current.addLayer({
        id: 'route-line',
        type: 'line',
        source: 'route',
        layout: {
          'line-join': 'round',
          'line-cap': 'round'
        },
        paint: {
          'line-color': ['get', 'color'],
          'line-width': 4,
        },
      });

      // Secondary Route (Proximity)
      map.current.addSource('secondary-route', {
        type: 'geojson',
        data: { type: 'FeatureCollection', features: [] }
      });

      map.current.addLayer({
        id: 'secondary-route-line',
        type: 'line',
        source: 'secondary-route',
        layout: {
          'line-join': 'round',
          'line-cap': 'round'
        },
        paint: {
          'line-color': '#fb923c', // Orange-400
          'line-width': 3,
          'line-dasharray': [2, 2]
        },
      });

      // Secondary Points (for hover tooltip)
      map.current.addSource('secondary-points', {
        type: 'geojson',
        data: { type: 'FeatureCollection', features: [] }
      });

      map.current.addLayer({
        id: 'secondary-route-points',
        type: 'circle',
        source: 'secondary-points',
        paint: {
          'circle-radius': 3,
          'circle-color': '#fb923c',
          'circle-stroke-width': 1,
          'circle-stroke-color': '#ffffff'
        }
      });

      map.current.addSource('points', {
        type: 'geojson',
        data: {
          type: 'FeatureCollection',
          features: []
        },
      });

      map.current.addLayer({
        id: 'route-points',
        type: 'circle',
        source: 'points',
        paint: {
          'circle-radius': 4,
          'circle-color': ['get', 'color'],
          'circle-stroke-width': 1,
          'circle-stroke-color': '#ffffff'
        }
      });

      // AI Highlight Segment Layer
      map.current.addSource('ai-highlight-segment', {
        type: 'geojson',
        data: { type: 'FeatureCollection', features: [] }
      });

      map.current.addLayer({
        id: 'ai-highlight-segment-line',
        type: 'line',
        source: 'ai-highlight-segment',
        layout: {
          'line-join': 'round',
          'line-cap': 'round'
        },
        paint: {
          'line-color': '#f59e0b', // Amber-500
          'line-width': 8,
          'line-opacity': 0.8
        }
      });

      // AI Highlight Segment Glow (underneath)
      map.current.addLayer({
        id: 'ai-highlight-segment-glow',
        type: 'line',
        source: 'ai-highlight-segment',
        layout: {
          'line-join': 'round',
          'line-cap': 'round'
        },
        paint: {
          'line-color': '#fbbf24',
          'line-width': 16,
          'line-opacity': 0.3,
          'line-blur': 4
        }
      }, 'ai-highlight-segment-line'); // Insert below the main line

      const popup = new maplibregl.Popup({
        closeButton: false,
        closeOnClick: false
      });

      const showPopup = (e: any) => {
        if (!map.current) return;
        map.current.getCanvas().style.cursor = 'pointer';

        const coordinates = (e.features![0].geometry as any).coordinates.slice();
        const props = e.features![0].properties;

        while (Math.abs(e.lngLat.lng - coordinates[0]) > 180) {
            coordinates[0] += e.lngLat.lng > coordinates[0] ? 360 : -360;
        }

        const timeStr = new Date(props.timestamp * 1000).toLocaleTimeString();

        popup.setLngLat(coordinates)
            .setHTML(`
                <div class="text-gray-900 p-1 text-xs font-sans">
                    <div class="font-bold border-b border-gray-300 pb-1 mb-1">${timeStr}</div>
                    <div>Alt: <span class="font-mono font-bold">${props.alt}</span> ft</div>
                    <div>Hdg: <span class="font-mono font-bold">${props.track}°</span></div>
                </div>
            `)
            .addTo(map.current);
      };

      const hidePopup = () => {
        if (!map.current) return;
        map.current.getCanvas().style.cursor = '';
        popup.remove();
      };

      map.current.on('mouseenter', 'route-points', showPopup);
      map.current.on('mouseleave', 'route-points', hidePopup);
      
      map.current.on('mouseenter', 'secondary-route-points', showPopup);
      map.current.on('mouseleave', 'secondary-route-points', hidePopup);
    });

    return () => {
       // Cleanup handled by React refs mostly
    };
  }, []);

  // Effect to handle AI highlighted point from props
  useEffect(() => {
    if (!map.current) return;
    
    if (aiHighlightedPoint) {
        // Remove existing AI marker
        if (aiMarkerRef.current) {
            aiMarkerRef.current.remove();
        }
        
        // Create pulsing marker element
        const el = document.createElement('div');
        el.className = 'ai-highlight-marker';
        el.innerHTML = `
            <div class="ai-marker-pulse"></div>
            <div class="ai-marker-center"></div>
        `;
        
        const marker = new maplibregl.Marker({ element: el })
            .setLngLat([aiHighlightedPoint.lon, aiHighlightedPoint.lat]);
        
        if (aiHighlightedPoint.label) {
            marker.setPopup(new maplibregl.Popup({ offset: 25 }).setHTML(
                `<div class="text-gray-900 p-2 text-sm font-sans font-bold">${aiHighlightedPoint.label}</div>`
            ));
        }
        
        marker.addTo(map.current);
        aiMarkerRef.current = marker;
        
        // Fly to the point
        map.current.flyTo({
            center: [aiHighlightedPoint.lon, aiHighlightedPoint.lat],
            zoom: Math.max(map.current.getZoom(), 10),
            duration: 1000
        });
    } else {
        // Clear marker
        if (aiMarkerRef.current) {
            aiMarkerRef.current.remove();
            aiMarkerRef.current = null;
        }
    }
  }, [aiHighlightedPoint]);

  // Effect to handle AI highlighted segment from props
  useEffect(() => {
    if (!map.current || !map.current.isStyleLoaded()) return;
    
    const source = map.current.getSource('ai-highlight-segment') as maplibregl.GeoJSONSource;
    if (!source) return;
    
    if (aiHighlightedSegment && points && points.length > 0) {
        const start = Math.max(0, aiHighlightedSegment.startIndex);
        const end = Math.min(points.length - 1, aiHighlightedSegment.endIndex);
        
        const segmentPoints = points.slice(start, end + 1);
        
        source.setData({
            type: 'FeatureCollection',
            features: [{
                type: 'Feature',
                properties: {},
                geometry: {
                    type: 'LineString',
                    coordinates: segmentPoints.map(p => [p.lon, p.lat])
                }
            }]
        });
        
        // Fit to segment bounds
        if (segmentPoints.length > 0) {
            const bounds = new maplibregl.LngLatBounds();
            segmentPoints.forEach(p => bounds.extend([p.lon, p.lat]));
            map.current.fitBounds(bounds, { padding: 100, duration: 1000 });
        }
    } else {
        // Clear segment
        source.setData({ type: 'FeatureCollection', features: [] });
    }
  }, [aiHighlightedSegment, points]);

  // Effect to display ML anomaly point markers
  useEffect(() => {
    if (!map.current || !map.current.isStyleLoaded()) return;

    // Remove existing ML markers
    mlMarkersRef.current.forEach(marker => marker.remove());
    mlMarkersRef.current = [];

    if (!showMLPoints || !mlAnomalyPoints || mlAnomalyPoints.length === 0) return;

    // Layer color mapping
    const layerColors: Record<string, string> = {
        'Deep Dense': '#8b5cf6',    // Purple
        'Deep CNN': '#f97316',      // Orange
        'Transformer': '#06b6d4',   // Cyan
        'Hybrid': '#ec4899'         // Pink
    };

    mlAnomalyPoints.forEach((pt) => {
        const color = layerColors[pt.layer] || '#f59e0b';
        
        // Create marker element
        const el = document.createElement('div');
        el.className = 'ml-anomaly-marker';
        el.style.cssText = `
            width: 16px;
            height: 16px;
            background: ${color};
            border: 2px solid white;
            border-radius: 50%;
            box-shadow: 0 0 8px ${color}80;
            cursor: pointer;
        `;
        
        const marker = new maplibregl.Marker({ element: el })
            .setLngLat([pt.lon, pt.lat])
            .setPopup(new maplibregl.Popup({ offset: 15 }).setHTML(`
                <div class="text-gray-900 p-2 text-xs font-sans">
                    <div class="font-bold border-b border-gray-300 pb-1 mb-1" style="color: ${color}">
                        ${pt.layer} Anomaly
                    </div>
                    <div>Time: <span class="font-mono">${new Date(pt.timestamp * 1000).toLocaleTimeString()}</span></div>
                    <div>Score: <span class="font-mono font-bold">${pt.point_score.toFixed(4)}</span></div>
                    <div class="text-gray-500 mt-1">${pt.lat.toFixed(4)}, ${pt.lon.toFixed(4)}</div>
                </div>
            `))
            .addTo(map.current!);
        
        mlMarkersRef.current.push(marker);
    });
  }, [mlAnomalyPoints, showMLPoints]);

  // Learned Layers Visualization (Paths, Turns, SIDs, STARs)
  useEffect(() => {
    if (!map.current) return;

    // Helper to add a source and layer safely
    const addLayerSafe = (id: string, color: string, dashArray: number[], type: 'line' | 'fill' = 'line') => {
        if (!map.current!.getSource(id)) {
            try {
                map.current!.addSource(id, {
                    type: 'geojson',
                    data: { type: 'FeatureCollection', features: [] }
                });
                
                const beforeId = map.current!.getLayer('route-line') ? 'route-line' : undefined;

                const paint: any = type === 'line' ? {
                    'line-color': color,
                    'line-width': 2,
                    'line-opacity': 0.7,
                } : {
                    'fill-color': color,
                    'fill-opacity': 0.2,
                    'fill-outline-color': color
                };

                if (type === 'line' && dashArray && dashArray.length > 0) {
                    paint['line-dasharray'] = dashArray;
                }

                map.current!.addLayer({
                    id: `${id}-layer`,
                    type: type,
                    source: id,
                    layout: { 'visibility': 'visible' },
                    paint: paint
                } as any, beforeId);
            } catch (e) {
                // Layer already exists or other error - ignore
            }
        } 
    };

    // Create all layer sources
    addLayerSafe('learned-paths', '#4CAF50', [], 'line');           // Green solid for paths
    addLayerSafe('learned-turns', '#FF9800', [], 'fill');           // Orange fill for turns
    addLayerSafe('learned-sids', '#2196F3', [5, 5], 'line');        // Blue dashed for SIDs
    addLayerSafe('learned-stars', '#E91E63', [5, 5], 'line');       // Pink dashed for STARs

    // Helper to create circle polygon from center and radius
    const createCirclePolygon = (lat: number, lon: number, radiusNm: number) => {
        const radiusKm = radiusNm * 1.852;
        const numPoints = 32;
        const coords: [number, number][] = [];
        const distanceX = radiusKm / (111.320 * Math.cos(lat * Math.PI / 180));
        const distanceY = radiusKm / 110.574;

        for (let i = 0; i < numPoints; i++) {
            const theta = (i / numPoints) * (2 * Math.PI);
            const x = distanceX * Math.cos(theta);
            const y = distanceY * Math.sin(theta);
            coords.push([lon + x, lat + y]);
        }
        coords.push(coords[0]); // Close the polygon
        return coords;
    };

    // Update paths source
    const updatePaths = () => {
        const source = map.current!.getSource('learned-paths') as maplibregl.GeoJSONSource;
        if (!source) return;

        if (showPaths && learnedLayers?.paths && learnedLayers.paths.length > 0) {
            const features = learnedLayers.paths
                .filter(path => path.centerline && path.centerline.length >= 2)
                .map(path => ({
                    type: 'Feature' as const,
                    properties: { 
                        id: path.id,
                        origin: path.origin,
                        destination: path.destination,
                        member_count: path.member_count
                    },
                    geometry: {
                        type: 'LineString' as const,
                        coordinates: path.centerline.map(p => [p.lon, p.lat])
                    }
                }));
            source.setData({ type: 'FeatureCollection', features });
        } else {
            source.setData({ type: 'FeatureCollection', features: [] });
        }
    };

    // Update turns source
    const updateTurns = () => {
        const source = map.current!.getSource('learned-turns') as maplibregl.GeoJSONSource;
        if (!source) return;

        if (showTurns && learnedLayers?.turns && learnedLayers.turns.length > 0) {
            const features = learnedLayers.turns.map(turn => ({
                type: 'Feature' as const,
                properties: { 
                    id: turn.id,
                    avg_alt_ft: turn.avg_alt_ft,
                    radius_nm: turn.radius_nm,
                    member_count: turn.member_count
                },
                geometry: {
                    type: 'Polygon' as const,
                    coordinates: [createCirclePolygon(turn.lat, turn.lon, turn.radius_nm)]
                }
            }));
            source.setData({ type: 'FeatureCollection', features });
        } else {
            source.setData({ type: 'FeatureCollection', features: [] });
        }
    };

    // Update SIDs source
    const updateSids = () => {
        const source = map.current!.getSource('learned-sids') as maplibregl.GeoJSONSource;
        if (!source) return;

        if (showSids && learnedLayers?.sids && learnedLayers.sids.length > 0) {
            const features = learnedLayers.sids
                .filter(proc => proc.centerline && proc.centerline.length >= 2)
                .map(proc => ({
                    type: 'Feature' as const,
                    properties: { 
                        id: proc.id,
                        airport: proc.airport,
                        type: proc.type,
                        member_count: proc.member_count
                    },
                    geometry: {
                        type: 'LineString' as const,
                        coordinates: proc.centerline.map(p => [p.lon, p.lat])
                    }
                }));
            source.setData({ type: 'FeatureCollection', features });
        } else {
            source.setData({ type: 'FeatureCollection', features: [] });
        }
    };

    // Update STARs source
    const updateStars = () => {
        const source = map.current!.getSource('learned-stars') as maplibregl.GeoJSONSource;
        if (!source) return;

        if (showStars && learnedLayers?.stars && learnedLayers.stars.length > 0) {
            const features = learnedLayers.stars
                .filter(proc => proc.centerline && proc.centerline.length >= 2)
                .map(proc => ({
                    type: 'Feature' as const,
                    properties: { 
                        id: proc.id,
                        airport: proc.airport,
                        type: proc.type,
                        member_count: proc.member_count
                    },
                    geometry: {
                        type: 'LineString' as const,
                        coordinates: proc.centerline.map(p => [p.lon, p.lat])
                    }
                }));
            source.setData({ type: 'FeatureCollection', features });
        } else {
            source.setData({ type: 'FeatureCollection', features: [] });
        }
    };

    // Update all sources
    updatePaths();
    updateTurns();
    updateSids();
    updateStars();
  }, [showPaths, showTurns, showSids, showStars, learnedLayers]);

  useEffect(() => {
    if (!map.current) return;
    
    if (!map.current.getSource('route')) {
        return;
    }

    const source = map.current.getSource('route') as maplibregl.GeoJSONSource;
    const pointsSource = map.current.getSource('points') as maplibregl.GeoJSONSource;
    const secondarySource = map.current.getSource('secondary-route') as maplibregl.GeoJSONSource;
    const secondaryPointsSource = map.current.getSource('secondary-points') as maplibregl.GeoJSONSource;
    
    // Reset markers (but preserve AI marker)
    const markers = document.getElementsByClassName('maplibregl-marker');
    const markersToRemove: Element[] = [];
    for (let i = 0; i < markers.length; i++) {
        if (!markers[i].classList.contains('ai-highlight-marker') && 
            !markers[i].querySelector('.ai-highlight-marker')) {
            markersToRemove.push(markers[i]);
        }
    }
    markersToRemove.forEach(m => m.remove());

    if (points.length === 0) {
        source.setData({
            type: 'FeatureCollection',
            features: []
        });
        if (pointsSource) {
            pointsSource.setData({
                type: 'FeatureCollection',
                features: []
            });
        }
        if (secondarySource) {
            secondarySource.setData({
                type: 'FeatureCollection',
                features: []
            });
        }
        if (secondaryPointsSource) {
            secondaryPointsSource.setData({
                type: 'FeatureCollection',
                features: []
            });
        }
        return;
    }

    // Update Secondary Track
    if (secondarySource) {
        if (secondaryPoints && secondaryPoints.length > 0) {
            secondarySource.setData({
                type: 'FeatureCollection',
                features: [{
                    type: 'Feature',
                    properties: {},
                    geometry: {
                        type: 'LineString',
                        coordinates: secondaryPoints.map(p => [p.lon, p.lat])
                    }
                }] as any
            });

            if (secondaryPointsSource) {
                 secondaryPointsSource.setData({
                    type: 'FeatureCollection',
                    features: secondaryPoints.map(p => ({
                        type: 'Feature',
                        properties: {
                            timestamp: p.timestamp,
                            alt: p.alt,
                            track: p.track ?? 0
                        },
                        geometry: {
                            type: 'Point',
                            coordinates: [p.lon, p.lat]
                        }
                    })) as any
                });
            }

        } else {
            secondarySource.setData({
                type: 'FeatureCollection',
                features: []
            });
            if (secondaryPointsSource) {
                secondaryPointsSource.setData({
                    type: 'FeatureCollection',
                    features: []
                });
            }
        }
    }

    // Fit bounds (include secondary points if available)
    const bounds = new maplibregl.LngLatBounds();
    points.forEach((p) => bounds.extend([p.lon, p.lat]));
    if (secondaryPoints && secondaryPoints.length > 0) {
        secondaryPoints.forEach((p) => bounds.extend([p.lon, p.lat]));
    }
    map.current.fitBounds(bounds, { padding: 50 });

    // Add Start Marker
    new maplibregl.Marker({ color: "#10b981" })
      .setLngLat([points[0].lon, points[0].lat])
      .setPopup(new maplibregl.Popup().setHTML("Start"))
      .addTo(map.current);

    // Add End Marker
    new maplibregl.Marker({ color: "#ef4444" })
        .setLngLat([points[points.length - 1].lon, points[points.length - 1].lat])
        .setPopup(new maplibregl.Popup().setHTML("End"))
        .addTo(map.current);

    // Create segments
    const features: any[] = [];
    
    // Create a Set for O(1) lookup
    const anomalySet = new Set(anomalyTimestamps);

    for (let i = 0; i < points.length - 1; i++) {
        const p1 = points[i];
        const p2 = points[i+1];
        
        // Check if this segment involves an anomaly
        // We consider the segment anomalous if either point is anomalous
        const isAnomaly = anomalySet.has(p1.timestamp) || anomalySet.has(p2.timestamp);
        
        features.push({
            type: 'Feature',
            properties: {
                color: isAnomaly ? '#ef4444' : '#3b82f6' // Red if anomaly, Blue otherwise
            },
            geometry: {
                type: 'LineString',
                coordinates: [
                    [p1.lon, p1.lat],
                    [p2.lon, p2.lat]
                ]
            }
        });
    }

    source.setData({
        type: 'FeatureCollection',
        features: features
    });

    if (pointsSource) {
        const pointFeatures = points.map(p => {
            const isAnomaly = anomalySet.has(p.timestamp);
            return {
                type: 'Feature',
                properties: {
                    color: isAnomaly ? '#ef4444' : '#3b82f6',
                    timestamp: p.timestamp,
                    alt: p.alt,
                    track: p.track ?? 0
                },
                geometry: {
                    type: 'Point',
                    coordinates: [p.lon, p.lat]
                }
            };
        });

        pointsSource.setData({
            type: 'FeatureCollection',
            features: pointFeatures as any
        });
    }

  }, [points, secondaryPoints, anomalyTimestamps]);

  return (
    <div className="relative w-full h-full" id="map-root">
        <div ref={mapContainer} className="w-full h-full" />
        
        {/* Clear AI Highlights Button */}
        {(aiHighlightedPoint || aiHighlightedSegment) && onClearAIHighlights && (
            <button
                onClick={onClearAIHighlights}
                className="absolute top-4 left-4 z-10 px-3 py-2 rounded-lg shadow-lg text-xs font-medium 
                           bg-amber-500 text-white hover:bg-amber-600 transition-colors
                           flex items-center gap-2"
            >
                <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
                </svg>
                Clear AI Highlight
            </button>
        )}
        
        <div className="absolute top-4 right-4 z-10 flex flex-col gap-2">
            <button 
                onClick={() => setShowPaths(!showPaths)}
                className={`px-3 py-2 rounded shadow text-xs font-medium opacity-90 transition-colors ${
                    showPaths ? 'bg-green-600 text-white' : 'bg-gray-800 text-gray-300 hover:bg-gray-700'
                }`}
                title={`${learnedLayers?.paths?.length || 0} flight paths`}
            >
                {showPaths ? "Hide Paths" : "Show Paths"}
            </button>
            <button 
                onClick={() => setShowTurns(!showTurns)}
                className={`px-3 py-2 rounded shadow text-xs font-medium opacity-90 transition-colors ${
                    showTurns ? 'bg-orange-600 text-white' : 'bg-gray-800 text-gray-300 hover:bg-gray-700'
                }`}
                title={`${learnedLayers?.turns?.length || 0} turn zones`}
            >
                {showTurns ? "Hide Turns" : "Show Turns"}
            </button>
            <button 
                onClick={() => setShowSids(!showSids)}
                className={`px-3 py-2 rounded shadow text-xs font-medium opacity-90 transition-colors ${
                    showSids ? 'bg-blue-600 text-white' : 'bg-gray-800 text-gray-300 hover:bg-gray-700'
                }`}
                title={`${learnedLayers?.sids?.length || 0} SID procedures`}
            >
                {showSids ? "Hide SIDs" : "Show SIDs"}
            </button>
            <button 
                onClick={() => setShowStars(!showStars)}
                className={`px-3 py-2 rounded shadow text-xs font-medium opacity-90 transition-colors ${
                    showStars ? 'bg-pink-600 text-white' : 'bg-gray-800 text-gray-300 hover:bg-gray-700'
                }`}
                title={`${learnedLayers?.stars?.length || 0} STAR procedures`}
            >
                {showStars ? "Hide STARs" : "Show STARs"}
            </button>
            {mlAnomalyPoints && mlAnomalyPoints.length > 0 && (
                <button 
                    onClick={() => setShowMLPoints(!showMLPoints)}
                    className={`px-3 py-2 rounded shadow text-xs font-medium opacity-90 transition-colors ${
                        showMLPoints ? 'bg-purple-600 text-white' : 'bg-gray-800 text-gray-300 hover:bg-gray-700'
                    }`}
                >
                    {showMLPoints ? "Hide ML Points" : "Show ML Points"}
                </button>
            )}
        </div>
        
        {/* CSS for AI Highlight Marker Animation */}
        <style>{`
            .ai-highlight-marker {
                position: relative;
                width: 24px;
                height: 24px;
            }
            
            .ai-marker-center {
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                width: 12px;
                height: 12px;
                background: #f59e0b;
                border: 2px solid white;
                border-radius: 50%;
                box-shadow: 0 0 8px rgba(245, 158, 11, 0.8);
            }
            
            .ai-marker-pulse {
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                width: 24px;
                height: 24px;
                background: rgba(245, 158, 11, 0.4);
                border-radius: 50%;
                animation: ai-pulse 1.5s ease-out infinite;
            }
            
            @keyframes ai-pulse {
                0% {
                    transform: translate(-50%, -50%) scale(1);
                    opacity: 1;
                }
                100% {
                    transform: translate(-50%, -50%) scale(2.5);
                    opacity: 0;
                }
            }
        `}</style>
    </div>
  );
});

// Display name for debugging
MapComponent.displayName = 'MapComponent';
