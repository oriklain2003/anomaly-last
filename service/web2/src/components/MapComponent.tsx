import React, { useEffect, useRef, useState } from 'react';
import maplibregl from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';
import type { TrackPoint } from '../types';
import { fetchLearnedPaths } from '../api';

interface MapComponentProps {
  points: TrackPoint[];
  anomalyTimestamps?: number[];
}

export const MapComponent: React.FC<MapComponentProps> = ({ points, anomalyTimestamps = [] }) => {
  const mapContainer = useRef<HTMLDivElement>(null);
  const map = useRef<maplibregl.Map | null>(null);
  const apiKey = 'r7kaQpfNDVZdaVp23F1r';

  const [learnedPaths, setLearnedPaths] = useState<any>(null);
  const [showStrict, setShowStrict] = useState(false);
  const [showLoose, setShowLoose] = useState(false);

  useEffect(() => {
    fetchLearnedPaths()
        .then(data => {
            console.log("[MapComponent] Fetched learned paths:", data);
            setLearnedPaths(data);
        })
        .catch(err => console.error("[MapComponent] Failed to load learned paths", err));
  }, []);

  useEffect(() => {
    if (map.current || !mapContainer.current) return;

    map.current = new maplibregl.Map({
      container: mapContainer.current,
      style: `https://api.maptiler.com/maps/darkmatter/style.json?key=${apiKey}`,
      center: [34.8516, 31.0461], // Israel center
      zoom: 6,
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

      const popup = new maplibregl.Popup({
        closeButton: false,
        closeOnClick: false
      });

      map.current.on('mouseenter', 'route-points', (e) => {
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
      });

      map.current.on('mouseleave', 'route-points', () => {
        if (!map.current) return;
        map.current.getCanvas().style.cursor = '';
        popup.remove();
      });
    });

    return () => {
       // Cleanup handled by React refs mostly
    };
  }, []);

  // Learned Paths Visualization
  useEffect(() => {
    if (!map.current) return;
    
    console.log("[MapComponent] Updating layers. ShowStrict:", showStrict, "ShowLoose:", showLoose, "Data Available:", !!learnedPaths);

    // Ideally we wait for style load, but simple check if getSource works usually suffices or we wrap in try-catch
    // or check map.current.loaded()
    if (!map.current.isStyleLoaded()) {
        console.log("[MapComponent] Map style not loaded yet, skipping layer update.");
        return;
    }

    const addLayerSafe = (id: string, color: string, dashArray: number[]) => {
        if (!map.current!.getSource(id)) {
            try {
                console.log(`[MapComponent] Adding source and layer: ${id}`);
                map.current!.addSource(id, {
                    type: 'geojson',
                    data: { type: 'FeatureCollection', features: [] }
                });
                
                const beforeId = map.current!.getLayer('route-line') ? 'route-line' : undefined;
                console.log(`[MapComponent] Inserting layer ${id} before: ${beforeId}`);
                
                const paint: any = {
                    'line-color': color,
                    'line-width': 2,
                    'line-opacity': 0.7,
                };

                if (dashArray && dashArray.length > 0) {
                    paint['line-dasharray'] = dashArray;
                }

                map.current!.addLayer({
                    id: `${id}-line`,
                    type: 'line',
                    source: id,
                    layout: { 'line-join': 'round', 'line-cap': 'round', 'visibility': 'visible' },
                    paint: paint
                }, beforeId);
            } catch (e) {
                console.error(`[MapComponent] Error adding ${id} layer`, e);
            }
        } else {
             console.log(`[MapComponent] Source ${id} already exists.`);
        }
    };

    addLayerSafe('paths-loose', '#a78bfa', [2, 2]);    // Purple dashed for loose
    addLayerSafe('paths-strict', '#10b981', []);       // Green solid for strict

    const updateSource = (id: string, visible: boolean, flows: any[]) => {
        const source = map.current!.getSource(id) as maplibregl.GeoJSONSource;
        if (!source) {
            console.warn(`[MapComponent] Source ${id} not found during update.`);
            return;
        }

        console.log(`[MapComponent] Updating ${id}: Visible=${visible}, Flows=${flows?.length}`);

        if (visible && flows) {
            const features = flows.map((flow: any) => ({
                type: 'Feature',
                properties: { flow_id: flow.flow_id },
                geometry: {
                    type: 'LineString',
                    coordinates: flow.centroid_path.map((p: any) => [p.lon, p.lat])
                }
            })) as any[];
            console.log(`[MapComponent] Setting ${features.length} features for ${id}`);
            source.setData({ type: 'FeatureCollection', features: features });
        } else {
            console.log(`[MapComponent] Clearing features for ${id}`);
            source.setData({ type: 'FeatureCollection', features: [] });
        }
    };

    if (learnedPaths && learnedPaths.layers) {
        const strictFlows = learnedPaths.layers.strict?.flows || [];
        const looseFlows = learnedPaths.layers.loose?.flows || [];
        
        updateSource('paths-strict', showStrict, strictFlows);
        updateSource('paths-loose', showLoose, looseFlows);
    } else {
        console.log("[MapComponent] No learned paths data structure found.");
    }

  }, [showStrict, showLoose, learnedPaths]);

  useEffect(() => {
    if (!map.current) return;
    
    if (!map.current.getSource('route')) {
        return;
    }

    const source = map.current.getSource('route') as maplibregl.GeoJSONSource;
    const pointsSource = map.current.getSource('points') as maplibregl.GeoJSONSource;
    
    // Reset markers
    const markers = document.getElementsByClassName('maplibregl-marker');
    while (markers.length > 0) {
      markers[0].remove();
    }

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
        return;
    }

    // Fit bounds
    const bounds = new maplibregl.LngLatBounds();
    points.forEach((p) => bounds.extend([p.lon, p.lat]));
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

  }, [points, anomalyTimestamps]);

  return (
    <div className="relative w-full h-full">
        <div ref={mapContainer} className="w-full h-full" />
        <div className="absolute top-4 right-4 z-10 flex flex-col gap-2">
            <button 
                onClick={() => setShowStrict(!showStrict)}
                className={`px-3 py-2 rounded shadow text-xs font-medium opacity-90 transition-colors ${
                    showStrict ? 'bg-green-600 text-white' : 'bg-gray-800 text-gray-300 hover:bg-gray-700'
                }`}
            >
                {showStrict ? "Hide Strict Paths" : "Show Strict Paths"}
            </button>
            <button 
                onClick={() => setShowLoose(!showLoose)}
                className={`px-3 py-2 rounded shadow text-xs font-medium opacity-90 transition-colors ${
                    showLoose ? 'bg-purple-600 text-white' : 'bg-gray-800 text-gray-300 hover:bg-gray-700'
                }`}
            >
                {showLoose ? "Hide Loose Paths" : "Show Loose Paths"}
            </button>
        </div>
    </div>
  );
};
