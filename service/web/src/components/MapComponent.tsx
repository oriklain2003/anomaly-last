import React, { useEffect, useRef } from 'react';
import maplibregl from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';
import type { Airport } from '../types';

interface MapComponentProps {
  path: [number, number][];
}

// Hardcoded list of airports from rule_config.json
const AIRPORTS: Airport[] = [
    { code: "LLBG", name: "Ben Gurion Intl", lat: 32.011389, lon: 34.886667 },
    { code: "LLER", name: "Ramon Intl", lat: 29.723704, lon: 35.01145 },
    { code: "LLHA", name: "Haifa", lat: 32.809444, lon: 35.043056 },
    { code: "LLSD", name: "Sde Dov", lat: 32.114722, lon: 34.781944 },
    { code: "LLBS", name: "Beersheba", lat: 31.287, lon: 34.723 },
    { code: "LLET", name: "Eilat (J. Hozman)", lat: 29.561111, lon: 34.960833 },
    { code: "LLOV", name: "Ovda", lat: 29.940, lon: 34.935 },
    { code: "LLNV", name: "Nevatim AFB", lat: 31.207, lon: 35.012 },
    { code: "LLMG", name: "Megiddo", lat: 32.597, lon: 35.228 },
    { code: "LLHZ", name: "Herzliya", lat: 32.186, lon: 34.835 },
    { code: "LCRA", name: "RAF Akrotiri", lat: 34.5900, lon: 32.9870 },
    { code: "OLBA", name: "Beirut Rafic Hariri Intl", lat: 33.820889, lon: 35.488389 },
    { code: "OLKA", name: "Rayak Air Base", lat: 33.850, lon: 35.987 },
    { code: "OJAI", name: "Queen Alia Intl (Amman)", lat: 31.722556, lon: 35.993214 },
    { code: "OJAM", name: "Amman–Marka Intl", lat: 31.972, lon: 35.991 },
    { code: "OJAQ", "name": "King Hussein Intl (Aqaba)", lat: 29.611, lon: 35.018 },
    { code: "OJMF", name: "Mafraq", lat: 32.356, lon: 36.259 },
    { code: "HEGR", name: "El Gora Airport", lat: 31.0686, lon: 34.1296 },
    { code: "ALJAWZAH", name: "Al-Jawzah Airport", lat: 31.7288, lon: 52.3827 },
    { code: "OSDI", name: "Damascus Intl", lat: 33.411, lon: 36.516 }
];

export const MapComponent: React.FC<MapComponentProps> = ({ path }) => {
  const mapContainer = useRef<HTMLDivElement>(null);
  const map = useRef<maplibregl.Map | null>(null);
  const apiKey = 'r7kaQpfNDVZdaVp23F1r';

  useEffect(() => {
    if (map.current || !mapContainer.current) return;

    // Enable RTL text plugin for correct Hebrew rendering
    if (maplibregl.getRTLTextPluginStatus() === 'unavailable') {
        maplibregl.setRTLTextPlugin(
            'https://unpkg.com/@mapbox/mapbox-gl-rtl-text@0.2.3/mapbox-gl-rtl-text.min.js',
            () => {}, // Error callback (lazy load)
            true // Lazy load
        );
    }

    map.current = new maplibregl.Map({
      container: mapContainer.current,
      style: `https://api.maptiler.com/maps/darkmatter/style.json?key=${apiKey}`,
      center: [34.8516, 31.0461], // Israel center
      zoom: 6,
    });

    map.current.on('load', () => {
      if (!map.current) return;

      // Add Airports Source
      map.current.addSource('airports', {
          type: 'geojson',
          data: {
              type: 'FeatureCollection',
              features: AIRPORTS.map(airport => ({
                  type: 'Feature',
                  geometry: {
                      type: 'Point',
                      coordinates: [airport.lon, airport.lat]
                  },
                  properties: {
                      code: airport.code,
                      name: airport.name
                  }
              }))
          }
      });

      // Add Airports Layer (Circles)
      map.current.addLayer({
          id: 'airports-circle',
          type: 'circle',
          source: 'airports',
          paint: {
              'circle-radius': 4,
              'circle-color': '#ffffff',
              'circle-stroke-width': 1,
              'circle-stroke-color': '#000000'
          }
      });

      // Add Airports Labels
      map.current.addLayer({
          id: 'airports-label',
          type: 'symbol',
          source: 'airports',
          layout: {
              'text-field': ['get', 'code'],
              'text-font': ['Noto Sans Bold'],
              'text-size': 10,
              'text-offset': [0, 1], // below the dot
              'text-anchor': 'top'
          },
          paint: {
              'text-color': '#cccccc',
              'text-halo-color': '#000000',
              'text-halo-width': 1
          }
      });

      map.current.addSource('route', {
        type: 'geojson',
        data: {
          type: 'Feature',
          properties: {},
          geometry: {
            type: 'LineString',
            coordinates: [],
          },
        },
      });

      map.current.addLayer({
        id: 'route-line',
        type: 'line',
        source: 'route',
        paint: {
          'line-color': '#3b82f6',
          'line-width': 4,
        },
      });
    });

    return () => {
        // Clean up map instance on unmount? 
        // MapLibre sometimes has issues with strict mode double-mount, but we keep it ref-guarded.
        // map.current?.remove(); 
        // Usually better to keep it if we can, or remove it.
    };
  }, []);

  useEffect(() => {
    if (!map.current || !map.current.getSource('route')) return;

    const source = map.current.getSource('route') as maplibregl.GeoJSONSource;
    
    // Reset markers
    const markers = document.getElementsByClassName('maplibregl-marker');
    while (markers.length > 0) {
      markers[0].remove();
    }

    if (path.length === 0) {
        source.setData({
            type: 'Feature',
            properties: {},
            geometry: { type: 'LineString', coordinates: [] }
        });
        return;
    }

    // Fit bounds immediately so we see the animation
    const bounds = new maplibregl.LngLatBounds();
    path.forEach((coord) => bounds.extend(coord));
    map.current.fitBounds(bounds, { padding: 50 });

    // Add Start Marker
    new maplibregl.Marker({ color: "#10b981" })
      .setLngLat(path[0])
      .setPopup(new maplibregl.Popup().setHTML("Start"))
      .addTo(map.current);

    // Animation Logic
    let animationFrameId: number;
    let currentIndex = 0;
    // Target ~2 seconds for full animation
    // 60fps * 2s = 120 frames. 
    // Steps per frame = total_points / 120
    const stepsPerFrame = Math.max(1, Math.ceil(path.length / 120));

    const animateLine = () => {
        currentIndex += stepsPerFrame;
        if (currentIndex > path.length) currentIndex = path.length;

        const currentPath = path.slice(0, currentIndex);
        
        source.setData({
            type: 'Feature',
            properties: {},
            geometry: {
                type: 'LineString',
                coordinates: currentPath,
            },
        });

        if (currentIndex < path.length) {
            animationFrameId = requestAnimationFrame(animateLine);
        } else {
            // Animation Complete - Add End Marker
            new maplibregl.Marker({ color: "#ef4444" })
                .setLngLat(path[path.length - 1])
                .setPopup(new maplibregl.Popup().setHTML("End"))
                .addTo(map.current!);
        }
    };

    animateLine();

    return () => {
        if (animationFrameId) cancelAnimationFrame(animationFrameId);
    };
  }, [path]);

  return <div ref={mapContainer} className="w-full h-full" />;
};
