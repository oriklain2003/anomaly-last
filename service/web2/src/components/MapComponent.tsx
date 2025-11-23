import React, { useEffect, useRef } from 'react';
import maplibregl from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';

interface MapComponentProps {
  path: [number, number][];
}

export const MapComponent: React.FC<MapComponentProps> = ({ path }) => {
  const mapContainer = useRef<HTMLDivElement>(null);
  const map = useRef<maplibregl.Map | null>(null);
  const apiKey = 'r7kaQpfNDVZdaVp23F1r';

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
       // Cleanup handled by React refs mostly
    };
  }, []);

  useEffect(() => {
    if (!map.current) return;
    
    // If source isn't loaded yet, try again in a bit (simple polling)
    // Or just rely on the fact that 'load' event sets it up.
    // Better: check if source exists.
    if (!map.current.getSource('route')) {
        // If we are here, map might be loading. 
        // We can attach a one-time handler if not loaded, but simplicity:
        // just update when ready.
        return;
    }

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

    // Fit bounds
    const bounds = new maplibregl.LngLatBounds();
    path.forEach((coord) => bounds.extend(coord));
    map.current.fitBounds(bounds, { padding: 50 });

    // Add Start Marker
    new maplibregl.Marker({ color: "#10b981" })
      .setLngLat(path[0])
      .setPopup(new maplibregl.Popup().setHTML("Start"))
      .addTo(map.current);

    // Add End Marker
    new maplibregl.Marker({ color: "#ef4444" })
        .setLngLat(path[path.length - 1])
        .setPopup(new maplibregl.Popup().setHTML("End"))
        .addTo(map.current);

    // Update line directly (No animation to ensure reliability)
    source.setData({
        type: 'Feature',
        properties: {},
        geometry: {
            type: 'LineString',
            coordinates: path,
        },
    });

  }, [path]);

  return <div ref={mapContainer} className="w-full h-full" />;
};
