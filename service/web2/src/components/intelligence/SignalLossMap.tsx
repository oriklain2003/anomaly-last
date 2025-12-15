import { useEffect, useRef, useState } from 'react';
import maplibregl from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';
import type { SignalLossLocation } from '../../types';

interface SignalLossMapProps {
  locations: SignalLossLocation[];
  height?: number;
}

export function SignalLossMap({ locations, height = 400 }: SignalLossMapProps) {
  const mapContainer = useRef<HTMLDivElement>(null);
  const map = useRef<maplibregl.Map | null>(null);
  const markersRef = useRef<maplibregl.Marker[]>([]);
  const [mapLoaded, setMapLoaded] = useState(false);
  
  const apiKey = 'r7kaQpfNDVZdaVp23F1r'; // Same key as main app

  // Initialize map
  useEffect(() => {
    if (!mapContainer.current || map.current) return;

    map.current = new maplibregl.Map({
      container: mapContainer.current,
      style: `https://api.maptiler.com/maps/dataviz-dark/style.json?key=${apiKey}`,
      center: [35.0, 32.0], // Israel center
      zoom: 5,
      attributionControl: false,
      renderWorldCopies: false,  // Prevent world wrapping
      maxBounds: [[-30, -10], [100, 60]]  // Limit to Europe/Middle East/Africa region
    });

    map.current.addControl(
      new maplibregl.NavigationControl({ showCompass: false }),
      'top-right'
    );

    map.current.on('load', () => {
      setMapLoaded(true);
    });

    return () => {
      markersRef.current.forEach(m => m.remove());
      map.current?.remove();
      map.current = null;
    };
  }, []);

  // Update markers when locations change
  useEffect(() => {
    if (!map.current || !mapLoaded) return;

    // Remove existing markers
    markersRef.current.forEach(m => m.remove());
    markersRef.current = [];

    if (locations.length === 0) return;

    // Filter locations to only include those within the map's maxBounds
    // maxBounds is [[-30, -10], [100, 60]] - SW to NE corners
    // This prevents markers from being placed at the edge of the container
    const validLocations = locations.filter(loc => 
      loc.lat >= -10 && loc.lat <= 60 && 
      loc.lon >= -30 && loc.lon <= 100
    );
    
    // Sort by count (most events first) and limit to prevent overcrowding
    const displayLocations = validLocations
      .sort((a, b) => b.count - a.count)
      .slice(0, 30);

    // Calculate max count for intensity scaling
    const maxCount = Math.max(...displayLocations.map(l => l.count), 1);

    // Add heatmap-style circles for each signal loss location
    displayLocations.forEach(loc => {
      const intensity = loc.count / maxCount;
      const size = 20 + intensity * 40; // 20-60px based on intensity
      
      // Create wrapper element for the marker (MapLibre will control this element's transform)
      const wrapper = document.createElement('div');
      wrapper.className = 'signal-loss-marker-wrapper';
      
      // Create the actual visual element inside (this can have animation without affecting position)
      const el = document.createElement('div');
      el.className = 'signal-loss-marker';
      el.style.cssText = `
        width: ${size}px;
        height: ${size}px;
        border-radius: 50%;
        background: radial-gradient(circle, 
          rgba(239, 68, 68, ${0.6 + intensity * 0.3}) 0%, 
          rgba(239, 68, 68, ${0.2 + intensity * 0.2}) 50%,
          rgba(239, 68, 68, 0) 100%);
        border: 2px solid rgba(239, 68, 68, ${0.5 + intensity * 0.4});
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 11px;
        font-weight: bold;
        color: white;
        text-shadow: 0 1px 2px rgba(0,0,0,0.8);
        animation: signal-pulse 2s ease-in-out infinite;
      `;
      
      wrapper.appendChild(el);
      
      // Show count for significant clusters
      if (loc.count >= 3) {
        el.textContent = loc.count.toString();
      }

      // Create popup with details
      const popup = new maplibregl.Popup({
        offset: 25,
        closeButton: false,
        className: 'signal-loss-popup'
      }).setHTML(`
        <div style="padding: 8px; background: #1f2937; border-radius: 8px; color: white; min-width: 180px;">
          <div style="font-weight: bold; color: #ef4444; margin-bottom: 6px; font-size: 13px;">
            ⚠️ Signal Loss Zone
          </div>
          <div style="display: grid; gap: 4px; font-size: 12px;">
            <div style="display: flex; justify-content: space-between;">
              <span style="color: #9ca3af;">Location:</span>
              <span>${loc.lat.toFixed(3)}°N, ${loc.lon.toFixed(3)}°E</span>
            </div>
            <div style="display: flex; justify-content: space-between;">
              <span style="color: #9ca3af;">Events:</span>
              <span style="color: #ef4444; font-weight: bold;">${loc.count}</span>
            </div>
            <div style="display: flex; justify-content: space-between;">
              <span style="color: #9ca3af;">Avg Gap:</span>
              <span>${Math.round(loc.avgDuration)}s</span>
            </div>
          </div>
          <div style="margin-top: 8px; padding-top: 6px; border-top: 1px solid #374151; font-size: 11px; color: #6b7280;">
            Possible causes: GPS jamming, terrain, coverage gap
          </div>
        </div>
      `);

      const marker = new maplibregl.Marker({ 
        element: wrapper,
        anchor: 'center'  // Ensure marker is centered on the coordinate
      })
        .setLngLat([loc.lon, loc.lat])
        .setPopup(popup)
        .addTo(map.current!);

      markersRef.current.push(marker);
    });

    // Fit bounds to show markers - focus on Middle East region
    if (displayLocations.length > 0) {
      const bounds = new maplibregl.LngLatBounds();
      displayLocations.forEach(loc => bounds.extend([loc.lon, loc.lat]));
      
      // Add some padding around the bounds
      map.current.fitBounds(bounds, {
        padding: { top: 60, bottom: 60, left: 60, right: 60 },
        maxZoom: 6,
        minZoom: 3
      });
    }
  }, [locations, mapLoaded]);

  // Add airport markers for reference
  useEffect(() => {
    if (!map.current || !mapLoaded) return;

    const airports = [
      { name: 'LLBG', lat: 32.01, lon: 34.89, label: 'Ben Gurion' },
      { name: 'LLER', lat: 29.94, lon: 35.00, label: 'Ramon' },
      { name: 'LLHA', lat: 32.81, lon: 35.04, label: 'Haifa' },
      { name: 'LLSD', lat: 32.11, lon: 34.78, label: 'Sde Dov' },
      { name: 'OJAI', lat: 31.72, lon: 35.99, label: 'Amman' },
      { name: 'OLBA', lat: 33.82, lon: 35.49, label: 'Beirut' },
      { name: 'LCLK', lat: 34.88, lon: 33.62, label: 'Larnaca' }
    ];

    airports.forEach(apt => {
      const el = document.createElement('div');
      el.style.cssText = `
        width: 10px;
        height: 10px;
        background: #10b981;
        border: 2px solid white;
        border-radius: 50%;
        cursor: pointer;
      `;

      const popup = new maplibregl.Popup({
        offset: 15,
        closeButton: false
      }).setHTML(`
        <div style="padding: 4px 8px; background: #1f2937; border-radius: 4px; color: white; font-size: 12px;">
          <strong style="color: #10b981;">${apt.name}</strong> - ${apt.label}
        </div>
      `);

      new maplibregl.Marker({ element: el, anchor: 'center' })
        .setLngLat([apt.lon, apt.lat])
        .setPopup(popup)
        .addTo(map.current!);
    });
  }, [mapLoaded]);

  return (
    <div className="relative rounded-lg overflow-hidden border border-white/10">
      <div 
        ref={mapContainer} 
        style={{ height: `${height}px`, width: '100%' }}
      />
      
      {/* Legend */}
      <div className="absolute bottom-3 left-3 bg-black/70 backdrop-blur-sm rounded-lg p-3 text-xs">
        <div className="flex items-center gap-2 mb-2">
          <div className="w-4 h-4 rounded-full bg-red-500/60 border border-red-500" />
          <span className="text-white">Signal Loss Zone</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-emerald-500 border border-white" />
          <span className="text-white">Airport</span>
        </div>
        <div className="text-white/50 mt-2 text-[10px]">
          Circle size = event frequency
        </div>
      </div>

      {/* No data overlay */}
      {locations.length === 0 && mapLoaded && (
        <div className="absolute inset-0 flex items-center justify-center bg-black/40">
          <div className="text-white/60 text-center">
            <div className="text-2xl mb-2">📡</div>
            <p>No signal loss zones detected</p>
            <p className="text-sm text-white/40">in selected time range</p>
          </div>
        </div>
      )}

      {/* CSS for pulse animation - uses signal-pulse to avoid conflicts with MapLibre transforms */}
      <style>{`
        @keyframes signal-pulse {
          0%, 100% { transform: scale(1); opacity: 1; }
          50% { transform: scale(1.1); opacity: 0.8; }
        }
        .maplibregl-popup-content {
          background: transparent !important;
          padding: 0 !important;
          box-shadow: none !important;
        }
        .maplibregl-popup-tip {
          display: none !important;
        }
      `}</style>
    </div>
  );
}

