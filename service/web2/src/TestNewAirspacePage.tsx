import { useEffect, useRef, useState } from 'react';
import maplibregl from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';

type PathEntry = { id: number; centroid: any; member_flights?: string[]; path_length_nm?: number; confidence?: number };
type TurnEntry = { id: number; sample_locations?: { lat: number; lon: number; alt?: number }[]; member_count?: number; confidence?: number };
type HoldingEntry = { id: number; center?: { radius_nm?: number; alt?: number; speed?: number }; hotspots?: { lat: number; lon: number }[]; member_count?: number; confidence?: number };
type SidStarEntry = { id: number; centroid: any; member_flights?: string[]; confidence?: number };
type RegionEntry = { description?: string; polygon?: [number, number][] };

type LoadedCounts = {
  paths?: number;
  turns?: number;
  holdings?: number;
  sids?: number;
  stars?: number;
  regions?: number;
};

const MAPTILER_KEY = 'r7kaQpfNDVZdaVp23F1r';
const DEFAULT_BASE = typeof window !== 'undefined'
  ? `${window.location.origin}/output/`
  : 'http://localhost:8000/output/';

export function TestNewAirspacePage() {
  const mapRef = useRef<HTMLDivElement | null>(null);
  const map = useRef<maplibregl.Map | null>(null);
  const bounds = useRef<maplibregl.LngLatBounds | null>(null);

  const [status, setStatus] = useState('Set a base URL or load files (manual file inputs avoid CORS).');
  const [baseUrl, setBaseUrl] = useState(DEFAULT_BASE);
  const [mapReady, setMapReady] = useState(false);
  const [showLayers, setShowLayers] = useState({
    paths: true,
    turns: true,
    holdings: true,
    sids: true,
    stars: true,
    regions: true,
    wkt: true,
  });
  const [counts, setCounts] = useState<LoadedCounts>({});
  const [wktInput, setWktInput] = useState('');
  const [wktFeatures, setWktFeatures] = useState<any[]>([]);

  useEffect(() => {
    if (map.current || !mapRef.current) return;

    map.current = new maplibregl.Map({
      container: mapRef.current,
      style: `https://api.maptiler.com/maps/darkmatter/style.json?key=${MAPTILER_KEY}`,
      center: [35, 31.5],
      zoom: 6,
    });

    map.current.addControl(new maplibregl.NavigationControl(), 'top-left');

    map.current.on('load', () => {
      bounds.current = new maplibregl.LngLatBounds();
      addBaseSourcesAndLayers();
      setMapReady(true);
      setStatus('Map ready. Load JSONs to visualize.');
    });
  }, []);

  const addBaseSourcesAndLayers = () => {
    if (!map.current) return;

    const ensureSource = (id: string, type: any) => {
      if (!map.current!.getSource(id)) {
        map.current!.addSource(id, type);
      }
    };

    ensureSource('paths', { type: 'geojson', data: emptyFeatureCollection() });
    ensureSource('turns', { type: 'geojson', data: emptyFeatureCollection() });
    ensureSource('holdings', { type: 'geojson', data: emptyFeatureCollection() });
    ensureSource('sids', { type: 'geojson', data: emptyFeatureCollection() });
    ensureSource('stars', { type: 'geojson', data: emptyFeatureCollection() });
    ensureSource('regions', { type: 'geojson', data: emptyFeatureCollection() });
    ensureSource('wkt', { type: 'geojson', data: emptyFeatureCollection() });

    const addLayerIfMissing = (id: string, opts: any, before?: string) => {
      if (!map.current!.getLayer(id)) {
        map.current!.addLayer(opts, before);
      }
    };

    addLayerIfMissing('paths-layer', {
      id: 'paths-layer',
      type: 'line',
      source: 'paths',
      paint: { 'line-color': '#22d3ee', 'line-width': 3, 'line-opacity': 0.8 },
      layout: { visibility: showLayers.paths ? 'visible' : 'none' },
    });

    addLayerIfMissing('turns-layer', {
      id: 'turns-layer',
      type: 'circle',
      source: 'turns',
      paint: {
        'circle-radius': 4,
        'circle-color': '#f97316',
        'circle-stroke-color': '#fff',
        'circle-stroke-width': 1,
        'circle-opacity': 0.8,
      },
      layout: { visibility: showLayers.turns ? 'visible' : 'none' },
    });

    addLayerIfMissing('holdings-layer', {
      id: 'holdings-layer',
      type: 'fill',
      source: 'holdings',
      paint: {
        'fill-color': '#a855f7',
        'fill-opacity': 0.15,
        'fill-outline-color': '#a855f7',
      },
      layout: { visibility: showLayers.holdings ? 'visible' : 'none' },
    });

    addLayerIfMissing('sids-layer', {
      id: 'sids-layer',
      type: 'line',
      source: 'sids',
      paint: { 'line-color': '#4ade80', 'line-width': 2.5, 'line-opacity': 0.9 },
      layout: { visibility: showLayers.sids ? 'visible' : 'none' },
    });

    addLayerIfMissing('stars-layer', {
      id: 'stars-layer',
      type: 'line',
      source: 'stars',
      paint: {
        'line-color': '#f59e0b',
        'line-width': 2.5,
        'line-opacity': 0.9,
        'line-dasharray': [3, 2],
      },
      layout: { visibility: showLayers.stars ? 'visible' : 'none' },
    });

    addLayerIfMissing('regions-layer', {
      id: 'regions-layer',
      type: 'fill',
      source: 'regions',
      paint: {
        'fill-color': '#ef4444',
        'fill-opacity': 0.12,
        'fill-outline-color': '#ef4444',
      },
      layout: { visibility: showLayers.regions ? 'visible' : 'none' },
    });

    addLayerIfMissing('wkt-fill', {
      id: 'wkt-fill',
      type: 'fill',
      source: 'wkt',
      paint: {
        'fill-color': '#22c55e',
        'fill-opacity': 0.2,
        'fill-outline-color': '#22c55e',
      },
      layout: { visibility: showLayers.wkt ? 'visible' : 'none' },
    });

    addLayerIfMissing('wkt-line', {
      id: 'wkt-line',
      type: 'line',
      source: 'wkt',
      paint: {
        'line-color': '#22c55e',
        'line-width': 3,
        'line-opacity': 0.8,
      },
      layout: { visibility: showLayers.wkt ? 'visible' : 'none' },
    });
  };

  const emptyFeatureCollection = () => ({ type: 'FeatureCollection', features: [] as any[] });

  const setVisibility = () => {
    if (!map.current || !mapReady || !map.current.isStyleLoaded()) return;
    map.current.setLayoutProperty('paths-layer', 'visibility', showLayers.paths ? 'visible' : 'none');
    map.current.setLayoutProperty('turns-layer', 'visibility', showLayers.turns ? 'visible' : 'none');
    map.current.setLayoutProperty('holdings-layer', 'visibility', showLayers.holdings ? 'visible' : 'none');
    map.current.setLayoutProperty('sids-layer', 'visibility', showLayers.sids ? 'visible' : 'none');
    map.current.setLayoutProperty('stars-layer', 'visibility', showLayers.stars ? 'visible' : 'none');
    map.current.setLayoutProperty('regions-layer', 'visibility', showLayers.regions ? 'visible' : 'none');
    map.current.setLayoutProperty('wkt-fill', 'visibility', showLayers.wkt ? 'visible' : 'none');
    map.current.setLayoutProperty('wkt-line', 'visibility', showLayers.wkt ? 'visible' : 'none');
  };

  useEffect(setVisibility, [showLayers, mapReady]);

  const loadAllFromBase = async () => {
    setStatus('Loading all JSONs…');
    try {
      const [paths, turns, holdings, sidStar, regions] = await Promise.all([
        fetchJson('path_library.json'),
        fetchJson('turn_library.json'),
        fetchJson('holding_library.json'),
        fetchJson('sid_star_library.json'),
        fetchJson('airspace_semantics.json'),
      ]);

      drawPaths(paths?.library || []);
      drawTurns(turns?.library || []);
      drawHoldings(holdings?.library || []);
      drawSidStar(sidStar?.sid || [], 'sids');
      drawSidStar(sidStar?.star || [], 'stars');
      drawRegions(regions?.regions || {});

      setCounts({
        paths: paths?.library?.length || 0,
        turns: turns?.library?.length || 0,
        holdings: holdings?.library?.length || 0,
        sids: sidStar?.sid?.length || 0,
        stars: sidStar?.star?.length || 0,
        regions: Object.keys(regions?.regions || {}).length,
      });
      fitBounds();
      setStatus('All layers loaded from base URL.');
    } catch (err: any) {
      console.error(err);
      const msg = (err?.message || '').toLowerCase().includes('fetch')
        ? 'Failed to load (maybe CORS). Try a same-origin base URL or use the file pickers below.'
        : `Failed to load: ${err?.message || err}`;
      setStatus(msg);
    }
  };

  const fetchJson = async (name: string) => {
    const url = baseUrl.replace(/\/+$/, '/') + name;
    const res = await fetch(url);
    if (!res.ok) throw new Error(`${name} HTTP ${res.status}`);
    return res.json();
  };

  const handleFileInput = async (e: React.ChangeEvent<HTMLInputElement>, kind: 'paths' | 'turns' | 'holdings' | 'sids' | 'stars' | 'regions') => {
    const file = e.target.files?.[0];
    if (!file) return;
    const text = await file.text();
    const json = JSON.parse(text);
    switch (kind) {
      case 'paths':
        drawPaths(json.library || []);
        setCounts(c => ({ ...c, paths: json.library?.length || 0 }));
        break;
      case 'turns':
        drawTurns(json.library || []);
        setCounts(c => ({ ...c, turns: json.library?.length || 0 }));
        break;
      case 'holdings':
        drawHoldings(json.library || []);
        setCounts(c => ({ ...c, holdings: json.library?.length || 0 }));
        break;
      case 'sids':
        drawSidStar(json.sid || [], 'sids');
        setCounts(c => ({ ...c, sids: json.sid?.length || 0 }));
        break;
      case 'stars':
        drawSidStar(json.star || [], 'stars');
        setCounts(c => ({ ...c, stars: json.star?.length || 0 }));
        break;
      case 'regions':
        drawRegions(json.regions || {});
        setCounts(c => ({ ...c, regions: Object.keys(json.regions || {}).length }));
        break;
    }
    fitBounds();
    setStatus(`Loaded ${file.name}`);
  };

  const drawPaths = (entries: PathEntry[]) => {
    if (!map.current) return;
    const features = entries.map((path) => {
      const coords = triplesToLngLat(path.centroid);
      coords.forEach(([lon, lat]) => extendBounds(lat, lon));
      return {
        type: 'Feature',
        properties: {
          id: path.id,
          members: path.member_flights?.length ?? 0,
          length_nm: path.path_length_nm ?? 0,
          confidence: path.confidence ?? 0,
        },
        geometry: { type: 'LineString', coordinates: coords },
      };
    });
    updateSource('paths', features);
  };

  const drawTurns = (entries: TurnEntry[]) => {
    if (!map.current) return;
    const features = entries.flatMap((turn) =>
      (turn.sample_locations || []).map((pt) => {
        if (!isFinite(pt.lat) || !isFinite(pt.lon)) return null;
        extendBounds(pt.lat, pt.lon);
        return {
          type: 'Feature',
          properties: {
            id: turn.id,
            alt: pt.alt ?? 0,
            members: turn.member_count ?? 0,
            confidence: turn.confidence ?? 0,
          },
          geometry: { type: 'Point', coordinates: [pt.lon, pt.lat] },
        };
      }).filter(Boolean) as any[]
    );
    updateSource('turns', features);
  };

  const drawHoldings = (entries: HoldingEntry[]) => {
    if (!map.current) return;
    const features: any[] = [];
    entries.forEach((holding) => {
      const radiusNm = holding.center?.radius_nm ?? 1;
      (holding.hotspots || []).forEach((spot) => {
        if (!isFinite(spot.lat) || !isFinite(spot.lon)) return;
        extendBounds(spot.lat, spot.lon);
        const poly = circlePolygon([spot.lon, spot.lat], radiusNm);
        features.push({
          type: 'Feature',
          properties: {
            id: holding.id,
            radius_nm: radiusNm,
            alt: holding.center?.alt ?? 0,
            speed: holding.center?.speed ?? 0,
            members: holding.member_count ?? 0,
            confidence: holding.confidence ?? 0,
          },
          geometry: { type: 'Polygon', coordinates: [poly] },
        });
      });
    });
    updateSource('holdings', features);
  };

  const drawSidStar = (entries: SidStarEntry[], target: 'sids' | 'stars') => {
    if (!map.current) return;
    const features = entries.map((entry) => {
      const coords = triplesToLngLat(entry.centroid);
      coords.forEach(([lon, lat]) => extendBounds(lat, lon));
      return {
        type: 'Feature',
        properties: {
          id: entry.id,
          members: entry.member_flights?.length ?? 0,
          confidence: entry.confidence ?? 0,
        },
        geometry: { type: 'LineString', coordinates: coords },
      };
    });
    updateSource(target, features);
  };

  const drawRegions = (regions: Record<string, RegionEntry>) => {
    if (!map.current) return;
    const features = Object.entries(regions).map(([name, region]) => {
      const coords: [number, number][] = (region.polygon || []).map(([lat, lon]): [number, number] => {
        extendBounds(lat, lon);
        return [lon, lat];
      });
      return {
        type: 'Feature',
        properties: { name, description: region.description || '' },
        geometry: { type: 'Polygon', coordinates: [coords] },
      };
    });
    updateSource('regions', features);
  };

  const addWktToMap = () => {
    if (!wktInput.trim()) return;
    const feature = wktToFeature(wktInput.trim());
    if (!feature) {
      setStatus('Invalid or unsupported WKT. Supported: POINT, LINESTRING, POLYGON.');
      return;
    }
    const features = [...wktFeatures, feature];
    setWktFeatures(features);
    updateSource('wkt', features);
    fitBounds();
    setStatus('WKT added to map.');
  };

  const updateSource = (id: string, features: any[]) => {
    if (!map.current) return;
    const src = map.current.getSource(id) as maplibregl.GeoJSONSource;
    if (!src) return;
    src.setData({ type: 'FeatureCollection', features });
  };

  const triplesToLngLat = (raw: any): [number, number][] => {
    if (!raw) return [];
    if (Array.isArray(raw) && Array.isArray(raw[0])) {
      const coords: [number, number][] = [];
      for (const t of raw) {
        const lat = Number(t[0]);
        const lon = Number(t[1]);
        if (isFinite(lat) && isFinite(lon)) coords.push([lon, lat]);
      }
      return coords;
    }
    const coords: [number, number][] = [];
    for (let i = 0; i < raw.length; i += 3) {
      const lat = Number(raw[i]);
      const lon = Number(raw[i + 1]);
      if (isFinite(lat) && isFinite(lon)) coords.push([lon, lat]);
    }
    return coords;
  };

  const circlePolygon = ([lon, lat]: [number, number], radiusNm: number, steps = 48) => {
    const radiusKm = radiusNm * 1.852;
    const coords: [number, number][] = [];
    const dLat = radiusKm / 110.574;
    const dLon = radiusKm / (111.320 * Math.cos((lat * Math.PI) / 180));
    for (let i = 0; i <= steps; i++) {
      const theta = (i / steps) * 2 * Math.PI;
      coords.push([lon + dLon * Math.cos(theta), lat + dLat * Math.sin(theta)]);
    }
    return coords;
  };

  const extendBounds = (lat: number, lon: number) => {
    if (!bounds.current) bounds.current = new maplibregl.LngLatBounds();
    if (isFinite(lat) && isFinite(lon)) bounds.current.extend([lon, lat]);
  };

  const fitBounds = () => {
    if (map.current && bounds.current && !bounds.current.isEmpty()) {
      map.current.fitBounds(bounds.current, { padding: 40, duration: 600 });
    }
  };

  const wktToFeature = (wkt: string): any | null => {
    const clean = wkt.trim().toUpperCase();
    const parseCoords = (body: string): [number, number][] => {
      const coords: [number, number][] = [];
      body.split(',').forEach(pair => {
        const [lonStr, latStr] = pair.trim().split(/\s+/);
        const lon = Number(lonStr);
        const lat = Number(latStr);
        if (isFinite(lon) && isFinite(lat)) {
          extendBounds(lat, lon);
          coords.push([lon, lat]);
        }
      });
      return coords;
    };

    if (clean.startsWith('POINT')) {
      const match = wkt.match(/POINT\s*\(\s*([^\)]+)\)/i);
      if (!match) return null;
      const [lonStr, latStr] = match[1].trim().split(/\s+/);
      const lon = Number(lonStr);
      const lat = Number(latStr);
      if (!isFinite(lon) || !isFinite(lat)) return null;
      extendBounds(lat, lon);
      return {
        type: 'Feature',
        properties: { source: 'wkt' },
        geometry: { type: 'Point', coordinates: [lon, lat] },
      };
    }

    if (clean.startsWith('LINESTRING')) {
      const match = wkt.match(/LINESTRING\s*\(\s*([^\)]+)\)/i);
      if (!match) return null;
      const coords = parseCoords(match[1]);
      if (coords.length < 2) return null;
      return {
        type: 'Feature',
        properties: { source: 'wkt' },
        geometry: { type: 'LineString', coordinates: coords },
      };
    }

    if (clean.startsWith('POLYGON')) {
      const match = wkt.match(/POLYGON\s*\(\s*\(\s*([^\)]+)\)\s*\)/i);
      if (!match) return null;
      const coords = parseCoords(match[1]);
      if (coords.length < 3) return null;
      const ring = coords[0][0] === coords[coords.length - 1][0] && coords[0][1] === coords[coords.length - 1][1]
        ? coords
        : [...coords, coords[0]];
      return {
        type: 'Feature',
        properties: { source: 'wkt' },
        geometry: { type: 'Polygon', coordinates: [ring] },
      };
    }

    return null;
  };

  const toggleLayer = (key: keyof typeof showLayers) => {
    setShowLayers((prev) => ({ ...prev, [key]: !prev[key] }));
  };

  return (
    <div className="min-h-screen bg-background-dark text-white flex flex-col">
      <header className="p-4 border-b border-white/10 flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="text-2xl font-semibold">Test New Airspace</h1>
          <p className="text-sm text-white/70">
            Load behavior model outputs onto the map with dedicated layers. If CORS blocks remote fetches, use a same-origin URL
            (e.g. move output/ into this app’s public/ folder or serve it on the same port) or load the JSON files via the pickers below.
          </p>
        </div>
        <div className="flex flex-wrap gap-2 items-center">
          <input
            className="bg-surface border border-white/10 rounded px-2 py-1 text-sm w-64"
            value={baseUrl}
            onChange={(e) => setBaseUrl(e.target.value)}
            placeholder="Base URL (e.g. http://localhost:8000/output/)"
          />
          <button
            onClick={loadAllFromBase}
            className="bg-primary/80 hover:bg-primary text-black font-semibold px-3 py-2 rounded border border-white/10"
          >
            Load all from base URL
          </button>
        </div>
      </header>

      <div className="flex flex-col md:flex-row flex-1 overflow-hidden">
        <aside className="w-full md:w-80 bg-surface border-r border-white/5 p-4 space-y-4 overflow-y-auto">
          <div className="text-sm text-white/80 border border-white/10 rounded p-3 bg-background-dark/60">
            Status: <span className="font-semibold text-primary">{status}</span>
          </div>

          <div className="space-y-3">
            <h2 className="text-lg font-semibold">Manual load</h2>
            {[
              ['Path Library', 'paths'] as const,
              ['Turn Library', 'turns'] as const,
              ['Holding Library', 'holdings'] as const,
              ['SID Library', 'sids'] as const,
              ['STAR Library', 'stars'] as const,
              ['Airspace Semantics', 'regions'] as const,
            ].map(([label, key]) => (
              <label key={key} className="flex flex-col gap-1 text-sm">
                <span className="text-white/80">{label}</span>
                <input
                  type="file"
                  accept=".json,application/json"
                  onChange={(e) => handleFileInput(e, key)}
                  className="text-xs text-white/80"
                />
              </label>
            ))}
          </div>

          <div className="space-y-2">
            <h2 className="text-lg font-semibold">Layers</h2>
        {(['paths', 'turns', 'holdings', 'sids', 'stars', 'regions', 'wkt'] as const).map((key) => (
              <label key={key} className="flex items-center gap-2 text-sm cursor-pointer">
                <input
                  type="checkbox"
                  checked={showLayers[key]}
                  onChange={() => toggleLayer(key)}
                  className="accent-cyan-400"
                />
                <span className="capitalize">{key}</span>
                {counts[key as keyof LoadedCounts] !== undefined && (
                  <span className="ml-auto text-white/60">{counts[key as keyof LoadedCounts]}</span>
                )}
              </label>
            ))}
          </div>

          <div className="space-y-2">
            <h2 className="text-lg font-semibold">Add WKT</h2>
            <textarea
              value={wktInput}
              onChange={(e) => setWktInput(e.target.value)}
              placeholder="Paste WKT (POINT, LINESTRING, POLYGON)..."
              className="w-full h-24 bg-background-dark/60 border border-white/10 rounded p-2 text-sm text-white/90"
            />
            <div className="flex gap-2">
              <button
                onClick={addWktToMap}
                className="bg-primary/80 hover:bg-primary text-black font-semibold px-3 py-2 rounded border border-white/10"
              >
                Add WKT
              </button>
              <button
                onClick={() => { setWktFeatures([]); updateSource('wkt', []); }}
                className="bg-surface hover:bg-white/10 text-white font-semibold px-3 py-2 rounded border border-white/10"
              >
                Clear WKT
              </button>
            </div>
          </div>
        </aside>

        <main className="flex-1 relative">
          <div ref={mapRef} className="absolute inset-0" />
        </main>
      </div>
    </div>
  );
}

