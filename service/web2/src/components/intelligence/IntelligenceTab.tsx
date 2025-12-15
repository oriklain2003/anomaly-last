import { useState, useEffect, useRef } from 'react';
import { Shield, Radar, TrendingUp, Info, MapPin, AlertTriangle, Clock, Search, Dna, Building2, Plane, Target } from 'lucide-react';
import { StatCard } from './StatCard';
import { TableCard, Column } from './TableCard';
import { ChartCard } from './ChartCard';
import { SignalLossMap } from './SignalLossMap';
import maplibregl from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';
import { 
  fetchAirlineEfficiency, 
  fetchHoldingPatterns, 
  fetchGPSJamming,
  fetchMilitaryPatterns,
  fetchPatternClusters,
  fetchAnomalyDNAEnhanced
} from '../../api';
import type { AirlineEfficiency, HoldingPatternAnalysis, GPSJammingPoint, MilitaryPattern, PatternCluster, AnomalyDNA } from '../../types';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';

interface IntelligenceTabProps {
  startTs: number;
  endTs: number;
  cacheKey?: number;
}

export function IntelligenceTab({ startTs, endTs, cacheKey = 0 }: IntelligenceTabProps) {
  const [airlineEfficiency, setAirlineEfficiency] = useState<AirlineEfficiency[]>([]);
  const [holdingPatterns, setHoldingPatterns] = useState<HoldingPatternAnalysis | null>(null);
  const [gpsJamming, setGpsJamming] = useState<GPSJammingPoint[]>([]);
  const [militaryPatterns, setMilitaryPatterns] = useState<MilitaryPattern[]>([]);
  const [patternClusters, setPatternClusters] = useState<PatternCluster[]>([]);
  const [loading, setLoading] = useState(true);
  
  // Anomaly DNA state
  const [dnaFlightId, setDnaFlightId] = useState('');
  const [anomalyDNA, setAnomalyDNA] = useState<AnomalyDNA | null>(null);
  const [dnaLoading, setDnaLoading] = useState(false);
  const [dnaError, setDnaError] = useState<string | null>(null);
  
  // Military map ref
  const militaryMapContainer = useRef<HTMLDivElement>(null);
  const militaryMap = useRef<maplibregl.Map | null>(null);
  const militaryMarkers = useRef<maplibregl.Marker[]>([]);

  useEffect(() => {
    loadData();
  }, [startTs, endTs, cacheKey]);

  const loadData = async () => {
    setLoading(true);
    try {
      const [efficiencyData, holdingData, jammingData, militaryData, clustersData] = await Promise.all([
        fetchAirlineEfficiency(startTs, endTs),
        fetchHoldingPatterns(startTs, endTs),
        fetchGPSJamming(startTs, endTs),
        fetchMilitaryPatterns(startTs, endTs),
        fetchPatternClusters(startTs, endTs)
      ]);
      setAirlineEfficiency(efficiencyData);
      setHoldingPatterns(holdingData);
      setGpsJamming(jammingData);
      setMilitaryPatterns(militaryData);
      setPatternClusters(clustersData);
    } catch (error) {
      console.error('Failed to load intelligence data:', error);
    } finally {
      setLoading(false);
    }
  };

  // Initialize military map
  useEffect(() => {
    if (!militaryMapContainer.current || militaryPatterns.length === 0) return;
    
    if (militaryMap.current) {
      // Clear existing markers
      militaryMarkers.current.forEach(m => m.remove());
      militaryMarkers.current = [];
    } else {
      // Initialize map
      militaryMap.current = new maplibregl.Map({
        container: militaryMapContainer.current,
        style: 'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json',
        center: [35.0, 31.5],
        zoom: 6
      });
      militaryMap.current.addControl(new maplibregl.NavigationControl(), 'top-right');
    }

    // Add markers for military patterns with locations
    militaryPatterns.forEach(pattern => {
      if (pattern.locations && pattern.locations.length > 0) {
        // Get first location for marker
        const loc = pattern.locations[0];
        if (loc && typeof loc.lat === 'number' && typeof loc.lon === 'number') {
          const el = document.createElement('div');
          el.className = 'military-marker';
          
          // Color by country
          const countryColors: Record<string, string> = {
            'US': '#3b82f6',
            'GB': '#ef4444',
            'RU': '#f59e0b',
            'IL': '#10b981',
            'NATO': '#8b5cf6'
          };
          const color = countryColors[pattern.country] || '#6b7280';
          
          // Pattern shape by type
          const patternShapes: Record<string, string> = {
            'orbit': '●',
            'racetrack': '◆',
            'transit': '▶'
          };
          const shape = patternShapes[pattern.pattern_type] || '■';
          
          el.style.cssText = `
            width: 28px;
            height: 28px;
            border-radius: ${pattern.pattern_type === 'orbit' ? '50%' : '4px'};
            background: ${color};
            border: 2px solid white;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 12px;
            color: white;
            cursor: pointer;
            box-shadow: 0 2px 8px rgba(0,0,0,0.4);
          `;
          el.textContent = shape;

          const popup = new maplibregl.Popup({ offset: 25 }).setHTML(`
            <div style="padding: 8px; max-width: 200px;">
              <div style="font-weight: bold; margin-bottom: 4px;">${pattern.callsign}</div>
              <div style="font-size: 12px; color: #666;">
                <div>Country: ${pattern.country}</div>
                <div>Type: ${pattern.type}</div>
                <div>Pattern: ${pattern.pattern_type}</div>
              </div>
            </div>
          `);

          const marker = new maplibregl.Marker({ element: el })
            .setLngLat([loc.lon, loc.lat])
            .setPopup(popup)
            .addTo(militaryMap.current!);
          
          militaryMarkers.current.push(marker);
        }
      }
    });

    return () => {
      militaryMarkers.current.forEach(m => m.remove());
    };
  }, [militaryPatterns]);

  // Fetch Anomaly DNA
  const fetchDNA = async () => {
    if (!dnaFlightId.trim()) {
      setDnaError('Please enter a flight ID');
      return;
    }
    
    setDnaLoading(true);
    setDnaError(null);
    setAnomalyDNA(null);
    
    try {
      const data = await fetchAnomalyDNAEnhanced(dnaFlightId.trim());
      setAnomalyDNA(data);
    } catch (error) {
      setDnaError(error instanceof Error ? error.message : 'Failed to fetch anomaly DNA');
    } finally {
      setDnaLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-white/60">Loading intelligence data...</div>
      </div>
    );
  }

  const airlineColumns: Column[] = [
    { key: 'airline', title: 'Airline' },
    { key: 'avg_flight_time_min', title: 'Avg Flight Time (min)' },
    { key: 'avg_holding_time_min', title: 'Avg Holding Time (min)' },
    { key: 'sample_count', title: 'Sample Size' }
  ];

  const patternClusterColumns: Column[] = [
    { key: 'pattern_id', title: 'Pattern ID' },
    { key: 'description', title: 'Description' },
    { key: 'occurrence_count', title: 'Occurrences' },
    { key: 'risk_level', title: 'Risk Level' },
    { 
      key: 'first_seen', 
      title: 'First Seen',
      render: (value: number) => value ? new Date(value * 1000).toLocaleDateString() : 'N/A'
    }
  ];

  const militaryColumns: Column[] = [
    { key: 'callsign', title: 'Callsign' },
    { key: 'country', title: 'Country' },
    { key: 'type', title: 'Type' },
    { key: 'pattern_type', title: 'Pattern' }
  ];

  const jammingColumns: Column[] = [
    { key: 'lat', title: 'Latitude', render: (val: number) => val.toFixed(3) },
    { key: 'lon', title: 'Longitude', render: (val: number) => val.toFixed(3) },
    { key: 'intensity', title: 'Intensity', render: (val: number) => (
      <span className={val > 70 ? 'text-red-500 font-bold' : val > 40 ? 'text-yellow-500' : ''}>
        {val}
      </span>
    )},
    { key: 'event_count', title: 'Events' },
    { key: 'affected_flights', title: 'Affected Flights' }
  ];

  return (
    <div className="space-y-6">
      {/* Level 2: Operational Insights */}
      <div className="border-b border-white/10 pb-4">
        <h2 className="text-white text-xl font-bold mb-4 flex items-center gap-2">
          <TrendingUp className="w-5 h-5" />
          Operational Insights
        </h2>
      </div>

      {/* Holding Pattern Analysis */}
      {holdingPatterns && (
        <div className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <StatCard
              title="Total Holding Time"
              value={`${holdingPatterns.total_time_hours}h`}
              subtitle="Wasted fuel time"
              icon={<Clock className="w-6 h-6" />}
            />
            <StatCard
              title="Estimated Fuel Cost"
              value={`$${holdingPatterns.estimated_fuel_cost_usd.toLocaleString()}`}
              subtitle="Approximate cost"
            />
            <StatCard
              title="Peak Holding Hours"
              value={holdingPatterns.peak_hours.slice(0, 3).map(h => `${h}:00`).join(', ')}
              subtitle="Busiest times"
            />
          </div>

          {/* Events by Airport Breakdown */}
          {holdingPatterns.events_by_airport && Object.keys(holdingPatterns.events_by_airport).length > 0 && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              {/* Bar Chart */}
              <ChartCard title="Holding Events by Airport">
                <ResponsiveContainer width="100%" height={250}>
                  <BarChart 
                    data={Object.entries(holdingPatterns.events_by_airport)
                      .sort(([,a], [,b]) => b - a)
                      .slice(0, 8)
                      .map(([airport, count]) => ({ airport, count }))}
                    layout="vertical"
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="#ffffff20" />
                    <XAxis type="number" stroke="#ffffff60" tick={{ fill: '#ffffff60' }} />
                    <YAxis 
                      type="category" 
                      dataKey="airport" 
                      stroke="#ffffff60" 
                      tick={{ fill: '#ffffff60' }}
                      width={60}
                    />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: '#1a1a1a',
                        border: '1px solid #ffffff20',
                        borderRadius: '8px'
                      }}
                    />
                    <Bar dataKey="count" fill="#f59e0b" radius={[0, 4, 4, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </ChartCard>

              {/* Pie Chart */}
              <ChartCard title="Distribution">
                <ResponsiveContainer width="100%" height={250}>
                  <PieChart>
                    <Pie
                      data={Object.entries(holdingPatterns.events_by_airport)
                        .sort(([,a], [,b]) => b - a)
                        .slice(0, 5)
                        .map(([airport, count]) => ({ name: airport, value: count }))}
                      cx="50%"
                      cy="50%"
                      innerRadius={50}
                      outerRadius={80}
                      paddingAngle={5}
                      dataKey="value"
                      label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                    >
                      {Object.entries(holdingPatterns.events_by_airport)
                        .slice(0, 5)
                        .map((_, index) => (
                          <Cell key={`cell-${index}`} fill={[
                            '#f59e0b', '#3b82f6', '#10b981', '#ef4444', '#8b5cf6'
                          ][index]} />
                        ))}
                    </Pie>
                    <Tooltip
                      contentStyle={{
                        backgroundColor: '#1a1a1a',
                        border: '1px solid #ffffff20',
                        borderRadius: '8px'
                      }}
                    />
                  </PieChart>
                </ResponsiveContainer>
              </ChartCard>
            </div>
          )}
        </div>
      )}

      {/* Airline Efficiency Comparison */}
      <ChartCard title="Airline Efficiency Comparison">
        {airlineEfficiency.length > 0 ? (
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={airlineEfficiency}>
              <CartesianGrid strokeDasharray="3 3" stroke="#ffffff20" />
              <XAxis dataKey="airline" stroke="#ffffff60" tick={{ fill: '#ffffff60' }} />
              <YAxis stroke="#ffffff60" tick={{ fill: '#ffffff60' }} />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#1a1a1a',
                  border: '1px solid #ffffff20',
                  borderRadius: '8px'
                }}
              />
              <Bar dataKey="avg_flight_time_min" fill="#3b82f6" name="Avg Flight Time (min)" />
              <Bar dataKey="avg_holding_time_min" fill="#ef4444" name="Avg Holding Time (min)" />
            </BarChart>
          </ResponsiveContainer>
        ) : (
          <div className="h-64 flex items-center justify-center text-white/40">
            No airline efficiency data available
          </div>
        )}
      </ChartCard>

      <TableCard
        title="Airline Efficiency Details"
        columns={airlineColumns}
        data={airlineEfficiency}
      />

      {/* Level 3: Deep Intelligence */}
      <div className="border-b border-white/10 pb-4 pt-8">
        <h2 className="text-white text-xl font-bold mb-4 flex items-center gap-2">
          <Shield className="w-5 h-5" />
          Deep Intelligence
        </h2>
      </div>

      {/* GPS Jamming Overview */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <StatCard
          title="GPS Jamming Zones"
          value={gpsJamming.length}
          subtitle="Detected interference areas"
          icon={<Radar className="w-6 h-6" />}
        />
        <StatCard
          title="Military Aircraft Tracked"
          value={militaryPatterns.length}
          subtitle="Foreign military presence"
          icon={<Shield className="w-6 h-6" />}
        />
      </div>

      {/* GPS Jamming Map Visualization */}
      <div className="bg-surface rounded-xl border border-white/10 overflow-hidden">
        <div className="px-6 py-4 border-b border-white/10">
          <div className="flex items-start justify-between">
            <div>
              <h3 className="text-lg font-semibold text-white flex items-center gap-2">
                <Radar className="w-5 h-5 text-red-500" />
                GPS Jamming Threat Map
              </h3>
              <p className="text-white/60 text-sm mt-1">
                Security-focused analysis of potential GPS interference zones
              </p>
              <div className="mt-2 inline-flex items-center gap-2 px-2 py-1 bg-red-500/20 rounded text-xs text-red-300">
                <AlertTriangle className="w-3 h-3" />
                <span>Security Intelligence - Potential hostile interference</span>
              </div>
            </div>
            <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-3 max-w-xs">
              <div className="flex items-start gap-2">
                <Info className="w-4 h-4 text-red-400 mt-0.5 shrink-0" />
                <div className="text-xs text-red-300">
                  <strong>Focus:</strong> Identifies zones where signal loss patterns suggest 
                  intentional GPS jamming. For general coverage gaps, see the <strong>Traffic Tab</strong>.
                </div>
              </div>
            </div>
          </div>
        </div>
        
        <div className="p-6">
          <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
            {/* Map */}
            <div className="xl:col-span-2">
              {gpsJamming.length > 0 ? (
                <SignalLossMap 
                  locations={gpsJamming.map(j => ({
                    lat: j.lat,
                    lon: j.lon,
                    count: j.event_count,
                    avgDuration: 300, // Default 5 min for jamming zones
                    intensity: j.intensity,
                    affected_flights: j.affected_flights
                  }))} 
                  height={400} 
                />
              ) : (
                <div className="h-[400px] flex items-center justify-center bg-surface-highlight rounded-lg border border-white/10">
                  <div className="text-white/40 text-center">
                    <Radar className="w-12 h-12 mx-auto mb-3 opacity-30" />
                    <p>No GPS jamming detected in this period</p>
                  </div>
                </div>
              )}
            </div>
            
            {/* Stats Panel */}
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-3">
                <div className="bg-surface-highlight rounded-lg p-4 text-center">
                  <div className="text-2xl font-bold text-red-400">{gpsJamming.length}</div>
                  <div className="text-xs text-white/50">Jamming Zones</div>
                </div>
                <div className="bg-surface-highlight rounded-lg p-4 text-center">
                  <div className="text-2xl font-bold text-orange-400">
                    {gpsJamming.reduce((sum, j) => sum + j.affected_flights, 0)}
                  </div>
                  <div className="text-xs text-white/50">Affected Flights</div>
                </div>
              </div>
              
              <div className="bg-surface-highlight rounded-lg p-4">
                <div className="flex items-center gap-2 mb-3">
                  <MapPin className="w-4 h-4 text-red-500" />
                  <span className="text-white/80 text-sm font-medium">High Intensity Zones</span>
                </div>
                <div className="space-y-2">
                  {gpsJamming.slice(0, 5).map((zone, idx) => (
                    <div key={idx} className="bg-black/20 rounded-lg p-3">
                      <div className="flex justify-between items-center mb-1">
                        <span className="text-white text-sm font-medium">
                          {zone.lat.toFixed(2)}°N, {zone.lon.toFixed(2)}°E
                        </span>
                        <span className={`font-bold text-sm ${
                          zone.intensity > 70 ? 'text-red-400' : 
                          zone.intensity > 40 ? 'text-orange-400' : 'text-yellow-400'
                        }`}>
                          {zone.intensity}%
                        </span>
                      </div>
                      <div className="text-xs text-white/50">
                        {zone.event_count} events • {zone.affected_flights} flights
                      </div>
                    </div>
                  ))}
                  {gpsJamming.length === 0 && (
                    <p className="text-white/40 text-sm text-center py-4">
                      ✓ No jamming zones detected
                    </p>
                  )}
                </div>
              </div>
              
              <div className="bg-gradient-to-br from-red-500/10 to-orange-500/10 border border-red-500/30 rounded-lg p-4">
                <h4 className="text-red-400 text-sm font-medium mb-2 flex items-center gap-2">
                  <AlertTriangle className="w-4 h-4" />
                  Security Implications
                </h4>
                <ul className="text-xs text-white/70 space-y-1.5">
                  <li className="flex items-start gap-2">
                    <span className="text-red-400">•</span>
                    <span>GPS jamming can indicate hostile activity</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-orange-400">•</span>
                    <span>May affect aircraft navigation systems</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-yellow-400">•</span>
                    <span>Report persistent zones to aviation authorities</span>
                  </li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </div>

      <TableCard
        title="GPS Jamming Zones (Geographic Data)"
        columns={jammingColumns}
        data={gpsJamming.slice(0, 15)}
      />

      {/* Military Aircraft Patterns */}
      <TableCard
        title="Military Aircraft Patterns"
        columns={militaryColumns}
        data={militaryPatterns.slice(0, 20)}
      />

      {/* Military Locations Map */}
      {militaryPatterns.length > 0 && (
        <div className="bg-surface rounded-xl border border-white/10 overflow-hidden">
          <div className="px-6 py-4 border-b border-white/10">
            <div className="flex items-start justify-between">
              <div>
                <h3 className="text-lg font-semibold text-white flex items-center gap-2">
                  <Target className="w-5 h-5 text-purple-500" />
                  Military Activity Map
                </h3>
                <p className="text-white/60 text-sm mt-1">
                  Geographic visualization of military aircraft patterns
                </p>
              </div>
              {/* Legend */}
              <div className="flex flex-wrap gap-3">
                <div className="flex items-center gap-2">
                  <div className="w-4 h-4 rounded-full bg-[#3b82f6]" />
                  <span className="text-white/60 text-xs">US</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-4 h-4 rounded-full bg-[#ef4444]" />
                  <span className="text-white/60 text-xs">GB</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-4 h-4 rounded-full bg-[#f59e0b]" />
                  <span className="text-white/60 text-xs">RU</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-4 h-4 rounded-full bg-[#10b981]" />
                  <span className="text-white/60 text-xs">IL</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-4 h-4 rounded-full bg-[#8b5cf6]" />
                  <span className="text-white/60 text-xs">NATO</span>
                </div>
              </div>
            </div>
            {/* Pattern Type Legend */}
            <div className="mt-3 flex gap-4">
              <div className="flex items-center gap-2">
                <span className="text-white text-sm">●</span>
                <span className="text-white/50 text-xs">Orbit</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-white text-sm">◆</span>
                <span className="text-white/50 text-xs">Racetrack</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-white text-sm">▶</span>
                <span className="text-white/50 text-xs">Transit</span>
              </div>
            </div>
          </div>
          <div 
            ref={militaryMapContainer} 
            className="h-[400px] w-full"
          />
        </div>
      )}

      {/* Pattern Analysis (Anomaly DNA) Section */}
      <div className="border-b border-white/10 pb-4 pt-8">
        <h2 className="text-white text-xl font-bold mb-4 flex items-center gap-2">
          <Radar className="w-5 h-5" />
          Pattern Analysis (Anomaly DNA)
        </h2>
        <p className="text-white/60 text-sm mb-4">
          Automatically detected recurring anomaly patterns and suspicious flight behaviors.
        </p>
      </div>

      {/* Pattern Clusters */}
      <TableCard
        title="Recurring Anomaly Clusters"
        columns={patternClusterColumns}
        data={patternClusters.slice(0, 10)}
      />
      
      {patternClusters.length === 0 && (
        <div className="text-white/40 text-center py-8">
          No recurring patterns detected in this time period
        </div>
      )}

      {/* Anomaly DNA Section */}
      <div className="border-b border-white/10 pb-4 pt-8">
        <h2 className="text-white text-xl font-bold mb-2 flex items-center gap-2">
          <Dna className="w-5 h-5 text-emerald-500" />
          Anomaly DNA Analysis
        </h2>
        <p className="text-white/60 text-sm">
          Deep analysis of flight anomalies with similar pattern matching and risk assessment
        </p>
      </div>

      {/* DNA Search */}
      <div className="bg-surface rounded-xl border border-white/10 p-6">
        <div className="flex gap-4 items-end">
          <div className="flex-1">
            <label className="block text-white/70 text-sm mb-2">Flight ID</label>
            <input
              type="text"
              value={dnaFlightId}
              onChange={(e) => setDnaFlightId(e.target.value)}
              placeholder="Enter flight ID to analyze (e.g., 3b86ff46)"
              className="w-full px-4 py-3 bg-surface-highlight border border-white/20 rounded-lg text-white placeholder-white/40 focus:outline-none focus:border-emerald-500"
              onKeyDown={(e) => e.key === 'Enter' && fetchDNA()}
            />
          </div>
          <button
            onClick={fetchDNA}
            disabled={dnaLoading}
            className="px-6 py-3 bg-emerald-600 hover:bg-emerald-700 disabled:bg-emerald-600/50 text-white font-medium rounded-lg flex items-center gap-2 transition-colors"
          >
            {dnaLoading ? (
              <>
                <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                Analyzing...
              </>
            ) : (
              <>
                <Dna className="w-4 h-4" />
                Analyze DNA
              </>
            )}
          </button>
        </div>

        {dnaError && (
          <div className="mt-4 p-3 bg-red-500/10 border border-red-500/30 rounded-lg text-red-400 text-sm">
            {dnaError}
          </div>
        )}
      </div>

      {/* DNA Results */}
      {anomalyDNA && (
        <div className="space-y-4">
          {/* Flight Info & Risk Assessment */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
            {/* Flight Info */}
            <div className="bg-surface rounded-xl border border-white/10 p-6">
              <h3 className="text-white font-bold mb-4 flex items-center gap-2">
                <Plane className="w-4 h-4 text-blue-400" />
                Flight Information
              </h3>
              <div className="space-y-3">
                <div className="flex justify-between">
                  <span className="text-white/60">Flight ID</span>
                  <span className="text-white font-mono">{anomalyDNA.flight_info.flight_id}</span>
                </div>
                {anomalyDNA.flight_info.callsign && (
                  <div className="flex justify-between">
                    <span className="text-white/60">Callsign</span>
                    <span className="text-white font-bold">{anomalyDNA.flight_info.callsign}</span>
                  </div>
                )}
                <div className="flex justify-between">
                  <span className="text-white/60">Pattern Type</span>
                  <span className="text-emerald-400">{anomalyDNA.recurring_pattern || 'Unique'}</span>
                </div>
              </div>
            </div>

            {/* Risk Assessment */}
            <div className={`bg-surface rounded-xl border-2 p-6 ${
              anomalyDNA.risk_assessment === 'high' ? 'border-red-500/50' :
              anomalyDNA.risk_assessment === 'medium' ? 'border-yellow-500/50' : 'border-green-500/50'
            }`}>
              <h3 className="text-white font-bold mb-4 flex items-center gap-2">
                <AlertTriangle className="w-4 h-4 text-yellow-400" />
                Risk Assessment
              </h3>
              <div className={`text-4xl font-bold mb-2 ${
                anomalyDNA.risk_assessment === 'high' ? 'text-red-400' :
                anomalyDNA.risk_assessment === 'medium' ? 'text-yellow-400' : 'text-green-400'
              }`}>
                {anomalyDNA.risk_assessment?.toUpperCase() || 'UNKNOWN'}
              </div>
              <p className="text-white/60 text-sm">
                Based on pattern analysis and historical data
              </p>
            </div>

            {/* Similar Flights Count */}
            <div className="bg-surface rounded-xl border border-white/10 p-6">
              <h3 className="text-white font-bold mb-4 flex items-center gap-2">
                <Search className="w-4 h-4 text-purple-400" />
                Similar Flights Found
              </h3>
              <div className="text-4xl font-bold text-purple-400 mb-2">
                {anomalyDNA.similar_flights?.length || 0}
              </div>
              <p className="text-white/60 text-sm">
                Flights with matching anomaly patterns
              </p>
            </div>
          </div>

          {/* Insights */}
          {anomalyDNA.insights && anomalyDNA.insights.length > 0 && (
            <div className="bg-gradient-to-br from-emerald-500/10 to-teal-500/10 border border-emerald-500/30 rounded-xl p-6">
              <h3 className="text-emerald-400 font-bold mb-4 flex items-center gap-2">
                <Info className="w-4 h-4" />
                Key Insights
              </h3>
              <ul className="space-y-2">
                {anomalyDNA.insights.map((insight, idx) => (
                  <li key={idx} className="flex items-start gap-2 text-white/80">
                    <span className="text-emerald-400 mt-1">•</span>
                    <span>{insight}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Detected Anomalies */}
          {anomalyDNA.anomalies_detected && anomalyDNA.anomalies_detected.length > 0 && (
            <div className="bg-surface rounded-xl border border-white/10 p-6">
              <h3 className="text-white font-bold mb-4 flex items-center gap-2">
                <AlertTriangle className="w-4 h-4 text-orange-400" />
                Detected Anomalies
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                {anomalyDNA.anomalies_detected.map((anomaly, idx) => (
                  <div key={idx} className="bg-surface-highlight rounded-lg p-4">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-white font-medium">Rule {anomaly.rule_id}</span>
                      <span className="text-orange-400 text-xs">
                        {new Date(anomaly.timestamp * 1000).toLocaleTimeString()}
                      </span>
                    </div>
                    <p className="text-white/60 text-sm">{anomaly.rule_name}</p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Similar Flights Table */}
          {anomalyDNA.similar_flights && anomalyDNA.similar_flights.length > 0 && (
            <div className="bg-surface rounded-xl border border-white/10 p-6">
              <h3 className="text-white font-bold mb-4 flex items-center gap-2">
                <Search className="w-4 h-4 text-purple-400" />
                Similar Flights
              </h3>
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-white/10">
                      <th className="text-left text-white/60 text-sm py-2 px-3">Flight ID</th>
                      <th className="text-left text-white/60 text-sm py-2 px-3">Callsign</th>
                      <th className="text-left text-white/60 text-sm py-2 px-3">Similarity</th>
                      <th className="text-left text-white/60 text-sm py-2 px-3">Pattern</th>
                      <th className="text-left text-white/60 text-sm py-2 px-3">Date</th>
                    </tr>
                  </thead>
                  <tbody>
                    {anomalyDNA.similar_flights.slice(0, 10).map((flight, idx) => (
                      <tr key={idx} className="border-b border-white/5 hover:bg-white/5">
                        <td className="py-3 px-3 text-white font-mono text-sm">{flight.flight_id}</td>
                        <td className="py-3 px-3 text-white">{flight.callsign || '-'}</td>
                        <td className="py-3 px-3">
                          <div className="flex items-center gap-2">
                            <div className="w-16 bg-black/30 rounded-full h-2">
                              <div 
                                className={`h-2 rounded-full ${
                                  flight.similarity_score > 70 ? 'bg-emerald-500' :
                                  flight.similarity_score > 40 ? 'bg-yellow-500' : 'bg-red-500'
                                }`}
                                style={{ width: `${flight.similarity_score}%` }}
                              />
                            </div>
                            <span className={`text-sm font-bold ${
                              flight.similarity_score > 70 ? 'text-emerald-400' :
                              flight.similarity_score > 40 ? 'text-yellow-400' : 'text-red-400'
                            }`}>
                              {flight.similarity_score}%
                            </span>
                          </div>
                        </td>
                        <td className="py-3 px-3 text-white/60 text-sm">{flight.pattern || '-'}</td>
                        <td className="py-3 px-3 text-white/60 text-sm">{flight.date || '-'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

