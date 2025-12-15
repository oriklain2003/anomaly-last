import { useState, useEffect } from 'react';
import { Plane, Signal, AlertTriangle, Info, Clock, MapPin, TrendingUp, ArrowRightLeft, Building2, BarChart3 } from 'lucide-react';
import { StatCard } from './StatCard';
import { TableCard, Column } from './TableCard';
import { ChartCard } from './ChartCard';
import { SignalLossMap } from './SignalLossMap';
import { 
  fetchFlightsPerDay, 
  fetchBusiestAirports, 
  fetchSignalLoss,
  fetchPeakHoursAnalysis,
  fetchDiversionStats,
  fetchAlternateAirportsData,
  fetchRunwayUsage
} from '../../api';
import type { FlightPerDay, SignalLossLocation } from '../../types';
import type { PeakHoursAnalysis, DiversionStats, AlternateAirport, RunwayUsage } from '../../api';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar, ComposedChart, Area } from 'recharts';

interface TrafficTabProps {
  startTs: number;
  endTs: number;
  cacheKey?: number;
}

export function TrafficTab({ startTs, endTs, cacheKey = 0 }: TrafficTabProps) {
  const [flightsPerDay, setFlightsPerDay] = useState<FlightPerDay[]>([]);
  const [airports, setAirports] = useState<any[]>([]);
  const [signalLoss, setSignalLoss] = useState<SignalLossLocation[]>([]);
  const [peakHours, setPeakHours] = useState<PeakHoursAnalysis | null>(null);
  const [diversionStats, setDiversionStats] = useState<DiversionStats | null>(null);
  const [alternateAirports, setAlternateAirports] = useState<AlternateAirport[]>([]);
  const [runwayUsage, setRunwayUsage] = useState<RunwayUsage[]>([]);
  const [selectedAirport, setSelectedAirport] = useState('LLBG');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadData();
  }, [startTs, endTs, cacheKey]);

  useEffect(() => {
    // Load runway usage when airport changes
    loadRunwayUsage();
  }, [selectedAirport, startTs, endTs]);

  const loadData = async () => {
    setLoading(true);
    try {
      const [flightsData, airportsData, signalData, peakData, diversionData, alternateData] = await Promise.all([
        fetchFlightsPerDay(startTs, endTs),
        fetchBusiestAirports(startTs, endTs, 10),
        fetchSignalLoss(startTs, endTs),
        fetchPeakHoursAnalysis(startTs, endTs).catch(() => null),
        fetchDiversionStats(startTs, endTs).catch(() => null),
        fetchAlternateAirportsData(startTs, endTs).catch(() => [])
      ]);
      setFlightsPerDay(flightsData);
      setAirports(airportsData);
      setSignalLoss(signalData);
      setPeakHours(peakData);
      setDiversionStats(diversionData);
      setAlternateAirports(alternateData);
    } catch (error) {
      console.error('Failed to load traffic data:', error);
    } finally {
      setLoading(false);
    }
  };

  const loadRunwayUsage = async () => {
    try {
      const data = await fetchRunwayUsage(selectedAirport, startTs, endTs);
      setRunwayUsage(data);
    } catch (error) {
      console.error('Failed to load runway usage:', error);
      setRunwayUsage([]);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-white/60">Loading traffic statistics...</div>
      </div>
    );
  }

  const totalFlights = flightsPerDay.reduce((sum, day) => sum + day.count, 0);
  const avgFlightsPerDay = flightsPerDay.length > 0 ? Math.round(totalFlights / flightsPerDay.length) : 0;
  const totalMilitary = flightsPerDay.reduce((sum, day) => sum + day.military_count, 0);
  const totalSignalLoss = signalLoss.reduce((sum, loc) => sum + loc.count, 0);

  const airportColumns: Column[] = [
    { key: 'airport', title: 'Airport' },
    { key: 'arrivals', title: 'Arrivals' },
    { key: 'departures', title: 'Departures' },
    { key: 'total', title: 'Total Operations' }
  ];

  const signalLossColumns: Column[] = [
    { key: 'lat', title: 'Latitude', render: (val) => val.toFixed(3) },
    { key: 'lon', title: 'Longitude', render: (val) => val.toFixed(3) },
    { key: 'count', title: 'Events' },
    { key: 'avgDuration', title: 'Avg Gap Duration (s)', render: (val) => Math.round(val) }
  ];

  return (
    <div className="space-y-6">
      {/* Key Traffic Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <StatCard
          title="Total Flights"
          value={totalFlights.toLocaleString()}
          subtitle="In selected period"
          icon={<Plane className="w-6 h-6" />}
        />
        <StatCard
          title="Avg Flights/Day"
          value={avgFlightsPerDay.toLocaleString()}
          subtitle="Daily average"
        />
        <StatCard
          title="Military Flights"
          value={totalMilitary.toLocaleString()}
          subtitle="Tracked military"
          icon={<Plane className="w-6 h-6" />}
        />
        <StatCard
          title="Signal Loss Events"
          value={totalSignalLoss.toLocaleString()}
          subtitle="GPS/tracking gaps"
          icon={<Signal className="w-6 h-6" />}
        />
      </div>

      {/* Flights Per Day Chart */}
      <ChartCard title="Flight Traffic Over Time">
        {flightsPerDay.length > 0 ? (
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={flightsPerDay}>
              <CartesianGrid strokeDasharray="3 3" stroke="#ffffff20" />
              <XAxis 
                dataKey="date" 
                stroke="#ffffff60"
                tick={{ fill: '#ffffff60' }}
              />
              <YAxis 
                stroke="#ffffff60"
                tick={{ fill: '#ffffff60' }}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#1a1a1a',
                  border: '1px solid #ffffff20',
                  borderRadius: '8px'
                }}
              />
              <Line 
                type="monotone" 
                dataKey="count" 
                stroke="#3b82f6" 
                strokeWidth={2}
                name="Total"
              />
              <Line 
                type="monotone" 
                dataKey="military_count" 
                stroke="#ef4444" 
                strokeWidth={2}
                name="Military"
              />
              <Line 
                type="monotone" 
                dataKey="civilian_count" 
                stroke="#10b981" 
                strokeWidth={2}
                name="Civilian"
              />
            </LineChart>
          </ResponsiveContainer>
        ) : (
          <div className="h-64 flex items-center justify-center text-white/40">
            No flight data available
          </div>
        )}
      </ChartCard>

      {/* Busiest Airports Table */}
      <TableCard
        title="Busiest Airports"
        columns={airportColumns}
        data={airports}
      />

      {/* Signal Loss / GPS Jamming Zones Section */}
      <div className="bg-surface rounded-xl border border-white/10 overflow-hidden">
        {/* Header with explanation */}
        <div className="px-6 py-4 border-b border-white/10">
          <div className="flex items-start justify-between">
            <div>
              <h3 className="text-lg font-semibold text-white flex items-center gap-2">
                <Signal className="w-5 h-5 text-red-500" />
                Signal Coverage Analysis
              </h3>
              <p className="text-white/60 text-sm mt-1">
                Operational view of areas where aircraft tracking signals were lost or interrupted
              </p>
              <div className="mt-2 inline-flex items-center gap-2 px-2 py-1 bg-blue-500/20 rounded text-xs text-blue-300">
                <Info className="w-3 h-3" />
                <span>For operational awareness - includes all coverage gaps</span>
              </div>
            </div>
            <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-3 max-w-xs">
              <div className="flex items-start gap-2">
                <Info className="w-4 h-4 text-blue-400 mt-0.5 shrink-0" />
                <div className="text-xs text-blue-300">
                  <strong>Purpose:</strong> Track operational coverage gaps from ADS-B receivers, 
                  terrain blocking, and equipment issues. For security-focused GPS jamming analysis, 
                  see the <strong>Intelligence Tab</strong>.
                </div>
              </div>
            </div>
          </div>
        </div>
        
        {/* Map and Details */}
        <div className="p-6">
          <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
            {/* Map - takes 2 columns on xl screens */}
            <div className="xl:col-span-2">
              <SignalLossMap locations={signalLoss} height={450} />
            </div>
            
            {/* Stats and Hotspots Panel */}
            <div className="space-y-4">
              {/* Quick Stats */}
              <div className="grid grid-cols-2 gap-3">
                <div className="bg-surface-highlight rounded-lg p-4 text-center">
                  <div className="text-2xl font-bold text-red-400">{totalSignalLoss}</div>
                  <div className="text-xs text-white/50">Total Events</div>
                </div>
                <div className="bg-surface-highlight rounded-lg p-4 text-center">
                  <div className="text-2xl font-bold text-orange-400">{signalLoss.length}</div>
                  <div className="text-xs text-white/50">Unique Zones</div>
                </div>
              </div>
              
              {/* Top Hotspots */}
              <div className="bg-surface-highlight rounded-lg p-4">
                <div className="flex items-center gap-2 mb-3">
                  <MapPin className="w-4 h-4 text-red-500" />
                  <span className="text-white/80 text-sm font-medium">Top Signal Loss Hotspots</span>
                </div>
                <div className="space-y-2">
                  {signalLoss.slice(0, 5).map((loc, idx) => (
                    <div key={idx} className="bg-black/20 rounded-lg p-3">
                      <div className="flex justify-between items-center mb-1">
                        <span className="text-white text-sm font-medium">
                          {loc.lat.toFixed(2)}°N, {loc.lon.toFixed(2)}°E
                        </span>
                        <span className="text-red-400 font-bold text-sm">{loc.count} events</span>
                      </div>
                      <div className="flex items-center gap-2 text-xs text-white/50">
                        <Clock className="w-3 h-3" />
                        <span>Avg gap: {Math.round(loc.avgDuration)}s</span>
                      </div>
                    </div>
                  ))}
                  {signalLoss.length === 0 && (
                    <p className="text-white/40 text-sm text-center py-4">
                      ✓ No signal loss zones detected
                    </p>
                  )}
                </div>
              </div>
              
              {/* Explanation Panel */}
              <div className="bg-gradient-to-br from-yellow-500/10 to-orange-500/10 border border-yellow-500/30 rounded-lg p-4">
                <h4 className="text-yellow-400 text-sm font-medium mb-2 flex items-center gap-2">
                  <AlertTriangle className="w-4 h-4" />
                  What causes signal loss?
                </h4>
                <ul className="text-xs text-white/70 space-y-1.5">
                  <li className="flex items-start gap-2">
                    <span className="text-red-400">•</span>
                    <span><strong className="text-white/90">GPS Jamming:</strong> Intentional interference with navigation signals</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-orange-400">•</span>
                    <span><strong className="text-white/90">Terrain:</strong> Mountains or buildings blocking signals</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-yellow-400">•</span>
                    <span><strong className="text-white/90">Coverage Gap:</strong> Areas with limited ADS-B receiver coverage</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-blue-400">•</span>
                    <span><strong className="text-white/90">Equipment:</strong> Temporary transponder issues on aircraft</span>
                  </li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Signal Loss Detailed Table */}
      {signalLoss.length > 0 && (
        <TableCard
          title="Signal Loss Zone Details"
          columns={signalLossColumns}
          data={signalLoss.slice(0, 15)}
        />
      )}

      {/* Peak Hours Analysis Section */}
      <div className="border-b border-white/10 pb-4 pt-8">
        <h2 className="text-white text-xl font-bold mb-2 flex items-center gap-2">
          <TrendingUp className="w-5 h-5 text-blue-500" />
          Peak Hours Analysis
        </h2>
        <p className="text-white/60 text-sm">
          Correlation between traffic volume and safety events by hour
        </p>
      </div>

      {peakHours && (
        <div className="space-y-4">
          {/* Correlation Score Card */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className={`bg-surface rounded-xl p-6 border-2 ${
              peakHours.correlation_score > 0.5 ? 'border-red-500/50' :
              peakHours.correlation_score > 0.2 ? 'border-yellow-500/50' : 'border-green-500/50'
            }`}>
              <div className="text-white/60 text-sm mb-1">Traffic-Safety Correlation</div>
              <div className={`text-4xl font-bold ${
                peakHours.correlation_score > 0.5 ? 'text-red-400' :
                peakHours.correlation_score > 0.2 ? 'text-yellow-400' : 'text-green-400'
              }`}>
                {(peakHours.correlation_score * 100).toFixed(0)}%
              </div>
              <div className="text-white/50 text-xs mt-2">
                {peakHours.correlation_score > 0.5 
                  ? 'High correlation - safety events increase with traffic'
                  : peakHours.correlation_score > 0.2 
                  ? 'Moderate correlation - some relationship'
                  : 'Low correlation - safety events independent of traffic'}
              </div>
            </div>
            <StatCard
              title="Peak Traffic Hours"
              value={peakHours.peak_traffic_hours.slice(0, 3).map(h => `${h}:00`).join(', ')}
              subtitle="Busiest times"
              icon={<Plane className="w-6 h-6" />}
            />
            <StatCard
              title="Peak Safety Hours"
              value={peakHours.peak_safety_hours.slice(0, 3).map(h => `${h}:00`).join(', ')}
              subtitle="Most events"
              icon={<AlertTriangle className="w-6 h-6" />}
            />
          </div>

          {/* Hourly Chart */}
          <ChartCard title="Traffic vs Safety Events by Hour">
            <ResponsiveContainer width="100%" height={300}>
              <ComposedChart data={peakHours.hourly_data}>
                <CartesianGrid strokeDasharray="3 3" stroke="#ffffff20" />
                <XAxis 
                  dataKey="hour" 
                  stroke="#ffffff60"
                  tick={{ fill: '#ffffff60' }}
                  tickFormatter={(h) => `${h}:00`}
                />
                <YAxis 
                  yAxisId="left"
                  stroke="#3b82f6"
                  tick={{ fill: '#3b82f6' }}
                />
                <YAxis 
                  yAxisId="right"
                  orientation="right"
                  stroke="#ef4444"
                  tick={{ fill: '#ef4444' }}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#1a1a1a',
                    border: '1px solid #ffffff20',
                    borderRadius: '8px'
                  }}
                  formatter={(value: number, name: string) => [
                    value,
                    name === 'traffic' ? 'Traffic' : 'Safety Events'
                  ]}
                />
                <Area 
                  yAxisId="left"
                  type="monotone" 
                  dataKey="traffic" 
                  fill="#3b82f620" 
                  stroke="#3b82f6"
                  strokeWidth={2}
                  name="traffic"
                />
                <Bar 
                  yAxisId="right"
                  dataKey="safety_events" 
                  fill="#ef4444" 
                  name="safety_events"
                  radius={[4, 4, 0, 0]}
                />
              </ComposedChart>
            </ResponsiveContainer>
          </ChartCard>
        </div>
      )}

      {!peakHours && (
        <div className="bg-surface rounded-xl p-8 border border-white/10 text-center">
          <TrendingUp className="w-12 h-12 mx-auto mb-3 text-white/20" />
          <p className="text-white/40">Peak hours analysis not available for this period</p>
        </div>
      )}

      {/* Diversion Statistics Section */}
      <div className="border-b border-white/10 pb-4 pt-8">
        <h2 className="text-white text-xl font-bold mb-2 flex items-center gap-2">
          <ArrowRightLeft className="w-5 h-5 text-orange-500" />
          Diversion Statistics
        </h2>
        <p className="text-white/60 text-sm">
          Flight diversions, route deviations, and holding patterns
        </p>
      </div>

      {diversionStats && (
        <div className="space-y-4">
          {/* Summary Stats */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <StatCard
              title="Total Diversions"
              value={diversionStats.total_diversions}
              subtitle="Flights diverted to alternate"
              icon={<ArrowRightLeft className="w-6 h-6" />}
            />
            <StatCard
              title="Large Deviations"
              value={diversionStats.total_large_deviations}
              subtitle=">20nm from planned route"
              icon={<AlertTriangle className="w-6 h-6" />}
            />
            <StatCard
              title="360° Holds"
              value={diversionStats.total_holding_360s}
              subtitle="Full orbit holds before landing"
              icon={<Clock className="w-6 h-6" />}
            />
          </div>

          {/* By Airport and Airline */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {/* By Airport */}
            <div className="bg-surface rounded-xl border border-white/10 p-6">
              <h3 className="text-white font-bold mb-4 flex items-center gap-2">
                <Building2 className="w-4 h-4 text-orange-400" />
                Diversions by Airport
              </h3>
              <div className="space-y-2">
                {Object.entries(diversionStats.by_airport)
                  .sort(([,a], [,b]) => b - a)
                  .slice(0, 6)
                  .map(([airport, count]) => (
                    <div key={airport} className="flex items-center justify-between bg-surface-highlight rounded-lg p-3">
                      <span className="text-white font-medium">{airport}</span>
                      <div className="flex items-center gap-3">
                        <div className="w-32 bg-black/30 rounded-full h-2">
                          <div 
                            className="bg-orange-500 h-2 rounded-full"
                            style={{ 
                              width: `${Math.min(100, (count / Math.max(...Object.values(diversionStats.by_airport))) * 100)}%` 
                            }}
                          />
                        </div>
                        <span className="text-orange-400 font-bold w-8 text-right">{count}</span>
                      </div>
                    </div>
                  ))}
                {Object.keys(diversionStats.by_airport).length === 0 && (
                  <p className="text-white/40 text-center py-4">No diversions recorded</p>
                )}
              </div>
            </div>

            {/* By Airline */}
            <div className="bg-surface rounded-xl border border-white/10 p-6">
              <h3 className="text-white font-bold mb-4 flex items-center gap-2">
                <Plane className="w-4 h-4 text-blue-400" />
                Diversions by Airline
              </h3>
              <div className="space-y-2">
                {Object.entries(diversionStats.by_airline)
                  .sort(([,a], [,b]) => b - a)
                  .slice(0, 6)
                  .map(([airline, count]) => (
                    <div key={airline} className="flex items-center justify-between bg-surface-highlight rounded-lg p-3">
                      <span className="text-white font-medium">{airline || 'Unknown'}</span>
                      <div className="flex items-center gap-3">
                        <div className="w-32 bg-black/30 rounded-full h-2">
                          <div 
                            className="bg-blue-500 h-2 rounded-full"
                            style={{ 
                              width: `${Math.min(100, (count / Math.max(...Object.values(diversionStats.by_airline))) * 100)}%` 
                            }}
                          />
                        </div>
                        <span className="text-blue-400 font-bold w-8 text-right">{count}</span>
                      </div>
                    </div>
                  ))}
                {Object.keys(diversionStats.by_airline).length === 0 && (
                  <p className="text-white/40 text-center py-4">No airline data</p>
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      {!diversionStats && (
        <div className="bg-surface rounded-xl p-8 border border-white/10 text-center">
          <ArrowRightLeft className="w-12 h-12 mx-auto mb-3 text-white/20" />
          <p className="text-white/40">Diversion statistics not available</p>
        </div>
      )}

      {/* Alternate Airports Section */}
      <div className="border-b border-white/10 pb-4 pt-8">
        <h2 className="text-white text-xl font-bold mb-2 flex items-center gap-2">
          <Building2 className="w-5 h-5 text-purple-500" />
          Alternate Airports Used
        </h2>
        <p className="text-white/60 text-sm">
          Airports used as alternates during diversions
        </p>
      </div>

      {alternateAirports.length > 0 ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {alternateAirports.slice(0, 6).map((alt, idx) => (
            <div key={alt.airport} className="bg-surface rounded-xl border border-white/10 p-5">
              <div className="flex items-start justify-between mb-3">
                <div>
                  <div className="text-white font-bold text-lg">{alt.airport}</div>
                  <div className="text-white/50 text-sm">Alternate airport</div>
                </div>
                <div className={`px-3 py-1 rounded-full text-sm font-bold ${
                  idx === 0 ? 'bg-purple-500/20 text-purple-400' :
                  idx === 1 ? 'bg-blue-500/20 text-blue-400' : 'bg-white/10 text-white/60'
                }`}>
                  #{idx + 1}
                </div>
              </div>
              <div className="flex items-center justify-between mb-3">
                <span className="text-white/60 text-sm">Times used</span>
                <span className="text-purple-400 font-bold text-xl">{alt.count}</span>
              </div>
              {alt.aircraft_types.length > 0 && (
                <div>
                  <div className="text-white/50 text-xs mb-1">Aircraft types</div>
                  <div className="flex flex-wrap gap-1">
                    {alt.aircraft_types.slice(0, 4).map(type => (
                      <span key={type} className="px-2 py-0.5 bg-surface-highlight rounded text-xs text-white/70">
                        {type}
                      </span>
                    ))}
                    {alt.aircraft_types.length > 4 && (
                      <span className="px-2 py-0.5 bg-surface-highlight rounded text-xs text-white/50">
                        +{alt.aircraft_types.length - 4}
                      </span>
                    )}
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      ) : (
        <div className="bg-surface rounded-xl p-8 border border-white/10 text-center">
          <Building2 className="w-12 h-12 mx-auto mb-3 text-white/20" />
          <p className="text-white/40">No alternate airports used in this period</p>
        </div>
      )}

      {/* Runway Usage Section */}
      <div className="border-b border-white/10 pb-4 pt-8">
        <h2 className="text-white text-xl font-bold mb-2 flex items-center gap-2">
          <BarChart3 className="w-5 h-5 text-cyan-500" />
          Runway Usage
        </h2>
        <p className="text-white/60 text-sm">
          Landing and takeoff distribution by runway
        </p>
      </div>

      <div className="bg-surface rounded-xl border border-white/10 p-6">
        {/* Airport Selector */}
        <div className="flex flex-col gap-3 mb-6">
          <label className="text-white/60 text-sm">Select Airport:</label>
          <div className="flex flex-wrap gap-2">
            {[
              { code: 'LLBG', name: 'Ben Gurion' },
              { code: 'LLER', name: 'Ramon' },
              { code: 'LLHA', name: 'Haifa' },
              { code: 'LLOV', name: 'Ovda' },
              { code: 'LLRD', name: 'Rosh Pina' },
              { code: 'LLET', name: 'Eilat' },
              { code: 'LLMZ', name: 'Mitzpe Ramon' }
            ].map(apt => (
              <button
                key={apt.code}
                onClick={() => setSelectedAirport(apt.code)}
                className={`px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                  selectedAirport === apt.code 
                    ? 'bg-cyan-500 text-white' 
                    : 'bg-surface-highlight text-white/60 hover:text-white hover:bg-white/10'
                }`}
                title={apt.name}
              >
                <span className="font-bold">{apt.code}</span>
                <span className="text-xs ml-1 opacity-70">({apt.name})</span>
              </button>
            ))}
          </div>
        </div>

        {runwayUsage.length > 0 ? (
          <div className="space-y-4">
            {/* Runway Bars */}
            <div className="space-y-3">
              {runwayUsage.map(rwy => (
                <div key={rwy.runway} className="bg-surface-highlight rounded-lg p-4">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-white font-bold text-lg">Runway {rwy.runway}</span>
                    <span className="text-cyan-400 font-bold">{rwy.total} ops</span>
                  </div>
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <div className="flex justify-between text-sm mb-1">
                        <span className="text-green-400">Landings</span>
                        <span className="text-white">{rwy.landings}</span>
                      </div>
                      <div className="w-full bg-black/30 rounded-full h-3">
                        <div 
                          className="bg-green-500 h-3 rounded-full"
                          style={{ 
                            width: `${rwy.total > 0 ? (rwy.landings / rwy.total) * 100 : 0}%` 
                          }}
                        />
                      </div>
                    </div>
                    <div>
                      <div className="flex justify-between text-sm mb-1">
                        <span className="text-blue-400">Takeoffs</span>
                        <span className="text-white">{rwy.takeoffs}</span>
                      </div>
                      <div className="w-full bg-black/30 rounded-full h-3">
                        <div 
                          className="bg-blue-500 h-3 rounded-full"
                          style={{ 
                            width: `${rwy.total > 0 ? (rwy.takeoffs / rwy.total) * 100 : 0}%` 
                          }}
                        />
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>

            {/* Summary Chart */}
            <ChartCard title={`${selectedAirport} Runway Distribution`}>
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={runwayUsage} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" stroke="#ffffff20" />
                  <XAxis type="number" stroke="#ffffff60" tick={{ fill: '#ffffff60' }} />
                  <YAxis 
                    type="category" 
                    dataKey="runway" 
                    stroke="#ffffff60" 
                    tick={{ fill: '#ffffff60' }}
                    width={80}
                    tickFormatter={(v) => `RWY ${v}`}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#1a1a1a',
                      border: '1px solid #ffffff20',
                      borderRadius: '8px'
                    }}
                  />
                  <Bar dataKey="landings" fill="#10b981" name="Landings" stackId="a" />
                  <Bar dataKey="takeoffs" fill="#3b82f6" name="Takeoffs" stackId="a" />
                </BarChart>
              </ResponsiveContainer>
            </ChartCard>
          </div>
        ) : (
          <div className="text-center py-8">
            <BarChart3 className="w-12 h-12 mx-auto mb-3 text-white/20" />
            <p className="text-white/40">No runway data available for {selectedAirport}</p>
          </div>
        )}
      </div>
    </div>
  );
}

