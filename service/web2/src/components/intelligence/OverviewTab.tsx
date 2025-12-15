import { useState, useEffect } from 'react';
import { AlertTriangle, Plane, TrendingUp, AlertCircle } from 'lucide-react';
import { StatCard } from './StatCard';
import { ChartCard } from './ChartCard';
import { fetchStatsOverview, fetchFlightsPerDay } from '../../api';
import type { OverviewStats, FlightPerDay } from '../../types';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

interface OverviewTabProps {
  startTs: number;
  endTs: number;
  cacheKey?: number; // Force refresh when this changes
}

export function OverviewTab({ startTs, endTs, cacheKey = 0 }: OverviewTabProps) {
  const [stats, setStats] = useState<OverviewStats | null>(null);
  const [flightsPerDay, setFlightsPerDay] = useState<FlightPerDay[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadData();
  }, [startTs, endTs, cacheKey]);

  const loadData = async () => {
    setLoading(true);
    try {
      const [overviewData, flightsData] = await Promise.all([
        fetchStatsOverview(startTs, endTs),
        fetchFlightsPerDay(startTs, endTs)
      ]);
      setStats(overviewData);
      setFlightsPerDay(flightsData);
    } catch (error) {
      console.error('Failed to load overview data:', error);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-white/60">Loading overview...</div>
      </div>
    );
  }

  if (!stats) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-white/60">No data available</div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          title="Total Flights"
          value={stats.total_flights.toLocaleString()}
          subtitle="Tracked flights"
          icon={<Plane className="w-6 h-6" />}
        />
        <StatCard
          title="Anomaly Flights"
          value={stats.total_anomalies.toLocaleString()}
          subtitle="Flights with anomalies"
          icon={<AlertTriangle className="w-6 h-6" />}
        />
        <StatCard
          title="Safety Events"
          value={stats.safety_events.toLocaleString()}
          subtitle="Critical incidents"
          icon={<AlertCircle className="w-6 h-6" />}
        />
        <StatCard
          title="Go-Arounds"
          value={stats.go_arounds.toLocaleString()}
          subtitle="Aborted landings"
          icon={<TrendingUp className="w-6 h-6" />}
        />
      </div>

      {/* Flights Per Day Chart */}
      <ChartCard title="Flights Per Day (Last 30 Days)">
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
                labelStyle={{ color: '#fff' }}
              />
              <Line 
                type="monotone" 
                dataKey="count" 
                stroke="#3b82f6" 
                strokeWidth={2}
                name="Total Flights"
              />
              <Line 
                type="monotone" 
                dataKey="military_count" 
                stroke="#ef4444" 
                strokeWidth={2}
                name="Military"
              />
            </LineChart>
          </ResponsiveContainer>
        ) : (
          <div className="h-64 flex items-center justify-center text-white/40">
            No flight data available
          </div>
        )}
      </ChartCard>

      {/* Additional Stats Row */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <StatCard
          title="Emergency Codes"
          value={stats.emergency_codes.toLocaleString()}
          subtitle="7700/7600/7500"
        />
        <StatCard
          title="Near-Miss Events"
          value={stats.near_miss.toLocaleString()}
          subtitle="Proximity violations"
        />
        <StatCard
          title="Detection Rate"
          value={`${((stats.total_anomalies / Math.max(stats.total_flights, 1)) * 100).toFixed(1)}%`}
          subtitle="Anomalies per flight"
        />
      </div>
    </div>
  );
}

