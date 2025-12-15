import { useState, useEffect } from 'react';
import { AlertTriangle, AlertOctagon, Activity } from 'lucide-react';
import { StatCard } from './StatCard';
import { TableCard, Column } from './TableCard';
import { ChartCard } from './ChartCard';
import { fetchEmergencyCodes, fetchNearMissEvents, fetchGoArounds } from '../../api';
import type { EmergencyCodeStat, NearMissEvent, GoAroundStat } from '../../types';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

interface SafetyTabProps {
  startTs: number;
  endTs: number;
  cacheKey?: number;
}

export function SafetyTab({ startTs, endTs, cacheKey = 0 }: SafetyTabProps) {
  const [emergencyCodes, setEmergencyCodes] = useState<EmergencyCodeStat[]>([]);
  const [nearMiss, setNearMiss] = useState<NearMissEvent[]>([]);
  const [goArounds, setGoArounds] = useState<GoAroundStat[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadData();
  }, [startTs, endTs, cacheKey]);

  const loadData = async () => {
    setLoading(true);
    try {
      const [codesData, nearMissData, goAroundData] = await Promise.all([
        fetchEmergencyCodes(startTs, endTs),
        fetchNearMissEvents(startTs, endTs),
        fetchGoArounds(startTs, endTs)
      ]);
      setEmergencyCodes(codesData);
      setNearMiss(nearMissData);
      setGoArounds(goAroundData);
    } catch (error) {
      console.error('Failed to load safety data:', error);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-white/60">Loading safety statistics...</div>
      </div>
    );
  }

  const totalEmergencies = emergencyCodes.reduce((sum, code) => sum + code.count, 0);
  const highSeverityNearMiss = nearMiss.filter(e => e.severity === 'high').length;
  const totalGoArounds = goArounds.reduce((sum, ga) => sum + ga.count, 0);

  const nearMissColumns: Column[] = [
    { key: 'timestamp', title: 'Time', render: (val) => new Date(val * 1000).toLocaleString() },
    { key: 'flight_id', title: 'Flight' },
    { key: 'other_flight_id', title: 'Other Flight' },
    { key: 'distance_nm', title: 'Distance (nm)' },
    { key: 'altitude_diff_ft', title: 'Alt Diff (ft)' },
    { 
      key: 'severity', 
      title: 'Severity',
      render: (val) => (
        <span className={val === 'high' ? 'text-red-500 font-bold' : 'text-yellow-500'}>
          {val.toUpperCase()}
        </span>
      )
    }
  ];

  return (
    <div className="space-y-6">
      {/* Key Safety Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <StatCard
          title="Emergency Codes"
          value={totalEmergencies}
          subtitle="7700/7600/7500 squawks"
          icon={<AlertTriangle className="w-6 h-6" />}
        />
        <StatCard
          title="Near-Miss Events"
          value={nearMiss.length}
          subtitle={`${highSeverityNearMiss} high severity`}
          icon={<AlertOctagon className="w-6 h-6" />}
        />
        <StatCard
          title="Go-Arounds"
          value={totalGoArounds}
          subtitle="Aborted landings"
          icon={<Activity className="w-6 h-6" />}
        />
      </div>

      {/* Emergency Codes Breakdown */}
      <ChartCard title="Emergency Codes by Type">
        {emergencyCodes.length > 0 ? (
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={emergencyCodes}>
              <CartesianGrid strokeDasharray="3 3" stroke="#ffffff20" />
              <XAxis dataKey="code" stroke="#ffffff60" tick={{ fill: '#ffffff60' }} />
              <YAxis stroke="#ffffff60" tick={{ fill: '#ffffff60' }} />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#1a1a1a',
                  border: '1px solid #ffffff20',
                  borderRadius: '8px'
                }}
              />
              <Bar dataKey="count" fill="#ef4444" />
            </BarChart>
          </ResponsiveContainer>
        ) : (
          <div className="h-64 flex items-center justify-center text-white/40">
            No emergency codes in this period
          </div>
        )}
      </ChartCard>

      {/* Go-Around Statistics by Airport */}
      <ChartCard title="Go-Arounds by Airport">
        {goArounds.length > 0 ? (
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={goArounds.slice(0, 10)}>
              <CartesianGrid strokeDasharray="3 3" stroke="#ffffff20" />
              <XAxis dataKey="airport" stroke="#ffffff60" tick={{ fill: '#ffffff60' }} />
              <YAxis stroke="#ffffff60" tick={{ fill: '#ffffff60' }} />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#1a1a1a',
                  border: '1px solid #ffffff20',
                  borderRadius: '8px'
                }}
              />
              <Bar dataKey="count" fill="#f59e0b" />
            </BarChart>
          </ResponsiveContainer>
        ) : (
          <div className="h-64 flex items-center justify-center text-white/40">
            No go-around events in this period
          </div>
        )}
      </ChartCard>

      {/* Near-Miss Events Table */}
      <TableCard
        title="Recent Near-Miss Events"
        columns={nearMissColumns}
        data={nearMiss.slice(0, 20)}
      />
    </div>
  );
}

