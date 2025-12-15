import { useState, useEffect } from 'react';
import { Activity, AlertTriangle } from 'lucide-react';
import { fetchAirspaceRisk } from '../api';
import type { AirspaceRisk } from '../types';

export function RiskScoreWidget() {
  const [risk, setRisk] = useState<AirspaceRisk | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadRisk();
    // Refresh every minute
    const interval = setInterval(loadRisk, 60000);
    return () => clearInterval(interval);
  }, []);

  const loadRisk = async () => {
    try {
      const data = await fetchAirspaceRisk();
      setRisk(data);
      setLoading(false);
    } catch (error) {
      console.error('Failed to load risk score:', error);
      setLoading(false);
    }
  };

  if (loading || !risk) {
    return null;
  }

  const getRiskColor = (level: string) => {
    switch (level) {
      case 'critical': return 'text-red-600 bg-red-500/20 border-red-500';
      case 'high': return 'text-orange-500 bg-orange-500/20 border-orange-500';
      case 'medium': return 'text-yellow-500 bg-yellow-500/20 border-yellow-500';
      case 'low': return 'text-green-500 bg-green-500/20 border-green-500';
      default: return 'text-white bg-surface border-white/10';
    }
  };

  return (
    <div className={`absolute top-4 right-4 rounded-xl p-4 border-2 backdrop-blur-sm z-10 ${getRiskColor(risk.risk_level)}`}>
      <div className="flex items-center gap-3">
        <Activity className="w-6 h-6" />
        <div>
          <div className="text-xs font-medium opacity-80">Airspace Risk</div>
          <div className="text-2xl font-bold">{risk.risk_score}</div>
          <div className="text-xs uppercase font-bold">{risk.risk_level}</div>
        </div>
      </div>
      {risk.risk_level === 'critical' || risk.risk_level === 'high' ? (
        <div className="mt-2 pt-2 border-t border-current/20">
          <div className="flex items-start gap-2">
            <AlertTriangle className="w-4 h-4 flex-shrink-0 mt-0.5" />
            <p className="text-xs">{risk.recommendation}</p>
          </div>
        </div>
      ) : null}
    </div>
  );
}

