import { useState, useEffect, useRef } from 'react';
import { Activity, AlertTriangle, TrendingUp, Shield, Search, Target, Navigation, MapPin, Crosshair } from 'lucide-react';
import { StatCard } from './StatCard';
import { ChartCard } from './ChartCard';
import { fetchAirspaceRisk, fetchSafetyForecast, fetchTrajectoryPrediction } from '../../api';
import type { AirspaceRisk, SafetyForecast, RiskFactor } from '../../types';
import type { TrajectoryPrediction } from '../../api';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import maplibregl from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8001';

interface HostileIntentResult {
  intent_score: number;
  risk_level: 'low' | 'medium' | 'high' | 'critical' | 'unknown';
  factors: { name: string; score: number; description: string }[];
  recommendation: string;
  confidence: number;
  track_points_analyzed: number;
}

export function PredictTab() {
  const [airspaceRisk, setAirspaceRisk] = useState<AirspaceRisk | null>(null);
  const [forecast, setForecast] = useState<SafetyForecast | null>(null);
  const [loading, setLoading] = useState(true);
  
  // Hostile intent analysis state
  const [flightIdInput, setFlightIdInput] = useState('');
  const [hostileResult, setHostileResult] = useState<HostileIntentResult | null>(null);
  const [hostileLoading, setHostileLoading] = useState(false);
  const [hostileError, setHostileError] = useState<string | null>(null);
  
  // Trajectory prediction state
  const [trajectoryFlightId, setTrajectoryFlightId] = useState('');
  const [trajectoryResult, setTrajectoryResult] = useState<TrajectoryPrediction | null>(null);
  const [trajectoryLoading, setTrajectoryLoading] = useState(false);
  const [trajectoryError, setTrajectoryError] = useState<string | null>(null);
  
  // Trajectory map refs
  const trajectoryMapContainer = useRef<HTMLDivElement>(null);
  const trajectoryMap = useRef<maplibregl.Map | null>(null);

  useEffect(() => {
    loadData();
    // Refresh risk score every minute
    const interval = setInterval(loadData, 60000);
    return () => clearInterval(interval);
  }, []);

  const loadData = async () => {
    setLoading(true);
    try {
      const [riskData, forecastData] = await Promise.all([
        fetchAirspaceRisk(),
        fetchSafetyForecast(24)
      ]);
      setAirspaceRisk(riskData);
      setForecast(forecastData);
    } catch (error) {
      console.error('Failed to load predictive data:', error);
    } finally {
      setLoading(false);
    }
  };

  const analyzeHostileIntent = async () => {
    if (!flightIdInput.trim()) {
      setHostileError('Please enter a flight ID');
      return;
    }
    
    setHostileLoading(true);
    setHostileError(null);
    setHostileResult(null);
    
    try {
      const response = await fetch(`${API_BASE}/api/predict/hostile-intent/${flightIdInput.trim()}`, {
        method: 'POST'
      });
      
      if (!response.ok) {
        if (response.status === 404) {
          throw new Error('Flight not found - no track data available');
        }
        throw new Error('Failed to analyze flight');
      }
      
      const data = await response.json();
      setHostileResult(data);
    } catch (error) {
      setHostileError(error instanceof Error ? error.message : 'Analysis failed');
    } finally {
      setHostileLoading(false);
    }
  };

  const getHostileRiskColor = (level: string) => {
    switch (level) {
      case 'critical': return 'text-red-500';
      case 'high': return 'text-orange-500';
      case 'medium': return 'text-yellow-500';
      case 'low': return 'text-green-500';
      default: return 'text-white/60';
    }
  };

  const getHostileRiskBg = (level: string) => {
    switch (level) {
      case 'critical': return 'bg-red-500/20 border-red-500';
      case 'high': return 'bg-orange-500/20 border-orange-500';
      case 'medium': return 'bg-yellow-500/20 border-yellow-500';
      case 'low': return 'bg-green-500/20 border-green-500';
      default: return 'bg-surface border-white/20';
    }
  };

  // Fetch trajectory prediction
  const analyzeTrajectory = async () => {
    if (!trajectoryFlightId.trim()) {
      setTrajectoryError('Please enter a flight ID');
      return;
    }
    
    setTrajectoryLoading(true);
    setTrajectoryError(null);
    setTrajectoryResult(null);
    
    try {
      const data = await fetchTrajectoryPrediction(trajectoryFlightId.trim());
      setTrajectoryResult(data);
    } catch (error) {
      setTrajectoryError(error instanceof Error ? error.message : 'Failed to predict trajectory');
    } finally {
      setTrajectoryLoading(false);
    }
  };

  // Initialize/update trajectory map when result changes
  useEffect(() => {
    if (!trajectoryMapContainer.current || !trajectoryResult) return;
    
    // Initialize map if needed
    if (!trajectoryMap.current) {
      trajectoryMap.current = new maplibregl.Map({
        container: trajectoryMapContainer.current,
        style: 'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json',
        center: [35.0, 31.5],
        zoom: 7
      });
      trajectoryMap.current.addControl(new maplibregl.NavigationControl(), 'top-right');
    }
    
    const map = trajectoryMap.current;
    
    // Wait for map to load
    const setupMap = () => {
      // Remove existing layers/sources
      if (map.getLayer('trajectory-line')) map.removeLayer('trajectory-line');
      if (map.getSource('trajectory')) map.removeSource('trajectory');
      if (map.getLayer('breach-point')) map.removeLayer('breach-point');
      if (map.getSource('breach')) map.removeSource('breach');
      
      // Add trajectory line
      if (trajectoryResult.predicted_path && trajectoryResult.predicted_path.length > 0) {
        const coordinates = trajectoryResult.predicted_path.map(p => [p.lon, p.lat]);
        
        map.addSource('trajectory', {
          type: 'geojson',
          data: {
            type: 'Feature',
            properties: {},
            geometry: {
              type: 'LineString',
              coordinates: coordinates
            }
          }
        });
        
        map.addLayer({
          id: 'trajectory-line',
          type: 'line',
          source: 'trajectory',
          layout: {
            'line-join': 'round',
            'line-cap': 'round'
          },
          paint: {
            'line-color': trajectoryResult.breach_warning ? '#ef4444' : '#3b82f6',
            'line-width': 4,
            'line-dasharray': [2, 2]
          }
        });
        
        // Fit bounds to trajectory
        const bounds = new maplibregl.LngLatBounds();
        coordinates.forEach(coord => bounds.extend(coord as [number, number]));
        map.fitBounds(bounds, { padding: 50 });
        
        // Add breach point marker if there's a warning
        if (trajectoryResult.breach_warning && trajectoryResult.predicted_path.length > 0) {
          // Find approximate breach point (use last point for now)
          const breachPoint = trajectoryResult.predicted_path[trajectoryResult.predicted_path.length - 1];
          
          map.addSource('breach', {
            type: 'geojson',
            data: {
              type: 'Feature',
              properties: {},
              geometry: {
                type: 'Point',
                coordinates: [breachPoint.lon, breachPoint.lat]
              }
            }
          });
          
          map.addLayer({
            id: 'breach-point',
            type: 'circle',
            source: 'breach',
            paint: {
              'circle-radius': 12,
              'circle-color': '#ef4444',
              'circle-stroke-width': 3,
              'circle-stroke-color': '#ffffff'
            }
          });
        }
      }
    };
    
    if (map.loaded()) {
      setupMap();
    } else {
      map.on('load', setupMap);
    }
  }, [trajectoryResult]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-white/60">Loading predictive analytics...</div>
      </div>
    );
  }

  const getRiskColor = (level: string) => {
    switch (level) {
      case 'critical': return 'text-red-600';
      case 'high': return 'text-orange-500';
      case 'medium': return 'text-yellow-500';
      case 'low': return 'text-green-500';
      default: return 'text-white';
    }
  };

  const getRiskBgColor = (level: string) => {
    switch (level) {
      case 'critical': return 'bg-red-500/20 border-red-500';
      case 'high': return 'bg-orange-500/20 border-orange-500';
      case 'medium': return 'bg-yellow-500/20 border-yellow-500';
      case 'low': return 'bg-green-500/20 border-green-500';
      default: return 'bg-surface';
    }
  };

  return (
    <div className="space-y-6">
      {/* Real-time Airspace Risk */}
      {airspaceRisk && (
        <>
          <div className="border-b border-white/10 pb-4">
            <h2 className="text-white text-xl font-bold mb-4 flex items-center gap-2">
              <Activity className="w-5 h-5" />
              Real-Time Airspace Risk Assessment
            </h2>
          </div>

          {/* Risk Score Display */}
          <div className={`rounded-xl p-8 border-2 ${getRiskBgColor(airspaceRisk.risk_level)}`}>
            <div className="text-center">
              <div className="text-white/60 text-sm mb-2">Current Risk Score</div>
              <div className={`text-6xl font-bold mb-2 ${getRiskColor(airspaceRisk.risk_level)}`}>
                {airspaceRisk.risk_score}
              </div>
              <div className={`text-2xl font-bold uppercase ${getRiskColor(airspaceRisk.risk_level)}`}>
                {airspaceRisk.risk_level}
              </div>
              <div className="text-white/80 text-sm mt-4 max-w-2xl mx-auto">
                {airspaceRisk.recommendation}
              </div>
            </div>
          </div>

          {/* Risk Factors Breakdown */}
          <ChartCard title="Risk Factors Contribution">
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={airspaceRisk.factors} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" stroke="#ffffff20" />
                <XAxis type="number" stroke="#ffffff60" tick={{ fill: '#ffffff60' }} />
                <YAxis 
                  type="category" 
                  dataKey="name" 
                  stroke="#ffffff60" 
                  tick={{ fill: '#ffffff60' }}
                  width={150}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#1a1a1a',
                    border: '1px solid #ffffff20',
                    borderRadius: '8px'
                  }}
                />
                <Bar dataKey="impact" fill="#3b82f6" />
              </BarChart>
            </ResponsiveContainer>
          </ChartCard>

          {/* Factor Details */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {airspaceRisk.factors.map((factor, idx) => (
              <div key={idx} className="bg-surface rounded-lg p-4 border border-white/10">
                <div className="flex justify-between items-start mb-2">
                  <h4 className="text-white font-bold">{factor.name}</h4>
                  <span className="text-white/60 text-sm">Weight: {factor.weight}</span>
                </div>
                <div className="text-white/80 text-sm mb-2">
                  {factor.description}
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-white/40 text-xs">Value: {factor.value}</span>
                  <span className="text-primary font-bold">Impact: {factor.impact}</span>
                </div>
              </div>
            ))}
          </div>
        </>
      )}

      {/* Safety Forecast */}
      {forecast && (
        <>
          <div className="border-b border-white/10 pb-4 pt-8">
            <h2 className="text-white text-xl font-bold mb-4 flex items-center gap-2">
              <TrendingUp className="w-5 h-5" />
              Safety Event Forecast (Next {forecast.forecast_period_hours}h)
            </h2>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <StatCard
              title="Expected Events"
              value={forecast.expected_events ?? 0}
              subtitle={`Forecast for next ${forecast.forecast_period_hours ?? 24} hours`}
              icon={<AlertTriangle className="w-6 h-6" />}
            />
            <StatCard
              title="Confidence Interval"
              value={forecast.confidence_interval ? `${forecast.confidence_interval[0]} - ${forecast.confidence_interval[1]}` : 'N/A'}
              subtitle="Range of expected events"
            />
            <StatCard
              title="Peak Risk Hours"
              value={forecast.peak_risk_hours?.length > 0 ? forecast.peak_risk_hours.join(', ') : 'None'}
              subtitle="Highest risk time periods"
            />
          </div>

          {/* Experimental Notice */}
          <div className="bg-yellow-500/10 border border-yellow-500/30 rounded-lg p-4">
            <div className="flex items-start gap-3">
              <AlertTriangle className="w-5 h-5 text-yellow-500 flex-shrink-0 mt-0.5" />
              <div>
                <h4 className="text-yellow-500 font-bold mb-1">Experimental Feature</h4>
                <p className="text-white/80 text-sm">
                  Predictive analytics are based on historical patterns and statistical models. 
                  Use these forecasts as supplementary information alongside standard operational procedures.
                </p>
              </div>
            </div>
          </div>
        </>
      )}

      {/* Trajectory Prediction with Restricted Zones */}
      <div className="border-b border-white/10 pb-4 pt-8">
        <h2 className="text-white text-xl font-bold mb-4 flex items-center gap-2">
          <Navigation className="w-5 h-5 text-cyan-500" />
          Trajectory Prediction & Restricted Zone Analysis
        </h2>
        <p className="text-white/60 text-sm">
          Predict flight trajectory and check for potential restricted airspace breaches
        </p>
      </div>

      {/* Trajectory Input */}
      <div className="bg-surface rounded-xl border border-white/10 p-6">
        <div className="flex gap-4 items-end">
          <div className="flex-1">
            <label className="block text-white/70 text-sm mb-2">Flight ID</label>
            <input
              type="text"
              value={trajectoryFlightId}
              onChange={(e) => setTrajectoryFlightId(e.target.value)}
              placeholder="Enter flight ID to predict trajectory"
              className="w-full px-4 py-3 bg-surface-highlight border border-white/20 rounded-lg text-white placeholder-white/40 focus:outline-none focus:border-cyan-500"
              onKeyDown={(e) => e.key === 'Enter' && analyzeTrajectory()}
            />
          </div>
          <button
            onClick={analyzeTrajectory}
            disabled={trajectoryLoading}
            className="px-6 py-3 bg-cyan-600 hover:bg-cyan-700 disabled:bg-cyan-600/50 text-white font-medium rounded-lg flex items-center gap-2 transition-colors"
          >
            {trajectoryLoading ? (
              <>
                <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                Predicting...
              </>
            ) : (
              <>
                <Navigation className="w-4 h-4" />
                Predict Path
              </>
            )}
          </button>
        </div>

        {trajectoryError && (
          <div className="mt-4 p-3 bg-red-500/10 border border-red-500/30 rounded-lg text-red-400 text-sm">
            {trajectoryError}
          </div>
        )}
      </div>

      {/* Trajectory Results */}
      {trajectoryResult && (
        <div className="space-y-4">
          {/* Status Cards */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {/* Breach Warning */}
            <div className={`rounded-xl p-6 border-2 ${
              trajectoryResult.breach_warning 
                ? 'bg-red-500/10 border-red-500' 
                : 'bg-green-500/10 border-green-500'
            }`}>
              <div className="flex items-center gap-3 mb-2">
                {trajectoryResult.breach_warning ? (
                  <AlertTriangle className="w-6 h-6 text-red-500" />
                ) : (
                  <Shield className="w-6 h-6 text-green-500" />
                )}
                <span className={`text-lg font-bold ${
                  trajectoryResult.breach_warning ? 'text-red-400' : 'text-green-400'
                }`}>
                  {trajectoryResult.breach_warning ? 'BREACH WARNING' : 'CLEAR PATH'}
                </span>
              </div>
              <p className="text-white/60 text-sm">
                {trajectoryResult.breach_warning 
                  ? 'Predicted trajectory may enter restricted airspace'
                  : 'No restricted zone breaches predicted'}
              </p>
            </div>

            {/* Confidence */}
            <div className="bg-surface rounded-xl border border-white/10 p-6">
              <div className="text-white/60 text-sm mb-1">Prediction Confidence</div>
              <div className="text-3xl font-bold text-cyan-400">
                {(trajectoryResult.prediction_confidence * 100).toFixed(0)}%
              </div>
              <div className="mt-2 w-full bg-black/30 rounded-full h-2">
                <div 
                  className="bg-cyan-500 h-2 rounded-full"
                  style={{ width: `${trajectoryResult.prediction_confidence * 100}%` }}
                />
              </div>
            </div>

            {/* Closest Zone */}
            {trajectoryResult.closest_zone && (
              <div className="bg-surface rounded-xl border border-white/10 p-6">
                <div className="text-white/60 text-sm mb-1">Closest Restricted Zone</div>
                <div className="text-xl font-bold text-white mb-1">
                  {trajectoryResult.closest_zone.name}
                </div>
                <div className="flex items-center gap-2">
                  <MapPin className="w-4 h-4 text-orange-400" />
                  <span className="text-orange-400 font-medium">
                    {trajectoryResult.closest_zone.distance_nm.toFixed(1)} nm away
                  </span>
                </div>
              </div>
            )}
          </div>

          {/* Breach Details */}
          {trajectoryResult.breach_warning && trajectoryResult.breach_zone && (
            <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-6">
              <div className="flex items-start gap-4">
                <Crosshair className="w-8 h-8 text-red-500 flex-shrink-0" />
                <div>
                  <h3 className="text-red-400 font-bold text-lg mb-2">Restricted Zone Breach Detected</h3>
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <div className="text-white/60 text-sm">Zone Name</div>
                      <div className="text-white font-bold">{trajectoryResult.breach_zone}</div>
                    </div>
                    <div>
                      <div className="text-white/60 text-sm">Severity</div>
                      <div className={`font-bold ${
                        trajectoryResult.breach_severity === 'critical' ? 'text-red-400' :
                        trajectoryResult.breach_severity === 'high' ? 'text-orange-400' : 'text-yellow-400'
                      }`}>
                        {trajectoryResult.breach_severity?.toUpperCase() || 'UNKNOWN'}
                      </div>
                    </div>
                  </div>
                  <p className="text-white/70 text-sm mt-3">
                    Immediate attention required. Consider contacting ATC and monitoring flight closely.
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* Trajectory Map */}
          <div className="bg-surface rounded-xl border border-white/10 overflow-hidden">
            <div className="px-6 py-4 border-b border-white/10">
              <h3 className="text-white font-bold flex items-center gap-2">
                <Navigation className="w-4 h-4 text-cyan-400" />
                Predicted Flight Path
              </h3>
              <p className="text-white/60 text-sm mt-1">
                Showing {trajectoryResult.predicted_path?.length || 0} predicted waypoints
              </p>
            </div>
            <div 
              ref={trajectoryMapContainer}
              className="h-[400px] w-full"
            />
            {/* Map Legend */}
            <div className="px-6 py-3 bg-surface-highlight border-t border-white/10 flex gap-6">
              <div className="flex items-center gap-2">
                <div className="w-8 h-1 bg-blue-500 rounded" style={{ borderStyle: 'dashed' }} />
                <span className="text-white/60 text-xs">Safe trajectory</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-8 h-1 bg-red-500 rounded" style={{ borderStyle: 'dashed' }} />
                <span className="text-white/60 text-xs">Breach trajectory</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 bg-red-500 rounded-full border-2 border-white" />
                <span className="text-white/60 text-xs">Breach point</span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Hostile Intent Analysis */}
      <div className="border-b border-white/10 pb-4 pt-8">
        <h2 className="text-white text-xl font-bold mb-4 flex items-center gap-2">
          <Shield className="w-5 h-5 text-red-500" />
          Hostile Intent Analysis
        </h2>
        <p className="text-white/60 text-sm">
          Analyze a specific flight for potential hostile behavior patterns
        </p>
      </div>

      {/* Flight ID Input */}
      <div className="bg-surface rounded-xl border border-white/10 p-6">
        <div className="flex gap-4 items-end">
          <div className="flex-1">
            <label className="block text-white/70 text-sm mb-2">Flight ID</label>
            <input
              type="text"
              value={flightIdInput}
              onChange={(e) => setFlightIdInput(e.target.value)}
              placeholder="Enter flight ID (e.g., 3b86ff46)"
              className="w-full px-4 py-3 bg-surface-highlight border border-white/20 rounded-lg text-white placeholder-white/40 focus:outline-none focus:border-primary"
              onKeyDown={(e) => e.key === 'Enter' && analyzeHostileIntent()}
            />
          </div>
          <button
            onClick={analyzeHostileIntent}
            disabled={hostileLoading}
            className="px-6 py-3 bg-red-600 hover:bg-red-700 disabled:bg-red-600/50 text-white font-medium rounded-lg flex items-center gap-2 transition-colors"
          >
            {hostileLoading ? (
              <>
                <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                Analyzing...
              </>
            ) : (
              <>
                <Target className="w-4 h-4" />
                Analyze Intent
              </>
            )}
          </button>
        </div>

        {hostileError && (
          <div className="mt-4 p-3 bg-red-500/10 border border-red-500/30 rounded-lg text-red-400 text-sm">
            {hostileError}
          </div>
        )}
      </div>

      {/* Hostile Intent Results */}
      {hostileResult && (
        <div className="space-y-4">
          {/* Main Score Display */}
          <div className={`rounded-xl p-6 border-2 ${getHostileRiskBg(hostileResult.risk_level)}`}>
            <div className="flex items-center justify-between">
              <div>
                <div className="text-white/60 text-sm mb-1">Intent Score</div>
                <div className={`text-5xl font-bold ${getHostileRiskColor(hostileResult.risk_level)}`}>
                  {hostileResult.intent_score}
                </div>
                <div className={`text-lg font-bold uppercase mt-1 ${getHostileRiskColor(hostileResult.risk_level)}`}>
                  {hostileResult.risk_level}
                </div>
              </div>
              <div className="text-right">
                <div className="text-white/60 text-sm mb-1">Confidence</div>
                <div className="text-2xl font-bold text-white">{(hostileResult.confidence * 100).toFixed(0)}%</div>
                <div className="text-white/40 text-xs mt-1">
                  {hostileResult.track_points_analyzed} points analyzed
                </div>
              </div>
            </div>
            <div className="mt-4 pt-4 border-t border-white/10">
              <div className="text-white/80 text-sm font-medium">
                {hostileResult.recommendation}
              </div>
            </div>
          </div>

          {/* Factor Breakdown */}
          <div className="bg-surface rounded-xl border border-white/10 p-6">
            <h3 className="text-white font-bold mb-4 flex items-center gap-2">
              <Search className="w-4 h-4" />
              Analysis Factors
            </h3>
            <div className="space-y-3">
              {hostileResult.factors.map((factor, idx) => (
                <div key={idx} className="bg-surface-highlight rounded-lg p-4">
                  <div className="flex justify-between items-start mb-2">
                    <span className="text-white font-medium">{factor.name}</span>
                    <span className={`font-bold ${factor.score >= 15 ? 'text-red-400' : factor.score >= 8 ? 'text-yellow-400' : 'text-green-400'}`}>
                      +{factor.score} pts
                    </span>
                  </div>
                  <p className="text-white/60 text-sm">{factor.description}</p>
                </div>
              ))}
            </div>
          </div>

          {/* Warning for high scores */}
          {hostileResult.intent_score >= 50 && (
            <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-4">
              <div className="flex items-start gap-3">
                <AlertTriangle className="w-5 h-5 text-red-500 flex-shrink-0 mt-0.5" />
                <div>
                  <h4 className="text-red-500 font-bold mb-1">High Intent Score Detected</h4>
                  <p className="text-white/80 text-sm">
                    This flight shows behavioral patterns that warrant immediate attention. 
                    Review the factor breakdown and consider escalating to security personnel.
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

