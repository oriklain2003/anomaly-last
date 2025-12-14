import React, { useState, useMemo } from 'react';
import type { AnalysisResult, RuleResult, RuleDefinition, AnomalyPoint } from '../types';
import { MapComponent, type MLAnomalyPoint } from './MapComponent';
import { AlertTriangle, CheckCircle, Hourglass, ChevronDown, MapPin } from 'lucide-react';
import clsx from 'clsx';

interface ResultsViewProps {
  data: AnalysisResult | null;
  flightId: string;
  isLoading: boolean;
  error: string | null;
}

export const ResultsView: React.FC<ResultsViewProps> = ({ data, flightId, isLoading, error }) => {
  const isAnomaly = data?.summary.is_anomaly;
  
  // Extract ML anomaly points for map visualization
  const mlAnomalyPoints = useMemo((): MLAnomalyPoint[] => {
    if (!data) return [];

    const points: MLAnomalyPoint[] = [];

    const layerMap: Record<string, string> = {
        'layer_3_deep_dense': 'Deep Dense',
        'layer_4_deep_cnn': 'Deep CNN',
        'layer_5_transformer': 'Transformer',
        'layer_6_hybrid': 'Hybrid'
    };

    Object.entries(layerMap).forEach(([key, layerName]) => {
        const layerData = (data as any)[key];
        if (layerData?.anomaly_points && layerData.is_anomaly) {
            layerData.anomaly_points.forEach((pt: AnomalyPoint) => {
                points.push({
                    lat: pt.lat,
                    lon: pt.lon,
                    timestamp: pt.timestamp,
                    point_score: pt.point_score,
                    layer: layerName
                });
            });
        }
    });

    return points;
  }, [data]);

  // Always render the map, use empty path if no data
  return (
    <div className="flex flex-col gap-6">
      {/* Header */}
      <div 
        className="flex flex-wrap items-center justify-between gap-4 p-4 rounded-xl bg-gray-50 dark:bg-white/5 border border-gray-200/50 dark:border-white/10 animate-in slide-in-from-bottom-4 fade-in duration-700 fill-mode-backwards"
      >
        <h2 className="text-xl sm:text-2xl font-bold leading-tight tracking-[-0.015em]">
          Results for Flight <span className="text-primary">{flightId || "..."}</span>
        </h2>
        <div className="flex items-center gap-3">
            {data && (
                <div className="hidden sm:flex items-center gap-2 px-3 py-1.5 rounded-full text-sm font-bold bg-blue-500/10 text-blue-600 dark:text-blue-400 border border-blue-500/20">
                   <span>Confidence: {data.summary.confidence_score}%</span>
                </div>
            )}
            <div
              className={clsx(
                "flex items-center gap-2 px-3 py-1.5 rounded-full text-sm font-semibold",
                isLoading
                  ? "bg-gray-500/10 text-gray-400"
                  : !data && !error 
                  ? "bg-gray-500/10 text-gray-400" // Initial state
                  : isAnomaly
                  ? "bg-red-500/10 text-red-400"
                  : "bg-green-500/10 text-green-400"
              )}
            >
              {isLoading ? (
                <>
                  <Hourglass className="h-4 w-4 animate-spin" />
                  <span>Processing...</span>
                </>
              ) : error ? (
                <>
                   <AlertTriangle className="h-4 w-4" />
                   <span>Error</span>
                </>
              ) : !data ? (
                <>
                   <span>Ready to Analyze</span>
                </>
              ) : isAnomaly ? (
                <>
                  <AlertTriangle className="h-4 w-4" />
                  <span>ANOMALY DETECTED</span>
                </>
              ) : (
                <>
                  <CheckCircle className="h-4 w-4" />
                  <span>NORMAL FLIGHT</span>
                </>
              )}
            </div>
        </div>
      </div>

      {error && (
        <div className="p-4 rounded-xl bg-red-50 dark:bg-red-900/10 border border-red-200 dark:border-red-900/20 flex items-center gap-3 text-red-600 dark:text-red-400 animate-in slide-in-from-bottom-4 fade-in duration-700 fill-mode-backwards" style={{ animationDelay: '100ms' }}>
          <AlertTriangle className="h-6 w-6" />
          <span className="font-bold">Error: {error}</span>
        </div>
      )}

      {/* Map - Always Visible */}
      <div 
        className="relative w-full aspect-[16/8] rounded-xl overflow-hidden bg-gray-200 dark:bg-gray-800 border border-gray-200/50 dark:border-white/10 animate-in slide-in-from-bottom-4 fade-in duration-700 fill-mode-backwards"
        style={{ animationDelay: '200ms' }}
      >
        <MapComponent 
          path={data?.summary.flight_path || []} 
          points={data?.track?.points}
          mlAnomalyPoints={mlAnomalyPoints}
        />
        <div className="absolute top-4 left-4 bg-black/70 backdrop-blur-sm text-white px-3 py-1.5 rounded-lg text-sm z-10 pointer-events-none">
          <p className="font-bold">Flight Path Visualization</p>
          {data && (
            <>
              <p className="text-xs opacity-80">Red segments indicate anomaly detections.</p>
              <p className="text-xs opacity-80">Analyzed Points: {data.summary.num_points}</p>
            </>
          )}
        </div>
      </div>

      {/* Cards Grid - Only show if we have data or loading */}
      {(data || isLoading) && (
        <div className={clsx("grid grid-cols-1 gap-6", isLoading && "opacity-50 pointer-events-none")}>
            
            {/* Rule Engine (Full Width) */}
            <RuleCard result={data?.layer_1_rules} delay="300ms" />

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                {/* XGBoost */}
                <ModelCard 
                  title="XGBoost Model"
                  status={data?.layer_2_xgboost?.status}
                  score={data?.layer_2_xgboost?.score}
                  isAnomaly={data?.layer_2_xgboost?.is_anomaly}
                  label="Anomaly Score"
                  barLabel="Probability"
                  delay="400ms"
                  anomalyPoints={data?.layer_2_xgboost?.anomaly_points}
                />

                {/* Deep Dense */}
                <ModelCard 
                  title="Deep Dense AE"
                  status={data?.layer_3_deep_dense?.status}
                  score={data?.layer_3_deep_dense?.severity}
                  isAnomaly={data?.layer_3_deep_dense?.is_anomaly}
                  label="Severity"
                  barLabel="Severity Ratio"
                  suffix="x"
                  scale={10} // Scale score * 10 for percentage if needed
                  delay="500ms"
                  anomalyPoints={data?.layer_3_deep_dense?.anomaly_points}
                />

                {/* Deep CNN */}
                <ModelCard 
                  title="Deep CNN"
                  status={data?.layer_4_deep_cnn?.status}
                  score={data?.layer_4_deep_cnn?.severity}
                  isAnomaly={data?.layer_4_deep_cnn?.is_anomaly}
                  label="Severity"
                  barLabel="Severity Ratio"
                  suffix="x"
                  scale={10}
                  delay="600ms"
                  anomalyPoints={data?.layer_4_deep_cnn?.anomaly_points}
                />

                {/* Transformer */}
                <ModelCard 
                  title="Transformer"
                  status={data?.layer_5_transformer?.status}
                  score={data?.layer_5_transformer?.score}
                  isAnomaly={data?.layer_5_transformer?.is_anomaly}
                  label="Anomaly Score"
                  barLabel="Probability"
                  delay="700ms"
                  anomalyPoints={data?.layer_5_transformer?.anomaly_points}
                />

                {/* Hybrid */}
                <ModelCard 
                  title="Hybrid CNN-Trans"
                  status={data?.layer_6_hybrid?.status}
                  score={data?.layer_6_hybrid?.score}
                  isAnomaly={data?.layer_6_hybrid?.is_anomaly}
                  label="Anomaly Score"
                  barLabel="Probability"
                  delay="800ms"
                  anomalyPoints={data?.layer_6_hybrid?.anomaly_points}
                />
            </div>
        </div>
      )}
    </div>
  );
};

const StatusBadge: React.FC<{ status?: string }> = ({ status = "PENDING" }) => {
  const isAnomaly = status === "ANOMALY";
  const isError = status === "ERROR";
  const isSkipped = status === "SKIPPED" || status === "SKIPPED_TOO_SHORT";
  
  let colorClass = "bg-gray-500/10 text-gray-400";
  if (isAnomaly) colorClass = "bg-red-500/10 text-red-400";
  else if (isError) colorClass = "bg-yellow-500/10 text-yellow-400";
  else if (!isSkipped && status !== "PENDING") colorClass = "bg-green-500/10 text-green-400";

  return (
    <span
      className={clsx(
        "text-sm font-semibold px-3 py-1 rounded-full uppercase",
        colorClass
      )}
    >
      {status}
    </span>
  );
};

const RuleCard: React.FC<{ result?: RuleResult; delay?: string }> = ({ result, delay }) => {
  const evaluations = result?.report?.evaluations || [];
  const hasEvaluations = evaluations.length > 0;

  // If we have old style triggers but no detailed report (backward compat)
  const simpleTriggers = result?.triggers || [];
  
  return (
    <div 
      className="flex flex-col gap-4 rounded-xl bg-gray-50 dark:bg-white/5 p-6 border border-gray-200/50 dark:border-white/10 transition-all hover:border-primary/50 animate-in slide-in-from-bottom-4 fade-in duration-700 fill-mode-backwards"
      style={{ animationDelay: delay }}
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
            <h3 className="text-lg font-bold">Rule Engine Analysis</h3>
            <span className="text-xs font-medium bg-gray-200 dark:bg-gray-700 px-2 py-0.5 rounded-full">Layer 1</span>
        </div>
        <StatusBadge status={result?.status} />
      </div>
      
      <div className="flex flex-col gap-3 text-sm">
        {!result ? (
           <div className="flex items-center gap-2 text-gray-500">
             <Hourglass className="h-4 w-4" />
             <span>Waiting for analysis...</span>
           </div>
        ) : (
          <>
             {/* If we have detailed evaluations */}
             {hasEvaluations ? (
                 <div className="flex flex-col gap-2">
                     {evaluations.map((rule) => (
                         <RuleItem key={rule.id} rule={rule} />
                     ))}
                 </div>
             ) : (
                 // Fallback for simple triggers
                 <div className="flex flex-col gap-2">
                     {simpleTriggers.length === 0 && result.status === "NORMAL" && (
                         <div className="p-3 rounded-lg bg-green-500/10 text-green-600 dark:text-green-400 flex items-center gap-2">
                             <CheckCircle className="h-4 w-4" />
                             <span>No rules triggered.</span>
                         </div>
                     )}
                     {simpleTriggers.map((t, i) => (
                         <div key={i} className="p-3 rounded-lg bg-red-500/10 text-red-600 dark:text-red-400 flex items-center gap-2">
                             <AlertTriangle className="h-4 w-4" />
                             <span>{t}</span>
                         </div>
                     ))}
                 </div>
             )}
          </>
        )}
      </div>
    </div>
  );
};

const RuleItem: React.FC<{ rule: RuleDefinition }> = ({ rule }) => {
    const [isOpen, setIsOpen] = useState(false);
    const hasEvents = rule.details?.events && rule.details.events.length > 0;
    const isPathDeviation = rule.id === 11 && rule.details?.segments;

    return (
        <div className={clsx(
            "flex flex-col rounded-lg border border-gray-200/10 overflow-hidden transition-colors",
            rule.matched ? "bg-red-500/5 border-red-500/20" : "bg-white/5 hover:bg-white/10"
        )}>
            <button 
                onClick={() => setIsOpen(!isOpen)}
                className="flex items-center justify-between p-3 w-full text-left"
            >
                <div className="flex items-center gap-3">
                    {rule.matched ? (
                        <AlertTriangle className="h-5 w-5 text-red-500 shrink-0" />
                    ) : (
                        <CheckCircle className="h-5 w-5 text-green-500 shrink-0" />
                    )}
                    <div className="flex flex-col">
                        <span className={clsx("font-medium", rule.matched ? "text-red-600 dark:text-red-400" : "text-gray-700 dark:text-gray-200")}>
                            {rule.name}
                        </span>
                        <span className="text-xs text-gray-500 dark:text-gray-400">{rule.summary}</span>
                    </div>
                </div>
                <div className="flex items-center gap-4">
                    {rule.matched && <span className="text-xs font-bold text-red-500 uppercase tracking-wider">Triggered</span>}
                    {(hasEvents || isPathDeviation) && (
                        <ChevronDown className={clsx("h-4 w-4 text-gray-400 transition-transform", isOpen && "rotate-180")} />
                    )}
                </div>
            </button>
            
            {/* Details Panel */}
            {isOpen && (
                 <div className="p-3 pt-0 pl-11 text-xs font-mono text-gray-600 dark:text-gray-300">
                     {hasEvents && (
                         <div className="p-2 bg-black/5 dark:bg-black/20 rounded border border-black/5 dark:border-white/5 overflow-x-auto mb-2">
                            <pre>{JSON.stringify(rule.details.events, null, 2)}</pre>
                         </div>
                     )}
                     
                     {isPathDeviation && (
                         <div className="space-y-2">
                             {Object.entries(rule.details.segments).map(([phase, data]: [string, any]) => (
                                 <div key={phase} className="p-2 bg-black/5 dark:bg-black/20 rounded border border-black/5 dark:border-white/5">
                                     <div className="flex justify-between mb-1">
                                         <span className="font-bold uppercase text-gray-500">{phase}</span>
                                         <span className={data.match_found ? "text-green-500" : "text-red-500"}>
                                             {data.match_found ? "MATCHED" : "NO MATCH"}
                                         </span>
                                     </div>
                                     <div className="grid grid-cols-2 gap-x-4 gap-y-1 opacity-70">
                                         {data.match_found ? (
                                             <>
                                                 <span>Flow ID: {data.flow_id}</span>
                                                 <span>Layer: {data.layer}</span>
                                                 <span>Distance: {data.dist_nm} NM</span>
                                             </>
                                         ) : (
                                             <span className="col-span-2 italic">No matching path found in library.</span>
                                         )}
                                         {data.closest_loose_dist_nm !== undefined && (
                                             <span className="col-span-2 mt-1 pt-1 border-t border-white/10 text-blue-400">
                                                 Distance to closest loose path: {data.closest_loose_dist_nm} NM
                                             </span>
                                         )}
                                     </div>
                                 </div>
                             ))}
                         </div>
                     )}
                 </div>
            )}
        </div>
    )
}

// Layer color mapping for ML models
const LAYER_COLORS: Record<string, { bg: string; border: string; text: string; accent: string }> = {
    'Deep Dense AE': { bg: 'bg-purple-500/15', border: 'border-purple-400', text: 'text-purple-400', accent: 'bg-purple-500/30' },
    'Deep CNN': { bg: 'bg-orange-500/15', border: 'border-orange-400', text: 'text-orange-400', accent: 'bg-orange-500/30' },
    'Transformer': { bg: 'bg-cyan-500/15', border: 'border-cyan-400', text: 'text-cyan-400', accent: 'bg-cyan-500/30' },
    'Hybrid CNN-Trans': { bg: 'bg-pink-500/15', border: 'border-pink-400', text: 'text-pink-400', accent: 'bg-pink-500/30' },
    'XGBoost Model': { bg: 'bg-blue-500/15', border: 'border-blue-400', text: 'text-blue-400', accent: 'bg-blue-500/30' },
};

interface ModelCardProps {
  title: string;
  status?: string;
  score?: number;
  isAnomaly?: boolean;
  label: string;
  barLabel: string;
  suffix?: string;
  scale?: number;
  delay?: string;
  anomalyPoints?: AnomalyPoint[];
}

const ModelCard: React.FC<ModelCardProps> = ({ title, status, score, isAnomaly, label, barLabel, suffix = "", scale = 100, delay, anomalyPoints = [] }) => {
  const percentage = Math.min((score || 0) * scale, 100);
  const [showPoints, setShowPoints] = useState(false);
  const layerColor = LAYER_COLORS[title] || { bg: 'bg-gray-500/15', border: 'border-gray-400', text: 'text-gray-400', accent: 'bg-gray-500/30' };
  
  return (
    <div 
      className="flex flex-col gap-4 rounded-xl bg-gray-50 dark:bg-white/5 p-6 border border-gray-200/50 dark:border-white/10 transition-all hover:border-primary/50 animate-in slide-in-from-bottom-4 fade-in duration-700 fill-mode-backwards"
      style={{ animationDelay: delay }}
    >
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-bold">{title}</h3>
        <StatusBadge status={status} />
      </div>
      <div className="flex flex-col gap-3 text-sm">
        <div className="flex justify-between items-center border-b border-gray-200/50 dark:border-white/10 pb-2">
          <span className="text-gray-500 dark:text-gray-400">{label}</span>
          <span className="font-medium">{score !== undefined ? score.toFixed(4) + suffix : "-"}</span>
        </div>
      </div>
      <div className="flex flex-col gap-2 pt-2">
        <label className="text-sm text-gray-500 dark:text-gray-400">{barLabel}</label>
        <div className="w-full bg-gray-200/50 dark:bg-white/10 rounded-full h-2.5 overflow-hidden">
          <div 
            className={clsx("h-2.5 rounded-full transition-all duration-1000", isAnomaly ? "bg-red-500" : "bg-green-500")} 
            style={{ width: `${percentage}%` }}
          ></div>
        </div>
      </div>
      
      {/* Anomaly Points Section */}
      <div className="mt-2 pt-3 border-t border-gray-200/50 dark:border-white/10">
        <button 
          onClick={() => setShowPoints(!showPoints)}
          className="flex items-center justify-between w-full text-left"
        >
          <div className="flex items-center gap-2">
            <MapPin className={clsx("h-4 w-4", isAnomaly ? layerColor.text : "text-gray-400")} />
            <span className={clsx("text-sm font-medium", isAnomaly ? layerColor.text : "text-gray-400")}>
              Detected Anomaly Locations
            </span>
          </div>
          <div className="flex items-center gap-2">
            {anomalyPoints.length > 0 && (
              <span className={clsx("text-xs px-2 py-0.5 rounded font-bold", layerColor.accent, layerColor.text)}>
                {anomalyPoints.length} points
              </span>
            )}
            <ChevronDown className={clsx("h-4 w-4 text-gray-400 transition-transform", showPoints && "rotate-180")} />
          </div>
        </button>
        
        {showPoints && (
          <div className="mt-3 space-y-2 animate-in slide-in-from-top-2 duration-200">
            {anomalyPoints.length > 0 ? (
              <div className="max-h-48 overflow-y-auto space-y-2 pr-1">
                {anomalyPoints.map((pt, idx) => (
                  <div 
                    key={idx} 
                    className={clsx("p-3 rounded-lg border-l-2 text-xs", layerColor.bg, layerColor.border)}
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <span className={clsx("w-5 h-5 rounded-full flex items-center justify-center font-bold text-[10px]", layerColor.accent, layerColor.text)}>
                          {idx + 1}
                        </span>
                        <span className="font-mono text-gray-700 dark:text-gray-200">
                          {new Date(pt.timestamp * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
                        </span>
                      </div>
                      <div className="flex items-center gap-1">
                        <span className="text-gray-500">Score:</span>
                        <span className={clsx("font-mono font-bold", layerColor.text)}>{pt.point_score.toFixed(4)}</span>
                      </div>
                    </div>
                    <div className="flex items-center justify-between mt-1.5 pl-7">
                      <span className="text-gray-500 font-mono text-[10px]">
                        {pt.lat.toFixed(4)}°, {pt.lon.toFixed(4)}°
                      </span>
                      <span className="text-gray-400 text-[10px]">
                        reconstruction error
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-xs text-gray-400 italic text-center py-3 bg-gray-100 dark:bg-white/5 rounded">
                {isAnomaly ? "No specific points identified" : "No anomalies detected"}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};
