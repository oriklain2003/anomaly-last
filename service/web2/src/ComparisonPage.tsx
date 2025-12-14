import React, { useEffect, useState } from 'react';

interface GlitchAnalysis {
  glitch_score: number;
  spike_ratio: number;
  freeze_ratio: number;
  metrics: any;
}

interface LLMAnalysis {
  explanation: string;
  logical_judgment: string;
  logical_anomaly_score: number;
  is_anomaly: boolean;
  reasoning: string;
}

interface TestResult {
  flight_id: string;
  original_anomaly: boolean;
  base_pipeline_anomaly: boolean;
  new_pipeline_anomaly: boolean;
  glitch_score: number;
  llm_explanation: string;
  llm_judgment: string;
  full_report: {
    glitch_analysis?: GlitchAnalysis;
    llm_analysis?: LLMAnalysis;
    [key: string]: any;
  };
}

export const ComparisonPage: React.FC = () => {
  const [results, setResults] = useState<TestResult[]>([]);
  const [loading, setLoading] = useState(true);
  const [showAll, setShowAll] = useState(true);

  useEffect(() => {
    fetch('/test_results.json')
      .then(res => res.json())
      .then(data => {
        setResults(data);
        setLoading(false);
      })
      .catch(err => {
        console.error("Failed to load results", err);
        setLoading(false);
      });
  }, []);

  if (loading) return <div className="p-8 text-white">Loading test results...</div>;

  // Filter: anomalies that the new pipeline said is normal 
  const reclassified = results.filter(r => 
    r.original_anomaly && (!r.new_pipeline_anomaly || r.llm_judgment === "normal but noisy")
  );

  const displayed = showAll ? results : reclassified;

  return (
    <div className="min-h-screen bg-slate-900 text-slate-100 p-8">
      <div className="flex justify-between items-center mb-6">
        <div>
            <h1 className="text-3xl font-bold">New Service Pipeline Test Results</h1>
            <p className="text-slate-400 mt-2">
                Processed {results.length} flights. {reclassified.length} reclassified as Normal.
            </p>
        </div>
        <div className="flex space-x-2 bg-slate-800 p-1 rounded-lg">
            <button 
                onClick={() => setShowAll(true)}
                className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${showAll ? 'bg-blue-600 text-white' : 'text-slate-400 hover:text-white'}`}
            >
                Show All ({results.length})
            </button>
            <button 
                onClick={() => setShowAll(false)}
                className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${!showAll ? 'bg-blue-600 text-white' : 'text-slate-400 hover:text-white'}`}
            >
                Show Reclassified Only ({reclassified.length})
            </button>
        </div>
      </div>

      <div className="space-y-6">
        {displayed.map(r => {
            const isReclassified = (!r.new_pipeline_anomaly || r.llm_judgment === "normal but noisy");
            return (
          <div key={r.flight_id} className={`rounded-lg p-6 border shadow-lg ${isReclassified ? 'bg-slate-800 border-green-900/50' : 'bg-slate-800/50 border-slate-700'}`}>
            <div className="flex justify-between items-start mb-4">
              <div>
                <div className="flex items-center gap-3">
                    <h2 className="text-xl font-semibold text-blue-400">{r.flight_id}</h2>
                    {isReclassified && <span className="px-2 py-0.5 bg-green-900 text-green-200 text-xs rounded border border-green-700">RECLASSIFIED</span>}
                </div>
                <div className="text-sm text-slate-400 mt-1">
                  Glitch Score: <span className={`font-mono ${r.glitch_score > 0.1 ? 'text-red-400' : 'text-green-400'}`}>{r.glitch_score}</span>
                </div>
              </div>
              <div className="text-right">
                <span className="px-3 py-1 rounded-full text-xs font-bold bg-red-900 text-red-200 mr-2">
                  Old: Anomaly
                </span>
                <span className={`px-3 py-1 rounded-full text-xs font-bold ${isReclassified ? 'bg-green-900 text-green-200' : 'bg-red-900 text-red-200'}`}>
                  New: {r.llm_judgment}
                </span>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Glitch Analysis */}
              <div className="bg-slate-900/50 p-4 rounded">
                <h3 className="text-sm font-uppercase text-slate-500 mb-2">Glitch Analysis</h3>
                <pre className="text-xs font-mono text-slate-300 overflow-auto max-h-40">
                  {JSON.stringify(r.full_report.glitch_analysis, null, 2)}
                </pre>
              </div>

              {/* LLM Analysis */}
              <div className="bg-slate-900/50 p-4 rounded border border-blue-900/30">
                <h3 className="text-sm font-uppercase text-blue-400 mb-2">LLM Assessment</h3>
                <div className="mb-2">
                  <span className="text-xs text-slate-500">Judgment:</span>
                  <div className="font-medium text-white">{r.llm_explanation}</div>
                </div>
                {r.full_report.llm_analysis?.reasoning && (
                  <div className="mt-2">
                    <span className="text-xs text-slate-500">Reasoning:</span>
                    <div className="text-sm text-slate-300 italic">"{r.full_report.llm_analysis.reasoning}"</div>
                  </div>
                )}
              </div>
            </div>
          </div>
        )})}

        {displayed.length === 0 && (
          <div className="text-center text-slate-500 py-12">
            No flights found matching criteria.
          </div>
        )}
      </div>
    </div>
  );
};
