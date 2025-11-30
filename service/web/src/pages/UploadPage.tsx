import React, { useState } from 'react';
import { ResultsView } from '../components/ResultsView';
import type { AnalysisResult } from '../types';
import { UploadCloud, AlertTriangle, CheckCircle } from 'lucide-react';

export const UploadPage: React.FC = () => {
  const [flightId, setFlightId] = useState('');
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [uploadStatus, setUploadStatus] = useState<string | null>(null);

  const handleAnalyzeAndUpload = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!flightId) return;

    setLoading(true);
    setError(null);
    setResult(null);
    setUploadStatus(null);

    try {
      // 1. Analyze Flight
      const API_BASE = import.meta.env.VITE_API_URL || '';
      const response = await fetch(`${API_BASE}/api/analyze/${flightId}`);
      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.detail || "Analysis failed");
      }
      const data: AnalysisResult = await response.json();
      setResult(data);

      // 2. If Anomaly, Upload to Feedback API
      if (data.summary.is_anomaly) {
        setUploadStatus('Saving to anomalies database...');
        try {
          const feedbackResponse = await fetch(`${API_BASE}/api/feedback`, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({
              flight_id: flightId,
              is_anomaly: true,
              comments: "Auto-uploaded via Upload Page"
            }),
          });
          
          if (!feedbackResponse.ok) {
             throw new Error("Failed to save feedback");
          }
          setUploadStatus('Successfully saved to anomalies database.');
        } catch (uploadErr) {
          console.error(uploadErr);
          setUploadStatus('Failed to save to anomalies database.');
        }
      } else {
        setUploadStatus('Flight is normal. Not saved to anomalies.');
      }

    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col gap-8">
      <div className="flex flex-col gap-2">
         <h1 className="text-2xl font-bold">Upload Anomaly</h1>
         <p className="text-gray-500">Enter a flight ID to analyze and automatically save it as an anomaly if detected.</p>
      </div>

      {/* Input Section */}
      <div className="flex flex-col gap-6">
        <form onSubmit={handleAnalyzeAndUpload} className="flex flex-col sm:flex-row items-end gap-4 p-4 rounded-xl bg-gray-50 dark:bg-white/5 border border-gray-200/50 dark:border-white/10">
          <label className="flex flex-col min-w-40 flex-1 gap-2">
            <span className="text-gray-600 dark:text-gray-300 text-base font-medium leading-normal">Flight ID</span>
            <input
              type="text"
              value={flightId}
              onChange={(e) => setFlightId(e.target.value)}
              className="flex w-full min-w-0 flex-1 resize-none overflow-hidden rounded-lg text-gray-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-primary border border-gray-300 dark:border-white/20 bg-white dark:bg-background-dark h-12 px-4 text-base"
              placeholder="Enter Flight ID"
            />
          </label>
          <button
            type="submit"
            disabled={loading}
            className="flex w-full sm:w-auto min-w-[84px] cursor-pointer items-center justify-center gap-2 rounded-lg h-12 px-6 bg-primary text-white text-base font-bold hover:bg-blue-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <UploadCloud className="h-5 w-5" />
            <span className="truncate">{loading ? 'Processing...' : 'Load & Upload'}</span>
          </button>
        </form>
      </div>

      {/* Upload Status Message */}
      {uploadStatus && (
        <div className={`p-4 rounded-xl border flex items-center gap-3 ${
            uploadStatus.includes('Success') ? 'bg-green-50 border-green-200 text-green-700 dark:bg-green-900/20 dark:border-green-900/30 dark:text-green-400' :
            uploadStatus.includes('Not saved') ? 'bg-blue-50 border-blue-200 text-blue-700 dark:bg-blue-900/20 dark:border-blue-900/30 dark:text-blue-400' :
            uploadStatus.includes('Saving') ? 'bg-gray-50 border-gray-200 text-gray-700 dark:bg-white/5 dark:border-white/10 dark:text-gray-300' :
            'bg-red-50 border-red-200 text-red-700 dark:bg-red-900/20 dark:border-red-900/30 dark:text-red-400'
        }`}>
            {uploadStatus.includes('Success') ? <CheckCircle className="h-5 w-5" /> : 
             uploadStatus.includes('Not saved') ? <CheckCircle className="h-5 w-5" /> :
             <AlertTriangle className="h-5 w-5" />}
            <span className="font-medium">{uploadStatus}</span>
        </div>
      )}

      {/* Results Section */}
      <ResultsView 
        data={result} 
        flightId={flightId} 
        isLoading={loading} 
        error={error} 
      />
    </div>
  );
};

