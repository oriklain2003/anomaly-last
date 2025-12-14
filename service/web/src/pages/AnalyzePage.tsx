import React, { useState } from 'react';
import { ResultsView } from '../components/ResultsView';
import { ChatInterface } from '../components/ChatInterface';
import type { AnalysisResult } from '../types';
import { Search, UploadCloud } from 'lucide-react';

export const AnalyzePage: React.FC = () => {
  const [flightId, setFlightId] = useState('3bc6854c');
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [uploadLoading, setUploadLoading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState<string | null>(null);

  const handleAnalyze = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!flightId) return;

    setLoading(true);
    setError(null);
    setResult(null);
    setUploadStatus(null);

    try {
      const API_BASE = import.meta.env.VITE_API_URL || '';
      const response = await fetch(`${API_BASE}/api/analyze/${flightId}`);
      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.detail || "Analysis failed");
      }
      const data: AnalysisResult = await response.json();
      setResult(data);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleUpload = async () => {
      if (!result || !result.summary.flight_id) return;
      
      setUploadLoading(true);
      setUploadStatus(null);
      
      try {
          const API_BASE = import.meta.env.VITE_API_URL || '';
          const feedbackResponse = await fetch(`${API_BASE}/api/feedback`, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({
              flight_id: result.summary.flight_id,
              is_anomaly: true,
              comments: "Manual upload via Analyze Page"
            }),
          });
          
          if (!feedbackResponse.ok) {
             throw new Error("Failed to save feedback");
          }
          setUploadStatus('Successfully saved to anomalies database.');
      } catch (e) {
          setUploadStatus('Failed to save to anomalies database.');
      } finally {
          setUploadLoading(false);
      }
  };

  return (
    <div className="flex flex-col gap-8">
      {/* Input Section */}
      <div className="flex flex-col gap-6">
        <form onSubmit={handleAnalyze} className="flex flex-col sm:flex-row items-end gap-4 p-4 rounded-xl bg-gray-50 dark:bg-white/5 border border-gray-200/50 dark:border-white/10">
          <label className="flex flex-col min-w-40 flex-1 gap-2">
            <span className="text-gray-600 dark:text-gray-300 text-base font-medium leading-normal">Flight ID</span>
            <input
              type="text"
              value={flightId}
              onChange={(e) => setFlightId(e.target.value)}
              className="flex w-full min-w-0 flex-1 resize-none overflow-hidden rounded-lg text-gray-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-primary border border-gray-300 dark:border-white/20 bg-white dark:bg-background-dark h-12 px-4 text-base"
              placeholder="Enter Flight ID (e.g., 3bc6854c)"
            />
          </label>
          <button
            type="submit"
            disabled={loading}
            className="flex w-full sm:w-auto min-w-[84px] cursor-pointer items-center justify-center gap-2 rounded-lg h-12 px-6 bg-primary text-white text-base font-bold hover:bg-blue-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Search className="h-5 w-5" />
            <span className="truncate">{loading ? 'Analyzing...' : 'Analyze Flight'}</span>
          </button>
        </form>
      </div>

      {/* Results Section */}
      <ResultsView 
        data={result} 
        flightId={flightId} 
        isLoading={loading} 
        error={error} 
      />

      {/* Upload Button Section */}
      {result && (
          <div className="flex flex-col items-end gap-2 px-4">
               <button 
                  onClick={handleUpload}
                  disabled={uploadLoading}
                  className="flex items-center gap-2 bg-red-500 hover:bg-red-600 text-white px-4 py-2 rounded-lg font-bold transition-colors disabled:opacity-50"
               >
                   <UploadCloud className="h-5 w-5" />
                   {uploadLoading ? 'Uploading...' : 'Upload as Anomaly'}
               </button>
               {uploadStatus && (
                   <span className={uploadStatus.includes('Success') ? 'text-green-500 font-medium' : 'text-red-500 font-medium'}>
                       {uploadStatus}
                   </span>
               )}
          </div>
      )}

      {/* Chat Interface - Only show when data is available */}
      {result && <ChatInterface data={result} flightId={flightId} />}
    </div>
  );
};
