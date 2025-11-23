import React, { useState, useEffect } from 'react';
import { ResultsView } from '../components/ResultsView';
import { ChatInterface } from '../components/ChatInterface';
import type { AnalysisResult } from '../types';
import { Search } from 'lucide-react';

export const SearchPage: React.FC = () => {
  const [callsign, setCallsign] = useState('CYF461');
  const [fromDate, setFromDate] = useState('');
  const [toDate, setToDate] = useState('');
  
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [searchedId, setSearchedId] = useState('');

  useEffect(() => {
    // Initialize dates
    const now = new Date();
    const fifteenMinsAgo = new Date(now.getTime() - 15 * 60 * 1000);
    const oneMinAgo = new Date(now.getTime() - 1 * 60 * 1000);

    const toLocalISO = (date: Date) => {
      const offset = date.getTimezoneOffset() * 60000;
      const local = new Date(date.getTime() - offset);
      return local.toISOString().slice(0, 16);
    };

    setFromDate(toLocalISO(fifteenMinsAgo));
    setToDate(toLocalISO(oneMinAgo));
  }, []);

  const handleSearch = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!callsign || !fromDate || !toDate) return;

    setLoading(true);
    setError(null);
    setResult(null);
    setSearchedId(`${callsign} (Searching...)`);

    try {
      const fromDt = new Date(fromDate).toISOString();
      const toDt = new Date(toDate).toISOString();

      const response = await fetch('/api/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          callsign,
          from_date: fromDt,
          to_date: toDt,
        }),
      });

      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.detail || "Search failed");
      }

      const data: AnalysisResult = await response.json();
      setResult(data);
      setSearchedId(`${data.summary.flight_id} (${callsign})`);
      
    } catch (err: any) {
      setError(err.message);
      setSearchedId(callsign);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col gap-8">
      {/* Input Section */}
      <div className="flex flex-col gap-6">
        <form onSubmit={handleSearch} className="flex flex-col gap-4 p-4 rounded-xl bg-gray-50 dark:bg-white/5 border border-gray-200/50 dark:border-white/10">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <label className="flex flex-col gap-2">
              <span className="text-gray-600 dark:text-gray-300 text-base font-medium leading-normal">Call Sign</span>
              <input
                type="text"
                value={callsign}
                onChange={(e) => setCallsign(e.target.value)}
                className="flex w-full rounded-lg text-gray-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-primary border border-gray-300 dark:border-white/20 bg-white dark:bg-background-dark h-12 px-4 text-base"
                placeholder="e.g., CYF461"
              />
            </label>
            <label className="flex flex-col gap-2">
              <span className="text-gray-600 dark:text-gray-300 text-base font-medium leading-normal">From Date</span>
              <input
                type="datetime-local"
                value={fromDate}
                onChange={(e) => setFromDate(e.target.value)}
                className="flex w-full rounded-lg text-gray-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-primary border border-gray-300 dark:border-white/20 bg-white dark:bg-background-dark h-12 px-4 text-base"
              />
            </label>
            <label className="flex flex-col gap-2">
              <span className="text-gray-600 dark:text-gray-300 text-base font-medium leading-normal">To Date</span>
              <input
                type="datetime-local"
                value={toDate}
                onChange={(e) => setToDate(e.target.value)}
                className="flex w-full rounded-lg text-gray-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-primary border border-gray-300 dark:border-white/20 bg-white dark:bg-background-dark h-12 px-4 text-base"
              />
            </label>
          </div>
          
          <div className="flex justify-end">
            <button
              type="submit"
              disabled={loading}
              className="flex w-full sm:w-auto min-w-[84px] cursor-pointer items-center justify-center gap-2 rounded-lg h-12 px-8 bg-primary text-white text-base font-bold hover:bg-blue-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <Search className="h-5 w-5" />
              <span className="truncate">{loading ? 'Searching...' : 'Search & Analyze'}</span>
            </button>
          </div>
        </form>
      </div>

      {/* Results Section */}
      <ResultsView 
        data={result} 
        flightId={searchedId} 
        isLoading={loading} 
        error={error} 
      />

      {/* Chat Interface - Only show when data is available */}
      {result && <ChatInterface data={result} flightId={searchedId} />}
    </div>
  );
};
