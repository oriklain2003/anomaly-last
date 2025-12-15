import { useState, useEffect, useCallback } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import { Settings, Bell, Home, Calendar, RefreshCw } from 'lucide-react';
import { OverviewTab } from './components/intelligence/OverviewTab';
import { SafetyTab } from './components/intelligence/SafetyTab';
import { TrafficTab } from './components/intelligence/TrafficTab';
import { IntelligenceTab } from './components/intelligence/IntelligenceTab';
import { PredictTab } from './components/intelligence/PredictTab';

type TabType = 'overview' | 'safety' | 'traffic' | 'intelligence' | 'predict';

// Parse URL params for initial state
function getInitialState() {
  const params = new URLSearchParams(window.location.search);
  
  // Parse tab
  const tabParam = params.get('tab');
  const validTabs = ['overview', 'safety', 'traffic', 'intelligence', 'predict'];
  const initialTab = validTabs.includes(tabParam || '') ? (tabParam as TabType) : 'overview';
  
  // Parse dates
  const startParam = params.get('start');
  const endParam = params.get('end');
  
  let startTs = Math.floor((Date.now() - 30 * 24 * 60 * 60 * 1000) / 1000); // Default: 30 days ago
  let endTs = Math.floor(Date.now() / 1000); // Default: now
  
  if (startParam && !isNaN(Number(startParam))) {
    startTs = Number(startParam);
  }
  if (endParam && !isNaN(Number(endParam))) {
    endTs = Number(endParam);
  }
  
  return { initialTab, startTs, endTs };
}

export function IntelligencePage() {
  const [, setSearchParams] = useSearchParams();
  const { initialTab, startTs: initialStartTs, endTs: initialEndTs } = getInitialState();
  
  const [activeTab, setActiveTab] = useState<TabType>(initialTab);
  const [dateRange, setDateRange] = useState({
    startTs: initialStartTs,
    endTs: initialEndTs
  });
  const [showCustomDate, setShowCustomDate] = useState(false);
  const [customStartDate, setCustomStartDate] = useState('');
  const [customEndDate, setCustomEndDate] = useState('');
  const [cacheKey, setCacheKey] = useState(0); // Used to force refresh data
  const [lastRefresh, setLastRefresh] = useState<Date | null>(null);
  
  // Update URL when tab or date range changes
  useEffect(() => {
    const newParams = new URLSearchParams();
    newParams.set('tab', activeTab);
    newParams.set('start', dateRange.startTs.toString());
    newParams.set('end', dateRange.endTs.toString());
    setSearchParams(newParams, { replace: true });
  }, [activeTab, dateRange, setSearchParams]);
  
  // Force refresh handler
  const handleForceRefresh = useCallback(() => {
    setCacheKey(prev => prev + 1);
    setLastRefresh(new Date());
  }, []);

  const tabs: { id: TabType; label: string }[] = [
    { id: 'overview', label: 'Overview' },
    { id: 'safety', label: 'Safety' },
    { id: 'traffic', label: 'Traffic' },
    { id: 'intelligence', label: 'Intelligence' },
    { id: 'predict', label: 'Predict' }
  ];

  const handleDateRangeChange = (value: string) => {
    if (value === 'custom') {
      setShowCustomDate(true);
      return;
    }
    
    const days = Number(value);
    const endTs = Math.floor(Date.now() / 1000);
    const startTs = Math.floor((Date.now() - days * 24 * 60 * 60 * 1000) / 1000);
    setDateRange({ startTs, endTs });
    setShowCustomDate(false);
    handleForceRefresh(); // Refresh data when date changes
  };

  const applyCustomDateRange = () => {
    if (customStartDate && customEndDate) {
      const startTs = Math.floor(new Date(customStartDate).getTime() / 1000);
      const endTs = Math.floor(new Date(customEndDate + 'T23:59:59').getTime() / 1000);
      setDateRange({ startTs, endTs });
      setShowCustomDate(false);
      handleForceRefresh(); // Refresh data when date changes
    }
  };
  
  // Calculate which preset matches current range
  const getSelectedPreset = () => {
    const now = Math.floor(Date.now() / 1000);
    const diffDays = Math.round((now - dateRange.startTs) / 86400);
    if (Math.abs(dateRange.endTs - now) < 3600) { // Within 1 hour of now
      if (diffDays <= 1) return '1';
      if (diffDays <= 7) return '7';
      if (diffDays <= 30) return '30';
      if (diffDays <= 90) return '90';
      if (diffDays <= 180) return '180';
      if (diffDays <= 365) return '365';
    }
    return 'custom';
  };

  return (
    <div className="flex h-screen w-full flex-col bg-background-light dark:bg-background-dark text-white overflow-hidden">
      {/* Header */}
      <header className="flex items-center justify-between whitespace-nowrap border-b border-solid border-white/10 px-6 py-3 shrink-0 bg-surface">
        <div className="flex items-center gap-4 text-white">
          <div className="size-6 text-primary">
            <svg fill="none" viewBox="0 0 48 48" xmlns="http://www.w3.org/2000/svg">
              <path d="M13.8261 17.4264C16.7203 18.1174 20.2244 18.5217 24 18.5217C27.7756 18.5217 31.2797 18.1174 34.1739 17.4264C36.9144 16.7722 39.9967 15.2331 41.3563 14.1648L24.8486 40.6391C24.4571 41.267 23.5429 41.267 23.1514 40.6391L6.64374 14.1648C8.00331 15.2331 11.0856 16.7722 13.8261 17.4264Z" fill="currentColor"></path>
              <path clipRule="evenodd" d="M39.998 12.236C39.9944 12.2537 39.9875 12.2845 39.9748 12.3294C39.9436 12.4399 39.8949 12.5741 39.8346 12.7175C39.8168 12.7597 39.7989 12.8007 39.7813 12.8398C38.5103 13.7113 35.9788 14.9393 33.7095 15.4811C30.9875 16.131 27.6413 16.5217 24 16.5217C20.3587 16.5217 17.0125 16.131 14.2905 15.4811C12.0012 14.9346 9.44505 13.6897 8.18538 12.8168C8.17384 12.7925 8.16216 12.767 8.15052 12.7408C8.09919 12.6249 8.05721 12.5114 8.02977 12.411C8.00356 12.3152 8.00039 12.2667 8.00004 12.2612C8.00004 12.261 8 12.2607 8.00004 12.2612C8.00004 12.2359 8.0104 11.9233 8.68485 11.3686C9.34546 10.8254 10.4222 10.2469 11.9291 9.72276C14.9242 8.68098 19.1919 8 24 8C28.8081 8 33.0758 8.68098 36.0709 9.72276C37.5778 10.2469 38.6545 10.8254 39.3151 11.3686C39.9006 11.8501 39.9857 12.1489 39.998 12.236ZM4.95178 15.2312L21.4543 41.6973C22.6288 43.5809 25.3712 43.5809 26.5457 41.6973L43.0534 15.223C43.0709 15.1948 43.0878 15.1662 43.104 15.1371L41.3563 14.1648C43.104 15.1371 43.1038 15.1374 43.104 15.1371L43.1051 15.135L43.1065 15.1325L43.1101 15.1261L43.1199 15.1082C43.1276 15.094 43.1377 15.0754 43.1497 15.0527C43.1738 15.0075 43.2062 14.9455 43.244 14.8701C43.319 14.7208 43.4196 14.511 43.5217 14.2683C43.6901 13.8679 44 13.0689 44 12.2609C44 10.5573 43.003 9.22254 41.8558 8.2791C40.6947 7.32427 39.1354 6.55361 37.385 5.94477C33.8654 4.72057 29.133 4 24 4C18.867 4 14.1346 4.72057 10.615 5.94478C8.86463 6.55361 7.30529 7.32428 6.14419 8.27911C4.99695 9.22255 3.99999 10.5573 3.99999 12.2609C3.99999 13.1275 4.29264 13.9078 4.49321 14.3607C4.60375 14.6102 4.71348 14.8196 4.79687 14.9689C4.83898 15.0444 4.87547 15.1065 4.9035 15.1529C4.91754 15.1762 4.92954 15.1957 4.93916 15.2111L4.94662 15.223L4.95178 15.2312ZM35.9868 18.996L24 38.22L12.0131 18.996C12.4661 19.1391 12.9179 19.2658 13.3617 19.3718C16.4281 20.1039 20.0901 20.5217 24 20.5217C27.9099 20.5217 31.5719 20.1039 34.6383 19.3718C35.082 19.2658 35.5339 19.1391 35.9868 18.996Z" fill="currentColor" fillRule="evenodd"></path>
            </svg>
          </div>
          <h2 className="text-white text-xl font-bold leading-tight tracking-[-0.015em]">Onyx Intelligence Dashboard</h2>
        </div>
        <div className="flex flex-1 justify-end gap-2">
          <Link
            to="/"
            className="flex h-10 px-3 items-center justify-center rounded-lg bg-surface-highlight text-white/80 hover:text-white transition-colors border border-white/10 text-sm font-bold no-underline"
          >
            <Home className="h-4 w-4 mr-2" />
            Main
          </Link>
          <Link
            to="/explorer"
            className="flex h-10 px-3 items-center justify-center rounded-lg bg-surface-highlight text-white/80 hover:text-white transition-colors border border-white/10 text-sm font-bold no-underline"
          >
            Explorer
          </Link>
          <button className="flex h-10 w-10 items-center justify-center rounded-lg bg-surface-highlight text-white/80 hover:text-white transition-colors border border-white/10">
            <Settings className="h-5 w-5" />
          </button>
          <button className="flex h-10 w-10 items-center justify-center rounded-lg bg-surface-highlight text-white/80 hover:text-white transition-colors border border-white/10">
            <Bell className="h-5 w-5" />
          </button>
          <div className="ml-2 size-10 rounded-full bg-gray-600" />
        </div>
      </header>

      {/* Tab Navigation */}
      <div className="border-b border-white/10 bg-surface px-6">
        <div className="flex items-center justify-between">
          <div className="flex gap-2">
            {tabs.map(tab => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`px-4 py-3 text-sm font-medium transition-colors border-b-2 ${
                  activeTab === tab.id
                    ? 'border-primary text-white'
                    : 'border-transparent text-white/60 hover:text-white'
                }`}
              >
                {tab.label}
              </button>
            ))}
          </div>

          {/* Date Range Selector with Display */}
          <div className="flex gap-3 items-center relative">
            {/* Refresh Button */}
            <button
              onClick={handleForceRefresh}
              className="flex h-10 items-center justify-center gap-2 rounded-lg bg-primary/20 text-primary hover:bg-primary/30 transition-colors border border-primary/30 px-3"
              title="Refresh data (bypass cache)"
            >
              <RefreshCw className="h-4 w-4" />
              <span className="text-sm font-medium">Refresh</span>
            </button>
            
            {/* Last Refresh Time */}
            {lastRefresh && (
              <span className="text-xs text-white/40">
                Updated {lastRefresh.toLocaleTimeString()}
              </span>
            )}
            
            {/* Current Date Range Display */}
            <div className="text-sm text-white/70 bg-surface-highlight px-3 py-2 rounded-lg border border-white/10">
              <span className="text-white/50">Showing: </span>
              <span className="text-white font-medium">
                {new Date(dateRange.startTs * 1000).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })}
              </span>
              <span className="text-white/50 mx-1">–</span>
              <span className="text-white font-medium">
                {new Date(dateRange.endTs * 1000).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })}
              </span>
            </div>
            
            {!showCustomDate ? (
              <>
                <select
                  onChange={(e) => handleDateRangeChange(e.target.value)}
                  value={getSelectedPreset()}
                  className="bg-surface-highlight text-white text-sm border border-white/10 rounded-lg px-3 py-2 focus:outline-none focus:border-primary"
                >
                  <option value="1">Last 24 Hours</option>
                  <option value="7">Last 7 Days</option>
                  <option value="30">Last 30 Days</option>
                  <option value="90">Last 90 Days</option>
                  <option value="180">Last 6 Months</option>
                  <option value="365">Last Year</option>
                  <option value="custom">Custom Range...</option>
                </select>
                <button
                  onClick={() => setShowCustomDate(true)}
                  className="flex h-10 items-center justify-center rounded-lg bg-surface-highlight text-white/80 hover:text-white transition-colors border border-white/10 px-3"
                  title="Custom date range"
                >
                  <Calendar className="h-4 w-4" />
                </button>
              </>
            ) : (
              <div className="flex gap-2 items-center bg-surface-highlight border border-white/10 rounded-lg p-2">
                <input
                  type="date"
                  value={customStartDate}
                  onChange={(e) => setCustomStartDate(e.target.value)}
                  className="bg-background-dark text-white text-sm border border-white/10 rounded px-2 py-1 focus:outline-none focus:border-primary"
                  placeholder="Start"
                />
                <span className="text-white/60">to</span>
                <input
                  type="date"
                  value={customEndDate}
                  onChange={(e) => setCustomEndDate(e.target.value)}
                  className="bg-background-dark text-white text-sm border border-white/10 rounded px-2 py-1 focus:outline-none focus:border-primary"
                  placeholder="End"
                />
                <button
                  onClick={applyCustomDateRange}
                  className="bg-primary text-white text-sm px-3 py-1 rounded hover:bg-primary/80 transition-colors font-medium"
                >
                  Apply
                </button>
                <button
                  onClick={() => setShowCustomDate(false)}
                  className="text-white/60 hover:text-white text-sm px-2"
                >
                  Cancel
                </button>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Tab Content */}
      <main className="flex-1 overflow-auto p-6">
        <div className="max-w-[1600px] mx-auto">
          {activeTab === 'overview' && <OverviewTab startTs={dateRange.startTs} endTs={dateRange.endTs} cacheKey={cacheKey} />}
          {activeTab === 'safety' && <SafetyTab startTs={dateRange.startTs} endTs={dateRange.endTs} cacheKey={cacheKey} />}
          {activeTab === 'traffic' && <TrafficTab startTs={dateRange.startTs} endTs={dateRange.endTs} cacheKey={cacheKey} />}
          {activeTab === 'intelligence' && <IntelligenceTab startTs={dateRange.startTs} endTs={dateRange.endTs} cacheKey={cacheKey} />}
          {activeTab === 'predict' && <PredictTab />}
        </div>
      </main>
    </div>
  );
}

