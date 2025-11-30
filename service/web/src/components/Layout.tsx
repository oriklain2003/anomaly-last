import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Settings, HelpCircle, Plane } from 'lucide-react';
import clsx from 'clsx';

export const Layout: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const location = useLocation();

  const tabs = [
    { name: 'Analyze', path: '/analyze' },
    { name: 'Search', path: '/search' },
    { name: 'Upload', path: '/upload' },
  ];

  // Handle root path active state
  const currentPath = location.pathname === '/' ? '/analyze' : location.pathname;

  return (
    <div className="relative flex h-auto min-h-screen w-full flex-col group/design-root overflow-x-hidden bg-background-light dark:bg-background-dark font-display text-gray-900 dark:text-white">
      <div className="layout-container flex h-full grow flex-col">
        
        {/* Header */}
        <header className="flex items-center justify-between whitespace-nowrap border-b border-solid border-gray-200/10 dark:border-white/10 px-4 sm:px-8 md:px-16 lg:px-24 py-4">
          <div className="flex items-center gap-4">
            <div className="text-primary">
              <Plane className="h-8 w-8" />
            </div>
            <h2 className="text-lg font-bold leading-tight tracking-[-0.015em]">Multi-Layer Flight Anomaly Detection</h2>
          </div>
          <div className="flex flex-1 justify-end items-center gap-4">
            <button className="flex items-center justify-center rounded-lg h-10 w-10 bg-gray-200/50 dark:bg-white/10 hover:bg-gray-300/50 dark:hover:bg-white/20 transition-colors">
              <Settings className="h-5 w-5" />
            </button>
            <button className="flex items-center justify-center rounded-lg h-10 w-10 bg-gray-200/50 dark:bg-white/10 hover:bg-gray-300/50 dark:hover:bg-white/20 transition-colors">
              <HelpCircle className="h-5 w-5" />
            </button>
            <div className="bg-center bg-no-repeat aspect-square bg-cover rounded-full size-10 bg-gray-300"></div>
          </div>
        </header>

        <main className="flex-1 px-4 sm:px-8 md:px-16 lg:px-24 py-8">
          <div className="flex flex-col gap-8 max-w-7xl mx-auto">
            
            <p className="text-4xl font-black leading-tight tracking-[-0.033em]">Flight Anomaly Detection System</p>

            {/* Navigation Tabs */}
            <div className="flex gap-4 border-b border-gray-200 dark:border-gray-700 mb-4">
              {tabs.map((tab) => (
                <Link
                  key={tab.path}
                  to={tab.path}
                  className={clsx(
                    "px-4 py-2 border-b-2 font-bold transition-colors",
                    currentPath === tab.path
                      ? "border-primary text-primary"
                      : "border-transparent text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300"
                  )}
                >
                  {tab.name}
                </Link>
              ))}
            </div>

            {children}
          </div>
        </main>
      </div>
    </div>
  );
};

