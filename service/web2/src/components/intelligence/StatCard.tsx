import { ReactNode } from 'react';
import { TrendingUp, TrendingDown } from 'lucide-react';

interface StatCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  icon?: ReactNode;
  trend?: { direction: 'up' | 'down'; value: string };
  onClick?: () => void;
}

export function StatCard({ title, value, subtitle, icon, trend, onClick }: StatCardProps) {
  return (
    <div
      className={`bg-surface rounded-xl p-6 border border-white/10 ${
        onClick ? 'cursor-pointer hover:bg-surface-highlight transition-colors' : ''
      }`}
      onClick={onClick}
    >
      <div className="flex items-start justify-between">
        <div className="flex-1">
          <p className="text-white/60 text-sm font-medium mb-2">{title}</p>
          <p className="text-white text-3xl font-bold">{value}</p>
          {subtitle && <p className="text-white/40 text-xs mt-1">{subtitle}</p>}
        </div>
        {icon && <div className="text-primary opacity-80">{icon}</div>}
      </div>
      
      {trend && (
        <div className="mt-4 flex items-center gap-1 text-sm">
          {trend.direction === 'up' ? (
            <TrendingUp className="w-4 h-4 text-green-500" />
          ) : (
            <TrendingDown className="w-4 h-4 text-red-500" />
          )}
          <span className={trend.direction === 'up' ? 'text-green-500' : 'text-red-500'}>
            {trend.value}
          </span>
          <span className="text-white/40">vs last period</span>
        </div>
      )}
    </div>
  );
}

