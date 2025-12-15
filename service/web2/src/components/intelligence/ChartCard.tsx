import { ReactNode } from 'react';

interface ChartCardProps {
  title: string;
  children: ReactNode;
  actions?: ReactNode;
  className?: string;
}

export function ChartCard({ title, children, actions, className = '' }: ChartCardProps) {
  return (
    <div className={`bg-surface rounded-xl p-6 border border-white/10 ${className}`}>
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-white text-lg font-bold">{title}</h3>
        {actions && <div className="flex items-center gap-2">{actions}</div>}
      </div>
      <div className="w-full">{children}</div>
    </div>
  );
}

