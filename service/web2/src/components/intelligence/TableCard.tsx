import { ReactNode, useState } from 'react';
import { ChevronUp, ChevronDown } from 'lucide-react';

export interface Column {
  key: string;
  title: string;
  render?: (value: any, row: any) => ReactNode;
  sortable?: boolean;
}

interface TableCardProps {
  title: string;
  columns: Column[];
  data: any[];
  sortable?: boolean;
  filterable?: boolean;
  className?: string;
}

export function TableCard({ title, columns, data, sortable = true, className = '' }: TableCardProps) {
  const [sortKey, setSortKey] = useState<string | null>(null);
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('asc');

  const handleSort = (columnKey: string) => {
    if (!sortable) return;
    
    if (sortKey === columnKey) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortKey(columnKey);
      setSortDirection('asc');
    }
  };

  const sortedData = sortKey
    ? [...data].sort((a, b) => {
        const aVal = a[sortKey];
        const bVal = b[sortKey];
        
        if (typeof aVal === 'number' && typeof bVal === 'number') {
          return sortDirection === 'asc' ? aVal - bVal : bVal - aVal;
        }
        
        const aStr = String(aVal || '');
        const bStr = String(bVal || '');
        return sortDirection === 'asc'
          ? aStr.localeCompare(bStr)
          : bStr.localeCompare(aStr);
      })
    : data;

  return (
    <div className={`bg-surface rounded-xl p-6 border border-white/10 ${className}`}>
      <h3 className="text-white text-lg font-bold mb-4">{title}</h3>
      
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead>
            <tr className="border-b border-white/10">
              {columns.map((col) => (
                <th
                  key={col.key}
                  className={`text-left text-white/60 text-sm font-medium py-3 px-4 ${
                    sortable && col.sortable !== false ? 'cursor-pointer hover:text-white' : ''
                  }`}
                  onClick={() => sortable && col.sortable !== false && handleSort(col.key)}
                >
                  <div className="flex items-center gap-2">
                    {col.title}
                    {sortable && col.sortable !== false && sortKey === col.key && (
                      sortDirection === 'asc' ? (
                        <ChevronUp className="w-4 h-4" />
                      ) : (
                        <ChevronDown className="w-4 h-4" />
                      )
                    )}
                  </div>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {sortedData.map((row, idx) => (
              <tr key={idx} className="border-b border-white/5 hover:bg-white/5">
                {columns.map((col) => (
                  <td key={col.key} className="text-white text-sm py-3 px-4">
                    {col.render ? col.render(row[col.key], row) : row[col.key]}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
        
        {sortedData.length === 0 && (
          <div className="text-center text-white/40 py-8">No data available</div>
        )}
      </div>
    </div>
  );
}

