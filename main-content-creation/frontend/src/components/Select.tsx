import type { ReactNode, SelectHTMLAttributes } from 'react';
import { Label } from './Label';

export function Select({ label, children, className = '', ...props }: { label?: string; children: ReactNode } & SelectHTMLAttributes<HTMLSelectElement>) {
  return (
    <div className="flex flex-col">
      {label && <Label text={label} />}
      <select
        className={`rounded-xl px-4 py-3 text-sm text-slate-200 outline-none transition-colors focus:border-violet-500/60 focus:ring-2 focus:ring-violet-500/20 ${className}`}
        style={{ background: '#1a2440', border: '1px solid rgba(139,92,246,0.18)' }}
        {...props}
      >
        {children}
      </select>
    </div>
  );
}
