import type { InputHTMLAttributes } from 'react';
import { Label } from './Label';

const inputBase = { background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(139,92,246,0.18)' };

export function Input({ label, className = '', ...props }: { label?: string } & InputHTMLAttributes<HTMLInputElement>) {
  return (
    <div className="flex flex-col">
      {label && <Label text={label} />}
      <input
        className={`rounded-xl px-4 py-3 text-sm text-slate-200 placeholder-slate-600 outline-none transition-colors focus:border-violet-500/60 focus:ring-2 focus:ring-violet-500/20 ${className}`}
        style={inputBase}
        {...props}
      />
    </div>
  );
}
