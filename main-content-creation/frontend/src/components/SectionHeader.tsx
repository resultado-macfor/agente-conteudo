import type { ReactNode } from 'react';

export function SectionHeader({ icon, title, subtitle }: {
  icon: ReactNode;
  title: string;
  subtitle?: string;
}) {
  return (
    <div className="flex items-center gap-3 mb-6">
      <div
        className="w-10 h-10 rounded-xl flex items-center justify-center shrink-0 text-violet-300"
        style={{ background: 'rgba(124,58,237,0.15)', border: '1px solid rgba(139,92,246,0.25)' }}
      >
        {icon}
      </div>
      <div>
        <h2 className="text-lg font-semibold text-slate-100 leading-tight">{title}</h2>
        {subtitle && <p className="text-xs text-slate-500 mt-0.5">{subtitle}</p>}
      </div>
    </div>
  );
}
