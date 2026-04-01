import type React from 'react';
import type { ReactNode } from 'react';

const surface = { background: '#111827', border: '1px solid rgba(139,92,246,0.15)' };

export function Card({
  children, className = '', style,
}: { children: ReactNode; className?: string; style?: React.CSSProperties }) {
  return (
    <div className={`rounded-2xl p-6 ${className}`} style={{ ...surface, ...style }}>
      {children}
    </div>
  );
}
