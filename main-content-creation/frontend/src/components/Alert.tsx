import type { ReactNode } from 'react';

const map = {
  error:   'bg-red-500/10   border-red-500/25   text-red-300',
  success: 'bg-emerald-500/10 border-emerald-500/25 text-emerald-300',
  info:    'bg-violet-500/10  border-violet-500/25 text-violet-300',
};

export function Alert({ type, children }: { type: 'error' | 'success' | 'info'; children: ReactNode }) {
  return (
    <div className={`rounded-xl border px-4 py-3 text-sm ${map[type]}`}>
      {children}
    </div>
  );
}
