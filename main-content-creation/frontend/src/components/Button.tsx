import type React from 'react';
import type { ReactNode } from 'react';

type BtnVariant = 'primary' | 'secondary' | 'ghost' | 'danger';

const variantStyles: Record<BtnVariant, { className: string; style?: React.CSSProperties }> = {
  primary:   { className: 'text-white shadow-lg hover:brightness-110 active:scale-[0.97]',            style: { background: 'linear-gradient(135deg, #4c1d95, #7c3aed)' } },
  secondary: { className: 'text-violet-300 hover:bg-violet-500/10 active:scale-[0.97]',               style: { border: '1px solid rgba(139,92,246,0.3)', background: 'rgba(139,92,246,0.07)' } },
  ghost:     { className: 'text-slate-400 hover:text-slate-200 hover:bg-white/5 active:scale-[0.97]', style: undefined },
  danger:    { className: 'text-red-400 hover:bg-red-500/15 active:scale-[0.97]',                     style: { border: '1px solid rgba(239,68,68,0.25)', background: 'rgba(239,68,68,0.08)' } },
};

const base = 'inline-flex items-center justify-center gap-2 rounded-xl px-5 py-2.5 text-sm font-semibold transition-all duration-150 cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed select-none';

export function Button({
  children, onClick, disabled, loading, variant = 'primary',
  className = '', type = 'button', fullWidth,
}: {
  children: ReactNode;
  onClick?: () => void;
  disabled?: boolean;
  loading?: boolean;
  variant?: BtnVariant;
  className?: string;
  type?: 'button' | 'submit';
  fullWidth?: boolean;
}) {
  const v = variantStyles[variant];
  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled || loading}
      className={`${base} ${v.className} ${fullWidth ? 'w-full' : ''} ${className}`}
      style={v.style}
    >
      {loading && (
        <svg className="animate-spin h-4 w-4 shrink-0" fill="none" viewBox="0 0 24 24">
          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z" />
        </svg>
      )}
      {children}
    </button>
  );
}
