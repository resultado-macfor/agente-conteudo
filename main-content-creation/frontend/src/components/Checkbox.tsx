export function Checkbox({ label, checked, onChange }: {
  label: string;
  checked: boolean;
  onChange: (v: boolean) => void;
}) {
  return (
    <label className="flex items-center gap-3 cursor-pointer text-sm text-slate-400 hover:text-slate-200 transition-colors select-none py-0.5">
      <div
        onClick={() => onChange(!checked)}
        className="w-5 h-5 rounded-md flex items-center justify-center shrink-0 transition-all"
        style={{
          background: checked ? 'linear-gradient(135deg,#4c1d95,#7c3aed)' : 'rgba(255,255,255,0.04)',
          border: checked ? 'none' : '1px solid rgba(139,92,246,0.35)',
        }}
      >
        {checked && (
          <svg width="11" height="11" viewBox="0 0 11 11" fill="none">
            <path d="M1.5 5.5L4.5 8.5L9.5 2.5" stroke="white" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        )}
      </div>
      {label}
    </label>
  );
}
