export function Stat({ label, value }: { label: string; value: string | number }) {
  return (
    <div
      className="rounded-xl p-4 text-center"
      style={{ background: 'rgba(0,0,0,0.25)', border: '1px solid rgba(139,92,246,0.15)' }}
    >
      <p className="text-xl font-bold text-violet-300">{value}</p>
      <p className="text-xs text-slate-500 mt-1">{label}</p>
    </div>
  );
}
