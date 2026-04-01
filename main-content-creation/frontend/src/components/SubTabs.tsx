export function SubTabs<T extends string>({
  tabs, active, onChange,
}: {
  tabs: { id: T; label: string }[];
  active: T;
  onChange: (t: T) => void;
}) {
  return (
    <div
      className="flex gap-1 mb-5 p-1 rounded-xl"
      style={{ background: 'rgba(0,0,0,0.2)', border: '1px solid rgba(139,92,246,0.12)' }}
    >
      {tabs.map((t) => (
        <button
          key={t.id}
          onClick={() => onChange(t.id)}
          className="flex-1 px-3 py-2 text-sm font-medium rounded-lg transition-all"
          style={{
            background: active === t.id ? 'linear-gradient(135deg, #4c1d95, #7c3aed)' : 'transparent',
            color: active === t.id ? '#fff' : '#64748b',
          }}
        >
          {t.label}
        </button>
      ))}
    </div>
  );
}
