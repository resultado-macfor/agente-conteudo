export function Spinner({ text = 'Processando...' }: { text?: string }) {
  return (
    <div className="flex items-center gap-3 py-4">
      <svg className="animate-spin h-5 w-5 text-violet-500 shrink-0" fill="none" viewBox="0 0 24 24">
        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z" />
      </svg>
      <span className="text-slate-400 text-sm">{text}</span>
    </div>
  );
}
