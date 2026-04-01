export function Label({ text }: { text: string }) {
  return (
    <label className="block text-xs font-semibold text-slate-400 uppercase tracking-widest mb-1.5">
      {text}
    </label>
  );
}
