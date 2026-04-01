import { Button } from './Button';

export function ResultBox({ content, filename }: { content: string; filename?: string }) {
  const copy = () => navigator.clipboard.writeText(content);

  const download = () => {
    const a = Object.assign(document.createElement('a'), {
      href: URL.createObjectURL(new Blob([content], { type: 'text/plain' })),
      download: filename ?? 'resultado.txt',
    });
    a.click();
  };

  return (
    <div className="flex flex-col gap-2">
      <div className="flex justify-end gap-2">
        <Button variant="ghost" onClick={copy} className="h-8 px-3 text-xs gap-1.5">
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <rect x="9" y="9" width="13" height="13" rx="2" />
            <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
          </svg>
          Copiar
        </Button>
        <Button variant="secondary" onClick={download} className="h-8 px-3 text-xs gap-1.5">
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
            <polyline points="7 10 12 15 17 10" />
            <line x1="12" y1="15" x2="12" y2="3" />
          </svg>
          Baixar
        </Button>
      </div>
      <div
        className="rounded-xl p-5 text-sm text-slate-300 whitespace-pre-wrap overflow-y-auto leading-relaxed"
        style={{
          background: 'rgba(0,0,0,0.3)',
          border: '1px solid rgba(139,92,246,0.12)',
          maxHeight: 540,
          fontFamily: 'ui-monospace, Consolas, monospace',
          fontSize: 13,
        }}
      >
        {content}
      </div>
    </div>
  );
}
