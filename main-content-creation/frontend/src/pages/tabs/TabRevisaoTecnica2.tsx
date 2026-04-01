import { useState } from 'react';
import { Search, RotateCcw } from 'lucide-react';
import { useStore } from '../../store/useStore';
import { buildContexto } from '../../utils/buildContexto';
import { contentApi } from '../../api/content';
import { Button, Card, Textarea, SectionHeader, Spinner, ResultBox, Alert } from '../../components/ui';

export default function TabRevisaoTecnica2() {
  const { selectedAgent, segmentos } = useStore();
  const [texto, setTexto] = useState('');
  const [resultado, setResultado] = useState('');
  const [ajuste, setAjuste] = useState('');
  const [loading, setLoading] = useState(false);
  const [loadingAjuste, setLoadingAjuste] = useState(false);
  const [error, setError] = useState('');

  const ctx = selectedAgent ? buildContexto(selectedAgent, segmentos) : '';

  const handleRevisar = async () => {
    if (!texto.trim()) return;
    setError(''); setLoading(true);
    try {
      const res = await contentApi.revisaoTecnicaSemRag({ texto, contextoAgente: ctx });
      setResultado(res.resultado);
    } catch (e: unknown) { setError((e as Error).message); }
    finally { setLoading(false); }
  };

  const handleAjustar = async () => {
    if (!ajuste.trim() || !resultado) return;
    setLoadingAjuste(true);
    try {
      const res = await contentApi.revisaoTecnicaSemRag({ texto: resultado, contextoAgente: ctx, ajuste });
      setResultado(res.resultado);
      setAjuste('');
    } catch (e: unknown) { setError((e as Error).message); }
    finally { setLoadingAjuste(false); }
  };

  return (
    <Card>
      <SectionHeader icon={<Search size={16} />} title="Revisão Técnica Sem RAG" subtitle="Revisão técnica profissional sem base vetorial" />
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="flex flex-col gap-3">
          <Textarea label="Conteúdo Original" value={texto} onChange={(e) => setTexto(e.target.value)} rows={14} placeholder="Cole aqui o conteúdo técnico para revisão…" />
          <Button onClick={handleRevisar} loading={loading} disabled={!texto.trim()} fullWidth>
            <Search size={14} /> Realizar Revisão Técnica
          </Button>
          {loading && <Spinner text="Revisando conteúdo…" />}
          {error && <Alert type="error">{error}</Alert>}
        </div>

        <div className="flex flex-col gap-3">
          <p className="text-xs font-semibold text-slate-400 uppercase tracking-wide">Conteúdo Revisado</p>
          {resultado ? (
            <>
              <ResultBox content={resultado} filename="revisao_tecnica.txt" />
              <div className="border-t pt-3" style={{ borderColor: 'var(--border)' }}>
                <Textarea label="Ajustes Incrementais" value={ajuste} onChange={(e) => setAjuste(e.target.value)} rows={3} placeholder="Solicite ajustes específicos no conteúdo revisado…" />
                <Button onClick={handleAjustar} loading={loadingAjuste} disabled={!ajuste.trim()} fullWidth className="mt-2">
                  <RotateCcw size={14} /> Aplicar Ajustes
                </Button>
              </div>
            </>
          ) : (
            <div className="rounded-xl border p-8 text-center text-slate-600 text-sm flex flex-col items-center gap-2" style={{ borderColor: 'var(--border)', background: 'rgba(0,0,0,0.15)' }}>
              <Search size={24} className="text-slate-700" />
              O conteúdo revisado aparecerá aqui.
            </div>
          )}
        </div>
      </div>
    </Card>
  );
}
