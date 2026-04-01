import { useState } from 'react';
import { ClipboardList, Sparkles, RotateCcw } from 'lucide-react';
import { useStore } from '../../store/useStore';
import { buildContexto } from '../../utils/buildContexto';
import { briefingsApi } from '../../api/content';
import { Button, Card, Textarea, SectionHeader, Spinner, ResultBox, Alert } from '../../components/ui';

export default function TabBriefings() {
  const { selectedAgent, segmentos } = useStore();
  const [descricao, setDescricao] = useState('');
  const [briefing, setBriefing] = useState('');
  const [ajuste, setAjuste] = useState('');
  const [loading, setLoading] = useState(false);
  const [loadingAjuste, setLoadingAjuste] = useState(false);
  const [error, setError] = useState('');

  const ctx = selectedAgent ? buildContexto(selectedAgent, segmentos) : '';

  const handleGerar = async () => {
    if (!descricao.trim()) return;
    setError(''); setLoading(true);
    try {
      const res = await briefingsApi.gerar({ descricao, contextoAgente: ctx });
      setBriefing(res.briefing);
    } catch (e: unknown) { setError((e as Error).message); }
    finally { setLoading(false); }
  };

  const handleAjustar = async () => {
    if (!ajuste.trim() || !briefing) return;
    setLoadingAjuste(true);
    try {
      const res = await briefingsApi.ajustar({ briefingAtual: briefing, ajuste, contextoAgente: ctx });
      setBriefing(res.briefing);
      setAjuste('');
    } catch (e: unknown) { setError((e as Error).message); }
    finally { setLoadingAjuste(false); }
  };

  return (
    <Card>
      <SectionHeader icon={<ClipboardList size={16} />} title="Gerador de Briefings" subtitle="Gere briefings completos e estruturados para seus projetos" />

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="flex flex-col gap-3">
          <Textarea
            label="Descrição do Projeto"
            value={descricao}
            onChange={(e) => setDescricao(e.target.value)}
            rows={10}
            placeholder={`Ex: Campanha para lançamento de fungicida para soja, público de produtores do Centro-Oeste, tom técnico, foco em eficácia e residual…`}
          />
          <Button onClick={handleGerar} loading={loading} disabled={!descricao.trim()} fullWidth>
            <Sparkles size={14} /> Gerar Briefing
          </Button>
          {loading && <Spinner text="Gerando briefing…" />}
          {error && <Alert type="error">{error}</Alert>}
        </div>

        <div className="flex flex-col gap-3">
          <p className="text-xs font-semibold text-slate-400 uppercase tracking-wide">Briefing Gerado</p>
          {briefing ? (
            <>
              <ResultBox content={briefing} filename="briefing.txt" />
              <div className="border-t pt-3" style={{ borderColor: 'var(--border)' }}>
                <Textarea label="Ajustar Briefing" value={ajuste} onChange={(e) => setAjuste(e.target.value)} rows={3} placeholder="Descreva os ajustes desejados…" />
                <Button onClick={handleAjustar} loading={loadingAjuste} disabled={!ajuste.trim()} fullWidth className="mt-2">
                  <RotateCcw size={14} /> Aplicar Ajustes
                </Button>
              </div>
            </>
          ) : (
            <div className="rounded-xl border p-8 text-center text-slate-600 text-sm flex flex-col items-center gap-2" style={{ borderColor: 'var(--border)', background: 'rgba(0,0,0,0.15)' }}>
              <ClipboardList size={24} className="text-slate-700" />
              O briefing gerado aparecerá aqui.
            </div>
          )}
        </div>
      </div>
    </Card>
  );
}
