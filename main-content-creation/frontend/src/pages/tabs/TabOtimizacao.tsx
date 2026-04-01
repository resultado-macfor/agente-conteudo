import { useState } from 'react';
import { TrendingUp, Search, Zap, RotateCcw, FileDown, CheckCircle } from 'lucide-react';
import { useStore } from '../../store/useStore';
import { buildContexto } from '../../utils/buildContexto';
import { contentApi } from '../../api/content';
import api from '../../api/client';
import { Button, Card, Textarea, Select, SectionHeader, Spinner, ResultBox, Alert, Checkbox, Stat } from '../../components/ui';

type ModoEntrada = 'Briefing + Conteúdo original' | 'Apenas Briefing' | 'Apenas Conteúdo original';
const MODOS: ModoEntrada[] = ['Briefing + Conteúdo original', 'Apenas Briefing', 'Apenas Conteúdo original'];
const HEADINGS = ['H2', 'H3', 'H1'];

export default function TabOtimizacao() {
  const { selectedAgent, segmentos } = useStore();
  const [modo, setModo] = useState<ModoEntrada>(MODOS[0]);
  const [briefing, setBriefing] = useState('');
  const [conteudoOriginal, setConteudoOriginal] = useState('');
  const [nivelHeading, setNivelHeading] = useState(HEADINGS[0]);
  const [qtdInternos, setQtdInternos] = useState(3);
  const [qtdExternos, setQtdExternos] = useState(2);
  const [usarBuscaWeb, setUsarBuscaWeb] = useState(false);
  const [avaliacao, setAvaliacao] = useState('');
  const [avaliacaoFeita, setAvaliacaoFeita] = useState(false);
  const [conteudoFinal, setConteudoFinal] = useState('');
  const [ajuste, setAjuste] = useState('');
  const [historicAjustes, setHistoricAjustes] = useState<string[]>([]);
  const [loadingAval, setLoadingAval] = useState(false);
  const [loadingGer, setLoadingGer] = useState(false);
  const [loadingAjuste, setLoadingAjuste] = useState(false);
  const [loadingDocx, setLoadingDocx] = useState(false);
  const [error, setError] = useState('');

  const usaBriefing = modo !== 'Apenas Conteúdo original';
  const usaConteudo = modo !== 'Apenas Briefing';
  const ctx = selectedAgent ? buildContexto(selectedAgent, segmentos) : '';

  const handleAvaliar = async () => {
    if (usaBriefing && !briefing.trim()) { setError('Cole o briefing de entrada.'); return; }
    if (usaConteudo && !conteudoOriginal.trim()) { setError('Cole o conteúdo original.'); return; }
    setError('');
    setAvaliacaoFeita(false);
    setConteudoFinal('');
    setLoadingAval(true);
    try {
      const res = await contentApi.otimizacaoSEO({
        briefing: usaBriefing ? briefing : '',
        conteudoOriginal: usaConteudo ? conteudoOriginal : '',
        contextoAgente: ctx, nivelHeading, qtdInternos, qtdExternos,
      });
      setAvaliacao(res.resultado);
      setAvaliacaoFeita(true);
    } catch (e: unknown) { setError((e as Error).message); }
    finally { setLoadingAval(false); }
  };

  const handleGerar = async () => {
    setError(''); setLoadingGer(true);
    try {
      let fontes = '';
      if (usarBuscaWeb && conteudoOriginal) {
        const r = await contentApi.perplexity(conteudoOriginal);
        fontes = r.resultado;
      }
      const res = await contentApi.otimizacaoSEO({
        briefing, conteudoOriginal, contextoAgente: ctx,
        avaliacao, fontes, nivelHeading, qtdInternos, qtdExternos,
      });
      setConteudoFinal(res.resultado);
    } catch (e: unknown) { setError((e as Error).message); }
    finally { setLoadingGer(false); }
  };

  const handleAjustar = async () => {
    if (!ajuste.trim() || !conteudoFinal) return;
    setLoadingAjuste(true);
    try {
      const prompt = `Aplique os ajustes ao conteúdo abaixo mantendo INTEGRALMENTE as regras do portal Mais Agro.\n\nCONTEÚDO ATUAL:\n${conteudoFinal}\n\nAJUSTES SOLICITADOS:\n${ajuste}\n\nRetorne o conteúdo completo com os ajustes aplicados.`;
      const res = await contentApi.otimizacaoSEO({ briefing, conteudoOriginal: conteudoFinal, contextoAgente: ctx, avaliacao: prompt, nivelHeading, qtdInternos, qtdExternos });
      setHistoricAjustes((p) => [...p, ajuste]);
      setConteudoFinal(res.resultado);
      setAjuste('');
    } catch (e: unknown) { setError((e as Error).message); }
    finally { setLoadingAjuste(false); }
  };

  const handleDownloadDocx = async () => {
    setLoadingDocx(true);
    try {
      const res = await api.post('/content/gerar-docx', { conteudo: conteudoFinal }, { responseType: 'blob' });
      const url = URL.createObjectURL(new Blob([res.data], { type: 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' }));
      const a = document.createElement('a');
      a.href = url;
      a.download = `conteudo_otimizado_${new Date().toISOString().slice(0, 10)}.docx`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (e: unknown) { setError('Erro ao gerar DOCX: ' + (e as Error).message); }
    finally { setLoadingDocx(false); }
  };

  return (
    <Card>
      <SectionHeader icon={<TrendingUp size={16} />} title="Otimização SEO de Conteúdo" subtitle="Avalie, gere e exporte conteúdo otimizado para o Mais Agro" />

      {/* ── Entradas ── */}
      <div className="flex flex-col gap-4">
        {/* Modo */}
        <div className="flex gap-1.5 flex-wrap">
          {MODOS.map((m) => (
            <button
              key={m}
              onClick={() => setModo(m)}
              className="px-3 py-1.5 text-xs font-medium rounded-lg transition-all border"
              style={{
                background: modo === m ? 'linear-gradient(135deg,#4c1d95,#7c3aed)' : 'rgba(255,255,255,0.03)',
                color: modo === m ? '#fff' : '#64748b',
                borderColor: modo === m ? 'transparent' : 'rgba(139,92,246,0.15)',
              }}
            >
              {m}
            </button>
          ))}
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <Textarea
            label={`Briefing${!usaBriefing ? ' (não utilizado)' : ''}`}
            value={briefing} onChange={(e) => setBriefing(e.target.value)}
            rows={10} disabled={!usaBriefing}
            placeholder="Cole o briefing com título H1, KWs, estrutura H2/H3, CTA, tom…"
          />
          <Textarea
            label={`Conteúdo Original${!usaConteudo ? ' (não utilizado)' : ''}`}
            value={conteudoOriginal} onChange={(e) => setConteudoOriginal(e.target.value)}
            rows={10} disabled={!usaConteudo}
            placeholder="Cole o conteúdo que será avaliado e otimizado."
          />
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          <Select label="Heading Corpo" value={nivelHeading} onChange={(e) => setNivelHeading(e.target.value)}>
            {HEADINGS.map((h) => <option key={h}>{h}</option>)}
          </Select>
          <div className="flex flex-col gap-1.5">
            <label className="text-xs font-semibold text-slate-400 uppercase tracking-widest">Links internos: {qtdInternos}</label>
            <input type="range" min={1} max={10} value={qtdInternos} onChange={(e) => setQtdInternos(+e.target.value)} className="accent-violet-500 mt-1" />
          </div>
          <div className="flex flex-col gap-1.5">
            <label className="text-xs font-semibold text-slate-400 uppercase tracking-widest">Links externos: {qtdExternos}</label>
            <input type="range" min={0} max={10} value={qtdExternos} onChange={(e) => setQtdExternos(+e.target.value)} className="accent-violet-500 mt-1" />
          </div>
          <Checkbox label="Busca web (Perplexity)" checked={usarBuscaWeb} onChange={setUsarBuscaWeb} />
        </div>

        {error && <Alert type="error">{error}</Alert>}

        {/* Etapa 1 — Avaliar */}
        <div className="border-t pt-4" style={{ borderColor: 'rgba(139,92,246,0.12)' }}>
          <p className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-3">Etapa 1 — Avaliação do Conteúdo</p>
          <Button onClick={handleAvaliar} loading={loadingAval} variant="secondary" fullWidth>
            <Search size={14} /> Avaliar Conteúdo
          </Button>
          {loadingAval && <Spinner text="Analisando conteúdo…" />}
        </div>

        {/* Etapa 2 — Gerar */}
        {avaliacaoFeita && !conteudoFinal && (
          <div className="border-t pt-4" style={{ borderColor: 'rgba(139,92,246,0.12)' }}>
            <p className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-3">Etapa 2 — Geração do Conteúdo Otimizado</p>
            <div
              className="flex items-center gap-3 rounded-xl px-4 py-3 mb-4"
              style={{ background: 'rgba(16,185,129,0.08)', border: '1px solid rgba(16,185,129,0.2)' }}
            >
              <CheckCircle size={16} className="text-emerald-400 shrink-0" />
              <span className="text-emerald-300 text-sm font-medium">Análise feita com sucesso</span>
            </div>
            <Button onClick={handleGerar} loading={loadingGer} fullWidth>
              <Zap size={14} /> Implementar e Gerar Conteúdo
            </Button>
            {loadingGer && <Spinner text="Gerando conteúdo otimizado…" />}
          </div>
        )}
      </div>

      {/* ── Resultado ── */}
      {conteudoFinal && (
        <div className="mt-6 flex flex-col gap-4">
          <div className="flex items-center justify-between">
            <p className="text-xs font-bold text-slate-500 uppercase tracking-widest">Conteúdo Otimizado</p>
            <div className="grid grid-cols-3 gap-2">
              <Stat label="Palavras"  value={conteudoFinal.split(' ').length} />
              <Stat label="Headings"  value={(conteudoFinal.match(/#{1,4} /g) ?? []).length} />
              <Stat label="CTA"       value={/maisagro|syngenta/i.test(conteudoFinal) ? '✓' : '✗'} />
            </div>
          </div>

          {/* Download DOCX — destaque */}
          <button
            onClick={handleDownloadDocx}
            disabled={loadingDocx}
            className="w-full flex items-center justify-center gap-2.5 rounded-xl py-3 text-sm font-bold text-white transition-all hover:brightness-110 active:scale-[0.98] disabled:opacity-40 shadow-lg"
            style={{ background: 'linear-gradient(135deg,#1a2d5a,#7c3aed)' }}
          >
            {loadingDocx ? (
              <>
                <svg className="animate-spin h-4 w-4" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z" />
                </svg>
                Gerando DOCX…
              </>
            ) : (
              <>
                <FileDown size={16} />
                Baixar Conteúdo Otimizado (.docx)
              </>
            )}
          </button>

          <ResultBox content={conteudoFinal} filename="conteudo_otimizado.md" />

          {/* Ajustes incrementais */}
          <div className="border-t pt-4" style={{ borderColor: 'rgba(139,92,246,0.12)' }}>
            <p className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-3">Ajustes Incrementais</p>
            <div className="flex gap-2">
              <Textarea
                value={ajuste}
                onChange={(e) => setAjuste(e.target.value)}
                rows={2}
                placeholder="Reescreva a introdução, ajuste o tom, aprofunde a seção X…"
                className="flex-1"
              />
              <Button onClick={handleAjustar} loading={loadingAjuste} disabled={!ajuste.trim()} className="self-end shrink-0">
                <RotateCcw size={14} />
              </Button>
            </div>
            {historicAjustes.length > 0 && (
              <p className="text-xs text-slate-600 mt-2">{historicAjustes.length} ajuste(s) aplicado(s)</p>
            )}
          </div>
        </div>
      )}
    </Card>
  );
}
