import { useState } from 'react';
import { TrendingUp, Search, Zap, RotateCcw } from 'lucide-react';
import { useStore } from '../../store/useStore';
import { buildContexto } from '../../utils/buildContexto';
import { contentApi } from '../../api/content';
import { Button, Card, Textarea, Select, SectionHeader, Spinner, ResultBox, Alert, Checkbox, Stat, SubTabs } from '../../components/ui';

type ModoEntrada = 'Briefing + Conteúdo original' | 'Apenas Briefing' | 'Apenas Conteúdo original';
const MODOS: ModoEntrada[] = ['Briefing + Conteúdo original', 'Apenas Briefing', 'Apenas Conteúdo original'];
const HEADINGS = ['H2', 'H3', 'H1'];
type Step = 'input' | 'avaliacao' | 'resultado';

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
  const [conteudoFinal, setConteudoFinal] = useState('');
  const [ajuste, setAjuste] = useState('');
  const [historicAjustes, setHistoricAjustes] = useState<string[]>([]);
  const [step, setStep] = useState<Step>('input');
  const [loadingAval, setLoadingAval] = useState(false);
  const [loadingGer, setLoadingGer] = useState(false);
  const [loadingAjuste, setLoadingAjuste] = useState(false);
  const [error, setError] = useState('');

  const usaBriefing = modo !== 'Apenas Conteúdo original';
  const usaConteudo = modo !== 'Apenas Briefing';
  const ctx = selectedAgent ? buildContexto(selectedAgent, segmentos) : '';

  const handleAvaliar = async () => {
    if (usaBriefing && !briefing.trim()) { setError('Cole o briefing de entrada.'); return; }
    if (usaConteudo && !conteudoOriginal.trim()) { setError('Cole o conteúdo original.'); return; }
    setError(''); setLoadingAval(true);
    try {
      const res = await contentApi.otimizacaoSEO({
        briefing: usaBriefing ? briefing : '',
        conteudoOriginal: usaConteudo ? conteudoOriginal : '',
        contextoAgente: ctx, nivelHeading, qtdInternos, qtdExternos,
      });
      setAvaliacao(res.resultado);
      setConteudoFinal('');
      setStep('avaliacao');
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
      setStep('resultado');
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

  const stepTabs = [
    { id: 'input' as Step, label: '1. Entrada' },
    { id: 'avaliacao' as Step, label: '2. Avaliação' },
    { id: 'resultado' as Step, label: '3. Resultado' },
  ];

  return (
    <Card>
      <SectionHeader icon={<TrendingUp size={16} />} title="Otimização SEO de Conteúdo" subtitle="Pipeline de 3 etapas: entrada → avaliação → geração otimizada" />

      <SubTabs tabs={stepTabs} active={step} onChange={setStep} />

      {/* Step 1 — Input */}
      {step === 'input' && (
        <div className="flex flex-col gap-4">
          {/* Modo */}
          <div className="flex gap-1.5 flex-wrap">
            {MODOS.map((m) => (
              <button key={m} onClick={() => setModo(m)}
                className="px-3 py-1.5 text-xs font-medium rounded-lg transition-all border"
                style={{
                  background: modo === m ? 'linear-gradient(135deg,#4c1d95,#7c3aed)' : 'rgba(255,255,255,0.03)',
                  color: modo === m ? '#fff' : '#64748b',
                  borderColor: modo === m ? 'transparent' : 'var(--border)',
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
              <label className="text-xs font-semibold text-slate-400 uppercase tracking-wide">Links internos: {qtdInternos}</label>
              <input type="range" min={1} max={10} value={qtdInternos} onChange={(e) => setQtdInternos(+e.target.value)} className="accent-violet-500 mt-1" />
            </div>
            <div className="flex flex-col gap-1.5">
              <label className="text-xs font-semibold text-slate-400 uppercase tracking-wide">Links externos: {qtdExternos}</label>
              <input type="range" min={0} max={10} value={qtdExternos} onChange={(e) => setQtdExternos(+e.target.value)} className="accent-violet-500 mt-1" />
            </div>
            <Checkbox label="Busca web (Perplexity)" checked={usarBuscaWeb} onChange={setUsarBuscaWeb} />
          </div>

          {error && <Alert type="error">{error}</Alert>}
          <Button onClick={handleAvaliar} loading={loadingAval} fullWidth>
            <Search size={14} /> Avaliar Conteúdo
          </Button>
          {loadingAval && <Spinner text="Analisando conteúdo…" />}
        </div>
      )}

      {/* Step 2 — Avaliação */}
      {step === 'avaliacao' && (
        <div className="flex flex-col gap-4">
          <Alert type="success">Avaliação concluída. Revise os pontos identificados e gere o conteúdo otimizado.</Alert>
          <ResultBox content={avaliacao} filename="avaliacao_seo.txt" />
          {error && <Alert type="error">{error}</Alert>}
          <Button onClick={handleGerar} loading={loadingGer} fullWidth>
            <Zap size={14} /> Gerar Conteúdo Otimizado
          </Button>
          {loadingGer && <Spinner text="Gerando conteúdo otimizado…" />}
        </div>
      )}

      {/* Step 3 — Resultado */}
      {step === 'resultado' && conteudoFinal && (
        <div className="flex flex-col gap-4">
          <div className="grid grid-cols-3 gap-3">
            <Stat label="Palavras" value={conteudoFinal.split(' ').length} />
            <Stat label="Headings" value={(conteudoFinal.match(/## |### /g) ?? []).length} />
            <Stat label="CTA" value={/maisagro|syngenta/i.test(conteudoFinal) ? '✓' : '✗'} />
          </div>
          <ResultBox content={conteudoFinal} filename="conteudo_otimizado.md" />

          {/* Ajustes */}
          <div className="border-t pt-4" style={{ borderColor: 'var(--border)' }}>
            <p className="text-xs font-semibold text-slate-400 uppercase tracking-wide mb-2">Ajustes Incrementais</p>
            <div className="flex gap-2">
              <Textarea value={ajuste} onChange={(e) => setAjuste(e.target.value)} rows={2} placeholder="Reescreva a introdução, ajuste o tom, aprofunde a seção X…" className="flex-1" />
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
