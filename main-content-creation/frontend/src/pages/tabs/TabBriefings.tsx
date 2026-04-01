import { useState } from 'react';
import { ClipboardList, Sparkles, RotateCcw, Upload, FileDown, Package, Eye } from 'lucide-react';
import { useStore } from '../../store/useStore';
import { buildContexto } from '../../utils/buildContexto';
import { briefingsApi, type BriefingGerado, type PautaExtraida } from '../../api/content';
import { Button, Card, Textarea, Input, SectionHeader, Spinner, Alert, SubTabs } from '../../components/ui';

type Modo = 'calendario' | 'individual';
type SubTab = 'visualizar' | 'ajuste' | 'historico';

function download(content: string, filename: string) {
  const a = Object.assign(document.createElement('a'), {
    href: URL.createObjectURL(new Blob([content], { type: 'text/plain' })),
    download: filename,
  });
  a.click();
}

function downloadZip(briefings: BriefingGerado[]) {
  // Gera um TXT consolidado (ZIP real requer lib adicional)
  const now = new Date().toISOString().slice(0, 10);
  let consolidado = `TODOS OS BRIEFINGS\nTotal: ${briefings.length}\n${'='.repeat(60)}\n\n`;
  for (const b of briefings) {
    consolidado += `BRIEFING ${b.indice}\n`;
    if (b.titulo) consolidado += `Título: ${b.titulo}\n`;
    else consolidado += `Pauta: ${b.conteudoOriginal}\n`;
    consolidado += `${'-'.repeat(40)}\n${b.briefing}\n${'='.repeat(60)}\n\n`;
  }
  download(consolidado, `todos_briefings_${now}.txt`);
}

export default function TabBriefings() {
  const { selectedAgent, segmentos } = useStore();
  const ctx = selectedAgent ? buildContexto(selectedAgent, segmentos) : '';

  const [modo, setModo] = useState<Modo>('individual');

  // Estado compartilhado
  const [briefings, setBriefings] = useState<BriefingGerado[]>([]);
  const [briefingAtual, setBriefingAtual] = useState<BriefingGerado | null>(null);
  const [subTab, setSubTab] = useState<SubTab>('visualizar');
  const [error, setError] = useState('');

  // Modo calendário
  const [csvText, setCsvText] = useState('');
  const [mesReferencia, setMesReferencia] = useState('JANEIRO 2026');
  const [contextoBriefings, setContextoBriefings] = useState('');
  const [pautasDetectadas, setPautasDetectadas] = useState<PautaExtraida[]>([]);
  const [loadingCSV, setLoadingCSV] = useState(false);
  const [loadingGerando, setLoadingGerando] = useState(false);
  const [progressoAtual, setProgressoAtual] = useState(0);
  const [progressoTotal, setProgressoTotal] = useState(0);

  // Modo individual
  const [tituloBriefing, setTituloBriefing] = useState('');
  const [mesReferenciaInd, setMesReferenciaInd] = useState('JANEIRO 2026');
  const [textoBase, setTextoBase] = useState('');
  const [contextoInd, setContextoInd] = useState('');
  const [loadingInd, setLoadingInd] = useState(false);

  // Ajuste
  const [ajuste, setAjuste] = useState('');
  const [loadingAjuste, setLoadingAjuste] = useState(false);

  // ── Calendário ──────────────────────────────────────────────────────────────

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setLoadingCSV(true);
    try {
      const text = await file.text();
      setCsvText(text);
      const res = await briefingsApi.extrairPautas(text);
      setPautasDetectadas(res.pautas);
    } catch (err: unknown) {
      setError('Erro ao ler CSV: ' + (err as Error).message);
    } finally {
      setLoadingCSV(false);
    }
  };

  const handleGerarDeCSV = async () => {
    if (!csvText.trim()) { setError('Nenhum CSV carregado.'); return; }
    setError(''); setLoadingGerando(true);
    setProgressoAtual(0); setProgressoTotal(pautasDetectadas.length);
    try {
      // Gera pauta por pauta para mostrar progresso
      const novos: BriefingGerado[] = [];
      for (let i = 0; i < pautasDetectadas.length; i++) {
        setProgressoAtual(i + 1);
        const pauta = pautasDetectadas[i];
        try {
          const res = await briefingsApi.gerarDePauta({
            conteudoPauta: pauta.conteudo,
            mesReferencia,
            contextoAdicional: contextoBriefings,
            contextoAgente: ctx,
          });
          novos.push({ indice: i + 1, conteudoOriginal: pauta.conteudo, briefing: res.briefing, mesReferencia });
        } catch (err: unknown) {
          novos.push({ indice: i + 1, conteudoOriginal: pauta.conteudo, briefing: `ERRO: ${(err as Error).message}`, mesReferencia });
        }
      }
      setBriefings((prev) => [...prev, ...novos]);
      if (novos.length > 0) setBriefingAtual(novos[0]);
    } catch (err: unknown) { setError((err as Error).message); }
    finally { setLoadingGerando(false); setProgressoAtual(0); setProgressoTotal(0); }
  };

  // ── Individual ──────────────────────────────────────────────────────────────

  const handleGerarIndividual = async () => {
    if (!textoBase.trim()) { setError('O texto base é obrigatório.'); return; }
    if (!tituloBriefing.trim()) { setError('O título do briefing é obrigatório.'); return; }
    setError(''); setLoadingInd(true);
    try {
      const res = await briefingsApi.gerarIndividual({
        titulo: tituloBriefing,
        mesReferencia: mesReferenciaInd,
        textoBase,
        contextoAdicional: contextoInd,
        contextoAgente: ctx,
      });
      const novo: BriefingGerado = {
        indice: briefings.length + 1,
        titulo: tituloBriefing,
        conteudoOriginal: textoBase,
        briefing: res.briefing,
        mesReferencia: mesReferenciaInd,
        tipo: 'individual',
      };
      setBriefings((prev) => [...prev, novo]);
      setBriefingAtual(novo);
      setSubTab('visualizar');
    } catch (err: unknown) { setError((err as Error).message); }
    finally { setLoadingInd(false); }
  };

  // ── Ajuste ──────────────────────────────────────────────────────────────────

  const handleAjustar = async () => {
    if (!ajuste.trim() || !briefingAtual) return;
    setLoadingAjuste(true);
    try {
      const res = await briefingsApi.ajustar({
        briefingAtual: briefingAtual.briefing,
        ajuste,
        tituloOuPauta: briefingAtual.titulo ?? briefingAtual.conteudoOriginal,
        mesReferencia: briefingAtual.mesReferencia,
        contextoAgente: ctx,
      });
      const atualizado = {
        ...briefingAtual,
        briefing: res.briefing,
        historicoAjustes: [
          ...(briefingAtual.historicoAjustes ?? []),
          { data: new Date().toLocaleString('pt-BR'), solicitacao: ajuste },
        ],
      };
      setBriefingAtual(atualizado);
      setBriefings((prev) => prev.map((b) => b.indice === atualizado.indice ? atualizado : b));
      setAjuste('');
    } catch (err: unknown) { setError((err as Error).message); }
    finally { setLoadingAjuste(false); }
  };

  const subTabs = [
    { id: 'visualizar' as SubTab, label: 'Visualizar / Editar' },
    { id: 'ajuste' as SubTab, label: 'Ajuste Pontual' },
    { id: 'historico' as SubTab, label: `Histórico (${briefingAtual?.historicoAjustes?.length ?? 0})` },
  ];

  return (
    <Card>
      <SectionHeader icon={<ClipboardList size={16} />} title="Gerador de Briefings" subtitle="Gere briefings estruturados a partir de pautas ou texto livre" />

      {/* Seletor de modo */}
      <div className="flex gap-1.5 mb-5">
        {([['calendario', 'Upload de Calendário (múltiplos briefings)'], ['individual', 'Texto Único (briefing individual)']] as const).map(([id, label]) => (
          <button
            key={id}
            onClick={() => setModo(id)}
            className="px-4 py-2 text-sm font-medium rounded-xl transition-all border"
            style={{
              background: modo === id ? 'linear-gradient(135deg,#4c1d95,#7c3aed)' : 'rgba(255,255,255,0.03)',
              color: modo === id ? '#fff' : '#64748b',
              borderColor: modo === id ? 'transparent' : 'rgba(139,92,246,0.18)',
            }}
          >
            {id === 'calendario' ? <Upload size={13} className="inline mr-1.5 -mt-0.5" /> : <ClipboardList size={13} className="inline mr-1.5 -mt-0.5" />}
            {label}
          </button>
        ))}
      </div>

      {/* ── Modo calendário ── */}
      {modo === 'calendario' && (
        <div className="flex flex-col gap-4 mb-5">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div>
              <label className="text-xs font-semibold text-slate-400 uppercase tracking-widest mb-1.5 block">Upload do Calendário CSV</label>
              <input
                type="file"
                accept=".csv"
                onChange={handleFileUpload}
                className="text-xs text-slate-500 file:mr-3 file:py-2 file:px-3 file:rounded-xl file:border-0 file:text-xs file:font-semibold file:text-white file:cursor-pointer w-full"
                style={{ ['--file-selector-button-bg' as string]: 'linear-gradient(135deg,#4c1d95,#7c3aed)' }}
              />
              {loadingCSV && <Spinner text="Lendo CSV e extraindo pautas…" />}
              {pautasDetectadas.length > 0 && !loadingCSV && (
                <div className="mt-2 flex items-center gap-2">
                  <Eye size={12} className="text-emerald-400" />
                  <span className="text-xs text-emerald-400 font-medium">{pautasDetectadas.length} pautas detectadas</span>
                </div>
              )}
            </div>
            <Input label="Mês de referência" value={mesReferencia} onChange={(e) => setMesReferencia(e.target.value)} placeholder="JANEIRO 2026" />
          </div>

          <Textarea
            label="Contexto adicional (opcional)"
            value={contextoBriefings}
            onChange={(e) => setContextoBriefings(e.target.value)}
            rows={2}
            placeholder="Foco em campanha de posicionamento, linguagem técnica mas acessível…"
          />

          {/* Preview das pautas detectadas */}
          {pautasDetectadas.length > 0 && (
            <div className="rounded-xl border p-4" style={{ borderColor: 'rgba(139,92,246,0.18)', background: 'rgba(0,0,0,0.15)' }}>
              <p className="text-xs font-semibold text-slate-400 uppercase tracking-widest mb-2">
                Primeiras 10 pautas detectadas ({pautasDetectadas.length} total)
              </p>
              <div className="flex flex-col gap-1">
                {pautasDetectadas.slice(0, 10).map((p) => (
                  <p key={p.indice} className="text-xs text-slate-400">
                    <span className="text-violet-400 font-medium">{p.indice}.</span> {p.conteudo}
                  </p>
                ))}
              </div>
            </div>
          )}

          {/* Progresso */}
          {loadingGerando && progressoTotal > 0 && (
            <div className="rounded-xl border p-4" style={{ borderColor: 'rgba(139,92,246,0.18)', background: 'rgba(0,0,0,0.15)' }}>
              <div className="flex justify-between text-xs text-slate-400 mb-2">
                <span>Gerando briefing {progressoAtual}/{progressoTotal}…</span>
                <span>{Math.round((progressoAtual / progressoTotal) * 100)}%</span>
              </div>
              <div className="rounded-full h-1.5 overflow-hidden" style={{ background: 'rgba(139,92,246,0.15)' }}>
                <div
                  className="h-full rounded-full transition-all duration-300"
                  style={{ width: `${(progressoAtual / progressoTotal) * 100}%`, background: 'linear-gradient(135deg,#4c1d95,#7c3aed)' }}
                />
              </div>
            </div>
          )}

          <Button onClick={handleGerarDeCSV} loading={loadingGerando} disabled={pautasDetectadas.length === 0} fullWidth>
            <Sparkles size={14} /> Processar Calendário e Gerar {pautasDetectadas.length > 0 ? `${pautasDetectadas.length} Briefings` : 'Briefings'}
          </Button>
        </div>
      )}

      {/* ── Modo individual ── */}
      {modo === 'individual' && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-5">
          <div className="flex flex-col gap-3">
            <Input label="Título do briefing" value={tituloBriefing} onChange={(e) => setTituloBriefing(e.target.value)} placeholder="Ex: Lançamento do produto X na cultura Y" />
            <Input label="Mês de referência" value={mesReferenciaInd} onChange={(e) => setMesReferenciaInd(e.target.value)} placeholder="JANEIRO 2026" />
            <Textarea
              label="Texto base para o briefing"
              value={textoBase}
              onChange={(e) => setTextoBase(e.target.value)}
              rows={6}
              placeholder="Cole aqui o texto que servirá de base: pauta, resumo, instruções do cliente…"
            />
            <Textarea
              label="Contexto adicional (opcional)"
              value={contextoInd}
              onChange={(e) => setContextoInd(e.target.value)}
              rows={2}
              placeholder="Informações complementares para orientar a criação…"
            />
            <Button onClick={handleGerarIndividual} loading={loadingInd} disabled={!textoBase.trim() || !tituloBriefing.trim()} fullWidth>
              <Sparkles size={14} /> Gerar Briefing Individual
            </Button>
          </div>

          {/* Lista de briefings gerados no modo individual */}
          {briefings.length > 0 && (
            <div className="flex flex-col gap-2">
              <p className="text-xs font-semibold text-slate-400 uppercase tracking-widest">Briefings gerados ({briefings.length})</p>
              <div className="flex flex-col gap-1.5 overflow-y-auto" style={{ maxHeight: 340 }}>
                {briefings.map((b) => (
                  <button
                    key={b.indice}
                    onClick={() => { setBriefingAtual(b); setSubTab('visualizar'); }}
                    className="text-left rounded-xl border px-3 py-2.5 text-sm transition-all"
                    style={{
                      borderColor: briefingAtual?.indice === b.indice ? 'rgba(124,58,237,0.5)' : 'rgba(139,92,246,0.15)',
                      background: briefingAtual?.indice === b.indice ? 'rgba(124,58,237,0.12)' : 'rgba(0,0,0,0.15)',
                    }}
                  >
                    <p className="text-violet-300 font-medium text-xs truncate">{b.titulo ?? b.conteudoOriginal.slice(0, 60)}</p>
                    <p className="text-slate-600 text-xs mt-0.5">{b.mesReferencia}</p>
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {error && <div className="mb-4"><Alert type="error">{error}</Alert></div>}

      {/* ── Lista de briefings (modo calendário) ── */}
      {modo === 'calendario' && briefings.length > 0 && (
        <div className="mb-4">
          <p className="text-xs font-semibold text-slate-400 uppercase tracking-widest mb-2">Selecionar briefing ({briefings.length} gerados)</p>
          <select
            className="w-full rounded-xl border px-3.5 py-2.5 text-sm text-slate-200 outline-none"
            style={{ background: '#1a2440', borderColor: 'rgba(139,92,246,0.18)' }}
            value={briefingAtual?.indice ?? ''}
            onChange={(e) => {
              const b = briefings.find((x) => x.indice === +e.target.value);
              if (b) { setBriefingAtual(b); setSubTab('visualizar'); }
            }}
          >
            {briefings.map((b) => (
              <option key={b.indice} value={b.indice}>
                {b.indice}. {(b.titulo ?? b.conteudoOriginal).slice(0, 70)}… ({b.mesReferencia})
              </option>
            ))}
          </select>
        </div>
      )}

      {/* ── Editor do briefing selecionado ── */}
      {briefingAtual && (
        <div className="border-t pt-5" style={{ borderColor: 'rgba(139,92,246,0.12)' }}>
          <div className="flex items-center justify-between mb-4 flex-wrap gap-2">
            <div>
              <p className="text-sm font-semibold text-slate-200">Briefing {briefingAtual.indice}</p>
              <p className="text-xs text-slate-500 mt-0.5">
                {briefingAtual.titulo ? `Título: ${briefingAtual.titulo}` : `Pauta: ${briefingAtual.conteudoOriginal.slice(0, 80)}…`}
                {' · '}{briefingAtual.mesReferencia}
              </p>
            </div>
          </div>

          <SubTabs tabs={subTabs} active={subTab} onChange={setSubTab} />

          {/* Visualizar / Editar direto */}
          {subTab === 'visualizar' && (
            <div className="flex flex-col gap-3">
              <Textarea
                label="Conteúdo do Briefing (edição direta)"
                value={briefingAtual.briefing}
                onChange={(e) => {
                  const atualizado = { ...briefingAtual, briefing: e.target.value };
                  setBriefingAtual(atualizado);
                  setBriefings((prev) => prev.map((b) => b.indice === atualizado.indice ? atualizado : b));
                }}
                rows={16}
              />
            </div>
          )}

          {/* Ajuste pontual */}
          {subTab === 'ajuste' && (
            <div className="flex flex-col gap-3">
              <p className="text-xs text-slate-500">Descreva o ajuste desejado. A estrutura original será mantida — apenas o solicitado será alterado.</p>
              <Textarea
                label="Solicitação de ajuste"
                value={ajuste}
                onChange={(e) => setAjuste(e.target.value)}
                rows={4}
                placeholder="Exemplos:&#10;- Adicione mais detalhes sobre o público-alvo&#10;- Inclua informações sobre o produto X na seção de produtos"
              />
              <Button onClick={handleAjustar} loading={loadingAjuste} disabled={!ajuste.trim()} fullWidth>
                <RotateCcw size={14} /> Aplicar Ajuste Pontual
              </Button>
              {loadingAjuste && <Spinner text="Aplicando ajuste…" />}
            </div>
          )}

          {/* Histórico */}
          {subTab === 'historico' && (
            <div className="flex flex-col gap-2">
              {(briefingAtual.historicoAjustes ?? []).length === 0 ? (
                <p className="text-slate-600 text-sm text-center py-6">Nenhum ajuste realizado ainda.</p>
              ) : (
                [...(briefingAtual.historicoAjustes ?? [])].map((h, i) => (
                  <div key={i} className="rounded-xl border p-3" style={{ borderColor: 'rgba(139,92,246,0.15)', background: 'rgba(0,0,0,0.15)' }}>
                    <p className="text-xs font-semibold text-violet-300">{i + 1}. {h.data}</p>
                    <p className="text-xs text-slate-400 mt-1">{h.solicitacao}</p>
                  </div>
                ))
              )}
            </div>
          )}

          {/* Downloads */}
          <div className="mt-4 pt-4 border-t grid grid-cols-1 sm:grid-cols-3 gap-2" style={{ borderColor: 'rgba(139,92,246,0.12)' }}>
            <Button
              variant="secondary"
              onClick={() => download(briefingAtual.briefing, `briefing_${briefingAtual.indice}.txt`)}
              fullWidth
            >
              <FileDown size={14} /> Baixar Este Briefing
            </Button>

            {(briefingAtual.historicoAjustes ?? []).length > 0 && (
              <Button
                variant="secondary"
                onClick={() => {
                  const conteudo = `# BRIEFING ${briefingAtual.indice}\n\n## BRIEFING ATUAL\n${briefingAtual.briefing}\n\n## HISTÓRICO\n` +
                    briefingAtual.historicoAjustes!.map((h, i) => `\n${i + 1}. ${h.data}\n   ${h.solicitacao}`).join('');
                  download(conteudo, `briefing_${briefingAtual.indice}_historico.txt`);
                }}
                fullWidth
              >
                <ClipboardList size={14} /> Baixar com Histórico
              </Button>
            )}

            {briefings.length > 1 && (
              <Button variant="secondary" onClick={() => downloadZip(briefings)} fullWidth>
                <Package size={14} /> Baixar Todos ({briefings.length})
              </Button>
            )}
          </div>
        </div>
      )}
    </Card>
  );
}
