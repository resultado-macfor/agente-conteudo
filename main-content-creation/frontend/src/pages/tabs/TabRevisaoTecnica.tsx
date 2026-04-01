import { useState } from 'react';
import { Wrench, Microscope, RotateCcw, FileDown, Package } from 'lucide-react';
import { useStore } from '../../store/useStore';
import { buildContexto } from '../../utils/buildContexto';
import { contentApi } from '../../api/content';
import { Button, Card, Textarea, Select, SectionHeader, Spinner, Alert, Checkbox, Stat, SubTabs } from '../../components/ui';

const TIPOS = ['Artigo Técnico', 'Material Comercial', 'Blog Post', 'Manual Técnico', 'Comunicado Técnico'];
const RIGOR = ['Leve', 'Moderado', 'Rigoroso', 'Especialista'];

type ResultTab = 'reescrito' | 'relatorio' | 'analise';

function download(content: string, filename: string, mime = 'text/plain') {
  const a = Object.assign(document.createElement('a'), {
    href: URL.createObjectURL(new Blob([content], { type: mime })),
    download: filename,
  });
  a.click();
}

export default function TabRevisaoTecnica() {
  const { selectedAgent, segmentos } = useStore();

  // Inputs
  const [texto, setTexto] = useState('');
  const [tipoConteudo, setTipoConteudo] = useState(TIPOS[0]);
  const [nivelRigor, setNivelRigor] = useState(RIGOR[1]);
  const [limiteDocumentos, setLimiteDocumentos] = useState(12);
  const [ragTaxonomia, setRagTaxonomia] = useState(true);
  const [ragEpidemiologia, setRagEpidemiologia] = useState(true);
  const [ragProdutos, setRagProdutos] = useState(true);
  const [ragGeral, setRagGeral] = useState(true);
  const [incluirRelatorio, setIncluirRelatorio] = useState(true);

  // Resultados
  const [textoReescrito, setTextoReescrito] = useState('');
  const [relatorio, setRelatorio] = useState('');
  const [statsRags, setStatsRags] = useState<Record<string, number>>({});
  const [resultTab, setResultTab] = useState<ResultTab>('reescrito');

  // Ajustes incrementais
  const [ajuste, setAjuste] = useState('');
  const [loadingAjuste, setLoadingAjuste] = useState(false);

  // Loading / error
  const [loading, setLoading] = useState(false);
  const [fase, setFase] = useState('');
  const [error, setError] = useState('');

  const ctx = selectedAgent ? buildContexto(selectedAgent, segmentos) : '';

  const handleRevisar = async () => {
    if (!texto.trim()) return;
    setError('');
    setTextoReescrito('');
    setRelatorio('');
    setStatsRags({});
    setLoading(true);

    try {
      setFase('Fase 1 — Buscando com RAGs especializados…');
      const res = await contentApi.revisaoTecnica({
        texto,
        rags: { taxonomia: ragTaxonomia, epidemiologia: ragEpidemiologia, produtos: ragProdutos, geral: ragGeral },
        limite: limiteDocumentos,
        contextoAgente: ctx,
        incluirRelatorio,
        nivelRigor,
        tipoConteudo,
      }) as { textoReescrito: string; relatorioMudancas: string; resultadosRags: Record<string, unknown[]> };

      const stats: Record<string, number> = {};
      for (const [k, v] of Object.entries(res.resultadosRags ?? {})) stats[k] = (v as unknown[]).length;
      setStatsRags(stats);

      setFase('Fase 2 — Reescrevendo conteúdo…');
      setTextoReescrito(res.textoReescrito);
      setRelatorio(res.relatorioMudancas ?? '');
      setResultTab('reescrito');
    } catch (e: unknown) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
      setFase('');
    }
  };

  const handleAjustar = async () => {
    if (!ajuste.trim() || !textoReescrito) return;
    setLoadingAjuste(true);
    try {
      const res = await contentApi.ajusteRevisaoRag(texto, textoReescrito, ajuste);
      setTextoReescrito(res.resultado);
      setAjuste('');
      setResultTab('reescrito');
    } catch (e: unknown) {
      setError((e as Error).message);
    } finally {
      setLoadingAjuste(false);
    }
  };

  const now = () => new Date().toISOString().slice(0, 16).replace('T', '_').replace(':', '');

  const handleDownloadPacote = () => {
    let pacote = `TEXTO ORIGINAL:\n${texto}\n\n${'='.repeat(60)}\n\nTEXTO REESCRITO COM RAGs:\n${textoReescrito}`;
    if (incluirRelatorio && relatorio) {
      pacote += `\n\n${'='.repeat(60)}\n\nRELATÓRIO DE MUDANÇAS:\n${relatorio}`;
    }
    download(pacote, `revisao_completa_rags_${now()}.txt`);
  };

  const resultTabs = [
    { id: 'reescrito' as ResultTab, label: 'Texto Reescrito' },
    { id: 'relatorio' as ResultTab, label: 'Relatório de Mudanças' },
    { id: 'analise' as ResultTab, label: 'Análise RAGs' },
  ];

  return (
    <Card>
      <SectionHeader
        icon={<Wrench size={16} />}
        title="Revisão Técnica com RAGs Especializados"
        subtitle="Análise em camadas: taxonomia, epidemiologia, produtos + reescrita final com relatório detalhado"
      />

      {/* Layout 2 colunas: original | revisado */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-4">
        {/* Coluna esquerda — original */}
        <div className="flex flex-col gap-2">
          <p className="text-xs font-semibold text-slate-400 uppercase tracking-widest">Conteúdo Original</p>
          <Textarea
            value={texto}
            onChange={(e) => setTexto(e.target.value)}
            rows={14}
            placeholder="Cole aqui o conteúdo técnico agrícola que precisa ser revisado..."
          />
        </div>

        {/* Coluna direita — revisado */}
        <div className="flex flex-col gap-2">
          <p className="text-xs font-semibold text-slate-400 uppercase tracking-widest">Conteúdo Revisado com RAG</p>
          {textoReescrito ? (
            <>
              <SubTabs tabs={resultTabs} active={resultTab} onChange={setResultTab} />

              {resultTab === 'reescrito' && (
                <textarea
                  readOnly
                  value={textoReescrito}
                  rows={10}
                  className="rounded-xl px-4 py-3 text-sm text-slate-300 outline-none resize-y font-mono w-full"
                  style={{ background: 'rgba(0,0,0,0.3)', border: '1px solid rgba(139,92,246,0.12)' }}
                />
              )}

              {resultTab === 'relatorio' && (
                relatorio ? (
                  <div
                    className="rounded-xl px-4 py-3 text-sm text-slate-300 overflow-y-auto whitespace-pre-wrap"
                    style={{ background: 'rgba(0,0,0,0.3)', border: '1px solid rgba(139,92,246,0.12)', maxHeight: 380 }}
                  >
                    {relatorio}
                  </div>
                ) : (
                  <div className="rounded-xl p-6 text-center text-slate-600 text-sm" style={{ background: 'rgba(0,0,0,0.2)', border: '1px solid rgba(139,92,246,0.1)' }}>
                    Nenhum relatório gerado.
                  </div>
                )
              )}

              {resultTab === 'analise' && (
                <div className="flex flex-col gap-3">
                  <div className="grid grid-cols-3 gap-2">
                    <Stat label="Palavras Original"  value={texto.split(' ').length} />
                    <Stat label="Palavras Reescrito" value={textoReescrito.split(' ').length} />
                    <Stat
                      label="Diferença"
                      value={`${textoReescrito.split(' ').length - texto.split(' ').length > 0 ? '+' : ''}${textoReescrito.split(' ').length - texto.split(' ').length}`}
                    />
                  </div>
                  <div className="grid grid-cols-2 gap-2">
                    {Object.entries(statsRags).map(([k, v]) => (
                      <Stat key={k} label={`RAG ${k}`} value={v} />
                    ))}
                  </div>
                </div>
              )}
            </>
          ) : (
            <div
              className="rounded-xl p-6 flex flex-col items-center justify-center text-center gap-2 flex-1"
              style={{ background: 'rgba(0,0,0,0.2)', border: '1px solid rgba(139,92,246,0.1)', minHeight: 200 }}
            >
              <Microscope size={28} className="text-slate-700" />
              <p className="text-slate-600 text-sm">
                {loading ? fase : 'Aguardando revisão com RAG…'}
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Configurações */}
      <div
        className="rounded-xl border p-4 mb-4 grid grid-cols-1 sm:grid-cols-3 gap-4"
        style={{ borderColor: 'rgba(139,92,246,0.15)', background: 'rgba(0,0,0,0.15)' }}
      >
        <Select label="Tipo de Conteúdo" value={tipoConteudo} onChange={(e) => setTipoConteudo(e.target.value)}>
          {TIPOS.map((t) => <option key={t}>{t}</option>)}
        </Select>

        <Select label="Nível de Rigor" value={nivelRigor} onChange={(e) => setNivelRigor(e.target.value)}>
          {RIGOR.map((r) => <option key={r}>{r}</option>)}
        </Select>

        <div className="flex flex-col gap-1.5">
          <label className="text-xs font-semibold text-slate-400 uppercase tracking-widest">Docs por RAG: {limiteDocumentos}</label>
          <input type="range" min={3} max={20} value={limiteDocumentos}
            onChange={(e) => setLimiteDocumentos(+e.target.value)} className="accent-violet-500 mt-1" />
        </div>

        <div className="flex flex-col gap-2">
          <p className="text-xs font-semibold text-slate-400 uppercase tracking-widest mb-1">RAGs Ativos</p>
          <Checkbox label="Taxonomia"    checked={ragTaxonomia}    onChange={setRagTaxonomia} />
          <Checkbox label="Epidemiologia" checked={ragEpidemiologia} onChange={setRagEpidemiologia} />
          <Checkbox label="Produtos"     checked={ragProdutos}     onChange={setRagProdutos} />
          <Checkbox label="Geral"        checked={ragGeral}        onChange={setRagGeral} />
        </div>

        <div className="flex flex-col gap-2 sm:col-span-2">
          <Checkbox label="Incluir relatório detalhado de mudanças" checked={incluirRelatorio} onChange={setIncluirRelatorio} />
        </div>
      </div>

      {/* Botão principal */}
      {error && <div className="mb-3"><Alert type="error">{error}</Alert></div>}

      <Button onClick={handleRevisar} loading={loading} disabled={!texto.trim()} fullWidth>
        <Microscope size={14} /> Realizar Revisão com RAGs Especializados
      </Button>

      {loading && (
        <div className="mt-3">
          <Spinner text={fase || 'Executando pipeline de RAGs…'} />
        </div>
      )}

      {/* Downloads — aparecem após gerar */}
      {textoReescrito && (
        <div className="mt-5 flex flex-col gap-4">
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            <Button
              variant="secondary"
              onClick={() => download(textoReescrito, `texto_reescrito_rags_${now()}.txt`)}
              fullWidth
            >
              <FileDown size={14} /> Baixar Texto Reescrito
            </Button>

            {incluirRelatorio && relatorio && (
              <Button
                variant="secondary"
                onClick={() => download(relatorio, `relatorio_mudancas_${now()}.md`, 'text/markdown')}
                fullWidth
              >
                <FileDown size={14} /> Baixar Relatório
              </Button>
            )}

            <Button variant="secondary" onClick={handleDownloadPacote} fullWidth>
              <Package size={14} /> Baixar Pacote Completo
            </Button>
          </div>

          {/* Ajustes incrementais */}
          <div className="border-t pt-4" style={{ borderColor: 'rgba(139,92,246,0.12)' }}>
            <p className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-3">
              Ajustes Incrementais para RAGs
            </p>
            <p className="text-xs text-slate-600 mb-3">
              Use o campo abaixo para solicitar ajustes específicos na última revisão com RAGs.
            </p>
            <div className="flex gap-2">
              <Textarea
                value={ajuste}
                onChange={(e) => setAjuste(e.target.value)}
                rows={3}
                placeholder="Exemplos:&#10;- Aumente o foco na taxonomia dos patógenos&#10;- Inclua mais informações epidemiológicas"
                className="flex-1"
              />
              <Button
                onClick={handleAjustar}
                loading={loadingAjuste}
                disabled={!ajuste.trim()}
                className="self-end shrink-0"
              >
                <RotateCcw size={14} /> Ajustar
              </Button>
            </div>
          </div>
        </div>
      )}
    </Card>
  );
}
