import { useState } from 'react';
import { Wrench, Microscope, RotateCcw } from 'lucide-react';
import { useStore } from '../../store/useStore';
import { buildContexto } from '../../utils/buildContexto';
import { contentApi } from '../../api/content';
import { Button, Card, Textarea, Select, SectionHeader, Spinner, ResultBox, Alert, Checkbox, Stat, SubTabs } from '../../components/ui';

const TIPOS = ['Artigo Técnico', 'Material Comercial', 'Blog Post', 'Manual Técnico', 'Comunicado Técnico'];
const RIGOR = ['Leve', 'Moderado', 'Rigoroso', 'Especialista'];

type SubTab = 'reescrito' | 'relatorio' | 'analise';

export default function TabRevisaoTecnica() {
  const { selectedAgent, segmentos } = useStore();
  const [texto, setTexto] = useState('');
  const [tipoConteudo, setTipoConteudo] = useState(TIPOS[0]);
  const [nivelRigor, setNivelRigor] = useState(RIGOR[1]);
  const [limiteDocumentos, setLimiteDocumentos] = useState(12);
  const [ragTaxonomia, setRagTaxonomia] = useState(true);
  const [ragEpidemiologia, setRagEpidemiologia] = useState(true);
  const [ragProdutos, setRagProdutos] = useState(true);
  const [ragGeral, setRagGeral] = useState(true);
  const [incluirRelatorio, setIncluirRelatorio] = useState(true);
  const [textoReescrito, setTextoReescrito] = useState('');
  const [relatorio, setRelatorio] = useState('');
  const [statsRags, setStatsRags] = useState<Record<string, number>>({});
  const [subTab, setSubTab] = useState<SubTab>('reescrito');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleRevisar = async () => {
    if (!texto.trim()) return;
    setError(''); setLoading(true);
    try {
      const ctx = selectedAgent ? buildContexto(selectedAgent, segmentos) : '';
      const res = await contentApi.revisaoTecnica({
        texto,
        rags: { taxonomia: ragTaxonomia, epidemiologia: ragEpidemiologia, produtos: ragProdutos, geral: ragGeral },
        limite: limiteDocumentos, contextoAgente: ctx, incluirRelatorio,
      }) as { textoReescrito: string; relatorioMudancas: string; resultadosRags: Record<string, unknown[]> };
      setTextoReescrito(res.textoReescrito);
      setRelatorio(res.relatorioMudancas ?? '');
      const stats: Record<string, number> = {};
      for (const [k, v] of Object.entries(res.resultadosRags ?? {})) stats[k] = (v as unknown[]).length;
      setStatsRags(stats);
    } catch (e: unknown) { setError((e as Error).message); }
    finally { setLoading(false); }
  };

  const subTabs = [
    { id: 'reescrito' as SubTab, label: 'Texto Reescrito' },
    { id: 'relatorio' as SubTab, label: 'Relatório de Mudanças' },
    { id: 'analise' as SubTab, label: 'Análise RAGs' },
  ];

  return (
    <Card>
      <SectionHeader icon={<Wrench size={16} />} title="Revisão Técnica com RAGs" subtitle="Taxonomia · Epidemiologia · Produtos · Geral" />

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 mb-4">
        {/* Content input */}
        <div className="lg:col-span-2">
          <Textarea label="Conteúdo Original" value={texto} onChange={(e) => setTexto(e.target.value)} rows={13} placeholder="Cole aqui o conteúdo técnico agrícola para revisão…" />
        </div>

        {/* Config */}
        <div className="flex flex-col gap-3">
          <Select label="Tipo de Conteúdo" value={tipoConteudo} onChange={(e) => setTipoConteudo(e.target.value)}>
            {TIPOS.map((t) => <option key={t}>{t}</option>)}
          </Select>
          <Select label="Nível de Rigor" value={nivelRigor} onChange={(e) => setNivelRigor(e.target.value)}>
            {RIGOR.map((r) => <option key={r}>{r}</option>)}
          </Select>
          <div className="flex flex-col gap-1.5">
            <label className="text-xs font-semibold text-slate-400 uppercase tracking-wide">Docs por RAG: {limiteDocumentos}</label>
            <input type="range" min={3} max={20} value={limiteDocumentos} onChange={(e) => setLimiteDocumentos(+e.target.value)} className="accent-violet-500" />
          </div>
          <div className="rounded-xl border p-3 flex flex-col gap-2" style={{ borderColor: 'var(--border)', background: 'rgba(0,0,0,0.15)' }}>
            <p className="text-xs font-semibold text-slate-400 uppercase tracking-wide flex items-center gap-1.5">
              <Microscope size={11} /> RAGs Ativos
            </p>
            <Checkbox label="Taxonomia" checked={ragTaxonomia} onChange={setRagTaxonomia} />
            <Checkbox label="Epidemiologia" checked={ragEpidemiologia} onChange={setRagEpidemiologia} />
            <Checkbox label="Produtos" checked={ragProdutos} onChange={setRagProdutos} />
            <Checkbox label="Geral" checked={ragGeral} onChange={setRagGeral} />
          </div>
          <Checkbox label="Incluir relatório de mudanças" checked={incluirRelatorio} onChange={setIncluirRelatorio} />
        </div>
      </div>

      <Button onClick={handleRevisar} loading={loading} disabled={!texto.trim()} fullWidth>
        <Microscope size={14} /> Executar Revisão com RAGs
      </Button>

      {loading && <Spinner text="Executando pipeline de RAGs especializados…" />}
      {error && <div className="mt-3"><Alert type="error">{error}</Alert></div>}

      {textoReescrito && (
        <div className="mt-5">
          <SubTabs tabs={subTabs} active={subTab} onChange={setSubTab} />

          {subTab === 'reescrito' && <ResultBox content={textoReescrito} filename="texto_reescrito.txt" />}
          {subTab === 'relatorio' && (
            relatorio
              ? <ResultBox content={relatorio} filename="relatorio_mudancas.md" />
              : <p className="text-slate-500 text-sm text-center py-6">Nenhum relatório gerado.</p>
          )}
          {subTab === 'analise' && (
            <div>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-4">
                {Object.entries(statsRags).map(([k, v]) => (
                  <Stat key={k} label={`RAG ${k}`} value={v} />
                ))}
              </div>
              <div className="grid grid-cols-3 gap-3">
                <Stat label="Palavras Original" value={texto.split(' ').length} />
                <Stat label="Palavras Reescrito" value={textoReescrito.split(' ').length} />
                <Stat label="Diferença" value={`${textoReescrito.split(' ').length - texto.split(' ').length > 0 ? '+' : ''}${textoReescrito.split(' ').length - texto.split(' ').length}`} />
              </div>
            </div>
          )}

          {/* Ajustes incrementais */}
          <div className="mt-4 pt-4 border-t" style={{ borderColor: 'var(--border)' }}>
            <p className="text-xs font-semibold text-slate-400 uppercase tracking-wide mb-2">Ajustes Incrementais</p>
            <div className="flex gap-2">
              <Textarea
                value={textoReescrito}
                onChange={(e) => setTextoReescrito(e.target.value)}
                rows={2}
                placeholder="Solicite ajustes na última revisão…"
                className="flex-1 hidden"
              />
            </div>
            <div className="flex gap-2 mt-2">
              <Button variant="secondary" onClick={() => navigator.clipboard.writeText(textoReescrito)} className="text-xs h-8">
                <RotateCcw size={12} /> Copiar texto
              </Button>
            </div>
          </div>
        </div>
      )}
    </Card>
  );
}
