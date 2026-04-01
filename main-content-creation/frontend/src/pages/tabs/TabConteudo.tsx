import { useState, useEffect } from 'react';
import { Sparkles, Upload, Database, Mic, RotateCcw } from 'lucide-react';
import { useStore } from '../../store/useStore';
import { buildContexto } from '../../utils/buildContexto';
import { contentApi, filesApi, briefingsApi } from '../../api/content';
import { Button, Card, Textarea, Input, Select, SectionHeader, Spinner, ResultBox, Alert, Checkbox, Stat } from '../../components/ui';

const TIPOS = ['Post Social', 'Artigo Blog', 'Email Marketing', 'Landing Page', 'Script Vídeo', 'Relatório Técnico', 'Press Release', 'Newsletter', 'Case Study'];
const TONS = ['Formal', 'Informal', 'Persuasivo', 'Educativo', 'Inspirador', 'Técnico', 'Jornalístico'];
const NIVEIS = ['Resumido', 'Balanceado', 'Detalhado', 'Completo'];
const FORMATOS = ['Texto Simples', 'Markdown', 'HTML Básico'];

interface BriefingDB { _id: string; nome_projeto: string; tipo: string; conteudo: string; createdAt: string; }

export default function TabConteudo() {
  const { selectedAgent, segmentos } = useStore();
  const [arquivos, setArquivos] = useState<File[]>([]);
  const [textosArquivos, setTextosArquivos] = useState('');
  const [briefingManual, setBriefingManual] = useState('');
  const [briefingsDB, setBriefingsDB] = useState<BriefingDB[]>([]);
  const [briefingDBSelecionado, setBriefingDBSelecionado] = useState('');
  const [instrucoes, setInstrucoes] = useState('');
  const [tipoConteudo, setTipoConteudo] = useState(TIPOS[0]);
  const [tomVoz, setTomVoz] = useState(TONS[0]);
  const [palavrasChave, setPalavrasChave] = useState('');
  const [numeroPalavras, setNumeroPalavras] = useState(800);
  const [nivelDetalhe, setNivelDetalhe] = useState(NIVEIS[1]);
  const [incluirCta, setIncluirCta] = useState(true);
  const [formatoSaida, setFormatoSaida] = useState(FORMATOS[1]);
  const [resultado, setResultado] = useState('');
  const [loading, setLoading] = useState(false);
  const [loadingFiles, setLoadingFiles] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    briefingsApi.listar().then(setBriefingsDB).catch(() => setBriefingsDB([]));
  }, []);

  const handleFilesChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files ?? []);
    if (!files.length) return;
    setArquivos(files);
    setLoadingFiles(true);
    try {
      const results = await filesApi.extractText(files);
      const texto = results.map((r) => `\n\n--- CONTEÚDO DE ${r.name.toUpperCase()} ---\n${r.text}`).join('');
      setTextosArquivos(texto);
    } catch (e: unknown) {
      setError('Erro ao extrair texto: ' + (e as Error).message);
    } finally {
      setLoadingFiles(false);
    }
  };

  const handleGerar = async () => {
    const briefingDB = briefingsDB.find((b) => b._id === briefingDBSelecionado);
    if (!textosArquivos && !briefingManual && !briefingDB) {
      setError('Forneça ao menos uma fonte de conteúdo.'); return;
    }
    setError(''); setLoading(true);
    try {
      const fontes = [];
      if (textosArquivos) fontes.push('### CONTEÚDO DOS ARQUIVOS:\n' + textosArquivos);
      if (briefingManual) fontes.push('### BRIEFING MANUAL:\n' + briefingManual);
      if (briefingDB) fontes.push('### BRIEFING DO BANCO:\n' + briefingDB.conteudo);
      const ctx = selectedAgent ? buildContexto(selectedAgent, segmentos) : '';
      const res = await contentApi.gerar({
        contextoAgente: ctx, tipoConteudo, tomVoz, palavrasChave,
        numeroPalavras, nivelDetalhe, incluirCta, formatoSaida,
        instrucoes, fontesTexto: '## FONTES DE CONTEÚDO COMBINADAS:\n\n' + fontes.join('\n\n'),
      });
      setResultado(res.conteudo);
    } catch (e: unknown) { setError((e as Error).message); }
    finally { setLoading(false); }
  };

  const fontesAtivas = (arquivos.length > 0 ? 1 : 0) + (briefingManual ? 1 : 0) + (briefingDBSelecionado ? 1 : 0);

  return (
    <Card>
      <SectionHeader icon={<Sparkles size={16} />} title="Geração de Conteúdo com Múltiplos Insumos" subtitle="Combine arquivos, briefings e transcrições para gerar conteúdo" />

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
        <div className="lg:col-span-2 flex flex-col gap-4">
          <div
            className="rounded-xl border-2 border-dashed p-4 transition-colors"
            style={{ borderColor: 'rgba(139,92,246,0.2)', background: 'rgba(139,92,246,0.03)' }}
          >
            <div className="flex items-center gap-2 mb-3">
              <Upload size={14} className="text-violet-400" />
              <span className="text-xs font-semibold text-slate-400 uppercase tracking-wide">Upload de Arquivos</span>
              <span className="text-xs text-slate-600">PDF, TXT, PPTX, DOCX</span>
            </div>
            <input
              type="file" multiple accept=".pdf,.txt,.pptx,.ppt,.docx,.doc"
              onChange={handleFilesChange}
              className="text-xs text-slate-500 file:mr-3 file:py-1.5 file:px-3 file:rounded-lg file:border-0 file:text-xs file:font-medium file:text-white file:cursor-pointer"
              style={{ ['--file-selector-button-bg' as string]: 'linear-gradient(135deg,#4c1d95,#7c3aed)' }}
            />
            {loadingFiles && <Spinner text="Extraindo texto dos arquivos…" />}
            {arquivos.length > 0 && !loadingFiles && (
              <p className="text-xs text-emerald-400 mt-2">✓ {arquivos.length} arquivo(s) processado(s)</p>
            )}
          </div>
          <div>
            <div className="flex items-center gap-2 mb-1.5">
              <Database size={13} className="text-violet-400" />
              <label className="text-xs font-semibold text-slate-400 uppercase tracking-wide">Briefing do Banco</label>
            </div>
            {briefingsDB.length > 0 ? (
              <select
                className="w-full rounded-xl px-5.5 py-2.5 text-sm text-slate-200 outline-none focus:border-violet-500/60"
                style={{ background: '#1d1d1f'}}
                value={briefingDBSelecionado}
                onChange={(e) => setBriefingDBSelecionado(e.target.value)}
              >
                <option value=""> Nenhum briefing do banco </option>
                {briefingsDB.map((b) => (
                  <option key={b._id} value={b._id}>
                    {b.nome_projeto} ({b.tipo}) · {new Date(b.createdAt).toLocaleDateString('pt-BR')}
                  </option>
                ))}
              </select>
            ) : (
              <p className="text-xs text-slate-600 py-2">Nenhum briefing no banco de dados.</p>
            )}
          </div>

          <Textarea label="Briefing Manual" value={briefingManual} onChange={(e) => setBriefingManual(e.target.value)} rows={5} placeholder="Cole aqui o briefing completo…" />

          <div>
            <div className="flex items-center gap-2 mb-1.5">
              <Mic size={13} className="text-violet-400" />
              <span className="text-xs font-semibold text-slate-400 uppercase tracking-wide">Transcrição de Mídia</span>
              <span className="text-xs text-slate-600">— em breve</span>
            </div>
          </div>

          <Textarea label="Instruções Específicas" value={instrucoes} onChange={(e) => setInstrucoes(e.target.value)} rows={3} placeholder="- Focar nos benefícios&#10;- Incluir estatísticas quando possível…" />
        </div>

        {/* Config column */}
        <div className="flex flex-col gap-3">
          <Select label="Tipo de Conteúdo" value={tipoConteudo} onChange={(e) => setTipoConteudo(e.target.value)}>
            {TIPOS.map((t) => <option key={t}>{t}</option>)}
          </Select>
          <Select label="Tom de Voz" value={tomVoz} onChange={(e) => setTomVoz(e.target.value)}>
            {TONS.map((t) => <option key={t}>{t}</option>)}
          </Select>
          <Input label="Palavras-chave" value={palavrasChave} onChange={(e) => setPalavrasChave(e.target.value)} placeholder="kw1, kw2, kw3…" />
          <div className="flex flex-col gap-1.5">
            <label className="text-xs font-semibold text-slate-400 uppercase tracking-wide">Nº de Palavras: {numeroPalavras}</label>
            <input type="range" min={100} max={3000} step={50} value={numeroPalavras} onChange={(e) => setNumeroPalavras(+e.target.value)} className="accent-violet-500" />
          </div>
          <Select label="Nível de Detalhe" value={nivelDetalhe} onChange={(e) => setNivelDetalhe(e.target.value)}>
            {NIVEIS.map((n) => <option key={n}>{n}</option>)}
          </Select>
          <Select label="Formato de Saída" value={formatoSaida} onChange={(e) => setFormatoSaida(e.target.value)}>
            {FORMATOS.map((f) => <option key={f}>{f}</option>)}
          </Select>
          <Checkbox label="Incluir Call-to-Action" checked={incluirCta} onChange={setIncluirCta} />

          <div className="rounded-xl border p-3 mt-1" style={{ borderColor: 'var(--border)', background: 'rgba(0,0,0,0.15)' }}>
            <p className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-2">Fontes Ativas</p>
            {[
              { label: `${arquivos.length} arquivo(s)`, active: arquivos.length > 0 },
              { label: 'Briefing do banco', active: !!briefingDBSelecionado },
              { label: 'Briefing manual', active: !!briefingManual },
            ].map(({ label, active }) => (
              <div key={label} className={`text-xs flex items-center gap-2 mb-1 ${active ? 'text-emerald-400' : 'text-slate-700'}`}>
                <span>{active ? '●' : '○'}</span> {label}
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="mt-5">
        <Button onClick={handleGerar} loading={loading} fullWidth disabled={fontesAtivas === 0}>
          <Sparkles size={14} /> Gerar Conteúdo com Todos os Insumos
        </Button>
      </div>

      {loading && <Spinner text="Processando insumos e gerando conteúdo…" />}
      {error && <div className="mt-3"><Alert type="error">{error}</Alert></div>}

      {resultado && (
        <div className="mt-5">
          <div className="grid grid-cols-3 gap-3 mb-4">
            <Stat label="Palavras" value={resultado.split(' ').length} />
            <Stat label="Arquivos" value={arquivos.length} />
            <Stat label="Fontes" value={fontesAtivas} />
          </div>
          <ResultBox content={resultado} filename="conteudo_gerado.md" />
          <div className="mt-3 flex justify-end">
            <Button variant="ghost" onClick={() => { setResultado(''); setArquivos([]); setTextosArquivos(''); setBriefingManual(''); setBriefingDBSelecionado(''); }} className="text-xs">
              <RotateCcw size={12} /> Nova geração
            </Button>
          </div>
        </div>
      )}
    </Card>
  );
}
