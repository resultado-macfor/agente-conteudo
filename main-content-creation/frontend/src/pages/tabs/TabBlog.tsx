import { useState, useEffect } from 'react';
import { useStore } from '../../store/useStore';
import { buildContexto } from '../../utils/buildContexto';
import { blogApi } from '../../api/content';
import { Button, Card, Textarea, Input, Select, SectionHeader, Spinner, ResultBox, Alert, Checkbox } from '../../components/ui';
import { BookOpen } from 'lucide-react';


const TONS = ['Técnico-científico', 'Jornalístico', 'Educativo', 'Consultivo'];
const HEADINGS = ['H2', 'H3', 'H1'];

interface Versao { versao: number; conteudo: string; data: Date; descricao: string; }

export default function TabBlog() {
  const { selectedAgent, segmentos } = useStore();
  const [briefing, setBriefing] = useState('');
  const [tomVoz, setTomVoz] = useState(TONS[0]);
  const [numeroPalavras, setNumeroPalavras] = useState(1500);
  const [palavrasChave, setPalavrasChave] = useState('');
  const [palavrasPrimeiraLinha, setPalavrasPrimeiraLinha] = useState('');
  const [densidadePalavras, setDensidadePalavras] = useState(3);
  const [nivelHeading, setNivelHeading] = useState(HEADINGS[0]);
  const [usarPerplexity, setUsarPerplexity] = useState(true);
  const [conteudo, setConteudo] = useState('');
  const [versoes, setVersoes] = useState<Versao[]>([]);
  const [fontes, setFontes] = useState<string[]>([]);
  const [ajuste, setAjuste] = useState('');
  const [loading, setLoading] = useState(false);
  const [loadingAjuste, setLoadingAjuste] = useState(false);
  const [error, setError] = useState('');
  const [activeSubTab, setActiveSubTab] = useState<'conteudo' | 'versoes' | 'fontes' | 'historico'>('conteudo');
  const [historico, setHistorico] = useState<Array<{ _id: string; briefing: string; conteudo: string; createdAt: string }>>([]);

  const ctx = selectedAgent ? buildContexto(selectedAgent, segmentos) : '';

  useEffect(() => {
    blogApi.historico().then(setHistorico).catch(() => setHistorico([]));
  }, []);

  const handleGerar = async () => {
    if (!briefing.trim()) { setError('Descreva o conteúdo que deseja gerar.'); return; }
    setError(''); setLoading(true);
    try {
      let fontesWeb = '';
      if (usarPerplexity) {
        const perp = await blogApi.buscarFontes(briefing);
        fontesWeb = perp.resultado;
        setFontes(perp.fontes);
      }
      const res = await blogApi.gerar({
        briefing, contextoAgente: ctx, tomVoz, numeroPalavras,
        palavrasChave: palavrasChave.split(',').map((p) => p.trim()).filter(Boolean),
        palavrasPrimeiraLinha: palavrasPrimeiraLinha.split(',').map((p) => p.trim()).filter(Boolean),
        densidadePalavras, nivelHeading, fontesWeb,
      });
      setConteudo(res.conteudo);
      setVersoes([{ versao: 1, conteudo: res.conteudo, data: new Date(), descricao: 'Geração inicial' }]);
      setActiveSubTab('conteudo');
      blogApi.salvar({ briefing, conteudo: res.conteudo, fontes, configuracoes: { tomVoz, palavrasChave, usarPerplexity } })
        .then(() => blogApi.historico().then(setHistorico))
        .catch(() => null);
    } catch (e: unknown) { setError((e as Error).message); }
    finally { setLoading(false); }
  };

  const handleAjustar = async () => {
    if (!ajuste.trim() || !conteudo) return;
    setLoadingAjuste(true);
    try {
      const novaVersao: Versao = { versao: versoes.length + 1, conteudo, data: new Date(), descricao: `Ajuste: ${ajuste.slice(0, 50)}...` };
      const res = await blogApi.ajustar({ conteudoAtual: conteudo, briefingOriginal: briefing, ajuste });
      setConteudo(res.conteudo);
      setVersoes((prev) => [...prev, novaVersao]);
      setAjuste('');
    } catch (e: unknown) { setError((e as Error).message); }
    finally { setLoadingAjuste(false); }
  };

  return (
    <Card>
      <SectionHeader icon={<BookOpen size={16} />} title="Blog Inteligente — Geração Avançada" />
      <p className="text-sm text-slate-400 mb-4">Cole tudo o que você quer abordar em uma única caixa de texto. O sistema fará o resto.</p>

      <Textarea
        label="DESCREVA AQUI O CONTEÚDO QUE VOCÊ QUER GERAR:"
        value={briefing}
        onChange={(e) => setBriefing(e.target.value)}
        rows={8}
        placeholder={`Exemplo:\nTítulo: Manejo de nematoides na cultura da soja\nCultura: Soja\nProblema: Meloidogyne e Heterodera\nProdutos: NemaControl, Victrato\nObjetivo: Educar o produtor...\nPúblico: Produtores do Centro-Oeste\nPalavras-chave: manejo de nematoides, bionematicida\nNúmero de palavras: 1500`}
      />

      <details className="mt-3 mb-3">
        <summary className="text-xs font-medium text-violet-300 cursor-pointer hover:text-violet-200"> Configurações Avançadas (opcional)</summary>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mt-3 p-3 rounded-lg border" style={{ borderColor: 'rgba(167,139,250,0.2)', background: 'rgba(0,0,0,0.2)' }}>
          <Input label="Palavras-chave (separadas por vírgula):" value={palavrasChave} onChange={(e) => setPalavrasChave(e.target.value)} placeholder="manejo, bionematicida, soja" />
          <Input label="Palavras obrigatórias na 1ª linha:" value={palavrasPrimeiraLinha} onChange={(e) => setPalavrasPrimeiraLinha(e.target.value)} placeholder="nematoides, soja" />
          <Select label="Tom de voz:" value={tomVoz} onChange={(e) => setTomVoz(e.target.value)}>
            {TONS.map((t) => <option key={t}>{t}</option>)}
          </Select>
          <Select label="Heading do corpo:" value={nivelHeading} onChange={(e) => setNivelHeading(e.target.value)}>
            {HEADINGS.map((h) => <option key={h}>{h}</option>)}
          </Select>
          <div className="flex flex-col gap-1">
            <label className="text-xs font-medium text-violet-300">Nº de palavras: {numeroPalavras}</label>
            <input type="range" min={500} max={5000} step={100} value={numeroPalavras} onChange={(e) => setNumeroPalavras(+e.target.value)} className="accent-violet-500" />
          </div>
          <div className="flex flex-col gap-1">
            <label className="text-xs font-medium text-violet-300">Densidade KW: {densidadePalavras}%</label>
            <input type="range" min={1} max={10} value={densidadePalavras} onChange={(e) => setDensidadePalavras(+e.target.value)} className="accent-violet-500" />
          </div>
          <Checkbox label="Buscar informações atualizadas na web" checked={usarPerplexity} onChange={setUsarPerplexity} />
        </div>
      </details>

      <Button onClick={handleGerar} loading={loading} fullWidth>Gerar Conteúdo Blog</Button>
      {loading && <Spinner text="Processando briefing e gerando conteúdo..." />}
      {error && <Alert type="error" >{error}</Alert>}

      {conteudo && (
        <div className="mt-4">
          <div className="grid grid-cols-4 gap-2 mb-3">
            <div className="rounded-lg border p-2 text-center" style={{ borderColor: 'rgba(167,139,250,0.2)', background: 'rgba(0,0,0,0.2)' }}>
              <p className="text-base font-bold text-violet-300">{conteudo.split(' ').length}</p>
              <p className="text-xs text-slate-400">Palavras</p>
            </div>
            <div className="rounded-lg border p-2 text-center" style={{ borderColor: 'rgba(167,139,250,0.2)', background: 'rgba(0,0,0,0.2)' }}>
              <p className="text-base font-bold text-violet-300">{versoes.length}</p>
              <p className="text-xs text-slate-400">Versões</p>
            </div>
            <div className="rounded-lg border p-2 text-center" style={{ borderColor: 'rgba(167,139,250,0.2)', background: 'rgba(0,0,0,0.2)' }}>
              <p className="text-base font-bold text-violet-300">{tomVoz.split('-')[0]}</p>
              <p className="text-xs text-slate-400">Tom</p>
            </div>
            <div className="rounded-lg border p-2 text-center" style={{ borderColor: 'rgba(167,139,250,0.2)', background: 'rgba(0,0,0,0.2)' }}>
              <p className="text-base font-bold text-violet-300">{fontes.length > 0 ? '' : '—'}</p>
              <p className="text-xs text-slate-400">Fontes Web</p>
            </div>
          </div>

          <div className="flex gap-1 mb-3">
            {(['conteudo', 'versoes', 'fontes', 'historico'] as const).map((t) => (
              <button key={t} onClick={() => setActiveSubTab(t)}
                className="px-3 py-1.5 text-xs rounded-lg transition-colors"
                style={{ background: activeSubTab === t ? 'rgba(124,58,237,0.3)' : 'rgba(255,255,255,0.05)', color: activeSubTab === t ? '#c4b5fd' : '#94a3b8' }}>
                {t === 'conteudo' ? ' Conteúdo' : t === 'versoes' ? ' Versões' : t === 'fontes' ? ' Fontes' : ` Histórico (${historico.length})`}
              </button>
            ))}
          </div>

          {activeSubTab === 'conteudo' && <ResultBox content={conteudo} filename="blog.md" />}
          {activeSubTab === 'versoes' && (
            <div className="flex flex-col gap-2">
              {[...versoes].reverse().map((v) => (
                <div key={v.versao} className="rounded-lg border p-3" style={{ borderColor: 'rgba(167,139,250,0.2)', background: 'rgba(0,0,0,0.2)' }}>
                  <div className="flex justify-between items-center mb-1">
                    <span className="text-sm font-medium text-violet-300">Versão {v.versao} — {v.descricao}</span>
                    <span className="text-xs text-slate-500">{v.data.toLocaleString('pt-BR')}</span>
                  </div>
                  <Button variant="secondary" onClick={() => setConteudo(v.conteudo)} className="text-xs">Restaurar esta versão</Button>
                </div>
              ))}
            </div>
          )}
          {activeSubTab === 'fontes' && (
            <div className="rounded-lg border p-3 text-sm text-slate-300" style={{ borderColor: 'rgba(167,139,250,0.2)', background: 'rgba(0,0,0,0.2)' }}>
              {fontes.length > 0 ? fontes.map((f, i) => <p key={i} className="text-xs mb-1 text-slate-400">{i + 1}. {f}</p>) : <p className="text-slate-500">Nenhuma fonte capturada.</p>}
            </div>
          )}

          {activeSubTab === 'historico' && (
            <div className="flex flex-col gap-2">
              {historico.length === 0 && <p className="text-slate-500 text-sm">Nenhum post no histórico.</p>}
              {historico.map((post) => (
                <div key={post._id} className="rounded-lg border p-3" style={{ borderColor: 'rgba(167,139,250,0.2)', background: 'rgba(0,0,0,0.2)' }}>
                  <div className="flex justify-between items-start mb-1">
                    <p className="text-xs text-violet-300">{new Date(post.createdAt).toLocaleString('pt-BR')}</p>
                    <Button variant="secondary" className="text-xs" onClick={() => { setConteudo(post.conteudo); setActiveSubTab('conteudo'); }}>
                      Carregar
                    </Button>
                  </div>
                  <p className="text-xs text-slate-400 line-clamp-2">{post.briefing.slice(0, 120)}...</p>
                  <p className="text-xs text-slate-500 mt-1">{post.conteudo.split(' ').length} palavras</p>
                </div>
              ))}
            </div>
          )}

   
          <div className="mt-4 border-t pt-4" style={{ borderColor: 'rgba(167,139,250,0.2)' }}>
            <p className="text-sm font-medium text-violet-300 mb-2"> Ajustar Conteúdo</p>
            <div className="flex gap-2">
              <Textarea value={ajuste} onChange={(e) => setAjuste(e.target.value)} rows={2} placeholder="Descreva os ajustes desejados..." className="flex-1" />
              <Button onClick={handleAjustar} loading={loadingAjuste} disabled={!ajuste.trim()} className="self-end"> Ajustar</Button>
            </div>
          </div>
        </div>
      )}
    </Card>
  );
}
