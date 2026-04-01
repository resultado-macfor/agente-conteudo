import { useState, useEffect } from 'react';
import { Pencil, Trash2, Settings, Link2, Check } from 'lucide-react';
import { useStore } from '../../store/useStore';
import { agentsApi } from '../../api/agents';
import type { Agent } from '../../types';
import { Button, Card, Input, Select, Textarea, SectionHeader, Alert, Checkbox, SubTabs } from '../../components/ui';

const CATEGORIAS = ['Social', 'SEO', 'Conteúdo'];
const ELEMENTOS = ['system_prompt', 'base_conhecimento', 'comments', 'planejamento'] as const;
const ELEM_LABELS: Record<string, string> = {
  system_prompt: 'System Prompt',
  base_conhecimento: 'Brand Guidelines',
  comments: 'Comentários',
  planejamento: 'Planejamento',
};

function AgentForm({ onSuccess, editAgent }: { onSuccess: () => void; editAgent?: Agent }) {
  useStore();
  const [nome, setNome] = useState(editAgent?.nome ?? '');
  const [categoria, setCategoria] = useState(editAgent?.categoria ?? 'Social');
  const [systemPrompt, setSystemPrompt] = useState(editAgent?.system_prompt ?? '');
  const [baseConhecimento, setBaseConhecimento] = useState(editAgent?.base_conhecimento ?? '');
  const [comments, setComments] = useState(editAgent?.comments ?? '');
  const [planejamento, setPlanejamento] = useState(editAgent?.planejamento ?? '');
  const [criarComoFilho, setCriarComoFilho] = useState(!!editAgent?.agente_mae_id);
  const [agenteMaeId, setAgenteMaeId] = useState(editAgent?.agente_mae_id ?? '');
  const [herdarElementos, setHerdarElementos] = useState<string[]>(editAgent?.herdar_elementos ?? []);
  const [agentesDisponiveis, setAgentesDisponiveis] = useState<Agent[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');

  useEffect(() => {
    agentsApi.listarParaHeranca(editAgent?._id).then(setAgentesDisponiveis).catch(console.error);
  }, [editAgent?._id]);

  const toggleElem = (el: string) =>
    setHerdarElementos((prev) => prev.includes(el) ? prev.filter((e) => e !== el) : [...prev, el]);

  const handleSubmit = async () => {
    if (!nome.trim()) { setError('Nome é obrigatório.'); return; }
    setError(''); setLoading(true);
    try {
      const data = {
        nome, system_prompt: systemPrompt, base_conhecimento: baseConhecimento,
        comments, planejamento, categoria,
        agente_mae_id: criarComoFilho && agenteMaeId ? agenteMaeId : null,
        herdar_elementos: criarComoFilho ? herdarElementos : [],
      };
      if (editAgent) {
        await agentsApi.atualizar(editAgent._id, data);
        setSuccess(`Agente '${nome}' atualizado com sucesso!`);
      } else {
        await agentsApi.criar(data);
        setSuccess(`Agente '${nome}' criado na categoria ${categoria}!`);
        setNome(''); setSystemPrompt(''); setBaseConhecimento(''); setComments(''); setPlanejamento('');
      }
      onSuccess();
    } catch (e: unknown) { setError((e as Error).message); }
    finally { setLoading(false); }
  };

  return (
    <div className="flex flex-col gap-4">
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        <Input label="Nome do Agente" value={nome} onChange={(e) => setNome(e.target.value)} placeholder="Ex: Agente Blog Soja" />
        <Select label="Categoria" value={categoria} onChange={(e) => setCategoria(e.target.value)}>
          {CATEGORIAS.map((c) => <option key={c}>{c}</option>)}
        </Select>
      </div>

      <Checkbox label="Criar como agente filho (herdar elementos de um agente mãe)" checked={criarComoFilho} onChange={setCriarComoFilho} />

      {criarComoFilho && agentesDisponiveis.length > 0 && (
        <div className="rounded-xl border p-4 flex flex-col gap-3" style={{ borderColor: 'var(--border)', background: 'rgba(0,0,0,0.15)' }}>
          <div className="flex items-center gap-2 text-xs font-semibold text-slate-400 uppercase tracking-wide">
            <Link2 size={12} /> Herança
          </div>
          <Select label="Agente Mãe" value={agenteMaeId} onChange={(e) => setAgenteMaeId(e.target.value)}>
            <option value="">-- Selecione --</option>
            {agentesDisponiveis.map((a) => <option key={a._id} value={a._id}>{a.nome} ({a.categoria})</option>)}
          </Select>
          <div>
            <p className="text-xs font-semibold text-slate-400 uppercase tracking-wide mb-2">Elementos para Herdar</p>
            <div className="grid grid-cols-2 gap-2">
              {ELEMENTOS.map((el) => <Checkbox key={el} label={ELEM_LABELS[el]} checked={herdarElementos.includes(el)} onChange={() => toggleElem(el)} />)}
            </div>
          </div>
        </div>
      )}

      <Textarea label="System Prompt" value={systemPrompt} onChange={(e) => setSystemPrompt(e.target.value)} rows={4} placeholder="Você é um assistente especializado em..." />
      <Textarea label="Brand Guidelines" value={baseConhecimento} onChange={(e) => setBaseConhecimento(e.target.value)} rows={5} placeholder="Cole aqui informações, diretrizes, dados da marca..." />
      <Textarea label="Comentários do Cliente" value={comments} onChange={(e) => setComments(e.target.value)} rows={3} placeholder="Feedbacks e observações do cliente..." />
      <Textarea label="Planejamento" value={planejamento} onChange={(e) => setPlanejamento(e.target.value)} rows={3} placeholder="Estratégias, cronogramas..." />

      {error && <Alert type="error">{error}</Alert>}
      {success && <Alert type="success">{success}</Alert>}

      <Button onClick={handleSubmit} loading={loading} fullWidth>
        <Check size={15} />
        {editAgent ? 'Atualizar Agente' : 'Criar Agente'}
      </Button>
    </div>
  );
}

type SubTab = 'criar' | 'editar' | 'gerenciar';

export default function TabGerenciamento() {
  const { user } = useStore();
  const [activeTab, setActiveTab] = useState<SubTab>('criar');
  const [agents, setAgents] = useState<Agent[]>([]);
  const [editTarget, setEditTarget] = useState<Agent | undefined>();
  const [categoria, setCategoria] = useState('Todos');
  const [loadingDelete, setLoadingDelete] = useState<string | null>(null);
  const [msg, setMsg] = useState('');

  const loadAgents = () => agentsApi.listar().then(setAgents).catch(console.error);
  useEffect(() => { loadAgents(); }, []);

  const filtrados = categoria === 'Todos' ? agents : agents.filter((a) => a.categoria === categoria);

  const handleDesativar = async (id: string, nome: string) => {
    setLoadingDelete(id);
    try {
      await agentsApi.desativar(id);
      setMsg(`Agente '${nome}' desativado.`);
      loadAgents();
    } catch (e: unknown) { setMsg('Erro: ' + (e as Error).message); }
    finally { setLoadingDelete(null); }
  };

  const handleEditar = (a: Agent) => { setEditTarget(a); setActiveTab('editar'); };

  const subTabs = [
    { id: 'criar' as SubTab, label: 'Criar' },
    { id: 'editar' as SubTab, label: 'Editar' },
    { id: 'gerenciar' as SubTab, label: 'Gerenciar' },
  ];

  return (
    <Card>
      <SectionHeader icon={<Settings size={16} />} title="Gerenciamento de Agentes" subtitle={user === 'admin' ? 'Modo administrador — todos os agentes visíveis' : `Agentes de ${user}`} />

      <SubTabs tabs={subTabs} active={activeTab} onChange={setActiveTab} />

      {activeTab === 'criar' && <AgentForm onSuccess={loadAgents} />}

      {activeTab === 'editar' && (
        <div className="flex flex-col gap-4">
          <Select label="Agente para editar" value={editTarget?._id ?? ''} onChange={(e) => {
            const a = agents.find((ag) => ag._id === e.target.value);
            setEditTarget(a);
          }}>
            <option value="">-- Selecione --</option>
            {agents.map((a) => <option key={a._id} value={a._id}>{a.nome} ({a.categoria})</option>)}
          </Select>
          {editTarget && <AgentForm key={editTarget._id} editAgent={editTarget} onSuccess={loadAgents} />}
        </div>
      )}

      {activeTab === 'gerenciar' && (
        <div>
          {msg && <div className="mb-3"><Alert type="info">{msg}</Alert></div>}

          {/* Category filter */}
          <div className="flex gap-1.5 mb-4 flex-wrap">
            {['Todos', ...CATEGORIAS].map((c) => (
              <button
                key={c}
                onClick={() => setCategoria(c)}
                className="px-3 py-1.5 text-xs font-medium rounded-lg transition-all"
                style={{
                  background: categoria === c ? 'linear-gradient(135deg,#4c1d95,#7c3aed)' : 'rgba(255,255,255,0.04)',
                  color: categoria === c ? '#fff' : '#64748b',
                  border: `1px solid ${categoria === c ? 'transparent' : 'var(--border)'}`,
                }}
              >
                {c}
              </button>
            ))}
          </div>

          <div className="flex flex-col gap-2">
            {filtrados.length === 0 && (
              <p className="text-slate-500 text-sm text-center py-8">Nenhum agente encontrado.</p>
            )}
            {filtrados.map((a) => (
              <div
                key={a._id}
                className="rounded-xl border p-4 flex items-start justify-between gap-3"
                style={{ borderColor: 'var(--border)', background: 'rgba(0,0,0,0.12)' }}
              >
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-1 flex-wrap">
                    <span className="text-sm font-semibold text-slate-200">{a.nome}</span>
                    <span
                      className="text-xs px-2 py-0.5 rounded-md font-medium"
                      style={{ background: 'var(--violet-muted)', color: 'var(--violet-light)', border: '1px solid rgba(139,92,246,0.2)' }}
                    >
                      {a.categoria}
                    </span>
                    {a.agente_mae_id && (
                      <span className="text-xs text-slate-500 flex items-center gap-1"><Link2 size={10} /> Herda</span>
                    )}
                  </div>
                  <p className="text-xs text-slate-500">{a.criado_por} · {a.createdAt ? new Date(a.createdAt).toLocaleDateString('pt-BR') : ''}</p>
                  {a.system_prompt && (
                    <p className="text-xs text-slate-600 mt-1.5 line-clamp-2">{a.system_prompt.slice(0, 120)}…</p>
                  )}
                </div>
                <div className="flex gap-2 shrink-0">
                  <Button variant="ghost" onClick={() => handleEditar(a)} className="h-8 px-3 text-xs">
                    <Pencil size={12} />
                  </Button>
                  <Button variant="danger" onClick={() => handleDesativar(a._id, a.nome)} loading={loadingDelete === a._id} className="h-8 px-3 text-xs">
                    <Trash2 size={12} />
                  </Button>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </Card>
  );
}
