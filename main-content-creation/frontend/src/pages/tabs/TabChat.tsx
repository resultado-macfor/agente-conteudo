import { useState, useRef, useEffect } from 'react';
import { MessageSquare, Send, Trash2 } from 'lucide-react';
import { useStore } from '../../store/useStore';
import { chatApi } from '../../api/content';
import { agentsApi } from '../../api/agents';
import { Button, Card, SectionHeader, Spinner } from '../../components/ui';

export default function TabChat() {
  const { selectedAgent, segmentos, messages, addMessage, clearMessages, user } = useStore();
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  if (!selectedAgent) {
    return (
      <Card>
        <div className="flex flex-col items-center justify-center py-16 gap-3 text-center">
          <div className="w-12 h-12 rounded-2xl flex items-center justify-center" style={{ background: 'var(--violet-muted)', border: '1px solid rgba(139,92,246,0.2)' }}>
            <MessageSquare size={20} className="text-violet-400" />
          </div>
          <p className="text-slate-400 text-sm">Selecione um agente para iniciar.</p>
        </div>
      </Card>
    );
  }

  const handleSend = async () => {
    const msg = input.trim();
    if (!msg || loading) return;
    setInput('');
    addMessage({ role: 'user', content: msg });
    setLoading(true);
    try {
      const res = await chatApi.sendMessage({ agenteId: selectedAgent._id, mensagem: msg, historico: messages, segmentos });
      addMessage({ role: 'assistant', content: res.resposta });
      await agentsApi.salvarConversa(selectedAgent._id, [...messages, { role: 'assistant', content: res.resposta }], segmentos);
    } catch (e: unknown) {
      addMessage({ role: 'assistant', content: `Erro: ${(e as Error).message}` });
    } finally {
      setLoading(false);
    }
  };

  return (
    <Card className="flex flex-col" style={{ height: 'calc(100vh - 280px)', minHeight: 400 }}>
      <div className="flex items-center justify-between mb-4">
        <SectionHeader icon={<MessageSquare size={16} />} title={`Chat — ${selectedAgent.nome}`} subtitle={`${selectedAgent.categoria} · ${segmentos.length} segmentos ativos`} />
        {messages.length > 0 && (
          <Button variant="ghost" onClick={clearMessages} className="h-8 px-3 text-xs shrink-0 -mt-5">
            <Trash2 size={12} /> Limpar
          </Button>
        )}
      </div>

      <div className="flex-1 overflow-y-auto flex flex-col gap-3 pr-1 mb-4">
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full gap-2 text-center">
            <MessageSquare size={28} className="text-slate-700" />
            <p className="text-slate-600 text-sm">Nenhuma mensagem. Comece a conversar!</p>
          </div>
        )}
        {messages.map((m, i) => (
          <div key={i} className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div
              className="rounded-2xl px-4 py-3 max-w-[80%] text-sm leading-relaxed"
              style={
                m.role === 'user'
                  ? { background: 'linear-gradient(135deg,#4c1d95,#7c3aed)', color: '#f1f5f9' }
                  : { background: 'var(--surface-2)', color: '#cbd5e1', border: '1px solid var(--border)' }
              }
            >
              <p className="text-[10px] font-semibold mb-1.5 opacity-60">
                {m.role === 'user' ? (user ?? 'Você') : selectedAgent.nome}
              </p>
              <p className="whitespace-pre-wrap">{m.content}</p>
            </div>
          </div>
        ))}
        {loading && (
          <div className="flex justify-start">
            <div className="rounded-2xl px-4 py-3" style={{ background: 'var(--surface-2)', border: '1px solid var(--border)' }}>
              <Spinner text="Pensando…" />
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div className="flex gap-2 items-end">
        <textarea
          className="flex-1 rounded-xl border px-3.5 py-2.5 text-sm text-slate-200 placeholder-slate-600 outline-none transition-colors focus:border-violet-500/60 resize-none"
          style={{ background: 'rgba(255,255,255,0.03)', borderColor: 'var(--border)', minHeight: 44, maxHeight: 120 }}
          placeholder="Digite sua mensagem… (Enter para enviar)"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend(); } }}
          disabled={loading}
          rows={1}
        />
        <Button onClick={handleSend} loading={loading} disabled={!input.trim()} className="h-11 px-4 shrink-0">
          <Send size={15} />
        </Button>
      </div>
    </Card>
  );
}
