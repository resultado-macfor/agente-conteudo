import { useEffect, useState } from 'react';
import { ChevronDown, Check, X, RefreshCw, SlidersHorizontal, Link2 } from 'lucide-react';
import { agentsApi } from '../api/agents';
import { useStore } from '../store/useStore';
import type { Agent, Segmento } from '../types';
import { SEGMENTO_LABELS, ALL_SEGMENTOS } from '../types';
import { Button, Checkbox } from './ui';

export default function AgentSelector() {
  const { selectedAgent, setSelectedAgent, segmentos, setSegmentos, clearMessages } = useStore();
  const [agents, setAgents] = useState<Agent[]>([]);
  const [selectedId, setSelectedId] = useState<string>('');
  const [showSegmentos, setShowSegmentos] = useState(false);
  const [tempSegmentos, setTempSegmentos] = useState<Segmento[]>(segmentos);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    agentsApi.listar().then(setAgents).catch(console.error);
  }, []);

  const handleApply = async () => {
    if (!selectedId) return;
    setLoading(true);
    try {
      const agente = await agentsApi.obterComHeranca(selectedId);
      setSelectedAgent(agente);
      clearMessages();
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setSelectedAgent(null);
    setSelectedId('');
    clearMessages();
  };

  const handleApplySegmentos = () => {
    setSegmentos(tempSegmentos);
    setShowSegmentos(false);
  };

  const toggleSeg = (seg: Segmento) => {
    setTempSegmentos((prev) =>
      prev.includes(seg) ? prev.filter((s) => s !== seg) : [...prev, seg],
    );
  };

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center gap-2 flex-wrap">
        <div className="relative flex-1 min-w-48">
          <ChevronDown size={14} className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-500 pointer-events-none" />
          <select
            className="w-full appearance-none rounded-xl  px-3.5 py-2 text-sm text-white outline-none transition-colors focus:border-violet-500/60 pr-8"
            style={{ background: '#1d1d1f' }}
            value={selectedId}
            onChange={(e) => setSelectedId(e.target.value)}
          >
            <option value="">Selecione um agente…</option>
            {agents.map((a) => (
              <option key={a._id} value={a._id}>
                {a.nome} ({a.categoria}){a.agente_mae_id ? ' ↗' : ''}
              </option>
            ))}
          </select>
        </div>

        <Button onClick={handleApply} disabled={!selectedId} loading={loading} className="h-9 px-4">
          <Check size={14} />
          Aplicar
        </Button>

        {selectedAgent ? (
          <Button variant="danger" onClick={handleClear} className="h-9 px-3">
            <X size={14} />
          </Button>
        ) : (
          <button
            onClick={() => agentsApi.listar().then(setAgents)}
            className="h-9 px-3 rounded-xl text-slate-500 hover:text-violet-400 hover:bg-violet-500/10 transition-all flex items-center justify-center"
            title="Recarregar agentes"
          >
            <RefreshCw size={14} />
          </button>
        )}

        {selectedAgent && (
          <Button
            variant="ghost"
            onClick={() => { setTempSegmentos(segmentos); setShowSegmentos((v) => !v); }}
            className="h-9 px-3 gap-1.5"
          >
            <SlidersHorizontal size={14} />
            <span className="text-xs hidden sm:inline">Segmentos</span>
          </Button>
        )}
      </div>

      {selectedAgent && (
        <div
          className="flex items-center gap-2.5 px-3 py-2 rounded-xl text-xs flex-wrap"
          style={{ background: 'rgba(124,58,237,0.08)', border: '1px solid rgba(124,58,237,0.2)' }}
        >
          <div className="w-1.5 h-1.5 rounded-full bg-emerald-400 shrink-0" />
          <span className="font-medium text-violet-200">{selectedAgent.nome}</span>
          <span className="text-slate-500">·</span>
          <span className="text-slate-400">{selectedAgent.categoria}</span>
          {selectedAgent.agente_mae_id && (
            <>
              <span className="text-slate-500">·</span>
              <span className="text-slate-500 flex items-center gap-1"><Link2 size={10} /> Herança</span>
            </>
          )}
          <span className="text-slate-500 ml-auto hidden sm:inline">
            {segmentos.map((s) => SEGMENTO_LABELS[s]).join(' · ')}
          </span>
        </div>
      )}

      {showSegmentos && (
        <div
          className="rounded-xl border p-4"
          style={{ background: 'rgba(0,0,0,0.2)', borderColor: 'var(--border)' }}
        >
          <p className="text-xs font-semibold text-slate-400 uppercase tracking-wide mb-3">Segmentos Ativos</p>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-3">
            {ALL_SEGMENTOS.map((s) => (
              <Checkbox
                key={s}
                label={SEGMENTO_LABELS[s]}
                checked={tempSegmentos.includes(s)}
                onChange={() => toggleSeg(s)}
              />
            ))}
          </div>
          <Button onClick={handleApplySegmentos} className="h-8 px-4 text-xs">
            <Check size={12} />
            Aplicar
          </Button>
        </div>
      )}
    </div>
  );
}
