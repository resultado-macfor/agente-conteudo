import { useState } from 'react';
import { Calendar, Sparkles } from 'lucide-react';
import { useStore } from '../../store/useStore';
import { buildContexto } from '../../utils/buildContexto';
import { calendarApi } from '../../api/content';
import { Button, Card, Input, Select, SectionHeader, Spinner, ResultBox, Alert } from '../../components/ui';

const CULTURAS = ['Soja', 'Milho', 'Algodão', 'Cana-de-açúcar', 'Trigo', 'Café', 'Citrus', 'Outra'];
const ESTADOS = ['Mato Grosso', 'Mato Grosso do Sul', 'Goiás', 'Paraná', 'São Paulo', 'Minas Gerais', 'Bahia', 'Rio Grande do Sul', 'Tocantins', 'Maranhão', 'Outro'];

export default function TabCalendario() {
  const { selectedAgent, segmentos } = useStore();
  const [cultura, setCultura] = useState('Soja');
  const [estado, setEstado] = useState('Mato Grosso');
  const [periodo, setPeriodo] = useState('');
  const [temas, setTemas] = useState('');
  const [resultado, setResultado] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleGerar = async () => {
    if (!periodo.trim()) { setError('Informe o período.'); return; }
    setError(''); setLoading(true);
    try {
      const ctx = selectedAgent ? buildContexto(selectedAgent, segmentos) : '';
      const res = await calendarApi.gerar({
        cultura, estado, periodo,
        temas: temas.split(',').map((t) => t.trim()).filter(Boolean),
        contextoAgente: ctx,
      });
      setResultado(res.calendario);
    } catch (e: unknown) { setError((e as Error).message); }
    finally { setLoading(false); }
  };

  return (
    <Card>
      <SectionHeader icon={<Calendar size={16} />} title="Criadora de Calendário" subtitle="Gere calendários de conteúdo alinhados ao calendário agrícola" />

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-4">
        <Select label="Cultura" value={cultura} onChange={(e) => setCultura(e.target.value)}>
          {CULTURAS.map((c) => <option key={c}>{c}</option>)}
        </Select>
        <Select label="Estado / Região" value={estado} onChange={(e) => setEstado(e.target.value)}>
          {ESTADOS.map((e) => <option key={e}>{e}</option>)}
        </Select>
        <Input label="Período (ex: Jan–Jun 2025)" value={periodo} onChange={(e) => setPeriodo(e.target.value)} placeholder="Jan–Jun 2025" />
        <Input label="Temas (separados por vírgula)" value={temas} onChange={(e) => setTemas(e.target.value)} placeholder="Doenças foliares, Fungicidas, Plantio…" />
      </div>

      <Button onClick={handleGerar} loading={loading} fullWidth>
        <Sparkles size={14} /> Gerar Calendário
      </Button>

      {loading && <Spinner text="Gerando calendário…" />}
      {error && <div className="mt-3"><Alert type="error">{error}</Alert></div>}
      {resultado && <div className="mt-4"><ResultBox content={resultado} filename="calendario.md" /></div>}
    </Card>
  );
}
