import { useState } from 'react';
import { CheckSquare, Search } from 'lucide-react';
import { useStore } from '../../store/useStore';
import { buildContexto } from '../../utils/buildContexto';
import { contentApi } from '../../api/content';
import { Button, Card, Textarea, SectionHeader, Spinner, ResultBox, Alert } from '../../components/ui';

export default function TabRevisaoOrtografica() {
  const { selectedAgent, segmentos } = useStore();
  const [texto, setTexto] = useState('');
  const [resultado, setResultado] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleRevisar = async () => {
    if (!texto.trim()) return;
    setError(''); setLoading(true);
    try {
      const ctx = selectedAgent ? buildContexto(selectedAgent, segmentos) : '';
      const res = await contentApi.revisaoOrtografica(texto, ctx);
      setResultado(res.resultado);
    } catch (e: unknown) { setError((e as Error).message); }
    finally { setLoading(false); }
  };

  return (
    <Card>
      <SectionHeader icon={<CheckSquare size={16} />} title="Revisão Ortográfica" subtitle="Correção de erros ortográficos e gramaticais" />
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="flex flex-col gap-3">
          <Textarea label="Texto para revisar" value={texto} onChange={(e) => setTexto(e.target.value)} rows={14} placeholder="Cole aqui o texto que deseja revisar…" />
          <Button onClick={handleRevisar} loading={loading} disabled={!texto.trim()} fullWidth>
            <Search size={14} /> Realizar Revisão
          </Button>
          {loading && <Spinner text="Revisando texto…" />}
          {error && <Alert type="error">{error}</Alert>}
        </div>
        <div className="flex flex-col gap-2">
          <p className="text-xs font-semibold text-slate-400 uppercase tracking-wide">Resultado</p>
          {resultado ? (
            <ResultBox content={resultado} filename="revisao_ortografica.txt" />
          ) : (
            <div className="rounded-xl border p-8 text-center text-slate-600 text-sm flex flex-col items-center gap-2" style={{ borderColor: 'var(--border)', background: 'rgba(0,0,0,0.15)' }}>
              <CheckSquare size={24} className="text-slate-700" />
              O resultado aparecerá aqui após a revisão.
            </div>
          )}
        </div>
      </div>
    </Card>
  );
}
