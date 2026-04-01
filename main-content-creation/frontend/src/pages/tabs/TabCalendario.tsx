import { useState } from 'react';
import { Calendar, Sparkles, FileDown, AlertCircle } from 'lucide-react';
import { useStore } from '../../store/useStore';
import { buildContexto } from '../../utils/buildContexto';
import { calendarApi } from '../../api/content';
import { Button, Card, Input, Textarea, SectionHeader, Spinner, Alert, Checkbox, SubTabs } from '../../components/ui';

interface ProdutoDirecional { produtos: string[]; culturas: string[]; tema: string; }

function parseProdutos(texto: string): ProdutoDirecional[] {
  return texto
    .split('\n')
    .map((l) => l.trim())
    .filter((l) => l && l.includes(' - '))
    .flatMap((l) => {
      const partes = l.split(' - ');
      if (partes.length < 3) return [];
      return [{
        produtos: partes[0].split(/\s+e\s+|,\s*/).map((p) => p.trim()).filter(Boolean),
        culturas: partes[1].split(/\s+e\s+|,\s*/).map((c) => c.trim()).filter(Boolean),
        tema: partes.slice(2).join(' - ').trim(),
      }];
    });
}

type ResultTab = 'csv' | 'xlsx';

export default function TabCalendario() {
  const { selectedAgent, segmentos } = useStore();

  // Config geral
  const [mesAno, setMesAno] = useState('FEVEREIRO 2026');
  const [dataInicio, setDataInicio] = useState('2026-02-01');
  const [dataFim, setDataFim] = useState('2026-02-28');

  // Culturas
  const [culturas, setCulturas] = useState('Soja, Milho, Cana-de-açúcar, Algodão, Soja e Milho, Soja e Cana');

  // Frequência
  const [diasCom1Pauta, setDiasCom1Pauta] = useState(5);
  const [diasCom2Pautas, setDiasCom2Pautas] = useState(15);
  const [diasCom3Pautas, setDiasCom3Pautas] = useState(3);

  // Produtos e direcionais
  const [produtosDirecionaisTexto, setProdutosDirecionaisTexto] = useState(
    `Verdavis, Megafol e Victrato - Soja e Milho - Tecnologia para feira
Elestal Neo - Soja - Controle de mosca-branca
Fortenza - Milho - Seedcare para cigarrinha
YieldOn - Soja - Bioativador para pegamento
Miravis - Soja - Fungicida para ferrugem
Victrato - Cana - Nematicida para cana-soca
Victrato pelo Brasil - Soja e Cana - Ação nacional`,
  );

  // Evento/feira
  const [semanaFeirasInicio, setSemanaFeirasInicio] = useState('2026-02-09');
  const [semanaFeirasFim, setSemanaFeirasFim] = useState('2026-02-13');
  const [produtosPrioritariosFeira, setProdutosPrioritariosFeira] = useState('Verdavis, Megafol, Victrato');

  // Pauta recorrente
  const [pautaRecorrenteTexto, setPautaRecorrenteTexto] = useState('Victrato pelo Brasil');
  const [pautaRecorrenteDias, setPautaRecorrenteDias] = useState<string[]>(['Terça', 'Quinta']);

  // Contexto
  const [contextoMensal, setContextoMensal] = useState(`FEVEREIRO 2026:
- Soja: colheita no centro-sul
- Milho: plantio da safrinha
- Cana: crescimento vegetativo
- Evento: Feira Nacional do Agronegócio (09-13/02)
- Foco: Verdavis, Megafol, Victrato na feira
- Pauta fixa: Victrato pelo Brasil (terças e quintas)`);

  // Controles
  const [evitarConsecutivosSemPautas, setEvitarConsecutivosSemPautas] = useState(true);
  const [maxRepeticoesTema, setMaxRepeticoesTema] = useState(2);

  // Estado
  const [calendarioCSV, setCalendarioCSV] = useState('');
  const [loading, setLoading] = useState(false);
  const [loadingXlsx, setLoadingXlsx] = useState(false);
  const [error, setError] = useState('');
  const [resultTab, setResultTab] = useState<ResultTab>('csv');

  const deltaDias = Math.max(0,
    Math.floor((new Date(dataFim).getTime() - new Date(dataInicio).getTime()) / 86400000) + 1
  );
  const diasSemPautas = deltaDias - diasCom1Pauta - diasCom2Pautas - diasCom3Pautas;
  const totalExcede = diasSemPautas < 0;

  const DIAS_SEMANA = ['Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta', 'Sábado', 'Domingo'];
  const toggleDia = (dia: string) => {
    setPautaRecorrenteDias((prev) =>
      prev.includes(dia) ? prev.filter((d) => d !== dia) : [...prev, dia]
    );
  };

  const handleGerar = async () => {
    if (!dataInicio || !dataFim || new Date(dataInicio) >= new Date(dataFim)) {
      setError('Data início deve ser anterior à data fim.'); return;
    }
    if (totalExcede) { setError('Total de dias com pautas excede o período.'); return; }
    setError(''); setLoading(true);
    try {
      const ctx = selectedAgent ? buildContexto(selectedAgent, segmentos) : '';
      const res = await calendarApi.gerar({
        contextoAgente: ctx,
        mesAno,
        dataInicio,
        dataFim,
        culturas: culturas.split(',').map((c) => c.trim()).filter(Boolean),
        diasCom1Pauta,
        diasCom2Pautas,
        diasCom3Pautas,
        produtosDirecionais: parseProdutos(produtosDirecionaisTexto),
        semanaFeirasInicio,
        semanaFeirasFim,
        produtosPrioritariosFeira,
        pautaRecorrenteTexto,
        pautaRecorrenteDias,
        contextoMensal,
        evitarConsecutivosSemPautas,
        maxRepeticoesTema,
      });
      setCalendarioCSV(res.calendario);
    } catch (e: unknown) { setError((e as Error).message); }
    finally { setLoading(false); }
  };

  const handleDownloadCsv = () => {
    const a = Object.assign(document.createElement('a'), {
      href: URL.createObjectURL(new Blob([calendarioCSV], { type: 'text/csv;charset=utf-8;' })),
      download: `calendario_${mesAno.replace(/\s+/g, '_').toLowerCase()}.csv`,
    });
    a.click();
  };

  const handleDownloadXlsx = async () => {
    setLoadingXlsx(true);
    try { await calendarApi.gerarXlsx(calendarioCSV, mesAno); }
    catch (e: unknown) { setError('Erro ao gerar XLSX: ' + (e as Error).message); }
    finally { setLoadingXlsx(false); }
  };

  const resultTabs = [
    { id: 'csv' as ResultTab, label: 'CSV' },
    { id: 'xlsx' as ResultTab, label: 'XLSX' },
  ];

  return (
    <Card>
      <SectionHeader icon={<Calendar size={16} />} title="Criadora de Calendário" subtitle="Gere calendários editoriais alinhados ao calendário agrícola" />

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
        {/* Coluna esquerda */}
        <div className="lg:col-span-2 flex flex-col gap-4">

          {/* Período */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            <Input label="Mês / Ano" value={mesAno} onChange={(e) => setMesAno(e.target.value)} placeholder="FEVEREIRO 2026" />
            <div className="flex flex-col">
              <label className="text-xs font-semibold text-slate-400 uppercase tracking-widest mb-1.5">Data início</label>
              <input type="date" value={dataInicio} onChange={(e) => setDataInicio(e.target.value)}
                className="rounded-xl px-4 py-3 text-sm text-slate-200 outline-none transition-colors focus:border-violet-500/60 focus:ring-2 focus:ring-violet-500/20"
                style={{ background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(139,92,246,0.18)' }} />
            </div>
            <div className="flex flex-col">
              <label className="text-xs font-semibold text-slate-400 uppercase tracking-widest mb-1.5">Data fim</label>
              <input type="date" value={dataFim} onChange={(e) => setDataFim(e.target.value)}
                className="rounded-xl px-4 py-3 text-sm text-slate-200 outline-none transition-colors focus:border-violet-500/60 focus:ring-2 focus:ring-violet-500/20"
                style={{ background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(139,92,246,0.18)' }} />
            </div>
          </div>

          <Textarea
            label="Culturas (separadas por vírgula, use 'e' para múltiplas)"
            value={culturas}
            onChange={(e) => setCulturas(e.target.value)}
            rows={2}
            placeholder="Soja, Milho, Cana-de-açúcar, Soja e Milho..."
          />

          {/* Produtos e direcionais */}
          <div>
            <label className="text-xs font-semibold text-slate-400 uppercase tracking-widest mb-1.5 block">
              Produtos e Direcionais <span className="normal-case font-normal text-slate-600">— formato: Produto(s) - Cultura(s) - Tema</span>
            </label>
            <Textarea
              value={produtosDirecionaisTexto}
              onChange={(e) => setProdutosDirecionaisTexto(e.target.value)}
              rows={7}
              placeholder="Verdavis, Megafol e Victrato - Soja e Milho - Tecnologia para feira"
            />
          </div>

          {/* Semana de evento e pauta recorrente */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div className="rounded-xl border p-4 flex flex-col gap-3" style={{ borderColor: 'rgba(139,92,246,0.18)', background: 'rgba(0,0,0,0.15)' }}>
              <p className="text-xs font-semibold text-slate-400 uppercase tracking-widest">Semana com Evento (1 post/dia)</p>
              <div className="flex flex-col">
                <label className="text-xs text-slate-500 mb-1">Início</label>
                <input type="date" value={semanaFeirasInicio} onChange={(e) => setSemanaFeirasInicio(e.target.value)}
                  className="rounded-xl px-3 py-2 text-sm text-slate-200 outline-none"
                  style={{ background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(139,92,246,0.18)' }} />
              </div>
              <div className="flex flex-col">
                <label className="text-xs text-slate-500 mb-1">Fim</label>
                <input type="date" value={semanaFeirasFim} onChange={(e) => setSemanaFeirasFim(e.target.value)}
                  className="rounded-xl px-3 py-2 text-sm text-slate-200 outline-none"
                  style={{ background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(139,92,246,0.18)' }} />
              </div>
              <Input label="Produtos prioritários" value={produtosPrioritariosFeira} onChange={(e) => setProdutosPrioritariosFeira(e.target.value)} placeholder="Verdavis, Megafol, Victrato" />
            </div>

            <div className="rounded-xl border p-4 flex flex-col gap-3" style={{ borderColor: 'rgba(139,92,246,0.18)', background: 'rgba(0,0,0,0.15)' }}>
              <p className="text-xs font-semibold text-slate-400 uppercase tracking-widest">Pauta Recorrente</p>
              <Input label="Texto da pauta fixa" value={pautaRecorrenteTexto} onChange={(e) => setPautaRecorrenteTexto(e.target.value)} placeholder="Victrato pelo Brasil" />
              <div>
                <label className="text-xs font-semibold text-slate-400 uppercase tracking-widest mb-2 block">Dias da semana</label>
                <div className="flex flex-wrap gap-2">
                  {DIAS_SEMANA.map((dia) => (
                    <button
                      key={dia}
                      onClick={() => toggleDia(dia)}
                      className="px-2.5 py-1 rounded-lg text-xs font-medium transition-all"
                      style={{
                        background: pautaRecorrenteDias.includes(dia) ? 'linear-gradient(135deg,#4c1d95,#7c3aed)' : 'rgba(255,255,255,0.04)',
                        color: pautaRecorrenteDias.includes(dia) ? '#fff' : '#64748b',
                        border: `1px solid ${pautaRecorrenteDias.includes(dia) ? 'transparent' : 'rgba(139,92,246,0.18)'}`,
                      }}
                    >
                      {dia}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          </div>

          <Textarea
            label="Contexto do mês"
            value={contextoMensal}
            onChange={(e) => setContextoMensal(e.target.value)}
            rows={6}
            placeholder="FEVEREIRO 2026:&#10;- Soja: colheita no centro-sul&#10;..."
          />
        </div>

        {/* Coluna direita — configurações */}
        <div className="flex flex-col gap-4">
          <div className="rounded-xl border p-4 flex flex-col gap-3" style={{ borderColor: 'rgba(139,92,246,0.18)', background: 'rgba(0,0,0,0.15)' }}>
            <p className="text-xs font-semibold text-slate-400 uppercase tracking-widest">Frequência de Pautas</p>
            <p className="text-xs text-slate-500">Período: <span className="text-violet-300 font-medium">{deltaDias} dias</span></p>

            {[
              { label: 'Dias com 1 pauta', value: diasCom1Pauta, set: setDiasCom1Pauta },
              { label: 'Dias com 2 pautas', value: diasCom2Pautas, set: setDiasCom2Pautas },
              { label: 'Dias com 3 pautas', value: diasCom3Pautas, set: setDiasCom3Pautas },
            ].map(({ label, value, set }) => (
              <div key={label} className="flex flex-col gap-1">
                <div className="flex justify-between">
                  <label className="text-xs text-slate-400">{label}</label>
                  <span className="text-xs text-violet-300 font-medium">{value}</span>
                </div>
                <input type="range" min={0} max={deltaDias} value={value}
                  onChange={(e) => set(+e.target.value)} className="accent-violet-500" />
              </div>
            ))}

            <div className={`text-xs px-3 py-2 rounded-lg flex items-center gap-2 ${totalExcede ? 'text-red-300 bg-red-500/10 border border-red-500/20' : 'text-slate-500 bg-black/20'}`}>
              {totalExcede && <AlertCircle size={12} />}
              {totalExcede ? 'Total excede o período!' : `Dias sem pautas: ${diasSemPautas}`}
            </div>
          </div>

          <div className="rounded-xl border p-4 flex flex-col gap-3" style={{ borderColor: 'rgba(139,92,246,0.18)', background: 'rgba(0,0,0,0.15)' }}>
            <p className="text-xs font-semibold text-slate-400 uppercase tracking-widest">Controles</p>
            <Checkbox label="Evitar dias consecutivos sem pautas" checked={evitarConsecutivosSemPautas} onChange={setEvitarConsecutivosSemPautas} />
            <div className="flex flex-col gap-1">
              <div className="flex justify-between">
                <label className="text-xs text-slate-400">Máx. repetições por tema</label>
                <span className="text-xs text-violet-300 font-medium">{maxRepeticoesTema}</span>
              </div>
              <input type="range" min={1} max={5} value={maxRepeticoesTema}
                onChange={(e) => setMaxRepeticoesTema(+e.target.value)} className="accent-violet-500" />
            </div>
          </div>
        </div>
      </div>

      {/* Gerar */}
      <div className="mt-5">
        {error && <div className="mb-3"><Alert type="error">{error}</Alert></div>}
        <Button onClick={handleGerar} loading={loading} disabled={totalExcede} fullWidth>
          <Sparkles size={14} /> Gerar Calendário
        </Button>
        {loading && <Spinner text="Gerando calendário editorial…" />}
      </div>

      {/* Resultado */}
      {calendarioCSV && (
        <div className="mt-6">
          <div className="flex items-center justify-between mb-3">
            <p className="text-xs font-bold text-slate-500 uppercase tracking-widest">Calendário — {mesAno}</p>
          </div>

          <SubTabs tabs={resultTabs} active={resultTab} onChange={setResultTab} />

          {resultTab === 'csv' && (
            <div className="flex flex-col gap-3">
              <textarea
                readOnly
                value={calendarioCSV}
                rows={14}
                className="rounded-xl px-4 py-3 text-xs text-slate-300 outline-none resize-y font-mono w-full"
                style={{ background: 'rgba(0,0,0,0.3)', border: '1px solid rgba(139,92,246,0.12)' }}
              />
              <Button variant="secondary" onClick={handleDownloadCsv} fullWidth>
                <FileDown size={14} /> Baixar CSV
              </Button>
            </div>
          )}

          {resultTab === 'xlsx' && (
            <div className="flex flex-col gap-4">
              <div
                className="rounded-xl border p-6 text-center"
                style={{ borderColor: 'rgba(139,92,246,0.18)', background: 'rgba(0,0,0,0.15)' }}
              >
                <FileDown size={32} className="text-violet-400 mx-auto mb-3" />
                <p className="text-sm text-slate-300 mb-1 font-medium">Exportar como planilha Excel</p>
                <p className="text-xs text-slate-500 mb-4">
                  Gera um arquivo .xlsx formatado com dias da semana, células com wrap e larguras ajustadas.
                </p>
                <button
                  onClick={handleDownloadXlsx}
                  disabled={loadingXlsx}
                  className="inline-flex items-center justify-center gap-2 rounded-xl px-6 py-3 text-sm font-bold text-white transition-all hover:brightness-110 active:scale-[0.98] disabled:opacity-40 shadow-lg"
                  style={{ background: 'linear-gradient(135deg,#1a2d5a,#7c3aed)' }}
                >
                  {loadingXlsx ? (
                    <>
                      <svg className="animate-spin h-4 w-4" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z" />
                      </svg>
                      Gerando XLSX…
                    </>
                  ) : (
                    <>
                      <FileDown size={15} /> Gerar e Baixar XLSX
                    </>
                  )}
                </button>
              </div>
            </div>
          )}
        </div>
      )}
    </Card>
  );
}
