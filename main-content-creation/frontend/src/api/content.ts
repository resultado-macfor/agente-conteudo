import api from './client';

export const contentApi = {
  gerar: (body: unknown) => api.post<{ conteudo: string }>('/content/gerar', body).then(
    (r) => r.data
  ),
  revisaoOrtografica: (texto: string, contextoAgente: string) =>
    api.post<{ resultado: string }>('/content/revisao-ortografica', { texto, contextoAgente }).then(
      (r) => r.data
    ),
  revisaoTecnica: (body: unknown) => api.post('/content/revisao-tecnica', body).then(
    (r) => r.data
  ),
  revisaoTecnicaSemRag: (body: unknown) =>
    api.post<{ resultado: string }>('/content/revisao-tecnica-sem-rag', body).then((r) => r.data),
  ajusteRevisaoRag: (textoOriginal: string, textoReescrito: string, ajuste: string) =>
    api.post<{ resultado: string }>('/content/ajuste-revisao-rag', { textoOriginal, textoReescrito, ajuste }).then((r) => r.data),
  otimizacaoSEO: (body: unknown) =>
    api.post<{ resultado: string }>('/content/otimizacao-seo', body).then(
      (r) => r.data
    ),
  perplexity: (texto: string) =>
    api.post<{ resultado: string }>('/content/perplexity', { texto }).then(
      (r) => r.data
    ),
};

export const blogApi = {
  buscarFontes: (briefing: string) =>
    api.post<{ resultado: string; fontes: string[] }>('/blog/perplexity', { briefing }).then(
      (r) => r.data
    ),
  gerar: (body: unknown) => api.post<{ conteudo: string }>('/blog/gerar', body).then(
    (r) => r.data
  ),
  ajustar: (body: unknown) => api.post<{ conteudo: string }>('/blog/ajustar', body).then(
    (r) => r.data
  ),
  historico: () => api.get<Array<{ _id: string; briefing: string; conteudo: string; createdAt: string }>>('/blog/historico').then(
    (r) => r.data
  ),
  salvar: (body: unknown) => api.post('/blog/salvar', body),
};

export interface BriefingGerado {
  indice: number;
  conteudoOriginal: string;
  titulo?: string;
  briefing: string;
  mesReferencia: string;
  tipo?: string;
  historicoAjustes?: Array<{ data: string; solicitacao: string }>;
}

export interface PautaExtraida {
  conteudo: string;
  linha: number;
  coluna: number;
  indice: number;
}

export const briefingsApi = {
  listar: () =>
    api.get<Array<{ _id: string; nome_projeto: string; tipo: string; conteudo: string; createdAt: string }>>('/briefings').then((r) => r.data),

  // Briefing individual a partir de título + texto base
  gerarIndividual: (body: {
    titulo: string;
    mesReferencia: string;
    textoBase: string;
    contextoAdicional: string;
    contextoAgente: string;
  }) => api.post<{ briefing: string }>('/briefings/gerar-individual', body).then((r) => r.data),

  // Briefing a partir de uma pauta do calendário
  gerarDePauta: (body: {
    conteudoPauta: string;
    mesReferencia: string;
    contextoAdicional: string;
    contextoAgente: string;
  }) => api.post<{ briefing: string }>('/briefings/gerar-de-pauta', body).then((r) => r.data),

  // Extrai pautas do CSV (preview)
  extrairPautas: (csvText: string) =>
    api.post<{ pautas: PautaExtraida[] }>('/briefings/extrair-pautas', { csvText }).then((r) => r.data),

  // Gera múltiplos briefings a partir de CSV
  gerarDeCSV: (body: {
    csvText: string;
    mesReferencia: string;
    contextoAdicional: string;
    contextoAgente: string;
  }) => api.post<BriefingGerado[]>('/briefings/gerar-de-csv', body).then((r) => r.data),

  // Ajuste pontual — mantém estrutura, altera apenas o solicitado
  ajustar: (body: {
    briefingAtual: string;
    ajuste: string;
    tituloOuPauta: string;
    mesReferencia: string;
    contextoAgente: string;
  }) => api.post<{ briefing: string }>('/briefings/ajustar', body).then((r) => r.data),
};

export const calendarApi = {
  gerar: (body: unknown) => api.post<{ calendario: string }>('/calendar/gerar', body).then((r) => r.data),
  gerarXlsx: async (csvText: string, mesAno: string): Promise<void> => {
    const res = await api.post('/calendar/gerar-xlsx', { csvText, mesAno }, { responseType: 'blob' });
    const url = URL.createObjectURL(new Blob([res.data], { type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' }));
    const a = Object.assign(document.createElement('a'), { href: url, download: `calendario_${mesAno.replace(/\s+/g, '_').toLowerCase()}.xlsx` });
    a.click();
    URL.revokeObjectURL(url);
  },
};

export const chatApi = {
  sendMessage: (body: unknown) => api.post<{ resposta: string }>('/chat/message', body).then(
    (r) => r.data
  ),
};

export const filesApi = {
  extractText: (files: File[]) => {
    const form = new FormData();
    files.forEach((f) => form.append('files', f));
    return api.post<Array<{ name: string; text: string }>>('/files/extract', form).then((r) => r.data);
  },
};
