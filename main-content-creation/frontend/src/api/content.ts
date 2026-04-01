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
    api.post<{ resultado: string }>('/content/revisao-tecnica-sem-rag', body).then(
      (r) => r.data
    ),
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

export const briefingsApi = {
  listar: () => api.get<Array<{ _id: string; nome_projeto: string; tipo: string; conteudo: string; createdAt: string }>>('/briefings').then(
    (r) => r.data
  ),
  gerar: (body: unknown) => api.post<{ briefing: string }>('/briefings/gerar', body).then(
    (r) => r.data
  ),
  ajustar: (body: unknown) => api.post<{ briefing: string }>('/briefings/ajustar', body).then(
    (r) => r.data
  ),
};

export const calendarApi = {
  gerar: (body: unknown) => api.post<{ calendario: string }>('/calendar/gerar', body).then(
    (r) => r.data
  ),
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
