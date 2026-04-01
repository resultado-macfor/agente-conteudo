import api from './client';
import type { Agent } from '../types';

export const agentsApi = {
  listar: () => api.get<Agent[]>('/agents').then(
    (r) => r.data
  ),
  listarParaHeranca: (exclude?: string) =>
    api.get<Agent[]>('/agents/heranca', { params: { exclude } }).then(
      (r) => r.data
    ),
  obter: (id: string) => api.get<Agent>(`/agents/${id}`).then(
    (r) => r.data
  ),
  obterComHeranca: (id: string) => api.get<Agent>(`/agents/${id}/completo`).then(
    (r) => r.data
  ),
  criar: (data: Partial<Agent>) => api.post<Agent>('/agents', data).then(
    (r) => r.data
  ),
  atualizar: (id: string, data: Partial<Agent>) => api.put<Agent>(`/agents/${id}`, data).then(
    (r) => r.data
  ),
  desativar: (id: string) => api.delete(`/agents/${id}`),
  salvarConversa: (id: string, mensagens: unknown[], segmentos: string[]) =>
    api.post(`/agents/${id}/conversas`, { mensagens, segmentos }),
};
