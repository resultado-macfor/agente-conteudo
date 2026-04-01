export interface Agent {
  _id: string;
  nome: string;
  system_prompt: string;
  base_conhecimento: string;
  comments: string;
  planejamento: string;
  categoria: string;
  agente_mae_id?: string | null;
  herdar_elementos?: string[];
  ativo: boolean;
  criado_por: string;
  createdAt?: string;
}

export interface Message {
  role: 'user' | 'assistant';
  content: string;
}

export type Segmento = 'system_prompt' | 'base_conhecimento' | 'comments' | 'planejamento';

export const SEGMENTO_LABELS: Record<Segmento, string> = {
  system_prompt: 'System Prompt',
  base_conhecimento: 'Brand Guidelines',
  comments: 'Comentários',
  planejamento: 'Planejamento',
};

export const ALL_SEGMENTOS: Segmento[] = ['system_prompt', 'base_conhecimento', 'comments', 'planejamento'];
