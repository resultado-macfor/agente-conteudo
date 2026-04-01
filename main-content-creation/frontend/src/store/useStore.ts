import { create } from 'zustand';
import type { Agent, Message, Segmento } from '../types';
import { ALL_SEGMENTOS } from '../types';

interface AppState {
  token: string | null;
  user: string | null;
  setAuth: (token: string, user: string) => void;
  logout: () => void;

  selectedAgent: Agent | null;
  setSelectedAgent: (agent: Agent | null) => void;
  segmentos: Segmento[];
  setSegmentos: (segs: Segmento[]) => void;

  messages: Message[];
  setMessages: (msgs: Message[]) => void;
  addMessage: (msg: Message) => void;
  clearMessages: () => void;
}

export const useStore = create<AppState>((set) => ({
  token: localStorage.getItem('token'),
  user: localStorage.getItem('user'),
  setAuth: (token, user) => {
    localStorage.setItem('token', token);
    localStorage.setItem('user', user);
    set({ token, user });
  },
  logout: () => {
    localStorage.removeItem('token');
    localStorage.removeItem('user');
    set({ token: null, user: null, selectedAgent: null, messages: [] });
  },

  selectedAgent: null,
  setSelectedAgent: (agent) => set({ selectedAgent: agent }),
  segmentos: ALL_SEGMENTOS,
  setSegmentos: (segs) => set({ segmentos: segs }),

  messages: [],
  setMessages: (msgs) => set({ messages: msgs }),
  addMessage: (msg) => set((s) => ({ messages: [...s.messages, msg] })),
  clearMessages: () => set({ messages: [] }),
}));
