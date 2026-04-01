import { useState } from 'react';
import type { FormEvent } from 'react';
import { useNavigate } from 'react-router-dom';
import { Lock, User, ArrowRight, AlertCircle } from 'lucide-react';
import api from '../api/client';
import { useStore } from '../store/useStore';
import Logo from '../assets/macLogo.png'
import Back from '../assets/bg.png'

export default function LoginPage() {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const setAuth = useStore((s) => s.setAuth);
  const navigate = useNavigate();

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setError('');
    setLoading(true);
    try {
      const res = await api.post<{ access_token: string; user: string }>('/auth/login', { username, password });
      setAuth(res.data.access_token, res.data.user);
      navigate('/');
    } catch {
      setError('Usuário ou senha incorretos');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="w-full min-h-screen flex" style={{ background: '#070d1f' }}>
      <div className="w-full lg:w-120 shrink-0 flex flex-col items-center justify-center px-8 py-12 relative z-10">

        <div
          className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-96 h-96 rounded-full pointer-events-none"
          style={{ background: 'radial-gradient(circle, rgba(124,58,237,0.12), transparent 70%)', filter: 'blur(40px)' }}
        />

        <div className="w-full max-w-sm relative">
          <div className="mb-10">
            <img src={Logo} alt="Logo" className="h-8 object-contain mb-6" />
            <h1 className="text-3xl font-bold text-white tracking-tight leading-tight">
              Criação de Conteúdo com IA
            </h1>
            <p className="text-slate-500 mt-2 text-sm">Entre com suas credenciais para continuar.</p>
          </div>
          <div
            className="rounded-2xl border p-7 shadow-2xl"
            style={{ background: '#111827', borderColor: 'rgba(139,92,246,0.18)' }}
          >
            <form onSubmit={handleSubmit} className="flex flex-col gap-5">

              <div className="flex flex-col gap-1.5">
                <label className="text-xs font-semibold text-slate-400 uppercase tracking-widest">Usuário</label>
                <div className="relative">
                  <User size={14} className="absolute left-3.5 top-1/2 -translate-y-1/2 text-slate-500" />
                  <input
                    type="text"
                    value={username}
                    onChange={(e) => setUsername(e.target.value)}
                    placeholder="Digite seu usuário"
                    required
                    autoFocus
                    className="w-full rounded-xl border pl-9 pr-4 py-3 text-sm text-slate-200 placeholder-slate-600 outline-none transition-all focus:border-violet-500/60 focus:ring-2 focus:ring-violet-500/20"
                    style={{ background: 'rgba(255,255,255,0.04)', borderColor: 'rgba(139,92,246,0.2)' }}
                  />
                </div>
              </div>

              <div className="flex flex-col gap-1.5">
                <label className="text-xs font-semibold text-slate-400 uppercase tracking-widest">Senha</label>
                <div className="relative">
                  <Lock size={14} className="absolute left-3.5 top-1/2 -translate-y-1/2 text-slate-500" />
                  <input
                    type="password"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    placeholder="Digite sua senha"
                    required
                    className="w-full rounded-xl border pl-9 pr-4 py-3 text-sm text-slate-200 placeholder-slate-600 outline-none transition-all focus:border-violet-500/60 focus:ring-2 focus:ring-violet-500/20"
                    style={{ background: 'rgba(255,255,255,0.04)', borderColor: 'rgba(139,92,246,0.2)' }}
                  />
                </div>
              </div>

              {error && (
                <div
                  className="flex items-center gap-2.5 rounded-xl border px-4 py-3 text-sm text-red-400"
                  style={{ background: 'rgba(239,68,68,0.08)', borderColor: 'rgba(239,68,68,0.2)' }}
                >
                  <AlertCircle size={14} className="shrink-0" />
                  {error}
                </div>
              )}

              <button
                type="submit"
                disabled={loading || !username || !password}
                className="mt-1 w-full flex items-center justify-center gap-2 rounded-xl py-3 text-sm font-bold text-white transition-all hover:brightness-110 active:scale-[0.98] disabled:opacity-40 disabled:cursor-not-allowed shadow-lg"
                style={{ background: 'linear-gradient(135deg,#4c1d95,#7c3aed)' }}
              >
                {loading ? (
                  <>
                    <svg className="animate-spin h-4 w-4" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z" />
                    </svg>
                    Entrando…
                  </>
                ) : (
                  <>
                    Entrar
                    <ArrowRight size={15} />
                  </>
                )}
              </button>
            </form>
          </div>
        </div>
      </div>
      <div className="hidden lg:flex flex-1 relative overflow-hidden">
        <img
          src={Back}
          alt=""
          className="absolute inset-0 w-full h-full object-cover"
        />
        <div
          className="absolute inset-0"
          style={{ background: 'linear-gradient(120deg, rgba(15,29,61,0.55) 0%, rgba(26,10,61,0.45) 50%, rgba(13,13,26,0.6) 100%)' }}
        />
      </div>
    </div>
  );
}
