import { useState } from 'react';
import {
  MessageSquare, Settings, Sparkles, BookOpen,
  CheckSquare, Wrench, TrendingUp, Calendar, ClipboardList,
  Search, LogOut, ChevronLeft, ChevronRight, Menu, X,
} from 'lucide-react';
import { useStore } from '../store/useStore';
import AgentSelector from '../components/AgentSelector';
import TabChat from './tabs/TabChat';
import TabGerenciamento from './tabs/TabGerenciamento';
import TabConteudo from './tabs/TabConteudo';
import TabBlog from './tabs/TabBlog';
import TabRevisaoOrtografica from './tabs/TabRevisaoOrtografica';
import TabRevisaoTecnica from './tabs/TabRevisaoTecnica';
import TabOtimizacao from './tabs/TabOtimizacao';
import TabCalendario from './tabs/TabCalendario';
import TabBriefings from './tabs/TabBriefings';
import TabRevisaoTecnica2 from './tabs/TabRevisaoTecnica2';
import Logo from '../assets/macLogo.png'

const TABS = [
  {
     id: 'chat', 
     label: 'Chat',               
     icon: MessageSquare,
     group: 'Principal' 
  },
  { id: 'gerenciamento', label: 'Gerenciar Agentes',   icon: Settings,       group: 'Principal' },
  { id: 'conteudo',      label: 'Geração de Conteúdo', icon: Sparkles,       group: 'Criação' },
  { id: 'blog',          label: 'Geração Blog',        icon: BookOpen,       group: 'Criação' },
  { id: 'briefings',     label: 'Briefings',           icon: ClipboardList,  group: 'Criação' },
  { id: 'calendario',    label: 'Calendário',          icon: Calendar,       group: 'Criação' },
  { id: 'ortografica',   label: 'Revisão Ortográfica', icon: CheckSquare,    group: 'Revisão' },
  { id: 'tecnica',       label: 'Revisão Técnica',     icon: Wrench,         group: 'Revisão' },
  { id: 'tecnica2',      label: 'Revisão Sem RAG',     icon: Search,         group: 'Revisão' },
  { id: 'otimizacao',    label: 'Otimização SEO',      icon: TrendingUp,     group: 'Revisão' },
] as const;

type TabId = typeof TABS[number]['id'];

const GROUPS = ['Principal', 'Criação', 'Revisão'];

export default function MainPage() {
  const { user, logout } = useStore();
  const [activeTab, setActiveTab] = useState<TabId>('chat');
  const [collapsed, setCollapsed] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);

  const renderTab = () => {
    switch (activeTab) {
      case 'chat':          return <TabChat />;
      case 'gerenciamento': return <TabGerenciamento />;
      case 'conteudo':      return <TabConteudo />;
      case 'blog':          return <TabBlog />;
      case 'ortografica':   return <TabRevisaoOrtografica />;
      case 'tecnica':       return <TabRevisaoTecnica />;
      case 'otimizacao':    return <TabOtimizacao />;
      case 'calendario':    return <TabCalendario />;
      case 'briefings':     return <TabBriefings />;
      case 'tecnica2':      return <TabRevisaoTecnica2 />;
    }
  };



  const SidebarContent = () => (
    <div className="flex flex-col h-full">
      <div
        className="flex items-center gap-3 px-4 py-5"
        style={{ borderBottom: '1px solid rgba(139,92,246,0.12)' }}
      >
        {!collapsed && (
          <div className="min-w-0">
            <img src={Logo} alt="Logo" width={200} height={30} className="object-contain" />
          </div>
        )}
      </div>

    
      <nav className="flex-1 overflow-y-auto py-4 px-2 space-y-5">
        {GROUPS.map((group) => {
          const groupTabs = TABS.filter((t) => t.group === group);
          return (
            <div key={group}>
              {!collapsed && (
                <p className="text-[10px] font-bold uppercase tracking-widest text-slate-600 px-3 mb-2">
                  {group}
                </p>
              )}
              <div className="space-y-0.5">
                {groupTabs.map((tab) => {
                  const Icon = tab.icon;
                  const isActive = activeTab === tab.id;
                  return (
                    <button
                      key={tab.id}
                      onClick={() => { setActiveTab(tab.id); setMobileOpen(false); }}
                      title={collapsed ? tab.label : undefined}
                      className="w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium transition-all duration-150"
                      style={{
                        background: isActive ? 'rgba(124,58,237,0.2)' : 'transparent',
                        color: isActive ? '#c4b5fd' : '#475569',
                        borderLeft: isActive ? '2px solid #7c3aed' : '2px solid transparent',
                      }}
                      onMouseEnter={(e) => { if (!isActive) (e.currentTarget as HTMLElement).style.color = '#94a3b8'; }}
                      onMouseLeave={(e) => { if (!isActive) (e.currentTarget as HTMLElement).style.color = '#475569'; }}
                    >
                      <Icon size={17} className="shrink-0" style={{ color: isActive ? '#a78bfa' : 'inherit' }} />
                      {!collapsed && <span className="truncate">{tab.label}</span>}
                    </button>
                  );
                })}
              </div>
            </div>
          );
        })}
      </nav>

     
      <div className="p-3" style={{ borderTop: '1px solid rgba(139,92,246,0.12)' }}>
        <div
          className="flex items-center gap-3 px-3 py-2.5 rounded-xl"
          style={{ background: 'rgba(255,255,255,0.04)' }}
        >
          <div
            className="w-8 h-8 rounded-lg flex items-center justify-center shrink-0 text-sm font-bold text-white"
            style={{ background: 'linear-gradient(135deg,#4c1d95,#7c3aed)' }}
          >
            {user?.[0]?.toUpperCase()}
          </div>
          {!collapsed && (
            <>
              <span className="text-sm text-slate-300 flex-1 truncate font-medium">{user}</span>
              <button
                onClick={logout}
                className="text-slate-600 hover:text-red-400 transition-colors p-1.5 rounded-lg hover:bg-red-500/10"
                title="Sair"
              >
                <LogOut size={15} />
              </button>
            </>
          )}
        </div>
        {collapsed && (
          <button
            onClick={logout}
            className="w-full mt-2 flex items-center justify-center p-2 rounded-xl text-slate-600 hover:text-red-400 transition-colors hover:bg-red-500/10"
          >
            <LogOut size={15} />
          </button>
        )}
      </div>
    </div>
  );

  const SIDEBAR_W = collapsed ? 72 : 256;

  return (
    <div className="flex h-screen w-full overflow-hidden" style={{ background: '#070d1f' }}>


      {mobileOpen && (
        <div className="fixed inset-0 bg-black/60 z-40 lg:hidden" onClick={() => setMobileOpen(false)} />
      )}

      <aside
        className="hidden lg:flex flex-col h-full shrink-0 relative transition-all duration-300"
        style={{ width: SIDEBAR_W, background: '#111827', borderRight: '1px solid rgba(139,92,246,0.12)' }}
      >
        <SidebarContent />
        <button
          onClick={() => setCollapsed(!collapsed)}
          className="absolute -right-3.5 top-7 w-7 h-7 rounded-full flex items-center justify-center transition-all hover:scale-110"
          style={{ background: '#1a2440', border: '1px solid rgba(139,92,246,0.25)', color: '#64748b' }}
        >
          {collapsed ? <ChevronRight size={13} /> : <ChevronLeft size={13} />}
        </button>
      </aside>


      <aside
        className="fixed left-0 top-0 h-full z-50 flex flex-col lg:hidden transition-transform duration-300"
        style={{ width: 256, background: '#111827', borderRight: '1px solid rgba(139,92,246,0.12)', transform: mobileOpen ? 'translateX(0)' : 'translateX(-100%)' }}
      >
        <SidebarContent />
      </aside>


      <div className="flex flex-col flex-1 min-w-0 overflow-hidden">
        <div
          className="shrink-0 flex items-center gap-3 px-4 py-3"
          style={{ background: 'rgba(0,0,0,0.2)', borderBottom: '1px solid rgba(139,92,246,0.1)' }}
        >

          <button
            className="lg:hidden shrink-0 w-9 h-9 flex items-center justify-center rounded-xl text-slate-400 hover:text-violet-400 hover:bg-violet-500/10 transition-all"
            onClick={() => setMobileOpen((v) => !v)}
            aria-label="Abrir menu"
          >
            {mobileOpen ? <X size={18} /> : <Menu size={18} />}
          </button>
          <div className="flex-1 min-w-0">
            <AgentSelector />
          </div>
        </div>

        <main className="flex-1 overflow-y-auto p-6 lg:p-8">
          {renderTab()}
        </main>
      </div>
    </div>
  );
}
