import streamlit as st
import google.generativeai as genai

from config.settings import GEMINI_API_KEY, MONGO_URI
from auth.auth import login, check_admin_password, get_current_user, require_login, logout
from services.database import get_mongo_client
from agent.agents import init_collections, listar_agentes, listar_agentes_para_heranca, obter_agente, obter_agente_com_heranca
from ui import (
    tab_chat,
    tab_gerenciamento,
    tab_conteudo,
    tab_blog,
    tab_revisao_ortografica,
    tab_revisao_tecnica,
    tab_otimizacao,
    tab_calendario,
    tab_briefings,
    tab_revisao_tecnica2,
)

st.set_page_config(
    layout="wide",
    page_title="Agente de Conteúdo",
    page_icon="🌱",
)

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    login()
    st.stop()

mongo_client = get_mongo_client()
db = mongo_client['agentes_personalizados']
init_collections(db['agentes'], db['conversas'])

if not GEMINI_API_KEY:
    st.error("GEM_API_KEY não encontrada nas variáveis de ambiente")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)
modelo_texto = genai.GenerativeModel("gemini-2.5-flash")
modelo_texto2 = genai.GenerativeModel("gemini-2.5-pro")

col_logo, col_title, col_logout = st.columns([1, 6, 1])
with col_logo:
    try:
        st.image('assets/macLogo.png', width=120)
    except Exception:
        st.markdown("🌱")
with col_title:
    st.title("Agente de Conteúdo")
    usuario_atual = get_current_user()
    st.caption(f"Usuário: **{usuario_atual}**")
with col_logout:
    st.markdown("<br><br>", unsafe_allow_html=True)
    if st.button("🚪 Sair", key="logout_btn", use_container_width=True):
        logout()
        st.rerun()

st.divider()

st.subheader("🤖 Base de Conhecimento")

if "agente_selecionado" not in st.session_state:
    st.session_state.agente_selecionado = None
if "segmentos_selecionados" not in st.session_state:
    st.session_state.segmentos_selecionados = ["system_prompt", "base_conhecimento", "comments", "planejamento"]

agentes = listar_agentes()
agente_options = {}
agente_selecionado_display = None

with st.container(border=True):
    col1, col2, col3 = st.columns([4, 1, 1])

    with col1:
        if agentes:
            agentes_por_categoria = {}
            for agente in agentes:
                categoria = agente.get('categoria', 'Social')
                agentes_por_categoria.setdefault(categoria, []).append(agente)

            agente_options = {}
            for categoria, agentes_cat in agentes_por_categoria.items():
                for agente in agentes_cat:
                    agente_completo = obter_agente_com_heranca(agente['_id'])
                    if agente_completo:
                        display_name = f"{agente['nome']} ({categoria})"
                        if agente.get('agente_mae_id'):
                            display_name += " 🔗"
                        if get_current_user() != "admin" and agente.get('criado_por'):
                            display_name += " 👤"
                        agente_options[display_name] = agente_completo

            if agente_options:
                agente_selecionado_display = st.selectbox(
                    "Selecione um agente para trabalhar:",
                    list(agente_options.keys()),
                    key="seletor_agente_global",
                )
            else:
                st.info("Nenhum agente disponível com as permissões atuais.")
        else:
            st.info("Nenhum agente disponível. Crie um agente primeiro na aba de Gerenciamento.")

    with col2:
        if agente_options:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("✅ Aplicar", key="aplicar_agente", use_container_width=True):
                st.session_state.agente_selecionado = agente_options[agente_selecionado_display]
                st.success(f"Agente '{agente_selecionado_display}' selecionado!")
                st.rerun()

    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.session_state.agente_selecionado:
            if st.button("🗑️ Limpar", key="limpar_agente", use_container_width=True):
                st.session_state.agente_selecionado = None
                st.session_state.messages = []
                st.rerun()
        else:
            if st.button("🔄 Atualizar", key="recarregar_agentes", use_container_width=True):
                st.rerun()

if st.session_state.agente_selecionado:
    agente_atual = st.session_state.agente_selecionado

    with st.container(border=True):
        col_info1, col_info2 = st.columns([3, 1])
        with col_info1:
            st.success(f"**✅ Agente Ativo:** {agente_atual['nome']} ({agente_atual.get('categoria', 'Social')})")
            if 'agente_mae_id' in agente_atual and agente_atual['agente_mae_id']:
                agente_original = obter_agente(agente_atual['_id'])
                if agente_original and agente_original.get('herdar_elementos'):
                    st.caption(f"🔗 Herda {len(agente_original['herdar_elementos'])} elementos do agente mãe")
        with col_info2:
            if st.button("⚙️ Segmentos", key="alterar_segmentos", use_container_width=True):
                st.session_state.mostrar_segmentos = not st.session_state.get('mostrar_segmentos', False)

        segmentos_labels = {
            "system_prompt": "System Prompt",
            "base_conhecimento": "Brand Guidelines",
            "comments": "Comentários",
            "planejamento": "Planejamento",
        }
        ativos = [segmentos_labels[s] for s in st.session_state.segmentos_selecionados if s in segmentos_labels]
        st.caption(f"📋 Segmentos ativos: {' · '.join(ativos)}")

        if st.session_state.get('mostrar_segmentos', False):
            with st.expander("🔧 Configurar Segmentos do Agente", expanded=True):
                col_seg1, col_seg2, col_seg3, col_seg4 = st.columns(4)
                with col_seg1:
                    system_prompt_ativado = st.checkbox("System Prompt", value="system_prompt" in st.session_state.segmentos_selecionados, key="seg_system")
                with col_seg2:
                    base_conhecimento_ativado = st.checkbox("Brand Guidelines", value="base_conhecimento" in st.session_state.segmentos_selecionados, key="seg_base")
                with col_seg3:
                    comments_ativado = st.checkbox("Comentários", value="comments" in st.session_state.segmentos_selecionados, key="seg_comments")
                with col_seg4:
                    planejamento_ativado = st.checkbox("Planejamento", value="planejamento" in st.session_state.segmentos_selecionados, key="seg_planejamento")

                if st.button("✅ Aplicar Segmentos", key="aplicar_segmentos"):
                    novos_segmentos = []
                    if system_prompt_ativado:
                        novos_segmentos.append("system_prompt")
                    if base_conhecimento_ativado:
                        novos_segmentos.append("base_conhecimento")
                    if comments_ativado:
                        novos_segmentos.append("comments")
                    if planejamento_ativado:
                        novos_segmentos.append("planejamento")
                    st.session_state.segmentos_selecionados = novos_segmentos
                    st.session_state.mostrar_segmentos = False
                    st.rerun()
else:
    st.info("💡 Selecione e aplique um agente acima para começar.")

st.divider()

(
    tab_chat_t,
    tab_gerenciamento_t,
    tab_conteudo_t,
    tab_blog_t,
    tab_revisao_ortografica_t,
    tab_revisao_tecnica_t,
    tab_otimizacao_t,
    tab_calendario_t,
    tab_briefings_t,
    tab_revisao_tecnica2_t,
) = st.tabs([
    "💬 Chat",
    "⚙️ Gerenciar Agentes",
    "✨ Geração de Conteúdo",
    "🌱 Geração de Conteúdo Blog",
    "📝 Revisão Ortográfica",
    "🔧 Revisão Técnica",
    "🚀 Otimização de Conteúdo",
    "📅 Criadora de Calendário",
    "📋 Gerador de Briefings",
    "🔍 Revisão Técnica Sem RAG",
])

tab_chat.render(tab_chat_t, modelo_texto)
tab_gerenciamento.render(tab_gerenciamento_t)
tab_conteudo.render(tab_conteudo_t, modelo_texto)
tab_blog.render(tab_blog_t, modelo_texto)
tab_revisao_ortografica.render(tab_revisao_ortografica_t, modelo_texto)
tab_revisao_tecnica.render(tab_revisao_tecnica_t, modelo_texto, modelo_texto2, db=db)
tab_otimizacao.render(tab_otimizacao_t, modelo_texto)
tab_calendario.render(tab_calendario_t, modelo_texto)
tab_briefings.render(tab_briefings_t, modelo_texto)
tab_revisao_tecnica2.render(tab_revisao_tecnica2_t, modelo_texto2)

st.markdown("""
<style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    [data-testid="stChatMessageContent"] {
        font-size: 1rem;
    }
    div[data-testid="stTabs"] {
        margin-top: -30px;
    }
    .segment-indicator {
        background-color: #f0f2f6;
        padding: 0.5rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #4CAF50;
    }
    .pipeline-step {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        border-left: 5px solid #4CAF50;
    }
    .pipeline-complete { border-left-color: #4CAF50; }
    .pipeline-current  { border-left-color: #2196F3; }
    .pipeline-pending  { border-left-color: #ff9800; }
</style>
""", unsafe_allow_html=True)
