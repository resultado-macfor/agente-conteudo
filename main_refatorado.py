"""
Agente de Conteúdo - Aplicação Streamlit Refatorada
====================================================

Aplicação para geração e revisão de conteúdo agrícola
usando IA generativa e bases de conhecimento vetorial.

Estrutura modular:
- config/: Configurações e credenciais
- database/: Conexões com MongoDB, AstraDB e Gemini
- auth/: Sistema de autenticação
- services/: Serviços (embeddings, RAG, transcrição)
- models/: Modelos e CRUD de agentes
- pages/: Páginas da aplicação
"""
import streamlit as st

# Configuração da página (DEVE ser a primeira chamada Streamlit)
st.set_page_config(
    layout="wide",
    page_title="Conteúdo"
)

# Imports após configuração da página
from auth import login, get_current_user, logout
from database import init_databases
from models import (
    listar_agentes,
    obter_agente,
    obter_agente_com_heranca
)
from pages import (
    chat,
    gerenciamento,
    conteudo,
    blog,
    revisao_ortografica,
    revisao_tecnica,
    otimizacao,
    calendario,
    briefings,
    revisao_tecnica2
)


def init_session_state():
    """Inicializa variáveis do session state."""
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "agente_selecionado" not in st.session_state:
        st.session_state.agente_selecionado = None
    if "segmentos_selecionados" not in st.session_state:
        st.session_state.segmentos_selecionados = [
            "system_prompt", "base_conhecimento", "comments", "planejamento"
        ]
    if "messages" not in st.session_state:
        st.session_state.messages = []


def render_agent_selector():
    """Renderiza o seletor de agentes."""
    st.header("🤖 Selecione a base de conhecimento")

    agentes = listar_agentes()

    with st.container():
        col1, col2, col3 = st.columns([3, 1, 1])

        with col1:
            if agentes:
                # Agrupar agentes por categoria
                agentes_por_categoria = {}
                for agente in agentes:
                    categoria = agente.get('categoria', 'Social')
                    if categoria not in agentes_por_categoria:
                        agentes_por_categoria[categoria] = []
                    agentes_por_categoria[categoria].append(agente)

                # Criar opções de seleção
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
                        key="seletor_agente_global"
                    )

                    if st.button("🔄 Aplicar Agente", key="aplicar_agente"):
                        st.session_state.agente_selecionado = agente_options[agente_selecionado_display]
                        st.success(f"Agente '{agente_selecionado_display}' selecionado!")
                        st.rerun()
                else:
                    st.info("Nenhum agente disponível com as permissões atuais.")
            else:
                st.info("Nenhum agente disponível. Crie um agente na aba de Gerenciamento.")

        with col2:
            if st.session_state.agente_selecionado:
                if st.button("🗑️ Limpar Agente", key="limpar_agente"):
                    st.session_state.agente_selecionado = None
                    st.session_state.messages = []
                    st.success("Agente removido!")
                    st.rerun()

        with col3:
            if st.button("🔄 Recarregar", key="recarregar_agentes"):
                st.rerun()


def render_agent_info():
    """Renderiza informações do agente selecionado."""
    if not st.session_state.agente_selecionado:
        st.warning("⚠️ Nenhum agente selecionado. Selecione um agente acima para começar.")
        return

    agente_atual = st.session_state.agente_selecionado

    with st.container():
        st.success(f"**✅ Agente Ativo:** {agente_atual['nome']} ({agente_atual.get('categoria', 'Social')})")

        # Mostrar informações de herança
        if agente_atual.get('agente_mae_id'):
            agente_original = obter_agente(agente_atual['_id'])
            if agente_original and agente_original.get('herdar_elementos'):
                st.info(f"🔗 Este agente herda {len(agente_original['herdar_elementos'])} elementos do agente mãe")

        st.info(f"📋 Segmentos ativos: {', '.join(st.session_state.segmentos_selecionados)}")

        # Configuração de segmentos
        if st.button("⚙️ Alterar Segmentos", key="alterar_segmentos"):
            if "mostrar_segmentos" not in st.session_state:
                st.session_state.mostrar_segmentos = True
            else:
                st.session_state.mostrar_segmentos = not st.session_state.mostrar_segmentos

        if st.session_state.get('mostrar_segmentos', False):
            with st.expander("🔧 Configurar Segmentos do Agente", expanded=True):
                st.write("Selecione quais elementos do agente serão utilizados:")

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    system_prompt_ativado = st.checkbox(
                        "System Prompt",
                        value="system_prompt" in st.session_state.segmentos_selecionados,
                        key="seg_system"
                    )
                with col2:
                    base_conhecimento_ativado = st.checkbox(
                        "Brand Guidelines",
                        value="base_conhecimento" in st.session_state.segmentos_selecionados,
                        key="seg_base"
                    )
                with col3:
                    comments_ativado = st.checkbox(
                        "Comentários",
                        value="comments" in st.session_state.segmentos_selecionados,
                        key="seg_comments"
                    )
                with col4:
                    planejamento_ativado = st.checkbox(
                        "Planejamento",
                        value="planejamento" in st.session_state.segmentos_selecionados,
                        key="seg_planejamento"
                    )

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
                    st.success(f"Segmentos atualizados: {', '.join(novos_segmentos)}")
                    st.session_state.mostrar_segmentos = False
                    st.rerun()


def main():
    """Função principal da aplicação."""
    # Inicializar session state
    init_session_state()

    # Verificar autenticação
    if not st.session_state.logged_in:
        login()
        st.stop()

    # Inicializar bancos de dados
    init_databases()

    # Header
    st.image('macLogo.png', width=300)
    st.title("Conteúdo")

    # Botão de logout
    if st.button("🚪 Sair", key="logout_btn"):
        logout()

    # Seletor de agente
    render_agent_selector()

    # Informações do agente
    render_agent_info()

    st.markdown("---")

    # Abas da aplicação
    tabs = st.tabs([
        "💬 Chat",
        "⚙️ Gerenciar Agentes",
        "✨ Geração de Conteúdo",
        "🌱 Geração de Conteúdo Blog",
        "📝 Revisão Ortográfica",
        "🔧 Revisão Técnica",
        "🚀 Otimização de Conteúdo",
        "📅 Criadora de Calendário",
        "📋 Gerador de Briefings",
        "Revisão Técnica Sem RAG"
    ])

    with tabs[0]:
        chat.render()

    with tabs[1]:
        gerenciamento.render()

    with tabs[2]:
        conteudo.render()

    with tabs[3]:
        blog.render()

    with tabs[4]:
        revisao_ortografica.render()

    with tabs[5]:
        revisao_tecnica.render()

    with tabs[6]:
        otimizacao.render()

    with tabs[7]:
        calendario.render()

    with tabs[8]:
        briefings.render()

    with tabs[9]:
        revisao_tecnica2.render()


if __name__ == "__main__":
    main()
