import streamlit as st
from auth.auth import get_current_user, check_admin_password
from agent.agents import (
    criar_agente, listar_agentes, listar_agentes_para_heranca,
    obter_agente, obter_agente_com_heranca, atualizar_agente, desativar_agente,
)


def render(tab):
    with tab:
        st.header("⚙️ Gerenciamento de Agentes")

        current_user = get_current_user()

        if current_user not in ["admin", "SYN", "SME", "Enterprise"]:
            st.warning("Acesso restrito a usuários autorizados")
            return

        if current_user == "admin":
            if not check_admin_password():
                st.warning("Digite a senha de administrador")
                return
            st.write("Bem-vindo administrador!")
        else:
            st.write(f"Bem-vindo {current_user}!")

        sub_tab1, sub_tab2, sub_tab3 = st.tabs(["Criar Agente", "Editar Agente", "Gerenciar Agentes"])

        with sub_tab1:
            _criar_agente_form()

        with sub_tab2:
            _editar_agente_form()

        with sub_tab3:
            _gerenciar_agentes(current_user)


def _criar_agente_form():
    st.subheader("Criar Novo Agente")

    with st.form("form_criar_agente"):
        nome_agente = st.text_input("Nome do Agente:")
        categoria = st.selectbox("Categoria:", ["Social", "SEO", "Conteúdo"],
                                 help="Organize o agente por área de atuação")
        criar_como_filho = st.checkbox("Criar como agente filho (herdar elementos)")

        agente_mae_id = None
        herdar_elementos = []

        if criar_como_filho:
            agentes_mae = listar_agentes_para_heranca()
            if agentes_mae:
                agente_mae_options = {
                    f"{a['nome']} ({a.get('categoria', 'Social')})": a['_id']
                    for a in agentes_mae
                }
                agente_mae_selecionado = st.selectbox("Agente Mãe:", list(agente_mae_options.keys()),
                                                       help="Selecione o agente do qual este agente irá herdar elementos")
                agente_mae_id = agente_mae_options[agente_mae_selecionado]

                st.subheader("Elementos para Herdar")
                herdar_elementos = st.multiselect(
                    "Selecione os elementos a herdar do agente mãe:",
                    ["system_prompt", "base_conhecimento", "comments", "planejamento"],
                    help="Estes elementos serão herdados do agente mãe se não preenchidos abaixo",
                )
            else:
                st.info("Nenhum agente disponível para herança. Crie primeiro um agente mãe.")

        system_prompt = st.text_area("Prompt de Sistema:", height=150,
                                     placeholder="Ex: Você é um assistente especializado em...",
                                     help="Deixe vazio se for herdar do agente mãe")
        base_conhecimento = st.text_area("Brand Guidelines:", height=200,
                                         placeholder="Cole aqui informações, diretrizes, dados...",
                                         help="Deixe vazio se for herdar do agente mãe")
        comments = st.text_area("Comentários do cliente:", height=200,
                                 placeholder="Cole aqui os comentários de ajuste do cliente (Se houver)",
                                 help="Deixe vazio se for herdar do agente mãe")
        planejamento = st.text_area("Planejamento:", height=200,
                                    placeholder="Estratégias, planejamentos, cronogramas...",
                                    help="Deixe vazio se for herdar do agente mãe")

        submitted = st.form_submit_button("Criar Agente")
        if submitted:
            if nome_agente:
                criar_agente(
                    nome_agente, system_prompt, base_conhecimento, comments, planejamento,
                    categoria,
                    agente_mae_id if criar_como_filho else None,
                    herdar_elementos if criar_como_filho else [],
                )
                st.success(f"Agente '{nome_agente}' criado com sucesso na categoria {categoria}!")
            else:
                st.error("Nome é obrigatório!")


def _editar_agente_form():
    st.subheader("Editar Agente Existente")

    agentes = listar_agentes()
    if not agentes:
        st.info("Nenhum agente criado ainda.")
        return

    agente_options = {agente['nome']: agente for agente in agentes}
    agente_selecionado_nome = st.selectbox("Selecione o agente para editar:", list(agente_options.keys()))

    if not agente_selecionado_nome:
        return

    agente = agente_options[agente_selecionado_nome]

    with st.form("form_editar_agente"):
        novo_nome = st.text_input("Nome do Agente:", value=agente['nome'])

        agente_mae_id = None
        herdar_elementos = []

        if agente.get('agente_mae_id'):
            agente_mae = obter_agente(agente['agente_mae_id'])
            if agente_mae:
                st.info(f"🔗 Este agente é filho de: {agente_mae['nome']}")
                st.write(f"Elementos herdados: {', '.join(agente.get('herdar_elementos', []))}")

            tornar_independente = st.checkbox("Tornar agente independente (remover herança)")
            if tornar_independente:
                agente_mae_id = None
                herdar_elementos = []
            else:
                agente_mae_id = agente.get('agente_mae_id')
                herdar_elementos = agente.get('herdar_elementos', [])
        else:
            adicionar_heranca = st.checkbox("Adicionar herança de agente mãe")
            if adicionar_heranca:
                agentes_mae = listar_agentes_para_heranca(agente['_id'])
                if agentes_mae:
                    agente_mae_options = {
                        f"{am['nome']} ({am.get('categoria', 'Social')})": am['_id']
                        for am in agentes_mae
                    }
                    if agente_mae_options:
                        agente_mae_selecionado = st.selectbox("Agente Mãe:", list(agente_mae_options.keys()),
                                                               help="Selecione o agente do qual este agente irá herdar elementos")
                        agente_mae_id = agente_mae_options[agente_mae_selecionado]
                        herdar_elementos = st.multiselect(
                            "Elementos para herdar:",
                            ["system_prompt", "base_conhecimento", "comments", "planejamento"],
                            default=herdar_elementos,
                        )
                    else:
                        st.info("Nenhum agente disponível para herança.")
                else:
                    st.info("Nenhum agente disponível para herança.")

        novo_prompt = st.text_area("Prompt de Sistema:", value=agente['system_prompt'], height=150)
        nova_base = st.text_area("Brand Guidelines:", value=agente.get('base_conhecimento', ''), height=200)
        nova_comment = st.text_area("Comentários:", value=agente.get('comments', ''), height=200)
        novo_planejamento = st.text_area("Planejamento:", value=agente.get('planejamento', ''), height=200)

        submitted = st.form_submit_button("Atualizar Agente")
        if submitted:
            if novo_nome:
                atualizar_agente(
                    agente['_id'], novo_nome, novo_prompt, nova_base,
                    nova_comment, novo_planejamento, agente.get('categoria', 'Social'),
                    agente_mae_id, herdar_elementos,
                )
                st.success(f"Agente '{novo_nome}' atualizado com sucesso!")
                st.rerun()
            else:
                st.error("Nome é obrigatório!")


def _gerenciar_agentes(current_user):
    st.subheader("Gerenciar Agentes")

    if current_user == "admin":
        st.info("👑 Modo Administrador: Visualizando todos os agentes do sistema")
    else:
        st.info(f"👤 Visualizando apenas seus agentes ({current_user})")

    categorias = ["Todos", "Social", "SEO", "Conteúdo"]
    categoria_filtro = st.selectbox("Filtrar por categoria:", categorias)

    agentes = listar_agentes()

    if categoria_filtro != "Todos":
        agentes = [a for a in agentes if a.get('categoria') == categoria_filtro]

    if not agentes:
        st.info("Nenhum agente encontrado para esta categoria.")
        return

    for i, agente in enumerate(agentes):
        with st.container():
            owner_info = ""
            if current_user == "admin" and agente.get('criado_por'):
                owner_info = f" | 👤 {agente['criado_por']}"

            st.write(f"**{agente['nome']} - {agente.get('categoria', 'Social')}{owner_info} - Criado em {agente['data_criacao'].strftime('%d/%m/%Y')}**")

            if agente.get('agente_mae_id'):
                agente_mae = obter_agente(agente['agente_mae_id'])
                if agente_mae:
                    st.write(f"**🔗 Herda de:** {agente_mae['nome']}")
                    st.write(f"**Elementos herdados:** {', '.join(agente.get('herdar_elementos', []))}")

            st.write(f"**Prompt de Sistema:** {agente['system_prompt'][:100]}..." if agente['system_prompt'] else "**Prompt de Sistema:** (herdado ou vazio)")
            if agente.get('base_conhecimento'):
                st.write(f"**Brand Guidelines:** {agente['base_conhecimento'][:200]}...")
            if agente.get('comments'):
                st.write(f"**Comentários do cliente:** {agente['comments'][:200]}...")
            if agente.get('planejamento'):
                st.write(f"**Planejamento:** {agente['planejamento'][:200]}...")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Selecionar para Chat", key=f"select_{i}"):
                    st.session_state.agente_selecionado = obter_agente_com_heranca(agente['_id'])
                    st.session_state.messages = []
                    st.success(f"Agente '{agente['nome']}' selecionado!")
            with col2:
                if st.button("Desativar", key=f"delete_{i}"):
                    desativar_agente(agente['_id'])
                    st.success(f"Agente '{agente['nome']}' desativado!")
                    st.rerun()
            st.divider()
