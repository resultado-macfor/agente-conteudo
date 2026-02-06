"""
Página de Geração de Briefings.
Geração de briefings a partir do calendário editorial.
"""
import streamlit as st
import datetime
from database import modelo_texto, get_blog_db
from models import construir_contexto


def render():
    """Renderiza a aba de geração de briefings."""
    st.header("📋 Gerador de Briefings a partir do Calendário")

    if not st.session_state.get('agente_selecionado'):
        st.warning("⚠️ Selecione um agente na parte superior do app.")
        return

    agente = st.session_state.agente_selecionado
    st.success(f"🎯 Gerando briefings com base no agente: **{agente['nome']}**")

    # Inicializar session state
    if 'briefings_gerados' not in st.session_state:
        st.session_state.briefings_gerados = []

    # Entrada do calendário
    st.subheader("📅 Calendário de Pautas")

    # Verificar se há calendário gerado
    if 'calendario_gerado' in st.session_state:
        st.info("📋 Calendário detectado na sessão!")
        usar_calendario_sessao = st.checkbox("Usar calendário da sessão", value=True)

        if usar_calendario_sessao:
            calendario_texto = st.session_state.calendario_gerado
            st.text_area("Calendário atual:", calendario_texto, height=200, disabled=True)
        else:
            calendario_texto = st.text_area(
                "Cole o calendário de pautas:",
                height=200,
                placeholder="Cole aqui o calendário CSV gerado..."
            )
    else:
        calendario_texto = st.text_area(
            "Cole o calendário de pautas:",
            height=200,
            placeholder="Cole aqui o calendário CSV gerado..."
        )

    # Configurações
    st.subheader("⚙️ Configurações dos Briefings")

    col1, col2 = st.columns(2)

    with col1:
        tipo_briefing = st.selectbox(
            "Tipo de Briefing:",
            ["Blog Post", "Post Social", "Artigo Técnico", "Email Marketing"]
        )

        nivel_detalhe = st.selectbox(
            "Nível de Detalhe:",
            ["Resumido", "Padrão", "Detalhado", "Completo"]
        )

    with col2:
        incluir_palavras_chave = st.checkbox("Incluir sugestões de palavras-chave", value=True)
        incluir_estrutura = st.checkbox("Incluir estrutura sugerida", value=True)
        incluir_referencias = st.checkbox("Incluir referências técnicas", value=True)

    # Filtros
    st.subheader("🔍 Filtros")

    col_filtro1, col_filtro2 = st.columns(2)

    with col_filtro1:
        filtrar_por_cultura = st.text_input(
            "Filtrar por cultura (opcional):",
            placeholder="Ex: Soja, Milho"
        )

    with col_filtro2:
        filtrar_por_produto = st.text_input(
            "Filtrar por produto (opcional):",
            placeholder="Ex: Verdavis, Miravis"
        )

    # Template de briefing
    template_briefing = """
    ## BRIEFING: {titulo}

    ### INFORMAÇÕES BÁSICAS
    - **Cultura:** {cultura}
    - **Produto(s):** {produtos}
    - **Tema:** {tema}
    - **Data sugerida:** {data}

    ### OBJETIVO
    {objetivo}

    ### PÚBLICO-ALVO
    {publico}

    ### PALAVRAS-CHAVE
    - Principal: {kw_principal}
    - Secundárias: {kw_secundarias}

    ### ESTRUTURA SUGERIDA
    {estrutura}

    ### REFERÊNCIAS TÉCNICAS
    {referencias}

    ### DIRETRIZES
    - Tom de voz: {tom}
    - Extensão sugerida: {extensao}
    - CTA: {cta}

    ---
    """

    # Botão de geração
    if st.button("📝 Gerar Briefings", type="primary", use_container_width=True):
        if not calendario_texto:
            st.error("Por favor, forneça um calendário de pautas.")
            return

        with st.spinner("Gerando briefings..."):
            try:
                # Construir contexto do agente
                contexto_agente = construir_contexto(
                    agente,
                    st.session_state.get('segmentos_selecionados', [])
                )

                # Preparar filtros
                filtros = ""
                if filtrar_por_cultura:
                    filtros += f"- Filtrar por culturas: {filtrar_por_cultura}\n"
                if filtrar_por_produto:
                    filtros += f"- Filtrar por produtos: {filtrar_por_produto}\n"

                # Prompt para geração de briefings
                prompt = f"""
                {contexto_agente}

                ## CALENDÁRIO DE PAUTAS:
                {calendario_texto}

                ## CONFIGURAÇÕES:
                - Tipo de briefing: {tipo_briefing}
                - Nível de detalhe: {nivel_detalhe}
                - Incluir palavras-chave: {incluir_palavras_chave}
                - Incluir estrutura: {incluir_estrutura}
                - Incluir referências: {incluir_referencias}

                ## FILTROS:
                {filtros if filtros else "Nenhum filtro aplicado - gerar para todas as pautas"}

                ## TAREFA:
                Para CADA pauta no calendário, gere um briefing completo seguindo este formato:

                ---
                ## BRIEFING: [Título da pauta]

                ### INFORMAÇÕES BÁSICAS
                - **Cultura:** [cultura]
                - **Produto(s):** [produtos]
                - **Tema:** [tema]
                - **Data sugerida:** [data]

                ### OBJETIVO
                [Objetivo claro e específico do conteúdo]

                ### PÚBLICO-ALVO
                [Descrição do público-alvo]

                ### PALAVRAS-CHAVE
                - Principal: [palavra-chave principal]
                - Secundárias: [lista de palavras-chave secundárias]

                ### ESTRUTURA SUGERIDA
                [Estrutura do conteúdo com seções H2/H3]

                ### REFERÊNCIAS TÉCNICAS
                [Fontes confiáveis para pesquisa: Embrapa, universidades, etc.]

                ### DIRETRIZES
                - Tom de voz: [tom adequado]
                - Extensão sugerida: [número de palavras]
                - CTA: [call-to-action sugerido]

                ---

                Gere briefings completos e profissionais para cada pauta identificada no calendário.
                Se houver filtros, aplique-os para gerar apenas os briefings relevantes.
                """

                resposta = modelo_texto.generate_content(prompt)
                briefings_gerados = resposta.text

                st.session_state.briefings_gerados = briefings_gerados

                st.success("✅ Briefings gerados com sucesso!")

            except Exception as e:
                st.error(f"❌ Erro ao gerar briefings: {str(e)}")

    # Exibir briefings gerados
    if st.session_state.briefings_gerados:
        st.subheader("📄 Briefings Gerados")

        st.markdown(st.session_state.briefings_gerados)

        # Estatísticas
        briefings_count = st.session_state.briefings_gerados.count("## BRIEFING:")
        st.info(f"📊 Total de briefings gerados: {briefings_count}")

        # Download
        st.download_button(
            "💾 Baixar Briefings",
            data=st.session_state.briefings_gerados,
            file_name=f"briefings_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.md",
            mime="text/markdown"
        )

        # Opção de salvar no banco
        db_blog = get_blog_db()
        if db_blog:
            if st.button("💾 Salvar no Banco de Dados"):
                try:
                    collection = db_blog['briefings_gerados']
                    documento = {
                        "briefings": st.session_state.briefings_gerados,
                        "tipo": tipo_briefing,
                        "filtro_cultura": filtrar_por_cultura,
                        "filtro_produto": filtrar_por_produto,
                        "data_criacao": datetime.datetime.now(),
                        "agente": agente['nome']
                    }
                    collection.insert_one(documento)
                    st.success("✅ Briefings salvos no banco de dados!")
                except Exception as e:
                    st.error(f"Erro ao salvar: {str(e)}")

        # Limpar
        if st.button("🗑️ Limpar Briefings"):
            st.session_state.briefings_gerados = []
            st.rerun()
