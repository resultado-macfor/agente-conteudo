"""
Página de Revisão Técnica Sem RAG.
Revisão técnica usando apenas o modelo de linguagem sem base vetorial.
"""
import streamlit as st
from database import modelo_texto
from models import construir_contexto


def render():
    """Renderiza a aba de revisão técnica sem RAG."""
    st.header("🔧 Revisão Técnica Sem RAG")
    st.markdown("**Revisão técnica usando apenas o conhecimento do modelo**")

    # Layout com duas colunas
    col_original, col_revisado = st.columns(2)

    with col_original:
        st.subheader("📄 Conteúdo Original")
        texto_tecnico = st.text_area(
            "Cole o conteúdo técnico para revisão:",
            height=400,
            placeholder="Cole aqui o conteúdo técnico agrícola...",
            key="texto_tecnico_sem_rag",
            label_visibility="collapsed"
        )

    with col_revisado:
        st.subheader("✨ Conteúdo Revisado")
        revisado_placeholder = st.empty()
        revisado_placeholder.info("📝 Aguardando revisão... O conteúdo revisado aparecerá aqui.")

    # Configurações
    st.markdown("---")
    st.subheader("⚙️ Configurações da Revisão")

    col_config1, col_config2 = st.columns(2)

    with col_config1:
        tipo_conteudo = st.selectbox(
            "Tipo de Conteúdo:",
            ["Artigo Técnico", "Material Comercial", "Blog Post", "Manual Técnico", "Comunicado Técnico"],
            key="tipo_sem_rag"
        )

        tom_voz = st.selectbox(
            "Tom de Voz:",
            ["Técnico", "Jornalístico", "Educativo", "Formal"],
            key="tom_sem_rag"
        )

    with col_config2:
        nivel_rigor = st.select_slider(
            "Nível de Rigor:",
            ["Leve", "Moderado", "Rigoroso", "Especialista"],
            key="rigor_sem_rag"
        )

        usar_contexto_agente = st.checkbox(
            "Usar contexto do agente",
            value=bool(st.session_state.get('agente_selecionado')),
            key="contexto_sem_rag"
        )

        incluir_relatorio = st.checkbox(
            "📋 Incluir relatório de mudanças",
            value=True,
            key="relatorio_sem_rag"
        )

    # Foco da revisão
    st.subheader("🎯 Foco da Revisão")

    col_foco1, col_foco2 = st.columns(2)

    with col_foco1:
        revisar_taxonomia = st.checkbox("Taxonomia e nomenclatura", value=True)
        revisar_dados = st.checkbox("Dados e estatísticas", value=True)
        revisar_terminologia = st.checkbox("Terminologia técnica", value=True)

    with col_foco2:
        revisar_coerencia = st.checkbox("Coerência científica", value=True)
        revisar_clareza = st.checkbox("Clareza e objetividade", value=True)
        revisar_fontes = st.checkbox("Verificar fontes citadas", value=True)

    # Instruções adicionais
    instrucoes_adicionais = st.text_area(
        "Instruções adicionais (opcional):",
        height=80,
        placeholder="Ex: Foque na correção de nomes científicos..."
    )

    # Botão de revisão
    st.markdown("---")
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])

    with col_btn2:
        if st.button("🔬 Realizar Revisão Técnica", type="primary", use_container_width=True, key="btn_sem_rag"):
            if texto_tecnico:
                # Construir contexto do agente
                contexto_agente = ""
                if usar_contexto_agente and st.session_state.get('agente_selecionado'):
                    agente = st.session_state.agente_selecionado
                    contexto_agente = construir_contexto(
                        agente,
                        st.session_state.get('segmentos_selecionados', [])
                    )

                # Construir lista de focos
                focos = []
                if revisar_taxonomia:
                    focos.append("Taxonomia e nomenclatura científica")
                if revisar_dados:
                    focos.append("Dados e estatísticas")
                if revisar_terminologia:
                    focos.append("Terminologia técnica")
                if revisar_coerencia:
                    focos.append("Coerência científica")
                if revisar_clareza:
                    focos.append("Clareza e objetividade")
                if revisar_fontes:
                    focos.append("Verificação de fontes citadas")

                with st.spinner("Realizando revisão técnica..."):
                    try:
                        # Prompt de revisão
                        prompt = f"""
                        {contexto_agente}

                        ## TEXTO ORIGINAL PARA REVISÃO TÉCNICA:
                        {texto_tecnico}

                        ## CONFIGURAÇÕES:
                        - Tipo de conteúdo: {tipo_conteudo}
                        - Tom de voz: {tom_voz}
                        - Nível de rigor: {nivel_rigor}

                        ## FOCOS DA REVISÃO:
                        {chr(10).join([f"- {foco}" for foco in focos])}

                        ## INSTRUÇÕES ADICIONAIS:
                        {instrucoes_adicionais if instrucoes_adicionais else "Nenhuma instrução adicional"}

                        ## TAREFA:
                        Realize uma revisão técnica completa do texto, focando nos aspectos selecionados.

                        {"## FORMATO DE SAÍDA:" if incluir_relatorio else ""}
                        {'''
                        ### 📝 TEXTO REVISADO
                        [Texto completo revisado]

                        ### 🔍 RELATÓRIO DE MUDANÇAS

                        #### 📊 RESUMO
                        - Total de correções: [N]
                        - Categorias: [lista]
                        - Impacto na precisão: [Alto/Médio/Baixo]

                        #### 📋 MUDANÇAS DETALHADAS
                        Para cada correção:
                        - **Original:** "texto original"
                        - **Corrigido:** "texto corrigido"
                        - **Justificativa:** explicação técnica

                        #### 🎯 RECOMENDAÇÕES
                        [Sugestões adicionais para melhorar o texto]
                        ''' if incluir_relatorio else '''
                        Retorne APENAS o texto revisado, sem comentários ou explicações.
                        '''}

                        ## CORREÇÕES OBRIGATÓRIAS:
                        1. **TAXONOMIA:** Validar nomes científicos, corrigir classificações
                        2. **DADOS:** Verificar consistência de números e estatísticas
                        3. **TERMINOLOGIA:** Usar termos técnicos precisos
                        4. **COERÊNCIA:** Garantir precisão científica
                        5. **CLAREZA:** Melhorar comunicação sem perder rigor técnico

                        Mantenha a estrutura original do texto.
                        """

                        resposta = modelo_texto.generate_content(prompt)
                        resultado = resposta.text

                        # Processar resultado
                        if incluir_relatorio:
                            if "### 📝 TEXTO REVISADO" in resultado and "### 🔍 RELATÓRIO" in resultado:
                                partes = resultado.split("### 🔍 RELATÓRIO")
                                texto_revisado = partes[0].replace("### 📝 TEXTO REVISADO", "").strip()
                                relatorio = "### 🔍 RELATÓRIO" + partes[1]
                            else:
                                texto_revisado = resultado
                                relatorio = "Relatório não gerado automaticamente."
                        else:
                            texto_revisado = resultado
                            relatorio = None

                        # Exibir resultado
                        with col_revisado:
                            revisado_placeholder.empty()
                            st.success("✅ Revisão concluída!")

                            if incluir_relatorio and relatorio:
                                tab1, tab2 = st.tabs(["📝 Texto Revisado", "📋 Relatório"])
                                with tab1:
                                    st.text_area(
                                        "Texto revisado:",
                                        texto_revisado,
                                        height=350,
                                        label_visibility="collapsed"
                                    )
                                with tab2:
                                    st.markdown(relatorio)
                            else:
                                st.text_area(
                                    "Texto revisado:",
                                    texto_revisado,
                                    height=350,
                                    label_visibility="collapsed"
                                )

                        # Estatísticas
                        st.subheader("📊 Estatísticas")
                        palavras_orig = len(texto_tecnico.split())
                        palavras_rev = len(texto_revisado.split())

                        col_s1, col_s2, col_s3 = st.columns(3)
                        with col_s1:
                            st.metric("Palavras Original", palavras_orig)
                        with col_s2:
                            st.metric("Palavras Revisado", palavras_rev)
                        with col_s3:
                            diff = palavras_rev - palavras_orig
                            st.metric("Diferença", f"{'+' if diff > 0 else ''}{diff}")

                        # Download
                        st.download_button(
                            "💾 Baixar Texto Revisado",
                            data=texto_revisado,
                            file_name="texto_revisado.txt",
                            mime="text/plain"
                        )

                    except Exception as e:
                        st.error(f"❌ Erro na revisão: {str(e)}")
            else:
                st.warning("Por favor, cole um conteúdo para revisão.")
