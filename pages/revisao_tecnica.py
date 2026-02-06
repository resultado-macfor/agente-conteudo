"""
Página de Revisão Técnica com RAGs Especializados.
Revisão técnica de conteúdo agrícola usando base de conhecimento vetorial.
"""
import streamlit as st
from typing import List, Dict
from config.settings import ASTRA_DB_COLLECTION
from database import modelo_texto, astra_client
from models import construir_contexto
from services import get_embedding


def render():
    """Renderiza a aba de revisão técnica."""
    st.header("🔧 Revisão Técnica com RAGs Especializados")
    st.markdown("**Análise em camadas: taxonomia, epidemiologia, produtos + reescrita final**")

    # Layout com duas colunas
    col_original, col_revisado = st.columns(2)

    with col_original:
        st.subheader("📄 Conteúdo Original")
        texto_tecnico = st.text_area(
            "Cole o conteúdo técnico para revisão:",
            height=300,
            placeholder="Cole aqui o conteúdo técnico agrícola...",
            key="texto_tecnico_rag",
            label_visibility="collapsed"
        )

    with col_revisado:
        st.subheader("✨ Conteúdo Revisado com RAG")
        revisado_placeholder = st.empty()
        revisado_placeholder.info("📝 Aguardando revisão... O conteúdo revisado aparecerá aqui.")

    # Configurações
    st.markdown("---")
    st.subheader("⚙️ Configurações da Revisão")

    col_config1, col_config2, col_config3 = st.columns([2, 1, 1])

    with col_config1:
        tipo_conteudo = st.selectbox(
            "Tipo de Conteúdo:",
            ["Artigo Técnico", "Material Comercial", "Blog Post", "Manual Técnico", "Comunicado Técnico"]
        )

    with col_config2:
        st.subheader("🔍 RAGs Especializados")
        rag_taxonomia = st.checkbox("RAG Taxonomia", value=True)
        rag_epidemiologia = st.checkbox("RAG Epidemiologia", value=True)
        rag_produtos = st.checkbox("RAG Produtos", value=True)
        rag_geral = st.checkbox("RAG Geral", value=True)

    with col_config3:
        st.subheader("⚙️ Configurações")
        nivel_rigor = st.select_slider("Nível de Rigor:", ["Leve", "Moderado", "Rigoroso", "Especialista"])
        limite_documentos = st.number_input("Docs por RAG", min_value=3, max_value=20, value=12)
        usar_contexto_agente = st.checkbox("Usar contexto do agente", value=bool(st.session_state.get('agente_selecionado')))
        incluir_relatorio = st.checkbox("📋 Incluir relatório de mudanças", value=True)

    # Funções RAG
    def realizar_rag_especializado(texto: str, perguntas: list, limite: int = 12) -> List[Dict]:
        """RAG com perguntas específicas"""
        documentos_combinados = []
        for pergunta in perguntas:
            query = f"{texto[:200]} {pergunta}"
            embedding = get_embedding(query)
            documentos = astra_client.vector_search(ASTRA_DB_COLLECTION, embedding, limit=limite // len(perguntas))
            documentos_combinados.extend(documentos)

        # Remover duplicados
        documentos_unicos = []
        ids_vistos = set()
        for doc in documentos_combinados:
            doc_id = str(doc.get('_id', ''))
            if doc_id not in ids_vistos:
                documentos_unicos.append(doc)
                ids_vistos.add(doc_id)
        return documentos_unicos[:limite]

    def realizar_rag_taxonomia(texto: str, limite: int = 12) -> List[Dict]:
        perguntas = [
            "classificação taxonômica", "fungo ou oomiceto", "nome científico patógeno",
            "agente causal doença", "oomiceto vs fungo diferença"
        ]
        return realizar_rag_especializado(texto, perguntas, limite)

    def realizar_rag_epidemiologia(texto: str, limite: int = 12) -> List[Dict]:
        perguntas = [
            "condições ambientais doença", "temperatura umidade molhamento foliar",
            "condições ideais infecção", "epidemiologia doença plantas"
        ]
        return realizar_rag_especializado(texto, perguntas, limite)

    def realizar_rag_produtos(texto: str, limite: int = 12) -> List[Dict]:
        perguntas = [
            "modo de ação produto", "aplicação dose recomendada",
            "eficácia controle doença", "características técnicas produto"
        ]
        return realizar_rag_especializado(texto, perguntas, limite)

    def realizar_rag_geral(texto: str, limite: int = 12) -> List[Dict]:
        embedding = get_embedding(texto[:800])
        return astra_client.vector_search(ASTRA_DB_COLLECTION, embedding, limit=limite)

    def processar_rags(texto: str, rags_ativos: dict, limite: int = 12) -> dict:
        resultados = {}
        if rags_ativos.get('taxonomia'):
            with st.spinner("🔬 Buscando informações de taxonomia..."):
                resultados['taxonomia'] = realizar_rag_taxonomia(texto, limite)
        if rags_ativos.get('epidemiologia'):
            with st.spinner("🌡️ Buscando informações epidemiológicas..."):
                resultados['epidemiologia'] = realizar_rag_epidemiologia(texto, limite)
        if rags_ativos.get('produtos'):
            with st.spinner("🧪 Buscando informações de produtos..."):
                resultados['produtos'] = realizar_rag_produtos(texto, limite)
        if rags_ativos.get('geral'):
            with st.spinner("📚 Buscando informações gerais..."):
                resultados['geral'] = realizar_rag_geral(texto, limite)
        return resultados

    def reescrever_com_relatorio(texto_original: str, resultados_rags: dict, contexto_agente: str = "") -> tuple:
        # Construir contexto dos RAGs
        contexto_rags = "## DOCUMENTOS TÉCNICOS DE REFERÊNCIA:\n\n"
        for categoria, documentos in resultados_rags.items():
            if documentos:
                contexto_rags += f"### {categoria.upper()} ({len(documentos)} docs):\n"
                for doc in documentos:
                    doc_content = str(doc)
                    doc_limpo = doc_content.replace('{', '').replace('}', '').replace("'", "").replace('"', '')[:300]
                    contexto_rags += f"- {doc_limpo}...\n"
                contexto_rags += "\n"

        prompt_reescrita = f"""
        {contexto_agente}

        ## TEXTO ORIGINAL:
        {texto_original}

        ## BASE TÉCNICA:
        {contexto_rags}

        ## TAREFA:
        1. Reescrever aplicando correções técnicas
        2. Gerar relatório detalhado das mudanças

        **FORMATO DE SAÍDA:**

        ### 📝 TEXTO REESCRITO
        [Texto completo reescrito]

        ### 🔍 RELATÓRIO DE MUDANÇAS

        #### 📊 RESUMO
        - Total de correções: [N]
        - Categorias: [lista]

        #### 📋 MUDANÇAS DETALHADAS
        **CORREÇÕES TAXONÔMICAS:**
        - Original: "..."
        - Corrigido: "..."
        - Justificativa: ...

        **PRECISÃO EPIDEMIOLÓGICA:**
        [Lista de correções]

        **INFORMAÇÕES DE PRODUTOS:**
        [Lista de correções]

        **CORREÇÕES OBRIGATÓRIAS:**
        - Precisão taxonômica (fungo vs oomiceto)
        - Especificidade epidemiológica
        - Informações de produtos
        - Terminologia técnica

        Mantenha a estrutura original.
        """

        try:
            resposta = modelo_texto.generate_content(prompt_reescrita)
            texto_completo = resposta.text

            if "### 📝 TEXTO REESCRITO" in texto_completo and "### 🔍 RELATÓRIO" in texto_completo:
                partes = texto_completo.split("### 🔍 RELATÓRIO")
                texto_reescrito = partes[0].replace("### 📝 TEXTO REESCRITO", "").strip()
                relatorio = "### 🔍 RELATÓRIO" + partes[1]
            else:
                texto_reescrito = texto_completo
                relatorio = "Relatório não gerado automaticamente."

            return texto_reescrito, relatorio

        except Exception as e:
            st.error(f"Erro na reescrita: {str(e)}")
            return texto_original, f"Erro: {str(e)}"

    # Botão de revisão
    st.markdown("---")
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])

    with col_btn2:
        if st.button("🔬 Realizar Revisão com RAGs", type="primary", use_container_width=True):
            if texto_tecnico:
                rags_ativos = {
                    'taxonomia': rag_taxonomia,
                    'epidemiologia': rag_epidemiologia,
                    'produtos': rag_produtos,
                    'geral': rag_geral
                }

                contexto_agente = ""
                if usar_contexto_agente and st.session_state.get('agente_selecionado'):
                    agente = st.session_state.agente_selecionado
                    contexto_agente = construir_contexto(agente, st.session_state.get('segmentos_selecionados', []))

                with st.spinner("🚀 Executando pipeline de RAGs..."):
                    try:
                        # Fase 1: RAGs
                        st.subheader("📡 Fase 1: Busca com RAGs")
                        resultados_rags = processar_rags(texto_tecnico, rags_ativos, limite_documentos)

                        # Métricas
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("RAG Taxonomia", len(resultados_rags.get('taxonomia', [])))
                        with col2:
                            st.metric("RAG Epidemiologia", len(resultados_rags.get('epidemiologia', [])))
                        with col3:
                            st.metric("RAG Produtos", len(resultados_rags.get('produtos', [])))
                        with col4:
                            st.metric("RAG Geral", len(resultados_rags.get('geral', [])))

                        # Fase 2: Reescrita
                        st.subheader("✍️ Fase 2: Reescrita")
                        with st.spinner("Reescrevendo..."):
                            texto_reescrito, relatorio = reescrever_com_relatorio(
                                texto_tecnico, resultados_rags, contexto_agente
                            )

                        # Resultados
                        st.subheader("📋 Resultados")

                        with col_revisado:
                            revisado_placeholder.empty()
                            st.success("✅ Conteúdo revisado!")

                            if incluir_relatorio:
                                tab1, tab2 = st.tabs(["📝 Texto Reescrito", "📋 Relatório"])
                                with tab1:
                                    st.text_area("Texto revisado:", texto_reescrito, height=300, label_visibility="collapsed")
                                with tab2:
                                    st.markdown(relatorio)
                            else:
                                st.text_area("Texto revisado:", texto_reescrito, height=300, label_visibility="collapsed")

                        # Estatísticas
                        palavras_orig = len(texto_tecnico.split())
                        palavras_rev = len(texto_reescrito.split())
                        col_s1, col_s2, col_s3 = st.columns(3)
                        with col_s1:
                            st.metric("Palavras Original", palavras_orig)
                        with col_s2:
                            st.metric("Palavras Reescrito", palavras_rev)
                        with col_s3:
                            diff = palavras_rev - palavras_orig
                            st.metric("Diferença", f"{'+' if diff > 0 else ''}{diff}")

                    except Exception as e:
                        st.error(f"❌ Erro: {str(e)}")
            else:
                st.warning("Por favor, cole um conteúdo para revisão.")
