import datetime
import streamlit as st
from utils.content_utils import construir_contexto
from agent.rag import (
    processar_rags_especializados,
    reescrever_com_relatorio_mudancas,
    reescrever_sem_relatorio,
)


def render(tab, modelo_texto, modelo_texto2, db=None):
    with tab:
        st.header("🔧 Revisão Técnica com RAGs Especializados")
        st.markdown("**Análise em camadas: taxonomia, epidemiologia, produtos + reescrita final com relatório detalhado**")

        col_original_rag, col_revisado_rag = st.columns(2)

        with col_original_rag:
            st.subheader("📄 Conteúdo Original")
            texto_tecnico = st.text_area(
                "Cole o conteúdo técnico para revisão:",
                height=300,
                placeholder="Cole aqui o conteúdo técnico agrícola que precisa ser revisado...",
                key="texto_tecnico_rag",
                label_visibility="collapsed",
            )

        with col_revisado_rag:
            st.subheader("✨ Conteúdo Revisado com RAG")
            revisado_rag_placeholder = st.empty()
            revisado_rag_placeholder.info("📝 Aguardando revisão com RAG... O conteúdo revisado aparecerá aqui.")

        st.markdown("---")
        st.subheader("⚙️ Configurações da Revisão")

        col_config1, col_config2, col_config3 = st.columns([2, 1, 1])

        with col_config1:
            tipo_conteudo = st.selectbox(
                "Tipo de Conteúdo:",
                ["Artigo Técnico", "Material Comercial", "Blog Post", "Manual Técnico", "Comunicado Técnico"],
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
            usar_contexto_agente = st.checkbox("Usar contexto do agente",
                                               value=bool(st.session_state.agente_selecionado))
            incluir_relatorio = st.checkbox("📋 Incluir relatório de mudanças", value=True)

        st.markdown("---")
        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])

        with col_btn2:
            if st.button("🔬 Realizar Revisão com RAGs Especializados", type="primary", use_container_width=True):
                if not texto_tecnico:
                    st.warning("Por favor, cole um conteúdo técnico para revisão.")
                    return

                rags_ativos = {
                    'taxonomia': rag_taxonomia,
                    'epidemiologia': rag_epidemiologia,
                    'produtos': rag_produtos,
                    'geral': rag_geral,
                }

                contexto_agente = ""
                if usar_contexto_agente and st.session_state.agente_selecionado:
                    agente = st.session_state.agente_selecionado
                    contexto_agente = construir_contexto(agente, st.session_state.segmentos_selecionados)

                with st.spinner("🚀 Executando pipeline de RAGs especializados..."):
                    try:
                        st.subheader("📡 Fase 1: Busca com RAGs Especializados")
                        resultados_rags = processar_rags_especializados(texto_tecnico, rags_ativos, limite_documentos)

                        col_rag1, col_rag2, col_rag3, col_rag4 = st.columns(4)
                        with col_rag1:
                            st.metric("RAG Taxonomia", len(resultados_rags.get('taxonomia', [])))
                        with col_rag2:
                            st.metric("RAG Epidemiologia", len(resultados_rags.get('epidemiologia', [])))
                        with col_rag3:
                            st.metric("RAG Produtos", len(resultados_rags.get('produtos', [])))
                        with col_rag4:
                            st.metric("RAG Geral", len(resultados_rags.get('geral', [])))

                        st.subheader("✍️ Fase 2: Reescrita com Base nos RAGs")

                        with st.spinner("Reescrevendo conteúdo e gerando relatório de mudanças..."):
                            if incluir_relatorio:
                                texto_reescrito, relatorio_mudancas = reescrever_com_relatorio_mudancas(
                                    texto_tecnico, resultados_rags, modelo_texto, contexto_agente
                                )
                            else:
                                texto_reescrito = reescrever_sem_relatorio(
                                    texto_tecnico, resultados_rags, modelo_texto, contexto_agente
                                )
                                relatorio_mudancas = None

                        st.subheader("📋 Fase 3: Resultados da Revisão")

                        with col_revisado_rag:
                            revisado_rag_placeholder.empty()
                            st.success("✅ Conteúdo revisado com RAGs!")

                            if incluir_relatorio and relatorio_mudancas:
                                tab_texto_reescrito, tab_relatorio_mudancas, tab_analise = st.tabs([
                                    "📝 Texto Reescrito", "📋 Relatório de Mudanças", "📊 Análise RAGs"
                                ])

                                with tab_texto_reescrito:
                                    st.text_area("Texto reescrito:", texto_reescrito, height=300,
                                                 label_visibility="collapsed")

                                with tab_relatorio_mudancas:
                                    st.markdown(relatorio_mudancas)

                                with tab_analise:
                                    palavras_orig = len(texto_tecnico.split())
                                    palavras_reesc = len(texto_reescrito.split())
                                    diff_palavras = palavras_reesc - palavras_orig
                                    col_s1, col_s2, col_s3 = st.columns(3)
                                    with col_s1:
                                        st.metric("Palavras Original", palavras_orig)
                                    with col_s2:
                                        st.metric("Palavras Reescrito", palavras_reesc)
                                    with col_s3:
                                        st.metric("Diferença",
                                                  f"{'+' if diff_palavras > 0 else ''}{diff_palavras}",
                                                  delta=f"{diff_palavras/palavras_orig*100:.1f}%" if palavras_orig > 0 else "0%")
                                    st.markdown("### 📊 Estatísticas dos RAGs")
                                    for categoria, documentos in resultados_rags.items():
                                        if documentos:
                                            st.write(f"**{categoria.capitalize()}:** {len(documentos)} documentos")
                            else:
                                st.text_area("Texto reescrito:", texto_reescrito, height=300,
                                             label_visibility="collapsed")

                        st.markdown("---")
                        col_dl1, col_dl2, col_dl3 = st.columns(3)

                        with col_dl1:
                            st.download_button(
                                "💾 Baixar Texto Reescrito",
                                data=texto_reescrito,
                                file_name=f"texto_reescrito_rags_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                                mime="text/plain",
                                use_container_width=True,
                            )

                        with col_dl2:
                            if incluir_relatorio and relatorio_mudancas:
                                st.download_button(
                                    "💾 Baixar Relatório",
                                    data=relatorio_mudancas,
                                    file_name=f"relatorio_mudancas_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.md",
                                    mime="text/markdown",
                                    use_container_width=True,
                                )

                        with col_dl3:
                            pacote = f"TEXTO ORIGINAL:\n{texto_tecnico}\n\n{'='*60}\n\nTEXTO REESCRITO COM RAGs:\n{texto_reescrito}"
                            if incluir_relatorio and relatorio_mudancas:
                                pacote += f"\n\n{'='*60}\n\nRELATÓRIO DE MUDANÇAS:\n{relatorio_mudancas}"
                            st.download_button(
                                "📦 Baixar Pacote Completo",
                                data=pacote,
                                file_name=f"revisao_completa_rags_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                                mime="text/plain",
                                use_container_width=True,
                            )

                        if db is not None:
                            try:
                                revisao_data = {
                                    "texto_original": texto_tecnico,
                                    "texto_reescrito": texto_reescrito,
                                    "relatorio_mudancas": relatorio_mudancas if incluir_relatorio else "Não gerado",
                                    "rags_utilizados": rags_ativos,
                                    "documentos_encontrados": {k: len(v) for k, v in resultados_rags.items()},
                                    "nivel_rigor": nivel_rigor,
                                    "incluiu_relatorio": incluir_relatorio,
                                    "data_criacao": datetime.datetime.now(),
                                }
                                if 'revisoes_rags' not in db.list_collection_names():
                                    db.create_collection('revisoes_rags')
                                db['revisoes_rags'].insert_one(revisao_data)
                                st.success("✅ Revisão salva no histórico!")
                            except Exception as e:
                                st.warning(f"Revisão concluída, mas não salva: {str(e)}")

                    except Exception as e:
                        st.error(f"❌ Erro no pipeline de RAGs: {str(e)}")
                        with col_revisado_rag:
                            revisado_rag_placeholder.error(f"❌ Erro: {str(e)}")

        if 'ultima_revisao' in st.session_state:
            st.markdown("---")
            st.subheader("🔄 Ajustes Incrementais para RAGs")
            st.info("Use o campo abaixo para solicitar ajustes específicos na última revisão com RAGs.")

            comando_ajuste_rag = st.text_area(
                "Comandos para ajustar a revisão RAG:",
                height=150,
                placeholder="Exemplos:\n- Aumente o foco na taxonomia dos patógenos\n- Inclua mais informações epidemiológicas",
                key="comando_ajuste_rag",
            )

            if st.button("🔄 Ajustar Revisão RAG", type="secondary", use_container_width=True):
                if comando_ajuste_rag:
                    with st.spinner("🔄 Aplicando ajustes na revisão RAG..."):
                        try:
                            prompt_ajuste_rag = f"""
                            VOCÊ É: Um especialista técnico agrícola.
                            SUA TAREFA: Ajustar a revisão técnica anterior com base nas solicitações específicas.

                            TEXTO ORIGINAL:
                            {texto_tecnico}

                            TEXTO REESCRITO COM RAGs:
                            {st.session_state.ultima_revisao}

                            SOLICITAÇÕES DE AJUSTE:
                            {comando_ajuste_rag}

                            INSTRUÇÕES:
                            1. Aplique TODOS os ajustes solicitados
                            2. Mantenha a precisão técnica
                            3. Retorne o texto reescrito ajustado.
                            """

                            resposta_ajuste_rag = modelo_texto2.generate_content(prompt_ajuste_rag)
                            texto_reescrito_ajustado = resposta_ajuste_rag.text

                            st.session_state.ultima_revisao = texto_reescrito_ajustado
                            st.success("✅ Revisão RAG ajustada!")
                            st.text_area("Texto reescrito ajustado:", texto_reescrito_ajustado, height=300,
                                         label_visibility="collapsed")

                            st.download_button(
                                "💾 Baixar Versão Ajustada",
                                data=texto_reescrito_ajustado,
                                file_name=f"revisao_rag_ajustada_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                                mime="text/plain",
                                use_container_width=True,
                            )

                        except Exception as e:
                            st.error(f"❌ Erro ao ajustar revisão RAG: {str(e)}")
