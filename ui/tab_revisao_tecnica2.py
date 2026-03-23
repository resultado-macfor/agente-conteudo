import datetime
import streamlit as st


def render(tab, modelo_texto2):
    with tab:
        st.header("🔬 Revisão Técnica Completa")
        st.markdown("**Análise rigorosa com expertise técnica em agronomia**")

        col_original, col_revisado = st.columns(2)

        with col_original:
            st.subheader("📄 Conteúdo Original")
            texto_tecnico = st.text_area(
                "Cole o conteúdo técnico agrícola para revisão:",
                height=300,
                placeholder="Cole aqui qualquer conteúdo agrícola que precisa ser revisado tecnicamente...",
                key="texto_tecnico_original",
                label_visibility="collapsed",
            )

        with col_revisado:
            st.subheader("✨ Conteúdo Revisado")
            revisao_placeholder = st.empty()
            revisao_placeholder.info("📝 Aguardando revisão... O conteúdo revisado aparecerá aqui.")

        st.markdown("---")
        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])

        with col_btn2:
            if st.button(
                "🔬 Realizar Revisão Técnica Completa",
                type="primary",
                key="revisao_inicial",
                use_container_width=True,
            ):
                if not texto_tecnico:
                    st.warning("Por favor, cole um conteúdo técnico para revisão.")
                else:
                    with st.spinner("🔍 Analisando conteúdo com rigor técnico..."):
                        try:
                            prompt_revisao = f"""
                            VOCÊ É: Um engenheiro agrônomo com ampla experiência técnica.

                            SUA TAREFA: Realizar uma revisão técnica completa do conteúdo fornecido seguindo EXATAMENTE o formato abaixo.

                            ANALISE ESTE CONTEÚDO:
                            {texto_tecnico}

                            RETORNE APENAS ESTE FORMATO EXATO:

                            ✅ O QUE ESTÁ CORRETO NO TEXTO (visão geral)
                            Antes das correções, é importante destacar que o texto está bem escrito, com boa estrutura, e a maior parte das informações está correta:
                            [Liste aqui os pontos que estão corretos em bullet points]
                            Ou seja: o conteúdo é bom, faltando apenas alguns ajustes e correções pontuais.

                            ❗ PONTOS INCORRETOS, IMPRECISOS OU QUE PRECISAM SER AJUSTADOS
                            Abaixo, estão todos os erros e imprecisões técnicas do texto, com explicação e sugestão.

                            ❌ 1. [Título do primeiro erro]
                            No trecho:
                            "[Citação exata do trecho problemático]"
                            Correção técnica:
                            [Explicação detalhada do erro]
                            ➡ Portanto, [conclusão técnica]
                            Como corrigir:
                            "[Sugestão de texto corrigido]"

                            [Continue numerando para cada erro encontrado...]

                            🧪 CONCLUSÃO TÉCNICA
                            O texto está bem escrito e majoritariamente correto, mas contém:
                            ✔ [X] erro(s) crítico(s)
                            ✔ [Y] afirmações que precisam correção ou moderação
                            ✔ [Z] pontos que não estão errados, mas precisam maior precisão
                            ✔ [W] pontos incompletos (não são erros, mas faltam informações-chave)

                            🔧 Se quiser, posso agora:
                            - Reescrever o texto totalmente revisado e técnico, já corrigido
                            - Criar uma versão mais curta para redes sociais
                            - Criar uma versão para material comercial

                            Seja direto e técnico. Mantenha o formato exato.
                            """

                            resposta = modelo_texto2.generate_content(prompt_revisao)
                            revisao_completa = resposta.text

                            st.session_state.ultima_revisao = revisao_completa
                            st.session_state.texto_original_revisao = texto_tecnico

                            with col_revisado:
                                revisao_placeholder.empty()
                                st.success("✅ Revisão concluída!")

                                tab_relatorio, tab_texto_corrigido = st.tabs(
                                    ["📋 Relatório Completo", "📝 Texto Corrigido"]
                                )

                                with tab_relatorio:
                                    st.markdown(revisao_completa)

                                with tab_texto_corrigido:
                                    st.info("📝 **Texto revisado com correções aplicadas:**")
                                    texto_corrigido_final = _aplicar_correcoes(texto_tecnico, revisao_completa)

                                    if texto_corrigido_final == texto_tecnico:
                                        st.warning("⚠️ Não foi possível extrair automaticamente o texto corrigido.")
                                        st.markdown(revisao_completa)
                                    else:
                                        st.text_area(
                                            "Texto com correções aplicadas:",
                                            texto_corrigido_final,
                                            height=300,
                                            label_visibility="collapsed",
                                        )

                            st.markdown("---")
                            col_dl1, col_dl2 = st.columns(2)

                            with col_dl1:
                                st.download_button(
                                    "💾 Baixar Relatório Completo",
                                    data=revisao_completa,
                                    file_name=f"revisao_tecnica_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                                    mime="text/plain",
                                    use_container_width=True,
                                )

                            with col_dl2:
                                texto_corrigido_final = _aplicar_correcoes(texto_tecnico, revisao_completa)
                                st.download_button(
                                    "💾 Baixar Texto Corrigido",
                                    data=texto_corrigido_final,
                                    file_name=f"texto_corrigido_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                                    mime="text/plain",
                                    use_container_width=True,
                                )

                        except Exception as e:
                            st.error(f"❌ Erro na revisão técnica: {str(e)}")
                            with col_revisado:
                                revisao_placeholder.error(f"❌ Erro: {str(e)}")

        if 'ultima_revisao' in st.session_state:
            st.markdown("---")
            st.subheader("🔄 Ajustes Incrementais")
            st.info("Use o campo abaixo para solicitar ajustes específicos na última revisão realizada.")

            comando_ajuste = st.text_area(
                "Comandos para ajustar a última revisão:",
                height=150,
                placeholder="Exemplos:\n- Foque mais na adubação nitrogenada\n- Adicione informações sobre irrigação\n- Corrija os termos técnicos sobre pragas",
                key="comando_ajuste_rev2",
            )

            if st.button("🔄 Revisar Novamente com Ajustes", type="secondary", use_container_width=True):
                if not comando_ajuste:
                    st.warning("Por favor, digite os comandos de ajuste desejados.")
                else:
                    with st.spinner("🔄 Aplicando ajustes solicitados..."):
                        try:
                            prompt_ajuste = f"""
                            VOCÊ É: Um engenheiro agrônomo com ampla experiência técnica.

                            SUA TAREFA: Revisar e ajustar o relatório técnico anterior com base nas solicitações específicas do usuário.

                            RELATÓRIO TÉCNICO ANTERIOR:
                            {st.session_state.ultima_revisao}

                            TEXTO ORIGINAL ANALISADO:
                            {st.session_state.texto_original_revisao}

                            SOLICITAÇÕES DE AJUSTE DO USUÁRIO:
                            {comando_ajuste}

                            INSTRUÇÕES:
                            1. Mantenha o MESMO FORMATO EXATO do relatório anterior
                            2. Aplique TODOS os ajustes solicitados pelo usuário
                            3. Mantenha a qualidade técnica e rigor científico

                            RETORNE APENAS O RELATÓRIO REVISADO NO MESMO FORMATO, SEM COMENTÁRIOS ADICIONAIS.
                            """

                            resposta_ajuste = modelo_texto2.generate_content(prompt_ajuste)
                            revisao_ajustada = resposta_ajuste.text

                            st.session_state.ultima_revisao = revisao_ajustada
                            st.success("✅ Revisão ajustada concluída!")

                            tab_rel_aj, tab_txt_aj = st.tabs(["📋 Relatório Ajustado", "📝 Texto Corrigido"])

                            with tab_rel_aj:
                                st.markdown(revisao_ajustada)

                            with tab_txt_aj:
                                st.info("📝 **Texto revisado com correções aplicadas:**")
                                texto_corrigido_final = _aplicar_correcoes(
                                    st.session_state.texto_original_revisao, revisao_ajustada
                                )

                                if texto_corrigido_final == st.session_state.texto_original_revisao:
                                    st.warning("⚠️ Não foi possível extrair automaticamente o texto corrigido.")
                                    st.markdown(revisao_ajustada)
                                else:
                                    st.text_area(
                                        "Texto com correções aplicadas:",
                                        texto_corrigido_final,
                                        height=300,
                                        label_visibility="collapsed",
                                    )

                            st.markdown("---")
                            col_dl1, col_dl2 = st.columns(2)

                            with col_dl1:
                                st.download_button(
                                    "💾 Baixar Relatório Ajustado",
                                    data=revisao_ajustada,
                                    file_name=f"revisao_ajustada_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                                    mime="text/plain",
                                    key="download_ajustado_rev2",
                                    use_container_width=True,
                                )

                            with col_dl2:
                                st.download_button(
                                    "💾 Baixar Texto Corrigido",
                                    data=texto_corrigido_final,
                                    file_name=f"texto_corrigido_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                                    mime="text/plain",
                                    use_container_width=True,
                                )

                        except Exception as e:
                            st.error(f"❌ Erro ao aplicar ajustes: {str(e)}")


def _aplicar_correcoes(texto_original: str, revisao: str) -> str:
    linhas = revisao.split('\n')
    texto_corrigido = texto_original

    for i, linha in enumerate(linhas):
        if "Como corrigir:" in linha and i + 1 < len(linhas):
            sugestao = linhas[i + 1].strip().strip('"')
            if sugestao:
                for j in range(i - 3, i):
                    if j >= 0 and "No trecho:" in linhas[j] and j + 1 < len(linhas):
                        trecho_original = linhas[j + 1].strip().strip('"')
                        if trecho_original:
                            texto_corrigido = texto_corrigido.replace(trecho_original, sugestao)

    return texto_corrigido
