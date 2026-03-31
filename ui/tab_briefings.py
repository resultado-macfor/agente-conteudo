import datetime
import io
import zipfile
import streamlit as st
from utils.content_utils import construir_contexto


def render(tab, modelo_texto):
    with tab:
        st.header("📋 Gerador de Briefings")

        if not st.session_state.agente_selecionado:
            st.warning("⚠️ Selecione um agente na parte superior do app para usar esta funcionalidade.")
            return

        agente = st.session_state.agente_selecionado
        st.success(f"🎯 Gerando briefings com base no agente: **{agente['nome']}**")

        for key, default in [
            ('briefings_gerados', []),
            ('briefing_atual_selecionado', None),
            ('briefing_em_edicao', None),
        ]:
            if key not in st.session_state:
                st.session_state[key] = default

        modo_entrada = st.radio(
            "Escolha o modo de entrada:",
            ["📅 Upload de Calendário (múltiplos briefings)", "📝 Texto Único (briefing individual)"],
            horizontal=True,
        )

        if modo_entrada == "📅 Upload de Calendário (múltiplos briefings)":
            st.subheader("📅 Gerar Múltiplos Briefings a partir do Calendário")

            col_upload1, col_upload2 = st.columns([2, 1])

            with col_upload1:
                usar_calendario_existente = st.checkbox(
                    "Usar calendário gerado anteriormente",
                    value='calendario_gerado' in st.session_state,
                )

                if not usar_calendario_existente or 'calendario_gerado' not in st.session_state:
                    arquivo_calendario = st.file_uploader("📅 Upload do calendário CSV:", type=['csv'])
                else:
                    st.info("✅ Usando calendário gerado anteriormente")
                    arquivo_calendario = None

            with col_upload2:
                mes_referencia = st.text_input("Mês de referência:", "JANEIRO 2026")
                ano_referencia = st.text_input("Ano de referência:", "2026")

            contexto_briefings = st.text_area(
                "Informações contextuais para orientar a criação dos briefings:",
                placeholder="Exemplo: Foco em campanha de posicionamento de produtos, linguagem técnica mas acessível...",
                height=80,
            )

            if st.button("🔄 Processar Calendário e Gerar Briefings", type="primary", use_container_width=True):
                conteudo_csv = ""

                if usar_calendario_existente and 'calendario_gerado' in st.session_state:
                    conteudo_csv = st.session_state.calendario_gerado
                    st.success("✅ Usando calendário da sessão")
                elif arquivo_calendario is not None:
                    try:
                        file_bytes = arquivo_calendario.getvalue()
                        try:
                            conteudo_csv = file_bytes.decode('utf-8')
                        except UnicodeDecodeError:
                            try:
                                conteudo_csv = file_bytes.decode('latin-1')
                            except UnicodeDecodeError:
                                conteudo_csv = file_bytes.decode('utf-8', errors='ignore')
                        st.success("✅ Arquivo CSV carregado")
                    except Exception as e:
                        st.error(f"❌ Erro ao ler arquivo: {str(e)}")
                        st.stop()
                else:
                    st.error("❌ Nenhum calendário disponível para processar")
                    st.stop()

                with st.spinner("📋 Processando calendário e extraindo pautas..."):
                    try:
                        linhas = conteudo_csv.split('\n')
                        todas_pautas = []

                        for linha_num, linha in enumerate(linhas):
                            linha_limpa = linha.strip().replace('\r', '').replace('\ufeff', '')
                            if not linha_limpa:
                                continue

                            celulas = linha_limpa.split(',')
                            for celula_num, celula in enumerate(celulas):
                                celula_limpa = celula.strip()

                                if (
                                    celula_limpa
                                    and len(celula_limpa) > 15
                                    and not celula_limpa.replace('.', '').isdigit()
                                    and not any(
                                        header in celula_limpa
                                        for header in ['DOMINGO', 'SEGUNDA', 'TERÇA', 'QUARTA', 'QUINTA', 'SEXTA', 'SÁBADO', 'CALENDÁRIO']
                                    )
                                    and 'CX,' not in celula_limpa
                                ):
                                    pautas_na_celula = []

                                    if '\n' in celula_limpa:
                                        for sub_pauta in celula_limpa.split('\n'):
                                            sub_pauta_limpa = sub_pauta.strip()
                                            if sub_pauta_limpa and len(sub_pauta_limpa) > 15:
                                                pautas_na_celula.append(sub_pauta_limpa)
                                    else:
                                        pautas_na_celula.append(celula_limpa)

                                    for pauta in pautas_na_celula:
                                        pauta_limpa = ' '.join(pauta.strip().split())
                                        todas_pautas.append({
                                            'conteudo': pauta_limpa,
                                            'linha': linha_num,
                                            'coluna': celula_num,
                                            'indice': len(todas_pautas) + 1,
                                        })

                        st.success(f"✅ Encontradas {len(todas_pautas)} pautas individuais no calendário")

                        if not todas_pautas:
                            st.error("❌ Nenhuma pauta válida encontrada no CSV")
                            st.stop()

                        with st.expander("👀 Visualizar Pautas Detectadas", expanded=True):
                            st.write(f"**Total de pautas detectadas:** {len(todas_pautas)}")
                            st.write("**Primeiras 10 pautas:**")
                            for i, pauta in enumerate(todas_pautas[:10]):
                                st.write(f"{i+1}. {pauta['conteudo']}")

                        st.subheader("📄 Gerando Briefings para Cada Pauta")

                        contexto_agente = construir_contexto(agente, st.session_state.segmentos_selecionados)
                        briefings_gerados = []
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        for idx, pauta in enumerate(todas_pautas):
                            status_text.text(f"Fazendo briefing da pauta {idx+1}/{len(todas_pautas)}: {pauta['conteudo'][:50]}...")
                            progress_bar.progress((idx + 1) / len(todas_pautas))

                            try:
                                prompt_briefing = _prompt_briefing(
                                    contexto_agente, pauta['conteudo'], mes_referencia, contexto_briefings
                                )
                                resposta = modelo_texto.generate_content(prompt_briefing)
                                briefing_limpo = resposta.text.strip().replace('```', '')

                                briefings_gerados.append({
                                    'indice': idx + 1,
                                    'conteudo_original': pauta['conteudo'],
                                    'briefing': briefing_limpo,
                                    'mes_referencia': mes_referencia,
                                })
                            except Exception as e:
                                st.error(f"❌ Erro ao gerar briefing para pauta {idx+1}: {str(e)}")
                                briefings_gerados.append({
                                    'indice': idx + 1,
                                    'conteudo_original': pauta['conteudo'],
                                    'briefing': f"ERRO: Não foi possível gerar o briefing.\n{str(e)}",
                                    'mes_referencia': mes_referencia,
                                })

                        progress_bar.empty()
                        status_text.empty()

                        st.session_state.briefings_gerados = briefings_gerados
                        st.success(f"✅ {len(briefings_gerados)} briefings gerados com sucesso!")

                    except Exception as e:
                        st.error(f"❌ Erro ao processar calendário: {str(e)}")

        else:
            st.subheader("📝 Gerar Briefing Individual a partir de Texto")

            col_texto1, col_texto2 = st.columns([2, 1])

            with col_texto1:
                titulo_briefing = st.text_input(
                    "Título do briefing:",
                    placeholder="Ex: Lançamento do produto X na cultura Y",
                    key="titulo_briefing_individual",
                )

            with col_texto2:
                mes_referencia_individual = st.text_input(
                    "Mês de referência:",
                    "JANEIRO 2026",
                    key="mes_ref_individual",
                )

            texto_base_briefing = st.text_area(
                "Texto base para gerar o briefing:",
                height=150,
                placeholder="Cole aqui o texto que servirá de base para o briefing. Pode ser uma pauta, um resumo, instruções do cliente, etc.",
                key="texto_base_individual",
            )

            contexto_individual = st.text_area(
                "Contexto adicional (opcional):",
                height=80,
                placeholder="Informações complementares para orientar a criação do briefing...",
                key="contexto_individual",
            )

            col_btn_ind1, col_btn_ind2, col_btn_ind3 = st.columns([1, 2, 1])
            with col_btn_ind2:
                if st.button("📄 GERAR BRIEFING INDIVIDUAL", type="primary", use_container_width=True):
                    if not texto_base_briefing:
                        st.error("❌ O texto base é obrigatório!")
                    elif not titulo_briefing:
                        st.error("❌ O título do briefing é obrigatório!")
                    else:
                        with st.spinner("🔄 Gerando briefing individual..."):
                            try:
                                contexto_agente = construir_contexto(agente, st.session_state.segmentos_selecionados)

                                prompt_briefing_individual = _prompt_briefing_individual(
                                    contexto_agente, titulo_briefing, mes_referencia_individual,
                                    texto_base_briefing, contexto_individual,
                                )

                                resposta = modelo_texto.generate_content(prompt_briefing_individual)
                                briefing_limpo = resposta.text.strip().replace('```', '')

                                novo_briefing = {
                                    'indice': len(st.session_state.briefings_gerados) + 1,
                                    'titulo': titulo_briefing,
                                    'conteudo_original': texto_base_briefing,
                                    'briefing': briefing_limpo,
                                    'mes_referencia': mes_referencia_individual,
                                    'tipo': 'individual',
                                }

                                st.session_state.briefings_gerados.append(novo_briefing)
                                st.session_state.briefing_atual_selecionado = novo_briefing

                                st.success(f"✅ Briefing '{titulo_briefing}' gerado com sucesso!")

                            except Exception as e:
                                st.error(f"❌ Erro ao gerar briefing: {str(e)}")

        if st.session_state.briefings_gerados:
            st.markdown("---")
            st.header("📋 Briefings Gerados")

            briefings = st.session_state.briefings_gerados

            briefing_options = {}
            for b in briefings:
                if 'titulo' in b:
                    label = f"{b['indice']}. {b['titulo']} ({b.get('mes_referencia', 'N/A')})"
                else:
                    label = f"{b['indice']}. {b['conteudo_original'][:60]}... ({b.get('mes_referencia', 'N/A')})"
                briefing_options[label] = b

            if briefing_options:
                col_sel1, col_sel2 = st.columns([3, 1])

                with col_sel1:
                    briefing_selecionado_label = st.selectbox(
                        "Selecione um briefing para visualizar/editar:",
                        list(briefing_options.keys()),
                        key="seletor_briefing_edicao",
                    )

                with col_sel2:
                    if st.button("🔄 Carregar Briefing", key="carregar_briefing"):
                        st.session_state.briefing_atual_selecionado = briefing_options[briefing_selecionado_label]
                        st.session_state.briefing_em_edicao = briefing_options[briefing_selecionado_label]['briefing']
                        st.rerun()

            if st.session_state.briefing_atual_selecionado:
                briefing_atual = st.session_state.briefing_atual_selecionado

                st.markdown("---")
                st.subheader(f"📄 Briefing {briefing_atual['indice']}")

                if 'titulo' in briefing_atual:
                    st.info(f"**Título:** {briefing_atual['titulo']}")
                else:
                    st.info(f"**Pauta original:** {briefing_atual['conteudo_original']}")

                st.write(f"**Mês referência:** {briefing_atual.get('mes_referencia', 'N/A')}")

                st.markdown("---")
                st.subheader("✏️ Ajuste Pontual do Briefing")
                st.markdown("**Mantenha a estrutura - altere apenas o solicitado**")

                col_ajuste1, col_ajuste2 = st.columns([3, 1])

                with col_ajuste1:
                    solicitacao_ajuste_briefing = st.text_area(
                        "Descreva o ajuste desejado:",
                        placeholder="Exemplos:\n- Adicione mais detalhes sobre o público-alvo\n- Inclua informações sobre o produto X na seção de produtos",
                        height=100,
                        key="ajuste_briefing",
                    )

                with col_ajuste2:
                    st.markdown("#####")
                    if st.button("✅ APLICAR AJUSTE", key="aplicar_ajuste_briefing", use_container_width=True):
                        if solicitacao_ajuste_briefing.strip():
                            with st.spinner("🔄 Aplicando ajuste pontual ao briefing..."):
                                try:
                                    contexto_agente = construir_contexto(agente, st.session_state.segmentos_selecionados)

                                    prompt_ajuste_briefing = f"""
                                    {contexto_agente}

                                    ## INSTRUÇÕES: AJUSTE PONTUAL DO BRIEFING
                                    ## MANTENHA A ESTRUTURA ORIGINAL - ALTERE APENAS O SOLICITADO

                                    ### BRIEFING ORIGINAL COMPLETO:
                                    {briefing_atual['briefing']}

                                    ### SOLICITAÇÃO ESPECÍFICA DE AJUSTE:
                                    "{solicitacao_ajuste_briefing}"

                                    ## INFORMAÇÕES DE CONTEXTO:
                                    **Título/Pauta original:** {briefing_atual.get('titulo', briefing_atual.get('conteudo_original', 'N/A'))}
                                    **Mês de referência:** {briefing_atual.get('mes_referencia', 'N/A')}

                                    ## REGRAS ABSOLUTAS:
                                    1. MANTENHA A ESTRUTURA ORIGINAL COMPLETA - NÃO remova seções, NÃO adicione novas seções
                                    2. ALTERE APENAS O ESTRITAMENTE SOLICITADO
                                    3. PRESERVE FORMATAÇÃO E ESTILO

                                    RETORNE APENAS O BRIEFING AJUSTADO, SEM COMENTÁRIOS ADICIONAIS.
                                    """

                                    resposta_ajuste = modelo_texto.generate_content(prompt_ajuste_briefing)
                                    briefing_ajustado = resposta_ajuste.text.replace('```', '')

                                    briefing_atual['briefing'] = briefing_ajustado
                                    briefing_atual.setdefault('historico_ajustes', []).append({
                                        'data': datetime.datetime.now(),
                                        'solicitacao': solicitacao_ajuste_briefing,
                                    })

                                    st.session_state.briefing_em_edicao = briefing_ajustado
                                    st.success("✅ Ajuste aplicado com sucesso!")
                                    st.rerun()

                                except Exception as e:
                                    st.error(f"❌ Erro ao aplicar ajuste: {str(e)}")
                        else:
                            st.warning("⚠️ Por favor, descreva o ajuste desejado.")

                briefing_para_mostrar = (
                    st.session_state.briefing_em_edicao
                    if st.session_state.briefing_em_edicao
                    else briefing_atual['briefing']
                )

                briefing_editado = st.text_area(
                    "📝 Conteúdo do Briefing (você pode editar diretamente):",
                    value=briefing_para_mostrar,
                    height=400,
                    key="editor_briefing_direto",
                )

                col_save1, col_save2, col_save3 = st.columns([1, 1, 2])

                with col_save1:
                    if st.button("💾 Salvar Edições Diretas", type="primary", use_container_width=True):
                        if briefing_editado != briefing_atual['briefing']:
                            briefing_atual['briefing'] = briefing_editado
                            briefing_atual.setdefault('historico_ajustes', []).append({
                                'data': datetime.datetime.now(),
                                'solicitacao': 'Edição direta no editor',
                            })
                            st.session_state.briefing_em_edicao = briefing_editado
                            st.success("✅ Briefing atualizado com sucesso!")
                            st.rerun()

                with col_save2:
                    if st.button("🔄 Restaurar Original", use_container_width=True):
                        briefing_atual['briefing'] = briefing_atual.get('briefing_original', briefing_atual['briefing'])
                        st.session_state.briefing_em_edicao = None
                        st.success("✅ Briefing original restaurado!")
                        st.rerun()

                if briefing_atual.get('historico_ajustes'):
                    with st.expander("📋 Histórico de Ajustes Realizados"):
                        for i, ajuste in enumerate(briefing_atual['historico_ajustes']):
                            data_ajuste = ajuste.get('data', '')
                            data_str = (
                                data_ajuste.strftime('%d/%m/%Y %H:%M:%S')
                                if isinstance(data_ajuste, datetime.datetime)
                                else 'Data desconhecida'
                            )
                            st.write(f"**{i+1}. {data_str}**")
                            st.write(f"*Solicitação:* {ajuste['solicitacao']}")
                            st.divider()

                st.markdown("---")
                col_dl1, col_dl2, col_dl3 = st.columns(3)

                with col_dl1:
                    nome_arquivo = f"briefing_{briefing_atual['indice']}.txt"
                    if 'titulo' in briefing_atual:
                        nome_arquivo = f"briefing_{briefing_atual['titulo'].replace(' ', '_')}.txt"
                    st.download_button(
                        "💾 Baixar Este Briefing",
                        data=briefing_atual['briefing'],
                        file_name=nome_arquivo,
                        mime="text/plain",
                        use_container_width=True,
                    )

                with col_dl2:
                    if briefing_atual.get('historico_ajustes'):
                        briefing_com_historico = (
                            f"# BRIEFING {briefing_atual['indice']}\n\n"
                            f"## INFORMAÇÕES ORIGINAIS\n"
                            f"- Título/Pauta: {briefing_atual.get('titulo', briefing_atual.get('conteudo_original', 'N/A'))}\n"
                            f"- Mês referência: {briefing_atual.get('mes_referencia', 'N/A')}\n\n"
                            f"## BRIEFING ATUAL\n{briefing_atual['briefing']}\n\n"
                            f"## HISTÓRICO DE AJUSTES\n"
                        )
                        for i, ajuste in enumerate(briefing_atual['historico_ajustes'], 1):
                            data_ajuste = ajuste.get('data', '')
                            data_str = (
                                data_ajuste.strftime('%d/%m/%Y %H:%M:%S')
                                if isinstance(data_ajuste, datetime.datetime)
                                else 'Data desconhecida'
                            )
                            briefing_com_historico += f"\n{i}. {data_str}\n   Solicitação: {ajuste['solicitacao']}\n"

                        st.download_button(
                            "📋 Baixar com Histórico",
                            data=briefing_com_historico,
                            file_name=f"briefing_{briefing_atual['indice']}_com_historico.txt",
                            mime="text/plain",
                            use_container_width=True,
                        )

                with col_dl3:
                    if len(briefings) > 1:
                        zip_buffer = io.BytesIO()
                        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                            for b in briefings:
                                nome_b = f"briefing_{b['indice']}.txt"
                                if 'titulo' in b:
                                    nome_b = f"briefing_{b['titulo'].replace(' ', '_')}.txt"
                                zip_file.writestr(nome_b, b['briefing'])

                            consolidado = f"TODOS OS BRIEFINGS\nTotal: {len(briefings)}\n{'='*60}\n\n"
                            for b in briefings:
                                consolidado += f"BRIEFING {b['indice']}\n"
                                if 'titulo' in b:
                                    consolidado += f"Título: {b['titulo']}\n"
                                else:
                                    consolidado += f"Pauta: {b['conteudo_original']}\n"
                                consolidado += f"{'-'*40}\n{b['briefing']}\n{'='*60}\n\n"

                            zip_file.writestr("briefings_consolidados.txt", consolidado)

                        st.download_button(
                            "📦 Baixar Todos (ZIP)",
                            data=zip_buffer.getvalue(),
                            file_name=f"todos_briefings_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.zip",
                            mime="application/zip",
                            use_container_width=True,
                        )


def _prompt_briefing(contexto_agente, conteudo_pauta, mes_referencia, contexto_adicional):
    return f"""
    {contexto_agente}

    ## TAREFA: GERAR BRIEFING COMPLETO PARA ESTA PAUTA ESPECÍFICA

    **PAUTA ESPECÍFICA:**
    {conteudo_pauta}

    **MÊS DE REFERÊNCIA:** {mes_referencia}

    **CONTEXTO ADICIONAL:**
    {contexto_adicional if contexto_adicional else "Nenhum contexto adicional fornecido."}

    Gere um briefing completo baseado APENAS nesta pauta específica.
    Use a base de conhecimento fornecida para identificar produtos, culturas e informações técnicas.
    Traga informações chave dos produtos exatamente como são, sem alterar o texto. Mas posicione, crie um tema,
    discorra sobre o produto, agregue o tema, após trazer as informações brutas dos produtos que não deve ser alterada,
    o posicione em termos de benefícios, como que ele deve ser discorrido.

    # [TÍTULO DO BRIEFING]

    ## 1. OBJETIVO DO CONTEÚDO
    ## 2. PÚBLICO-ALVO
    ## 3. TEMA PRINCIPAL E ABORDAGEM
    ## 4. PRODUTOS ENVOLVIDOS
    ## 5. CULTURAS ALVO
    ## 6. PONTOS-CHAVE OBRIGATÓRIOS
    ## 7. TOM DE VOZ E ESTILO
    ## 8. FORMATOS SUGERIDOS
    ## 9. PALAVRAS-CHAVE (SEO)
    ## 10. CALL TO ACTION (CTA) SUGERIDO
    ## 11. INFORMAÇÕES TÉCNICAS RELEVANTES
    ## 12. RESTRIÇÕES E CUIDADOS
    ## 13. REFERÊNCIAS SUGERIDAS

    Seja detalhado e específico. Quando trouxer informações de produtos, os traga exatamente como são sem reescrita.
    """


def _prompt_briefing_individual(contexto_agente, titulo, mes_referencia, texto_base, contexto_adicional):
    return f"""
    {contexto_agente}

    ## TAREFA: GERAR BRIEFING COMPLETO E ESTRUTURADO

    **TÍTULO DO BRIEFING:** {titulo}
    **MÊS DE REFERÊNCIA:** {mes_referencia}

    **TEXTO BASE:**
    {texto_base}

    **CONTEXTO ADICIONAL:**
    {contexto_adicional if contexto_adicional else "Nenhum contexto adicional fornecido."}

    ## INSTRUÇÕES PARA O FORMATO DO BRIEFING:
    Traga informações chave dos produtos exatamente como são, sem alterar o texto. Mas posicione, crie um tema,
    discorra sobre o produto, agregue o tema, após trazer as informações brutas dos produtos que não deve ser alterada,
    o posicione em termos de benefícios, como que ele deve ser discorrido.

    Gere um briefing completo seguindo EXATAMENTE esta estrutura:

    # [TÍTULO DO BRIEFING]

    ## 1. OBJETIVO DO CONTEÚDO
    ## 2. PÚBLICO-ALVO
    ## 3. TEMA PRINCIPAL E ABORDAGEM
    ## 4. PRODUTOS ENVOLVIDOS
    ## 5. CULTURAS ALVO
    ## 6. PONTOS-CHAVE OBRIGATÓRIOS
    ## 7. TOM DE VOZ E ESTILO
    ## 8. FORMATOS SUGERIDOS
    ## 9. PALAVRAS-CHAVE (SEO)
    ## 10. CALL TO ACTION (CTA) SUGERIDO
    ## 11. INFORMAÇÕES TÉCNICAS RELEVANTES
    ## 12. RESTRIÇÕES E CUIDADOS
    ## 13. REFERÊNCIAS SUGERIDAS

    Seja detalhado e específico. Quando trouxer informações de produtos, os traga exatamente como são sem reescrita.
    """
