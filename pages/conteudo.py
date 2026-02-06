"""
Página de Geração de Conteúdo.
Geração de conteúdo com múltiplos insumos.
"""
import streamlit as st
import datetime
from database import modelo_texto, get_briefings_db
from models import construir_contexto
from services import extrair_texto_arquivo, transcrever_audio_video


def render():
    """Renderiza a aba de geração de conteúdo."""
    st.header("✨ Geração de Conteúdo com Múltiplos Insumos")

    # Conexão com MongoDB para briefings
    db_briefings = get_briefings_db()
    mongo_connected = db_briefings is not None

    if not mongo_connected:
        st.warning("Conexão com MongoDB não disponível para briefings")

    # Layout principal
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📝 Fontes de Conteúdo")

        # Upload de arquivos
        st.write("📎 Upload de Arquivos (PDF, TXT, PPTX, DOCX):")
        arquivos_upload = st.file_uploader(
            "Selecione um ou mais arquivos:",
            type=['pdf', 'txt', 'pptx', 'ppt', 'docx', 'doc'],
            accept_multiple_files=True,
            help="Arquivos serão convertidos para texto"
        )

        textos_arquivos = ""
        if arquivos_upload:
            st.success(f"✅ {len(arquivos_upload)} arquivo(s) carregado(s)")
            with st.expander("📋 Visualizar Conteúdo dos Arquivos", expanded=False):
                for i, arquivo in enumerate(arquivos_upload):
                    st.write(f"**{arquivo.name}** ({arquivo.size} bytes)")
                    with st.spinner(f"Processando {arquivo.name}..."):
                        texto_extraido = extrair_texto_arquivo(arquivo)
                        textos_arquivos += f"\n\n--- CONTEÚDO DE {arquivo.name.upper()} ---\n{texto_extraido}"
                        preview = texto_extraido[:500] + "..." if len(texto_extraido) > 500 else texto_extraido
                        st.text_area(f"Preview - {arquivo.name}", value=preview, height=100, key=f"preview_{i}")

        # Briefing do banco de dados
        st.write("🗃️ Briefing do Banco de Dados:")
        briefing_data = None
        if mongo_connected:
            collection_briefings = db_briefings['briefings']
            briefings_disponiveis = list(collection_briefings.find().sort("data_criacao", -1).limit(20))
            if briefings_disponiveis:
                briefing_options = {
                    f"{b['nome_projeto']} ({b['tipo']}) - {b['data_criacao'].strftime('%d/%m/%Y')}": b
                    for b in briefings_disponiveis
                }
                briefing_selecionado = st.selectbox("Escolha um briefing:", list(briefing_options.keys()))
                if briefing_selecionado:
                    briefing_data = briefing_options[briefing_selecionado]
                    st.info(f"Briefing selecionado: {briefing_data['nome_projeto']}")
            else:
                st.info("Nenhum briefing encontrado no banco de dados.")

        # Briefing manual
        st.write("✍️ Briefing Manual:")
        briefing_manual = st.text_area(
            "Ou cole o briefing completo aqui:", height=150,
            placeholder="Exemplo:\nTítulo: Campanha de Lançamento\nObjetivo: Divulgar novo produto..."
        )

        # Transcrição de mídia
        st.write("🎤 Transcrição de Áudio/Video:")
        arquivos_midia = st.file_uploader(
            "Áudios/Vídeos para transcrição:",
            type=['mp3', 'wav', 'mp4', 'mov', 'avi'],
            accept_multiple_files=True,
            help="Arquivos de mídia serão transcritos automaticamente"
        )

        transcricoes_texto = ""
        if arquivos_midia:
            st.info(f"🎬 {len(arquivos_midia)} arquivo(s) de mídia carregado(s)")
            if st.button("🔄 Transcrever Todos os Arquivos de Mídia"):
                with st.spinner("Transcrevendo arquivos de mídia..."):
                    for arquivo in arquivos_midia:
                        tipo = "audio" if arquivo.type.startswith('audio') else "video"
                        transcricao = transcrever_audio_video(arquivo, tipo)
                        transcricoes_texto += f"\n\n--- TRANSCRIÇÃO DE {arquivo.name.upper()} ---\n{transcricao}"
                        st.success(f"✅ {arquivo.name} transcrito!")

    with col2:
        st.subheader("⚙️ Configurações")

        tipo_conteudo = st.selectbox(
            "Tipo de Conteúdo:",
            ["Post Social", "Artigo Blog", "Email Marketing", "Landing Page",
             "Script Vídeo", "Relatório Técnico", "Press Release", "Newsletter", "Case Study"]
        )

        tom_voz = st.selectbox(
            "Tom de Voz:",
            ["Formal", "Informal", "Persuasivo", "Educativo", "Inspirador", "Técnico", "Jornalístico"],
            key='tom_voz_conteudo'
        )

        palavras_chave = st.text_input(
            "Palavras-chave (opcional):",
            placeholder="separadas por vírgula"
        )

        numero_palavras = st.slider("Número de Palavras:", 100, 3000, 800)

        with st.expander("🔧 Configurações Avançadas"):
            usar_contexto_agente = st.checkbox(
                "Usar contexto do agente selecionado",
                value=bool(st.session_state.get('agente_selecionado'))
            )
            nivel_detalhe = st.select_slider(
                "Nível de Detalhe:",
                ["Resumido", "Balanceado", "Detalhado", "Completo"]
            )
            incluir_cta = st.checkbox("Incluir Call-to-Action", value=True)
            formato_saida = st.selectbox(
                "Formato de Saída:",
                ["Texto Simples", "Markdown", "HTML Básico"]
            )

    # Instruções específicas
    st.subheader("🎯 Instruções Específicas")
    instrucoes_especificas = st.text_area(
        "Diretrizes adicionais para geração:",
        placeholder="Exemplos:\n- Focar nos benefícios para o usuário final\n- Incluir estatísticas quando possível",
        height=100
    )

    # Botão de geração
    if st.button("🚀 Gerar Conteúdo com Todos os Insumos", type="primary", use_container_width=True):
        tem_conteudo = arquivos_upload or briefing_manual or briefing_data or arquivos_midia

        if not tem_conteudo:
            st.error("❌ Por favor, forneça pelo menos uma fonte de conteúdo")
            return

        with st.spinner("Processando todos os insumos e gerando conteúdo..."):
            try:
                # Construir contexto
                contexto_completo = "## FONTES DE CONTEÚDO COMBINADAS:\n\n"

                if textos_arquivos:
                    contexto_completo += "### CONTEÚDO DOS ARQUIVOS:\n" + textos_arquivos + "\n\n"

                if briefing_manual:
                    contexto_completo += "### BRIEFING MANUAL:\n" + briefing_manual + "\n\n"
                elif briefing_data:
                    contexto_completo += "### BRIEFING DO BANCO:\n" + briefing_data.get('conteudo', '') + "\n\n"

                if transcricoes_texto:
                    contexto_completo += "### TRANSCRIÇÕES DE MÍDIA:\n" + transcricoes_texto + "\n\n"

                contexto_agente = ""
                if usar_contexto_agente and st.session_state.get('agente_selecionado'):
                    agente = st.session_state.agente_selecionado
                    contexto_agente = construir_contexto(
                        agente,
                        st.session_state.get('segmentos_selecionados', [])
                    )

                prompt_final = f"""
                {contexto_agente}

                ## INSTRUÇÕES PARA GERAÇÃO DE CONTEÚDO:

                **TIPO DE CONTEÚDO:** {tipo_conteudo}
                **TOM DE VOZ:** {tom_voz}
                **PALAVRAS-CHAVE:** {palavras_chave if palavras_chave else 'Não especificadas'}
                **NÚMERO DE PALAVRAS:** {numero_palavras} (±10%)
                **NÍVEL DE DETALHE:** {nivel_detalhe}
                **INCLUIR CALL-TO-ACTION:** {incluir_cta}

                **INSTRUÇÕES ESPECÍFICAS:**
                {instrucoes_especificas if instrucoes_especificas else 'Nenhuma instrução específica fornecida.'}

                ## FONTES E REFERÊNCIAS:
                {contexto_completo}

                ## TAREFA:
                Gere um conteúdo do tipo {tipo_conteudo} que:
                1. Combine e sintetize informações de todas as fontes
                2. Mantenha consistência com as informações originais
                3. Agregue insights de valor
                4. Engaje o público-alvo
                5. Comunique ideias de forma acessível

                **FORMATO DE SAÍDA:** {formato_saida}

                Gere um conteúdo completo e profissional.
                """

                resposta = modelo_texto.generate_content(prompt_final)
                conteudo_gerado = resposta.text

                # Converter para HTML se necessário
                if formato_saida == "HTML Básico":
                    import re
                    conteudo_gerado = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', conteudo_gerado)
                    conteudo_gerado = re.sub(r'\*(.*?)\*', r'<em>\1</em>', conteudo_gerado)
                    conteudo_gerado = re.sub(r'### (.*?)\n', r'<h3>\1</h3>', conteudo_gerado)
                    conteudo_gerado = re.sub(r'## (.*?)\n', r'<h2>\1</h2>', conteudo_gerado)
                    conteudo_gerado = re.sub(r'# (.*?)\n', r'<h1>\1</h1>', conteudo_gerado)
                    conteudo_gerado = conteudo_gerado.replace('\n', '<br>')

                st.subheader("📄 Conteúdo Gerado")

                if formato_saida == "HTML Básico":
                    st.components.v1.html(conteudo_gerado, height=400, scrolling=True)
                else:
                    st.markdown(conteudo_gerado)

                # Estatísticas
                palavras_count = len(conteudo_gerado.split())
                col_stat1, col_stat2, col_stat3 = st.columns(3)
                with col_stat1:
                    st.metric("Palavras Geradas", palavras_count)
                with col_stat2:
                    st.metric("Arquivos Processados", len(arquivos_upload) if arquivos_upload else 0)
                with col_stat3:
                    fontes = (1 if arquivos_upload else 0) + (1 if briefing_manual or briefing_data else 0) + (1 if transcricoes_texto else 0)
                    st.metric("Fontes Utilizadas", fontes)

                # Download
                extensao = ".html" if formato_saida == "HTML Básico" else ".md" if formato_saida == "Markdown" else ".txt"
                st.download_button(
                    f"💾 Baixar Conteúdo ({formato_saida})",
                    data=conteudo_gerado,
                    file_name=f"conteudo_gerado_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}{extensao}",
                    mime="text/html" if formato_saida == "HTML Básico" else "text/plain"
                )

            except Exception as e:
                st.error(f"❌ Erro ao gerar conteúdo: {str(e)}")
                st.info("💡 Dica: Verifique se os arquivos não estão corrompidos e tente novamente.")
