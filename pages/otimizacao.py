"""
Página de Otimização de Conteúdo.
Otimização de conteúdo para SEO, engajamento e conversão.
"""
import streamlit as st
import datetime
import os
from database import modelo_texto
from models import construir_contexto


def render():
    """Renderiza a aba de otimização de conteúdo."""
    st.header("🚀 Otimização de Conteúdo")

    # Inicializar session state
    if 'conteudo_otimizado' not in st.session_state:
        st.session_state.conteudo_otimizado = None
    if 'ajustes_realizados' not in st.session_state:
        st.session_state.ajustes_realizados = []
    if 'fontes_busca_web' not in st.session_state:
        st.session_state.fontes_busca_web = ""

    # Entrada do conteúdo
    texto_para_otimizar = st.text_area("Cole o conteúdo para otimização:", height=300)

    # Configurações
    col_config1, col_config2 = st.columns([2, 1])

    with col_config1:
        tipo_otimizacao = st.selectbox("Tipo de Otimização:", ["SEO", "Engajamento", "Conversão", "Clareza"])

    with col_config2:
        tom_voz = st.text_input("Tom de Voz:", value="Técnico", key="tom_voz_otimizacao")
        nivel_heading = st.selectbox("Nível de Heading:", ["H1", "H2", "H3", "H4"])

    # Busca web e links
    st.subheader("🔍 Busca Web e Links")
    usar_busca_web = st.checkbox("Usar busca web para enriquecer conteúdo", value=False)
    incluir_links_internos = st.checkbox("Incluir links internos", value=True)

    instrucoes_briefing = st.text_area("Instruções do briefing (opcional):", height=80)

    # Função de busca web
    def realizar_busca_web(texto):
        try:
            from perplexity import Perplexity

            perp_api_key = os.getenv("PERP_API_KEY")
            if not perp_api_key:
                return "PERP_API_KEY não encontrada"

            client = Perplexity(api_key=perp_api_key)

            prompt = f"""
            Busque informações atualizadas sobre:
            {texto[:500]}

            Fontes confiáveis: Embrapa, universidades, institutos de pesquisa
            Retorne no máximo 10 fontes relevantes.
            """

            response = client.chat.completions.create(
                model="sonar",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=10000
            )

            if response and response.choices:
                return response.choices[0].message.content
            return "Nenhuma resposta recebida"

        except Exception as e:
            return f"Erro na busca: {str(e)}"

    # Botão de otimização
    if st.button("🚀 Otimizar Conteúdo", type="primary", use_container_width=True):
        if texto_para_otimizar:
            with st.spinner("Processando otimização..."):
                try:
                    # Busca web
                    fontes_encontradas = ""
                    if usar_busca_web:
                        st.info("🔍 Buscando na web...")
                        fontes_encontradas = realizar_busca_web(texto_para_otimizar)
                        if not fontes_encontradas.startswith("Erro"):
                            st.session_state.fontes_busca_web = fontes_encontradas
                            st.success(f"✅ Busca concluída")

                    # Contexto do agente
                    contexto_agente = ""
                    if st.session_state.get('agente_selecionado'):
                        agente = st.session_state.agente_selecionado
                        contexto_agente = construir_contexto(agente, st.session_state.get('segmentos_selecionados', []))

                    # Prompt de otimização
                    prompt = f"""
                    {contexto_agente}

                    **TEXTO ORIGINAL:**
                    {texto_para_otimizar}

                    **FONTES DA BUSCA WEB:**
                    {fontes_encontradas if fontes_encontradas else "Nenhuma fonte externa disponível."}

                    **INSTRUÇÕES DO BRIEFING:**
                    {instrucoes_briefing if instrucoes_briefing else 'Sem briefing específico'}

                    **CONFIGURAÇÕES:**
                    - Tipo: {tipo_otimizacao}
                    - Tom: {tom_voz}
                    - Heading level: {nivel_heading}

                    ## REQUISITOS:

                    1. **META TAGS:**
                       Gere 3 opções de meta title (≤60 chars) e description (≤155 chars)

                    2. **BULLETS:**
                       Use bullets para listas de benefícios e características

                    3. **HEADING LEVEL {nivel_heading}:**
                       Todos os headings devem ser {nivel_heading}

                    4. **CORREÇÕES:**
                       - Remova introduções genéricas
                       - Quebre parágrafos longos
                       - Remova repetições
                       - Melhore escaneabilidade

                    5. **LINKS INTERNOS:**
                       Sugira 3-5 links relevantes

                    Retorne o conteúdo otimizado com todas as melhorias aplicadas.
                    """

                    # Gerar otimização
                    resposta = modelo_texto.generate_content(prompt)
                    resultado = resposta.text

                    st.session_state.conteudo_otimizado = resultado

                    # Exibir resultados
                    st.success("✅ Conteúdo otimizado!")

                    st.subheader("📝 Conteúdo Otimizado")
                    st.markdown(resultado)

                    # Verificações
                    st.subheader("🔍 Verificação")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        bullets = resultado.count("- ") + resultado.count("* ")
                        st.metric("Bullet Points", bullets)
                    with col2:
                        has_heading = nivel_heading.lower() in resultado.lower()
                        st.metric(f"Heading {nivel_heading}", "✅" if has_heading else "❌")
                    with col3:
                        has_meta = 'title' in resultado[:500].lower() or 'description' in resultado[:500].lower()
                        st.metric("Meta Tags", "✅" if has_meta else "❌")

                    # Download
                    st.download_button(
                        "💾 Baixar Conteúdo Otimizado",
                        data=resultado,
                        file_name=f"conteudo_otimizado_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                        mime="text/plain"
                    )

                except Exception as e:
                    st.error(f"❌ Erro na otimização: {str(e)}")
        else:
            st.warning("Por favor, cole um conteúdo para otimizar")

    # Ajustes incrementais
    if st.session_state.conteudo_otimizado:
        st.divider()
        st.subheader("🔄 Ajustes Incrementais")

        comando_ajuste = st.text_area(
            "Ajustes desejados:",
            height=80,
            placeholder="Ex: Adicione mais bullets, corrija headings..."
        )

        if st.button("🔄 Aplicar Ajustes"):
            if comando_ajuste:
                with st.spinner("Aplicando ajustes..."):
                    try:
                        prompt_ajuste = f"""
                        **CONTEÚDO ATUAL:** {st.session_state.conteudo_otimizado[:1000]}

                        **AJUSTES SOLICITADOS:** {comando_ajuste}

                        **MANTENHA:**
                        - Meta tags existentes
                        - Heading level {nivel_heading}
                        - Bullets onde aplicável

                        Aplique os ajustes e retorne APENAS o conteúdo atualizado.
                        """

                        resposta = modelo_texto.generate_content(prompt_ajuste)
                        st.session_state.conteudo_otimizado = resposta.text
                        st.session_state.ajustes_realizados.append(comando_ajuste)

                        st.success("✅ Ajustes aplicados!")
                        st.markdown(resposta.text)

                    except Exception as e:
                        st.error(f"Erro: {str(e)}")
            else:
                st.warning("Digite os ajustes desejados")

        if st.button("🗑️ Limpar Histórico de Ajustes"):
            st.session_state.ajustes_realizados = []
            st.success("Histórico limpo")
