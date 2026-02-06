"""
Página de Revisão Ortográfica.
Revisão ortográfica e gramatical de textos.
"""
import streamlit as st
from database import modelo_texto
from models import construir_contexto


def render():
    """Renderiza a aba de revisão ortográfica."""
    st.header("📝 Revisão Ortográfica")

    texto_para_revisao = st.text_area(
        "Cole o texto que deseja revisar:",
        height=300
    )

    if st.button("🔍 Realizar Revisão Ortográfica", type="primary"):
        if not texto_para_revisao:
            st.warning("Por favor, cole um texto para revisão.")
            return

        with st.spinner("Revisando texto..."):
            try:
                # Usar contexto do agente selecionado se disponível
                if st.session_state.agente_selecionado:
                    agente = st.session_state.agente_selecionado
                    contexto = construir_contexto(
                        agente,
                        st.session_state.segmentos_selecionados
                    )
                    prompt = f"""
                    {contexto}

                    Faça uma revisão ortográfica e gramatical completa do seguinte texto:

                    ###BEGIN TEXTO A SER REVISADO###
                    {texto_para_revisao}
                    ###END TEXTO A SER REVISADO###

                    MANTENHA A ESTRUTURA DO TEXTO ORIGINAL. APENAS CORRIJA ERROS ORTOGRÁFICOS (SE PRESENTES) E APONTE QUAIS FORAM OS ERROS CORRIGIDOS
                    """
                else:
                    prompt = f"""
                    Faça uma revisão ortográfica e gramatical completa do seguinte texto:

                    ###BEGIN TEXTO A SER REVISADO###
                    {texto_para_revisao}
                    ###END TEXTO A SER REVISADO###

                    MANTENHA A ESTRUTURA DO TEXTO ORIGINAL. APENAS CORRIJA ERROS ORTOGRÁFICOS (SE PRESENTES) E APONTE QUAIS FORAM OS ERROS CORRIGIDOS
                    """

                resposta = modelo_texto.generate_content(prompt)
                st.subheader("📋 Resultado da Revisão")
                st.markdown(resposta.text)

            except Exception as e:
                st.error(f"Erro na revisão: {str(e)}")
