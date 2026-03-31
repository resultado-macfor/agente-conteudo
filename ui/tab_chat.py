import streamlit as st
from agent.agents import salvar_conversa
from utils.content_utils import construir_contexto


def render(tab, modelo_texto):
    with tab:
        st.header("💬 Chat com Agente")

        if "messages" not in st.session_state:
            st.session_state.messages = []

        if not st.session_state.agente_selecionado:
            st.info("Selecione um agente na parte superior do app para iniciar o chat.")
            return

        agente = st.session_state.agente_selecionado
        st.subheader(f"Conversando com: {agente['nome']}")

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Digite sua mensagem..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            contexto = construir_contexto(
                agente,
                st.session_state.segmentos_selecionados,
                st.session_state.messages,
            )

            with st.chat_message("assistant"):
                with st.spinner('Pensando...'):
                    try:
                        resposta = modelo_texto.generate_content(contexto)
                        st.markdown(resposta.text)
                        st.session_state.messages.append({"role": "assistant", "content": resposta.text})
                        salvar_conversa(
                            agente['_id'],
                            st.session_state.messages,
                            st.session_state.segmentos_selecionados,
                        )
                    except Exception as e:
                        st.error(f"Erro ao gerar resposta: {str(e)}")
