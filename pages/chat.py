"""
Página de Chat com Agente.
Interface de chat interativo usando agentes personalizados.
"""
import streamlit as st
from database import modelo_texto
from models import construir_contexto, salvar_conversa


def render():
    """Renderiza a aba de chat."""
    st.header("💬 Chat com Agente")

    # Inicializar estado da sessão
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Verificar se há agente selecionado
    if not st.session_state.agente_selecionado:
        st.info("Selecione um agente na parte superior do app para iniciar o chat.")
        return

    agente = st.session_state.agente_selecionado
    st.subheader(f"Conversando com: {agente['nome']}")

    # Exibir histórico de mensagens
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input do usuário
    if prompt := st.chat_input("Digite sua mensagem..."):
        # Adicionar mensagem do usuário ao histórico
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Construir contexto com segmentos selecionados
        contexto = construir_contexto(
            agente,
            st.session_state.segmentos_selecionados,
            st.session_state.messages
        )

        # Gerar resposta
        with st.chat_message("assistant"):
            with st.spinner('Pensando...'):
                try:
                    resposta = modelo_texto.generate_content(contexto)
                    st.markdown(resposta.text)

                    # Adicionar ao histórico
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": resposta.text
                    })

                    # Salvar conversa com segmentos utilizados
                    salvar_conversa(
                        agente['_id'],
                        st.session_state.messages,
                        st.session_state.segmentos_selecionados
                    )

                except Exception as e:
                    st.error(f"Erro ao gerar resposta: {str(e)}")
