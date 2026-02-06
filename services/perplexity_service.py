"""
Serviço de busca usando Perplexity AI.
Realiza buscas na web para enriquecer conteúdo.
"""
import streamlit as st
from perplexity import Perplexity
from config.settings import PERP_API_KEY

# Inicializa o cliente Perplexity
perplexity_client = None
if PERP_API_KEY:
    try:
        perplexity_client = Perplexity(api_key=PERP_API_KEY)
    except Exception as e:
        st.warning(f"Erro ao inicializar Perplexity: {str(e)}")


def buscar_perplexity(prompt: str) -> str:
    """
    Realiza busca usando Perplexity AI.

    Args:
        prompt: Texto/pergunta para buscar

    Returns:
        Resultado da busca ou mensagem de erro
    """
    if not perplexity_client:
        return "Cliente Perplexity não disponível. Verifique a API key."

    try:
        response = perplexity_client.search(prompt)
        if response and 'answer' in response:
            return response['answer']
        return "Nenhum resultado encontrado."
    except Exception as e:
        return f"Erro na busca Perplexity: {str(e)}"


def buscar_fontes_para_otimizacao(conteudo: str, tipo: str, tom: str) -> str:
    """
    Busca fontes relevantes para otimização de conteúdo.

    Args:
        conteudo: Conteúdo a ser otimizado
        tipo: Tipo de conteúdo (ex: "blog", "social")
        tom: Tom de voz desejado

    Returns:
        Informações relevantes encontradas
    """
    if not perplexity_client:
        return ""

    try:
        # Extrai palavras-chave do conteúdo
        palavras = conteudo[:500].split()[:10]
        query = f"Informações atualizadas sobre: {' '.join(palavras)} para {tipo} com tom {tom}"

        response = perplexity_client.search(query)
        if response and 'answer' in response:
            return response['answer']
        return ""
    except Exception:
        return ""


def is_perplexity_available() -> bool:
    """Verifica se o cliente Perplexity está disponível."""
    return perplexity_client is not None
