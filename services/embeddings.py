"""
Serviço de embeddings.
Gera embeddings de texto usando OpenAI.
"""
import hashlib
import streamlit as st
import openai
from typing import List
from config.settings import OPENAI_API_KEY, MODELO_EMBEDDING


def get_embedding(text: str) -> List[float]:
    """
    Obtém embedding do texto usando OpenAI.

    Args:
        text: Texto para gerar embedding

    Returns:
        Lista de floats representando o embedding (1536 dimensões)
    """
    try:
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        response = client.embeddings.create(
            input=text,
            model=MODELO_EMBEDDING
        )
        return response.data[0].embedding
    except Exception as e:
        st.warning(f"Embedding OpenAI não disponível: {str(e)}")
        # Fallback para embedding simples baseado em hash
        return _fallback_embedding(text)


def _fallback_embedding(text: str) -> List[float]:
    """
    Gera embedding simples como fallback quando OpenAI não está disponível.

    Args:
        text: Texto para gerar embedding

    Returns:
        Lista de 1536 floats
    """
    text_hash = hashlib.md5(text.encode()).hexdigest()
    vector = [float(int(text_hash[i:i+2], 16) / 255.0) for i in range(0, 32, 2)]
    # Preenche com zeros para ter 1536 dimensões
    while len(vector) < 1536:
        vector.append(0.0)
    return vector[:1536]
