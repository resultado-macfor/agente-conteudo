"""
Serviço de RAG (Retrieval-Augmented Generation).
Reescreve conteúdo usando base de conhecimento vetorial.
"""
import streamlit as st
from config.settings import ASTRA_DB_COLLECTION
from database import astra_client, modelo_texto
from .embeddings import get_embedding


def reescrever_com_rag_blog(content: str, tom_voz: str = "Formal") -> str:
    """
    Reescreve conteúdo de blog usando RAG.

    Args:
        content: Conteúdo original para reescrever
        tom_voz: Tom de voz desejado

    Returns:
        Conteúdo reescrito
    """
    try:
        # Gera embedding para busca
        embedding = get_embedding(content[:800])

        # Busca documentos relevantes
        relevant_docs = astra_client.vector_search(ASTRA_DB_COLLECTION, embedding, limit=10)

        # Constrói contexto dos documentos
        rag_context = _build_rag_context(relevant_docs, "INFORMAÇÕES TÉCNICAS RELEVANTES DA BASE")

        # Prompt de entendimento RAG
        rewrite_prompt = f"""
        Entenda o que no texto original de fato é enriquecido e corrigido pelo referencial teórico.
        Considere que você não pode tangenciar o assunto do texto original.

        ###BEGIN TEXTO ORIGINAL###
        {content}
        ###END TEXTO ORIGINAL###

        ###BEGIN REFERENCIAL TEÓRICO###
        {rag_context}
        ###END REFERENCIAL TEÓRICO###
        """

        # Gera conteúdo pré-processado
        pre_response = modelo_texto.generate_content(rewrite_prompt)

        # Prompt final
        final_prompt = f"""
        ###BEGIN TEXTO ORIGINAL###
        {content}
        ###END TEXTO ORIGINAL###

        ###BEGIN REFERENCIAL TEÓRICO###
        {pre_response}
        ###END REFERENCIAL TEÓRICO###

        Aplique isso ao texto original:

        1. SUBSTITUA termos vagos por terminologia técnica precisa da área agrícola que são relevantes ao texto original.
        2. CORRIGIR automaticamente qualquer imprecisão técnica ou científica no texto original
        3. ENRIQUECER com dados concretos, números e informações específicas da base
        4. MANTER tom {tom_voz} mas com precisão técnica absoluta
        5. MANTENHA a estrutura do texto original. Não reescreva por inteiro. Apenas corrija
        7. O agente revisor precisaria entregar o texto exatamente como no original, mas apontando os ajustes técnicos necessários/feitos, sem reescrever tudo automaticamente OU reescrevendo e sinalizando o que foi alterado no texto, mostrando como estava > como ficou > fonte/referência utilizada.
        8. NÃO acrescente informações que tangem o tema do texto original
        9. Mantenha o tamanho do texto original (com um delta de no máximo 5%)

        ESTRUTURA OBRIGATÓRIA:
        - Mantenha a estrutura original. O seu papel é REVISAR TECNICAMENTE O CONTEÚDO DE ENTRADA ENRIQUECENDO-O E, QUANDO NECESSÁRIO, CORRIJINDO-O COM O REFERENCIAL TEÓRICO.

        RETORNE O CONTEÚDO REEESCRITO FINAL, apontando as mudanças em uma subseção ao final.
        """

        response = modelo_texto.generate_content(final_prompt)
        return response.text

    except Exception as e:
        st.error(f"Erro no RAG rewrite para blog: {str(e)}")
        return content


def reescrever_com_rag_revisao_SEO(content: str, tom_voz: str = "Formal") -> str:
    """
    Reescreve conteúdo técnico para revisão SEO.

    Args:
        content: Conteúdo original
        tom_voz: Tom de voz desejado

    Returns:
        Conteúdo reescrito
    """
    try:
        embedding = get_embedding(content[:800])
        relevant_docs = astra_client.vector_search(ASTRA_DB_COLLECTION, embedding, limit=10)
        rag_context = _build_rag_context(relevant_docs, "DOCUMENTAÇÃO TÉCNICA ESPECIALIZADA", max_chars=400)

        rewrite_prompt = f"""
        CONTEÚDO TÉCNICO ORIGINAL PARA REESCRITA COMPLETA:
        {content}

        BASE DE CONHECIMENTO TÉCNICO:
        {rag_context}

        Aplique isso ao texto original:

        1. SUBSTITUA termos vagos por terminologia técnica precisa da área agrícola que são relevantes ao texto original.
        2. CORRIGIR automaticamente qualquer imprecisão técnica ou científica no texto original
        3. ENRIQUECER com dados concretos, números e informações específicas da base
        4. MANTER tom {tom_voz} mas com precisão técnica absoluta
        5. MANTENHA a estrutura do texto original. Não reescreva por inteiro. Apenas corrija
        7. O agente revisor precisaria entregar o texto exatamente como no original, mas apontando os ajustes técnicos necessários/feitos, sem reescrever tudo automaticamente OU reescrevendo e sinalizando o que foi alterado no texto, mostrando como estava > como ficou > fonte/referência utilizada.
        8. NÃO acrescente informações que tangem o tema do texto original
        9. Mantenha o tamanho do texto original (com um delta de no máximo 5%)

        ESTRUTURA OBRIGATÓRIA:
        - Mantenha a estrutura original. O seu papel é REVISAR TECNICAMENTE O CONTEÚDO DE ENTRADA ENRIQUECENDO-O COM O REFERENCIAL TEÓRICO.

        RETORNE O CONTEÚDO REEESCRITO FINAL, apontando as mudanças em uma subseção ao final.
        """

        response = modelo_texto.generate_content(rewrite_prompt)
        return response.text

    except Exception as e:
        st.error(f"Erro no RAG rewrite técnico: {str(e)}")
        return content


def reescrever_com_rag_revisao_NORM(content: str, tom_voz: str = "Formal") -> str:
    """
    Reescreve conteúdo técnico para revisão normalizada (sem bullets).

    Args:
        content: Conteúdo original
        tom_voz: Tom de voz desejado

    Returns:
        Conteúdo reescrito
    """
    try:
        embedding = get_embedding(content[:800])
        relevant_docs = astra_client.vector_search(ASTRA_DB_COLLECTION, embedding, limit=10)
        rag_context = _build_rag_context(relevant_docs, "DOCUMENTAÇÃO TÉCNICA ESPECIALIZADA", max_chars=400)

        rewrite_prompt = f"""
        CONTEÚDO TÉCNICO ORIGINAL PARA REESCRITA COMPLETE:
        {content}

        BASE DE CONHECIMENTO TÉCNICO:
        {rag_context}

        Aplique isso ao texto original:

        1. SUBSTITUA termos vagos por terminologia técnica precisa da área agrícola que são relevantes ao texto original.
        2. CORRIGIR automaticamente qualquer imprecisão técnica ou científica no texto original
        3. ENRIQUECER com dados concretos, números e informações específicas da base
        4. MANTER tom {tom_voz} mas com precisão técnica absoluta
        5. MANTENHA a estrutura do texto original. Não reescreva por inteiro. Apenas corrija
        7. O agente revisor precisaria entregar o texto exatamente como no original, mas apontando os ajustes técnicos necessários/feitos, sem reescrever tudo automaticamente OU reescrevendo e sinalizando o que foi alterado no texto, mostrando como estava > como ficou > fonte/referência utilizada.
        8. NÃO acrescente informações que tangem o tema do texto original
        9. Mantenha o tamanho do texto original (com um delta de no máximo 5%)
        10. NÃO USE BULLETS EM TODO O CONTEÚDO. MANTENHA OS ORIGINAIS E O RESTO DEVE VIR EM FORMATO DE PARÁGRAFO

        ESTRUTURA OBRIGATÓRIA:
        - Mantenha a estrutura original. O seu papel é REVISAR TECNICAMENTE O CONTEÚDO DE ENTRADA ENRIQUECENDO-O COM O REFERENCIAL TEÓRICO.

        RETORNE O CONTEÚDO REEESCRITO FINAL, apontando as mudanças em uma subseção ao final.
        """

        response = modelo_texto.generate_content(rewrite_prompt)
        return response.text

    except Exception as e:
        st.error(f"Erro no RAG rewrite técnico: {str(e)}")
        return content


def _build_rag_context(docs: list, header: str, max_chars: int = 500) -> str:
    """
    Constrói contexto RAG a partir dos documentos encontrados.

    Args:
        docs: Lista de documentos do AstraDB
        header: Cabeçalho do contexto
        max_chars: Máximo de caracteres por documento

    Returns:
        String formatada com o contexto
    """
    if not docs:
        return "Base de conhecimento não retornou resultados específicos."

    context = f"{header}:\n"
    for i, doc in enumerate(docs, 1):
        doc_content = str(doc)
        doc_clean = doc_content.replace('{', '').replace('}', '').replace("'", "").replace('"', '')
        context += f"--- Fonte {i} ---\n{doc_clean[:max_chars]}...\n\n"

    return context
