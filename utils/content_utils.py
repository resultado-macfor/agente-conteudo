import streamlit as st
from google.genai import types
from config.settings import GEMINI_API_KEY


def construir_contexto(agente, segmentos_selecionados, historico_mensagens=None):
    """Constrói o contexto com base nos segmentos selecionados do agente."""
    contexto = ""

    if "system_prompt" in segmentos_selecionados and agente.get('system_prompt'):
        contexto += f"### INSTRUÇÕES DO SISTEMA ###\n{agente['system_prompt']}\n\n"

    if "base_conhecimento" in segmentos_selecionados and agente.get('base_conhecimento'):
        contexto += f"### BASE DE CONHECIMENTO ###\n{agente['base_conhecimento']}\n\n"

    if "comments" in segmentos_selecionados and agente.get('comments'):
        contexto += f"### COMENTÁRIOS DO CLIENTE ###\n{agente['comments']}\n\n"

    if "planejamento" in segmentos_selecionados and agente.get('planejamento'):
        contexto += f"### PLANEJAMENTO ###\n{agente['planejamento']}\n\n"

    if historico_mensagens:
        contexto += "### HISTÓRICO DA CONVERSA ###\n"
        for msg in historico_mensagens:
            contexto += f"{msg['role']}: {msg['content']}\n"
        contexto += "\n"

    contexto += "### RESPOSTA ATUAL ###\nassistant:"
    return contexto


def transcrever_audio_video(arquivo, tipo_arquivo):
    """Transcreve áudio ou vídeo usando a API do Gemini."""
    try:
        import google.generativeai as genai
        client = genai.Client(api_key=GEMINI_API_KEY)

        if tipo_arquivo == "audio":
            mime_type = f"audio/{arquivo.name.split('.')[-1]}"
        else:
            mime_type = f"video/{arquivo.name.split('.')[-1]}"

        arquivo_bytes = arquivo.read()

        if len(arquivo_bytes) > 20 * 1024 * 1024:  # 20 MB
            uploaded_file = client.files.upload(file=arquivo_bytes, mime_type=mime_type)
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=["Transcreva este arquivo em detalhes:", uploaded_file],
            )
        else:
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[
                    "Transcreva este arquivo em detalhes:",
                    types.Part.from_bytes(data=arquivo_bytes, mime_type=mime_type),
                ],
            )

        return response.text
    except Exception as e:
        return f"Erro na transcrição: {str(e)}"


def buscar_perplexity(prompt: str, perplexity_client) -> str:
    """Realiza busca na web usando a biblioteca Perplexity."""
    try:
        response = perplexity_client.chat.completions.create(
            model="sonar",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ Erro na busca Perplexity: {str(e)}"


def buscar_fontes_para_otimizacao(conteudo: str, tipo: str, tom: str, perplexity_client) -> str:
    """Busca fontes específicas para otimização de conteúdo agrícola."""
    prompt = f"""
    DADOS TÉCNICOS ATUALIZADOS para este conteúdo:
    {conteudo[:800]}
    """
    return buscar_perplexity(prompt, perplexity_client)


def realizar_busca_web_perplexity(texto: str, tipo_otimizacao: str, tom_voz: str) -> str:
    """Realiza busca web via Perplexity para otimização de conteúdo."""
    try:
        from perplexity import Perplexity
        import os

        perp_api_key = os.getenv("PERP_API_KEY")
        if not perp_api_key:
            return "❌ ERRO: PERP_API_KEY não encontrada nas variáveis de ambiente"

        client = Perplexity(api_key=perp_api_key)

        prompt = f"""
        Você é um assistente especializado em pesquisa. Busque informações atualizadas e confiáveis sobre:

        TÓPICO PRINCIPAL: {texto}

        CRITÉRIOS DE PESQUISA:
        1. Fontes confiáveis: Embrapa, universidades, órgãos governamentais, institutos de pesquisa
        2. Informações técnicas atualizadas (últimos 2-3 anos)
        3. Dados concretos: números, estatísticas, resultados de pesquisa
        4. Melhores práticas agrícolas
        5. Soluções tecnológicas inovadoras

        FORMATO DE RESPOSTA:
        Para CADA fonte encontrada, forneça:
        - TÍTULO: Título do artigo/referência
        - CONTEÚDO: Resumo das informações relevantes (máx 200 palavras)
        - URL: Link completo para a fonte
        - RELEVÂNCIA: Por que esta fonte é relevante para o tópico

        Retorne no máximo 20 fontes mais relevantes.
        """

        response = client.chat.completions.create(
            model="sonar",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=20000,
        )

        if response and response.choices:
            return response.choices[0].message.content
        return "❌ ERRO: Nenhuma resposta recebida do Perplexity"

    except ImportError as e:
        return f"❌ ERRO: Biblioteca perplexity-api não instalada.\nDetalhes: {str(e)}"
    except Exception as e:
        return f"❌ ERRO na busca web: {str(e)}"
