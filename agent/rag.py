import streamlit as st
import openai
from typing import List, Dict
from config.settings import OPENAI_API_KEY, ASTRA_DB_COLLECTION
from services.database import astra_client




def get_embedding(text: str) -> List[float]:
    try:
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        response = client.embeddings.create(
            input=text,
            model="text-embedding-3-small",
        )
        return response.data[0].embedding
    except Exception as e:
        st.warning(f"Embedding OpenAI não disponível: {str(e)}")
        import hashlib
        text_hash = hashlib.md5(text.encode()).hexdigest()
        vector = [float(int(text_hash[i:i + 2], 16) / 255.0) for i in range(0, 32, 2)]
        while len(vector) < 1536:
            vector.append(0.0)
        return vector[:1536]




def realizar_rag_taxonomia(texto: str, limite: int = 12) -> List[Dict]:
    perguntas = [
        "classificação taxonômica",
        "fungo ou oomiceto",
        "nome científico patógeno",
        "reino filo classe ordem",
        "agente causal doença",
        "Peronospora Phakopsora Corynespora",
        "oomiceto vs fungo diferença",
        "taxonomia fitopatologia",
    ]
    return _busca_multi_query(texto, perguntas, limite)


def realizar_rag_epidemiologia(texto: str, limite: int = 12) -> List[Dict]:
    perguntas = [
        "condições ambientais doença",
        "temperatura umidade molhamento foliar",
        "condições ideais infecção",
        "epidemiologia doença plantas",
        "período molhamento temperatura ótima",
        "umidade relativa infecção",
        "condições climáticas favoráveis",
        "fatores epidemiológicos",
    ]
    return _busca_multi_query(texto, perguntas, limite)


def realizar_rag_produtos(texto: str, limite: int = 12) -> List[Dict]:
    perguntas = [
        "modo de ação produto",
        "aplicação dose recomendada",
        "eficácia controle doença",
        "características técnicas produto",
        "benefícios produto agrícola",
        "tecnologia aplicação",
        "resultados eficácia",
        "recomendações uso produto",
    ]
    return _busca_multi_query(texto, perguntas, limite)


def realizar_rag_geral(texto: str, limite: int = 12) -> List[Dict]:
    embedding = get_embedding(texto[:800])
    return astra_client.vector_search(ASTRA_DB_COLLECTION, embedding, limit=limite)


def processar_rags_especializados(texto: str, rags_ativos: dict, limite: int = 12) -> dict:
    resultados = {}

    if rags_ativos.get('taxonomia'):
        with st.spinner("🔬 Buscando informações de taxonomia..."):
            resultados['taxonomia'] = realizar_rag_taxonomia(texto, limite)

    if rags_ativos.get('epidemiologia'):
        with st.spinner("🌡️ Buscando informações epidemiológicas..."):
            resultados['epidemiologia'] = realizar_rag_epidemiologia(texto, limite)

    if rags_ativos.get('produtos'):
        with st.spinner("🧪 Buscando informações de produtos..."):
            resultados['produtos'] = realizar_rag_produtos(texto, limite)

    if rags_ativos.get('geral'):
        with st.spinner("📚 Buscando informações gerais..."):
            resultados['geral'] = realizar_rag_geral(texto, limite)

    return resultados


# ─── Rewrite com RAG ─────────────────────────────────────────────────────────

def reescrever_com_rag_blog(content: str, modelo_texto, tom_voz: str) -> str:
    """Reescreve conteúdo de blog usando RAG."""
    try:
        embedding = get_embedding(content[:800])
        relevant_docs = astra_client.vector_search(ASTRA_DB_COLLECTION, embedding, limit=10)
        rag_context = _build_rag_context(relevant_docs, "INFORMAÇÕES TÉCNICAS RELEVANTES DA BASE:", limit=500)

        pre_response = modelo_texto.generate_content(f"""
        Entenda o que no texto original de fato é enriquecido e corrigido pelo referencial teórico.
        Considere que você não pode tangenciar o assunto do texto original.

        ###BEGIN TEXTO ORIGINAL###
        {content}
        ###END TEXTO ORIGINAL###

        ###BEGIN REFERENCIAL TEÓRICO###
        {rag_context}
        ###END REFERENCIAL TEÓRICO###
        """)

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


def reescrever_com_rag_revisao_SEO(content: str, modelo_texto, tom_voz: str) -> str:
    """Reescreve conteúdo técnico para revisão SEO."""
    try:
        embedding = get_embedding(content[:800])
        relevant_docs = astra_client.vector_search(ASTRA_DB_COLLECTION, embedding, limit=10)
        rag_context = _build_rag_context(relevant_docs, "DOCUMENTAÇÃO TÉCNICA ESPECIALIZADA:", limit=400)

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


def reescrever_com_rag_revisao_NORM(content: str, modelo_texto, tom_voz: str) -> str:
    """Reescreve conteúdo técnico para revisão normalizada (sem bullets)."""
    try:
        embedding = get_embedding(content[:800])
        relevant_docs = astra_client.vector_search(ASTRA_DB_COLLECTION, embedding, limit=10)
        rag_context = _build_rag_context(relevant_docs, "DOCUMENTAÇÃO TÉCNICA ESPECIALIZADA:", limit=400)

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


def reescrever_com_relatorio_mudancas(texto_original: str, resultados_rags: dict, modelo_texto, contexto_agente: str = "") -> tuple:
    """Reescreve o conteúdo e gera relatório detalhado das mudanças."""
    contexto_rags = _build_rags_context(resultados_rags)

    prompt_reescrita = f"""
    {contexto_agente}

    ## TEXTO ORIGINAL PARA REESCRITA:
    {texto_original}

    ## BASE TÉCNICA DE REFERÊNCIA:
    {contexto_rags}

    ## INSTRUÇÕES CRÍTICAS:

    **SUA TAREFA:**
    1. Reescrever o texto original aplicando correções técnicas baseadas nos documentos de referência
    2. Gerar um relatório DETALHADO de TODAS as mudanças realizadas
    3. Você deve manter a estrutura original do texto. Você deve realizar apenas mudanças e enriquecimentos conforme o contexto novo vindo da base técnica de referência. O texto original deve sempre ser o molde a ser seguido.

    **FORMATO DE SAÍDA EXIGIDO (use exatamente esta estrutura):**

    ### 📝 TEXTO REESCRITO
    [AQUI VOCÊ COLA O TEXTO COMPLETO REESCRITO E CORRIGIDO]

    ### 🔍 RELATÓRIO DETALHADO DE MUDANÇAS

    #### 📊 RESUMO EXECUTIVO
    - Total de correções aplicadas: [N]
    - Principais categorias de ajustes: [lista categorias]
    - Impacto na precisão técnica: [Alto/Médio/Baixo]

    #### 📋 MUDANÇAS DETALHADAS

    **1. CORREÇÕES TAXONÔMICAS:**
    [Lista cada correção taxonômica no formato:
    - **Original:** "texto original"
    - **Corrigido:** "texto corrigido"
    - **Justificativa:** explicação técnica baseada nos documentos]

    **2. PRECISÃO EPIDEMIOLÓGICA:**
    [Lista cada correção epidemiológica no formato:
    - **Original:** "texto original"
    - **Corrigido:** "texto corrigido"
    - **Justificativa:** explicação com base científica]

    **3. INFORMAÇÕES DE PRODUTOS:**
    [Lista cada correção de produtos no formato:
    - **Original:** "texto original"
    - **Corrigido:** "texto corrigido"
    - **Justificativa:** ajuste técnico necessário]

    **4. TERMINOLOGIA TÉCNICA:**
    [Lista cada ajuste de terminologia no formato:
    - **Original:** "termo vago/impreciso"
    - **Corrigido:** "termo técnico preciso"
    - **Justificativa:** padronização técnica]

    **5. DADOS E ESTATÍSTICAS:**
    [Lista cada correção de dados no formato:
    - **Original:** "dado impreciso"
    - **Corrigido:** "dado corrigido"
    - **Justificativa:** fonte/documento de referência]

    #### 🎯 IMPACTO DAS CORREÇÕES
    - Melhorias na precisão científica: [lista específica]
    - Ajustes na comunicação técnica: [lista específica]
    - Correções de segurança da informação: [lista específica]

    **CORREÇÕES TÉCNICAS OBRIGATÓRIAS:**
    1. **PRECISÃO TAXONÔMICA:** Corrigir "fungo" para "oomiceto" quando aplicável
    2. **ESPECIFICIDADE EPIDEMIOLÓGICA:** Substituir termos vagos por faixas específicas
    3. **DESCRIÇÃO PRECISA DE SINTOMAS:** Corrigir descrições imprecisas
    4. **MANEJO E TIMING:** Alinhar mensagens sobre timing de aplicação
    5. **INFORMAÇÕES DE PRODUTOS:** Corrigir claims imprecisos

    **REGRAS ADICIONAIS:**
    - Mantenha a estrutura e formatação do original
    - Apenas corrija o conteúdo técnico, não reinvente a estrutura
    - Para CADA mudança, forneça justificativa técnica específica

    **RETORNE EXATAMENTE no formato especificado acima.**
    """

    try:
        resposta = modelo_texto.generate_content(prompt_reescrita)
        texto_completo = resposta.text

        if "### 📝 TEXTO REESCRITO" in texto_completo and "### 🔍 RELATÓRIO DETALHADO DE MUDANÇAS" in texto_completo:
            partes = texto_completo.split("### 🔍 RELATÓRIO DETALHADO DE MUDANÇAS")
            texto_reescrito = partes[0].replace("### 📝 TEXTO REESCRITO", "").strip()
            relatorio_mudancas = "### 🔍 RELATÓRIO DETALHADO DE MUDANÇAS" + partes[1]
        else:
            texto_reescrito = texto_completo
            relatorio_mudancas = "### ❌ Relatório não gerado automaticamente\nO modelo não seguiu o formato solicitado."

        return texto_reescrito, relatorio_mudancas

    except Exception as e:
        st.error(f"Erro na reescrita: {str(e)}")
        return texto_original, f"### ❌ Erro na geração do relatório\n{str(e)}"


def reescrever_sem_relatorio(texto_original: str, resultados_rags: dict, modelo_texto, contexto_agente: str = "") -> str:
    """Reescreve o conteúdo sem gerar relatório."""
    contexto_rags = _build_rags_context(resultados_rags)

    prompt_rapido = f"""
    {contexto_agente}

    ## TEXTO ORIGINAL PARA REESCRITA:
    {texto_original}

    ## BASE TÉCNICA DE REFERÊNCIA:
    {contexto_rags}

    **REESCREVA o texto aplicando correções técnicas baseadas nos documentos.**
    **RETORNE APENAS o texto reescrito, sem comentários ou relatórios.**

    Correções obrigatórias:
    - Precisão taxonômica (fungo vs oomiceto)
    - Especificidade epidemiológica (temperaturas, umidades)
    - Informações precisas de produtos
    - Terminologia técnica adequada

    Mantenha a estrutura original.
    """

    resposta = modelo_texto.generate_content(prompt_rapido)
    return resposta.text.strip()




def _busca_multi_query(texto: str, perguntas: list, limite: int) -> List[Dict]:
    docs = []
    ids_vistos = set()
    per_query = max(1, limite // len(perguntas))

    for pergunta in perguntas:
        query = f"{texto[:200]} {pergunta}"
        embedding = get_embedding(query)
        resultados = astra_client.vector_search(ASTRA_DB_COLLECTION, embedding, limit=per_query)
        for doc in resultados:
            doc_id = str(doc.get('_id', ''))
            if doc_id not in ids_vistos:
                docs.append(doc)
                ids_vistos.add(doc_id)

    return docs[:limite]


def _build_rag_context(docs: List[Dict], header: str, limit: int = 500) -> str:
    if not docs:
        return "Base de conhecimento não retornou resultados específicos."
    context = header + "\n"
    for i, doc in enumerate(docs, 1):
        doc_clean = str(doc).replace('{', '').replace('}', '').replace("'", "").replace('"', '')
        context += f"--- Fonte {i} ---\n{doc_clean[:limit]}...\n\n"
    return context


def _build_rags_context(resultados_rags: dict) -> str:
    context = "## DOCUMENTOS TÉCNICOS DE REFERÊNCIA:\n\n"
    for categoria, documentos in resultados_rags.items():
        if documentos:
            context += f"### {categoria.upper()} ({len(documentos)} documentos):\n"
            for doc in documentos:
                doc_limpo = str(doc).replace('{', '').replace('}', '').replace("'", "").replace('"', '')
                if len(doc_limpo) > 300:
                    doc_limpo = doc_limpo[:300] + "..."
                context += f"- {doc_limpo}\n"
            context += "\n"
    return context
