import streamlit as st
import io
import google.generativeai as genai
from PIL import Image
import requests
import datetime
import os
from pymongo import MongoClient
from bson import ObjectId
import json
import hashlib
from google.genai import types
import uuid
from typing import List, Dict
import openai
import pandas as pd
import csv
from perplexity import Perplexity

# Configure a API key do Perplexity
perp_api_key = os.getenv("PERP_API_KEY")
if perp_api_key:
    perplexity_client = Perplexity(api_key=perp_api_key)
else:
    st.warning("PERP_API_KEY não encontrada. Busca web estará desativada.")
    perplexity_client = None

# Configurações das credenciais
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
ASTRA_DB_API_ENDPOINT = os.getenv('ASTRA_DB_API_ENDPOINT')
ASTRA_DB_APPLICATION_TOKEN = os.getenv('ASTRA_DB_APPLICATION_TOKEN')
ASTRA_DB_NAMESPACE = os.getenv('ASTRA_DB_NAMESPACE')
ASTRA_DB_COLLECTION = os.getenv('ASTRA_DB_COLLECTION')

class AstraDBClient:
    def __init__(self):
        self.base_url = f"{ASTRA_DB_API_ENDPOINT}/api/json/v1/{ASTRA_DB_NAMESPACE}"
        self.headers = {
            "Content-Type": "application/json",
            "x-cassandra-token": ASTRA_DB_APPLICATION_TOKEN,
            "Accept": "application/json"
        }
    
    def vector_search(self, collection: str, vector: List[float], limit: int = 6) -> List[Dict]:
        """Realiza busca por similaridade vetorial"""
        url = f"{self.base_url}/{collection}"
        payload = {
            "find": {
                "sort": {"$vector": vector},
                "options": {"limit": limit}
            }
        }
        try:
            response = requests.post(url, json=payload, headers=self.headers, timeout=30)
            response.raise_for_status()
            data = response.json()
            return data.get("data", {}).get("documents", [])
        except Exception as e:
            st.error(f"Erro na busca vetorial: {str(e)}")
            return []

# Inicializa o cliente AstraDB
astra_client = AstraDBClient()

def get_embedding(text: str) -> List[float]:
    """Obtém embedding do texto usando OpenAI"""
    try:
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        response = client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding
    except Exception as e:
        st.warning(f"Embedding OpenAI não disponível: {str(e)}")
        # Fallback para embedding simples
        import hashlib
        import numpy as np
        text_hash = hashlib.md5(text.encode()).hexdigest()
        vector = [float(int(text_hash[i:i+2], 16) / 255.0) for i in range(0, 32, 2)]
        # Preenche com valores aleatórios para ter 1536 dimensões
        while len(vector) < 1536:
            vector.append(0.0)
        return vector[:1536]

def reescrever_com_rag_blog(content: str) -> str:
    """REESCREVE conteúdo de blog usando RAG - SAÍDA DIRETA DO CONTEÚDO REESCRITO"""
    try:
        # Gera embedding para busca
        embedding = get_embedding(content[:800])
        
        # Busca documentos relevantes
        relevant_docs = astra_client.vector_search(ASTRA_DB_COLLECTION, embedding, limit=10)
        
        # Constrói contexto dos documentos
        rag_context = ""
        if relevant_docs:
            rag_context = "INFORMAÇÕES TÉCNICAS RELEVANTES DA BASE:\n"
            for i, doc in enumerate(relevant_docs, 1):
                doc_content = str(doc)
                # Limpa e formata o documento
                doc_clean = doc_content.replace('{', '').replace('}', '').replace("'", "").replace('"', '')
                rag_context += f"--- Fonte {i} ---\n{doc_clean[:500]}...\n\n"
        else:
            rag_context = "Base de conhecimento não retornou resultados específicos."

        # Prompt de entendimento RAG
        rewrite_prompt = f"""

        Entenda o que no texto original de fato é enriquecido e corrigido pelo referencial teórico. Considere que você não pode tangenciar o assunto do texto original.
    
        ###BEGIN TEXTO ORIGINAL###
        {content}
        ###END TEXTO ORIGINAL###

        ###BEGIN REFERENCIAL TEÓRICO###
        {rag_context}
        ###END REFERENCIAL TEÓRICO###
        
        
        """

        # Gera conteúdo REEESCRITO
        pre_response = modelo_texto.generate_content(rewrite_prompt)

        # Saída final
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

def reescrever_com_rag_revisao_SEO(content: str) -> str:
    """REESCREVE conteúdo técnico para revisão - SAÍDA DIRETA DO CONTEÚDO REESCRITO"""
    try:
        # Gera embedding para busca
        embedding = get_embedding(content[:800])
        
        # Busca documentos relevantes
        relevant_docs = astra_client.vector_search(ASTRA_DB_COLLECTION, embedding, limit=10)
        
        # Constrói contexto dos documentos
        rag_context = ""
        if relevant_docs:
            rag_context = "DOCUMENTAÇÃO TÉCNICA ESPECIALIZADA:\n"
            for i, doc in enumerate(relevant_docs, 1):
                doc_content = str(doc)
                doc_clean = doc_content.replace('{', '').replace('}', '').replace("'", "").replace('"', '')
                rag_context += f"--- Documento Técnico {i} ---\n{doc_clean[:400]}...\n\n"
        else:
            rag_context = "Consulta técnica não retornou documentos específicos."

        # Prompt de REWRITE TÉCNICO AVANÇADO
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

        # Gera conteúdo técnico REEESCRITO
        response = modelo_texto.generate_content(rewrite_prompt)
        return response.text
        
    except Exception as e:
        st.error(f"Erro no RAG rewrite técnico: {str(e)}")
        return content

def reescrever_com_rag_revisao_NORM(content: str) -> str:
    """REESCREVE conteúdo técnico para revisão - SAÍDA DIRETA DO CONTEÚDO REESCRITO"""
    try:
        # Gera embedding para busca
        embedding = get_embedding(content[:800])
        
        # Busca documentos relevantes
        relevant_docs = astra_client.vector_search(ASTRA_DB_COLLECTION, embedding, limit=10)
        
        # Constrói contexto dos documentos
        rag_context = ""
        if relevant_docs:
            rag_context = "DOCUMENTAÇÃO TÉCNICA ESPECIALIZADA:\n"
            for i, doc in enumerate(relevant_docs, 1):
                doc_content = str(doc)
                doc_clean = doc_content.replace('{', '').replace('}', '').replace("'", "").replace('"', '')
                rag_context += f"--- Documento Técnico {i} ---\n{doc_clean[:400]}...\n\n"
        else:
            rag_context = "Consulta técnica não retornou documentos específicos."

        # Prompt de REWRITE TÉCNICO AVANÇADO
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

        # Gera conteúdo técnico REEESCRITO
        response = modelo_texto.generate_content(rewrite_prompt)
        return response.text
        
    except Exception as e:
        st.error(f"Erro no RAG rewrite técnico: {str(e)}")
        return content

# Configuração inicial
st.set_page_config(
    layout="wide",
    page_title="Conteúdo")

# --- Sistema de Autenticação ---
def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password, hashed_text):
    return make_hashes(password) == hashed_text

# Dados de usuário (em produção, isso deve vir de um banco de dados seguro)
users = {
    "admin": make_hashes("senha1234"),  # admin/senha1234
    "SYN": make_hashes("senha1"),  # user1/password1
    "SME": make_hashes("senha2"),   # user2/password2
    "Enterprise": make_hashes("senha3")   # user2/password2
}

def get_current_user():
    """Retorna o usuário atual da sessão"""
    return st.session_state.get('user', 'unknown')

def login():
    """Formulário de login"""
    
    with st.form("login_form"):
        username = st.text_input("Usuário")
        password = st.text_input("Senha", type="password")
        submit_button = st.form_submit_button("Login")
        
        if submit_button:
            if username in users and check_hashes(password, users[username]):
                st.session_state.logged_in = True
                st.session_state.user = username
                st.success("Login realizado com sucesso!")
                st.rerun()
            else:
                st.error("Usuário ou senha incorretos")

# Verificar se o usuário está logado
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    login()
    st.stop()

# --- CONEXÃO MONGODB (após login) ---
client = MongoClient("mongodb+srv://gustavoromao3345:RqWFPNOJQfInAW1N@cluster0.5iilj.mongodb.net/auto_doc?retryWrites=true&w=majority&ssl=true&ssl_cert_reqs=CERT_NONE&tlsAllowInvalidCertificates=true")
db = client['agentes_personalizados']
collection_agentes = db['agentes']
collection_conversas = db['conversas']

# Configuração da API do Gemini
gemini_api_key = os.getenv("GEM_API_KEY")
if not gemini_api_key:
    st.error("GEMINI_API_KEY não encontrada nas variáveis de ambiente")
    st.stop()

genai.configure(api_key=gemini_api_key)
modelo_vision = genai.GenerativeModel("gemini-2.5-flash", generation_config={"temperature": 0.0})
modelo_texto = genai.GenerativeModel("gemini-2.5-flash")
modelo_texto2 = genai.GenerativeModel("gemini-2.5-pro")

# --- Funções CRUD para Agentes ---
def criar_agente(nome, system_prompt, base_conhecimento, comments, planejamento, categoria, agente_mae_id=None, herdar_elementos=None):
    """Cria um novo agente no MongoDB"""
    agente = {
        "nome": nome,
        "system_prompt": system_prompt,
        "base_conhecimento": base_conhecimento,
        "comments": comments,
        "planejamento": planejamento,
        "categoria": categoria,
        "agente_mae_id": agente_mae_id,
        "herdar_elementos": herdar_elementos or [],
        "ativo": True,
        "data_criacao": datetime.datetime.now(),
        "criado_por": get_current_user()  # NOVO CAMPO
    }
    result = collection_agentes.insert_one(agente)
    return result.inserted_id

def listar_agentes():
    """Retorna todos os agentes ativos do usuário atual ou todos se admin"""
    current_user = get_current_user()
    if current_user == "admin":
        return list(collection_agentes.find({"ativo": True}).sort("data_criacao", -1))
    else:
        return list(collection_agentes.find({
            "ativo": True, 
            "criado_por": current_user
        }).sort("data_criacao", -1))

def listar_agentes_para_heranca(agente_atual_id=None):
    """Retorna todos os agentes ativos que podem ser usados como mãe"""
    current_user = get_current_user()
    query = {"ativo": True}
    
    # Filtro por usuário (admin vê todos, outros só os seus)
    if current_user != "admin":
        query["criado_por"] = current_user
    
    if agente_atual_id:
        # Excluir o próprio agente da lista de opções para evitar auto-herança
        if isinstance(agente_atual_id, str):
            agente_atual_id = ObjectId(agente_atual_id)
        query["_id"] = {"$ne": agente_atual_id}
    
    return list(collection_agentes.find(query).sort("data_criacao", -1))

def obter_agente(agente_id):
    """Obtém um agente específico pelo ID com verificação de permissão"""
    if isinstance(agente_id, str):
        agente_id = ObjectId(agente_id)
    
    agente = collection_agentes.find_one({"_id": agente_id})
    
    # Verificar permissão
    if agente and agente.get('ativo', True):
        current_user = get_current_user()
        if current_user == "admin" or agente.get('criado_por') == current_user:
            return agente
    
    return None

def atualizar_agente(agente_id, nome, system_prompt, base_conhecimento, comments, planejamento, categoria, agente_mae_id=None, herdar_elementos=None):
    """Atualiza um agente existente com verificação de permissão"""
    if isinstance(agente_id, str):
        agente_id = ObjectId(agente_id)
    
    # Verificar se o usuário tem permissão para editar este agente
    agente_existente = obter_agente(agente_id)
    if not agente_existente:
        raise PermissionError("Agente não encontrado ou sem permissão de edição")
    
    return collection_agentes.update_one(
        {"_id": agente_id},
        {
            "$set": {
                "nome": nome,
                "system_prompt": system_prompt,
                "base_conhecimento": base_conhecimento,
                "comments": comments,
                "planejamento": planejamento,
                "categoria": categoria,
                "agente_mae_id": agente_mae_id,
                "herdar_elementos": herdar_elementos or [],
            }
        }
    )

def desativar_agente(agente_id):
    """Desativa um agente (soft delete) com verificação de permissão"""
    if isinstance(agente_id, str):
        agente_id = ObjectId(agente_id)
    
    # Verificar se o usuário tem permissão para desativar este agente
    agente_existente = obter_agente(agente_id)
    if not agente_existente:
        raise PermissionError("Agente não encontrado ou sem permissão para desativar")
    
    return collection_agentes.update_one(
        {"_id": agente_id},
        {"$set": {"ativo": False}}
    )

def obter_agente_com_heranca(agente_id):
    """Obtém um agente com os elementos herdados aplicados"""
    agente = obter_agente(agente_id)
    if not agente or not agente.get('agente_mae_id'):
        return agente
    
    agente_mae = obter_agente(agente['agente_mae_id'])
    if not agente_mae:
        return agente
    
    elementos_herdar = agente.get('herdar_elementos', [])
    agente_completo = agente.copy()
    
    for elemento in elementos_herdar:
        if elemento == 'system_prompt' and not agente_completo.get('system_prompt'):
            agente_completo['system_prompt'] = agente_mae.get('system_prompt', '')
        elif elemento == 'base_conhecimento' and not agente_completo.get('base_conhecimento'):
            agente_completo['base_conhecimento'] = agente_mae.get('base_conhecimento', '')
        elif elemento == 'comments' and not agente_completo.get('comments'):
            agente_completo['comments'] = agente_mae.get('comments', '')
        elif elemento == 'planejamento' and not agente_completo.get('planejamento'):
            agente_completo['planejamento'] = agente_mae.get('planejamento', '')
    
    return agente_completo

def salvar_conversa(agente_id, mensagens, segmentos_utilizados=None):
    """Salva uma conversa no histórico"""
    if isinstance(agente_id, str):
        agente_id = ObjectId(agente_id)
    conversa = {
        "agente_id": agente_id,
        "mensagens": mensagens,
        "segmentos_utilizados": segmentos_utilizados,
        "data_criacao": datetime.datetime.now()
    }
    return collection_conversas.insert_one(conversa)

def obter_conversas(agente_id, limite=10):
    """Obtém o histórico de conversas de um agente"""
    if isinstance(agente_id, str):
        agente_id = ObjectId(agente_id)
    return list(collection_conversas.find(
        {"agente_id": agente_id}
    ).sort("data_criacao", -1).limit(limite))

# --- Função para construir contexto com segmentos selecionados ---
def construir_contexto(agente, segmentos_selecionados, historico_mensagens=None):
    """Constrói o contexto com base nos segmentos selecionados"""
    contexto = ""
    
    if "system_prompt" in segmentos_selecionados and agente.get('system_prompt'):
        contexto += f"### INSTRUÇÕES DO SISTEMA ###\n{agente['system_prompt']}\n\n"
    
    if "base_conhecimento" in segmentos_selecionados and agente.get('base_conhecimento'):
        contexto += f"### BASE DE CONHECIMENTO ###\n{agente['base_conhecimento']}\n\n"
    
    if "comments" in segmentos_selecionados and agente.get('comments'):
        contexto += f"### COMENTÁRIOS DO CLIENTE ###\n{agente['comments']}\n\n"
    
    if "planejamento" in segmentos_selecionados and agente.get('planejamento'):
        contexto += f"### PLANEJAMENTO ###\n{agente['planejamento']}\n\n"
    
    # Adicionar histórico se fornecido
    if historico_mensagens:
        contexto += "### HISTÓRICO DA CONVERSA ###\n"
        for msg in historico_mensagens:
            contexto += f"{msg['role']}: {msg['content']}\n"
        contexto += "\n"
    
    contexto += "### RESPOSTA ATUAL ###\nassistant:"
    
    return contexto

# --- Funções para Transcrição de Áudio/Video ---
def transcrever_audio_video(arquivo, tipo_arquivo):
    """Transcreve áudio ou vídeo usando a API do Gemini"""
    try:
        client = genai.Client(api_key=gemini_api_key)
        
        if tipo_arquivo == "audio":
            mime_type = f"audio/{arquivo.name.split('.')[-1]}"
        else:  # video
            mime_type = f"video/{arquivo.name.split('.')[-1]}"
        
        # Lê os bytes do arquivo
        arquivo_bytes = arquivo.read()
        
        # Para arquivos maiores, usa upload
        if len(arquivo_bytes) > 20 * 1024 * 1024:  # 20MB
            uploaded_file = client.files.upload(file=arquivo_bytes, mime_type=mime_type)
            response = client.models.generate_content(
                model="gemini-2.0-flash", 
                contents=["Transcreva este arquivo em detalhes:", uploaded_file]
            )
        else:
            # Para arquivos menores, usa inline
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[
                    "Transcreva este arquivo em detalhes:",
                    types.Part.from_bytes(data=arquivo_bytes, mime_type=mime_type)
                ]
            )
        
        return response.text
    except Exception as e:
        return f"Erro na transcrição: {str(e)}"

# --- Configuração de Autenticação de Administrador ---
def check_admin_password():
    """Retorna True se o usuário fornecer a senha de admin correta."""
    
    def admin_password_entered():
        """Verifica se a senha de admin está correta."""
        if st.session_state["admin_password"] == "senha123":
            st.session_state["admin_password_correct"] = True
            st.session_state["admin_user"] = "admin"
            del st.session_state["admin_password"]
        else:
            st.session_state["admin_password_correct"] = False

    if "admin_password_correct" not in st.session_state:
        # Mostra o input para senha de admin
        st.text_input(
            "Senha de Administrador", 
            type="password", 
            on_change=admin_password_entered, 
            key="admin_password"
        )
        return False
    elif not st.session_state["admin_password_correct"]:
        # Senha incorreta, mostra input + erro
        st.text_input(
            "Senha de Administrador", 
            type="password", 
            on_change=admin_password_entered, 
            key="admin_password"
        )
        st.error("😕 Senha de administrador incorreta")
        return False
    else:
        # Senha correta
        return True

# ========== SELEÇÃO EXTERNA DE AGENTE ==========
st.image('macLogo.png', width=300)
st.title("Conteúdo")

# Botão de logout na sidebar
if st.button("🚪 Sair", key="logout_btn"):
    for key in ["logged_in", "user", "admin_password_correct", "admin_user"]:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()

# --- SELEÇÃO DE AGENTE EXTERNA ---
st.header("🤖 Selecione a base de conhecimento")

# Inicializar estado da sessão para agente selecionado
if "agente_selecionado" not in st.session_state:
    st.session_state.agente_selecionado = None
if "segmentos_selecionados" not in st.session_state:
    st.session_state.segmentos_selecionados = ["system_prompt", "base_conhecimento", "comments", "planejamento"]

# Carregar agentes (agora filtrados por usuário)
agentes = listar_agentes()

# Container para seleção de agente
with st.container():
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col1:
        if agentes:
            # Agrupar agentes por categoria
            agentes_por_categoria = {}
            for agente in agentes:
                categoria = agente.get('categoria', 'Social')
                if categoria not in agentes_por_categoria:
                    agentes_por_categoria[categoria] = []
                agentes_por_categoria[categoria].append(agente)
            
            # Criar opções de seleção com agrupamento
            agente_options = {}
            for categoria, agentes_cat in agentes_por_categoria.items():
                for agente in agentes_cat:
                    agente_completo = obter_agente_com_heranca(agente['_id'])
                    if agente_completo:  # Só adiciona se tiver permissão
                        display_name = f"{agente['nome']} ({categoria})"
                        if agente.get('agente_mae_id'):
                            display_name += " 🔗"
                        # Adicionar indicador de proprietário se não for admin
                        if get_current_user() != "admin" and agente.get('criado_por'):
                            display_name += f" 👤"
                        agente_options[display_name] = agente_completo
            
            if agente_options:
                # Seletor de agente
                agente_selecionado_display = st.selectbox(
                    "Selecione um agente para trabalhar:", 
                    list(agente_options.keys()),
                    key="seletor_agente_global"
                )
                
                # Botão para aplicar agente
                if st.button("🔄 Aplicar Agente", key="aplicar_agente"):
                    st.session_state.agente_selecionado = agente_options[agente_selecionado_display]
                    st.success(f"Agente '{agente_selecionado_display}' selecionado!")
                    st.rerun()
            else:
                st.info("Nenhum agente disponível com as permissões atuais.")
        
        else:
            st.info("Nenhum agente disponível. Crie um agente primeiro na aba de Gerenciamento.")
    
    with col2:
        # Botão para limpar agente selecionado
        if st.session_state.agente_selecionado:
            if st.button("🗑️ Limpar Agente", key="limpar_agente"):
                st.session_state.agente_selecionado = None
                st.session_state.messages = []
                st.success("Agente removido!")
                st.rerun()
    
    with col3:
        # Botão para recarregar lista
        if st.button("🔄 Recarregar", key="recarregar_agentes"):
            st.rerun()

# Mostrar agente atual selecionado
if st.session_state.agente_selecionado:
    agente_atual = st.session_state.agente_selecionado
    
    # Container para informações do agente
    with st.container():
        st.success(f"**✅ Agente Ativo:** {agente_atual['nome']} ({agente_atual.get('categoria', 'Social')})")
        
        # Mostrar informações de herança se aplicável
        if 'agente_mae_id' in agente_atual and agente_atual['agente_mae_id']:
            agente_original = obter_agente(agente_atual['_id'])
            if agente_original and agente_original.get('herdar_elementos'):
                st.info(f"🔗 Este agente herda {len(agente_original['herdar_elementos'])} elementos do agente mãe")
        
        # Mostrar segmentos ativos
        st.info(f"📋 Segmentos ativos: {', '.join(st.session_state.segmentos_selecionados)}")
        
        # Botão para alterar segmentos
        if st.button("⚙️ Alterar Segmentos", key="alterar_segmentos"):
            # Toggle para mostrar/ocultar configuração de segmentos
            if "mostrar_segmentos" not in st.session_state:
                st.session_state.mostrar_segmentos = True
            else:
                st.session_state.mostrar_segmentos = not st.session_state.mostrar_segmentos
        
        # Mostrar configuração de segmentos se solicitado
        if st.session_state.get('mostrar_segmentos', False):
            with st.expander("🔧 Configurar Segmentos do Agente", expanded=True):
                st.write("Selecione quais elementos do agente serão utilizados:")
                
                col_seg1, col_seg2, col_seg3, col_seg4 = st.columns(4)
                
                with col_seg1:
                    system_prompt_ativado = st.checkbox("System Prompt", 
                                                      value="system_prompt" in st.session_state.segmentos_selecionados,
                                                      key="seg_system")
                with col_seg2:
                    base_conhecimento_ativado = st.checkbox("Brand Guidelines", 
                                                          value="base_conhecimento" in st.session_state.segmentos_selecionados,
                                                          key="seg_base")
                with col_seg3:
                    comments_ativado = st.checkbox("Comentários", 
                                                 value="comments" in st.session_state.segmentos_selecionados,
                                                 key="seg_comments")
                with col_seg4:
                    planejamento_ativado = st.checkbox("Planejamento", 
                                                     value="planejamento" in st.session_state.segmentos_selecionados,
                                                     key="seg_planejamento")
                
                if st.button("✅ Aplicar Segmentos", key="aplicar_segmentos"):
                    novos_segmentos = []
                    if system_prompt_ativado:
                        novos_segmentos.append("system_prompt")
                    if base_conhecimento_ativado:
                        novos_segmentos.append("base_conhecimento")
                    if comments_ativado:
                        novos_segmentos.append("comments")
                    if planejamento_ativado:
                        novos_segmentos.append("planejamento")
                    
                    st.session_state.segmentos_selecionados = novos_segmentos
                    st.success(f"Segmentos atualizados: {', '.join(novos_segmentos)}")
                    st.session_state.mostrar_segmentos = False
                    st.rerun()

else:
    st.warning("⚠️ Nenhum agente selecionado. Selecione um agente acima para começar.")

st.markdown("---")

# Menu de abas - AGORA COM A NOVA ABA DE CALENDÁRIO
tab_chat, tab_gerenciamento, tab_conteudo, tab_blog, tab_revisao_ortografica, tab_revisao_tecnica, tab_otimizacao, tab_calendario, tab_briefings, tab_revisao_tecnica2 = st.tabs([
    "💬 Chat", 
    "⚙️ Gerenciar Agentes",
    "✨ Geração de Conteúdo", 
    "🌱 Geração de Conteúdo Blog",
    "📝 Revisão Ortográfica",
    "🔧 Revisão Técnica",
    "🚀 Otimização de Conteúdo",
    "📅 Criadora de Calendário",
    "📋 Gerador de Briefings",
    "Revisão Técnica Sem RAG" # NOVA ABA
])

# ========== ABA: CHAT ==========
with tab_chat:
    st.header("💬 Chat com Agente")
    
    # Inicializar estado da sessão
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Verificar se há agente selecionado
    if not st.session_state.agente_selecionado:
        st.info("Selecione um agente na parte superior do app para iniciar o chat.")
    else:
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
                        st.session_state.messages.append({"role": "assistant", "content": resposta.text})
                        
                        # Salvar conversa com segmentos utilizados
                        salvar_conversa(
                            agente['_id'], 
                            st.session_state.messages,
                            st.session_state.segmentos_selecionados
                        )
                        
                    except Exception as e:
                        st.error(f"Erro ao gerar resposta: {str(e)}")

# ========== ABA: GERENCIAMENTO DE AGENTES ==========
with tab_gerenciamento:
    st.header("⚙️ Gerenciamento de Agentes")
    
    # Verificar autenticação apenas para gerenciamento
    current_user = get_current_user()
    
    if current_user not in ["admin", "SYN", "SME", "Enterprise"]:
        st.warning("Acesso restrito a usuários autorizados")
    else:
        # Para admin, verificar senha adicional
        if current_user == "admin":
            if not check_admin_password():
                st.warning("Digite a senha de administrador")
            else:
                st.write(f'Bem-vindo administrador!')
        else:
            st.write(f'Bem-vindo {current_user}!')
            
        # Subabas para gerenciamento
        sub_tab1, sub_tab2, sub_tab3 = st.tabs(["Criar Agente", "Editar Agente", "Gerenciar Agentes"])
        
        with sub_tab1:
            st.subheader("Criar Novo Agente")
            
            with st.form("form_criar_agente"):
                nome_agente = st.text_input("Nome do Agente:")
                
                # Seleção de categoria
                categoria = st.selectbox(
                    "Categoria:",
                    ["Social", "SEO", "Conteúdo"],
                    help="Organize o agente por área de atuação"
                )
                
                # Opção para criar como agente filho
                criar_como_filho = st.checkbox("Criar como agente filho (herdar elementos)")
                
                agente_mae_id = None
                herdar_elementos = []
                
                if criar_como_filho:
                    # Listar TODOS os agentes disponíveis para herança
                    agentes_mae = listar_agentes_para_heranca()
                    if agentes_mae:
                        agente_mae_options = {f"{agente['nome']} ({agente.get('categoria', 'Social')})": agente['_id'] for agente in agentes_mae}
                        agente_mae_selecionado = st.selectbox(
                            "Agente Mãe:",
                            list(agente_mae_options.keys()),
                            help="Selecione o agente do qual este agente irá herdar elementos"
                        )
                        agente_mae_id = agente_mae_options[agente_mae_selecionado]
                        
                        st.subheader("Elementos para Herdar")
                        herdar_elementos = st.multiselect(
                            "Selecione os elementos a herdar do agente mãe:",
                            ["system_prompt", "base_conhecimento", "comments", "planejamento"],
                            help="Estes elementos serão herdados do agente mãe se não preenchidos abaixo"
                        )
                    else:
                        st.info("Nenhum agente disponível para herança. Crie primeiro um agente mãe.")
                
                system_prompt = st.text_area("Prompt de Sistema:", height=150, 
                                            placeholder="Ex: Você é um assistente especializado em...",
                                            help="Deixe vazio se for herdar do agente mãe")
                base_conhecimento = st.text_area("Brand Guidelines:", height=200,
                                               placeholder="Cole aqui informações, diretrizes, dados...",
                                               help="Deixe vazio se for herdar do agente mãe")
                comments = st.text_area("Comentários do cliente:", height=200,
                                               placeholder="Cole aqui os comentários de ajuste do cliente (Se houver)",
                                               help="Deixe vazio se for herdar do agente mãe")
                planejamento = st.text_area("Planejamento:", height=200,
                                           placeholder="Estratégias, planejamentos, cronogramas...",
                                           help="Deixe vazio se for herdar do agente mãe")
                
                submitted = st.form_submit_button("Criar Agente")
                if submitted:
                    if nome_agente:
                        agente_id = criar_agente(
                            nome_agente, 
                            system_prompt, 
                            base_conhecimento, 
                            comments, 
                            planejamento,
                            categoria,
                            agente_mae_id if criar_como_filho else None,
                            herdar_elementos if criar_como_filho else []
                        )
                        st.success(f"Agente '{nome_agente}' criado com sucesso na categoria {categoria}!")
                    else:
                        st.error("Nome é obrigatório!")
        
        with sub_tab2:
            st.subheader("Editar Agente Existente")
            
            agentes = listar_agentes()
            if agentes:
                agente_options = {agente['nome']: agente for agente in agentes}
                agente_selecionado_nome = st.selectbox("Selecione o agente para editar:", 
                                                     list(agente_options.keys()))
                
                if agente_selecionado_nome:
                    agente = agente_options[agente_selecionado_nome]
                    
                    with st.form("form_editar_agente"):
                        novo_nome = st.text_input("Nome do Agente:", value=agente['nome'])
                        
                        # Categoria
                        
                        
                        # Informações de herança
                        if agente.get('agente_mae_id'):
                            agente_mae = obter_agente(agente['agente_mae_id'])
                            if agente_mae:
                                st.info(f"🔗 Este agente é filho de: {agente_mae['nome']}")
                                st.write(f"Elementos herdados: {', '.join(agente.get('herdar_elementos', []))}")
                        
                        # Opção para tornar independente
                        if agente.get('agente_mae_id'):
                            tornar_independente = st.checkbox("Tornar agente independente (remover herança)")
                            if tornar_independente:
                                agente_mae_id = None
                                herdar_elementos = []
                            else:
                                agente_mae_id = agente.get('agente_mae_id')
                                herdar_elementos = agente.get('herdar_elementos', [])
                        else:
                            agente_mae_id = None
                            herdar_elementos = []
                            # Opção para adicionar herança
                            adicionar_heranca = st.checkbox("Adicionar herança de agente mãe")
                            if adicionar_heranca:
                                # Listar TODOS os agentes disponíveis para herança (excluindo o próprio)
                                agentes_mae = listar_agentes_para_heranca(agente['_id'])
                                if agentes_mae:
                                    agente_mae_options = {f"{agente_mae['nome']} ({agente_mae.get('categoria', 'Social')})": agente_mae['_id'] for agente_mae in agentes_mae}
                                    if agente_mae_options:
                                        agente_mae_selecionado = st.selectbox(
                                            "Agente Mãe:",
                                            list(agente_mae_options.keys()),
                                            help="Selecione o agente do qual este agente irá herdar elementos"
                                        )
                                        agente_mae_id = agente_mae_options[agente_mae_selecionado]
                                        herdar_elementos = st.multiselect(
                                            "Elementos para herdar:",
                                            ["system_prompt", "base_conhecimento", "comments", "planejamento"],
                                            default=herdar_elementos
                                        )
                                    else:
                                        st.info("Nenhum agente disponível para herança.")
                                else:
                                    st.info("Nenhum agente disponível para herança.")
                        
                        novo_prompt = st.text_area("Prompt de Sistema:", value=agente['system_prompt'], height=150)
                        nova_base = st.text_area("Brand Guidelines:", value=agente.get('base_conhecimento', ''), height=200)
                        nova_comment = st.text_area("Comentários:", value=agente.get('comments', ''), height=200)
                        novo_planejamento = st.text_area("Planejamento:", value=agente.get('planejamento', ''), height=200)
                        
                        submitted = st.form_submit_button("Atualizar Agente")
                        if submitted:
                            if novo_nome:
                                atualizar_agente(
                                    agente['_id'], 
                                    novo_nome, 
                                    novo_prompt, 
                                    nova_base, 
                                    nova_comment, 
                                    novo_planejamento,
                                    agente_mae_id,
                                    herdar_elementos
                                )
                                st.success(f"Agente '{novo_nome}' atualizado com sucesso!")
                                st.rerun()
                            else:
                                st.error("Nome é obrigatório!")
            else:
                st.info("Nenhum agente criado ainda.")
        
        with sub_tab3:
            st.subheader("Gerenciar Agentes")
            
            # Mostrar informações do usuário atual
            if current_user == "admin":
                st.info("👑 Modo Administrador: Visualizando todos os agentes do sistema")
            else:
                st.info(f"👤 Visualizando apenas seus agentes ({current_user})")
            
            # Filtros por categoria
            categorias = ["Todos", "Social", "SEO", "Conteúdo"]
            categoria_filtro = st.selectbox("Filtrar por categoria:", categorias)
            
            agentes = listar_agentes()
            
            # Aplicar filtro
            if categoria_filtro != "Todos":
                agentes = [agente for agente in agentes if agente.get('categoria') == categoria_filtro]
            
            if agentes:
                for i, agente in enumerate(agentes):
                    with st.container():
                        # Mostrar proprietário se for admin
                        owner_info = ""
                        if current_user == "admin" and agente.get('criado_por'):
                            owner_info = f" | 👤 {agente['criado_por']}"
                        
                        st.write(f"**{agente['nome']} - {agente.get('categoria', 'Social')}{owner_info} - Criado em {agente['data_criacao'].strftime('%d/%m/%Y')}**")
                        
                        # Mostrar informações de herança
                        if agente.get('agente_mae_id'):
                            agente_mae = obter_agente(agente['agente_mae_id'])
                            if agente_mae:
                                st.write(f"**🔗 Herda de:** {agente_mae['nome']}")
                                st.write(f"**Elementos herdados:** {', '.join(agente.get('herdar_elementos', []))}")
                        
                        st.write(f"**Prompt de Sistema:** {agente['system_prompt'][:100]}..." if agente['system_prompt'] else "**Prompt de Sistema:** (herdado ou vazio)")
                        if agente.get('base_conhecimento'):
                            st.write(f"**Brand Guidelines:** {agente['base_conhecimento'][:200]}...")
                        if agente.get('comments'):
                            st.write(f"**Comentários do cliente:** {agente['comments'][:200]}...")
                        if agente.get('planejamento'):
                            st.write(f"**Planejamento:** {agente['planejamento'][:200]}...")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("Selecionar para Chat", key=f"select_{i}"):
                                st.session_state.agente_selecionado = obter_agente_com_heranca(agente['_id'])
                                st.session_state.messages = []
                                st.success(f"Agente '{agente['nome']}' selecionado!")
                        with col2:
                            if st.button("Desativar", key=f"delete_{i}"):
                                desativar_agente(agente['_id'])
                                st.success(f"Agente '{agente['nome']}' desativado!")
                                st.rerun()
                        st.divider()
            else:
                st.info("Nenhum agente encontrado para esta categoria.")


# ========== ABA: GERAÇÃO DE CONTEÚDO ==========
with tab_conteudo:
    st.header("✨ Geração de Conteúdo com Múltiplos Insumos")
    
    # Conexão com MongoDB para briefings
    try:
        client2 = MongoClient("mongodb+srv://gustavoromao3345:RqWFPNOJQfInAW1N@cluster0.5iilj.mongodb.net/auto_doc?retryWrites=true&w=majority&ssl=true&ssl_cert_reqs=CERT_NONE&tlsAllowInvalidCertificates=true")
        db_briefings = client2['briefings_Broto_Tecnologia']
        collection_briefings = db_briefings['briefings']
        mongo_connected_conteudo = True
    except Exception as e:
        st.error(f"Erro na conexão com MongoDB: {str(e)}")
        mongo_connected_conteudo = False

    # Função para extrair texto de diferentes tipos de arquivo
    def extrair_texto_arquivo(arquivo):
        """Extrai texto de diferentes formatos de arquivo"""
        try:
            extensao = arquivo.name.split('.')[-1].lower()
            
            if extensao == 'pdf':
                return extrair_texto_pdf(arquivo)
            elif extensao == 'txt':
                return extrair_texto_txt(arquivo)
            elif extensao in ['pptx', 'ppt']:
                return extrair_texto_pptx(arquivo)
            elif extensao in ['docx', 'doc']:
                return extrair_texto_docx(arquivo)
            else:
                return f"Formato {extensao} não suportado para extração de texto."
                
        except Exception as e:
            return f"Erro ao extrair texto do arquivo {arquivo.name}: {str(e)}"

    def extrair_texto_pdf(arquivo):
        """Extrai texto de arquivos PDF"""
        try:
            import PyPDF2
            pdf_reader = PyPDF2.PdfReader(arquivo)
            texto = ""
            for pagina in pdf_reader.pages:
                texto += pagina.extract_text() + "\n"
            return texto
        except Exception as e:
            return f"Erro na leitura do PDF: {str(e)}"

    def extrair_texto_txt(arquivo):
        """Extrai texto de arquivos TXT"""
        try:
            return arquivo.read().decode('utf-8')
        except:
            try:
                return arquivo.read().decode('latin-1')
            except Exception as e:
                return f"Erro na leitura do TXT: {str(e)}"

    def extrair_texto_pptx(arquivo):
        """Extrai texto de arquivos PowerPoint"""
        try:
            from pptx import Presentation
            import io
            prs = Presentation(io.BytesIO(arquivo.read()))
            texto = ""
            for slide in prs.slides:
                for shape in slide.shapes:
                    if hasattr(shape, "text"):
                        texto += shape.text + "\n"
            return texto
        except Exception as e:
            return f"Erro na leitura do PowerPoint: {str(e)}"

    def extrair_texto_docx(arquivo):
        """Extrai texto de arquivos Word"""
        try:
            import docx
            import io
            doc = docx.Document(io.BytesIO(arquivo.read()))
            texto = ""
            for para in doc.paragraphs:
                texto += para.text + "\n"
            return texto
        except Exception as e:
            return f"Erro na leitura do Word: {str(e)}"

    # Layout principal
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📝 Fontes de Conteúdo")
        
        # Opção 1: Upload de múltiplos arquivos
        st.write("📎 Upload de Arquivos (PDF, TXT, PPTX, DOCX):")
        arquivos_upload = st.file_uploader(
            "Selecione um ou mais arquivos:",
            type=['pdf', 'txt', 'pptx', 'ppt', 'docx', 'doc'],
            accept_multiple_files=True,
            help="Arquivos serão convertidos para texto e usados como base para geração de conteúdo"
        )
        
        # Processar arquivos uploadados
        textos_arquivos = ""
        if arquivos_upload:
            st.success(f"✅ {len(arquivos_upload)} arquivo(s) carregado(s)")
            
            with st.expander("📋 Visualizar Conteúdo dos Arquivos", expanded=False):
                for i, arquivo in enumerate(arquivos_upload):
                    st.write(f"**{arquivo.name}** ({arquivo.size} bytes)")
                    with st.spinner(f"Processando {arquivo.name}..."):
                        texto_extraido = extrair_texto_arquivo(arquivo)
                        textos_arquivos += f"\n\n--- CONTEÚDO DE {arquivo.name.upper()} ---\n{texto_extraido}"
                        
                        # Mostrar preview
                        if len(texto_extraido) > 500:
                            st.text_area(f"Preview - {arquivo.name}", 
                                       value=texto_extraido[:500] + "...", 
                                       height=100,
                                       key=f"preview_{i}")
                        else:
                            st.text_area(f"Preview - {arquivo.name}", 
                                       value=texto_extraido, 
                                       height=100,
                                       key=f"preview_{i}")
        
        # Opção 2: Selecionar briefing do banco de dados
        st.write("🗃️ Briefing do Banco de Dados:")
        if mongo_connected_conteudo:
            briefings_disponiveis = list(collection_briefings.find().sort("data_criacao", -1).limit(20))
            if briefings_disponiveis:
                briefing_options = {f"{briefing['nome_projeto']} ({briefing['tipo']}) - {briefing['data_criacao'].strftime('%d/%m/%Y')}": briefing for briefing in briefings_disponiveis}
                briefing_selecionado = st.selectbox("Escolha um briefing:", list(briefing_options.keys()))
                
                if briefing_selecionado:
                    briefing_data = briefing_options[briefing_selecionado]
                    st.info(f"Briefing selecionado: {briefing_data['nome_projeto']}")
            else:
                st.info("Nenhum briefing encontrado no banco de dados.")
        else:
            st.warning("Conexão com MongoDB não disponível")
        
        # Opção 3: Inserir briefing manualmente
        st.write("✍️ Briefing Manual:")
        briefing_manual = st.text_area("Ou cole o briefing completo aqui:", height=150,
                                      placeholder="""Exemplo:
Título: Campanha de Lançamento
Objetivo: Divulgar novo produto
Público-alvo: Empresários...
Pontos-chave: [lista os principais pontos]""")
        
        # Transcrição de áudio/vídeo
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
        
        tipo_conteudo = st.selectbox("Tipo de Conteúdo:", 
                                   ["Post Social", "Artigo Blog", "Email Marketing", 
                                    "Landing Page", "Script Vídeo", "Relatório Técnico",
                                    "Press Release", "Newsletter", "Case Study"])
        
        tom_voz = st.selectbox("Tom de Voz:", 
                              ["Formal", "Informal", "Persuasivo", "Educativo", 
                               "Inspirador", "Técnico", "Jornalístico"], key = 'qq')
        
        palavras_chave = st.text_input("Palavras-chave (opcional):",
                                      placeholder="separadas por vírgula")
        
        numero_palavras = st.slider("Número de Palavras:", 100, 3000, 800)
        
        # Configurações avançadas
        with st.expander("🔧 Configurações Avançadas"):
            usar_contexto_agente = st.checkbox("Usar contexto do agente selecionado", 
                                             value=bool(st.session_state.agente_selecionado))
            
            nivel_detalhe = st.select_slider("Nível de Detalhe:", 
                                           ["Resumido", "Balanceado", "Detalhado", "Completo"])
            
            incluir_cta = st.checkbox("Incluir Call-to-Action", value=True)
            
            formato_saida = st.selectbox("Formato de Saída:", 
                                       ["Texto Simples", "Markdown", "HTML Básico"])

    # Área de instruções específicas
    st.subheader("🎯 Instruções Específicas")
    instrucoes_especificas = st.text_area(
        "Diretrizes adicionais para geração:",
        placeholder="""Exemplos:
- Focar nos benefícios para o usuário final
- Incluir estatísticas quando possível
- Manter linguagem acessível
- Evitar jargões técnicos excessivos
- Seguir estrutura: problema → solução → benefícios""",
        height=100
    )

    # Botão para gerar conteúdo
    if st.button("🚀 Gerar Conteúdo com Todos os Insumos", type="primary", use_container_width=True):
        # Verificar se há pelo menos uma fonte de conteúdo
        tem_conteudo = (arquivos_upload or 
                       briefing_manual or 
                       ('briefing_data' in locals() and briefing_data) or
                       arquivos_midia)
        
        if not tem_conteudo:
            st.error("❌ Por favor, forneça pelo menos uma fonte de conteúdo (arquivos, briefing ou mídia)")
        else:
            with st.spinner("Processando todos os insumos e gerando conteúdo..."):
                try:
                    # Construir o contexto combinado de todas as fontes
                    contexto_completo = "## FONTES DE CONTEÚDO COMBINADAS:\n\n"
                    
                    # Adicionar conteúdo dos arquivos uploadados
                    if textos_arquivos:
                        contexto_completo += "### CONTEÚDO DOS ARQUIVOS:\n" + textos_arquivos + "\n\n"
                    
                    # Adicionar briefing do banco ou manual
                    if briefing_manual:
                        contexto_completo += "### BRIEFING MANUAL:\n" + briefing_manual + "\n\n"
                    elif 'briefing_data' in locals() and briefing_data:
                        contexto_completo += "### BRIEFING DO BANCO:\n" + briefing_data['conteudo'] + "\n\n"
                    
                    # Adicionar transcrições
                    if transcricoes_texto:
                        contexto_completo += "### TRANSCRIÇÕES DE MÍDIA:\n" + transcricoes_texto + "\n\n"
                    
                    # Adicionar contexto do agente se selecionado
                    contexto_agente = ""
                    if usar_contexto_agente and st.session_state.agente_selecionado:
                        agente = st.session_state.agente_selecionado
                        contexto_agente = construir_contexto(agente, st.session_state.segmentos_selecionados)
                    
                    # Construir prompt final
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
                    Com base em TODAS as fontes fornecidas acima, gere um conteúdo do tipo {tipo_conteudo} que:
                    
                    1. **Síntese Eficiente:** Combine e sintetize informações de todas as fontes
                    2. **Coerência:** Mantenha consistência com as informações originais
                    3. **Valor Agregado:** Vá além da simples cópia, agregando insights
                    4. **Engajamento:** Crie conteúdo que engaje o público-alvo
                    5. **Clareza:** Comunique ideias complexas de forma acessível
                    
                    **FORMATO DE SAÍDA:** {formato_saida}
                    
                    Gere um conteúdo completo e profissional.
                    """
                    
                    resposta = modelo_texto.generate_content(prompt_final)
                    
                    # Processar saída baseada no formato selecionado
                    conteudo_gerado = resposta.text
                    
                    if formato_saida == "HTML Básico":
                        # Converter markdown para HTML básico
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
                        st.metric("Fontes Utilizadas", 
                                 (1 if arquivos_upload else 0) + 
                                 (1 if briefing_manual or 'briefing_data' in locals() else 0) +
                                 (1 if transcricoes_texto else 0))
                    
                    # Botões de download
                    extensao = ".html" if formato_saida == "HTML Básico" else ".md" if formato_saida == "Markdown" else ".txt"
                    
                    st.download_button(
                        f"💾 Baixar Conteúdo ({formato_saida})",
                        data=conteudo_gerado,
                        file_name=f"conteudo_gerado_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}{extensao}",
                        mime="text/html" if formato_saida == "HTML Básico" else "text/plain"
                    )
                    
                    # Salvar no histórico se MongoDB disponível
                    if mongo_connected_conteudo:
                        try:
                            from bson import ObjectId
                            historico_data = {
                                "tipo_conteudo": tipo_conteudo,
                                "tom_voz": tom_voz,
                                "palavras_chave": palavras_chave,
                                "numero_palavras": numero_palavras,
                                "conteudo_gerado": conteudo_gerado,
                                "fontes_utilizadas": {
                                    "arquivos_upload": [arquivo.name for arquivo in arquivos_upload] if arquivos_upload else [],
                                    "briefing_manual": bool(briefing_manual),
                                    "transcricoes": len(arquivos_midia) if arquivos_midia else 0
                                },
                                "data_criacao": datetime.datetime.now()
                            }
                            db_briefings['historico_geracao'].insert_one(historico_data)
                            st.success("✅ Conteúdo salvo no histórico!")
                        except Exception as e:
                            st.warning(f"Conteúdo gerado, mas não salvo no histórico: {str(e)}")
                    
                except Exception as e:
                    st.error(f"❌ Erro ao gerar conteúdo: {str(e)}")
                    st.info("💡 Dica: Verifique se os arquivos não estão corrompidos e tente novamente.")

    # Seção de histórico rápido
    if mongo_connected_conteudo:
        with st.expander("📚 Histórico de Gerações Recentes"):
            try:
                historico = list(db_briefings['historico_geracao'].find().sort("data_criacao", -1).limit(5))
                if historico:
                    for item in historico:
                        st.write(f"**{item['tipo_conteudo']}** - {item['data_criacao'].strftime('%d/%m/%Y %H:%M')}")
                        st.caption(f"Palavras-chave: {item.get('palavras_chave', 'Nenhuma')} | Tom: {item['tom_voz']}")
                        with st.expander("Ver conteúdo"):
                            st.write(item['conteudo_gerado'][:500] + "..." if len(item['conteudo_gerado']) > 500 else item['conteudo_gerado'])
                else:
                    st.info("Nenhuma geração no histórico")
            except Exception as e:
                st.warning(f"Erro ao carregar histórico: {str(e)}")

# ========== ABA: GERAÇÃO DE CONTEÚDO BLOG AGRÍCOLA ==========
with tab_blog:
    st.title("🌱 Gerador de Blog Posts Agrícolas")
    st.markdown("Crie conteúdos especializados para o agronegócio seguindo a estrutura profissional")

    # Conexão com MongoDB
    try:
        client_mongo = MongoClient("mongodb+srv://gustavoromao3345:RqWFPNOJQfInAW1N@cluster0.5iilj.mongodb.net/auto_doc?retryWrites=true&w=majority&ssl=true&ssl_cert_reqs=CERT_NONE&tlsAllowInvalidCertificates=true")
        db = client_mongo['blog_posts_agricolas']
        collection_posts = db['posts_gerados']
        collection_briefings = db['briefings']
        collection_kbf = db['kbf_produtos']
        mongo_connected_blog = True
    except Exception as e:
        st.error(f"Erro na conexão com MongoDB: {str(e)}")
        mongo_connected_blog = False

    # Funções para o banco de dados
    def salvar_post(titulo, cultura, editoria, mes_publicacao, objetivo_post, url, texto_gerado, palavras_chave, palavras_proibidas, tom_voz, estrutura, palavras_contagem, meta_title, meta_descricao, linha_fina, links_internos=None):
        if mongo_connected_blog:
            documento = {
                "id": str(uuid.uuid4()),
                "titulo": titulo,
                "cultura": cultura,
                "editoria": editoria,
                "mes_publicacao": mes_publicacao,
                "objetivo_post": objetivo_post,
                "url": url,
                "texto_gerado": texto_gerado,
                "palavras_chave": palavras_chave,
                "palavras_proibidas": palavras_proibidas,
                "tom_voz": tom_voz,
                "estrutura": estrutura,
                "palavras_contagem": palavras_contagem,
                "meta_title": meta_title,
                "meta_descricao": meta_descricao,
                "linha_fina": linha_fina,
                "links_internos": links_internos or [],
                "versao": "2.1"  # Atualizado para versão 2.1
            }
            collection_posts.insert_one(documento)
            return True
        return False

    def carregar_kbf_produtos():
        if mongo_connected_blog:
            try:
                kbf_docs = list(collection_kbf.find({}))
                return kbf_docs
            except:
                return []
        return []

    def salvar_briefing(briefing_data):
        if mongo_connected_blog:
            documento = {
                "id": str(uuid.uuid4()),
                "briefing": briefing_data,
            }
            collection_briefings.insert_one(documento)
            return True
        return False

    def carregar_posts_anteriores():
        if mongo_connected_blog:
            try:
                posts = list(collection_posts.find({}).sort("data_criacao", -1).limit(10))
                return posts
            except:
                return []
        return []

    # ASSINATURA PADRÃO E BOX INICIAL
    ASSINATURA_PADRAO = """
---

**Sobre o Mais Agro**
O Mais Agro é uma plataforma de conteúdo especializado em agronegócio, trazendo informações técnicas, análises de mercado e soluções inovadoras para produtores rurais e profissionais do setor.

📞 **Fale conosco:** [contato@maisagro.com.br](mailto:contato@maisagro.com.br)
🌐 **Site:** [www.maisagro.com.br](https://www.maisagro.com.br)
📱 **Redes sociais:** @maisagrooficial

*Este conteúdo foi desenvolvido pela equipe técnica do Mais Agro para apoiar o produtor rural com informações confiáveis e atualizadas.*
"""

    BOX_INICIAL = """
> 📌 **Destaque do Artigo**
> 
> *[Este box deve conter um resumo executivo de 2-3 linhas com os pontos mais importantes do artigo, destacando o problema principal e a solução abordada. Exemplo: "Neste artigo você vai entender como o manejo integrado de nematoides pode aumentar em até 30% a produtividade da soja, com estratégias práticas para implementação imediata."]*
"""

    # Regras base do sistema - ATUALIZADAS COM CORREÇÕES
    regras_base = '''
    **REGRAS DE REPLICAÇÃO - ESTRUTURA PROFISSIONAL:**

    **1. ESTRUTURA DO DOCUMENTO:**
    - Título principal impactante e com chamada para ação (máx 65 caracteres)
    - BOX INICIAL com resumo executivo (usar template fornecido)
    - Linha fina resumindo o conteúdo (máx 200 caracteres)
    - Meta-title otimizado para SEO (máx 60 caracteres)
    - Meta-descrição atrativa (máx 155 caracteres)
    - Introdução contextualizando o problema e impacto (EVITAR padrão "cultura X é importante")
    - Seção de Problema: Detalhamento técnico dos desafios
    - Seção de Produto/Solução: Informações específicas sobre o produto e sua aplicação
    - Seção de Benefícios: Vantagens mensuráveis da solução
    - Seção de Implementação Prática: Como aplicar no campo
    - ASSINATURA PADRÃO (usar template fornecido)

    **2. LINGUAGEM E TOM:**
    - {tom_voz}
    - Linguagem {nivel_tecnico} técnica e profissional
    - Uso de terminologia específica do agronegócio
    - Persuasão baseada em benefícios e solução de problemas
    - Evitar repetição de informações entre seções
    - NÃO usar "Conclusão" como subtítulo - finalizar com chamada para ação natural
    - NÃO usar letras maiúsculas em excesso - apenas onde gramaticalmente necessário

    **3. ELEMENTOS TÉCNICOS OBRIGATÓRIOS:**
    - Nomes científicos entre parênteses quando aplicável
    - Citação EXPLÍCITA de fontes confiáveis (Embrapa, universidades, etc.) mencionando o órgão/instituição no corpo do texto
    - Destaque para termos técnicos-chave e nomes de produtos
    - Descrição detalhada de danos e benefícios
    - Dados concretos e informações mensuráveis com referências específicas

    **4. FORMATAÇÃO E ESTRUTURA:**
    - Parágrafos curtos (máximo 4-5 linhas cada)
    - Listas de tópicos com no máximo 5 itens cada
    - Evitar blocos extensos de texto
    - Usar subtítulos para quebrar o conteúdo
    - NÃO usar os termos "Solução Genérica" e "Solução Específica" nos subtítulos

    **5. RESTRIÇÕES E FILTROS:**
    - PALAVRAS PROIBIDAS ABSOLUTAS: {palavras_proibidas_efetivas}
    - NÃO USAR as palavras acima em nenhuma circunstância
    - Evitar viés comercial explícito
    - Manter abordagem {abordagem_problema}
    - Número de palavras: {numero_palavras} (±5%)
    - NÃO INVENTAR SOLUÇÕES ou informações não fornecidas
    - Seguir EXATAMENTE o formato e informações do briefing
    - EVITAR introduções genéricas sobre importância da cultura
    - Focar em problemas específicos e soluções práticas desde o início
    '''

    # CONFIGURAÇÕES DO BLOG (agora dentro da aba)
    st.header("📋 Configurações do Blog Agrícola")
    
    col_config1, col_config2 = st.columns(2)
    
    with col_config1:
        # Modo de entrada - Briefing ou Campos Individuais
        modo_entrada = st.radio("Modo de Entrada:", ["Campos Individuais", "Briefing Completo"])
        
        # Controle de palavras - MAIS RESTRITIVO
        numero_palavras = st.slider("Número de Palavras:", min_value=300, max_value=2500, value=1500, step=100)
        st.info(f"Meta: {numero_palavras} palavras (±5%)")
        
        # Palavras-chave
        st.subheader("🔑 Palavras-chave")
        palavra_chave_principal = st.text_input("Palavra-chave Principal:")
        palavras_chave_secundarias = st.text_area("Palavras-chave Secundárias (separadas por vírgula):")
        
        # Configurações de estilo
        st.subheader("🎨 Configurações de Estilo")
        tom_voz = st.selectbox("Tom de Voz:", ["Jornalístico", "Especialista Técnico", "Educativo", "Persuasivo"], key = 'uu')
        nivel_tecnico = st.selectbox("Nível Técnico:", ["Básico", "Intermediário", "Avançado"])
        abordagem_problema = st.text_area("Aborde o problema de tal forma que:", "seja claro, técnico e focando na solução prática para o produtor")
    
    with col_config2:
        # Restrições - MELHOR CONTROLE DE PALAVRAS PROIBIDAS
        st.subheader("🚫 Restrições")
        palavras_proibidas_input = st.text_area("Palavras Proibidas (separadas por vírgula):", "melhor, número 1, líder, insuperável, invenção, inventado, solução mágica, revolucionário, único, exclusivo")
        
        # Processar palavras proibidas para garantir efetividade
        palavras_proibidas_lista = [palavra.strip().lower() for palavra in palavras_proibidas_input.split(",") if palavra.strip()]
        palavras_proibidas_efetivas = ", ".join(palavras_proibidas_lista)
        
        if palavras_proibidas_lista:
            st.info(f"🔒 {len(palavras_proibidas_lista)} palavra(s) proibida(s) serão filtradas")
        
        # Estrutura do texto - REMOVIDAS SEÇÕES PROBLEMÁTICAS
        st.subheader("📐 Estrutura do Texto")
        estrutura_opcoes = st.multiselect("Seções do Post:", 
                                         ["Introdução", "Problema/Desafio", "Solução/Produto", 
                                          "Benefícios", "Implementação Prática", "Considerações Finais", "Fontes"],
                                         default=["Introdução", "Problema/Desafio", "Solução/Produto", "Benefícios", "Implementação Prática"])
        
        # KBF de Produtos
        st.subheader("📦 KBF de Produtos")
        kbf_produtos = carregar_kbf_produtos()
        if kbf_produtos:
            produtos_disponiveis = [prod['nome'] for prod in kbf_produtos]
            produto_selecionado = st.selectbox("Selecionar Produto do KBF:", ["Nenhum"] + produtos_disponiveis)
            if produto_selecionado != "Nenhum":
                produto_info = next((prod for prod in kbf_produtos if prod['nome'] == produto_selecionado), None)
                if produto_info:
                    st.info(f"**KBF Fixo:** {produto_info.get('caracteristicas', 'Informações do produto')}")
        else:
            st.info("Nenhum KBF cadastrado no banco de dados")

    # Área principal baseada no modo de entrada
    if modo_entrada == "Campos Individuais":
        col1, col2 = st.columns(2)
        
        with col1:
            st.header("📝 Informações Básicas")
            titulo_blog = st.text_input("Título do Blog:", "Proteja sua soja de nematoides e pragas de solo")
            cultura = st.text_input("Cultura:", "Soja")
            editoria = st.text_input("Editoria:", "Manejo e Proteção")
            mes_publicacao = st.text_input("Mês de Publicação:", "08/2025")
            objetivo_post = st.text_area("Objetivo do Post:", "Explicar a importância do manejo de nematoides e apresentar soluções via tratamento de sementes")
            url = st.text_input("URL:", "/manejo-e-protecao/proteja-sua-soja-de-nematoides")
            
            st.header("🔧 Conteúdo Técnico")
            problema_principal = st.text_area("Problema Principal/Contexto:", "Solos compactados e com palhada de milho têm favorecido a explosão populacional de nematoides")
            pragas_alvo = st.text_area("Pragas/Alvo Principal:", "Nematoide das galhas (Meloidogyne incognita), Nematoide de cisto (Heterodera glycines)")
            danos_causados = st.text_area("Danos Causados:", "Formação de galhas nas raízes que impedem a absorção de água e nutrientes")
        
        with col2:
            st.header("🏭 Informações da Empresa")
            nome_empresa = st.text_input("Nome da Empresa/Marca:")
            nome_central = st.text_input("Nome da Central de Conteúdos:")
            
            st.header("💡 Soluções e Produtos")
            nome_produto = st.text_input("Nome do Produto:")
            principio_ativo = st.text_input("Princípio Ativo/Diferencial:")
            beneficios_produto = st.text_area("Benefícios do Produto:")
            espectro_acao = st.text_area("Espectro de Ação:")
            modo_acao = st.text_area("Modo de Ação:")
            aplicacao_pratica = st.text_area("Aplicação Prática:")
            
            st.header("🎯 Diretrizes Específicas")
            diretrizes_usuario = st.text_area("Diretrizes Adicionais:", 
                                            "NÃO INVENTE SOLUÇÕES. Use apenas informações fornecidas. Incluir dicas práticas para implementação no campo. Manter linguagem acessível mas técnica. EVITAR introduções genéricas sobre importância da cultura.")
            fontes_pesquisa = st.text_area("Fontes para Pesquisa/Referência (cite órgãos específicos):", 
                                         "Embrapa Soja, Universidade de São Paulo - ESALQ, Instituto Biológico de São Paulo, Artigos técnicos sobre nematoides")
            
            # Upload de MÚLTIPLOS arquivos estratégicos
            arquivos_estrategicos = st.file_uploader("📎 Upload de Múltiplos Arquivos Estratégicos", 
                                                   type=['txt', 'pdf', 'docx', 'mp3', 'wav', 'mp4', 'mov'], 
                                                   accept_multiple_files=True)
            if arquivos_estrategicos:
                st.success(f"{len(arquivos_estrategicos)} arquivo(s) carregado(s) com sucesso!")
    
    else:  # Modo Briefing
        st.header("📄 Briefing Completo")
        
        st.warning("""
        **ATENÇÃO:** Para conteúdos técnicos complexos (especialmente Syngenta), 
        recomenda-se usar o modo "Campos Individuais" para melhor controle da qualidade.
        """)
        
        briefing_texto = st.text_area("Cole aqui o briefing completo:", height=300,
                                     placeholder="""EXEMPLO DE BRIEFING:
Título: Controle Eficiente de Nematoides na Soja
Cultura: Soja
Problema: Aumento da população de nematoides em solos com palhada de milho
Objetivo: Educar produtores sobre manejo integrado
Produto: NemaControl
Público-alvo: Produtores de soja técnica
Tom: Técnico-jornalístico
Palavras-chave: nematoide, soja, tratamento sementes, manejo integrado

IMPORTANTE: NÃO INVENTE SOLUÇÕES. Use apenas informações fornecidas aqui.""")
        
        if briefing_texto:
            if st.button("Processar Briefing"):
                salvar_briefing(briefing_texto)
                st.success("Briefing salvo no banco de dados!")

    # NOVO CAMPO: LINKS INTERNOS
    st.header("🔗 Links Internos")
    st.info("Adicione links internos que serão automaticamente inseridos no corpo do texto como âncoras")
    
    links_internos = []
    num_links = st.number_input("Número de links internos a adicionar:", min_value=0, max_value=10, value=0)
    
    for i in range(num_links):
        col_link1, col_link2 = st.columns([3, 1])
        with col_link1:
            texto_ancora = st.text_input(f"Texto âncora {i+1}:", placeholder="Ex: manejo integrado de pragas")
            url_link = st.text_input(f"URL do link {i+1}:", placeholder="Ex: /blog/manejo-integrado-pragas")
        with col_link2:
            posicao = st.selectbox(f"Posição {i+1}:", ["Automática", "Introdução", "Problema", "Solução", "Benefícios", "Implementação"])
        
        if texto_ancora and url_link:
            links_internos.append({
                "texto_ancora": texto_ancora,
                "url": url_link,
                "posicao": posicao
            })
    
    if links_internos:
        st.success(f"✅ {len(links_internos)} link(s) interno(s) configurado(s)")

    # Configurações avançadas
    with st.expander("⚙️ Configurações Avançadas"):
        col_av1, col_av2 = st.columns(2)
        
        with col_av1:
            st.subheader("Opcionais")
            usar_pesquisa_web = st.checkbox("🔍 Habilitar Pesquisa Web", value=False)
            gerar_blocos_dinamicos = st.checkbox("🔄 Gerar Blocos Dinamicamente", value=True)
            incluir_fontes = st.checkbox("📚 Incluir Referências de Fontes", value=True)
            incluir_assinatura = st.checkbox("✍️ Incluir Assinatura Padrão", value=True, help="Assinatura padrão do Mais Agro será incluída automaticamente")
            incluir_box_inicial = st.checkbox("📌 Incluir Box Inicial", value=True, help="Box de destaque no início do artigo")
            
        with col_av2:
            st.subheader("Controles de Qualidade")
            evitar_repeticao = st.slider("Nível de Evitar Repetição:", 1, 10, 8)
            profundidade_conteudo = st.selectbox("Profundidade do Conteúdo:", ["Superficial", "Moderado", "Detalhado", "Especializado"])
            
            # Configurações de formatação
            st.subheader("📐 Formatação")
            max_paragrafos = st.slider("Máximo de linhas por parágrafo:", 3, 8, 5)
            max_lista_itens = st.slider("Máximo de itens por lista:", 3, 8, 5)
            
            # MÚLTIPLOS arquivos para transcrição
            st.subheader("🎤 Transcrição de Mídia")
            arquivos_midia = st.file_uploader("Áudios/Vídeos para Transcrição (múltiplos)", 
                                            type=['mp3', 'wav', 'mp4', 'mov'], 
                                            accept_multiple_files=True)
            
            if arquivos_midia:
                st.info(f"{len(arquivos_midia)} arquivo(s) de mídia carregado(s)")
                if st.button("🎬 Transcrever Mídia"):
                    with st.spinner("Transcrevendo arquivos de mídia..."):
                        for arquivo in arquivos_midia:
                            tipo = "audio" if arquivo.type.startswith('audio') else "video"
                            transcricao = transcrever_audio_video(arquivo, tipo)
                            st.write(f"**Transcrição de {arquivo.name}:**")
                            st.write(transcricao)

    # Metadados para SEO
    st.header("🔍 Metadados para SEO")
    col_meta1, col_meta2 = st.columns(2)
    
    with col_meta1:
        meta_title = st.text_input("Meta Title (máx 60 caracteres):", 
                                 max_chars=60,
                                 help="Título para SEO - aparecerá nos resultados de busca")
        st.info(f"Caracteres: {len(meta_title)}/60")
        
        linha_fina = st.text_area("Linha Fina (máx 200 caracteres):",
                                max_chars=200,
                                help="Resumo executivo que aparece abaixo do título")
        st.info(f"Caracteres: {len(linha_fina)}/200")
    
    with col_meta2:
        meta_descricao = st.text_area("Meta Descrição (máx 155 caracteres):",
                                    max_chars=155,
                                    help="Descrição que aparece nos resultados de busca")
        st.info(f"Caracteres: {len(meta_descricao)}/155")

    # Área de geração
    st.header("🔄 Geração do Conteúdo")
    
    if st.button("🚀 Gerar Blog Post", type="primary", use_container_width=True):
        with st.spinner("Gerando conteúdo... Isso pode levar alguns minutos"):
            try:
                # Processar transcrições se houver arquivos
                transcricoes_texto = ""
                if 'arquivos_midia' in locals() and arquivos_midia:
                    for arquivo in arquivos_midia:
                        tipo = "audio" if arquivo.type.startswith('audio') else "video"
                        transcricao = transcrever_audio_video(arquivo, tipo)
                        transcricoes_texto += f"\n\n--- TRANSCRIÇÃO DE {arquivo.name} ---\n{transcricao}"
                    st.info(f"Processadas {len(arquivos_midia)} transcrição(ões)")
                
                # Construir prompt personalizado - CORRIGIDO
                regras_personalizadas = regras_base.format(
                    tom_voz=tom_voz,
                    nivel_tecnico=nivel_tecnico,
                    palavras_proibidas_efetivas=palavras_proibidas_efetivas,
                    abordagem_problema=abordagem_problema,
                    numero_palavras=numero_palavras
                )
                
                # Adicionar instruções sobre links internos se houver
                instrucoes_links = ""
                if links_internos:
                    instrucoes_links = "\n\n**INSTRUÇÕES PARA LINKS INTERNOS:**\n"
                    instrucoes_links += "INSIRA os seguintes links internos DENTRO do texto, como âncoras naturais:\n"
                    for link in links_internos:
                        instrucoes_links += f"- [{link['texto_ancora']}]({link['url']}) - Posição: {link['posicao']}\n"
                    instrucoes_links += "\n**IMPORTANTE:** Insira os links de forma natural no contexto, sem forçar. Use como referência para criar âncoras relevantes."
                
                # Instruções específicas para BOX INICIAL e ASSINATURA
                instrucoes_estrutura = ""
                if incluir_box_inicial:
                    instrucoes_estrutura += f"\n\n**BOX INICIAL OBRIGATÓRIO:**\n{BOX_INICIAL}"
                
                if incluir_assinatura:
                    instrucoes_estrutura += f"\n\n**ASSINATURA PADRÃO OBRIGATÓRIA:**\n{ASSINATURA_PADRAO}"

                prompt_final = f"""
                **INSTRUÇÕES PARA CRIAÇÃO DE BLOG POST AGRÍCOLA:**

                {regras_personalizadas}
                
                **INFORMAÇÕES ESPECÍFICAS:**
                - Título: {titulo_blog if 'titulo_blog' in locals() else 'A definir'}
                - Cultura: {cultura if 'cultura' in locals() else 'A definir'}
                - Palavra-chave Principal: {palavra_chave_principal}
                - Palavras-chave Secundárias: {palavras_chave_secundarias}
                
                {instrucoes_links}
                {instrucoes_estrutura}

                **METADADOS:**
                - Meta Title: {meta_title}
                - Meta Description: {meta_descricao}
                - Linha Fina: {linha_fina}
                
                **CONFIGURAÇÕES DE FORMATAÇÃO:**
                - Parágrafos máximos: {max_paragrafos} linhas
                - Listas máximas: {max_lista_itens} itens
                - Estrutura: {', '.join(estrutura_opcoes)}
                - Profundidade: {profundidade_conteudo}
                - Evitar repetição: Nível {evitar_repeticao}/10
                
                **DIRETRIZES CRÍTICAS:**
                - NÃO INVENTE SOLUÇÕES OU INFORMAÇÕES
                - Use APENAS dados fornecidos no briefing
                - Cite fontes específicas no corpo do texto
                - Mantenha parágrafos e listas CURTOS
                - INSIRA OS LINKS INTERNOS de forma natural no texto
                - EVITE letras maiúsculas em excesso
                - NÃO USE "Conclusão" como subtítulo
                - EVITE introduções genéricas sobre importância da cultura
                - FOCAR em problemas específicos desde o início
                - FILTRAR as palavras proibidas: {palavras_proibidas_efetivas}
                
                **CONTEÚDO DE TRANSCRIÇÕES:**
                {transcricoes_texto if transcricoes_texto else 'Nenhuma transcrição fornecida'}
                
                **INFORMAÇÕES SOBRE PRODUTO:**
                - Nome do Produto: {nome_produto if 'nome_produto' in locals() else 'Não especificado'}
                - Princípio Ativo: {principio_ativo if 'principio_ativo' in locals() else 'Não especificado'}
                - Benefícios: {beneficios_produto if 'beneficios_produto' in locals() else 'Não especificado'}
                - Modo de Ação: {modo_acao if 'modo_acao' in locals() else 'Não especificado'}
                - Aplicação Prática: {aplicacao_pratica if 'aplicacao_pratica' in locals() else 'Não especificado'}
                
                **DIRETRIZES ADICIONAIS:** {diretrizes_usuario if 'diretrizes_usuario' in locals() else 'Nenhuma'}
                
                Gere um conteúdo {profundidade_conteudo.lower()} com EXATAMENTE {numero_palavras} palavras (±5%).
                """
                
                response = modelo_texto.generate_content(prompt_final)
                
                texto_gerado = response.text
                
                # VERIFICAÇÃO E APLICAÇÃO DE FILTROS
                # 1. Verificar palavras proibidas
                palavras_proibidas_encontradas = []
                for palavra in palavras_proibidas_lista:
                    if palavra.lower() in texto_gerado.lower():
                        palavras_proibidas_encontradas.append(palavra)
                
                if palavras_proibidas_encontradas:
                    st.warning(f"⚠️ Palavras proibidas encontradas: {', '.join(palavras_proibidas_encontradas)}")
                    # Substituir palavras proibidas
                    for palavra in palavras_proibidas_encontradas:
                        texto_gerado = texto_gerado.replace(palavra, "[FILTRADO]")
                        texto_gerado = texto_gerado.replace(palavra.capitalize(), "[FILTRADO]")
                
                # 2. Verificar contagem de palavras
                palavras_count = len(texto_gerado.split())
                st.info(f"📊 Contagem de palavras geradas: {palavras_count} (meta: {numero_palavras})")
                
                if abs(palavras_count - numero_palavras) > numero_palavras * 0.1:
                    st.warning("⚠️ A contagem de palavras está significativamente diferente da meta")
                
                # 3. Verificar estrutura
                if "Conclusão" in texto_gerado:
                    st.warning("⚠️ O texto contém 'Conclusão' como subtítulo - isso deve ser evitado")
                
                # Salvar no MongoDB
                if salvar_post(
                    titulo_blog if 'titulo_blog' in locals() else "Título gerado",
                    cultura if 'cultura' in locals() else "Cultura não especificada",
                    editoria if 'editoria' in locals() else "Editoria geral",
                    mes_publicacao if 'mes_publicacao' in locals() else datetime.datetime.now().strftime("%m/%Y"),
                    objetivo_post if 'objetivo_post' in locals() else "Objetivo não especificado",
                    url if 'url' in locals() else "/",
                    texto_gerado,
                    f"{palavra_chave_principal}, {palavras_chave_secundarias}",
                    palavras_proibidas_efetivas,
                    tom_voz,
                    ', '.join(estrutura_opcoes),
                    palavras_count,
                    meta_title,
                    meta_descricao,
                    linha_fina,
                    links_internos
                ):
                    st.success("✅ Post gerado e salvo no banco de dados!")
                
                st.subheader("📝 Conteúdo Gerado")
                st.markdown(texto_gerado)
                
                st.download_button(
                    "💾 Baixar Post",
                    data=texto_gerado,
                    file_name=f"blog_post_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                    mime="text/plain"
                )
                
            except Exception as e:
                st.error(f"Erro na geração: {str(e)}")

    # Banco de textos gerados
    st.header("📚 Banco de Textos Gerados")
    
    posts_anteriores = carregar_posts_anteriores()
    if posts_anteriores:
        for post in posts_anteriores:
            with st.expander(f"{post.get('titulo', 'Sem título')}"):
                st.write(f"**Cultura:** {post.get('cultura', 'N/A')}")
                st.write(f"**Palavras:** {post.get('palavras_contagem', 'N/A')}")
                
                # Mostrar metadados salvos
                if post.get('meta_title'):
                    st.write(f"**Meta Title:** {post.get('meta_title')}")
                if post.get('meta_descricao'):
                    st.write(f"**Meta Descrição:** {post.get('meta_descricao')}")
                
                # Mostrar palavras proibidas filtradas
                if post.get('palavras_proibidas'):
                    st.write(f"**Palavras proibidas filtradas:** {post.get('palavras_proibidas')}")
                
                # Mostrar links internos se existirem
                if post.get('links_internos'):
                    st.write("**Links Internos:**")
                    for link in post['links_internos']:
                        st.write(f"- [{link.get('texto_ancora', 'N/A')}]({link.get('url', '#')})")
                
                st.text_area("Conteúdo:", value=post.get('texto_gerado', ''), height=200, key=post['id'])
                
                col_uso1, col_uso2 = st.columns(2)
                with col_uso1:
                    if st.button("Reutilizar", key=f"reuse_{post['id']}"):
                        st.session_state.texto_gerado = post.get('texto_gerado', '')
                        st.success("Conteúdo carregado para reutilização!")
                with col_uso2:
                    st.download_button(
                        label="📥 Download",
                        data=post.get('texto_gerado', ''),
                        file_name=f"blog_post_{post.get('titulo', 'post').lower().replace(' ', '_')}.txt",
                        mime="text/plain",
                        key=f"dl_btn_{post['id']}"
                    )
    else:
        st.info("Nenhum post encontrado no banco de dados.")

# ========== ABA: REVISÃO ORTOGRÁFICA ==========
with tab_revisao_ortografica:
    st.header("📝 Revisão Ortográfica")
    
    texto_para_revisao = st.text_area("Cole o texto que deseja revisar:", height=300)
    
    if st.button("🔍 Realizar Revisão Ortográfica", type="primary"):
        if texto_para_revisao:
            with st.spinner("Revisando texto..."):
                try:
                    # Usar contexto do agente selecionado se disponível
                    if st.session_state.agente_selecionado:
                        agente = st.session_state.agente_selecionado
                        contexto = construir_contexto(agente, st.session_state.segmentos_selecionados)
                        prompt = f"""
                        
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
        else:
            st.warning("Por favor, cole um texto para revisão.")

# ========== ABA: REVISÃO TÉCNICA (VERSÃO COMPLETA COM RELATÓRIO DE MUDANÇAS) ==========
with tab_revisao_tecnica:
    st.header("🔧 Revisão Técnica com RAGs Especializados")
    st.markdown("**Análise em camadas: taxonomia, epidemiologia, produtos + reescrita final com relatório detalhado**")
    
    # Configurações da revisão
    col_config1, col_config2, col_config3 = st.columns([2, 1, 1])
    
    with col_config1:
        texto_tecnico = st.text_area(
            "Cole o conteúdo técnico para revisão:", 
            height=300,
            placeholder="Cole aqui o conteúdo técnico agrícola que precisa ser revisado..."
        )
        
        # Tipo de conteúdo específico
        tipo_conteudo = st.selectbox(
            "Tipo de Conteúdo:",
            ["Artigo Técnico", "Material Comercial", "Blog Post", "Manual Técnico", "Comunicado Técnico"],
            help="Define o rigor da revisão"
        )
    
    with col_config2:
        st.subheader("🔍 RAGs Especializados")
        
        rag_taxonomia = st.checkbox("RAG Taxonomia", value=True, 
                                  help="Busca específica por classificação de patógenos")
        rag_epidemiologia = st.checkbox("RAG Epidemiologia", value=True,
                                      help="Busca específica por condições ambientais")
        rag_produtos = st.checkbox("RAG Produtos", value=True,
                                 help="Busca específica por informações de produtos")
        rag_geral = st.checkbox("RAG Geral", value=True,
                              help="Busca geral por similaridade semântica")
    
    with col_config3:
        st.subheader("⚙️ Configurações")
        
        nivel_rigor = st.select_slider(
            "Nível de Rigor:",
            ["Leve", "Moderado", "Rigoroso", "Especialista"]
        )
        
        limite_documentos = st.number_input("Docs por RAG", min_value=3, max_value=20, value=12,
                                          help="Número de documentos resgatados por RAG especializado")
        
        usar_contexto_agente = st.checkbox("Usar contexto do agente", 
                                         value=bool(st.session_state.agente_selecionado))
        
        # NOVA OPÇÃO: Incluir relatório detalhado
        incluir_relatorio = st.checkbox("📋 Incluir relatório de mudanças", value=True,
                                      help="Gera um relatório detalhado mostrando todas as alterações")

    # Funções para RAGs especializados
    def realizar_rag_taxonomia(texto: str, limite: int = 12) -> List[Dict]:
        """RAG especializado em taxonomia e classificação de patógenos"""
        perguntas_especificas = [
            "classificação taxonômica",
            "fungo ou oomiceto",
            "nome científico patógeno", 
            "reino filo classe ordem",
            "agente causal doença",
            "Peronospora Phakopsora Corynespora",
            "oomiceto vs fungo diferença",
            "taxonomia fitopatologia"
        ]
        
        documentos_combinados = []
        for pergunta in perguntas_especificas:
            query = f"{texto[:200]} {pergunta}"
            embedding = get_embedding(query)
            documentos = astra_client.vector_search(ASTRA_DB_COLLECTION, embedding, limit=limite//len(perguntas_especificas))
            documentos_combinados.extend(documentos)
        
        # Remover duplicados
        documentos_unicos = []
        ids_vistos = set()
        for doc in documentos_combinados:
            doc_id = str(doc.get('_id', ''))
            if doc_id not in ids_vistos:
                documentos_unicos.append(doc)
                ids_vistos.add(doc_id)
        
        return documentos_unicos[:limite]

    def realizar_rag_epidemiologia(texto: str, limite: int = 12) -> List[Dict]:
        """RAG especializado em condições epidemiológicas"""
        perguntas_especificas = [
            "condições ambientais doença",
            "temperatura umidade molhamento foliar",
            "condições ideais infecção",
            "epidemiologia doença plantas",
            "período molhamento temperatura ótima",
            "umidade relativa infecção",
            "condições climáticas favoráveis",
            "fatores epidemiológicos"
        ]
        
        documentos_combinados = []
        for pergunta in perguntas_especificas:
            query = f"{texto[:200]} {pergunta}"
            embedding = get_embedding(query)
            documentos = astra_client.vector_search(ASTRA_DB_COLLECTION, embedding, limit=limite//len(perguntas_especificas))
            documentos_combinados.extend(documentos)
        
        # Remover duplicados
        documentos_unicos = []
        ids_vistos = set()
        for doc in documentos_combinados:
            doc_id = str(doc.get('_id', ''))
            if doc_id not in ids_vistos:
                documentos_unicos.append(doc)
                ids_vistos.add(doc_id)
        
        return documentos_unicos[:limite]

    def realizar_rag_produtos(texto: str, limite: int = 12) -> List[Dict]:
        """RAG especializado em informações de produtos"""
        perguntas_especificas = [
            "modo de ação produto",
            "aplicação dose recomendada",
            "eficácia controle doença",
            "características técnicas produto",
            "benefícios produto agrícola",
            "tecnologia aplicação",
            "resultados eficácia",
            "recomendações uso produto"
        ]
        
        documentos_combinados = []
        for pergunta in perguntas_especificas:
            query = f"{texto[:200]} {pergunta}"
            embedding = get_embedding(query)
            documentos = astra_client.vector_search(ASTRA_DB_COLLECTION, embedding, limit=limite//len(perguntas_especificas))
            documentos_combinados.extend(documentos)
        
        # Remover duplicados
        documentos_unicos = []
        ids_vistos = set()
        for doc in documentos_combinados:
            doc_id = str(doc.get('_id', ''))
            if doc_id not in ids_vistos:
                documentos_unicos.append(doc)
                ids_vistos.add(doc_id)
        
        return documentos_unicos[:limite]

    def realizar_rag_geral(texto: str, limite: int = 12) -> List[Dict]:
        """RAG geral por similaridade semântica"""
        embedding = get_embedding(texto[:800])
        documentos = astra_client.vector_search(ASTRA_DB_COLLECTION, embedding, limit=limite)
        return documentos

    def processar_rags_especializados(texto: str, rags_ativos: dict, limite: int = 12) -> dict:
        """Executa todos os RAGs especializados e retorna resultados consolidados"""
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

    # NOVA FUNÇÃO: Reescrita com relatório detalhado de mudanças
    def reescrever_com_relatorio_mudancas(texto_original: str, resultados_rags: dict, contexto_agente: str = "") -> tuple:
        """Reescreve o conteúdo e gera um relatório detalhado das mudanças"""
        
        # Construir contexto consolidado dos RAGs
        contexto_rags = "## DOCUMENTOS TÉCNICOS DE REFERÊNCIA:\n\n"
        
        for categoria, documentos in resultados_rags.items():
            if documentos:
                contexto_rags += f"### {categoria.upper()} ({len(documentos)} documentos):\n"
                for i, doc in enumerate(documentos, 1):
                    doc_content = str(doc)
                    doc_limpo = doc_content.replace('{', '').replace('}', '').replace("'", "").replace('"', '')
                    if len(doc_limpo) > 300:
                        doc_limpo = doc_limpo[:300] + "..."
                    contexto_rags += f"- {doc_limpo}\n"
                contexto_rags += "\n"

        # Prompt para reescrita COM relatório
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

        1. **PRECISÃO TAXONÔMICA:**
           - Corrigir "fungo" para "oomiceto" quando aplicável
           - Validar nomes científicos e classificação
           - Ajustar descrições de ciclo de vida

        2. **ESPECIFICIDADE EPIDEMIOLÓGICA:**
           - Substituir termos vagos por faixas específicas
           - Especificar temperaturas exatas
           - Definir períodos de molhamento foliar
           - Vincular condições ao fechamento do dossel

        3. **DESCRIÇÃO PRECISA DE SINTOMAS:**
           - Corrigir descrições imprecisas
           - Especificar localização nas plantas
           - Detalhar evolução dos sintomas
           - Ajustar terminologia técnica

        4. **MANEJO E TIMING:**
           - Alinhar mensagens sobre timing de aplicação
           - Esclarecer momentos diferentes
           - Especificar rotação de MoA

        5. **INFORMAÇÕES DE PRODUTOS:**
           - Corrigir claims imprecisos
           - Especificar "conforme bula" quando necessário
           - Validar números de eficácia
           - Ajustar claims técnicos com precisão

        **REGRAS ADICIONAIS:**
        - Mantenha a estrutura e formatação do original
        - Preserve títulos, subtítulos e marcações
        - Apenas corrija o conteúdo técnico, não reinvente a estrutura
        - Se não houver informações nos RAGs para corrigir algo específico, mantenha o original
        - Para CADA mudança, forneça justificativa técnica específica

        **RETORNE EXATAMENTE no formato especificado acima.**
        """

        try:
            resposta = modelo_texto.generate_content(prompt_reescrita)
            texto_completo = resposta.text
            
            # Separar o texto reescrito do relatório
            if "### 📝 TEXTO REESCRITO" in texto_completo and "### 🔍 RELATÓRIO DETALHADO DE MUDANÇAS" in texto_completo:
                partes = texto_completo.split("### 🔍 RELATÓRIO DETALHADO DE MUDANÇAS")
                texto_reescrito = partes[0].replace("### 📝 TEXTO REESCRITO", "").strip()
                relatorio_mudancas = "### 🔍 RELATÓRIO DETALHADO DE MUDANÇAS" + partes[1]
            else:
                # Fallback se o formato não for seguido
                texto_reescrito = texto_completo
                relatorio_mudancas = "### ❌ Relatório não gerado automaticamente\nO modelo não seguiu o formato solicitado para o relatório."
            
            return texto_reescrito, relatorio_mudancas
            
        except Exception as e:
            st.error(f"Erro na reescrita: {str(e)}")
            return texto_original, f"### ❌ Erro na geração do relatório\n{str(e)}"

    def reescrever_sem_relatorio(texto_original: str, resultados_rags: dict, contexto_agente: str = "") -> str:
        """Reescreve o conteúdo sem gerar relatório (para opção rápida)"""
        
        contexto_rags = "## DOCUMENTOS TÉCNICOS DE REFERÊNCIA:\n\n"
        
        for categoria, documentos in resultados_rags.items():
            if documentos:
                contexto_rags += f"### {categoria.upper()} ({len(documentos)} documentos):\n"
                for i, doc in enumerate(documentos, 1):
                    doc_content = str(doc)
                    doc_limpo = doc_content.replace('{', '').replace('}', '').replace("'", "").replace('"', '')
                    if len(doc_limpo) > 300:
                        doc_limpo = doc_limpo[:300] + "..."
                    contexto_rags += f"- {doc_limpo}\n"
                contexto_rags += "\n"

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

    # Botão de revisão técnica com RAGs especializados
    if st.button("🔬 Realizar Revisão com RAGs Especializados", type="primary"):
        if texto_tecnico:
            # Configurar RAGs ativos
            rags_ativos = {
                'taxonomia': rag_taxonomia,
                'epidemiologia': rag_epidemiologia, 
                'produtos': rag_produtos,
                'geral': rag_geral
            }
            
            # Construir contexto do agente se solicitado
            contexto_agente = ""
            if usar_contexto_agente and st.session_state.agente_selecionado:
                agente = st.session_state.agente_selecionado
                contexto_agente = construir_contexto(agente, st.session_state.segmentos_selecionados)
            
            with st.spinner("🚀 Executando pipeline de RAGs especializados..."):
                try:
                    # FASE 1: Executar RAGs especializados
                    st.subheader("📡 Fase 1: Busca com RAGs Especializados")
                    
                    resultados_rags = processar_rags_especializados(texto_tecnico, rags_ativos, limite_documentos)
                    
                    # Mostrar estatísticas dos RAGs
                    col_rag1, col_rag2, col_rag3, col_rag4 = st.columns(4)
                    with col_rag1:
                        st.metric("RAG Taxonomia", 
                                 len(resultados_rags.get('taxonomia', [])),
                                 help="Documentos sobre classificação de patógenos")
                    with col_rag2:
                        st.metric("RAG Epidemiologia", 
                                 len(resultados_rags.get('epidemiologia', [])),
                                 help="Documentos sobre condições ambientais")
                    with col_rag3:
                        st.metric("RAG Produtos", 
                                 len(resultados_rags.get('produtos', [])),
                                 help="Documentos sobre produtos e eficácia")
                    with col_rag4:
                        st.metric("RAG Geral", 
                                 len(resultados_rags.get('geral', [])),
                                 help="Documentos por similaridade semântica")
                    
                    # Mostrar relatório detalhado dos RAGs
                    with st.expander("📊 Detalhes dos RAGs Executados", expanded=False):
                        relatorio_rags = "## 📊 RELATÓRIO DOS RAGs ESPECIALIZADOS\n\n"
                        for categoria, documentos in resultados_rags.items():
                            relatorio_rags += f"### {categoria.upper()}\n"
                            relatorio_rags += f"Documentos encontrados: {len(documentos)}\n\n"
                            
                            for i, doc in enumerate(documentos, 1):
                                doc_content = str(doc)
                                doc_limpo = doc_content.replace('{', '').replace('}', '').replace("'", "").replace('"', '')
                                relatorio_rags += f"**Documento {i}:** {doc_limpo[:200]}...\n\n"
                        st.markdown(relatorio_rags)
                    
                    # FASE 2: Reescrita com LLM
                    st.subheader("✍️ Fase 2: Reescrita com Base nos RAGs")
                    
                    with st.spinner("Reescrevendo conteúdo e gerando relatório de mudanças..."):
                        # Escolher qual função de reescrita usar baseado na configuração
                        if incluir_relatorio:
                            texto_reescrito, relatorio_mudancas = reescrever_com_relatorio_mudancas(
                                texto_tecnico, resultados_rags, contexto_agente
                            )
                        else:
                            texto_reescrito = reescrever_sem_relatorio(texto_tecnico, resultados_rags, contexto_agente)
                            relatorio_mudancas = None
                    
                    # FASE 3: Apresentação dos resultados
                    st.subheader("📋 Fase 3: Resultados da Revisão")
                    
                    # Abas para diferentes visualizações - AGORA COM RELATÓRIO
                    if incluir_relatorio:
                        tab_original, tab_reescrito, tab_comparacao, tab_relatorio = st.tabs([
                            "📄 Original", "✨ Reescrito", "⚖️ Comparação", "📋 Relatório de Mudanças"
                        ])
                    else:
                        tab_original, tab_reescrito, tab_comparacao = st.tabs([
                            "📄 Original", "✨ Reescrito", "⚖️ Comparação"
                        ])
                    
                    with tab_original:
                        st.markdown("### Conteúdo Original")
                        st.text_area("Original", texto_tecnico, height=400, key="original_rag", label_visibility="collapsed")
                        
                        # Estatísticas do original
                        palavras_orig = len(texto_tecnico.split())
                        linhas_orig = texto_tecnico.count('\n') + 1
                        col_orig1, col_orig2 = st.columns(2)
                        with col_orig1:
                            st.metric("Palavras Originais", palavras_orig)
                        with col_orig2:
                            st.metric("Linhas Originais", linhas_orig)
                    
                    with tab_reescrito:
                        st.markdown("### Conteúdo Reescrito com RAGs")
                        st.text_area("Reescrito", texto_reescrito, height=400, key="reescrito_rag", label_visibility="collapsed")
                        
                        # Estatísticas do reescrito
                        palavras_reesc = len(texto_reescrito.split())
                        linhas_reesc = texto_reescrito.count('\n') + 1
                        col_reesc1, col_reesc2 = st.columns(2)
                        with col_reesc1:
                            st.metric("Palavras Reescritas", palavras_reesc)
                        with col_reesc2:
                            st.metric("Linhas Reescritas", linhas_reesc)
                        
                        # Botões de ação
                        col_dl1, col_dl2 = st.columns(2)
                        with col_dl1:
                            st.download_button(
                                "💾 Baixar Texto Reescrito",
                                data=texto_reescrito,
                                file_name=f"texto_reescrito_rags_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                                mime="text/plain"
                            )
                        with col_dl2:
                            if st.button("📋 Copiar para Área de Transferência", key="copy_reescrito"):
                                st.code(texto_reescrito, language='markdown')
                                st.success("Texto copiado!")
                    
                    with tab_comparacao:
                        st.markdown("### 🔍 Análise Comparativa")
                        
                        # Métricas de comparação
                        col_comp1, col_comp2, col_comp3, col_comp4 = st.columns(4)
                        with col_comp1:
                            diff_palavras = palavras_reesc - palavras_orig
                            st.metric("Diferença de Palavras", 
                                     f"{'+' if diff_palavras > 0 else ''}{diff_palavras}",
                                     delta=f"{diff_palavras/palavras_orig*100:.1f}%" if palavras_orig > 0 else "0%")
                        
                        with col_comp2:
                            diff_linhas = linhas_reesc - linhas_orig
                            st.metric("Diferença de Linhas",
                                     f"{'+' if diff_linhas > 0 else ''}{diff_linhas}")
                        
                        with col_comp3:
                            documentos_total = sum(len(docs) for docs in resultados_rags.values())
                            st.metric("Documentos Utilizados", documentos_total)
                        
                        with col_comp4:
                            rags_utilizados = sum(1 for docs in resultados_rags.values() if docs)
                            st.metric("RAGs com Resultados", rags_utilizados)
                        
                        # Análise qualitativa
                        st.markdown("#### 📈 Impacto dos RAGs Especializados")
                        
                        analise_categorias = []
                        if resultados_rags.get('taxonomia'):
                            analise_categorias.append("✅ **Taxonomia**: Classificação de patógenos validada")
                        else:
                            analise_categorias.append("⚠️ **Taxonomia**: Nenhum documento específico encontrado")
                        
                        if resultados_rags.get('epidemiologia'):
                            analise_categorias.append("✅ **Epidemiologia**: Condições ambientais validadas") 
                        else:
                            analise_categorias.append("⚠️ **Epidemiologia**: Nenhum documento específico encontrado")
                        
                        if resultados_rags.get('produtos'):
                            analise_categorias.append("✅ **Produtos**: Informações técnicas validadas")
                        else:
                            analise_categorias.append("⚠️ **Produtos**: Nenhum documento específico encontrado")
                        
                        for analise in analise_categorias:
                            st.write(analise)
                    
                    # NOVA ABA: Relatório de Mudanças
                    if incluir_relatorio and relatorio_mudancas:
                        with tab_relatorio:
                            st.markdown("### 📋 Relatório Detalhado de Mudanças")
                            st.markdown(relatorio_mudancas)
                            
                            # Botões de download para o relatório
                            col_rel1, col_rel2 = st.columns(2)
                            with col_rel1:
                                st.download_button(
                                    "💾 Baixar Relatório de Mudanças",
                                    data=relatorio_mudancas,
                                    file_name=f"relatorio_mudancas_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.md",
                                    mime="text/markdown"
                                )
                            with col_rel2:
                                st.download_button(
                                    "📦 Baixar Pacote Completo (ZIP)",
                                    data=texto_reescrito + "\n\n" + "="*50 + "\n\n" + relatorio_mudancas,
                                    file_name=f"revisao_completa_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                                    mime="text/plain"
                                )
                    
                    # Salvar no histórico se MongoDB disponível
                    if mongo_connected_blog:
                        try:
                            revisao_data = {
                                "texto_original": texto_tecnico,
                                "texto_reescrito": texto_reescrito,
                                "relatorio_mudancas": relatorio_mudancas if incluir_relatorio else "Não gerado",
                                "rags_utilizados": rags_ativos,
                                "documentos_encontrados": {k: len(v) for k, v in resultados_rags.items()},
                                "nivel_rigor": nivel_rigor,
                                "incluiu_relatorio": incluir_relatorio,
                                "data_criacao": datetime.datetime.now()
                            }
                            if 'revisoes_rags' not in db.list_collection_names():
                                db.create_collection('revisoes_rags')
                            db['revisoes_rags'].insert_one(revisao_data)
                            st.success("✅ Revisão salva no histórico!")
                        except Exception as e:
                            st.warning(f"Revisão concluída, mas não salva: {str(e)}")
                
                except Exception as e:
                    st.error(f"❌ Erro no pipeline de RAGs: {str(e)}")
        else:
            st.warning("Por favor, cole um conteúdo técnico para revisão.")

    # Ferramentas avançadas para análise
    st.header("🛠️ Ferramentas de Análise Avançada")
    
    col_tools1, col_tools2 = st.columns(2)
    
    with col_tools1:
        with st.expander("🔍 Analisador de Precisão Taxonômica"):
            st.info("Foco específico em classificação de patógenos")
            
            texto_analise = st.text_area("Texto para análise taxonômica:", height=150,
                                       placeholder="Cole trecho com descrição de patógenos...",
                                       key="taxonomia_analyzer")
            
            if st.button("Analisar Taxonomia", key="btn_taxonomia"):
                if texto_analise:
                    with st.spinner("Analisando precisão taxonômica..."):
                        try:
                            # RAG específico para taxonomia
                            resultados_tax = realizar_rag_taxonomia(texto_analise, limite=8)
                            
                            if resultados_tax:
                                st.success(f"📚 Encontrados {len(resultados_tax)} documentos de taxonomia")
                                
                                prompt_analise = f"""
                                Analise a precisão taxonômica deste texto:

                                **TEXTO:** {texto_analise}

                                **DOCUMENTOS DE REFERÊNCIA:**
                                {str(resultados_tax)[:2000]}

                                Foque em:
                                1. Classificação correta (fungo vs oomiceto)
                                2. Nomes científicos precisos
                                3. Descrição de ciclo de vida
                                4. Terminologia técnica adequada

                                Retorne análise concisa.
                                """
                                
                                resposta = modelo_texto.generate_content(prompt_analise)
                                st.markdown(resposta.text)
                            else:
                                st.warning("Nenhum documento de taxonomia encontrado na base")
                                
                        except Exception as e:
                            st.error(f"Erro na análise: {str(e)}")
    
    with col_tools2:
        with st.expander("🌡️ Validador Epidemiológico"):
            st.info("Valida condições ambientais e epidemiológicas")
            
            texto_epidemio = st.text_area("Texto para análise epidemiológica:", height=150,
                                        placeholder="Cole trecho com condições ambientais...",
                                        key="epidemiologia_validator")
            
            if st.button("Validar Epidemiologia", key="btn_epidemiologia"):
                if texto_epidemio:
                    with st.spinner("Validando condições epidemiológicas..."):
                        try:
                            # RAG específico para epidemiologia
                            resultados_epi = realizar_rag_epidemiologia(texto_epidemio, limite=8)
                            
                            if resultados_epi:
                                st.success(f"📚 Encontrados {len(resultados_epi)} documentos epidemiológicos")
                                
                                prompt_validacao = f"""
                                Valide as condições epidemiológicas deste texto:

                                **TEXTO:** {texto_epidemio}

                                **DOCUMENTOS DE REFERÊNCIA:**
                                {str(resultados_epi)[:2000]}

                                Verifique:
                                1. Temperaturas específicas
                                2. Períodos de molhamento foliar
                                3. Umidade relativa ideal
                                4. Condições ambientais precisas

                                Retorne validação concisa.
                                """
                                
                                resposta = modelo_texto.generate_content(prompt_validacao)
                                st.markdown(resposta.text)
                            else:
                                st.warning("Nenhum documento epidemiológico encontrado na base")
                                
                        except Exception as e:
                            st.error(f"Erro na validação: {str(e)}")

    # Seção de histórico e exemplos
    with st.expander("📚 Histórico e Exemplos de Revisão"):
        st.markdown("""
        ### 🎯 Exemplos de Correções com RAGs Especializados
        
        **Cenário 1: Correção Taxonômica**
        ```
        Original: "O fungo Peronospora manshurica causa o míldio"
        Corrigido: "O oomiceto Peronospora manshurica causa o míldio"
        Relatório: 
        - Categoria: Taxonomia
        - Justificativa: Peronospora spp. são oomicetos, não fungos verdadeiros
        - Fonte: Documentos taxonômicos da base
        ```
        
        **Cenário 2: Precisão Epidemiológica**
        ```
        Original: "A ferrugem se desenvolve em condições úmidas"
        Corrigido: "A ferrugem-asiática requer 6-10 horas de molhamento foliar com temperaturas entre 18-26°C"
        Relatório:
        - Categoria: Epidemiologia  
        - Justificativa: Especificação de condições ambientais precisas
        - Fonte: Documentos epidemiológicos sobre Phakopsora pachyrhizi
        ```
        
        **Cenário 3: Informações de Produtos**
        ```
        Original: "Produto X tem ação curativa"
        Corrigido: "Produto X apresenta ação preventiva e pós-infecção inicial (antiesporulante)"
        Relatório:
        - Categoria: Produtos
        - Justificativa: Correção de claims técnicos conforme documentação
        - Fonte: Informações técnicas do produto na base
        ```
        
        ### 📊 Métricas de Qualidade
        - **RAGs Ativos**: Número de especializações utilizadas
        - **Documentos/Especialidade**: Precisão por área técnica  
        - **Taxa de Correção**: Impacto na precisão técnica
        - **Relatório de Mudanças**: Detalhamento completo das alterações
        """)

    # Informações de uso
    with st.expander("ℹ️ Como Usar os RAGs Especializados"):
        st.markdown("""
        ### 🎯 Guia de Uso dos RAGs Especializados
        
        **📋 NOVO: Relatório de Mudanças**
        - Ative a opção "Incluir relatório de mudanças" para obter um detalhamento completo
        - O relatório mostra cada alteração com justificativa técnica
        - Inclui categorização por tipo de correção
        - Fornece métricas de impacto das mudanças
        
        **1. RAG Taxonomia**
        - Foca em classificação científica de patógenos
        - Corrige "fungo" vs "oomiceto" 
        - Valida nomes científicos
        - Ajusta descrições de ciclo de vida
        
        **2. RAG Epidemiologia**
        - Especifica condições ambientais exatas
        - Valida temperaturas e umidades ideais
        - Corrige períodos de molhamento foliar
        - Ajusta gatilhos epidemiológicos
        
        **3. RAG Produtos**
        - Valida modos de ação técnicos
        - Corrige doses e épocas de aplicação
        - Ajusta claims de eficácia
        - Valida recomendações de uso
        
        **4. RAG Geral**
        - Busca por similaridade semântica
        - Complementa informações gerais
        - Mantém coerência contextual
        - Fornece base ampla de conhecimento
        
        ### ⚡ Dicas para Melhores Resultados
        - Ative **todos os RAGs** para cobertura completa
        - Use **limite de 12 documentos** por RAG para balancear qualidade/velocidade
        - Configure nível **"Especialista"** para revisões críticas
        - **Ative o relatório** para auditoria completa das mudanças
        - Utilize as **ferramentas de análise específicas** para validação pontual
        """)


# --- FUNÇÃO ATUALIZADA PARA BUSCA WEB COM PERPLEXITY ---
def buscar_perplexity(prompt: str) -> str:
    """Realiza busca na web usando a biblioteca Perplexity"""
    try:
        if not perplexity_available or perplexity_client is None:
            return "❌ Cliente Perplexity não disponível"
        
        # Enviar prompt para o Perplexity
        response = perplexity_client.chat.completions.create(
            model="sonar",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.0  # Baixa temperatura para respostas mais precisas
        )
        
        # Pegar a resposta
        resposta = response.choices[0].message.content
        
        # Adicionar informações da resposta
        resposta_completa = f"""{resposta}"""
        
        return resposta_completa
        
    except Exception as e:
        return f"❌ Erro na busca Perplexity: {str(e)}"

# --- FUNÇÃO ESPECÍFICA PARA OTIMIZAÇÃO DE CONTEÚDO ---
def buscar_fontes_para_otimizacao(conteudo: str, tipo: str, tom: str) -> str:
    """Busca fontes específicas para otimização de conteúdo agrícola"""
    if not perplexity_available:
        return "Busca web desativada"
    
    prompt = f"""
    
   
    DADOS TÉCNICOS ATUALIZADOS para este conteúdo:
    {conteudo[:800]}
    
    
    """
    
    return buscar_perplexity(prompt)
        

# ========== ABA: OTIMIZAÇÃO DE CONTEÚDO ==========
with tab_otimizacao:
    st.header("🚀 Otimização de Conteúdo")
    
    # Inicializar session state
    if 'conteudo_otimizado' not in st.session_state:
        st.session_state.conteudo_otimizado = None
    if 'ultima_otimizacao' not in st.session_state:
        st.session_state.ultima_otimizacao = None
    if 'ajustes_realizados' not in st.session_state:
        st.session_state.ajustes_realizados = []
    
    # Área para entrada do conteúdo
    texto_para_otimizar = st.text_area("Cole o conteúdo para otimização:", height=300)
    
    # Configurações
    col_config1, col_config2 = st.columns([2, 1])
    
    with col_config1:
        tipo_otimizacao = st.selectbox("Tipo de Otimização:", 
                                      ["SEO", "Engajamento", "Conversão", "Clareza"])
        
    with col_config2:
        tom_voz = st.text_input("Tom de Voz (ex: Técnico, Persuasivo):", 
                               value="Técnico",
                               key="tom_voz_otimizacao")
        
        nivel_heading = st.selectbox("Nível de Heading Solicitado:", 
                                   ["H1", "H2", "H3", "H4"],
                                   help="Nível de heading que foi solicitado no briefing. CORRIJA se o texto usar nível diferente")

    # CONFIGURAÇÕES DE BUSCA WEB
    st.subheader("🔍 Busca Web e Links")
    
    usar_busca_web = st.checkbox("Usar busca web para enriquecer conteúdo", 
                               value=True,
                               help="Ativa a busca no Perplexity para encontrar informações atualizadas")
    
    incluir_links_internos = st.checkbox("Incluir links internos", 
                                       value=True,
                                       help="Sugere e ancora links relevantes no texto")

    # Área para briefing
    instrucoes_briefing = st.text_area(
        "Instruções do briefing (opcional):",
        height=80
    )

    # Botão de otimização
    if st.button("🚀 Otimizar Conteúdo", type="primary", use_container_width=True):
        if texto_para_otimizar:
            with st.spinner("Otimizando conteúdo com busca web..."):
                try:
                    # FASE 1: BUSCA WEB (se ativada)
                    fontes_encontradas = ""
                    if usar_busca_web:
                        st.info("🔍 Buscando informações atualizadas e técnicas...")
                        
                        # Construir query de busca baseada no conteúdo
                        query_base = f"""
                        Informações técnicas atualizadas e fontes confiáveis:
                        
                        CONTEÚDO PARA OTIMIZAR:
                        {texto_para_otimizar[:300]}
                        
            
           
                        
                        Seja conciso e direto. Liste as informações em tópicos claros.
                        """
                        
                        # Buscar fontes relevantes usando Perplexity
                        resultado_busca = buscar_perplexity(query_base)
                        
                        if resultado_busca and not resultado_busca.startswith("❌"):
                            fontes_encontradas = resultado_busca
                            st.success(f"✅ Fontes técnicas encontradas")
                        else:
                            st.warning("⚠️ Busca web não retornou resultados. Continuando sem fontes externas.")
                    
                    # Contexto do agente
                    contexto_agente = ""
                    if st.session_state.agente_selecionado:
                        agente = st.session_state.agente_selecionado
                        contexto_agente = construir_contexto(agente, st.session_state.segmentos_selecionados)
                    
                    # Prompt de otimização COMPLETO
                    prompt = f"""
                    {contexto_agente}

                    ## TAREFA: OTIMIZAR CONTEÚDO SEGUINDO TODAS AS ESPECIFICAÇÕES

                    **TEXTO ORIGINAL PARA OTIMIZAÇÃO:**
                    {texto_para_otimizar}

                    **FONTES TÉCNICAS DA BUSCA WEB:**
                    {fontes_encontradas if fontes_encontradas else "Nenhuma fonte adicional encontrada na busca web."}

                    **INSTRUÇÕES DO BRIEFING:**
                    {instrucoes_briefing if instrucoes_briefing else 'Sem briefing específico'}

                    **CONFIGURAÇÕES:**
                    - Tipo de otimização: {tipo_otimizacao}
                    - Tom de voz: {tom_voz}
                    - Nível de heading solicitado: {nivel_heading}
                    - Incluir links internos: {"Sim" if incluir_links_internos else "Não"}
                    - Busca web utilizada: {"Sim" if usar_busca_web else "Não"}

                    ## ESPECIFICAÇÕES OBRIGATÓRIAS:

                    1. **SUGESTÕES DE TITLES E DESCRIPTIONS (OBRIGATÓRIO):**
                       - Gere 3 opções de meta title (máx 60 caracteres cada)
                       - Gere 3 opções de meta description (máx 155 caracteres cada)
                       - Inclua palavras-chave principais
                       - Title deve ter chamada para ação
                       - Description deve ser atrativa e incluir benefício

                    2. **BULLETS QUANDO APLICÁVEL EM SEO:**
                       - Para listas de benefícios: use bullets
                       - Para características técnicas: use bullets
                       - Para etapas de processo: use bullets
                       - Para comparativos: use bullets
                       - Limite de 3-5 itens por lista
                       - Cada bullet: máximo 1 linha

                    3. **NÍVEL DE HEADING DA TAG (CORREÇÃO OBRIGATÓRIA):**
                       - Verifique níveis de heading no conteúdo original
                       - Corrija se usar H4 quando foi solicitado {nivel_heading}
                       - Todos os headings principais devem ser {nivel_heading}
                       - Sub-headings devem seguir hierarquia apropriada

                    4. **CORREÇÕES AUTOMÁTICAS (TODAS DEVEM SER APLICADAS):**
                       - Reescreva introduções genéricas
                       - Quebre parágrafos longos (máx 3-4 frases)
                       - Remova informações duplicadas
                       - Remova tópicos fora do briefing
                       - Melhore escaneabilidade
                       - Divida frases muito longas
                       - Reforce público-alvo
                       - Use dados da busca web quando disponíveis

                    5. **LINKS INTERNOS (se solicitado):**
                       - Sugira 3-5 links internos relevantes
                       - Formato: [Texto âncora](URL) - inserir naturalmente no texto

                    ## FORMATO DE SAÍDA OBRIGATÓRIO:

                    ### 📊 SUGESTÕES DE TITLES E DESCRIPTIONS

                    **Opção 1 (Recomendada):**
                    Title: [title com até 60 caracteres]
                    Description: [description com até 155 caracteres]

                    **Opção 2:**
                    Title: [title com até 60 caracteres]
                    Description: [description com até 155 caracteres]

                    **Opção 3:**
                    Title: [title com até 60 caracteres]
                    Description: [description com até 155 caracteres]

                    ### ✅ CORREÇÕES APLICADAS
                    [Liste cada correção aplicada com detalhes]

                    ### 🔗 LINKS INTERNOS SUGERIDOS (se aplicável)
                    [Liste 3-5 links internos com contexto]

                    ### 📝 CONTEÚDO OTIMIZADO (COM TODAS AS CORREÇÕES)
                    [AQUI O CONTEÚDO COMPLETO OTIMIZADO com:
                    - Meta title e description selecionadas (usar a Opção 1 no início)
                    - Bullet points onde aplicável em SEO
                    - Headings corrigidos para {nivel_heading}
                    - Todas as correções aplicadas
                    - Fontes ancoradas quando usar dados da busca web
                    - Links internos inseridos (se solicitado)]

                    Aplique TODAS as especificações acima automaticamente.
                    """

                    resposta = modelo_texto.generate_content(prompt)
                    resultado = resposta.text
                    
                    # Processar resultado
                    partes_do_resultado = {}
                    
                    # Extrair seções
                    if "### 📊 SUGESTÕES DE TITLES E DESCRIPTIONS" in resultado:
                        partes = resultado.split("### 📊 SUGESTÕES DE TITLES E DESCRIPTIONS")
                        if len(partes) > 1:
                            meta_part = partes[1]
                            if "### ✅ CORREÇÕES APLICADAS" in meta_part:
                                meta_section = meta_part.split("### ✅ CORREÇÕES APLICADAS")[0]
                                partes_do_resultado["📊 SUGESTÕES DE TITLES E DESCRIPTIONS"] = meta_section.strip()
                            
                            # Extrair correções
                            if "### ✅ CORREÇÕES APLICADAS" in partes[1]:
                                corr_part = partes[1].split("### ✅ CORREÇÕES APLICADAS")[1]
                                if "### 🔗 LINKS INTERNOS SUGERIDOS" in corr_part:
                                    corr_section = corr_part.split("### 🔗 LINKS INTERNOS SUGERIDOS")[0]
                                    partes_do_resultado["✅ CORREÇÕES APLICADAS"] = corr_section.strip()
                                    
                                    # Extrair links
                                    links_part = corr_part.split("### 🔗 LINKS INTERNOS SUGERIDOS")[1]
                                    if "### 📝 CONTEÚDO OTIMIZADO" in links_part:
                                        links_section = links_part.split("### 📝 CONTEÚDO OTIMIZADO")[0]
                                        partes_do_resultado["🔗 LINKS INTERNOS SUGERIDOS"] = links_section.strip()
                                        
                                        # Extrair conteúdo
                                        content_part = links_part.split("### 📝 CONTEÚDO OTIMIZADO")[1]
                                        partes_do_resultado["📝 CONTEÚDO OTIMIZADO"] = content_part.strip()
                    
                    # Se não extraiu corretamente, usar resultado completo
                    if not partes_do_resultado:
                        partes_do_resultado["📝 CONTEÚDO OTIMIZADO"] = resultado
                    
                    # Salvar no session state
                    st.session_state.conteudo_otimizado = partes_do_resultado.get("📝 CONTEÚDO OTIMIZADO", resultado)
                    st.session_state.ultima_otimizacao = resultado
                    st.session_state.texto_original = texto_para_otimizar
                    st.session_state.fontes_encontradas = fontes_encontradas
                    st.session_state.partes_resultado = partes_do_resultado
                    
                    # Exibir resultados
                    st.success("✅ Conteúdo otimizado com todas as especificações!")
                    
                    # 1. Mostrar Meta Tags (OBRIGATÓRIO)
                    st.subheader("📊 Sugestões de Titles e Descriptions (Obrigatório)")
                    if "📊 SUGESTÕES DE TITLES E DESCRIPTIONS" in partes_do_resultado:
                        st.markdown(partes_do_resultado["📊 SUGESTÕES DE TITLES E DESCRIPTIONS"])
                        
                        # Extrair a primeira opção para usar no conteúdo
                        meta_content = partes_do_resultado["📊 SUGESTÕES DE TITLES E DESCRIPTIONS"]
                        if "**Opção 1 (Recomendada):**" in meta_content:
                            # Tentar extrair title e description da opção 1
                            lines = meta_content.split('\n')
                            for i, line in enumerate(lines):
                                if "Title:" in line:
                                    meta_title = line.replace("Title:", "").strip()
                                if "Description:" in line:
                                    meta_description = line.replace("Description:", "").strip()
                                    break
                    else:
                        st.warning("Meta tags não foram geradas no formato esperado")
                        # Tentar encontrar meta tags no resultado completo
                        lines = resultado.split('\n')
                        meta_found = []
                        for line in lines:
                            if 'title' in line.lower() or 'description' in line.lower() or 'meta' in line.lower():
                                meta_found.append(line)
                        if meta_found:
                            st.info("Possíveis meta tags encontradas:")
                            for line in meta_found[:6]:
                                st.write(line)
                    
                    # 2. Mostrar Correções Aplicadas
                    if "✅ CORREÇÕES APLICADAS" in partes_do_resultado:
                        with st.expander("✅ Correções Aplicadas (Detalhado)", expanded=True):
                            st.markdown(partes_do_resultado["✅ CORREÇÕES APLICADAS"])
                    
                    # 3. Mostrar Busca Web (se aplicável)
                    if fontes_encontradas and usar_busca_web:
                        with st.expander("🔍 Fontes Encontradas na Busca Web"):
                            st.markdown(fontes_encontradas)
                    
                    # 4. Mostrar Links Internos (se aplicável)
                    if "🔗 LINKS INTERNOS SUGERIDOS" in partes_do_resultado and incluir_links_internos:
                        with st.expander("🔗 Links Internos Sugeridos"):
                            st.markdown(partes_do_resultado["🔗 LINKS INTERNOS SUGERIDOS"])
                    
                    # 5. Mostrar Conteúdo Otimizado
                    st.subheader("📝 Conteúdo Otimizado (Com Todas as Correções)")
                    conteudo_final = partes_do_resultado.get("📝 CONTEÚDO OTIMIZADO", resultado)
                    st.markdown(conteudo_final)
                    
                    # Verificar especificações críticas
                    st.subheader("🔍 Verificação de Especificações")
                    
                    col_check1, col_check2, col_check3 = st.columns(3)
                    
                    with col_check1:
                        # Verificar meta tags
                        meta_found = 'title' in conteudo_final.lower() or 'description' in conteudo_final.lower()
                        st.metric("Meta Tags", "✅ Geradas" if meta_found else "⚠️ Verificar")
                    
                    with col_check2:
                        # Verificar bullets (contar no conteúdo)
                        bullet_count = conteudo_final.count("- ") + conteudo_final.count("* ")
                        st.metric("Bullet Points", bullet_count)
                    
                    with col_check3:
                        # Verificar heading level
                        heading_correct = nivel_heading.lower() in conteudo_final.lower()
                        st.metric(f"Heading {nivel_heading}", 
                                "✅ Presente" if heading_correct else "⚠️ Verificar")
                    
                    # Botão de download
                    st.download_button(
                        "💾 Baixar Conteúdo Otimizado",
                        data=conteudo_final,
                        file_name=f"conteudo_otimizado_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                        mime="text/plain"
                    )
                    
                    # Botão para download das meta tags separadas
                    if "📊 SUGESTÕES DE TITLES E DESCRIPTIONS" in partes_do_resultado:
                        st.download_button(
                            "📋 Baixar Apenas Meta Tags",
                            data=partes_do_resultado["📊 SUGESTÕES DE TITLES E DESCRIPTIONS"],
                            file_name=f"meta_tags_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                            mime="text/plain"
                        )
                
                except Exception as e:
                    st.error(f"Erro na otimização: {str(e)}")
        else:
            st.warning("Cole um conteúdo para otimizar")

    # Ajustes incrementais
    if st.session_state.conteudo_otimizado:
        st.divider()
        st.subheader("🔄 Ajustes Incrementais")
        
        comando_ajuste = st.text_area(
            "Ajustes desejados:",
            height=80,
            placeholder="""Exemplos:
- Adicione mais bullet points para benefícios
- Corrija todos os headings para H3
- Melhore as meta tags
- Adicione mais dados técnicos
- Simplifique a linguagem"""
        )
        
        if st.button("🔄 Aplicar Ajustes"):
            if comando_ajuste:
                with st.spinner("Aplicando ajustes..."):
                    try:
                        historico = ""
                        if st.session_state.ajustes_realizados:
                            historico = "**Histórico de ajustes:**\n"
                            for i, a in enumerate(st.session_state.ajustes_realizados, 1):
                                historico += f"{i}. {a}\n"
                        
                        prompt_ajuste = f"""
                        ## AJUSTE INCREMENTAL COM ESPECIFICAÇÕES

                        **CONTEÚDO ATUAL (JÁ OTIMIZADO):**
                        {st.session_state.conteudo_otimizado}

                        **CONFIGURAÇÕES ORIGINAIS:**
                        - Tipo: {tipo_otimizacao}
                        - Tom: {tom_voz}
                        - Heading solicitado: {nivel_heading}
                        - Meta tags obrigatórias: SIM
                        - Bullets quando aplicável: SIM

                        {historico}

                        **NOVOS AJUSTES SOLICITADOS:**
                        {comando_ajuste}

                        **REGRAS DE AJUSTE (MANTENHA):**
                        1. Meta tags devem ser mantidas ou melhoradas
                        2. Bullet points devem ser usados quando aplicável
                        3. Heading level {nivel_heading} deve ser mantido
                        4. Parágrafos curtos (3-4 frases máx)
                        5. Escaneabilidade preservada

                        **FORMATO DE RESPOSTA:**
                        ### ✅ AJUSTES APLICADOS:
                        [Liste ajustes aplicados]

                        ### 📊 META TAGS ATUALIZADAS (se necessário):
                        [Meta title e description atualizadas]

                        ### 📝 CONTEÚDO ATUALIZADO:
                        [Conteúdo completo com ajustes aplicados]

                        Aplique os ajustes mantendo TODAS as especificações anteriores.
                        """

                        resposta_ajuste = modelo_texto.generate_content(prompt_ajuste)
                        resultado_ajuste = resposta_ajuste.text
                        
                        # Extrair conteúdo atualizado
                        if "### 📝 CONTEÚDO ATUALIZADO:" in resultado_ajuste:
                            partes = resultado_ajuste.split("### 📝 CONTEÚDO ATUALIZADO:")
                            if len(partes) > 1:
                                conteudo_atualizado = partes[1].strip()
                            else:
                                conteudo_atualizado = resultado_ajuste.strip()
                        else:
                            conteudo_atualizado = resultado_ajuste.strip()
                        
                        # Atualizar session state
                        st.session_state.conteudo_otimizado = conteudo_atualizado
                        st.session_state.ajustes_realizados.append(comando_ajuste)
                        
                        st.success("✅ Ajustes aplicados mantendo especificações")
                        st.markdown(conteudo_atualizado)
                        
                        st.download_button(
                            "💾 Baixar Versão Atualizada",
                            data=conteudo_atualizado,
                            file_name=f"conteudo_ajustado_{len(st.session_state.ajustes_realizados)}_v_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                            mime="text/plain"
                        )
                        
                    except Exception as e:
                        st.error(f"Erro ao aplicar ajustes: {str(e)}")
            else:
                st.warning("Digite ajustes desejados")
        
        # Limpar histórico
        if st.button("🗑️ Limpar Histórico de Ajustes"):
            st.session_state.ajustes_realizados = []
            st.success("Histórico limpo")
# ========== ABA: CRIADORA DE CALENDÁRIO ==========
with tab_calendario:
    st.header("📅 Criadora de Calendário de Conteúdo")
    st.markdown("Gere um calendário de pautas no formato visual para conteúdo agrícola")
    
    # Verificar se há agente selecionado
    if not st.session_state.agente_selecionado:
        st.warning("⚠️ Selecione um agente na parte superior do app para usar esta funcionalidade.")
        st.info("O calendário será gerado com base nos produtos e informações do agente selecionado.")
    else:
        agente = st.session_state.agente_selecionado
        st.success(f"🎯 Gerando calendário com base no agente: **{agente['nome']}**")
        
        # Configurações do calendário
        col_cal1, col_cal2, col_cal3 = st.columns([2, 1, 1])
        
        with col_cal1:
            # Time frame e mês/ano
            st.subheader("📅 Período do Calendário")
            mes_ano = st.text_input("Mês/Ano para o calendário:", "AGOSTO 2025")
            data_inicio = st.date_input("Data de início:", value=datetime.date(2025, 8, 1))
            data_fim = st.date_input("Data de término:", value=datetime.date(2025, 8, 31))
            
            # Validar datas
            if data_inicio >= data_fim:
                st.error("❌ A data de início deve ser anterior à data de término")
            
            # Calcular número de dias
            delta_dias = (data_fim - data_inicio).days + 1
            dias_uteis = sum(1 for i in range(delta_dias) if (data_inicio + datetime.timedelta(days=i)).weekday() < 5)
            st.info(f"📆 Período de {delta_dias} dias ({dias_uteis} dias úteis)")
            
            # Configuração de culturas
            st.subheader("🌱 Culturas para Foco")
            culturas_prioritarias = st.text_area(
                "Lista de culturas para focar no calendário (separadas por vírgula):",
                "Soja, Milho, Cana-de-açúcar, Algodão, Café, Trigo, HF",
                help="Digite as culturas que serão priorizadas no calendário, separadas por vírgula"
            )
            # Processar culturas
            culturas_lista = [cultura.strip() for cultura in culturas_prioritarias.split(",") if cultura.strip()]
        
        with col_cal2:
            st.subheader("🎯 Configurações de Pautas")
            
            # Configuração de pautas apenas de culturas
            st.write("**Pautas apenas de culturas:**")
            pautas_apenas_culturas = st.slider(
                "Número de pautas sem produtos:",
                min_value=0,
                max_value=30,
                value=8,
                help="Número total de pautas que serão apenas sobre culturas (sem produtos específicos)",
                key="pautas_culturas"
            )
            
            # Controles separados para dias com 2 e 3 pautas
            st.write("**Dias com 2 pautas:**")
            dias_com_2_pautas = st.slider(
                "Número de dias com 2 pautas:",
                min_value=0,
                max_value=delta_dias,
                value=max(0, delta_dias - 5),  # Valor padrão: total - 5
                help="Dias que terão 2 pautas",
                key="dias_2_pautas"
            )
            
            st.write("**Dias com 3 pautas:**")
            dias_com_3_pautas = st.slider(
                "Número de dias com 3 pautas:",
                min_value=0,
                max_value=delta_dias,
                value=min(5, delta_dias),  # Valor padrão: 5 ou total se menor
                help="Dias que terão 3 pautas",
                key="dias_3_pautas"
            )
            
            # Configuração básica
            usar_contexto_completo = st.checkbox("Usar contexto completo do agente", value=True,
                                               help="Inclui brand guidelines, planejamento e comentários do agente")
        
        with col_cal3:
            st.subheader("📋 Diretrizes")
            
            # Configurações específicas
            incluir_fins_semana = st.checkbox("Incluir sábado e domingo", value=False,
                                            help="Incluir pautas nos fins de semana (opcional)")
            verificar_datas_comemorativas = st.checkbox("Verificar datas comemorativas", value=True,
                                                      help="Incluir pautas relacionadas a datas comemorativas do agro")
            considerar_temporalidade = st.checkbox("Considerar temporalidade", value=True,
                                                 help="Considerar época de uso dos produtos nas culturas")
            
            # Estatísticas automáticas com validação
            st.subheader("📊 Estatísticas")
            st.write(f"**Total de dias:** {delta_dias}")
            
            # Validar se a soma dos dias não excede o total
            total_dias_configurados = dias_com_2_pautas + dias_com_3_pautas
            dias_restantes = delta_dias - total_dias_configurados
            
            if total_dias_configurados > delta_dias:
                st.error(f"❌ Total de dias configurados ({total_dias_configurados}) excede período ({delta_dias})")
            else:
                st.write(f"**Dias com 2 pautas:** {dias_com_2_pautas}")
                st.write(f"**Dias com 3 pautas:** {dias_com_3_pautas}")
                if dias_restantes > 0:
                    st.write(f"**Dias sem pautas:** {dias_restantes}")
                else:
                    st.write("**Dias sem pautas:** 0")
                
                st.write(f"**Pautas só culturas:** {pautas_apenas_culturas}")
                total_pautas = (dias_com_2_pautas * 2) + (dias_com_3_pautas * 3)
                st.write(f"**Total de pautas:** {total_pautas}")
                st.write(f"**Culturas:** {len(culturas_lista)}")
        
        # Input de produtos com culturas e direcionais
        st.subheader("🏭 Produtos e Direcionais")
        
        st.info("💡 **Formato:** Produto - Cultura - Direcional (ex: Elestal Neo - Soja - Controle de mosca-branca)")
        
        produtos_direcionais = st.text_area(
            "Lista de produtos com culturas e direcionais (um por linha):",
            """Elestal Neo - Soja - Controle de mosca-branca
YieldOn - Soja - Bioativador para pegamento de flores
Miravis - Soja - Fungicida para ferrugem
Fortenza - Milho - Seedcare para cigarrinha
Victrato - Cana - Nematicida para cana-soca""",
            height=150,
            help="Digite um produto por linha no formato: PRODUTO - CULTURA - DIRECIONAL"
        )
        
        # Processar produtos com direcionais
        produtos_com_direcionais = []
        if produtos_direcionais:
            for linha in produtos_direcionais.split('\n'):
                linha = linha.strip()
                if linha and ' - ' in linha:
                    partes = linha.split(' - ')
                    if len(partes) >= 3:
                        produto = partes[0].strip()
                        cultura = partes[1].strip()
                        direcional = ' - '.join(partes[2:]).strip()
                        produtos_com_direcionais.append({
                            'produto': produto,
                            'cultura': cultura,
                            'direcional': direcional
                        })
        
        # Contexto adicional
        st.subheader("📝 Contexto do Mês (Opcional)")
        contexto_mensal = st.text_area(
            "Informações contextuais do mês para orientar as pautas:",
            placeholder="""Exemplo para Janeiro:
- Soja: colheita, enchimento, florescimento tardio
- Milho safrinha: início do plantio
- Cana: crescimento vegetativo máximo
- Algodão: pico de plantio
- Clima: chuvas intensas, atenção a doenças e pragas
- Datas comemorativas: Dia do Solo (15/04), Dia do Milho (24/05)""",
            height=120,
            help="Contexto mensal para orientar as pautas, incluindo datas comemorativas se aplicável"
        )
        
        # Editorias disponíveis
        st.subheader("📰 Editorias")
        editorias_disponiveis = {
            "🔵": "Dia a dia do campo",
            "🟠": "Inovações e tendências", 
            "🟢": "Sustentabilidade",
            "🔴": "Mercado e safra",
            "🟣": "Especialistas"
        }
        
        editorias_selecionadas = st.multiselect(
            "Selecione as editorias para o calendário:",
            list(editorias_disponiveis.values()),
            default=list(editorias_disponiveis.values()),
            help="Editorias que serão utilizadas no calendário"
        )
        
        # Função auxiliar para descrições das editorias
        def get_editoria_description(editoria_nome):
            descriptions = {
                "Dia a dia do campo": "Pautas sobre cotidiano dos produtores, pragas, doenças, plantas daninhas, manejo e aplicação de produtos",
                "Inovações e tendências": "Novas tecnologias, moléculas, equipamentos e futuro do agro",
                "Sustentabilidade": "Práticas sustentáveis, produtos biológicos, saúde do solo, uso consciente",
                "Mercado e safra": "Notícias sobre mercado, expectativas de safra, clima, preços e colheita", 
                "Especialistas": "Conteúdos chancelados por autoridades no assunto"
            }
            return descriptions.get(editoria_nome, "Conteúdos diversos sobre o agro")
        
        # Botão para gerar calendário
        if st.button("🔄 Gerar Calendário", type="primary", use_container_width=True):
            if data_inicio >= data_fim:
                st.error("❌ Corrija as datas antes de gerar o calendário")
            elif not culturas_lista:
                st.error("❌ Digite pelo menos uma cultura")
            elif (dias_com_2_pautas + dias_com_3_pautas) > delta_dias:
                st.error(f"❌ Total de dias configurados ({dias_com_2_pautas + dias_com_3_pautas}) excede período ({delta_dias})")
            else:
                with st.spinner("🔄 Analisando temporalidade e gerando calendário visual..."):
                    try:
                        # Construir contexto completo do agente
                        contexto_agente = ""
                        if usar_contexto_completo:
                            contexto_agente = construir_contexto(agente, st.session_state.segmentos_selecionados)
                        else:
                            # Usar apenas base de conhecimento se disponível
                            if "base_conhecimento" in st.session_state.segmentos_selecionados and agente.get('base_conhecimento'):
                                contexto_agente = f"### BASE DE CONHECIMENTO ###\n{agente['base_conhecimento']}\n\n"
                        
                        # Mapear editorias selecionadas para emojis
                        editorias_emoji = {}
                        for emoji, nome in editorias_disponiveis.items():
                            if nome in editorias_selecionadas:
                                editorias_emoji[emoji] = nome
                        
                        # Calcular distribuição
                        total_pautas = (dias_com_2_pautas * 2) + (dias_com_3_pautas * 3)
                        dias_sem_pautas = delta_dias - (dias_com_2_pautas + dias_com_3_pautas)
                        
                        # Preparar informações de produtos com direcionais
                        info_produtos_direcionais = ""
                        if produtos_com_direcionais:
                            info_produtos_direcionais = f"""
                        **PRODUTOS COM CULTURAS E DIRECIONAIS:**
                        {chr(10).join([f'- {p["produto"]} - {p["cultura"]} - {p["direcional"]}' for p in produtos_com_direcionais])}
                        
                        Use estes produtos específicos com as culturas e direcionais indicados.
                        Um mesmo produto pode aparecer com culturas diferentes para propósitos diferentes.
                            """
                        else:
                            info_produtos_direcionais = "**PRODUTOS:** Use produtos relevantes da base de conhecimento, considerando a temporalidade de uso."
                        
                        # Usar a função definida localmente
                        editorias_info = "\n".join([f'   - {emoji} {nome}: {get_editoria_description(nome)}' for emoji, nome in editorias_emoji.items()])
                        
                        # Prompt para gerar o calendário
                        prompt_calendario = f"""
                        {contexto_agente}

                        ## TAREFA: GERAR CALENDÁRIO DE PAUTAS CONSIDERANDO TEMPORALIDADE

                        **INFORMAÇÕES:**
                        - Mês/Ano: {mes_ano}
                        - Data início: {data_inicio.strftime('%d/%m/%Y')}
                        - Data término: {data_fim.strftime('%d/%m/%Y')}
                        - Período: {delta_dias} dias ({dias_uteis} dias úteis)
                        - Culturas prioritárias: {', '.join(culturas_lista)}
                        - Editorias disponíveis: {', '.join([f'{emoji} {nome}' for emoji, nome in editorias_emoji.items()])}
                        
                        {info_produtos_direcionais}
                        
                        **DISTRIBUIÇÃO DE PAUTAS:**
                        - Pautas apenas de culturas (sem produtos): {pautas_apenas_culturas} pautas
                        - Dias com 2 pautas: {dias_com_2_pautas} dias
                        - Dias com 3 pautas: {dias_com_3_pautas} dias
                        - Dias sem pautas: {dias_sem_pautas} dias
                        - Total de pautas: {total_pautas}
                        - Incluir fins de semana: {incluir_fins_semana}

                        **CONTEXTO DO MÊS:**
                        {contexto_mensal if contexto_mensal else "Nenhum contexto mensal específico fornecido."}

                        ## FORMATO EXATO EXIGIDO:

                        Você DEVE gerar um CSV que quando aberto no Excel forme uma visualização de calendário mensal, seguindo EXATAMENTE este formato:

                        Linha 1: ,,CALENDÁRIO DE PAUTAS,,,,[MÊS],,[ANO],,,,,,,,,,,,,,,,
                        Linha 2: ,,{',,'.join([f'{emoji}- {nome}' for emoji, nome in editorias_emoji.items()])},,,,,,,,,,,,,,,,,
                        Linha 3: ,,DOMINGO,SEGUNDA,TERÇA,QUARTA,QUINTA,SEXTA,SÁBADO,,,,,,,,,,,,,,,,,

                        Depois, para cada semana:
                        - Uma linha com os números dos dias (ex: ,,3,4,5,6,7,8,9)
                        - Linhas com as pautas (2 ou 3 por dia conforme distribuição)

                        ## REGRAS ESTRITAS:

                        1. **FORMATO DAS PAUTAS:**
                           - Pautas de produtos: "[EMOJI] NomeProduto – Cultura - TítuloMacro - BreveDescrição"
                           - Pautas de culturas: "[EMOJI] Cultura - TítuloMacro - BreveDescrição"

                        2. **DISTRIBUIÇÃO OBRIGATÓRIA:**
                           - {pautas_apenas_culturas} pautas devem ser APENAS de culturas (sem produtos)
                           - {dias_com_2_pautas} dias devem ter EXATAMENTE 2 pautas cada
                           - {dias_com_3_pautas} dias devem ter EXATAMENTE 3 pautas cada
                           - {dias_sem_pautas} dias devem ficar SEM PAUTAS
                           - Priorizar segunda a sexta-feira, mas pode incluir sábado/domingo se solicitado

                        3. **PRODUTOS E CULTURAS:**
                           - Um mesmo produto pode aparecer com culturas diferentes para propósitos diferentes
                           - Use os produtos com direcionais específicos quando fornecidos
                           - Considere a temporalidade de uso dos produtos nas culturas

                        4. **EXEMPLOS CORRETOS:**
                           - Produto: "🔵 Elestal Neo - Soja - Ação contra mosca-branca - Reforçar uso do produto no controle da mosca-branca"
                           - Produto com outra cultura: "🔵 Elestal Neo - Algodão - Proteção contra pulgões - Destaque para controle eficiente"
                           - Cultura: "🟢 Soja - Pegamento de flores - Explicar fatores que causam abortamento floral"

                        5. **BALANCEAMENTO:**
                           - Distribuir pautas de culturas ao longo do mês (não concentrar em dias específicos)
                           - Evitar repetição excessiva dos mesmos produtos no mesmo dia
                           - Balancear editorias ao longo do mês
                           - Variar culturas entre os dias
                           - Dias com 3 pautas podem ter mais variedade de temas

                        6. **CONTEÚDO RELEVANTE:**
                           - Temas em linha com o momento do campo no mês
                           - Considerar temporalidade de uso dos produtos
                           - Destacar produtos mais relevantes para o período
                           - {f"Incluir datas comemorativas se aplicável" if verificar_datas_comemorativas else "Não é necessário incluir datas comemorativas"}

                        7. **TEMPORALIDADE:**
                           - Analise qual é a época de uso de cada produto nas culturas
                           - Priorize produtos que são mais relevantes para o período do calendário
                           - Considere o estágio fenológico das culturas no mês
                           - Destaque produtos com época de aplicação coincidente com o período

                        8. **EDITORIAS:**
                        {editorias_info}

                        ## ESTRATÉGIA DE DISTRIBUIÇÃO:

                        Para as {pautas_apenas_culturas} pautas apenas de culturas:
                        - Distribua ao longo dos dias COM pautas, não concentre em dias específicos
                        - Use para temas macro: expectativas de safra, clima, tendências
                        - Exemplo: "🔴 Soja - Andamento da colheita - Análise do avanço da colheita e perspectivas de mercado"

                        Para os {dias_com_2_pautas} dias com 2 pautas:
                        - Mantenha foco em temas do dia a dia do campo
                        - Balance entre produtos e culturas
                        - Pode ser 1 produto + 1 cultura, ou 2 produtos diferentes

                        Para os {dias_com_3_pautas} dias com 3 pautas:
                        - Distribua entre diferentes culturas e editorias
                        - Inclua variedade: 1 pauta de cultura + 2 de produtos, ou 3 produtos diferentes
                        - Use para temas mais complexos ou campanhas especiais

                        Para os {dias_sem_pautas} dias sem pautas:
                        - Deixe as células vazias no calendário
                        - Distribua esses dias de forma estratégica (fins de semana, feriados, etc.)

                        **IMPORTANTE:**
                        - Retorne APENAS o CSV completo, sem texto adicional
                        - Formate EXATAMENTE como o exemplo fornecido
                        - Garanta que seja um calendário visual legível no Excel
                        - Siga a distribuição EXATA: {dias_com_2_pautas} dias com 2 pautas, {dias_com_3_pautas} dias com 3 pautas
                        - {dias_sem_pautas} dias devem ficar completamente sem pautas
                        - Considere a temporalidade de uso dos produtos
                        - Um produto pode aparecer com múltiplas culturas
                        - Distribua pautas de culturas ao longo do mês
                        """

                        # Gerar o calendário
                        resposta = modelo_texto.generate_content(prompt_calendario)
                        calendario_csv = resposta.text
                        
                        # Processar e exibir o resultado
                        st.success("✅ Calendário visual gerado com sucesso!")
                        
                        # Limpar o CSV de possíveis markdown
                        calendario_limpo = calendario_csv.strip()
                        if '```csv' in calendario_limpo:
                            calendario_limpo = calendario_limpo.replace('```csv', '').replace('```', '')
                        if '```' in calendario_limpo:
                            calendario_limpo = calendario_limpo.replace('```', '')
                        
                        # Salvar na sessão para usar na visualização
                        st.session_state.calendario_gerado = calendario_limpo
                        st.session_state.mes_ano_calendario = mes_ano
                        st.session_state.distribuicao_pautas = {
                            'pautas_apenas_culturas': pautas_apenas_culturas,
                            'dias_2_pautas': dias_com_2_pautas,
                            'dias_3_pautas': dias_com_3_pautas,
                            'dias_sem_pautas': dias_sem_pautas,
                            'total_pautas': total_pautas,
                            'culturas': len(culturas_lista),
                            'produtos_com_direcionais': len(produtos_com_direcionais)
                        }
                    
                    except Exception as e:
                        st.error(f"❌ Erro ao gerar calendário: {str(e)}")
                        st.info("💡 Dica: Verifique se o agente selecionado possui base de conhecimento relevante")

        # Seção de visualização do calendário gerado
        if 'calendario_gerado' in st.session_state and st.session_state.calendario_gerado:
            st.markdown("---")
            st.subheader(f"📊 Visualização do Calendário - {st.session_state.mes_ano_calendario}")
            
            # Mostrar distribuição utilizada
            if hasattr(st.session_state, 'distribuicao_pautas'):
                dist = st.session_state.distribuicao_pautas
                col_dist1, col_dist2, col_dist3, col_dist4, col_dist5, col_dist6 = st.columns(6)
                with col_dist1:
                    st.metric("Dias 2 pautas", dist['dias_2_pautas'])
                with col_dist2:
                    st.metric("Dias 3 pautas", dist['dias_3_pautas'])
                with col_dist3:
                    st.metric("Dias sem pautas", dist['dias_sem_pautas'])
                with col_dist4:
                    st.metric("Pautas culturas", dist['pautas_apenas_culturas'])
                with col_dist5:
                    st.metric("Total pautas", dist['total_pautas'])
                with col_dist6:
                    st.metric("Produtos", dist['produtos_com_direcionais'])
            
            # Abas para diferentes visualizações
            tab_csv, tab_tabela, tab_analise = st.tabs(["📋 CSV Original", "🗓️ Visualização em Tabela", "📈 Análise"])
            
            with tab_csv:
                st.subheader("📋 CSV para Download")
                st.text_area("Conteúdo CSV:", st.session_state.calendario_gerado, height=400, key="csv_calendario")
                
                # Botão de download do CSV
                st.download_button(
                    "💾 Baixar Calendário CSV",
                    data=st.session_state.calendario_gerado,
                    file_name=f"calendario_pautas_{mes_ano.replace(' ', '_').lower()}_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                    type="primary"
                )
                
                st.info("💡 **Dica:** Abra este arquivo CSV no Excel para visualizar o calendário em formato de grade")
            
            with tab_tabela:
                st.subheader("🗓️ Visualização em Tabela")
                try:
                    # Converter CSV para DataFrame para visualização
                    import io
                    import pandas as pd
                    
                    # Ler o CSV
                    csv_data = io.StringIO(st.session_state.calendario_gerado)
                    df = pd.read_csv(csv_data, header=None)
                    
                    # Mostrar a tabela
                    st.dataframe(df, use_container_width=True, hide_index=True)
                    
                    # Estatísticas da tabela
                    st.subheader("📊 Estatísticas da Tabela")
                    col_stat1, col_stat2, col_stat3 = st.columns(3)
                    with col_stat1:
                        st.metric("Linhas", df.shape[0])
                    with col_stat2:
                        st.metric("Colunas", df.shape[1])
                    with col_stat3:
                        celulas_preenchidas = df.notna().sum().sum()
                        st.metric("Células com conteúdo", celulas_preenchidas)
                        
                except Exception as e:
                    st.error(f"Erro ao processar tabela: {str(e)}")
                    st.info("Visualizando como texto simples:")
                    st.text(st.session_state.calendario_gerado)
            
            with tab_analise:
                st.subheader("📈 Análise do Calendário")
                
                # Análise automática do conteúdo gerado
                try:
                    linhas = st.session_state.calendario_gerado.split('\n')
                    pautas_encontradas = 0
                    culturas_utilizadas = set()
                    produtos_utilizados = set()
                    editorias_utilizadas = set()
                    
                    # Mapeamento de emojis para editorias
                    emoji_editoria = {
                        '🔵': 'Dia a dia do campo',
                        '🟠': 'Inovações e tendências',
                        '🟢': 'Sustentabilidade', 
                        '🔴': 'Mercado e safra',
                        '🟣': 'Especialistas'
                    }
                    
                    for linha in linhas:
                        for emoji, editoria in emoji_editoria.items():
                            if emoji in linha:
                                editorias_utilizadas.add(editoria)
                                pautas_encontradas += 1
                                
                                # Tentar extrair cultura
                                if ' - ' in linha:
                                    partes = linha.split(' - ')
                                    if len(partes) >= 2:
                                        # Segunda parte geralmente tem a cultura
                                        cultura_candidata = partes[1].strip()
                                        if cultura_candidata and len(cultura_candidata) < 50:  # Filtro simples
                                            culturas_utilizadas.add(cultura_candidata)
                                        
                                        # Tentar identificar produto (primeira parte sem emoji)
                                        produto_candidato = partes[0].replace(emoji, '').strip()
                                        if produto_candidato and produto_candidato not in list(emoji_editoria.values()):
                                            produtos_utilizados.add(produto_candidato)
                    
                    # Mostrar análise
                    col_ana1, col_ana2, col_ana3 = st.columns(3)
                    with col_ana1:
                        st.metric("Pautas encontradas", pautas_encontradas)
                    with col_ana2:
                        st.metric("Culturas utilizadas", len(culturas_utilizadas))
                    with col_ana3:
                        st.metric("Produtos utilizados", len(produtos_utilizados))
                    
                    # Detalhes
                    with st.expander("🔍 Detalhes das Culturas"):
                        if culturas_utilizadas:
                            st.write("**Culturas encontradas:**")
                            for cultura in sorted(culturas_utilizadas):
                                st.write(f"- {cultura}")
                        else:
                            st.info("Nenhuma cultura identificada automaticamente")
                    
                    with st.expander("🏭 Detalhes dos Produtos"):
                        if produtos_utilizados:
                            st.write("**Produtos encontrados:**")
                            for produto in sorted(produtos_utilizados):
                                st.write(f"- {produto}")
                        else:
                            st.info("Nenhum produto identificado automaticamente")
                    
                    with st.expander("📰 Distribuição por Editoria"):
                        if editorias_utilizadas:
                            st.write("**Editorias utilizadas:**")
                            for editoria in sorted(editorias_utilizadas):
                                st.write(f"- {editoria}")
                        else:
                            st.info("Nenhuma editoria identificada automaticamente")
                            
                except Exception as e:
                    st.error(f"Erro na análise: {str(e)}")

        # Seção de informações
        with st.expander("ℹ️ Instruções de Uso"):
            st.markdown("""
            ### 🎯 Controles de Distribuição

            **Dias com 2 pautas:**
            - Configure quantos dias terão exatamente 2 pautas
            - Ideal para dias com temas mais focados
            - Exemplo: 15 dias com 2 pautas = 30 pautas

            **Dias com 3 pautas:**
            - Configure quantos dias terão exatamente 3 pautas  
            - Para dias com maior variedade de conteúdo
            - Exemplo: 10 dias com 3 pautas = 30 pautas

            **Dias sem pautas:**
            - Dias que ficarão vazios no calendário
            - Calculado automaticamente: Total - (Dias 2 pautas + Dias 3 pautas)
            - Útil para fins de semana, feriados, ou dias de menor movimento

            **Pautas apenas de culturas:**
            - Número total de pautas sem produtos específicos
            - Distribuídas estrategicamente entre os dias com pautas

            ### 💡 Dicas para Melhor Visualização

            1. **No Excel:** Abra o arquivo CSV diretamente para ver o calendário em formato de grade
            2. **No Google Sheets:** Importe o CSV para visualização similar
            3. **Formatação:** As células estão organizadas para formar um calendário visual
            4. **Cores:** Os emojis ajudam a identificar rapidamente o tipo de conteúdo
            """)

# ========== ABA: GERADOR DE BRIEFINGS ==========
with tab_briefings:
    st.header("📋 Gerador de Briefings a partir do Calendário")
    
    # Verificar se há agente selecionado
    if not st.session_state.agente_selecionado:
        st.warning("⚠️ Selecione um agente na parte superior do app para usar esta funcionalidade.")
    else:
        agente = st.session_state.agente_selecionado
        st.success(f"🎯 Gerando briefings com base no agente: **{agente['nome']}**")
        
        # Inicializar session_state para briefings
        if 'briefings_gerados' not in st.session_state:
            st.session_state.briefings_gerados = []
        
        # Upload do CSV ou usar o gerado
        col_upload1, col_upload2 = st.columns([2, 1])
        
        with col_upload1:
            usar_calendario_existente = st.checkbox("Usar calendário gerado anteriormente", 
                                                  value='calendario_gerado' in st.session_state)
            
            if not usar_calendario_existente or 'calendario_gerado' not in st.session_state:
                arquivo_calendario = st.file_uploader("📅 Upload do calendário CSV:", type=['csv'])
            else:
                st.info("✅ Usando calendário gerado anteriormente")
                arquivo_calendario = None
        
        with col_upload2:
            mes_referencia = st.text_input("Mês de referência:", "JANEIRO 2026")
            ano_referencia = st.text_input("Ano de referência:", "2026")
        
        # Contexto adicional para os briefings
        contexto_briefings = st.text_area(
            "Informações contextuais para orientar a criação dos briefings:",
            placeholder="Exemplo: Foco em campanha de posicionamento de produtos, linguagem técnica mas acessível...",
            height=80
        )
        
        # Botão para processar e gerar briefings
        if st.button("🔄 Processar Calendário e Gerar Briefings", type="primary", use_container_width=True):
            # Obter o conteúdo do CSV
            conteudo_csv = ""
            
            if usar_calendario_existente and 'calendario_gerado' in st.session_state:
                conteudo_csv = st.session_state.calendario_gerado
                st.success("✅ Usando calendário da sessão")
            elif arquivo_calendario is not None:
                try:
                    # Tentar diferentes encodings
                    file_bytes = arquivo_calendario.getvalue()
                    
                    # Tentar UTF-8 primeiro
                    try:
                        conteudo_csv = file_bytes.decode('utf-8')
                    except UnicodeDecodeError:
                        # Tentar Latin-1 (ISO-8859-1)
                        try:
                            conteudo_csv = file_bytes.decode('latin-1')
                        except UnicodeDecodeError:
                            # Tentar UTF-8 com tratamento de erros
                            conteudo_csv = file_bytes.decode('utf-8', errors='ignore')
                    
                    st.success("✅ Arquivo CSV carregado")
                except Exception as e:
                    st.error(f"❌ Erro ao ler arquivo: {str(e)}")
                    st.stop()
            else:
                st.error("❌ Nenhum calendário disponível para processar")
                st.stop()
            
            # Processar o CSV para extrair TODAS as células de conteúdo
            with st.spinner("📋 Processando calendário e extraindo pautas..."):
                try:
                    linhas = conteudo_csv.split('\n')
                    todas_pautas = []
                    
                    # Processar cada linha do CSV para encontrar TODAS as pautas
                    for linha_num, linha in enumerate(linhas):
                        # Limpar a linha de caracteres problemáticos
                        linha_limpa = linha.strip().replace('\r', '').replace('﻿', '')  # Remove BOM
                        if not linha_limpa:
                            continue
                            
                        celulas = linha_limpa.split(',')
                        for celula_num, celula in enumerate(celulas):
                            celula_limpa = celula.strip()
                            
                            # CRITÉRIO SIMPLES: qualquer conteúdo com mais de 15 caracteres que não seja apenas números
                            if (celula_limpa and 
                                len(celula_limpa) > 15 and 
                                not celula_limpa.replace('.', '').isdigit() and  # Não é apenas número
                                not any(header in celula_limpa for header in ['DOMINGO', 'SEGUNDA', 'TERÇA', 'QUARTA', 'QUINTA', 'SEXTA', 'SÁBADO', 'CALENDÁRIO']) and
                                'CX,' not in celula_limpa):
                                
                                # É uma pauta - processar cada uma separadamente
                                pautas_na_celula = []
                                
                                # Dividir por quebras de linha para pegar múltiplas pautas na mesma célula
                                if '\n' in celula_limpa:
                                    # Célula com múltiplas pautas (2 ou 3 pautas por dia)
                                    sub_pautas = celula_limpa.split('\n')
                                    for sub_pauta in sub_pautas:
                                        sub_pauta_limpa = sub_pauta.strip()
                                        if sub_pauta_limpa and len(sub_pauta_limpa) > 15:
                                            pautas_na_celula.append(sub_pauta_limpa)
                                else:
                                    # Célula com uma única pauta
                                    pautas_na_celula.append(celula_limpa)
                                
                                # Adicionar cada pauta individualmente
                                for pauta in pautas_na_celula:
                                    # Limpar e padronizar a pauta
                                    pauta_limpa = pauta.strip()
                                    pauta_limpa = ' '.join(pauta_limpa.split())
                                    
                                    todas_pautas.append({
                                        'conteudo': pauta_limpa,
                                        'linha': linha_num,
                                        'coluna': celula_num,
                                        'indice': len(todas_pautas) + 1
                                    })
                    
                    st.success(f"✅ Encontradas {len(todas_pautas)} pautas individuais no calendário")
                    
                    if not todas_pautas:
                        st.error("❌ Nenhuma pauta válida encontrada no CSV")
                        st.info("💡 **Dica:** O sistema procura por qualquer conteúdo com mais de 15 caracteres")
                        st.stop()
                    
                    # Mostrar preview das pautas encontradas
                    with st.expander("👀 Visualizar Pautas Detectadas", expanded=True):
                        st.write(f"**Total de pautas detectadas:** {len(todas_pautas)}")
                        st.write("**Primeiras 10 pautas:**")
                        for i, pauta in enumerate(todas_pautas[:10]):
                            st.write(f"{i+1}. {pauta['conteudo']}")
                    
                    # Gerar briefings para CADA pauta individual
                    st.subheader("📄 Gerando Briefings para Cada Pauta")
                    
                    # Construir contexto do agente
                    contexto_agente = construir_contexto(agente, st.session_state.segmentos_selecionados)
                    
                    # Processar TODAS as pautas
                    pautas_processar = todas_pautas
                    st.info(f"🔄 Gerando {len(pautas_processar)} briefings")
                    
                    briefings_gerados = []
                    
                    # Barra de progresso
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for idx, pauta in enumerate(pautas_processar):
                        status_text.text(f"Fazendo briefing da pauta {idx+1}/{len(pautas_processar)}: {pauta['conteudo'][:50]}...")
                        progress_bar.progress((idx + 1) / len(pautas_processar))
                        
                        try:
                            # Prompt SIMPLES e DIRETO para gerar o briefing
                            prompt_briefing = f"""
                            {contexto_agente}

                            ## TAREFA: GERAR BRIEFING COMPLETO PARA ESTA PAUTA ESPECÍFICA

                            **PAUTA ESPECÍFICA:**
                            {pauta['conteudo']}

                            **MÊS DE REFERÊNCIA:** {mes_referencia}

                            **CONTEXTO ADICIONAL:**
                            {contexto_briefings if contexto_briefings else "Nenhum contexto adicional fornecido."}

                            Gere um briefing completo baseado APENAS nesta pauta específica.
                            Use a base de conhecimento fornecida para identificar produtos, culturas e informações técnicas.
                            Formato completo com contexto, objetivos e formatos.
                            """

                            # Gerar o briefing
                            resposta = modelo_texto.generate_content(prompt_briefing)
                            briefing_gerado = resposta.text
                            
                            # Limpar possíveis markdown
                            briefing_limpo = briefing_gerado.strip()
                            if '```' in briefing_limpo:
                                briefing_limpo = briefing_limpo.replace('```', '')
                            
                            # Armazenar briefing
                            briefings_gerados.append({
                                'indice': idx + 1,
                                'conteudo_original': pauta['conteudo'],
                                'briefing': briefing_limpo
                            })
                            
                        except Exception as e:
                            st.error(f"❌ Erro ao gerar briefing para pauta {idx+1}: {str(e)}")
                            briefings_gerados.append({
                                'indice': idx + 1,
                                'conteudo_original': pauta['conteudo'],
                                'briefing': f"ERRO: Não foi possível gerar o briefing.\n{str(e)}"
                            })
                    
                    # Limpar barra de progresso
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Salvar briefings na session_state
                    st.session_state.briefings_gerados = briefings_gerados
                    st.success(f"✅ {len(briefings_gerados)} briefings gerados com sucesso!")
                    
                except Exception as e:
                    st.error(f"❌ Erro ao processar calendário: {str(e)}")

        # MOSTRAR BRIEFINGS GERADOS (sempre que existirem na session_state)
        if st.session_state.briefings_gerados:
            st.markdown("---")
            st.subheader("📄 Briefings Gerados")
            
            briefings_gerados = st.session_state.briefings_gerados
            
            # Abas para organizar os briefings
            tab_individual, tab_lote = st.tabs(["📄 Briefings Individuais", "📦 Download em Lote"])
            
            with tab_individual:
                st.write(f"**Total de briefings gerados:** {len(briefings_gerados)}")
                
                for briefing in briefings_gerados:
                    with st.expander(f"📋 Briefing {briefing['indice']}: {briefing['conteudo_original'][:60]}...", expanded=False):
                        st.write(f"**Pauta original:** {briefing['conteudo_original']}")
                        st.text_area(f"Conteúdo do Briefing {briefing['indice']}", 
                                   briefing['briefing'], 
                                   height=300, 
                                   key=f"briefing_{briefing['indice']}")
                        
                        # Botões de ação para cada briefing
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            nome_arquivo = f"briefing_{briefing['indice']}.txt"
                            st.download_button(
                                f"💾 Baixar Briefing {briefing['indice']}",
                                data=briefing['briefing'],
                                file_name=nome_arquivo,
                                mime="text/plain",
                                key=f"dl_single_{briefing['indice']}"
                            )
            
            with tab_lote:
                st.subheader("📦 Download em Lote")
                
                # Criar ZIP sem usar with statement para evitar fechamento prematuro
                import zipfile
                import io
                
                # Criar o buffer e o arquivo ZIP
                zip_buffer = io.BytesIO()
                zip_file = zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED)
                
                try:
                    # Adicionar briefings individuais
                    for briefing in briefings_gerados:
                        nome_arquivo = f"briefing_{briefing['indice']}.txt"
                        zip_file.writestr(nome_arquivo, briefing['briefing'])
                    
                    # Criar arquivo consolidado
                    consolidado = f"BRIEFINGS - {mes_referencia}\n"
                    consolidado += f"Total de briefings: {len(briefings_gerados)}\n"
                    consolidado += "="*60 + "\n\n"
                    
                    for briefing in briefings_gerados:
                        consolidado += f"BRIEFING {briefing['indice']}\n"
                        consolidado += f"Pauta: {briefing['conteudo_original']}\n"
                        consolidado += "-"*40 + "\n"
                        consolidado += f"{briefing['briefing']}\n\n"
                        consolidado += "="*60 + "\n\n"
                    
                    # Adicionar arquivo consolidado
                    zip_file.writestr(f"briefings_consolidados_{mes_referencia.replace(' ', '_').lower()}.txt", consolidado)
                    
                finally:
                    # Fechar o arquivo ZIP manualmente
                    zip_file.close()
                
                # Botão de download
                st.download_button(
                    "📥 Baixar Todos os Briefings (ZIP)",
                    data=zip_buffer.getvalue(),
                    file_name=f"briefings_completos_{mes_referencia.replace(' ', '_').lower()}_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.zip",
                    mime="application/zip",
                    type="primary"
                )

# ========== ABA: REVISÃO TÉCNICA (VERSÃO COMPLETA SEM RAG) ==========
with tab_revisao_tecnica2:
    st.header("🔬 Revisão Técnica Completa")
    st.markdown("**Análise rigorosa com expertise técnica em agronomia**")
    
    texto_tecnico = st.text_area(
        "Cole o conteúdo técnico agrícola para revisão:", 
        height=300,
        placeholder="Cole aqui qualquer conteúdo agrícola que precisa ser revisado tecnicamente...",
        key="texto_tecnico_original"
    )

    # Botão para realizar revisão técnica completa
    if st.button("🔬 Realizar Revisão Técnica Completa", type="primary", key="revisao_inicial"):
        if texto_tecnico:
            with st.spinner("🔍 Analisando conteúdo com rigor técnico..."):
                try:
                    # Prompt para revisão técnica no formato específico
                    prompt_revisao = f"""
                    VOCÊ É: Um engenheiro agrônomo com ampla experiência técnica.

                    SUA TAREFA: Realizar uma revisão técnica completa do conteúdo fornecido seguindo EXATAMENTE o formato abaixo.

                    ANALISE ESTE CONTEÚDO:
                    {texto_tecnico}

                    RETORNE APENAS ESTE FORMATO EXATO:

                    ✅ O QUE ESTÁ CORRETO NO TEXTO (visão geral)
                    Antes das correções, é importante destacar que o texto está bem escrito, com boa estrutura, e a maior parte das informações está correta:
                    [Liste aqui os pontos que estão corretos em bullet points]
                    Ou seja: o conteúdo é bom, faltando apenas alguns ajustes e correções pontuais.

                    ❗ PONTOS INCORRETOS, IMPRECISOS OU QUE PRECISAM SER AJUSTADOS
                    Abaixo, estão todos os erros e imprecisões técnicas do texto, com explicação e sugestão.

                    ❌ 1. [Título do primeiro erro]
                    No trecho:
                    "[Citação exata do trecho problemático]"
                    Correção técnica:
                    [Explicação detalhada do erro]
                    ➡ Portanto, [conclusão técnica]
                    Como corrigir:
                    "[Sugestão de texto corrigido]"

                    ❌ 2. [Título do segundo erro]
                    No trecho:
                    "[Citação exata do trecho problemático]"
                    Correção técnica:
                    [Explicação detalhada do erro]
                    ➡ Portanto, [conclusão técnica]
                    Como corrigir:
                    "[Sugestão de texto corrigido]"

                    [Continue numerando para cada erro encontrado...]

                    🧪 CONCLUSÃO TÉCNICA
                    O texto está bem escrito e majoritariamente correto, mas contém:
                    ✔ [X] erro(s) crítico(s)
                    [Descrição dos erros críticos]
                    ✔ [Y] afirmações que precisam correção ou moderação
                    [Descrição das correções necessárias]
                    ✔ [Z] pontos que não estão errados, mas precisam maior precisão
                    [Descrição dos pontos que precisam de precisão]
                    ✔ [W] pontos incompletos (não são erros, mas faltam informações-chave)
                    [Descrição dos pontos incompletos]

                    🔧 Se quiser, posso agora:
                    - Reescrever o texto totalmente revisado e técnico, já corrigido
                    - Criar uma versão mais curta para redes sociais
                    - Criar uma versão para material comercial
                    - Montar um quadro comparativo entre técnicas/culturas
                    - Fazer uma versão para cultura específica

                    Seja direto e técnico. Mantenha o formato exato.
                    """

                    resposta = modelo_texto2.generate_content(prompt_revisao)
                    revisao_completa = resposta.text
                    
                    # Salvar no session state para uso posterior
                    st.session_state.ultima_revisao = revisao_completa
                    st.session_state.texto_original_revisao = texto_tecnico
                    
                    # Exibir resultados
                    st.success("✅ Revisão técnica completa concluída!")
                    st.markdown(revisao_completa)
                    
                    # Botão de download
                    st.download_button(
                        "💾 Baixar Relatório Completo",
                        data=revisao_completa,
                        file_name=f"revisao_tecnica_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                        mime="text/plain"
                    )
                
                except Exception as e:
                    st.error(f"❌ Erro na revisão técnica: {str(e)}")
        else:
            st.warning("Por favor, cole um conteúdo técnico para revisão.")

    # Seção para ajustes incrementais (só aparece após a primeira revisão)
    if 'ultima_revisao' in st.session_state:
        st.divider()
        st.subheader("🔄 Ajustes Incrementais")
        
        st.info("Use o campo abaixo para solicitar ajustes específicos na última revisão realizada.")
        
        # Caixa de texto para comandos de ajuste
        comando_ajuste = st.text_area(
            "Comandos para ajustar a última revisão:",
            height=150,
            placeholder="Exemplos:\n- Foque mais na adubação nitrogenada\n- Adicione informações sobre irrigação\n- Corrija os termos técnicos sobre pragas\n- Simplifique a linguagem para produtores\n- Inclua recomendações para clima tropical",
            key="comando_ajuste"
        )
        
        # Botão para revisar novamente com base nos ajustes
        if st.button("🔄 Revisar Novamente com Ajustes", type="secondary"):
            if comando_ajuste:
                with st.spinner("🔄 Aplicando ajustes solicitados..."):
                    try:
                        # Prompt para revisão com ajustes
                        prompt_ajuste = f"""
                        VOCÊ É: Um engenheiro agrônomo com ampla experiência técnica.

                        SUA TAREFA: Revisar e ajustar o relatório técnico anterior com base nas solicitações específicas do usuário.

                        RELATÓRIO TÉCNICO ANTERIOR:
                        {st.session_state.ultima_revisao}

                        TEXTO ORIGINAL ANALISADO:
                        {st.session_state.texto_original_revisao}

                        SOLICITAÇÕES DE AJUSTE DO USUÁRIO:
                        {comando_ajuste}

                        INSTRUÇÕES:
                        1. Mantenha o MESMO FORMATO EXATO do relatório anterior
                        2. Aplique TODOS os ajustes solicitados pelo usuário
                        3. Mantenha a qualidade técnica e rigor científico
                        4. Se o ajuste solicitar foco em algum aspecto específico, dê mais ênfase a esse tópico
                        5. Se o ajuste pedir adição de informações, inclua-as de forma coerente
                        6. Se o ajuste for sobre estilo ou linguagem, adapte conforme solicitado

                        RETORNE APENAS O RELATÓRIO REVISADO NO MESMO FORMATO, SEM COMENTÁRIOS ADICIONAIS.
                        """

                        resposta_ajuste = modelo_texto2.generate_content(prompt_ajuste)
                        revisao_ajustada = resposta_ajuste.text
                        
                        # Atualizar o session state com a nova versão
                        st.session_state.ultima_revisao = revisao_ajustada
                        
                        # Exibir resultados
                        st.success("✅ Revisão ajustada concluída!")
                        st.markdown(revisao_ajustada)
                        
                        # Botão de download da versão ajustada
                        st.download_button(
                            "💾 Baixar Relatório Ajustado",
                            data=revisao_ajustada,
                            file_name=f"revisao_ajustada_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                            mime="text/plain",
                            key="download_ajustado"
                        )
                    
                    except Exception as e:
                        st.error(f"❌ Erro ao aplicar ajustes: {str(e)}")
            else:
                st.warning("Por favor, digite os comandos de ajuste desejados.")
        
        # Mostrar a última revisão salva se existir
        with st.expander("📋 Última Revisão Salva (para referência)"):
            st.markdown(st.session_state.ultima_revisao)
            
# --- Estilização ---
st.markdown("""
<style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    [data-testid="stChatMessageContent"] {
        font-size: 1rem;
    }
    div[data-testid="stTabs"] {
        margin-top: -30px;
    }
    .segment-indicator {
        background-color: #f0f2f6;
        padding: 0.5rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #4CAF50;
    }
    /* Estilo para o pipeline */
    .pipeline-step {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        border-left: 5px solid #4CAF50;
    }
    .pipeline-complete {
        border-left-color: #4CAF50;
    }
    .pipeline-current {
        border-left-color: #2196F3;
    }
    .pipeline-pending {
        border-left-color: #ff9800;
    }
</style>
""", unsafe_allow_html=True)
