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
from google.cloud import bigquery
import openai
import pandas as pd
import csv
from perplexity import Perplexity
import openpyxl
from openpyxl.styles import Font, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from typing import Optional, List, Dict
from sentence_transformers import SentenceTransformer
import openai

def get_bigquery_client():
    """Função auxiliar para obter cliente BigQuery de forma segura"""
    try:
        from google.cloud import bigquery
        from google.oauth2 import service_account
        import json
        
        if 'bigquery_credentials' in st.secrets:
            # Usar credenciais dos secrets
            credentials_json = st.secrets['bigquery_credentials']
            credentials_info = json.loads(credentials_json)
            credentials = service_account.Credentials.from_service_account_info(credentials_info)
            
            client = bigquery.Client(
                credentials=credentials,
                project=credentials_info['project_id']
            )
            return client
        else:
            # Fallback para ADC
            client = bigquery.Client()
            return client
            
    except Exception as e:
        st.error(f"❌ Não foi possível criar cliente BigQuery: {str(e)}")
        return None
        

# Configure a API key do Perplexity
perp_api_key = st.secrets["PERP_API_KEY"]
if perp_api_key:
    perplexity_client = Perplexity(api_key=perp_api_key)
else:
    st.warning("PERP_API_KEY não encontrada. Busca web estará desativada.")
    perplexity_client = None

# Configurações das credenciais
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
ASTRA_DB_API_ENDPOINT = st.secrets.get("ASTRA_DB_API_ENDPOINT", "")
ASTRA_DB_APPLICATION_TOKEN = st.secrets.get("ASTRA_DB_APPLICATION_TOKEN", "")
ASTRA_DB_NAMESPACE = st.secrets.get("ASTRA_DB_NAMESPACE", "default_keyspace")
ASTRA_DB_COLLECTION = st.secrets.get("ASTRA_DB_COLLECTION", "documents")
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]


@st.cache_resource

def load_resource_models():
    """Carrega todos os modelos necessários incluindo BigQuery com credenciais seguras"""
    
    # 1. Configurar Gemini
    try:
        # Usar API key do secrets ou variável de ambiente
        gemini_api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            st.error("❌ GEMINI_API_KEY não encontrada nos secrets ou variáveis de ambiente")
            st.stop()
        
        genai.configure(api_key=gemini_api_key)
        gemini_model = genai.GenerativeModel('gemini-2.0-flash')
        print("✅ Gemini model inicializado")
    except Exception as e:
        st.error(f"❌ Erro ao configurar Gemini: {str(e)}")
        gemini_model = None
    
    # 2. Modelo para embeddings
    try:
        st_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ SentenceTransformer inicializado")
    except Exception as e:
        st.error(f"❌ Erro ao carregar SentenceTransformer: {str(e)}")
        st_model = None
    
    # 3. Inicializar BigQuery usando os secrets do Streamlit
    bq_client = None
    try:
        import json
        from google.oauth2 import service_account
        from google.cloud import bigquery
        
        # Verificar se temos credenciais nos secrets
        if 'bigquery_credentials' in st.secrets:
            # Carregar credenciais como JSON string
            credentials_json = st.secrets['bigquery_credentials']
            
            # Converter string para dicionário
            credentials_info = json.loads(credentials_json)
            
            # Criar credenciais
            credentials = service_account.Credentials.from_service_account_info(credentials_info)
            
            # Inicializar cliente BigQuery
            bq_client = bigquery.Client(
                credentials=credentials,
                project=credentials_info.get('project_id', 'gen-lang-client-0949885382')
            )
            
            print(f"✅ BigQuery inicializado com credenciais do projeto: {credentials.project_id}")
            
            # Testar a conexão
            try:
                # Query simples para testar
                test_query = "SELECT 1 as test"
                test_job = bq_client.query(test_query)
                test_job.result()  # Aguardar conclusão
                print("✅ Conexão com BigQuery testada com sucesso")
            except Exception as test_error:
                print(f"⚠️ Aviso ao testar BigQuery: {test_error}")
                # Continuar mesmo com erro de teste
                
        else:
            print("⚠️ bigquery_credentials não encontrado nos secrets, tentando ADC...")
            try:
                # Tentar Application Default Credentials
                bq_client = bigquery.Client()
                print("✅ BigQuery inicializado com ADC")
            except Exception as adc_error:
                print(f"❌ Falha ao usar ADC: {adc_error}")
                
    except json.JSONDecodeError as e:
        st.error(f"❌ Erro ao decodificar JSON das credenciais do BigQuery: {str(e)}")
    except ImportError as e:
        st.error(f"❌ Biblioteca do Google Cloud não instalada: {str(e)}")
        st.info("💡 Execute: pip install google-cloud-bigquery google-auth")
    except Exception as e:
        st.error(f"❌ Erro ao inicializar BigQuery: {str(e)}")
        print(f"Detalhes do erro BigQuery: {type(e).__name__}: {str(e)}")
    
    return gemini_model, st_model, bq_client


class BigQueryClient:
    """Classe wrapper para busca vetorial no BigQuery."""
    def __init__(self, client):
        self.client = client
        print("✅ BigQueryClient inicializado para busca vetorial.")
        
    def vector_search(self, colecao: str, vector: List[float], limit: int = 10) -> List[Dict]:
        """Realiza busca por similaridade vetorial na tabela nova."""
        if not colecao or colecao == "ERRO":
            return []
            
        try:
            vector_str = str(vector)
            table_id = "gen-lang-client-0949885382.teste_julia.teste_tabela"
            
            query = f"""
            SELECT 
                chunk_id,
                chunk_text,
                fonte,
                colecao,
                ML.DISTANCE(
                    CAST(embedding AS ARRAY<FLOAT64>), 
                    CAST({vector_str} AS ARRAY<FLOAT64>), 
                    'COSINE'
                ) AS similarity_score
            FROM `{table_id}`
            WHERE colecao = '{colecao}'
            ORDER BY similarity_score ASC
            LIMIT {limit}
            """
            
            query_job = self.client.query(query)
            results = query_job.result()
            
            documents = []
            for row in results:
                doc = {
                    "chunk_id": row.chunk_id,
                    "chunk_text": row.chunk_text,
                    "fonte": row.fonte,
                    "similarity_score": row.similarity_score
                }
                documents.append(doc)
            return documents


        except Exception as e:
            st.error(f"❌ ERRO na busca BigQuery: {str(e)}")
            return []

model, st_model, bigquery_client = load_resource_models()

# Verificar se o BigQuery foi inicializado corretamente
if bigquery_client is None:
    st.warning("⚠️ Conexão com BigQuery não estabelecida. Algumas funcionalidades estarão limitadas.")
else:
    print("✅ Conexão com BigQuery estabelecida com sucesso")



# -----------------------------------------------------------
# V. CLASSE LLMClient (Mantida igual)
# -----------------------------------------------------------


class LLMClient:
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model


    def generate_content(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Você é um agente de revisão técnica altamente preciso."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"ERRO NA GERAÇÃO DO LLM: {str(e)}"


modelo_texto_openai = LLMClient(api_key=OPENAI_API_KEY)


#FUNÇÕES ESPECÍFICAS DESSA ABA DE REVISÃO TÉCNICA 
def classificar_texto(texto: str) -> Optional[str]:
    prompt = f"""Analise o texto e classifique-o em: PRODUTO, CULTURA ou OUTROS.
    Texto: "{texto}"
    Retorne apenas a palavra em capslook: PRODUTO, CULTURA OU OUTROS."""


    try:
        response = modelo_texto.generate_content(prompt)
        resposta = response.text.strip().upper()
        if any(cat in resposta for cat in ["PRODUTO", "CULTURA", "OUTROS"]):
            return "PRODUTO" if "PRODUTO" in resposta else "CULTURA" if "CULTURA" in resposta else "OUTROS"
        return "OUTROS"
    except Exception:
        return "ERRO"


def get_embedding(text: str) -> List[float]:
    """Usa o SentenceTransformer carregado no cache."""
    return st_model.encode(text).tolist()


# -----------------------------------------------------------
# VII. FUNÇÕES PRINCIPAIS (RAG e Incremental)
# -----------------------------------------------------------


def reescrever_revisor(content: str, colecao_override: Optional[str] = None) -> str:
    bq_search_client = BigQueryClient(bigquery_client)
    # 1. Classificação
    if colecao_override and colecao_override != "Automática (Classificação Gemini)":
        colecao = colecao_override
    else:
        colecao = classificar_texto(content)
    
    if colecao in ["ERRO", None]:
        return "Erro na classificação da coleção."


    # 2. Busca Vetorial
    embedding = get_embedding(content[:800])
    relevant_docs = bq_search_client.vector_search(colecao, embedding, limit=5)
    
    # 3. Contexto RAG
    rag_context = ""
    if relevant_docs:
        rag_context = "### REFERENCIAL TEÓRICO BUSCADO (BigQuery) ###\n"
        for i, doc in enumerate(relevant_docs, 1):
            rag_context += f"--- Fonte: {doc['fonte']} (Similaridade: {doc['similarity_score']:.4f}) ---\n"
            rag_context += f"{doc['chunk_text']}\n\n"
    
    modelo_texto_openai = LLMClient(api_key=OPENAI_API_KEY)
    
    # 4. Prompt Final
    final_prompt = f"""
    Você é um **Revisor Técnico Sênior** com foco na área agrícola.
    CORRIGIR imprecisões e ENRIQUECER o texto com os dados do referencial.
    
    TEXTO ORIGINAL:
    {content}
    
    {rag_context}


    ## ESTRUTURA DE RETORNO:
    1. TEXTO REVISADO E CORRIGIDO
    2. 🛠️ Ajustes Técnicos e Correções (lista de alterações e fontes usadas)
    3. Você deve dizer todas as fontes utilizadas 
    """
    
    return modelo_texto_openai.generate_content(final_prompt)


def ajuste_incremental(texto_revisado: str, instrucao_incremental: str) -> str:
    if not instrucao_incremental: return texto_revisado
    
    partes = texto_revisado.split("🛠️ Ajustes Técnicos e Correções")
    texto_principal = partes[0].strip()
    
    prompt = f"Aplique esta mudança: {instrucao_incremental}\n\nTEXTO: {texto_principal}"
    return modelo_texto.generate_content(prompt)



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

client = MongoClient("mongodb+srv://gustavoromao3345:RqWFPNOJQfInAW1N@cluster0.5iilj.mongodb.net/auto_doc?retryWrites=true&w=majority&ssl=true&ssl_cert_reqs=CERT_NONE&tlsAllowInvalidCertificates=true")
db = client['agentes_personalizados']
collection_agentes = db['agentes']
collection_conversas = db['conversas']

# Configuração da API do Gemini
gemini_api_key = st.secrets["GEMINI_API_KEY"]  # Mude de os.getenv para st.secrets
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
    - Destaque para termos técnicos-chave e nomes de produto
