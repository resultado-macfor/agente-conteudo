"""
Configurações e credenciais da aplicação.
Todas as variáveis de ambiente são centralizadas aqui.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

ENV_PATH = Path(__file__).parent / ".env"
load_dotenv(ENV_PATH)

# API KEYS
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PERP_API_KEY = os.getenv("PERP_API_KEY")
GEMINI_API_KEY = os.getenv("GEM_API_KEY")

# MONGODB
MONGO_URI = os.getenv('MONGO_URI')


# ASTRA DB (Vector Database)
ASTRA_DB_API_ENDPOINT = os.getenv('ASTRA_DB_API_ENDPOINT')
ASTRA_DB_APPLICATION_TOKEN = os.getenv('ASTRA_DB_APPLICATION_TOKEN')
ASTRA_DB_NAMESPACE = os.getenv('ASTRA_DB_NAMESPACE')
ASTRA_DB_COLLECTION = os.getenv('ASTRA_DB_COLLECTION')

# SENHAS DE USUÁRIOS
SENHA_ADMIN = os.getenv('SENHA_ADMIN')
SENHA_SYN = os.getenv('SENHA_SYN')
SENHA_SME = os.getenv('SENHA_SME')
SENHA_ENT = os.getenv('SENHA_ENT')


# CONFIGURAÇÕES DE MODELOS
MODELO_VISION = "gemini-2.5-flash"
MODELO_TEXTO = "gemini-2.5-flash"
MODELO_TEXTO_PRO = "gemini-2.5-pro"
MODELO_TRANSCRICAO = "gemini-2.0-flash"
MODELO_EMBEDDING = "text-embedding-3-small"
