import os
from dotenv import load_dotenv

load_dotenv()

PERP_API_KEY = os.getenv("PERP_API_KEY")
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
ASTRA_DB_API_ENDPOINT = os.getenv('ASTRA_DB_API_ENDPOINT')
ASTRA_DB_APPLICATION_TOKEN = os.getenv('ASTRA_DB_APPLICATION_TOKEN')
ASTRA_DB_NAMESPACE = os.getenv('ASTRA_DB_NAMESPACE')
ASTRA_DB_COLLECTION = os.getenv('ASTRA_DB_COLLECTION')
MONGO_URI = os.getenv('MONGO_URI')
GEMINI_API_KEY = os.getenv("GEM_API_KEY")


SENHA_ADMIN = os.getenv('SENHA_ADMIN')
SENHA_SYN = os.getenv('SENHA_SYN')
SENHA_SME = os.getenv('SENHA_SME')
SENHA_ENT = os.getenv('SENHA_ENT')
