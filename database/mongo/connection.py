"""
Conexões MongoDB.
Gerencia todas as conexões com o banco de dados MongoDB.
"""
from pymongo import MongoClient
from config.settings import MONGO_URI

# =============================================================================
# CONEXÕES MONGODB (Singleton Pattern)
# =============================================================================
_mongo_client = None
_db_agentes = None
_db_briefings = None
_db_blog = None


def get_mongo_client():
    """Retorna o cliente MongoDB (singleton)."""
    global _mongo_client
    if _mongo_client is None and MONGO_URI:
        _mongo_client = MongoClient(MONGO_URI)
    return _mongo_client


def get_agentes_db():
    """Retorna o banco de dados de agentes."""
    global _db_agentes
    if _db_agentes is None:
        client = get_mongo_client()
        if client:
            _db_agentes = client['agentes_personalizados']
    return _db_agentes


def get_briefings_db():
    """Retorna o banco de dados de briefings."""
    global _db_briefings
    if _db_briefings is None:
        client = get_mongo_client()
        if client:
            _db_briefings = client['briefings_Broto_Tecnologia']
    return _db_briefings


def get_blog_db():
    """Retorna o banco de dados de blog posts."""
    global _db_blog
    if _db_blog is None:
        client = get_mongo_client()
        if client:
            _db_blog = client['blog_posts_agricolas']
    return _db_blog


def init_databases():
    """Inicializa todas as conexões de banco de dados."""
    get_agentes_db()
    get_briefings_db()
    get_blog_db()
    return True
