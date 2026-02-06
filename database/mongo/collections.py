"""
Coleções MongoDB.
Define e gerencia as coleções utilizadas na aplicação.
"""
from .connection import get_agentes_db

# Coleções globais (inicializadas sob demanda)
collection_agentes = None
collection_conversas = None


def get_collection_agentes():
    """Retorna a coleção de agentes."""
    global collection_agentes
    if collection_agentes is None:
        db = get_agentes_db()
        if db is not None:
            collection_agentes = db['agentes']
    return collection_agentes


def get_collection_conversas():
    """Retorna a coleção de conversas."""
    global collection_conversas
    if collection_conversas is None:
        db = get_agentes_db()
        if db is not None:
            collection_conversas = db['conversas']
    return collection_conversas
