import requests
import streamlit as st
from pymongo import MongoClient
from typing import List, Dict
from config.settings import (
    ASTRA_DB_API_ENDPOINT,
    ASTRA_DB_APPLICATION_TOKEN,
    ASTRA_DB_NAMESPACE,
    MONGO_URI,
)


class AstraDBClient:
    def __init__(self):
        self.base_url = f"{ASTRA_DB_API_ENDPOINT}/api/json/v1/{ASTRA_DB_NAMESPACE}"
        self.headers = {
            "Content-Type": "application/json",
            "x-cassandra-token": ASTRA_DB_APPLICATION_TOKEN,
            "Accept": "application/json",
        }

    def vector_search(self, collection: str, vector: List[float], limit: int = 6) -> List[Dict]:
        url = f"{self.base_url}/{collection}"
        payload = {
            "find": {
                "sort": {"$vector": vector},
                "options": {"limit": limit},
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

astra_client = AstraDBClient()


def get_mongo_client():
    return MongoClient(MONGO_URI)


def get_agentes_db():
    client = get_mongo_client()
    db = client['agentes_personalizados']
    return client, db, db['agentes'], db['conversas']


def get_briefings_db():
    client = get_mongo_client()
    db = client['briefings_Broto_Tecnologia']
    return client, db, db['briefings']


def get_blog_rag_db():
    client = get_mongo_client()
    db = client['blog_rag_tecnico']
    return client, db, db['posts_rag'], db['versoes_ajustes']
