"""
Cliente para AstraDB (banco de dados vetorial).
Gerencia buscas por similaridade vetorial.
"""
import requests
from typing import List, Dict
import streamlit as st

from config.settings import (
    ASTRA_DB_API_ENDPOINT,
    ASTRA_DB_APPLICATION_TOKEN,
    ASTRA_DB_NAMESPACE
)


class AstraDBClient:
    """Cliente para interação com AstraDB."""

    def __init__(self):
        self.base_url = f"{ASTRA_DB_API_ENDPOINT}/api/json/v1/{ASTRA_DB_NAMESPACE}"
        self.headers = {
            "Content-Type": "application/json",
            "x-cassandra-token": ASTRA_DB_APPLICATION_TOKEN,
            "Accept": "application/json"
        }

    def vector_search(self, collection: str, vector: List[float], limit: int = 6) -> List[Dict]:
        """
        Realiza busca por similaridade vetorial.

        Args:
            collection: Nome da coleção no AstraDB
            vector: Vetor de embedding para busca
            limit: Número máximo de resultados

        Returns:
            Lista de documentos similares
        """
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


# Instância global do cliente AstraDB
astra_client = AstraDBClient()
