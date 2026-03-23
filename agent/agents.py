import datetime
import streamlit as st
from bson import ObjectId
from auth.auth import get_current_user


_collection_conversas = None


def init_collections(collection_agentes, collection_conversas):
    global _collection_agentes, _collection_conversas
    _collection_agentes = collection_agentes
    _collection_conversas = collection_conversas



def criar_agente(nome, system_prompt, base_conhecimento, comments, planejamento, categoria, agente_mae_id=None, herdar_elementos=None):
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
        "criado_por": get_current_user(),
    }
    result = _collection_agentes.insert_one(agente)
    return result.inserted_id


def listar_agentes():
    current_user = get_current_user()
    if current_user == "admin":
        return list(_collection_agentes.find({"ativo": True}).sort("data_criacao", -1))
    return list(_collection_agentes.find({
        "ativo": True,
        "criado_por": current_user,
    }).sort("data_criacao", -1))


def listar_agentes_para_heranca(agente_atual_id=None):
    current_user = get_current_user()
    query = {"ativo": True}

    if current_user != "admin":
        query["criado_por"] = current_user

    if agente_atual_id:
        if isinstance(agente_atual_id, str):
            agente_atual_id = ObjectId(agente_atual_id)
        query["_id"] = {"$ne": agente_atual_id}

    return list(_collection_agentes.find(query).sort("data_criacao", -1))


def obter_agente(agente_id):
    if isinstance(agente_id, str):
        agente_id = ObjectId(agente_id)

    agente = _collection_agentes.find_one({"_id": agente_id})

    if agente and agente.get('ativo', True):
        current_user = get_current_user()
        if current_user == "admin" or agente.get('criado_por') == current_user:
            return agente

    return None


def atualizar_agente(agente_id, nome, system_prompt, base_conhecimento, comments, planejamento, categoria, agente_mae_id=None, herdar_elementos=None):
    if isinstance(agente_id, str):
        agente_id = ObjectId(agente_id)

    agente_existente = obter_agente(agente_id)
    if not agente_existente:
        raise PermissionError("Agente não encontrado ou sem permissão de edição")

    return _collection_agentes.update_one(
        {"_id": agente_id},
        {"$set": {
            "nome": nome,
            "system_prompt": system_prompt,
            "base_conhecimento": base_conhecimento,
            "comments": comments,
            "planejamento": planejamento,
            "categoria": categoria,
            "agente_mae_id": agente_mae_id,
            "herdar_elementos": herdar_elementos or [],
        }}
    )


def desativar_agente(agente_id):
    if isinstance(agente_id, str):
        agente_id = ObjectId(agente_id)

    agente_existente = obter_agente(agente_id)
    if not agente_existente:
        raise PermissionError("Agente não encontrado ou sem permissão para desativar")

    return _collection_agentes.update_one(
        {"_id": agente_id},
        {"$set": {"ativo": False}}
    )


def obter_agente_com_heranca(agente_id):
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
    if isinstance(agente_id, str):
        agente_id = ObjectId(agente_id)
    conversa = {
        "agente_id": agente_id,
        "mensagens": mensagens,
        "segmentos_utilizados": segmentos_utilizados,
        "data_criacao": datetime.datetime.now(),
    }
    return _collection_conversas.insert_one(conversa)


def obter_conversas(agente_id, limite=10):
    if isinstance(agente_id, str):
        agente_id = ObjectId(agente_id)
    return list(_collection_conversas.find(
        {"agente_id": agente_id}
    ).sort("data_criacao", -1).limit(limite))
