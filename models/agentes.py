import datetime
from bson import ObjectId
from database.mongo import get_collection_agentes, get_collection_conversas
from auth import get_current_user


def criar_agente(nome, system_prompt, base_conhecimento, comments, planejamento,
                 categoria, agente_mae_id=None, herdar_elementos=None):
    """
    Cria um novo agente no MongoDB.

    Args:
        nome: Nome do agente
        system_prompt: Prompt do sistema
        base_conhecimento: Base de conhecimento/brand guidelines
        comments: Comentários do cliente
        planejamento: Planejamento estratégico
        categoria: Categoria (Social, SEO, Conteúdo)
        agente_mae_id: ID do agente mãe para herança
        herdar_elementos: Lista de elementos a herdar

    Returns:
        ID do agente criado
    """
    collection = get_collection_agentes()
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
        "criado_por": get_current_user()
    }
    result = collection.insert_one(agente)
    return result.inserted_id


def listar_agentes():
    """
    Retorna todos os agentes ativos do usuário atual ou todos se admin.

    Returns:
        Lista de agentes
    """
    collection = get_collection_agentes()
    current_user = get_current_user()

    if current_user == "admin":
        return list(collection.find({"ativo": True}).sort("data_criacao", -1))
    else:
        return list(collection.find({
            "ativo": True,
            "criado_por": current_user
        }).sort("data_criacao", -1))


def listar_agentes_para_heranca(agente_atual_id=None):
    """
    Retorna todos os agentes ativos que podem ser usados como mãe.

    Args:
        agente_atual_id: ID do agente atual (será excluído da lista)

    Returns:
        Lista de agentes disponíveis para herança
    """
    collection = get_collection_agentes()
    current_user = get_current_user()
    query = {"ativo": True}

    # Filtro por usuário (admin vê todos, outros só os seus)
    if current_user != "admin":
        query["criado_por"] = current_user

    if agente_atual_id:
        # Excluir o próprio agente da lista para evitar auto-herança
        if isinstance(agente_atual_id, str):
            agente_atual_id = ObjectId(agente_atual_id)
        query["_id"] = {"$ne": agente_atual_id}

    return list(collection.find(query).sort("data_criacao", -1))


def obter_agente(agente_id):
    """
    Obtém um agente específico pelo ID com verificação de permissão.

    Args:
        agente_id: ID do agente

    Returns:
        Documento do agente ou None
    """
    collection = get_collection_agentes()
    if isinstance(agente_id, str):
        agente_id = ObjectId(agente_id)

    agente = collection.find_one({"_id": agente_id})

    # Verificar permissão
    if agente and agente.get('ativo', True):
        current_user = get_current_user()
        if current_user == "admin" or agente.get('criado_por') == current_user:
            return agente

    return None


def atualizar_agente(agente_id, nome, system_prompt, base_conhecimento, comments,
                     planejamento, categoria, agente_mae_id=None, herdar_elementos=None):
    """
    Atualiza um agente existente com verificação de permissão.

    Args:
        agente_id: ID do agente
        nome: Novo nome
        system_prompt: Novo prompt
        base_conhecimento: Nova base de conhecimento
        comments: Novos comentários
        planejamento: Novo planejamento
        categoria: Nova categoria
        agente_mae_id: Novo ID do agente mãe
        herdar_elementos: Novos elementos a herdar

    Returns:
        Resultado da operação de update
    """
    collection = get_collection_agentes()
    if isinstance(agente_id, str):
        agente_id = ObjectId(agente_id)

    # Verificar permissão
    agente_existente = obter_agente(agente_id)
    if not agente_existente:
        raise PermissionError("Agente não encontrado ou sem permissão de edição")

    return collection.update_one(
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
    """
    Desativa um agente (soft delete) com verificação de permissão.

    Args:
        agente_id: ID do agente

    Returns:
        Resultado da operação
    """
    collection = get_collection_agentes()
    if isinstance(agente_id, str):
        agente_id = ObjectId(agente_id)

    # Verificar permissão
    agente_existente = obter_agente(agente_id)
    if not agente_existente:
        raise PermissionError("Agente não encontrado ou sem permissão para desativar")

    return collection.update_one(
        {"_id": agente_id},
        {"$set": {"ativo": False}}
    )


def obter_agente_com_heranca(agente_id):
    """
    Obtém um agente com os elementos herdados aplicados.

    Args:
        agente_id: ID do agente

    Returns:
        Documento do agente com herança aplicada
    """
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
    """
    Salva uma conversa no histórico.

    Args:
        agente_id: ID do agente
        mensagens: Lista de mensagens
        segmentos_utilizados: Segmentos do agente usados

    Returns:
        Resultado da inserção
    """
    collection = get_collection_conversas()
    if isinstance(agente_id, str):
        agente_id = ObjectId(agente_id)

    conversa = {
        "agente_id": agente_id,
        "mensagens": mensagens,
        "segmentos_utilizados": segmentos_utilizados,
        "data_criacao": datetime.datetime.now()
    }
    return collection.insert_one(conversa)


def obter_conversas(agente_id, limite=10):
    """
    Obtém o histórico de conversas de um agente.

    Args:
        agente_id: ID do agente
        limite: Número máximo de conversas

    Returns:
        Lista de conversas
    """
    collection = get_collection_conversas()
    if isinstance(agente_id, str):
        agente_id = ObjectId(agente_id)

    return list(collection.find(
        {"agente_id": agente_id}
    ).sort("data_criacao", -1).limit(limite))


def construir_contexto(agente, segmentos_selecionados, historico_mensagens=None):
    """
    Constrói o contexto com base nos segmentos selecionados.

    Args:
        agente: Documento do agente
        segmentos_selecionados: Lista de segmentos a incluir
        historico_mensagens: Histórico de mensagens do chat

    Returns:
        String com o contexto completo
    """
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
