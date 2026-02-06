"""
Módulo de fallback com dados padrão.
Usado quando o agente não possui dados específicos na base de conhecimento.
"""

from .dados_safra import (
    INFO_ALGODAO,
    INFO_ARROZ,
    INFO_SOJA,
    INFO_MILHO,
    INFO_TRIGO_CANA,
    get_dados_safra_completos
)

__all__ = [
    'INFO_ALGODAO',
    'INFO_ARROZ',
    'INFO_SOJA',
    'INFO_MILHO',
    'INFO_TRIGO_CANA',
    'get_dados_safra_completos'
]
