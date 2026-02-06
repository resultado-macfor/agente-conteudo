"""
Dados de safras agrícolas por estado (fallback).
Usado quando o agente não possui esses dados na base de conhecimento.
"""

INFO_ALGODAO = """
Tocantins: Plantio de novembro (2ª quinzena) até fevereiro (2ª quinzena), com pico intenso em janeiro. Colheita de abril (2ª quinzena) até agosto (1ª quinzena), com pico intenso em junho e julho.
Maranhão: Plantio de dezembro (1ª quinzena) até março (2ª quinzena), com pico intenso em janeiro. Colheita de maio (2ª quinzena) até agosto (2ª quinzena), com pico intenso em junho e julho.
Piauí: Plantio de dezembro (2ª quinzena) até março (2ª quinzena), com pico intenso em janeiro. Colheita de maio (2ª quinzena) até agosto (1ª quinzena), com pico intenso em junho e julho.
Ceará: Plantio de janeiro (1ª quinzena) até maio (1ª quinzena), com pico intenso em fevereiro e março. Colheita de junho (1ª quinzena) até outubro (2ª quinzena), com pico intenso em junho, julho e agosto.
Rio Grande do Norte: Plantio de janeiro (1ª quinzena) até abril (2ª quinzena), com pico intenso em fevereiro e março. Colheita de julho (1ª quinzena) até novembro (2ª quinzena), com pico intenso em agosto e setembro.
Paraíba: Plantio de fevereiro (1ª quinzena) até maio (1ª quinzena), com pico intenso em março. Colheita de agosto (1ª quinzena) até novembro (2ª quinzena), com pico intenso em agosto e setembro.
Pernambuco: Plantio de janeiro (1ª quinzena) até junho (2ª quinzena), com pico intenso em março. Colheita de agosto (1ª quinzena) até dezembro (1ª quinzena), com pico intenso em agosto e setembro.
Alagoas: Plantio de maio (2ª quinzena) até agosto (2ª quinzena), com pico intenso em junho. Colheita de outubro (2ª quinzena) até janeiro (2ª quinzena), com pico intenso em novembro e dezembro.
Bahia: Plantio de novembro (2ª quinzena) até fevereiro (1ª quinzena), com pico intenso em dezembro. Colheita de abril (2ª quinzena) até setembro (1ª quinzena), com pico intenso em maio e junho.
Mato Grosso: Plantio de dezembro (1ª quinzena) até fevereiro (2ª quinzena), com pico intenso em janeiro. Colheita de abril (2ª quinzena) até agosto (2ª quinzena), com pico intenso em junho.
Mato Grosso do Sul: Plantio de outubro (1ª quinzena) até janeiro (1ª quinzena), com pico intenso em novembro e dezembro. Colheita de março (2ª quinzena) até junho (1ª quinzena), com pico intenso em abril.
Goiás: Plantio de outubro (2ª quinzena) até janeiro (1ª quinzena), com pico intenso em novembro e dezembro. Colheita de março (2ª quinzena) até junho (2ª quinzena), com pico intenso em maio.
Distrito Federal: Plantio de outubro (2ª quinzena) até janeiro (1ª quinzena), com pico intenso em novembro e dezembro. Colheita de abril (1ª quinzena) até junho (2ª quinzena), com pico intenso em maio.
Minas Gerais: Plantio de outubro (1ª quinzena) até janeiro (1ª quinzena), com pico intenso em novembro e dezembro. Colheita de março (2ª quinzena) até junho (1ª quinzena), com pico intenso em abril e maio.
São Paulo: Plantio de outubro (1ª quinzena) até janeiro (1ª quinzena), com pico intenso em novembro e dezembro. Colheita de março (1ª quinzena) até junho (1ª quinzena), com pico intenso em abril e maio.
Paraná: Plantio de setembro (2ª quinzena) até dezembro (1ª quinzena), com pico intenso em outubro e novembro. Colheita de março (1ª quinzena) até maio (2ª quinzena), com pico intenso em abril.
"""

INFO_ARROZ = """
Roraima: Plantio de maio (1ª quinzena) até agosto (2ª quinzena), com pico intenso em maio. Colheita de julho (2ª quinzena) até novembro (2ª quinzena), com pico intenso em setembro.
Rondônia: Plantio de setembro (2ª quinzena) até janeiro (1ª quinzena), com pico intenso em novembro e dezembro. Colheita de janeiro (1ª quinzena) até maio (1ª quinzena), com pico intenso em fevereiro e março.
Acre: Plantio de setembro (2ª quinzena) até janeiro (1ª quinzena), com pico intenso em novembro e dezembro. Colheita de janeiro (1ª quinzena) até maio (1ª quinzena), com pico intenso em fevereiro e março.
Amazonas: Plantio de setembro (2ª quinzena) até janeiro (1ª quinzena), com pico intenso em novembro e dezembro. Colheita de janeiro (2ª quinzena) até maio (1ª quinzena), com pico intenso em março.
Amapá: Plantio de janeiro (1ª quinzena) até abril (1ª quinzena), com pico intenso em fevereiro. Colheita de maio (2ª quinzena) até agosto (2ª quinzena), com pico intenso em junho.
Pará: Plantio de dezembro (1ª quinzena) até abril (2ª quinzena), com pico intenso em janeiro. Colheita de abril (1ª quinzena) até agosto (2ª quinzena), com pico intenso em abril e maio.
Tocantins: Plantio de outubro (2ª quinzena) até janeiro (2ª quinzena), com pico intenso em novembro e dezembro. Colheita de fevereiro (1ª quinzena) até maio (1ª quinzena), com pico intenso em março e abril.
Maranhão: Plantio de novembro (1ª quinzena) até março (1ª quinzena), com pico intenso em janeiro. Colheita de abril (1ª quinzena) até julho (1ª quinzena), com pico intenso em abril e maio.
Piauí: Plantio de novembro (1ª quinzena) até março (1ª quinzena), com pico intenso em janeiro. Colheita de abril (1ª quinzena) até julho (1ª quinzena), com pico intenso em abril e maio.
Ceará: Plantio de janeiro (1ª quinzena) até abril (2ª quinzena), com pico intenso em janeiro e fevereiro. Colheita de maio (1ª quinzena) até julho (1ª quinzena), com pico intenso em maio e junho.
Rio Grande do Norte: Plantio de janeiro (2ª quinzena) até maio (1ª quinzena), com pico intenso em março. Colheita de junho (1ª quinzena) até outubro (1ª quinzena), com pico intenso em agosto.
Paraíba: Plantio de janeiro (1ª quinzena) até abril (1ª quinzena), com pico intenso em janeiro e fevereiro. Colheita de maio (2ª quinzena) até agosto (1ª quinzena), com pico intenso em junho.
Pernambuco: Plantio de janeiro (1ª quinzena) até abril (1ª quinzena), com pico intenso em fevereiro. Colheita de maio (1ª quinzena) até agosto (1ª quinzena), com pico intenso em junho.
Alagoas: Plantio de setembro (2ª quinzena) até dezembro (1ª quinzena), com pico intenso em outubro e novembro. Colheita de janeiro (1ª quinzena) até março (2ª quinzena), com pico intenso em fevereiro e março.
Sergipe: Plantio de setembro (2ª quinzena) até novembro (2ª quinzena), com pico intenso em outubro. Colheita de janeiro (1ª quinzena) até março (2ª quinzena), com pico intenso em fevereiro.
Bahia: Plantio de setembro (2ª quinzena) até dezembro (1ª quinzena), com pico intenso em outubro e novembro. Colheita de janeiro (1ª quinzena) até abril (1ª quinzena), com pico intenso em fevereiro e março.
Mato Grosso: Plantio de setembro (1ª quinzena) até janeiro (1ª quinzena), com pico intenso em novembro e dezembro. Colheita de janeiro (1ª quinzena) até maio (1ª quinzena), com pico intenso em março.
Mato Grosso do Sul: Plantio de setembro (1ª quinzena) até dezembro (2ª quinzena), com pico intenso em outubro e novembro. Colheita de janeiro (1ª quinzena) até abril (2ª quinzena), com pico intenso em fevereiro.
Goiás: Plantio de outubro (1ª quinzena) até janeiro (1ª quinzena), com pico intenso em novembro e dezembro. Colheita de fevereiro (1ª quinzena) até abril (2ª quinzena), com pico intenso em março.
Distrito Federal: Plantio de outubro (1ª quinzena) até janeiro (1ª quinzena), com pico intenso em novembro e dezembro. Colheita de fevereiro (1ª quinzena) até abril (2ª quinzena), com pico intenso em março.
Minas Gerais: Plantio de outubro (1ª quinzena) até janeiro (1ª quinzena), com pico intenso em novembro e dezembro. Colheita de fevereiro (1ª quinzena) até maio (1ª quinzena), com pico intenso em março e abril.
Espírito Santo: Plantio de outubro (1ª quinzena) até janeiro (1ª quinzena), com pico intenso em novembro e dezembro. Colheita de fevereiro (1ª quinzena) até maio (1ª quinzena), com pico intenso em março e abril.
Rio de Janeiro: Plantio de outubro (1ª quinzena) até janeiro (1ª quinzena), com pico intenso em novembro e dezembro. Colheita de janeiro (1ª quinzena) até maio (2ª quinzena), com pico intenso em março e abril.
São Paulo: Plantio de outubro (1ª quinzena) até janeiro (1ª quinzena), com pico intenso em novembro e dezembro. Colheita de fevereiro (1ª quinzena) até maio (1ª quinzena), com pico intenso em março e abril.
Paraná: Plantio de setembro (1ª quinzena) até janeiro (1ª quinzena), com pico intenso em novembro e dezembro. Colheita de janeiro (1ª quinzena) até maio (1ª quinzena), com pico intenso em março e abril.
Santa Catarina: Plantio de agosto (2ª quinzena) até dezembro (1ª quinzena), com pico intenso em outubro e novembro. Colheita de janeiro (1ª quinzena) até abril (2ª quinzena), com pico intenso em fevereiro e março.
Rio Grande do Sul: Plantio de setembro (1ª quinzena) até dezembro (1ª quinzena), com pico intenso em outubro e novembro. Colheita de fevereiro (1ª quinzena) até maio (1ª quinzena), com pico intenso em março e abril.
"""

INFO_SOJA = """
Roraima: Plantio de abril (2ª quinzena) até junho (2ª quinzena), com pico intenso em maio. Colheita de julho (2ª quinzena) até novembro (2ª quinzena), com pico intenso em setembro.
Rondônia: Plantio de setembro (2ª quinzena) até janeiro (1ª quinzena), com pico intenso em novembro. Colheita de janeiro (1ª quinzena) até abril (1ª quinzena), com pico intenso em março.
Amazonas: Plantio de setembro (1ª quinzena) até dezembro (1ª quinzena), com pico intenso em setembro e outubro. Colheita de dezembro (2ª quinzena) até março (2ª quinzena), com pico intenso em fevereiro.
Pará: Plantio de outubro (1ª quinzena) até janeiro (2ª quinzena), com pico intenso em março. Colheita de fevereiro (2ª quinzena) até agosto (2ª quinzena), com pico intenso em março e julho.
Tocantins: Plantio de outubro (1ª quinzena) até janeiro (1ª quinzena), com pico intenso em novembro e dezembro. Colheita de fevereiro (1ª quinzena) até maio (1ª quinzena), com pico intenso em março e abril.
Maranhão: Plantio de outubro (1ª quinzena) até janeiro (2ª quinzena), com pico intenso em novembro e dezembro. Colheita de fevereiro (2ª quinzena) até maio (2ª quinzena), com pico intenso em março e abril.
Piauí: Plantio de outubro (1ª quinzena) até janeiro (1ª quinzena), com pico intenso em novembro e dezembro. Colheita de março (1ª quinzena) até maio (2ª quinzena), com pico intenso em abril.
Bahia: Plantio de outubro (1ª quinzena) até janeiro (2ª quinzena), com pico intenso em novembro e dezembro. Colheita de fevereiro (2ª quinzena) até maio (1ª quinzena), com pico intenso em março e abril.
Mato Grosso: Plantio de setembro (1ª quinzena) até dezembro (2ª quinzena), com pico intenso em outubro e novembro. Colheita de janeiro (1ª quinzena) até abril (1ª quinzena), com pico intenso em fevereiro e março.
Mato Grosso do Sul: Plantio de setembro (2ª quinzena) até dezembro (2ª quinzena), com pico intenso em outubro e novembro. Colheita de janeiro (2ª quinzena) até abril (1ª quinzena), com pico intenso em março.
Goiás: Plantio de setembro (2ª quinzena) até janeiro (1ª quinzena), com pico intenso em outubro e novembro. Colheita de janeiro (2ª quinzena) até abril (1ª quinzena), com pico intenso em março.
Distrito Federal: Plantio de outubro (1ª quinzena) até janeiro (1ª quinzena), com pico intenso em novembro e dezembro. Colheita de fevereiro (1ª quinzena) até maio (2ª quinzena), com pico intenso em março e abril.
Minas Gerais: Plantio de outubro (1ª quinzena) até janeiro (1ª quinzena), com pico intenso em novembro e dezembro. Colheita de fevereiro (1ª quinzena) até maio (1ª quinzena), com pico intenso em março e abril.
São Paulo: Plantio de setembro (2ª quinzena) até janeiro (1ª quinzena), com pico intenso em outubro e novembro. Colheita de janeiro (1ª quinzena) até abril (2ª quinzena), com pico intenso em março.
Paraná: Plantio de setembro (2ª quinzena) até dezembro (2ª quinzena), com pico intenso em outubro e novembro. Colheita de janeiro (2ª quinzena) até abril (2ª quinzena), com pico intenso em março.
Santa Catarina: Plantio de outubro (1ª quinzena) até dezembro (2ª quinzena), com pico intenso em novembro e dezembro. Colheita de janeiro (1ª quinzena) até maio (2ª quinzena), com pico intenso em março e abril.
Rio Grande do Sul: Plantio de outubro (1ª quinzena) até janeiro (1ª quinzena), com pico intenso em novembro e dezembro. Colheita de fevereiro (1ª quinzena) até maio (1ª quinzena), com pico intenso em março e abril.
"""

INFO_MILHO = """
Calendário de Safra: Milho 1ª Safra (Ciclo 120-180 dias)
Rondônia: Plantio de agosto (2ª quinzena) até novembro (1ª quinzena), com pico intenso em setembro. Colheita de janeiro (2ª quinzena) até abril (2ª quinzena), com pico intenso em fevereiro.
Acre: Plantio de setembro (2ª quinzena) até dezembro (1ª quinzena), com pico intenso em outubro. Colheita de fevereiro (1ª quinzena) até maio (2ª quinzena), com pico intenso em março.
Amazonas: Plantio de outubro (1ª quinzena) até dezembro (2ª quinzena), com pico intenso em novembro. Colheita de março (2ª quinzena) até junho (2ª quinzena), com pico intenso em abril.
Pará: Plantio de outubro (1ª quinzena) até janeiro (1ª quinzena), com pico intenso em novembro. Colheita de março (2ª quinzena) até junho (2ª quinzena), com pico intenso em maio.
Tocantins: Plantio de outubro (1ª quinzena) até janeiro (1ª quinzena), com pico intenso em novembro e dezembro. Colheita de março (2ª quinzena) até junho (2ª quinzena), com pico intenso em maio.
Maranhão: Plantio de outubro (1ª quinzena) até janeiro (2ª quinzena), com pico intenso em novembro. Colheita de março (2ª quinzena) até junho (2ª quinzena), com pico intenso em abril.
Piauí: Plantio de outubro (1ª quinzena) até janeiro (2ª quinzena), com pico intenso em novembro e dezembro. Colheita de abril (1ª quinzena) até junho (2ª quinzena), com pico intenso em maio.
Pernambuco: Plantio de outubro (2ª quinzena) até janeiro (2ª quinzena), com pico intenso em dezembro. Colheita de abril (2ª quinzena) até junho (2ª quinzena), com pico intenso em maio.
Bahia: Plantio de outubro (1ª quinzena) até fevereiro (1ª quinzena), com pico intenso em novembro e dezembro. Colheita de março (2ª quinzena) até julho (1ª quinzena), com pico intenso em abril e maio.
Mato Grosso: Plantio de setembro (2ª quinzena) até dezembro (1ª quinzena), com pico intenso em outubro e novembro. Colheita de fevereiro (1ª quinzena) até maio (1ª quinzena), com pico intenso em março e abril.
Mato Grosso do Sul: Plantio de agosto (2ª quinzena) até novembro (2ª quinzena), com pico intenso em setembro e outubro. Colheita de janeiro (2ª quinzena) até abril (1ª quinzena), com pico intenso em março.
Goiás: Plantio de setembro (2ª quinzena) até janeiro (1ª quinzena), com pico intenso em outubro e novembro. Colheita de janeiro (2ª quinzena) até junho (1ª quinzena), com pico intenso em março e abril.
Distrito Federal: Plantio de setembro (2ª quinzena) até dezembro (2ª quinzena), com pico intenso em outubro e novembro. Colheita de fevereiro (2ª quinzena) até junho (1ª quinzena), com pico intenso em abril.
Minas Gerais: Plantio de setembro (2ª quinzena) até janeiro (1ª quinzena), com pico intenso em outubro, novembro e dezembro. Colheita de fevereiro (2ª quinzena) até junho (1ª quinzena), com pico intenso em maio.
Espírito Santo: Plantio de agosto (2ª quinzena) até dezembro (2ª quinzena), com pico intenso em setembro e outubro. Colheita de janeiro (2ª quinzena) até maio (2ª quinzena), com pico intenso em março.
Rio de Janeiro: Plantio de setembro (1ª quinzena) até dezembro (2ª quinzena), com pico intenso em outubro e novembro. Colheita de fevereiro (1ª quinzena) até junho (1ª quinzena), com pico intenso em março e abril.
São Paulo: Plantio de setembro (1ª quinzena) até dezembro (2ª quinzena), com pico intenso em outubro e novembro. Colheita de janeiro (1ª quinzena) até julho (1ª quinzena), com pico intenso em março e abril.
Paraná: Plantio de agosto (2ª quinzena) até dezembro (1ª quinzena), com pico intenso em setembro e outubro. Colheita de janeiro (2ª quinzena) até junho (2ª quinzena), com pico intenso em março.
Santa Catarina: Plantio de agosto (1ª quinzena) até dezembro (1ª quinzena), com pico intenso em setembro e outubro. Colheita de janeiro (1ª quinzena) até maio (2ª quinzena), com pico intenso em março e abril.
Rio Grande do Sul: Plantio de agosto (1ª quinzena) até novembro (2ª quinzena), com pico intenso em setembro e outubro. Colheita de dezembro (2ª quinzena) até maio (2ª quinzena), com pico intenso em fevereiro e março.

Calendário de Safra: Milho 2ª Safra (Ciclo 120-180 dias)
Roraima: Plantio de maio (1ª quinzena) até junho (2ª quinzena), com pico intenso em maio. Colheita de setembro (2ª quinzena) até novembro (2ª quinzena), com pico intenso em outubro.
Rondônia: Plantio de janeiro (2ª quinzena) até março (1ª quinzena), com pico intenso em fevereiro. Colheita de maio (2ª quinzena) até agosto (2ª quinzena), com pico intenso em julho e agosto.
Amapá: Plantio de fevereiro (1ª quinzena) até março (2ª quinzena), com pico intenso em fevereiro. Colheita de maio (2ª quinzena) até julho (1ª quinzena), com pico intenso em maio e junho.
Pará: Plantio de janeiro (1ª quinzena) até março (1ª quinzena), com pico intenso em janeiro e fevereiro. Colheita de abril (2ª quinzena) até novembro (2ª quinzena), com pico intenso em maio.
Tocantins: Plantio de janeiro (1ª quinzena) até março (2ª quinzena), com pico intenso de janeiro a março. Colheita de maio (2ª quinzena) até agosto (1ª quinzena), com pico intenso em julho.
Maranhão: Plantio de janeiro (1ª quinzena) até março (2ª quinzena), com pico intenso em janeiro e fevereiro. Colheita de maio (1ª quinzena) até agosto (2ª quinzena), com pico intenso em junho e julho.
Piauí: Plantio de janeiro (1ª quinzena) até março (1ª quinzena), com pico intenso em janeiro e fevereiro. Colheita de maio (1ª quinzena) até agosto (1ª quinzena), com pico intenso em junho e julho.
Ceará: Plantio de janeiro (1ª quinzena) até março (2ª quinzena), com pico intenso em janeiro e fevereiro. Colheita de maio (2ª quinzena) até agosto (1ª quinzena), com pico intenso em julho.
Rio Grande do Norte: Plantio de fevereiro (1ª quinzena) até março (2ª quinzena), com pico intenso em fevereiro. Colheita de julho (1ª quinzena) até setembro (2ª quinzena), com pico intenso em agosto.
Paraíba: Plantio de março (1ª quinzena) até abril (2ª quinzena), com pico intenso em março e abril. Colheita de julho (1ª quinzena) até setembro (2ª quinzena), com pico intenso em agosto.
Pernambuco: Plantio de março (1ª quinzena) até abril (2ª quinzena), com pico intenso em março e abril. Colheita de julho (1ª quinzena) até outubro (2ª quinzena), com pico intenso em agosto e setembro.
Alagoas: Plantio de abril (2ª quinzena) até junho (2ª quinzena), com pico intenso em maio. Colheita de setembro (1ª quinzena) até dezembro (1ª quinzena), com pico intenso em outubro.
Sergipe: Plantio de abril (2ª quinzena) até junho (1ª quinzena), com pico intenso em maio. Colheita de setembro (1ª quinzena) até dezembro (1ª quinzena), com pico intenso em outubro e novembro.
Bahia: Plantio de abril (2ª quinzena) até junho (1ª quinzena), com pico intenso em maio. Colheita de agosto (2ª quinzena) até novembro (2ª quinzena), com pico intenso em outubro.
Mato Grosso: Plantio de janeiro (2ª quinzena) até março (1ª quinzena), com pico intenso em fevereiro. Colheita de maio (2ª quinzena) até agosto (1ª quinzena), com pico intenso em junho e julho.
Mato Grosso do Sul: Plantio de janeiro (1ª quinzena) até março (2ª quinzena), com pico intenso em fevereiro. Colheita de maio (2ª quinzena) até setembro (1ª quinzena), com pico intenso em julho.
Goiás: Plantio de janeiro (1ª quinzena) até março (1ª quinzena), com pico intenso em fevereiro. Colheita de maio (1ª quinzena) até setembro (1ª quinzena), com pico intenso em junho e julho.
Distrito Federal: Plantio de janeiro (1ª quinzena) até fevereiro (2ª quinzena), com pico intenso em janeiro e fevereiro. Colheita de maio (2ª quinzena) até agosto (2ª quinzena), com pico intenso em junho e julho.
Minas Gerais: Plantio de janeiro (1ª quinzena) até março (2ª quinzena), com pico intenso em fevereiro. Colheita de maio (2ª quinzena) até setembro (1ª quinzena), com pico intenso em julho.
Espírito Santo: Plantio de fevereiro (1ª quinzena) até março (1ª quinzena), com pico intenso em fevereiro. Colheita de junho (1ª quinzena) até agosto (2ª quinzena), com pico intenso em julho.
Rio de Janeiro: Plantio de fevereiro (1ª quinzena) até março (1ª quinzena), com pico intenso em fevereiro. Colheita de junho (1ª quinzena) até agosto (1ª quinzena), com pico intenso em julho.
São Paulo: Plantio de janeiro (2ª quinzena) até março (2ª quinzena), com pico intenso em fevereiro e março. Colheita de junho (1ª quinzena) até setembro (2ª quinzena), com pico intenso em julho e agosto.
Paraná: Plantio de janeiro (2ª quinzena) até abril (1ª quinzena), com pico intenso em março. Colheita de junho (1ª quinzena) até outubro (1ª quinzena), com pico intenso em agosto e setembro.
Santa Catarina: Plantio de janeiro (1ª quinzena) até fevereiro (1ª quinzena), com pico intenso em janeiro. Colheita de maio (1ª quinzena) até junho (2ª quinzena), com pico intenso em maio e junho.
"""

INFO_TRIGO_CANA = """
Calendário de Safra: Trigo (Ciclo 120-135 dias)
Mato Grosso do Sul: Plantio de março (2ª quinzena) até maio (2ª quinzena), com pico intenso em abril. Colheita de agosto (1ª quinzena) até setembro (2ª quinzena), com pico intenso em agosto.
Goiás: Plantio de abril (1ª quinzena) até maio (2ª quinzena), com pico intenso em maio. Colheita de agosto (1ª quinzena) até outubro (1ª quinzena), com pico intenso em setembro.
Distrito Federal: Plantio de abril (1ª quinzena) até maio (2ª quinzena), com pico intenso em maio. Colheita de agosto (1ª quinzena) até outubro (1ª quinzena), com pico intenso em setembro.
Minas Gerais: Plantio de fevereiro (2ª quinzena) até maio (2ª quinzena), com pico intenso em março e abril. Colheita de julho (1ª quinzena) até setembro (1ª quinzena), com pico intenso em julho e agosto.
São Paulo: Plantio de março (2ª quinzena) até junho (1ª quinzena), com pico intenso em abril e maio. Colheita de julho (2ª quinzena) até outubro (2ª quinzena), com pico intenso em agosto e setembro.
Paraná: Plantio de abril (1ª quinzena) até julho (1ª quinzena), com pico intenso em maio e junho. Colheita de agosto (2ª quinzena) até novembro (2ª quinzena), com pico intenso em setembro e outubro.
Santa Catarina: Plantio de maio (2ª quinzena) até agosto (2ª quinzena), com pico intenso em junho e julho. Colheita de outubro (2ª quinzena) até dezembro (2ª quinzena), com pico intenso em novembro e dezembro.
Rio Grande do Sul: Plantio de maio (1ª quinzena) até agosto (1ª quinzena), com pico intenso em junho e julho. Colheita de outubro (1ª quinzena) até dezembro (2ª quinzena), com pico intenso em novembro e dezembro.

Calendário de Safra: Cana-de-Açúcar
(Diferente dos grãos, a cana possui ciclos de colheita e plantio mais extensos e contínuos em várias regiões)
Centro-Oeste: Plantio de janeiro (1ª quinzena) até julho (1ª quinzena) e de outubro (1ª quinzena) até dezembro (2ª quinzena). Colheita de abril (1ª quinzena) até novembro (2ª quinzena).
Nordeste: Plantio de janeiro (1ª quinzena) até abril (2ª quinzena) e de setembro (1ª quinzena) até dezembro (2ª quinzena). Colheita de janeiro (2ª quinzena) até maio (1ª quinzena) e de agosto (2ª quinzena) até outubro (2ª quinzena).
Norte: Plantio de outubro (1ª quinzena) até dezembro (2ª quinzena). Colheita de maio (1ª quinzena) até outubro (2ª quinzena).
Sudeste: Plantio de janeiro (1ª quinzena) até julho (1ª quinzena) e de outubro (1ª quinzena) até dezembro (2ª quinzena). Colheita de abril (1ª quinzena) até novembro (2ª quinzena).
Sul: Plantio de janeiro (1ª quinzena) até julho (1ª quinzena) e de outubro (1ª quinzena) até dezembro (2ª quinzena). Colheita de abril (1ª quinzena) até novembro (2ª quinzena).
"""


def get_dados_safra_completos():
    """Retorna todos os dados de safra formatados."""
    return f"""
### BEGIN DADOS_SAFRA ###
{INFO_ALGODAO}
{INFO_ARROZ}
{INFO_SOJA}
{INFO_MILHO}
{INFO_TRIGO_CANA}
### END DADOS_SAFRA ###
"""
