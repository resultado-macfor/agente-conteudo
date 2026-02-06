"""
Página de Geração de Briefings.
Geração de briefings a partir do calendário editorial ou individual.
"""
import streamlit as st
import datetime
import re
from database import modelo_texto, get_blog_db
from models import construir_contexto

PALAVRAS_PROIBIDAS = [
    "garantir", "garantia", "garantido", "garantimos",
    "certeza", "com certeza", "certamente",
    "sempre", "nunca",
    "100%", "totalmente",
    "promessa", "prometemos",
    "infalível", "perfeito",
    "melhor do mercado", "único no mercado",
    "revolucionário",
]

PRODUTOS_BIOLOGICOS = [
    "Neture", "Qualitas", "Taegro", "Timorex Gold",
    "Rizos", "Votivo Prime", "Avicta"
]


def extrair_celulas_do_calendario(calendario_csv: str) -> list:
    """
    Extrai todas as células (pautas) do calendário CSV.

    Returns:
        Lista de dicts com: {'celula': texto, 'dia_semana': str, 'linha': int}
    """
    celulas = []
    linhas = calendario_csv.strip().split('\n')

    # Pular cabeçalho se existir
    inicio = 0
    if linhas and any(dia in linhas[0].upper() for dia in ['DOMINGO', 'SEGUNDA', 'DOM', 'SEG']):
        inicio = 1

    dias_semana = ['Domingo', 'Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta', 'Sábado']

    for idx, linha in enumerate(linhas[inicio:], start=inicio+1):
        colunas = linha.split(',')
        for col_idx, celula in enumerate(colunas):
            celula_limpa = celula.strip()
            if celula_limpa and len(celula_limpa) > 3:
                celula_display = re.sub(r'^[^\w\s]+\s*', '', celula_limpa).strip()
                if celula_display:
                    dia = dias_semana[col_idx] if col_idx < len(dias_semana) else f"Col {col_idx+1}"
                    celulas.append({
                        'celula_original': celula_limpa,
                        'celula_display': celula_display,
                        'dia_semana': dia,
                        'linha': idx
                    })

    return celulas


def extrair_produto_cultura_tema(celula: str) -> dict:
    """
    Extrai produto, cultura e tema de uma célula do calendário.
    Formato esperado: "Produto - Cultura - Tema" ou variações.
    """
    celula_limpa = re.sub(r'^[^\w\s]+\s*', '', celula).strip()

    resultado = {
        'produto': '',
        'cultura': '',
        'tema': celula_limpa 
    }
    partes = celula_limpa.split(' - ')

    if len(partes) >= 3:
        resultado['produto'] = partes[0].strip()
        resultado['cultura'] = partes[1].strip()
        resultado['tema'] = ' - '.join(partes[2:]).strip()
    elif len(partes) == 2:
        resultado['produto'] = partes[0].strip()
        resultado['tema'] = partes[1].strip()

    return resultado


def extrair_info_produto_do_contexto(contexto_agente: str, nome_produto: str) -> dict:
    """
    Extrai informações específicas de um produto do contexto do agente.
    """
    info_produto = {
        "nome": nome_produto,
        "slogan": "",
        "kbfs": [],
        "assinatura": "",
        "posicionamento": "",
        "culturas": [],
        "alvos": [],
        "categoria": "",
        "encontrado": False
    }

    linhas = contexto_agente.split('\n')
    em_secao_produto = False

    for linha in linhas:
        linha_lower = linha.lower()
        produto_lower = nome_produto.lower()

        if produto_lower in linha_lower and any(x in linha_lower for x in ['produto', 'sobre', '###', '##']):
            em_secao_produto = True
            info_produto["encontrado"] = True
            continue

        if em_secao_produto and linha.startswith('##'):
            if produto_lower not in linha_lower:
                em_secao_produto = False
                continue

        if em_secao_produto:
            if 'slogan' in linha_lower or 'assinatura' in linha_lower:
                partes = linha.split(':')
                if len(partes) > 1:
                    info_produto["slogan"] = partes[1].strip().strip('"\'')

            if 'kbf' in linha_lower or 'benefício' in linha_lower or 'diferencial' in linha_lower:
                if '-' in linha:
                    kbf = linha.split('-')[-1].strip()
                    if kbf:
                        info_produto["kbfs"].append(kbf)

            if 'posicionamento' in linha_lower or 'categoria' in linha_lower:
                partes = linha.split(':')
                if len(partes) > 1:
                    info_produto["posicionamento"] = partes[1].strip()

            if any(x in linha_lower for x in ['controla', 'controle de', 'alvo', 'praga', 'doença']):
                if '-' in linha:
                    alvo = linha.split('-')[-1].strip()
                    if alvo and len(alvo) > 3:
                        info_produto["alvos"].append(alvo)

            if 'cultura' in linha_lower:
                if '-' in linha:
                    cultura = linha.split('-')[-1].strip()
                    if cultura:
                        info_produto["culturas"].append(cultura)

    return info_produto


def validar_palavras_proibidas(texto: str) -> list:
    """Verifica se o texto contém palavras proibidas."""
    encontradas = []
    texto_lower = texto.lower()
    for palavra in PALAVRAS_PROIBIDAS:
        if palavra.lower() in texto_lower:
            encontradas.append(palavra)
    return encontradas


def render():
    """Renderiza a aba de geração de briefings."""
    st.header("📋 Gerador de Briefings")

    if not st.session_state.get('agente_selecionado'):
        st.warning("Selecione um agente na parte superior do app.")
        return

    agente = st.session_state.agente_selecionado
    st.success(f"Agente: **{agente['nome']}**")

    if 'briefings_gerados' not in st.session_state:
        st.session_state.briefings_gerados = []
    if 'briefing_individual' not in st.session_state:
        st.session_state.briefing_individual = ""
    if 'briefing_celula' not in st.session_state:
        st.session_state.briefing_celula = ""

    tab_celula, tab_individual, tab_lote = st.tabs([
        "🎯 Briefing de Célula do Calendário",
        "📝 Briefing Manual",
        "📅 Briefings em Lote"
    ])
    
    # BRIEFING DE CÉLULA DO CALENDÁRIO
    with tab_celula:
        st.subheader("Gerar Briefing de uma Célula Específica")
        st.info("Selecione uma pauta diretamente do calendário e adicione contexto específico para gerar um briefing mais assertivo.")

        calendario_texto = ""
        if 'calendario_gerado' in st.session_state and st.session_state.calendario_gerado:
            st.success("📅 Calendário encontrado na sessão!")
            usar_sessao = st.checkbox("Usar calendário da sessão", value=True, key="usar_sessao_celula")

            if usar_sessao:
                calendario_texto = st.session_state.calendario_gerado
            else:
                calendario_texto = st.text_area(
                    "Ou cole o calendário aqui:",
                    height=150,
                    placeholder="Cole o calendário CSV...",
                    key="calendario_manual_celula"
                )
        else:
            calendario_texto = st.text_area(
                "Cole o calendário de pautas:",
                height=150,
                placeholder="Cole o calendário CSV gerado na aba de Calendário...",
                key="calendario_input_celula"
            )

        if calendario_texto:
            # Extrair células do calendário
            celulas = extrair_celulas_do_calendario(calendario_texto)

            if celulas:
                st.write(f"**{len(celulas)} pautas encontradas no calendário:**")
                opcoes_celulas = [
                    f"{c['dia_semana']} (linha {c['linha']}): {c['celula_display'][:60]}..."
                    if len(c['celula_display']) > 60
                    else f"{c['dia_semana']} (linha {c['linha']}): {c['celula_display']}"
                    for c in celulas
                ]

                celula_selecionada_idx = st.selectbox(
                    "Selecione a pauta:",
                    range(len(opcoes_celulas)),
                    format_func=lambda x: opcoes_celulas[x],
                    key="celula_selecionada"
                )

                celula_info = celulas[celula_selecionada_idx]

                st.markdown("---")
                st.markdown("**Pauta selecionada:**")
                st.code(celula_info['celula_original'])

                dados_extraidos = extrair_produto_cultura_tema(celula_info['celula_original'])

                col1, col2 = st.columns(2)
                with col1:
                    produto_celula = st.text_input(
                        "Produto (extraído/edite se necessário):",
                        value=dados_extraidos['produto'],
                        key="produto_celula"
                    )
                    cultura_celula = st.text_input(
                        "Cultura (extraída/edite se necessário):",
                        value=dados_extraidos['cultura'],
                        key="cultura_celula"
                    )

                with col2:
                    st.write("**Canais:**")
                    canal_pecas_cel = st.checkbox("Peças sociais (IG/FB)", value=True, key="canal_pecas_cel")
                    canal_blog_cel = st.checkbox("Blog (Mais Agro)", value=False, key="canal_blog_cel")
                    canal_webstories_cel = st.checkbox("Webstories", value=False, key="canal_webstories_cel")

                tema_celula = st.text_area(
                    "Tema (extraído/edite se necessário):",
                    value=dados_extraidos['tema'],
                    height=80,
                    key="tema_celula"
                )

                # Contexto adicional
                st.markdown("---")
                st.subheader("📌 Direcionais e Contexto da Pauta")
                st.caption("Este é o campo mais importante! Adicione aqui tudo que a IA precisa saber para gerar um briefing assertivo.")

                contexto_celula = st.text_area(
                    "Contexto estratégico e direcionais:",
                    height=180,
                    placeholder="""Exemplos do que incluir aqui:

• Esta pauta faz parte de uma ação do plano de [produto]
• O foco é no conceito "cigarrinha-do-milho+" que é diferencial do produto
• Estamos em momento de transição: colheita da soja e plantio do milho
• Usar o mote "4 dimensões" do Vaniva (proteção abaixo do solo, acima do solo, simples, sustentável)
• A pauta deve reforçar a safra 2025/26
• É um reforço de pauta sobre controle de bipolaris (não é reforço de aplicação)
• Não posicionar o produto para controle de nematoides
• Priorizar as mensagens do KV: [listar mensagens]""",
                    key="contexto_celula",
                    help="Quanto mais contexto você fornecer, mais assertivo será o briefing gerado"
                )

                # Configurações extras 
                with st.expander("⚙️ Configurações adicionais"):
                    col_cfg1, col_cfg2 = st.columns(2)

                    with col_cfg1:
                        tipo_conteudo_cel = st.selectbox(
                            "Tipo de conteúdo:",
                            [
                                "Produto + Cultura (padrão)",
                                "Glossário de Alvos",
                                "Depoimento de Produtor",
                                "Diário de Campo",
                                "Pauta Recorrente",
                                "Cobertura de Evento"
                            ],
                            key="tipo_cel"
                        )

                    with col_cfg2:
                        gerar_captacao_cel = st.checkbox(
                            "Gerar direcional de captação (vídeos)",
                            value=False,
                            key="captacao_cel"
                        )

                # Botão de geração
                if st.button("🎯 Gerar Briefing desta Pauta", type="primary", use_container_width=True, key="gerar_celula"):
                    if not tema_celula:
                        st.error("O tema da pauta é obrigatório.")
                        return

                    with st.spinner("Gerando briefing..."):
                        try:
                            contexto_agente = construir_contexto(
                                agente,
                                st.session_state.get('segmentos_selecionados', [])
                            )

                            # Buscar info do produto
                            secao_produto = ""
                            if produto_celula:
                                info_prod = extrair_info_produto_do_contexto(contexto_agente, produto_celula)
                                if info_prod["encontrado"]:
                                    secao_produto = f"""
INFORMAÇÕES DO PRODUTO ENCONTRADAS:
- Produto: {info_prod['nome']}
- Slogan: {info_prod['slogan'] or '[buscar no contexto]'}
- KBFs: {', '.join(info_prod['kbfs']) or '[buscar no contexto]'}
- Posicionamento: {info_prod['posicionamento'] or '[buscar no contexto]'}
- Alvos: {', '.join(info_prod['alvos']) or '[buscar no contexto]'}
"""
                            canais = []
                            if canal_pecas_cel:
                                canais.append("Peças sociais (IG/FB)")
                            if canal_blog_cel:
                                canais.append("Blog (mais técnico e denso)")
                            if canal_webstories_cel:
                                canais.append("Webstories")

                            # Instruções por tipo
                            instrucoes_tipo = ""
                            if tipo_conteudo_cel == "Glossário de Alvos":
                                instrucoes_tipo = """
FORMATO GLOSSÁRIO: Incluir nome comum, nome científico (itálico), sintomas, danos, ciclo.
Seguir padrão visual dos glossários anteriores."""
                            elif tipo_conteudo_cel == "Depoimento de Produtor":
                                instrucoes_tipo = """
FORMATO DEPOIMENTO: Briefing focado no vídeo. Se solicitado, incluir direcional de captação."""

                            prompt = f'''
{contexto_agente}

=== GERAR BRIEFING PARA CÉLULA DO CALENDÁRIO ===

PAUTA ORIGINAL DO CALENDÁRIO:
"{celula_info['celula_original']}"

DADOS DA PAUTA:
- Produto: {produto_celula or "Não especificado"}
- Cultura: {cultura_celula or "Não especificada"}
- Tema: {tema_celula}
- Dia: {celula_info['dia_semana']}
- Canais: {", ".join(canais) if canais else "Peças sociais"}

{secao_produto}

=== CONTEXTO ESTRATÉGICO FORNECIDO PELO USUÁRIO ===
{contexto_celula if contexto_celula else "Nenhum contexto adicional fornecido. Usar apenas as informações da pauta."}

{instrucoes_tipo}

=== REGRAS CRÍTICAS ===

1. PALAVRAS PROIBIDAS (NÃO USE): {", ".join(PALAVRAS_PROIBIDAS)}

2. NOMES CIENTÍFICOS: sempre em itálico (*Glycine max*)

3. FIDELIDADE AO TEMA:
   - Use o tema EXATAMENTE como fornecido
   - NÃO modifique nem complemente com frases adicionais
   - O tema já contém a ideia estratégica

4. FIDELIDADE AO PRODUTO:
   - Use APENAS informações do contexto do agente
   - NÃO invente slogans, KBFs ou posicionamentos
   - Se não encontrar, indique "[Buscar informação oficial]"
   - Respeite o posicionamento correto (ex: Victrato = tratamento de sementes)

5. CONTEXTO:
   - RESPEITE 100% o contexto estratégico fornecido pelo usuário acima
   - Cite APENAS pragas/doenças que o produto controla
   - NÃO mencione problemas que o produto não resolve

=== ESTRUTURA DO BRIEFING ===

## BRIEFING: {tema_celula}

### INFORMAÇÕES BÁSICAS
- **Produto:** {produto_celula or "[produto]"}
- **Cultura:** {cultura_celula or "[cultura]"}
- **Tema:** {tema_celula}
- **Canais:** {", ".join(canais)}

### CONTEXTO
[Cenário da cultura no campo - APENAS problemas que o produto resolve]

### SOBRE O PRODUTO
[Informações oficiais: slogan, KBFs, assinaturas]
[Se não encontrar: "[Buscar informação oficial de X]"]

### OBJETIVO DO CONTEÚDO
[Objetivo claro baseado no contexto fornecido]

### INFORMAÇÕES-CHAVE / ARGUMENTOS
[Pontos técnicos relevantes]
[Mensagens prioritárias do KV]
[Diferenciais a comunicar]

### ESTRUTURA SUGERIDA
{"#### Para Blog:\n[Estrutura H2/H3]" if canal_blog_cel else ""}
{"#### Para Peças Sociais:\n[Sugestões de cards/carrossel]" if canal_pecas_cel else ""}

### REFERÊNCIAS TÉCNICAS
[Embrapa, universidades, Conab, etc.]

### DIRETRIZES
- Tom: [adequado ao agente]
- Extensão: [conforme canais]

{"""
### DIRECIONAL DE CAPTAÇÃO
- Objetivo da captação
- Recomendações
- Sugestão de roteiro
""" if gerar_captacao_cel else ""}

---
Gere o briefing seguindo RIGOROSAMENTE o contexto fornecido pelo usuário.
'''

                            resposta = modelo_texto.generate_content(prompt)
                            briefing = resposta.text

                            palavras = validar_palavras_proibidas(briefing)
                            if palavras:
                                st.warning(f"Palavras proibidas encontradas: {', '.join(palavras)}")

                            st.session_state.briefing_celula = briefing
                            st.success("Briefing gerado!")

                        except Exception as e:
                            st.error(f"Erro: {str(e)}")

                # Exibir briefing gerado
                if st.session_state.briefing_celula:
                    st.markdown("---")
                    st.subheader("📄 Briefing Gerado")
                    st.markdown(st.session_state.briefing_celula)

                    col_b1, col_b2 = st.columns(2)
                    with col_b1:
                        st.download_button(
                            "💾 Baixar Briefing",
                            data=st.session_state.briefing_celula,
                            file_name=f"briefing_{produto_celula or 'pauta'}_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.md",
                            mime="text/markdown",
                            key="download_celula"
                        )
                    with col_b2:
                        if st.button("🗑️ Limpar", key="limpar_celula"):
                            st.session_state.briefing_celula = ""
                            st.rerun()
            else:
                st.warning("Nenhuma pauta encontrada no calendário. Verifique o formato.")
        else:
            st.info("Cole ou carregue um calendário para selecionar uma pauta.")


    #  BRIEFING MANUAL (sem calendário)
    with tab_individual:
        st.subheader("Criar Briefing do Zero")
        st.info("Para pautas que não estão no calendário ou quando você precisa criar um briefing manualmente.")

        col1, col2 = st.columns(2)

        with col1:
            produto_individual = st.text_input(
                "Produto:",
                placeholder="Ex: Verdavis, Miravis Duo",
                key="produto_manual"
            )

            cultura_individual = st.text_input(
                "Cultura(s):",
                placeholder="Ex: Soja, Milho, Soja e Milho",
                key="cultura_manual"
            )

            tema_individual = st.text_area(
                "Tema/Título da pauta:",
                placeholder="Ex: Como os produtores têm lidado com os percevejos na safra 2025/26",
                height=80,
                key="tema_manual"
            )

        with col2:
            st.write("**Canais de publicação:**")
            canal_pecas = st.checkbox("Peças para redes sociais (IG/FB)", value=True, key="canal_pecas_manual")
            canal_blog = st.checkbox("Blog (Mais Agro)", value=False, key="canal_blog_manual")
            canal_webstories = st.checkbox("Webstories", value=False, key="canal_ws_manual")
            canal_video = st.checkbox("Vídeo/Depoimento", value=False, key="canal_video_manual")

            tipo_conteudo = st.selectbox(
                "Tipo de conteúdo:",
                [
                    "Produto + Cultura",
                    "Glossário de Alvos",
                    "Depoimento de Produtor",
                    "Diário de Campo",
                    "Institucional",
                    "Pauta Recorrente",
                    "Cobertura de Evento/Feira"
                ],
                key="tipo_manual"
            )

        st.subheader("📌 Contexto Estratégico")
        contexto_estrategico = st.text_area(
            "Direcionais e contexto da pauta:",
            placeholder="""Adicione aqui:
- Ações de plano relacionadas
- Conceitos/motes a usar
- Momento da safra
- Mensagens prioritárias
- O que NÃO incluir""",
            height=150,
            key="contexto_manual"
        )

        with st.expander("⚙️ Configurações Avançadas"):
            col_adv1, col_adv2 = st.columns(2)
            with col_adv1:
                incluir_sobre_produto = st.checkbox("Incluir seção 'Sobre o Produto'", value=True, key="sobre_prod_manual")
            with col_adv2:
                gerar_direcional_captacao = st.checkbox("Gerar direcional de captação", value=False, key="captacao_manual")

        if st.button("📝 Gerar Briefing", type="primary", use_container_width=True, key="gerar_manual"):
            if not tema_individual:
                st.error("Preencha pelo menos o tema da pauta.")
                return

            with st.spinner("Gerando briefing..."):
                try:
                    contexto_agente = construir_contexto(
                        agente,
                        st.session_state.get('segmentos_selecionados', [])
                    )

                    secao_sobre_produto = ""
                    if produto_individual and incluir_sobre_produto:
                        info_produto = extrair_info_produto_do_contexto(contexto_agente, produto_individual)
                        if info_produto["encontrado"]:
                            secao_sobre_produto = f"""
INFORMAÇÕES DO PRODUTO:
- Produto: {info_produto['nome']}
- Slogan: {info_produto['slogan'] or '[buscar]'}
- KBFs: {', '.join(info_produto['kbfs']) or '[buscar]'}
- Posicionamento: {info_produto['posicionamento'] or '[buscar]'}
"""

                    canais_selecionados = []
                    if canal_pecas:
                        canais_selecionados.append("Peças sociais (IG/FB)")
                    if canal_blog:
                        canais_selecionados.append("Blog (Mais Agro)")
                    if canal_webstories:
                        canais_selecionados.append("Webstories")
                    if canal_video:
                        canais_selecionados.append("Vídeo/Depoimento")

                    prompt_individual = f'''
{contexto_agente}

=== GERAR BRIEFING ===

DADOS DA PAUTA:
- Produto: {produto_individual or "Não especificado"}
- Cultura(s): {cultura_individual or "Não especificada"}
- Tema: {tema_individual}
- Tipo: {tipo_conteudo}
- Canais: {", ".join(canais_selecionados) if canais_selecionados else "Peças sociais"}

CONTEXTO ESTRATÉGICO:
{contexto_estrategico if contexto_estrategico else "Nenhum contexto adicional"}

{secao_sobre_produto}

=== REGRAS ===
1. PALAVRAS PROIBIDAS: {", ".join(PALAVRAS_PROIBIDAS)}
2. Nomes científicos em itálico
3. Use o tema EXATAMENTE como escrito
4. NÃO invente informações do produto
5. Cite APENAS pragas/doenças que o produto controla

=== ESTRUTURA ===

## BRIEFING: {tema_individual}

### INFORMAÇÕES BÁSICAS
- **Produto:** [produto]
- **Cultura:** [cultura]
- **Tema:** [tema exato]
- **Canais:** [canais]

### CONTEXTO
[Cenário da cultura - apenas problemas que o produto resolve]

### SOBRE O PRODUTO
[Informações oficiais ou "[Buscar informação oficial]"]

### OBJETIVO
[Objetivo claro]

### INFORMAÇÕES-CHAVE
[Pontos técnicos e mensagens do KV]

### ESTRUTURA SUGERIDA
[Estrutura para os canais selecionados]

### REFERÊNCIAS
[Fontes técnicas]

### DIRETRIZES
- Tom: [tom do agente]
- Extensão: [conforme canal]

{"""
### DIRECIONAL DE CAPTAÇÃO
- Objetivo
- Recomendações
- Roteiro
""" if gerar_direcional_captacao else ""}

---
'''

                    resposta = modelo_texto.generate_content(prompt_individual)
                    briefing_gerado = resposta.text

                    palavras_encontradas = validar_palavras_proibidas(briefing_gerado)
                    if palavras_encontradas:
                        st.warning(f"Palavras proibidas: {', '.join(palavras_encontradas)}")

                    st.session_state.briefing_individual = briefing_gerado
                    st.success("Briefing gerado!")

                except Exception as e:
                    st.error(f"Erro: {str(e)}")

        if st.session_state.briefing_individual:
            st.subheader("📄 Briefing Gerado")
            st.markdown(st.session_state.briefing_individual)

            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                st.download_button(
                    "💾 Baixar Briefing",
                    data=st.session_state.briefing_individual,
                    file_name=f"briefing_{produto_individual or 'pauta'}_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.md",
                    mime="text/markdown",
                    key="download_manual"
                )
            with col_btn2:
                if st.button("🗑️ Limpar", key="limpar_manual"):
                    st.session_state.briefing_individual = ""
                    st.rerun()

    # BRIEFINGS EM LOTE
    with tab_lote:
        st.subheader("Gerar Briefings em Lote")
        st.info("Gera briefings para todas as pautas do calendário de uma vez. Use filtros para selecionar apenas algumas.")

        if 'calendario_gerado' in st.session_state:
            st.success("📋 Calendário detectado!")
            usar_calendario_sessao = st.checkbox("Usar calendário da sessão", value=True, key="usar_sessao_lote")

            if usar_calendario_sessao:
                calendario_texto_lote = st.session_state.calendario_gerado
                st.text_area("Calendário:", calendario_texto_lote, height=150, disabled=True, key="cal_preview_lote")
            else:
                calendario_texto_lote = st.text_area(
                    "Cole o calendário:",
                    height=150,
                    key="calendario_lote"
                )
        else:
            calendario_texto_lote = st.text_area(
                "Cole o calendário de pautas:",
                height=150,
                placeholder="Cole o calendário CSV...",
                key="calendario_lote_input"
            )

        col1, col2 = st.columns(2)

        with col1:
            tipo_briefing_lote = st.selectbox(
                "Tipo de Briefing:",
                ["Peças Sociais", "Blog (mais técnico)", "Misto (Social + Blog)"],
                key="tipo_lote"
            )

        with col2:
            nivel_detalhe = st.selectbox(
                "Nível de Detalhe:",
                ["Padrão", "Detalhado", "Resumido"],
                key="nivel_lote"
            )

        st.subheader("🔍 Filtros")
        col_f1, col_f2 = st.columns(2)

        with col_f1:
            filtrar_cultura = st.text_input("Filtrar por cultura:", placeholder="Ex: Soja", key="filtro_cultura_lote")

        with col_f2:
            filtrar_produto = st.text_input("Filtrar por produto:", placeholder="Ex: Verdavis", key="filtro_produto_lote")

        info_contextual_lote = st.text_area(
            "Contexto geral para todos os briefings:",
            placeholder="Ex: Estamos na safra 2025/26, colheita de soja no centro-sul...",
            height=100,
            key="contexto_lote"
        )

        if st.button("📝 Gerar Briefings em Lote", type="primary", use_container_width=True, key="gerar_lote"):
            if not calendario_texto_lote:
                st.error("Forneça um calendário.")
                return

            with st.spinner("Gerando briefings..."):
                try:
                    contexto_agente = construir_contexto(
                        agente,
                        st.session_state.get('segmentos_selecionados', [])
                    )

                    filtros = ""
                    if filtrar_cultura:
                        filtros += f"- Culturas: {filtrar_cultura}\n"
                    if filtrar_produto:
                        filtros += f"- Produtos: {filtrar_produto}\n"

                    prompt_lote = f'''
{contexto_agente}

## CALENDÁRIO:
{calendario_texto_lote}

## CONFIGURAÇÕES:
- Tipo: {tipo_briefing_lote}
- Detalhe: {nivel_detalhe}

## FILTROS:
{filtros if filtros else "Nenhum - gerar para todas"}

## CONTEXTO:
{info_contextual_lote if info_contextual_lote else "Usar contexto padrão"}

## REGRAS:
1. PALAVRAS PROIBIDAS: {", ".join(PALAVRAS_PROIBIDAS)}
2. Nomes científicos em itálico
3. Temas EXATOS do calendário
4. "Soja e Milho" = UMA pauta única
5. NÃO invente informações

## ESTRUTURA POR BRIEFING:

---
## BRIEFING: [Tema exato]

### INFORMAÇÕES BÁSICAS
- Produto, Cultura, Tema, Data

### CONTEXTO
[Cenário - apenas problemas que o produto resolve]

### SOBRE O PRODUTO
[Informações oficiais ou "[Buscar]"]

### OBJETIVO
[Objetivo claro]

### INFORMAÇÕES-CHAVE
[Pontos técnicos]

### ESTRUTURA SUGERIDA
[Conforme tipo]

### REFERÊNCIAS
[Fontes]

---

Gere para cada pauta do calendário.
'''

                    resposta = modelo_texto.generate_content(prompt_lote)
                    briefings_gerados = resposta.text

                    palavras = validar_palavras_proibidas(briefings_gerados)
                    if palavras:
                        st.warning(f"Palavras proibidas: {', '.join(set(palavras))}")

                    st.session_state.briefings_gerados = briefings_gerados
                    st.success("Briefings gerados!")

                except Exception as e:
                    st.error(f"Erro: {str(e)}")

        if st.session_state.briefings_gerados:
            st.subheader("📄 Briefings Gerados")
            st.markdown(st.session_state.briefings_gerados)

            count = st.session_state.briefings_gerados.count("## BRIEFING:")
            st.info(f"Total: {count} briefings")

            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                st.download_button(
                    "💾 Baixar Briefings",
                    data=st.session_state.briefings_gerados,
                    file_name=f"briefings_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.md",
                    mime="text/markdown",
                    key="download_lote"
                )

            with col_dl2:
                db_blog = get_blog_db()
                if db_blog:
                    if st.button("💾 Salvar no Banco", key="salvar_lote"):
                        try:
                            collection = db_blog['briefings_gerados']
                            documento = {
                                "briefings": st.session_state.briefings_gerados,
                                "tipo": tipo_briefing_lote,
                                "data_criacao": datetime.datetime.now(),
                                "agente": agente['nome']
                            }
                            collection.insert_one(documento)
                            st.success("Salvos!")
                        except Exception as e:
                            st.error(f"Erro: {str(e)}")

            if st.button("🗑️ Limpar", key="limpar_lote"):
                st.session_state.briefings_gerados = []
                st.rerun()
