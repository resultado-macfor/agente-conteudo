import datetime
import streamlit as st
from utils.content_utils import construir_contexto
from services.database import get_blog_rag_db


def render(tab, modelo_texto):
    with tab:
        st.header("🌱 Blog Inteligente - Geração Avançada")
        st.markdown("**Cole tudo o que você quer abordar em uma única caixa de texto. O sistema fará o resto.**")

        try:
            _, db_blog_rag, collection_posts_rag, _ = get_blog_rag_db()
            mongo_connected = True
        except Exception as e:
            st.error(f"❌ Erro na conexão com MongoDB: {str(e)}")
            mongo_connected = False

        for key, default in [
            ('conteudo_gerado_blog', None),
            ('versoes_blog', []),
            ('relatorio_fontes_blog', None),
            ('briefing_original_blog', None),
            ('fontes_perplexity_blog', []),
            ('usou_perplexity_blog', False),
        ]:
            if key not in st.session_state:
                st.session_state[key] = default

        st.markdown("---")

        texto_briefing = st.text_area(
            "📋 **DESCREVA O CONTEÚDO QUE VOCÊ QUER GERAR**",
            height=250,
            placeholder="""Exemplo de briefing completo:

Título: Manejo de nematoides na cultura da soja com produtos biológicos

Cultura: Soja
Problema: Aumento da população de nematoides (Meloidogyne e Heterodera) em solos com palhada de milho
Produtos: NemaControl (bionematicida) e Victrato (bioativador)

Objetivo: Educar o produtor sobre a importância do manejo biológico de nematoides, mostrando resultados práticos e posicionando os produtos como solução eficaz.

Público-alvo: Produtores de soja do Centro-Oeste, nível técnico médio a alto.

Palavras-chave principais: manejo de nematoides, bionematicida, soja
Palavras-chave secundárias: Meloidogyne, Heterodera, tratamento de sementes, produtividade

Observações importantes:
- Tom técnico mas acessível
- Incluir dados de eficácia dos produtos
- Citar resultados de pesquisas da Embrapa
- Texto com ~1500 palavras
- Incluir CTA para falar com consultor técnico
""",
            key="briefing_unico",
        )

        st.markdown("---")

        with st.expander("⚙️ Configurações Avançadas (opcional)", expanded=False):
            col_adv1, col_adv2 = st.columns(2)

            with col_adv1:
                palavras_chave_input = st.text_input(
                    "Palavras-chave (separadas por vírgula):",
                    placeholder="ex: manejo de nematoides, bionematicida, soja",
                )
                densidade_palavras = st.slider("Densidade desejada para palavras-chave (%):", 1, 10, 3)
                palavras_primeira_linha = st.text_input(
                    "Palavras que devem aparecer na primeira linha:",
                    placeholder="ex: nematoides, soja, manejo",
                )

            with col_adv2:
                usar_perplexity_blog = st.checkbox("🌐 Buscar informações atualizadas na web", value=True)
                if usar_perplexity_blog:
                    profundidade_busca = st.select_slider(
                        "Profundidade da busca:", options=["Básica", "Moderada", "Avançada"], value="Avançada"
                    )
                else:
                    profundidade_busca = "Avançada"

                tom_voz = st.selectbox("Tom de voz:",
                                       ["Técnico-científico", "Jornalístico", "Educativo", "Consultivo"], index=0)
                numero_palavras = st.number_input("Número aproximado de palavras:", 500, 5000, 1500, step=100)

        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
        with col_btn2:
            if st.button("🚀 GERAR CONTEÚDO DO BLOG", type="primary", use_container_width=True):
                if not texto_briefing.strip():
                    st.error("❌ Por favor, descreva o conteúdo que deseja gerar.")
                    return

                with st.spinner("🔄 Processando briefing e gerando conteúdo..."):
                    try:
                        st.session_state.briefing_original_blog = texto_briefing

                        palavras_chave_lista = [p.strip() for p in palavras_chave_input.split(',') if p.strip()] if palavras_chave_input else []
                        palavras_primeira_linha_lista = [p.strip() for p in palavras_primeira_linha.split(',') if p.strip()] if palavras_primeira_linha else []

                        
                        resultados_perplexity = {"resultado": None, "fontes": [], "erro": None}
                        if usar_perplexity_blog:
                            with st.spinner("🌐 Buscando informações atualizadas na web..."):
                                resultados_perplexity = _buscar_perplexity_blog(texto_briefing, profundidade_busca)
                                if resultados_perplexity.get('erro'):
                                    st.warning(f"⚠️ Busca web: {resultados_perplexity['erro']}")
                                else:
                                    fontes_count = len(resultados_perplexity.get('fontes', []))
                                    st.success(f"✅ {fontes_count} fontes encontradas na web")
                                    st.session_state.fontes_perplexity_blog = resultados_perplexity.get('fontes', [])
                                    st.session_state.usou_perplexity_blog = True
                        else:
                            st.session_state.usou_perplexity_blog = False

                        contexto_agente = ""
                        if st.session_state.agente_selecionado:
                            agente = st.session_state.agente_selecionado
                            contexto_agente = construir_contexto(agente, st.session_state.segmentos_selecionados)

                        nivel_h_corpo = "H2"
                        import re as _re
                        m = _re.search(r'\bH([23])\b', texto_briefing, _re.IGNORECASE)
                        if m:
                            nivel_h_corpo = f"H{m.group(1)}"

                        prompt_geracao_blog = f"""Você é um redator técnico especializado em agronegócio, escrevendo para o portal Mais Agro da Syngenta.

{f"###CONTEXTO DO AGENTE###\\n{contexto_agente}\\n###FIM DO CONTEXTO###\\n" if contexto_agente else ""}

---
###BRIEFING###
{texto_briefing}
###FIM DO BRIEFING###

---
###FONTES WEB (use para enriquecer com dados e links ancorados)###
{resultados_perplexity.get('resultado', 'Nenhuma informação da web disponível.')}
###FIM DAS FONTES###

---
###CONFIGURAÇÕES###
- Tom de voz: {tom_voz}
- Número de palavras: {numero_palavras} (±10%)
- Palavras-chave: {', '.join(palavras_chave_lista) if palavras_chave_lista else 'extraídas do briefing'}
- Densidade de palavras-chave: {densidade_palavras}%
- Palavras obrigatórias na primeira linha: {', '.join(palavras_primeira_linha_lista) if palavras_primeira_linha_lista else 'não especificadas'}
- Nível de heading do corpo: {nivel_h_corpo}
###FIM DAS CONFIGURAÇÕES###

---
## REGRAS OBRIGATÓRIAS — SIGA À RISCA:

### METADADOS (coloque SEMPRE no topo, antes do H1):
```
META TITLE: [até 60 caracteres, com KW principal]
META DESCRIPTION: [até 155 caracteres, com KW e chamada para ação]
URL: /[slug-amigavel]
CATEGORIA: [categoria sugerida]
ALT TEXT CAPA: [texto descritivo com KW]
```

### ESTRUTURA EXATA DO ARTIGO — siga esta ordem:

**Bloco 1 — Introdução:**
- 2 a 3 parágrafos diretos e factuais sobre o tema

**Bloco 2 — Corpo do artigo:**
- Use exatamente as seções {nivel_h_corpo} listadas no briefing, nessa ordem
- Cada seção: 2 a 4 parágrafos + bullets quando há 3 ou mais itens paralelos
- Bullets com termo em negrito: `- **Termo:** explicação da característica`
- Listas numeradas para sequências de etapas (1. 2. 3.)

**Bloco 3 — Produtos Syngenta recomendados:**
- Seção com heading próprio ao final do corpo (antes da CTA)
- Mencione nome comercial + registro (ex: CALARIS®, Dual Gold®, Grover®)
- Tom institucional: "A Syngenta oferece qualidade e tecnologia..."

**Bloco 4 — CTA final:**
- Insira a CTA do briefing
- Última frase: "Confira a central de conteúdos [Mais Agro](URL-da-CTA) para ficar por dentro de tudo o que está acontecendo no campo."

### INTRODUÇÃO — PROIBIÇÕES:
- PROIBIDO: "No dinâmico cenário do agronegócio...", "No contexto atual...", qualquer abertura genérica
- PROIBIDO: apresentar o leitor ("Se você é produtor...", "Este guia é para...")
- PROIBIDO: frases meta ("Neste artigo, vamos explorar...", "Ao longo deste conteúdo...")
- OBRIGATÓRIO: comece com dado factual, afirmação direta ou contexto imediato do tema

### QUALIDADE TEXTUAL:
- Parágrafos com NO MÁXIMO 3 frases curtas (~20 palavras por frase)
- Negrito (`**texto**`) em: termo técnico na 1ª ocorrência, dados-chave, nome do item em bullet
- A KW principal em negrito SOMENTE na primeira vez — nas demais, sem negrito
- Cada informação aparece UMA única vez — não repita entre seções

### LINKS — regras obrigatórias:
**Formato:** SEMPRE Markdown `[texto descritivo](https://url-completa.com)`

**PROIBIDO:**
- URLs cruas no corpo do texto
- Formato `texto → URL` com seta
- Numeração estilo Wikipedia `[1]`, `[2]`
- Qualquer empresa agroquímica que não seja Syngenta: BASF, Bayer, Corteva, FMC, UPL, Adama, Helm, Nufarm

**Links internos (Mais Agro) — 3 a 4 obrigatórios distribuídos no corpo:**
- Ancore cada link no parágrafo onde o tema é mencionado, dentro da frase
- Exemplo correto: O [manejo integrado de plantas daninhas](https://maisagro.syngenta.com.br/manejo-plantas-daninhas) combina práticas culturais e químicas para reduzir o banco de sementes no solo.
- Exemplo errado: lista de links no final ou agrupados numa seção separada
- Use URLs no padrão `https://maisagro.syngenta.com.br/[slug-do-tema]`

**Links externos — 2 a 3 obrigatórios no corpo:**
- Somente fontes neutras: Embrapa, universidades, institutos governamentais
- Ancore no dado/estudo: `De acordo com [pesquisa da Embrapa Soja](https://www.embrapa.br/...), a espécie...`
- NUNCA "clique aqui" ou anchor genérica

### FORMATAÇÃO — PROIBIÇÕES ABSOLUTAS:
**NUNCA use tags HTML**: `<strong>`, `<em>`, `<b>`, `<i>`, `<a>`, `<br>`, `<p>`, `<ul>`, `<li>`, `<h1>` etc.
Use EXCLUSIVAMENTE Markdown puro: `**negrito**`, `[link](url)`, `## Heading`, `- item`, `1. item`.

### TABELAS:
- Markdown puro — nunca HTML:
  | Coluna 1 | Coluna 2 |
  |----------|----------|
  | Dado A   | Dado B   |

---
**LEMBRETE FINAL:** Markdown puro, zero HTML, zero `[n]` Wikipedia, zero concorrentes Syngenta, links internos ancorados nos parágrafos onde o tema é discutido, CTA com link âncora no final.

Gere o artigo completo seguindo todas as regras acima.
"""

                        resposta = modelo_texto.generate_content(prompt_geracao_blog)
                        conteudo_gerado = resposta.text

                        relatorio_fontes = "## 📚 REFERÊNCIAS E FONTES UTILIZADAS\n\n"
                        if resultados_perplexity.get('fontes'):
                            relatorio_fontes += "### Fontes da Web:\n"
                            for i, fonte in enumerate(resultados_perplexity['fontes'], 1):
                                relatorio_fontes += f"{i}. {fonte}\n"
                        else:
                            relatorio_fontes += "*Nenhuma fonte web específica foi capturada.*\n"

                        st.session_state.conteudo_gerado_blog = conteudo_gerado
                        st.session_state.relatorio_fontes_blog = relatorio_fontes
                        st.session_state.versoes_blog = [{
                            "versao": 1,
                            "conteudo": conteudo_gerado,
                            "data": datetime.datetime.now(),
                            "descricao": "Geração inicial",
                        }]

                        if mongo_connected:
                            try:
                                collection_posts_rag.insert_one({
                                    "briefing": texto_briefing,
                                    "conteudo": conteudo_gerado,
                                    "fontes": resultados_perplexity.get('fontes', []),
                                    "configuracoes": {
                                        "tom_voz": tom_voz,
                                        "palavras_chave": palavras_chave_lista,
                                        "usou_perplexity": usar_perplexity_blog,
                                    },
                                    "data_criacao": datetime.datetime.now(),
                                })
                            except Exception as e:
                                st.warning(f"⚠️ Conteúdo gerado mas não salvo no banco: {str(e)}")

                        st.success("✅ Conteúdo gerado com sucesso!")
                        st.rerun()

                    except Exception as e:
                        st.error(f"❌ Erro na geração: {str(e)}")

        if st.session_state.conteudo_gerado_blog:
            st.markdown("---")

            palavras_count = len(st.session_state.conteudo_gerado_blog.split())
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1:
                st.metric("📊 Palavras", palavras_count)
            with col_m2:
                st.metric("📋 Versões", len(st.session_state.versoes_blog))
            with col_m3:
                st.metric("🎯 Tom", st.session_state.get('tom_voz_blog', 'Técnico-científico'))
            with col_m4:
                usou_perplexity = st.session_state.get('usou_perplexity_blog', False)
                tem_fontes = len(st.session_state.get('fontes_perplexity_blog', [])) > 0
                st.metric("🌐 Fontes", "✅" if usou_perplexity and tem_fontes else "❌")

            tab_conteudo_b, tab_ref, tab_versoes, tab_export = st.tabs([
                "📝 Conteúdo Gerado", "📚 Referências", "📋 Histórico", "💾 Exportar"
            ])

            with tab_conteudo_b:
                st.markdown(st.session_state.conteudo_gerado_blog)

            with tab_ref:
                if st.session_state.relatorio_fontes_blog:
                    st.markdown(st.session_state.relatorio_fontes_blog)
                else:
                    st.info("Nenhuma referência disponível")

            with tab_versoes:
                if st.session_state.versoes_blog:
                    for versao in reversed(st.session_state.versoes_blog[-5:]):
                        data_str = versao['data'].strftime('%d/%m/%Y %H:%M') if isinstance(versao['data'], datetime.datetime) else 'Data desconhecida'
                        with st.expander(f"Versão {versao['versao']} - {data_str} - {versao['descricao']}"):
                            conteudo_versao = versao['conteudo']
                            st.text_area(f"Conteúdo da versão {versao['versao']}",
                                         value=conteudo_versao[:500] + "..." if len(conteudo_versao) > 500 else conteudo_versao,
                                         height=200, key=f"versao_{versao['versao']}")
                            if st.button(f"Restaurar versão {versao['versao']}", key=f"restore_{versao['versao']}"):
                                st.session_state.conteudo_gerado_blog = versao['conteudo']
                                st.success(f"✅ Versão {versao['versao']} restaurada!")
                                st.rerun()
                else:
                    st.info("Nenhuma versão disponível")

            with tab_export:
                col_exp1, col_exp2 = st.columns(2)
                with col_exp1:
                    st.download_button("📥 Baixar como TXT", data=st.session_state.conteudo_gerado_blog,
                                       file_name=f"blog_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                                       mime="text/plain", use_container_width=True)
                    st.download_button("📥 Baixar como MD", data=st.session_state.conteudo_gerado_blog,
                                       file_name=f"blog_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.md",
                                       mime="text/markdown", use_container_width=True)
                with col_exp2:
                    if st.session_state.relatorio_fontes_blog:
                        st.download_button("📥 Baixar Referências", data=st.session_state.relatorio_fontes_blog,
                                           file_name=f"referencias_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.md",
                                           mime="text/markdown", use_container_width=True)
                    pacote = f"""# BLOG POST - {datetime.datetime.now().strftime('%d/%m/%Y')}

## BRIEFING ORIGINAL
{st.session_state.briefing_original_blog if st.session_state.briefing_original_blog else 'N/A'}

## CONTEÚDO GERADO
{st.session_state.conteudo_gerado_blog}

## REFERÊNCIAS
{st.session_state.relatorio_fontes_blog if st.session_state.relatorio_fontes_blog else 'N/A'}
"""
                    st.download_button("📦 Pacote Completo", data=pacote,
                                       file_name=f"pacote_completo_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                                       mime="text/plain", use_container_width=True)

            st.markdown("---")
            st.subheader("🔄 Ajustar Conteúdo")

            col_ajuste1, col_ajuste2 = st.columns([3, 1])
            with col_ajuste1:
                solicitacao_ajuste = st.text_area(
                    "Descreva os ajustes desejados:",
                    placeholder="Exemplos:\n- Aprofunde mais na seção sobre modo de ação dos produtos\n- Adicione mais dados de eficácia com fontes\n- Melhore a narrativa, conectando melhor problema e solução",
                    height=100,
                    key="campo_ajuste_blog",
                )
            with col_ajuste2:
                st.markdown("#####")
                if st.button("✅ APLICAR AJUSTES", type="secondary", use_container_width=True):
                    if solicitacao_ajuste.strip():
                        with st.spinner("🔄 Aplicando ajustes..."):
                            try:
                                prompt_ajuste = f"""Você é um redator técnico especializado em agronegócio, escrevendo para o portal Mais Agro da Syngenta.

## CONTEÚDO ATUAL:
{st.session_state.conteudo_gerado_blog}

## BRIEFING ORIGINAL:
{st.session_state.briefing_original_blog if st.session_state.briefing_original_blog else 'N/A'}

## AJUSTES SOLICITADOS:
{solicitacao_ajuste}

## REGRAS PARA APLICAR OS AJUSTES — mantenha INTEGRALMENTE:
- Metadados SEO (META TITLE, META DESCRIPTION, URL, CATEGORIA, ALT TEXT) no topo
- Hierarquia de headings existente (não altere os níveis)
- Parágrafos com no máximo 3 frases curtas
- Bullets com `**Termo:** explicação`
- Links ancorados distribuídos no corpo (não agrupe no final)
- Seção de produtos Syngenta recomendados
- CTA final com link âncora para o Mais Agro
- Markdown puro — zero tags HTML, zero citações `[n]`, zero concorrentes Syngenta
- NÃO adicione seções fora do briefing original

RETORNE O CONTEÚDO COMPLETO COM OS AJUSTES APLICADOS.
"""

                                resposta_ajuste = modelo_texto.generate_content(prompt_ajuste)
                                conteudo_ajustado = resposta_ajuste.text

                                nova_versao = {
                                    "versao": len(st.session_state.versoes_blog) + 1,
                                    "conteudo": st.session_state.conteudo_gerado_blog,
                                    "data": datetime.datetime.now(),
                                    "descricao": f"Ajuste: {solicitacao_ajuste[:50]}...",
                                }
                                st.session_state.versoes_blog.append(nova_versao)
                                st.session_state.conteudo_gerado_blog = conteudo_ajustado

                                st.success("✅ Ajustes aplicados com sucesso!")
                                st.rerun()

                            except Exception as e:
                                st.error(f"❌ Erro ao aplicar ajustes: {str(e)}")
                    else:
                        st.warning("⚠️ Descreva os ajustes desejados.")

        if mongo_connected:
            st.markdown("---")
            st.subheader("📚 Histórico de Gerações")
            try:
                historico = list(collection_posts_rag.find().sort("data_criacao", -1).limit(5))
                if historico:
                    for post in historico:
                        data_str = post.get('data_criacao', '').strftime('%d/%m/%Y %H:%M') if post.get('data_criacao') else 'Data desconhecida'
                        with st.expander(f"📄 {data_str} - Briefing: {post.get('briefing', '')[:100]}..."):
                            st.write(f"**Palavras:** {len(post.get('conteudo', '').split())}")
                            st.write(f"**Fontes:** {len(post.get('fontes', []))}")
                            if st.button(f"Carregar este post", key=f"load_{post.get('_id')}"):
                                st.session_state.conteudo_gerado_blog = post.get('conteudo', '')
                                st.session_state.briefing_original_blog = post.get('briefing', '')
                                st.success("✅ Post carregado!")
                                st.rerun()
                else:
                    st.info("Nenhum post no histórico")
            except Exception as e:
                st.warning(f"Erro ao carregar histórico: {str(e)}")


def _buscar_perplexity_blog(briefing: str, profundidade: str) -> dict:
    """Busca informações atualizadas na web via Perplexity."""
    try:
        import re
        import os
        from perplexity import Perplexity

        perp_api_key = os.getenv("PERP_API_KEY")
        if not perp_api_key:
            return {"erro": "PERP_API_KEY não encontrada", "resultado": None, "fontes": []}

        client = Perplexity(api_key=perp_api_key)

        prompt_busca = f"""
        Você é um pesquisador agrícola. Busque informações técnicas atualizadas e confiáveis sobre:

        {briefing[:800]}

        REQUISITOS:
        1. Fontes: Embrapa, universidades, artigos científicos, boletins técnicos
        2. Dados concretos: números, estatísticas, resultados de pesquisa
        3. Informações dos últimos 2-3 anos sempre que possível
        4. Para CADA informação, forneça a fonte completa

        FORMATO:
        ## INFORMAÇÕES ENCONTRADAS

        ### [Tópico 1]
        - Informação: [dado técnico]
        - Fonte: [instituição, ano]
        - Relevância: [por que é relevante para o tema]
        - URL/Link: [se disponível]

        ## LISTA DE FONTES
        [Lista numerada com todas as fontes utilizadas]
        """

        response = client.chat.completions.create(
            model="sonar",
            messages=[{"role": "user", "content": prompt_busca}],
            temperature=0.0,
            max_tokens=20000,
        )

        if response and response.choices:
            resultado = response.choices[0].message.content
            # Remove citações numéricas estilo Wikipedia [1], [2], [9] etc.
            resultado = re.sub(r'\s*\[\d+\]', '', resultado)

            fontes = []
            for linha in resultado.split('\n'):
                if 'http://' in linha or 'https://' in linha:
                    urls = re.findall(r'(https?://[^\s\)]+)', linha)
                    fontes.extend(urls)
                elif 'Fonte:' in linha and '[' not in linha:
                    fontes.append(linha.strip())

            return {"erro": None, "resultado": resultado, "fontes": list(set(fontes))[:15]}
        else:
            return {"erro": "Sem resposta", "resultado": None, "fontes": []}

    except Exception as e:
        return {"erro": str(e), "resultado": None, "fontes": []}
