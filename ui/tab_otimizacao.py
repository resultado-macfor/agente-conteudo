import datetime
import io
import re
import streamlit as st
from docx import Document
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from docx.shared import Pt
from utils.content_utils import construir_contexto, realizar_busca_web_perplexity


def _adicionar_paragrafo_com_links(doc, texto: str):
    """Adiciona um parágrafo ao documento, convertendo [texto](url) em hyperlinks clicáveis."""
    partes = re.split(r'(\[([^\]]+)\]\((https?://[^\)]+)\))', texto)
    if len(partes) == 1:
        doc.add_paragraph(texto)
        return

    p = doc.add_paragraph()
    for parte in partes:
        m = re.fullmatch(r'\[([^\]]+)\]\((https?://[^\)]+)\)', parte)
        if m:
            link_texto, url = m.group(1), m.group(2)
            r_id = p.part.relate_to(url, 'http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink', is_external=True)
            hyperlink = OxmlElement('w:hyperlink')
            hyperlink.set(qn('r:id'), r_id)
            run_elem = OxmlElement('w:r')
            rPr = OxmlElement('w:rPr')
            rStyle = OxmlElement('w:rStyle')
            rStyle.set(qn('w:val'), 'Hyperlink')
            rPr.append(rStyle)
            run_elem.append(rPr)
            t = OxmlElement('w:t')
            t.text = link_texto
            run_elem.append(t)
            hyperlink.append(run_elem)
            p._p.append(hyperlink)
        elif parte and not re.search(r'\[([^\]]+)\]\((https?://[^\)]+)\)', parte):
            run = p.add_run(parte)


def _gerar_docx(texto: str) -> bytes:
    doc = Document()
    for linha in texto.split("\n"):
        linha = linha.rstrip()
        if linha.startswith("#### "):
            doc.add_heading(linha[5:], level=4)
        elif linha.startswith("### "):
            doc.add_heading(linha[4:], level=3)
        elif linha.startswith("## "):
            doc.add_heading(linha[3:], level=2)
        elif linha.startswith("# "):
            doc.add_heading(linha[2:], level=1)
        else:
            _adicionar_paragrafo_com_links(doc, linha)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def render(tab, modelo_texto):
    with tab:
        st.header("🚀 Otimização SEO de Conteúdo")

        for key, default in [
            ('conteudo_otimizado', None),
            ('ajustes_realizados', []),
            ('fontes_busca_web', ""),
        ]:
            if key not in st.session_state:
                st.session_state[key] = default

        col_esq, col_dir = st.columns(2)
        with col_esq:
            briefing_entrada = st.text_area(
                "📋 Briefing de entrada:",
                height=300,
                placeholder="Cole aqui o briefing com título H1, objetivo, KWs, estrutura H2/H3, CTA, tom, etc.",
            )
        with col_dir:
            conteudo_original = st.text_area(
                "📄 Conteúdo original (a ser otimizado):",
                height=300,
                placeholder="Cole aqui o conteúdo existente que será reescrito/otimizado com base no briefing.",
            )

        col1, col2 = st.columns([2, 1])
        with col1:
            usar_busca_web = st.checkbox("🔍 Enriquecer com busca web (Perplexity)", value=False)
        with col2:
            nivel_heading = st.selectbox(
                "Heading de partida do corpo:",
                ["H2", "H3", "H1"],
                help="Nível do primeiro heading do corpo do artigo (conforme briefing)",
            )

        if st.button("🚀 Otimizar Conteúdo", type="primary", use_container_width=True):
            if not briefing_entrada:
                st.warning("Cole o briefing de entrada.")
                return
            if not conteudo_original:
                st.warning("Cole o conteúdo original a ser otimizado.")
                return

            with st.spinner("Processando otimização..."):
                try:
                    fontes_encontradas = ""
                    if usar_busca_web:
                        st.info("🔍 Buscando fontes no Perplexity...")
                        busca_ph = st.empty()
                        try:
                            resultado_busca = realizar_busca_web_perplexity(
                                conteudo_original, "SEO", "técnico"
                            )
                            if resultado_busca and not resultado_busca.startswith("❌"):
                                fontes_encontradas = resultado_busca
                                st.session_state.fontes_busca_web = resultado_busca
                                busca_ph.success(f"✅ Busca concluída ({len(resultado_busca.split())} palavras)")
                                with st.expander("Prévia das fontes", expanded=False):
                                    st.markdown(resultado_busca[:1000] + ("..." if len(resultado_busca) > 1000 else ""))
                            else:
                                busca_ph.warning("⚠️ Busca não retornou resultados válidos")
                        except Exception as e:
                            busca_ph.error(f"❌ Erro na busca: {e}")

                    contexto_agente = ""
                    if st.session_state.get("agente_selecionado"):
                        agente = st.session_state.agente_selecionado
                        contexto_agente = construir_contexto(agente, st.session_state.get("segmentos_selecionados", []))

                    prompt = f"""Você é um especialista em SEO e redação técnica para o agronegócio.

Sua tarefa é otimizar e reescrever o CONTEÚDO ORIGINAL com base nas diretrizes do BRIEFING, produzindo um artigo completo, estruturado e pronto para publicação.

{f"###CONTEXTO DO AGENTE###\\n{contexto_agente}\\n###FIM DO CONTEXTO###\\n" if contexto_agente else ""}

---
###BRIEFING###
{briefing_entrada}
###FIM DO BRIEFING###

---
###CONTEÚDO ORIGINAL###
{conteudo_original}
###FIM DO CONTEÚDO ORIGINAL###

{f"---\\n###FONTES WEB###\\n{fontes_encontradas}\\n###FIM DAS FONTES###\\n" if fontes_encontradas else ""}

---
## INSTRUÇÕES DE OTIMIZAÇÃO

### 1. LEITURA DO BRIEFING
O briefing pode conter campos estruturados — interprete-os assim:
- `TÍTULO H1 OBRIGATORIO:` ou `TÍTULO/H1 desejado:` → use como H1 exato do artigo
- `H2: [texto]`, `H3: [texto]` listados na estrutura → são as seções a gerar, na ordem exata
- `link da CTA:` → URL a ancorar no texto da CTA
- `CTA OBRIGATÓRIA:` ou `CTA FINAL OBRIGATÓRIA:` → texto a inserir antes da conclusão
- `Palavra-chave principal (KW1):` → incluir no META TITLE, META DESCRIPTION e no texto
- `Diretrizes de tom/estilo:` → tom a seguir

### 2. METADADOS SEO — coloque SEMPRE no topo, antes do H1
```
META TITLE: [até 60 caracteres, com KW principal]
META DESCRIPTION: [até 155 caracteres, com KW e chamada para ação]
URL: /[slug-amigavel-baseado-no-h1]
CATEGORIA: [categoria sugerida]
ALT TEXT CAPA: [texto descritivo com KW]
```

### 3. ESTRUTURA DO ARTIGO
- Use o H1 exatamente como indicado no briefing
- Gere **exatamente** as seções H2 e H3 listadas no briefing, na mesma ordem — NADA a mais
- Heading de partida do corpo: **{nivel_heading}** (ou conforme indicado no briefing)
- H3 deve ser subseção de H2; não use H4 quando o briefing pede H3

### 4. INTRODUÇÃO
- PROIBIDO: "No dinâmico cenário do agronegócio..." ou variações genéricas
- PROIBIDO: apresentar o público-alvo ("Se você é produtor...", "Este guia é para...")
- Comece com dado factual ou afirmação direta sobre o tema do artigo

### 5. QUALIDADE TEXTUAL
- Parágrafos com NO MÁXIMO 3 frases curtas (~20 palavras por frase)
- Use bullet points (`-`) para listas de 3 ou mais itens
- Negrito apenas em termos técnicos ou dados-chave
- Cada informação aparece UMA única vez — não repita entre seções

### 6. CTA OBRIGATÓRIA
- Insira o texto da CTA exatamente como no briefing
- Ancore o link da CTA: [Confira a central de conteúdos Mais Agro](URL do briefing)

### 7. LINKS ANCORADOS — obrigatório, clicáveis no DOCX e na web
- Formato: [texto descritivo](URL completa)
- Ancore TODOS os links: CTA, interlinks do briefing, fontes web, produtos citados
- NUNCA escreva URLs cruas no corpo — sempre ancore em texto descritivo
- Ao final do artigo:
  ```
  LINKS USADOS:
  - [texto âncora] → URL completa
  ```

### 8. TABELAS
- Use SOMENTE Markdown puro — nunca HTML ou estilos inline:
  | Coluna 1 | Coluna 2 |
  |----------|----------|
  | Dado A   | Dado B   |

---
Gere o artigo completo seguindo todas as instruções acima.
"""

                    resposta = modelo_texto.generate_content(prompt)
                    resultado = resposta.text

                    st.session_state.conteudo_otimizado = resultado
                    st.success("✅ Conteúdo otimizado com sucesso!")

                    if "META TITLE:" in resultado:
                        linhas = resultado.split("\n")
                        meta_linhas = [l for l in linhas if any(
                            l.startswith(k) for k in ("META TITLE:", "META DESCRIPTION:", "URL:", "CATEGORIA:", "ALT TEXT")
                        )]
                        if meta_linhas:
                            st.subheader("📊 Metadados SEO")
                            for l in meta_linhas:
                                st.markdown(f"**{l}**")

                    st.subheader("📝 Conteúdo Otimizado")
                    st.markdown(resultado)

                    col_m1, col_m2, col_m3 = st.columns(3)
                    palavras = len(resultado.split())
                    with col_m1:
                        st.metric("Palavras", palavras)
                    with col_m2:
                        headings = resultado.count("## ") + resultado.count("### ")
                        st.metric("Headings", headings)
                    with col_m3:
                        tem_cta = "maisagro" in resultado.lower() or "syngenta" in resultado.lower()
                        st.metric("CTA", "✅" if tem_cta else "❌")

                    st.download_button(
                        "💾 Baixar Conteúdo Otimizado (.docx)",
                        data=_gerar_docx(resultado),
                        file_name=f"conteudo_otimizado_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        use_container_width=True,
                    )

                except Exception as e:
                    st.error(f"❌ Erro na otimização: {e}")

        if st.session_state.conteudo_otimizado:
            st.divider()
            st.subheader("🔄 Ajustes Incrementais")

            comando_ajuste = st.text_area(
                "Descreva os ajustes:",
                height=80,
                placeholder="Ex: Reescreva a introdução, ajuste o tom, corrija o heading da seção X...",
                key="ajuste_text",
            )

            if st.button("🔄 Aplicar Ajustes", key="btn_ajuste"):
                if not comando_ajuste:
                    st.warning("Descreva os ajustes desejados.")
                    return

                with st.spinner("Aplicando ajustes..."):
                    try:
                        prompt_ajuste = f"""Você é um especialista em SEO e redação técnica para o agronegócio.

Aplique os ajustes solicitados ao conteúdo abaixo, mantendo:
- Os metadados SEO (META TITLE, META DESCRIPTION, URL, CATEGORIA, ALT TEXT) no topo
- A hierarquia de headings existente (não altere os níveis de heading)
- Apenas as seções previstas no briefing original — não adicione seções extras
- Parágrafos com no máximo 3 frases curtas
- Bullet points para listas de 3 ou mais itens
- Links ancorados existentes
- A CTA com link âncora

**CONTEÚDO ATUAL:**
{st.session_state.conteudo_otimizado}

**AJUSTES SOLICITADOS:**
{comando_ajuste}

Retorne o conteúdo completo com os ajustes aplicados.
"""
                        resposta = modelo_texto.generate_content(prompt_ajuste)
                        st.session_state.conteudo_otimizado = resposta.text
                        st.session_state.ajustes_realizados.append(comando_ajuste)

                        st.success("✅ Ajustes aplicados!")
                        st.markdown(resposta.text)

                        st.download_button(
                            "💾 Baixar versão ajustada (.docx)",
                            data=_gerar_docx(resposta.text),
                            file_name=f"conteudo_ajustado_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.docx",
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                            key="dl_ajuste",
                        )

                    except Exception as e:
                        st.error(f"Erro: {e}")

            if st.session_state.ajustes_realizados:
                with st.expander(f"Histórico de ajustes ({len(st.session_state.ajustes_realizados)})"):
                    for i, aj in enumerate(st.session_state.ajustes_realizados, 1):
                        st.markdown(f"{i}. {aj}")
                if st.button("🗑️ Limpar histórico"):
                    st.session_state.ajustes_realizados = []
                    st.rerun()
