"""
Página de Geração de Blog Posts Agrícolas.
Geração especializada de conteúdo para agronegócio.
"""
import streamlit as st
import datetime
import uuid
from database import modelo_texto, get_blog_db
from models import construir_contexto
from services import transcrever_audio_video, reescrever_com_rag_blog


def render():
    """Renderiza a aba de geração de blog agrícola."""
    st.title("🌱 Gerador de Blog Posts Agrícolas")
    st.markdown("Crie conteúdos especializados para o agronegócio seguindo a estrutura profissional")

    # Conexão com MongoDB
    db_blog = get_blog_db()
    mongo_connected = db_blog is not None

    if not mongo_connected:
        st.warning("Conexão com MongoDB não disponível")

    # Funções auxiliares
    def salvar_post(titulo, cultura, editoria, mes_publicacao, objetivo_post, url,
                    texto_gerado, palavras_chave, palavras_proibidas, tom_voz,
                    estrutura, palavras_contagem, meta_title, meta_descricao,
                    linha_fina, links_internos=None):
        if mongo_connected:
            collection_posts = db_blog['posts_gerados']
            documento = {
                "id": str(uuid.uuid4()),
                "titulo": titulo,
                "cultura": cultura,
                "editoria": editoria,
                "mes_publicacao": mes_publicacao,
                "objetivo_post": objetivo_post,
                "url": url,
                "texto_gerado": texto_gerado,
                "palavras_chave": palavras_chave,
                "palavras_proibidas": palavras_proibidas,
                "tom_voz": tom_voz,
                "estrutura": estrutura,
                "palavras_contagem": palavras_contagem,
                "meta_title": meta_title,
                "meta_descricao": meta_descricao,
                "linha_fina": linha_fina,
                "links_internos": links_internos or [],
                "versao": "2.1",
                "data_criacao": datetime.datetime.now()
            }
            collection_posts.insert_one(documento)
            return True
        return False

    def carregar_posts_anteriores():
        if mongo_connected:
            try:
                collection_posts = db_blog['posts_gerados']
                return list(collection_posts.find({}).sort("data_criacao", -1).limit(10))
            except:
                return []
        return []

    def carregar_kbf_produtos():
        if mongo_connected:
            try:
                collection_kbf = db_blog['kbf_produtos']
                return list(collection_kbf.find({}))
            except:
                return []
        return []

    # Assinatura e box inicial
    ASSINATURA_PADRAO = """
---

**Sobre o Mais Agro**
O Mais Agro é uma plataforma de conteúdo especializado em agronegócio.

📞 **Fale conosco:** [contato@maisagro.com.br](mailto:contato@maisagro.com.br)
🌐 **Site:** [www.maisagro.com.br](https://www.maisagro.com.br)
"""

    BOX_INICIAL = """
> 📌 **Destaque do Artigo**
>
> *[Resumo executivo de 2-3 linhas com os pontos mais importantes]*
"""

    # Configurações
    st.header("📋 Configurações do Blog Agrícola")

    col_config1, col_config2 = st.columns(2)

    with col_config1:
        modo_entrada = st.radio("Modo de Entrada:", ["Campos Individuais", "Briefing Completo"])
        numero_palavras = st.slider("Número de Palavras:", min_value=300, max_value=2500, value=1500, step=100)

        st.subheader("🔑 Palavras-chave")
        palavra_chave_principal = st.text_input("Palavra-chave Principal:")
        palavras_chave_secundarias = st.text_area("Palavras-chave Secundárias (separadas por vírgula):")

        st.subheader("🎨 Configurações de Estilo")
        tom_voz = st.selectbox("Tom de Voz:", ["Jornalístico", "Especialista Técnico", "Educativo", "Persuasivo"], key='tom_blog')
        nivel_tecnico = st.selectbox("Nível Técnico:", ["Básico", "Intermediário", "Avançado"])
        abordagem_problema = st.text_area("Aborde o problema de tal forma que:", "seja claro, técnico e focando na solução prática")

    with col_config2:
        st.subheader("🚫 Restrições")
        palavras_proibidas_input = st.text_area(
            "Palavras Proibidas (separadas por vírgula):",
            "melhor, número 1, líder, insuperável, revolucionário, único, exclusivo"
        )
        palavras_proibidas_lista = [p.strip().lower() for p in palavras_proibidas_input.split(",") if p.strip()]

        st.subheader("📐 Estrutura do Texto")
        estrutura_opcoes = st.multiselect(
            "Seções do Post:",
            ["Introdução", "Problema/Desafio", "Solução/Produto", "Benefícios", "Implementação Prática", "Considerações Finais", "Fontes"],
            default=["Introdução", "Problema/Desafio", "Solução/Produto", "Benefícios", "Implementação Prática"]
        )

        st.subheader("📦 KBF de Produtos")
        kbf_produtos = carregar_kbf_produtos()
        if kbf_produtos:
            produtos_disponiveis = [prod['nome'] for prod in kbf_produtos]
            produto_selecionado = st.selectbox("Selecionar Produto do KBF:", ["Nenhum"] + produtos_disponiveis)
        else:
            st.info("Nenhum KBF cadastrado")

    # Campos do blog
    if modo_entrada == "Campos Individuais":
        col1, col2 = st.columns(2)

        with col1:
            st.header("📝 Informações Básicas")
            titulo_blog = st.text_input("Título do Blog:", "Proteja sua soja de nematoides")
            cultura = st.text_input("Cultura:", "Soja")
            editoria = st.text_input("Editoria:", "Manejo e Proteção")
            mes_publicacao = st.text_input("Mês de Publicação:", "08/2025")
            objetivo_post = st.text_area("Objetivo do Post:", "Explicar a importância do manejo de nematoides")
            url = st.text_input("URL:", "/manejo-e-protecao/proteja-sua-soja")

            st.header("🔧 Conteúdo Técnico")
            problema_principal = st.text_area("Problema Principal/Contexto:")
            pragas_alvo = st.text_area("Pragas/Alvo Principal:")
            danos_causados = st.text_area("Danos Causados:")

        with col2:
            st.header("🏭 Informações da Empresa")
            nome_empresa = st.text_input("Nome da Empresa/Marca:")
            nome_central = st.text_input("Nome da Central de Conteúdos:")

            st.header("💡 Soluções e Produtos")
            nome_produto = st.text_input("Nome do Produto:")
            principio_ativo = st.text_input("Princípio Ativo/Diferencial:")
            beneficios_produto = st.text_area("Benefícios do Produto:")
            espectro_acao = st.text_area("Espectro de Ação:")
            modo_acao = st.text_area("Modo de Ação:")
            aplicacao_pratica = st.text_area("Aplicação Prática:")

            st.header("🎯 Diretrizes Específicas")
            diretrizes_usuario = st.text_area("Diretrizes Adicionais:", "NÃO INVENTE SOLUÇÕES. Use apenas informações fornecidas.")
            fontes_pesquisa = st.text_area("Fontes para Pesquisa/Referência:", "Embrapa Soja, ESALQ")
    else:
        st.header("📄 Briefing Completo")
        briefing_texto = st.text_area("Cole aqui o briefing completo:", height=300)

    # Metadados SEO
    st.header("🔍 Metadados para SEO")
    col_meta1, col_meta2 = st.columns(2)

    with col_meta1:
        meta_title = st.text_input("Meta Title (máx 60 caracteres):", max_chars=60)
        st.info(f"Caracteres: {len(meta_title)}/60")
        linha_fina = st.text_area("Linha Fina (máx 200 caracteres):", max_chars=200)

    with col_meta2:
        meta_descricao = st.text_area("Meta Descrição (máx 155 caracteres):", max_chars=155)
        st.info(f"Caracteres: {len(meta_descricao)}/155")

    # Configurações avançadas
    with st.expander("⚙️ Configurações Avançadas"):
        col_av1, col_av2 = st.columns(2)
        with col_av1:
            usar_pesquisa_web = st.checkbox("🔍 Habilitar Pesquisa Web", value=False)
            incluir_assinatura = st.checkbox("✍️ Incluir Assinatura Padrão", value=True)
            incluir_box_inicial = st.checkbox("📌 Incluir Box Inicial", value=True)
        with col_av2:
            evitar_repeticao = st.slider("Nível de Evitar Repetição:", 1, 10, 8)
            profundidade_conteudo = st.selectbox("Profundidade:", ["Superficial", "Moderado", "Detalhado", "Especializado"])
            max_paragrafos = st.slider("Máximo de linhas por parágrafo:", 3, 8, 5)

    # Geração
    st.header("🔄 Geração do Conteúdo")

    if st.button("🚀 Gerar Blog Post", type="primary", use_container_width=True):
        with st.spinner("Gerando conteúdo..."):
            try:
                palavras_proibidas_efetivas = ", ".join(palavras_proibidas_lista)

                regras_base = f'''
                **REGRAS DE REPLICAÇÃO:**

                1. **ESTRUTURA:** Título, Box inicial, Linha fina, Meta-title, Meta-descrição, Seções
                2. **LINGUAGEM:** Tom {tom_voz}, nível {nivel_tecnico}
                3. **ELEMENTOS TÉCNICOS:** Nomes científicos, fontes confiáveis, dados concretos
                4. **FORMATAÇÃO:** Parágrafos curtos (máx {max_paragrafos} linhas), listas de até 5 itens
                5. **RESTRIÇÕES:** PALAVRAS PROIBIDAS: {palavras_proibidas_efetivas}
                6. **NÚMERO DE PALAVRAS:** {numero_palavras} (±5%)
                '''

                instrucoes_estrutura = ""
                if incluir_box_inicial:
                    instrucoes_estrutura += f"\n\n**BOX INICIAL:**\n{BOX_INICIAL}"
                if incluir_assinatura:
                    instrucoes_estrutura += f"\n\n**ASSINATURA:**\n{ASSINATURA_PADRAO}"

                prompt_final = f"""
                {regras_base}

                **INFORMAÇÕES:**
                - Título: {titulo_blog if modo_entrada == "Campos Individuais" else "Extrair do briefing"}
                - Cultura: {cultura if modo_entrada == "Campos Individuais" else "Extrair do briefing"}
                - Palavra-chave Principal: {palavra_chave_principal}
                - Palavras-chave Secundárias: {palavras_chave_secundarias}

                {instrucoes_estrutura}

                **METADADOS:**
                - Meta Title: {meta_title}
                - Meta Description: {meta_descricao}
                - Linha Fina: {linha_fina}

                **DIRETRIZES:**
                - NÃO INVENTE SOLUÇÕES
                - Cite fontes específicas
                - Estrutura: {', '.join(estrutura_opcoes)}
                - Profundidade: {profundidade_conteudo}

                {f"**PRODUTO:** {nome_produto}" if modo_entrada == "Campos Individuais" and nome_produto else ""}
                {f"**BRIEFING:** {briefing_texto}" if modo_entrada != "Campos Individuais" else ""}

                Gere um conteúdo {profundidade_conteudo.lower()} com EXATAMENTE {numero_palavras} palavras (±5%).
                """

                resposta = modelo_texto.generate_content(prompt_final)
                texto_gerado = resposta.text

                # Filtrar palavras proibidas
                palavras_encontradas = []
                for palavra in palavras_proibidas_lista:
                    if palavra.lower() in texto_gerado.lower():
                        palavras_encontradas.append(palavra)
                        texto_gerado = texto_gerado.replace(palavra, "[FILTRADO]")
                        texto_gerado = texto_gerado.replace(palavra.capitalize(), "[FILTRADO]")

                if palavras_encontradas:
                    st.warning(f"⚠️ Palavras proibidas filtradas: {', '.join(palavras_encontradas)}")

                palavras_count = len(texto_gerado.split())
                st.info(f"📊 Palavras geradas: {palavras_count} (meta: {numero_palavras})")

                # Salvar no MongoDB
                if modo_entrada == "Campos Individuais":
                    salvar_post(
                        titulo_blog, cultura, editoria, mes_publicacao, objetivo_post, url,
                        texto_gerado, f"{palavra_chave_principal}, {palavras_chave_secundarias}",
                        palavras_proibidas_efetivas, tom_voz, ', '.join(estrutura_opcoes),
                        palavras_count, meta_title, meta_descricao, linha_fina
                    )
                    st.success("✅ Post gerado e salvo!")

                st.subheader("📝 Conteúdo Gerado")
                st.markdown(texto_gerado)

                st.download_button(
                    "💾 Baixar Post",
                    data=texto_gerado,
                    file_name=f"blog_post_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                    mime="text/plain"
                )

            except Exception as e:
                st.error(f"Erro na geração: {str(e)}")

    # Histórico
    st.header("📚 Banco de Textos Gerados")
    posts_anteriores = carregar_posts_anteriores()
    if posts_anteriores:
        for post in posts_anteriores:
            with st.expander(f"{post.get('titulo', 'Sem título')}"):
                st.write(f"**Cultura:** {post.get('cultura', 'N/A')}")
                st.write(f"**Palavras:** {post.get('palavras_contagem', 'N/A')}")
                st.text_area("Conteúdo:", value=post.get('texto_gerado', ''), height=200, key=post['id'])
    else:
        st.info("Nenhum post encontrado.")
