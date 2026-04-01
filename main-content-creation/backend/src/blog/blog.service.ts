import { Injectable } from '@nestjs/common';
import { InjectModel } from '@nestjs/mongoose';
import { Model } from 'mongoose';
import { ConfigService } from '@nestjs/config';
import { GoogleGenerativeAI } from '@google/generative-ai';
import axios from 'axios';
import { BlogPost, BlogPostDocument } from '../common/schemas/blog-post.schema';

@Injectable()
export class BlogService {
  private genAI: GoogleGenerativeAI;

  constructor(
    private config: ConfigService,
    @InjectModel(BlogPost.name, 'blog') private blogPostModel: Model<BlogPostDocument>,
  ) {
    this.genAI = new GoogleGenerativeAI(config.get<string>('GEM_API_KEY') ?? '');
  }

  async listarHistorico(limite = 5): Promise<BlogPostDocument[]> {
    return this.blogPostModel.find().sort({ createdAt: -1 }).limit(limite);
  }

  async salvarPost(data: { briefing: string; conteudo: string; fontes: string[]; configuracoes: Record<string, unknown> }): Promise<void> {
    await this.blogPostModel.create(data);
  }

  async buscarPerplexityBlog(briefing: string): Promise<{ resultado: string; fontes: string[] }> {
    const apiKey = this.config.get<string>('PERP_API_KEY');
    if (!apiKey) return { resultado: 'PERP_API_KEY não configurada', fontes: [] };

    const prompt = `Você é um pesquisador agrícola. Busque informações técnicas atualizadas sobre:\n${briefing.slice(0, 800)}
REQUISITOS: Fontes Embrapa, universidades, artigos científicos. Dados concretos. Para cada informação forneça fonte completa.
FORMATO: ## INFORMAÇÕES ENCONTRADAS / ### [Tópico] / - Informação / - Fonte / - URL`;

    try {
      const response = await axios.post(
        'https://api.perplexity.ai/chat/completions',
        { model: 'sonar', messages: [{ role: 'user', content: prompt }], temperature: 0, max_tokens: 20000 },
        { headers: { Authorization: `Bearer ${apiKey}`, 'Content-Type': 'application/json' } },
      );
      const resultado = response.data.choices[0].message.content.replace(/\s*\[\d+\]/g, '');
      const fontes: string[] = [];
      for (const linha of resultado.split('\n')) {
        const urls = linha.match(/(https?:\/\/[^\s)]+)/g) ?? [];
        fontes.push(...urls);
        if (linha.includes('Fonte:') && !linha.includes('[')) fontes.push(linha.trim());
      }
      return { resultado, fontes: [...new Set(fontes)].slice(0, 15) };
    } catch (e) {
      return { resultado: `Erro: ${e.message}`, fontes: [] };
    }
  }

  async gerarBlog(body: {
    briefing: string;
    contextoAgente: string;
    tomVoz: string;
    numeroPalavras: number;
    palavrasChave: string[];
    palavrasPrimeiraLinha: string[];
    densidadePalavras: number;
    nivelHeading: string;
    fontesWeb: string;
  }): Promise<string> {
    const model = this.genAI.getGenerativeModel({ model: 'gemini-2.5-flash' });

    const prompt = `Você é um redator técnico especializado em agronegócio, escrevendo para o portal Mais Agro da Syngenta.

${body.contextoAgente ? `###CONTEXTO DO AGENTE###\n${body.contextoAgente}\n###FIM DO CONTEXTO###\n` : ''}

---
###BRIEFING###
${body.briefing}
###FIM DO BRIEFING###

---
###FONTES WEB (use para enriquecer com dados e links ancorados)###
${body.fontesWeb || 'Nenhuma informação da web disponível.'}
###FIM DAS FONTES###

---
###CONFIGURAÇÕES###
- Tom de voz: ${body.tomVoz}
- Número de palavras: ${body.numeroPalavras} (±10%)
- Palavras-chave: ${body.palavrasChave.join(', ') || 'extraídas do briefing'}
- Densidade de palavras-chave: ${body.densidadePalavras}%
- Palavras obrigatórias na primeira linha: ${body.palavrasPrimeiraLinha.join(', ') || 'não especificadas'}
- Nível de heading do corpo: ${body.nivelHeading}
###FIM DAS CONFIGURAÇÕES###

---
## REGRAS OBRIGATÓRIAS — SIGA À RISCA:

### METADADOS (coloque SEMPRE no topo, antes do H1):
\`\`\`
META TITLE: [até 60 caracteres, com KW principal]
META DESCRIPTION: [até 155 caracteres, com KW e chamada para ação]
URL: /[slug-amigavel]
CATEGORIA: [categoria sugerida]
ALT TEXT CAPA: [texto descritivo com KW]
\`\`\`

### ESTRUTURA EXATA DO ARTIGO — siga esta ordem:

**Bloco 1 — Introdução:**
- 2 a 3 parágrafos diretos e factuais sobre o tema

**Bloco 2 — Corpo do artigo:**
- Use exatamente as seções ${body.nivelHeading} listadas no briefing, nessa ordem
- Cada seção: 2 a 4 parágrafos + bullets quando há 3 ou mais itens paralelos
- Bullets com termo em negrito: \`- **Termo:** explicação da característica\`
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
- Negrito (\`**texto**\`) em: termo técnico na 1ª ocorrência, dados-chave, nome do item em bullet
- A KW principal em negrito SOMENTE na primeira vez — nas demais, sem negrito
- Cada informação aparece UMA única vez — não repita entre seções

### LINKS — regras obrigatórias:
**Formato:** SEMPRE Markdown \`[texto descritivo](https://url-completa.com)\`

**PROIBIDO:**
- URLs cruas no corpo do texto
- Formato \`texto → URL\` com seta
- Numeração estilo Wikipedia \`[1]\`, \`[2]\`
- Qualquer empresa agroquímica que não seja Syngenta: BASF, Bayer, Corteva, FMC, UPL, Adama, Helm, Nufarm

**Links internos (Mais Agro) — 3 a 4 obrigatórios distribuídos no corpo:**
- Ancore cada link no parágrafo onde o tema é mencionado, dentro da frase
- Exemplo correto: O [manejo integrado de plantas daninhas](https://maisagro.syngenta.com.br/manejo-plantas-daninhas) combina práticas culturais e químicas para reduzir o banco de sementes no solo.
- Exemplo errado: lista de links no final ou agrupados numa seção separada
- Use URLs no padrão \`https://maisagro.syngenta.com.br/[slug-do-tema]\`

**Links externos — 2 a 3 obrigatórios no corpo:**
- Somente fontes neutras: Embrapa, universidades, institutos governamentais
- Ancore no dado/estudo: \`De acordo com [pesquisa da Embrapa Soja](https://www.embrapa.br/...), a espécie...\`
- NUNCA "clique aqui" ou anchor genérica

### FORMATAÇÃO — PROIBIÇÕES ABSOLUTAS:
**NUNCA use tags HTML**: \`<strong>\`, \`<em>\`, \`<b>\`, \`<i>\`, \`<a>\`, \`<br>\`, \`<p>\`, \`<ul>\`, \`<li>\`, \`<h1>\` etc.
Use EXCLUSIVAMENTE Markdown puro: \`**negrito**\`, \`[link](url)\`, \`## Heading\`, \`- item\`, \`1. item\`.

### TABELAS:
- Markdown puro — nunca HTML

---
**LEMBRETE FINAL:** Markdown puro, zero HTML, zero \`[n]\` Wikipedia, zero concorrentes Syngenta, links internos ancorados nos parágrafos onde o tema é discutido, CTA com link âncora no final.

Gere o artigo completo seguindo todas as regras acima.`;

    const result = await model.generateContent(prompt);
    return result.response.text();
  }

  async ajustarBlog(body: { conteudoAtual: string; briefingOriginal: string; ajuste: string }): Promise<string> {
    const model = this.genAI.getGenerativeModel({ model: 'gemini-2.5-flash' });
    const prompt = `Você é um redator técnico especializado em agronegócio. Aplique os ajustes solicitados mantendo INTEGRALMENTE as regras do portal Mais Agro da Syngenta.

## CONTEÚDO ATUAL:
${body.conteudoAtual}

## BRIEFING ORIGINAL:
${body.briefingOriginal}

## AJUSTES SOLICITADOS:
${body.ajuste}

Regras: metadados SEO no topo, headings existentes, links ancorados no corpo, CTA final, Markdown puro zero HTML.
RETORNE O CONTEÚDO COMPLETO COM OS AJUSTES APLICADOS.`;

    const result = await model.generateContent(prompt);
    return result.response.text();
  }
}
