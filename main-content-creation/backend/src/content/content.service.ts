import { Injectable } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { GoogleGenerativeAI } from '@google/generative-ai';
import { RagService } from '../rag/rag.service';
import axios from 'axios';

@Injectable()
export class ContentService {
  private genAI: GoogleGenerativeAI;

  constructor(
    private config: ConfigService,
    private ragService: RagService,
  ) {
    this.genAI = new GoogleGenerativeAI(config.get<string>('GEM_API_KEY') ?? '');
  }

  async gerarConteudo(body: {
    contextoAgente: string;
    tipoConteudo: string;
    tomVoz: string;
    palavrasChave: string;
    numeroPalavras: number;
    nivelDetalhe: string;
    incluirCta: boolean;
    formatoSaida: string;
    instrucoes: string;
    fontesTexto: string;
  }): Promise<string> {
    const model = this.genAI.getGenerativeModel({ model: 'gemini-2.5-flash' });
    const prompt = `${body.contextoAgente}

## INSTRUÇÕES PARA GERAÇÃO DE CONTEÚDO:
**TIPO DE CONTEÚDO:** ${body.tipoConteudo}
**TOM DE VOZ:** ${body.tomVoz}
**PALAVRAS-CHAVE:** ${body.palavrasChave || 'Não especificadas'}
**NÚMERO DE PALAVRAS:** ${body.numeroPalavras} (±10%)
**NÍVEL DE DETALHE:** ${body.nivelDetalhe}
**INCLUIR CALL-TO-ACTION:** ${body.incluirCta}
**INSTRUÇÕES ESPECÍFICAS:** ${body.instrucoes || 'Nenhuma'}

## FONTES E REFERÊNCIAS:
${body.fontesTexto}

## TAREFA:
Gere um conteúdo do tipo ${body.tipoConteudo} sintetizando todas as fontes acima. Formato de saída: ${body.formatoSaida}.`;

    const result = await model.generateContent(prompt);
    return result.response.text();
  }

  async revisaoOrtografica(texto: string, contextoAgente: string): Promise<string> {
    const model = this.genAI.getGenerativeModel({ model: 'gemini-2.5-flash' });
    const prompt = `${contextoAgente}

Faça uma revisão ortográfica e gramatical completa do seguinte texto:

###BEGIN TEXTO A SER REVISADO###
${texto}
###END TEXTO A SER REVISADO###

MANTENHA A ESTRUTURA DO TEXTO ORIGINAL. APENAS CORRIJA ERROS ORTOGRÁFICOS (SE PRESENTES) E APONTE QUAIS FORAM OS ERROS CORRIGIDOS`;

    const result = await model.generateContent(prompt);
    return result.response.text();
  }

  async revisaoTecnicaComRag(body: {
    texto: string;
    rags: Record<string, boolean>;
    limite: number;
    contextoAgente: string;
    incluirRelatorio: boolean;
  }): Promise<{ textoReescrito: string; relatorioMudancas: string; resultadosRags: Record<string, unknown[]> }> {
    const model = this.genAI.getGenerativeModel({ model: 'gemini-2.5-flash' });
    const resultadosRags = await this.ragService.processarRags(body.texto, body.rags, body.limite);
    const contextoRags = this.ragService.buildRagsContext(resultadosRags);

    if (body.incluirRelatorio) {
      const prompt = `${body.contextoAgente}

## TEXTO ORIGINAL PARA REESCRITA:
${body.texto}

## BASE TÉCNICA DE REFERÊNCIA:
${contextoRags}

## INSTRUÇÕES CRÍTICAS:

**SUA TAREFA:**
1. Reescrever o texto original aplicando correções técnicas baseadas nos documentos de referência
2. Gerar um relatório DETALHADO de TODAS as mudanças realizadas
3. Você deve manter a estrutura original do texto. Você deve realizar apenas mudanças e enriquecimentos conforme o contexto novo vindo da base técnica de referência. O texto original deve sempre ser o molde a ser seguido.

**FORMATO DE SAÍDA EXIGIDO (use exatamente esta estrutura):**

### 📝 TEXTO REESCRITO
[AQUI VOCÊ COLA O TEXTO COMPLETO REESCRITO E CORRIGIDO]

### 🔍 RELATÓRIO DETALHADO DE MUDANÇAS

#### 📊 RESUMO EXECUTIVO
- Total de correções aplicadas: [N]
- Principais categorias de ajustes: [lista categorias]
- Impacto na precisão técnica: [Alto/Médio/Baixo]

#### 📋 MUDANÇAS DETALHADAS

**1. CORREÇÕES TAXONÔMICAS:**
[Lista cada correção taxonômica no formato:
- **Original:** "texto original"
- **Corrigido:** "texto corrigido"
- **Justificativa:** explicação técnica baseada nos documentos]

**2. PRECISÃO EPIDEMIOLÓGICA:**
[Lista cada correção epidemiológica no formato:
- **Original:** "texto original"
- **Corrigido:** "texto corrigido"
- **Justificativa:** explicação com base científica]

**3. INFORMAÇÕES DE PRODUTOS:**
[Lista cada correção de produtos no formato:
- **Original:** "texto original"
- **Corrigido:** "texto corrigido"
- **Justificativa:** ajuste técnico necessário]

**4. TERMINOLOGIA TÉCNICA:**
[Lista cada ajuste de terminologia no formato:
- **Original:** "termo vago/impreciso"
- **Corrigido:** "termo técnico preciso"
- **Justificativa:** padronização técnica]

**5. DADOS E ESTATÍSTICAS:**
[Lista cada correção de dados no formato:
- **Original:** "dado impreciso"
- **Corrigido:** "dado corrigido"
- **Justificativa:** fonte/documento de referência]

#### 🎯 IMPACTO DAS CORREÇÕES
- Melhorias na precisão científica: [lista específica]
- Ajustes na comunicação técnica: [lista específica]
- Correções de segurança da informação: [lista específica]

**CORREÇÕES TÉCNICAS OBRIGATÓRIAS:**
1. **PRECISÃO TAXONÔMICA:** Corrigir "fungo" para "oomiceto" quando aplicável
2. **ESPECIFICIDADE EPIDEMIOLÓGICA:** Substituir termos vagos por faixas específicas
3. **DESCRIÇÃO PRECISA DE SINTOMAS:** Corrigir descrições imprecisas
4. **MANEJO E TIMING:** Alinhar mensagens sobre timing de aplicação
5. **INFORMAÇÕES DE PRODUTOS:** Corrigir claims imprecisos

**REGRAS ADICIONAIS:**
- Mantenha a estrutura e formatação do original
- Apenas corrija o conteúdo técnico, não reinvente a estrutura
- Para CADA mudança, forneça justificativa técnica específica

**RETORNE EXATAMENTE no formato especificado acima.**`;

      const result = await model.generateContent(prompt);
      const full = result.response.text();
      if (full.includes('### 📝 TEXTO REESCRITO') && full.includes('### 🔍 RELATÓRIO DETALHADO DE MUDANÇAS')) {
        const parts = full.split('### 🔍 RELATÓRIO DETALHADO DE MUDANÇAS');
        return {
          textoReescrito: parts[0].replace('### 📝 TEXTO REESCRITO', '').trim(),
          relatorioMudancas: '### 🔍 RELATÓRIO DETALHADO DE MUDANÇAS' + parts[1],
          resultadosRags,
        };
      }
      return { textoReescrito: full, relatorioMudancas: '', resultadosRags };
    } else {
      const prompt = `${body.contextoAgente}

## TEXTO ORIGINAL PARA REESCRITA:
${body.texto}

## BASE TÉCNICA DE REFERÊNCIA:
${contextoRags}

**REESCREVA o texto aplicando correções técnicas baseadas nos documentos.**
**RETORNE APENAS o texto reescrito, sem comentários ou relatórios.**

Correções obrigatórias:
- Precisão taxonômica (fungo vs oomiceto)
- Especificidade epidemiológica (temperaturas, umidades)
- Informações precisas de produtos
- Terminologia técnica adequada

Mantenha a estrutura original.`;
      const result = await model.generateContent(prompt);
      return { textoReescrito: result.response.text().trim(), relatorioMudancas: '', resultadosRags };
    }
  }

  async revisaoTecnicaSemRag(body: { texto: string; contextoAgente: string; ajuste?: string }): Promise<string> {
    const model = this.genAI.getGenerativeModel({ model: 'gemini-2.5-pro' });

    const textoBase = body.ajuste
      ? `VOCÊ É: Um especialista técnico agrícola.
SUA TAREFA: Ajustar a revisão técnica anterior com base nas solicitações específicas.

TEXTO ORIGINAL:
${body.texto}

SOLICITAÇÕES DE AJUSTE:
${body.ajuste}

INSTRUÇÕES:
1. Aplique TODOS os ajustes solicitados
2. Mantenha a precisão técnica
3. Retorne o texto reescrito ajustado.`
      : `${body.contextoAgente}

Faça uma revisão técnica profissional completa do seguinte texto:

###BEGIN TEXTO A SER REVISADO###
${body.texto}
###END TEXTO A SER REVISADO###

INSTRUÇÕES:
1. Identifique e corrija todas as imprecisões técnicas
2. Mantenha a estrutura original do texto
3. Aponte as correções realizadas com justificativas técnicas
4. Inclua um relatório estruturado das mudanças ao final`;

    const result = await model.generateContent(textoBase);
    return result.response.text();
  }

  async otimizacaoSEO(body: {
    briefing: string;
    conteudoOriginal: string;
    contextoAgente: string;
    avaliacao?: string;
    fontes?: string;
    nivelHeading: string;
    qtdInternos: number;
    qtdExternos: number;
  }): Promise<string> {
    const model = this.genAI.getGenerativeModel({ model: 'gemini-2.5-flash' });

    const contextoBloco = body.contextoAgente ? `###CONTEXTO DO AGENTE###\n${body.contextoAgente}\n###FIM DO CONTEXTO###\n\n` : '';
    const briefingBloco = body.briefing
      ? `###BRIEFING DE REFERÊNCIA###\n${body.briefing}\n###FIM DO BRIEFING###\n`
      : '(Briefing não fornecido — avalie apenas o conteúdo.)';
    const conteudoBloco = body.conteudoOriginal
      ? `###CONTEÚDO ORIGINAL###\n${body.conteudoOriginal}\n###FIM DO CONTEÚDO ORIGINAL###\n`
      : '(Conteúdo original não fornecido — gere a partir do briefing.)';
    const fontesBloco = body.fontes ? `---\n###FONTES WEB###\n${body.fontes}\n###FIM DAS FONTES###\n` : '';

    const isAvaliacao = !body.avaliacao;

    const prompt = isAvaliacao
      ? `Você é um editor sênior especializado em SEO para o agronegócio, revisando conteúdos para o portal Mais Agro da Syngenta.

Analise o CONTEÚDO ORIGINAL abaixo e gere um relatório detalhado de melhorias necessárias, avaliando CADA critério da lista abaixo.

${contextoBloco}---
${briefingBloco}

---
${conteudoBloco}

---
## CRITÉRIOS DE AVALIAÇÃO — verifique cada um:

### 1. METADADOS SEO
- Existe META TITLE (até 60 caracteres com KW principal)?
- Existe META DESCRIPTION (até 155 caracteres com KW e CTA)?
- Existe URL slug amigável, CATEGORIA e ALT TEXT CAPA?

### 2. ESTRUTURA E HEADINGS
- O H1 corresponde exatamente ao título do briefing?
- As seções H2/H3 seguem a estrutura do briefing, na ordem correta?
- Há headings extras não previstos no briefing?

### 3. INTRODUÇÃO
- A introdução começa com dado factual ou afirmação direta?
- Contém frases genéricas proibidas ("No dinâmico cenário...", "Neste artigo vamos...")?
- Apresenta o leitor explicitamente ("Se você é produtor...")?

### 4. QUALIDADE E PROFUNDIDADE
- O conteúdo é raso ou superficial? Falta aprofundamento técnico?
- Os produtos Syngenta são qualificados adequadamente (nome comercial, registro, modo de ação, benefícios)?
- Há informações repetidas entre seções?
- Os parágrafos têm mais de 3 frases ou frases muito longas (>20 palavras)?
- Listas de 3+ itens usam bullet points?

### 5. NEGRITO E FORMATAÇÃO
- A palavra-chave principal está em negrito em MAIS de uma ocorrência? (deve aparecer em negrito APENAS na 1ª vez)
- O negrito é usado em excesso ou em contextos inadequados?
- Existem asteriscos literais \`**palavra**\` aparecendo como texto (negrito não renderizado)?
- Há tags HTML no texto (\`<strong>\`, \`<br>\`, \`<p>\`, \`<a>\`, etc.)?

### 6. LINKS INTERNOS (Mais Agro)
- Existem exatamente ${body.qtdInternos} links internos \`maisagro.syngenta.com.br\`?
- Estão ancorados nos parágrafos onde o tema é discutido (não agrupados no final)?
- Os textos âncora são descritivos e relevantes?

### 7. LINKS EXTERNOS
- Existem exatamente ${body.qtdExternos} links externos de fontes neutras (Embrapa, universidades, institutos)?
- Há links para concorrentes da Syngenta (BASF, Bayer, Corteva, FMC, UPL, Adama, Helm, Nufarm)?
- Há URLs cruas no texto ou formato \`texto → URL\` com seta?
- Há citações numéricas estilo Wikipedia \`[1]\`, \`[2]\`?

### 8. CTA FINAL
- A CTA do briefing está presente e com o texto exato?
- O link da CTA está ancorado corretamente?
- A última frase referencia a central Mais Agro?

---
## FORMATO DO RELATÓRIO:

Para cada critério com problema, escreva:
**[CRITÉRIO]** — Problema identificado: [descrição objetiva do que está errado]
Correção necessária: [o que exatamente deve ser feito]

Ao final, escreva:
**RESUMO:** X problemas críticos, Y melhorias recomendadas.
**PRIORIDADE ALTA:** [lista dos 3 problemas mais impactantes para SEO e qualidade]`
      : `Você é um especialista em SEO e redação técnica para o agronegócio, escrevendo para o portal Mais Agro da Syngenta.

Reescreva e otimize o CONTEÚDO ORIGINAL aplicando TODOS os ajustes indicados no RELATÓRIO DE AVALIAÇÃO, seguindo o BRIEFING como referência estrutural.

${contextoBloco}---
${briefingBloco}

---
${conteudoBloco}

---
###RELATÓRIO DE AVALIAÇÃO (aplique TODOS os pontos)###
${body.avaliacao}
###FIM DO RELATÓRIO###

${fontesBloco}

---
## INSTRUÇÕES DE GERAÇÃO

### 1. METADADOS SEO — coloque SEMPRE no topo, antes do H1
\`\`\`
META TITLE: [até 60 caracteres, com KW principal]
META DESCRIPTION: [até 155 caracteres, com KW e chamada para ação]
URL: /[slug-amigavel-baseado-no-h1]
CATEGORIA: [categoria sugerida]
ALT TEXT CAPA: [texto descritivo com KW]
\`\`\`

### 2. ESTRUTURA DO ARTIGO
- H1 exatamente como indicado no briefing
- Heading de partida do corpo: **${body.nivelHeading}**
- Seções na ordem exata do briefing — nada a mais, nada a menos
- Cada seção: 2 a 4 parágrafos + bullets quando há 3+ itens paralelos
- Bullets: \`- **Termo:** explicação da característica\`
- Listas numeradas para sequências de etapas (1. 2. 3.)

### 3. INTRODUÇÃO
- Comece com dado factual ou afirmação direta sobre o tema
- PROIBIDO: aberturas genéricas, apresentar o leitor, frases meta

### 4. QUALIDADE TEXTUAL
- Parágrafos com NO MÁXIMO 3 frases curtas (~20 palavras)
- KW principal em negrito SOMENTE na 1ª ocorrência — nas demais, sem negrito
- Cada informação aparece UMA única vez — sem repetição entre seções
- Qualifique os produtos Syngenta: nome comercial + registro + modo de ação + benefícios

### 5. LINKS
**Formato obrigatório:** \`[texto descritivo](https://url-completa.com)\`

**PROIBIDO:** URLs cruas, formato \`texto → URL\`, citações \`[n]\`, concorrentes Syngenta (BASF, Bayer, Corteva, FMC, UPL, Adama, Helm, Nufarm)

**Internos (Mais Agro) — exatamente ${body.qtdInternos} distribuídos no corpo:**
- Ancorados no parágrafo onde o tema é discutido
- Padrão: \`https://maisagro.syngenta.com.br/[slug]\`
- Exemplo: O [manejo integrado de plantas daninhas](https://maisagro.syngenta.com.br/manejo-plantas-daninhas) reduz o banco de sementes...

**Externos — exatamente ${body.qtdExternos} distribuídos no corpo:**
- Apenas Embrapa, universidades, institutos governamentais
- Ancorados no dado/estudo: \`De acordo com [pesquisa da Embrapa](https://embrapa.br/...), a espécie...\`

### 6. PRODUTOS SYNGENTA
- Seção própria antes da CTA com heading específico
- Mencione nome comercial + registro (ex: CALARIS®, Dual Gold®, Grover®)
- Tom: "A Syngenta oferece qualidade e tecnologia..."

### 7. CTA FINAL
- Texto exato do briefing
- Última frase: "Confira a central de conteúdos [Mais Agro](URL-da-CTA) para ficar por dentro de tudo o que está acontecendo no campo."

### 8. FORMATAÇÃO — PROIBIÇÕES ABSOLUTAS
- NUNCA tags HTML: \`<strong>\`, \`<em>\`, \`<b>\`, \`<i>\`, \`<a>\`, \`<br>\`, \`<p>\`, \`<ul>\`, \`<li>\`
- NUNCA asteriscos literais como texto — use Markdown: \`**negrito**\`
- Markdown puro em todo o artigo

---
**LEMBRETE:** Markdown puro, zero HTML, zero \`[n]\`, zero concorrentes, links ancorados nos parágrafos, CTA com link no final.

Gere o artigo completo.`;

    const result = await model.generateContent(prompt);
    return result.response.text();
  }

  async buscarPerplexity(texto: string): Promise<string> {
    const apiKey = this.config.get<string>('PERP_API_KEY');
    if (!apiKey) return 'PERP_API_KEY não configurada';
    try {
      const response = await axios.post(
        'https://api.perplexity.ai/chat/completions',
        {
          model: 'sonar',
          messages: [{ role: 'user', content: `Busque informações técnicas atualizadas e confiáveis sobre agronegócio: ${texto.slice(0, 800)}\nRetorne fontes com URL, conteúdo e relevância.` }],
          temperature: 0,
          max_tokens: 20000,
        },
        { headers: { Authorization: `Bearer ${apiKey}`, 'Content-Type': 'application/json' } },
      );
      return response.data.choices[0].message.content.replace(/\s*\[\d+\]/g, '');
    } catch (e) {
      return `Erro na busca Perplexity: ${e.message}`;
    }
  }
}
