import { Injectable } from '@nestjs/common';
import { InjectModel } from '@nestjs/mongoose';
import { Model } from 'mongoose';
import { ConfigService } from '@nestjs/config';
import { GoogleGenerativeAI } from '@google/generative-ai';
import { Briefing, BriefingDocument } from '../common/schemas/briefing.schema';

@Injectable()
export class BriefingsService {
  private genAI: GoogleGenerativeAI;

  constructor(
    private config: ConfigService,
    @InjectModel(Briefing.name, 'briefings') private briefingModel: Model<BriefingDocument>,
  ) {
    this.genAI = new GoogleGenerativeAI(config.get<string>('GEM_API_KEY') ?? '');
  }

  async listar(limite = 20): Promise<BriefingDocument[]> {
    return this.briefingModel.find().sort({ createdAt: -1 }).limit(limite);
  }
 
  async gerarBriefingDePauta(body: {
    conteudoPauta: string;
    mesReferencia: string;
    contextoAdicional: string;
    contextoAgente: string;
  }): Promise<string> {
    const model = this.genAI.getGenerativeModel({ model: 'gemini-2.5-flash' });
    const prompt = `
    ${body.contextoAgente}

    ## TAREFA: GERAR BRIEFING COMPLETO PARA ESTA PAUTA ESPECÍFICA

    **PAUTA ESPECÍFICA:**
    ${body.conteudoPauta}

    **MÊS DE REFERÊNCIA:** ${body.mesReferencia}

    **CONTEXTO ADICIONAL:**
    ${body.contextoAdicional || 'Nenhum contexto adicional fornecido.'}

    Gere um briefing completo baseado APENAS nesta pauta específica.
    Use a base de conhecimento fornecida para identificar produtos, culturas e informações técnicas.
    Traga informações chave dos produtos exatamente como são, sem alterar o texto. Mas posicione, crie um tema,
    discorra sobre o produto, agregue o tema, após trazer as informações brutas dos produtos que não deve ser alterada,
    o posicione em termos de benefícios, como que ele deve ser discorrido.

    # [TÍTULO DO BRIEFING]

    ## 1. OBJETIVO DO CONTEÚDO
    ## 2. PÚBLICO-ALVO
    ## 3. TEMA PRINCIPAL E ABORDAGEM
    ## 4. PRODUTOS ENVOLVIDOS
    ## 5. CULTURAS ALVO
    ## 6. PONTOS-CHAVE OBRIGATÓRIOS
    ## 7. TOM DE VOZ E ESTILO
    ## 8. FORMATOS SUGERIDOS
    ## 9. PALAVRAS-CHAVE (SEO)
    ## 10. CALL TO ACTION (CTA) SUGERIDO
    ## 11. INFORMAÇÕES TÉCNICAS RELEVANTES
    ## 12. RESTRIÇÕES E CUIDADOS
    ## 13. REFERÊNCIAS SUGERIDAS

    Seja detalhado e específico. Quando trouxer informações de produtos, os traga exatamente como são sem reescrita.
    `;

    const result = await model.generateContent(prompt);
    return result.response.text().replace(/```/g, '').trim();
  }

  async gerarBriefingIndividual(body: {
    titulo: string;
    mesReferencia: string;
    textoBase: string;
    contextoAdicional: string;
    contextoAgente: string;
  }): Promise<string> {
    const model = this.genAI.getGenerativeModel({ model: 'gemini-2.5-flash' });
    const prompt = `
    ${body.contextoAgente}

    ## TAREFA: GERAR BRIEFING COMPLETO E ESTRUTURADO

    **TÍTULO DO BRIEFING:** ${body.titulo}
    **MÊS DE REFERÊNCIA:** ${body.mesReferencia}

    **TEXTO BASE:**
    ${body.textoBase}

    **CONTEXTO ADICIONAL:**
    ${body.contextoAdicional || 'Nenhum contexto adicional fornecido.'}

    ## INSTRUÇÕES PARA O FORMATO DO BRIEFING:
    Traga informações chave dos produtos exatamente como são, sem alterar o texto. Mas posicione, crie um tema,
    discorra sobre o produto, agregue o tema, após trazer as informações brutas dos produtos que não deve ser alterada,
    o posicione em termos de benefícios, como que ele deve ser discorrido.

    Gere um briefing completo seguindo EXATAMENTE esta estrutura:

    # [TÍTULO DO BRIEFING]

    ## 1. OBJETIVO DO CONTEÚDO
    ## 2. PÚBLICO-ALVO
    ## 3. TEMA PRINCIPAL E ABORDAGEM
    ## 4. PRODUTOS ENVOLVIDOS
    ## 5. CULTURAS ALVO
    ## 6. PONTOS-CHAVE OBRIGATÓRIOS
    ## 7. TOM DE VOZ E ESTILO
    ## 8. FORMATOS SUGERIDOS
    ## 9. PALAVRAS-CHAVE (SEO)
    ## 10. CALL TO ACTION (CTA) SUGERIDO
    ## 11. INFORMAÇÕES TÉCNICAS RELEVANTES
    ## 12. RESTRIÇÕES E CUIDADOS
    ## 13. REFERÊNCIAS SUGERIDAS

    Seja detalhado e específico. Quando trouxer informações de produtos, os traga exatamente como são sem reescrita.
    `;

    const result = await model.generateContent(prompt);
    return result.response.text().replace(/```/g, '').trim();
  }

  async ajustarBriefing(body: {
    briefingAtual: string;
    ajuste: string;
    tituloOuPauta: string;
    mesReferencia: string;
    contextoAgente: string;
  }): Promise<string> {
    const model = this.genAI.getGenerativeModel({ model: 'gemini-2.5-flash' });
    const prompt = `
    ${body.contextoAgente}

    ## INSTRUÇÕES: AJUSTE PONTUAL DO BRIEFING
    ## MANTENHA A ESTRUTURA ORIGINAL - ALTERE APENAS O SOLICITADO

    ### BRIEFING ORIGINAL COMPLETO:
    ${body.briefingAtual}

    ### SOLICITAÇÃO ESPECÍFICA DE AJUSTE:
    "${body.ajuste}"

    ## INFORMAÇÕES DE CONTEXTO:
    **Título/Pauta original:** ${body.tituloOuPauta}
    **Mês de referência:** ${body.mesReferencia}

    ## REGRAS ABSOLUTAS:
    1. MANTENHA A ESTRUTURA ORIGINAL COMPLETA - NÃO remova seções, NÃO adicione novas seções
    2. ALTERE APENAS O ESTRITAMENTE SOLICITADO
    3. PRESERVE FORMATAÇÃO E ESTILO

    RETORNE APENAS O BRIEFING AJUSTADO, SEM COMENTÁRIOS ADICIONAIS.
    `;

    const result = await model.generateContent(prompt);
    return result.response.text().replace(/```/g, '').trim();
  }

  extrairPautasDoCSV(csvText: string): Array<{ conteudo: string; linha: number; coluna: number; indice: number }> {
    const HEADERS = ['DOMINGO', 'SEGUNDA', 'TERÇA', 'QUARTA', 'QUINTA', 'SEXTA', 'SÁBADO', 'CALENDÁRIO'];
    const pautas: Array<{ conteudo: string; linha: number; coluna: number; indice: number }> = [];

    const linhas = csvText.split('\n');
    for (let linhaNum = 0; linhaNum < linhas.length; linhaNum++) {
      const linhaLimpa = linhas[linhaNum].replace(/\r/g, '').replace(/\ufeff/g, '').trim();
      if (!linhaLimpa) continue;

      const celulas = linhaLimpa.split(',');
      for (let celulaNum = 0; celulaNum < celulas.length; celulaNum++) {
        const celulaLimpa = celulas[celulaNum].trim();

        const isHeader = HEADERS.some((h) => celulaLimpa.includes(h));
        const isNumeric = celulaLimpa.replace(/\./g, '').match(/^\d+$/);
        const isCX = celulaLimpa.includes('CX,');

        if (celulaLimpa && celulaLimpa.length > 15 && !isHeader && !isNumeric && !isCX) {
          const subPautas = celulaLimpa.includes('\n')
            ? celulaLimpa.split('\n').filter((p) => p.trim().length > 15)
            : [celulaLimpa];

          for (const sub of subPautas) {
            const pautaLimpa = sub.trim().replace(/\s+/g, ' ');
            pautas.push({ conteudo: pautaLimpa, linha: linhaNum, coluna: celulaNum, indice: pautas.length + 1 });
          }
        }
      }
    }

    return pautas;
  }

  async gerarBriefingsDePautas(body: {
    csvText: string;
    mesReferencia: string;
    contextoAdicional: string;
    contextoAgente: string;
  }): Promise<Array<{ indice: number; conteudoOriginal: string; briefing: string; mesReferencia: string }>> {
    const pautas = this.extrairPautasDoCSV(body.csvText);
    const resultados: Array<{ indice: number; conteudoOriginal: string; briefing: string; mesReferencia: string }> = [];

    for (const pauta of pautas) {
      try {
        const briefing = await this.gerarBriefingDePauta({
          conteudoPauta: pauta.conteudo,
          mesReferencia: body.mesReferencia,
          contextoAdicional: body.contextoAdicional,
          contextoAgente: body.contextoAgente,
        });
        resultados.push({
          indice: pauta.indice,
          conteudoOriginal: pauta.conteudo,
          briefing,
          mesReferencia: body.mesReferencia,
        });
      } catch (e: unknown) {
        resultados.push({
          indice: pauta.indice,
          conteudoOriginal: pauta.conteudo,
          briefing: `ERRO: Não foi possível gerar o briefing.\n${(e as Error).message}`,
          mesReferencia: body.mesReferencia,
        });
      }
    }

    return resultados;
  }
}
