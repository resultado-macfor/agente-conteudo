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

  async gerarBriefing(body: { descricao: string; contextoAgente: string }): Promise<string> {
    const model = this.genAI.getGenerativeModel({ model: 'gemini-2.5-flash' });
    const prompt = `${body.contextoAgente}

Gere um briefing completo e estruturado com base na seguinte descrição:
${body.descricao}

O briefing deve incluir:
1. Título do projeto
2. Objetivo principal
3. Público-alvo
4. Tom de voz
5. Palavras-chave principais
6. Palavras-chave secundárias
7. Estrutura do conteúdo (H1, H2, H3)
8. Pontos obrigatórios a abordar
9. Produtos/serviços a mencionar
10. Call-to-action
11. Observações técnicas
12. Fontes sugeridas
13. Número de palavras estimado

Retorne o briefing completo e formatado.`;

    const result = await model.generateContent(prompt);
    return result.response.text();
  }

  async ajustarBriefing(body: { briefingAtual: string; ajuste: string; contextoAgente: string }): Promise<string> {
    const model = this.genAI.getGenerativeModel({ model: 'gemini-2.5-flash' });
    const prompt = `${body.contextoAgente}

BRIEFING ATUAL:
${body.briefingAtual}

AJUSTES SOLICITADOS:
${body.ajuste}

Aplique os ajustes e retorne o briefing completo atualizado.`;

    const result = await model.generateContent(prompt);
    return result.response.text();
  }

  async gerarMultiplosBriefings(body: { descricoes: string[]; contextoAgente: string }): Promise<string[]> {
    return Promise.all(
      body.descricoes.map((descricao) =>
        this.gerarBriefing({ descricao, contextoAgente: body.contextoAgente }),
      ),
    );
  }
}
