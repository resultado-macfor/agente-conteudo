import { Injectable } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { AstraService } from './astra.service';
import OpenAI from 'openai';

@Injectable()
export class RagService {
  private openai: OpenAI;
  private collection: string;

  constructor(
    private config: ConfigService,
    private astra: AstraService,
  ) {
    this.openai = new OpenAI({ apiKey: config.get<string>('OPENAI_API_KEY') ?? '' });
    this.collection = config.get<string>('ASTRA_DB_COLLECTION') ?? 'documents';
  }

  async getEmbedding(text: string): Promise<number[]> {
    const response = await this.openai.embeddings.create({
      input: text.slice(0, 800),
      model: 'text-embedding-3-small',
    });
    return response.data[0].embedding;
  }

  async ragTaxonomia(texto: string, limite = 12): Promise<unknown[]> {
    const perguntas = [
      'classificação taxonômica', 'fungo ou oomiceto', 'nome científico patógeno',
      'reino filo classe ordem', 'agente causal doença', 'oomiceto vs fungo diferença',
    ];
    return this.buscaMultiQuery(texto, perguntas, limite);
  }

  async ragEpidemiologia(texto: string, limite = 12): Promise<unknown[]> {
    const perguntas = [
      'condições ambientais doença', 'temperatura umidade molhamento foliar',
      'condições ideais infecção', 'epidemiologia doença plantas',
      'período molhamento temperatura ótima', 'fatores epidemiológicos',
    ];
    return this.buscaMultiQuery(texto, perguntas, limite);
  }

  async ragProdutos(texto: string, limite = 12): Promise<unknown[]> {
    const perguntas = [
      'modo de ação produto', 'aplicação dose recomendada',
      'eficácia controle doença', 'características técnicas produto',
      'benefícios produto agrícola', 'recomendações uso produto',
    ];
    return this.buscaMultiQuery(texto, perguntas, limite);
  }

  async ragGeral(texto: string, limite = 12): Promise<unknown[]> {
    const embedding = await this.getEmbedding(texto);
    return this.astra.vectorSearch(this.collection, embedding, limite);
  }

  async processarRags(texto: string, rags: { taxonomia?: boolean; epidemiologia?: boolean; produtos?: boolean; geral?: boolean }, limite = 12) {
    const resultados: Record<string, unknown[]> = {};
    if (rags.taxonomia) resultados.taxonomia = await this.ragTaxonomia(texto, limite);
    if (rags.epidemiologia) resultados.epidemiologia = await this.ragEpidemiologia(texto, limite);
    if (rags.produtos) resultados.produtos = await this.ragProdutos(texto, limite);
    if (rags.geral) resultados.geral = await this.ragGeral(texto, limite);
    return resultados;
  }

  buildRagsContext(resultados: Record<string, unknown[]>): string {
    let context = '## DOCUMENTOS TÉCNICOS DE REFERÊNCIA:\n\n';
    for (const [categoria, documentos] of Object.entries(resultados)) {
      if (documentos?.length) {
        context += `### ${categoria.toUpperCase()} (${documentos.length} documentos):\n`;
        for (const doc of documentos) {
          const docStr = JSON.stringify(doc).slice(0, 300);
          context += `- ${docStr}\n`;
        }
        context += '\n';
      }
    }
    return context;
  }

  private async buscaMultiQuery(texto: string, perguntas: string[], limite: number): Promise<unknown[]> {
    const docs: unknown[] = [];
    const idsVistos = new Set<string>();
    const perQuery = Math.max(1, Math.floor(limite / perguntas.length));

    for (const pergunta of perguntas) {
      const query = `${texto.slice(0, 200)} ${pergunta}`;
      const embedding = await this.getEmbedding(query);
      const resultados = await this.astra.vectorSearch(this.collection, embedding, perQuery);
      for (const doc of resultados) {
        const id = String((doc as Record<string, unknown>)._id ?? '');
        if (!idsVistos.has(id)) {
          docs.push(doc);
          idsVistos.add(id);
        }
      }
    }
    return docs.slice(0, limite);
  }
}
