import { Injectable } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { GoogleGenerativeAI } from '@google/generative-ai';
import { AgentsService } from '../agents/agents.service';

@Injectable()
export class ChatService {
  private genAI: GoogleGenerativeAI;

  constructor(
    private config: ConfigService,
    private agentsService: AgentsService,
  ) {
    this.genAI = new GoogleGenerativeAI(config.get<string>('GEM_API_KEY') ?? '');
  }

  buildContexto(agente: Record<string, string>, segmentos: string[], historico: Array<{ role: string; content: string }> = []): string {
    let contexto = '';
    if (segmentos.includes('system_prompt') && agente.system_prompt)
      contexto += `### INSTRUÇÕES DO SISTEMA ###\n${agente.system_prompt}\n\n`;
    if (segmentos.includes('base_conhecimento') && agente.base_conhecimento)
      contexto += `### BASE DE CONHECIMENTO ###\n${agente.base_conhecimento}\n\n`;
    if (segmentos.includes('comments') && agente.comments)
      contexto += `### COMENTÁRIOS DO CLIENTE ###\n${agente.comments}\n\n`;
    if (segmentos.includes('planejamento') && agente.planejamento)
      contexto += `### PLANEJAMENTO ###\n${agente.planejamento}\n\n`;
    if (historico.length) {
      contexto += '### HISTÓRICO DA CONVERSA ###\n';
      for (const msg of historico) contexto += `${msg.role}: ${msg.content}\n`;
      contexto += '\n';
    }
    contexto += '### RESPOSTA ATUAL ###\nassistant:';
    return contexto;
  }

  async enviarMensagem(body: {
    agenteId: string;
    mensagem: string;
    historico: Array<{ role: string; content: string }>;
    segmentos: string[];
    user: string;
  }): Promise<string> {
    const agente = await this.agentsService.obterComHeranca(body.agenteId, body.user);
    const contexto = this.buildContexto(agente as unknown as Record<string, string>, body.segmentos, body.historico);

    const model = this.genAI.getGenerativeModel({ model: 'gemini-2.5-flash' });
    const result = await model.generateContent(contexto + '\n\nUsuário: ' + body.mensagem);
    return result.response.text();
  }
}
