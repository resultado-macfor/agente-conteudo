import { Injectable, NotFoundException, ForbiddenException } from '@nestjs/common';
import { InjectModel } from '@nestjs/mongoose';
import { Model, Types } from 'mongoose';
import { Agent, AgentDocument } from '../common/schemas/agent.schema';
import { Conversa, ConversaDocument } from '../common/schemas/conversa.schema';

@Injectable()
export class AgentsService {
  constructor(
    @InjectModel(Agent.name) private agentModel: Model<AgentDocument>,
    @InjectModel(Conversa.name) private conversaModel: Model<ConversaDocument>,
  ) {}

  async listar(user: string): Promise<AgentDocument[]> {
    const query = user === 'admin' ? { ativo: true } : { ativo: true, criado_por: user };
    return this.agentModel.find(query).sort({ createdAt: -1 });
  }

  async listarParaHeranca(user: string, agenteAtualId?: string): Promise<AgentDocument[]> {
    const query: Record<string, unknown> = { ativo: true };
    if (user !== 'admin') query.criado_por = user;
    if (agenteAtualId) query._id = { $ne: new Types.ObjectId(agenteAtualId) };
    return this.agentModel.find(query).sort({ createdAt: -1 });
  }

  async obter(id: string, user: string): Promise<AgentDocument> {
    const agente = await this.agentModel.findById(id);
    if (!agente || !agente.ativo) throw new NotFoundException('Agente não encontrado');
    if (user !== 'admin' && agente.criado_por !== user) throw new ForbiddenException();
    return agente;
  }

  async obterComHeranca(id: string, user: string): Promise<AgentDocument> {
    const agente = await this.obter(id, user);
    if (!agente.agente_mae_id) return agente;

    let agenteMae: AgentDocument;
    try {
      agenteMae = await this.agentModel.findById(agente.agente_mae_id) as AgentDocument;
    } catch {
      return agente;
    }
    if (!agenteMae) return agente;

    const completo = agente.toObject() as Record<string, unknown>;
    for (const elem of agente.herdar_elementos) {
      if (['system_prompt', 'base_conhecimento', 'comments', 'planejamento'].includes(elem)) {
        if (!completo[elem]) completo[elem] = agenteMae[elem as keyof Agent];
      }
    }
    return completo as unknown as AgentDocument;
  }

  async criar(data: {
    nome: string;
    system_prompt: string;
    base_conhecimento: string;
    comments: string;
    planejamento: string;
    categoria: string;
    agente_mae_id?: string;
    herdar_elementos?: string[];
    user: string;
  }): Promise<AgentDocument> {
    return this.agentModel.create({
      nome: data.nome,
      system_prompt: data.system_prompt,
      base_conhecimento: data.base_conhecimento,
      comments: data.comments,
      planejamento: data.planejamento,
      categoria: data.categoria,
      agente_mae_id: data.agente_mae_id ? new Types.ObjectId(data.agente_mae_id) : null,
      herdar_elementos: data.herdar_elementos ?? [],
      ativo: true,
      criado_por: data.user,
    });
  }

  async atualizar(id: string, user: string, data: Partial<Agent>): Promise<AgentDocument> {
    await this.obter(id, user);
    return this.agentModel.findByIdAndUpdate(id, { $set: data }, { new: true }) as unknown as AgentDocument;
  }

  async desativar(id: string, user: string): Promise<void> {
    await this.obter(id, user);
    await this.agentModel.findByIdAndUpdate(id, { $set: { ativo: false } });
  }

  async salvarConversa(
    agenteId: string,
    mensagens: Array<{ role: string; content: string }>,
    segmentos: string[],
  ): Promise<void> {
    await this.conversaModel.create({
      agente_id: new Types.ObjectId(agenteId),
      mensagens,
      segmentos_utilizados: segmentos,
    });
  }

  async obterConversas(agenteId: string, limite = 10) {
    return this.conversaModel
      .find({ agente_id: new Types.ObjectId(agenteId) })
      .sort({ createdAt: -1 })
      .limit(limite);
  }
}
