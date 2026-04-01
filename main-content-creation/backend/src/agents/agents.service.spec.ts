import { Test } from '@nestjs/testing';
import { getModelToken } from '@nestjs/mongoose';
import { NotFoundException, ForbiddenException } from '@nestjs/common';
import { AgentsService } from './agents.service';
import { Agent } from '../common/schemas/agent.schema';
import { Conversa } from '../common/schemas/conversa.schema';

const mockAgent = {
  _id: 'agent123',
  nome: 'Agente Teste',
  system_prompt: 'Você é um assistente de teste',
  base_conhecimento: 'Conhecimento de teste',
  comments: '',
  planejamento: '',
  categoria: 'Social',
  agente_mae_id: null,
  herdar_elementos: [],
  ativo: true,
  criado_por: 'admin',
  toObject: jest.fn().mockReturnThis(),
};

const mockAgentModel = {
  find: jest.fn().mockReturnValue({ sort: jest.fn().mockResolvedValue([mockAgent]) }),
  findById: jest.fn().mockResolvedValue(mockAgent),
  findByIdAndUpdate: jest.fn().mockResolvedValue(mockAgent),
  create: jest.fn().mockResolvedValue(mockAgent),
};

const mockConversaModel = {
  find: jest.fn().mockReturnValue({ sort: jest.fn().mockReturnValue({ limit: jest.fn().mockResolvedValue([]) }) }),
  create: jest.fn().mockResolvedValue({}),
};

describe('AgentsService', () => {
  let service: AgentsService;

  beforeEach(async () => {
    const module = await Test.createTestingModule({
      providers: [
        AgentsService,
        { provide: getModelToken(Agent.name), useValue: mockAgentModel },
        { provide: getModelToken(Conversa.name), useValue: mockConversaModel },
      ],
    }).compile();
    service = module.get(AgentsService);
  });

  afterEach(() => jest.clearAllMocks());

  describe('listar', () => {
    it('admin vê todos os agentes', async () => {
      const result = await service.listar('admin');
      expect(mockAgentModel.find).toHaveBeenCalledWith({ ativo: true });
      expect(result).toHaveLength(1);
    });

    it('usuário comum vê apenas seus agentes', async () => {
      await service.listar('SYN');
      expect(mockAgentModel.find).toHaveBeenCalledWith({ ativo: true, criado_por: 'SYN' });
    });
  });

  describe('obter', () => {
    it('retorna agente se usuário tem permissão', async () => {
      const result = await service.obter('agent123', 'admin');
      expect(result).toBeDefined();
      expect(result.nome).toBe('Agente Teste');
    });

    it('lança NotFoundException se agente não existe', async () => {
      mockAgentModel.findById.mockResolvedValueOnce(null);
      await expect(service.obter('inexistente', 'admin')).rejects.toThrow(NotFoundException);
    });

    it('lança ForbiddenException se usuário não tem permissão', async () => {
      const agenteDeOutro = { ...mockAgent, criado_por: 'outro_user', ativo: true, toObject: jest.fn() };
      mockAgentModel.findById.mockResolvedValueOnce(agenteDeOutro);
      await expect(service.obter('agent123', 'SYN')).rejects.toThrow(ForbiddenException);
    });
  });

  describe('criar', () => {
    it('cria agente com os dados corretos', async () => {
      await service.criar({
        nome: 'Novo Agente',
        system_prompt: 'Prompt',
        base_conhecimento: 'Base',
        comments: '',
        planejamento: '',
        categoria: 'SEO',
        user: 'admin',
      });
      expect(mockAgentModel.create).toHaveBeenCalledWith(
        expect.objectContaining({ nome: 'Novo Agente', categoria: 'SEO', criado_por: 'admin' }),
      );
    });
  });

  describe('desativar', () => {
    it('desativa agente (soft delete)', async () => {
      await service.desativar('agent123', 'admin');
      expect(mockAgentModel.findByIdAndUpdate).toHaveBeenCalledWith(
        expect.anything(),
        { $set: { ativo: false } },
      );
    });
  });

  describe('salvarConversa', () => {
    it('salva conversa no banco', async () => {
      const validId = '507f1f77bcf86cd799439011';
      await service.salvarConversa(validId, [{ role: 'user', content: 'Oi' }], ['system_prompt']);
      expect(mockConversaModel.create).toHaveBeenCalledWith(
        expect.objectContaining({ mensagens: [{ role: 'user', content: 'Oi' }] }),
      );
    });
  });
});
