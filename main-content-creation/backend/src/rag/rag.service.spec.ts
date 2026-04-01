import { Test } from '@nestjs/testing';
import { ConfigService } from '@nestjs/config';
import { RagService } from './rag.service';
import { AstraService } from './astra.service';

const mockConfig = {
  get: (key: string) => {
    const map: Record<string, string> = {
      OPENAI_API_KEY: 'test_openai_key',
      ASTRA_DB_COLLECTION: 'documents',
    };
    return map[key];
  },
};

const mockDocs = [
  { _id: 'doc1', content: 'Conteúdo sobre soja e ferrugem asiática' },
  { _id: 'doc2', content: 'Classificação taxonômica de Phakopsora pachyrhizi' },
];

const mockAstra = {
  vectorSearch: jest.fn().mockResolvedValue(mockDocs),
};

const mockEmbeddingCreate = jest.fn().mockResolvedValue({
  data: [{ embedding: new Array(1536).fill(0.1) }],
});

// Mock compatível com ESM/CJS do pacote openai
jest.mock('openai', () => {
  const MockOpenAI = jest.fn().mockImplementation(() => ({
    embeddings: { create: mockEmbeddingCreate },
  }));
  return { default: MockOpenAI, __esModule: true };
});

describe('RagService', () => {
  let service: RagService;

  beforeEach(async () => {
    const module = await Test.createTestingModule({
      providers: [
        RagService,
        { provide: ConfigService, useValue: mockConfig },
        { provide: AstraService, useValue: mockAstra },
      ],
    }).compile();
    service = module.get(RagService);
  });

  afterEach(() => jest.clearAllMocks());

  it('getEmbedding retorna vetor de 1536 dimensões', async () => {
    const embedding = await service.getEmbedding('ferrugem asiática soja');
    expect(embedding).toHaveLength(1536);
    expect(typeof embedding[0]).toBe('number');
  });

  it('ragGeral chama vectorSearch e retorna documentos', async () => {
    const result = await service.ragGeral('soja ferrugem', 6);
    expect(mockAstra.vectorSearch).toHaveBeenCalledWith('documents', expect.any(Array), 6);
    expect(result).toHaveLength(2);
  });

  it('ragTaxonomia retorna documentos via multi-query', async () => {
    const result = await service.ragTaxonomia('Phakopsora', 12);
    expect(mockAstra.vectorSearch).toHaveBeenCalled();
    expect(Array.isArray(result)).toBe(true);
  });

  it('ragEpidemiologia retorna documentos', async () => {
    const result = await service.ragEpidemiologia('temperatura umidade', 12);
    expect(Array.isArray(result)).toBe(true);
  });

  it('ragProdutos retorna documentos', async () => {
    const result = await service.ragProdutos('fungicida modo de ação', 12);
    expect(Array.isArray(result)).toBe(true);
  });

  it('processarRags executa apenas os RAGs ativos', async () => {
    const result = await service.processarRags('texto teste', {
      taxonomia: true,
      epidemiologia: false,
      produtos: false,
      geral: true,
    });
    expect(result).toHaveProperty('taxonomia');
    expect(result).toHaveProperty('geral');
    expect(result).not.toHaveProperty('epidemiologia');
    expect(result).not.toHaveProperty('produtos');
  });

  it('buildRagsContext formata o contexto corretamente', () => {
    const ctx = service.buildRagsContext({ taxonomia: mockDocs, epidemiologia: [] });
    expect(ctx).toContain('TAXONOMIA');
    expect(ctx).toContain('2 documentos');
  });
});
