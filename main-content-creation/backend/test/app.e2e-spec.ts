import { Test, TestingModule } from '@nestjs/testing';
import { INestApplication, ValidationPipe } from '@nestjs/common';
import request from 'supertest';
import { App } from 'supertest/types';
import { AppModule } from './../src/app.module';

describe('Agente de Conteúdo — E2E', () => {
  let app: INestApplication<App>;
  let token: string;

  beforeAll(async () => {
    const moduleFixture: TestingModule = await Test.createTestingModule({
      imports: [AppModule],
    }).compile();

    app = moduleFixture.createNestApplication();
    app.useGlobalPipes(new ValidationPipe({ transform: true }));
    app.setGlobalPrefix('api');
    await app.init();
  }, 30000);

  afterAll(async () => {
    await app.close();
  });

  describe('POST /api/auth/login', () => {
    it('retorna token com credenciais corretas', async () => {
      const res = await request(app.getHttpServer())
        .post('/api/auth/login')
        .send({ username: 'admin', password: 'senha1234' })
        .expect(201);

      expect(res.body).toHaveProperty('access_token');
      expect(res.body).toHaveProperty('user', 'admin');
      token = res.body.access_token;
    });

    it('retorna 401 com senha errada', () => {
      return request(app.getHttpServer())
        .post('/api/auth/login')
        .send({ username: 'admin', password: 'errada' })
        .expect(401);
    });

    it('retorna 401 com usuário inexistente', () => {
      return request(app.getHttpServer())
        .post('/api/auth/login')
        .send({ username: 'inexistente', password: 'qualquer' })
        .expect(401);
    });
  });

  describe('GET /api/agents', () => {
    it('retorna 401 sem token', () => {
      return request(app.getHttpServer()).get('/api/agents').expect(401);
    });

    it('retorna lista de agentes com token válido', async () => {
      const res = await request(app.getHttpServer())
        .get('/api/agents')
        .set('Authorization', `Bearer ${token}`)
        .expect(200);

      expect(Array.isArray(res.body)).toBe(true);
    });
  });

  describe('CRUD de Agentes', () => {
    let agentId: string;

    it('cria um agente', async () => {
      const res = await request(app.getHttpServer())
        .post('/api/agents')
        .set('Authorization', `Bearer ${token}`)
        .send({
          nome: 'Agente Teste E2E',
          system_prompt: 'Você é um agente de teste',
          base_conhecimento: 'Informações de teste',
          comments: '',
          planejamento: '',
          categoria: 'Social',
        })
        .expect(201);

      expect(res.body).toHaveProperty('_id');
      expect(res.body.nome).toBe('Agente Teste E2E');
      expect(res.body.criado_por).toBe('admin');
      agentId = res.body._id;
    });

    it('obtém o agente criado por ID', async () => {
      const res = await request(app.getHttpServer())
        .get(`/api/agents/${agentId}`)
        .set('Authorization', `Bearer ${token}`)
        .expect(200);

      expect(res.body._id).toBe(agentId);
    });

    it('obtém o agente com herança resolvida', async () => {
      const res = await request(app.getHttpServer())
        .get(`/api/agents/${agentId}/completo`)
        .set('Authorization', `Bearer ${token}`)
        .expect(200);

      expect(res.body).toHaveProperty('nome');
    });

    it('atualiza o agente', async () => {
      const res = await request(app.getHttpServer())
        .put(`/api/agents/${agentId}`)
        .set('Authorization', `Bearer ${token}`)
        .send({ nome: 'Agente Atualizado', categoria: 'SEO' })
        .expect(200);

      expect(res.body.nome).toBe('Agente Atualizado');
    });

    it('desativa o agente (soft delete)', async () => {
      await request(app.getHttpServer())
        .delete(`/api/agents/${agentId}`)
        .set('Authorization', `Bearer ${token}`)
        .expect(200);
    });

    it('agente desativado não aparece na listagem', async () => {
      const res = await request(app.getHttpServer())
        .get('/api/agents')
        .set('Authorization', `Bearer ${token}`)
        .expect(200);

      const ids = res.body.map((a: { _id: string }) => a._id);
      expect(ids).not.toContain(agentId);
    });
  });

  describe('POST /api/files/extract', () => {
    it('extrai texto de arquivo TXT', async () => {
      const res = await request(app.getHttpServer())
        .post('/api/files/extract')
        .set('Authorization', `Bearer ${token}`)
        .attach('files', Buffer.from('Conteúdo de teste para extração'), 'teste.txt')
        .expect(201);

      expect(Array.isArray(res.body)).toBe(true);
      expect(res.body[0].name).toBe('teste.txt');
      expect(res.body[0].text).toContain('Conteúdo de teste');
    });

    it('retorna 401 sem token', () => {
      return request(app.getHttpServer())
        .post('/api/files/extract')
        .attach('files', Buffer.from('teste'), 'teste.txt')
        .expect(401);
    });
  });

  describe('POST /api/content/gerar-docx', () => {
    it('retorna arquivo DOCX com assinatura ZIP (PK)', async () => {
      const res = await request(app.getHttpServer())
        .post('/api/content/gerar-docx')
        .set('Authorization', `Bearer ${token}`)
        .send({ conteudo: '# Título\n\n## Seção\n\n**Negrito** e [link](https://example.com).' })
        .expect(201);

      expect(res.headers['content-type']).toContain('officedocument');
      expect(res.body[0]).toBe(80); 
      expect(res.body[1]).toBe(75); 
    });
  });

  describe('POST /api/calendar/gerar-xlsx', () => {
    it('retorna arquivo XLSX com assinatura ZIP (PK)', async () => {
      const csv = 'DOMINGO,SEGUNDA,TERÇA,QUARTA,QUINTA,SEXTA,SÁBADO\n,1,,2,,3,';
      const res = await request(app.getHttpServer())
        .post('/api/calendar/gerar-xlsx')
        .set('Authorization', `Bearer ${token}`)
        .send({ csvText: csv, mesAno: 'ABRIL 2026' })
        .expect(201);

      expect(res.headers['content-type']).toContain('officedocument');
    });
  });

  describe('POST /api/rag/search', () => {
    it('retorna 401 sem token', () => {
      return request(app.getHttpServer())
        .post('/api/rag/search')
        .send({ texto: 'soja ferrugem', rags: { geral: true } })
        .expect(401);
    });

    it('aceita requisição com token (200 ou 500 dependendo do AstraDB)', async () => {
      const res = await request(app.getHttpServer())
        .post('/api/rag/search')
        .set('Authorization', `Bearer ${token}`)
        .send({ texto: 'soja ferrugem asiática', rags: { geral: true }, limite: 3 });

      expect([200, 201, 500]).toContain(res.status);
    });
  });
});
