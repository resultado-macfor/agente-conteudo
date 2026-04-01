import { Test } from '@nestjs/testing';
import { FilesService } from './files.service';

describe('FilesService', () => {
  let service: FilesService;

  beforeEach(async () => {
    const module = await Test.createTestingModule({
      providers: [FilesService],
    }).compile();
    service = module.get(FilesService);
  });

  it('extrai texto de arquivo TXT (UTF-8)', async () => {
    const buffer = Buffer.from('Conteúdo de teste em texto simples', 'utf-8');
    const result = await service.extractText(buffer, 'documento.txt');
    expect(result).toBe('Conteúdo de teste em texto simples');
  });

  it('extrai texto de TXT com encoding latin-1', async () => {
    const buffer = Buffer.from('Texto simples', 'latin1');
    const result = await service.extractText(buffer, 'arquivo.txt');
    expect(result).toContain('Texto');
  });

  it('retorna mensagem de formato não suportado para extensão desconhecida', async () => {
    const buffer = Buffer.from('dados');
    const result = await service.extractText(buffer, 'arquivo.xyz');
    expect(result).toContain('não suportado');
  });

  it('identifica extensão corretamente para .doc (alias de docx)', async () => {
    const buffer = Buffer.from('não é docx real');
    const result = await service.extractText(buffer, 'arquivo.doc');
    expect(typeof result).toBe('string');
  });

  it('identifica extensão corretamente para .ppt (alias de pptx)', async () => {
    const buffer = Buffer.from('não é pptx real');
    const result = await service.extractText(buffer, 'arquivo.ppt');
    expect(typeof result).toBe('string');
  });
});
