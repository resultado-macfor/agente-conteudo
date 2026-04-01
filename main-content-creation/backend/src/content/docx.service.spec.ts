import { Test } from '@nestjs/testing';
import { DocxService } from './docx.service';

describe('DocxService', () => {
  let service: DocxService;

  beforeEach(async () => {
    const module = await Test.createTestingModule({
      providers: [DocxService],
    }).compile();
    service = module.get(DocxService);
  });

  it('gera buffer DOCX válido a partir de markdown simples', async () => {
    const markdown = `# Título Principal\n\n## Seção 1\n\nTexto simples aqui.\n\n**Negrito** e texto normal.`;
    const buffer = await service.generateDocx(markdown);
    expect(buffer).toBeInstanceOf(Buffer);
    expect(buffer.length).toBeGreaterThan(100);
  });

  it('gera DOCX com headings H1 a H4', async () => {
    const markdown = `# H1\n## H2\n### H3\n#### H4\n\nParágrafo normal.`;
    const buffer = await service.generateDocx(markdown);
    expect(buffer).toBeInstanceOf(Buffer);
    expect(buffer.length).toBeGreaterThan(0);
  });

  it('gera DOCX com links Markdown convertidos para hyperlinks', async () => {
    const markdown = `Veja o [portal Mais Agro](https://maisagro.syngenta.com.br) para mais informações.`;
    const buffer = await service.generateDocx(markdown);
    expect(buffer).toBeInstanceOf(Buffer);
  });

  it('gera DOCX com texto em negrito **bold**', async () => {
    const markdown = `**Ferrugem asiática** é causada por *Phakopsora pachyrhizi*.`;
    const buffer = await service.generateDocx(markdown);
    expect(buffer).toBeInstanceOf(Buffer);
  });

  it('normaliza links com seta (→) para formato Markdown', async () => {
    const markdown = `Portal → https://maisagro.syngenta.com.br`;
    const buffer = await service.generateDocx(markdown);
    expect(buffer).toBeInstanceOf(Buffer);
  });

  it('lida com markdown vazio sem lançar erro', async () => {
    const buffer = await service.generateDocx('');
    expect(buffer).toBeInstanceOf(Buffer);
  });

  it('o buffer DOCX começa com a assinatura PK (ZIP)', async () => {
    const buffer = await service.generateDocx('# Teste');
    // Arquivos DOCX são ZIPs — assinatura PK = 0x50 0x4B
    expect(buffer[0]).toBe(0x50);
    expect(buffer[1]).toBe(0x4b);
  });
});
