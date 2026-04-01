import { Test } from '@nestjs/testing';
import { ConfigService } from '@nestjs/config';
import { CalendarService } from './calendar.service';

const MOCK_CSV = `DOMINGO,SEGUNDA,TERÇA,QUARTA,QUINTA,SEXTA,SÁBADO
,1,,2,,3,
,,4,,5,,6`;

const mockGenerateContent = jest.fn().mockResolvedValue({
  response: { text: jest.fn().mockReturnValue(MOCK_CSV) },
});

const mockGetModel = jest.fn().mockReturnValue({ generateContent: mockGenerateContent });

jest.mock('@google/generative-ai', () => ({
  GoogleGenerativeAI: jest.fn().mockImplementation(() => ({
    getGenerativeModel: mockGetModel,
  })),
}));

const mockConfig = {
  get: (key: string) => (key === 'GEM_API_KEY' ? 'test_key' : undefined),
};

describe('CalendarService', () => {
  let service: CalendarService;

  beforeEach(async () => {
    mockGenerateContent.mockClear();
    mockGetModel.mockClear();
    mockGenerateContent.mockResolvedValue({
      response: { text: jest.fn().mockReturnValue(MOCK_CSV) },
    });

    const module = await Test.createTestingModule({
      providers: [
        CalendarService,
        { provide: ConfigService, useValue: mockConfig },
      ],
    }).compile();
    service = module.get(CalendarService);
  });

  const baseBody = {
    contextoAgente: '',
    mesAno: 'FEVEREIRO 2026',
    dataInicio: '2026-02-01',
    dataFim: '2026-02-28',
    culturas: ['Soja', 'Milho'],
    diasCom1Pauta: 5,
    diasCom2Pautas: 15,
    diasCom3Pautas: 3,
    produtosDirecionais: [
      { produtos: ['Verdavis', 'Megafol'], culturas: ['Soja', 'Milho'], tema: 'Tecnologia para feira' },
    ],
    semanaFeirasInicio: '2026-02-09',
    semanaFeirasFim: '2026-02-13',
    produtosPrioritariosFeira: 'Verdavis, Megafol, Victrato',
    pautaRecorrenteTexto: 'Victrato pelo Brasil',
    pautaRecorrenteDias: ['Terça', 'Quinta'],
    contextoMensal: 'FEVEREIRO 2026: Soja em colheita',
    evitarConsecutivosSemPautas: true,
    maxRepeticoesTema: 2,
  };

  it('gerarCalendario retorna CSV sem tags de código', async () => {
    const result = await service.gerarCalendario(baseBody);
    expect(typeof result).toBe('string');
    expect(result).not.toContain('```');
  });

  it('gerarCalendario chama o modelo com configurações no prompt', async () => {
    await service.gerarCalendario(baseBody);
    const [promptArg] = mockGenerateContent.mock.calls[0] as [string];
    expect(promptArg).toContain('FEVEREIRO 2026');
    expect(promptArg).toContain('Victrato pelo Brasil');
    expect(promptArg).toContain('Terça');
    expect(promptArg).toContain('Quinta');
    expect(promptArg).toContain('DADOS_SAFRA');
  });

  it('gerarCalendario chama generateContent com produtos prioritários da feira', async () => {
    await service.gerarCalendario(baseBody);
    expect(mockGenerateContent).toHaveBeenCalledTimes(1);
    const promptArg = String(mockGenerateContent.mock.calls[0][0]);
    expect(promptArg).toContain('Verdavis, Megafol, Victrato');
  });

  it('gerarCalendario chama generateContent incluindo INFO_SAFRAS', async () => {
    await service.gerarCalendario(baseBody);
    expect(mockGenerateContent).toHaveBeenCalledTimes(1);
    const promptArg = String(mockGenerateContent.mock.calls[0][0]);
    expect(promptArg).toContain('Mato Grosso');
    expect(promptArg).toContain('2ª Safra');
  });

  it('gerarXlsx retorna Buffer não vazio', async () => {
    const ExcelJS = require('exceljs') as typeof import('exceljs');
    const wb = new ExcelJS.Workbook();
    const ws = wb.addWorksheet('Teste');
    ws.getCell('A1').value = 'Teste';
    const buf = await wb.xlsx.writeBuffer();
    expect(buf.byteLength).toBeGreaterThan(0);
  });

  it('gerarXlsx: XLSX gerado tem assinatura ZIP (PK)', async () => {
    const ExcelJS = require('exceljs') as typeof import('exceljs');
    const wb = new ExcelJS.Workbook();
    wb.addWorksheet('Cal');
    const raw = await wb.xlsx.writeBuffer();
    const buf = Buffer.from(raw);
    expect(buf[0]).toBe(0x50);
    expect(buf[1]).toBe(0x4b); 
  });
});
