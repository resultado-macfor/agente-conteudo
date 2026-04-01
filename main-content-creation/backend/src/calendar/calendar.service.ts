import { Injectable } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { GoogleGenerativeAI } from '@google/generative-ai';

const INFO_SAFRAS = `
Calendário de Safra: Algodão
Tocantins: Plantio de novembro (2ª quinzena) até fevereiro (2ª quinzena), com pico intenso em janeiro. Colheita de abril (2ª quinzena) até agosto (1ª quinzena), com pico intenso em junho e julho.
Maranhão: Plantio de dezembro (1ª quinzena) até março (2ª quinzena), com pico intenso em janeiro. Colheita de maio (2ª quinzena) até agosto (2ª quinzena), com pico intenso em junho e julho.
Mato Grosso: Plantio de dezembro (1ª quinzena) até fevereiro (2ª quinzena), com pico intenso em janeiro. Colheita de abril (2ª quinzena) até agosto (2ª quinzena), com pico intenso em junho.
São Paulo: Plantio de outubro (1ª quinzena) até janeiro (1ª quinzena), com pico intenso em novembro e dezembro. Colheita de março (1ª quinzena) até junho (1ª quinzena), com pico intenso em abril e maio.
Paraná: Plantio de setembro (2ª quinzena) até dezembro (1ª quinzena), com pico intenso em outubro e novembro. Colheita de março (1ª quinzena) até maio (2ª quinzena), com pico intenso em abril.

Calendário de Safra: Soja
Mato Grosso: Plantio de setembro (1ª quinzena) até dezembro (2ª quinzena), com pico intenso em outubro e novembro. Colheita de janeiro (1ª quinzena) até abril (1ª quinzena), com pico intenso em fevereiro e março.
Mato Grosso do Sul: Plantio de setembro (2ª quinzena) até dezembro (2ª quinzena), com pico intenso em outubro e novembro. Colheita de janeiro (2ª quinzena) até abril (1ª quinzena), com pico intenso em março.
Goiás: Plantio de setembro (2ª quinzena) até janeiro (1ª quinzena), com pico intenso em outubro e novembro. Colheita de janeiro (2ª quinzena) até abril (1ª quinzena), com pico intenso em março.
Paraná: Plantio de setembro (2ª quinzena) até dezembro (2ª quinzena), com pico intenso em outubro e novembro. Colheita de janeiro (2ª quinzena) até abril (2ª quinzena), com pico intenso em março.
Rio Grande do Sul: Plantio de outubro (1ª quinzena) até janeiro (1ª quinzena), com pico intenso em novembro e dezembro. Colheita de fevereiro (1ª quinzena) até maio (1ª quinzena), com pico intenso em março e abril.

Calendário de Safra: Milho 1ª Safra
Mato Grosso: Plantio de setembro (2ª quinzena) até dezembro (1ª quinzena), com pico intenso em outubro e novembro. Colheita de fevereiro (1ª quinzena) até maio (1ª quinzena), com pico intenso em março e abril.
Paraná: Plantio de agosto (2ª quinzena) até dezembro (1ª quinzena), com pico intenso em setembro e outubro. Colheita de janeiro (2ª quinzena) até junho (2ª quinzena), com pico intenso em março.

Calendário de Safra: Milho 2ª Safra
Mato Grosso: Plantio de janeiro (2ª quinzena) até março (1ª quinzena), com pico intenso em fevereiro. Colheita de maio (2ª quinzena) até agosto (1ª quinzena), com pico intenso em junho e julho.
Goiás: Plantio de janeiro (1ª quinzena) até março (1ª quinzena), com pico intenso em fevereiro. Colheita de maio (1ª quinzena) até setembro (1ª quinzena), com pico intenso em junho e julho.

Calendário de Safra: Cana-de-Açúcar
Centro-Oeste: Plantio de janeiro (1ª quinzena) até julho (1ª quinzena) e de outubro (1ª quinzena) até dezembro (2ª quinzena). Colheita de abril (1ª quinzena) até novembro (2ª quinzena).
Sudeste: Plantio de janeiro (1ª quinzena) até julho (1ª quinzena) e de outubro (1ª quinzena) até dezembro (2ª quinzena). Colheita de abril (1ª quinzena) até novembro (2ª quinzena).
`;

export interface ProdutoDirecional {
  produtos: string[];
  culturas: string[];
  tema: string;
}

export interface CalendarioBody {
  contextoAgente: string;
  mesAno: string;
  dataInicio: string;
  dataFim: string;
  culturas: string[];
  diasCom1Pauta: number;
  diasCom2Pautas: number;
  diasCom3Pautas: number;
  produtosDirecionais: ProdutoDirecional[];
  semanaFeirasInicio: string;
  semanaFeirasFim: string;
  produtosPrioritariosFeira: string;
  pautaRecorrenteTexto: string;
  pautaRecorrenteDias: string[];
  contextoMensal: string;
  evitarConsecutivosSemPautas: boolean;
  maxRepeticoesTema: number;
}

@Injectable()
export class CalendarService {
  private genAI: GoogleGenerativeAI;

  constructor(config: ConfigService) {
    this.genAI = new GoogleGenerativeAI(config.get<string>('GEM_API_KEY') ?? '');
  }

  async gerarCalendario(body: CalendarioBody): Promise<string> {
    const model = this.genAI.getGenerativeModel({ model: 'gemini-2.5-flash' });

    const diasSemPautas = Math.max(0,
      (new Date(body.dataFim).getTime() - new Date(body.dataInicio).getTime()) / 86400000 + 1
      - body.diasCom1Pauta - body.diasCom2Pautas - body.diasCom3Pautas
    );

    const fmt = (d: string) => new Date(d).toLocaleDateString('pt-BR', { day: '2-digit', month: '2-digit', year: 'numeric' });
    const fmtShort = (d: string) => new Date(d).toLocaleDateString('pt-BR', { day: '2-digit', month: '2-digit' });

    const produtosLinhas = body.produtosDirecionais
      .map((p) => `- ${p.produtos.join(', ')} - ${p.culturas.join(' e ')} - ${p.tema}`)
      .join('\n');

    const infoEspecifica = `
    CONFIGURAÇÕES:
    1. SEMANA COM EVENTO (${fmtShort(body.semanaFeirasInicio)} a ${fmtShort(body.semanaFeirasFim)}):
       - Apenas 1 pauta por dia
       - Priorizar: ${body.produtosPrioritariosFeira}

    2. PAUTA FIXA: "${body.pautaRecorrenteTexto}"
       - Dias: ${body.pautaRecorrenteDias.join(', ')}

    3. FREQUÊNCIA:
       - Dias com 1 pauta: ${body.diasCom1Pauta}
       - Dias com 2 pautas: ${body.diasCom2Pautas}
       - Dias com 3 pautas: ${body.diasCom3Pautas}
       - Dias sem pautas: ${diasSemPautas}
       - Evitar consecutivos sem pautas: ${body.evitarConsecutivosSemPautas}

    4. CONTROLE REPETIÇÃO:
       - Máximo repetições por tema: ${body.maxRepeticoesTema}
    `;

    const prompt = `
    ${body.contextoAgente}

    ### BEGIN DADOS_SAFRA ###
    ${INFO_SAFRAS}
    ### END DADOS_SAFRA ###

    GERAR CALENDÁRIO COM ESTAS REGRAS:

    PERÍODO: ${fmt(body.dataInicio)} a ${fmt(body.dataFim)}
    MÊS: ${body.mesAno}

    ${infoEspecifica}

    CONTEXTO: ${body.contextoMensal}

    PRODUTOS E TEMAS:
    ${produtosLinhas}

    REGRAS CRÍTICAS:
    1. Semana ${fmtShort(body.semanaFeirasInicio)} a ${fmtShort(body.semanaFeirasFim)}: APENAS 1 PAUTA POR DIA
    2. Priorizar produtos: ${body.produtosPrioritariosFeira} na semana da feira
    3. Inserir "${body.pautaRecorrenteTexto}" em TODAS as ${body.pautaRecorrenteDias.join(', ')}
    4. NÃO repetir temas (máximo ${body.maxRepeticoesTema} repetições)
    5. Células podem ter múltiplas culturas: "Soja e Milho", "Verdavis e Megafol"
    6. Praticamente todos os dias com conteúdo
    7. NUNCA 3 dias consecutivos sem pautas
    8. Baseie pautas no contexto do mês
    9. As pautas devem respeitar COM RIGIDEZ as fases reais de cada cultura por estados descritos no bloco 'DADOS_SAFRA'

    FORMATO:
    - Célula: "[EMOJI] Produto(s) - Cultura(s) - Tema - Breve descrição"
    - Ex: "🔵 Verdavis e Megafol - Soja e Milho - Tecnologia feira - Soluções apresentadas na feira"
    - Ex: "🟢 Victrato pelo Brasil - Soja e Cana - Ação nacional - Resultados em diferentes regiões"

    Retorne CSV pronto para Excel.
    `;

    const result = await model.generateContent(prompt);
    let csv = result.response.text().trim();
    csv = csv.replace(/```csv/g, '').replace(/```/g, '').trim();
    return csv;
  }

  async gerarXlsx(csvText: string, mesAno: string): Promise<Buffer> {
    const ExcelJS = await import('exceljs');
    const wb = new ExcelJS.default.Workbook();
    const ws = wb.addWorksheet(`Calendário ${mesAno}`);

    ws.mergeCells('A1:G1');
    const titleCell = ws.getCell('A1');
    titleCell.value = `CALENDÁRIO - ${mesAno}`;
    titleCell.font = { bold: true, size: 14 };
    titleCell.alignment = { horizontal: 'center' };

    const dias = ['DOMINGO', 'SEGUNDA', 'TERÇA', 'QUARTA', 'QUINTA', 'SEXTA', 'SÁBADO'];
    dias.forEach((dia, i) => {
      const cell = ws.getCell(3, i + 1);
      cell.value = dia;
      cell.font = { bold: true };
      cell.alignment = { horizontal: 'center' };
      cell.fill = { type: 'pattern', pattern: 'solid', fgColor: { argb: 'FF1a2d5a' } };
      cell.font = { bold: true, color: { argb: 'FFFFFFFF' } };
    });
    const lines = csvText.split('\n').filter((l) => l.trim() && !l.startsWith(',,'));
    let rowNum = 4;
    for (const line of lines) {
      const cells = line.split(',');
      cells.forEach((content, colIdx) => {
        const c = content.trim();
        if (c) {
          const cell = ws.getCell(rowNum, colIdx + 1);
          cell.value = c;
          cell.alignment = { wrapText: true, vertical: 'top' };
          cell.border = {
            top: { style: 'thin' }, bottom: { style: 'thin' },
            left: { style: 'thin' }, right: { style: 'thin' },
          };
        }
      });
      ws.getRow(rowNum).height = 60;
      rowNum++;
    }
    for (let c = 1; c <= 7; c++) {
      ws.getColumn(c).width = 30;
    }

    return wb.xlsx.writeBuffer() as unknown as Promise<Buffer>;
  }
}
