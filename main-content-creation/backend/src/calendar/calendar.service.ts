import { Injectable } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { GoogleGenerativeAI } from '@google/generative-ai';

const INFO_SAFRAS = `
Calendário de Safra: Soja
Mato Grosso: Plantio set-dez, pico out-nov. Colheita jan-abr, pico fev-mar.
Paraná: Plantio set-dez, pico out-nov. Colheita jan-abr, pico mar.
Rio Grande do Sul: Plantio out-dez, pico nov. Colheita fev-abr, pico mar-abr.
Goiás: Plantio set-jan, pico out-nov. Colheita jan-abr, pico mar.

Calendário de Safra: Milho
Mato Grosso 1ª Safra: Plantio out-dez, pico nov. Colheita fev-abr, pico mar.
Paraná 1ª Safra: Plantio set-nov, pico out. Colheita jan-mar, pico fev.
Safrinha (2ª safra) Mato Grosso: Plantio jan-fev. Colheita jun-ago.

Calendário de Safra: Algodão
Mato Grosso: Plantio dez-fev, pico jan. Colheita abr-ago, pico jun.
Bahia: Plantio dez-fev. Colheita jun-set.

Calendário de Safra: Cana-de-açúcar
São Paulo: Plantio jan-mar e ago-out. Colheita abr-nov, pico jun-ago.
`;

@Injectable()
export class CalendarService {
  private genAI: GoogleGenerativeAI;

  constructor(private config: ConfigService) {
    this.genAI = new GoogleGenerativeAI(config.get<string>('GEM_API_KEY') ?? '');
  }

  async gerarCalendario(body: {
    cultura: string;
    estado: string;
    periodo: string;
    temas: string[];
    contextoAgente: string;
  }): Promise<string> {
    const model = this.genAI.getGenerativeModel({ model: 'gemini-2.5-flash' });
    const prompt = `${body.contextoAgente}

INFORMAÇÕES DE SAFRA:
${INFO_SAFRAS}

Gere um calendário de conteúdo completo para:
- Cultura: ${body.cultura}
- Estado/Região: ${body.estado}
- Período: ${body.periodo}
- Temas a abordar: ${body.temas.join(', ')}

O calendário deve conter:
1. Uma linha por publicação, com colunas: Data | Tema | Tipo de Conteúdo | Título Sugerido | Palavras-chave | Observações
2. Considerar as fases fenológicas da cultura (plantio, desenvolvimento, floração, colheita)
3. Incluir datas de eventos importantes (feiras, reuniões técnicas) quando relevante
4. Variar os tipos de conteúdo: blog post, post social, email, vídeo, infográfico
5. Alinhar os temas ao calendário de safra da região

Retorne o calendário formatado como tabela Markdown.`;

    const result = await model.generateContent(prompt);
    return result.response.text();
  }
}
