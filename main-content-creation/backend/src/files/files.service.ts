import { Injectable } from '@nestjs/common';
import * as path from 'path';

@Injectable()
export class FilesService {
  async extractText(buffer: Buffer, originalname: string): Promise<string> {
    const ext = path.extname(originalname).toLowerCase().slice(1);
    try {
      if (ext === 'pdf') return await this.extractPdf(buffer);
      if (ext === 'txt') return buffer.toString('utf-8');
      if (['pptx', 'ppt'].includes(ext)) return await this.extractPptx(buffer);
      if (['docx', 'doc'].includes(ext)) return await this.extractDocx(buffer);
      return `Formato .${ext} não suportado`;
    } catch (e) {
      return `Erro ao extrair texto de ${originalname}: ${e.message}`;
    }
  }

  private async extractPdf(buffer: Buffer): Promise<string> {
    const pdfParse = require('pdf-parse') as (buf: Buffer) => Promise<{ text: string }>;
    const data = await pdfParse(buffer);
    return data.text;
  }

  private async extractPptx(buffer: Buffer): Promise<string> {
    const JSZip = (await import('jszip')).default;
    const zip = await JSZip.loadAsync(buffer);
    let text = '';
    for (const [name, file] of Object.entries(zip.files)) {
      if (name.startsWith('ppt/slides/slide') && name.endsWith('.xml')) {
        const xml = await file.async('string');
        const matches = xml.match(/<a:t>(.*?)<\/a:t>/g) ?? [];
        text += matches.map((m) => m.replace(/<\/?a:t>/g, '')).join('\n') + '\n';
      }
    }
    return text;
  }

  private async extractDocx(buffer: Buffer): Promise<string> {
    const mammoth = await import('mammoth');
    const result = await mammoth.extractRawText({ buffer });
    return result.value;
  }
}
