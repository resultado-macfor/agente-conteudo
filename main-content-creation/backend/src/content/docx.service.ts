import { Injectable } from '@nestjs/common';
import {
  Document, Packer, Paragraph, TextRun, HeadingLevel,
  ExternalHyperlink, AlignmentType,
} from 'docx';


const TOKEN_RE = /(\*\*(.+?)\*\*)|(\[([^\]]+)\]\((https?:\/\/[^)\s]+)\))/g;
const LINK_ARROW_RE = /(.+?)\s*→\s*(https?:\/\/\S+)/;

function normaliseLine(line: string): string {
  return line.replace(LINK_ARROW_RE, (_m, label, url) => {
    const clean = label.replace(/^- /, '').trim();
    const prefix = label.startsWith('- ') ? '- ' : '';
    return `${prefix}[${clean}](${url})`;
  });
}

function stripHtml(text: string): string {
  return text
    .replace(/<[^>]+>/g, '')
    .replace(/&amp;/g, '&')
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&nbsp;/g, ' ')
    .replace(/&quot;/g, '"');
}

function parseRuns(text: string): (TextRun | ExternalHyperlink)[] {
  const parts: (TextRun | ExternalHyperlink)[] = [];
  let pos = 0;
  TOKEN_RE.lastIndex = 0;

  for (const m of text.matchAll(TOKEN_RE)) {
    if (m.index! > pos) {
      parts.push(new TextRun(text.slice(pos, m.index)));
    }
    if (m[1]) {
      parts.push(new TextRun({ text: m[2], bold: true }));
    } else if (m[3]) {
      parts.push(
        new ExternalHyperlink({
          link: m[5],
          children: [new TextRun({ text: m[4], style: 'Hyperlink' })],
        }),
      );
    }
    pos = m.index! + m[0].length;
  }
  if (pos < text.length) {
    parts.push(new TextRun(text.slice(pos)));
  }
  return parts;
}

@Injectable()
export class DocxService {
  async generateDocx(markdown: string): Promise<Buffer> {
    const paragraphs: Paragraph[] = [];

    for (const rawLine of markdown.split('\n')) {
      const line = stripHtml(normaliseLine(rawLine.trimEnd()));

      if (line.startsWith('#### ')) {
        paragraphs.push(new Paragraph({ heading: HeadingLevel.HEADING_4, children: [new TextRun(line.slice(5))] }));
      } else if (line.startsWith('### ')) {
        paragraphs.push(new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun(line.slice(4))] }));
      } else if (line.startsWith('## ')) {
        paragraphs.push(new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(line.slice(3))] }));
      } else if (line.startsWith('# ')) {
        paragraphs.push(new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(line.slice(2))] }));
      } else if (TOKEN_RE.test(line)) {
        TOKEN_RE.lastIndex = 0;
        paragraphs.push(new Paragraph({ children: parseRuns(line), alignment: AlignmentType.LEFT }));
      } else {
        paragraphs.push(new Paragraph({ children: [new TextRun(line)], alignment: AlignmentType.LEFT }));
      }
    }

    const doc = new Document({
      sections: [{ properties: {}, children: paragraphs }],
    });

    return Packer.toBuffer(doc) as unknown as Buffer;
  }
}
