import { Controller, Post, Get, Body, Res, UseGuards, UseInterceptors, UploadedFile } from '@nestjs/common';
import { FileInterceptor } from '@nestjs/platform-express';
import type { Response } from 'express';
import { JwtAuthGuard } from '../auth/jwt-auth.guard';
import { ContentService } from './content.service';
import { DocxService } from './docx.service';

@Controller('content')
@UseGuards(JwtAuthGuard)
export class ContentController {
  constructor(
    private contentService: ContentService,
    private docxService: DocxService,
  ) {}

  @Post('gerar')
  gerarConteudo(@Body() body) {
    return this.contentService.gerarConteudo(body).then((text) => ({ conteudo: text }));
  }

  @Get('historico')
  listarHistorico() {
    return this.contentService.listarHistoricoGeracao();
  }

  // Transcrição de áudio/vídeo via Gemini (igual ao Python legado)
  @Post('transcrever-midia')
  @UseInterceptors(FileInterceptor('file', { limits: { fileSize: 100 * 1024 * 1024 } }))
  async transcreverMidia(@UploadedFile() file: Express.Multer.File) {
    const resultado = await this.contentService.transcreverMidia(file.buffer, file.mimetype);
    return { transcricao: resultado, nome: file.originalname };
  }

  @Post('revisao-ortografica')
  revisaoOrtografica(@Body() body: { texto: string; contextoAgente: string }) {
    return this.contentService.revisaoOrtografica(body.texto, body.contextoAgente).then((text) => ({ resultado: text }));
  }

  @Post('revisao-tecnica')
  revisaoTecnica(@Body() body) {
    return this.contentService.revisaoTecnicaComRag(body);
  }

  @Post('revisao-tecnica-sem-rag')
  revisaoTecnicaSemRag(@Body() body) {
    return this.contentService.revisaoTecnicaSemRag(body).then((text) => ({ resultado: text }));
  }

  @Post('ajuste-revisao-rag')
  ajusteRevisaoRag(@Body() body: { textoOriginal: string; textoReescrito: string; ajuste: string }) {
    return this.contentService.ajusteRevisaoRag(body).then((text) => ({ resultado: text }));
  }

  @Post('otimizacao-seo')
  otimizacaoSEO(@Body() body) {
    return this.contentService.otimizacaoSEO(body).then((text) => ({ resultado: text }));
  }

  @Post('perplexity')
  buscarPerplexity(@Body() body: { texto: string }) {
    return this.contentService.buscarPerplexity(body.texto).then((text) => ({ resultado: text }));
  }

  @Post('gerar-docx')
  async gerarDocx(@Body() body: { conteudo: string }, @Res() res: Response) {
    const buffer = await this.docxService.generateDocx(body.conteudo);
    res.set({
      'Content-Type': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
      'Content-Disposition': `attachment; filename="conteudo_otimizado.docx"`,
      'Content-Length': buffer.length,
    });
    res.end(buffer);
  }
}
