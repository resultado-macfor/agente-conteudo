import { Controller, Post, Body, UseGuards } from '@nestjs/common';
import { JwtAuthGuard } from '../auth/jwt-auth.guard';
import { ContentService } from './content.service';

@Controller('content')
@UseGuards(JwtAuthGuard)
export class ContentController {
  constructor(private contentService: ContentService) {}

  @Post('gerar')
  gerarConteudo(@Body() body) {
    return this.contentService.gerarConteudo(body).then((text) => ({ conteudo: text }));
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

  @Post('otimizacao-seo')
  otimizacaoSEO(@Body() body) {
    return this.contentService.otimizacaoSEO(body).then((text) => ({ resultado: text }));
  }

  @Post('perplexity')
  buscarPerplexity(@Body() body: { texto: string }) {
    return this.contentService.buscarPerplexity(body.texto).then((text) => ({ resultado: text }));
  }
}
