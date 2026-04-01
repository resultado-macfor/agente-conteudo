import { Controller, Post, Get, Body, UseGuards } from '@nestjs/common';
import { JwtAuthGuard } from '../auth/jwt-auth.guard';
import { BlogService } from './blog.service';

@Controller('blog')
@UseGuards(JwtAuthGuard)
export class BlogController {
  constructor(private blogService: BlogService) {}

  @Get('historico')
  listarHistorico() {
    return this.blogService.listarHistorico();
  }

  @Post('salvar')
  salvarPost(@Body() body) {
    return this.blogService.salvarPost(body);
  }

  @Post('perplexity')
  async buscarFontes(@Body() body: { briefing: string }) {
    return this.blogService.buscarPerplexityBlog(body.briefing);
  }

  @Post('gerar')
  async gerarBlog(@Body() body) {
    const conteudo = await this.blogService.gerarBlog(body);
    return { conteudo };
  }

  @Post('ajustar')
  async ajustarBlog(@Body() body) {
    const conteudo = await this.blogService.ajustarBlog(body);
    return { conteudo };
  }
}
