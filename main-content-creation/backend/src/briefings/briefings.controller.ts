import { Controller, Post, Get, Body, UseGuards } from '@nestjs/common';
import { JwtAuthGuard } from '../auth/jwt-auth.guard';
import { BriefingsService } from './briefings.service';

@Controller('briefings')
@UseGuards(JwtAuthGuard)
export class BriefingsController {
  constructor(private briefingsService: BriefingsService) {}

  @Get()
  listar() {
    return this.briefingsService.listar();
  }

  @Post('gerar')
  gerar(@Body() body) {
    return this.briefingsService.gerarBriefing(body).then((b) => ({ briefing: b }));
  }

  @Post('ajustar')
  ajustar(@Body() body) {
    return this.briefingsService.ajustarBriefing(body).then((b) => ({ briefing: b }));
  }

  @Post('gerar-multiplos')
  gerarMultiplos(@Body() body) {
    return this.briefingsService.gerarMultiplosBriefings(body).then((list) => ({ briefings: list }));
  }
}
