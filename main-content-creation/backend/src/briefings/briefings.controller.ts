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

  @Post('gerar-individual')
  async gerarIndividual(@Body() body: Parameters<BriefingsService['gerarBriefingIndividual']>[0]) {
    const briefing = await this.briefingsService.gerarBriefingIndividual(body);
    return { briefing };
  }

  @Post('gerar-de-pauta')
  async gerarDePauta(@Body() body: Parameters<BriefingsService['gerarBriefingDePauta']>[0]) {
    const briefing = await this.briefingsService.gerarBriefingDePauta(body);
    return { briefing };
  }


  @Post('gerar-de-csv')
  gerarDeCSV(@Body() body: Parameters<BriefingsService['gerarBriefingsDePautas']>[0]) {
    return this.briefingsService.gerarBriefingsDePautas(body);
  }

  @Post('extrair-pautas')
  extrairPautas(@Body() body: { csvText: string }) {
    return { pautas: this.briefingsService.extrairPautasDoCSV(body.csvText) };
  }

  @Post('ajustar')
  async ajustar(@Body() body: Parameters<BriefingsService['ajustarBriefing']>[0]) {
    const briefing = await this.briefingsService.ajustarBriefing(body);
    return { briefing };
  }
}
