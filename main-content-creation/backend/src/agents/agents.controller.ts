import { Controller, Get, Post, Put, Delete, Body, Param, Query, UseGuards, Request } from '@nestjs/common';
import { JwtAuthGuard } from '../auth/jwt-auth.guard';
import { AgentsService } from './agents.service';

@Controller('agents')
@UseGuards(JwtAuthGuard)
export class AgentsController {
  constructor(private agentsService: AgentsService) {}

  @Get()
  listar(@Request() req) {
    return this.agentsService.listar(req.user.username);
  }

  @Get('heranca')
  listarParaHeranca(@Request() req, @Query('exclude') exclude?: string) {
    return this.agentsService.listarParaHeranca(req.user.username, exclude);
  }

  @Get(':id')
  obter(@Param('id') id: string, @Request() req) {
    return this.agentsService.obter(id, req.user.username);
  }

  @Get(':id/completo')
  obterComHeranca(@Param('id') id: string, @Request() req) {
    return this.agentsService.obterComHeranca(id, req.user.username);
  }

  @Post()
  criar(@Body() body, @Request() req) {
    return this.agentsService.criar({ ...body, user: req.user.username });
  }

  @Put(':id')
  atualizar(@Param('id') id: string, @Body() body, @Request() req) {
    return this.agentsService.atualizar(id, req.user.username, body);
  }

  @Delete(':id')
  desativar(@Param('id') id: string, @Request() req) {
    return this.agentsService.desativar(id, req.user.username);
  }

  @Post(':id/conversas')
  salvarConversa(@Param('id') id: string, @Body() body) {
    return this.agentsService.salvarConversa(id, body.mensagens, body.segmentos);
  }

  @Get(':id/conversas')
  obterConversas(@Param('id') id: string) {
    return this.agentsService.obterConversas(id);
  }
}
