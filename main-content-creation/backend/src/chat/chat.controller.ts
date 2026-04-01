import { Controller, Post, Body, UseGuards, Request } from '@nestjs/common';
import { JwtAuthGuard } from '../auth/jwt-auth.guard';
import { ChatService } from './chat.service';

@Controller('chat')
@UseGuards(JwtAuthGuard)
export class ChatController {
  constructor(private chatService: ChatService) {}

  @Post('message')
  async sendMessage(
    @Body() body: {
      agenteId: string;
      mensagem: string;
      historico: Array<{ role: string; content: string }>;
      segmentos: string[];
    },
    @Request() req,
  ) {
    const resposta = await this.chatService.enviarMensagem({
      ...body,
      user: req.user.username,
    });
    return { resposta };
  }
}
