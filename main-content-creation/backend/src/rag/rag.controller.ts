import { Controller, Post, Body, UseGuards } from '@nestjs/common';
import { JwtAuthGuard } from '../auth/jwt-auth.guard';
import { RagService } from './rag.service';

@Controller('rag')
@UseGuards(JwtAuthGuard)
export class RagController {
  constructor(private ragService: RagService) {}

  @Post('search')
  async search(@Body() body: { texto: string; rags: Record<string, boolean>; limite?: number }) {
    return this.ragService.processarRags(body.texto, body.rags, body.limite ?? 12);
  }
}
