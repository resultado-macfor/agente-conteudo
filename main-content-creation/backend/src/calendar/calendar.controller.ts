import { Controller, Post, Body, Res, UseGuards } from '@nestjs/common';
import type { Response } from 'express';
import { JwtAuthGuard } from '../auth/jwt-auth.guard';
import { CalendarService } from './calendar.service';

@Controller('calendar')
@UseGuards(JwtAuthGuard)
export class CalendarController {
  constructor(private calendarService: CalendarService) {}

  @Post('gerar')
  async gerar(@Body() body) {
    const calendario = await this.calendarService.gerarCalendario(body);
    return { calendario };
  }

  @Post('gerar-xlsx')
  async gerarXlsx(@Body() body: { csvText: string; mesAno: string }, @Res() res: Response) {
    const buffer = await this.calendarService.gerarXlsx(body.csvText, body.mesAno);
    res.set({
      'Content-Type': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
      'Content-Disposition': `attachment; filename="calendario_${body.mesAno.replace(/\s+/g, '_').toLowerCase()}.xlsx"`,
      'Content-Length': buffer.length,
    });
    res.end(buffer);
  }
}
