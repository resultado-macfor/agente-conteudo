import { Controller, Post, Body, UseGuards } from '@nestjs/common';
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
}
