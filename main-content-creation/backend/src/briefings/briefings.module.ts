import { Module } from '@nestjs/common';
import { MongooseModule } from '@nestjs/mongoose';
import { BriefingsController } from './briefings.controller';
import { BriefingsService } from './briefings.service';
import { Briefing, BriefingSchema } from '../common/schemas/briefing.schema';

@Module({
  imports: [
    MongooseModule.forFeature(
      [{ name: Briefing.name, schema: BriefingSchema, collection: 'briefings' }],
      'briefings',
    ),
  ],
  controllers: [BriefingsController],
  providers: [BriefingsService],
  exports: [BriefingsService],
})
export class BriefingsModule {}
