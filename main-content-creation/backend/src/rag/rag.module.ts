import { Module } from '@nestjs/common';
import { RagController } from './rag.controller';
import { RagService } from './rag.service';
import { AstraService } from './astra.service';

@Module({
  controllers: [RagController],
  providers: [RagService, AstraService],
  exports: [RagService],
})
export class RagModule {}
