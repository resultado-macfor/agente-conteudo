import { Module } from '@nestjs/common';
import { MongooseModule } from '@nestjs/mongoose';
import { ContentController } from './content.controller';
import { ContentService } from './content.service';
import { DocxService } from './docx.service';
import { RagModule } from '../rag/rag.module';
import { HistoricoGeracao, HistoricoGeracaoSchema } from '../common/schemas/historico-geracao.schema';

@Module({
  imports: [
    RagModule,
    MongooseModule.forFeature(
      [{ name: HistoricoGeracao.name, schema: HistoricoGeracaoSchema, collection: 'historico_geracao' }],
      'briefings',
    ),
  ],
  controllers: [ContentController],
  providers: [ContentService, DocxService],
})
export class ContentModule {}
