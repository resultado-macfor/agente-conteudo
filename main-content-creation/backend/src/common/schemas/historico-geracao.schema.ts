import { Prop, Schema, SchemaFactory } from '@nestjs/mongoose';
import { Document } from 'mongoose';

export type HistoricoGeracaoDocument = HistoricoGeracao & Document;

@Schema({ timestamps: true })
export class HistoricoGeracao {
  @Prop({ required: true }) tipo_conteudo: string;
  @Prop({ default: '' }) tom_voz: string;
  @Prop({ default: '' }) palavras_chave: string;
  @Prop({ default: 0 }) numero_palavras: number;
  @Prop({ default: '' }) conteudo_gerado: string;
  @Prop({ type: Object, default: {} }) fontes_utilizadas: Record<string, unknown>;
}

export const HistoricoGeracaoSchema = SchemaFactory.createForClass(HistoricoGeracao);
