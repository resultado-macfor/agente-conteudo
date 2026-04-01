import { Prop, Schema, SchemaFactory } from '@nestjs/mongoose';
import { Document } from 'mongoose';

export type BriefingDocument = Briefing & Document;

@Schema({ timestamps: true })
export class Briefing {
  @Prop({ required: true }) nome_projeto: string;
  @Prop({ default: '' }) tipo: string;
  @Prop({ default: '' }) conteudo: string;
}

export const BriefingSchema = SchemaFactory.createForClass(Briefing);
