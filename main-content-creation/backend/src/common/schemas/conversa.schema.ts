import { Prop, Schema, SchemaFactory } from '@nestjs/mongoose';
import { Document, Types } from 'mongoose';

export type ConversaDocument = Conversa & Document;

@Schema({ timestamps: true })
export class Conversa {
  @Prop({ type: Types.ObjectId, required: true }) agente_id: Types.ObjectId;
  @Prop({ type: [Object], default: [] }) mensagens: Array<{ role: string; content: string }>;
  @Prop({ type: [String], default: [] }) segmentos_utilizados: string[];
}

export const ConversaSchema = SchemaFactory.createForClass(Conversa);
