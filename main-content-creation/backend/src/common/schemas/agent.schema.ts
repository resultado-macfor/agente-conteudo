import { Prop, Schema, SchemaFactory } from '@nestjs/mongoose';
import { Document, Types } from 'mongoose';

export type AgentDocument = Agent & Document;

@Schema({ timestamps: true })
export class Agent {
  @Prop({ required: true }) nome: string;
  @Prop({ default: '' }) system_prompt: string;
  @Prop({ default: '' }) base_conhecimento: string;
  @Prop({ default: '' }) comments: string;
  @Prop({ default: '' }) planejamento: string;
  @Prop({ default: 'Social' }) categoria: string;
  @Prop({ type: Types.ObjectId, default: null }) agente_mae_id: Types.ObjectId | null;
  @Prop({ type: [String], default: [] }) herdar_elementos: string[];
  @Prop({ default: true }) ativo: boolean;
  @Prop({ required: true }) criado_por: string;
}

export const AgentSchema = SchemaFactory.createForClass(Agent);
