import { Prop, Schema, SchemaFactory } from '@nestjs/mongoose';
import { Document } from 'mongoose';

export type BlogPostDocument = BlogPost & Document;

@Schema({ timestamps: true })
export class BlogPost {
  @Prop({ default: '' }) briefing: string;
  @Prop({ default: '' }) conteudo: string;
  @Prop({ type: [String], default: [] }) fontes: string[];
  @Prop({ type: Object, default: {} }) configuracoes: Record<string, unknown>;
}

export const BlogPostSchema = SchemaFactory.createForClass(BlogPost);
