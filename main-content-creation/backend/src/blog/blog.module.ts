import { Module } from '@nestjs/common';
import { MongooseModule } from '@nestjs/mongoose';
import { BlogController } from './blog.controller';
import { BlogService } from './blog.service';
import { BlogPost, BlogPostSchema } from '../common/schemas/blog-post.schema';

@Module({
  imports: [
    MongooseModule.forFeature(
      [{ name: BlogPost.name, schema: BlogPostSchema, collection: 'posts_rag' }],
      'blog',
    ),
  ],
  controllers: [BlogController],
  providers: [BlogService],
})
export class BlogModule {}
