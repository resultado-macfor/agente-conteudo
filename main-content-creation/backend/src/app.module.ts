import { Module } from '@nestjs/common';
import { ConfigModule, ConfigService } from '@nestjs/config';
import { MongooseModule } from '@nestjs/mongoose';
import { AuthModule } from './auth/auth.module';
import { AgentsModule } from './agents/agents.module';
import { RagModule } from './rag/rag.module';
import { ContentModule } from './content/content.module';
import { BlogModule } from './blog/blog.module';
import { CalendarModule } from './calendar/calendar.module';
import { BriefingsModule } from './briefings/briefings.module';
import { FilesModule } from './files/files.module';
import { ChatModule } from './chat/chat.module';

@Module({
  imports: [
    ConfigModule.forRoot({ isGlobal: true }),
    // Banco principal: agentes_personalizados (agentes + conversas)
    MongooseModule.forRootAsync({
      useFactory: (config: ConfigService) => ({
        uri: `${config.get<string>('MONGO_URI_BASE')}/agentes_personalizados?retryWrites=true&w=majority`,
        tlsAllowInvalidCertificates: true,
        serverSelectionTimeoutMS: 10000,
      }),
      inject: [ConfigService],
    }),
    // Banco de briefings
    MongooseModule.forRootAsync({
      connectionName: 'briefings',
      useFactory: (config: ConfigService) => ({
        uri: `${config.get<string>('MONGO_URI_BASE')}/briefings_Broto_Tecnologia?retryWrites=true&w=majority`,
        tlsAllowInvalidCertificates: true,
        serverSelectionTimeoutMS: 10000,
      }),
      inject: [ConfigService],
    }),
    // Banco de blog
    MongooseModule.forRootAsync({
      connectionName: 'blog',
      useFactory: (config: ConfigService) => ({
        uri: `${config.get<string>('MONGO_URI_BASE')}/blog_rag_tecnico?retryWrites=true&w=majority`,
        tlsAllowInvalidCertificates: true,
        serverSelectionTimeoutMS: 10000,
      }),
      inject: [ConfigService],
    }),
    AuthModule,
    AgentsModule,
    RagModule,
    ContentModule,
    BlogModule,
    CalendarModule,
    BriefingsModule,
    FilesModule,
    ChatModule,
  ],
})
export class AppModule {}
