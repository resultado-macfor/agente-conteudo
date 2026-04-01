import { Module } from '@nestjs/common';
import { MongooseModule } from '@nestjs/mongoose';
import { AgentsController } from './agents.controller';
import { AgentsService } from './agents.service';
import { Agent, AgentSchema } from '../common/schemas/agent.schema';
import { Conversa, ConversaSchema } from '../common/schemas/conversa.schema';

@Module({
  imports: [
    MongooseModule.forFeature([
      { name: Agent.name, schema: AgentSchema, collection: 'agentes' },
      { name: Conversa.name, schema: ConversaSchema, collection: 'conversas' },
    ]),
  ],
  controllers: [AgentsController],
  providers: [AgentsService],
  exports: [AgentsService],
})
export class AgentsModule {}
