import { Module } from '@nestjs/common';
import { RagController } from './rag.controller';
import { RagEngineService } from './rag-engine.service';

@Module({
  controllers: [RagController],
  providers: [RagEngineService],
})
export class RagModule {}
