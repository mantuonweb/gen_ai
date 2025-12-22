import { Module } from '@nestjs/common';
import { RagModule } from './rag/rag.module';

@Module({
  imports: [RagModule],
})
export class AppModule {}
