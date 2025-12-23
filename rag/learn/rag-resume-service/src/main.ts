import { NestFactory } from '@nestjs/core';
import { SwaggerModule, DocumentBuilder } from '@nestjs/swagger';
import { ValidationPipe } from '@nestjs/common';
import { AppModule } from './app.module';

async function bootstrap() {
  const app = await NestFactory.create(AppModule);

  // Enable CORS
  app.enableCors();

  // Enable validation
  app.useGlobalPipes(new ValidationPipe());

  // Swagger setup
  const config = new DocumentBuilder()
    .setTitle('RAG Resume Microservice')
    .setDescription(
      'Upload resumes and perform intelligent search using RAG + Ollama',
    )
    .setVersion('1.0')
    .build();

  const document = SwaggerModule.createDocument(app, config);
  SwaggerModule.setup('docs', app, document);

  await app.listen(3000);

  console.log('\n' + '='.repeat(50));
  console.log('🚀 RAG Resume Microservice (NestJS + Ollama)');
  console.log('='.repeat(50));
  console.log('📖 API Docs: http://localhost:3000/docs');
  console.log('🔗 API URL: http://localhost:3000');
  console.log('='.repeat(50) + '\n');
}
bootstrap();
