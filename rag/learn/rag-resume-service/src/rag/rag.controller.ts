import {
  Controller,
  Get,
  Post,
  Delete,
  Body,
  Param,
  UploadedFile,
  UseInterceptors,
  BadRequestException,
  NotFoundException,
} from '@nestjs/common';
import { FileInterceptor } from '@nestjs/platform-express';
import { ApiTags, ApiOperation, ApiConsumes, ApiBody } from '@nestjs/swagger';
import { RagEngineService } from './rag-engine.service';
import { SearchQueryDto } from './dto/search-query.dto';
import { SearchResponseDto } from './dto/search-response.dto';
import { UploadResponseDto } from './dto/upload-response.dto';
import { StatusResponseDto } from './dto/status-response.dto';
import { v4 as uuidv4 } from 'uuid';
import * as fs from 'fs/promises';
import * as path from 'path';

@ApiTags('RAG Resume Service')
@Controller()
export class RagController {
  private readonly uploadDir = 'uploads';

  constructor(private readonly ragEngine: RagEngineService) {
    this.ensureUploadDir();
  }

  private async ensureUploadDir() {
    try {
      await fs.mkdir(this.uploadDir, { recursive: true });
      // eslint-disable-next-line @typescript-eslint/no-unused-vars
    } catch (error) {
      // Directory already exists
    }
  }

  @Get()
  @ApiOperation({ summary: 'Service status' })
  async getStatus(): Promise<StatusResponseDto> {
    const ollamaRunning = await this.ragEngine.checkOllamaConnection();

    return {
      service: 'RAG Resume Microservice (NestJS + Ollama)',
      status: 'running',
      total_resumes: this.ragEngine.getTotalResumes(),
      embedding_model: 'Xenova/all-MiniLM-L6-v2',
      llm_model: 'llama3.2',
      ollama_running: ollamaRunning,
    };
  }

  @Post('upload')
  @ApiOperation({ summary: 'Upload a resume' })
  @ApiConsumes('multipart/form-data')
  @ApiBody({
    schema: {
      type: 'object',
      properties: {
        file: {
          type: 'string',
          format: 'binary',
        },
      },
    },
  })
  @UseInterceptors(FileInterceptor('file'))
  async uploadResume(
    @UploadedFile() file: Express.Multer.File,
  ): Promise<UploadResponseDto> {
    if (!file) {
      throw new BadRequestException('No file uploaded');
    }

    if (!file.originalname.endsWith('.txt')) {
      throw new BadRequestException('Only .txt files are allowed');
    }

    const resumeId = uuidv4();
    const filePath = path.join(
      this.uploadDir,
      `${resumeId}_${file.originalname}`,
    );

    await fs.writeFile(filePath, file.buffer);

    const content = file.buffer.toString('utf-8');
    await this.ragEngine.addResume(resumeId, content, file.originalname);

    return {
      id: resumeId,
      filename: file.originalname,
      message: 'Resume uploaded and indexed successfully',
    };
  }

  @Post('search')
  @ApiOperation({ summary: 'Search resumes using RAG' })
  async searchResumes(
    @Body() searchQuery: SearchQueryDto,
  ): Promise<SearchResponseDto> {
    const results = await this.ragEngine.search(
      searchQuery.query,
      searchQuery.top_k,
    );

    let answer: string | undefined;
    if (searchQuery.generate_answer && results.length > 0) {
      answer = await this.ragEngine.generateAnswer(searchQuery.query, results);
    }

    return {
      query: searchQuery.query,
      results,
      answer,
      total_resumes: this.ragEngine.getTotalResumes(),
    };
  }

  @Get('resumes')
  @ApiOperation({ summary: 'List all uploaded resumes' })
  listResumes() {
    return {
      total: this.ragEngine.getTotalResumes(),
      resumes: this.ragEngine.getResumes().map((r) => ({
        id: r.id,
        filename: r.filename,
      })),
    };
  }

  @Delete('resumes/:id')
  @ApiOperation({ summary: 'Delete a resume' })
  async deleteResume(@Param('id') id: string) {
    const deleted = await this.ragEngine.deleteResume(id);

    if (!deleted) {
      throw new NotFoundException('Resume not found');
    }

    return {
      message: 'Resume deleted successfully',
      remaining: this.ragEngine.getTotalResumes(),
    };
  }

  @Get('health')
  @ApiOperation({ summary: 'Health check' })
  async healthCheck() {
    const ollamaRunning = await this.ragEngine.checkOllamaConnection();

    return {
      status: 'healthy',
      resumes_loaded: this.ragEngine.getTotalResumes(),
      embedding_model: 'Xenova/all-MiniLM-L6-v2',
      llm_model: 'llama3.2',
      ollama_running: ollamaRunning,
    };
  }

  @Get('models')
  @ApiOperation({ summary: 'List available Ollama models' })
  async listModels() {
    const models = await this.ragEngine.listAvailableModels();

    return {
      current_model: 'llama3.2',
      available_models: models,
    };
  }
}
