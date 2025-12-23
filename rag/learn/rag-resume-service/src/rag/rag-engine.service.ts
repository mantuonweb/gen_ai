import { Injectable, Logger, OnModuleInit } from '@nestjs/common';
import { pipeline } from '@xenova/transformers';
import ollama from 'ollama';
import * as fs from 'fs/promises';

interface Resume {
  id: string;
  filename: string;
  content: string;
}

interface SearchResult {
  id: string;
  filename: string;
  content: string;
  score: number;
}

@Injectable()
export class RagEngineService implements OnModuleInit {
  private readonly logger = new Logger(RagEngineService.name);
  private embedder: any;
  private resumes: Resume[] = [];
  private embeddings: number[][] = [];
  private readonly llmModel = 'llama3.2';
  private readonly stateFile = 'rag_state.json';
  private readonly ollamaClient: typeof ollama;

  constructor() {
    // Initialize Ollama client with explicit host
    this.ollamaClient = ollama;
  }

  async onModuleInit() {
    this.logger.log('🔄 Initializing RAG Engine...');
    
    // Load embedding model
    this.embedder = await pipeline(
      'feature-extraction',
      'Xenova/all-MiniLM-L6-v2',
    );
    
    this.logger.log('✅ RAG Engine initialized');
    
    // Check Ollama connection
    const isConnected = await this.checkOllamaConnection();
    if (isConnected) {
      this.logger.log('✅ Ollama is connected');
    } else {
      this.logger.warn('⚠️ Ollama is not running - AI answers will be disabled');
    }
    
    // Load saved state
    await this.loadState();
  }

  async addResume(id: string, content: string, filename: string): Promise<void> {
    this.resumes.push({ id, filename, content });
    
    // Generate embedding
    const embedding = await this.generateEmbedding(content);
    this.embeddings.push(embedding);
    
    this.logger.log(`✅ Added resume: ${filename} (Total: ${this.resumes.length})`);
    
    // Save state
    await this.saveState();
  }

  async search(query: string, topK: number = 3): Promise<SearchResult[]> {
    if (this.resumes.length === 0) {
      return [];
    }

    // Generate query embedding
    const queryEmbedding = await this.generateEmbedding(query);

    // Calculate similarities
    const similarities = this.embeddings.map((embedding) =>
      this.cosineSimilarity(queryEmbedding, embedding),
    );

    // Get top K indices
    const indices = similarities
      .map((score, idx) => ({ score, idx }))
      .sort((a, b) => b.score - a.score)
      .slice(0, topK);

    // Build results
    return indices.map(({ score, idx }) => ({
      id: this.resumes[idx].id,
      filename: this.resumes[idx].filename,
      content: this.resumes[idx].content,
      score,
    }));
  }

  async generateAnswer(query: string, relevantResumes: SearchResult[]): Promise<string> {
    if (relevantResumes.length === 0) {
      return 'No relevant resumes found.';
    }

    // Check Ollama connection first
    const isConnected = await this.checkOllamaConnection();
    if (!isConnected) {
      return 'Ollama is not running. Please start Ollama with: ollama serve';
    }

    // Prepare context
    const context = relevantResumes
      .map((r) => `Resume: ${r.filename}\n${r.content.substring(0, 800)}`)
      .join('\n\n---\n\n');

    const prompt = `Based on the following resumes, answer this question: ${query}

Resumes:
${context}

Provide a clear, concise answer based only on the information in these resumes.`;

    try {
      this.logger.log('🤖 Generating AI answer with Ollama...');
      
      const response = await this.ollamaClient.chat({
        model: this.llmModel,
        messages: [
          {
            role: 'system',
            content: 'You are an HR assistant analyzing resumes. Be concise and factual.',
          },
          {
            role: 'user',
            content: prompt,
          },
        ],
        options: {
          temperature: 0.7,
          num_predict: 200,
        },
      });

      this.logger.log('✅ AI answer generated');
      return response.message.content;
    } catch (error) {
      this.logger.error(`❌ Error generating answer: ${error.message}`);
      
      // Provide helpful error message
      if (error.message.includes('fetch failed') || error.message.includes('ECONNREFUSED')) {
        return 'Cannot connect to Ollama. Please ensure Ollama is running:\n\n1. Start Ollama: ollama serve\n2. Verify it\'s running: curl http://localhost:11434/api/tags';
      }
      
      return `Error generating answer: ${error.message}`;
    }
  }

  async checkOllamaConnection(): Promise<boolean> {
    try {
      await this.ollamaClient.list();
      return true;
    } catch (error) {
      this.logger.warn(`⚠️ Ollama connection failed: ${error.message}`);
      return false;
    }
  }

  async listAvailableModels(): Promise<string[]> {
    try {
      const result = await this.ollamaClient.list();
      return result.models.map((m: any) => m.name || m.model);
    } catch (error) {
      this.logger.error(`Error listing models: ${error.message}`);
      return [];
    }
  }

  getResumes(): Resume[] {
    return this.resumes;
  }

  getTotalResumes(): number {
    return this.resumes.length;
  }

  async deleteResume(id: string): Promise<boolean> {
    const index = this.resumes.findIndex((r) => r.id === id);
    
    if (index === -1) {
      return false;
    }

    this.resumes.splice(index, 1);
    this.embeddings.splice(index, 1);
    
    await this.saveState();
    return true;
  }

  private async generateEmbedding(text: string): Promise<number[]> {
    const output = await this.embedder(text, {
      pooling: 'mean',
      normalize: true,
    });
    return Array.from(output.data);
  }

  private cosineSimilarity(a: number[], b: number[]): number {
    const dotProduct = a.reduce((sum, val, i) => sum + val * b[i], 0);
    const magnitudeA = Math.sqrt(a.reduce((sum, val) => sum + val * val, 0));
    const magnitudeB = Math.sqrt(b.reduce((sum, val) => sum + val * val, 0));
    return dotProduct / (magnitudeA * magnitudeB);
  }

  private async saveState(): Promise<void> {
    const state = {
      resumes: this.resumes,
      embeddings: this.embeddings,
    };
    
    await fs.writeFile(this.stateFile, JSON.stringify(state, null, 2));
    this.logger.log(`💾 Saved state to ${this.stateFile}`);
  }

  private async loadState(): Promise<void> {
    try {
      const data = await fs.readFile(this.stateFile, 'utf-8');
      const state = JSON.parse(data);
      
      this.resumes = state.resumes || [];
      this.embeddings = state.embeddings || [];
      
      this.logger.log(`📂 Loaded ${this.resumes.length} resumes from ${this.stateFile}`);
    } catch (error) {
      this.logger.log('No previous state found, starting fresh');
    }
  }
}
