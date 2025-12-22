import { ApiProperty } from '@nestjs/swagger';

export class StatusResponseDto {
  @ApiProperty()
  service: string;

  @ApiProperty()
  status: string;

  @ApiProperty()
  total_resumes: number;

  @ApiProperty()
  embedding_model: string;

  @ApiProperty()
  llm_model: string;

  @ApiProperty()
  ollama_running: boolean;
}