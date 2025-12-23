import { ApiProperty } from '@nestjs/swagger';

export class SearchResultDto {
  @ApiProperty()
  id: string;

  @ApiProperty()
  filename: string;

  @ApiProperty()
  score: number;

  @ApiProperty()
  content: string;
}

export class SearchResponseDto {
  @ApiProperty()
  query: string;

  @ApiProperty({ type: [SearchResultDto] })
  results: SearchResultDto[];

  @ApiProperty({ required: false })
  answer?: string;

  @ApiProperty()
  total_resumes: number;
}