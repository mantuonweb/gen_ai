import { ApiProperty } from '@nestjs/swagger';
import { IsString, IsInt, IsBoolean, IsOptional, Min, Max } from 'class-validator';

export class SearchQueryDto {
  @ApiProperty({ example: 'Python developer with ML experience' })
  @IsString()
  query: string;

  @ApiProperty({ example: 3, default: 3 })
  @IsOptional()
  @IsInt()
  @Min(1)
  @Max(10)
  top_k?: number = 3;

  @ApiProperty({ example: true, default: true })
  @IsOptional()
  @IsBoolean()
  generate_answer?: boolean = true;
}