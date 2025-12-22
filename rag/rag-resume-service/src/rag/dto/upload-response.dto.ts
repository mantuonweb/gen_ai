import { ApiProperty } from '@nestjs/swagger';

export class UploadResponseDto {
  @ApiProperty()
  id: string;

  @ApiProperty()
  filename: string;

  @ApiProperty()
  message: string;
}