import { Controller, Post, UseInterceptors, UploadedFiles, UseGuards } from '@nestjs/common';
import { FilesInterceptor } from '@nestjs/platform-express';
import { JwtAuthGuard } from '../auth/jwt-auth.guard';
import { FilesService } from './files.service';

@Controller('files')
@UseGuards(JwtAuthGuard)
export class FilesController {
  constructor(private filesService: FilesService) {}

  @Post('extract')
  @UseInterceptors(FilesInterceptor('files', 10, { limits: { fileSize: 50 * 1024 * 1024 } }))
  async extractText(@UploadedFiles() files: Express.Multer.File[]) {
    const results = await Promise.all(
      files.map(async (file) => ({
        name: file.originalname,
        text: await this.filesService.extractText(file.buffer, file.originalname),
      })),
    );
    return results;
  }
}
