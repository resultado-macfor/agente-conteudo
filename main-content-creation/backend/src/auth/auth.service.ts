import { Injectable, UnauthorizedException } from '@nestjs/common';
import { JwtService } from '@nestjs/jwt';
import { ConfigService } from '@nestjs/config';
import * as crypto from 'crypto';

@Injectable()
export class AuthService {
  private readonly users: Record<string, string>;

  constructor(
    private jwtService: JwtService,
    private config: ConfigService,
  ) {
    this.users = this.buildUsers();
  }

  private hash(password: string): string {
    return crypto.createHash('sha256').update(password).digest('hex');
  }

  private buildUsers(): Record<string, string> {
    const map: Record<string, string> = {};
    const entries = [
      ['admin', this.config.get<string>('SENHA_ADMIN')],
      ['SYN', this.config.get<string>('SENHA_SYN')],
      ['SME', this.config.get<string>('SENHA_SME')],
      ['Enterprise', this.config.get<string>('SENHA_ENT')],
    ];
    for (const [username, senha] of entries) {
      if (username && senha) map[username as string] = this.hash(senha as string);
    }
    return map;
  }

  login(username: string, password: string): { access_token: string; user: string } {
    const hashed = this.hash(password);
    if (!this.users[username] || this.users[username] !== hashed) {
      throw new UnauthorizedException('Usuário ou senha incorretos');
    }
    const payload = { sub: username, username };
    return {
      access_token: this.jwtService.sign(payload),
      user: username,
    };
  }
}
