import { Test } from '@nestjs/testing';
import { JwtService } from '@nestjs/jwt';
import { ConfigService } from '@nestjs/config';
import { UnauthorizedException } from '@nestjs/common';
import { AuthService } from './auth.service';

const mockConfig = {
  get: (key: string) => {
    const map: Record<string, string> = {
      JWT_SECRET: 'test_secret',
      SENHA_ADMIN: 'senha1234',
      SENHA_SYN: 'senha1',
      SENHA_SME: 'senha2',
      SENHA_ENT: 'senha3',
    };
    return map[key];
  },
};

const mockJwt = {
  sign: jest.fn().mockReturnValue('mock_token'),
};

describe('AuthService', () => {
  let service: AuthService;

  beforeEach(async () => {
    const module = await Test.createTestingModule({
      providers: [
        AuthService,
        { provide: JwtService, useValue: mockJwt },
        { provide: ConfigService, useValue: mockConfig },
      ],
    }).compile();
    service = module.get(AuthService);
  });

  it('deve fazer login com credenciais corretas (admin)', () => {
    const result = service.login('admin', 'senha1234');
    expect(result.access_token).toBe('mock_token');
    expect(result.user).toBe('admin');
  });

  it('deve fazer login com credenciais corretas (SYN)', () => {
    const result = service.login('SYN', 'senha1');
    expect(result.user).toBe('SYN');
  });

  it('deve lançar UnauthorizedException com senha errada', () => {
    expect(() => service.login('admin', 'errada')).toThrow(UnauthorizedException);
  });

  it('deve lançar UnauthorizedException com usuário inexistente', () => {
    expect(() => service.login('inexistente', 'qualquer')).toThrow(UnauthorizedException);
  });

  it('deve retornar token JWT ao fazer login', () => {
    service.login('admin', 'senha1234');
    expect(mockJwt.sign).toHaveBeenCalledWith({ sub: 'admin', username: 'admin' });
  });
});
