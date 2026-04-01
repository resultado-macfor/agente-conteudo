import { Injectable } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import axios from 'axios';

@Injectable()
export class AstraService {
  private baseUrl: string;
  private headers: Record<string, string>;

  constructor(private config: ConfigService) {
    const endpoint = config.get<string>('ASTRA_DB_API_ENDPOINT') ?? '';
    const namespace = config.get<string>('ASTRA_DB_NAMESPACE') ?? '';
    this.baseUrl = `${endpoint}/api/json/v1/${namespace}`;
    this.headers = {
      'Content-Type': 'application/json',
      'x-cassandra-token': config.get<string>('ASTRA_DB_APPLICATION_TOKEN') ?? '',
      Accept: 'application/json',
    };
  }

  async vectorSearch(collection: string, vector: number[], limit = 6): Promise<unknown[]> {
    const url = `${this.baseUrl}/${collection}`;
    const payload = {
      find: {
        sort: { $vector: vector },
        options: { limit },
      },
    };
    try {
      const response = await axios.post(url, payload, { headers: this.headers, timeout: 30000 });
      return response.data?.data?.documents ?? [];
    } catch {
      return [];
    }
  }
}
