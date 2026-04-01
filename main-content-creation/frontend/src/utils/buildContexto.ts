import type { Agent } from '../types';

export function buildContexto(agent: Agent, segmentos: string[]): string {
  let ctx = '';
  if (segmentos.includes('system_prompt') && agent.system_prompt)
    ctx += `### INSTRUÇÕES DO SISTEMA ###\n${agent.system_prompt}\n\n`;
  if (segmentos.includes('base_conhecimento') && agent.base_conhecimento)
    ctx += `### BASE DE CONHECIMENTO ###\n${agent.base_conhecimento}\n\n`;
  if (segmentos.includes('comments') && agent.comments)
    ctx += `### COMENTÁRIOS DO CLIENTE ###\n${agent.comments}\n\n`;
  if (segmentos.includes('planejamento') && agent.planejamento)
    ctx += `### PLANEJAMENTO ###\n${agent.planejamento}\n\n`;
  return ctx;
}
