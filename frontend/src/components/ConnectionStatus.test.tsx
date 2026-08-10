import {act} from 'react';
import {createRoot, type Root} from 'react-dom/client';
import {afterEach, beforeEach, describe, expect, it, vi} from 'vitest';

const apiMocks = vi.hoisted(() => ({
  checkHealth: vi.fn(),
}));

vi.mock('../services/api', async (importOriginal) => ({
  ...(await importOriginal<typeof import('../services/api')>()),
  checkHealth: apiMocks.checkHealth,
}));

import ConnectionStatus from './ConnectionStatus';

let container: HTMLDivElement;
let root: Root;

beforeEach(() => {
  apiMocks.checkHealth.mockReset();
  container = document.createElement('div');
  document.body.appendChild(container);
  root = createRoot(container);
});

afterEach(async () => {
  await act(async () => root.unmount());
  container.remove();
});

describe('ConnectionStatus configured provider', () => {
  it('labels a healthy local-mode backend as Local AI even when Groq is reachable', async () => {
    apiMocks.checkHealth.mockResolvedValue({
      status: 'ok',
      usable: true,
      reasons: [],
      inference_mode: 'local',
      groq_available: true,
      ollama_connected: true,
      model: 'qwen3-nothink:latest',
      groq_model: 'llama-3.3-70b-versatile',
      courses_count: 1021,
    });

    await act(async () => root.render(<ConnectionStatus />));

    expect(container.querySelector('[role="status"]')?.getAttribute('aria-label')).toBe(
      'Connected via Local AI',
    );
  });

  it('uses Groq as the active provider in healthy hybrid mode', async () => {
    apiMocks.checkHealth.mockResolvedValue({
      status: 'ok',
      usable: true,
      reasons: [],
      inference_mode: 'hybrid',
      groq_available: true,
      ollama_connected: true,
      model: 'qwen3-nothink:latest',
      groq_model: 'llama-3.3-70b-versatile',
      courses_count: 1021,
    });

    await act(async () => root.render(<ConnectionStatus />));

    expect(container.querySelector('[role="status"]')?.getAttribute('aria-label')).toBe(
      'Connected via Groq Cloud',
    );
  });
});
