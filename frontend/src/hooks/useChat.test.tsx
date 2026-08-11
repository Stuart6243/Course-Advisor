import {act} from 'react';
import {createRoot, type Root} from 'react-dom/client';
import {afterEach, beforeEach, describe, expect, it, vi} from 'vitest';
import type {ChatStreamCallbacks} from '../services/api';
import type {StreamSourcesEvent} from '../services/sse';

const apiMocks = vi.hoisted(() => ({
  safeUUID: vi.fn(),
  sendMessageStream: vi.fn(),
}));

vi.mock('../services/api', () => apiMocks);

import {localizeStreamError, useChat} from './useChat';

type ChatHook = ReturnType<typeof useChat>;

let container: HTMLDivElement;
let root: Root;
let current: ChatHook;
let idCounter = 0;

const sourceEvent = (uid: string, courseCode: string): StreamSourcesEvent => {
  const shared = {
    uid,
    course_code: courseCode,
    title: `${courseCode} title`,
    citation_label: 'S1',
    source_label: 'Columbia Engineering Bulletin 2025–2026',
    offerings: [
      {
        term: 'Fall 2025',
        section_id: '001/10001',
        meeting_time: 'M 10:10am - 11:25am',
        location: 'Room 101',
      },
    ],
  };
  return {
    type: 'sources',
    schema_version: 2,
    courses: [courseCode],
    answer_sources: [
      {...shared, role: 'answer_source', citation_status: 'verified'},
    ],
    prompt_basis: [{...shared, role: 'prompt_basis', citation_status: 'candidate'}],
  };
};

function Harness() {
  current = useChat('en', {maxHistoryTurns: 10, maxResults: 5});
  return <div>{current.messages.map((message) => message.content).join('|')}</div>;
}

beforeEach(async () => {
  idCounter = 0;
  apiMocks.safeUUID.mockImplementation(() => `id-${++idCounter}`);
  apiMocks.sendMessageStream.mockReset();
  container = document.createElement('div');
  document.body.appendChild(container);
  root = createRoot(container);
  await act(async () => root.render(<Harness />));
});

afterEach(async () => {
  await act(async () => root.unmount());
  container.remove();
});

describe('useChat stream lifecycle', () => {
  it('localizes client-generated stream errors in all supported languages', () => {
    const event = {
      type: 'error' as const,
      code: 'early_eof' as const,
      message: 'English fallback text',
    };

    expect(localizeStreamError(event, 'en')).toContain('connection');
    expect(localizeStreamError(event, 'zh')).toContain('连接');
    expect(localizeStreamError(event, 'es')).toContain('conexión');
    expect(localizeStreamError(event, 'fr')).toContain('connexion');
  });

  it('replaces Groq partial text when Ollama fallback resets the answer', async () => {
    apiMocks.sendMessageStream.mockImplementation(
      async (
        _message: string,
        _conversationId: string,
        _language: string,
        _settings: unknown,
        callbacks: ChatStreamCallbacks,
      ) => {
        callbacks.onMeta?.({
          type: 'meta',
          provider: 'groq',
          fallback_available: true,
          history_turns: 0,
          revision: 0,
        });
        callbacks.onChunk('Groq partial');
        callbacks.onSources(sourceEvent('groq-uid', 'COMS W1001'));
        callbacks.onFallback?.({
          type: 'fallback',
          action: 'reset',
          from: 'groq',
          to: 'ollama',
          reason: 'timeout',
        });
        callbacks.onChunk('Ollama full answer');
        callbacks.onSources(sourceEvent('ollama-uid', 'COMS W1004'));
        callbacks.onDone({
          type: 'done',
          provider: 'ollama',
          fallback_used: true,
          fallback_reason: 'timeout',
        });
      },
    );

    await act(async () => current.sendMessage('question'));

    const assistant = current.messages.find((message) => message.role === 'assistant')!;
    expect(assistant.content).toBe('Ollama full answer');
    expect(assistant.content).not.toContain('Groq partial');
    expect(assistant.provider).toBe('ollama');
    expect(assistant.fallbackUsed).toBe(true);
    expect(assistant.status).toBe('complete');
    expect(assistant.sources?.schema_version).toBe(2);
    expect(
      assistant.sources?.schema_version === 2
        ? assistant.sources.answer_sources.map((source) => source.uid)
        : [],
    ).toEqual(['ollama-uid']);
  });

  it('keeps sources pending until a complete done event commits them atomically', async () => {
    let callbacks!: ChatStreamCallbacks;
    let resolveRequest!: () => void;
    apiMocks.sendMessageStream.mockImplementation(
      async (
        _message: string,
        _conversationId: string,
        _language: string,
        _settings: unknown,
        streamCallbacks: ChatStreamCallbacks,
      ) => {
        callbacks = streamCallbacks;
        await new Promise<void>((resolve) => {
          resolveRequest = resolve;
        });
      },
    );

    let pending!: Promise<void>;
    await act(async () => {
      pending = current.sendMessage('question');
      await Promise.resolve();
    });
    await act(async () => {
      callbacks.onMeta?.({
        type: 'meta',
        provider: 'deterministic',
        fallback_available: false,
        history_turns: 0,
        revision: 0,
      });
      callbacks.onChunk('complete answer');
      callbacks.onSources(sourceEvent('final-uid', 'COMS W1004'));
      await Promise.resolve();
    });

    expect(current.messages.find((message) => message.role === 'assistant')?.sources).toBeUndefined();

    await act(async () => {
      callbacks.onDone({
        type: 'done',
        provider: 'deterministic',
        fallback_used: false,
        fallback_reason: null,
      });
      resolveRequest();
      await pending;
    });

    const assistant = current.messages.find((message) => message.role === 'assistant')!;
    expect(assistant.status).toBe('complete');
    expect(assistant.sources).toEqual(sourceEvent('final-uid', 'COMS W1004'));
  });

  it('discards pending sources when the stream ends in an error', async () => {
    apiMocks.sendMessageStream.mockImplementation(
      async (
        _message: string,
        _conversationId: string,
        _language: string,
        _settings: unknown,
        callbacks: ChatStreamCallbacks,
      ) => {
        callbacks.onChunk('partial');
        callbacks.onSources(sourceEvent('uncommitted-uid', 'COMS W1004'));
        callbacks.onError({
          type: 'error',
          message: 'generation failed',
          interrupted: true,
        });
      },
    );

    await act(async () => current.sendMessage('question'));

    const assistant = current.messages.find((message) => message.role === 'assistant')!;
    expect(assistant.status).toBe('interrupted');
    expect(assistant.sources).toBeUndefined();
  });

  it('uses server revisions to distinguish a backend restart from normal continuation', async () => {
    apiMocks.sendMessageStream
      .mockImplementationOnce(
        async (
          _message: string,
          _conversationId: string,
          _language: string,
          _settings: unknown,
          callbacks: ChatStreamCallbacks,
        ) => {
          callbacks.onMeta?.({
            type: 'meta',
            provider: 'groq',
            fallback_available: false,
            revision: 0,
            history_turns: 0,
          });
          callbacks.onChunk('first');
          callbacks.onSources({type: 'sources', schema_version: 1, courses: []});
          callbacks.onDone({
            type: 'done',
            provider: 'groq',
            fallback_used: false,
            fallback_reason: null,
          });
        },
      )
      .mockImplementationOnce(
        async (
          _message: string,
          _conversationId: string,
          _language: string,
          _settings: unknown,
          callbacks: ChatStreamCallbacks,
        ) => {
          // A healthy continuation would report revision 1. Revision 0 means
          // the server forgot the completed first turn.
          callbacks.onMeta?.({
            type: 'meta',
            provider: 'groq',
            fallback_available: false,
            revision: 0,
            history_turns: 0,
          });
          callbacks.onChunk('second');
          callbacks.onSources({type: 'sources', schema_version: 1, courses: []});
          callbacks.onDone({
            type: 'done',
            provider: 'groq',
            fallback_used: false,
            fallback_reason: null,
          });
        },
      );

    await act(async () => current.sendMessage('one'));
    expect(current.contextLost).toBe(false);
    await act(async () => current.sendMessage('two'));
    expect(current.contextLost).toBe(true);
  });

  it('removes an empty assistant bubble when the user stops generation', async () => {
    apiMocks.sendMessageStream.mockImplementation(
      async (
        _message: string,
        _conversationId: string,
        _language: string,
        _settings: unknown,
        callbacks: ChatStreamCallbacks,
        signal: AbortSignal,
      ) =>
        await new Promise<void>((resolve) => {
          signal.addEventListener('abort', () => {
            // fetch/ReadableStream cancellation reports AbortError asynchronously.
            queueMicrotask(() => {
              callbacks.onAbort();
              resolve();
            });
          });
        }),
    );

    let pending!: Promise<void>;
    await act(async () => {
      pending = current.sendMessage('question');
      await Promise.resolve();
    });
    expect(current.isLoading).toBe(true);

    await act(async () => {
      current.stopGeneration();
      await pending;
    });

    expect(current.isLoading).toBe(false);
    expect(current.messages.map((message) => message.role)).toEqual(['user']);
  });

  it('keeps partial text and marks it stopped when the user aborts', async () => {
    apiMocks.sendMessageStream.mockImplementation(
      async (
        _message: string,
        _conversationId: string,
        _language: string,
        _settings: unknown,
        callbacks: ChatStreamCallbacks,
        signal: AbortSignal,
      ) => {
        callbacks.onMeta?.({
          type: 'meta',
          provider: 'groq',
          fallback_available: false,
          history_turns: 0,
          revision: 0,
        });
        callbacks.onChunk('partial answer');
        callbacks.onSources(sourceEvent('stopped-uid', 'COMS W1004'));
        await new Promise<void>((resolve) => {
          signal.addEventListener('abort', () => {
            callbacks.onAbort();
            resolve();
          });
        });
      },
    );

    let pending!: Promise<void>;
    await act(async () => {
      pending = current.sendMessage('question');
      await Promise.resolve();
    });
    await act(async () => {
      current.stopGeneration();
      await pending;
    });

    const assistant = current.messages.find((message) => message.role === 'assistant')!;
    expect(assistant.content).toBe('partial answer');
    expect(assistant.provider).toBe('groq');
    expect(assistant.status).toBe('stopped');
    expect(assistant.sources).toBeUndefined();
  });

  it('ignores callbacks from an old request after starting a new chat', async () => {
    let staleCallbacks!: ChatStreamCallbacks;
    let resolveRequest!: () => void;
    apiMocks.sendMessageStream.mockImplementation(
      async (
        _message: string,
        _conversationId: string,
        _language: string,
        _settings: unknown,
        callbacks: ChatStreamCallbacks,
      ) => {
        staleCallbacks = callbacks;
        await new Promise<void>((resolve) => {
          resolveRequest = resolve;
        });
      },
    );

    let pending!: Promise<void>;
    await act(async () => {
      pending = current.sendMessage('old question');
      await Promise.resolve();
    });
    await act(async () => current.newChat());
    await act(async () => {
      staleCallbacks.onSources(sourceEvent('stale-uid', 'COMS W1004'));
      staleCallbacks.onChunk('late text');
      staleCallbacks.onDone({
        type: 'done',
        provider: 'groq',
        fallback_used: false,
        fallback_reason: null,
      });
      resolveRequest();
      await pending;
    });

    expect(current.messages).toEqual([]);
    expect(current.isLoading).toBe(false);
  });

  it('restores and attributes the Groq partial when Ollama fallback fails', async () => {
    apiMocks.sendMessageStream.mockImplementation(
      async (
        _message: string,
        _conversationId: string,
        _language: string,
        _settings: unknown,
        callbacks: ChatStreamCallbacks,
      ) => {
        callbacks.onMeta?.({
          type: 'meta',
          provider: 'groq',
          fallback_available: true,
          history_turns: 0,
          revision: 0,
        });
        callbacks.onChunk('Groq partial');
        callbacks.onFallback?.({
          type: 'fallback',
          action: 'reset',
          from: 'groq',
          to: 'ollama',
          reason: 'timeout',
        });
        callbacks.onChunk('Ollama fragment');
        callbacks.onError({
          type: 'error',
          message: 'Local generation failed.',
          provider: 'ollama',
          fallback_used: true,
          fallback_reason: 'timeout',
          interrupted: true,
          partial_content: 'Groq partial',
          partial_provider: 'groq',
        });
      },
    );

    await act(async () => current.sendMessage('question'));

    const assistant = current.messages.find((message) => message.role === 'assistant')!;
    expect(assistant.content).toContain('Groq partial');
    expect(assistant.content).not.toContain('Ollama fragment');
    expect(assistant.provider).toBe('groq');
    expect(assistant.fallbackFailed).toBe(true);
    expect(assistant.status).toBe('interrupted');
  });

  it('restores the Groq snapshot when the client loses the stream after reset', async () => {
    apiMocks.sendMessageStream.mockImplementation(
      async (
        _message: string,
        _conversationId: string,
        _language: string,
        _settings: unknown,
        callbacks: ChatStreamCallbacks,
      ) => {
        callbacks.onMeta?.({
          type: 'meta',
          provider: 'groq',
          fallback_available: true,
          history_turns: 0,
          revision: 0,
        });
        callbacks.onChunk('Groq snapshot');
        callbacks.onFallback?.({
          type: 'fallback',
          action: 'reset',
          from: 'groq',
          to: 'ollama',
          reason: 'timeout',
        });
        callbacks.onError({
          type: 'error',
          code: 'early_eof',
          message: 'The connection closed.',
          interrupted: true,
        });
      },
    );

    await act(async () => current.sendMessage('question'));

    const assistant = current.messages.find((message) => message.role === 'assistant')!;
    expect(assistant.content).toContain('Groq snapshot');
    expect(assistant.provider).toBe('groq');
    expect(assistant.fallbackFailed).toBe(true);
  });
});
