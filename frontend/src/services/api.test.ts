import {afterEach, describe, expect, it, vi} from 'vitest';
import {importFile, sendMessageStream, type ChatStreamCallbacks} from './api';

const encoder = new TextEncoder();
const META =
  'data: {"type":"meta","provider":"groq","fallback_available":false,"history_turns":0,"revision":0}\n\n';
const FALLBACK_META =
  'data: {"type":"meta","provider":"groq","fallback_available":true,"history_turns":0,"revision":0}\n\n';
const GROQ_DONE =
  'data: {"type":"done","provider":"groq","fallback_used":false,"fallback_reason":null}\n\n';

function responseFor(...chunks: string[]): Response {
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const chunk of chunks) {
        controller.enqueue(encoder.encode(chunk));
      }
      controller.close();
    },
  });
  return new Response(stream, {status: 200, headers: {'Content-Type': 'text/event-stream'}});
}

function callbacks() {
  return {
    onChunk: vi.fn(),
    onSources: vi.fn(),
    onDone: vi.fn(),
    onError: vi.fn(),
    onAbort: vi.fn(),
    onMeta: vi.fn(),
    onFallback: vi.fn(),
  } satisfies ChatStreamCallbacks;
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('sendMessageStream state machine', () => {
  it('reports a distinct first-event timeout without marking success', async () => {
    vi.useFakeTimers();
    try {
      vi.stubGlobal(
        'fetch',
        vi.fn().mockImplementation(
          (_url: string, init: RequestInit) =>
            new Promise<Response>((_resolve, reject) => {
              init.signal?.addEventListener('abort', () => {
                reject(new DOMException('aborted', 'AbortError'));
              });
            }),
        ),
      );
      const handlers = callbacks();
      const pending = sendMessageStream(
        'q',
        'c',
        'en',
        undefined,
        handlers,
        undefined,
        {firstEventMs: 10, idleMs: 100},
      );

      await vi.advanceTimersByTimeAsync(11);
      await pending;

      expect(handlers.onError).toHaveBeenCalledWith(
        expect.objectContaining({code: 'first_event_timeout', interrupted: false}),
      );
      expect(handlers.onDone).not.toHaveBeenCalled();
    } finally {
      vi.useRealTimers();
    }
  });

  it('delivers reset fallback and exactly one successful terminal event', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        responseFor(
          FALLBACK_META,
          'data: {"type":"chunk","content":"partial"}\n\n',
          'data: {"type":"fallback","action":"reset","from":"groq","to":"ollama","reason":"429"}\n\n',
          'data: {"type":"chunk","content":"full answer"}\n\n',
          'data: {"type":"sources","courses":["COMS W4111"]}\n\n',
          'data: {"type":"done","provider":"ollama","fallback_used":true,"fallback_reason":"429"}\n\n',
        ),
      ),
    );
    const handlers = callbacks();

    await sendMessageStream('q', 'c', 'en', undefined, handlers);

    expect(handlers.onChunk.mock.calls.map(([text]) => text)).toEqual([
      'partial',
      'full answer',
    ]);
    expect(handlers.onFallback).toHaveBeenCalledWith(
      expect.objectContaining({action: 'reset', from: 'groq', to: 'ollama'}),
    );
    expect(handlers.onDone).toHaveBeenCalledTimes(1);
    expect(handlers.onError).not.toHaveBeenCalled();
  });

  it('treats EOF without done as an interrupted error', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        responseFor(META, 'data: {"type":"chunk","content":"partial"}\n\n'),
      ),
    );
    const handlers = callbacks();

    await sendMessageStream('q', 'c', 'en', undefined, handlers);

    expect(handlers.onDone).not.toHaveBeenCalled();
    expect(handlers.onError).toHaveBeenCalledWith(
      expect.objectContaining({code: 'early_eof', interrupted: true}),
    );
  });

  it('rejects duplicate terminal events', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        responseFor(
          META,
          'data: {"type":"chunk","content":"answer"}\n\n',
          'data: {"type":"sources","courses":[]}\n\n',
          GROQ_DONE,
          GROQ_DONE,
        ),
      ),
    );
    const handlers = callbacks();

    await sendMessageStream('q', 'c', 'en', undefined, handlers);

    expect(handlers.onDone).not.toHaveBeenCalled();
    expect(handlers.onError).toHaveBeenCalledWith(
      expect.objectContaining({message: expect.stringContaining('terminal')}),
    );
  });

  it('does not convert an error event into success', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        responseFor('data: {"type":"error","message":"interrupted"}\n\n'),
      ),
    );
    const handlers = callbacks();

    await sendMessageStream('q', 'c', 'en', undefined, handlers);

    expect(handlers.onError).toHaveBeenCalledTimes(1);
    expect(handlers.onDone).not.toHaveBeenCalled();
  });

  it('accepts a no-fallback primary error with its provider failure reason', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        responseFor(
          META,
          'data: {"type":"chunk","content":"partial"}\n\n',
          'data: {"type":"error","message":"generation failed","provider":"groq","fallback_used":false,"fallback_reason":"provider_error","interrupted":true,"partial_content":"partial","partial_provider":"groq"}\n\n',
        ),
      ),
    );
    const handlers = callbacks();

    await sendMessageStream('q', 'c', 'en', undefined, handlers);

    expect(handlers.onError).toHaveBeenCalledWith(
      expect.objectContaining({
        provider: 'groq',
        fallback_used: false,
        fallback_reason: 'provider_error',
        partial_provider: 'groq',
      }),
    );
    expect(handlers.onDone).not.toHaveBeenCalled();
  });

  it('preserves explicit Groq ownership on a failed Ollama fallback error', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        responseFor(
          FALLBACK_META,
          'data: {"type":"chunk","content":"Groq partial"}\n\n',
          'data: {"type":"fallback","action":"reset","from":"groq","to":"ollama","reason":"timeout"}\n\n',
          'data: {"type":"chunk","content":"Ollama fragment"}\n\n',
          'data: {"type":"error","message":"fallback failed","provider":"ollama","fallback_used":true,"fallback_reason":"timeout","interrupted":true,"partial_content":"Groq partial","partial_provider":"groq"}\n\n',
        ),
      ),
    );
    const handlers = callbacks();

    await sendMessageStream('q', 'c', 'en', undefined, handlers);

    expect(handlers.onError).toHaveBeenCalledWith(
      expect.objectContaining({provider: 'ollama', partial_provider: 'groq'}),
    );
    expect(handlers.onDone).not.toHaveBeenCalled();
  });

  it('rejects a blank successful response before sources/done', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        responseFor(
          META,
          'data: {"type":"sources","courses":[]}\n\n',
          GROQ_DONE,
        ),
      ),
    );
    const handlers = callbacks();

    await sendMessageStream('q', 'c', 'en', undefined, handlers);

    expect(handlers.onDone).not.toHaveBeenCalled();
    expect(handlers.onError).toHaveBeenCalledWith(
      expect.objectContaining({code: 'protocol_error'}),
    );
  });

  it('reports an idle timeout after valid metadata without treating it as user abort', async () => {
    vi.useFakeTimers();
    try {
      vi.stubGlobal(
        'fetch',
        vi.fn().mockImplementation((_url: string, init: RequestInit) => {
          const stream = new ReadableStream<Uint8Array>({
            start(controller) {
              controller.enqueue(encoder.encode(META));
              init.signal?.addEventListener('abort', () => {
                controller.error(new DOMException('aborted', 'AbortError'));
              });
            },
          });
          return Promise.resolve(new Response(stream, {status: 200}));
        }),
      );
      const handlers = callbacks();
      const pending = sendMessageStream(
        'q',
        'c',
        'en',
        undefined,
        handlers,
        undefined,
        {firstEventMs: 100, idleMs: 10},
      );

      await vi.advanceTimersByTimeAsync(11);
      await pending;

      expect(handlers.onError).toHaveBeenCalledWith(
        expect.objectContaining({code: 'idle_timeout', interrupted: true}),
      );
      expect(handlers.onAbort).not.toHaveBeenCalled();
      expect(handlers.onDone).not.toHaveBeenCalled();
    } finally {
      vi.useRealTimers();
    }
  });

  it('keeps an external stop distinct from timeout and completion', async () => {
    const controller = new AbortController();
    vi.stubGlobal(
      'fetch',
      vi.fn().mockImplementation((_url: string, init: RequestInit) => {
        const stream = new ReadableStream<Uint8Array>({
          start(streamController) {
            streamController.enqueue(encoder.encode(META));
            init.signal?.addEventListener('abort', () => {
              streamController.error(new DOMException('aborted', 'AbortError'));
            });
          },
        });
        return Promise.resolve(new Response(stream, {status: 200}));
      }),
    );
    const handlers = callbacks();
    const pending = sendMessageStream('q', 'c', 'en', undefined, handlers, controller.signal);

    await Promise.resolve();
    controller.abort();
    await pending;

    expect(handlers.onAbort).toHaveBeenCalledTimes(1);
    expect(handlers.onError).not.toHaveBeenCalled();
    expect(handlers.onDone).not.toHaveBeenCalled();
  });

  it('cancels the active reader when fetch does not propagate abort after headers', async () => {
    const controller = new AbortController();
    const readerCancelled = vi.fn();
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        new Response(
          new ReadableStream<Uint8Array>({
            start(streamController) {
              streamController.enqueue(encoder.encode(META));
            },
            cancel: readerCancelled,
          }),
          {status: 200},
        ),
      ),
    );
    const handlers = callbacks();
    const pending = sendMessageStream(
      'q',
      'c',
      'en',
      undefined,
      handlers,
      controller.signal,
    );

    await Promise.resolve();
    await Promise.resolve();
    controller.abort();
    await pending;

    expect(readerCancelled).toHaveBeenCalledTimes(1);
    expect(handlers.onAbort).toHaveBeenCalledTimes(1);
    expect(handlers.onError).not.toHaveBeenCalled();
    expect(handlers.onDone).not.toHaveBeenCalled();
  });
});

describe('syllabus upload lifecycle', () => {
  it('preserves a structured 422 response so the UI can request manual fields', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        new Response(
          JSON.stringify({
            success: false,
            needs_manual_input: true,
            status: 'rejected',
            message: 'Credits and section identity are required.',
            missing_fields: ['points_raw', 'section_id'],
            partial_data: {course_code: 'COMS W4111'},
          }),
          {status: 422, headers: {'Content-Type': 'application/json'}},
        ),
      ),
    );

    await expect(importFile(new File(['pdf'], 'course.pdf'))).resolves.toMatchObject({
      success: false,
      needs_manual_input: true,
      message: 'Credits and section identity are required.',
      partial_data: {course_code: 'COMS W4111'},
    });
  });

  it('reports the upload timeout with a localizable error code', async () => {
    vi.useFakeTimers();
    try {
      vi.stubGlobal(
        'fetch',
        vi.fn().mockImplementation(
          (_url: string, init: RequestInit) =>
            new Promise<Response>((_resolve, reject) => {
              init.signal?.addEventListener('abort', () => {
                reject(new DOMException('aborted', 'AbortError'));
              });
            }),
        ),
      );
      const pending = importFile(new File(['pdf'], 'course.pdf'));
      const assertion = expect(pending).rejects.toMatchObject({
        code: 'import_timeout',
      });

      await vi.advanceTimersByTimeAsync(60_001);
      await assertion;
    } finally {
      vi.useRealTimers();
    }
  });

  it('preserves user cancellation instead of reporting a timeout', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockImplementation(
        (_url: string, init: RequestInit) =>
          new Promise<Response>((_resolve, reject) => {
            init.signal?.addEventListener('abort', () => {
              reject(new DOMException('aborted', 'AbortError'));
            });
          }),
      ),
    );
    const controller = new AbortController();
    const pending = importFile(new File(['pdf'], 'course.pdf'), controller.signal);
    controller.abort();

    await expect(pending).rejects.toMatchObject({name: 'AbortError'});
  });
});
