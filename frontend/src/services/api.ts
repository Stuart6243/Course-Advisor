/**
 * 后端 API 调用封装。
 * 所有 HTTP 请求通过 Vite proxy 转发到后端 localhost:8000。
 */

import {ChatSettings, HealthStatus, ImportResult, ManualCourseData} from '../types';
import {
  ChatSseDecoder,
  SseProtocolError,
  type StreamDoneEvent,
  type StreamErrorEvent,
  type StreamFallbackEvent,
  type StreamMetaEvent,
} from './sse';

const configuredApiOrigin = import.meta.env.VITE_API_BASE_URL?.trim().replace(/\/$/, '');
const API_BASE = configuredApiOrigin
  ? configuredApiOrigin.endsWith('/api')
    ? configuredApiOrigin
    : `${configuredApiOrigin}/api`
  : '/api';

/** 首个合法 SSE event 与后续 event 分开计时，避免长回答被绝对 90s 截断。 */
const FIRST_EVENT_TIMEOUT_MS = 30_000;
const STREAM_IDLE_TIMEOUT_MS = 45_000;
const TERMINAL_DRAIN_TIMEOUT_MS = 250;

export type ChatStreamCallbacks = {
  onChunk: (text: string) => void;
  onSources: (courses: string[]) => void;
  onDone: (event: StreamDoneEvent) => void;
  onError: (event: StreamErrorEvent) => void;
  onAbort: () => void;
  onMeta?: (event: StreamMetaEvent) => void;
  onFallback?: (event: StreamFallbackEvent) => void;
};

export type StreamTimeouts = {
  firstEventMs?: number;
  idleMs?: number;
};

export type ApiRequestErrorCode =
  | 'import_timeout'
  | 'import_http_error'
  | 'import_invalid_response'
  | 'import_network_error';

export class ApiRequestError extends Error {
  constructor(
    public readonly code: ApiRequestErrorCode,
    message: string,
  ) {
    super(message);
    this.name = 'ApiRequestError';
  }
}

const isImportResult = (value: unknown): value is ImportResult => {
  if (typeof value !== 'object' || value === null) {
    return false;
  }
  const result = value as Partial<ImportResult>;
  return typeof result.success === 'boolean' && typeof result.message === 'string';
};

async function decodeImportResponse(res: Response): Promise<ImportResult> {
  let payload: unknown;
  try {
    payload = await res.json();
  } catch {
    if (!res.ok) {
      throw new ApiRequestError(
        'import_http_error',
        `Import failed: HTTP ${res.status}`,
      );
    }
    throw new ApiRequestError(
      'import_invalid_response',
      'The import service returned invalid JSON.',
    );
  }

  // Validation/manual-input responses deliberately use 4xx while retaining a
  // structured body the UI must display. Server failures remain transport errors.
  if (res.status >= 500) {
    throw new ApiRequestError('import_http_error', `Import failed: HTTP ${res.status}`);
  }
  if (isImportResult(payload)) {
    return payload;
  }
  if (!res.ok) {
    throw new ApiRequestError('import_http_error', `Import failed: HTTP ${res.status}`);
  }
  throw new ApiRequestError(
    'import_invalid_response',
    'The import service returned an invalid response.',
  );
}

/**
 * crypto.randomUUID() 只在 secure context（https 或 localhost）可用。
 * dev server 监听 0.0.0.0，从局域网 IP 访问时它是 undefined，
 * 旧代码会直接抛 "crypto.randomUUID is not a function" 导致整个 App 白屏。
 */
export function safeUUID(): string {
  const c = globalThis.crypto;
  if (c && typeof c.randomUUID === 'function') {
    return c.randomUUID();
  }
  if (c && typeof c.getRandomValues === 'function') {
    const bytes = c.getRandomValues(new Uint8Array(16));
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    const hex = Array.from(bytes, (b) => b.toString(16).padStart(2, '0')).join('');
    return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
  }
  return `id-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
}

export async function checkHealth(): Promise<HealthStatus> {
  try {
    const res = await fetch(`${API_BASE}/health`);
    if (!res.ok) {
      throw new Error(`HTTP ${res.status}`);
    }
    const data = await res.json();
    const groqAvailable = Boolean(data.groq_available);
    const ollamaConnected = Boolean(data.ollama_connected ?? data.ollama_available);
    return {
      status: data.status ?? 'error',
      // 兼容旧后端：没返回 usable 时退回按两个后端是否可用推断
      usable: Boolean(data.usable ?? (groqAvailable || ollamaConnected)),
      reasons: Array.isArray(data.reasons) ? data.reasons.map(String) : [],
      inference_mode: data.inference_mode ?? 'unknown',
      groq_available: groqAvailable,
      ollama_connected: ollamaConnected,
      model: data.model ?? '',
      groq_model: data.groq_model ?? '',
      courses_count: Number(data.courses_count ?? 0),
    };
  } catch {
    return {
      status: 'error',
      usable: false,
      reasons: [],
      inference_mode: 'unknown',
      groq_available: false,
      ollama_connected: false,
      model: '',
      groq_model: '',
      courses_count: 0,
    };
  }
}

export async function sendMessageStream(
  message: string,
  conversationId: string,
  language: string,
  settings: ChatSettings | undefined,
  callbacks: ChatStreamCallbacks,
  signal?: AbortSignal,
  timeouts: StreamTimeouts = {},
): Promise<void> {
  const payload: Record<string, unknown> = {
    message,
    conversation_id: conversationId,
    language,
  };
  if (settings) {
    payload.max_history_turns = settings.maxHistoryTurns;
    payload.max_results = settings.maxResults;
  }

  const requestController = new AbortController();
  let abortCause: 'external' | 'first-event-timeout' | 'idle-timeout' | null = null;
  let firstEventTimer: ReturnType<typeof setTimeout> | null = null;
  let idleTimer: ReturnType<typeof setTimeout> | null = null;
  let terminalDrainTimer: ReturnType<typeof setTimeout> | null = null;
  let activeReader: ReadableStreamDefaultReader<Uint8Array> | null = null;
  let sawEvent = false;

  const abortFor = (cause: Exclude<typeof abortCause, null>) => {
    if (requestController.signal.aborted) {
      return;
    }
    abortCause = cause;
    requestController.abort();
  };

  const onExternalAbort = () => {
    abortFor('external');
    // Once fetch has resolved with response headers, some browsers no longer
    // interrupt an outstanding reader.read() from the request signal alone.
    // Cancel the reader explicitly so Stop always reaches a terminal UI state.
    void activeReader?.cancel().catch(() => undefined);
  };
  signal?.addEventListener('abort', onExternalAbort);
  if (signal?.aborted) {
    abortFor('external');
  }

  const firstEventMs = timeouts.firstEventMs ?? FIRST_EVENT_TIMEOUT_MS;
  const idleMs = timeouts.idleMs ?? STREAM_IDLE_TIMEOUT_MS;
  firstEventTimer = setTimeout(() => abortFor('first-event-timeout'), firstEventMs);

  const noteEvent = () => {
    sawEvent = true;
    if (firstEventTimer !== null) {
      clearTimeout(firstEventTimer);
      firstEventTimer = null;
    }
    if (idleTimer !== null) {
      clearTimeout(idleTimer);
    }
    idleTimer = setTimeout(() => abortFor('idle-timeout'), idleMs);
  };

  const cleanup = () => {
    if (firstEventTimer !== null) {
      clearTimeout(firstEventTimer);
    }
    if (idleTimer !== null) {
      clearTimeout(idleTimer);
    }
    if (terminalDrainTimer !== null) {
      clearTimeout(terminalDrainTimer);
    }
    signal?.removeEventListener('abort', onExternalAbort);
  };

  const timeoutError = (): StreamErrorEvent => ({
    type: 'error',
    code: abortCause === 'first-event-timeout' ? 'first_event_timeout' : 'idle_timeout',
    message:
      abortCause === 'first-event-timeout'
        ? 'The model did not start responding in time. Please try again.'
        : 'The response stream stalled. Please try again.',
    interrupted: sawEvent,
  });

  let res: Response;
  try {
    res = await fetch(`${API_BASE}/chat`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload),
      signal: requestController.signal,
    });
  } catch (err) {
    cleanup();
    if (abortCause === 'external' || signal?.aborted) {
      callbacks.onAbort();
      return;
    }
    if (err instanceof DOMException && err.name === 'AbortError') {
      callbacks.onError(timeoutError());
      return;
    }
    callbacks.onError({
      type: 'error',
      code: 'network_error',
      message: err instanceof Error ? err.message : 'Unable to reach the server.',
    });
    return;
  }

  if (!res.ok || !res.body) {
    cleanup();
    callbacks.onError({
      type: 'error',
      code: 'http_error',
      message: `Request failed: HTTP ${res.status}`,
    });
    return;
  }

  const reader = res.body.getReader();
  activeReader = reader;
  const decoder = new TextDecoder();
  const eventDecoder = new ChatSseDecoder();
  let terminal: StreamDoneEvent | StreamErrorEvent | null = null;
  let fallbackSeen = false;
  let fallbackEvent: StreamFallbackEvent | null = null;
  let metaSeen = false;
  let metaEvent: StreamMetaEvent | null = null;
  let sourcesSeen = false;
  let answerChunkSeen = false;

  const scheduleTerminalDrain = () => {
    if (terminalDrainTimer !== null) {
      return;
    }
    terminalDrainTimer = setTimeout(() => {
      void reader.cancel().catch(() => undefined);
    }, TERMINAL_DRAIN_TIMEOUT_MS);
  };

  const processEvents = (events: ReturnType<ChatSseDecoder['push']>) => {
    for (const event of events) {
      if (terminal) {
        throw new SseProtocolError('The server sent more than one terminal event.');
      }
      noteEvent();

      switch (event.type) {
        case 'meta':
          if (metaSeen || sourcesSeen) {
            throw new SseProtocolError('The server sent metadata out of order.');
          }
          metaSeen = true;
          metaEvent = event;
          callbacks.onMeta?.(event);
          break;
        case 'chunk':
          if (!metaSeen || sourcesSeen) {
            throw new SseProtocolError('The server sent answer text out of order.');
          }
          answerChunkSeen = true;
          callbacks.onChunk(event.content);
          break;
        case 'fallback':
          if (
            !metaSeen ||
            !metaEvent?.fallback_available ||
            event.from !== metaEvent.provider ||
            sourcesSeen ||
            fallbackSeen
          ) {
            throw new SseProtocolError('The server sent an invalid fallback sequence.');
          }
          fallbackSeen = true;
          fallbackEvent = event;
          answerChunkSeen = false;
          callbacks.onFallback?.(event);
          break;
        case 'sources':
          if (!metaSeen || !answerChunkSeen || sourcesSeen) {
            throw new SseProtocolError('The server sent course sources out of order.');
          }
          sourcesSeen = true;
          callbacks.onSources(event.courses);
          break;
        case 'error':
          if (
            metaEvent &&
            ((event.provider !== undefined &&
              event.provider !== (fallbackEvent?.to ?? metaEvent.provider)) ||
              (event.fallback_used !== undefined && event.fallback_used !== fallbackSeen) ||
              (fallbackEvent !== null && event.fallback_reason !== fallbackEvent.reason))
          ) {
            throw new SseProtocolError('The server failed with inconsistent provider metadata.');
          }
          terminal = event;
          scheduleTerminalDrain();
          break;
        case 'done':
          if (!metaSeen || !sourcesSeen) {
            throw new SseProtocolError('The server completed an invalid response sequence.');
          }
          if (
            !metaEvent ||
            event.fallback_used !== fallbackSeen ||
            event.provider !== (fallbackEvent?.to ?? metaEvent.provider) ||
            event.fallback_reason !== (fallbackEvent?.reason ?? null)
          ) {
            throw new SseProtocolError('The server completed with inconsistent provider metadata.');
          }
          terminal = event;
          scheduleTerminalDrain();
          break;
      }
    }
  };

  try {
    while (true) {
      const {value, done} = await reader.read();
      if (done) {
        processEvents(eventDecoder.push(decoder.decode(), true));
        break;
      }

      processEvents(eventDecoder.push(decoder.decode(value, {stream: true})));
    }
  } catch (err) {
    // A validated terminal event is authoritative even if the transport resets
    // before a clean EOF. Protocol violations after it (for example a second
    // terminal event) still fail closed.
    if (!terminal || err instanceof SseProtocolError) {
      if (abortCause === 'external' || signal?.aborted) {
        cleanup();
        callbacks.onAbort();
        return;
      }
      cleanup();
      if (err instanceof DOMException && err.name === 'AbortError') {
        callbacks.onError(timeoutError());
        return;
      }
      callbacks.onError({
        type: 'error',
        code: err instanceof SseProtocolError ? 'protocol_error' : 'stream_failed',
        message:
          err instanceof SseProtocolError
            ? err.message
            : err instanceof Error
              ? err.message
              : 'The response stream failed.',
        interrupted: sawEvent,
      });
      return;
    }
  }

  cleanup();

  if (abortCause === 'external' || signal?.aborted) {
    callbacks.onAbort();
    return;
  }

  // terminal is assigned inside processEvents; keep the narrowing explicit for TypeScript.
  const terminalEvent = terminal as StreamDoneEvent | StreamErrorEvent | null;
  if (!terminalEvent) {
    callbacks.onError({
      type: 'error',
      code: 'early_eof',
      message: 'The connection closed before the answer completed.',
      interrupted: sawEvent,
    });
    return;
  }

  if (terminalEvent.type === 'error') {
    callbacks.onError(terminalEvent);
    return;
  }

  callbacks.onDone(terminalEvent);
}

export async function importFile(file: File, signal?: AbortSignal): Promise<ImportResult> {
  const formData = new FormData();
  formData.append('file', file);

  const controller = new AbortController();
  let timedOut = false;
  const timer = setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, 60_000);
  const onAbort = () => controller.abort();
  signal?.addEventListener('abort', onAbort);
  if (signal?.aborted) {
    controller.abort();
  }

  try {
    let res: Response;
    try {
      res = await fetch(`${API_BASE}/import`, {
        method: 'POST',
        body: formData,
        signal: controller.signal,
      });
    } catch (error) {
      if (timedOut) {
        throw new ApiRequestError('import_timeout', 'Import timed out after 60 seconds.');
      }
      if (controller.signal.aborted) {
        throw error;
      }
      throw new ApiRequestError(
        'import_network_error',
        error instanceof Error ? error.message : 'Unable to reach the import service.',
      );
    }

    try {
      return await decodeImportResponse(res);
    } catch (error) {
      if (timedOut) {
        throw new ApiRequestError('import_timeout', 'Import timed out after 60 seconds.');
      }
      throw error;
    }
  } finally {
    clearTimeout(timer);
    signal?.removeEventListener('abort', onAbort);
  }
}

export async function importManual(data: ManualCourseData): Promise<ImportResult> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE}/import/manual`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(data),
    });
  } catch (error) {
    throw new ApiRequestError(
      'import_network_error',
      error instanceof Error ? error.message : 'Unable to reach the import service.',
    );
  }

  return await decodeImportResponse(res);
}

export async function exportChat(
  messages: Array<{role: string; content: string}>,
  format: 'markdown' | 'json',
): Promise<void> {
  const res = await fetch(`${API_BASE}/export`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({messages, format}),
  });

  if (!res.ok) {
    throw new Error(`Export failed: HTTP ${res.status}`);
  }

  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const date = new Date();
  const y = String(date.getFullYear());
  const m = String(date.getMonth() + 1).padStart(2, '0');
  const d = String(date.getDate()).padStart(2, '0');
  const ext = format === 'markdown' ? 'md' : 'json';
  const filename = `chat_export_${y}${m}${d}.${ext}`;

  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}
