export type Provider = 'groq' | 'ollama' | 'deterministic';
export type IntentProvider = 'rule' | 'groq' | 'ollama' | 'minimal';

export type StreamMetaEvent = {
  type: 'meta';
  provider: Provider;
  fallback_available: boolean;
  history_turns: number;
  revision: number;
  intent_provider?: IntentProvider;
  intent_fallback_used?: boolean;
  intent_fallback_reason?: string | null;
};

export type StreamChunkEvent = {
  type: 'chunk';
  content: string;
};

export type StreamFallbackEvent = {
  type: 'fallback';
  action: 'reset';
  from: Provider;
  to: Provider;
  reason: string;
};

export type StreamSourcesEvent = {
  type: 'sources';
  courses: string[];
};

export type StreamErrorEvent = {
  type: 'error';
  message: string;
  code?:
    | 'first_event_timeout'
    | 'idle_timeout'
    | 'network_error'
    | 'http_error'
    | 'protocol_error'
    | 'stream_failed'
    | 'early_eof';
  provider?: Provider;
  fallback_used?: boolean;
  fallback_reason?: string | null;
  interrupted?: boolean;
  partial_content?: string;
  partial_provider?: Provider;
};

const CLIENT_ERROR_CODES = new Set<NonNullable<StreamErrorEvent['code']>>([
  'first_event_timeout',
  'idle_timeout',
  'network_error',
  'http_error',
  'protocol_error',
  'stream_failed',
  'early_eof',
]);

export type StreamDoneEvent = {
  type: 'done';
  provider: Provider;
  fallback_used: boolean;
  fallback_reason: string | null;
};

export type ChatStreamEvent =
  | StreamMetaEvent
  | StreamChunkEvent
  | StreamFallbackEvent
  | StreamSourcesEvent
  | StreamErrorEvent
  | StreamDoneEvent;

export class SseProtocolError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'SseProtocolError';
  }
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

const isNonEmptyString = (value: unknown): value is string =>
  typeof value === 'string' && Boolean(value.trim());

const isNonZeroLengthString = (value: unknown): value is string =>
  typeof value === 'string' && value.length > 0;

const PROVIDERS = new Set<Provider>(['groq', 'ollama', 'deterministic']);
const INTENT_PROVIDERS = new Set<IntentProvider>(['rule', 'groq', 'ollama', 'minimal']);

const isProvider = (value: unknown): value is Provider =>
  typeof value === 'string' && PROVIDERS.has(value as Provider);

const isIntentProvider = (value: unknown): value is IntentProvider =>
  typeof value === 'string' && INTENT_PROVIDERS.has(value as IntentProvider);

const isNullableString = (value: unknown): value is string | null =>
  value === null || typeof value === 'string';

function validateMetaEvent(parsed: Record<string, unknown>): StreamMetaEvent {
  if (
    !isProvider(parsed.provider) ||
    typeof parsed.fallback_available !== 'boolean' ||
    !Number.isInteger(parsed.history_turns) ||
    Number(parsed.history_turns) < 0 ||
    Number(parsed.history_turns) > 10 ||
    !Number.isInteger(parsed.revision) ||
    Number(parsed.revision) < 0 ||
    (parsed.fallback_available && parsed.provider !== 'groq')
  ) {
    throw new SseProtocolError('The server sent invalid stream metadata.');
  }
  if (
    (parsed.intent_provider !== undefined && !isIntentProvider(parsed.intent_provider)) ||
    (parsed.intent_fallback_used !== undefined &&
      typeof parsed.intent_fallback_used !== 'boolean') ||
    (parsed.intent_fallback_reason !== undefined &&
      !isNullableString(parsed.intent_fallback_reason)) ||
    (typeof parsed.intent_fallback_reason === 'string' &&
      !isNonEmptyString(parsed.intent_fallback_reason))
  ) {
    throw new SseProtocolError('The server sent invalid intent metadata.');
  }
  return parsed as StreamMetaEvent;
}

function validateDoneEvent(parsed: Record<string, unknown>): StreamDoneEvent {
  if (
    !isProvider(parsed.provider) ||
    typeof parsed.fallback_used !== 'boolean' ||
    !isNullableString(parsed.fallback_reason)
  ) {
    throw new SseProtocolError('The server sent an invalid completion event.');
  }
  if (
    (parsed.fallback_used && !isNonEmptyString(parsed.fallback_reason)) ||
    (!parsed.fallback_used && parsed.fallback_reason !== null)
  ) {
    throw new SseProtocolError('The server sent inconsistent fallback completion metadata.');
  }
  return parsed as StreamDoneEvent;
}

function parseEventPayload(payload: string): ChatStreamEvent {
  let parsed: unknown;
  try {
    parsed = JSON.parse(payload);
  } catch {
    throw new SseProtocolError('The server sent a malformed stream event.');
  }

  if (!isRecord(parsed) || typeof parsed.type !== 'string') {
    throw new SseProtocolError('The server sent an invalid stream event.');
  }

  switch (parsed.type) {
    case 'meta':
      return validateMetaEvent(parsed);
    case 'chunk':
      if (!isNonZeroLengthString(parsed.content)) {
        throw new SseProtocolError('A stream chunk did not contain text.');
      }
      return parsed as StreamChunkEvent;
    case 'fallback':
      if (
        parsed.action !== 'reset' ||
        !isProvider(parsed.from) ||
        !isProvider(parsed.to) ||
        parsed.from !== 'groq' ||
        parsed.to !== 'ollama' ||
        !isNonEmptyString(parsed.reason)
      ) {
        throw new SseProtocolError('The server sent an invalid fallback event.');
      }
      return parsed as StreamFallbackEvent;
    case 'sources':
      if (
        !Array.isArray(parsed.courses) ||
        !parsed.courses.every((item) => isNonEmptyString(item))
      ) {
        throw new SseProtocolError('The server sent invalid course sources.');
      }
      return parsed as StreamSourcesEvent;
    case 'error':
      if (typeof parsed.message !== 'string' || !parsed.message.trim()) {
        throw new SseProtocolError('The server sent an invalid error event.');
      }
      if (
        parsed.code !== undefined &&
        (!isNonEmptyString(parsed.code) ||
          !CLIENT_ERROR_CODES.has(parsed.code as NonNullable<StreamErrorEvent['code']>))
      ) {
        throw new SseProtocolError('The server sent an invalid error code.');
      }
      if (
        (parsed.provider !== undefined && !isProvider(parsed.provider)) ||
        (parsed.fallback_used !== undefined && typeof parsed.fallback_used !== 'boolean') ||
        (parsed.fallback_reason !== undefined && !isNullableString(parsed.fallback_reason)) ||
        (typeof parsed.fallback_reason === 'string' &&
          !isNonEmptyString(parsed.fallback_reason)) ||
        (parsed.interrupted !== undefined && typeof parsed.interrupted !== 'boolean') ||
        (parsed.partial_content !== undefined &&
          !isNonZeroLengthString(parsed.partial_content)) ||
        (parsed.partial_provider !== undefined && !isProvider(parsed.partial_provider)) ||
        ((parsed.partial_content === undefined) !== (parsed.partial_provider === undefined)) ||
        (parsed.partial_content !== undefined && parsed.interrupted !== true)
      ) {
        throw new SseProtocolError('The server sent invalid error metadata.');
      }
      if (
        (parsed.fallback_used === true && !isNonEmptyString(parsed.fallback_reason)) ||
        (parsed.fallback_used === undefined && parsed.fallback_reason !== undefined)
      ) {
        throw new SseProtocolError('The server sent inconsistent fallback error metadata.');
      }
      return parsed as StreamErrorEvent;
    case 'done':
      return validateDoneEvent(parsed);
    default:
      throw new SseProtocolError(`Unsupported stream event type: ${parsed.type}`);
  }
}

function parseEventBlock(block: string): ChatStreamEvent | null {
  const dataLines = block
    .split(/\r?\n/)
    .filter((line) => line.startsWith('data:'))
    .map((line) => line.slice(5).replace(/^ /, ''));

  if (dataLines.length === 0) {
    return null;
  }

  const payload = dataLines.join('\n').trim();
  if (!payload) {
    throw new SseProtocolError('The server sent an empty stream event.');
  }
  return parseEventPayload(payload);
}

/**
 * Incremental SSE decoder. It keeps event boundaries intact even when a JSON
 * payload or CRLF delimiter is split across network chunks.
 */
export class ChatSseDecoder {
  private buffer = '';

  push(text: string, final = false): ChatStreamEvent[] {
    this.buffer += text;
    const events: ChatStreamEvent[] = [];

    while (true) {
      const boundary = this.buffer.match(/\r?\n\r?\n/);
      if (!boundary || boundary.index === undefined) {
        break;
      }
      const block = this.buffer.slice(0, boundary.index);
      this.buffer = this.buffer.slice(boundary.index + boundary[0].length);
      const event = parseEventBlock(block);
      if (event) {
        events.push(event);
      }
    }

    if (final && this.buffer.trim()) {
      const event = parseEventBlock(this.buffer);
      this.buffer = '';
      if (event) {
        events.push(event);
      }
    }

    return events;
  }
}
