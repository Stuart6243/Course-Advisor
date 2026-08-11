import type {
  CourseCitationStatus,
  CourseSource,
  CourseSourceOffering,
  CourseSourceRole,
  CourseSourcesEvent,
} from '../types';

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

export type StreamSourcesEvent = CourseSourcesEvent;

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

const isNullableNonEmptyString = (value: unknown): value is string | null =>
  value === null || isNonEmptyString(value);

const SOURCE_ROLES = new Set<CourseSourceRole>(['answer_source', 'prompt_basis']);
const CITATION_STATUSES = new Set<CourseCitationStatus>([
  'verified',
  'deterministic',
  'candidate',
]);

const isSourceRole = (value: unknown): value is CourseSourceRole =>
  typeof value === 'string' && SOURCE_ROLES.has(value as CourseSourceRole);

const isCitationStatus = (value: unknown): value is CourseCitationStatus =>
  typeof value === 'string' && CITATION_STATUSES.has(value as CourseCitationStatus);

function validateOffering(value: unknown): CourseSourceOffering {
  if (
    !isRecord(value) ||
    !isNullableNonEmptyString(value.term) ||
    !isNullableNonEmptyString(value.section_id) ||
    !isNullableNonEmptyString(value.meeting_time) ||
    !isNullableNonEmptyString(value.location)
  ) {
    throw new SseProtocolError('The server sent an invalid course-source offering.');
  }
  return value as CourseSourceOffering;
}

function validateCourseSource(value: unknown, expectedRole: CourseSourceRole): CourseSource {
  if (
    !isRecord(value) ||
    !isNonEmptyString(value.uid) ||
    !isNonEmptyString(value.course_code) ||
    !isNonEmptyString(value.title) ||
    !isNonEmptyString(value.citation_label) ||
    !/^S[1-9]\d*$/.test(value.citation_label) ||
    !isNonEmptyString(value.source_label) ||
    !isSourceRole(value.role) ||
    !isCitationStatus(value.citation_status) ||
    !Array.isArray(value.offerings) ||
    value.role !== expectedRole
  ) {
    throw new SseProtocolError('The server sent an invalid structured course source.');
  }

  const validStatus =
    expectedRole === 'answer_source'
      ? value.citation_status === 'verified' || value.citation_status === 'deterministic'
      : value.citation_status === 'candidate';
  if (!validStatus) {
    throw new SseProtocolError('The server sent an inconsistent course-source role.');
  }

  const offerings = value.offerings.map(validateOffering);
  return {...value, offerings} as CourseSource;
}

function validateSourceArray(value: unknown, role: CourseSourceRole): CourseSource[] {
  if (!Array.isArray(value)) {
    throw new SseProtocolError('The server sent an invalid structured source list.');
  }
  const sources = value.map((item) => validateCourseSource(item, role));
  const uids = new Set<string>();
  for (const source of sources) {
    if (uids.has(source.uid)) {
      throw new SseProtocolError('The server sent duplicate course-source UIDs.');
    }
    uids.add(source.uid);
  }
  return sources;
}

function validateSourcesEvent(parsed: Record<string, unknown>): StreamSourcesEvent {
  if (
    !Array.isArray(parsed.courses) ||
    !parsed.courses.every((item) => isNonEmptyString(item))
  ) {
    throw new SseProtocolError('The server sent invalid course sources.');
  }

  if (parsed.schema_version === undefined || parsed.schema_version === 1) {
    if (parsed.answer_sources !== undefined || parsed.prompt_basis !== undefined) {
      throw new SseProtocolError('The server sent an incomplete structured source event.');
    }
    return {
      type: 'sources',
      schema_version: 1,
      courses: [...parsed.courses] as string[],
    };
  }

  if (parsed.schema_version !== 2) {
    throw new SseProtocolError('The server sent an unsupported source schema version.');
  }

  const answerSources = validateSourceArray(parsed.answer_sources, 'answer_source');
  const promptBasis = validateSourceArray(parsed.prompt_basis, 'prompt_basis');
  const basisByUid = new Map(promptBasis.map((source) => [source.uid, source]));
  if (promptBasis.some((source, index) => source.citation_label !== `S${index + 1}`)) {
    throw new SseProtocolError('Prompt-basis citation labels were not sequential.');
  }
  for (const source of answerSources) {
    const basisSource = basisByUid.get(source.uid);
    if (!basisSource) {
      throw new SseProtocolError('An answer source was not present in the prompt basis.');
    }
    const offeringsMatch =
      source.offerings.length === basisSource.offerings.length &&
      source.offerings.every((offering, index) => {
        const basisOffering = basisSource.offerings[index];
        return (
          offering.term === basisOffering.term &&
          offering.section_id === basisOffering.section_id &&
          offering.meeting_time === basisOffering.meeting_time &&
          offering.location === basisOffering.location
        );
      });
    if (
      source.course_code !== basisSource.course_code ||
      source.title !== basisSource.title ||
      source.citation_label !== basisSource.citation_label ||
      source.source_label !== basisSource.source_label ||
      !offeringsMatch
    ) {
      throw new SseProtocolError('An answer source did not match its prompt-basis record.');
    }
  }

  const courses = parsed.courses as string[];
  if (
    courses.length !== answerSources.length ||
    courses.some((courseCode, index) => courseCode !== answerSources[index].course_code)
  ) {
    throw new SseProtocolError('The legacy source mirror did not match the answer sources.');
  }

  return {
    type: 'sources',
    schema_version: 2,
    courses: [...courses],
    answer_sources: answerSources,
    prompt_basis: promptBasis,
  };
}

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
      return validateSourcesEvent(parsed);
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
