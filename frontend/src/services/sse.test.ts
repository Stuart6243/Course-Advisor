import {describe, expect, it} from 'vitest';
import {ChatSseDecoder, SseProtocolError} from './sse';

describe('ChatSseDecoder', () => {
  it('preserves a CRLF event boundary split across chunks', () => {
    const decoder = new ChatSseDecoder();
    expect(decoder.push('data: {"type":"chunk","content":"hel')).toEqual([]);
    expect(decoder.push('lo"}\r')).toEqual([]);
    expect(decoder.push('\n\r\n')).toEqual([{type: 'chunk', content: 'hello'}]);
  });

  it('decodes the complete fallback contract', () => {
    const decoder = new ChatSseDecoder();
    const text = [
      'data: {"type":"meta","provider":"groq","fallback_available":true,"history_turns":0,"revision":0}',
      'data: {"type":"chunk","content":"partial"}',
      'data: {"type":"fallback","action":"reset","from":"groq","to":"ollama","reason":"timeout"}',
      'data: {"type":"chunk","content":"complete"}',
      'data: {"type":"sources","courses":["COMS W4111"]}',
      'data: {"type":"done","provider":"ollama","fallback_used":true,"fallback_reason":"timeout"}',
    ].join('\n\n') + '\n\n';

    expect(decoder.push(text).map((event) => event.type)).toEqual([
      'meta',
      'chunk',
      'fallback',
      'chunk',
      'sources',
      'done',
    ]);
  });

  it('rejects malformed JSON instead of silently discarding it', () => {
    const decoder = new ChatSseDecoder();
    expect(() => decoder.push('data: {not-json}\n\n')).toThrow(SseProtocolError);
  });

  it('preserves whitespace-only token chunks', () => {
    const decoder = new ChatSseDecoder();
    expect(decoder.push('data: {"type":"chunk","content":" \\n"}\n\n')).toEqual([
      {type: 'chunk', content: ' \n'},
    ]);
  });

  it('accepts a primary failure reason when no fallback was available', () => {
    const decoder = new ChatSseDecoder();
    expect(
      decoder.push(
        'data: {"type":"error","message":"interrupted","provider":"groq","fallback_used":false,"fallback_reason":"provider_error","interrupted":true,"partial_content":" ","partial_provider":"groq"}\n\n',
      ),
    ).toEqual([
      expect.objectContaining({
        type: 'error',
        fallback_used: false,
        fallback_reason: 'provider_error',
        partial_content: ' ',
      }),
    ]);
  });

  it.each([
    '{"type":"meta","provider":42,"fallback_available":false,"history_turns":0,"revision":0}',
    '{"type":"meta","provider":"unknown","fallback_available":false,"history_turns":0,"revision":0}',
    '{"type":"meta","provider":"groq","fallback_available":false,"history_turns":0,"revision":0,"intent_provider":"other"}',
    '{"type":"meta","provider":"groq","fallback_available":"yes","history_turns":0,"revision":0}',
    '{"type":"meta","provider":"groq","fallback_available":false,"history_turns":11,"revision":0}',
    '{"type":"meta","provider":"ollama","fallback_available":true,"history_turns":0,"revision":0}',
    '{"type":"fallback","action":"reset","from":"groq","to":"ollama"}',
    '{"type":"fallback","action":"reset","from":"ollama","to":"groq","reason":"timeout"}',
    '{"type":"error","message":"broken","interrupted":true,"partial_content":"partial"}',
    '{"type":"done","provider":"groq","fallback_used":false}',
    '{"type":"done","provider":"unknown","fallback_used":false,"fallback_reason":null}',
  ])('rejects a malformed typed payload: %s', (payload) => {
    const decoder = new ChatSseDecoder();
    expect(() => decoder.push(`data: ${payload}\n\n`)).toThrow(SseProtocolError);
  });
});
