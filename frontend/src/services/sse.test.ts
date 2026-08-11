import {describe, expect, it} from 'vitest';
import {ChatSseDecoder, SseProtocolError} from './sse';

const offering = {
  term: 'Fall 2025',
  section_id: '001/10001',
  meeting_time: 'M 10:10am - 11:25am',
  location: 'Room 101',
};

const promptSource = {
  uid: 'uid-1',
  course_code: 'COMS W1004',
  title: 'Introduction to Computer Science and Programming in Java',
  citation_label: 'S1',
  source_label: 'Columbia Engineering Bulletin 2025–2026',
  role: 'prompt_basis',
  citation_status: 'candidate',
  offerings: [offering],
};

const answerSource = {
  ...promptSource,
  role: 'answer_source',
  citation_status: 'verified',
};

const structuredSources = {
  type: 'sources',
  schema_version: 2,
  courses: ['COMS W1004'],
  answer_sources: [answerSource],
  prompt_basis: [promptSource],
};

const decodePayload = (payload: unknown) =>
  new ChatSseDecoder().push(`data: ${JSON.stringify(payload)}\n\n`)[0];

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

  it('normalizes a legacy string-array source event without claiming verified citations', () => {
    expect(decodePayload({type: 'sources', courses: ['COMS W4111']})).toEqual({
      type: 'sources',
      schema_version: 1,
      courses: ['COMS W4111'],
    });
  });

  it('validates and preserves a structured v2 source event', () => {
    expect(decodePayload(structuredSources)).toEqual(structuredSources);
    expect(
      decodePayload({
        ...structuredSources,
        answer_sources: [{...answerSource, citation_status: 'deterministic'}],
      }),
    ).toMatchObject({
      answer_sources: [{uid: 'uid-1', citation_status: 'deterministic'}],
    });
  });

  it.each([
    {
      name: 'missing required source title',
      payload: {...structuredSources, answer_sources: [{...answerSource, title: ''}]},
    },
    {
      name: 'answer source with candidate status',
      payload: {
        ...structuredSources,
        answer_sources: [{...answerSource, citation_status: 'candidate'}],
      },
    },
    {
      name: 'prompt basis with answer role',
      payload: {...structuredSources, prompt_basis: [{...promptSource, role: 'answer_source'}]},
    },
    {
      name: 'duplicate UID in one source array',
      payload: {
        ...structuredSources,
        courses: ['COMS W1004', 'COMS W1004'],
        answer_sources: [answerSource, answerSource],
      },
    },
    {
      name: 'duplicate UID in the prompt-basis array',
      payload: {
        ...structuredSources,
        prompt_basis: [promptSource, promptSource],
      },
    },
    {
      name: 'offering missing a required nullable field',
      payload: {
        ...structuredSources,
        answer_sources: [
          {
            ...answerSource,
            offerings: [
              {
                term: offering.term,
                section_id: offering.section_id,
                meeting_time: offering.meeting_time,
              },
            ],
          },
        ],
      },
    },
    {
      name: 'answer UID absent from prompt basis',
      payload: {
        ...structuredSources,
        prompt_basis: [{...promptSource, uid: 'another-uid'}],
      },
    },
    {
      name: 'answer fields differ from the matching prompt basis',
      payload: {
        ...structuredSources,
        answer_sources: [{...answerSource, title: 'A different title'}],
      },
    },
    {
      name: 'prompt citation labels are not sequential',
      payload: {
        ...structuredSources,
        prompt_basis: [{...promptSource, citation_label: 'S2'}],
      },
    },
    {
      name: 'legacy courses mirror with a different code',
      payload: {...structuredSources, courses: ['CSEE W4121']},
    },
    {
      name: 'legacy courses mirror out of answer-source order',
      payload: {
        ...structuredSources,
        courses: ['CSEE W4121', 'COMS W1004'],
        answer_sources: [
          answerSource,
          {...answerSource, uid: 'uid-2', course_code: 'CSEE W4121'},
        ],
        prompt_basis: [
          promptSource,
          {...promptSource, uid: 'uid-2', course_code: 'CSEE W4121'},
        ],
      },
    },
    {
      name: 'unsupported schema version',
      payload: {...structuredSources, schema_version: 3},
    },
  ])('rejects $name', ({payload}) => {
    expect(() => decodePayload(payload)).toThrow(SseProtocolError);
  });

  it('preserves duplicate term/section offerings from distinct catalog records', () => {
    const duplicateOfferings = {
      ...structuredSources,
      answer_sources: [{...answerSource, offerings: [offering, offering]}],
      prompt_basis: [{...promptSource, offerings: [offering, offering]}],
    };

    expect(decodePayload(duplicateOfferings)).toMatchObject({
      answer_sources: [{offerings: [offering, offering]}],
    });
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
