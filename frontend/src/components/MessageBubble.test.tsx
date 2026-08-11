import {act} from 'react';
import {createRoot, type Root} from 'react-dom/client';
import {afterEach, beforeEach, describe, expect, it} from 'vitest';

import MessageBubble from './MessageBubble';
import type {CourseSource} from '../types';

let container: HTMLDivElement;
let root: Root;

beforeEach(() => {
  container = document.createElement('div');
  document.body.appendChild(container);
  root = createRoot(container);
});

afterEach(async () => {
  await act(async () => root.unmount());
  container.remove();
});

describe('MessageBubble response metadata', () => {
  it('shows the provider and an explicit incomplete-fallback status', async () => {
    await act(async () =>
      root.render(
        <MessageBubble
          index={0}
          message={{
            id: 'answer',
            role: 'assistant',
            content: 'Groq partial',
            time: '12:00',
            provider: 'groq',
            fallbackUsed: true,
            fallbackFailed: true,
            fallbackReason: 'timeout',
            status: 'interrupted',
          }}
        />,
      ),
    );

    expect(container.textContent).toContain('groq');
    expect(container.textContent).toContain('Local AI did not complete');
    expect(container.querySelector('[aria-label*="timeout"]')).not.toBeNull();
  });

  it('wraps a wide Markdown table in its own horizontal scroller', async () => {
    await act(async () =>
      root.render(
        <MessageBubble
          index={0}
          message={{
            id: 'table',
            role: 'assistant',
            content: '| Course | Schedule |\n| --- | --- |\n| COMS W4111 | Monday |',
            time: '12:00',
            status: 'complete',
          }}
        />,
      ),
    );

    const table = container.querySelector('table')!;
    expect(table.parentElement?.className).toContain('overflow-x-auto');
  });

  it('labels legacy string sources as generation candidates, not verified sources', async () => {
    await act(async () =>
      root.render(
        <MessageBubble
          index={0}
          message={{
            id: 'legacy-sources',
            role: 'assistant',
            content: 'Legacy answer',
            time: '12:00',
            status: 'complete',
            sources: {
              type: 'sources',
              schema_version: 1,
              courses: ['COMS W4111'],
            },
          }}
        />,
      ),
    );

    expect(container.textContent).toContain('Courses considered for generation');
    expect(container.textContent).not.toContain('Verified answer sources');
    expect(container.querySelector('[data-source-legacy="true"]')?.textContent).toBe(
      'COMS W4111',
    );
    expect(container.querySelector('[data-source-schema-version="1"]')).not.toBeNull();
  });

  it('renders duplicate course codes with exact UID, source, and offering identities', async () => {
    const makeSource = (
      uid: string,
      sectionId: string,
      sourceLabel: string,
      role: CourseSource['role'],
      citationStatus: CourseSource['citation_status'],
    ): CourseSource => ({
      uid,
      course_code: 'ORCA E2500',
      title: 'Foundations of Data Science',
      citation_label: uid === 'uid-a' ? 'S1' : 'S2',
      source_label: sourceLabel,
      role,
      citation_status: citationStatus,
      offerings: [
        {
          term: 'Fall 2025',
          section_id: sectionId,
          meeting_time: 'F 10:10am - 12:40pm',
          location: '329 Pupin Laboratories',
        },
      ],
    });
    const answerA = makeSource(
      'uid-a',
      '001/11859',
      'Computer Science Bulletin',
      'answer_source',
      'verified',
    );
    const answerB = makeSource(
      'uid-b',
      '001/13321',
      'Applied Mathematics Bulletin',
      'answer_source',
      'deterministic',
    );
    const basisA = {...answerA, role: 'prompt_basis', citation_status: 'candidate'} as CourseSource;
    const basisB = {...answerB, role: 'prompt_basis', citation_status: 'candidate'} as CourseSource;
    const candidate = {
      ...makeSource(
        'uid-c',
        '002/22222',
        'Operations Research Bulletin',
        'prompt_basis',
        'candidate',
      ),
      course_code: 'CSOR W4246',
      title: 'Algorithms for Data Science',
      citation_label: 'S3',
    };

    await act(async () =>
      root.render(
        <MessageBubble
          index={0}
          message={{
            id: 'structured-sources',
            role: 'assistant',
            content: 'Structured answer',
            time: '12:00',
            status: 'complete',
            sources: {
              type: 'sources',
              schema_version: 2,
              courses: ['ORCA E2500', 'ORCA E2500'],
              answer_sources: [answerA, answerB],
              prompt_basis: [basisA, basisB, candidate],
            },
          }}
        />,
      ),
    );

    expect(container.textContent).toContain('Verified answer sources');
    expect(container.textContent).toContain('Courses considered for generation');
    expect(
      container.querySelectorAll('[data-source-group="answer_sources"] [data-source-uid]'),
    ).toHaveLength(2);
    expect(
      container.querySelector('[data-source-uid="uid-a"] [data-source-section="001/11859"]'),
    ).not.toBeNull();
    expect(
      container.querySelector('[data-source-uid="uid-b"] [data-source-section="001/13321"]'),
    ).not.toBeNull();
    expect(container.querySelector('[data-source-uid="uid-c"]')).not.toBeNull();
    expect(container.textContent).toContain('Computer Science Bulletin');
    expect(container.textContent).toContain('Applied Mathematics Bulletin');
    expect(container.textContent).toContain('uid-a');
    expect(container.textContent).toContain('uid-b');
  });

  it('preserves repeated offerings with the same term and section', async () => {
    const repeated: CourseSource = {
      uid: 'uid-repeat',
      course_code: 'ORCA E2500',
      title: 'Foundations of Data Science',
      citation_label: 'S1',
      source_label: 'Engineering Bulletin',
      role: 'answer_source',
      citation_status: 'deterministic',
      offerings: [
        {term: 'Fall 2025', section_id: '001/11859', meeting_time: null, location: null},
        {term: 'Fall 2025', section_id: '001/11859', meeting_time: null, location: null},
      ],
    };

    await act(async () =>
      root.render(
        <MessageBubble
          index={0}
          message={{
            id: 'repeated-offerings',
            role: 'assistant',
            content: 'Answer',
            time: '12:00',
            status: 'complete',
            sources: {
              type: 'sources',
              schema_version: 2,
              courses: ['ORCA E2500'],
              answer_sources: [repeated],
              prompt_basis: [
                {...repeated, role: 'prompt_basis', citation_status: 'candidate'},
              ],
            },
          }}
        />,
      ),
    );

    const offerings = container.querySelectorAll(
      '[data-source-uid="uid-repeat"] [data-source-section="001/11859"]',
    );
    expect(offerings).toHaveLength(2);
    expect(offerings[0].getAttribute('data-source-offering-index')).toBe('0');
    expect(offerings[1].getAttribute('data-source-offering-index')).toBe('1');
  });
});
