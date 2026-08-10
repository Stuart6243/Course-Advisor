import {act} from 'react';
import {createRoot, type Root} from 'react-dom/client';
import {afterEach, beforeEach, describe, expect, it} from 'vitest';

import MessageBubble from './MessageBubble';

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
});
