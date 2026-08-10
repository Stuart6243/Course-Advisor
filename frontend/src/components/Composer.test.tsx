import {act} from 'react';
import {createRoot, type Root} from 'react-dom/client';
import {afterEach, beforeEach, describe, expect, it, vi} from 'vitest';
import ChatView from './ChatView';
import LandingView from './LandingView';

let container: HTMLDivElement;
let root: Root;

function setTextareaValue(textarea: HTMLTextAreaElement, value: string) {
  const setter = Object.getOwnPropertyDescriptor(
    HTMLTextAreaElement.prototype,
    'value',
  )?.set;
  setter?.call(textarea, value);
  textarea.dispatchEvent(new Event('input', {bubbles: true}));
}

beforeEach(async () => {
  container = document.createElement('div');
  document.body.appendChild(container);
  root = createRoot(container);
});

afterEach(async () => {
  await act(async () => root.unmount());
  container.remove();
});

describe('message composer', () => {
  it('does not submit Enter while an IME composition is active', async () => {
    const onStart = vi.fn();
    await act(async () => root.render(<LandingView onStart={onStart} />));
    const textarea = container.querySelector('textarea')!;

    await act(async () => setTextareaValue(textarea, '中文问题'));
    await act(async () => {
      textarea.dispatchEvent(new CompositionEvent('compositionstart', {bubbles: true}));
      textarea.dispatchEvent(
        new KeyboardEvent('keydown', {key: 'Enter', bubbles: true, cancelable: true}),
      );
    });
    expect(onStart).not.toHaveBeenCalled();

    await act(async () => {
      textarea.dispatchEvent(new CompositionEvent('compositionend', {bubbles: true}));
      textarea.dispatchEvent(
        new KeyboardEvent('keydown', {key: 'Enter', bubbles: true, cancelable: true}),
      );
    });
    expect(onStart).toHaveBeenCalledWith('中文问题');
  });

  it('disables whitespace and blocks a 4001-character message with an inline error', async () => {
    await act(async () => root.render(<LandingView onStart={vi.fn()} />));
    const textarea = container.querySelector('textarea')!;
    const send = container.querySelector<HTMLButtonElement>('button[aria-label="Send message"]')!;

    await act(async () => setTextareaValue(textarea, '   '));
    expect(send.disabled).toBe(true);

    await act(async () => setTextareaValue(textarea, 'x'.repeat(4001)));
    expect(send.disabled).toBe(true);
    expect(container.textContent).toContain('4001/4000');
  });

  it('also guards IME Enter in the in-conversation composer', async () => {
    const onSend = vi.fn();
    await act(async () =>
      root.render(<ChatView messages={[]} isLoading={false} onSend={onSend} />),
    );
    const textarea = container.querySelector('textarea')!;

    await act(async () => setTextareaValue(textarea, '连续追问'));
    await act(async () => {
      textarea.dispatchEvent(new CompositionEvent('compositionstart', {bubbles: true}));
      textarea.dispatchEvent(
        new KeyboardEvent('keydown', {key: 'Enter', bubbles: true, cancelable: true}),
      );
    });
    expect(onSend).not.toHaveBeenCalled();

    await act(async () => {
      textarea.dispatchEvent(new CompositionEvent('compositionend', {bubbles: true}));
      textarea.dispatchEvent(
        new KeyboardEvent('keydown', {key: 'Enter', bubbles: true, cancelable: true}),
      );
    });
    expect(onSend).toHaveBeenCalledWith('连续追问');
  });
});
