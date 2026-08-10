import {act} from 'react';
import {createRoot, type Root} from 'react-dom/client';
import {afterEach, beforeEach, describe, expect, it, vi} from 'vitest';
import SettingsDrawer from './SettingsDrawer';

let container: HTMLDivElement;
let root: Root;

const baseProps = {
  onClose: vi.fn(),
  language: 'en' as const,
  onLanguageChange: vi.fn(),
  messages: [],
  maxHistoryTurns: 10,
  maxResults: 5,
  onSettingsChange: vi.fn(),
};

function render(isOpen: boolean) {
  root.render(<SettingsDrawer {...baseProps} isOpen={isOpen} />);
}

function setInputValue(input: HTMLInputElement, value: string) {
  const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')?.set;
  setter?.call(input, value);
  input.dispatchEvent(new Event('input', {bubbles: true}));
}

beforeEach(() => {
  vi.clearAllMocks();
  container = document.createElement('div');
  document.body.appendChild(container);
  root = createRoot(container);
});

afterEach(async () => {
  await act(async () => root.unmount());
  container.remove();
});

describe('SettingsDrawer drafts', () => {
  it('restores saved values after closing without saving and caps turns at ten', async () => {
    await act(async () => render(true));
    let historyInput = container.querySelector<HTMLInputElement>('#max-history-turns')!;
    expect(historyInput.max).toBe('10');

    await act(async () => setInputValue(historyInput, '3'));
    expect(historyInput.value).toBe('3');

    await act(async () => render(false));
    await act(async () => render(true));
    historyInput = container.querySelector<HTMLInputElement>('#max-history-turns')!;
    expect(historyInput.value).toBe('10');
  });

  it('clamps a manually entered history value to ten when saving', async () => {
    await act(async () => render(true));
    const historyInput = container.querySelector<HTMLInputElement>('#max-history-turns')!;
    await act(async () => setInputValue(historyInput, '37'));
    const save = Array.from(container.querySelectorAll('button')).find(
      (button) => button.textContent === 'Save Settings',
    )!;

    await act(async () => save.click());

    expect(baseProps.onSettingsChange).toHaveBeenCalledWith({
      maxHistoryTurns: 10,
      maxResults: 5,
    });
  });
});
