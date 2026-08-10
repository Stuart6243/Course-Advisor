import {act} from 'react';
import {createRoot, type Root} from 'react-dom/client';
import {afterEach, beforeEach, describe, expect, it, vi} from 'vitest';

import ManualImportForm from './ManualImportForm';

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

describe('ManualImportForm publishing guard', () => {
  it('marks identity/points as required and blocks submission without credits', async () => {
    const onSubmit = vi.fn();
    await act(async () =>
      root.render(
        <ManualImportForm
          isOpen
          onClose={vi.fn()}
          partialData={{
            course_code: 'COMS W4111',
            title: 'Introduction to Databases',
            term: 'Spring 2026',
            section_id: '001',
          }}
          onSubmit={onSubmit}
        />,
      ),
    );

    const inputs = Array.from(container.querySelectorAll<HTMLInputElement>('input'));
    expect(inputs).toHaveLength(5);
    expect(inputs.every((input) => input.getAttribute('aria-required') === 'true')).toBe(true);

    await act(async () => {
      container.querySelector('form')?.dispatchEvent(
        new SubmitEvent('submit', {bubbles: true, cancelable: true}),
      );
    });

    expect(onSubmit).not.toHaveBeenCalled();
    expect(container.textContent).toContain('Credits are required before publishing.');
    expect(inputs[4].getAttribute('aria-invalid')).toBe('true');
    expect(inputs[4].getAttribute('aria-describedby')).toBe('manual-import-error');
  });

  it('keeps focus inside the dialog when tabbing past the final control', async () => {
    await act(async () =>
      root.render(
        <ManualImportForm isOpen onClose={vi.fn()} onSubmit={vi.fn()} />,
      ),
    );
    const submit = container.querySelector<HTMLButtonElement>('button[type="submit"]')!;
    const close = container.querySelector<HTMLButtonElement>(
      'button[aria-label="Close syllabus form"]',
    )!;
    submit.focus();

    document.dispatchEvent(
      new KeyboardEvent('keydown', {key: 'Tab', bubbles: true, cancelable: true}),
    );

    expect(document.activeElement).toBe(close);
  });
});
