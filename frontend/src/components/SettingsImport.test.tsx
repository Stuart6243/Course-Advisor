import {act} from 'react';
import {createRoot, type Root} from 'react-dom/client';
import {afterEach, beforeEach, describe, expect, it, vi} from 'vitest';

const apiMocks = vi.hoisted(() => ({
  importFile: vi.fn(),
}));

vi.mock('../services/api', async (importOriginal) => ({
  ...(await importOriginal<typeof import('../services/api')>()),
  importFile: apiMocks.importFile,
}));

import SettingsDrawer from './SettingsDrawer';

let container: HTMLDivElement;
let root: Root;

beforeEach(() => {
  apiMocks.importFile.mockReset();
  container = document.createElement('div');
  document.body.appendChild(container);
  root = createRoot(container);
});

afterEach(async () => {
  await act(async () => root.unmount());
  container.remove();
});

describe('Settings syllabus import status', () => {
  it('rejects DOCX locally without calling the import API', async () => {
    await act(async () =>
      root.render(
        <SettingsDrawer
          isOpen
          onClose={vi.fn()}
          language="en"
          onLanguageChange={vi.fn()}
          messages={[]}
          maxHistoryTurns={10}
          maxResults={5}
          onSettingsChange={vi.fn()}
        />,
      ),
    );
    const input = container.querySelector<HTMLInputElement>('input[type="file"]')!;
    Object.defineProperty(input, 'files', {
      configurable: true,
      value: [new File(['docx'], 'course.docx')],
    });

    await act(async () => {
      input.dispatchEvent(new Event('change', {bubbles: true}));
    });

    expect(apiMocks.importFile).not.toHaveBeenCalled();
    expect(container.textContent).toContain('Only PDF, HTML, and HTM');
  });

  it('states that a review syllabus is not searchable', async () => {
    apiMocks.importFile.mockResolvedValue({
      success: true,
      status: 'review',
      search_visible: false,
      message: 'attached for review',
      course: {course_code: 'COMS W4111', title: 'Introduction to Databases'},
    });
    await act(async () =>
      root.render(
        <SettingsDrawer
          isOpen
          onClose={vi.fn()}
          language="en"
          onLanguageChange={vi.fn()}
          messages={[]}
          maxHistoryTurns={10}
          maxResults={5}
          onSettingsChange={vi.fn()}
        />,
      ),
    );
    const input = container.querySelector<HTMLInputElement>('input[type="file"]')!;
    Object.defineProperty(input, 'files', {
      configurable: true,
      value: [new File(['syllabus'], 'course.pdf', {type: 'application/pdf'})],
    });

    await act(async () => {
      input.dispatchEvent(new Event('change', {bubbles: true}));
    });

    expect(container.textContent).toContain('Saved for review and not searchable yet');
    expect(container.textContent).not.toContain('Course imported successfully!');
  });

  it('lets a keyboard user cancel an in-flight upload', async () => {
    apiMocks.importFile.mockImplementation(
      async (_file: File, signal: AbortSignal) =>
        await new Promise((_resolve, reject) => {
          signal.addEventListener('abort', () => {
            reject(new DOMException('aborted', 'AbortError'));
          });
        }),
    );
    await act(async () =>
      root.render(
        <SettingsDrawer
          isOpen
          onClose={vi.fn()}
          language="en"
          onLanguageChange={vi.fn()}
          messages={[]}
          maxHistoryTurns={10}
          maxResults={5}
          onSettingsChange={vi.fn()}
        />,
      ),
    );
    const input = container.querySelector<HTMLInputElement>('input[type="file"]')!;
    expect(input.getAttribute('aria-label')).toBe('Choose file');
    Object.defineProperty(input, 'files', {
      configurable: true,
      value: [new File(['syllabus'], 'course.html', {type: 'text/html'})],
    });

    await act(async () => {
      input.dispatchEvent(new Event('change', {bubbles: true}));
      await Promise.resolve();
    });
    const cancel = Array.from(container.querySelectorAll('button')).find(
      (button) => button.textContent === 'Cancel import',
    )!;
    cancel.focus();
    await act(async () => {
      cancel.click();
      await Promise.resolve();
    });

    expect(container.textContent).toContain('Import cancelled.');
  });

  it('opens the manual form and shows the backend message for a structured 422 result', async () => {
    apiMocks.importFile.mockResolvedValue({
      success: false,
      needs_manual_input: true,
      status: 'rejected',
      message: 'Credits and section identity are required.',
      missing_fields: ['points_raw', 'section_id'],
      partial_data: {
        course_code: 'COMS W4111',
        title: 'Introduction to Databases',
      },
    });
    await act(async () =>
      root.render(
        <SettingsDrawer
          isOpen
          onClose={vi.fn()}
          language="en"
          onLanguageChange={vi.fn()}
          messages={[]}
          maxHistoryTurns={10}
          maxResults={5}
          onSettingsChange={vi.fn()}
        />,
      ),
    );
    const input = container.querySelector<HTMLInputElement>('input[type="file"]')!;
    Object.defineProperty(input, 'files', {
      configurable: true,
      value: [new File(['syllabus'], 'course.pdf', {type: 'application/pdf'})],
    });

    await act(async () => {
      input.dispatchEvent(new Event('change', {bubbles: true}));
    });

    const manualDialog = container.querySelector('[aria-labelledby="manual-import-title"]')!;
    expect(manualDialog).not.toBeNull();
    expect(manualDialog.textContent).toContain('Credits and section identity are required.');
    expect(container.querySelector<HTMLInputElement>('input[aria-invalid="true"]')).not.toBeNull();
  });
});
