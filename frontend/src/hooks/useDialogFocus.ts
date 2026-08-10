import {type RefObject, useEffect, useRef} from 'react';

const FOCUSABLE_SELECTOR = [
  'a[href]',
  'button:not([disabled])',
  'input:not([disabled])',
  'select:not([disabled])',
  'textarea:not([disabled])',
  '[tabindex]:not([tabindex="-1"])',
].join(',');

type DialogFocusOptions = {
  isOpen: boolean;
  isTopmost?: boolean;
  canClose?: boolean;
  onClose: () => void;
  initialFocusRef?: RefObject<HTMLElement | null>;
};

/**
 * Keeps keyboard focus inside a modal, handles Escape, and restores the
 * invoking control when the modal closes. Mutable option refs avoid tearing
 * down the focus lifecycle when submission/topmost state changes.
 */
export function useDialogFocus({
  isOpen,
  isTopmost = true,
  canClose = true,
  onClose,
  initialFocusRef,
}: DialogFocusOptions) {
  const dialogRef = useRef<HTMLDivElement>(null);
  const onCloseRef = useRef(onClose);
  const topmostRef = useRef(isTopmost);
  const canCloseRef = useRef(canClose);
  const initialFocusRefRef = useRef(initialFocusRef);

  onCloseRef.current = onClose;
  topmostRef.current = isTopmost;
  canCloseRef.current = canClose;
  initialFocusRefRef.current = initialFocusRef;

  useEffect(() => {
    if (!isOpen) {
      return;
    }

    const previouslyFocused = document.activeElement as HTMLElement | null;
    const focusInitial = () => {
      const dialog = dialogRef.current;
      const requested = initialFocusRefRef.current?.current;
      const firstFocusable = dialog?.querySelector<HTMLElement>(FOCUSABLE_SELECTOR);
      (requested ?? firstFocusable ?? dialog)?.focus();
    };
    const frame = window.requestAnimationFrame(focusInitial);

    const onKeyDown = (event: KeyboardEvent) => {
      if (!topmostRef.current) {
        return;
      }
      if (event.key === 'Escape' && canCloseRef.current) {
        event.preventDefault();
        onCloseRef.current();
        return;
      }
      if (event.key !== 'Tab') {
        return;
      }

      const dialog = dialogRef.current;
      if (!dialog) {
        return;
      }
      const focusable = Array.from(dialog.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTOR)).filter(
        (element) => !element.hidden && element.getAttribute('aria-hidden') !== 'true',
      );
      if (!focusable.length) {
        event.preventDefault();
        dialog.focus();
        return;
      }

      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      const active = document.activeElement;
      if (event.shiftKey && (active === first || !dialog.contains(active))) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && (active === last || !dialog.contains(active))) {
        event.preventDefault();
        first.focus();
      }
    };

    document.addEventListener('keydown', onKeyDown);
    return () => {
      window.cancelAnimationFrame(frame);
      document.removeEventListener('keydown', onKeyDown);
      if (previouslyFocused?.isConnected) {
        previouslyFocused.focus();
      }
    };
  }, [isOpen]);

  return dialogRef;
}
