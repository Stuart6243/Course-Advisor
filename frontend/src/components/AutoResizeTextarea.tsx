import {
  type ChangeEvent,
  type CompositionEvent,
  type KeyboardEvent,
  useEffect,
  useRef,
} from 'react';

type Props = {
  value: string;
  onChange: (e: ChangeEvent<HTMLTextAreaElement>) => void;
  onKeyDown: (e: KeyboardEvent<HTMLTextAreaElement>) => void;
  placeholder: string;
  className: string;
  minHeight: string;
  disabled?: boolean;
  maxLength?: number;
  ariaLabel?: string;
  ariaDescribedBy?: string;
  onCompositionStart?: (event: CompositionEvent<HTMLTextAreaElement>) => void;
  onCompositionEnd?: (event: CompositionEvent<HTMLTextAreaElement>) => void;
};

export default function AutoResizeTextarea({
  value,
  onChange,
  onKeyDown,
  placeholder,
  className,
  minHeight,
  disabled,
  maxLength,
  ariaLabel,
  ariaDescribedBy,
  onCompositionStart,
  onCompositionEnd,
}: Props) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`;
    }
  }, [value]);

  return (
    <textarea
      ref={textareaRef}
      value={value}
      onChange={onChange}
      onKeyDown={onKeyDown}
      placeholder={placeholder}
      className={className}
      rows={1}
      style={{minHeight}}
      disabled={disabled}
      maxLength={maxLength}
      aria-label={ariaLabel}
      aria-describedby={ariaDescribedBy}
      onCompositionStart={onCompositionStart}
      onCompositionEnd={onCompositionEnd}
    />
  );
}
