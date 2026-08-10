import {useCallback, useEffect, useRef, useState} from 'react';
import {AlertTriangle, ArrowDown, ArrowUp, Square} from 'lucide-react';
import {useTranslation} from 'react-i18next';
import {MAX_MESSAGE_LENGTH} from '../constants';
import {Message} from '../types';
import AutoResizeTextarea from './AutoResizeTextarea';
import MessageBubble from './MessageBubble';

type Props = {
  messages: Message[];
  isLoading: boolean;
  onSend: (text: string) => Promise<void> | void;
  onStop?: () => void;
  contextLost?: boolean;
};

/** 距底部多少像素以内算「贴着底部」。 */
const STICK_TO_BOTTOM_THRESHOLD = 120;

export default function ChatView({
  messages,
  isLoading,
  onSend,
  onStop,
  contextLost,
}: Props) {
  const {t} = useTranslation();
  const [inputText, setInputText] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const containerRef = useRef<HTMLElement>(null);
  const [stickToBottom, setStickToBottom] = useState(true);
  const composingRef = useRef(false);
  const messageTooLong = inputText.length > MAX_MESSAGE_LENGTH;

  const handleScroll = useCallback(() => {
    const el = containerRef.current;
    if (!el) {
      return;
    }
    const distanceFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
    setStickToBottom(distanceFromBottom <= STICK_TO_BOTTOM_THRESHOLD);
  }, []);

  // 只有用户本来就贴着底部时才自动滚动。
  // 旧版每来一个 token 就无条件 scrollIntoView({behavior:'smooth'})，
  // 导致生成期间没法往上翻看历史消息，一滚就被拽回底部。
  useEffect(() => {
    if (!stickToBottom) {
      return;
    }
    const isStreaming = messages.some((msg) => msg.isStreaming);
    messagesEndRef.current?.scrollIntoView({
      // 流式期间用 auto，避免 smooth 动画被后续 token 不断打断而抖动
      behavior: isStreaming ? 'auto' : 'smooth',
    });
  }, [messages, stickToBottom]);

  const scrollToBottom = () => {
    setStickToBottom(true);
    messagesEndRef.current?.scrollIntoView({behavior: 'smooth'});
  };

  const handleSend = () => {
    if (!inputText.trim() || messageTooLong || isLoading) {
      return;
    }
    const text = inputText;
    setInputText('');
    void onSend(text);
  };

  return (
    <>
      <main
        ref={containerRef}
        onScroll={handleScroll}
        role="log"
        aria-live="polite"
        aria-relevant="additions text"
        aria-busy={isLoading}
        className="flex-1 overflow-y-auto relative pb-32"
        id="chat-container"
      >
        <div className="max-w-3xl mx-auto px-4 py-8 flex flex-col gap-6">
          {contextLost ? (
            <div
              className="flex items-start gap-2 rounded-lg border border-amber-500/30 bg-amber-500/10 px-4 py-3 text-sm text-amber-200"
              role="alert"
            >
              <AlertTriangle className="w-4 h-4 mt-0.5 flex-none" />
              <span>{t('chat.contextLost')}</span>
            </div>
          ) : null}
          {messages.map((msg, idx) => (
            <MessageBubble key={msg.id} message={msg} index={idx} />
          ))}
          <div ref={messagesEndRef} className="h-12" />
        </div>
      </main>

      {isLoading && onStop ? (
        <div className="fixed bottom-36 left-1/2 -translate-x-1/2 z-40">
          <button
            type="button"
            onClick={onStop}
            className="flex items-center gap-2 rounded-full bg-surface-dark border border-border-dark px-4 py-2 text-sm text-slate-200 shadow-lg hover:bg-slate-700 hover:text-white transition-colors"
          >
            <Square className="w-3.5 h-3.5 fill-current" />
            {t('chat.stop')}
          </button>
        </div>
      ) : null}

      {!stickToBottom ? (
        <div className="fixed bottom-28 right-8 z-30">
          <button
            type="button"
            onClick={scrollToBottom}
            aria-label={t('chat.scrollToBottom')}
            title={t('chat.scrollToBottom')}
            className="w-10 h-10 rounded-full bg-border-dark border border-slate-700 text-slate-300 shadow-lg hover:bg-slate-700 hover:text-white flex items-center justify-center transition-all group"
          >
            <ArrowDown className="w-5 h-5 group-hover:translate-y-0.5 transition-transform" />
          </button>
        </div>
      ) : null}

      <div className="fixed bottom-0 left-0 w-full bg-gradient-to-t from-background-dark via-background-dark/95 to-transparent pt-10 pb-6 px-4 z-40 pointer-events-none">
        <div className="max-w-3xl mx-auto relative pointer-events-auto">
          <div className="relative bg-surface-dark border border-border-dark rounded-xl shadow-2xl focus-within:ring-2 focus-within:ring-primary/50 focus-within:border-primary transition-all duration-200">
            <AutoResizeTextarea
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              onKeyDown={(e) => {
                if (
                  e.key === 'Enter' &&
                  !e.shiftKey &&
                  !e.nativeEvent.isComposing &&
                  !composingRef.current
                ) {
                  e.preventDefault();
                  handleSend();
                }
              }}
              onCompositionStart={() => {
                composingRef.current = true;
              }}
              onCompositionEnd={() => {
                composingRef.current = false;
              }}
              className="w-full bg-transparent text-slate-200 placeholder-slate-500 text-[15px] p-4 pr-12 rounded-xl focus:ring-0 focus:outline-none resize-none overflow-hidden max-h-48"
              placeholder={t('chat.placeholder')}
              ariaLabel={t('chat.placeholder')}
              ariaDescribedBy="chat-composer-help"
              minHeight="56px"
              disabled={isLoading}
            />
            <button
              type="button"
              onClick={handleSend}
              disabled={!inputText.trim() || messageTooLong || isLoading}
              aria-label={t('chat.send')}
              className="absolute bottom-2 right-2 p-1.5 rounded-lg bg-primary text-white hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              <ArrowUp className="w-5 h-5" />
            </button>
          </div>
          <div
            id="chat-composer-help"
            className="mt-1 min-h-5 px-1 text-xs"
            aria-live="polite"
          >
            {messageTooLong ? (
              <p className="text-red-400">
                {t('chat.messageTooLong', {max: MAX_MESSAGE_LENGTH, count: inputText.length})}
              </p>
            ) : inputText.length > MAX_MESSAGE_LENGTH * 0.9 ? (
              <p className="text-right text-slate-500">
                {inputText.length}/{MAX_MESSAGE_LENGTH}
              </p>
            ) : null}
          </div>
          <div className="text-center mt-2">
            <p className="text-[11px] text-slate-500">{t('chat.disclaimer')}</p>
          </div>
        </div>
      </div>
    </>
  );
}
