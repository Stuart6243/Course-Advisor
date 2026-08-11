import {useCallback, useEffect, useRef, useState} from 'react';
import {ChatSettings, Language, Message} from '../types';
import {safeUUID, sendMessageStream} from '../services/api';
import {MAX_MESSAGE_LENGTH} from '../constants';
import i18n from '../i18n';
import type {StreamErrorEvent, StreamSourcesEvent} from '../services/sse';

const nowTime = () =>
  new Date().toLocaleTimeString([], {hour: '2-digit', minute: '2-digit'});

export const localizeStreamError = (event: StreamErrorEvent, language: Language) =>
  event.code ? i18n.getFixedT(language)(`errors.${event.code}`) : event.message;

export function useChat(language: Language, settings: ChatSettings) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [conversationId, setConversationId] = useState(() => safeUUID());
  const [contextLost, setContextLost] = useState(false);
  const loadingRef = useRef(false);
  const abortRef = useRef<AbortController | null>(null);
  const generationRef = useRef(0);
  const expectedRevisionRef = useRef<number | null>(null);
  // 供 onMeta 回调读取当前消息数，避免把 messages 加进 sendMessage 的依赖数组
  const messagesRef = useRef<Message[]>([]);
  messagesRef.current = messages;

  const setLoading = (next: boolean) => {
    loadingRef.current = next;
    setIsLoading(next);
  };

  // 组件卸载时中止在途请求，避免向已卸载组件 setState。
  useEffect(() => {
    return () => {
      generationRef.current += 1;
      abortRef.current?.abort();
    };
  }, []);

  const sendMessage = useCallback(
    async (text: string) => {
      const trimmed = text.trim();
      if (!trimmed || trimmed.length > MAX_MESSAGE_LENGTH || loadingRef.current) {
        return;
      }

      const userId = safeUUID();
      const assistantId = safeUUID();

      const userMessage: Message = {
        id: userId,
        role: 'user',
        content: trimmed,
        time: nowTime(),
      };

      const assistantMessage: Message = {
        id: assistantId,
        role: 'assistant',
        content: '',
        time: nowTime(),
        isStreaming: true,
        status: 'streaming',
      };

      setMessages((prev) => [...prev, userMessage, assistantMessage]);
      setLoading(true);

      const controller = new AbortController();
      abortRef.current = controller;
      const generation = ++generationRef.current;
      let requestRevision: number | null = null;
      let streamedContent = '';
      let fallbackSnapshot: {content: string; provider: string} | null = null;
      let pendingSources: StreamSourcesEvent | null = null;
      let activeProvider: string | undefined;
      const isCurrentGeneration = () =>
        generationRef.current === generation && abortRef.current === controller;

      const finishAssistant = (
        update: (message: Message) => Message,
      ) => {
        if (!isCurrentGeneration()) {
          return;
        }
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === assistantId ? update(msg) : msg,
          ),
        );
        abortRef.current = null;
        setLoading(false);
      };

      try {
        await sendMessageStream(
          trimmed,
          conversationId,
          language,
          settings,
          {
            onChunk: (chunk) => {
              if (!isCurrentGeneration()) {
                return;
              }
              streamedContent += chunk;
              setMessages((prev) =>
                prev.map((msg) =>
                  msg.id === assistantId
                    ? {
                        ...msg,
                        content: msg.content + chunk,
                      }
                    : msg,
                ),
              );
            },
            onSources: (event) => {
              if (!isCurrentGeneration()) {
                return;
              }
              pendingSources = event;
            },
            onDone: (event) => {
              activeProvider = event.provider;
              if (isCurrentGeneration() && requestRevision !== null) {
                expectedRevisionRef.current = requestRevision + 1;
              }
              finishAssistant((msg) => ({
                ...msg,
                isStreaming: false,
                status: 'complete',
                provider: event.provider ?? msg.provider,
                fallbackUsed: event.fallback_used ?? msg.fallbackUsed,
                fallbackFailed: false,
                fallbackReason: event.fallback_reason ?? msg.fallbackReason,
                sources: pendingSources ?? undefined,
              }));
            },
            onError: (event) => {
              pendingSources = null;
              finishAssistant((msg) => {
                const fallbackPrimary = fallbackSnapshot?.content ?? '';
                const hasServerPartial = event.partial_content !== undefined;
                const shouldRestoreClientSnapshot = Boolean(
                  !hasServerPartial && fallbackPrimary,
                );
                const recovered = hasServerPartial
                  ? event.partial_content ?? ''
                  : shouldRestoreClientSnapshot
                    ? fallbackPrimary
                    : streamedContent || msg.content;
                const errorMessage = localizeStreamError(event, language);
                const content = recovered
                  ? `${recovered}\n\n_⚠️ ${errorMessage}_`
                  : errorMessage;
                return {
                  ...msg,
                  content,
                  isStreaming: false,
                  status: recovered ? 'interrupted' : 'error',
                  provider: hasServerPartial
                    ? event.partial_provider ?? event.provider ?? msg.provider
                    : shouldRestoreClientSnapshot
                      ? fallbackSnapshot?.provider
                      : event.provider ?? activeProvider ?? msg.provider,
                  fallbackUsed: event.fallback_used ?? msg.fallbackUsed,
                  fallbackFailed: Boolean(fallbackSnapshot),
                  fallbackReason: event.fallback_reason ?? msg.fallbackReason,
                  sources: undefined,
                };
              });
            },
            onAbort: () => {
              if (!isCurrentGeneration()) {
                return;
              }
              pendingSources = null;
              setMessages((prev) => {
                const stoppedContent = streamedContent || fallbackSnapshot?.content || '';
                if (!stoppedContent) {
                  return prev.filter((msg) => msg.id !== assistantId);
                }
                return prev.map((msg) =>
                  msg.id === assistantId
                    ? {
                        ...msg,
                        content: stoppedContent,
                        isStreaming: false,
                        status: 'stopped',
                        provider: streamedContent
                          ? activeProvider ?? msg.provider
                          : fallbackSnapshot?.provider ?? msg.provider,
                        fallbackFailed: Boolean(!streamedContent && fallbackSnapshot?.content),
                        sources: undefined,
                      }
                    : msg,
                );
              });
              abortRef.current = null;
              setLoading(false);
            },
            onFallback: (event) => {
              if (!isCurrentGeneration()) {
                return;
              }
              fallbackSnapshot = {
                content: streamedContent,
                provider: event.from,
              };
              pendingSources = null;
              streamedContent = '';
              activeProvider = event.to;
              // replace/reset 语义：Groq 局部回答不得与 Ollama 完整回答拼接。
              setMessages((prev) =>
                prev.map((msg) =>
                  msg.id === assistantId
                    ? {
                        ...msg,
                        content: '',
                        sources: undefined,
                        provider: event.to,
                        fallbackUsed: true,
                        fallbackFailed: false,
                        fallbackReason: event.reason,
                      }
                    : msg,
                ),
              );
            },
            onMeta: (event) => {
              if (!isCurrentGeneration()) {
                return;
              }
              if (event.provider) {
                activeProvider = event.provider;
                setMessages((prev) =>
                  prev.map((msg) =>
                    msg.id === assistantId ? {...msg, provider: event.provider} : msg,
                  ),
                );
              }

              let revisionDiverged = false;
              if (typeof event.revision === 'number') {
                requestRevision = event.revision;
                if (
                  expectedRevisionRef.current !== null &&
                  event.revision !== expectedRevisionRef.current
                ) {
                  revisionDiverged = true;
                }
              }

              const priorTurns = Math.floor(
                messagesRef.current.filter((m) => m.role === 'user').length,
              );
              const historyTurns = Number(event.history_turns ?? 0);
              if (revisionDiverged || (priorTurns > 1 && historyTurns === 0)) {
                setContextLost(true);
              }
            },
          },
          controller.signal,
        );
      } catch (err) {
        if (!isCurrentGeneration()) {
          return;
        }
        pendingSources = null;
        const msg = err instanceof Error
          ? err.message
          : i18n.getFixedT(language)('errors.request_failed');
        finishAssistant((message) => ({
          ...message,
          content: message.content
            ? `${message.content}\n\n_⚠️ ${msg}_`
            : msg,
          isStreaming: false,
          status: message.content ? 'interrupted' : 'error',
          sources: undefined,
        }));
      }
    },
    [conversationId, language, settings],
  );

  /** 停止生成：保留已经流出的内容。 */
  const stopGeneration = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  const newChat = useCallback(() => {
    // 切换会话前先中止在途请求，否则旧的流会继续写入并让状态错乱。
    generationRef.current += 1;
    abortRef.current?.abort();
    abortRef.current = null;
    setMessages([]);
    setConversationId(safeUUID());
    expectedRevisionRef.current = null;
    setContextLost(false);
    setLoading(false);
  }, []);

  return {messages, isLoading, sendMessage, stopGeneration, newChat, contextLost};
}
