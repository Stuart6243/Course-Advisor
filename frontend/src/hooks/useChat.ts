import {useCallback, useEffect, useRef, useState} from 'react';
import {ChatSettings, Language, Message} from '../types';
import {safeUUID, sendMessageStream} from '../services/api';

const nowTime = () =>
  new Date().toLocaleTimeString([], {hour: '2-digit', minute: '2-digit'});

export function useChat(language: Language, settings: ChatSettings) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [conversationId, setConversationId] = useState(() => safeUUID());
  const loadingRef = useRef(false);
  const abortRef = useRef<AbortController | null>(null);

  const setLoading = (next: boolean) => {
    loadingRef.current = next;
    setIsLoading(next);
  };

  // 组件卸载时中止在途请求，避免向已卸载组件 setState。
  useEffect(() => {
    return () => {
      abortRef.current?.abort();
    };
  }, []);

  const sendMessage = useCallback(
    async (text: string) => {
      const trimmed = text.trim();
      if (!trimmed || loadingRef.current) {
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
      };

      setMessages((prev) => [...prev, userMessage, assistantMessage]);
      setLoading(true);

      const controller = new AbortController();
      abortRef.current = controller;

      const finishAssistant = (fallbackText?: string) => {
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === assistantId
              ? {
                  ...msg,
                  // 出错时：如果已经流出了部分内容，保留它并在末尾附一行错误提示，
                  // 而不是把已显示的回答整段替换成错误信息。
                  content:
                    typeof fallbackText === 'string'
                      ? msg.content
                        ? `${msg.content}\n\n_⚠️ ${fallbackText}_`
                        : fallbackText
                      : msg.content,
                  isStreaming: false,
                }
              : msg,
          ),
        );
        if (abortRef.current === controller) {
          abortRef.current = null;
        }
        setLoading(false);
      };

      try {
        await sendMessageStream(
          trimmed,
          conversationId,
          language,
          settings,
          (chunk) => {
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
          (courses) => {
            // 后端在回答结束时给出本轮引用的课程代码，挂到助手消息上用于展示。
            if (courses && courses.length) {
              setMessages((prev) =>
                prev.map((msg) =>
                  msg.id === assistantId ? {...msg, sources: courses} : msg,
                ),
              );
            }
          },
          () => {
            finishAssistant();
          },
          (errorMsg) => {
            finishAssistant(errorMsg || 'Request failed.');
          },
          controller.signal,
        );
      } catch (err) {
        if (controller.signal.aborted) {
          finishAssistant();
          return;
        }
        const msg = err instanceof Error ? err.message : 'Request failed.';
        finishAssistant(msg);
      }
    },
    [conversationId, language, settings],
  );

  /** 停止生成：保留已经流出的内容。 */
  const stopGeneration = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
  }, []);

  const newChat = useCallback(() => {
    // 切换会话前先中止在途请求，否则旧的流会继续写入并让状态错乱。
    abortRef.current?.abort();
    abortRef.current = null;
    setMessages([]);
    setConversationId(safeUUID());
    setLoading(false);
  }, []);

  return {messages, isLoading, sendMessage, stopGeneration, newChat};
}
