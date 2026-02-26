import {useCallback, useRef, useState} from 'react';
import {ChatSettings, Language, Message} from '../types';
import {sendMessageStream} from '../services/api';

const nowTime = () =>
  new Date().toLocaleTimeString([], {hour: '2-digit', minute: '2-digit'});

export function useChat(language: Language, settings: ChatSettings) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [conversationId, setConversationId] = useState(() => crypto.randomUUID());
  const loadingRef = useRef(false);

  const setLoading = (next: boolean) => {
    loadingRef.current = next;
    setIsLoading(next);
  };

  const sendMessage = useCallback(
    async (text: string) => {
      const trimmed = text.trim();
      if (!trimmed || loadingRef.current) {
        return;
      }

      const userId = crypto.randomUUID();
      const assistantId = crypto.randomUUID();

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

      const finishAssistant = (fallbackText?: string) => {
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === assistantId
              ? {
                  ...msg,
                  content:
                    typeof fallbackText === 'string' ? fallbackText : msg.content,
                  isStreaming: false,
                }
              : msg,
          ),
        );
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
          () => {
            // Sources are already provided by backend; no UI rendering needed here.
          },
          () => {
            finishAssistant();
          },
          (errorMsg) => {
            finishAssistant(errorMsg || 'Request failed.');
          },
        );
      } catch (err) {
        const msg = err instanceof Error ? err.message : 'Request failed.';
        finishAssistant(msg);
      }
    },
    [conversationId, language, settings],
  );

  const newChat = useCallback(() => {
    setMessages([]);
    setConversationId(crypto.randomUUID());
    setLoading(false);
  }, []);

  return {messages, isLoading, sendMessage, newChat};
}
