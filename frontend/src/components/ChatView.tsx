import {useEffect, useRef, useState} from 'react';
import {ArrowDown, ArrowUp} from 'lucide-react';
import {useTranslation} from 'react-i18next';
import {Message} from '../types';
import AutoResizeTextarea from './AutoResizeTextarea';
import MessageBubble from './MessageBubble';

type Props = {
  messages: Message[];
  isLoading: boolean;
  onSend: (text: string) => Promise<void> | void;
};

export default function ChatView({messages, isLoading, onSend}: Props) {
  const {t} = useTranslation();
  const [inputText, setInputText] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({behavior: 'smooth'});
  }, [messages]);

  const handleSend = () => {
    if (!inputText.trim() || isLoading) {
      return;
    }
    const text = inputText;
    setInputText('');
    void onSend(text);
  };

  return (
    <>
      <main className="flex-1 overflow-y-auto relative scroll-smooth pb-32" id="chat-container">
        <div className="max-w-3xl mx-auto px-4 py-8 flex flex-col gap-6">
          {messages.map((msg, idx) => (
            <MessageBubble key={msg.id} message={msg} index={idx} />
          ))}
          <div ref={messagesEndRef} className="h-12" />
        </div>
      </main>

      <div className="fixed bottom-28 right-8 z-30">
        <button
          onClick={() => messagesEndRef.current?.scrollIntoView({behavior: 'smooth'})}
          className="w-10 h-10 rounded-full bg-border-dark border border-slate-700 text-slate-300 shadow-lg hover:bg-slate-700 hover:text-white flex items-center justify-center transition-all group"
        >
          <ArrowDown className="w-5 h-5 group-hover:translate-y-0.5 transition-transform" />
        </button>
      </div>

      <div className="fixed bottom-0 left-0 w-full bg-gradient-to-t from-background-dark via-background-dark/95 to-transparent pt-10 pb-6 px-4 z-40 pointer-events-none">
        <div className="max-w-3xl mx-auto relative pointer-events-auto">
          <div className="relative bg-surface-dark border border-border-dark rounded-xl shadow-2xl focus-within:ring-2 focus-within:ring-primary/50 focus-within:border-primary transition-all duration-200">
            <AutoResizeTextarea
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleSend();
                }
              }}
              className="w-full bg-transparent text-slate-200 placeholder-slate-500 text-[15px] p-4 pr-12 rounded-xl focus:ring-0 focus:outline-none resize-none overflow-hidden max-h-48"
              placeholder={t('chat.placeholder')}
              minHeight="56px"
              disabled={isLoading}
            />
            <button
              onClick={handleSend}
              disabled={!inputText.trim() || isLoading}
              className="absolute bottom-2 right-2 p-1.5 rounded-lg bg-primary text-white hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              <ArrowUp className="w-5 h-5" />
            </button>
          </div>
          <div className="text-center mt-2">
            <p className="text-[11px] text-slate-500">{t('chat.disclaimer')}</p>
          </div>
        </div>
      </div>
    </>
  );
}
