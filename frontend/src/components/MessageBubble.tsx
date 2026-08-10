import {useState} from 'react';
import {Bot, Check, Copy, User} from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import {useTranslation} from 'react-i18next';
import {Message} from '../types';

type Props = {
  message: Message;
  index: number;
};

export default function MessageBubble({message, index}: Props) {
  const [copied, setCopied] = useState(false);
  const {t} = useTranslation();
  const sources = message.sources ?? [];

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(message.content);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1200);
    } catch {
      setCopied(false);
    }
  };

  return (
    <div
      className={`flex gap-4 p-2 animate-fade-in ${message.role === 'user' ? 'flex-row-reverse' : ''}`}
      style={{animationDelay: `${Math.min(index * 0.1, 0.5)}s`}}
    >
      <div className="flex-none">
        {message.role === 'assistant' ? (
          <div className="w-8 h-8 rounded-full bg-primary/20 flex items-center justify-center text-primary ring-1 ring-primary/30">
            <Bot className="w-5 h-5" />
          </div>
        ) : (
          <div className="w-8 h-8 rounded-full bg-surface-dark flex items-center justify-center text-slate-300 ring-1 ring-border-dark">
            <User className="w-5 h-5" />
          </div>
        )}
      </div>
      <div
        className={`flex-1 space-y-2 flex flex-col ${message.role === 'user' ? 'items-end' : 'items-start'}`}
      >
        <div
          className={`flex items-baseline gap-2 ${message.role === 'user' ? 'flex-row-reverse' : ''}`}
        >
          <span className="font-bold text-sm">
            {message.role === 'assistant' ? t('chat.assistantName') : t('chat.userName')}
          </span>
          <span className="text-xs text-slate-500">{message.time}</span>
        </div>
        <div
          className={`${message.role === 'user' ? 'bg-surface-dark text-slate-100 rounded-2xl rounded-tr-sm px-5 py-3 max-w-[85%]' : 'text-slate-300 w-full'} text-[15px] leading-[1.55]`}
        >
          {message.role === 'user' ? (
            message.content
          ) : (
            <div className="space-y-2">
              <div className="leading-[1.65] [&_p]:my-2 [&_ul]:list-disc [&_ol]:list-decimal [&_ul]:pl-5 [&_ol]:pl-5 [&_code]:rounded [&_code]:bg-[#161b22] [&_code]:px-1.5 [&_code]:py-0.5 [&_pre]:overflow-x-auto [&_pre]:rounded-xl [&_pre]:border [&_pre]:border-border-dark [&_pre]:bg-[#0d1117] [&_pre]:p-3 [&_table]:w-full [&_table]:border-collapse [&_th]:border [&_th]:border-border-dark [&_th]:bg-[#161b22] [&_th]:px-3 [&_th]:py-2 [&_td]:border [&_td]:border-border-dark [&_td]:px-3 [&_td]:py-2">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.content}</ReactMarkdown>
                {message.isStreaming ? (
                  <span className="inline-block ml-1 animate-pulse text-primary">▍</span>
                ) : null}
              </div>
              {!message.isStreaming && sources.length > 0 ? (
                <div className="flex flex-wrap items-center gap-1.5 pt-1">
                  <span className="text-xs text-slate-500 mr-0.5">
                    {t('chat.sources')}
                  </span>
                  {sources.map((code) => (
                    <span
                      key={code}
                      className="text-xs font-medium text-primary bg-primary/10 ring-1 ring-primary/20 rounded-md px-2 py-0.5"
                    >
                      {code}
                    </span>
                  ))}
                </div>
              ) : null}
              <div className="flex justify-end">
                <button
                  onClick={handleCopy}
                  className="flex items-center gap-1 text-xs text-slate-400 hover:text-white transition-colors"
                  aria-label={t('chat.copy')}
                >
                  {copied ? <Check className="w-3.5 h-3.5" /> : <Copy className="w-3.5 h-3.5" />}
                  {copied ? t('chat.copied') : t('chat.copy')}
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
