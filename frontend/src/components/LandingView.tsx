import {useRef, useState} from 'react';
import {ArrowUp, GraduationCap} from 'lucide-react';
import {useTranslation} from 'react-i18next';
import {MAX_MESSAGE_LENGTH} from '../constants';
import AutoResizeTextarea from './AutoResizeTextarea';

type Props = {
  onStart: (text: string) => Promise<void> | void;
};

export default function LandingView({onStart}: Props) {
  const {t} = useTranslation();
  const [inputText, setInputText] = useState('');
  const composingRef = useRef(false);
  const messageTooLong = inputText.length > MAX_MESSAGE_LENGTH;

  const handleSend = () => {
    if (!inputText.trim() || messageTooLong) {
      return;
    }
    const text = inputText;
    setInputText('');
    void onStart(text);
  };

  return (
    <main className="flex-1 flex flex-col items-center justify-center w-full max-w-5xl mx-auto px-4 sm:px-6 relative">
      <div className="mb-8 flex justify-center">
        <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-primary/20 to-transparent flex items-center justify-center ring-1 ring-white/5 shadow-[0_0_20px_-5px_rgba(25,127,230,0.3)] backdrop-blur-sm">
          <GraduationCap className="w-8 h-8 text-primary" />
        </div>
      </div>
      <h1 className="text-3xl md:text-4xl font-bold tracking-tight text-center mb-8 bg-gradient-to-b from-white to-slate-400 bg-clip-text text-transparent pb-1">
        {t('landing.title')}
      </h1>
      <div className="w-full max-w-[820px] relative">
        <div className="relative flex flex-col w-full bg-[#1e2024] border border-border-dark rounded-full focus-within:shadow-[0_0_20px_-5px_rgba(25,127,230,0.3)] focus-within:border-primary/50 transition-all duration-300 overflow-hidden">
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
            className="w-full bg-transparent text-base md:text-lg text-white placeholder-slate-500 px-6 py-4 pr-14 focus:outline-none resize-none max-h-[200px] overflow-y-auto leading-relaxed"
            placeholder={t('landing.placeholder')}
            ariaLabel={t('landing.placeholder')}
            ariaDescribedBy="landing-composer-help"
            minHeight="60px"
          />
          <div className="absolute bottom-3 right-3 flex items-center gap-2">
            <button
              type="button"
              onClick={handleSend}
              disabled={!inputText.trim() || messageTooLong}
              aria-label={t('chat.send')}
              className="flex items-center justify-center w-9 h-9 rounded-full bg-[#2a2d33] text-slate-500 disabled:opacity-50 disabled:cursor-not-allowed transition-all hover:bg-primary hover:text-white group/send"
            >
              <ArrowUp className="w-5 h-5 translate-y-0.5 group-hover/send:-translate-y-0.5 transition-transform" />
            </button>
          </div>
        </div>
        <div
          id="landing-composer-help"
          className="mt-2 min-h-5 px-4 text-xs"
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
      </div>
      <footer className="absolute bottom-4 py-4 px-6 text-center w-full">
        <p className="text-xs text-slate-600">{t('chat.disclaimer')}</p>
      </footer>
    </main>
  );
}
