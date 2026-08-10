import {useEffect, useState} from 'react';
import {useTranslation} from 'react-i18next';
import ChatView from './components/ChatView';
import Header from './components/Header';
import LandingView from './components/LandingView';
import SettingsDrawer from './components/SettingsDrawer';
import {useChat} from './hooks/useChat';
import {usePersistentState} from './hooks/usePersistentState';
import './i18n';
import {ChatSettings, Language} from './types';

const SUPPORTED_LANGUAGES: Language[] = ['en', 'zh', 'es', 'fr'];

const isLanguage = (value: unknown): value is Language =>
  typeof value === 'string' && (SUPPORTED_LANGUAGES as string[]).includes(value);

const isChatSettings = (value: unknown): value is ChatSettings => {
  if (typeof value !== 'object' || value === null) {
    return false;
  }
  const v = value as Partial<ChatSettings>;
  return (
    typeof v.maxHistoryTurns === 'number' &&
    typeof v.maxResults === 'number' &&
    v.maxHistoryTurns >= 1 &&
    v.maxHistoryTurns <= 50 &&
    v.maxResults >= 1 &&
    v.maxResults <= 20
  );
};

export default function App() {
  // 语言和查询设置持久化到 localStorage，刷新后不再重置。
  const [language, setLanguage] = usePersistentState<Language>(
    'course-advisor.language',
    'en',
    isLanguage,
  );
  const [chatSettings, setChatSettings] = usePersistentState<ChatSettings>(
    'course-advisor.settings',
    {maxHistoryTurns: 10, maxResults: 5},
    isChatSettings,
  );
  const {messages, isLoading, sendMessage, stopGeneration, newChat, contextLost} =
    useChat(language, chatSettings);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const {i18n} = useTranslation();

  const hasStarted = messages.length > 0;

  // 从 localStorage 恢复的语言要同步给 i18next，否则界面文案仍是英文。
  useEffect(() => {
    if (i18n.language !== language) {
      void i18n.changeLanguage(language);
    }
  }, [i18n, language]);

  const handleLanguageChange = (lang: Language) => {
    setLanguage(lang);
    void i18n.changeLanguage(lang);
  };

  const handleSettingsChange = (settings: ChatSettings) => {
    setChatSettings(settings);
  };

  return (
    <div className="min-h-screen flex flex-col overflow-hidden bg-background-dark text-slate-100 font-display selection:bg-primary/30 selection:text-white">
      <Header
        onNewChat={newChat}
        onOpenSettings={() => setIsSettingsOpen(true)}
      />

      {hasStarted ? (
        <ChatView
          messages={messages}
          isLoading={isLoading}
          onSend={sendMessage}
          onStop={stopGeneration}
          contextLost={contextLost}
        />
      ) : (
        <LandingView onStart={sendMessage} />
      )}

      <SettingsDrawer
        isOpen={isSettingsOpen}
        onClose={() => setIsSettingsOpen(false)}
        language={language}
        onLanguageChange={handleLanguageChange}
        messages={messages}
        maxHistoryTurns={chatSettings.maxHistoryTurns}
        maxResults={chatSettings.maxResults}
        onSettingsChange={handleSettingsChange}
      />
    </div>
  );
}
