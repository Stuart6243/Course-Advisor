import {useState} from 'react';
import {useTranslation} from 'react-i18next';
import ChatView from './components/ChatView';
import Header from './components/Header';
import LandingView from './components/LandingView';
import SettingsDrawer from './components/SettingsDrawer';
import {useChat} from './hooks/useChat';
import './i18n';
import {ChatSettings, Language} from './types';

export default function App() {
  const [language, setLanguage] = useState<Language>('en');
  const [chatSettings, setChatSettings] = useState<ChatSettings>({
    maxHistoryTurns: 10,
    maxResults: 5,
  });
  const {messages, isLoading, sendMessage, newChat} = useChat(language, chatSettings);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const {i18n} = useTranslation();

  const hasStarted = messages.length > 0;

  const handleLanguageChange = (lang: Language) => {
    setLanguage(lang);
    i18n.changeLanguage(lang);
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
