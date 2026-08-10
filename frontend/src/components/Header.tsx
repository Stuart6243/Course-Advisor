import {GraduationCap, Plus, Settings} from 'lucide-react';
import {useTranslation} from 'react-i18next';
import ConnectionStatus from './ConnectionStatus';

type Props = {
  onNewChat: () => void;
  onOpenSettings: () => void;
};

export default function Header({onNewChat, onOpenSettings}: Props) {
  const {t} = useTranslation();

  return (
    <header className="flex-none flex items-center justify-between whitespace-nowrap border-b border-border-dark px-4 md:px-6 py-3 bg-background-dark z-20">
      <div className="flex items-center gap-3 text-white">
        <div className="w-8 h-8 flex items-center justify-center text-primary">
          <GraduationCap className="w-7 h-7" />
        </div>
        <h2 className="text-lg font-bold leading-tight tracking-[-0.015em]">
          {t('header.title')}
        </h2>
        <ConnectionStatus />
      </div>
      <div className="flex items-center gap-4">
        <button
          type="button"
          onClick={onNewChat}
          aria-label={t('header.newChat')}
          title={t('header.newChat')}
          className="md:hidden w-9 h-9 cursor-pointer items-center justify-center rounded-lg bg-primary text-white hover:bg-blue-600 transition-colors flex"
        >
          <Plus className="w-4 h-4" />
        </button>
        <button
          type="button"
          onClick={onNewChat}
          aria-label={t('header.newChat')}
          className="hidden md:flex min-w-[100px] cursor-pointer items-center justify-center overflow-hidden rounded-lg h-9 px-4 bg-primary text-white text-sm font-bold leading-normal tracking-[0.015em] hover:bg-blue-600 transition-colors"
        >
          <span className="truncate">{t('header.newChat')}</span>
          <Plus className="w-4 h-4 ml-2" />
        </button>
        <button
          type="button"
          onClick={onOpenSettings}
          aria-label={t('settings.title')}
          className="w-9 h-9 rounded-full flex items-center justify-center text-slate-300 ring-2 ring-border-dark hover:ring-primary hover:text-white transition-colors"
        >
          <Settings className="w-5 h-5" />
        </button>
      </div>
    </header>
  );
}
