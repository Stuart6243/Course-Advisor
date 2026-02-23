import {GraduationCap, Plus} from 'lucide-react';
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
          onClick={onNewChat}
          className="hidden md:flex min-w-[100px] cursor-pointer items-center justify-center overflow-hidden rounded-lg h-9 px-4 bg-primary text-white text-sm font-bold leading-normal tracking-[0.015em] hover:bg-blue-600 transition-colors"
        >
          <span className="truncate">{t('header.newChat')}</span>
          <Plus className="w-4 h-4 ml-2" />
        </button>
        <button
          onClick={onOpenSettings}
          className="w-9 h-9 rounded-full overflow-hidden ring-2 ring-border-dark hover:ring-primary transition-colors"
        >
          <img
            src="https://picsum.photos/seed/avatar/100/100"
            alt="User"
            className="w-full h-full object-cover"
            referrerPolicy="no-referrer"
          />
        </button>
      </div>
    </header>
  );
}
