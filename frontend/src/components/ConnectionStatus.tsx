import {useEffect, useState} from 'react';
import {useTranslation} from 'react-i18next';
import {checkHealth} from '../services/api';
import {HealthStatus} from '../types';

const defaultHealth: HealthStatus = {
  status: 'error',
  inference_mode: 'unknown',
  groq_available: false,
  ollama_connected: false,
  model: '',
  groq_model: '',
  courses_count: 0,
};

export default function ConnectionStatus() {
  const {t} = useTranslation();
  const [health, setHealth] = useState<HealthStatus>(defaultHealth);

  useEffect(() => {
    let mounted = true;

    const load = async () => {
      const status = await checkHealth();
      if (mounted) {
        setHealth(status);
      }
    };

    load();
    const timer = window.setInterval(load, 30000);

    return () => {
      mounted = false;
      window.clearInterval(timer);
    };
  }, []);

  let dotClass = 'bg-red-400 animate-pulse';
  let tooltip = t('connection.disconnected');

  if (health.groq_available) {
    dotClass = 'bg-emerald-400';
    tooltip = t('connection.groqConnected');
  } else if (health.ollama_connected) {
    dotClass = 'bg-amber-400';
    tooltip = t('connection.ollamaConnected');
  }

  return (
    <div className="group relative flex items-center">
      <span
        title={tooltip}
        className={`w-2.5 h-2.5 rounded-full ${dotClass}`}
        aria-label={tooltip}
      />
      <div className="pointer-events-none absolute left-4 top-1/2 hidden -translate-y-1/2 whitespace-nowrap rounded-md border border-border-dark bg-surface-dark px-2 py-1 text-xs text-slate-300 shadow-lg group-hover:block">
        {tooltip}
      </div>
    </div>
  );
}
