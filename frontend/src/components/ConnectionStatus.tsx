import {useEffect, useState} from 'react';
import {useTranslation} from 'react-i18next';
import {checkHealth} from '../services/api';
import {HealthStatus} from '../types';

const defaultHealth: HealthStatus = {
  status: 'error',
  usable: false,
  reasons: [],
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

  const configuredProvider =
    health.inference_mode === 'local'
      ? health.ollama_connected
        ? 'ollama'
        : null
      : health.inference_mode === 'groq'
        ? health.groq_available
          ? 'groq'
          : null
        : health.groq_available
          ? 'groq'
          : health.ollama_connected
            ? 'ollama'
            : null;

  if (health.usable && configuredProvider === 'groq') {
    dotClass = 'bg-emerald-400';
    tooltip = t('connection.groqConnected');
  } else if (health.usable && configuredProvider === 'ollama') {
    dotClass = 'bg-amber-400';
    tooltip = t('connection.ollamaConnected');
  } else if (health.status === 'degraded') {
    // 后端起来了但当前 INFERENCE_MODE 下没有可用模型
    // （最常见：忘了 export GROQ_API_KEY）。把原因直接显示出来。
    dotClass = 'bg-red-400 animate-pulse';
    tooltip = health.reasons.length
      ? `${t('connection.degraded')}: ${health.reasons.join('; ')}`
      : t('connection.degraded');
  }

  return (
    <div className="group relative flex items-center">
      <span
        title={tooltip}
        className={`w-2.5 h-2.5 rounded-full ${dotClass}`}
        role="status"
        aria-live="polite"
        aria-label={tooltip}
      />
      <div className="pointer-events-none absolute left-4 top-1/2 hidden -translate-y-1/2 whitespace-nowrap rounded-md border border-border-dark bg-surface-dark px-2 py-1 text-xs text-slate-300 shadow-lg group-hover:block">
        {tooltip}
      </div>
    </div>
  );
}
