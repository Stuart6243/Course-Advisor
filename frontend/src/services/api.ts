/**
 * 后端 API 调用封装。
 * 所有 HTTP 请求通过 Vite proxy 转发到后端 localhost:8000。
 */

import {ChatSettings, HealthStatus, ImportResult, ManualCourseData} from '../types';

const API_BASE = '/api';

/** 单次请求的最长等待时间。本地 Ollama 冷启动加载模型可能要 30s+。 */
const REQUEST_TIMEOUT_MS = 90_000;

/**
 * crypto.randomUUID() 只在 secure context（https 或 localhost）可用。
 * dev server 监听 0.0.0.0，从局域网 IP 访问时它是 undefined，
 * 旧代码会直接抛 "crypto.randomUUID is not a function" 导致整个 App 白屏。
 */
export function safeUUID(): string {
  const c = globalThis.crypto;
  if (c && typeof c.randomUUID === 'function') {
    return c.randomUUID();
  }
  if (c && typeof c.getRandomValues === 'function') {
    const bytes = c.getRandomValues(new Uint8Array(16));
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    const hex = Array.from(bytes, (b) => b.toString(16).padStart(2, '0')).join('');
    return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
  }
  return `id-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
}

export async function checkHealth(): Promise<HealthStatus> {
  try {
    const res = await fetch(`${API_BASE}/health`);
    if (!res.ok) {
      throw new Error(`HTTP ${res.status}`);
    }
    const data = await res.json();
    const groqAvailable = Boolean(data.groq_available);
    const ollamaConnected = Boolean(data.ollama_connected ?? data.ollama_available);
    return {
      status: data.status ?? 'error',
      // 兼容旧后端：没返回 usable 时退回按两个后端是否可用推断
      usable: Boolean(data.usable ?? (groqAvailable || ollamaConnected)),
      reasons: Array.isArray(data.reasons) ? data.reasons.map(String) : [],
      inference_mode: data.inference_mode ?? 'unknown',
      groq_available: groqAvailable,
      ollama_connected: ollamaConnected,
      model: data.model ?? '',
      groq_model: data.groq_model ?? '',
      courses_count: Number(data.courses_count ?? 0),
    };
  } catch {
    return {
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
  }
}

export async function sendMessageStream(
  message: string,
  conversationId: string,
  language: string,
  settings: ChatSettings | undefined,
  onChunk: (text: string) => void,
  onSources: (courses: string[]) => void,
  onDone: () => void,
  onError: (msg: string) => void,
  signal?: AbortSignal,
  onMeta?: (meta: {historyTurns: number}) => void,
): Promise<void> {
  const payload: Record<string, unknown> = {
    message,
    conversation_id: conversationId,
    language,
  };
  if (settings) {
    payload.max_history_turns = settings.maxHistoryTurns;
    payload.max_results = settings.maxResults;
  }

  // 超时和用户主动中止合并成一个 signal：
  // 旧版没有任何超时/中止机制，后端卡住时输入框被永久 disable，只能刷新页面。
  const timeoutController = new AbortController();
  const timer = window.setTimeout(() => timeoutController.abort(), REQUEST_TIMEOUT_MS);
  const onExternalAbort = () => timeoutController.abort();
  signal?.addEventListener('abort', onExternalAbort);

  const cleanup = () => {
    window.clearTimeout(timer);
    signal?.removeEventListener('abort', onExternalAbort);
  };

  let res: Response;
  try {
    res = await fetch(`${API_BASE}/chat`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload),
      signal: timeoutController.signal,
    });
  } catch (err) {
    cleanup();
    if (signal?.aborted) {
      onDone();
      return;
    }
    if (err instanceof DOMException && err.name === 'AbortError') {
      onError('The request timed out. Please try again.');
      return;
    }
    throw err;
  }

  if (!res.ok || !res.body) {
    cleanup();
    onError(`Request failed: HTTP ${res.status}`);
    return;
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let finished = false;

  const processLine = (rawLine: string) => {
    const line = rawLine.trim();
    if (!line || !line.startsWith('data:')) {
      return;
    }

    const payload = line.slice(5).trim();
    if (!payload) {
      return;
    }

    try {
      const event = JSON.parse(payload);
      if (event.type === 'chunk' && typeof event.content === 'string') {
        onChunk(event.content);
        return;
      }
      if (event.type === 'meta') {
        onMeta?.({historyTurns: Number(event.history_turns ?? 0)});
        return;
      }
      if (event.type === 'sources' && Array.isArray(event.courses)) {
        onSources(event.courses);
        return;
      }
      if (event.type === 'done') {
        finished = true;
        onDone();
        return;
      }
      if (event.type === 'error') {
        finished = true;
        onError(String(event.message || 'Unknown error'));
      }
    } catch {
      // Ignore malformed SSE lines.
    }
  };

  try {
    while (true) {
      const {value, done} = await reader.read();
      if (done) {
        break;
      }

      buffer += decoder.decode(value, {stream: true});
      const lines = buffer.split('\n');
      buffer = lines.pop() ?? '';

      for (const line of lines) {
        processLine(line);
      }
    }

    if (buffer.trim()) {
      processLine(buffer);
    }
  } catch (err) {
    // 用户点了「停止生成」：保留已经流出的内容，正常收尾。
    if (signal?.aborted) {
      cleanup();
      onDone();
      return;
    }
    cleanup();
    if (err instanceof DOMException && err.name === 'AbortError') {
      onError('The request timed out. Please try again.');
      return;
    }
    throw err;
  }

  cleanup();

  if (!finished) {
    onDone();
  }
}

export async function importFile(file: File): Promise<ImportResult> {
  const formData = new FormData();
  formData.append('file', file);

  const res = await fetch(`${API_BASE}/import`, {
    method: 'POST',
    body: formData,
  });

  return await res.json();
}

export async function importManual(data: ManualCourseData): Promise<ImportResult> {
  const res = await fetch(`${API_BASE}/import/manual`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(data),
  });

  return await res.json();
}

export async function exportChat(
  messages: Array<{role: string; content: string}>,
  format: 'markdown' | 'json',
): Promise<void> {
  const res = await fetch(`${API_BASE}/export`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({messages, format}),
  });

  if (!res.ok) {
    throw new Error(`Export failed: HTTP ${res.status}`);
  }

  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const date = new Date();
  const y = String(date.getFullYear());
  const m = String(date.getMonth() + 1).padStart(2, '0');
  const d = String(date.getDate()).padStart(2, '0');
  const ext = format === 'markdown' ? 'md' : 'json';
  const filename = `chat_export_${y}${m}${d}.${ext}`;

  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}
