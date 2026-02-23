/**
 * 后端 API 调用封装。
 * 所有 HTTP 请求通过 Vite proxy 转发到后端 localhost:8000。
 */

import {HealthStatus, ImportResult, ManualCourseData} from '../types';

const API_BASE = '/api';

export async function checkHealth(): Promise<HealthStatus> {
  try {
    const res = await fetch(`${API_BASE}/health`);
    if (!res.ok) {
      throw new Error(`HTTP ${res.status}`);
    }
    const data = await res.json();
    return {
      status: data.status ?? 'error',
      inference_mode: data.inference_mode ?? 'unknown',
      groq_available: Boolean(data.groq_available),
      ollama_connected: Boolean(data.ollama_connected ?? data.ollama_available),
      model: data.model ?? '',
      groq_model: data.groq_model ?? '',
      courses_count: Number(data.courses_count ?? 0),
    };
  } catch {
    return {
      status: 'error',
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
  onChunk: (text: string) => void,
  onSources: (courses: string[]) => void,
  onDone: () => void,
  onError: (msg: string) => void,
): Promise<void> {
  const res = await fetch(`${API_BASE}/chat`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      message,
      conversation_id: conversationId,
      language,
    }),
  });

  if (!res.ok || !res.body) {
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
