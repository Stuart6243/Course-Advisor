export type Message = {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  time: string;
  isStreaming?: boolean;
  sources?: string[];
};

export type Language = 'en' | 'zh' | 'es' | 'fr';

export type ChatSettings = {
  maxHistoryTurns: number;
  maxResults: number;
};

export type HealthStatus = {
  status: string;
  /** 当前 INFERENCE_MODE 下是否真的有可用模型 */
  usable: boolean;
  /** 不可用时的具体原因，用于 tooltip 提示 */
  reasons: string[];
  inference_mode: string;
  groq_available: boolean;
  ollama_connected: boolean;
  model: string;
  groq_model: string;
  courses_count: number;
};

export type ManualCourseData = {
  course_code: string;
  title: string;
  points_raw?: string;
  points_min?: number;
  points_max?: number;
  description?: string;
  prerequisites_text?: string;
  department_or_group?: string;
  sections?: Array<{
    term?: string;
    times?: string;
    instructor?: string;
    location?: string;
  }>;
};

export type ImportResult = {
  success: boolean;
  course?: {course_code: string; title: string; points?: string};
  message: string;
  needs_manual_input?: boolean;
  partial_data?: Partial<ManualCourseData>;
  missing_fields?: string[];
  extracted_text_preview?: string;
};
