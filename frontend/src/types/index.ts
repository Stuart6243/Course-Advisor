export type CourseSourceRole = 'answer_source' | 'prompt_basis';

export type CourseCitationStatus = 'verified' | 'deterministic' | 'candidate';

export type CourseSourceOffering = {
  term: string | null;
  section_id: string | null;
  meeting_time: string | null;
  location: string | null;
};

export type CourseSource = {
  uid: string;
  course_code: string;
  title: string;
  citation_label: string;
  source_label: string;
  role: CourseSourceRole;
  citation_status: CourseCitationStatus;
  offerings: CourseSourceOffering[];
};

export type LegacyCourseSourcesEvent = {
  type: 'sources';
  /** Normalized client-side marker; legacy servers omit this field on the wire. */
  schema_version: 1;
  courses: string[];
};

export type StructuredCourseSourcesEvent = {
  type: 'sources';
  schema_version: 2;
  /** Legacy mirror containing only ordered, actual answer-source course codes. */
  courses: string[];
  answer_sources: CourseSource[];
  prompt_basis: CourseSource[];
};

export type CourseSourcesEvent = LegacyCourseSourcesEvent | StructuredCourseSourcesEvent;

export type Message = {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  time: string;
  isStreaming?: boolean;
  sources?: CourseSourcesEvent;
  provider?: string;
  fallbackUsed?: boolean;
  fallbackFailed?: boolean;
  fallbackReason?: string;
  status?: 'streaming' | 'complete' | 'stopped' | 'interrupted' | 'error';
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
  term: string;
  section_id: string;
  points_raw?: string;
  points_min?: number;
  points_max?: number;
  description?: string;
  prerequisites_text?: string;
  department_or_group?: string;
  sections?: Array<{
    term?: string;
    section_id?: string;
    section_call_number?: string;
    times?: string;
    instructor?: string;
    location?: string;
  }>;
};

export type ImportResult = {
  success: boolean;
  status?: 'rejected' | 'review' | 'published';
  search_visible?: boolean;
  quality_score?: number;
  quality_issues?: string[];
  course?: {
    course_code: string;
    title: string;
    points?: string;
    term?: string;
    section_id?: string;
  };
  message: string;
  needs_manual_input?: boolean;
  partial_data?: Partial<ManualCourseData>;
  missing_fields?: string[];
  extracted_text_preview?: string;
};
