"""
项目配置常量。
所有路径、模型名称、超时设置集中管理。
"""

from __future__ import annotations

import os
from pathlib import Path


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().casefold() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, *, minimum: int, maximum: int) -> int:
    raw = os.getenv(name, "").strip()
    try:
        value = int(raw) if raw else default
    except ValueError:
        return default
    return max(minimum, min(maximum, value))

# ============================================================
# Ollama 配置
# ============================================================
OLLAMA_BASE_URL = (
    os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").strip().rstrip("/")
    or "http://127.0.0.1:11434"
)
OLLAMA_MODEL = "qwen3-nothink:latest"  # 主模型：回答生成
OLLAMA_FALLBACK_MODEL = "qwen2.5:7b"  # 备用模型：JSON 输出稳定
OLLAMA_INTENT_MODEL = OLLAMA_FALLBACK_MODEL
OLLAMA_TIMEOUT = 60  # 秒

# ============================================================
# 数据路径（相对于 backend/ 目录）
# ============================================================
DATA_DIR = Path(__file__).parent.parent / "data"
COURSES_DIR = DATA_DIR / "courses_flat"
RAW_INDEX_PATH = DATA_DIR / "courses_flat_index.json"
ENRICHED_INDEX_PATH = DATA_DIR / "courses_enriched_index.json"

# ============================================================
# 文件导入
# ============================================================
SUPPORTED_IMPORT_FORMATS = [".pdf", ".html", ".htm"]
MAX_IMPORT_SIZE_MB = 25
AUTO_PUBLISH_QUALITY_SCORE = 80
IMPORT_MULTIPART_OVERHEAD_BYTES = 128 * 1024
MAX_PDF_PAGES = _env_int("MAX_PDF_PAGES", 100, minimum=1, maximum=500)
MAX_EXTRACTED_TEXT_CHARS = _env_int(
    "MAX_EXTRACTED_TEXT_CHARS", 100_000, minimum=4_000, maximum=1_000_000
)
PDF_PARSE_TIMEOUT_SECONDS = _env_int(
    "PDF_PARSE_TIMEOUT_SECONDS", 30, minimum=1, maximum=120
)
MAX_IMPORTED_SECTIONS = _env_int(
    "MAX_IMPORTED_SECTIONS", 50, minimum=1, maximum=200
)

# Persistent overlay capacity is a hard fail-safe. The store never starts a
# generation write after one of these limits has been reached.
SYLLABUS_STORE_MAX_INDEX_BYTES = _env_int(
    "SYLLABUS_STORE_MAX_INDEX_BYTES", 16 * 1024 * 1024, minimum=64 * 1024, maximum=128 * 1024 * 1024
)
SYLLABUS_STORE_MAX_VERSIONS = _env_int(
    "SYLLABUS_STORE_MAX_VERSIONS", 2_000, minimum=1, maximum=100_000
)
SYLLABUS_STORE_MAX_GENERATIONS = _env_int(
    "SYLLABUS_STORE_MAX_GENERATIONS", 500, minimum=1, maximum=10_000
)

# ============================================================
# HTTP security controls
# ============================================================
# The token is backend-only. Never expose it through a VITE_* variable.
API_AUTH_TOKEN = os.getenv("COURSE_ADVISOR_API_TOKEN", "").strip()
API_ALLOW_LOOPBACK_WITHOUT_AUTH = _env_bool(
    "COURSE_ADVISOR_ALLOW_LOOPBACK_WITHOUT_AUTH", True
)
CHAT_REQUEST_MAX_BYTES = 16 * 1024
MANUAL_IMPORT_REQUEST_MAX_BYTES = 128 * 1024
EXPORT_REQUEST_MAX_BYTES = 3 * 1024 * 1024
API_RATE_WINDOW_SECONDS = _env_int(
    "API_RATE_WINDOW_SECONDS", 60, minimum=1, maximum=3_600
)
API_CHAT_RATE_LIMIT = _env_int("API_CHAT_RATE_LIMIT", 30, minimum=1, maximum=10_000)
API_WRITE_RATE_LIMIT = _env_int("API_WRITE_RATE_LIMIT", 6, minimum=1, maximum=1_000)
API_EXPORT_RATE_LIMIT = _env_int("API_EXPORT_RATE_LIMIT", 20, minimum=1, maximum=1_000)
API_RATE_MAX_CLIENTS = _env_int("API_RATE_MAX_CLIENTS", 2_048, minimum=16, maximum=100_000)
API_CHAT_MAX_CONCURRENCY = _env_int(
    "API_CHAT_MAX_CONCURRENCY", 4, minimum=1, maximum=100
)
API_WRITE_MAX_CONCURRENCY = _env_int(
    "API_WRITE_MAX_CONCURRENCY", 1, minimum=1, maximum=20
)
API_EXPORT_MAX_CONCURRENCY = _env_int(
    "API_EXPORT_MAX_CONCURRENCY", 2, minimum=1, maximum=20
)

# ============================================================
# LLM 输出控制
# ============================================================
INTENT_MAX_TOKENS = 256
INTENT_TIMEOUT = 30

# ---------------- 回答生成 token 预算 ----------------
#
# 为什么必须设上限（而不是不传 max_tokens 让模型自己决定）：
#   - Groq 省略 max_tokens 时会放到模型上限（70b 为 32768）。一次跑飞
#     既拖慢响应，也会啃掉免费额度（70b 每天仅 1000 次请求）。
#   - Ollama 的 num_predict 默认 -1，会一直生成到填满上下文窗口。
#
# 为什么要按课程数动态算（而不是固定值）：
#   实测（1021 门课真实索引）：
#     max_results=5  → 课程上下文 545 tok，回答需求 ≈580 tok
#     max_results=20 → 课程上下文 1865 tok，回答需求 ≈2000 tok
#   固定 1536 在 5 门课时浪费 2.6 倍余量，在 20 门课时又不够、继续被截断。
#
# 本地 qwen3-nothink 的 Modelfile 写死 num_ctx 8192，是这里的硬约束：
#   最坏情况 prompt(1865) + 输出(2912) = 4777 < 8192，安全。
RESPONSE_BASE_TOKENS = 512        # 开场白 + 结尾建议的固定开销
RESPONSE_TOKENS_PER_COURSE = 120  # 每门课的叙述预算
RESPONSE_MIN_TOKENS = 768         # 无检索结果的 follow-up 也要够用
RESPONSE_MAX_TOKENS = 4096        # 绝对上限，防跑飞


def response_token_budget(course_count: int) -> int:
    """按本轮实际送进上下文的课程数计算输出预算。"""
    budget = RESPONSE_BASE_TOKENS + RESPONSE_TOKENS_PER_COURSE * max(0, course_count)
    return max(RESPONSE_MIN_TOKENS, min(RESPONSE_MAX_TOKENS, budget))


# ---------------- 文件导入 token 预算 ----------------
#
# 导入时模型要输出一整个课程 JSON（含完整 description 与 sections 数组），
# 复用回答的 token 上限会让 JSON 在中途断裂，
# 表现为 description 残缺或解析失败后误触发「需要手动录入」。
#
# 实测 400 门真实课程需要模型输出的 JSON 体积：
#   中位数 270 tok / p90 488 / p99 3421 / 最大 6120
# 3000 覆盖约 98%；再往上会挤爆本地 8192 的上下文窗口
#   （输入 12000 字符 ≈ 3000 tok + 系统提示 ≈ 400 + 输出 3000 = 6400）。
IMPORT_MAX_TOKENS = 3000

# 送进 LLM 的导入原文长度上限（字符）。太短会截掉 syllabus 后半段；
# 太长会和 IMPORT_MAX_TOKENS 一起超出本地模型 8192 的窗口。
IMPORT_INPUT_MAX_CHARS = 12000

# ============================================================
# 检索控制
# ============================================================
MAX_RETRIEVAL_RESULTS = 5
MAX_COURSE_CONTEXT_CHARS = 200

# ============================================================
# 对话历史缓存
# ============================================================
CONVERSATION_MAX_TURNS = 10
CONVERSATION_MAX_SESSIONS = 200
# Character budget for stored/prompted history after the 10-turn cap.  Roughly
# 3k tokens leaves room in the local 8k context for course evidence + output.
CONVERSATION_MAX_CHARS = 12000

# ============================================================
# Groq 云 API 配置
# ============================================================
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_INTENT_MODEL = "llama-3.1-8b-instant"
GROQ_RESPONSE_MODEL = "llama-3.3-70b-versatile"
GROQ_TIMEOUT = 15

# ============================================================
# 模型预热（仅在使用本地 Ollama 时有效）
# ============================================================
WARMUP_ON_STARTUP = True

# ============================================================
# 推理模式选择
# ============================================================
# "groq"   = 使用 Groq 云 API
# "local"  = 使用本地 Ollama
# "hybrid" = 优先 Groq，失败时 fallback 到 Ollama
INFERENCE_MODE = os.getenv("INFERENCE_MODE", "hybrid").strip().lower() or "hybrid"
