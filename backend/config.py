"""
项目配置常量。
所有路径、模型名称、超时设置集中管理。
"""

from __future__ import annotations

import os
from pathlib import Path

# ============================================================
# Ollama 配置
# ============================================================
OLLAMA_BASE_URL = "http://localhost:11434"
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
IMPORT_MIN_QUALITY_SCORE = 40

# ============================================================
# LLM 输出控制
# ============================================================
INTENT_MAX_TOKENS = 256
INTENT_TIMEOUT = 30
RESPONSE_MAX_TOKENS = 512

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
