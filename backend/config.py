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

# 回答生成上限。512 太小：比较/推荐 5 门课时回答必然被腰斩。
# 1536 tokens ≈ 1000 英文单词 / 700 汉字，足够覆盖 5 门课的详细对比。
RESPONSE_MAX_TOKENS = 1536

# 文件导入转换上限。导入时模型要输出一整个课程 JSON（含完整 description
# 与 sections 数组），复用 RESPONSE_MAX_TOKENS 会导致 JSON 在中途断裂，
# 表现为 description 残缺或解析失败后误触发「需要手动录入」。
IMPORT_MAX_TOKENS = 3000

# 送进 LLM 的导入原文长度上限（字符）。太短会截掉 syllabus 后半段。
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
