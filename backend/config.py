"""
项目配置常量。
所有路径、模型名称、超时设置集中管理。
"""

import os
from pathlib import Path

# Ollama 配置
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen3-nothink:latest"  # 主模型：通用回答 + 意图提取
OLLAMA_FALLBACK_MODEL = "qwen2.5:7b"  # 备选模型：JSON 输出更稳定时用于意图提取
OLLAMA_TIMEOUT = 60  # 秒（M4 16GB 跑 8B 模型应足够）

# 数据路径（相对于 backend/ 目录）
DATA_DIR = Path(__file__).parent.parent / "data"
COURSES_DIR = DATA_DIR / "courses_flat"
RAW_INDEX_PATH = DATA_DIR / "courses_flat_index.json"
ENRICHED_INDEX_PATH = DATA_DIR / "courses_enriched_index.json"

# 文件导入
SUPPORTED_IMPORT_FORMATS = [".pdf", ".html", ".htm"]
MAX_IMPORT_SIZE_MB = 25


# 在现有内容后面添加这几行：

# LLM 输出控制
INTENT_MAX_TOKENS = 256       # 意图提取最大输出 token
RESPONSE_MAX_TOKENS = 512     # 回答生成最大输出 token

# 检索控制
MAX_RETRIEVAL_RESULTS = 5     # 从 10 减到 5
MAX_COURSE_CONTEXT_CHARS = 200  # 每门课喂给 LLM 的最大字符数








# ============================================================
# Groq 云 API 配置（免费计划）
# 注册: https://console.groq.com → 获取 API Key
# ============================================================
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_INTENT_MODEL = "llama-3.1-8b-instant"       # 意图提取 fallback（快+免费额度高）
GROQ_RESPONSE_MODEL = "llama-3.3-70b-versatile"   # 回答生成（质量最高）
GROQ_TIMEOUT = 15  # 秒

# ============================================================
# 意图提取配置
# ============================================================
INTENT_MAX_TOKENS = 256
INTENT_TIMEOUT = 30  # 本地 Ollama fallback 的超时

# ============================================================
# 回答生成配置
# ============================================================
RESPONSE_MAX_TOKENS = 512

# ============================================================
# 检索控制
# ============================================================
MAX_RETRIEVAL_RESULTS = 5

# ============================================================
# 模型预热（仅在使用本地 Ollama 时有效）
# ============================================================
WARMUP_ON_STARTUP = True  # 如果使用 Groq 可以设为 False 节省启动时间

# ============================================================
# 推理模式选择
# ============================================================
# "groq"   = 使用 Groq 云 API（推荐，快速 + 免费）
# "local"  = 使用本地 Ollama（离线可用，但慢）
# "hybrid" = 优先 Groq，失败时 fallback 到 Ollama
INFERENCE_MODE = "hybrid"
