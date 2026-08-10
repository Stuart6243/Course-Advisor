"""
FastAPI 主入口。
定义 API 路由并管理应用生命周期。
"""

from __future__ import annotations

import asyncio
import copy
import io
import json
import re
from collections import OrderedDict
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Optional

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator

import config
from conversation_scope import (
    Attribute as ScopeAttribute,
    ConversationScope,
    Operation as ScopeOperation,
    Scope,
    parse_conversation_scope,
)
from course_codes import extract_course_codes
from course_index import (
    build_enriched_index_from_dir,
    save_enriched_index,
)
from course_retriever import retrieve_courses
from export_handler import export_as_json, export_as_markdown
from file_importer import (
    import_file,
    import_manual_syllabus,
)
from groq_client import GroqClient
from ollama_client import OllamaClient
from prerequisites import compare_course_prerequisites
from provider_errors import classify_provider_failure
from query_parser import IntentExtractionResult, extract_query_intent_result
from response_generator import (
    format_prerequisite_answer,
    generate_response_stream,
    is_conversation_recall_query,
    select_courses_for_context,
)
from syllabus_store import SyllabusStore, apply_published_overlays


class ExportRequest(BaseModel):
    # 限制条数，避免一次导出请求占用大量内存（旧版 4MB 载荷可直接通过）
    messages: list[dict[str, Any]] = Field(max_length=2000)
    format: Literal["markdown", "json"]


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    # conversation_id 是内存字典的 key，限长避免被塞入超大字符串
    conversation_id: str = Field(min_length=1, max_length=128)
    language: Literal["en", "zh", "es", "fr"] = "en"
    max_history_turns: Optional[int] = None
    max_results: Optional[int] = None

    @field_validator("message")
    @classmethod
    def validate_message(cls, value: str) -> str:
        text = (value or "").strip()
        if not text:
            raise ValueError("message must not be empty")
        return text

    @field_validator("max_history_turns")
    @classmethod
    def validate_max_history_turns(cls, value: Optional[int]) -> Optional[int]:
        if value is not None and not (1 <= value <= config.CONVERSATION_MAX_TURNS):
            raise ValueError(
                f"max_history_turns must be between 1 and {config.CONVERSATION_MAX_TURNS}"
            )
        return value

    @field_validator("max_results")
    @classmethod
    def validate_max_results(cls, value: Optional[int]) -> Optional[int]:
        if value is not None and not (1 <= value <= 20):
            raise ValueError("max_results must be between 1 and 20")
        return value


class ManualImportRequest(BaseModel):
    course_code: str
    title: str
    points_raw: Optional[str] = ""
    points_min: Optional[float] = None
    points_max: Optional[float] = None
    term: Optional[str] = ""
    section_id: Optional[str] = ""
    times: Optional[str] = ""
    location: Optional[str] = ""
    instructor: Optional[str] = ""
    enrollment_raw: Optional[str] = ""
    enrollment_current: Optional[int] = None
    enrollment_capacity: Optional[int] = None
    description: Optional[str] = ""
    prerequisites_text: Optional[str] = ""
    notes_text: Optional[str] = ""
    department_or_group: Optional[str] = ""
    sections: list[dict[str, Any]] = Field(default_factory=list)


ERROR_MESSAGES: dict[str, dict[str, str]] = {
    "en": {
        "missing_key": "The cloud model isn't configured. Set GROQ_API_KEY in the terminal that runs the backend, or switch INFERENCE_MODE to \"local\" to use Ollama.",
        "rate_limited": "The cloud model is rate-limited right now. Please wait a moment and try again.",
        "timeout": "The model took too long to respond. Please try again, or ask a more specific question.",
        "truncated": "The model reached its response limit before finishing. Please narrow the question and try again.",
        "unreachable": "Couldn't reach the model service. Check that Ollama is running (or that you're online for Groq).",
        "generic": "Something went wrong while generating the answer. Please try again.",
    },
    "zh": {
        "missing_key": "云端模型未配置。请在运行后端的终端里设置 GROQ_API_KEY，或把 INFERENCE_MODE 改成 \"local\" 以使用本地 Ollama。",
        "rate_limited": "云端模型当前触发了限流，请稍等片刻再试。",
        "timeout": "模型响应超时。请重试，或把问题问得更具体一些。",
        "truncated": "模型在完整回答前达到了输出上限。请缩小问题范围后重试。",
        "unreachable": "无法连接模型服务。请确认 Ollama 已启动（或网络可访问 Groq）。",
        "generic": "生成回答时出错了，请重试。",
    },
    "es": {
        "missing_key": "El modelo en la nube no está configurado. Define GROQ_API_KEY en la terminal del backend, o cambia INFERENCE_MODE a \"local\" para usar Ollama.",
        "rate_limited": "El modelo en la nube está limitado por ahora. Espera un momento e inténtalo de nuevo.",
        "timeout": "El modelo tardó demasiado en responder. Inténtalo de nuevo o haz una pregunta más específica.",
        "truncated": "El modelo alcanzó el límite de respuesta antes de terminar. Reduce el alcance de la pregunta e inténtalo de nuevo.",
        "unreachable": "No se pudo conectar con el servicio del modelo. Verifica que Ollama esté en ejecución.",
        "generic": "Ocurrió un error al generar la respuesta. Inténtalo de nuevo.",
    },
    "fr": {
        "missing_key": "Le modèle cloud n'est pas configuré. Définissez GROQ_API_KEY dans le terminal du backend, ou passez INFERENCE_MODE à \"local\" pour utiliser Ollama.",
        "rate_limited": "Le modèle cloud est actuellement limité. Patientez un instant puis réessayez.",
        "timeout": "Le modèle a mis trop de temps à répondre. Réessayez ou posez une question plus précise.",
        "truncated": "Le modèle a atteint sa limite de réponse avant de terminer. Réduisez la portée de la question et réessayez.",
        "unreachable": "Impossible de joindre le service du modèle. Vérifiez qu'Ollama est démarré.",
        "generic": "Une erreur s'est produite lors de la génération de la réponse. Veuillez réessayer.",
    },
}

GENERIC_RECOMMEND_KEYWORDS = {
    "other",
    "another",
    "more",
    "additional",
    "else",
}
# 指代词/回指表达：出现这些词，且当前问题没有给出新的系别/课程代码/教授，
# 说明用户是在追问“上一轮那些课”，此时应复用上一轮的课程，而不是重新全库检索。
#
# 注意 its/their/which one 等：旧版只匹配 \bit\b，
# "what are its prerequisites?" 因为 its≠it 没被识别成回指，
# 于是拿 ['prerequisites'] 去全库检索，把 AERO 的对话拽到一堆 ELEN 课上。
_REFERENCE_WORD_RE = re.compile(
    r"\b("
    r"those|them|they|these|their|theirs|its|his|her|hers|"
    r"it|that one|this one|the same|"
    r"which (?:one|ones|of)|any of|one of|each of|both|the rest|the other|"
    r"the (?:first|second|third|fourth|fifth|last|former|latter|ones?)|"
    r"above|aforementioned|previous(?:ly)?|earlier|so far|"
    r"(?:recommend|suggest|pick|choose|show)\s+(?:me\s+)?(?:one|another|any|some)"
    r")\b",
    re.I,
)
_REFERENCE_CJK = (
    "它", "他们", "它们", "那些", "这些", "那几", "这几", "上面", "前面",
    "刚才", "第一", "第二", "第三", "之前", "那个", "这个", "其中",
    "哪个", "哪一个", "哪门", "这门", "那门", "再推荐", "还有别的", "刚说",
)

# “属性型”追问词：用户在问已讨论课程的某个属性，而不是发起新检索。
# 这类问题往往一个指代词都没有（"what are the prerequisites?"），
# 但拿这些词去全库检索只会捞回一堆无关课程。
_ATTRIBUTE_WORDS = frozenset({
    "prerequisite", "prerequisites", "prereq", "prereqs",
    "credit", "credits", "point", "points", "unit", "units",
    "instructor", "instructors", "professor", "professors", "teacher",
    "teach", "teaches", "teaching", "taught",
    "time", "times", "schedule", "when", "where", "location", "room",
    "enrollment", "capacity", "full", "seats", "spots",
    "difficulty", "difficult", "harder", "easier", "workload",
    "syllabus", "description", "detail", "details", "info", "information",
    "offered", "semester", "term", "level", "meet", "meets",
    "fewer", "fewest", "least", "most", "better", "cheapest",
    "shortest", "longest", "earliest", "latest", "sophomore", "junior",
    "senior", "freshman", "beginner", "advanced",
    "morning", "afternoon", "evening", "monday", "tuesday",
    "wednesday", "thursday", "friday", "saturday", "sunday",
    "one", "ones", "another", "first", "second", "third", "fourth",
    "fifth", "former", "latter",
})

_FOLLOWUP_NON_TOPIC_WORDS = _ATTRIBUTE_WORDS | frozenset({
    "fewer", "fewest", "least", "more", "most", "lowest", "highest",
    "which", "those", "these", "them", "they", "their", "its", "among",
    "prerrequisito", "prerrequisitos", "requisito", "requisitos",
    "credito", "creditos", "horario", "cuando", "profesor", "profesores",
    "prerequis", "prealable", "prealables", "credits", "horaire", "quand",
    "professeur", "professeurs", "moins", "plus", "lequel", "laquelle",
    "primero", "segundo", "premier", "deuxieme", "cuantos", "combien",
    "vaut", "cuando", "reune", "quand", "lieu", "comparalo", "compare-le",
    "lista", "liste", "estos", "estas", "esos", "esas", "ces", "cinq",
    "five", "cinco", "menos", "parmi", "entre", "eux", "ellos",
})


def _is_reference_message(text: str) -> bool:
    if not text:
        return False
    if _REFERENCE_WORD_RE.search(text):
        return True
    return any(tok in text for tok in _REFERENCE_CJK)


def _is_attribute_question(intent: dict) -> bool:
    """关键词全是属性词 -> 在问上一轮课程的属性，而不是新检索。"""
    keywords = [str(k).strip().lower() for k in (intent.get("keywords") or []) if str(k).strip()]
    if not keywords:
        return True
    return all(k in _ATTRIBUTE_WORDS for k in keywords)


def _intent_has_new_search_anchor(intent: dict) -> bool:
    """Return whether intent contains a genuine new catalog-search anchor."""
    if intent.get("course_codes") or intent.get("department") or intent.get("instructor"):
        return True
    keywords = {
        str(keyword).strip().lower()
        for keyword in (intent.get("keywords") or [])
        if str(keyword).strip()
    }
    return any(keyword not in _FOLLOWUP_NON_TOPIC_WORDS for keyword in keywords)


def _is_referential_followup(intent: dict, message: str, is_followup: bool) -> bool:
    """是否为“指代上一轮课程”的追问。"""
    if not is_followup:
        return False
    parsed = parse_conversation_scope(
        message,
        previous_count=2,
        has_current_focus=True,
        new_search_anchor=_intent_has_new_search_anchor(intent),
    )
    return parsed.scope is not Scope.NEW_SEARCH


def _course_identity(course: dict) -> str:
    return str(course.get("course_uid") or course.get("course_code") or "").strip().upper()


def _append_unique_course(target: list[dict], course: dict | None) -> None:
    if not isinstance(course, dict):
        return
    identity = _course_identity(course)
    if identity and any(_course_identity(existing) == identity for existing in target):
        return
    target.append(course)


def _resolve_focus_course(
    raw_focus: object,
    previous_courses: list[dict],
) -> dict | None:
    if isinstance(raw_focus, dict):
        return raw_focus
    if isinstance(raw_focus, str):
        wanted = raw_focus.strip().upper()
        for course in previous_courses:
            if _course_identity(course) == wanted or str(
                course.get("course_code") or ""
            ).strip().upper() == wanted:
                return course
    return None


def _courses_for_conversation_scope(
    parsed_scope: ConversationScope,
    previous_courses: list[dict],
    current_course: dict | None,
) -> tuple[list[dict], dict | None, str | None]:
    """Resolve previous-result ordinals/focus without touching the catalog."""
    if parsed_scope.scope is Scope.NEW_SEARCH:
        return [], current_course, None

    if parsed_scope.scope is Scope.CURRENT_COURSE:
        selected: dict | None = None
        if parsed_scope.ordinal is not None:
            position = parsed_scope.ordinal - 1
            if 0 <= position < len(previous_courses):
                selected = previous_courses[position]
            else:
                return [], current_course, "ordinal_out_of_range"
        elif current_course is not None:
            selected = current_course
        elif len(previous_courses) == 1:
            selected = previous_courses[0]
        if selected is None:
            return [], current_course, "current_course_unavailable"
        return [selected], selected, None

    selected_courses: list[dict] = []
    if parsed_scope.operation is ScopeOperation.COMPARE and (
        parsed_scope.uses_focus or parsed_scope.ordinals
    ):
        if parsed_scope.uses_focus:
            _append_unique_course(selected_courses, current_course)
        for ordinal in parsed_scope.ordinals:
            position = ordinal - 1
            if 0 <= position < len(previous_courses):
                _append_unique_course(selected_courses, previous_courses[position])
            else:
                return [], current_course, "ordinal_out_of_range"
        if selected_courses:
            return selected_courses, current_course, None

    return list(previous_courses), current_course, None


def _clip_message_content(content: str, limit: int) -> str:
    if len(content) <= limit:
        return content
    if limit <= 1:
        return content[:limit]
    marker = "\n…\n"
    if limit <= len(marker) + 2:
        return content[:limit]
    usable = limit - len(marker)
    head = (usable * 2) // 3
    return content[:head] + marker + content[-(usable - head) :]


def _trim_conversation_history(
    history: list[dict[str, str]],
    *,
    max_turns: int,
    max_chars: int | None = None,
) -> list[dict[str, str]]:
    """Keep complete recent turns under both turn and character limits."""
    char_budget = (
        config.CONVERSATION_MAX_CHARS
        if max_chars is None
        else max(0, max_chars)
    )
    capped = list(history[-max_turns * 2 :])
    turns = [capped[index : index + 2] for index in range(0, len(capped), 2)]
    selected: list[list[dict[str, str]]] = []
    remaining = char_budget

    for turn in reversed(turns):
        turn_size = sum(len(str(message.get("content") or "")) for message in turn)
        if turn_size <= remaining:
            selected.append([dict(message) for message in turn])
            remaining -= turn_size
            continue
        if not selected and remaining > 0:
            per_message = max(1, remaining // max(1, len(turn)))
            clipped_turn: list[dict[str, str]] = []
            for message in turn:
                copied = dict(message)
                copied["content"] = _clip_message_content(
                    str(message.get("content") or ""), per_message
                )
                clipped_turn.append(copied)
            selected.append(clipped_turn)
        break

    return [message for turn in reversed(selected) for message in turn]


IMPORT_READ_CHUNK_BYTES = 1024 * 1024
IMPORT_PIPELINE_TIMEOUT_SECONDS = 120.0


async def _read_upload_limited(upload: UploadFile, limit_bytes: int) -> bytes:
    """Read at most limit+1 bytes, never materializing an oversized upload."""

    chunks: list[bytes] = []
    total = 0
    while True:
        remaining_probe = limit_bytes + 1 - total
        chunk = await upload.read(min(IMPORT_READ_CHUNK_BYTES, remaining_probe))
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)
        total += len(chunk)
        if total > limit_bytes:
            raise ValueError(f"File exceeds {config.MAX_IMPORT_SIZE_MB}MB limit.")


def _ensure_import_state(application: FastAPI) -> None:
    if not isinstance(
        getattr(application.state, "seed_enriched_index", None), list
    ):
        application.state.seed_enriched_index = copy.deepcopy(
            getattr(application.state, "enriched_index", []) or []
        )
    if not isinstance(
        getattr(application.state, "syllabus_store", None), SyllabusStore
    ):
        application.state.syllabus_store = SyllabusStore(
            Path(config.DATA_DIR) / "syllabus_store"
        )
    if getattr(application.state, "import_lock", None) is None:
        application.state.import_lock = asyncio.Lock()


def _refresh_runtime_overlays(application: FastAPI) -> None:
    """Rebuild from immutable seed + the complete published overlay snapshot."""

    _ensure_import_state(application)
    seed = application.state.seed_enriched_index
    overlays = application.state.syllabus_store.effective_overlays()
    application.state.enriched_index = apply_published_overlays(seed, overlays)


def _manual_payload_data(payload: ManualImportRequest) -> dict[str, Any]:
    """Accept direct fields while retaining compatibility with sections[0]."""

    data = payload.model_dump()
    legacy = data.get("sections", [None])[0] if data.get("sections") else None
    if isinstance(legacy, dict):
        aliases = {
            "term": ("term",),
            "section_id": ("section_id", "section_call_number"),
            "points_raw": ("points_raw", "points"),
            "times": ("times",),
            "location": ("location",),
            "instructor": ("instructor",),
            "enrollment_raw": ("enrollment_raw",),
            "enrollment_current": ("enrollment_current",),
            "enrollment_capacity": ("enrollment_capacity",),
        }
        for target, sources in aliases.items():
            if data.get(target) not in (None, ""):
                continue
            for source in sources:
                if legacy.get(source) not in (None, ""):
                    data[target] = legacy[source]
                    break
    data.pop("sections", None)
    return data


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.ollama = OllamaClient(
        config.OLLAMA_BASE_URL,
        config.OLLAMA_MODEL,
        config.OLLAMA_TIMEOUT,
    )
    # Intent extraction uses the JSON-stable model and its shorter task timeout.
    app.state.ollama_intent = OllamaClient(
        config.OLLAMA_BASE_URL,
        config.OLLAMA_INTENT_MODEL,
        config.INTENT_TIMEOUT,
    )
    app.state.groq = GroqClient()
    app.state.groq_intent = GroqClient(timeout=config.INTENT_TIMEOUT)
    app.state.enriched_index = []
    app.state.seed_enriched_index = []
    app.state.syllabus_store = SyllabusStore(Path(config.DATA_DIR) / "syllabus_store")
    app.state.import_lock = asyncio.Lock()
    app.state.conversations: OrderedDict[str, list[dict[str, str]]] = OrderedDict()
    app.state.conversations_meta: OrderedDict[str, dict[str, Any]] = OrderedDict()

    groq_available = await app.state.groq.is_available()
    if groq_available:
        print("✅ Groq API connected")
    else:
        print("⚠️ Groq API unavailable, will use local Ollama")

    if config.ENRICHED_INDEX_PATH.exists():
        try:
            with config.ENRICHED_INDEX_PATH.open("r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, list):
                app.state.enriched_index = loaded
            print(f"✅ Loaded {len(app.state.enriched_index)} courses")
        except Exception:
            app.state.enriched_index = []

    # 索引缺失或为空时，直接从 courses_flat 目录自动构建一次，
    # 避免新机器/首次运行因为没有 enriched 索引而 0 门课、所有查询返回空。
    if not app.state.enriched_index and config.COURSES_DIR.exists():
        print("⚙️  Enriched index missing/empty — building from courses_flat ...")
        try:
            built = build_enriched_index_from_dir(str(config.COURSES_DIR))
            if built:
                app.state.enriched_index = built
                save_enriched_index(built, str(config.ENRICHED_INDEX_PATH))
                print(f"✅ Built and saved {len(built)} courses")
        except Exception as exc:
            print(f"⚠️ Failed to auto-build index: {exc}")

    # Keep the loaded catalog immutable in memory.  Runtime search is always a
    # fresh seed copy plus the complete published-only overlay snapshot.
    app.state.seed_enriched_index = copy.deepcopy(app.state.enriched_index)
    try:
        _refresh_runtime_overlays(app)
    except Exception as exc:
        app.state.enriched_index = copy.deepcopy(app.state.seed_enriched_index)
        print(f"⚠️ Failed to load syllabus overlays: {exc}")

    if config.WARMUP_ON_STARTUP and config.INFERENCE_MODE in ("local", "hybrid"):
        print("Warming up local LLM...")
        try:
            await app.state.ollama.chat(
                [{"role": "user", "content": "hi"}],
                system_prompt="Reply ok.",
                max_tokens=4,
            )
            print("  ✅ Local model warmed up")
        except Exception as exc:
            print(f"  ⚠️ Local warmup failed: {exc}")

    try:
        yield
    finally:
        app.state.ollama = None
        app.state.ollama_intent = None
        app.state.groq = None
        app.state.groq_intent = None
        app.state.enriched_index = []
        app.state.seed_enriched_index = []
        app.state.syllabus_store = None
        app.state.import_lock = None
        app.state.conversations = OrderedDict()
        app.state.conversations_meta = OrderedDict()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def get_llm_client(request: Request, task: str = "response"):
    """
    根据 INFERENCE_MODE 选择 LLM 客户端。
    返回 (client, provider)，provider 为 "groq" 或 "ollama"。
    """
    _ = task
    mode = config.INFERENCE_MODE
    groq = request.app.state.groq
    ollama = request.app.state.ollama

    if mode == "groq":
        return groq, "groq"
    if mode == "local":
        return ollama, "ollama"

    # Hybrid means Groq is the primary runtime.  Do not preflight either provider here:
    # the actual request is authoritative and failures are handled lazily by the stream
    # orchestrator.  In particular, a normal Groq request must never wait on Ollama's
    # potentially 60-second /api/tags probe.
    return groq, "groq"


async def get_fallback_client(request: Request, primary_provider: str):
    """
    hybrid 模式下的运行时兜底客户端。

    Groq 免费额度下限流是常态而非边缘情况，所以这里提供本地客户端供
    SSE orchestrator 在运行时执行一次 reset-and-replace fallback。这里只返回
    客户端引用；不得在健康的 Groq 请求前探测或加载 Ollama。
    """
    if config.INFERENCE_MODE != "hybrid" or primary_provider != "groq":
        return None
    ollama = request.app.state.ollama
    # Availability is intentionally not probed here.  Calling chat_stream only after the
    # primary fails is both faster on the healthy path and avoids a stale health result.
    return ollama


async def _extract_intent_for_request(
    payload: ChatRequest,
    request: Request,
) -> IntentExtractionResult:
    """Run rule -> configured primary -> lazy Ollama fallback -> minimal intent."""
    mode = config.INFERENCE_MODE
    groq = getattr(request.app.state, "groq_intent", None) or request.app.state.groq
    ollama_intent = getattr(request.app.state, "ollama_intent", None)
    if ollama_intent is None:
        ollama_intent = OllamaClient(
            config.OLLAMA_BASE_URL,
            config.OLLAMA_INTENT_MODEL,
            config.INTENT_TIMEOUT,
        )

    if mode == "local":
        return await extract_query_intent_result(
            payload.message,
            ollama_intent,
            model=config.OLLAMA_INTENT_MODEL,
            primary_source="ollama",
            timeout=config.INTENT_TIMEOUT,
        )

    fallback_client = ollama_intent if mode == "hybrid" else None
    return await extract_query_intent_result(
        payload.message,
        groq,
        model=config.GROQ_INTENT_MODEL,
        primary_source="groq",
        fallback_client=fallback_client,
        fallback_model=config.OLLAMA_INTENT_MODEL,
        fallback_source="ollama",
        timeout=config.INTENT_TIMEOUT,
    )


def _fallback_reason(exc: Exception) -> str:
    return classify_provider_failure(exc)


def _user_facing_error(exc: Exception, language: str) -> str:
    """把内部异常转成面向用户的提示，不直接暴露 str(exc)。"""
    text = str(exc).lower()
    if "groq_api_key" in text or "api key" in text:
        key = "missing_key"
    else:
        reason = classify_provider_failure(exc)
        actionable = {"rate_limited", "timeout", "truncated", "unreachable"}
        key = reason if reason in actionable else "generic"
    messages = ERROR_MESSAGES.get(language) or ERROR_MESSAGES["en"]
    return messages[key]


def _is_ambiguous_recommend_followup(intent: dict) -> bool:
    if (intent.get("query_type") or "").lower() != "recommend":
        return False
    if intent.get("department") or intent.get("course_codes"):
        return False
    keywords = [str(k).strip().lower() for k in (intent.get("keywords") or []) if k]
    if not keywords:
        return True
    return all(k in GENERIC_RECOMMEND_KEYWORDS for k in keywords)


def _extract_single_department_from_codes(course_codes: list[str]) -> str | None:
    prefixes: set[str] = set()
    for code in course_codes:
        text = str(code).strip().upper()
        if not text:
            continue
        prefix = text.split(" ", 1)[0]
        if prefix:
            prefixes.add(prefix)
    if len(prefixes) == 1:
        return next(iter(prefixes))
    return None


def _infer_department_from_context(
    history: list[dict[str, str]],
    conversation_meta: dict[str, Any] | None,
) -> str | None:
    if conversation_meta:
        last_intent = conversation_meta.get("last_intent") or {}
        if last_intent.get("department"):
            return str(last_intent["department"]).strip().upper()
        dept = _extract_single_department_from_codes(last_intent.get("course_codes") or [])
        if dept:
            return dept

    for msg in reversed(history):
        content = str(msg.get("content") or "").upper()
        codes = extract_course_codes(content)
        if not codes:
            continue
        prefixes = {code.split(" ", 1)[0] for code in codes}
        if len(prefixes) == 1:
            return next(iter(prefixes))
    return None


STATS_HEADERS = {
    "en": "We currently have **{total}** courses across **{depts}** departments:",
    "zh": "当前数据库共有 **{total}** 门课程，覆盖 **{depts}** 个系别：",
    "es": "Tenemos **{total}** cursos en total, distribuidos en **{depts}** departamentos:",
    "fr": "Nous avons **{total}** cours au total, répartis sur **{depts}** départements :",
}

# 列表项里的「门课 / courses / cursos / cours」也要跟着语言走，
# 旧版四种语言都硬编码成英文 "courses"。
STATS_UNIT = {"en": "courses", "zh": "门", "es": "cursos", "fr": "cours"}

STATS_MORE = {
    "en": "_...and {n} more departments._",
    "zh": "_……还有 {n} 个系别未列出。_",
    "es": "_...y {n} departamentos más._",
    "fr": "_...et {n} autres départements._",
}

# 一次列 56 个系太长，默认只列前 N 个。
STATS_MAX_DEPTS = 20


def _format_stats_message(index_data: list[dict], language: str) -> str:
    lang = language if language in STATS_HEADERS else "en"

    dept_counts: dict[str, int] = {}
    for entry in index_data:
        dept = (entry.get("department_prefix") or "UNKNOWN").strip().upper() or "UNKNOWN"
        dept_counts[dept] = dept_counts.get(dept, 0) + 1

    sorted_depts = sorted(dept_counts.items(), key=lambda item: (-item[1], item[0]))
    unit = STATS_UNIT[lang]

    lines = [
        STATS_HEADERS[lang].format(total=len(index_data), depts=len(sorted_depts)),
        "",
    ]
    lines.extend(
        f"- **{dept}**: {count} {unit}" for dept, count in sorted_depts[:STATS_MAX_DEPTS]
    )

    remaining = len(sorted_depts) - STATS_MAX_DEPTS
    if remaining > 0:
        lines.append("")
        lines.append(STATS_MORE[lang].format(n=remaining))

    return "\n".join(lines)


@app.post("/api/chat")
async def chat(payload: ChatRequest, request: Request):
    async def stream():
        response_provider = "deterministic"
        actual_provider = response_provider
        fallback_used = False
        fallback_reason: str | None = None
        full_response = ""
        deterministic_response: str | None = None

        async def client_disconnected() -> bool:
            try:
                return await request.is_disconnected()
            except Exception:
                return False

        try:
            index_data = request.app.state.enriched_index
            convos: OrderedDict[str, list[dict[str, str]]] = request.app.state.conversations
            convos_meta: OrderedDict[str, dict[str, Any]] = request.app.state.conversations_meta
            cid = (payload.conversation_id or "").strip() or "default"
            history = list(convos.get(cid, []))
            conversation_meta = convos_meta.get(cid, {})
            max_history_turns = (
                payload.max_history_turns
                if payload.max_history_turns is not None
                else config.CONVERSATION_MAX_TURNS
            )
            max_history_turns = min(max_history_turns, config.CONVERSATION_MAX_TURNS)
            max_results = (
                payload.max_results
                if payload.max_results is not None
                else config.MAX_RETRIEVAL_RESULTS
            )

            # The model budget includes the current request, not only cached
            # history.  ChatRequest caps one message at 4k, so the current user
            # text always fits and is preserved verbatim while older complete
            # turns are discarded/clipped first.
            history = _trim_conversation_history(
                history,
                max_turns=max_history_turns,
                max_chars=config.CONVERSATION_MAX_CHARS - len(payload.message),
            )

            intent_result = await _extract_intent_for_request(payload, request)
            intent = intent_result.intent
            is_stats_query = (intent.get("query_type") or "").lower() == "stats"
            messages_for_llm = history + [{"role": "user", "content": payload.message}]
            is_followup = len(history) >= 2
            recall_query = is_conversation_recall_query(intent, messages_for_llm)
            prev_courses = [
                course
                for course in (conversation_meta.get("last_courses") or [])
                if isinstance(course, dict)
            ]
            current_course = _resolve_focus_course(
                conversation_meta.get("current_course"), prev_courses
            )
            revision = int(conversation_meta.get("revision") or 0)
            # Capture explicit anchors before contextual department inheritance;
            # inherited state must not masquerade as a new user-provided topic.
            new_search_anchor = _intent_has_new_search_anchor(intent)
            ambiguous_recommend = _is_ambiguous_recommend_followup(intent)

            if ambiguous_recommend:
                inferred_dept = _infer_department_from_context(history, conversation_meta)
                if inferred_dept:
                    intent["department"] = inferred_dept
                    keywords = intent.get("keywords") or []
                    intent["keywords"] = [
                        kw for kw in keywords
                        if str(kw).strip().lower() not in GENERIC_RECOMMEND_KEYWORDS
                    ]
                    # "Any other courses?" is a deliberate new search within
                    # inherited context, rather than a detail about old results.
                    new_search_anchor = True

            parsed_scope = parse_conversation_scope(
                payload.message,
                previous_count=len(prev_courses),
                has_current_focus=current_course is not None,
                new_search_anchor=new_search_anchor,
                force_new_search=ambiguous_recommend,
            )
            intent["conversation_scope"] = parsed_scope.as_dict()
            scope_error: str | None = None
            next_current_course = current_course
            reused_previous_results = False

            if is_stats_query:
                courses: list[dict] = []
                prompt_basis: list[dict] = []
                response_client = None
                fallback_client = None
                response_provider = "deterministic"
                deterministic_response = _format_stats_message(
                    index_data, payload.language
                )
            else:
                if recall_query:
                    courses = []
                    reused_previous_results = True
                elif parsed_scope.scope is not Scope.NEW_SEARCH:
                    reused_previous_results = True
                    courses, next_current_course, scope_error = _courses_for_conversation_scope(
                        parsed_scope, prev_courses, current_course
                    )
                    if scope_error:
                        intent["scope_error"] = scope_error
                else:
                    courses = retrieve_courses(
                        index_data,
                        intent,
                        str(config.COURSES_DIR),
                        max_results=max_results,
                    )
                    next_current_course = courses[0] if len(courses) == 1 else None

                prompt_basis = select_courses_for_context(courses, max_results)
                if (
                    parsed_scope.attribute is ScopeAttribute.PREREQUISITES
                    and parsed_scope.operation
                    in (ScopeOperation.ARGMIN, ScopeOperation.ARGMAX)
                    and prompt_basis
                ):
                    comparison = compare_course_prerequisites(
                        prompt_basis, operation=parsed_scope.operation.value
                    )
                    comparison_facts = comparison.as_dict()
                    winner_ids = {
                        winner.strip().upper() for winner in comparison.winners
                    }
                    excluded_ids = {
                        identifier.strip().upper()
                        for identifier in comparison.excluded_unknown
                    }
                    comparison_facts["winner_course_codes"] = [
                        str(course.get("course_code") or "").strip()
                        for course in prompt_basis
                        if _course_identity(course) in winner_ids
                    ]
                    comparison_facts["excluded_unknown_course_codes"] = [
                        str(course.get("course_code") or "").strip()
                        for course in prompt_basis
                        if _course_identity(course) in excluded_ids
                    ]
                    intent["prerequisite_comparison"] = comparison_facts
                    if len(comparison.winners) == 1:
                        winner = comparison.winners[0].strip().upper()
                        next_current_course = next(
                            (
                                course
                                for course in prompt_basis
                                if _course_identity(course) == winner
                            ),
                            next_current_course,
                        )
                    elif comparison.tied:
                        next_current_course = None

                if parsed_scope.attribute is ScopeAttribute.PREREQUISITES:
                    response_client = None
                    fallback_client = None
                    response_provider = "deterministic"
                    deterministic_response = format_prerequisite_answer(
                        prompt_basis,
                        payload.language,
                        operation=parsed_scope.operation.value,
                    )
                else:
                    response_client, response_provider = await get_llm_client(
                        request, "response"
                    )
                    fallback_client = await get_fallback_client(
                        request, response_provider
                    )

            # A genuine new catalog search with zero matches must not expose
            # old course turns to the response model.  Passing no history makes
            # generate_response_stream return the localized EMPTY_RESULT text
            # directly (without invoking chat_stream).  Referential/recall
            # empty-basis turns retain their separate history semantics.
            generation_history = messages_for_llm
            if (
                parsed_scope.scope is Scope.NEW_SEARCH
                and not prompt_basis
                and not recall_query
                and response_provider != "deterministic"
            ):
                generation_history = None

            actual_provider = response_provider
            fallback_available = bool(
                response_provider == "groq" and fallback_client is not None
            )
            meta_event = {
                "type": "meta",
                "provider": response_provider,
                "fallback_available": fallback_available,
                "history_turns": len(history) // 2,
                "revision": revision,
                "intent_provider": intent_result.source,
                "intent_fallback_used": intent_result.fallback_used,
                "intent_fallback_reason": intent_result.fallback_reason,
            }
            yield f"data: {json.dumps(meta_event, ensure_ascii=False)}\n\n"

            if response_provider == "deterministic":
                full_response = deterministic_response or ""
                yield f"data: {json.dumps({'type': 'chunk', 'content': full_response}, ensure_ascii=False)}\n\n"
            else:
                try:
                    async for chunk in generate_response_stream(
                        intent=intent,
                        courses=prompt_basis,
                        ollama=response_client,
                        language=payload.language,
                        conversation_history=generation_history,
                        max_results=max_results,
                    ):
                        if await client_disconnected():
                            return
                        full_response += chunk
                        event = {"type": "chunk", "content": chunk}
                        yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                except asyncio.CancelledError:
                    # Browser stop/disconnect cancels the ASGI task.  It is not a provider
                    # failure and must never start an Ollama generation.
                    raise
                except Exception as primary_exc:
                    if await client_disconnected():
                        return

                    primary_partial = full_response
                    primary_reason = _fallback_reason(primary_exc)
                    if fallback_client is None:
                        error_event: dict[str, Any] = {
                            "type": "error",
                            "message": _user_facing_error(primary_exc, payload.language),
                            "provider": response_provider,
                            "fallback_used": False,
                            "fallback_reason": primary_reason,
                            "interrupted": bool(primary_partial),
                        }
                        if primary_partial:
                            error_event["partial_content"] = primary_partial
                            error_event["partial_provider"] = response_provider
                        yield f"data: {json.dumps(error_event, ensure_ascii=False)}\n\n"
                        return

                    fallback_used = True
                    fallback_reason = primary_reason
                    actual_provider = "ollama"
                    fallback_event = {
                        "type": "fallback",
                        "action": "reset",
                        "from": response_provider,
                        "to": "ollama",
                        "reason": fallback_reason,
                    }
                    yield f"data: {json.dumps(fallback_event, ensure_ascii=False)}\n\n"
                    if await client_disconnected():
                        return

                    # Reset the server accumulator at the same boundary as the frontend.
                    # The fallback receives the same prompt/history and generates from zero.
                    full_response = ""
                    try:
                        async for chunk in generate_response_stream(
                            intent=intent,
                            courses=prompt_basis,
                            ollama=fallback_client,
                            language=payload.language,
                            conversation_history=generation_history,
                            max_results=max_results,
                        ):
                            if await client_disconnected():
                                return
                            full_response += chunk
                            event = {"type": "chunk", "content": chunk}
                            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                    except asyncio.CancelledError:
                        raise
                    except Exception as fallback_exc:
                        if await client_disconnected():
                            return
                        error_event = {
                            "type": "error",
                            "message": _user_facing_error(fallback_exc, payload.language),
                            "provider": "ollama",
                            "fallback_used": True,
                            "fallback_reason": fallback_reason,
                            "interrupted": True,
                        }
                        # The UI may already have reset and displayed Ollama chunks.  Send
                        # the recoverable Groq partial so it can restore the last coherent
                        # answer instead of leaving an empty/half-local success state.
                        if primary_partial:
                            error_event["partial_content"] = primary_partial
                            error_event["partial_provider"] = response_provider
                        yield f"data: {json.dumps(error_event, ensure_ascii=False)}\n\n"
                        return

            if await client_disconnected():
                return

            # 回答完成后更新会话历史和 meta。
            # 注意：这里基于「写回时的最新历史」追加，而不是请求开始时读到的快照。
            # 否则同一 conversation_id 并发两条消息时，后写会覆盖先写（lost update）。
            latest = convos.get(cid, [])
            history = list(latest) if len(latest) >= len(history) else history
            history.append({"role": "user", "content": payload.message})
            history.append({"role": "assistant", "content": full_response})
            history = _trim_conversation_history(
                history, max_turns=max_history_turns
            )

            convos[cid] = history
            convos.move_to_end(cid)
            if is_stats_query or reused_previous_results:
                next_last_courses = prev_courses
            else:
                # A genuine zero-result new search clears stale results instead
                # of making the next pronoun refer to an older unrelated query.
                next_last_courses = prompt_basis
            focus_code = ""
            if isinstance(next_current_course, dict):
                focus_code = _course_identity(next_current_course)
            convos_meta[cid] = {
                "last_intent": dict(intent),
                "last_courses": next_last_courses,
                "current_course": focus_code or None,
                "revision": revision + 1,
            }
            convos_meta.move_to_end(cid)
            while len(convos) > config.CONVERSATION_MAX_SESSIONS:
                old_cid, _ = convos.popitem(last=False)
                convos_meta.pop(old_cid, None)

            source_codes: list[str] = []
            for course in prompt_basis:
                code = (course.get("course_code") or "").strip()
                if code:
                    source_codes.append(code)
            yield f"data: {json.dumps({'type': 'sources', 'courses': source_codes}, ensure_ascii=False)}\n\n"
            done_event = {
                "type": "done",
                "provider": actual_provider,
                "fallback_used": fallback_used,
                "fallback_reason": fallback_reason,
            }
            yield f"data: {json.dumps(done_event, ensure_ascii=False)}\n\n"
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if await client_disconnected():
                return
            print(f"[ERROR] /api/chat failed: {type(exc).__name__}: {exc}")
            friendly = _user_facing_error(exc, payload.language)
            error_event = {
                "type": "error",
                "message": friendly,
                "provider": actual_provider,
                "fallback_used": fallback_used,
                "fallback_reason": fallback_reason,
                "interrupted": bool(full_response),
            }
            if full_response:
                error_event["partial_content"] = full_response
                error_event["partial_provider"] = actual_provider
            yield f"data: {json.dumps(error_event, ensure_ascii=False)}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.post("/api/import")
async def import_course(request: Request, file: UploadFile = File(...)):
    """Attach a PDF/HTML syllabus without ever mutating catalog seed files."""

    limit_bytes = config.MAX_IMPORT_SIZE_MB * 1024 * 1024
    content_length = request.headers.get("content-length")
    if content_length and content_length.isdigit() and int(content_length) > (
        limit_bytes + IMPORT_READ_CHUNK_BYTES
    ):
        await file.close()
        return JSONResponse(
            status_code=413,
            content={
                "success": False,
                "status": "rejected",
                "message": f"File exceeds {config.MAX_IMPORT_SIZE_MB}MB limit.",
            },
        )

    try:
        try:
            file_bytes = await _read_upload_limited(file, limit_bytes)
        except ValueError as exc:
            return JSONResponse(
                status_code=413,
                content={"success": False, "status": "rejected", "message": str(exc)},
            )
        if await request.is_disconnected():
            raise asyncio.CancelledError

        _ensure_import_state(request.app)
        llm_client, _ = await get_llm_client(request, task="response")

        async def pre_commit_check() -> None:
            if await request.is_disconnected():
                raise asyncio.CancelledError

        async with request.app.state.import_lock:
            runtime_candidate: list[dict[str, Any]] | None = None

            def prepare_runtime_candidate(overlays: list[dict[str, Any]]) -> None:
                nonlocal runtime_candidate
                runtime_candidate = apply_published_overlays(
                    request.app.state.seed_enriched_index, overlays
                )

            try:
                result = await asyncio.wait_for(
                    import_file(
                        file_bytes=file_bytes,
                        filename=file.filename or "unknown",
                        llm_client=llm_client,
                        courses_dir=str(config.COURSES_DIR),
                        enriched_index=request.app.state.seed_enriched_index,
                        enriched_index_path=str(config.ENRICHED_INDEX_PATH),
                        syllabus_store=request.app.state.syllabus_store,
                        pre_commit_check=pre_commit_check,
                        before_store_commit=prepare_runtime_candidate,
                    ),
                    timeout=IMPORT_PIPELINE_TIMEOUT_SECONDS,
                )
            except TimeoutError:
                return JSONResponse(
                    status_code=504,
                    content={
                        "success": False,
                        "status": "rejected",
                        "message": "Syllabus import timed out; no runtime view was changed.",
                    },
                )
            if result.get("success"):
                if runtime_candidate is None:
                    return JSONResponse(
                        status_code=500,
                        content={
                            "success": False,
                            "status": "rejected",
                            "message": (
                                "Could not validate the syllabus search view; "
                                "no overlay was published."
                            ),
                        },
                    )
                request.app.state.enriched_index = runtime_candidate

        if result.get("success"):
            return result
        if result.get("error_code") in {
            "store_commit_failed",
            "import_pipeline_failed",
        }:
            return JSONResponse(status_code=500, content=result)
        status_code = 422 if result.get("status") == "rejected" else 400
        return JSONResponse(status_code=status_code, content=result)
    except asyncio.CancelledError:
        # Cancellation never refreshes the runtime view.  Store commits are
        # atomic, so a cancellation cannot expose a partial generation.
        raise
    finally:
        await file.close()


@app.post("/api/import/manual")
async def import_manual(payload: ManualImportRequest, request: Request):
    """Attach direct input to an existing versioned syllabus identity."""

    _ensure_import_state(request.app)
    data = _manual_payload_data(payload)
    async with request.app.state.import_lock:
        runtime_candidate: list[dict[str, Any]] | None = None

        def prepare_runtime_candidate(overlays: list[dict[str, Any]]) -> None:
            nonlocal runtime_candidate
            runtime_candidate = apply_published_overlays(
                request.app.state.seed_enriched_index, overlays
            )

        try:
            result = import_manual_syllabus(
                data=data,
                enriched_index=request.app.state.seed_enriched_index,
                syllabus_store=request.app.state.syllabus_store,
                before_store_commit=prepare_runtime_candidate,
            )
        except Exception:
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "status": "rejected",
                    "message": (
                        "Could not validate and commit the syllabus version; "
                        "no overlay was published."
                    ),
                },
            )
        if result.get("success"):
            if runtime_candidate is None:
                return JSONResponse(
                    status_code=500,
                    content={
                        "success": False,
                        "status": "rejected",
                        "message": (
                            "Could not validate the syllabus search view; "
                            "no overlay was published."
                        ),
                    },
                )
            request.app.state.enriched_index = runtime_candidate

    if result.get("success"):
        return result
    return JSONResponse(status_code=422, content=result)


@app.get("/api/health")
async def health(request: Request):
    ollama_ok = await request.app.state.ollama.is_available()
    groq_ok = await request.app.state.groq.is_available()
    index_data = request.app.state.enriched_index
    total = len(index_data) if isinstance(index_data, list) else 0

    # 如实反映「当前 INFERENCE_MODE 下到底能不能用」，
    # 旧版无论配置多离谱都返回 status: ok，导致忘记 export GROQ_API_KEY 时
    # 前端状态点和健康检查都看不出问题，直到发消息才报错。
    mode = config.INFERENCE_MODE
    if mode == "groq":
        model_ok = groq_ok
    elif mode == "local":
        model_ok = ollama_ok
    else:
        model_ok = groq_ok or ollama_ok

    reasons: list[str] = []
    if not model_ok:
        if mode in ("groq", "hybrid"):
            if not config.GROQ_API_KEY:
                reasons.append("GROQ_API_KEY is not set")
            else:
                reasons.append("Groq API unreachable or rejected the key")
        if mode in ("local", "hybrid"):
            reasons.append(
                f"Ollama not reachable at {config.OLLAMA_BASE_URL} "
                f"or model '{config.OLLAMA_MODEL}' not pulled"
            )

    # 课程库为空时同样是不可用状态：模型再正常，用户也只会一直看到
    # 「找不到匹配课程」，而完全意识不到是数据没加载。
    if total == 0:
        reasons.append(
            f"No courses loaded — check that {config.COURSES_DIR} contains course JSON files"
        )

    usable = model_ok and total > 0

    return {
        "status": "ok" if usable else "degraded",
        "usable": usable,
        "reasons": reasons,
        "inference_mode": mode,
        "groq_available": groq_ok,
        "ollama_available": ollama_ok,
        "ollama_connected": ollama_ok,
        "model": config.OLLAMA_MODEL,
        "groq_model": config.GROQ_RESPONSE_MODEL,
        "courses_count": total,
    }


@app.post("/api/export")
async def export_chat(payload: ExportRequest):
    if payload.format == "markdown":
        content = export_as_markdown(payload.messages)
        media_type = "text/markdown; charset=utf-8"
        extension = "md"
    else:
        content = export_as_json(payload.messages)
        media_type = "application/json; charset=utf-8"
        extension = "json"

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"course-advisor-chat-{timestamp}.{extension}"
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}

    return StreamingResponse(
        io.BytesIO(content.encode("utf-8")),
        media_type=media_type,
        headers=headers,
    )


@app.get("/api/courses/stats")
async def courses_stats(request: Request):
    index_data = request.app.state.enriched_index
    if not isinstance(index_data, list):
        return {"total": 0, "departments": [], "terms": []}

    departments = sorted(
        {
            (entry.get("department_prefix") or "").strip()
            for entry in index_data
            if (entry.get("department_prefix") or "").strip()
        }
    )
    terms = sorted(
        {
            term.strip()
            for entry in index_data
            for term in (entry.get("all_terms") or [])
            if isinstance(term, str) and term.strip()
        }
    )
    return {"total": len(index_data), "departments": departments, "terms": terms}
