"""
FastAPI 主入口。
定义 API 路由并管理应用生命周期。
"""

from __future__ import annotations

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
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

import config
from course_index import (
    add_to_index,
    build_enriched_entry,
    build_enriched_index_from_dir,
    save_enriched_index,
)
from course_retriever import retrieve_courses
from export_handler import export_as_json, export_as_markdown
from file_importer import (
    _find_existing_by_code as find_existing_by_code,
    backfill_points,
    complete_course_json,
    generate_course_uid,
    import_file,
    normalize_course_code,
    validate_course_code,
)
from groq_client import GroqClient
from ollama_client import OllamaClient
from query_parser import extract_query_intent
from response_generator import generate_response_stream, is_conversation_recall_query


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
        if value is not None and not (1 <= value <= 50):
            raise ValueError("max_history_turns must be between 1 and 50")
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
    points_min: Optional[float] = 0.0
    points_max: Optional[float] = 0.0
    description: Optional[str] = ""
    prerequisites_text: Optional[str] = ""
    department_or_group: Optional[str] = ""
    sections: Optional[list] = []


ERROR_MESSAGES: dict[str, dict[str, str]] = {
    "en": {
        "missing_key": "The cloud model isn't configured. Set GROQ_API_KEY in the terminal that runs the backend, or switch INFERENCE_MODE to \"local\" to use Ollama.",
        "rate_limited": "The cloud model is rate-limited right now. Please wait a moment and try again.",
        "timeout": "The model took too long to respond. Please try again, or ask a more specific question.",
        "unreachable": "Couldn't reach the model service. Check that Ollama is running (or that you're online for Groq).",
        "generic": "Something went wrong while generating the answer. Please try again.",
    },
    "zh": {
        "missing_key": "云端模型未配置。请在运行后端的终端里设置 GROQ_API_KEY，或把 INFERENCE_MODE 改成 \"local\" 以使用本地 Ollama。",
        "rate_limited": "云端模型当前触发了限流，请稍等片刻再试。",
        "timeout": "模型响应超时。请重试，或把问题问得更具体一些。",
        "unreachable": "无法连接模型服务。请确认 Ollama 已启动（或网络可访问 Groq）。",
        "generic": "生成回答时出错了，请重试。",
    },
    "es": {
        "missing_key": "El modelo en la nube no está configurado. Define GROQ_API_KEY en la terminal del backend, o cambia INFERENCE_MODE a \"local\" para usar Ollama.",
        "rate_limited": "El modelo en la nube está limitado por ahora. Espera un momento e inténtalo de nuevo.",
        "timeout": "El modelo tardó demasiado en responder. Inténtalo de nuevo o haz una pregunta más específica.",
        "unreachable": "No se pudo conectar con el servicio del modelo. Verifica que Ollama esté en ejecución.",
        "generic": "Ocurrió un error al generar la respuesta. Inténtalo de nuevo.",
    },
    "fr": {
        "missing_key": "Le modèle cloud n'est pas configuré. Définissez GROQ_API_KEY dans le terminal du backend, ou passez INFERENCE_MODE à \"local\" pour utiliser Ollama.",
        "rate_limited": "Le modèle cloud est actuellement limité. Patientez un instant puis réessayez.",
        "timeout": "Le modèle a mis trop de temps à répondre. Réessayez ou posez une question plus précise.",
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
COURSE_CODE_RE = re.compile(r"\b([A-Z]{2,4})[\s\-_]*[A-Z]?\d{4}\b")

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


def _is_referential_followup(intent: dict, message: str, is_followup: bool) -> bool:
    """是否为“指代上一轮课程”的追问。"""
    if not is_followup:
        return False
    # 如果用户明确给了新的锚点（课程代码/系别/教授），说明是新查询，不算回指。
    if intent.get("course_codes") or intent.get("department") or intent.get("instructor"):
        return False
    # 显式指代词，或整句只在问属性（没有任何新主题词）
    return _is_reference_message(message) or _is_attribute_question(intent)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.ollama = OllamaClient(
        config.OLLAMA_BASE_URL,
        config.OLLAMA_MODEL,
        config.OLLAMA_TIMEOUT,
    )
    app.state.groq = GroqClient()
    app.state.enriched_index = []
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
        app.state.groq = None
        app.state.enriched_index = []
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

    if await groq.is_available():
        return groq, "groq"
    return ollama, "ollama"


async def get_fallback_client(request: Request, primary_provider: str):
    """
    hybrid 模式下的运行时兜底客户端。

    get_llm_client 只在请求开始前探测一次可用性，一旦选中 Groq，
    后续调用失败（429 限流 / 超时 / 网关抖动）就直接报错，不会切到本地。
    Groq 免费额度下限流是常态而非边缘情况，所以这里额外提供一个兜底客户端，
    供 generate_response_stream 在「首个 token 之前失败」时整轮重试。
    """
    if config.INFERENCE_MODE != "hybrid" or primary_provider != "groq":
        return None
    ollama = request.app.state.ollama
    if ollama is None:
        return None
    if not await ollama.is_available():
        return None
    return ollama


def _user_facing_error(exc: Exception, language: str) -> str:
    """把内部异常转成面向用户的提示，不直接暴露 str(exc)。"""
    text = str(exc).lower()
    if "groq_api_key" in text or "api key" in text:
        key = "missing_key"
    elif "429" in text or "rate limit" in text or "too many requests" in text:
        key = "rate_limited"
    elif "timeout" in text or "timed out" in text:
        key = "timeout"
    elif "connect" in text or "connection" in text:
        key = "unreachable"
    else:
        key = "generic"
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
        matches = COURSE_CODE_RE.findall(content)
        if not matches:
            continue
        prefixes = {dept.strip().upper() for dept in matches if dept}
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
            max_results = (
                payload.max_results
                if payload.max_results is not None
                else config.MAX_RETRIEVAL_RESULTS
            )

            history_limit = max_history_turns * 2
            if len(history) > history_limit:
                history = history[-history_limit:]

            # 告知前端后端侧还记得多少轮。对话历史存在内存里，
            # 进程一重启（开发时 --reload 每次存盘都会重启）就清空，
            # 而前端 UI 仍完整显示着之前的对话 —— 用户会看到助手突然"失忆"
            # 却不知道为什么。前端据此提示"上下文已重置"。
            yield (
                "data: "
                + json.dumps(
                    {"type": "meta", "history_turns": len(history) // 2},
                    ensure_ascii=False,
                )
                + "\n\n"
            )

            intent_client, intent_provider = await get_llm_client(request, "intent")
            # 意图提取只看当前问题，不使用历史。Groq 用轻量 8b 模型（省 70b 配额）。
            intent_model = (
                config.GROQ_INTENT_MODEL if intent_provider == "groq" else ""
            )
            intent = await extract_query_intent(
                payload.message, intent_client, model=intent_model
            )
            messages_for_llm = history + [{"role": "user", "content": payload.message}]
            is_followup = len(history) >= 2
            recall_query = is_conversation_recall_query(intent, messages_for_llm)
            prev_courses = conversation_meta.get("last_courses") or []

            if _is_ambiguous_recommend_followup(intent):
                inferred_dept = _infer_department_from_context(history, conversation_meta)
                if inferred_dept:
                    intent["department"] = inferred_dept
                    keywords = intent.get("keywords") or []
                    intent["keywords"] = [
                        kw for kw in keywords
                        if str(kw).strip().lower() not in GENERIC_RECOMMEND_KEYWORDS
                    ]

            if (intent.get("query_type") or "").lower() == "stats":
                full_response = _format_stats_message(index_data, payload.language)
                yield f"data: {json.dumps({'type': 'chunk', 'content': full_response}, ensure_ascii=False)}\n\n"
                courses: list[dict] = []
            else:
                if recall_query:
                    courses = []
                elif _is_referential_followup(intent, payload.message, is_followup) and prev_courses:
                    # 回指追问（“那些课里…”“tell me more about it”）：
                    # 复用上一轮实际展示过的课程，避免重新全库检索拉来无关课程。
                    courses = prev_courses
                else:
                    courses = retrieve_courses(
                        index_data,
                        intent,
                        str(config.COURSES_DIR),
                        max_results=max_results,
                    )

                response_client, response_provider = await get_llm_client(
                    request, "response"
                )
                fallback_client = await get_fallback_client(request, response_provider)
                full_response = ""
                async for chunk in generate_response_stream(
                    intent=intent,
                    courses=courses,
                    ollama=response_client,
                    language=payload.language,
                    conversation_history=messages_for_llm,
                    max_results=max_results,
                    fallback_client=fallback_client,
                ):
                    full_response += chunk
                    event = {"type": "chunk", "content": chunk}
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

            # 回答完成后更新会话历史和 meta。
            # 注意：这里基于「写回时的最新历史」追加，而不是请求开始时读到的快照。
            # 否则同一 conversation_id 并发两条消息时，后写会覆盖先写（lost update）。
            latest = convos.get(cid, [])
            history = list(latest) if len(latest) >= len(history) else history
            history.append({"role": "user", "content": payload.message})
            history.append({"role": "assistant", "content": full_response})

            max_msgs = max_history_turns * 2
            if len(history) > max_msgs:
                history = history[-max_msgs:]

            convos[cid] = history
            convos.move_to_end(cid)
            # 记住这一轮展示的课程，供下一轮回指追问复用。
            # 若本轮没有检索到课程（例如纯回忆型问题），保留上一轮的课程上下文。
            convos_meta[cid] = {
                "last_intent": dict(intent),
                "last_courses": courses if courses else prev_courses,
            }
            convos_meta.move_to_end(cid)
            while len(convos) > config.CONVERSATION_MAX_SESSIONS:
                old_cid, _ = convos.popitem(last=False)
                convos_meta.pop(old_cid, None)

            seen: set[str] = set()
            source_codes: list[str] = []
            for course in courses:
                code = (course.get("course_code") or "").strip()
                if code and code not in seen:
                    seen.add(code)
                    source_codes.append(code)
            yield f"data: {json.dumps({'type': 'sources', 'courses': source_codes}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"
        except Exception as exc:
            print(f"[ERROR] /api/chat failed: {type(exc).__name__}: {exc}")
            friendly = _user_facing_error(exc, payload.language)
            yield f"data: {json.dumps({'type': 'error', 'message': friendly}, ensure_ascii=False)}\n\n"
            # 始终补一个 done，前端无需依赖异常路径来收尾。
            yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.post("/api/import")
async def import_course(request: Request, file: UploadFile = File(...)):
    """接收上传文件并导入课程库。"""
    file_bytes = await file.read()
    if len(file_bytes) > config.MAX_IMPORT_SIZE_MB * 1024 * 1024:
        return {
            "success": False,
            "message": f"File exceeds {config.MAX_IMPORT_SIZE_MB}MB limit.",
        }

    llm_client, _ = await get_llm_client(request, task="response")
    result = await import_file(
        file_bytes=file_bytes,
        filename=file.filename or "unknown",
        llm_client=llm_client,
        courses_dir=str(config.COURSES_DIR),
        enriched_index=request.app.state.enriched_index,
        enriched_index_path=str(config.ENRICHED_INDEX_PATH),
    )
    return result


@app.post("/api/import/manual")
async def import_manual(payload: ManualImportRequest, request: Request):
    """接收用户手动填写的课程信息，直接入库（不需要 LLM）。"""
    data = payload.model_dump()
    code = normalize_course_code(data.get("course_code") or "")
    title = (data.get("title") or "").strip()

    if not code or not title:
        return {"success": False, "message": "course_code and title are required."}
    if len(title) < 3:
        return {
            "success": False,
            "message": f"Title too short: '{title}'. Minimum 3 characters.",
        }
    if not validate_course_code(code):
        return {
            "success": False,
            "message": (
                f"Invalid course_code format: '{code}'. Expected pattern: XXXX Y1234 "
                "(e.g., CIEN E3125, COMS W4111)."
            ),
        }

    data["course_code"] = code
    data["title"] = title
    # 表单只收 points_raw，这里回填 points_min/max，
    # 否则手动录入的课程永远无法被「3 学分的课」这类查询命中。
    data = backfill_points(data)

    uid = generate_course_uid(code, title)

    # 按 course_code 去重（旧版按 sha1(code|title)，换个标题就能重复入库）。
    existing = find_existing_by_code(request.app.state.enriched_index, code)
    if existing is not None:
        return {
            "success": False,
            "message": (
                f"Course {code} already exists in database "
                f"(\"{existing.get('title', '')}\")."
            ),
        }

    full_json = complete_course_json(data, uid)

    # 磁盘只读/权限不足/磁盘满时，旧版让 PermissionError 直接冒泡成 500，
    # 前端只会看到一个没有 JSON body 的错误。这里捕获并返回可读信息。
    try:
        courses_dir = Path(config.COURSES_DIR)
        courses_dir.mkdir(parents=True, exist_ok=True)
        save_path = courses_dir / f"{uid}.json"
        save_path.write_text(
            json.dumps(full_json, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        raw_entry = {
            "course_uid": uid,
            "course_code": code,
            "title": title,
            "file_name": f"{uid}.json",
            "path": f"courses_flat/{uid}.json",
        }
        enriched = build_enriched_entry(raw_entry, full_json)
        add_to_index(request.app.state.enriched_index, enriched)
        save_enriched_index(
            request.app.state.enriched_index, str(config.ENRICHED_INDEX_PATH)
        )
    except OSError as exc:
        print(f"[ERROR] /api/import/manual write failed: {exc}")
        return {
            "success": False,
            "message": (
                f"Could not write the course file: {exc.strerror or exc}. "
                f"Check that {config.COURSES_DIR} exists and is writable."
            ),
        }

    return {
        "success": True,
        "course": {
            "course_code": code,
            "title": title,
            "points": data.get("points_raw", ""),
        },
        "message": f"Successfully imported {code}: {title}",
    }


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
