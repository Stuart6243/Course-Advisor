"""
FastAPI 主入口。
定义 API 路由并管理应用生命周期。
"""

from __future__ import annotations

import io
import json
from collections import OrderedDict
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Optional

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import config
from course_index import add_to_index, build_enriched_entry, save_enriched_index
from course_retriever import retrieve_courses
from export_handler import export_as_json, export_as_markdown
from file_importer import (
    complete_course_json,
    generate_course_uid,
    import_file,
    normalize_course_code,
    validate_course_code,
)
from groq_client import GroqClient
from ollama_client import OllamaClient
from query_parser import extract_query_intent
from response_generator import generate_response_stream


class ExportRequest(BaseModel):
    messages: list[dict[str, Any]]
    format: Literal["markdown", "json"]


class ChatRequest(BaseModel):
    message: str
    conversation_id: str
    language: Literal["en", "zh", "es", "fr"] = "en"


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


@app.post("/api/chat")
async def chat(payload: ChatRequest, request: Request):
    async def stream():
        try:
            index_data = request.app.state.enriched_index
            convos: OrderedDict[str, list[dict[str, str]]] = request.app.state.conversations
            cid = (payload.conversation_id or "").strip() or "default"
            history = list(convos.get(cid, []))

            intent_client, _ = await get_llm_client(request, "intent")
            # 意图提取只看当前问题，不使用历史。
            intent = await extract_query_intent(payload.message, intent_client)

            courses = retrieve_courses(
                index_data,
                intent,
                str(config.COURSES_DIR),
                max_results=config.MAX_RETRIEVAL_RESULTS,
            )

            response_client, _ = await get_llm_client(request, "response")
            messages_for_llm = history + [{"role": "user", "content": payload.message}]

            full_response = ""
            async for chunk in generate_response_stream(
                intent=intent,
                courses=courses,
                ollama=response_client,
                language=payload.language,
                conversation_history=messages_for_llm,
            ):
                full_response += chunk
                event = {"type": "chunk", "content": chunk}
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

            # 回答完成后更新会话历史。
            history.append({"role": "user", "content": payload.message})
            history.append({"role": "assistant", "content": full_response})

            max_msgs = config.CONVERSATION_MAX_TURNS * 2
            if len(history) > max_msgs:
                history = history[-max_msgs:]

            convos[cid] = history
            convos.move_to_end(cid)
            while len(convos) > config.CONVERSATION_MAX_SESSIONS:
                convos.popitem(last=False)

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
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)}, ensure_ascii=False)}\n\n"

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

    uid = generate_course_uid(code, title)

    for entry in request.app.state.enriched_index:
        if entry.get("course_uid") == uid:
            return {
                "success": False,
                "message": f"Course {code} already exists in database.",
            }

    full_json = complete_course_json(data, uid)
    save_path = Path(config.COURSES_DIR) / f"{uid}.json"
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

    return {
        "status": "ok",
        "inference_mode": config.INFERENCE_MODE,
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
