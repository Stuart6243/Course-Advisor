"""
FastAPI 主入口。
定义所有 API 路由，管理应用生命周期。
"""
# 在现有 import 区域添加：
from groq_client import GroqClient

from __future__ import annotations

import io
import json
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import config
from course_retriever import retrieve_courses
from export_handler import export_as_json, export_as_markdown
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. 本地 Ollama 客户端（离线 fallback）
    app.state.ollama = OllamaClient(
        config.OLLAMA_BASE_URL,
        config.OLLAMA_MODEL,
        config.OLLAMA_TIMEOUT,
    )

    # 2. Groq 云 API 客户端
    app.state.groq = GroqClient()
    groq_available = await app.state.groq.is_available()
    if groq_available:
        print("✅ Groq API connected")
    else:
        print("⚠️ Groq API unavailable, will use local Ollama")

    # 3. 加载 enriched index
    app.state.enriched_index = []
    if config.ENRICHED_INDEX_PATH.exists():
        try:
            with config.ENRICHED_INDEX_PATH.open("r", encoding="utf-8") as f:
                loaded = json.load(f)
                if isinstance(loaded, list):
                    app.state.enriched_index = loaded
            print(f"✅ Loaded {len(app.state.enriched_index)} courses")
        except Exception:
            app.state.enriched_index = []

    # 4. 预热本地模型（仅在 local/hybrid 模式且有需要时）
    if config.WARMUP_ON_STARTUP and config.INFERENCE_MODE in ("local", "hybrid"):
        print("Warming up local LLM...")
        try:
            await app.state.ollama.chat(
                [{"role": "user", "content": "hi"}],
                system_prompt="Reply ok.", max_tokens=4
            )
            print("  ✅ Local model warmed up")
        except Exception as e:
            print(f"  ⚠️ Local warmup failed: {e}")

    try:
        yield
    finally:
        app.state.ollama = None
        app.state.groq = None
        app.state.enriched_index = []


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
    根据 INFERENCE_MODE 和可用性选择 LLM 客户端。
    
    task: "intent" | "response"
    返回: (client, client_type) — client_type 是 "groq" 或 "ollama"
    """
    mode = config.INFERENCE_MODE
    groq = request.app.state.groq
    ollama = request.app.state.ollama

    if mode == "groq":
        return groq, "groq"
    elif mode == "local":
        return ollama, "ollama"
    else:  # hybrid
        if await groq.is_available():
            return groq, "groq"
        return ollama, "ollama"













@app.post("/api/chat")
async def chat(payload: ChatRequest, request: Request):
    async def stream():
        try:
            index_data = request.app.state.enriched_index

            # 1. 意图提取（规则引擎优先）
            llm_client, client_type = await get_llm_client(request, "intent")
            intent = await extract_query_intent(payload.message, llm_client)

            # 2. 检索
            courses = retrieve_courses(
                index_data, intent, str(config.COURSES_DIR),
                max_results=config.MAX_RETRIEVAL_RESULTS
            )

            # 3. 回答生成（选择最佳客户端）
            resp_client, resp_type = await get_llm_client(request, "response")
            
            # 如果用 Groq，指定模型
            if resp_type == "groq":
                # 临时 monkey-patch 让 generate_response_stream 用 Groq
                async for chunk in generate_response_stream(
                    intent=intent, courses=courses,
                    ollama=resp_client,  # GroqClient 接口兼容
                    language=payload.language,
                ):
                    event = {"type": "chunk", "content": chunk}
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            else:
                async for chunk in generate_response_stream(
                    intent=intent, courses=courses,
                    ollama=resp_client,
                    language=payload.language,
                ):
                    event = {"type": "chunk", "content": chunk}
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

            # 4. sources + done
            seen = set()
            source_codes = []
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





@app.get("/api/health")
async def health(request: Request):
    ollama_ok = await request.app.state.ollama.is_available()
    groq_ok = await request.app.state.groq.is_available()
    index_data = request.app.state.enriched_index
    
    return {
        "status": "ok",
        "inference_mode": config.INFERENCE_MODE,
        "groq_available": groq_ok,
        "ollama_available": ollama_ok,
        "model": config.OLLAMA_MODEL,
        "groq_model": config.GROQ_RESPONSE_MODEL,
        "courses_count": len(index_data) if isinstance(index_data, list) else 0,
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

    return StreamingResponse(io.BytesIO(content.encode("utf-8")), media_type=media_type, headers=headers)


@app.post("/api/chat")
async def chat(payload: ChatRequest, request: Request):
    async def stream():
        try:
            ollama = request.app.state.ollama
            index_data = request.app.state.enriched_index

            intent = await extract_query_intent(payload.message, ollama)
            courses = retrieve_courses(index_data, intent, str(config.COURSES_DIR))

            async for chunk in generate_response_stream(
                intent=intent,
                courses=courses,
                ollama=ollama,
                language=payload.language,
            ):
                event = {"type": "chunk", "content": chunk}
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

            seen: set[str] = set()
            source_codes: list[str] = []
            for course in courses:
                code = (course.get("course_code") or "").strip()
                if code and code not in seen:
                    seen.add(code)
                    source_codes.append(code)

            sources_event = {"type": "sources", "courses": source_codes}
            yield f"data: {json.dumps(sources_event, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"
        except Exception as exc:
            error_event = {"type": "error", "message": str(exc)}
            yield f"data: {json.dumps(error_event, ensure_ascii=False)}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.post("/api/import")
async def import_not_implemented():
    raise HTTPException(status_code=501, detail="Not implemented")


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
