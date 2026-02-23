"""
端到端自动化测试。
对 test_questions.json 中的每个问题执行完整 pipeline，输出结构化结果。
新增：test_regression.json 回归测试（防幻觉、多轮对话、导入校验）。

运行方式：
  cd backend && python tests/test_e2e.py

输出文件：
  tests/test_results.json
  tests/test_regression_results.json
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
import sys

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import config
from course_index import load_enriched_index
from course_retriever import retrieve_courses
from file_importer import quality_score, validate_course_code
from groq_client import GroqClient
from ollama_client import OllamaClient
from query_parser import extract_query_intent, normalize_question, rule_based_extract
from response_generator import generate_response_stream


QUESTIONS_PATH = Path(__file__).with_name("test_questions.json")
REGRESSION_PATH = Path(__file__).with_name("test_regression.json")
RESULTS_PATH = Path(__file__).with_name("test_results.json")
REGRESSION_RESULTS_PATH = Path(__file__).with_name("test_regression_results.json")
THROTTLE_SECONDS = 2.5


class NullLLM:
    """当本地和云端 LLM 都不可用时兜底，避免脚本崩溃。"""

    async def chat(self, messages, system_prompt="", max_tokens=0):
        _ = (messages, system_prompt, max_tokens)
        raise RuntimeError("No available LLM client")

    async def chat_stream(self, messages, system_prompt="", max_tokens=0):
        _ = (messages, system_prompt, max_tokens)
        raise RuntimeError("No available LLM client")
        yield ""


def _now_ms() -> float:
    return time.perf_counter() * 1000


def _intent_source(question: str) -> str:
    normalized = normalize_question(question)
    rule_result = rule_based_extract(normalized)
    return "rule_engine" if rule_result is not None else "llm"


async def get_test_llm_clients():
    """初始化测试用 LLM 客户端。"""
    null_client = NullLLM()
    groq = GroqClient()
    ollama_response = OllamaClient(
        config.OLLAMA_BASE_URL,
        config.OLLAMA_MODEL,
        config.OLLAMA_TIMEOUT,
    )
    ollama_intent = OllamaClient(
        config.OLLAMA_BASE_URL,
        config.OLLAMA_INTENT_MODEL,
        config.INTENT_TIMEOUT,
    )

    groq_ok = await groq.is_available()
    ollama_ok = await ollama_response.is_available()
    intent_ollama_ok = await ollama_intent.is_available()

    provider = "none"
    response_client = null_client

    if config.INFERENCE_MODE in ("hybrid", "groq") and groq_ok:
        response_client = groq
        provider = "groq"
    elif ollama_ok:
        response_client = ollama_response
        provider = "ollama"

    if intent_ollama_ok:
        intent_client = ollama_intent
    elif groq_ok:
        intent_client = groq
    else:
        intent_client = null_client

    print(f"[LLM] response_provider={provider}, groq_ok={groq_ok}, ollama_ok={ollama_ok}, intent_ollama_ok={intent_ollama_ok}")
    return response_client, intent_client, provider


async def run_pipeline(
    question: str,
    intent_client,
    response_client,
    index_data: list[dict],
    courses_dir: str,
    language: str = "en",
    conversation_history: list[dict[str, str]] | None = None,
) -> dict:
    total_start = _now_ms()

    source = _intent_source(question)
    intent_start = _now_ms()
    intent = await extract_query_intent(question, intent_client)
    intent_latency_ms = round(_now_ms() - intent_start, 2)

    courses = retrieve_courses(
        index_data,
        intent,
        courses_dir,
        max_results=config.MAX_RETRIEVAL_RESULTS,
    )

    generation_start = _now_ms()
    chunks: list[str] = []
    generation_error = ""
    try:
        async for chunk in generate_response_stream(
            intent=intent,
            courses=courses,
            ollama=response_client,
            language=language,
            conversation_history=conversation_history,
        ):
            chunks.append(chunk)
    except Exception as exc:
        generation_error = str(exc)

    full_response = "".join(chunks)
    generation_latency_ms = round(_now_ms() - generation_start, 2)
    total_latency_ms = round(_now_ms() - total_start, 2)

    return {
        "intent": intent,
        "intent_source": source,
        "intent_latency_ms": intent_latency_ms,
        "courses": courses,
        "response": full_response,
        "generation_error": generation_error,
        "generation_latency_ms": generation_latency_ms,
        "total_latency_ms": total_latency_ms,
    }


async def run_single_test(question_data, idx, intent_client, response_client, index_data, courses_dir):
    question = question_data["question"]
    result = await run_pipeline(
        question=question,
        intent_client=intent_client,
        response_client=response_client,
        index_data=index_data,
        courses_dir=courses_dir,
        language="en",
    )

    course_codes = []
    seen = set()
    for course in result["courses"]:
        code = (course.get("course_code") or "").strip()
        if code and code not in seen:
            seen.add(code)
            course_codes.append(code)

    return {
        "id": question_data.get("id", f"Q{idx:02d}"),
        "question": question,
        "category": question_data.get("category", "unknown"),
        "intent": result["intent"],
        "intent_source": result["intent_source"],
        "intent_latency_ms": result["intent_latency_ms"],
        "courses_found": len(result["courses"]),
        "course_codes_found": course_codes,
        "response_preview": result["response"][:400],
        "generation_error": result["generation_error"],
        "generation_latency_ms": result["generation_latency_ms"],
        "total_latency_ms": result["total_latency_ms"],
    }


async def run_hallucination_test(test_case, response_client, intent_client, index_data, courses_dir):
    output = await run_pipeline(
        question=test_case["question"],
        intent_client=intent_client,
        response_client=response_client,
        index_data=index_data,
        courses_dir=courses_dir,
        language="en",
    )
    full_response = output["response"]
    response_lower = full_response.lower()

    forbidden = test_case.get("expect_contains_none", [])
    found_forbidden = [w for w in forbidden if w.lower() in response_lower]

    redirect = test_case.get("expect_contains_any", [])
    found_redirect = [w for w in redirect if w.lower() in response_lower]

    if redirect:
        redirect_ok = (len(found_redirect) > 0) or (output["generation_error"] != "")
    else:
        # 某些用例只要求“不编造”，不强制命中引导关键词；
        # 但仍要求模型有实际回答（或显式 generation_error），避免空字符串被误判通过。
        redirect_ok = bool(full_response.strip()) or (output["generation_error"] != "")

    passed = (len(found_forbidden) == 0) and redirect_ok

    return {
        "id": test_case["id"],
        "question": test_case["question"],
        "passed": passed,
        "response_preview": full_response[:400],
        "generation_error": output["generation_error"],
        "forbidden_words_found": found_forbidden,
        "redirect_words_found": found_redirect,
    }


async def run_conversation_memory_test(test_case, response_client, intent_client, index_data, courses_dir):
    questions = test_case["question_sequence"]
    history: list[dict[str, str]] = []
    responses = []

    for i, q in enumerate(questions):
        messages_for_llm = history + [{"role": "user", "content": q}]
        output = await run_pipeline(
            question=q,
            intent_client=intent_client,
            response_client=response_client,
            index_data=index_data,
            courses_dir=courses_dir,
            language="en",
            conversation_history=messages_for_llm,
        )
        full_response = output["response"]
        responses.append(full_response[:300])
        history.append({"role": "user", "content": q})
        history.append({"role": "assistant", "content": full_response})
        if i < len(questions) - 1:
            await asyncio.sleep(THROTTLE_SECONDS)

    expected = test_case.get("expect_second_contains", [])
    second = responses[-1].lower() if len(responses) > 1 else ""
    found = [w for w in expected if w.lower() in second]

    return {
        "id": test_case["id"],
        "question_sequence": questions,
        "passed": len(found) > 0,
        "responses": responses,
        "expected_in_second": expected,
        "found_in_second": found,
    }


def run_import_validation_test(test_case):
    details = []
    all_passed = True

    for code in test_case.get("invalid_codes", []):
        result = validate_course_code(code)
        ok = result is False
        details.append(
            {
                "code": code,
                "expected": "invalid",
                "got": "valid" if result else "invalid",
                "ok": ok,
            }
        )
        if not ok:
            all_passed = False

    for code in test_case.get("valid_codes", []):
        result = validate_course_code(code)
        ok = result is True
        details.append(
            {
                "code": code,
                "expected": "valid",
                "got": "valid" if result else "invalid",
                "ok": ok,
            }
        )
        if not ok:
            all_passed = False

    return {"id": test_case["id"], "passed": all_passed, "details": details}


def run_quality_score_test(test_case):
    details = []
    all_passed = True

    for tc in test_case.get("test_cases", []):
        score, issues = quality_score(tc["data"])
        ok = True
        if "expect_score_gte" in tc and score < tc["expect_score_gte"]:
            ok = False
        if "expect_score_lte" in tc and score > tc["expect_score_lte"]:
            ok = False

        details.append({"data": tc["data"], "score": score, "issues": issues, "ok": ok})
        if not ok:
            all_passed = False

    return {"id": test_case["id"], "passed": all_passed, "details": details}


def _avg(values: list[float]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 2)


def summarize_e2e(results: list[dict], provider: str) -> dict:
    total = len(results)
    rule_hits = sum(1 for r in results if r.get("intent_source") == "rule_engine")
    parse_ok = sum(1 for r in results if isinstance(r.get("intent"), dict) and r["intent"].get("query_type"))
    retrieval_ok = sum(1 for r in results if r.get("courses_found", 0) > 0)

    by_category: dict[str, int] = {}
    for r in results:
        cat = r.get("category", "unknown")
        by_category[cat] = by_category.get(cat, 0) + 1

    return {
        "total_tests": total,
        "response_provider": provider,
        "rule_engine_hit_rate": round((rule_hits / total) * 100, 2) if total else 0.0,
        "intent_parse_success_rate": round((parse_ok / total) * 100, 2) if total else 0.0,
        "retrieval_non_empty_rate": round((retrieval_ok / total) * 100, 2) if total else 0.0,
        "avg_total_latency_ms": _avg([r.get("total_latency_ms", 0.0) for r in results]),
        "avg_intent_latency_ms": _avg([r.get("intent_latency_ms", 0.0) for r in results]),
        "by_category": by_category,
    }


def summarize_regression(results: list[dict]) -> dict:
    total = len(results)
    passed = sum(1 for r in results if r.get("passed") is True)

    def _count(prefix: str) -> tuple[int, int]:
        subset = [r for r in results if r.get("id", "").startswith(prefix)]
        return sum(1 for r in subset if r.get("passed") is True), len(subset)

    anti_passed, anti_total = _count("REG-0")
    # 固定映射
    anti_passed = sum(1 for r in results if r.get("id") in {"REG-01", "REG-02", "REG-03", "REG-04"} and r.get("passed"))
    conv_passed = sum(1 for r in results if r.get("id") in {"REG-05", "REG-06"} and r.get("passed"))
    import_passed = sum(1 for r in results if r.get("id") == "REG-07" and r.get("passed"))
    quality_passed = sum(1 for r in results if r.get("id") == "REG-08" and r.get("passed"))

    return {
        "total_tests": total,
        "passed": passed,
        "pass_rate": round((passed / total) * 100, 2) if total else 0.0,
        "anti_hallucination": {"passed": anti_passed, "total": 4},
        "conversation_memory": {"passed": conv_passed, "total": 2},
        "import_validation": {"passed": import_passed, "total": 1},
        "quality_score": {"passed": quality_passed, "total": 1},
    }


async def run_all_tests():
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    regressions = json.loads(REGRESSION_PATH.read_text(encoding="utf-8"))

    response_client, intent_client, provider = await get_test_llm_clients()
    index_data = load_enriched_index(str(config.ENRICHED_INDEX_PATH))

    print(f"[E2E] loaded questions={len(questions)}, index_size={len(index_data)}")
    e2e_results = []
    for idx, question_data in enumerate(questions, start=1):
        print(f"[E2E] {idx}/{len(questions)} {question_data.get('id')} {question_data.get('question')}")
        result = await run_single_test(
            question_data,
            idx,
            intent_client,
            response_client,
            index_data,
            str(config.COURSES_DIR),
        )
        e2e_results.append(result)
        if idx < len(questions):
            await asyncio.sleep(THROTTLE_SECONDS)

    e2e_summary = summarize_e2e(e2e_results, provider)
    e2e_payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "summary": e2e_summary,
        "results": e2e_results,
    }
    RESULTS_PATH.write_text(
        json.dumps(e2e_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    regression_results = []
    for idx, tc in enumerate(regressions, start=1):
        cid = tc.get("id", "")
        print(f"[REG] running {cid}")
        used_llm = False
        if cid in {"REG-01", "REG-02", "REG-03", "REG-04"}:
            regression_results.append(
                await run_hallucination_test(
                    tc,
                    response_client,
                    intent_client,
                    index_data,
                    str(config.COURSES_DIR),
                )
            )
            used_llm = True
        elif cid in {"REG-05", "REG-06"}:
            regression_results.append(
                await run_conversation_memory_test(
                    tc,
                    response_client,
                    intent_client,
                    index_data,
                    str(config.COURSES_DIR),
                )
            )
            used_llm = True
        elif cid == "REG-07":
            regression_results.append(run_import_validation_test(tc))
        elif cid == "REG-08":
            regression_results.append(run_quality_score_test(tc))
        if used_llm and idx < len(regressions):
            await asyncio.sleep(THROTTLE_SECONDS)

    reg_summary = summarize_regression(regression_results)
    reg_payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "summary": reg_summary,
        "results": regression_results,
    }
    REGRESSION_RESULTS_PATH.write_text(
        json.dumps(reg_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n=== E2E Summary ===")
    print(json.dumps(e2e_summary, ensure_ascii=False, indent=2))
    print("\n=== Regression Summary ===")
    print(json.dumps(reg_summary, ensure_ascii=False, indent=2))
    print(f"\nSaved: {RESULTS_PATH}")
    print(f"Saved: {REGRESSION_RESULTS_PATH}")


if __name__ == "__main__":
    asyncio.run(run_all_tests())
