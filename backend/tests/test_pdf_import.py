"""
真实 PDF 导入端到端验证脚本。

运行方式：
  cd backend
  python tests/test_pdf_import.py
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
import sys

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import config
from course_index import load_enriched_index, save_enriched_index, search_by_keywords
from file_importer import extract_text_from_pdf, generate_course_uid, import_file
from groq_client import GroqClient
from ollama_client import OllamaClient


async def _pick_llm_client():
    groq = GroqClient()
    if await groq.is_available():
        print("Using Groq client for PDF import test.")
        return groq

    ollama = OllamaClient(
        config.OLLAMA_BASE_URL,
        config.OLLAMA_MODEL,
        config.OLLAMA_TIMEOUT,
    )
    if await ollama.is_available():
        print(f"Using Ollama client for PDF import test ({config.OLLAMA_MODEL}).")
        return ollama

    raise RuntimeError("No available LLM client (Groq unavailable and Ollama model not ready).")


async def run_pdf_import_test() -> None:
    pdf_path = Path("tests/fixtures/test_real_course.pdf")
    assert pdf_path.exists(), f"请先将真实课程 PDF 放在 {pdf_path}"

    print("=========== Step 1: PDF text extraction ===========")
    pdf_bytes = pdf_path.read_bytes()
    text = extract_text_from_pdf(pdf_bytes)
    assert text.strip(), "PDF text extraction returned empty text."
    assert len(text) > 50, f"Extracted text too short: {len(text)} chars"
    print("✅ Step 1: PDF text extraction OK")
    print(f"   Extracted {len(text)} characters")
    print(f"   Preview: {text[:300]}")

    print("\n=========== Step 2: Import pipeline ===========")
    llm_client = await _pick_llm_client()
    base_index = load_enriched_index(str(config.ENRICHED_INDEX_PATH))
    before_count = len(base_index)

    with tempfile.TemporaryDirectory(prefix="pdf-import-test-") as tmpdir:
        tmp_root = Path(tmpdir)
        tmp_courses_dir = tmp_root / "courses_flat"
        tmp_courses_dir.mkdir(parents=True, exist_ok=True)
        tmp_index_path = tmp_root / "courses_enriched_index.json"
        working_index = list(base_index)
        save_enriched_index(working_index, str(tmp_index_path))

        result = await import_file(
            file_bytes=pdf_bytes,
            filename=pdf_path.name,
            llm_client=llm_client,
            courses_dir=str(tmp_courses_dir),
            enriched_index=working_index,
            enriched_index_path=str(tmp_index_path),
        )
        assert result.get("success") is True, f"Import failed: {result}"
        print("✅ Step 2: Import completed")
        print(f"   Course: {result['course']}")

        print("\n=========== Step 3: JSON validation ===========")
        course_code = result["course"]["course_code"]
        course_title = result["course"]["title"]
        uid = generate_course_uid(course_code, course_title)
        new_json_path = tmp_courses_dir / f"{uid}.json"
        assert new_json_path.exists(), f"JSON file not found: {new_json_path}"

        course_json = json.loads(new_json_path.read_text(encoding="utf-8"))
        assert (course_json.get("course_code") or "").strip()
        assert (course_json.get("title") or "").strip()
        assert (course_json.get("course_uid") or "").strip()
        assert isinstance(course_json.get("points_min"), (int, float))
        assert isinstance(course_json.get("sections"), list)
        print("✅ Step 3: JSON file validation OK")
        print(json.dumps(course_json, ensure_ascii=False, indent=2))

        print("\n=========== Step 4: Index update ===========")
        reloaded_index = load_enriched_index(str(tmp_index_path))
        assert len(reloaded_index) == before_count + 1
        matched = next((e for e in reloaded_index if e.get("course_uid") == uid), None)
        assert matched is not None, "Imported course not found in enriched index."
        assert (matched.get("department_prefix") or "").strip()
        assert (matched.get("searchable_text") or "").strip()
        assert matched.get("all_instructors") or matched.get("all_terms")
        print("✅ Step 4: Index update OK")

        print("\n=========== Step 5: Search verification ===========")
        title_keywords = [w.lower() for w in course_title.split() if len(w) >= 4][:2]
        if not title_keywords:
            title_keywords = [course_code.split()[0].lower()]
        results = search_by_keywords(reloaded_index, title_keywords)
        matched_codes = {r.get("course_code") for r in results}
        assert course_code in matched_codes, (
            f"Course '{course_code}' not found in search results for keywords {title_keywords}"
        )
        print("✅ Step 5: Search verification OK")
        print(f"   Course '{course_code}' is searchable!")

    print("\n============================")
    print("✅ ALL PDF IMPORT TESTS PASSED")
    print("============================")
    print("⚠️ 请人工确认：")
    print("  1. Step 1 打印的文本是否包含课程信息？")
    print("  2. Step 3 打印的 JSON 与 PDF 原文是否一致？")


if __name__ == "__main__":
    asyncio.run(run_pdf_import_test())
