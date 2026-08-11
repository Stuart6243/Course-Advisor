"""Deterministic PDF attach-only integration test (no network or formal writes)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sys

import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from course_index import build_enriched_entry
from course_retriever import retrieve_courses
from file_importer import extract_text_from_pdf, import_file
from syllabus_store import SyllabusStore, apply_published_overlays


FIXTURE = Path(__file__).parent / "fixtures" / "test_synthetic_course.pdf"
COURSE_UID = "b" * 40


class FakeLLM:
    def __init__(self, response: dict) -> None:
        self.response = json.dumps(response, ensure_ascii=False)
        self.calls: list[dict] = []

    async def chat(self, messages, system_prompt="", max_tokens=0) -> str:
        self.calls.append(
            {
                "messages": list(messages),
                "system_prompt": system_prompt,
                "max_tokens": max_tokens,
            }
        )
        return self.response


def _intent() -> dict:
    return {
        "query_type": "detail",
        "course_codes": ["COMS E9999"],
        "keywords": [],
        "department": None,
        "department_terms": [],
        "instructor": None,
        "time_preference": None,
        "day_preference": [],
        "points_range": [3.0, 3.0],
        "term": "Fall 2025",
        "comparison_targets": [],
        "original_question": "COMS E9999 in Fall 2025",
    }


def test_pdf_import_attaches_existing_seed_without_network_or_seed_mutation(
    tmp_path: Path,
) -> None:
    pytest.importorskip("pdfplumber")
    pdf_bytes = FIXTURE.read_bytes()
    actual_text = extract_text_from_pdf(pdf_bytes)
    assert "COMS E9999: EVIDENCE-GROUNDED SYSTEMS" in actual_text
    assert actual_text.index("Course Times") < actual_text.index("COURSE DESCRIPTION")

    verified_description = (
        "This synthetic course introduces evidence-grounded retrieval, "
        "deterministic fact rendering, source attribution, streaming interfaces, "
        "and resilient model fallback."
    )

    data_dir = tmp_path / "data"
    courses_dir = data_dir / "courses_flat"
    courses_dir.mkdir(parents=True)
    seed_detail = {
        "course_uid": COURSE_UID,
        "course_code": "COMS E9999",
        "title": "EVIDENCE-GROUNDED SYSTEMS",
        "points_raw": "3.00 points",
        "points_min": 3.0,
        "points_max": 3.0,
        "description": "Seed catalog summary.",
        "prerequisites_text": "",
        "sections": [],
    }
    filename = f"{COURSE_UID}.json"
    course_path = courses_dir / filename
    course_path.write_text(json.dumps(seed_detail, indent=2), encoding="utf-8")
    raw_entry = {
        "course_uid": COURSE_UID,
        "course_code": seed_detail["course_code"],
        "title": seed_detail["title"],
        "file_name": filename,
    }
    seed_index = [build_enriched_entry(raw_entry, seed_detail)]
    index_path = data_dir / "courses_enriched_index.json"
    index_path.write_text(json.dumps(seed_index, indent=2), encoding="utf-8")
    before_course = course_path.read_bytes()
    before_index = index_path.read_bytes()
    before_files = sorted(path.name for path in courses_dir.iterdir())

    payload = {
        "course_code": "COMS E9999",
        "title": "EVIDENCE-GROUNDED SYSTEMS",
        "points_raw": "3.00 points",
        "points_min": 3.0,
        "points_max": 3.0,
        "description": verified_description,
        "prerequisites_text": "",
        "notes_text": "",
        "sections": [
            {
                "term": "Fall 2025",
                "course_number": "COMS 9999",
                "section_call_number": "001/11111",
                "times": "T 9:00am - 12:15pm",
                "location": "Example Room 101",
                "instructor": "Ada Example",
                "points": "3.00",
                "enrollment_raw": "",
                "enrollment_current": None,
                "enrollment_capacity": None,
            }
        ],
    }
    llm = FakeLLM(payload)
    store = SyllabusStore(tmp_path / "syllabus-store")

    result = asyncio.run(
        import_file(
            file_bytes=pdf_bytes,
            filename=FIXTURE.name,
            llm_client=llm,
            courses_dir=str(courses_dir),
            enriched_index=seed_index,
            enriched_index_path=str(index_path),
            syllabus_store=store,
        )
    )

    assert result["success"] is True
    assert result["status"] == "published"
    assert course_path.read_bytes() == before_course
    assert index_path.read_bytes() == before_index
    assert sorted(path.name for path in courses_dir.iterdir()) == before_files
    assert store.manifest()["identity_count"] == 1
    assert store.manifest()["effective_published_count"] == 1

    runtime = apply_published_overlays(seed_index, store.effective_overlays())
    found = retrieve_courses(runtime, _intent(), str(courses_dir))
    assert len(found) == 1
    assert found[0]["course_uid"] == COURSE_UID
    assert found[0]["matched_sections"][0]["section_call_number"] == "001/11111"
    assert "evidence-grounded" in found[0]["description"].lower()

    assert len(llm.calls) == 1
    call = llm.calls[0]
    assert "不可信" in call["system_prompt"]
    assert call["messages"][0]["content"].startswith(
        "<UNTRUSTED_COURSE_DOCUMENT>\n"
    )
    assert "</UNTRUSTED_COURSE_DOCUMENT>" in call["messages"][0]["content"]
