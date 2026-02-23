"""
Track D: 文件导入模块测试。
"""

from __future__ import annotations

import json
import asyncio
from pathlib import Path
import sys

import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from file_importer import (
    extract_text_from_html,
    extract_text_from_pdf,
    generate_course_uid,
    import_file,
    validate_course_json,
)


FIXTURES_DIR = Path(__file__).parent / "fixtures"


class DummyLLM:
    def __init__(self, response_text: str):
        self.response_text = response_text

    async def chat(self, messages, system_prompt="", max_tokens=0) -> str:
        _ = (messages, system_prompt, max_tokens)
        return self.response_text


def test_extract_text_from_html() -> None:
    html_path = FIXTURES_DIR / "test_course.html"
    raw = html_path.read_bytes()
    text = extract_text_from_html(raw)
    assert "TEST E9999" in text
    assert "Introduction to Testing" in text


def test_extract_text_from_pdf() -> None:
    pytest.importorskip("pdfplumber")
    pdf_path = FIXTURES_DIR / "test_real_course.pdf"
    assert pdf_path.exists(), "Missing fixture: test_real_course.pdf"
    text = extract_text_from_pdf(pdf_path.read_bytes())
    assert text.strip()


def test_validate_course_json() -> None:
    valid = {
        "course_code": "TEST E9999",
        "title": "Introduction to Testing",
        "points_min": 3.0,
        "points_max": 3.0,
    }
    ok, msg = validate_course_json(valid)
    assert ok is True
    assert msg == ""

    invalid = {
        "title": "Missing code",
        "points_min": 3.0,
        "points_max": 3.0,
    }
    ok2, msg2 = validate_course_json(invalid)
    assert ok2 is False
    assert "course_code" in msg2


def test_generate_course_uid_deterministic() -> None:
    uid1 = generate_course_uid("TEST E9999", "Introduction to Testing")
    uid2 = generate_course_uid("TEST E9999", "Introduction to Testing")
    assert uid1 == uid2
    assert len(uid1) == 40


def test_import_file_html_and_duplicate_and_unsupported(tmp_path: Path) -> None:
    html_path = FIXTURES_DIR / "test_course.html"
    html_bytes = html_path.read_bytes()

    llm_payload = {
        "course_code": "TEST E9999",
        "title": "Introduction to Testing",
        "points_raw": "3.00 points",
        "points_min": 3.0,
        "points_max": 3.0,
        "bulletin_year": "2025-2026",
        "department_or_group": "Computer Science",
        "description": "This course covers fundamental principles of software testing.",
        "prerequisites_text": "COMS W3134",
        "notes_text": "",
        "sections": [
            {
                "term": "Spring 2026",
                "course_number": "TEST 9999",
                "section_call_number": "001/12345",
                "times": "M W 10:00am - 11:15am",
                "location": "100 Mudd Building",
                "instructor": "Dr. Test Professor",
                "points": "3.00",
                "enrollment_current": 10,
                "enrollment_capacity": 30,
            }
        ],
    }
    dummy_llm = DummyLLM(json.dumps(llm_payload, ensure_ascii=False))

    courses_dir = tmp_path / "courses_flat"
    index_path = tmp_path / "courses_enriched_index.json"
    enriched_index: list[dict] = []

    first = asyncio.run(
        import_file(
            file_bytes=html_bytes,
            filename="test_course.html",
            llm_client=dummy_llm,
            courses_dir=str(courses_dir),
            enriched_index=enriched_index,
            enriched_index_path=str(index_path),
        )
    )
    assert first["success"] is True
    assert first["course"]["course_code"] == "TEST E9999"

    uid = generate_course_uid("TEST E9999", "Introduction to Testing")
    saved_json = courses_dir / f"{uid}.json"
    assert saved_json.exists()

    second = asyncio.run(
        import_file(
            file_bytes=html_bytes,
            filename="test_course.html",
            llm_client=dummy_llm,
            courses_dir=str(courses_dir),
            enriched_index=enriched_index,
            enriched_index_path=str(index_path),
        )
    )
    assert second["success"] is False
    assert "already exists" in second["message"]

    unsupported = asyncio.run(
        import_file(
            file_bytes=b"abc",
            filename="test_course.txt",
            llm_client=dummy_llm,
            courses_dir=str(courses_dir),
            enriched_index=enriched_index,
            enriched_index_path=str(index_path),
        )
    )
    assert unsupported["success"] is False
    assert "Unsupported file format" in unsupported["message"]
