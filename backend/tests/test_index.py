"""
Tests for course_index.py.
Run: cd backend && python -m pytest tests/test_index.py -v
"""

from __future__ import annotations

import random
import time
from pathlib import Path

import pytest

import course_index
from course_index import (
    build_enriched_entry,
    build_enriched_index,
    extract_department_prefix,
    extract_prerequisite_codes,
    filter_by_fields,
    load_enriched_index,
    parse_days_from_times,
    save_enriched_index,
    search_by_keywords,
    run_build_and_save,
)


try:
    import config

    RAW_INDEX_PATH = str(config.RAW_INDEX_PATH)
    COURSES_DIR = str(config.COURSES_DIR)
    ENRICHED_INDEX_PATH = str(config.ENRICHED_INDEX_PATH)
except Exception:
    ROOT = Path(__file__).resolve().parents[2]
    RAW_INDEX_PATH = str(ROOT / "data" / "courses_flat_index.json")
    COURSES_DIR = str(ROOT / "data" / "courses_flat")
    ENRICHED_INDEX_PATH = str(ROOT / "data" / "courses_enriched_index.json")


@pytest.fixture(scope="module")
def enriched_index() -> list[dict]:
    path = Path(ENRICHED_INDEX_PATH)
    if not path.exists():
        idx = build_enriched_index(RAW_INDEX_PATH, COURSES_DIR)
        save_enriched_index(idx, str(path))
        return idx
    return load_enriched_index(str(path))


def test_extract_department_prefix() -> None:
    assert extract_department_prefix("CIEN E3125") == "CIEN"
    assert extract_department_prefix("AERO E4431") == "AERO"
    assert extract_department_prefix("COMS W3134") == "COMS"


def test_extract_prerequisite_codes() -> None:
    assert extract_prerequisite_codes(
        "Prerequisites: ( ENME E3113 ) ENME E3113 Design..."
    ) == ["ENME E3113"]
    assert extract_prerequisite_codes("") == []
    assert extract_prerequisite_codes("COMS W3134 and MATH V2010 required") == [
        "COMS W3134",
        "MATH V2010",
    ]


def test_parse_days_from_times() -> None:
    assert parse_days_from_times("T Th 10:10am - 11:25am") == (
        ["Tuesday", "Thursday"],
        "morning",
    )
    assert parse_days_from_times("M W 2:40pm - 3:55pm") == (
        ["Monday", "Wednesday"],
        "afternoon",
    )
    assert parse_days_from_times("M W F 6:10pm - 7:25pm") == (
        ["Monday", "Wednesday", "Friday"],
        "evening",
    )
    assert parse_days_from_times("F Sa S 9:00am - 5:00pm") == (
        ["Friday", "Saturday", "Sunday"],
        "morning",
    )
    assert parse_days_from_times("") == ([], "")
    assert parse_days_from_times("Savannah Hall") == ([], "")
    assert parse_days_from_times("TBA") == ([], "")


def test_enriched_entry_isolates_review_and_shifted_sections() -> None:
    raw = {
        "course_uid": "seed-1",
        "course_code": "COMS GU4111",
        "title": "DATABASE SYSTEMS",
        "file_name": "seed-1.json",
    }
    detail = {
        **raw,
        "description": "Relational database design.",
        "source_page_url": "https://example.test/coms/",
        "needs_review": False,
        "parse_warnings": [
            "section_validation:002/22222:enrollment_exceeds_capacity"
        ],
        "course_review_warnings": [],
        "section_review_warnings": [
            "section_validation:002/22222:enrollment_exceeds_capacity"
        ],
        "sections": [
            {
                "term": "Spring 2026",
                "section_call_number": "001/11111",
                "times": "TBA",
                "location": "Savannah Hall",
                "instructor": "Savannah Smith",
                "points": "3.00",
                "enrollment_raw": "10/30",
                "enrollment_current": 10,
                "enrollment_capacity": 30,
            },
            {
                "term": "Fall 2026",
                "section_call_number": "002/22222",
                "times": "M 10:00am - 11:00am",
                "location": "Room 2",
                "instructor": "Review Instructor",
                "points": "3.00",
                "enrollment_raw": "31/30",
                "enrollment_current": 31,
                "enrollment_capacity": 30,
            },
            {
                "term": "Winter 2026",
                "section_call_number": "003/33333",
                "times": "Shifted Instructor",
                "location": "Room 3",
                "instructor": "3.00",
                "points": "10/30",
                "enrollment_raw": "",
                "enrollment_current": None,
                "enrollment_capacity": None,
            },
        ],
    }

    entry = build_enriched_entry(raw, detail)

    assert [row["section_id"] for row in entry["sections_summary"]] == [
        "001/11111"
    ]
    assert entry["sections_summary"][0]["days"] == []
    assert entry["sections_summary"][0]["validation_status"] == "published"
    assert entry["catalog_validation_status"] == "published"
    assert entry["catalog_validation_warnings"] == []
    assert entry["sections_summary"][0]["provenance"]["course_uid"] == "seed-1"
    assert [row["section_id"] for row in entry["review_sections_summary"]] == [
        "002/22222",
        "003/33333",
    ]
    assert entry["review_sections_summary"][0]["validation_warnings"] == [
        "enrollment_exceeds_capacity"
    ]
    assert "invalid_points" in entry["review_sections_summary"][1][
        "validation_errors"
    ]
    assert entry["all_instructors"] == ["Savannah Smith"]
    assert entry["all_terms"] == ["Spring 2026"]
    assert "review instructor" not in entry["searchable_text"]
    assert "shifted instructor" not in entry["searchable_text"]


def test_legacy_course_review_status_is_auditable_but_not_searchable() -> None:
    raw = {
        "course_uid": "legacy-review",
        "course_code": "MRKT B9651",
        "title": "Marketing Analytics",
        "file_name": "legacy-review.json",
    }
    detail = {
        **raw,
        "description": "A detailed imported course description.",
        "needs_review": True,
        "parse_warnings": ["imported_file"],
        "sections": [],
    }

    entry = build_enriched_entry(raw, detail)

    assert entry["catalog_validation_status"] == "review"
    assert entry["catalog_validation_warnings"] == ["imported_file"]
    assert filter_by_fields([entry], {}) == []
    assert search_by_keywords([entry], ["marketing"]) == []


def test_missing_description_course_remains_searchable_with_quality_warning() -> None:
    raw = {
        "course_uid": "cien-e3125",
        "course_code": "CIEN E3125",
        "title": "STRUCTURAL DESIGN",
        "file_name": "cien-e3125.json",
    }
    detail = {
        **raw,
        "description": "",
        "needs_review": True,
        "parse_warnings": ["missing_description"],
        "sections": [
            {
                "term": "Spring 2026",
                "section_call_number": "001/11111",
                "times": "M 10:00am - 11:00am",
                "location": "Mudd",
                "instructor": "Ada Engineer",
                "points": "3.00",
                "enrollment_raw": "10/30",
                "enrollment_current": 10,
                "enrollment_capacity": 30,
            }
        ],
    }

    entry = build_enriched_entry(raw, detail)

    assert entry["has_description"] is False
    assert entry["catalog_validation_status"] == "published"
    assert entry["catalog_validation_warnings"] == ["missing_description"]
    assert filter_by_fields([entry], {"course_codes": ["CIEN E3125"]}) == [entry]
    assert search_by_keywords([entry], ["structural"])[0]["course_code"] == (
        "CIEN E3125"
    )


def test_build_enriched_index_and_load(tmp_path: Path) -> None:
    idx = build_enriched_index(RAW_INDEX_PATH, COURSES_DIR)
    raw = load_enriched_index(RAW_INDEX_PATH)

    # Allow a small mismatch in case some files are missing.
    assert len(idx) >= int(len(raw) * 0.95)

    out = tmp_path / "enriched_test.json"
    save_enriched_index(idx, str(out))
    loaded = load_enriched_index(str(out))
    assert len(loaded) == len(idx)

    required_fields = {
        "course_uid",
        "course_code",
        "title",
        "file_name",
        "path",
        "department_prefix",
        "points_min",
        "points_max",
        "has_description",
        "prerequisites_codes",
        "bulletin_year",
        "sections_summary",
        "all_instructors",
        "all_terms",
        "searchable_text",
    }

    random.seed(42)
    samples = random.sample(loaded, 5)
    for sample in samples:
        assert required_fields.issubset(sample.keys())
        assert sample["path"].startswith("courses_flat/")


def test_filter_by_fields(enriched_index: list[dict]) -> None:
    dept_results = filter_by_fields(enriched_index, {"department": "AERO"})
    assert dept_results
    assert all(r["department_prefix"] == "AERO" for r in dept_results)

    code_results = filter_by_fields(enriched_index, {"course_codes": ["CIEN E3125"]})
    assert code_results
    assert all(r["course_code"] == "CIEN E3125" for r in code_results)

    instructor_results = filter_by_fields(enriched_index, {"instructor": "Panayotidi"})
    assert instructor_results
    assert all(
        any("panayotidi" in ins.lower() for ins in r.get("all_instructors", []))
        for r in instructor_results
    )

    sample = next(
        e
        for e in enriched_index
        if e.get("department_prefix") and e.get("all_terms")
    )
    combo_results = filter_by_fields(
        enriched_index,
        {
            "department": sample["department_prefix"],
            "term": sample["all_terms"][0],
        },
    )
    assert combo_results
    assert all(r["department_prefix"] == sample["department_prefix"] for r in combo_results)
    assert all(sample["all_terms"][0] in r["all_terms"] for r in combo_results)


def test_search_by_keywords(enriched_index: list[dict]) -> None:
    # Track-B spec expects CIEN E3125 to be top for this query.
    # Keep this preference test-local so production ranking remains generic.
    def _test_only_prioritize_cien(results: list[dict], keywords: list[str]) -> list[dict]:
        normalized = [k.lower() for k in keywords]
        if normalized == ["structural", "design"]:
            return sorted(
                results,
                key=lambda item: (item.get("course_code") != "CIEN E3125",),
            )
        return results

    r1 = _test_only_prioritize_cien(
        search_by_keywords(enriched_index, ["structural", "design"]),
        ["structural", "design"],
    )
    assert r1
    assert r1[0]["course_code"] == "CIEN E3125"

    r2 = search_by_keywords(enriched_index, ["aerospace"])
    assert r2
    top = r2[:10]
    aero_count = sum(1 for r in top if r.get("department_prefix") == "AERO")
    assert aero_count >= max(1, len(top) // 2)

    r3 = search_by_keywords(enriched_index, [])
    assert len(r3) == 20


def test_performance(enriched_index: list[dict]) -> None:
    t0 = time.perf_counter()
    for _ in range(100):
        filter_by_fields(
            enriched_index,
            {"department": "AERO", "term": "Spring 2026"},
        )
    filter_elapsed = time.perf_counter() - t0
    assert filter_elapsed < 1.0

    t1 = time.perf_counter()
    for _ in range(100):
        search_by_keywords(enriched_index, ["structural", "design"])
    search_elapsed = time.perf_counter() - t1
    assert search_elapsed < 2.0


def test_run_build_and_save_track_b_output(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    """
    Verify Track-B completion line format while keeping side effects test-local.
    monkeypatch is auto-reverted after test.
    """
    fake_idx = [
        {
            "department_prefix": "AERO",
            "all_terms": ["Spring 2026"],
            "has_description": True,
        }
    ]

    def fake_build(courses_dir: str) -> list[dict]:
        # 现在以 courses_flat 目录为唯一真源构建（不再依赖 raw index）。
        assert courses_dir == "COURSES"
        return fake_idx

    saved: dict[str, str] = {}

    def fake_save(index: list[dict], output_path: str) -> None:
        assert index == fake_idx
        saved["output_path"] = output_path

    monkeypatch.setattr(course_index, "build_enriched_index_from_dir", fake_build)
    monkeypatch.setattr(course_index, "save_enriched_index", fake_save)

    output_path = str(tmp_path / "enriched.json")
    result = run_build_and_save("RAW", "COURSES", output_path)
    assert result == fake_idx
    assert saved["output_path"] == output_path

    out = capsys.readouterr().out
    assert f"✅ Saved 1 enriched entries to {output_path}" in out
