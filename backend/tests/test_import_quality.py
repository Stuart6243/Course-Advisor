from __future__ import annotations

from pathlib import Path
import sys

import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import config
import file_importer
from file_importer import assess_import, parse_points_raw, validate_course_code


def high_quality_payload() -> dict:
    return {
        "course_code": "COMS GU4111",
        "title": "Introduction to Databases",
        "points_raw": "3.00 points",
        "points_min": 3.0,
        "points_max": 3.0,
        "description": "A detailed study of database design, queries, and transactions.",
        "sections": [
            {
                "term": "Spring 2026",
                "course_number": "COMS 4111",
                "section_call_number": "001/12345",
                "times": "M W 10:00am - 11:15am",
                "instructor": "Ada Lovelace",
                "points": "3.00",
                "enrollment_raw": "10/30",
                "enrollment_current": 10,
                "enrollment_capacity": 30,
            }
        ],
    }


def source_text() -> str:
    return (
        "COMS GU4111 Introduction to Databases 3.00 points Spring 2026 "
        "section 001/12345 3.00 M W 10:00am - 11:15am Ada Lovelace "
        "A detailed study of database design, queries, and transactions."
    )


def test_quality_gate_all_three_statuses_are_reachable() -> None:
    published = assess_import(high_quality_payload(), source_text())
    assert published.status == "published"
    assert published.hard_errors == ()

    suspicious = high_quality_payload()
    suspicious["sections"][0]["enrollment_raw"] = "31/30"
    suspicious["sections"][0]["enrollment_current"] = 31
    review = assess_import(suspicious, source_text())
    assert review.status == "review"
    assert any("enrollment_exceeds_capacity" in issue for issue in review.quality_issues)

    malformed = high_quality_payload()
    malformed["sections"][0]["times"] = "Savannah Eisner"
    rejected = assess_import(malformed, source_text())
    assert rejected.status == "rejected"
    assert any("invalid_times" in issue for issue in rejected.hard_errors)


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (config.AUTO_PUBLISH_QUALITY_SCORE - 1, "review"),
        (config.AUTO_PUBLISH_QUALITY_SCORE, "published"),
    ],
)
def test_auto_publish_threshold_has_one_configured_boundary(
    monkeypatch: pytest.MonkeyPatch, score: int, expected: str
) -> None:
    monkeypatch.setattr(file_importer, "quality_score", lambda _data: (score, []))
    assert assess_import(high_quality_payload(), source_text()).status == expected


def test_unverified_identity_evidence_cannot_auto_publish() -> None:
    assessment = assess_import(
        high_quality_payload(), "COMS GU4111 Introduction to Databases 3.00 points"
    )
    assert assessment.status == "review"
    assert "unverified_evidence:section_0.term" in assessment.quality_issues
    assert "unverified_evidence:section_0.section_id" in assessment.quality_issues


@pytest.mark.parametrize(
    "code",
    ["PSAM UN3707", "BINF GU4001", "EESC GR5400", "COMS W4111"],
)
def test_supported_course_code_level_designators(code: str) -> None:
    assert validate_course_code(code)


def test_level_less_course_code_is_not_in_current_seed_grammar() -> None:
    assert not validate_course_code("AERO 3001")


@pytest.mark.parametrize(
    "raw", ["course 4111 worth 3 points", "3 points trailing instructions", "40"]
)
def test_points_do_not_accept_embedded_or_out_of_bounds_numbers(raw: str) -> None:
    assert parse_points_raw(raw) is None
