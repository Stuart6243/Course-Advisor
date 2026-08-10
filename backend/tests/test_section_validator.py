from __future__ import annotations

from pathlib import Path
import sys

import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from section_validator import (
    parse_day_tokens,
    parse_points_value,
    validate_catalog_record,
    validate_section,
)


def valid_section(**overrides):
    section = {
        "term": "Spring 2026",
        "course_number": "COMS 4111",
        "section_call_number": "001/12345",
        "times": "T Th 10:10am - 11:25am",
        "location": "Room 1",
        "instructor": "Savannah Eisner",
        "points": "3.00",
        "enrollment_raw": "10/30",
        "enrollment_current": 10,
        "enrollment_capacity": 30,
    }
    section.update(overrides)
    return section


def test_day_tokens_have_alphabetic_boundaries() -> None:
    assert parse_day_tokens("T Th 10:10am - 11:25am") == ["Tuesday", "Thursday"]
    assert parse_day_tokens("Savannah Eisner") == []
    assert parse_day_tokens("TBA") == []
    assert parse_day_tokens("Saturday Sa 9:00am - 10:00am") == ["Saturday"]
    assert parse_day_tokens("F Sa S 9:00am - 5:00pm") == [
        "Friday",
        "Saturday",
        "Sunday",
    ]


def test_saved_standalone_s_schedule_is_publishable_as_sunday() -> None:
    result = validate_section(
        valid_section(times="F Sa S 9:00am - 5:00pm"), require_identity=True
    )
    assert result.status == "published"
    assert result.errors == ()
    assert result.days == ("Friday", "Saturday", "Sunday")


def test_empty_schedule_and_instructor_are_valid_placeholders() -> None:
    result = validate_section(
        valid_section(times="", instructor=""), require_identity=True
    )
    assert result.valid
    assert result.status == "published"
    assert result.filterable_fields["schedule"] is True


@pytest.mark.parametrize(
    ("overrides", "expected_error"),
    [
        ({"times": "Savannah Eisner"}, "invalid_times"),
        ({"times": "M 10:00am - 11:00am Ada Lovelace"}, "invalid_times"),
        ({"instructor": "3.00"}, "invalid_instructor"),
        ({"instructor": "10/30"}, "invalid_instructor"),
        ({"points": "enrollment 10/30"}, "invalid_points"),
        ({"enrollment_raw": "Professor Smith"}, "invalid_enrollment_raw"),
        ({"enrollment_raw": "10/30", "enrollment_current": 9}, "enrollment_current_mismatch"),
        ({"term": "2026 Spring"}, "invalid_term"),
    ],
)
def test_shifted_or_malformed_columns_are_not_publishable(
    overrides, expected_error
) -> None:
    result = validate_section(valid_section(**overrides), require_identity=True)
    assert result.status == "review"
    assert expected_error in result.errors


def test_enrollment_over_capacity_is_review_not_hard_rejection() -> None:
    result = validate_section(
        valid_section(
            enrollment_raw="31/30", enrollment_current=31, enrollment_capacity=30
        ),
        require_identity=True,
    )
    assert result.valid
    assert result.status == "review"
    assert result.warnings == ("enrollment_exceeds_capacity",)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("3.00 points", (3.0, 3.0)),
        ("1.5-6 credits", (1.5, 6.0)),
        ("course 4111 has 3 points", None),
        ("6-1", None),
        ("31", None),
    ],
)
def test_points_parser_requires_a_complete_bounded_expression(raw, expected) -> None:
    assert parse_points_value(raw) == expected


def test_identity_is_required_for_imported_sections() -> None:
    result = validate_section(
        valid_section(term="", section_call_number=""), require_identity=True
    )
    assert set(result.errors) >= {"missing_term", "missing_section_id"}


def test_catalog_missing_description_is_nonblocking_but_imported_file_is_not() -> None:
    missing = validate_catalog_record(
        {"needs_review": True, "parse_warnings": ["missing_description"]}
    )
    assert missing.status == "published"
    assert missing.warnings == ("missing_description",)
    assert missing.blocking_warnings == ()

    imported = validate_catalog_record(
        {"needs_review": True, "parse_warnings": ["imported_file"]}
    )
    assert imported.status == "review"
    assert imported.blocking_warnings == ("imported_file",)

    unknown = validate_catalog_record({"needs_review": True, "parse_warnings": []})
    assert unknown.status == "review"
    assert unknown.blocking_warnings == ("legacy_needs_review",)
