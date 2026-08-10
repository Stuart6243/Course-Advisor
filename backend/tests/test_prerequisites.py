from __future__ import annotations

import pytest

from prerequisites import (
    PrerequisiteRelationship,
    PrerequisiteStatus,
    compare_course_prerequisites,
    compare_prerequisite_counts,
    parse_prerequisites,
)


@pytest.mark.parametrize("text", [None, "", "   ", "Unknown", "未列出", "Non indiqué"])
def test_missing_evidence_is_unknown_and_never_zero(text: str | None) -> None:
    parsed = parse_prerequisites(text)
    assert parsed.status is PrerequisiteStatus.UNKNOWN
    assert parsed.relationship is PrerequisiteRelationship.UNKNOWN
    assert parsed.minimum_required_count is None
    assert parsed.required_codes == ()


@pytest.mark.parametrize(
    "text",
    [
        "No prerequisites.",
        "无先修课要求。",
        "Sin prerrequisitos.",
        "Aucun prérequis.",
    ],
)
def test_explicit_none_requires_explicit_source_text(text: str) -> None:
    parsed = parse_prerequisites(text)
    assert parsed.status is PrerequisiteStatus.EXPLICIT_NONE
    assert parsed.minimum_required_count == 0
    assert not parsed.recommended_only


@pytest.mark.parametrize(
    "text",
    [
        "No prerequisites listed.",
        "Prerequisites not provided.",
        "先修课未提供。",
        "Prerrequisitos no indicados.",
        "Prérequis non renseignés.",
    ],
)
def test_not_listed_is_not_explicit_none(text: str) -> None:
    parsed = parse_prerequisites(text)
    assert parsed.status is PrerequisiteStatus.UNKNOWN
    assert parsed.minimum_required_count is None


def test_and_relationship_has_deterministic_count() -> None:
    parsed = parse_prerequisites("COMS W3134 and MATH V2010 are required.")
    assert parsed.status is PrerequisiteStatus.LISTED
    assert parsed.required_codes == ("COMS W3134", "MATH V2010")
    assert parsed.relationship is PrerequisiteRelationship.AND
    assert parsed.minimum_required_count == 2


def test_or_relationship_has_deterministic_count() -> None:
    parsed = parse_prerequisites("COMS W3134 or MATH V2010.")
    assert parsed.relationship is PrerequisiteRelationship.OR
    assert parsed.minimum_required_count == 1


def test_mixed_relationship_respects_parentheses() -> None:
    parsed = parse_prerequisites(
        "(COMS W3134 or MATH V2010) and STAT GU4001."
    )
    assert parsed.relationship is PrerequisiteRelationship.MIXED
    assert parsed.minimum_required_count == 2


def test_unparenthesized_mixed_expression_uses_and_precedence() -> None:
    parsed = parse_prerequisites(
        "COMS W3134 or MATH V2010 and STAT GU4001."
    )
    assert parsed.relationship is PrerequisiteRelationship.MIXED
    assert parsed.minimum_required_count == 1


def test_ambiguous_list_and_prose_have_unknown_count() -> None:
    ambiguous = parse_prerequisites("COMS W3134, MATH V2010")
    prose = parse_prerequisites("Permission of the instructor.")
    assert ambiguous.relationship is PrerequisiteRelationship.UNKNOWN
    assert ambiguous.minimum_required_count is None
    assert prose.status is PrerequisiteStatus.LISTED
    assert prose.minimum_required_count is None


def test_oxford_list_inherits_final_conjunction() -> None:
    parsed = parse_prerequisites(
        "COMS W3134, MATH V2010, and STAT GU4001 are required."
    )
    assert parsed.relationship is PrerequisiteRelationship.AND
    assert parsed.minimum_required_count == 3


@pytest.mark.parametrize(
    "text",
    [
        "Recommended: COMS W3134 or MATH V2010.",
        "建议先学 COMS W3134 或 MATH V2010。",
        "Se recomienda COMS W3134 o MATH V2010.",
        "COMS W3134 ou MATH V2010 est recommandé.",
    ],
)
def test_recommended_only_is_listed_but_not_a_required_course(text: str) -> None:
    parsed = parse_prerequisites(text)
    assert parsed.status is PrerequisiteStatus.LISTED
    assert parsed.relationship is PrerequisiteRelationship.OR
    assert parsed.recommended_only
    assert parsed.minimum_required_count == 0


def test_full_text_is_preserved_without_truncation() -> None:
    text = (
        "Students must complete COMS W3134 and MATH V2010 with a grade of B "
        "or better before enrollment; instructor permission may also be required. "
        * 4
    ).strip()
    parsed = parse_prerequisites(text)
    assert len(text) > 300
    assert parsed.full_text == text
    assert parsed.as_dict()["full_text"] == text


def test_argmin_returns_ties_and_excludes_unknown_in_input_order() -> None:
    candidates = [
        ("A", parse_prerequisites("No prerequisites.")),
        ("B", parse_prerequisites("COMS W3134 and MATH V2010")),
        ("C", parse_prerequisites("COMS W3134 or MATH V2010")),
        ("D", parse_prerequisites("Permission of instructor")),
        ("E", parse_prerequisites("无先修课要求")),
    ]
    result = compare_prerequisite_counts(candidates, operation="argmin")
    assert result.winners == ("A", "E")
    assert result.winning_count == 0
    assert result.tied
    assert result.excluded_unknown == ("D",)


def test_argmax_is_deterministic() -> None:
    candidates = [
        ("A", parse_prerequisites("No prerequisites.")),
        ("B", parse_prerequisites("COMS W3134 and MATH V2010")),
        ("C", parse_prerequisites("COMS W3134 or MATH V2010")),
    ]
    result = compare_prerequisite_counts(candidates, operation="most")
    assert result.operation == "argmax"
    assert result.winners == ("B",)
    assert result.winning_count == 2
    assert not result.tied


def test_all_unknown_has_no_winner_instead_of_zero_winner() -> None:
    result = compare_prerequisite_counts(
        [
            ("A", parse_prerequisites("")),
            ("B", parse_prerequisites("Instructor permission")),
        ],
        operation="argmin",
    )
    assert result.winners == ()
    assert result.winning_count is None
    assert result.excluded_unknown == ("A", "B")


def test_course_dictionary_adapter_is_non_mutating_and_handles_missing_codes() -> None:
    courses = [
        {"course_code": "COMS W1001", "prerequisites_text": "No prerequisites."},
        {"title": "Synthetic", "prerequisites_text": "COMS W1001"},
    ]
    before = [dict(course) for course in courses]
    result = compare_course_prerequisites(courses, operation="argmin")
    assert result.winners == ("COMS W1001",)
    assert courses == before


def test_course_dictionary_adapter_uses_uid_for_duplicate_course_codes() -> None:
    courses = [
        {
            "course_uid": "uid-a",
            "course_code": "COMS W1001",
            "prerequisites_text": "COMS W3134 and MATH V2010",
        },
        {
            "course_uid": "uid-b",
            "course_code": "COMS W1001",
            "prerequisites_text": "No prerequisites.",
        },
    ]
    result = compare_course_prerequisites(courses, operation="argmin")
    assert result.winners == ("uid-b",)


def test_invalid_comparison_operation_is_rejected() -> None:
    with pytest.raises(ValueError):
        compare_prerequisite_counts([], operation="median")
