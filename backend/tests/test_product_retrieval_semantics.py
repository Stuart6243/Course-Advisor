"""Product-semantic retrieval regressions using read-only formal data."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from course_index import load_enriched_index
from course_retriever import retrieve_courses, select_distinct_code_first
from query_parser import contains_ascii_alias, normalize_question, rule_based_extract
from suitability import SuitabilityReason, SuitabilityStatus, assess_beginner_suitability


ROOT = Path(__file__).resolve().parents[2]
REAL_COURSES_DIR = ROOT / "data" / "courses_flat"
REAL_INDEX_PATH = ROOT / "data" / "courses_enriched_index.json"


@pytest.fixture(scope="module")
def real_index() -> list[dict]:
    return load_enriched_index(str(REAL_INDEX_PATH))


def _intent(question: str) -> dict:
    parsed = rule_based_extract(normalize_question(question))
    assert parsed is not None, question
    return parsed


@pytest.mark.parametrize("question", ["cs课", "目前cs有什么……"])
def test_ascii_cs_alias_matches_when_adjacent_to_cjk(question: str) -> None:
    intent = _intent(question)
    assert intent["department"] == "COMS"
    assert intent["department_terms"] == ["cs"]


@pytest.mark.parametrize("text", ["scs课", "cs50课", "_cs课"])
def test_ascii_cs_alias_rejects_substrings_and_identifier_neighbors(text: str) -> None:
    assert contains_ascii_alias(text, "cs") is False
    intent = rule_based_extract(normalize_question(text))
    assert intent is None or intent["department"] != "COMS"


@pytest.mark.parametrize(
    "question",
    ["无基础的计算机课程", "计算机入门课", "零基础编程课", "数学方面的基础课"],
)
def test_beginner_wording_is_structured_suitability_not_title_translation(
    question: str,
) -> None:
    normalized = normalize_question(question)
    assert "fundamentals" not in normalized.lower()
    assert "introduction" not in normalized.lower()
    assert _intent(question)["suitability"] == "beginner"


def test_generic_mathematics_defaults_to_apma_but_explicit_anchors_win() -> None:
    math = _intent("数学方面的基础课")
    assert math["department"] == "APMA"
    assert math["department_defaulted_from"] == "mathematics"

    explicit_code = _intent("COMS W3203 数学方面的基础课")
    assert explicit_code["course_codes"] == ["COMS W3203"]
    assert explicit_code["department"] != "APMA"
    assert explicit_code["department_defaulted_from"] is None

    explicit_department = _intent("computer science mathematics courses")
    assert explicit_department["department"] == "COMS"
    assert explicit_department["department_defaulted_from"] is None


def test_suitability_evidence_keeps_blank_prerequisites_unknown() -> None:
    evidence = assess_beginner_suitability(
        {
            "course_code": "COMS W1004",
            "title": "PROGRAMMING IN JAVA",
            "description": "Assumes no prior programming background.",
            "prerequisites_text": "",
        }
    )
    assert evidence.status is SuitabilityStatus.POSITIVE
    assert evidence.reason is SuitabilityReason.EXPLICIT_NO_PRIOR_BACKGROUND
    assert evidence.prerequisite_status.value == "unknown"


def test_title_words_never_override_explicit_prerequisites_or_advanced_level() -> None:
    introduction_with_prerequisite = assess_beginner_suitability(
        {
            "course_code": "COMS W4111",
            "title": "INTRODUCTION TO DATABASES",
            "description": "An introduction to database systems.",
            "prerequisites_text": "Prerequisites: COMS W3134.",
        }
    )
    assert introduction_with_prerequisite.status is SuitabilityStatus.NEGATIVE
    assert introduction_with_prerequisite.reason is SuitabilityReason.LISTED_PREREQUISITES

    advanced_with_missing_prerequisite_data = assess_beginner_suitability(
        {
            "course_code": "TEST E3500",
            "title": "ADVANCED PROGRAMMING",
            "description": "Covers programming fundamentals.",
            "prerequisites_text": "",
        }
    )
    assert advanced_with_missing_prerequisite_data.status is SuitabilityStatus.NEGATIVE
    assert advanced_with_missing_prerequisite_data.reason is SuitabilityReason.ADVANCED_LEVEL
    assert advanced_with_missing_prerequisite_data.prerequisite_status.value == "unknown"


def test_narrow_reliable_intro_evidence_precedes_counter_evidence() -> None:
    explicitly_for_beginners = assess_beginner_suitability(
        {
            "course_code": "TEST E3500",
            "title": "SPECIAL TOPICS",
            "description": "This course is designed for complete beginners.",
            "prerequisites_text": "Prerequisites: TEST E2000.",
        }
    )
    assert explicitly_for_beginners.status is SuitabilityStatus.POSITIVE
    assert (
        explicitly_for_beginners.reason
        is SuitabilityReason.RELIABLE_INTRODUCTORY_EVIDENCE
    )
    assert explicitly_for_beginners.prerequisite_status.value == "listed"

    explicit_basic_introduction = assess_beginner_suitability(
        {
            "course_code": "TEST E1001",
            "title": "INFORMATION SCIENCE",
            "description": "A basic introduction to information science.",
            "prerequisites_text": "",
        }
    )
    assert explicit_basic_introduction.status is SuitabilityStatus.POSITIVE
    assert (
        explicit_basic_introduction.reason
        is SuitabilityReason.RELIABLE_INTRODUCTORY_EVIDENCE
    )

    generic_introduction = assess_beginner_suitability(
        {
            "course_code": "TEST E2500",
            "title": "INTRODUCTION TO SYSTEMS",
            "description": "A general introduction to modern systems.",
            "prerequisites_text": "Prerequisites: TEST E1000.",
        }
    )
    assert generic_introduction.status is SuitabilityStatus.NEGATIVE
    assert generic_introduction.reason is SuitabilityReason.LISTED_PREREQUISITES


def test_beginner_retrieval_uses_complete_detail_and_excludes_negative_courses(
    real_index: list[dict],
) -> None:
    computer_intent = _intent("无基础的计算机课程")
    computer = retrieve_courses(
        real_index, computer_intent, str(REAL_COURSES_DIR), max_results=5
    )
    assert computer[0]["course_code"] == "COMS W1004"
    assert computer[0]["suitability"]["status"] == "positive"
    assert computer[0]["suitability"]["reason"] == "explicit_no_prior_background"
    assert computer[0]["suitability"]["prerequisite_status"] == "unknown"
    assert "COMS W3157" not in {course["course_code"] for course in computer}
    assert "COMS W4111" not in {course["course_code"] for course in computer}
    assert all(course["suitability"]["status"] != "negative" for course in computer)

    programming_intent = _intent("零基础编程课")
    programming = retrieve_courses(
        real_index, programming_intent, str(REAL_COURSES_DIR), max_results=5
    )
    assert [course["course_code"] for course in programming] == ["COMS W1004"]


def test_real_upper_level_intro_titles_have_negative_suitability(
    real_index: list[dict],
) -> None:
    by_code = {entry["course_code"]: entry for entry in real_index}
    for code in ("COMS W3157", "COMS W4111"):
        detail = json.loads((ROOT / "data" / by_code[code]["path"]).read_text())
        evidence = assess_beginner_suitability(detail)
        assert evidence.status is SuitabilityStatus.NEGATIVE
        assert evidence.reason is SuitabilityReason.LISTED_PREREQUISITES


def test_mathematics_beginner_fallback_is_apma_and_explicitly_unknown(
    real_index: list[dict],
) -> None:
    intent = _intent("数学方面的基础课")
    courses = retrieve_courses(real_index, intent, str(REAL_COURSES_DIR), max_results=5)
    assert courses
    assert all(course["course_code"].startswith("APMA ") for course in courses)
    assert all(course["suitability"]["status"] == "unknown" for course in courses)


def test_distinct_code_first_refills_duplicates_only_after_representatives() -> None:
    ranked = [
        {"course_uid": "a1", "course_code": "TEST E1000"},
        {"course_uid": "a2", "course_code": "TEST E1000"},
        {"course_uid": "a3", "course_code": "TEST E1000"},
        {"course_uid": "b1", "course_code": "TEST E2000"},
        {"course_uid": "b2", "course_code": "TEST E2000"},
    ]
    selected = select_distinct_code_first(ranked, 5)
    assert [entry["course_uid"] for entry in selected] == [
        "a1",
        "b1",
        "a2",
        "a3",
        "b2",
    ]


def test_real_data_science_top_five_are_distinct_and_exact(
    real_index: list[dict],
) -> None:
    intent = _intent("给我推荐一些 Data Science 课程")
    courses = retrieve_courses(real_index, intent, str(REAL_COURSES_DIR), max_results=5)
    assert [
        (course["course_code"], course["course_uid"])
        for course in courses
    ] == [
        ("ORCA E2500", "11b0b5906008531ed6a10935e2e0c5b1f73eaf36"),
        ("CSEE W4121", "41ad00da31895ad6b3c216f19aa5800251b02366"),
        ("CSOR W4246", "5ac34094d3018c02e45baa691cd0796a5936f882"),
        ("MECE E4520", "1eb6fb3b6181447c76b6b488340cda8a7a3ea468"),
        ("ORCA E4500", "2299bcb12007fafa0ce640a006473e1508a632c5"),
    ]
    assert intent["retrieval_metadata"] == {
        "total_matches": 84,
        "displayed": 5,
        "truncated": True,
    }


def test_formal_course_files_remain_one_file_per_uid() -> None:
    rows = json.loads((ROOT / "data" / "courses_flat_index.json").read_text())
    assert len(rows) == 1021
    assert len({row["course_uid"] for row in rows}) == 1021
    assert len(list(REAL_COURSES_DIR.glob("*.json"))) == 1021
