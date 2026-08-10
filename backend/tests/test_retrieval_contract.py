"""Regression tests for the exact Phase-C retrieval contract.

Synthetic course/detail files live exclusively under ``tmp_path``.  The few
real-data acceptance checks are read-only and guard the concrete failures from
the implementation handoff.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from course_index import (
    filter_by_fields,
    load_enriched_index,
    parse_days_from_times,
)
from course_retriever import retrieve_courses
from query_parser import normalize_question, rule_based_extract
from response_generator import build_answer_prompt
from syllabus_store import SyllabusStore, apply_published_overlays


ROOT = Path(__file__).resolve().parents[2]
REAL_COURSES_DIR = ROOT / "data" / "courses_flat"
REAL_INDEX_PATH = ROOT / "data" / "courses_enriched_index.json"


def _intent(**overrides) -> dict:
    result = {
        "query_type": "search",
        "course_codes": [],
        "keywords": [],
        "department": None,
        "department_terms": [],
        "instructor": None,
        "time_preference": None,
        "day_preference": [],
        "points_range": None,
        "term": None,
        "comparison_targets": [],
        "original_question": "test",
    }
    result.update(overrides)
    return result


def _make_index(tmp_path: Path, courses: list[dict]) -> tuple[list[dict], Path]:
    data_dir = tmp_path / "data"
    courses_dir = data_dir / "courses_flat"
    courses_dir.mkdir(parents=True)

    index: list[dict] = []
    for position, raw_course in enumerate(courses):
        course = dict(raw_course)
        course["sections"] = [
            {
                "points": "3.00",
                "enrollment_raw": "",
                "enrollment_current": None,
                "enrollment_capacity": None,
                **section,
            }
            for section in (raw_course.get("sections") or [])
        ]
        filename = f"course-{position}.json"
        (courses_dir / filename).write_text(
            json.dumps(course, ensure_ascii=False), encoding="utf-8"
        )

        sections_summary = []
        for section in course.get("sections") or []:
            days, time_of_day = parse_days_from_times(section.get("times") or "")
            sections_summary.append(
                {
                    "term": section.get("term", ""),
                    "times": section.get("times", ""),
                    "days": days,
                    "time_of_day": time_of_day,
                    "instructor": section.get("instructor", ""),
                    "location": section.get("location", ""),
                    "points": section.get("points", "3.00"),
                    "enrollment_raw": section.get("enrollment_raw", ""),
                    "enrollment_current": section.get("enrollment_current"),
                    "enrollment_capacity": section.get("enrollment_capacity"),
                }
            )

        code = course["course_code"]
        title = course.get("title", "")
        description = course.get("description", "")
        index.append(
            {
                "course_uid": f"uid-{position}",
                "course_code": code,
                "title": title,
                "file_name": filename,
                "path": f"courses_flat/{filename}",
                "department_prefix": code.split()[0],
                "points_min": course.get("points_min", 3.0),
                "points_max": course.get("points_max", 3.0),
                "has_description": bool(description),
                "sections_summary": sections_summary,
                "all_instructors": [
                    section["instructor"]
                    for section in sections_summary
                    if section.get("instructor")
                ],
                "all_terms": list(
                    dict.fromkeys(
                        section["term"]
                        for section in sections_summary
                        if section.get("term")
                    )
                ),
                "searchable_text": f"{code} {title} {description}".lower(),
            }
        )

    return index, courses_dir


def test_section_constraints_must_match_one_section(tmp_path: Path) -> None:
    index, courses_dir = _make_index(
        tmp_path,
        [
            {
                "course_code": "CSEE W4119",
                "title": "COMPUTER NETWORKS",
                "sections": [
                    {
                        "term": "Fall 2025",
                        "times": "M W 4:10pm - 5:25pm",
                        "instructor": "Henning Schulzrinne",
                    },
                    {
                        "term": "Spring 2026",
                        "times": "T Th 11:40am - 12:55pm",
                        "instructor": "Xia Zhou",
                    },
                ],
            }
        ],
    )

    courses = retrieve_courses(
        index,
        _intent(
            query_type="detail",
            course_codes=["CSEE W4119"],
            day_preference=["Monday"],
            time_preference="morning",
        ),
        str(courses_dir),
    )

    assert courses == [], "Monday and morning must not come from different sections"


def test_term_and_instructor_must_match_one_section() -> None:
    index = [
        {
            "course_code": "TEST W1000",
            "department_prefix": "TEST",
            "points_min": 3.0,
            "points_max": 3.0,
            "sections_summary": [
                {"term": "Fall 2025", "times": "M 10:00am", "instructor": "Ada One"},
                {"term": "Spring 2026", "times": "M 10:00am", "instructor": "Grace Two"},
            ],
        }
    ]

    assert filter_by_fields(
        index, {"term": "Spring 2026", "instructor": "Ada One"}
    ) == []


def test_zero_results_are_not_silently_relaxed(tmp_path: Path) -> None:
    index, courses_dir = _make_index(
        tmp_path,
        [
            {
                "course_code": "COMS W4111",
                "title": "INTRODUCTION TO DATABASES",
                "sections": [
                    {"term": "Fall 2025", "times": "M W 4:10pm - 5:25pm"}
                ],
            }
        ],
    )

    courses = retrieve_courses(
        index,
        _intent(
            query_type="detail",
            course_codes=["COMS W4111"],
            day_preference=["Monday"],
            time_preference="evening",
        ),
        str(courses_dir),
    )

    assert courses == []


def test_compare_targets_defensively_drive_retrieval_without_course_codes(
    tmp_path: Path,
) -> None:
    index, courses_dir = _make_index(
        tmp_path,
        [
            {"course_code": "CIEN E3125", "title": "STRUCTURAL DESIGN"},
            {"course_code": "ENME E3113", "title": "MECHANICS OF SOLIDS"},
            {"course_code": "COMS W4111", "title": "DATABASE SYSTEMS"},
        ],
    )

    courses = retrieve_courses(
        index,
        _intent(
            query_type="compare",
            course_codes=[],
            comparison_targets=["CIEN E3125", "ENME E3113"],
        ),
        str(courses_dir),
    )

    assert len(courses) == 2
    assert {course["course_code"] for course in courses} == {
        "CIEN E3125",
        "ENME E3113",
    }


def test_matched_sections_contains_only_complete_matches(tmp_path: Path) -> None:
    index, courses_dir = _make_index(
        tmp_path,
        [
            {
                "course_code": "COMS W4111",
                "title": "INTRODUCTION TO DATABASES",
                "sections": [
                    {
                        "term": "Fall 2025",
                        "times": "M W 4:10pm - 5:25pm",
                        "instructor": "Kenneth Ross",
                    },
                    {
                        "term": "Spring 2026",
                        "times": "F 10:10am - 12:40pm",
                        "instructor": "Donald Ferguson",
                    },
                ],
            }
        ],
    )

    courses = retrieve_courses(
        index,
        _intent(
            query_type="detail",
            course_codes=["COMS W4111"],
            term="Spring 2026",
            day_preference=["Friday"],
            time_preference="morning",
            instructor="Ferguson",
        ),
        str(courses_dir),
    )

    assert len(courses) == 1
    assert len(courses[0]["sections"]) == 2, "full detail remains available"
    assert len(courses[0]["matched_sections"]) == 1
    matched = courses[0]["matched_sections"][0]
    assert matched["term"] == "Spring 2026"
    assert matched["instructor"] == "Donald Ferguson"


def test_pure_keyword_results_are_not_backfilled(tmp_path: Path) -> None:
    index, courses_dir = _make_index(
        tmp_path,
        [
            {"course_code": "PSAM UN3707", "title": "Persuasion at Scale", "sections": []},
            {"course_code": "CHEN E1000", "title": "Chemical Principles", "sections": []},
            {"course_code": "COMS W1004", "title": "Introduction to Java", "sections": []},
        ],
    )

    courses = retrieve_courses(
        index,
        _intent(keywords=["persuasion"], original_question="persuasion courses"),
        str(courses_dir),
        max_results=5,
    )

    assert [course["course_code"] for course in courses] == ["PSAM UN3707"]


def test_department_only_query_still_uses_quality_sort(tmp_path: Path) -> None:
    index, courses_dir = _make_index(
        tmp_path,
        [
            {"course_code": "COMS E9900", "title": "RESEARCH", "sections": []},
            {
                "course_code": "COMS W1004",
                "title": "PROGRAMMING IN JAVA",
                "description": "A complete introductory programming course.",
                "sections": [{"term": "Spring 2026", "times": "M W 10:10am"}],
            },
        ],
    )

    courses = retrieve_courses(
        index,
        _intent(
            department="COMS",
            department_terms=["computer", "science"],
            keywords=["computer", "science"],
        ),
        str(courses_dir),
    )

    assert [course["course_code"] for course in courses] == ["COMS W1004", "COMS E9900"]


def test_robotics_keyword_can_cross_departments_without_parser_lock(tmp_path: Path) -> None:
    index, courses_dir = _make_index(
        tmp_path,
        [
            {"course_code": "MECE E4602", "title": "INTRODUCTION TO ROBOTICS", "sections": []},
            {
                "course_code": "COMS W4733",
                "title": "COMPUTATIONAL ASPECTS OF ROBOTICS",
                "sections": [],
            },
            {"course_code": "MECE E1008", "title": "THERMODYNAMICS", "sections": []},
        ],
    )

    courses = retrieve_courses(
        index,
        _intent(keywords=["robotics"], department=None),
        str(courses_dir),
        max_results=10,
    )

    assert {course["course_code"] for course in courses} == {
        "MECE E4602",
        "COMS W4733",
    }


def test_day_parser_uses_token_boundaries() -> None:
    assert parse_days_from_times("Savannah") == ([], "")
    assert parse_days_from_times("TBA") == ([], "")
    assert parse_days_from_times("Sa 11:00am - 1:30pm") == (["Saturday"], "morning")


def test_section_points_are_authoritative_and_sectionless_uses_course_range(
    tmp_path: Path,
) -> None:
    index, courses_dir = _make_index(
        tmp_path,
        [
            {
                "course_code": "TEST E1001",
                "title": "MULTI-TERM CREDITS",
                # Deliberately stale: a section-level match must not be rejected
                # by this aggregate before the detail predicate is evaluated.
                "points_min": 4.0,
                "points_max": 4.0,
                "sections": [
                    {
                        "term": "Spring 2026",
                        "times": "M 10:00am - 11:00am",
                        "points": "3.00",
                    },
                    {
                        "term": "Fall 2026",
                        "times": "T 10:00am - 11:00am",
                        "points": "4.00",
                    },
                ],
            },
            {
                "course_code": "TEST E1002",
                "title": "SECTIONLESS VARIABLE CREDIT",
                "points_min": 1.0,
                "points_max": 6.0,
                "sections": [],
            },
            {
                "course_code": "TEST E1003",
                "title": "SECTIONLESS FOUR CREDIT",
                "points_min": 4.0,
                "points_max": 4.0,
                "sections": [],
            },
        ],
    )

    spring_three = retrieve_courses(
        index,
        _intent(
            query_type="detail",
            course_codes=["TEST E1001"],
            term="Spring 2026",
            points_range=[3.0, 3.0],
        ),
        str(courses_dir),
    )
    assert [course["course_code"] for course in spring_three] == ["TEST E1001"]
    assert [section["term"] for section in spring_three[0]["matched_sections"]] == [
        "Spring 2026"
    ]
    assert retrieve_courses(
        index,
        _intent(
            query_type="detail",
            course_codes=["TEST E1001"],
            term="Spring 2026",
            points_range=[4.0, 4.0],
        ),
        str(courses_dir),
    ) == []

    sectionless = retrieve_courses(
        index,
        _intent(
            query_type="detail",
            course_codes=["TEST E1002", "TEST E1003"],
            points_range=[3.0, 3.0],
        ),
        str(courses_dir),
    )
    assert [course["course_code"] for course in sectionless] == ["TEST E1002"]
    assert sectionless[0]["matched_sections"] == []


def _attach_overlay(
    store: SyllabusStore,
    *,
    term: str,
    section_id: str,
    description: str,
    prerequisite: str,
    points: str,
    status: str,
    source: bytes,
) -> dict:
    return store.attach_syllabus(
        course_code="COMS GU4111",
        term=term,
        section_id=section_id,
        payload={
            "title": "DATABASE SYSTEMS",
            "description": description,
            "prerequisites_text": prerequisite,
            "points_min": float(points),
            "points_max": float(points),
            "section": {
                "term": term,
                "section_call_number": section_id,
                "times": "M 10:00am - 11:00am",
                "instructor": "Overlay Instructor",
                "points": points,
                "enrollment_raw": "10/30",
                "enrollment_current": 10,
                "enrollment_capacity": 30,
            },
        },
        source_bytes=source,
        status=status,
        provenance={"seed_course_uids": ["uid-0"], "source_type": "html"},
        evidence={"verified": True},
        quality_score=95,
        quality_issues=[],
    )


def test_published_overlay_is_filter_aware_and_review_is_invisible(
    tmp_path: Path,
) -> None:
    seed_index, courses_dir = _make_index(
        tmp_path,
        [
            {
                "course_uid": "uid-0",
                "course_code": "COMS GU4111",
                "title": "DATABASE SYSTEMS",
                "description": "Seed catalog description.",
                "prerequisites_text": "Seed prerequisite.",
                "sections": [
                    {
                        "term": "Fall 2025",
                        "section_call_number": "001/11111",
                        "times": "T 2:00pm - 3:00pm",
                        "location": "Seed Room",
                        "instructor": "Seed Instructor",
                    }
                ],
            }
        ],
    )
    store = SyllabusStore(tmp_path / "syllabus-store")
    # Attach Spring first and Fall second.  effective_overlays sorts by term
    # (Fall before Spring), so a term/list-order last-wins implementation would
    # incorrectly choose the older Spring course fields for an unfiltered query.
    spring = _attach_overlay(
        store,
        term="Spring 2026",
        section_id="002/22222",
        description="Spring-only overlay description.",
        prerequisite="COMS W3157",
        points="3.00",
        status="published",
        source=b"spring published",
    )
    fall = _attach_overlay(
        store,
        term="Fall 2025",
        section_id="001/11111",
        description="Fall-only overlay description.",
        prerequisite="COMS W3134",
        points="4.00",
        status="published",
        source=b"fall published",
    )
    review = _attach_overlay(
        store,
        term="Summer 2026",
        section_id="003/33333",
        description="Review-only hidden description.",
        prerequisite="COMS W9999",
        points="3.00",
        status="review",
        source=b"summer review",
    )

    runtime = apply_published_overlays(seed_index, store.effective_overlays())
    assert seed_index[0].get("published_syllabus_overlays") is None
    assert set(runtime[0]["syllabus_overlay_versions"]) == {
        fall["version_id"],
        spring["version_id"],
    }
    assert review["version_id"] not in runtime[0]["syllabus_overlay_versions"]
    assert (runtime[0]["points_min"], runtime[0]["points_max"]) == (3.0, 4.0)
    reversed_runtime = apply_published_overlays(
        seed_index, list(reversed(store.effective_overlays()))
    )
    assert (reversed_runtime[0]["points_min"], reversed_runtime[0]["points_max"]) == (
        3.0,
        4.0,
    )

    spring_intent = _intent(
        query_type="detail",
        course_codes=["COMS GU4111"],
        term="Spring 2026",
    )
    spring_courses = retrieve_courses(
        runtime, spring_intent, str(courses_dir), max_results=5
    )
    assert len(spring_courses) == 1
    spring_detail = spring_courses[0]
    assert spring_detail["course_uid"] == "uid-0"
    assert spring_detail["description"] == "Spring-only overlay description."
    assert spring_detail["prerequisites_text"] == "COMS W3157"
    assert [row["section_call_number"] for row in spring_detail["matched_sections"]] == [
        "002/22222"
    ]
    system_prompt, messages = build_answer_prompt(
        spring_intent, spring_courses, "en", max_results=5
    )
    prompt = system_prompt + "\n" + "\n".join(
        message["content"] for message in messages
    )
    assert "Spring-only overlay description." in prompt
    assert "COMS W3157" in prompt
    assert "Fall-only overlay description." not in prompt
    assert "Review-only hidden description." not in prompt

    assert retrieve_courses(
        runtime,
        _intent(
            query_type="detail",
            course_codes=["COMS GU4111"],
            term="Spring 2026",
            points_range=[4.0, 4.0],
        ),
        str(courses_dir),
    ) == []
    assert len(
        retrieve_courses(
            runtime,
            _intent(
                query_type="detail",
                course_codes=["COMS GU4111"],
                term="Spring 2026",
                points_range=[3.0, 3.0],
            ),
            str(courses_dir),
        )
    ) == 1

    fall_courses = retrieve_courses(
        runtime,
        _intent(
            query_type="detail",
            course_codes=["COMS GU4111"],
            term="Fall 2025",
        ),
        str(courses_dir),
    )
    assert fall_courses[0]["description"] == "Fall-only overlay description."
    assert fall_courses[0]["matched_sections"][0]["location"] == "Seed Room"

    assert retrieve_courses(
        runtime,
        _intent(
            query_type="detail",
            course_codes=["COMS GU4111"],
            term="Summer 2026",
        ),
        str(courses_dir),
    ) == []
    unfiltered = retrieve_courses(
        runtime,
        _intent(query_type="detail", course_codes=["COMS GU4111"]),
        str(courses_dir),
    )[0]
    reversed_unfiltered = retrieve_courses(
        reversed_runtime,
        _intent(query_type="detail", course_codes=["COMS GU4111"]),
        str(courses_dir),
    )[0]
    assert unfiltered["description"] == "Fall-only overlay description."
    assert unfiltered["prerequisites_text"] == "COMS W3134"
    assert reversed_unfiltered["description"] == unfiltered["description"]
    assert reversed_unfiltered["prerequisites_text"] == unfiltered[
        "prerequisites_text"
    ]
    assert reversed_unfiltered["sections"] == unfiltered["sections"]
    assert reversed_unfiltered["syllabus_overlay_versions"] == unfiltered[
        "syllabus_overlay_versions"
    ]
    assert "003/33333" not in {
        row.get("section_call_number") for row in unfiltered["sections"]
    }
    assert "Review-only hidden description." not in json.dumps(unfiltered)


@pytest.mark.skipif(not REAL_INDEX_PATH.exists(), reason="real enriched index unavailable")
def test_real_handoff_section_and_keyword_acceptance() -> None:
    index = load_enriched_index(str(REAL_INDEX_PATH))

    for question in (
        "Recommend five computer science courses",
        "Find five computer science courses and list their course codes.",
        "Recomiéndame cinco cursos de ciencias de la computación",
        "Recommandez-moi cinq cours d’informatique",
    ):
        intent = rule_based_extract(normalize_question(question))
        assert intent is not None
        courses = retrieve_courses(
            index, intent, str(REAL_COURSES_DIR), max_results=5
        )
        assert len(courses) == 5
        assert all(
            course["course_code"].startswith("COMS ") for course in courses
        )

    impossible_combo = retrieve_courses(
        index,
        _intent(
            query_type="detail",
            course_codes=["CSEE W4119"],
            day_preference=["Monday"],
            time_preference="morning",
        ),
        str(REAL_COURSES_DIR),
    )
    assert impossible_combo == []

    spring_database = retrieve_courses(
        index,
        _intent(
            query_type="detail",
            course_codes=["COMS W4111"],
            term="Spring 2026",
        ),
        str(REAL_COURSES_DIR),
    )
    assert len(spring_database) == 1
    assert spring_database[0]["matched_sections"]
    assert {
        section.get("term") for section in spring_database[0]["matched_sections"]
    } == {"Spring 2026"}

    persuasion = retrieve_courses(
        index,
        _intent(keywords=["persuasion"]),
        str(REAL_COURSES_DIR),
        max_results=5,
    )
    assert [course["course_code"] for course in persuasion] == ["PSAM UN3707"]

    robotics = retrieve_courses(
        index,
        _intent(keywords=["robotics"], department=None),
        str(REAL_COURSES_DIR),
        max_results=10,
    )
    assert robotics
    assert any(
        not course["course_code"].startswith("MECE ") for course in robotics
    ), "a topical robotics query must not be restricted to MECE"
    assert all("robot" in (course.get("title") or "").lower() for course in robotics)
