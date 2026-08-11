from __future__ import annotations

import copy

import pytest

from source_contract import (
    DEFAULT_SOURCE_LABEL,
    SourceContractError,
    build_sources_event,
    extract_answer_source_uids,
    validate_sources_event,
)


def _course(
    number: int,
    *,
    uid: str | None = None,
    code: str | None = None,
    matched_sections: list[dict] | None = None,
) -> dict:
    course = {
        "course_uid": uid or f"uid-{number}",
        "course_code": code or f"TEST E{number:04d}",
        "title": f"Synthetic Course {number}",
        "sections": [
            {
                "term": "Legacy Term",
                "section_call_number": "LEGACY/1",
                "times": "F 1:00pm - 2:00pm",
                "location": "Legacy Room",
            }
        ],
    }
    if matched_sections is not None:
        course["matched_sections"] = matched_sections
    return course


def test_five_candidates_emit_only_two_verified_answer_sources_in_answer_order() -> None:
    basis = [_course(number) for number in range(1001, 1006)]
    final_text = (
        "Start with TEST E1004 because it is the closest match; "
        "TEST E1002 is the second course I used."
    )

    answer_uids = extract_answer_source_uids(final_text, basis)
    event = build_sources_event(basis, answer_uids, "verified")

    assert answer_uids == ["uid-1004", "uid-1002"]
    assert event["type"] == "sources"
    assert event["schema_version"] == 2
    assert event["courses"] == ["TEST E1004", "TEST E1002"]
    assert [record["uid"] for record in event["answer_sources"]] == answer_uids
    assert [record["role"] for record in event["answer_sources"]] == [
        "answer_source",
        "answer_source",
    ]
    assert [record["citation_status"] for record in event["answer_sources"]] == [
        "verified",
        "verified",
    ]
    assert [record["citation_label"] for record in event["prompt_basis"]] == [
        "S1",
        "S2",
        "S3",
        "S4",
        "S5",
    ]
    assert all(record["role"] == "prompt_basis" for record in event["prompt_basis"])
    assert all(
        record["citation_status"] == "candidate"
        for record in event["prompt_basis"]
    )


def test_duplicate_course_code_is_not_a_reliable_code_citation_but_token_is() -> None:
    basis = [
        _course(1, uid="uid-a", code="COMS W4111"),
        _course(2, uid="uid-b", code="COMS W4111"),
    ]

    assert extract_answer_source_uids("Use COMS W4111.", basis) == []
    assert extract_answer_source_uids("Use [S2].", basis) == ["uid-b"]

    event = build_sources_event(basis, ["uid-b", "uid-a"], "deterministic")
    assert event["courses"] == ["COMS W4111", "COMS W4111"]
    assert [record["uid"] for record in event["answer_sources"]] == [
        "uid-b",
        "uid-a",
    ]


def test_square_and_cjk_tokens_map_exact_uids_and_preserve_text_order() -> None:
    basis = [_course(number) for number in range(1001, 1004)]
    text = (
        "先看【S3】，再比较 [s1]；未知 [S99] 不应产生来源，"
        "最后重复 [S3]。"
    )

    assert extract_answer_source_uids(text, basis) == ["uid-1003", "uid-1001"]


def test_exact_course_code_uses_ascii_boundaries_and_flexible_whitespace() -> None:
    basis = [_course(1, uid="uid-coms", code="COMS W4111")]

    assert extract_answer_source_uids("XCOMS W4111Y", basis) == []
    assert extract_answer_source_uids("internal_COMS W4111_identifier", basis) == []
    assert extract_answer_source_uids("课程 COMS\nW4111，很合适。", basis) == ["uid-coms"]


def test_offerings_use_all_matched_sections_and_never_leak_legacy_sections() -> None:
    matched = [
        {
            "term": "Fall 2025",
            "section_call_number": "001/10001",
            "times": "M W 10:10am - 11:25am",
            "location": "Mudd 123",
        },
        {
            "term": "Spring 2026",
            "section_id": "002/20002",
            "times": "",
            "location": None,
        },
    ]
    basis = [_course(1001, matched_sections=matched)]

    event = build_sources_event(basis, ["uid-1001"], "deterministic")
    answer = event["answer_sources"][0]

    assert answer["source_label"] == DEFAULT_SOURCE_LABEL
    assert answer["offerings"] == [
        {
            "term": "Fall 2025",
            "section_id": "001/10001",
            "meeting_time": "M W 10:10am - 11:25am",
            "location": "Mudd 123",
        },
        {
            "term": "Spring 2026",
            "section_id": "002/20002",
            "meeting_time": None,
            "location": None,
        },
    ]
    assert answer["offerings"] == event["prompt_basis"][0]["offerings"]
    assert all(row["term"] != "Legacy Term" for row in answer["offerings"])


def test_sections_are_used_only_for_legacy_rows_without_matched_sections_key() -> None:
    legacy = _course(1001)
    authoritative_empty = _course(1002, matched_sections=[])

    event = build_sources_event(
        [legacy, authoritative_empty],
        ["uid-1001", "uid-1002"],
        "verified",
    )

    assert event["answer_sources"][0]["offerings"][0]["term"] == "Legacy Term"
    assert event["answer_sources"][1]["offerings"] == []


def test_duplicate_offering_identities_are_preserved_without_data_cleanup() -> None:
    repeated_identity = [
        {
            "term": "Fall 2025",
            "section_call_number": "001/13544",
            "times": "M 10:00am - 11:00am",
            "location": "Room A",
        },
        {
            "term": "Fall 2025",
            "section_call_number": "001/13544",
            "times": "W 10:00am - 11:00am",
            "location": "Room B",
        },
    ]

    event = build_sources_event(
        [_course(1001, matched_sections=repeated_identity)],
        ["uid-1001"],
        "deterministic",
    )

    offerings = event["answer_sources"][0]["offerings"]
    assert len(offerings) == 2
    assert [offering["meeting_time"] for offering in offerings] == [
        "M 10:00am - 11:00am",
        "W 10:00am - 11:00am",
    ]


@pytest.mark.parametrize(
    "basis,answer_uids,status,match",
    [
        ([{"course_code": "COMS W4111", "title": "Databases"}], [], "verified", "course_uid"),
        ([_course(1), _course(2, uid="uid-1")], [], "verified", "duplicate UID"),
        ([_course(1)], ["missing-uid"], "verified", "not in prompt_basis"),
        ([_course(1)], ["uid-1", "uid-1"], "verified", "duplicate UID"),
        ([_course(1)], [], "candidate", "verified or deterministic"),
    ],
)
def test_build_rejects_ambiguous_or_invalid_identity_contracts(
    basis: list[dict],
    answer_uids: list[str],
    status: str,
    match: str,
) -> None:
    with pytest.raises(SourceContractError, match=match):
        build_sources_event(basis, answer_uids, status)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "mutate,match",
    [
        (
            lambda event: event.update(courses=["WRONG E0000"]),
            "legacy courses",
        ),
        (
            lambda event: event["answer_sources"][0].update(uid="unknown-uid"),
            "not in prompt_basis",
        ),
        (
            lambda event: event["answer_sources"][0].update(title="Wrong title"),
            "does not match",
        ),
        (
            lambda event: event["prompt_basis"][0].update(
                citation_status="verified"
            ),
            "must be candidate",
        ),
        (
            lambda event: event["prompt_basis"][0].update(citation_label="S2"),
            "sequential",
        ),
    ],
)
def test_validate_sources_event_rejects_inconsistent_payloads(mutate, match: str) -> None:
    valid = build_sources_event([_course(1)], ["uid-1"], "verified")
    invalid = copy.deepcopy(valid)
    mutate(invalid)

    with pytest.raises(SourceContractError, match=match):
        validate_sources_event(invalid)
