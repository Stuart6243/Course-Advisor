"""Structured answer-source contract for chat SSE responses.

The retrieval result (``prompt_basis``) and the courses actually referenced by
the completed answer are deliberately separate.  This module has no FastAPI or
provider dependencies so the same validation can be reused by the server and
by deterministic renderers.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Literal, Sequence


DEFAULT_SOURCE_LABEL = "Columbia Engineering Bulletin 2025–2026"
ANSWER_CITATION_STATUSES = frozenset({"verified", "deterministic"})


class SourceContractError(ValueError):
    """Raised when a source payload cannot be represented without ambiguity."""


def _required_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SourceContractError(f"{field} must be a non-empty string")
    return value.strip()


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _course_uid(course: dict[str, Any]) -> str:
    raw_uid = course.get("course_uid")
    if raw_uid is None:
        # ``uid`` makes the helper usable with an already structured source
        # record, while catalog/detail rows continue to use ``course_uid``.
        raw_uid = course.get("uid")
    return _required_text(raw_uid, "course_uid")


def _offering_rows(course: dict[str, Any]) -> list[dict[str, str | None]]:
    # An explicitly present matched_sections key is authoritative, including
    # when it is empty.  Falling back to all sections in that case could leak a
    # term or schedule that did not match the user's query.
    raw_offerings = (
        course.get("matched_sections")
        if "matched_sections" in course
        else course.get("sections", [])
    )
    if raw_offerings is None:
        raw_offerings = []
    if not isinstance(raw_offerings, list):
        raise SourceContractError("course offerings must be a list")

    offerings: list[dict[str, str | None]] = []
    for section in raw_offerings:
        if not isinstance(section, dict):
            raise SourceContractError("each course offering must be an object")
        offerings.append(
            {
                "term": _optional_text(section.get("term")),
                "section_id": _optional_text(
                    section.get("section_call_number") or section.get("section_id")
                ),
                "meeting_time": _optional_text(section.get("times")),
                "location": _optional_text(section.get("location")),
            }
        )
    return offerings


def _source_label(course: dict[str, Any]) -> str:
    explicit = course.get("source_label")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    return DEFAULT_SOURCE_LABEL


def _candidate_record(
    course: dict[str, Any], position: int
) -> dict[str, Any]:
    return {
        "uid": _course_uid(course),
        "course_code": _required_text(course.get("course_code"), "course_code"),
        "title": _required_text(course.get("title"), "title"),
        "citation_label": f"S{position}",
        "source_label": _source_label(course),
        "role": "prompt_basis",
        "citation_status": "candidate",
        "offerings": _offering_rows(course),
    }


def _prepare_prompt_sources(
    prompt_basis: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    if isinstance(prompt_basis, (str, bytes)) or not isinstance(prompt_basis, Sequence):
        raise SourceContractError("prompt_basis must be a sequence of course objects")

    records: list[dict[str, Any]] = []
    seen_uids: set[str] = set()
    for position, course in enumerate(prompt_basis, start=1):
        if not isinstance(course, dict):
            raise SourceContractError("each prompt-basis course must be an object")
        record = _candidate_record(course, position)
        uid = record["uid"]
        if uid in seen_uids:
            raise SourceContractError(f"prompt_basis contains duplicate UID: {uid}")
        seen_uids.add(uid)
        records.append(record)
    return records


_CITATION_TOKEN_RE = re.compile(
    r"【\s*(S[1-9]\d*)\s*】|\[\s*(S[1-9]\d*)\s*\]", re.IGNORECASE
)


def _course_code_pattern(course_code: str) -> re.Pattern[str]:
    parts = re.split(r"\s+", course_code.strip())
    body = r"\s+".join(re.escape(part) for part in parts)
    # ASCII-aware lookarounds prevent a code from matching inside an identifier
    # such as XCOMS W4111Y while still working next to CJK and punctuation.
    return re.compile(rf"(?<![A-Za-z0-9_]){body}(?![A-Za-z0-9_])", re.IGNORECASE)


def extract_answer_source_uids(
    final_text: str,
    basis: Sequence[dict[str, Any]],
) -> list[str]:
    """Extract reliable answer-source UIDs in first-mention order.

    ``[S2]`` and ``【S2】`` tokens map directly to basis order.  A full course
    code is also accepted when that code belongs to exactly one UID in the
    basis.  Titles and ambiguous duplicate course codes are intentionally not
    treated as citations.
    """

    if not isinstance(final_text, str):
        raise SourceContractError("final_text must be a string")

    records = _prepare_prompt_sources(basis)
    label_to_uid = {
        str(record["citation_label"]).upper(): str(record["uid"])
        for record in records
    }
    occurrences: list[tuple[int, int, str]] = []

    for match in _CITATION_TOKEN_RE.finditer(final_text):
        label = (match.group(1) or match.group(2) or "").upper()
        uid = label_to_uid.get(label)
        if uid is not None:
            occurrences.append((match.start(), match.end(), uid))

    normalized_codes = [
        re.sub(r"\s+", " ", str(record["course_code"]).strip()).upper()
        for record in records
    ]
    code_counts = Counter(normalized_codes)
    for record, normalized_code in zip(records, normalized_codes):
        if code_counts[normalized_code] != 1:
            continue
        pattern = _course_code_pattern(str(record["course_code"]))
        for match in pattern.finditer(final_text):
            occurrences.append((match.start(), match.end(), str(record["uid"])))

    occurrences.sort(key=lambda item: (item[0], item[1]))
    ordered_uids: list[str] = []
    seen: set[str] = set()
    for _start, _end, uid in occurrences:
        if uid in seen:
            continue
        seen.add(uid)
        ordered_uids.append(uid)
    return ordered_uids


_SOURCE_RECORD_FIELDS = frozenset(
    {
        "uid",
        "course_code",
        "title",
        "citation_label",
        "source_label",
        "role",
        "citation_status",
        "offerings",
    }
)
_OFFERING_FIELDS = frozenset(
    {"term", "section_id", "meeting_time", "location"}
)


def _validate_source_record(
    record: object,
    *,
    expected_role: Literal["prompt_basis", "answer_source"],
) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise SourceContractError("each source record must be an object")
    missing = _SOURCE_RECORD_FIELDS.difference(record)
    if missing:
        raise SourceContractError(
            f"source record is missing fields: {', '.join(sorted(missing))}"
        )

    for field in ("uid", "course_code", "title", "citation_label", "source_label"):
        _required_text(record.get(field), field)
    if not re.fullmatch(r"S[1-9]\d*", str(record["citation_label"])):
        raise SourceContractError("citation_label must have the form S1")
    if record.get("role") != expected_role:
        raise SourceContractError(f"source record role must be {expected_role}")

    status = record.get("citation_status")
    if expected_role == "prompt_basis":
        if status != "candidate":
            raise SourceContractError("prompt-basis citation_status must be candidate")
    elif status not in ANSWER_CITATION_STATUSES:
        raise SourceContractError(
            "answer-source citation_status must be verified or deterministic"
        )

    offerings = record.get("offerings")
    if not isinstance(offerings, list):
        raise SourceContractError("offerings must be a list")
    for offering in offerings:
        if not isinstance(offering, dict):
            raise SourceContractError("each offering must be an object")
        missing_offering = _OFFERING_FIELDS.difference(offering)
        if missing_offering:
            raise SourceContractError(
                "offering is missing fields: "
                + ", ".join(sorted(missing_offering))
            )
        if any(
            value is not None
            and (not isinstance(value, str) or not value.strip())
            for value in (offering[field] for field in _OFFERING_FIELDS)
        ):
            raise SourceContractError(
                "offering fields must be non-empty strings or null"
            )
    return record


def validate_sources_event(event: object) -> None:
    """Validate v2 roles, identity mappings, order, and legacy compatibility."""

    if not isinstance(event, dict):
        raise SourceContractError("sources event must be an object")
    if event.get("type") != "sources" or event.get("schema_version") != 2:
        raise SourceContractError("sources event must use schema_version 2")

    courses = event.get("courses")
    answer_sources = event.get("answer_sources")
    prompt_sources = event.get("prompt_basis")
    if not isinstance(courses, list) or not all(
        isinstance(code, str) and code.strip() for code in courses
    ):
        raise SourceContractError("courses must be a list of non-empty strings")
    if not isinstance(answer_sources, list) or not isinstance(prompt_sources, list):
        raise SourceContractError("answer_sources and prompt_basis must be lists")

    prompt_by_uid: dict[str, dict[str, Any]] = {}
    expected_labels: list[str] = []
    for position, raw_record in enumerate(prompt_sources, start=1):
        record = _validate_source_record(raw_record, expected_role="prompt_basis")
        uid = str(record["uid"])
        if uid in prompt_by_uid:
            raise SourceContractError(f"prompt_basis contains duplicate UID: {uid}")
        prompt_by_uid[uid] = record
        expected_labels.append(f"S{position}")
    if [record.get("citation_label") for record in prompt_sources] != expected_labels:
        raise SourceContractError("prompt-basis citation labels must be sequential")

    answer_uids: set[str] = set()
    expected_codes: list[str] = []
    comparable_fields = (
        "uid",
        "course_code",
        "title",
        "citation_label",
        "source_label",
        "offerings",
    )
    for raw_record in answer_sources:
        record = _validate_source_record(raw_record, expected_role="answer_source")
        uid = str(record["uid"])
        if uid in answer_uids:
            raise SourceContractError(f"answer_sources contains duplicate UID: {uid}")
        answer_uids.add(uid)
        candidate = prompt_by_uid.get(uid)
        if candidate is None:
            raise SourceContractError(f"answer source UID is not in prompt_basis: {uid}")
        if any(record[field] != candidate[field] for field in comparable_fields):
            raise SourceContractError(
                f"answer source does not match its prompt-basis record: {uid}"
            )
        expected_codes.append(str(record["course_code"]))

    if courses != expected_codes:
        raise SourceContractError(
            "legacy courses must exactly match ordered answer-source course codes"
        )


def build_sources_event(
    prompt_basis: Sequence[dict[str, Any]],
    answer_uids: Sequence[str],
    citation_status: Literal["verified", "deterministic"],
) -> dict[str, Any]:
    """Build and validate a source-v2 SSE event.

    ``answer_uids`` order is authoritative and may preserve identical course
    codes belonging to different UIDs.  Every answer UID must resolve to one
    exact prompt-basis row.
    """

    if citation_status not in ANSWER_CITATION_STATUSES:
        raise SourceContractError(
            "citation_status must be verified or deterministic"
        )
    if isinstance(answer_uids, (str, bytes)) or not isinstance(answer_uids, Sequence):
        raise SourceContractError("answer_uids must be a sequence")

    prompt_sources = _prepare_prompt_sources(prompt_basis)
    by_uid = {str(record["uid"]): record for record in prompt_sources}
    answer_sources: list[dict[str, Any]] = []
    seen_answer_uids: set[str] = set()
    for raw_uid in answer_uids:
        uid = _required_text(raw_uid, "answer UID")
        if uid in seen_answer_uids:
            raise SourceContractError(f"answer_uids contains duplicate UID: {uid}")
        seen_answer_uids.add(uid)
        candidate = by_uid.get(uid)
        if candidate is None:
            raise SourceContractError(f"answer UID is not in prompt_basis: {uid}")
        answer_sources.append(
            {
                **candidate,
                "role": "answer_source",
                "citation_status": citation_status,
            }
        )

    event = {
        "type": "sources",
        "schema_version": 2,
        "courses": [record["course_code"] for record in answer_sources],
        "answer_sources": answer_sources,
        "prompt_basis": prompt_sources,
    }
    validate_sources_event(event)
    return event


__all__ = [
    "ANSWER_CITATION_STATUSES",
    "DEFAULT_SOURCE_LABEL",
    "SourceContractError",
    "build_sources_event",
    "extract_answer_source_uids",
    "validate_sources_event",
]
