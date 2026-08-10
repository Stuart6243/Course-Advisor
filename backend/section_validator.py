"""Shared validation helpers for catalog and syllabus section records.

The scraper and the syllabus importer both produce the same section-shaped
objects.  Keeping their semantic checks here prevents a valid-looking JSON
object from entering the search index with shifted columns.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable


MAX_CREDITS = 30.0
MAX_ENROLLMENT = 100_000
NON_BLOCKING_CATALOG_WARNINGS = frozenset({"missing_description"})

DAY_NAMES = {
    "m": "Monday",
    "mon": "Monday",
    "monday": "Monday",
    "t": "Tuesday",
    "tu": "Tuesday",
    "tue": "Tuesday",
    "tues": "Tuesday",
    "tuesday": "Tuesday",
    "w": "Wednesday",
    "wed": "Wednesday",
    "wednesday": "Wednesday",
    "th": "Thursday",
    "thu": "Thursday",
    "thur": "Thursday",
    "thurs": "Thursday",
    "thursday": "Thursday",
    "f": "Friday",
    "fri": "Friday",
    "friday": "Friday",
    "sa": "Saturday",
    "sat": "Saturday",
    "saturday": "Saturday",
    # The saved Columbia schedule source uses ``F Sa S`` for a
    # Friday/Saturday/Sunday intensive.  ``Sa`` is already Saturday there, so
    # the standalone ``S`` is unambiguously Sunday for this dataset.
    "s": "Sunday",
    "su": "Sunday",
    "sun": "Sunday",
    "sunday": "Sunday",
}

# The alphabetic look-arounds are essential: without them ``Savannah`` is
# interpreted as ``Sa`` (Saturday), and ``TBA`` as ``T`` (Tuesday).
DAY_TOKEN_RE = re.compile(
    r"(?<![A-Za-z])(?P<day>Monday|Mon|M|Tuesday|Tues|Tue|Tu|T|"
    r"Wednesday|Wed|W|Thursday|Thurs|Thur|Thu|Th|"
    r"Friday|Fri|F|Saturday|Sat|Sa|Sunday|Sun|Su|S)(?![A-Za-z])",
    re.IGNORECASE,
)
TIME_RANGE_RE = re.compile(
    r"\b(?P<start_h>\d{1,2}):(?P<start_m>\d{2})\s*(?P<start_ap>am|pm)?"
    r"\s*(?:-|–|—|to)\s*"
    r"(?P<end_h>\d{1,2}):(?P<end_m>\d{2})\s*(?P<end_ap>am|pm)?\b",
    re.IGNORECASE,
)
ALLOWED_TIME_TEXT_RE = re.compile(
    r"^\s*(?:tba|to be announced|online|online only|arranged|by arrangement|"
    r"by appointment|asynchronous|synchronous)\s*$",
    re.IGNORECASE,
)
POINTS_RE = re.compile(
    r"^\s*(?P<lo>\d+(?:\.\d+)?)"
    r"(?:\s*(?:-|–|—|to)\s*(?P<hi>\d+(?:\.\d+)?))?"
    r"(?:\s*(?:points?|credits?|pts?))?\s*$",
    re.IGNORECASE,
)
ENROLLMENT_PAIR_RE = re.compile(r"^\s*(?P<current>\d+)\s*/\s*(?P<capacity>\d+)\s*$")
ENROLLMENT_SINGLE_RE = re.compile(r"^\s*(?P<current>\d+)\s*$")
TERM_RE = re.compile(r"^(?:Fall|Spring|Summer|Winter)\s+\d{4}$", re.IGNORECASE)


@dataclass(frozen=True)
class SectionValidationResult:
    """Semantic validation outcome for one section."""

    status: str
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    days: tuple[str, ...]
    normalized: dict[str, Any]
    filterable_fields: dict[str, bool]

    @property
    def valid(self) -> bool:
        return not self.errors

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "valid": self.valid,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "days": list(self.days),
            "normalized": dict(self.normalized),
            "filterable_fields": dict(self.filterable_fields),
        }


@dataclass(frozen=True)
class CatalogValidationResult:
    """Course-level seed quality independent of per-section validation."""

    status: str
    warnings: tuple[str, ...]
    blocking_warnings: tuple[str, ...]


def validate_catalog_record(record: Any) -> CatalogValidationResult:
    """Classify legacy catalog warnings without hiding good sibling sections.

    ``missing_description`` lowers answer quality but does not make course
    identity/schedule evidence suspicious.  Section validation issues are
    audited on their individual summaries and never promote the whole course
    to review.  Imported-file provenance and unknown course-level warnings are
    conservative blockers.
    """

    if not isinstance(record, dict):
        return CatalogValidationResult(
            status="review",
            warnings=("catalog_record_not_object",),
            blocking_warnings=("catalog_record_not_object",),
        )

    explicit = record.get("course_review_warnings")
    if isinstance(explicit, list):
        raw_warnings = explicit
    else:
        raw_warnings = [
            item
            for item in (record.get("parse_warnings") or [])
            if item != "enrollment_parse_warning"
            and not str(item).startswith("section_validation:")
        ]
    warnings = _dedupe(
        _text(item) for item in raw_warnings if _text(item)
    )
    if record.get("needs_review") and not warnings:
        warnings = ("legacy_needs_review",)
    blocking = tuple(
        warning
        for warning in warnings
        if warning not in NON_BLOCKING_CATALOG_WARNINGS
    )
    return CatalogValidationResult(
        status="review" if blocking else "published",
        warnings=warnings,
        blocking_warnings=blocking,
    )


def _text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value).replace("\xa0", " ")).strip()


def parse_day_tokens(times: str) -> list[str]:
    """Extract schedule-day tokens without matching inside ordinary words."""

    value = _text(times)
    if not value or ALLOWED_TIME_TEXT_RE.fullmatch(value):
        return []

    days: list[str] = []
    for match in DAY_TOKEN_RE.finditer(value):
        name = DAY_NAMES[match.group("day").lower()]
        if name not in days:
            days.append(name)
    return days


def parse_points_value(value: Any) -> tuple[float, float] | None:
    """Parse a complete points expression; arbitrary embedded numbers fail."""

    match = POINTS_RE.fullmatch(_text(value))
    if not match:
        return None
    low = float(match.group("lo"))
    high = float(match.group("hi") or match.group("lo"))
    if low > high or low < 0 or high > MAX_CREDITS:
        return None
    return low, high


def parse_enrollment_value(value: Any) -> tuple[int | None, int | None] | None:
    """Parse ``current/capacity`` or a source-provided current-only integer."""

    text = _text(value)
    if not text:
        return None, None
    pair = ENROLLMENT_PAIR_RE.fullmatch(text)
    if pair:
        current = int(pair.group("current"))
        capacity = int(pair.group("capacity"))
    else:
        single = ENROLLMENT_SINGLE_RE.fullmatch(text)
        if not single:
            return None
        current = int(single.group("current"))
        capacity = None
    if current > MAX_ENROLLMENT or (capacity is not None and capacity > MAX_ENROLLMENT):
        return None
    return current, capacity


def _strict_nonnegative_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str) and re.fullmatch(r"\d+", value.strip()):
        parsed = int(value.strip())
    else:
        return None
    if parsed < 0 or parsed > MAX_ENROLLMENT:
        return None
    return parsed


def _valid_clock_match(times: str) -> re.Match[str] | None:
    match = TIME_RANGE_RE.search(times)
    if not match:
        return None
    for key in ("start_h", "end_h"):
        hour = int(match.group(key))
        if not 1 <= hour <= 12:
            return None
    for key in ("start_m", "end_m"):
        minute = int(match.group(key))
        if not 0 <= minute <= 59:
            return None
    return match


def contains_valid_clock_range(times: Any) -> bool:
    """Return whether *times* contains a numerically valid clock range."""

    return _valid_clock_match(_text(times)) is not None


def _validate_clock(times: str) -> bool:
    match = _valid_clock_match(times)
    if match is None:
        return False

    # Outside the clock range only day tokens and punctuation/whitespace are
    # permitted.  Checking both sides rejects an instructor name shifted
    # either before or after an otherwise valid schedule.
    prefix = times[: match.start()]
    prefix = DAY_TOKEN_RE.sub(" ", prefix)
    suffix = times[match.end() :]
    return not any(character.isalpha() for character in prefix + suffix)


def _looks_like_non_name(instructor: str) -> bool:
    if not instructor:
        return False
    return bool(
        POINTS_RE.fullmatch(instructor)
        or ENROLLMENT_PAIR_RE.fullmatch(instructor)
        or ENROLLMENT_SINGLE_RE.fullmatch(instructor)
        or TIME_RANGE_RE.search(instructor)
    )


def _dedupe(items: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(items))


def validate_section(section: Any, *, require_identity: bool = False) -> SectionValidationResult:
    """Validate one section and return normalized data plus filter eligibility.

    Empty time and instructor cells are legitimate.  Empty/malformed points are
    not: they are the strongest signal of a shifted schedule row and cannot be
    used for published syllabus search.
    """

    if not isinstance(section, dict):
        return SectionValidationResult(
            status="rejected",
            errors=("section_not_object",),
            warnings=(),
            days=(),
            normalized={},
            filterable_fields={
                "schedule": False,
                "instructor": False,
                "points": False,
                "enrollment": False,
            },
        )

    normalized = {
        "term": _text(section.get("term")),
        "course_number": _text(section.get("course_number")),
        "section_call_number": _text(
            section.get("section_call_number") or section.get("section_id")
        ),
        "times": _text(section.get("times")),
        "location": _text(section.get("location")),
        "instructor": _text(section.get("instructor")),
        "points": _text(section.get("points")),
        "enrollment_raw": _text(section.get("enrollment_raw")),
        "enrollment_current": section.get("enrollment_current"),
        "enrollment_capacity": section.get("enrollment_capacity"),
    }

    errors: list[str] = []
    warnings: list[str] = []

    term = normalized["term"]
    if term and not TERM_RE.fullmatch(term):
        errors.append("invalid_term")
    elif require_identity and not term:
        errors.append("missing_term")

    section_id = normalized["section_call_number"]
    if require_identity and not section_id:
        errors.append("missing_section_id")

    times = normalized["times"]
    schedule_ok = not times or bool(ALLOWED_TIME_TEXT_RE.fullmatch(times)) or _validate_clock(times)
    if not schedule_ok:
        errors.append("invalid_times")

    instructor = normalized["instructor"]
    instructor_ok = not _looks_like_non_name(instructor)
    if not instructor_ok:
        errors.append("invalid_instructor")

    points_parsed = parse_points_value(normalized["points"])
    points_ok = points_parsed is not None
    if not points_ok:
        errors.append("invalid_points")

    raw_enrollment = parse_enrollment_value(normalized["enrollment_raw"])
    if raw_enrollment is None:
        errors.append("invalid_enrollment_raw")
        enrollment_ok = False
    else:
        enrollment_ok = True
        raw_current, raw_capacity = raw_enrollment
        current_value = normalized["enrollment_current"]
        capacity_value = normalized["enrollment_capacity"]
        current = _strict_nonnegative_int(current_value)
        capacity = _strict_nonnegative_int(capacity_value)

        if current_value not in (None, "") and current is None:
            errors.append("invalid_enrollment_current")
            enrollment_ok = False
        if capacity_value not in (None, "") and capacity is None:
            errors.append("invalid_enrollment_capacity")
            enrollment_ok = False
        if raw_current is not None and current is not None and raw_current != current:
            errors.append("enrollment_current_mismatch")
            enrollment_ok = False
        if raw_capacity is not None and capacity is not None and raw_capacity != capacity:
            errors.append("enrollment_capacity_mismatch")
            enrollment_ok = False

        # A section may be intentionally over-enrolled, so this is review
        # metadata rather than evidence of a shifted column.
        effective_current = current if current is not None else raw_current
        effective_capacity = capacity if capacity is not None else raw_capacity
        if (
            effective_current is not None
            and effective_capacity is not None
            and effective_current > effective_capacity
        ):
            warnings.append("enrollment_exceeds_capacity")

        if effective_capacity is None and effective_current is not None:
            warnings.append("enrollment_capacity_unknown")

    errors_tuple = _dedupe(errors)
    warnings_tuple = _dedupe(warnings)
    if errors_tuple:
        status = "review"
    elif warnings_tuple:
        status = "review"
    else:
        status = "published"

    return SectionValidationResult(
        status=status,
        errors=errors_tuple,
        warnings=warnings_tuple,
        days=tuple(parse_day_tokens(times)) if schedule_ok else (),
        normalized=normalized,
        filterable_fields={
            "schedule": schedule_ok,
            "instructor": instructor_ok,
            "points": points_ok,
            "enrollment": enrollment_ok,
        },
    )


def validate_sections(
    sections: Any, *, require_identity: bool = False
) -> list[SectionValidationResult]:
    if not isinstance(sections, list):
        return [validate_section(sections, require_identity=require_identity)]
    return [validate_section(section, require_identity=require_identity) for section in sections]
