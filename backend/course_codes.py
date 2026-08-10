"""Canonical Columbia-style course-code parsing.

The catalog currently contains the single-letter levels B/C/E/W and the
two-letter levels UN/GU/GR.  The parser accepts any single alphabetic level for
backward compatibility with prerequisite references such as ``MATH V2010``,
but only the known two-letter levels are accepted.  All consumers should use
this module so query, history, import, and prerequisite parsing cannot drift.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterator


CURRENT_SINGLE_LETTER_LEVELS = frozenset({"B", "C", "E", "W"})
CURRENT_DOUBLE_LETTER_LEVELS = frozenset({"UN", "GU", "GR"})

DEPARTMENT_PATTERN = r"[A-Z]{2,4}"
LEVEL_PATTERN = r"(?:UN|GU|GR|[A-Z])"
COURSE_CODE_BODY_PATTERN = (
    rf"(?:{DEPARTMENT_PATTERN}[\s_-]+{LEVEL_PATTERN}[\s_-]*\d{{4}}|"
    rf"{DEPARTMENT_PATTERN}{LEVEL_PATTERN}\d{{4}})"
)

_SEPARATED_CODE_RE = re.compile(
    rf"^\s*(?P<department>{DEPARTMENT_PATTERN})[\s_-]+"
    rf"(?P<level>{LEVEL_PATTERN})[\s_-]*(?P<number>\d{{4}})\s*$",
    re.IGNORECASE,
)
_COMPACT_CODE_RE = re.compile(
    r"^\s*(?P<prefix>[A-Z]{3,6})(?P<number>\d{4})\s*$", re.IGNORECASE
)
_SEARCH_CODE_RE = re.compile(
    rf"(?<![A-Z0-9])(?P<code>{COURSE_CODE_BODY_PATTERN})(?![A-Z0-9])",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class CourseCode:
    department: str
    level: str
    number: str

    @property
    def canonical(self) -> str:
        return f"{self.department} {self.level}{self.number}"

    @property
    def is_current_level(self) -> bool:
        return self.level in CURRENT_SINGLE_LETTER_LEVELS | CURRENT_DOUBLE_LETTER_LEVELS

    def __str__(self) -> str:
        return self.canonical


@dataclass(frozen=True, slots=True)
class CourseCodeMatch:
    code: CourseCode
    start: int
    end: int


def parse_course_code(value: str | None) -> CourseCode | None:
    """Parse one complete code, returning ``None`` for malformed input."""
    if not isinstance(value, str) or not value.strip():
        return None
    separated = _SEPARATED_CODE_RE.fullmatch(value)
    if separated:
        return CourseCode(
            department=separated.group("department").upper(),
            level=separated.group("level").upper(),
            number=separated.group("number"),
        )

    compact = _COMPACT_CODE_RE.fullmatch(value)
    if not compact:
        return None

    prefix = compact.group("prefix").upper()
    for level in sorted(CURRENT_DOUBLE_LETTER_LEVELS, key=len, reverse=True):
        if prefix.endswith(level) and 2 <= len(prefix) - len(level) <= 4:
            return CourseCode(prefix[: -len(level)], level, compact.group("number"))

    department, level = prefix[:-1], prefix[-1:]
    if not (2 <= len(department) <= 4):
        return None
    return CourseCode(department, level, compact.group("number"))


def normalize_course_code(value: str | None) -> str | None:
    """Return ``DEPT LEVEL1234`` or ``None`` when parsing fails."""
    parsed = parse_course_code(value)
    return parsed.canonical if parsed else None


def is_valid_course_code(value: str | None) -> bool:
    return parse_course_code(value) is not None


def iter_course_code_matches(text: str | None) -> Iterator[CourseCodeMatch]:
    """Yield canonical course codes and source spans from arbitrary text."""
    if not isinstance(text, str) or not text:
        return
    for match in _SEARCH_CODE_RE.finditer(text):
        parsed = parse_course_code(match.group("code"))
        if parsed is not None:
            yield CourseCodeMatch(parsed, match.start("code"), match.end("code"))


def extract_course_codes(text: str | None, *, dedupe: bool = True) -> list[str]:
    """Extract canonical codes in source order."""
    result: list[str] = []
    seen: set[str] = set()
    for match in iter_course_code_matches(text):
        canonical = match.code.canonical
        if dedupe and canonical in seen:
            continue
        result.append(canonical)
        seen.add(canonical)
    return result
