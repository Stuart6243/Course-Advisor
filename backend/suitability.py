"""Deterministic, evidence-backed beginner suitability assessment.

Suitability is deliberately separate from topical relevance.  Words such as
``Introduction`` or ``Fundamentals`` in a title do not prove that a student can
take a course with no background, and an empty prerequisite field remains
unknown rather than being converted to "none".
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re
import unicodedata
from typing import Mapping

from prerequisites import PrerequisiteStatus, parse_prerequisites


class SuitabilityStatus(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    UNKNOWN = "unknown"


class SuitabilityReason(str, Enum):
    EXPLICIT_NO_PRIOR_BACKGROUND = "explicit_no_prior_background"
    EXPLICIT_NO_PREREQUISITES = "explicit_no_prerequisites"
    RELIABLE_INTRODUCTORY_EVIDENCE = "reliable_introductory_evidence"
    LISTED_PREREQUISITES = "listed_prerequisites"
    ADVANCED_LEVEL = "advanced_level"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class BeginnerSuitability:
    status: SuitabilityStatus
    reason: SuitabilityReason
    evidence_text: str
    prerequisite_status: PrerequisiteStatus
    priority: int

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "reason": self.reason.value,
            "evidence_text": self.evidence_text,
            "prerequisite_status": self.prerequisite_status.value,
            "priority": self.priority,
        }


def _fold(text: object) -> str:
    if not isinstance(text, str):
        return ""
    decomposed = unicodedata.normalize("NFKD", text)
    folded = "".join(
        character
        for character in decomposed
        if not unicodedata.combining(character)
    )
    return re.sub(r"\s+", " ", folded).strip()


_NO_PRIOR_BACKGROUND_RE = re.compile(
    r"\b(?:assumes?|requires?|expects?)\s+no\s+(?:prior|previous)\s+"
    r"(?:(?:programming|computing|technical|subject[- ]matter)\s+)?"
    r"(?:background|experience|knowledge)\b|"
    r"\bno\s+(?:prior|previous)\s+"
    r"(?:(?:programming|computing|technical|subject[- ]matter)\s+)?"
    r"(?:background|experience|knowledge)\s+(?:is\s+)?required\b|"
    r"\bwithout\s+(?:prior|previous)\s+"
    r"(?:(?:programming|computing|technical|subject[- ]matter)\s+)?"
    r"(?:background|experience|knowledge)\b",
    re.IGNORECASE,
)

_RELIABLE_INTRO_DESCRIPTION_RE = re.compile(
    # These phrases make an explicit audience/level claim in the description.
    # A generic "introduction to X" or "introductory concepts" is deliberately
    # excluded because upper-level courses commonly use that wording while
    # still listing prerequisites.
    r"\b(?:intended|designed|suitable|appropriate)\s+for\s+"
    r"(?:(?:complete|absolute)\s+)?(?:beginners?|novices?|non[- ]?majors?)\b|"
    r"\bbasic\s+introduction\s+to\b|"
    r"\bintroductory\s+course\s+(?:intended|designed)\s+for\s+"
    r"(?:(?:complete|absolute)\s+)?(?:beginners?|novices?|non[- ]?majors?)\b",
    re.IGNORECASE,
)

_ADVANCED_TITLE_RE = re.compile(r"\badvanced\b", re.IGNORECASE)
_COURSE_LEVEL_RE = re.compile(r"(\d{4})\s*$")


def _course_level(course_code: object) -> int | None:
    match = _COURSE_LEVEL_RE.search(str(course_code or "").strip())
    return int(match.group(1)) if match else None


def _sentence_containing(text: str, match: re.Match[str]) -> str:
    """Return a compact, human-auditable sentence around one evidence match."""

    left = max(text.rfind(".", 0, match.start()), text.rfind("。", 0, match.start()))
    right_candidates = [
        position
        for position in (text.find(".", match.end()), text.find("。", match.end()))
        if position >= 0
    ]
    right = min(right_candidates) + 1 if right_candidates else len(text)
    excerpt = re.sub(r"\s+", " ", text[left + 1 : right]).strip()
    return excerpt[:500]


def assess_beginner_suitability(
    course: Mapping[str, object],
) -> BeginnerSuitability:
    """Assess one complete course detail using deterministic source evidence.

    Only narrow, explicit introductory-description signals qualify, and they
    take the documented priority over counter-evidence.  A title containing
    ``Introduction`` or ``Fundamentals`` is never positive evidence by itself.
    """

    raw_description = _fold(course.get("description"))
    raw_title = _fold(course.get("title"))
    raw_prerequisites = course.get("prerequisites_text")
    prerequisite = parse_prerequisites(
        raw_prerequisites if isinstance(raw_prerequisites, str) else None
    )

    no_prior_match = _NO_PRIOR_BACKGROUND_RE.search(raw_description)
    if no_prior_match:
        return BeginnerSuitability(
            SuitabilityStatus.POSITIVE,
            SuitabilityReason.EXPLICIT_NO_PRIOR_BACKGROUND,
            _sentence_containing(raw_description, no_prior_match),
            prerequisite.status,
            0,
        )

    if prerequisite.status is PrerequisiteStatus.EXPLICIT_NONE:
        return BeginnerSuitability(
            SuitabilityStatus.POSITIVE,
            SuitabilityReason.EXPLICIT_NO_PREREQUISITES,
            prerequisite.full_text.strip(),
            prerequisite.status,
            1,
        )

    # Only the narrow, explicit description patterns above qualify at this
    # priority.  They intentionally precede level/prerequisite counter-evidence
    # to implement the published product ordering; generic introductory wording
    # never reaches this branch.
    intro_match = _RELIABLE_INTRO_DESCRIPTION_RE.search(raw_description)
    if intro_match:
        return BeginnerSuitability(
            SuitabilityStatus.POSITIVE,
            SuitabilityReason.RELIABLE_INTRODUCTORY_EVIDENCE,
            _sentence_containing(raw_description, intro_match),
            prerequisite.status,
            1,
        )

    if prerequisite.status is PrerequisiteStatus.LISTED:
        return BeginnerSuitability(
            SuitabilityStatus.NEGATIVE,
            SuitabilityReason.LISTED_PREREQUISITES,
            prerequisite.full_text.strip(),
            prerequisite.status,
            2,
        )

    level = _course_level(course.get("course_code"))
    if _ADVANCED_TITLE_RE.search(raw_title) or (level is not None and level >= 3000):
        evidence = raw_title or str(course.get("course_code") or "").strip()
        return BeginnerSuitability(
            SuitabilityStatus.NEGATIVE,
            SuitabilityReason.ADVANCED_LEVEL,
            evidence,
            prerequisite.status,
            2,
        )

    return BeginnerSuitability(
        SuitabilityStatus.UNKNOWN,
        SuitabilityReason.UNKNOWN,
        "",
        prerequisite.status,
        3,
    )
