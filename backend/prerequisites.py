"""Structured prerequisite parsing and deterministic comparison.

The catalog's empty prerequisite fields mean "not collected", not "none".
This module therefore keeps evidence status separate from the minimum number
of required courses and never assigns a numeric value to unknown prose.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re
from typing import Iterable, Mapping, Sequence
import unicodedata

from course_codes import COURSE_CODE_BODY_PATTERN, extract_course_codes


class PrerequisiteStatus(str, Enum):
    EXPLICIT_NONE = "explicit_none"
    LISTED = "listed"
    UNKNOWN = "unknown"


class PrerequisiteRelationship(str, Enum):
    AND = "AND"
    OR = "OR"
    MIXED = "MIXED"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class PrerequisiteInfo:
    status: PrerequisiteStatus
    required_codes: tuple[str, ...]
    relationship: PrerequisiteRelationship
    recommended_only: bool
    full_text: str
    minimum_required_count: int | None

    def as_dict(self) -> dict:
        return {
            "status": self.status.value,
            "required_codes": list(self.required_codes),
            "relationship": self.relationship.value,
            "recommended_only": self.recommended_only,
            "full_text": self.full_text,
            "minimum_required_count": self.minimum_required_count,
        }


@dataclass(frozen=True, slots=True)
class PrerequisiteComparison:
    operation: str
    winners: tuple[str, ...]
    winning_count: int | None
    tied: bool
    excluded_unknown: tuple[str, ...]

    def as_dict(self) -> dict:
        return {
            "operation": self.operation,
            "winners": list(self.winners),
            "winning_count": self.winning_count,
            "tied": self.tied,
            "excluded_unknown": list(self.excluded_unknown),
        }


def _fold(text: str | None) -> str:
    if not isinstance(text, str):
        return ""
    decomposed = unicodedata.normalize("NFKD", text)
    folded = "".join(char for char in decomposed if not unicodedata.combining(char))
    return re.sub(r"\s+", " ", folded.lower()).strip()


# These phrases state that the evidence is missing.  They must be checked
# before explicit-none phrases because "no prerequisites listed" contains
# both "no prerequisites" and "listed".
_UNKNOWN_RE = re.compile(
    r"\b(?:unknown|not\s+(?:listed|available|provided|specified)|"
    r"no\s+(?:prerequisite|prerequisites|prereq|prereqs)\s+(?:listed|provided)|"
    r"to\s+be\s+(?:announced|determined)|tba|tbd|n/?a|"
    r"no\s+(?:consta|indicado|indicados|disponible)|"
    r"non\s+(?:indique|indiques|renseigne|renseignes|disponible))\b|"
    r"未列出|未提供|未说明|未知|待定|暂无信息"
)

_EXPLICIT_NONE_RE = re.compile(
    r"^(?:none|no\s+(?:prerequisites?|prereqs?)(?:\s+(?:are\s+)?required)?|"
    r"(?:there\s+are\s+)?no\s+(?:prerequisites?|prereqs?)|"
    r"without\s+(?:prerequisites?|prereqs?)|"
    r"sin\s+(?:prerrequisitos?|requisitos?\s+previos?)|"
    r"no\s+se\s+requieren?\s+(?:prerrequisitos?|requisitos?\s+previos?)|"
    r"aucun(?:e)?\s+(?:prerequis|prealable)s?|"
    r"pas\s+de\s+(?:prerequis|prealable)s?|"
    r"(?:无|没有|无需)(?:任何)?(?:先修|前置)(?:课|课程)?(?:要求)?|"
    r"不需要(?:任何)?(?:先修|前置)(?:课|课程)?)"
    r"[.!。！;；\s]*$"
)

_RECOMMENDED_RE = re.compile(
    r"\b(?:recommended|recommendation|advised|suggested|"
    r"recomendad[oa]s?|se\s+recomienda|aconsejad[oa]s?|"
    r"recommandees?|recommande|conseillees?|conseille)\b|建议|推荐"
)
_MANDATORY_RE = re.compile(
    r"\b(?:required|requires?|must|mandatory|need(?:ed|s)?|"
    r"obligatori[oa]s?|se\s+requiere|debe(?:n)?|"
    r"obligatoires?|exigees?|requis(?:e|es|s)?|doit|doivent)\b|"
    r"必须|要求|需先|须先"
)

_EXPRESSION_TOKEN_RE = re.compile(
    rf"(?P<code>{COURSE_CODE_BODY_PATTERN})|"
    rf"(?P<and>\b(?:and|y|et)\b|并且|以及|和|与|及|且)|"
    rf"(?P<or>\b(?:or|o|ou)\b|或者|或)|"
    rf"(?P<lparen>\()|(?P<rparen>\))|(?P<comma>[,，;；])",
    re.IGNORECASE,
)


def _expression_tokens(text: str) -> tuple[list[str], set[str]]:
    """Return the course/operator expression and its explicit operators."""
    raw_tokens: list[str] = []
    code_positions: list[int] = []
    explicit_operators: set[str] = set()

    for match in _EXPRESSION_TOKEN_RE.finditer(_fold(text)):
        kind = match.lastgroup
        if kind == "code":
            code_positions.append(len(raw_tokens))
            raw_tokens.append("COURSE")
        elif kind == "and":
            explicit_operators.add("AND")
            raw_tokens.append("AND")
        elif kind == "or":
            explicit_operators.add("OR")
            raw_tokens.append("OR")
        elif kind == "lparen":
            raw_tokens.append("(")
        elif kind == "rparen":
            raw_tokens.append(")")
        else:
            raw_tokens.append("COMMA")

    if not code_positions:
        return [], explicit_operators

    start = code_positions[0]
    while start > 0 and raw_tokens[start - 1] == "(":
        start -= 1
    end = code_positions[-1] + 1
    while end < len(raw_tokens) and raw_tokens[end] == ")":
        end += 1
    tokens = raw_tokens[start:end]

    # Oxford-style lists inherit their final explicit conjunction.  A comma-
    # only list remains unknown rather than being guessed as AND.
    if len(explicit_operators) == 1:
        inherited = next(iter(explicit_operators))
        tokens = [inherited if token == "COMMA" else token for token in tokens]
        tokens = [
            token
            for index, token in enumerate(tokens)
            if not (
                index > 0
                and token == inherited
                and tokens[index - 1] == inherited
            )
        ]
    return tokens, explicit_operators


class _MinimumExpressionParser:
    """Evaluate minimum required leaves with AND taking precedence over OR."""

    def __init__(self, tokens: Sequence[str]) -> None:
        self.tokens = tokens
        self.position = 0

    def parse(self) -> int | None:
        value = self._or_expression()
        if value is None or self.position != len(self.tokens):
            return None
        return value

    def _peek(self) -> str | None:
        if self.position >= len(self.tokens):
            return None
        return self.tokens[self.position]

    def _take(self, expected: str) -> bool:
        if self._peek() != expected:
            return False
        self.position += 1
        return True

    def _or_expression(self) -> int | None:
        value = self._and_expression()
        if value is None:
            return None
        while self._take("OR"):
            right = self._and_expression()
            if right is None:
                return None
            value = min(value, right)
        return value

    def _and_expression(self) -> int | None:
        value = self._atom()
        if value is None:
            return None
        while self._take("AND"):
            right = self._atom()
            if right is None:
                return None
            value += right
        return value

    def _atom(self) -> int | None:
        if self._take("COURSE"):
            return 1
        if self._take("("):
            value = self._or_expression()
            if value is None or not self._take(")"):
                return None
            return value
        return None


def _relationship(
    codes: Sequence[str], explicit_operators: set[str]
) -> PrerequisiteRelationship:
    if not codes:
        return PrerequisiteRelationship.UNKNOWN
    if explicit_operators == {"AND", "OR"}:
        return PrerequisiteRelationship.MIXED
    if explicit_operators == {"OR"}:
        return PrerequisiteRelationship.OR
    if explicit_operators == {"AND"} or len(codes) == 1:
        return PrerequisiteRelationship.AND
    return PrerequisiteRelationship.UNKNOWN


def parse_prerequisites(text: str | None) -> PrerequisiteInfo:
    """Parse prerequisite evidence without turning missing data into none."""
    full_text = text if isinstance(text, str) else ""
    normalized = _fold(full_text)

    if not normalized or _UNKNOWN_RE.search(normalized):
        return PrerequisiteInfo(
            PrerequisiteStatus.UNKNOWN,
            (),
            PrerequisiteRelationship.UNKNOWN,
            False,
            full_text,
            None,
        )

    if _EXPLICIT_NONE_RE.fullmatch(normalized):
        return PrerequisiteInfo(
            PrerequisiteStatus.EXPLICIT_NONE,
            (),
            PrerequisiteRelationship.UNKNOWN,
            False,
            full_text,
            0,
        )

    codes = tuple(extract_course_codes(full_text))
    tokens, explicit_operators = _expression_tokens(full_text)
    relationship = _relationship(codes, explicit_operators)
    recommended_only = bool(_RECOMMENDED_RE.search(normalized)) and not bool(
        _MANDATORY_RE.search(normalized)
    )

    minimum: int | None
    if recommended_only:
        minimum = 0
    elif relationship is PrerequisiteRelationship.AND and len(codes) == 1:
        minimum = 1
    elif relationship is PrerequisiteRelationship.UNKNOWN:
        minimum = None
    else:
        minimum = _MinimumExpressionParser(tokens).parse()

    return PrerequisiteInfo(
        PrerequisiteStatus.LISTED,
        codes,
        relationship,
        recommended_only,
        full_text,
        minimum,
    )


def _normalize_operation(operation: str) -> str:
    normalized = operation.strip().lower()
    if normalized in {"argmin", "min", "minimum", "fewest", "least"}:
        return "argmin"
    if normalized in {"argmax", "max", "maximum", "most"}:
        return "argmax"
    raise ValueError("operation must be argmin or argmax")


def compare_prerequisite_counts(
    candidates: Iterable[tuple[str, PrerequisiteInfo]],
    *,
    operation: str,
) -> PrerequisiteComparison:
    """Select all deterministic minima/maxima, preserving candidate order.

    Candidates whose minimum count cannot be proven are reported in
    ``excluded_unknown`` and never participate as zero.
    """
    normalized_operation = _normalize_operation(operation)
    known: list[tuple[str, int]] = []
    excluded: list[str] = []
    for identifier, prerequisite in candidates:
        count = prerequisite.minimum_required_count
        if count is None:
            excluded.append(identifier)
        else:
            known.append((identifier, count))

    if not known:
        return PrerequisiteComparison(
            normalized_operation, (), None, False, tuple(excluded)
        )

    selector = min if normalized_operation == "argmin" else max
    winning_count = selector(count for _, count in known)
    winners = tuple(
        identifier for identifier, count in known if count == winning_count
    )
    return PrerequisiteComparison(
        normalized_operation,
        winners,
        winning_count,
        len(winners) > 1,
        tuple(excluded),
    )


def compare_course_prerequisites(
    courses: Sequence[Mapping[str, object]],
    *,
    operation: str,
    identity_key: str = "course_uid",
    code_key: str = "course_code",
    text_key: str = "prerequisites_text",
) -> PrerequisiteComparison:
    """Convenience adapter for the course dictionaries used by the server.

    Runtime records are identified by ``course_uid`` when available.  A course
    code is only a display label and is not unique in the catalog, so using it
    as the comparison key could focus the wrong record after an argmin/argmax
    operation.  Synthetic/legacy rows without a UID continue to fall back to
    ``course_code`` and finally their input position.
    """
    candidates: list[tuple[str, PrerequisiteInfo]] = []
    for position, course in enumerate(courses, start=1):
        raw_identifier = course.get(identity_key) or course.get(code_key)
        identifier = (
            str(raw_identifier).strip() if raw_identifier is not None else ""
        ) or f"course_{position}"
        raw_text = course.get(text_key)
        candidates.append(
            (identifier, parse_prerequisites(raw_text if isinstance(raw_text, str) else None))
        )
    return compare_prerequisite_counts(candidates, operation=operation)
