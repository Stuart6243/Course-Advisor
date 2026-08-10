"""Deterministic multilingual credit-constraint parsing."""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata


_NUMBER = r"\d+(?:\.\d+)?"
_UNIT = r"(?:credits?|creditos?|points?|puntos?|units?|unidades?|学分)"

_LOWER_PREFIX_RE = re.compile(
    rf"(?:at\s+least|no\s+less\s+than|not\s+less\s+than|"
    rf"minimum(?:\s+of)?|al\s+menos|como\s+minimo|no\s+menos\s+de|"
    rf"au\s+moins|minimum(?:\s+de)?|至少|最少|不低于)\s*(?P<value>{_NUMBER})"
)
_UPPER_PREFIX_RE = re.compile(
    rf"(?:at\s+most|no\s+more\s+than|not\s+more\s+than|"
    rf"maximum(?:\s+of)?|up\s+to|como\s+maximo|a\s+lo\s+sumo|"
    rf"no\s+mas\s+de|au\s+plus|maximum(?:\s+de)?|最多|至多|不超过)"
    rf"\s*(?P<value>{_NUMBER})"
)
_LOWER_SUFFIX_RE = re.compile(
    rf"(?P<value>{_NUMBER})\s*(?:{_UNIT}\s*)?"
    rf"(?:or\s+more|and\s+above|o\s+mas|ou\s+plus|以上|起)"
)
_UPPER_SUFFIX_RE = re.compile(
    rf"(?P<value>{_NUMBER})\s*(?:{_UNIT}\s*)?"
    rf"(?:or\s+less|or\s+fewer|o\s+menos|ou\s+moins|以下|以内)"
)
_BETWEEN_RE = re.compile(
    rf"(?:between|entre)\s*(?P<low>{_NUMBER})\s*(?:and|y|et)\s*"
    rf"(?P<high>{_NUMBER})"
)
_RANGE_RE = re.compile(
    rf"(?P<low>{_NUMBER})\s*(?:-|~|to|through|a|hasta|au|到|至)\s*"
    rf"(?P<high>{_NUMBER})"
)
_EXACT_RE = re.compile(rf"(?P<value>{_NUMBER})\s*{_UNIT}\b")


def _normalize(text: str | None) -> str:
    if not isinstance(text, str):
        return ""
    decomposed = unicodedata.normalize("NFKD", text)
    folded = "".join(char for char in decomposed if not unicodedata.combining(char))
    folded = folded.lower().replace("–", "-").replace("—", "-").replace("−", "-")
    folded = re.sub(r"(?<=\d),(?=\d)", ".", folded)
    return re.sub(r"\s+", " ", folded).strip()


@dataclass(frozen=True, slots=True)
class CreditConstraint:
    minimum: float | None
    maximum: float | None

    def __post_init__(self) -> None:
        if self.minimum is None and self.maximum is None:
            raise ValueError("a credit constraint needs at least one bound")
        if self.minimum is not None and self.minimum < 0:
            raise ValueError("minimum credits cannot be negative")
        if self.maximum is not None and self.maximum < 0:
            raise ValueError("maximum credits cannot be negative")
        if (
            self.minimum is not None
            and self.maximum is not None
            and self.minimum > self.maximum
        ):
            raise ValueError("minimum credits cannot exceed maximum credits")

    def as_list(self) -> list[float | None]:
        return [self.minimum, self.maximum]

    def overlaps(self, course_minimum: float, course_maximum: float) -> bool:
        low, high = sorted((float(course_minimum), float(course_maximum)))
        if self.minimum is not None and high < self.minimum:
            return False
        if self.maximum is not None and low > self.maximum:
            return False
        return True


def _value(match: re.Match[str], group: str = "value") -> float:
    return float(match.group(group))


def parse_credit_constraint(text: str | None) -> CreditConstraint | None:
    """Parse exact, range, and open-bound credit expressions in four languages.

    Supported languages are English, Simplified Chinese, Spanish, and French.
    Accent folding and decimal-comma normalization happen before matching.
    """
    normalized = _normalize(text)
    if not normalized:
        return None

    lower = _LOWER_PREFIX_RE.search(normalized) or _LOWER_SUFFIX_RE.search(normalized)
    upper = _UPPER_PREFIX_RE.search(normalized) or _UPPER_SUFFIX_RE.search(normalized)
    if lower or upper:
        low_value = _value(lower) if lower else None
        high_value = _value(upper) if upper else None
        if low_value is not None and high_value is not None and low_value > high_value:
            return None
        return CreditConstraint(low_value, high_value)

    between = _BETWEEN_RE.search(normalized)
    if between:
        low, high = _value(between, "low"), _value(between, "high")
        return CreditConstraint(min(low, high), max(low, high))

    range_match = _RANGE_RE.search(normalized)
    if range_match:
        low, high = _value(range_match, "low"), _value(range_match, "high")
        return CreditConstraint(min(low, high), max(low, high))

    exact = _EXACT_RE.search(normalized)
    if exact:
        value = _value(exact)
        return CreditConstraint(value, value)

    return None


def parse_points_range(text: str | None) -> list[float | None] | None:
    """Compatibility helper for the query intent ``points_range`` field."""
    constraint = parse_credit_constraint(text)
    return constraint.as_list() if constraint else None
