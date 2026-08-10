"""Deterministic multilingual conversation-scope parsing.

This module deliberately does not know about FastAPI, retrieval, or model
clients.  It converts a current utterance plus small conversation metadata into
the scope/attribute/operation contract used before retrieval.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re
import unicodedata

from course_codes import extract_course_codes


class Scope(str, Enum):
    PREVIOUS_RESULTS = "previous_results"
    NEW_SEARCH = "new_search"
    CURRENT_COURSE = "current_course"


class Operation(str, Enum):
    DETAIL = "detail"
    COMPARE = "compare"
    ARGMIN = "argmin"
    ARGMAX = "argmax"
    LIST = "list"


class Attribute(str, Enum):
    PREREQUISITES = "prerequisites"
    CREDITS = "credits"
    SCHEDULE = "schedule"
    INSTRUCTOR = "instructor"
    DESCRIPTION = "description"
    ENROLLMENT = "enrollment"
    DIFFICULTY = "difficulty"


@dataclass(frozen=True, slots=True)
class ConversationScope:
    scope: Scope
    attribute: Attribute | None
    operation: Operation
    ordinal: int | None = None
    ordinals: tuple[int, ...] = ()
    uses_focus: bool = False

    def as_dict(self) -> dict:
        return {
            "scope": self.scope.value,
            "attribute": self.attribute.value if self.attribute else None,
            "operation": self.operation.value,
            "ordinal": self.ordinal,
            "ordinals": list(self.ordinals),
            "uses_focus": self.uses_focus,
        }


def _fold(text: str | None) -> str:
    if not isinstance(text, str):
        return ""
    decomposed = unicodedata.normalize("NFKD", text)
    folded = "".join(char for char in decomposed if not unicodedata.combining(char))
    return re.sub(r"\s+", " ", folded.lower()).strip()


_COMPARE_RE = re.compile(
    r"\b(?:compare|comparison|versus|vs|differ|difference|"
    r"comparar|compara|comparalo|comparez)\b|比较|对比|compare-le"
)
_ARGMIN_RE = re.compile(
    r"\b(?:fewest|least|lowest|minimum|shortest|earliest|"
    r"fewer(?:\s+(?:prerequisites?|credits?|points?))?|"
    r"menos|menor|minimo|le\s+moins|plus\s+tot)\b|"
    r"最少|最低|最短|最早|更少"
)
_ARGMAX_RE = re.compile(
    r"\b(?:most|highest|maximum|longest|latest|"
    r"more\s+(?:prerequisites?|credits?|points?)|"
    r"mas|mayor|maximo|le\s+plus|plus\s+tard)\b|"
    r"最多|最高|最长|最晚|更多"
)
_LIST_RE = re.compile(
    r"\b(?:list|recall|summarize|enumerate|lista|listar|resume|liste|lister)\b|"
    r"列出|罗列|回顾"
)

_ATTRIBUTE_PATTERNS: tuple[tuple[Attribute, re.Pattern[str]], ...] = (
    (
        Attribute.PREREQUISITES,
        re.compile(
            r"\b(?:prerequisites?|prereqs?|prerrequisitos?|requisitos?\s+previos?|"
            r"prerequis|prealables?)\b|先修|前置"
        ),
    ),
    (
        Attribute.CREDITS,
        re.compile(
            r"\b(?:credits?|points?|units?|creditos?|puntos?|unidades?)\b|学分"
        ),
    ),
    (
        Attribute.SCHEDULE,
        re.compile(
            r"\b(?:schedule|time|times|when|meet|meets|meeting|horario|cuando|"
            r"reune|horaire|quand|lieu)\b|什么时候|几点|上课|时间"
        ),
    ),
    (
        Attribute.INSTRUCTOR,
        re.compile(
            r"\b(?:instructors?|professors?|teachers?|teach|teaches|taught|"
            r"profesores?|docentes?|ensena|professeurs?|enseignants?|enseigne)\b|"
            r"教授|老师|谁教"
        ),
    ),
    (
        Attribute.ENROLLMENT,
        re.compile(
            r"\b(?:enrollment|capacity|seats?|spots?|inscripcion|cupos?|"
            r"inscription|places?)\b|人数|容量|名额"
        ),
    ),
    (
        Attribute.DESCRIPTION,
        re.compile(
            r"\b(?:description|details?|about|information|descripcion|detalle|"
            r"description|informations?)\b|介绍|内容|详情"
        ),
    ),
    (
        Attribute.DIFFICULTY,
        re.compile(
            r"\b(?:difficulty|difficult|harder|easier|workload|"
            r"dificultad|dificil|facil|carga\s+de\s+trabajo|"
            r"difficulte|difficile|facile|charge\s+de\s+travail)\b|"
            r"难度|更难|更简单|工作量"
        ),
    ),
)

_PLURAL_REFERENCE_RE = re.compile(
    r"\b(?:these|those|them|they|their|among\s+them|among\s+these|"
    r"which\s+of\s+(?:these|those|them)|previous\s+results?|courses?\s+above|"
    r"the\s+(?:five|ones)|estos|estas|esos|esas|ellos|ellas|"
    r"entre\s+estos|de\s+estos|los\s+cinco|anteriores|"
    r"ces\s+cours|ceux|celles|parmi\s+ces|parmi\s+eux|les\s+cinq|precedents?)\b|"
    r"这些(?:课|课程)?|那些(?:课|课程)?|这五门|那五门|这几门|那几门|"
    r"它们|他们|其中|当中|上面(?:的)?|前面(?:的)?"
)
_SINGULAR_REFERENCE_RE = re.compile(
    r"\b(?:it|its|this\s+one|that\s+one|the\s+same|former|latter|"
    r"este\s+curso|esta\s+clase|ese\s+curso|esa\s+clase|el\s+mismo|la\s+misma|"
    r"comparalo|ce\s+cours|ce\s+dernier|celui-ci|celle-ci|a-t-il|il|elle|"
    r"(?:recommend|suggest|pick|choose)\s+(?:me\s+)?(?:one|another)|"
    r"(?:recomienda|sugiere)\s+(?:uno|otro)|"
    r"(?:recommande|suggere)\s+(?:un|autre))\b|"
    r"(?<!它)它(?!们)|这门|那门|该课|再推荐(?:一门)?|compare-le"
)
_WHICH_ONE_RE = re.compile(
    r"\b(?:which\s+(?:one|course)|cual(?:es)?|lequel|laquelle)\b|"
    r"哪一门|哪门|哪一个|哪个"
)

_ORDINAL_WORDS = {
    # English
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
    "tenth": 10,
    # Spanish (accent-folded)
    "primero": 1,
    "primera": 1,
    "primer": 1,
    "segundo": 2,
    "segunda": 2,
    "tercero": 3,
    "tercera": 3,
    "cuarto": 4,
    "cuarta": 4,
    "quinto": 5,
    "quinta": 5,
    "sexto": 6,
    "sexta": 6,
    "septimo": 7,
    "septima": 7,
    "octavo": 8,
    "octava": 8,
    "noveno": 9,
    "novena": 9,
    "decimo": 10,
    "decima": 10,
    # French (accent-folded)
    "premier": 1,
    "premiere": 1,
    "deuxieme": 2,
    "second": 2,
    "seconde": 2,
    "troisieme": 3,
    "quatrieme": 4,
    "cinquieme": 5,
    "sixieme": 6,
    "septieme": 7,
    "huitieme": 8,
    "neuvieme": 9,
    "dixieme": 10,
}
_CHINESE_ORDINALS = {
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
}
_ORDINAL_WORD_RE = re.compile(
    r"\b(?:" + "|".join(sorted(_ORDINAL_WORDS, key=len, reverse=True)) + r")\b"
)
_ORDINAL_NUMBER_RE = re.compile(r"\b(?P<number>\d{1,2})(?:st|nd|rd|th)\b")
_CHINESE_ORDINAL_RE = re.compile(r"第(?P<number>\d{1,2}|[一二两三四五六七八九十])(?:门|个|项)?")


def _extract_ordinals(text: str) -> tuple[int, ...]:
    located: list[tuple[int, int]] = []
    for match in _ORDINAL_WORD_RE.finditer(text):
        located.append((match.start(), _ORDINAL_WORDS[match.group(0)]))
    for match in _ORDINAL_NUMBER_RE.finditer(text):
        located.append((match.start(), int(match.group("number"))))
    for match in _CHINESE_ORDINAL_RE.finditer(text):
        raw = match.group("number")
        value = int(raw) if raw.isdigit() else _CHINESE_ORDINALS[raw]
        located.append((match.start(), value))

    located.sort(key=lambda item: item[0])
    result: list[int] = []
    for _, value in located:
        if value not in result:
            result.append(value)
    return tuple(result)


def _attribute(text: str) -> Attribute | None:
    for attribute, pattern in _ATTRIBUTE_PATTERNS:
        if pattern.search(text):
            return attribute
    return None


def _operation(text: str) -> Operation:
    if _COMPARE_RE.search(text):
        return Operation.COMPARE
    if _ARGMIN_RE.search(text):
        return Operation.ARGMIN
    if _ARGMAX_RE.search(text):
        return Operation.ARGMAX
    if _LIST_RE.search(text):
        return Operation.LIST
    return Operation.DETAIL


def parse_conversation_scope(
    text: str | None,
    *,
    previous_count: int = 0,
    has_current_focus: bool = False,
    new_search_anchor: bool = False,
    force_new_search: bool = False,
) -> ConversationScope:
    """Parse scope before retrieval.

    ``new_search_anchor`` should be set by the caller when the current intent
    contains a possible new department, instructor, or topic.  Explicit
    references to prior results take precedence over that broad signal.  Course
    codes are detected directly here and always start a new search.

    ``force_new_search`` is reserved for caller-resolved cases whose semantics
    are unambiguously a new request even if their wording contains a reference
    (for example, an inherited-context "recommend another" request).  With no
    previous results, a reference phrase remains a new search instead of
    inventing conversation state.
    """
    folded = _fold(text)
    attribute = _attribute(folded)
    operation = _operation(folded)
    ordinals = _extract_ordinals(folded)
    ordinal = ordinals[0] if ordinals else None

    if previous_count <= 0:
        return ConversationScope(
            Scope.NEW_SEARCH, attribute, operation, ordinal, ordinals, False
        )

    plural_reference = bool(_PLURAL_REFERENCE_RE.search(folded))
    singular_reference = bool(_SINGULAR_REFERENCE_RE.search(folded))
    which_one = bool(_WHICH_ONE_RE.search(folded))
    explicit_code = bool(extract_course_codes(text))

    if explicit_code or force_new_search:
        return ConversationScope(
            Scope.NEW_SEARCH, attribute, operation, ordinal, ordinals, False
        )

    uses_focus = has_current_focus and singular_reference

    if operation is Operation.COMPARE and (
        plural_reference or singular_reference or ordinals
    ):
        return ConversationScope(
            Scope.PREVIOUS_RESULTS,
            attribute,
            operation,
            ordinal,
            ordinals,
            uses_focus,
        )

    if operation in (Operation.ARGMIN, Operation.ARGMAX) and (
        plural_reference or singular_reference or which_one or ordinals
    ):
        return ConversationScope(
            Scope.PREVIOUS_RESULTS,
            attribute,
            operation,
            ordinal,
            ordinals,
            uses_focus,
        )

    if ordinals:
        return ConversationScope(
            Scope.CURRENT_COURSE, attribute, operation, ordinal, ordinals, False
        )

    if plural_reference or which_one:
        return ConversationScope(
            Scope.PREVIOUS_RESULTS, attribute, operation, ordinal, ordinals, False
        )

    if singular_reference:
        scope = (
            Scope.CURRENT_COURSE
            if has_current_focus or previous_count == 1
            else Scope.PREVIOUS_RESULTS
        )
        return ConversationScope(
            scope, attribute, operation, ordinal, ordinals, uses_focus
        )

    # Broad model/rule keywords are only a new-search signal after explicit
    # references have been resolved.  This keeps questions such as "Which of
    # these is introductory?" inside the prior result set without weakening
    # the hard course-code/force-new-search cases above.
    if new_search_anchor:
        return ConversationScope(
            Scope.NEW_SEARCH, attribute, operation, ordinal, ordinals, False
        )

    # Elliptical compare/argmin/list requests can still refer to the current
    # conversation when no explicit new topic was extracted.
    if operation is Operation.COMPARE and has_current_focus:
        return ConversationScope(
            Scope.PREVIOUS_RESULTS, attribute, operation, ordinal, ordinals, False
        )
    if operation in (Operation.ARGMIN, Operation.ARGMAX) and previous_count > 1:
        return ConversationScope(
            Scope.PREVIOUS_RESULTS, attribute, operation, ordinal, ordinals, False
        )
    if operation is Operation.LIST:
        return ConversationScope(
            Scope.PREVIOUS_RESULTS, attribute, operation, ordinal, ordinals, False
        )

    # Elliptical attribute follow-ups such as "How many credits?" or
    # "¿Cuándo se reúne?" refer to a selected focus when one exists.
    if attribute is not None and has_current_focus:
        return ConversationScope(
            Scope.CURRENT_COURSE, attribute, operation, ordinal, ordinals, True
        )
    if attribute is not None and previous_count == 1:
        return ConversationScope(
            Scope.CURRENT_COURSE, attribute, operation, ordinal, ordinals, False
        )

    return ConversationScope(
        Scope.NEW_SEARCH, attribute, operation, ordinal, ordinals, False
    )
