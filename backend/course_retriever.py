"""
两阶段课程检索模块。
Stage 1: 结构化字段粗筛（程序化，不调用 LLM）
Stage 2: searchable_text 关键词精筛（评分排序）
最后加载匹配课程的完整 JSON。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from course_index import (
    filter_by_fields,
    has_points_filters,
    has_schedule_filters,
    has_section_filters,
    load_course_detail,
    matching_sections,
    points_range_matches,
    search_by_keywords,
    section_matches_filters,
    sort_by_quality,
)
from section_validator import validate_section


COURSE_OVERLAY_FIELDS = frozenset(
    {
        "description",
        "prerequisites_text",
        "notes_text",
        "points_raw",
        "points_min",
        "points_max",
    }
)
SECTION_OVERLAY_FIELDS = frozenset(
    {
        "term",
        "catalog_ref",
        "course_number",
        "section_call_number",
        "section_id",
        "times",
        "location",
        "instructor",
        "points",
        "enrollment_raw",
        "enrollment_current",
        "enrollment_capacity",
    }
)


def _section_identity(section: dict[str, Any]) -> tuple[str, str]:
    term = str(section.get("term") or "").strip().casefold()
    section_id = str(
        section.get("section_call_number") or section.get("section_id") or ""
    ).strip().casefold()
    return term, section_id


def _overlay_version_key(overlay: dict[str, Any]) -> tuple[int, int, str]:
    """Return an explicit, input-order-independent overlay recency key."""

    def numeric(value: object) -> int:
        if isinstance(value, bool):
            return -1
        try:
            return int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return -1

    return (
        numeric(overlay.get("created_at_ns")),
        numeric(overlay.get("revision")),
        str(overlay.get("version_id") or ""),
    )


def _merge_published_overlays(detail: dict, entry: dict, filters: dict) -> dict:
    """Merge allowlisted published payload fields into a detail copy."""

    merged = dict(detail)
    sections = [
        dict(section)
        for section in (detail.get("sections") or [])
        if isinstance(section, dict)
    ]
    applied_versions: list[str] = []
    applied_provenance: list[dict[str, Any]] = []
    prepared: list[tuple[dict, dict, dict, tuple[str, str]]] = []
    for overlay in entry.get("published_syllabus_overlays") or []:
        if not isinstance(overlay, dict) or overlay.get("status") != "published":
            continue
        payload = overlay.get("payload")
        if not isinstance(payload, dict):
            continue
        overlay_section = payload.get("section")
        if not isinstance(overlay_section, dict):
            continue
        sanitized_section = {
            key: value
            for key, value in overlay_section.items()
            if key in SECTION_OVERLAY_FIELDS
        }
        sanitized_section["term"] = str(
            overlay.get("term") or sanitized_section.get("term") or ""
        ).strip()
        sanitized_section["section_call_number"] = str(
            overlay.get("section_id")
            or sanitized_section.get("section_call_number")
            or sanitized_section.get("section_id")
            or ""
        ).strip()
        sanitized_section.pop("section_id", None)
        identity = _section_identity(sanitized_section)
        if not all(identity):
            continue
        prepared.append((overlay, payload, sanitized_section, identity))

    # Canonical ordering makes section replacement, provenance, and returned
    # section order independent of the store/list iteration order.  If a
    # defensive caller supplies multiple published versions for one identity,
    # the explicit latest version is applied last.
    prepared.sort(key=lambda item: (item[3], _overlay_version_key(item[0])))
    eligible_course_overlays: list[tuple[dict, dict]] = []
    for overlay, payload, sanitized_section, identity in prepared:
        for position, existing in enumerate(sections):
            if _section_identity(existing) == identity:
                existing_allowlisted = {
                    key: value
                    for key, value in existing.items()
                    if key in SECTION_OVERLAY_FIELDS
                }
                sections[position] = {**existing_allowlisted, **sanitized_section}
                sanitized_section = sections[position]
                break
        else:
            sections.append(sanitized_section)
        if not has_section_filters(filters) or section_matches_filters(
            sanitized_section, filters
        ):
            eligible_course_overlays.append((overlay, payload))
        version_id = str(overlay.get("version_id") or "").strip()
        if version_id:
            applied_versions.append(version_id)
        provenance = overlay.get("provenance")
        if isinstance(provenance, dict):
            applied_provenance.append(dict(provenance))

    # Description/prerequisite/course-credit fields describe one syllabus
    # version and therefore must come from one deterministic winner.  Applying
    # each overlay in term/list order made an unfiltered code/topic query use
    # whichever term happened to sort last.
    if eligible_course_overlays:
        _winner, winner_payload = max(
            eligible_course_overlays,
            key=lambda item: _overlay_version_key(item[0]),
        )
        for field in COURSE_OVERLAY_FIELDS:
            if field in winner_payload:
                merged[field] = winner_payload[field]

    sections.sort(key=_section_identity)
    merged["sections"] = sections
    if applied_versions:
        merged["syllabus_overlay_versions"] = list(dict.fromkeys(applied_versions))
        merged["syllabus_provenance"] = applied_provenance
    return merged


def _valid_detail_sections(detail: dict) -> list[dict]:
    """Exclude whole rows with semantic errors before filters or prompting."""

    return [
        dict(section)
        for section in (detail.get("sections") or [])
        if isinstance(section, dict)
        and validate_section(section).status == "published"
    ]


def _as_float(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    return None


def build_filters_from_intent(intent: dict) -> dict:
    """
    将 query_intent 转换为 filter_by_fields 可接受的 filters dict。

    映射规则：
      intent["course_codes"] → filters["course_codes"]（非空时）
      intent["department"] → filters["department"]（非 None 时）
      intent["instructor"] → filters["instructor"]（非 None 时）
      intent["points_range"] → filters["points_min"], filters["points_max"]
      intent["term"] → filters["term"]（非 None 时）
      intent["day_preference"] → filters["days"]（非空时）
      intent["time_preference"] → filters["time_of_day"]（非 None 时）

    只包含非空/非 None 的条件。
    """
    filters: dict = {}

    course_codes = intent.get("course_codes") or []
    if course_codes:
        filters["course_codes"] = course_codes

    department = intent.get("department")
    if department:
        filters["department"] = department

    instructor = intent.get("instructor")
    if instructor:
        filters["instructor"] = instructor

    points_range = intent.get("points_range")
    if isinstance(points_range, (list, tuple)) and len(points_range) == 2:
        min_points, max_points = _as_float(points_range[0]), _as_float(points_range[1])
        if min_points is not None:
            filters["points_min"] = min_points
        if max_points is not None:
            filters["points_max"] = max_points

    term = intent.get("term")
    if term:
        filters["term"] = term

    days = intent.get("day_preference") or []
    if days:
        filters["days"] = days

    time_of_day = intent.get("time_preference")
    if time_of_day:
        filters["time_of_day"] = time_of_day

    return filters


_CREDIT_CONSTRAINT_TERMS = frozenset(
    {
        "credit",
        "credits",
        "point",
        "points",
        "unit",
        "units",
        "least",
        "most",
        "minimum",
        "maximum",
        "min",
        "max",
        "between",
        "menos",
        "entre",
        "credito",
        "creditos",
        "punto",
        "puntos",
        "unidad",
        "unidades",
        "como",
        "maximo",
        "mas",
        "maximum",
        "moins",
        "plus",
    }
)

_ATTRIBUTE_QUERY_TERMS = frozenset(
    {
        "prerequisite",
        "prerequisites",
        "prereq",
        "prereqs",
        "credit",
        "credits",
        "point",
        "points",
        "unit",
        "units",
        "instructor",
        "instructors",
        "professor",
        "professors",
        "teacher",
        "teach",
        "teaches",
        "teaching",
        "taught",
        "time",
        "times",
        "schedule",
        "when",
        "where",
        "location",
        "room",
        "meet",
        "meets",
        "offered",
        "semester",
        "term",
        "enrollment",
        "capacity",
        "seat",
        "seats",
        "syllabus",
        "description",
        "detail",
        "details",
        "info",
        "information",
        # Spanish/French attribute words retained by normalize_question.
        "prerrequisito",
        "prerrequisitos",
        "requisito",
        "requisitos",
        "previos",
        "credito",
        "creditos",
        "punto",
        "puntos",
        "unidad",
        "unidades",
        "profesor",
        "profesores",
        "docente",
        "docentes",
        "horario",
        "cuando",
        "reune",
        "descripcion",
        "detalle",
        "inscripcion",
        "cupo",
        "cupos",
        "prerequis",
        "prealable",
        "prealables",
        "professeur",
        "professeurs",
        "horaire",
        "quand",
        "inscription",
        "places",
    }
)


def _ranking_keywords(intent: dict, filters: dict) -> list[str]:
    """Remove words already represented by deterministic field filters.

    Once silent candidate backfilling is removed, structured words such as
    ``Monday`` or ``credits`` must not become mandatory topical keywords.
    Otherwise a perfectly valid structured match would score zero simply
    because course titles do not contain the word "credits".
    """
    ignored: set[str] = set(_ATTRIBUTE_QUERY_TERMS)

    ignored.update(
        str(term).strip().lower()
        for term in (intent.get("department_terms") or [])
        if str(term).strip()
    )

    for code in filters.get("course_codes") or []:
        ignored.update(part.lower() for part in str(code).replace("-", " ").split())

    # An explicitly named course is itself the relevance decision.  Free-form
    # question words (especially untranslated multilingual words) must not
    # suppress that record; all additional structured filters still apply.
    if filters.get("course_codes"):
        return []

    for day in filters.get("days") or []:
        day_text = str(day).strip().lower()
        if day_text:
            ignored.add(day_text)
            ignored.add(f"{day_text}s")
        ignored.update(
            {
                "lunes",
                "martes",
                "miercoles",
                "jueves",
                "viernes",
                "sabado",
                "domingo",
                "lundi",
                "mardi",
                "mercredi",
                "jeudi",
                "vendredi",
                "samedi",
                "dimanche",
            }
        )

    time_of_day = str(filters.get("time_of_day") or "").strip().lower()
    if time_of_day:
        ignored.add(time_of_day)
        ignored.update(
            {"manana", "tarde", "noche", "matin", "soir", "apres", "midi"}
        )

    term = str(filters.get("term") or "").strip().lower()
    if term:
        ignored.update(term.split())

    if "points_min" in filters or "points_max" in filters:
        ignored.update(_CREDIT_CONSTRAINT_TERMS)

    instructor = str(filters.get("instructor") or "").strip().lower()
    if instructor:
        ignored.update(instructor.split())
        ignored.update({"professor", "prof", "instructor", "teacher"})

    return [
        text
        for keyword in (intent.get("keywords") or [])
        if (text := str(keyword).strip()) and text.lower() not in ignored
    ]


def retrieve_courses(
    enriched_index: list[dict],
    intent: dict,
    courses_dir: str,
    max_results: int = 10,
) -> list[dict]:
    """
    完整检索流程。

    1. 如果 query_type == "general" → 返回空列表（通用问题不需要课程数据）

    2. 构造 filters: build_filters_from_intent(intent)

    3. Stage 1 粗筛: filter_by_fields(enriched_index, filters)

    4. Stage 2 精筛:
       - 如果 intent["keywords"] 非空 → search_by_keywords(candidates, keywords)
       - 否则截取 candidates[:max_results]

    5. 加载课程详情:
       对精筛结果中的每个课程，用其 path 字段加载完整 JSON。
       路径拼接：Path(courses_dir).parent / entry["path"]
       （因为 path 格式是 "courses_flat/xxx.json"，而 courses_dir 是 data/courses_flat）

    6. 返回完整课程 JSON 列表

    边界情况：
    - 粗筛 0 结果 → 精确返回空列表，不静默删除时间/星期条件
    - 关键词命中不足 max_results → 返回实际命中，不用无关课程补齐
    - 加载某个 JSON 失败 → 跳过该课程，继续其他
    """
    if (intent.get("query_type") or "").lower() in ("general", "stats"):
        return []

    filters = build_filters_from_intent(intent)
    has_structural = bool(filters)
    candidates = filter_by_fields(enriched_index, filters)

    if not candidates:
        return []

    # 已经按 department 结构化过滤过了，用户用来指代系别的词
    # （"computer science" → COMS）在 Stage-2 打分中不再具备区分度：
    # 它们对全系课程一律命中，导致同分并退化成课号字母序，
    # 且会把标题字面含系别名的占位课（TOPICS IN COMPUTER SCIENCE）顶到第一。
    keywords = _ranking_keywords(intent, filters)

    if keywords:
        # 剩下的都是主题词：必须命中课号/标题级别的相关度
        # (score >= 3)；只在 searchable_text 里蹭到一个泛化词不算。
        ranked = search_by_keywords(
            candidates, keywords, limit=max_results, min_score=3
        )
        if not ranked:
            return []
        top_entries = ranked
    else:
        # 无关键词又无结构化条件 -> 无锚点，返回空而不是整库前 N 门。
        if not has_structural:
            return []
        # 有结构化条件但没有区分性关键词（例如「有哪些计算机课」）：
        # 按课程质量排序，而不是原索引顺序（等价于课号字母序）。
        top_entries = sort_by_quality(candidates)[:max_results]

    data_root = Path(courses_dir).parent
    detailed_courses: list[dict] = []
    for entry in top_entries:
        rel_path = entry.get("path")
        if not rel_path:
            continue
        detail_path = data_root / rel_path
        detail = load_course_detail(str(detail_path))
        if detail is None:
            continue
        detail.setdefault("course_uid", entry.get("course_uid", ""))

        detail = _merge_published_overlays(detail, entry, filters)
        detail_sections = _valid_detail_sections(detail)
        matched_detail_sections = matching_sections(detail_sections, filters)
        if has_section_filters(filters) and not matched_detail_sections:
            if detail_sections or has_schedule_filters(filters):
                # Keep the detail-level contract exact even if an index is stale.
                continue
            # A course with no published sections can still answer a pure
            # credit query from its course-level range.  It cannot satisfy any
            # term/day/time/instructor condition.
            if not has_points_filters(filters) or not points_range_matches(
                detail.get("points_min"), detail.get("points_max"), filters
            ):
                continue

        detail["sections"] = detail_sections
        detail["matched_sections"] = [dict(section) for section in matched_detail_sections]
        detailed_courses.append(detail)

    return detailed_courses
