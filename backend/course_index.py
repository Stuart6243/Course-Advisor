"""
Course index management: build enriched index and provide search helpers.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from course_codes import extract_course_codes
from section_validator import (
    parse_points_value,
    validate_catalog_record,
    validate_section,
)


# Common department expansions used in searchable_text.
DEPARTMENT_NAMES = {
    # 航空航天
    "AERO": "aerospace aeronautical flight space",
    "AEME": "aerospace mechanical turbomachinery propulsion aerodynamics",
    # 生物医学
    "BMEN": "biomedical engineering biology medical",
    "BMCS": "biomedical computational statistics genomics",
    "BMEB": "biomedical electrical neuroscience brain",
    "BMEE": "biomedical electrical neural control instrumentation",
    "BINF": "bioinformatics computational biomedical health",
    "APBM": "applied physics biomedical anatomy",
    "MEBM": "mechanical biomedical modeling morphogenesis",
    "EEBM": "electrical biomedical neuroscience neuroengineering",
    # 化学工程
    "CHEN": "chemical engineering chemistry",
    "CHEE": "chemical engineering thermodynamics colloid",
    "CBMF": "chemical biochemical molecular",
    "CHAP": "chemical applied physics statistical mechanics",
    # 土木工程
    "CIEN": "civil engineering structural construction",
    "CIEE": "civil environmental infrastructure risk",
    "CEEM": "civil engineering mechanics research",
    "CEOR": "civil engineering operations infrastructure optimization",
    # 计算机科学
    "COMS": "computer science programming software algorithms",
    "CSOR": "operations research computer science optimization",
    "CSEE": "computer science electrical engineering embedded systems",
    "ORCS": "operations research computer science data decision",
    # 地球环境
    "EAEE": "earth environmental engineering sustainability climate",
    "EACE": "earth environmental civil hydrosystems pollution",
    "ECIA": "environmental civil infrastructure water management",
    "EESC": "earth environmental science climate dynamics",
    # 电子电气
    "EECS": "electrical engineering computer science",
    "ELEN": "electrical engineering circuits signals electronics",
    "ECBM": "electrical computer biomedical",
    "EEEL": "electrical engineering energy economics optimization",
    "EEME": "electrical engineering mechanical control systems",
    # 工业工程与运筹
    "IEOR": "industrial engineering operations research optimization",
    "EEOR": "electrical engineering operations research convex optimization",
    "DROM": "decision risk operations management analytics healthcare",
    "ORCA": "operations research data science foundations",
    # 机械工程
    "MECE": "mechanical engineering design manufacturing robotics",
    "MECS": "mechanical computer science evolutionary robotics",
    "MECH": "mechanical engineering combustion",
    "MEEM": "mechanical engineering materials small scale",
    "MEEE": "mechanical electrical acoustics signal",
    "MEIE": "mechanical engineering innovation space flight human",
    "IEME": "industrial engineering mechanical human centered design innovation",
    "MACH": "mechanical applied combustion",
    # 材料科学
    "MSAE": "materials science applied engineering crystallography",
    # 应用数学/物理
    "APMA": "applied mathematics analysis numerical methods",
    "APPH": "applied physics quantum optics condensed matter",
    "APAM": "applied physics applied mathematics research",
    "APCH": "applied physics chemistry soft condensed matter",
    # 工程通识
    "ENGI": "engineering general introduction gateway art",
    "ENME": "engineering mechanics dynamics structures",
    # 其他
    "PLCE": "planning conservation environment urban",
    "PSAM": "quantitative methods social science",
    "GRAP": "graphics computer engineering visualization",
    "HSAM": "humanities science applied mathematics data history",
    "AHCE": "humanities civil engineering history roman",
    "MRKT": "marketing product management",
}


def load_raw_index(path: str) -> list[dict]:
    """Load the raw index JSON file."""
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def load_course_detail(path: str) -> Optional[dict]:
    """Load a course detail JSON file. Return None if missing."""
    p = Path(path)
    if not p.exists():
        print(f"[WARN] Course file missing: {p}")
        return None
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:  # pragma: no cover - defensive logging
        print(f"[WARN] Failed to parse course file {p}: {exc}")
        return None


def extract_department_prefix(course_code: str) -> str:
    """
    Extract department prefix from course code.
    Example: 'CIEN E3125' -> 'CIEN'
    """
    if not course_code:
        return ""
    return course_code.strip().split(" ", 1)[0].upper()


def extract_prerequisite_codes(prereq_text: str) -> list[str]:
    """
    Extract course codes from prerequisite text and deduplicate in order.
    """
    return extract_course_codes(prereq_text)


def parse_days_from_times(times_str: str) -> tuple[list[str], str]:
    """
    Parse times field into (days, time_of_day).
    """
    if not times_str:
        return [], ""

    day_map = {
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
        "su": "Sunday",
        "sun": "Sunday",
        "sunday": "Sunday",
    }

    # Day abbreviations are tokens, not arbitrary substrings.  Without the
    # alphabetic boundaries below, instructor/location text such as
    # "Savannah" is interpreted as Saturday and "TBA" as Tuesday.
    day_tokens = re.findall(
        r"(?<![A-Za-z])(?:Monday|Mon|M|Tuesday|Tues|Tue|Tu|T|"
        r"Wednesday|Wed|W|Thursday|Thurs|Thur|Thu|Th|"
        r"Friday|Fri|F|Saturday|Sat|Sa|Sunday|Sun|Su)(?![A-Za-z])",
        times_str,
        flags=re.IGNORECASE,
    )
    days: list[str] = []
    for token in day_tokens:
        day_name = day_map[token.lower()]
        if day_name not in days:
            days.append(day_name)

    time_of_day = ""
    match = re.search(r"(\d{1,2}):(\d{2})\s*(am|pm)", times_str, flags=re.IGNORECASE)
    if match:
        hour = int(match.group(1))
        am_pm = match.group(3).lower()
        if am_pm == "pm" and hour != 12:
            hour += 12
        elif am_pm == "am" and hour == 12:
            hour = 0

        if hour < 12:
            time_of_day = "morning"
        elif hour < 17:
            time_of_day = "afternoon"
        else:
            time_of_day = "evening"

    return days, time_of_day


SCHEDULE_SECTION_FILTER_KEYS = frozenset(
    {"term", "days", "time_of_day", "instructor"}
)
POINT_SECTION_FILTER_KEYS = frozenset({"points_min", "points_max"})
SECTION_FILTER_KEYS = SCHEDULE_SECTION_FILTER_KEYS | POINT_SECTION_FILTER_KEYS


def has_section_filters(filters: dict) -> bool:
    """Return whether *filters* contains an active section-level condition."""
    return has_schedule_filters(filters) or has_points_filters(filters)


def has_schedule_filters(filters: dict) -> bool:
    """Return whether term/day/time/instructor constrains an offering."""
    return any(filters.get(key) for key in SCHEDULE_SECTION_FILTER_KEYS)


def has_points_filters(filters: dict) -> bool:
    """Return whether either (possibly zero-valued) credit bound is active."""
    return any(filters.get(key) is not None for key in POINT_SECTION_FILTER_KEYS)


def points_range_matches(
    actual_min: object,
    actual_max: object,
    filters: dict,
) -> bool:
    """Return whether a course/section credit range overlaps the request."""
    if not has_points_filters(filters):
        return True
    if actual_min is None and actual_max is None:
        return False
    try:
        low = float(actual_min if actual_min is not None else actual_max)
        high = float(actual_max if actual_max is not None else actual_min)
        wanted_low = (
            float(filters["points_min"])
            if filters.get("points_min") is not None
            else float("-inf")
        )
        wanted_high = (
            float(filters["points_max"])
            if filters.get("points_max") is not None
            else float("inf")
        )
    except (TypeError, ValueError):
        return False
    low, high = sorted((low, high))
    return not (high < wanted_low or low > wanted_high)


def _instructor_matches(actual: str, requested: str) -> bool:
    actual_lower = (actual or "").strip().lower()
    requested_lower = (requested or "").strip().lower()
    if not actual_lower or not requested_lower:
        return False

    # Support a surname or given-name-only query while keeping all requested
    # name parts on the same section/instructor string.
    needle_parts = [part for part in requested_lower.split() if len(part) > 1]
    return requested_lower in actual_lower or (
        bool(needle_parts) and all(part in actual_lower for part in needle_parts)
    )


def section_matches_filters(section: dict, filters: dict) -> bool:
    """Evaluate every section-level condition against one section.

    This is deliberately a single predicate.  A course must not satisfy
    Monday on one section and morning (or a term/instructor) on another.
    The helper accepts both enriched ``sections_summary`` rows and full course
    detail sections; missing derived day/time fields are recomputed from the
    raw ``times`` value.
    """
    requested_term = (filters.get("term") or "").strip().lower()
    if requested_term:
        actual_term = (section.get("term") or "").strip().lower()
        if actual_term != requested_term:
            return False

    requested_instructor = (filters.get("instructor") or "").strip()
    if requested_instructor and not _instructor_matches(
        str(section.get("instructor") or ""), requested_instructor
    ):
        return False

    # When a term/day/instructor filter selects an actual offering, a credit
    # constraint must hold on that same section.  The course-level aggregate is
    # only a broad Stage-1 prefilter and cannot prove a term-specific value.
    requested_points_min = filters.get("points_min")
    requested_points_max = filters.get("points_max")
    if requested_points_min is not None or requested_points_max is not None:
        section_points = parse_points_value(section.get("points"))
        if section_points is None:
            return False
        if not points_range_matches(section_points[0], section_points[1], filters):
            return False

    requested_days = {
        str(day).strip().lower()
        for day in (filters.get("days") or [])
        if str(day).strip()
    }
    requested_time = (filters.get("time_of_day") or "").strip().lower()
    if requested_days or requested_time:
        times = str(section.get("times") or "").strip()
        parsed_days, parsed_time = parse_days_from_times(times)

        # Prefer deriving from the raw value when it exists.  This also avoids
        # trusting stale indexes built by the former substring-based day parser.
        if times:
            actual_days = {day.lower() for day in parsed_days}
            actual_time = parsed_time.lower()
        else:
            actual_days = {
                str(day).strip().lower()
                for day in (section.get("days") or [])
                if str(day).strip()
            }
            actual_time = (section.get("time_of_day") or "").strip().lower()

        if requested_days and not requested_days.issubset(actual_days):
            return False
        if requested_time and actual_time != requested_time:
            return False

    return True


def matching_sections(sections: list[dict], filters: dict) -> list[dict]:
    """Return the sections that satisfy the complete section predicate."""
    if not has_section_filters(filters):
        return list(sections or [])
    return [section for section in (sections or []) if section_matches_filters(section, filters)]


def build_searchable_text(course_detail: dict, enriched_entry: dict) -> str:
    """
    Build lower-cased searchable text blob for fuzzy matching.
    """
    parts: list[str] = []

    parts.append(enriched_entry.get("course_code", ""))
    parts.append(enriched_entry.get("title", ""))
    parts.extend(enriched_entry.get("all_instructors", []))

    department_prefix = enriched_entry.get("department_prefix", "")
    if department_prefix:
        parts.append(DEPARTMENT_NAMES.get(department_prefix, ""))

    parts.extend(enriched_entry.get("all_terms", []))

    section_days: list[str] = []
    section_times: list[str] = []
    for section in enriched_entry.get("sections_summary", []):
        section_days.extend(section.get("days", []))
        tod = section.get("time_of_day", "")
        if tod:
            section_times.append(tod)
    parts.extend(section_days)
    parts.extend(section_times)

    description = (course_detail.get("description") or "").strip()
    if description:
        parts.append(description[:200])

    prereq_codes = enriched_entry.get("prerequisites_codes", [])
    if prereq_codes:
        parts.extend(prereq_codes)
    else:
        parts.extend(extract_prerequisite_codes(course_detail.get("prerequisites_text", "")))

    text = " ".join(p for p in parts if p)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def build_enriched_entry(raw_entry: dict, course_detail: dict) -> dict:
    """
    Build one enriched index entry from raw index row + full course detail.
    """
    file_name = raw_entry.get("file_name") or Path(raw_entry.get("path", "")).name
    path_value = f"courses_flat/{file_name}" if file_name else raw_entry.get("path", "")

    course_code = course_detail.get("course_code") or raw_entry.get("course_code", "")
    department_prefix = extract_department_prefix(course_code)
    prerequisites_codes = extract_prerequisite_codes(course_detail.get("prerequisites_text", ""))

    sections = course_detail.get("sections", []) or []
    sections_summary: list[dict] = []
    review_sections_summary: list[dict] = []
    all_instructors: list[str] = []
    all_terms: list[str] = []

    for section in sections:
        validation = validate_section(section)
        term = validation.normalized["term"]
        times = validation.normalized["times"]
        instructor = validation.normalized["instructor"]
        location = validation.normalized["location"]
        days, time_of_day = (
            parse_days_from_times(times) if not validation.errors else ([], "")
        )

        summary = {
            "section_id": validation.normalized["section_call_number"],
            "term": term,
            "times": times,
            "days": days,
            "time_of_day": time_of_day,
            "instructor": instructor,
            "location": location,
            "points": validation.normalized["points"],
            "enrollment_current": section.get("enrollment_current"),
            "enrollment_capacity": section.get("enrollment_capacity"),
            "validation_status": validation.status,
            "validation_errors": list(validation.errors),
            "validation_warnings": list(validation.warnings),
            "provenance": {
                "source": "catalog_seed",
                "course_uid": raw_entry.get("course_uid")
                or course_detail.get("course_uid", ""),
                "source_page_url": course_detail.get("source_page_url", ""),
            },
        }
        if validation.status != "published":
            # Retain review metadata for audit/UI, but never feed it into
            # section filters, all_* fields, or searchable_text.
            review_sections_summary.append(summary)
            continue
        sections_summary.append(summary)

        if instructor and instructor not in all_instructors:
            all_instructors.append(instructor)
        if term and term not in all_terms:
            all_terms.append(term)

    description = (course_detail.get("description") or "").strip()
    catalog_validation = validate_catalog_record(course_detail)
    enriched = {
        "course_uid": raw_entry.get("course_uid") or course_detail.get("course_uid", ""),
        "course_code": course_code,
        "title": course_detail.get("title") or raw_entry.get("title", ""),
        "file_name": file_name,
        "path": path_value,
        "department_prefix": department_prefix,
        "points_min": course_detail.get("points_min"),
        "points_max": course_detail.get("points_max"),
        "has_description": bool(description),
        "prerequisites_codes": prerequisites_codes,
        "bulletin_year": course_detail.get("bulletin_year", ""),
        "catalog_validation_status": catalog_validation.status,
        "catalog_validation_warnings": list(catalog_validation.warnings),
        "sections_summary": sections_summary,
        "review_sections_summary": review_sections_summary,
        "all_instructors": all_instructors,
        "all_terms": all_terms,
    }
    enriched["searchable_text"] = build_searchable_text(course_detail, enriched)
    return enriched


def _resolve_course_path(raw_entry: dict, courses_dir: Path) -> Path:
    """
    Resolve course JSON path using common raw path variants.
    """
    file_name = raw_entry.get("file_name")
    raw_path = raw_entry.get("path", "")

    candidates: list[Path] = []
    if file_name:
        candidates.append(courses_dir / file_name)

    if raw_path:
        rp = Path(raw_path)
        if rp.is_absolute():
            candidates.append(rp)
        else:
            repo_root = courses_dir.parent.parent
            candidates.append(repo_root / raw_path)
            if raw_path.startswith("data/"):
                candidates.append(repo_root / raw_path[5:])
            candidates.append(courses_dir.parent / rp.name)
            candidates.append(courses_dir / rp.name)

    # Keep order, drop duplicates.
    deduped: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            deduped.append(candidate)

    for candidate in deduped:
        if candidate.exists():
            return candidate

    return deduped[0] if deduped else courses_dir / (file_name or "")


def build_enriched_index(raw_index_path: str, courses_dir: str) -> list[dict]:
    """
    Build enriched index by iterating every item in the raw index.
    """
    raw_index = load_raw_index(raw_index_path)
    courses_path = Path(courses_dir)

    enriched: list[dict] = []
    failed = 0
    for raw_entry in raw_index:
        detail_path = _resolve_course_path(raw_entry, courses_path)
        detail = load_course_detail(str(detail_path))
        if detail is None:
            failed += 1
            continue
        enriched.append(build_enriched_entry(raw_entry, detail))

    print(
        f"[Index Build] total={len(raw_index)}, success={len(enriched)}, failed={failed}"
    )
    return enriched


def build_enriched_index_from_dir(courses_dir: str) -> list[dict]:
    """
    直接扫描 courses_flat 目录下的所有课程 JSON 构建 enriched index。

    以课程文件目录为「唯一真源」，避免依赖可能过期/缺条目的 raw index
    （raw index 与实际课程文件数量曾出现漂移）。
    """
    courses_path = Path(courses_dir)
    if not courses_path.exists():
        return []

    enriched: list[dict] = []
    failed = 0
    for json_file in sorted(courses_path.glob("*.json")):
        detail = load_course_detail(str(json_file))
        if detail is None:
            failed += 1
            continue
        raw_entry = {
            "course_uid": detail.get("course_uid", ""),
            "course_code": detail.get("course_code", ""),
            "title": detail.get("title", ""),
            "file_name": json_file.name,
            "path": f"courses_flat/{json_file.name}",
        }
        enriched.append(build_enriched_entry(raw_entry, detail))

    print(f"[Index Build/dir] files={len(enriched) + failed}, success={len(enriched)}, failed={failed}")
    return enriched


def build_raw_index_from_dir(courses_dir: str) -> list[dict]:
    """从 courses_flat 目录重建 raw index（{uid, code, title, file_name, path}）。"""
    courses_path = Path(courses_dir)
    if not courses_path.exists():
        return []

    raw: list[dict] = []
    for json_file in sorted(courses_path.glob("*.json")):
        detail = load_course_detail(str(json_file))
        if detail is None:
            continue
        raw.append(
            {
                "course_uid": detail.get("course_uid", ""),
                "course_code": detail.get("course_code", ""),
                "title": detail.get("title", ""),
                "file_name": json_file.name,
                "path": f"courses_flat/{json_file.name}",
            }
        )
    return raw


def save_enriched_index(index: list[dict], output_path: str) -> None:
    """Save enriched index JSON."""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)


def load_enriched_index(path: str) -> list[dict]:
    """Load enriched index JSON."""
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def filter_by_fields(index: list[dict], filters: dict) -> list[dict]:
    """
    Stage-1 filtering using structured fields.
    """
    # Legacy seed records marked needs_review stay auditable in the index but
    # are not candidates.  A published overlay may explicitly elevate a
    # runtime copy without modifying the immutable course detail.
    index = [
        entry
        for entry in index
        if entry.get("catalog_validation_status", "published") == "published"
    ]
    if not filters:
        return list(index)

    # 显式点名课程代码时，以课程代码为主筛条件，但不再直接 return —
    # 旧版在这里提前返回，导致同一条查询里的 term / days / time_of_day
    # 等附加条件被静默忽略（"CIEN E3125 春季有开吗" 的学期条件不生效）。
    course_codes = [c.strip().upper() for c in (filters.get("course_codes") or []) if c]
    code_set = set(course_codes)
    if code_set:
        index = [e for e in index if (e.get("course_code", "").upper() in code_set)]

    department = (filters.get("department") or "").strip().lower()
    schedule_filtering = has_schedule_filters(filters)
    points_filtering = has_points_filters(filters)
    schedule_only_filters = {
        key: value
        for key, value in filters.items()
        if key not in POINT_SECTION_FILTER_KEYS
    }

    results: list[dict] = []
    for entry in index:
        if department:
            if (entry.get("department_prefix") or "").lower() != department:
                continue

        sections = entry.get("sections_summary", []) or []
        schedule_matches = matching_sections(sections, schedule_only_filters)
        if schedule_filtering and not schedule_matches:
            continue

        matched = schedule_matches
        if points_filtering:
            # New indexes carry section points, allowing the complete
            # term/day/time/instructor/credits predicate to be proved on one
            # offering.  The checked-in legacy index lacks summary points, so
            # use its course range only as a broad prefilter and let detail
            # loading perform the exact section check.  This avoids both a
            # false rejection and cross-section Spring-3/Fall-4 leakage.
            summaries_have_points = bool(schedule_matches) and all(
                parse_points_value(section.get("points")) is not None
                for section in schedule_matches
            )
            if summaries_have_points:
                matched = [
                    section
                    for section in schedule_matches
                    if section_matches_filters(section, filters)
                ]
                if not matched:
                    continue
            elif not points_range_matches(
                entry.get("points_min"), entry.get("points_max"), filters
            ):
                continue

        # Preserve the exact summaries that passed the combined predicate for
        # downstream ranking/detail loading.  Never mutate the shared index.
        if schedule_filtering or (points_filtering and bool(sections)):
            entry = dict(entry)
            entry["matched_sections"] = matched

        results.append(entry)

    return results


# 占位型课程：独立研究、研讨、专题、论文等。对本科生选课几乎没有参考价值，
# 但它们的标题往往字面包含系别名（"TOPICS IN COMPUTER SCIENCE"），
# 在关键词打分里反而排第一。这里显式降权。
PLACEHOLDER_TITLE_RE = re.compile(
    r"\b(tutorial|seminar|projects?|topics?|research|readings?|"
    r"independent\s+study|special\s+topics?|fieldwork|internship|"
    r"thesis|dissertation|colloquium|practicum|advanced\s+study)\b",
    re.IGNORECASE,
)


def extract_course_level(course_code: str) -> int:
    """从课程代码中取出 4 位数字课号。取不到返回 9999。"""
    match = re.search(r"(\d{4})\s*$", (course_code or "").strip())
    return int(match.group(1)) if match else 9999


def course_quality_score(entry: dict) -> int:
    """
    本科生视角的课程「可选性」评分，用作检索排序的 tie-break。

    解决的问题：按系别检索时所有课的关键词得分相同，
    排序退化成课号字母序，Top-5 全是 E69xx/E99xx 的占位课程。
    """
    score = 0
    title = entry.get("title") or ""
    level = extract_course_level(entry.get("course_code") or "")

    # 有真实课程描述 —— 最重要的信号，否则没法回答"这门课讲什么"
    if entry.get("has_description"):
        score += 30

    # 本学期真的开课，且有时间/教师信息
    sections = entry.get("sections_summary") or []
    if sections:
        score += 25
        if any((s.get("times") or "").strip() for s in sections):
            score += 10
        if any((s.get("instructor") or "").strip() for s in sections):
            score += 5

    # 占位/科研类课程降权
    if PLACEHOLDER_TITLE_RE.search(title):
        score -= 45

    # 课号层级：本科与硕士基础课优先，9000+ 基本是博士科研
    if level < 6000:
        score += 20
    elif level >= 9000:
        score -= 30
    else:
        score -= 10

    return score


def sort_by_quality(entries: list[dict]) -> list[dict]:
    """无关键词区分度时，按课程质量排序（而不是课号字母序）。"""
    entries = [
        entry
        for entry in entries
        if entry.get("catalog_validation_status", "published") == "published"
    ]
    return sorted(
        entries,
        key=lambda e: (
            -course_quality_score(e),
            extract_course_level(e.get("course_code") or ""),
            e.get("course_code", ""),
        ),
    )


def search_by_keywords(
    candidates: list[dict],
    keywords: list[str],
    limit: int = 15,
    min_score: int = 1,
) -> list[dict]:
    """
    Stage-2 scoring and ranking using keywords and searchable_text.

    同分时用 course_quality_score 做 tie-break，避免退化成课号字母序。

    min_score: 相关度下限。没有任何结构化锚点（系别/课号/教授）时应调高，
    否则一个泛化词命中 searchable_text 就足以"凑"出结果
    （"what is the meaning of life" 会因为 life 命中 LIFE CYCLE ASSESSMENT 而返回课程）。
    """
    candidates = [
        entry
        for entry in candidates
        if entry.get("catalog_validation_status", "published") == "published"
    ]
    normalized = [kw.strip().lower() for kw in keywords if kw and kw.strip()]
    if not normalized:
        return sort_by_quality(candidates)[:20]

    scored: list[tuple[int, dict]] = []
    for entry in candidates:
        code = (entry.get("course_code") or "").lower()
        title = (entry.get("title") or "").lower()
        searchable = (entry.get("searchable_text") or "").lower()

        score = 0
        for kw in normalized:
            if kw in code:
                score += 5
            if kw in title:
                score += 3
            if kw in searchable:
                score += 1

        if score >= min_score:
            scored.append((score, entry))

    scored.sort(
        key=lambda item: (
            -item[0],
            -course_quality_score(item[1]),
            extract_course_level(item[1].get("course_code") or ""),
            item[1].get("course_code", ""),
            item[1].get("title", ""),
        )
    )
    return [entry for _, entry in scored[:limit]]


def add_to_index(index: list[dict], new_entry: dict) -> None:
    """
    Add or replace an entry by course_uid.
    """
    new_uid = new_entry.get("course_uid")
    if not new_uid:
        index.append(new_entry)
        return

    for i, entry in enumerate(index):
        if entry.get("course_uid") == new_uid:
            index[i] = new_entry
            return
    index.append(new_entry)


def run_build_and_save(raw_index_path: str, courses_dir: str, enriched_index_path: str) -> list[dict]:
    """
    Build and save enriched index, printing Track-B style progress summary.
    """
    print(f"Building enriched index...")
    print(f"  Courses dir (source of truth): {courses_dir}")

    # 以 courses_flat 目录为唯一真源重建，避免 raw index 缺条目导致丢课。
    idx = build_enriched_index_from_dir(courses_dir)
    save_enriched_index(idx, enriched_index_path)

    print(f"\n✅ Saved {len(idx)} enriched entries to {enriched_index_path}")

    depts = set(e["department_prefix"] for e in idx)
    terms = set(t for e in idx for t in e["all_terms"])
    with_desc = sum(1 for e in idx if e["has_description"])
    print(f"  Departments: {len(depts)}")
    print(f"  Terms: {terms}")
    print(f"  With description: {with_desc}/{len(idx)}")
    return idx


if __name__ == "__main__":
    import sys
    import config

    run_build_and_save(
        str(config.RAW_INDEX_PATH),
        str(config.COURSES_DIR),
        str(config.ENRICHED_INDEX_PATH),
    )
