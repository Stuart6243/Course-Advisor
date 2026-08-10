"""
Course index management: build enriched index and provide search helpers.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional


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
    if not prereq_text:
        return []

    raw_codes = re.findall(r"[A-Z]{4}\s+[A-Z]?\d{4}", prereq_text.upper())
    unique_codes: list[str] = []
    seen: set[str] = set()
    for code in raw_codes:
        normalized = re.sub(r"\s+", " ", code).strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique_codes.append(normalized)
    return unique_codes


def parse_days_from_times(times_str: str) -> tuple[list[str], str]:
    """
    Parse times field into (days, time_of_day).
    """
    if not times_str:
        return [], ""

    day_map = {
        "M": "Monday",
        "T": "Tuesday",
        "W": "Wednesday",
        "Th": "Thursday",
        "F": "Friday",
        "Sa": "Saturday",
        "Su": "Sunday",
    }

    day_tokens = re.findall(r"(Th|Su|Sa|[MTWF])", times_str)
    days: list[str] = []
    for token in day_tokens:
        day_name = day_map[token]
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
    all_instructors: list[str] = []
    all_terms: list[str] = []

    for section in sections:
        term = (section.get("term") or "").strip()
        times = (section.get("times") or "").strip()
        instructor = (section.get("instructor") or "").strip()
        location = (section.get("location") or "").strip()
        days, time_of_day = parse_days_from_times(times)

        sections_summary.append(
            {
                "term": term,
                "times": times,
                "days": days,
                "time_of_day": time_of_day,
                "instructor": instructor,
                "location": location,
                "enrollment_current": section.get("enrollment_current"),
                "enrollment_capacity": section.get("enrollment_capacity"),
            }
        )

        if instructor and instructor not in all_instructors:
            all_instructors.append(instructor)
        if term and term not in all_terms:
            all_terms.append(term)

    description = (course_detail.get("description") or "").strip()
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
        "sections_summary": sections_summary,
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
    if not filters:
        return list(index)

    course_codes = [c.strip().upper() for c in (filters.get("course_codes") or []) if c]
    if course_codes:
        code_set = set(course_codes)
        return [e for e in index if (e.get("course_code", "").upper() in code_set)]

    department = (filters.get("department") or "").strip().lower()
    instructor = (filters.get("instructor") or "").strip().lower()
    points_min = filters.get("points_min")
    points_max = filters.get("points_max")
    term = (filters.get("term") or "").strip().lower()
    days = [d.strip().lower() for d in (filters.get("days") or []) if d]
    time_of_day = (filters.get("time_of_day") or "").strip().lower()

    results: list[dict] = []
    for entry in index:
        if department:
            if (entry.get("department_prefix") or "").lower() != department:
                continue

        if instructor:
            instructors = [i.lower() for i in entry.get("all_instructors", []) if i]
            if not any(instructor in name for name in instructors):
                continue

        if points_min is not None:
            val = entry.get("points_min")
            if val is None or float(val) < float(points_min):
                continue

        if points_max is not None:
            val = entry.get("points_max")
            if val is None or float(val) > float(points_max):
                continue

        if term:
            terms = [t.lower() for t in entry.get("all_terms", []) if t]
            if term not in terms:
                continue

        if days:
            needed = set(days)
            sections = entry.get("sections_summary", []) or []
            if not any(
                needed.issubset({d.lower() for d in (section.get("days") or [])})
                for section in sections
            ):
                continue

        if time_of_day:
            sections = entry.get("sections_summary", []) or []
            if not any(
                (section.get("time_of_day") or "").lower() == time_of_day
                for section in sections
            ):
                continue

        results.append(entry)

    return results


def search_by_keywords(candidates: list[dict], keywords: list[str]) -> list[dict]:
    """
    Stage-2 scoring and ranking using keywords and searchable_text.
    """
    normalized = [kw.strip().lower() for kw in keywords if kw and kw.strip()]
    if not normalized:
        return candidates[:20]

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

        if score > 0:
            scored.append((score, entry))

    scored.sort(
        key=lambda item: (
            -item[0],
            item[1].get("course_code", ""),
            item[1].get("title", ""),
        )
    )
    return [entry for _, entry in scored[:15]]


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
