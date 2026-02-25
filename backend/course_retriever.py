"""
两阶段课程检索模块。
Stage 1: 结构化字段粗筛（程序化，不调用 LLM）
Stage 2: searchable_text 关键词精筛（评分排序）
最后加载匹配课程的完整 JSON。
"""

from __future__ import annotations

from pathlib import Path

from course_index import filter_by_fields, load_course_detail, search_by_keywords


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


def _relaxed_filters(filters: dict) -> dict:
    relaxed = dict(filters)
    relaxed.pop("time_of_day", None)
    relaxed.pop("days", None)
    return relaxed


def _retry_relaxed_candidates(enriched_index: list[dict], filters: dict) -> list[dict]:
    if len(filters) <= 1:
        return []

    # Prefer dropping time constraints first, then day constraints.
    if "time_of_day" in filters:
        relaxed = dict(filters)
        relaxed.pop("time_of_day", None)
        candidates = filter_by_fields(enriched_index, relaxed)
        if candidates:
            return candidates

    if "days" in filters:
        relaxed = dict(filters)
        relaxed.pop("days", None)
        candidates = filter_by_fields(enriched_index, relaxed)
        if candidates:
            return candidates

    relaxed = _relaxed_filters(filters)
    if relaxed != filters:
        return filter_by_fields(enriched_index, relaxed)
    return []


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
    - 粗筛 0 结果 + filters 有多个条件 → 尝试放宽（移除 time_of_day 或 days 重试）
    - 最终 0 结果 → 返回空列表
    - 加载某个 JSON 失败 → 跳过该课程，继续其他
    """
    if (intent.get("query_type") or "").lower() in ("general", "stats"):
        return []

    filters = build_filters_from_intent(intent)
    candidates = filter_by_fields(enriched_index, filters)

    if not candidates:
        candidates = _retry_relaxed_candidates(enriched_index, filters)

    if not candidates:
        return []

    keywords = intent.get("keywords") or []
    if keywords:
        safe_keywords = [str(k) for k in keywords if k is not None]
        ranked = search_by_keywords(candidates, safe_keywords)
        if len(ranked) < max_results:
            seen_paths = {entry.get("path") for entry in ranked}
            for entry in candidates:
                path = entry.get("path")
                if path in seen_paths:
                    continue
                ranked.append(entry)
                seen_paths.add(path)
                if len(ranked) >= max_results:
                    break
        top_entries = ranked[:max_results]
    else:
        top_entries = candidates[:max_results]

    data_root = Path(courses_dir).parent
    detailed_courses: list[dict] = []
    for entry in top_entries:
        rel_path = entry.get("path")
        if not rel_path:
            continue
        detail_path = data_root / rel_path
        detail = load_course_detail(str(detail_path))
        if detail is not None:
            detailed_courses.append(detail)

    return detailed_courses
