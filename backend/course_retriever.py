"""
两阶段课程检索模块。
Stage 1: 结构化字段粗筛（程序化，不调用 LLM）
Stage 2: searchable_text 关键词精筛（评分排序）
最后加载匹配课程的完整 JSON。
"""

from __future__ import annotations

from pathlib import Path

from course_index import (
    course_quality_score,
    filter_by_fields,
    load_course_detail,
    search_by_keywords,
    sort_by_quality,
)


def _entry_richness(entry: dict) -> tuple:
    """同一课号的多条索引记录里，判断哪条信息更完整。"""
    sections = entry.get("sections_summary") or []
    return (
        1 if entry.get("has_description") else 0,
        len(sections),
        sum(1 for s in sections if (s.get("instructor") or "").strip()),
        len(entry.get("searchable_text") or ""),
    )


def dedupe_by_course_code(entries: list[dict]) -> list[dict]:
    """
    按 course_code 去重，保留信息最完整的那条。

    索引里存在 ~147 条重复记录（同一课号来自不同 bulletin 页面，course_uid 不同，
    因此 add_to_index 的 uid 去重拦不住）。不去重会导致同一门课在回答里出现两次，
    并白白挤占 max_results 名额。这里只在查询链路去重，不改动 data/ 下的原始文件。
    """
    best_index: dict[str, int] = {}
    result: list[dict] = []

    for entry in entries:
        code = (entry.get("course_code") or "").strip().upper()
        if not code:
            result.append(entry)
            continue

        if code not in best_index:
            best_index[code] = len(result)
            result.append(entry)
            continue

        pos = best_index[code]
        if _entry_richness(entry) > _entry_richness(result[pos]):
            # 保持原位置（相关度顺序不变），但换成信息更全的那条
            result[pos] = entry

    return result


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
    has_structural = bool(filters)
    candidates = filter_by_fields(enriched_index, filters)

    if not candidates:
        candidates = _retry_relaxed_candidates(enriched_index, filters)

    if not candidates:
        return []

    # 先去重，避免重复课号占掉 max_results 名额。
    candidates = dedupe_by_course_code(candidates)

    # 已经按 department 结构化过滤过了，用户用来指代系别的词
    # （"computer science" → COMS）在 Stage-2 打分中不再具备区分度：
    # 它们对全系课程一律命中，导致同分并退化成课号字母序，
    # 且会把标题字面含系别名的占位课（TOPICS IN COMPUTER SCIENCE）顶到第一。
    keywords = [str(k) for k in (intent.get("keywords") or []) if k is not None]
    if intent.get("department"):
        dept_terms = {
            str(t).strip().lower()
            for t in (intent.get("department_terms") or [])
            if str(t).strip()
        }
        if dept_terms:
            keywords = [k for k in keywords if k.strip().lower() not in dept_terms]

    if keywords:
        # 无结构化锚点时抬高相关度门槛：必须命中课号或标题（score>=3），
        # 只在 searchable_text 里蹭到一个泛化词（score=1）不算相关。
        min_score = 1 if has_structural else 3
        ranked = search_by_keywords(
            candidates, keywords, limit=max(15, max_results), min_score=min_score
        )
        # 没有任何结构化过滤条件（系别/课程代码/学分/时间…），
        # 且关键词一个都没命中 -> 不要用整库的前 N 门课来凑数（否则会返回一堆无关课程）。
        if not has_structural and not ranked:
            return []
        if len(ranked) < max_results:
            # 补齐时只从已过滤的候选集里取，不引入无关课程；补齐部分按质量排序。
            seen_codes = {(e.get("course_code") or "").upper() for e in ranked}
            for entry in sort_by_quality(candidates):
                code = (entry.get("course_code") or "").upper()
                if code in seen_codes:
                    continue
                ranked.append(entry)
                seen_codes.add(code)
                if len(ranked) >= max_results:
                    break
        top_entries = ranked[:max_results]
    else:
        # 无关键词又无结构化条件 -> 无锚点，返回空而不是整库前 N 门。
        if not has_structural:
            return []
        # 有结构化条件但没有区分性关键词（例如「有哪些计算机课」）：
        # 按课程质量排序，而不是原索引顺序（等价于课号字母序）。
        top_entries = sort_by_quality(candidates)[:max_results]

    data_root = Path(courses_dir).parent
    detailed_courses: list[dict] = []
    seen_codes: set[str] = set()
    for entry in top_entries:
        rel_path = entry.get("path")
        if not rel_path:
            continue
        detail_path = data_root / rel_path
        detail = load_course_detail(str(detail_path))
        if detail is None:
            continue
        # 详情加载后再兜底去重一次：索引与详情文件的 course_code 可能不一致。
        code = (detail.get("course_code") or "").strip().upper()
        if code and code in seen_codes:
            continue
        if code:
            seen_codes.add(code)
        detailed_courses.append(detail)

    return detailed_courses
