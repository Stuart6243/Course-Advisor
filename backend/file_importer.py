"""
PDF/HTML 文件导入模块。
将任意来源的课程文件转换为标准课程 JSON 格式。

流程：提取文本 -> LLM 结构化转换 -> 验证 -> 保存 -> 更新索引
"""

from __future__ import annotations

import hashlib
import io
import json
import re
from pathlib import Path
from typing import Any

import config
from course_index import DEPARTMENT_NAMES, add_to_index, build_enriched_entry, save_enriched_index


CONVERSION_SYSTEM_PROMPT = """你是一个课程信息提取专家。你的任务是从原始文本中提取课程信息，并转换为精确的 JSON 格式。

只返回 JSON，不要任何其他文字。

目标 JSON 结构：
{
  "course_code": "(必填) 格式如 'CIEN E3125'，即 大写字母系别缩写 + 空格 + 字母数字代码",
  "title": "(必填) 课程英文全称",
  "points_raw": "如 '3.00 points'",
  "points_min": 3.0,
  "points_max": 3.0,
  "bulletin_year": "如 '2025-2026'，不确定则空字符串",
  "department_or_group": "系别全称，不确定则空字符串",
  "description": "课程描述文字（最重要字段），没有则空字符串",
  "prerequisites_text": "先修课要求的完整文字，没有则空字符串",
  "notes_text": "其他备注信息，没有则空字符串",
  "sections": [
    {
      "term": "如 'Spring 2026'",
      "course_number": "如 'CIEN 3125'",
      "section_call_number": "如 '001/11855'，不确定则空字符串",
      "times": "如 'T Th 10:10am - 11:25am'",
      "location": "教室位置",
      "instructor": "教授全名",
      "points": "如 '3.00'",
      "enrollment_current": 0,
      "enrollment_capacity": 0
    }
  ]
}

严格规则：
1. course_code 和 title 是必填字段。如果无法从文本中提取这两个字段，返回：{"error": "Cannot extract course code or title from the provided text"}
2. 所有可选字段：如果文本中没有对应信息，用空字符串 ""、空数组 []、或 0
3. 绝对不要编造文本中不存在的信息
4. sections 数组可以为空（如果文本中没有具体的开课信息）
5. points_min 和 points_max 必须是数字（float），如果无法确定学分，都设为 0.0
6. 如果文本包含多门课程的信息，只提取第一门
7. description 字段必须尽量完整：如果文档包含模块/主题/教学目标/工具（如 Python、Excel）/评分方式，请全部整合到 description
8. description 不得为空且不得只写一句空泛话；若有 syllabus/module breakdown，要覆盖所有关键模块"""


COURSE_CODE_PATTERN = re.compile(r"^[A-Z]{2,4}\s+[A-Z]?\d{4}$")
MULTI_SPACE_RE = re.compile(r"\s+")
KNOWN_DEPARTMENT_PREFIXES = set(DEPARTMENT_NAMES.keys())
DESCRIPTION_FALLBACK_PATTERNS = (
    re.compile(
        r"(?:course\s+description|description)\s*[:：]?\s*(.{80,2000}?)(?:\n\s*\n|$)",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"(?:module|topics?|outline|syllabus)\s*[:：]?\s*(.{80,2000}?)(?:\n\s*\n|$)",
        re.IGNORECASE | re.DOTALL,
    ),
)


def _as_float(value: Any, default: float | None = 0.0) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return default
    return default


def _as_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return default
        try:
            return int(float(text))
        except ValueError:
            return default
    return default


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def normalize_course_code(code: str) -> str:
    """归一 course_code：大写 + 多空格压缩。"""
    text = _safe_str(code).upper()
    return MULTI_SPACE_RE.sub(" ", text).strip()


def validate_course_code(code: str) -> bool:
    """严格校验课程代码格式。"""
    raw = _safe_str(code)
    normalized = normalize_course_code(raw)
    # validate_course_code 保持“严格输入”语义：必须已是规范大写格式。
    if raw != normalized:
        return False
    return bool(COURSE_CODE_PATTERN.match(raw))


def quality_score(data: dict) -> tuple[int, list[str]]:
    """导入质量评分：返回 (0-100, issues)。"""
    score = 100
    issues: list[str] = []

    code = normalize_course_code(data.get("course_code", ""))
    title = _safe_str(data.get("title"))

    if not validate_course_code(code):
        score -= 40
        issues.append("invalid_course_code")

    if len(title) < 3:
        score -= 30
        issues.append("title_too_short")
    elif len(title) < 8:
        score -= 10
        issues.append("title_suspiciously_short")

    points_raw = _safe_str(data.get("points_raw"))
    points_min = _as_float(data.get("points_min"), default=None)
    if (points_min is None or points_min <= 0) and not points_raw:
        score -= 15
        issues.append("missing_points")

    if not _safe_str(data.get("description")):
        score -= 10
        issues.append("missing_description")

    prefix = code.split()[0] if code else ""
    if prefix and prefix not in KNOWN_DEPARTMENT_PREFIXES:
        score -= 20
        issues.append(f"unknown_department:{prefix}")

    return max(0, score), issues


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """从 PDF 字节流中提取纯文本。"""
    try:
        import pdfplumber
    except ModuleNotFoundError as exc:
        raise RuntimeError("pdfplumber is required for PDF import") from exc

    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            pages_text = [page.extract_text() or "" for page in pdf.pages]
    except Exception as exc:
        raise ValueError("Invalid PDF file") from exc
    return "\n\n".join(pages_text).strip()


def extract_text_from_html(file_bytes: bytes) -> str:
    """从 HTML 字节流中提取可见文本。"""
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(file_bytes, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        return soup.get_text(separator="\n", strip=True)
    except ModuleNotFoundError:
        text = file_bytes.decode("utf-8", errors="ignore")
        text = re.sub(
            r"<(script|style|nav|footer|header)[^>]*>.*?</\1>",
            "",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        )
        text = re.sub(r"<[^>]+>", "\n", text)
        return re.sub(r"\s+", " ", text).strip()


def _try_parse_json(text: str) -> dict | None:
    if not text:
        return None
    try:
        parsed = json.loads(text.strip())
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _parse_first_json_object(text: str) -> dict | None:
    decoder = json.JSONDecoder()
    for idx, char in enumerate(text):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _remove_think_tags(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)


def parse_conversion_response(raw_text: str) -> dict:
    """
    从 LLM 返回文本中提取课程 JSON。

    容错链：
    1. 直接 json.loads
    2. 正则提取 {...}
    3. 过滤 <think> 标签后重试
    """
    text = raw_text or ""
    parsed = _try_parse_json(text)
    if parsed is None:
        parsed = _parse_first_json_object(text)
    if parsed is None:
        stripped = _remove_think_tags(text)
        parsed = _try_parse_json(stripped)
        if parsed is None:
            parsed = _parse_first_json_object(stripped)

    if parsed is None:
        return {"error": "Failed to parse LLM response"}
    if "error" in parsed:
        return parsed
    return parsed


def validate_course_json(data: dict) -> tuple[bool, str]:
    """验证转换结果字段。"""
    if not isinstance(data, dict):
        return False, "Invalid conversion result."

    if "error" in data:
        return False, _safe_str(data.get("error")) or "Conversion failed."

    course_code = normalize_course_code(data.get("course_code"))
    title = _safe_str(data.get("title"))

    if not course_code:
        return False, "Missing required field: course_code"
    if not title:
        return False, "Missing required field: title"
    if len(title) < 3:
        return False, f"Title too short: '{title}'. Minimum 3 characters."
    if not validate_course_code(course_code):
        return (
            False,
            f"Invalid course_code format: '{course_code}'. Expected pattern: XXXX Y1234",
        )

    points_min = _as_float(data.get("points_min"), default=None)
    points_max = _as_float(data.get("points_max"), default=None)
    points_raw = _safe_str(data.get("points_raw"))

    if points_min is None and points_max is None and not points_raw:
        return False, "Missing points information."

    return True, ""


def _find_existing_by_code(enriched_index: list[dict], course_code: str) -> dict | None:
    """按 course_code 在索引中查找已存在的课程（大小写/空格不敏感）。"""
    target = normalize_course_code(course_code)
    if not target:
        return None
    for entry in enriched_index:
        if normalize_course_code(entry.get("course_code", "")) == target:
            return entry
    return None


def generate_course_uid(course_code: str, title: str) -> str:
    """生成课程唯一 ID。"""
    key = f"{normalize_course_code(course_code)}|{_safe_str(title)}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


def complete_course_json(data: dict, uid: str) -> dict:
    """补全课程 JSON 中的系统字段和默认字段。"""
    course_code = normalize_course_code(data.get("course_code"))
    title = _safe_str(data.get("title"))
    points_raw = _safe_str(data.get("points_raw"))
    points_min = _as_float(data.get("points_min"), default=0.0) or 0.0
    points_max = _as_float(data.get("points_max"), default=0.0) or 0.0

    raw_sections = data.get("sections")
    sections: list[dict] = []
    if isinstance(raw_sections, list):
        for section in raw_sections:
            if not isinstance(section, dict):
                continue
            sections.append(
                {
                    "term": _safe_str(section.get("term")),
                    "course_number": _safe_str(section.get("course_number")),
                    "section_call_number": _safe_str(section.get("section_call_number")),
                    "times": _safe_str(section.get("times")),
                    "location": _safe_str(section.get("location")),
                    "instructor": _safe_str(section.get("instructor")),
                    "points": _safe_str(section.get("points")),
                    "enrollment_current": _as_int(section.get("enrollment_current"), default=0),
                    "enrollment_capacity": _as_int(section.get("enrollment_capacity"), default=0),
                }
            )

    raw_title_text = f"{course_code} {title}. {points_raw}".strip()
    if raw_title_text.endswith("."):
        raw_title_text = raw_title_text[:-1]

    return {
        "dedup_key": uid,
        "course_uid": uid,
        "bulletin_year": _safe_str(data.get("bulletin_year")),
        "source_page_url": "",
        "source_page_title": "",
        "department_or_group": _safe_str(data.get("department_or_group")),
        "course_code": course_code,
        "title": title,
        "points_raw": points_raw,
        "points_min": points_min,
        "points_max": points_max,
        "description": _safe_str(data.get("description")),
        "description_source": "imported" if _safe_str(data.get("description")) else "none",
        "prerequisites_text": _safe_str(data.get("prerequisites_text")),
        "prerequisites_source": (
            "imported" if _safe_str(data.get("prerequisites_text")) else "none"
        ),
        "notes_text": _safe_str(data.get("notes_text")),
        "sections": sections,
        "needs_review": True,
        "parse_warnings": ["imported_file"],
        "raw_title_text": raw_title_text,
    }


def _identify_missing_fields(data: dict) -> list[str]:
    """识别缺失的关键字段。"""
    missing: list[str] = []
    if not normalize_course_code(data.get("course_code", "")):
        missing.append("course_code")
    if len(_safe_str(data.get("title"))) < 3:
        missing.append("title")

    points_min = _as_float(data.get("points_min"), default=None)
    points_max = _as_float(data.get("points_max"), default=None)
    if points_min is None and points_max is None and not _safe_str(data.get("points_raw")):
        missing.append("points")
    return missing


def _extract_description_fallback(extracted_text: str) -> str:
    """当 LLM description 为空时，从原文中回退提取描述段落。"""
    text = extracted_text or ""
    if not text.strip():
        return ""

    normalized = re.sub(r"\r\n?", "\n", text)
    for pattern in DESCRIPTION_FALLBACK_PATTERNS:
        match = pattern.search(normalized)
        if not match:
            continue
        candidate = re.sub(r"\s+", " ", match.group(1)).strip()
        if len(candidate) >= 20:
            return candidate[:1200]

    collapsed = re.sub(r"\s+", " ", normalized).strip()
    if len(collapsed) >= 120:
        return collapsed[:1200]
    return ""


async def import_file(
    file_bytes: bytes,
    filename: str,
    llm_client,
    courses_dir: str,
    enriched_index: list[dict],
    enriched_index_path: str,
) -> dict:
    """完整导入流程。"""
    try:
        ext = Path(filename).suffix.lower()
        if ext not in config.SUPPORTED_IMPORT_FORMATS:
            return {
                "success": False,
                "message": "Unsupported file format. Use PDF or HTML.",
            }

        if ext == ".pdf":
            extracted_text = extract_text_from_pdf(file_bytes)
        else:
            extracted_text = extract_text_from_html(file_bytes)

        if not extracted_text.strip():
            return {"success": False, "message": "Could not extract text from file."}

        input_text = extracted_text[: config.IMPORT_INPUT_MAX_CHARS]
        messages = [{"role": "user", "content": input_text}]
        # 用独立的 IMPORT_MAX_TOKENS：模型这里要输出一整个课程 JSON
        # （含完整 description + sections），沿用 512 的回答上限会把 JSON 截断，
        # 表现为 description 残缺或解析失败后误触发「需要手动录入」。
        raw_response = await llm_client.chat(
            messages,
            system_prompt=CONVERSION_SYSTEM_PROMPT,
            max_tokens=config.IMPORT_MAX_TOKENS,
        )

        parsed = parse_conversion_response(raw_response)
        if len(_safe_str(parsed.get("description"))) < 20:
            fallback_description = _extract_description_fallback(extracted_text)
            if fallback_description:
                parsed["description"] = fallback_description

        valid, error_msg = validate_course_json(parsed)
        if not valid:
            return {
                "success": False,
                "needs_manual_input": True,
                "partial_data": parsed,
                "missing_fields": _identify_missing_fields(parsed),
                "extracted_text_preview": input_text[:500],
                "message": error_msg,
            }

        parsed["course_code"] = normalize_course_code(parsed.get("course_code"))
        parsed["title"] = _safe_str(parsed.get("title"))

        score, issues = quality_score(parsed)
        if score < config.IMPORT_MIN_QUALITY_SCORE:
            issue_text = ", ".join(issues) if issues else "unknown"
            return {
                "success": False,
                "needs_manual_input": True,
                "partial_data": parsed,
                "missing_fields": _identify_missing_fields(parsed),
                "quality_score": score,
                "quality_issues": issues,
                "extracted_text_preview": input_text[:500],
                "message": (
                    f"Import quality too low ({score}/100): {issue_text}. "
                    "Please review and correct manually."
                ),
            }

        course_code = parsed["course_code"]
        title = parsed["title"]
        uid = generate_course_uid(course_code, title)

        # 去重必须按 course_code，而不是 sha1(code|title)：
        # 后者只要标题略有差别就能把同一门课重复入库，
        # 这正是索引里积累出 100+ 条重复记录的原因之一。
        existing = _find_existing_by_code(enriched_index, course_code)
        if existing is not None:
            return {
                "success": False,
                "message": (
                    f"Course {course_code} already exists in database "
                    f"(\"{existing.get('title', '')}\")."
                ),
                "existing_course": {
                    "course_code": existing.get("course_code", ""),
                    "title": existing.get("title", ""),
                },
            }

        completed = complete_course_json(parsed, uid)

        courses_path = Path(courses_dir)
        courses_path.mkdir(parents=True, exist_ok=True)
        output_file = courses_path / f"{uid}.json"
        with output_file.open("w", encoding="utf-8") as f:
            json.dump(completed, f, ensure_ascii=False, indent=2)

        raw_entry = {
            "course_uid": uid,
            "course_code": completed["course_code"],
            "title": completed["title"],
            "file_name": output_file.name,
            "path": f"courses_flat/{output_file.name}",
        }
        enriched_entry = build_enriched_entry(raw_entry, completed)
        add_to_index(enriched_index, enriched_entry)
        save_enriched_index(enriched_index, enriched_index_path)

        return {
            "success": True,
            "course": {
                "course_code": completed["course_code"],
                "title": completed["title"],
                "points": completed["points_raw"],
                "description_length": len(completed.get("description", "")),
            },
            "message": f"Successfully imported {completed['course_code']}: {completed['title']}",
        }
    except Exception as exc:
        return {"success": False, "message": str(exc)}
