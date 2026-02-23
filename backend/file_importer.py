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
from course_index import add_to_index, build_enriched_entry, save_enriched_index


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
  "description": "课程描述文字，没有则空字符串",
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
6. 如果文本包含多门课程的信息，只提取第一门"""


def _as_float(value: Any, default: float = 0.0) -> float:
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
    """验证转换结果的字段完整性。"""
    if not isinstance(data, dict):
        return False, "Invalid conversion result."

    if "error" in data:
        return False, _safe_str(data.get("error")) or "Conversion failed."

    course_code = _safe_str(data.get("course_code"))
    title = _safe_str(data.get("title"))
    if not course_code:
        return False, "Missing required field: course_code"
    if not title:
        return False, "Missing required field: title"

    has_letters = re.search(r"[A-Z]", course_code.upper()) is not None
    has_numbers = re.search(r"\d", course_code) is not None
    if not (has_letters and has_numbers):
        return False, "Invalid course_code format."

    points_min = data.get("points_min")
    points_max = data.get("points_max")
    if not isinstance(points_min, (int, float)) and not (
        isinstance(points_min, str) and points_min.strip()
    ):
        return False, "points_min must be a number."
    if not isinstance(points_max, (int, float)) and not (
        isinstance(points_max, str) and points_max.strip()
    ):
        return False, "points_max must be a number."

    return True, ""















def generate_course_uid(course_code: str, title: str) -> str:
    """生成课程唯一 ID。"""
    key = f"{course_code.strip()}|{title.strip()}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


def complete_course_json(data: dict, uid: str) -> dict:
    """补全课程 JSON 中的系统字段和默认字段。"""
    course_code = _safe_str(data.get("course_code"))
    title = _safe_str(data.get("title"))
    points_raw = _safe_str(data.get("points_raw"))
    points_min = _as_float(data.get("points_min"), default=0.0)
    points_max = _as_float(data.get("points_max"), default=0.0)

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
    missing = []
    if not data.get("course_code", "").strip():
        missing.append("course_code")
    if not data.get("title", "").strip():
        missing.append("title")
    if not data.get("points_min") and not data.get("points_raw"):
        missing.append("points")
    return missing








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

        input_text = extracted_text[:5000]
        messages = [{"role": "user", "content": input_text}]
        raw_response = await llm_client.chat(
            messages,
            system_prompt=CONVERSION_SYSTEM_PROMPT,
            max_tokens=config.RESPONSE_MAX_TOKENS,
        )

        parsed = parse_conversion_response(raw_response)
        
        
        # 在 import_file() 的 validate_course_json 失败分支中，改为：
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

        course_code = _safe_str(parsed.get("course_code"))
        title = _safe_str(parsed.get("title"))
        uid = generate_course_uid(course_code, title)

        exists = any((entry.get("course_uid") or "") == uid for entry in enriched_index)
        if exists:
            return {
                "success": False,
                "message": f"Course {course_code} already exists in database.",
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
            },
            "message": f"Successfully imported {completed['course_code']}: {completed['title']}",
        }
    except Exception as exc:
        return {"success": False, "message": str(exc)}
