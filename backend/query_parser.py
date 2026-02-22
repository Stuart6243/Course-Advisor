"""
LLM 意图提取模块。
从学生的自然语言问题中提取结构化查询条件。

工作原理：
  1. 用精心设计的 system prompt（含 few-shot examples）指导 LLM
  2. LLM 返回 JSON 格式的查询意图
  3. 容错解析：处理 LLM 可能的格式偏差
"""

from __future__ import annotations

import json
import re

from ollama_client import OllamaClient


EXTRACTION_SYSTEM_PROMPT = """你是一个查询条件提取器。你的唯一任务是从学生的课程相关问题中提取结构化查询条件。

只返回 JSON，不要任何其他文字。/no_think

JSON Schema：
{
  "query_type": "search | compare | recommend | detail | schedule | general",
  "course_codes": [],
  "keywords": [],
  "department": null,
  "instructor": null,
  "time_preference": null,
  "day_preference": [],
  "points_range": null,
  "term": null,
  "comparison_targets": [],
  "original_question": ""
}

字段说明：
- query_type: search=搜索课程, compare=对比课程, recommend=推荐课程, detail=课程详情, schedule=时间安排, general=通用问题
- course_codes: 用户明确提到的课程代码（如 "CIEN E3125"），没有就留空数组
- keywords: 从问题中提取的关键词（英文），用于模糊搜索
- department: 系别前缀，如 "CIEN","AERO","COMS","MECE","IEOR" 等。可从语义推断（"aerospace"→"AERO"）
- instructor: 教授姓名（姓即可）
- time_preference: "morning" | "afternoon" | "evening"，null 表示无偏好
- day_preference: 星期列表 ["Monday","Tuesday",...]
- points_range: [min, max]，如 [3,3] 表示恰好 3 学分
- term: "Spring 2026" | "Fall 2025" 等
- comparison_targets: 需要对比的课程代码列表（query_type=compare 时使用）
- original_question: 原封不动复制用户的问题

规则：
- 不确定的字段留 null 或空数组
- course_codes 只在用户明确写出课程代码时填写
- department 可以从关键词推断（"computer science"→"COMS", "mechanical"→"MECE"）
- 如果用户问的不是课程相关问题（如 "how do I register", "what's the weather"），query_type 设为 "general"

示例：

Q: "What 3-credit courses are offered on Tuesdays in Spring 2026?"
A: {"query_type":"search","course_codes":[],"keywords":[],"department":null,"instructor":null,"time_preference":null,"day_preference":["Tuesday"],"points_range":[3,3],"term":"Spring 2026","comparison_targets":[],"original_question":"What 3-credit courses are offered on Tuesdays in Spring 2026?"}

Q: "Tell me about Professor Panayotidi's courses"
A: {"query_type":"search","course_codes":[],"keywords":[],"department":null,"instructor":"Panayotidi","time_preference":null,"day_preference":[],"points_range":null,"term":null,"comparison_targets":[],"original_question":"Tell me about Professor Panayotidi's courses"}

Q: "Compare CIEN E3125 and ENME E3113"
A: {"query_type":"compare","course_codes":["CIEN E3125","ENME E3113"],"keywords":[],"department":null,"instructor":null,"time_preference":null,"day_preference":[],"points_range":null,"term":null,"comparison_targets":["CIEN E3125","ENME E3113"],"original_question":"Compare CIEN E3125 and ENME E3113"}

Q: "I'm interested in aerospace, what do you recommend?"
A: {"query_type":"recommend","course_codes":[],"keywords":["aerospace"],"department":"AERO","instructor":null,"time_preference":null,"day_preference":[],"points_range":null,"term":null,"comparison_targets":[],"original_question":"I'm interested in aerospace, what do you recommend?"}

Q: "What are the morning classes available?"
A: {"query_type":"search","course_codes":[],"keywords":[],"department":null,"instructor":null,"time_preference":"morning","day_preference":[],"points_range":null,"term":null,"comparison_targets":[],"original_question":"What are the morning classes available?"}

Q: "我想学机器人相关的课程"
A: {"query_type":"recommend","course_codes":[],"keywords":["robotics","robot"],"department":"MECE","instructor":null,"time_preference":null,"day_preference":[],"points_range":null,"term":null,"comparison_targets":[],"original_question":"我想学机器人相关的课程"}

Q: "How do I register for classes?"
A: {"query_type":"general","course_codes":[],"keywords":[],"department":null,"instructor":null,"time_preference":null,"day_preference":[],"points_range":null,"term":null,"comparison_targets":[],"original_question":"How do I register for classes?"}"""


DEFAULT_INTENT = {
    "query_type": "general",
    "course_codes": [],
    "keywords": [],
    "department": None,
    "instructor": None,
    "time_preference": None,
    "day_preference": [],
    "points_range": None,
    "term": None,
    "comparison_targets": [],
    "original_question": "",
}


def _default_intent_copy() -> dict:
    return {
        "query_type": "general",
        "course_codes": [],
        "keywords": [],
        "department": None,
        "instructor": None,
        "time_preference": None,
        "day_preference": [],
        "points_range": None,
        "term": None,
        "comparison_targets": [],
        "original_question": "",
    }


def _merge_with_default(parsed: dict) -> dict:
    merged = _default_intent_copy()
    for key in DEFAULT_INTENT:
        if key in parsed:
            merged[key] = parsed[key]

    # Defensive normalization for downstream filters.
    if not isinstance(merged.get("course_codes"), list):
        merged["course_codes"] = []
    if not isinstance(merged.get("keywords"), list):
        merged["keywords"] = []
    if not isinstance(merged.get("day_preference"), list):
        merged["day_preference"] = []
    if not isinstance(merged.get("comparison_targets"), list):
        merged["comparison_targets"] = []

    points = merged.get("points_range")
    if not (
        points is None
        or (isinstance(points, (list, tuple)) and len(points) == 2)
    ):
        merged["points_range"] = None
    return merged


def _try_parse_json(text: str) -> dict | None:
    if not text:
        return None
    try:
        parsed = json.loads(text.strip())
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _parse_by_json_block(text: str) -> dict | None:
    for match in re.finditer(r"\{.*?\}", text, flags=re.DOTALL):
        parsed = _try_parse_json(match.group(0))
        if parsed is not None:
            return parsed
    return None


def _parse_from_fenced_json(text: str) -> dict | None:
    match = re.search(
        r"```(?:json)?\s*(\{.*?\})\s*```",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if not match:
        return None
    return _try_parse_json(match.group(1))


def _remove_think_tags(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)


def parse_extraction_response(raw_text: str) -> dict:
    """
    从 LLM 返回文本中提取 JSON。

    容错处理链：
    1. 直接 json.loads(raw_text.strip())
    2. 失败 → 用正则 r'\\{.*\\}' (re.DOTALL) 找 JSON 块再 json.loads
    3. 失败 → 过滤掉 <think>...</think> 后重试步骤 1 和 2
    4. 全部失败 → 返回 DEFAULT_INTENT 的副本

    成功解析后，用 DEFAULT_INTENT 补全缺失字段。
    """
    text = raw_text or ""

    parsed = _try_parse_json(text)
    if parsed is None:
        parsed = _parse_from_fenced_json(text)
    if parsed is None:
        parsed = _parse_by_json_block(text)
    if parsed is None:
        without_think = _remove_think_tags(text)
        parsed = _try_parse_json(without_think)
        if parsed is None:
            parsed = _parse_from_fenced_json(without_think)
        if parsed is None:
            parsed = _parse_by_json_block(without_think)

    if parsed is None:
        return _default_intent_copy()

    return _merge_with_default(parsed)


async def extract_query_intent(question: str, ollama: OllamaClient) -> dict:
    """
    完整意图提取流程。

    1. 构造 messages: [{"role":"user","content":question}]
    2. 调用 ollama.chat(messages, system_prompt=EXTRACTION_SYSTEM_PROMPT)
    3. parse_extraction_response 解析返回
    4. 确保 original_question 字段等于输入的 question
    5. 返回解析后的 intent dict

    异常处理：Ollama 调用失败时返回 DEFAULT_INTENT + original_question
    """
    messages = [{"role": "user", "content": question}]
    try:
        raw_response = await ollama.chat(
            messages=messages,
            system_prompt=EXTRACTION_SYSTEM_PROMPT,
            max_tokens=256,
        )
        intent = parse_extraction_response(raw_response)
    except Exception:
        intent = _default_intent_copy()

    intent["original_question"] = question
    return intent
