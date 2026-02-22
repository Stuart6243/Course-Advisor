

"""
LLM 回答生成模块。
将检索到的课程信息组织为 LLM 上下文，生成流式自然语言回答。
"""

from __future__ import annotations

from typing import AsyncGenerator

from ollama_client import OllamaClient

import config


LANGUAGE_NAMES = {
    "en": "English",
    "zh": "Chinese (Simplified)",
    "es": "Spanish",
    "fr": "French",
}

ANSWER_SYSTEM_PROMPT_TEMPLATE = """You are a Columbia University AI Course Advisor. Answer based ONLY on the course data below. Do not invent information. If data is insufficient, say so. Respond in {language_name}.

Question: {original_question}
Type: {query_type}

Courses:
{formatted_courses}"""

NO_RESULTS_MESSAGES = {
    "en": "I couldn't find any courses matching your criteria. Could you try rephrasing your question or broadening your search? For example, you could ask about a specific department, instructor, or time slot.",
    "zh": "很抱歉，没有找到符合您条件的课程。您可以尝试换一种方式提问，或者扩大搜索范围。比如可以问某个系别、某位教授或某个时间段的课程。",
    "es": "No pude encontrar cursos que coincidan con tus criterios. ¿Podrías reformular tu pregunta o ampliar tu búsqueda?",
    "fr": "Je n'ai pas trouvé de cours correspondant à vos critères. Pourriez-vous reformuler votre question ou élargir votre recherche?",
}


def _language_name(language: str) -> str:
    return LANGUAGE_NAMES.get(language, LANGUAGE_NAMES["en"])


def _format_points(course: dict) -> str:
    raw = (course.get("points_raw") or "").strip()
    if raw:
        return raw

    min_points = course.get("points_min")
    max_points = course.get("points_max")
    if min_points is None and max_points is None:
        return "N/A"
    if min_points == max_points:
        return f"{min_points}"
    return f"{min_points}-{max_points}"


# 修改 4a: format_course_for_context — 精简到 ~100 tokens/课
# 替换原来的 format_course_for_context 函数
# ============================================================

def format_course_for_context(course: dict) -> str:
    """精简版：每门课控制在 ~100 tokens。"""
    code = (course.get("course_code") or "").strip() or "?"
    title = (course.get("title") or "").strip() or "?"
    points = _format_points(course)

    # 先修课：只取前 80 字符
    prereqs = (course.get("prerequisites_text") or "").strip()
    if len(prereqs) > 80:
        prereqs = prereqs[:80] + "..."
    prereq_line = prereqs if prereqs else "None"

    # 描述：只取前 100 字符
    desc = (course.get("description") or "").strip()
    desc_line = ""
    if desc:
        if len(desc) > 100:
            desc = desc[:100] + "..."
        desc_line = f"\n  Desc: {desc}"

    # Sections：最多 2 个
    section_lines = []
    for sec in (course.get("sections") or [])[:2]:
        term = (sec.get("term") or "").strip() or "?"
        times = (sec.get("times") or "").strip() or "TBA"
        instructor = (sec.get("instructor") or "").strip() or "TBA"
        location = (sec.get("location") or "").strip() or "TBA"
        current = sec.get("enrollment_current", "?")
        capacity = sec.get("enrollment_capacity", "?")
        section_lines.append(f"  {term}: {times}, {instructor}, {location}, {current}/{capacity}")

    sections_text = "\n".join(section_lines) if section_lines else "  No sections"

    return f"[{code}] {title} | {points}\n  Prereqs: {prereq_line}{desc_line}\n{sections_text}"



# ============================================================
# 修改 4b: build_answer_prompt + generate_response_stream
# 替换原来的这两个函数
# ============================================================

def build_answer_prompt(intent: dict, courses: list[dict], language: str) -> tuple[str, list[dict]]:
    """
    构造发给 LLM 的 system prompt 和 messages。
    课程数量限制在前 5 门。
    """
    # 关键改动：只取前 5 门课
    courses_to_use = courses[:config.MAX_RETRIEVAL_RESULTS]

    formatted_courses = "(No courses found)"
    if courses_to_use:
        formatted_courses = "\n\n".join(format_course_for_context(c) for c in courses_to_use)

    system_prompt = ANSWER_SYSTEM_PROMPT_TEMPLATE.format(
        language_name=_language_name(language),
        original_question=intent.get("original_question") or "",
        query_type=intent.get("query_type") or "general",
        formatted_courses=formatted_courses,
    )
    messages = [
        {
            "role": "user",
            "content": intent.get("original_question") or "",
        }
    ]
    return system_prompt, messages


async def generate_response_stream(
    intent: dict,
    courses: list[dict],
    ollama: OllamaClient,
    language: str,
) -> AsyncGenerator[str, None]:
    """
    流式生成回答。
    """
    query_type = (intent.get("query_type") or "general").lower()
    lang = language if language in LANGUAGE_NAMES else "en"

    if query_type != "general" and not courses:
        yield NO_RESULTS_MESSAGES.get(lang, NO_RESULTS_MESSAGES["en"])
        return

    if query_type == "general":
        system_prompt = (
            "You are a Columbia University AI Course Advisor. "
            "If the question is outside your scope, say so clearly and suggest asking course-related questions. "
            f"Respond in {_language_name(lang)}."
        )
        messages = [{"role": "user", "content": intent.get("original_question") or ""}]
    else:
        system_prompt, messages = build_answer_prompt(intent, courses, lang)

    # 关键改动：传入 max_tokens=512 限制输出长度
    async for token in ollama.chat_stream(messages, system_prompt=system_prompt, max_tokens=config.RESPONSE_MAX_TOKENS):
        if token:
            yield token
