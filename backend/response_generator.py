"""
LLM 回答生成模块。
将检索到的课程信息组织为 LLM 上下文，生成流式自然语言回答。
"""

from __future__ import annotations

from typing import Any, AsyncGenerator

import config


LANGUAGE_NAMES = {
    "en": "English",
    "zh": "Chinese (Simplified)",
    "es": "Spanish",
    "fr": "French",
}

ANTI_HALLUCINATION_PREAMBLE = """ABSOLUTE RULE: You are a Columbia University course advisor. You ONLY answer based on the course data provided below.
- If the provided course data is empty or does not match the question, respond with a course-search guidance message.
- NEVER generate encyclopedic/Wikipedia-style knowledge about any topic.
- NEVER fabricate course names, codes, instructors, or schedules that are not in the provided data."""

FOLLOWUP_GUIDANCE = """## Conversation Context Rules
- If the user refers to previous messages (e.g., "those", "the ones I mentioned", "which of those", "上面那些"), use the conversation history to understand references.
- You may reference courses discussed in prior turns of this conversation.
- If the user states a preference (e.g., "my favorite department is AERO"), remember it for follow-up questions.
- Even if no new course data is provided in this turn, you can still answer based on earlier turns."""

ANSWER_SYSTEM_PROMPT_TEMPLATE = """{anti_hallucination}
Respond in {language_name}.

Question: {original_question}
Type: {query_type}

Courses:
{formatted_courses}"""

EMPTY_RESULT_MESSAGES = {
    "en": "I couldn't find any Columbia courses matching your query. Try asking about a specific department (e.g., 'computer science courses'), course code (e.g., 'COMS W4111'), or instructor name.",
    "zh": "未找到匹配的哥大课程。请尝试提问具体的系别（如「计算机系课程」）、课程代码（如「COMS W4111」）或教授姓名。",
    "es": "No encontré cursos de Columbia que coincidan con tu consulta. Intenta preguntar sobre un departamento, código de curso o nombre de instructor específico.",
    "fr": "Je n'ai trouvé aucun cours de Columbia correspondant à votre recherche. Essayez de demander un département, un code de cours ou un nom d'instructeur spécifique.",
}

# 兼容旧测试与旧调用命名
NO_RESULTS_MESSAGES = EMPTY_RESULT_MESSAGES


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


def format_course_for_context(course: dict) -> str:
    """精简版课程上下文：减少 token 占用。"""
    code = (course.get("course_code") or "").strip() or "?"
    title = (course.get("title") or "").strip() or "?"
    points = _format_points(course)

    prereqs = (course.get("prerequisites_text") or "").strip()
    if len(prereqs) > 80:
        prereqs = prereqs[:80] + "..."
    prereq_line = prereqs if prereqs else "None"

    desc = (course.get("description") or "").strip()
    desc_line = ""
    if desc:
        if len(desc) > 100:
            desc = desc[:100] + "..."
        desc_line = f"\n  Desc: {desc}"

    section_lines = []
    for sec in (course.get("sections") or [])[:2]:
        term = (sec.get("term") or "").strip() or "?"
        times = (sec.get("times") or "").strip() or "TBA"
        instructor = (sec.get("instructor") or "").strip() or "TBA"
        location = (sec.get("location") or "").strip() or "TBA"
        current = sec.get("enrollment_current", "?")
        capacity = sec.get("enrollment_capacity", "?")
        section_lines.append(
            f"  {term}: {times}, {instructor}, {location}, {current}/{capacity}"
        )

    sections_text = "\n".join(section_lines) if section_lines else "  No sections"
    return f"[{code}] {title} | {points}\n  Prereqs: {prereq_line}{desc_line}\n{sections_text}"


def build_answer_prompt(
    intent: dict,
    courses: list[dict],
    language: str,
    conversation_history: list[dict[str, str]] | None = None,
) -> tuple[str, list[dict[str, str]]]:
    """构造发给 LLM 的 system prompt 和 messages。"""
    is_followup = conversation_history is not None and len(conversation_history) > 1
    courses_to_use = courses[: config.MAX_RETRIEVAL_RESULTS]

    formatted_courses = "(No courses found)"
    if courses_to_use:
        formatted_courses = "\n\n".join(
            format_course_for_context(c) for c in courses_to_use
        )
    elif is_followup:
        formatted_courses = (
            "(No new course data for this turn. Answer based on conversation history if applicable.)"
        )

    anti_hallucination = ANTI_HALLUCINATION_PREAMBLE
    if is_followup:
        anti_hallucination += f"\n\n{FOLLOWUP_GUIDANCE}"

    system_prompt = ANSWER_SYSTEM_PROMPT_TEMPLATE.format(
        anti_hallucination=anti_hallucination,
        language_name=_language_name(language),
        original_question=intent.get("original_question") or "",
        query_type=intent.get("query_type") or "general",
        formatted_courses=formatted_courses,
    )

    if conversation_history:
        messages = list(conversation_history)
    else:
        messages = [{"role": "user", "content": intent.get("original_question") or ""}]

    return system_prompt, messages


async def generate_response_stream(
    intent: dict,
    courses: list[dict],
    ollama: Any,
    language: str,
    conversation_history: list[dict[str, str]] | None = None,
) -> AsyncGenerator[str, None]:
    """流式生成回答。"""
    lang = language if language in LANGUAGE_NAMES else "en"
    is_followup = conversation_history is not None and len(conversation_history) > 1

    if not courses and not is_followup:
        yield EMPTY_RESULT_MESSAGES.get(lang, EMPTY_RESULT_MESSAGES["en"])
        return

    system_prompt, messages = build_answer_prompt(
        intent,
        courses,
        lang,
        conversation_history=conversation_history,
    )

    async for token in ollama.chat_stream(
        messages,
        system_prompt=system_prompt,
        max_tokens=config.RESPONSE_MAX_TOKENS,
    ):
        if token:
            yield token
