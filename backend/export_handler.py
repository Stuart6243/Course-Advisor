"""
对话导出功能。
将聊天消息数组转换为 Markdown 或 JSON 格式。
"""

from __future__ import annotations

import json
from datetime import datetime


def _speaker_label(role: str) -> str:
    if role == "user":
        return "You"
    if role == "assistant":
        return "Advisor AI"
    return role.capitalize() if role else "Message"


def export_as_markdown(messages: list[dict]) -> str:
    """将消息导出为 Markdown 格式。"""
    exported_on = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        "# Course Advisor AI - Chat Export",
        f"_Exported on {exported_on}_",
        "",
        "---",
        "",
    ]

    for idx, message in enumerate(messages):
        role = str(message.get("role", ""))
        content = str(message.get("content", ""))

        lines.append(f"**{_speaker_label(role)}**:")
        lines.append(content)
        lines.append("")

        if idx != len(messages) - 1:
            lines.append("---")
            lines.append("")

    return "\n".join(lines).strip() + "\n"


def export_as_json(messages: list[dict]) -> str:
    """将消息导出为格式化 JSON。"""
    payload = {
        "exported_at": datetime.now().replace(microsecond=0).isoformat(),
        "messages": messages,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)
