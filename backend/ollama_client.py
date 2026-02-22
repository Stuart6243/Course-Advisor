"""
Ollama API 通信封装。
处理与本地 Ollama 服务的所有 HTTP 交互。
"""

from __future__ import annotations

import json
import re
from typing import AsyncGenerator

import httpx


class OllamaClient:
    """Ollama API 客户端，支持非流式和流式调用。"""

    def __init__(self, base_url: str, model: str, timeout: int):
        """初始化客户端。"""
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    @staticmethod
    def _strip_think_tags(text: str) -> str:
        """移除 Qwen3 的思考标签 <think>...</think>。"""
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    def _build_messages(self, messages: list[dict], system_prompt: str) -> list[dict]:
        if not system_prompt:
            return list(messages)
        return [{"role": "system", "content": system_prompt}, *messages]

    @staticmethod
    def _runtime_error_message(response: httpx.Response) -> str:
        try:
            payload = response.json()
            if isinstance(payload, dict) and payload.get("error"):
                return str(payload["error"])
        except Exception:
            pass
        return f"Ollama API error: HTTP {response.status_code}"

    async def chat(self, messages: list[dict], system_prompt: str = "", max_tokens: int = 0) -> str:
        """非流式聊天。发送消息列表，返回完整回答文本。"""
        request_messages = self._build_messages(messages, system_prompt)
        body = {
            "model": self.model,
            "messages": request_messages,
            "stream": False,
        }
        if max_tokens > 0:
            body["options"] = {"num_predict": max_tokens}
		
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(f"{self.base_url}/api/chat", json=body)
        except httpx.TimeoutException as exc:
            raise TimeoutError("Ollama request timed out") from exc
        except httpx.RequestError as exc:
            raise ConnectionError("Failed to connect to Ollama") from exc

        if response.status_code >= 400:
            raise RuntimeError(self._runtime_error_message(response))

        try:
            data = response.json()
            content = data["message"]["content"]
        except Exception as exc:
            raise RuntimeError("Invalid Ollama chat response format") from exc

        return self._strip_think_tags(str(content))

    async def chat_stream(
        self, messages: list[dict], system_prompt: str = "", max_tokens: int = 0
    ) -> AsyncGenerator[str, None]:
        """流式聊天。逐个 yield token 字符串。"""
        request_messages = self._build_messages(messages, system_prompt)
        body = {
            "model": self.model,
            "messages": request_messages,
            "stream": True,
        }
        if max_tokens > 0:
            body["options"] = {"num_predict": max_tokens}

        start_tag = "<think>"
        end_tag = "</think>"
        outside_tail = ""
        inside_think = False
        inside_tail = ""

        def filter_stream_text(text: str, final: bool = False) -> str:
            nonlocal outside_tail, inside_think, inside_tail
            output: list[str] = []
            remaining = text

            while remaining:
                if inside_think:
                    segment = inside_tail + remaining
                    end_idx = segment.find(end_tag)
                    if end_idx == -1:
                        keep = min(len(segment), len(end_tag) - 1)
                        inside_tail = segment[-keep:] if keep else ""
                        remaining = ""
                    else:
                        inside_think = False
                        inside_tail = ""
                        remaining = segment[end_idx + len(end_tag) :]
                else:
                    segment = outside_tail + remaining
                    start_idx = segment.find(start_tag)
                    if start_idx == -1:
                        if final:
                            output.append(segment)
                            outside_tail = ""
                            remaining = ""
                        else:
                            keep_len = len(start_tag) - 1
                            flush_len = max(0, len(segment) - keep_len)
                            output.append(segment[:flush_len])
                            outside_tail = segment[flush_len:]
                            remaining = ""
                    else:
                        output.append(segment[:start_idx])
                        outside_tail = ""
                        inside_think = True
                        remaining = segment[start_idx + len(start_tag) :]

            if final:
                if not inside_think and outside_tail:
                    output.append(outside_tail)
                    outside_tail = ""
                if inside_think:
                    inside_tail = ""

            return "".join(output)

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream(
                    "POST", f"{self.base_url}/api/chat", json=body
                ) as response:
                    if response.status_code >= 400:
                        message = self._runtime_error_message(response)
                        raise RuntimeError(message)

                    async for line in response.aiter_lines():
                        if not line:
                            continue

                        try:
                            chunk = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        content = str(chunk.get("message", {}).get("content", ""))
                        if content:
                            cleaned = filter_stream_text(content, final=False)
                            if cleaned:
                                yield cleaned

                        if chunk.get("done") is True:
                            tail = filter_stream_text("", final=True)
                            if tail:
                                yield tail
                            break
        except httpx.TimeoutException as exc:
            raise TimeoutError("Ollama request timed out") from exc
        except httpx.RequestError as exc:
            raise ConnectionError("Failed to connect to Ollama") from exc

    async def is_available(self) -> bool:
        """检查 Ollama 服务是否在线且目标模型存在。"""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            data = response.json()
            models = data.get("models", [])
            for model in models:
                if not isinstance(model, dict):
                    continue
                name = model.get("name") or model.get("model")
                if name == self.model:
                    return True
            return False
        except Exception:
            return False
