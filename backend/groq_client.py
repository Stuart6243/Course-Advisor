"""
Groq 云 API 客户端。
免费计划：30 RPM, 14400 RPD (8b) / 1000 RPD (70b)
API 兼容 OpenAI 格式。
"""

import httpx
import json
from typing import AsyncGenerator
import config


class GroqClient:
    """Groq API 客户端，接口与 OllamaClient 对齐。"""

    def __init__(
        self,
        api_key: str = "",
        api_url: str = "",
        timeout: int = 15,
    ):
        self.api_key = api_key or config.GROQ_API_KEY
        self.api_url = api_url or config.GROQ_API_URL
        self.timeout = timeout or config.GROQ_TIMEOUT

    def _ensure_api_key(self) -> None:
        if not self.api_key:
            raise ValueError("GROQ_API_KEY is not set. Please configure it in your local environment.")

    async def is_available(self) -> bool:
        """检查 Groq API 是否可用。"""
        if not self.api_key:
            return False
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(
                    "https://api.groq.com/openai/v1/models",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                )
                return resp.status_code == 200
        except Exception:
            return False

    async def chat(
        self,
        messages: list[dict],
        system_prompt: str = "",
        max_tokens: int = 512,
        model: str = "",
    ) -> str:
        """同步聊天，返回完整文本。"""
        self._ensure_api_key()
        model = model or config.GROQ_RESPONSE_MODEL
        payload = self._build_payload(messages, system_prompt, max_tokens, model, stream=False)

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                self.api_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]

    async def chat_stream(
        self,
        messages: list[dict],
        system_prompt: str = "",
        max_tokens: int = 512,
        model: str = "",
    ) -> AsyncGenerator[str, None]:
        """流式聊天，逐 token yield。"""
        self._ensure_api_key()
        model = model or config.GROQ_RESPONSE_MODEL
        payload = self._build_payload(messages, system_prompt, max_tokens, model, stream=True)

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                "POST",
                self.api_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        delta = data["choices"][0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue

    def _build_payload(
        self,
        messages: list[dict],
        system_prompt: str,
        max_tokens: int,
        model: str,
        stream: bool,
    ) -> dict:
        msgs = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        msgs.extend(messages)

        return {
            "model": model,
            "messages": msgs,
            "max_tokens": max_tokens,
            "stream": stream,
            "temperature": 0.3,  # 低温度提高确定性
        }
