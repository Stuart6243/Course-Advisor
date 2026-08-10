"""
Groq 云 API 客户端。
免费计划：30 RPM, 14400 RPD (8b) / 1000 RPD (70b)
API 兼容 OpenAI 格式。
"""

import asyncio
import httpx
import json
import time
from typing import AsyncGenerator
import config


class GroqStreamProtocolError(RuntimeError):
    """Groq returned a malformed or prematurely terminated SSE stream."""


class GroqClient:
    """Groq API 客户端，接口与 OllamaClient 对齐。"""

    # 可用性探测结果缓存时长（秒）。避免每条消息都打一次 /models 探活，
    # 减少每次对话的额外网络往返和延迟。
    AVAILABILITY_TTL = 30

    # 瞬时错误（限流/网关抖动）自动重试一次，缓解 groq-only 模式下无本地兜底的问题。
    RETRY_STATUS = frozenset({429, 500, 502, 503, 529})
    RETRY_BACKOFF = 0.8

    def __init__(
        self,
        api_key: str = "",
        api_url: str = "",
        timeout: int = 15,
    ):
        self.api_key = api_key or config.GROQ_API_KEY
        self.api_url = api_url or config.GROQ_API_URL
        self.timeout = timeout or config.GROQ_TIMEOUT
        self._avail_cache: bool | None = None
        self._avail_ts: float = 0.0

    def _ensure_api_key(self) -> None:
        if not self.api_key:
            raise ValueError("GROQ_API_KEY is not set. Please configure it in your local environment.")

    async def is_available(self, force: bool = False) -> bool:
        """检查 Groq API 是否可用（带 TTL 缓存）。"""
        if not self.api_key:
            return False
        now = time.monotonic()
        if (
            not force
            and self._avail_cache is not None
            and (now - self._avail_ts) < self.AVAILABILITY_TTL
        ):
            return self._avail_cache
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(
                    "https://api.groq.com/openai/v1/models",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                )
                available = resp.status_code == 200
        except Exception:
            available = False
        self._avail_cache = available
        self._avail_ts = now
        return available

    async def chat(
        self,
        messages: list[dict],
        system_prompt: str = "",
        max_tokens: int = 512,
        model: str = "",
    ) -> str:
        """同步聊天，返回完整文本。瞬时错误自动重试一次。"""
        self._ensure_api_key()
        model = model or config.GROQ_RESPONSE_MODEL
        payload = self._build_payload(messages, system_prompt, max_tokens, model, stream=False)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        for attempt in range(2):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    resp = await client.post(self.api_url, headers=headers, json=payload)
                if resp.status_code in self.RETRY_STATUS and attempt == 0:
                    await asyncio.sleep(self.RETRY_BACKOFF)
                    continue
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"]
            except (httpx.TimeoutException, httpx.TransportError):
                if attempt == 0:
                    await asyncio.sleep(self.RETRY_BACKOFF)
                    continue
                raise
        # 不会走到这里，兜底避免 lint 报缺少返回值。
        raise RuntimeError("Groq request failed after retry")

    async def chat_stream(
        self,
        messages: list[dict],
        system_prompt: str = "",
        max_tokens: int = 512,
        model: str = "",
    ) -> AsyncGenerator[str, None]:
        """流式聊天，逐 token yield。

        瞬时错误（429/网关抖动/超时）只在「首个 token 之前」重试一次，
        已经流出内容后不再重试，避免重复输出。
        """
        self._ensure_api_key()
        model = model or config.GROQ_RESPONSE_MODEL
        payload = self._build_payload(messages, system_prompt, max_tokens, model, stream=True)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        for attempt in range(2):
            yielded = False
            try:
                completed = False
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    async with client.stream(
                        "POST", self.api_url, headers=headers, json=payload
                    ) as resp:
                        if resp.status_code in self.RETRY_STATUS and attempt == 0:
                            await resp.aread()  # 排空响应体后重试
                            await asyncio.sleep(self.RETRY_BACKOFF)
                            continue
                        resp.raise_for_status()
                        async for line in resp.aiter_lines():
                            if not line.startswith("data: "):
                                continue
                            data_str = line[6:]
                            if data_str.strip() == "[DONE]":
                                completed = True
                                break
                            try:
                                data = json.loads(data_str)
                            except json.JSONDecodeError as exc:
                                raise GroqStreamProtocolError(
                                    "Groq stream contained malformed JSON"
                                ) from exc
                            try:
                                choices = data.get("choices", [])
                                if not choices:
                                    continue
                                delta = choices[0].get("delta", {})
                                content = delta.get("content", "")
                            except (AttributeError, KeyError, IndexError, TypeError) as exc:
                                raise GroqStreamProtocolError(
                                    "Groq stream contained an invalid event"
                                ) from exc
                            if content:
                                yielded = True
                                yield str(content)
                if not completed:
                    raise GroqStreamProtocolError(
                        "Groq stream ended before the [DONE] marker"
                    )
                if not yielded:
                    raise GroqStreamProtocolError(
                        "Groq stream completed without answer content"
                    )
                return
            except (httpx.TimeoutException, httpx.TransportError, GroqStreamProtocolError):
                if not yielded and attempt == 0:
                    await asyncio.sleep(self.RETRY_BACKOFF)
                    continue
                raise

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
