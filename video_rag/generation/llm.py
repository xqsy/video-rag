from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from collections.abc import Mapping
import re
import time
from typing import Any

from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from video_rag.config import settings


Message = dict[str, str]


class OpenRouterRateLimitError(RuntimeError):
    pass


def _is_rate_limit_error(exc: Exception) -> bool:
    if isinstance(exc, OpenRouterRateLimitError):
        return True
    if getattr(exc, "status_code", None) == 429:
        return True
    text = str(exc).lower()
    return "rate limit" in text or "free-models-per-min" in text


def _should_retry_exception(exc: Exception) -> bool:
    return not _is_rate_limit_error(exc)


def _extract_rate_limit_reset_ms(value: Any) -> int | None:
    if isinstance(value, Mapping):
        error = value.get("error")
        if isinstance(error, Mapping):
            metadata = error.get("metadata")
            if isinstance(metadata, Mapping):
                headers = metadata.get("headers")
                if isinstance(headers, Mapping):
                    reset = headers.get("X-RateLimit-Reset")
                    if isinstance(reset, str) and reset.isdigit():
                        return int(reset)
        return None
    if isinstance(value, str):
        match = re.search(r"X-RateLimit-Reset['\"]?:\s*['\"]?(\d{10,})", value)
        if match:
            return int(match.group(1))
    return None


class OpenRouterLLM:
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self.api_key = api_key or settings.openrouter_api_key
        self.model = model or settings.openrouter_model
        self.base_url = base_url or settings.openrouter_base_url
        self.timeout = timeout or settings.request_timeout_seconds
        self._client = None

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception(_should_retry_exception),
    )
    def generate(
        self,
        messages: Sequence[Message],
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1200,
    ) -> str:
        requested_model = model or self.model
        response = self._create_completion(
            model=requested_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        choice = response.choices[0] if response.choices else None
        if choice is not None and getattr(choice, "finish_reason", None) == "length" and max_tokens < 4800:
            response = self._create_completion(
                model=requested_model,
                messages=messages,
                temperature=temperature,
                max_tokens=min(max_tokens * 2, 4800),
            )
            choice = response.choices[0] if response.choices else None
        if choice is None or choice.message is None:
            raise RuntimeError("OpenRouter returned an empty response")
        return self._coerce_content(choice.message.content)

    def stream_generate(
        self,
        messages: Sequence[Message],
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1200,
    ) -> Iterator[str]:
        stream = self._create_completion(
            model=model or self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            content = getattr(delta, "content", None)
            if isinstance(content, str) and content:
                yield content

    def _create_completion(
        self,
        *,
        model: str,
        messages: Sequence[Message],
        temperature: float,
        max_tokens: int,
        stream: bool = False,
    ):
        try:
            return self._get_client().chat.completions.create(
                model=model,
                messages=list(messages),
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream,
            )
        except Exception as exc:
            if _is_rate_limit_error(exc):
                raise OpenRouterRateLimitError(self._format_rate_limit_message(exc)) from exc
            raise

    def _format_rate_limit_message(self, exc: Exception) -> str:
        reset_ms = _extract_rate_limit_reset_ms(getattr(exc, "body", None))
        if reset_ms is None:
            reset_ms = _extract_rate_limit_reset_ms(str(exc))
        message = "Превышен лимит запросов OpenRouter для бесплатной модели."
        if reset_ms is not None:
            wait_seconds = max(0, int((reset_ms - time.time() * 1000 + 999) // 1000))
            if 0 < wait_seconds <= 180:
                message += f" Повторите попытку примерно через {wait_seconds} сек."
            elif wait_seconds > 180:
                message += " Повторите попытку позже."
        return message

    def _get_client(self):
        if self._client is None:
            if not self.api_key:
                raise RuntimeError(
                    "OPENROUTER_API_KEY is not configured. Add it to .env before using generation."
                )
            from openai import OpenAI

            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
                default_headers={
                    "HTTP-Referer": "https://localhost/video-rag",
                    "X-Title": settings.app_name,
                },
            )
        return self._client

    def _coerce_content(self, content: Any) -> str:
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, Iterable):
            parts: list[str] = []
            for item in content:
                text = getattr(item, "text", None)
                if text:
                    parts.append(str(text))
                    continue
                if isinstance(item, dict) and item.get("text"):
                    parts.append(str(item["text"]))
            return "".join(parts).strip()
        return str(content).strip()
