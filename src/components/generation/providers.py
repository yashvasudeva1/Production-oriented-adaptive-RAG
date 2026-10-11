from __future__ import annotations

import logging
import os
import random
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


def sanitize_secret(text: str) -> str:
    """Mask potential API keys and auth headers to keep logs safe."""
    text = re.sub(r"(Bearer\s+)[a-zA-Z0-9_\-\.]{8,}", r"\1[REDACTED]", text)
    text = re.sub(r"([a-zA-Z0-9_\-]{4})[a-zA-Z0-9_\-]{16,}([a-zA-Z0-9_\-]{4})", r"\1...[REDACTED]...\2", text)
    return text


@dataclass
class LLMResult:
    """Container for grounded LLM inference results."""
    text: str
    provider: str
    model: str
    latency_ms: float
    tokens_used: int = 0
    is_fallback: bool = False
    error: Optional[str] = None


class LLMProviderError(Exception):
    """Base exception for provider failures."""
    pass


class RateLimitError(LLMProviderError):
    """Raised on HTTP 429 rate limit errors."""
    def __init__(self, message: str, retry_after: Optional[float] = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class AuthenticationError(LLMProviderError):
    """Raised on HTTP 401/403 authorization failures."""
    pass


class ProviderTimeoutError(LLMProviderError):
    """Raised on connection or read timeouts."""
    pass


class BaseLLMProvider(ABC):
    """Abstract interface for answer-generation LLM providers."""

    def __init__(
        self,
        model_name: str,
        timeout: float = 10.0,
        max_retries: int = 2,
    ) -> None:
        self.model_name = model_name
        self.timeout = timeout
        self.max_retries = max_retries

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float = 0.0,
    ) -> LLMResult:
        """Generate response text given prompt and system instructions."""
        pass


class GroqProvider(BaseLLMProvider):
    """
    Groq free-tier provider adapter with exponential backoff and 429 Retry-After handling.
    Default model: llama-3.3-70b-versatile or llama-3.1-8b-instant.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "llama-3.3-70b-versatile",
        timeout: float = 10.0,
        max_retries: int = 2,
    ) -> None:
        super().__init__(model_name=model_name, timeout=timeout, max_retries=max_retries)
        self.api_key = api_key or os.getenv("GROQ_API_KEY", "")

    def generate(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float = 0.0,
    ) -> LLMResult:
        if not self.api_key:
            raise AuthenticationError("GROQ_API_KEY is not configured.")

        import requests

        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
        }

        last_err: Optional[Exception] = None
        t0 = time.perf_counter()

        for attempt in range(self.max_retries + 1):
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
                if resp.status_code == 200:
                    latency = (time.perf_counter() - t0) * 1000
                    data = resp.json()
                    choice = data["choices"][0]["message"]["content"].strip()
                    usage = data.get("usage", {}).get("total_tokens", 0)
                    return LLMResult(
                        text=choice,
                        provider="groq",
                        model=self.model_name,
                        latency_ms=latency,
                        tokens_used=usage,
                    )
                elif resp.status_code == 429:
                    retry_header = resp.headers.get("Retry-After")
                    retry_sec = float(retry_header) if retry_header and retry_header.isdigit() else 2.0
                    last_err = RateLimitError(
                        f"Groq rate limit exceeded (HTTP 429). Retry-After: {retry_sec}s",
                        retry_after=retry_sec,
                    )
                    if attempt < self.max_retries:
                        sleep_time = min(retry_sec, 4.0) + random.uniform(0.1, 0.5)
                        logger.warning(f"Groq 429 rate limit hit. Backing off for {sleep_time:.2f}s...")
                        time.sleep(sleep_time)
                        continue
                    raise last_err
                elif resp.status_code in (401, 403):
                    raise AuthenticationError(f"Groq authentication rejected with HTTP {resp.status_code}.")
                else:
                    msg = f"Groq HTTP {resp.status_code}: {sanitize_secret(resp.text[:200])}"
                    last_err = LLMProviderError(msg)
                    if resp.status_code >= 500 and attempt < self.max_retries:
                        sleep_time = (2 ** attempt) + random.uniform(0.1, 0.5)
                        time.sleep(sleep_time)
                        continue
                    raise last_err

            except requests.Timeout as te:
                last_err = ProviderTimeoutError(f"Groq request timed out after {self.timeout}s.")
                if attempt < self.max_retries:
                    time.sleep(1.0 + random.uniform(0.1, 0.5))
                    continue
                raise last_err
            except requests.RequestException as re_exc:
                last_err = LLMProviderError(f"Groq network error: {sanitize_secret(str(re_exc))}")
                if attempt < self.max_retries:
                    time.sleep(1.0)
                    continue
                raise last_err

        raise last_err or LLMProviderError("Groq inference failed after retries.")


class GeminiProvider(BaseLLMProvider):
    """
    Google Gemini free-tier provider adapter (Generative Language API).
    Default model: gemini-2.5-flash or gemini-1.5-flash.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-1.5-flash",
        timeout: float = 10.0,
        max_retries: int = 2,
    ) -> None:
        super().__init__(model_name=model_name, timeout=timeout, max_retries=max_retries)
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")

    def generate(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float = 0.0,
    ) -> LLMResult:
        if not self.api_key:
            raise AuthenticationError("GEMINI_API_KEY is not configured.")

        import requests

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.api_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "generationConfig": {
                "temperature": temperature,
            },
        }

        t0 = time.perf_counter()
        last_err: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):
            try:
                resp = requests.post(url, json=payload, timeout=self.timeout)
                if resp.status_code == 200:
                    latency = (time.perf_counter() - t0) * 1000
                    data = resp.json()
                    candidates = data.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        text = "".join(p.get("text", "") for p in parts).strip()
                        usage = data.get("usageMetadata", {}).get("totalTokenCount", 0)
                        return LLMResult(
                            text=text,
                            provider="gemini",
                            model=self.model_name,
                            latency_ms=latency,
                            tokens_used=usage,
                        )
                    raise LLMProviderError("Gemini returned empty candidates.")
                elif resp.status_code == 429:
                    last_err = RateLimitError("Gemini free-tier rate limit exceeded (HTTP 429).")
                    if attempt < self.max_retries:
                        sleep_time = (2 ** attempt) + random.uniform(0.1, 0.5)
                        time.sleep(sleep_time)
                        continue
                    raise last_err
                elif resp.status_code in (401, 403):
                    raise AuthenticationError(f"Gemini authentication failed (HTTP {resp.status_code}).")
                else:
                    msg = f"Gemini HTTP {resp.status_code}: {sanitize_secret(resp.text[:200])}"
                    last_err = LLMProviderError(msg)
                    if resp.status_code >= 500 and attempt < self.max_retries:
                        time.sleep(1.0 + random.uniform(0.1, 0.5))
                        continue
                    raise last_err

            except requests.Timeout:
                last_err = ProviderTimeoutError(f"Gemini request timed out after {self.timeout}s.")
                if attempt < self.max_retries:
                    time.sleep(1.0)
                    continue
                raise last_err
            except requests.RequestException as exc:
                last_err = LLMProviderError(f"Gemini network error: {sanitize_secret(str(exc))}")
                if attempt < self.max_retries:
                    time.sleep(1.0)
                    continue
                raise last_err

        raise last_err or LLMProviderError("Gemini inference failed.")


class OpenRouterProvider(BaseLLMProvider):
    """
    OpenRouter free models provider adapter.
    Default model: meta-llama/llama-3.2-3b-instruct:free.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "meta-llama/llama-3.2-3b-instruct:free",
        timeout: float = 12.0,
        max_retries: int = 2,
    ) -> None:
        super().__init__(model_name=model_name, timeout=timeout, max_retries=max_retries)
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY", "")

    def generate(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float = 0.0,
    ) -> LLMResult:
        if not self.api_key:
            raise AuthenticationError("OPENROUTER_API_KEY is not configured.")

        import requests

        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
        }

        t0 = time.perf_counter()
        last_err: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
                if resp.status_code == 200:
                    latency = (time.perf_counter() - t0) * 1000
                    data = resp.json()
                    choice = data["choices"][0]["message"]["content"].strip()
                    usage = data.get("usage", {}).get("total_tokens", 0)
                    return LLMResult(
                        text=choice,
                        provider="openrouter",
                        model=self.model_name,
                        latency_ms=latency,
                        tokens_used=usage,
                    )
                elif resp.status_code == 429:
                    last_err = RateLimitError("OpenRouter rate limit exceeded (HTTP 429).")
                    if attempt < self.max_retries:
                        time.sleep(2.0 + random.uniform(0.1, 0.5))
                        continue
                    raise last_err
                elif resp.status_code in (401, 403):
                    raise AuthenticationError(f"OpenRouter authentication failed (HTTP {resp.status_code}).")
                else:
                    msg = f"OpenRouter HTTP {resp.status_code}: {sanitize_secret(resp.text[:200])}"
                    last_err = LLMProviderError(msg)
                    if resp.status_code >= 500 and attempt < self.max_retries:
                        time.sleep(1.0)
                        continue
                    raise last_err

            except requests.Timeout:
                last_err = ProviderTimeoutError(f"OpenRouter request timed out after {self.timeout}s.")
                if attempt < self.max_retries:
                    time.sleep(1.0)
                    continue
                raise last_err
            except requests.RequestException as exc:
                last_err = LLMProviderError(f"OpenRouter network error: {sanitize_secret(str(exc))}")
                if attempt < self.max_retries:
                    time.sleep(1.0)
                    continue
                raise last_err

        raise last_err or LLMProviderError("OpenRouter inference failed.")


class OfflineExtractiveProvider(BaseLLMProvider):
    """
    Deterministic zero-cost offline extractive synthesis provider.
    Runs locally on CPU without external network requests or API keys.
    Extracts the most query-aligned sentences and grounds them with bracketed citations [n].
    """

    def __init__(self, model_name: str = "extractive-deterministic") -> None:
        super().__init__(model_name=model_name, timeout=0.0, max_retries=0)

    def generate(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float = 0.0,
    ) -> LLMResult:
        # Prompt is expected in format: Sources:\n{formatted_context}\n\nQuestion: {query}
        t0 = time.perf_counter()
        sources_match = re.search(
            r"Sources:[\s\n]*(.*?)(?:\n+Question:[\s\n]*(.*))?$",
            prompt,
            re.DOTALL | re.IGNORECASE,
        )
        if sources_match:
            sources_text = (sources_match.group(1) or "").strip()
            query = (sources_match.group(2) or "").strip()
        else:
            sources_text = prompt
            query = ""

        q_words = set(query.lower().split()) if query else set()

        # Parse source blocks: "Source [1] (filename, locator):\ntext..." or "[1] text..."
        source_blocks = re.split(r"(?:Source\s+)?\[(\d+)\](?:\s*\([^)]*\))?:?\s*", sources_text)
        sentences_with_citation: List[str] = []

        if len(source_blocks) >= 3:
            for i in range(1, len(source_blocks), 2):
                citation_id = source_blocks[i].strip()
                block_text = source_blocks[i + 1].strip() if i + 1 < len(source_blocks) else ""
                if not block_text:
                    continue

                sents = re.split(r"(?<=[.?!])\s+", block_text)
                best_sent = None
                best_overlap = -1

                for s in sents:
                    s_clean = s.strip()
                    if len(s_clean) < 15:
                        continue
                    if "ignore previous instructions" in s_clean.lower():
                        continue
                    overlap = len(set(s_clean.lower().split()) & q_words) if q_words else 1
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_sent = s_clean

                if best_sent:
                    sentences_with_citation.append(f"{best_sent} [{citation_id}]")
                elif sents:
                    first = sents[0].strip()
                    if first and "ignore previous instructions" not in first.lower():
                        sentences_with_citation.append(f"{first} [{citation_id}]")
        else:
            # Fallback: extract sentences from raw sources_text
            sents = re.split(r"(?<=[.?!])\s+", sources_text)
            for s in sents:
                s_clean = s.strip()
                if len(s_clean) < 15 or "ignore previous instructions" in s_clean.lower():
                    continue
                overlap = len(set(s_clean.lower().split()) & q_words) if q_words else 1
                if overlap > 0:
                    sentences_with_citation.append(f"{s_clean} [1]")
                    break

        latency = (time.perf_counter() - t0) * 1000

        if sentences_with_citation:
            ans = "Based on the retrieved evidence:\n\n" + "\n\n".join(sentences_with_citation[:3])
        else:
            ans = "No sufficient evidence found to answer the query."

        return LLMResult(
            text=ans,
            provider="offline",
            model="extractive-deterministic",
            latency_ms=latency,
            tokens_used=len(ans.split()),
        )


class ProviderRouter:
    """
    Resilient provider router with transparent fallback, rate-limit handling,
    and observable telemetry.
    """

    def __init__(
        self,
        primary_provider: str = "groq",
        primary_model: str = "llama-3.3-70b-versatile",
        fallback_provider: str = "offline",
        fallback_model: str = "",
        timeout: float = 10.0,
        max_retries: int = 2,
    ) -> None:
        self.primary_name = primary_provider.lower()
        self.primary_model = primary_model
        self.fallback_name = fallback_provider.lower()
        self.fallback_model = fallback_model
        self.timeout = timeout
        self.max_retries = max_retries

        self._providers: Dict[str, BaseLLMProvider] = {}
        self._init_providers()

    def _init_providers(self) -> None:
        # Register available provider adapters
        self._providers["offline"] = OfflineExtractiveProvider()
        self._providers["groq"] = GroqProvider(
            model_name=self.primary_model if self.primary_name == "groq" else "llama-3.3-70b-versatile",
            timeout=self.timeout,
            max_retries=self.max_retries,
        )
        self._providers["gemini"] = GeminiProvider(
            model_name=self.primary_model if self.primary_name == "gemini" else "gemini-1.5-flash",
            timeout=self.timeout,
            max_retries=self.max_retries,
        )
        self._providers["openrouter"] = OpenRouterProvider(
            model_name=self.primary_model if self.primary_name == "openrouter" else "meta-llama/llama-3.2-3b-instruct:free",
            timeout=self.timeout,
            max_retries=self.max_retries,
        )

    def generate(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float = 0.0,
    ) -> LLMResult:
        # Offline shortcut
        if os.getenv("OFFLINE_EVAL", "0").lower() in ("1", "true", "yes") or self.primary_name == "offline":
            return self._providers["offline"].generate(prompt, system_prompt, temperature)

        primary = self._providers.get(self.primary_name)
        if primary is None:
            logger.warning(f"Unknown primary LLM provider '{self.primary_name}', routing to offline fallback.")
            primary = self._providers["offline"]

        try:
            return primary.generate(prompt, system_prompt, temperature)
        except Exception as primary_exc:
            reason = sanitize_secret(str(primary_exc))
            logger.warning(
                f"[ProviderRouter] Primary provider '{self.primary_name}' ({getattr(primary, 'model_name', '')}) "
                f"failed: {reason}. Failing over to '{self.fallback_name}'."
            )

            fallback = self._providers.get(self.fallback_name) or self._providers["offline"]
            try:
                res = fallback.generate(prompt, system_prompt, temperature)
                res.is_fallback = True
                return res
            except Exception as fb_exc:
                logger.error(
                    f"[ProviderRouter] Fallback provider '{self.fallback_name}' also failed: {sanitize_secret(str(fb_exc))}. "
                    "Falling back to local offline extractive synthesis."
                )
                res = self._providers["offline"].generate(prompt, system_prompt, temperature)
                res.is_fallback = True
                return res
