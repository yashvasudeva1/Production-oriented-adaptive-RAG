from __future__ import annotations

import json
import logging
import os
import random
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
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

    _last_request_time: float = 0.0

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "qwen/qwen3.8-27b",
        timeout: float = 16.0,
        max_retries: int = 5,
        min_request_interval: float = 7.0,
    ) -> None:
        super().__init__(model_name=model_name, timeout=timeout, max_retries=max_retries)
        self.api_key = api_key or os.getenv("GROQ_API_KEY", "")
        self.min_request_interval = min_request_interval

    def generate(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float = 0.0,
    ) -> LLMResult:
        if not self.api_key:
            raise AuthenticationError("GROQ_API_KEY is not configured.")

        # Enforce rate limit interval for free tier TPM / RPM limits
        now = time.time()
        elapsed = now - GroqProvider._last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        GroqProvider._last_request_time = time.time()

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
            "max_tokens": 1024,
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
                    reset_tokens = resp.headers.get("x-ratelimit-reset-tokens")
                    retry_sec = 3.0
                    if reset_tokens:
                        m_sec = re.search(r"([\d\.]+)s", reset_tokens)
                        if m_sec:
                            retry_sec = float(m_sec.group(1))
                    elif retry_header and retry_header.replace(".", "", 1).isdigit():
                        # Cap at 5.0s because Retry-After often contains the whole day/hour reset
                        retry_sec = min(float(retry_header), 5.0)

                    last_err = RateLimitError(
                        f"Groq rate limit exceeded (HTTP 429). Retry-After: {retry_sec}s",
                        retry_after=retry_sec,
                    )
                    if attempt < self.max_retries:
                        sleep_time = min(max(retry_sec, 3.0), 6.0) + (attempt * 1.5) + random.uniform(0.2, 0.8)
                        logger.warning(f"Groq 429 rate limit hit. Backing off for {sleep_time:.2f}s...")
                        time.sleep(sleep_time)
                        GroqProvider._last_request_time = time.time()
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
    Default model: gemini-2.5-flash-lite or gemini-flash-lite-latest.
    """

    _last_request_time: float = 0.0

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-3.8-flash",
        timeout: float = 14.0,
        max_retries: int = 3,
        min_request_interval: float = 4.1,
    ) -> None:
        super().__init__(model_name=model_name, timeout=timeout, max_retries=max_retries)
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.min_request_interval = min_request_interval

    def generate(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float = 0.0,
    ) -> LLMResult:
        if not self.api_key:
            raise AuthenticationError("GEMINI_API_KEY is not configured.")

        # Enforce rate limit interval for free tier (15 requests/minute = 1 req per 4.0s)
        now = time.time()
        elapsed = now - GeminiProvider._last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        GeminiProvider._last_request_time = time.time()

        import requests

        clean_model = self.model_name.replace("models/", "")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{clean_model}:generateContent?key={self.api_key}"
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
                        sleep_time = 12.0 + (attempt * 6.0) + random.uniform(0.5, 2.0)
                        logger.warning(f"Gemini 429 rate limit hit. Backing off for {sleep_time:.2f}s...")
                        time.sleep(sleep_time)
                        GeminiProvider._last_request_time = time.time()
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
    Default model: nvidia/nemotron-3-super-120b-a12b:free.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "nvidia/nemotron-3-super-120b-a12b:free",
        timeout: float = 20.0,
        max_retries: int = 3,
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


class OpenCodeProvider(BaseLLMProvider):
    """
    OpenCode local server provider adapter (opencode serve/API).
    Default endpoint: http://127.0.0.1:4096 or discovered from ~/.local/state/opencode/service.json.
    Default model: opencode/nemotron-3.5-lightning-free.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model_name: str = "opencode/nemotron-3.5-lightning-free",
        timeout: float = 20.0,
        max_retries: int = 2,
    ) -> None:
        super().__init__(model_name=model_name, timeout=timeout, max_retries=max_retries)
        self.base_url = base_url or os.getenv("OPENCODE_API_URL", "")
        self.password = os.getenv("OPENCODE_SERVER_PASSWORD", "")

        # Auto-discover local opencode service if not explicitly specified
        if not self.base_url:
            svc_path = Path.home() / ".local" / "state" / "opencode" / "service.json"
            if svc_path.exists():
                try:
                    with open(svc_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        self.base_url = data.get("url", "http://127.0.0.1:4096")
                        if not self.password:
                            self.password = data.get("password", "")
                except Exception:
                    pass
        if not self.base_url:
            self.base_url = "http://127.0.0.1:4096"

    def generate(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float = 0.0,
    ) -> LLMResult:
        import requests

        url = f"{self.base_url.rstrip('/')}/api/experimental/generate"
        auth = ("opencode", self.password) if self.password else None
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "opencode/2.0.18",
        }

        clean_model = self.model_name.replace("opencode/", "")
        payload = {
            "prompt": f"{system_prompt}\n\n{prompt}",
            "model": {
                "providerID": "opencode",
                "id": clean_model,
            },
        }

        t0 = time.perf_counter()
        last_err: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):
            try:
                resp = requests.post(url, headers=headers, json=payload, auth=auth, timeout=self.timeout)
                if resp.status_code == 200:
                    latency = (time.perf_counter() - t0) * 1000
                    data = resp.json()
                    choice = data.get("data", {}).get("text", "").strip()
                    return LLMResult(
                        text=choice,
                        provider="opencode",
                        model=self.model_name,
                        latency_ms=latency,
                        tokens_used=len(choice.split()),
                    )
                elif resp.status_code == 429:
                    last_err = RateLimitError("OpenCode rate limit exceeded (HTTP 429).")
                    if attempt < self.max_retries:
                        time.sleep(2.0)
                        continue
                    raise last_err
                elif resp.status_code in (401, 403):
                    raise AuthenticationError(f"OpenCode server authorization rejected (HTTP {resp.status_code}).")
                else:
                    msg = f"OpenCode HTTP {resp.status_code}: {sanitize_secret(resp.text[:200])}"
                    last_err = LLMProviderError(msg)
                    if attempt < self.max_retries and resp.status_code >= 500 and resp.status_code != 503:
                        time.sleep(1.0)
                        continue
                    raise last_err
            except requests.Timeout:
                last_err = ProviderTimeoutError(f"OpenCode request timed out after {self.timeout}s.")
                if attempt < self.max_retries:
                    time.sleep(1.0)
                    continue
                raise last_err
            except requests.RequestException as re_exc:
                last_err = LLMProviderError(f"OpenCode connection error: {sanitize_secret(str(re_exc))}")
                if attempt < self.max_retries:
                    time.sleep(1.0)
                    continue
                raise last_err

        raise last_err or LLMProviderError("OpenCode inference failed.")


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

        q_lower = query.lower()
        q_words = set(re.findall(r"\b[a-zA-Z0-9_\-\.]{3,}\b", q_lower))

        # Identify target answer categories
        is_multi_hop = "multi-hop" in system_prompt.lower() or len(query.split()) >= 12
        needs_ranking = bool(re.search(r"\b(rank|tallest|largest|highest|lowest|first|second|third|fourth|fifth|last)\b", q_lower))
        needs_year = bool(re.search(r"\b(year|years|when|date|month)\b", q_lower))
        needs_quantity = bool(re.search(r"\b(how many|how much|duration|count|percentage|length|rate|difference|score|dimension|marks)\b", q_lower))
        needs_name = bool(re.search(r"\b(who|whose|whom|which person|name of the|founder|president|first lady|inventor|author|director|creator|ceo)\b", q_lower))

        # Parse source blocks: "Source [1] (filename, locator):\ntext..." or "[1] text..."
        source_blocks = re.split(r"(?:Source\s+)?\[(\d+)\](?:\s*\([^)]*\))?:?\s*", sources_text)
        sentences_with_citation: List[str] = []
        extracted_facts: List[Tuple[str, str, int]] = []
        scored_candidates: List[Tuple[float, str]] = []

        seen_canonical = set()
        if len(source_blocks) >= 3:
            for i in range(1, len(source_blocks), 2):
                citation_id = source_blocks[i].strip()
                block_text = source_blocks[i + 1].strip() if i + 1 < len(source_blocks) else ""
                if not block_text:
                    continue

                # Normalize whitespace and strip internal linebreaks within the block
                block_text = re.sub(r"[\r\n]+", " ", block_text)
                sents = re.split(r"(?<=[.?!])\s+", block_text)
                scored_sents = []
                for s in sents:
                    s_clean = re.sub(r"\s+", " ", s.strip())
                    # Strip leading section headers like "3.2.2 Multi-Head Attention " or "Table 3: "
                    s_clean = re.sub(
                        r"^(?:\d+(?:\.\d+)*\s+[A-Z][A-Za-z\s-]{2,40}\b|Table\s+\d+:?|Figure\s+\d+:?)\s*",
                        "",
                        s_clean,
                    ).strip()
                    if len(s_clean) < 15 or "ignore previous instructions" in s_clean.lower():
                        continue

                    # Deduplication check
                    canonical = re.sub(r"[^a-zA-Z0-9]", "", s_clean.lower())[:80]
                    if canonical in seen_canonical:
                        continue

                    s_words = set(re.findall(r"\b[a-zA-Z0-9_\-\.]{3,}\b", s_clean.lower()))
                    overlap = len(s_words & q_words)

                    salience = overlap * 2
                    if needs_ranking and re.search(r"\b\d+(?:st|nd|rd|th)\b|\b(tallest|largest|fourth|first)\b", s_clean, re.IGNORECASE):
                        salience += 6
                    if needs_year and re.search(r"\b(18|19|20)\d{2}\b", s_clean):
                        salience += 5
                    if needs_quantity and re.search(r"\b\d+(?:,\d+)*(?:\.\d+)?%?\b", s_clean):
                        salience += 5
                    if needs_name and re.search(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", s_clean):
                        salience += 4

                    scored_sents.append((salience, s_clean, canonical))

                if scored_sents:
                    scored_sents.sort(key=lambda x: x[0], reverse=True)
                    best_sal, best_sent, best_can = scored_sents[0]
                    seen_canonical.add(best_can)
                    extracted_facts.append((citation_id, best_sent, best_sal))

                    # Extract candidate answer spans from high-salience sentences with prioritized scoring
                    if needs_ranking:
                        ord_matches = re.findall(r"\b\d+(?:st|nd|rd|th)\b", best_sent, re.IGNORECASE)
                        for om in ord_matches:
                            scored_candidates.append((best_sal + 25.0, om))
                    if needs_quantity:
                        qty_matches = re.findall(r"\b\d+(?:,\d+)*(?:\.\d+)?(?:\s+(?:years?|days?|months?|cm|minutes?|seconds?|percent|%))?\b", best_sent, re.IGNORECASE)
                        for qm in qty_matches:
                            if qm.strip() and qm.strip().lower() not in q_lower:
                                scored_candidates.append((best_sal + 20.0, qm.strip()))
                    if needs_year:
                        year_matches = re.findall(r"\b(18|19|20)\d{2}\b", best_sent)
                        for ym in year_matches:
                            if ym not in query:
                                scored_candidates.append((best_sal + 15.0, ym))
                    if needs_name:
                        name_matches = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", best_sent)
                        for nm in name_matches:
                            if not all(w in q_lower for w in nm.lower().split()):
                                scored_candidates.append((best_sal + 15.0, nm))
                    if not (needs_ranking or needs_quantity or needs_year or needs_name):
                        for nm in re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", best_sent):
                            if len(nm) >= 3 and nm.lower() not in q_lower:
                                scored_candidates.append((best_sal + 5.0, nm))

            # Ensure multi-hop representation across sources
            for cit_id, sent, _ in extracted_facts:
                sentences_with_citation.append(f"{sent} [{cit_id}]")
        else:
            block_text = re.sub(r"[\r\n]+", " ", sources_text)
            sents = re.split(r"(?<=[.?!])\s+", block_text)
            for s in sents:
                s_clean = re.sub(r"\s+", " ", s.strip())
                s_clean = re.sub(
                    r"^(?:\d+(?:\.\d+)*\s+[A-Z][A-Za-z\s-]{2,40}\b|Table\s+\d+:?|Figure\s+\d+:?)\s*",
                    "",
                    s_clean,
                ).strip()
                if len(s_clean) < 15 or "ignore previous instructions" in s_clean.lower():
                    continue
                canonical = re.sub(r"[^a-zA-Z0-9]", "", s_clean.lower())[:80]
                if canonical in seen_canonical:
                    continue
                seen_canonical.add(canonical)
                sentences_with_citation.append(f"{s_clean} [1]")
                if len(sentences_with_citation) >= 3:
                    break

        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        answer_candidates = [cand for _, cand in scored_candidates]

        latency = (time.perf_counter() - t0) * 1000

        if sentences_with_citation:
            if len(sentences_with_citation) == 1:
                base_answer = f"According to the source documentation, {sentences_with_citation[0]}"
            else:
                formatted_items = "\n".join([f"- {s}" for s in sentences_with_citation[:4]])
                base_answer = f"Based on the retrieved evidence:\n\n{formatted_items}"
            if is_multi_hop and answer_candidates:
                top_ans = answer_candidates[0]
                ans = f"{base_answer}\n\n**Key Finding**: {top_ans}"
            else:
                ans = base_answer
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
        timeout: float = 16.0,
        max_retries: int = 5,
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
            model_name=self.primary_model if self.primary_name == "groq" else (self.fallback_model if self.fallback_model and "qwen" not in self.fallback_model else "llama-3.3-70b-versatile"),
            timeout=self.timeout,
            max_retries=self.max_retries,
        )
        self._providers["gemini"] = GeminiProvider(
            model_name=self.primary_model if self.primary_name == "gemini" else "gemini-2.5-flash-lite",
            timeout=self.timeout,
            max_retries=self.max_retries,
        )
        self._providers["openrouter"] = OpenRouterProvider(
            model_name=self.primary_model if self.primary_name == "openrouter" else (self.fallback_model if self.fallback_name == "openrouter" and self.fallback_model else "nvidia/nemotron-3-super-120b-a12b:free"),
            timeout=self.timeout,
            max_retries=self.max_retries,
        )
        self._providers["opencode"] = OpenCodeProvider(
            model_name=self.primary_model if self.primary_name == "opencode" else (self.fallback_model if self.fallback_name == "opencode" and self.fallback_model else "opencode/nemotron-3.5-lightning-free"),
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

            fallback = self._providers.get(self.fallback_name)
            if fallback is not None and fallback != primary:
                try:
                    res = fallback.generate(prompt, system_prompt, temperature)
                    res.is_fallback = True
                    return res
                except Exception as fb_exc:
                    logger.warning(
                        f"[ProviderRouter] Fallback provider '{self.fallback_name}' failed: {sanitize_secret(str(fb_exc))}."
                    )

            # If both primary and fallback failed (or fallback was offline), try OpenRouter as secondary online safety net
            if self.primary_name != "openrouter" and self.fallback_name != "openrouter" and os.getenv("OPENROUTER_API_KEY"):
                try:
                    logger.info("[ProviderRouter] Attempting tertiary online failover via OpenRouter...")
                    or_prov = self._providers.get("openrouter")
                    if or_prov:
                        res = or_prov.generate(prompt, system_prompt, temperature)
                        res.is_fallback = True
                        return res
                except Exception as or_exc:
                    logger.warning(f"[ProviderRouter] OpenRouter tertiary failover failed: {sanitize_secret(str(or_exc))}")

            # Ultimate fallback to offline extractive
            logger.error("[ProviderRouter] All online providers failed. Falling back to offline extractive synthesis.")
            res = self._providers["offline"].generate(prompt, system_prompt, temperature)
            res.is_fallback = True
            return res
