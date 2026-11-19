"""OpenAI-compatible chat client with free-tier friendly fallbacks.

Providers are tried in order: the main model, Groq fallback models, then Google Gemini (if GEMINI_API_KEY is set).
When a model returns HTTP 429 (rate limit) it is put on cooldown for the time the API asks, so later turns
skip it instantly instead of wasting a request.
"""
import asyncio
import logging
import re
import time

import httpx

from . import config

log = logging.getLogger("llm")
_client = httpx.AsyncClient(timeout=config.LLM_TIMEOUT)
_cooldown: dict[str, float] = {}  # model -> unix time when it can be used again

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai"


def _providers() -> list[tuple[str, str, str]]:
    """(base_url, api_key, model) in priority order."""
    # main model -> Gemini (better than the small Groq models) -> smaller Groq models
    out = [(config.LLM_BASE_URL, config.LLM_API_KEY, config.LLM_MODEL)]
    if config.GEMINI_API_KEY:
        out += [(GEMINI_URL, config.GEMINI_API_KEY, m) for m in config.GEMINI_MODELS]
    out += [(config.LLM_BASE_URL, config.LLM_API_KEY, m) for m in config.LLM_FALLBACK_MODELS if m != config.LLM_MODEL]
    return out


def _payload(model: str, messages: list[dict], tools: list[dict] | None) -> dict:
    p = {"model": model, "messages": messages, "temperature": 0.3, "max_tokens": 400}
    if "gpt-oss" in model:
        p["reasoning_effort"] = "low"
        p["include_reasoning"] = False
    if "gemini" in model:
        p["reasoning_effort"] = "none"  # fastest answers, no thinking tokens
    if tools:
        p["tools"] = tools
        p["tool_choice"] = "auto"
    return p


def _retry_after(r: httpx.Response) -> float:
    m = re.search(r"try again in (?:(\d+)m)?([\d.]+)s", r.text)
    if m:
        return int(m.group(1) or 0) * 60 + float(m.group(2))
    try:
        return float(r.headers.get("retry-after", 30))
    except ValueError:
        return 30


async def chat(messages: list[dict], tools: list[dict] | None = None) -> dict:
    last_err = "no LLM provider available"
    providers = _providers()
    # if every provider is cooling down, still try the one that frees up first
    ready = [p for p in providers if _cooldown.get(p[2], 0) <= time.time()] or \
        sorted(providers, key=lambda p: _cooldown.get(p[2], 0))[:1]
    for base, key, model in ready:
        for attempt in range(2):
            t0 = time.time()
            try:
                r = await _client.post(f"{base}/chat/completions", headers={"Authorization": f"Bearer {key}"},
                                       json=_payload(model, messages, tools))
            except httpx.HTTPError as e:
                last_err = f"{model}: {e!r}"
                log.warning(last_err)
                break
            if r.status_code == 429:
                wait = _retry_after(r)
                if attempt == 0 and wait <= 2:
                    await asyncio.sleep(wait + 0.2)
                    continue
                _cooldown[model] = time.time() + wait
                last_err = f"429 on {model}"
                log.warning("Rate limited on %s for %.0fs -> next provider", model, wait)
                break
            if r.status_code >= 400:
                last_err = f"LLM error {r.status_code} on {model}: {r.text[:300]}"
                log.warning(last_err)
                break
            msg = r.json()["choices"][0]["message"]
            if not (msg.get("content") or "").strip() and not msg.get("tool_calls"):
                last_err = f"{model} returned an empty answer"
                log.warning(last_err + " -> next provider")
                break
            log.info("LLM %s answered in %.1fs", model, time.time() - t0)
            return msg
    raise RuntimeError(last_err)
