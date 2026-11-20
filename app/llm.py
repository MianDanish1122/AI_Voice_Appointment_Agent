"""OpenAI-compatible chat client with free-tier friendly fallbacks and optional token streaming.

Providers are tried in order: the main model, Gemini (if GEMINI_API_KEY is set), then smaller Groq models.
When a model returns HTTP 429 (rate limit) it is put on cooldown for the time the API asks, so later turns
skip it instantly instead of wasting a request. An empty answer counts as a failure.

Streaming: Groq and Gemini both implement the OpenAI `stream=true` server-sent-events protocol, so when the
caller passes `on_text` the answer's text is delivered token by token (the agent speaks it sentence by
sentence). This is LLM-output streaming only - speech recognition and text-to-speech are separate stages.
"""
import asyncio
import json
import logging
import re
import time

import httpx

from . import config

log = logging.getLogger("llm")
# One shared client = connection reuse (TLS handshake paid once, not on every turn).
_client = httpx.AsyncClient(timeout=httpx.Timeout(config.LLM_TIMEOUT, connect=8.0),
                            limits=httpx.Limits(max_keepalive_connections=8, keepalive_expiry=120))
_cooldown: dict[str, float] = {}  # model -> unix time when it can be used again

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai"


def _providers() -> list[tuple[str, str, str]]:
    """(base_url, api_key, model) in priority order."""
    out = [(config.LLM_BASE_URL, config.LLM_API_KEY, config.LLM_MODEL)]
    if config.GEMINI_API_KEY:
        out += [(GEMINI_URL, config.GEMINI_API_KEY, m) for m in config.GEMINI_MODELS]
    out += [(config.LLM_BASE_URL, config.LLM_API_KEY, m) for m in config.LLM_FALLBACK_MODELS if m != config.LLM_MODEL]
    return out


def _payload(model: str, messages: list[dict], tools: list[dict] | None) -> dict:
    p = {"model": model, "messages": messages, "temperature": 0.3, "max_tokens": 300}
    if "gpt-oss" in model:
        p["reasoning_effort"] = "low"
        p["include_reasoning"] = False
    if "gemini" in model:
        p["reasoning_effort"] = "none"  # fastest answers, no thinking tokens
    if tools:
        p["tools"] = tools
        p["tool_choice"] = "auto"
    return p


def _retry_after(text: str, headers) -> float:
    m = re.search(r"try again in (?:(\d+)m)?([\d.]+)s", text)
    if m:
        return int(m.group(1) or 0) * 60 + float(m.group(2))
    try:
        return float(headers.get("retry-after", 30))
    except ValueError:
        return 30


async def _post(base, key, model, messages, tools):
    r = await _client.post(f"{base}/chat/completions", headers={"Authorization": f"Bearer {key}"},
                           json=_payload(model, messages, tools))
    if r.status_code >= 400:
        return r.status_code, r.text, r.headers, None
    return 200, "", r.headers, r.json()["choices"][0]["message"]


async def _post_stream(base, key, model, messages, tools, on_text, stats):
    payload = _payload(model, messages, tools)
    payload["stream"] = True
    t0 = time.perf_counter()
    async with _client.stream("POST", f"{base}/chat/completions", headers={"Authorization": f"Bearer {key}"},
                              json=payload) as r:
        if r.status_code >= 400:
            return r.status_code, (await r.aread()).decode("utf-8", "ignore"), r.headers, None
        content: list[str] = []
        calls: dict[int, dict] = {}
        async for line in r.aiter_lines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                delta = (json.loads(data).get("choices") or [{}])[0].get("delta") or {}
            except (json.JSONDecodeError, IndexError):
                continue
            if delta.get("content"):
                if "ttft_ms" not in stats:
                    stats["ttft_ms"] = int((time.perf_counter() - t0) * 1000)
                content.append(delta["content"])
                on_text(delta["content"])
            for tc in delta.get("tool_calls") or []:
                idx = tc.get("index", 0)
                if tc.get("id") and idx in calls and calls[idx]["id"] and calls[idx]["id"] != tc["id"]:
                    idx = max(calls) + 1                      # providers that omit/repeat `index`
                cur = calls.setdefault(idx, {"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                if tc.get("id"):
                    cur["id"] = tc["id"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    cur["function"]["name"] += fn["name"]
                if fn.get("arguments"):
                    cur["function"]["arguments"] += fn["arguments"]
                if "ttft_ms" not in stats:
                    stats["ttft_ms"] = int((time.perf_counter() - t0) * 1000)
    msg = {"role": "assistant", "content": "".join(content) or None}
    if calls:
        msg["tool_calls"] = [dict(v, id=v["id"] or f"call_{i}") for i, v in sorted(calls.items())]
    return 200, "", r.headers, msg


async def chat(messages: list[dict], tools: list[dict] | None = None, on_text=None, stats: dict | None = None) -> dict:
    """Return the assistant message. `on_text(delta)` enables streaming; `stats` receives model/ttft/total ms."""
    stats = stats if stats is not None else {}
    last_err = "no LLM provider available"
    providers = _providers()
    # if every provider is cooling down, still try the one that frees up first
    ready = [p for p in providers if _cooldown.get(p[2], 0) <= time.time()] or \
        sorted(providers, key=lambda p: _cooldown.get(p[2], 0))[:1]
    emitted = False

    def forward(delta: str):
        nonlocal emitted
        emitted = True
        on_text(delta)

    for base, key, model in ready:
        for attempt in range(2):
            t0 = time.perf_counter()
            stats.pop("ttft_ms", None)
            try:
                if on_text:
                    status, body, headers, msg = await _post_stream(base, key, model, messages, tools, forward, stats)
                else:
                    status, body, headers, msg = await _post(base, key, model, messages, tools)
            except httpx.HTTPError as e:
                last_err = f"{model}: {e!r}"
                log.warning(last_err)
                if emitted:
                    raise RuntimeError(last_err)
                break
            if status == 429:
                wait = _retry_after(body, headers)
                if attempt == 0 and wait <= 2:
                    await asyncio.sleep(wait + 0.2)
                    continue
                _cooldown[model] = time.time() + wait
                last_err = f"429 on {model}"
                log.warning("Rate limited on %s for %.0fs -> next provider", model, wait)
                break
            if status >= 400:
                last_err = f"LLM error {status} on {model}: {body[:300]}"
                log.warning(last_err)
                if emitted:
                    raise RuntimeError(last_err)
                break
            if not (msg.get("content") or "").strip() and not msg.get("tool_calls"):
                last_err = f"{model} returned an empty answer"
                log.warning(last_err + " -> next provider")
                if emitted:
                    raise RuntimeError(last_err)
                break
            stats.update(model=model, total_ms=int((time.perf_counter() - t0) * 1000), streamed=bool(on_text))
            log.info("LLM %s answered in %.1fs%s", model, time.perf_counter() - t0, " (streamed)" if on_text else "")
            return msg
    raise RuntimeError(last_err)


async def warm():
    """Open the TLS connection to the providers before the first call (saves ~0.3-0.8 s on the first turn)."""
    async def ping(base, key):
        try:
            await _client.get(f"{base}/models", headers={"Authorization": f"Bearer {key}"}, timeout=10)
        except Exception as e:  # warm-up only
            log.info("LLM warm-up skipped (%s): %r", base, e)
    jobs = [ping(config.LLM_BASE_URL, config.LLM_API_KEY)]
    if config.GEMINI_API_KEY:
        jobs.append(ping(GEMINI_URL, config.GEMINI_API_KEY))
    await asyncio.gather(*jobs)
