"""Optional Vapi path.

Vapi (a managed voice platform) runs the call - speech-to-text, the LLM, text-to-speech, the phone number - and calls this
server over HTTPS ("server URL") whenever its LLM wants to use a tool. Everything that matters stays the same code as the
built-in path: the shared booking state, validation, guard rails, the read-back + explicit "yes", and Google Calendar.

Shapes follow the public Vapi docs (server-url/events, tools/custom-tools, assistants):
  request : {"message": {"type": "tool-calls", "toolCallList": [{"id", "function": {"name", "arguments": {...}}}],
                         "artifact": {"messages": [{"role", "message"}...]}, "call": {"id", ...}}}
  response: {"results": [{"toolCallId", "result": "<single-line string>"}]}   (always HTTP 200; use "error" for failures)
The conversation is read from artifact.messages / conversation-update, so the caller's own words feed the same
deterministic extraction (dates, times, emails, names, purpose, yes/no) as in the built-in path.
"""
from __future__ import annotations

import asyncio
import hmac
import json
import logging
import time

from . import agent, config
from .agent import get_session
from .kb import kb

log = logging.getLogger("vapi")

TOOL_NAMES = ("check_available_slots", "update_booking_details", "book_appointment")
CALL_TO_SID: dict[str, str] = {}      # Vapi call id -> our session id (so the web form can follow a phone/web call)


def authorized(header_value: str | None) -> bool:
    """Vapi sends the shared secret in X-Vapi-Secret (we put it in server.headers when creating the assistant)."""
    if not config.VAPI_WEBHOOK_SECRET:
        return True
    return hmac.compare_digest(header_value or "", config.VAPI_WEBHOOK_SECRET)


# ------------------------------------------------------------------ what we create in Vapi (scripts/vapi_setup.py)
def system_prompt() -> str:
    open_days = ", ".join(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][d] for d in config.WORK_DAYS)
    kb_text = kb.full_text[:6000] or "(no knowledge base loaded)"
    return f"""You are {config.AGENT_NAME}, the receptionist at {config.BUSINESS_NAME}, on a live voice call.
Speak like a warm, natural person: 1-2 short sentences, contractions, no lists. English only. Say dates and times naturally
("Friday, November 20th at 3 PM"). End every reply with ONE question, except when saying goodbye.
Answer business questions ONLY from the BUSINESS FACTS below; if it is not there, say you are not sure and offer a team member.

BOOKING ({open_days}, {config.OPEN_TIME}-{config.CLOSE_TIME} {config.TIMEZONE}, {config.SLOT_MINUTES}-minute slots). Collect: date, time, full name, email, purpose.
- Session id for every tool call: {{{{sid}}}}  -> pass it as session_id, exactly.
- Whenever the caller states or CHANGES date, time, name, email or purpose, call update_booking_details with only what they said.
  Never invent values. Today is {{{{"now" | date: "%A %Y-%m-%d %H:%M", "{config.TIMEZONE}"}}}}.
- To offer times for a date call check_available_slots. Never guess availability or weekdays.
- Ask only for the next missing item, in this order: date, time, name, email, purpose. Never repeat a question you just asked.
- Every tool result may contain a field `say`. If it does, say exactly that text to the caller, word for word, and wait for the answer.
- When everything is known, call book_appointment. It reads a summary back; if the caller says yes, call book_appointment again.
- NEVER say a meeting is booked unless a book_appointment result says success true.
- If the caller wants a human, say you will arrange for a team member to call back and end politely.

BUSINESS FACTS:
{kb_text}"""


def first_message() -> str:
    return f"Hello, thank you for calling {config.BUSINESS_NAME}. This is {config.AGENT_NAME}. How can I help you today?"


def tool_payloads(server_url: str, secret: str | None) -> list[dict]:
    """Vapi 'function' tools whose server is this app (POST /tool)."""
    out = []
    for t in agent.TOOLS:
        f = t["function"]
        if f["name"] not in TOOL_NAMES:
            continue
        params = json.loads(json.dumps(f["parameters"]))
        params.setdefault("properties", {})["session_id"] = {
            "type": "string", "description": "The session id from the system prompt. Pass it exactly."}
        server = {"url": server_url, "timeoutSeconds": 20}
        if secret:
            server["headers"] = {"X-Vapi-Secret": secret}
        out.append({"type": "function", "async": False,
                    "function": {"name": f["name"], "description": TOOL_DESCRIPTIONS[f["name"]], "parameters": params},
                    "server": server})
    return out


TOOL_DESCRIPTIONS = {   # written for Vapi's LLM (the built-in path has its own wording in agent.TOOLS)
    "check_available_slots": "Get the free meeting times for one date (YYYY-MM-DD).",
    "update_booking_details": "Save what the caller just said or changed: date, time, name, email, purpose. "
                              "Pass ONLY what they stated in their latest message; call again when they correct something.",
    "book_appointment": "Finalize once all five details are known. Call it once to get the confirmation summary to read out, "
                        "and again after the caller says yes. Takes no arguments except session_id.",
}


def assistant_payload(server_url: str, secret: str | None, tool_ids: list[str], voice_id: str = "Elliot",
                      model: str = "gpt-4o-mini") -> dict:
    server = {"url": server_url, "timeoutSeconds": 20}
    if secret:
        server["headers"] = {"X-Vapi-Secret": secret}
    return {
        "name": f"{config.AGENT_NAME} - {config.BUSINESS_NAME}",
        "firstMessage": first_message(),
        "firstMessageMode": "assistant-speaks-first",
        "model": {"provider": "openai", "model": model, "temperature": 0.3, "maxTokens": 250,
                  "messages": [{"role": "system", "content": system_prompt()}], "toolIds": tool_ids},
        "voice": {"provider": "vapi", "voiceId": voice_id},
        "transcriber": {"provider": "deepgram", "model": "nova-2", "language": "en"},
        "server": server,
        "serverMessages": ["tool-calls", "conversation-update", "end-of-call-report", "status-update"],
        "maxDurationSeconds": 600,
        "silenceTimeoutSeconds": 30,
    }


# ------------------------------------------------------------------ webhook handling
def _text(m: dict) -> str:
    t = m.get("message") if m.get("message") is not None else m.get("content")
    return t.strip() if isinstance(t, str) else ""


async def ingest(session, messages: list[dict]) -> str | None:
    """Feed the caller's utterances (read from Vapi's conversation) through the SAME rules as the built-in path.
    Returns the ready-made reply for the newest utterance, if the rules have one."""
    session.external = True
    session.call_active = True
    reply, n_user, last_agent = None, 0, ""
    for m in messages or []:
        role, text = m.get("role"), _text(m)
        if not text:
            continue
        if role in ("assistant", "bot"):
            last_agent = text
        elif role == "user":
            n_user += 1
            if n_user <= session.vapi_processed:
                continue
            session.external_user_texts.append(text)
            session.external_last_agent = last_agent
            session.turn += 1
            session.tools_this_turn = 0
            session.last_user_text, session.via, session.forced_reply = text, "voice", None
            reply = await session._deterministic(text)
    session.vapi_processed = max(session.vapi_processed, n_user)
    session.last_active = time.time()
    return reply


def _flat(obj) -> str:
    """Vapi wants a flat single-line string as the tool result."""
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":")).replace("\n", " ")


def _session_for(sid: str, call: dict):
    call_id = (call or {}).get("id") or ""
    from_overrides = (((call or {}).get("assistantOverrides") or {}).get("variableValues") or {}).get("sid")
    sid = sid or from_overrides or CALL_TO_SID.get(call_id) or (f"vapi-{call_id}" if call_id else "vapi-unknown")
    if call_id:
        CALL_TO_SID[call_id] = sid
    number = ((call or {}).get("customer") or {}).get("number")
    return get_session(sid, number or "browser")


async def handle_tool_calls(msg: dict) -> dict:
    messages = (msg.get("artifact") or {}).get("messages") or msg.get("messages") or []
    results = []
    for tc in msg.get("toolCallList") or []:
        tc_id, fn = tc.get("id"), tc.get("function") or {}
        name, args = fn.get("name"), fn.get("arguments") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        sid = str(args.pop("session_id", "") or "").strip()
        if "{" in sid or sid.lower() in ("none", "null", "unknown", "undefined"):
            sid = ""                            # the {{sid}} variable was not filled in (e.g. a phone call): use the call id
        try:
            if name not in TOOL_NAMES:
                results.append({"toolCallId": tc_id, "error": f"Unknown tool {name}."})
                continue
            session = _session_for(sid, msg.get("call") or {})
            reply = await ingest(session, messages)
            session.forced_reply = None
            session.tools_this_turn = 0
            res = await asyncio.to_thread(session._run_tool, name, json.dumps(args))
            say = reply or session.forced_reply
            session.forced_reply = None
            if say:
                res = {**res, "say": say, "instruction": "Say the `say` text to the caller word for word, then wait."}
            results.append({"toolCallId": tc_id, "result": _flat(res)})
        except Exception:
            log.exception("tool %s failed", name)
            results.append({"toolCallId": tc_id, "error": "Something went wrong on our side. Ask the caller to repeat that."})
    return {"results": results}


async def handle_message(msg: dict) -> dict:
    """Dispatch one Vapi server message. Only tool-calls needs a real answer; the rest is informational."""
    typ = msg.get("type")
    if typ == "tool-calls":
        return await handle_tool_calls(msg)
    call = msg.get("call") or {}
    if typ == "conversation-update":
        # keep the shared booking state (and the web form) in step with what the caller says, tool call or not
        await ingest(_session_for("", call), msg.get("messages") or msg.get("conversation") or [])
    elif typ in ("end-of-call-report", "status-update"):
        ended = typ == "end-of-call-report" or msg.get("status") == "ended"
        sid = CALL_TO_SID.get(call.get("id", ""))
        if ended and sid and sid in agent.SESSIONS:
            agent.SESSIONS[sid].call_active = False
            log.info("Vapi call %s ended (%s)", call.get("id"), msg.get("endedReason", ""))
    return {}
