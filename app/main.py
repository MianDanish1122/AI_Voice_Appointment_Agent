import asyncio
import json
import logging
import uuid
from xml.sax.saxutils import escape

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import calendar_store, config, gcal, llm, vapi
from .agent import SESSIONS, end_session, get_session
from .booking import pretty_when
from .kb import kb

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
log = logging.getLogger("main")

app = FastAPI(title="AI Call Agent")
app.mount("/static", StaticFiles(directory=config.ROOT / "static"), name="static")   # tokens.css for the page

_BG: set = set()


def spawn(coro) -> asyncio.Task:
    """create_task() that keeps a strong reference: the event loop only holds weak references to tasks, so a
    fire-and-forget task can be garbage-collected mid-flight (we saw a stream that never finished)."""
    task = asyncio.create_task(coro)
    _BG.add(task)
    task.add_done_callback(_BG.discard)
    return task


# ---------------------------------------------------------------- Twilio phone calls
def _say(text: str) -> str:
    return f'<Say voice="{config.TTS_VOICE}" language="{config.TTS_LANGUAGE}">{escape(text)}</Say>'


def _listen(text: str) -> str:
    """Speak text, then listen for the caller's speech. If silent, go to /twilio/noinput."""
    return (
        f'<Gather input="speech" action="/twilio/gather" method="POST" speechTimeout="2" timeout="6" '
        f'speechModel="phone_call" language="{config.STT_LANGUAGE}">{_say(text)}</Gather>'
        f'<Redirect method="POST">/twilio/noinput</Redirect>'
    )


def _twiml(body: str) -> Response:
    return Response(f'<?xml version="1.0" encoding="UTF-8"?><Response>{body}</Response>', media_type="application/xml")


def _after_reply(session, reply: str) -> Response:
    if session.action == "transfer":
        end_session(session.id)
        return _twiml(_say(reply) + f"<Dial>{escape(config.HUMAN_PHONE)}</Dial>")
    if session.action == "hangup":
        end_session(session.id)
        return _twiml(_say(reply) + "<Hangup/>")
    return _twiml(_listen(reply))


@app.post("/twilio/voice")
async def twilio_voice(request: Request):
    form = await request.form()
    sid, caller = form.get("CallSid", str(uuid.uuid4())), form.get("From")
    log.info("Incoming call %s from %s", sid, caller)
    session = get_session(sid, caller)
    return _twiml(_listen(session.greeting()))


@app.post("/twilio/gather")
async def twilio_gather(request: Request):
    form = await request.form()
    sid, caller = form.get("CallSid", ""), form.get("From")
    speech = (form.get("SpeechResult") or "").strip()
    session = get_session(sid, caller)
    if not speech:
        return _twiml(_listen("Sorry, I didn't catch that. Could you say it again?"))
    log.info("[%s] caller: %s", sid, speech)
    reply = await session.respond(speech)
    log.info("[%s] agent: %s", sid, reply)
    return _after_reply(session, reply)


@app.post("/twilio/noinput")
async def twilio_noinput(request: Request):
    form = await request.form()
    session = get_session(form.get("CallSid", ""), form.get("From"))
    session.silence = getattr(session, "silence", 0) + 1
    if session.silence >= 2:
        end_session(session.id)
        return _twiml(_say("I haven't heard anything, so I'll end the call now. Goodbye!") + "<Hangup/>")
    return _twiml(_listen("Are you still there? How can I help you?"))


@app.post("/twilio/status")
async def twilio_status(request: Request):
    form = await request.form()
    if form.get("CallStatus") in ("completed", "failed", "busy", "no-answer", "canceled"):
        end_session(form.get("CallSid", ""))
    return Response(status_code=204)


# ---------------------------------------------------------------- Web app: voice + typed chat + editable form
# All three feed ONE shared booking state per session (see app/booking.py).
class StartReq(BaseModel):
    silent: bool = False        # create the session without a greeting (the form is used on its own)
    session_id: str | None = None   # reuse a session (e.g. the form was filled before the call started)


class WebMsg(BaseModel):
    session_id: str
    text: str
    via: str = "voice"          # voice | typed


class StateReq(BaseModel):
    session_id: str
    fields: dict


class KeyReq(BaseModel):
    session_id: str
    key: str = ""


class SidReq(BaseModel):
    session_id: str


class ResetReq(BaseModel):
    session_id: str
    keep_contact: bool = True   # "Book another" keeps name + email; "Clear" wipes everything


@app.get("/api/config")
async def web_config():
    tz = config.TIMEZONE
    return {
        "business": config.BUSINESS_NAME, "agent": config.AGENT_NAME, "timezone": tz,
        "timezone_label": calendar_store.now().strftime("%Z") or tz, "open": config.OPEN_TIME, "close": config.CLOSE_TIME,
        "slot_minutes": config.SLOT_MINUTES, "work_days": config.WORK_DAYS,
        "today": calendar_store.now().date().isoformat(), "max_days": 60,
        "calendar": "google" if gcal.enabled() else "local", "invites": gcal.oauth_enabled() if gcal.enabled() else False,
        "model": config.LLM_MODEL, "human_transfer": bool(config.HUMAN_PHONE),
        # the public key is meant for browsers; the private key and webhook secret never leave the server
        "vapi": {"enabled": bool(config.VAPI_PUBLIC_KEY and config.VAPI_ASSISTANT_ID),
                 "public_key": config.VAPI_PUBLIC_KEY if config.VAPI_ASSISTANT_ID else "",
                 "assistant_id": config.VAPI_ASSISTANT_ID},
    }


# ---------------------------------------------------------------- Vapi (optional managed voice platform)
@app.post("/api/vapi/webhook")
async def vapi_webhook(request: Request):
    """Vapi's 'server URL'. Same booking rules as the built-in path; tool results are always HTTP 200 (Vapi's contract)."""
    if not vapi.authorized(request.headers.get("x-vapi-secret")):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({}, status_code=200)
    return JSONResponse(await vapi.handle_message(body.get("message") or {}))


class LinkReq(BaseModel):
    call_id: str
    session_id: str


@app.post("/api/vapi/link")
async def vapi_link(req: LinkReq):
    """The page tells us which Vapi call belongs to which web session, so the form follows the call."""
    vapi.CALL_TO_SID[req.call_id] = req.session_id
    return {"ok": True}


@app.post("/api/web/start")
async def web_start(req: StartReq | None = None):
    sid = (req.session_id if req and req.session_id else None) or "web-" + uuid.uuid4().hex[:8]
    session = get_session(sid, "browser")
    reply = None if (req and req.silent) else session.greeting()
    return {"session_id": sid, "reply": reply, "action": None, "state": await session.public_state()}


@app.post("/api/web/end")
async def web_end(req: SidReq):
    """The call ended; keep the booking state so the form still works."""
    session = get_session(req.session_id, "browser")
    session.call_active = False
    session.pending_key = session.awaiting = None
    return await session.public_state()


def _finish_payload(session, reply: str, state: dict, timings: dict) -> dict:
    action = session.action
    return {"reply": reply, "action": action, "human_phone": config.HUMAN_PHONE if action == "transfer" else None,
            "state": state, "timings": timings}


@app.post("/api/web/message")
async def web_message(msg: WebMsg):
    """Same as /api/web/stream but returns one JSON answer when the reply is complete."""
    session = get_session(msg.session_id, "browser")
    timings: dict = {}
    reply = await session.respond(msg.text, via=msg.via, timings=timings)
    out = _finish_payload(session, reply, await session.public_state(fresh=True), timings)
    if session.action:
        end_session(session.id)
    return out


@app.post("/api/web/stream")
async def web_stream(msg: WebMsg):
    """The reply as newline-delimited JSON: {t:sentence}* {t:state}* then {t:done}. Each sentence is sent as soon as
    the LLM has produced it (providers stream tokens), so speech can start before the whole answer exists."""
    session = get_session(msg.session_id, "browser")
    loop = asyncio.get_running_loop()
    q: asyncio.Queue = asyncio.Queue()
    timings: dict = {}

    def push(item):                      # safe from the event loop and from worker threads (tool calls)
        loop.call_soon_threadsafe(q.put_nowait, item)

    async def run():
        try:
            reply = await session.respond(msg.text, via=msg.via, emit=lambda s: push({"t": "sentence", "text": s}),
                                          on_state=lambda: push({"t": "state_dirty"}), timings=timings)
            payload = _finish_payload(session, reply, await session.public_state(fresh=True), timings)
            if session.action:
                end_session(session.id)
            push({"t": "done", **payload})
        except Exception as e:
            log.exception("stream failed")
            push({"t": "error", "error": str(e)})
        finally:
            push(None)

    spawn(run())

    async def gen():
        while True:
            item = await q.get()
            if item is None:
                break
            if item["t"] == "state_dirty":
                item = {"t": "state", "state": await session.public_state()}
            yield json.dumps(item) + "\n"

    return StreamingResponse(gen(), media_type="application/x-ndjson",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/web/state/{sid}")
async def web_get_state(sid: str):
    return await get_session(sid, "browser").public_state()      # polled every ~1 s during a Vapi call: cached availability is fine


@app.post("/api/web/state")
async def web_set_state(req: StateReq):
    """The caller edited the form. Manual input wins; the agent continues from the corrected value."""
    return await get_session(req.session_id, "browser").apply_manual(req.fields)


@app.post("/api/web/review")
async def web_review(req: SidReq):
    """Validate everything server-side and return the appointment summary to show before confirming."""
    session = get_session(req.session_id, "browser")
    state = await session.public_state(fresh=True)
    st, v = session.booking, session.booking.values
    issues = [f"{f}: {st.errors[f]}" if f in st.errors else f"{f} is missing" for f in st.missing()]
    slot = state["slot"] or {}
    if not issues and not slot.get("ok"):
        issues.append(slot.get("error") or "That time is not available.")
    if st.booked and st.booked.get("key") == st.key():
        issues.append("This appointment is already booked.")
    summary = None if issues else {
        "when": pretty_when(v["date"], v["time"]), "name": v["name"], "email": v["email"], "purpose": v["purpose"],
        "timezone": config.TIMEZONE, "calendar": "Google Calendar" if gcal.enabled() else "local calendar only",
    }
    return {"ok": not issues, "issues": issues, "summary": summary, "key": st.key(), "state": state}


@app.post("/api/web/confirm")
async def web_confirm(req: KeyReq):
    """Book. Only for exactly the summary the user reviewed (key), and success is reported only after the
    calendar API confirmed the event."""
    session = get_session(req.session_id, "browser")
    res = await session.confirm_from_ui(req.key)
    return {"success": bool(res.get("success")), "result": res, "state": await session.public_state(fresh=True)}


@app.post("/api/web/reset")
async def web_reset(req: ResetReq):
    session = get_session(req.session_id, "browser")
    session.booking.new_booking(keep_contact=req.keep_contact)
    session.pending_key = session.awaiting = None
    return await session.public_state()


# ---------------------------------------------------------------- Natural voice (free Edge neural TTS)
_tts_cache: dict[str, bytes] = {}


@app.on_event("startup")
async def _warm_tts():
    """Warm everything the first caller would otherwise wait for: LLM + Google connections (TLS, access token)
    and the greeting audio, so the first turn is as fast as the rest."""
    spawn(llm.warm())
    asyncio.get_running_loop().run_in_executor(None, gcal.warm)

    async def warm():
        from .agent import CallSession
        text = CallSession("warm").greeting()
        try:
            import edge_tts
            data = b"".join([c["data"] async for c in edge_tts.Communicate(text, config.WEB_TTS_VOICE, rate="+8%").stream()
                             if c["type"] == "audio"])
            _tts_cache[text] = data
            log.info("Greeting audio cached (%d bytes)", len(data))
        except Exception as e:
            log.warning("Could not pre-generate greeting audio: %r", e)
    spawn(warm())


@app.get("/api/tts")
async def tts(text: str):
    """Stream MP3 speech for `text`. The page falls back to the browser voice if this fails."""
    text = text.strip()[:800]
    if text in _tts_cache:
        return Response(_tts_cache[text], media_type="audio/mpeg")
    import edge_tts

    stream = edge_tts.Communicate(text, config.WEB_TTS_VOICE, rate="+8%").stream()
    try:  # get the first audio chunk before answering so errors become a clean 503
        first = b""
        while not first:
            chunk = await stream.__anext__()
            if chunk["type"] == "audio":
                first = chunk["data"]
    except Exception as e:
        log.warning("TTS failed: %r", e)
        return Response(status_code=503)

    async def body():
        parts = [first]
        yield first
        try:
            async for chunk in stream:
                if chunk["type"] == "audio":
                    parts.append(chunk["data"])
                    yield chunk["data"]
        except Exception as e:
            log.warning("TTS stream broke: %r", e)
            return
        if len(text) < 200 and len(_tts_cache) < 100:
            _tts_cache[text] = b"".join(parts)

    return StreamingResponse(body(), media_type="audio/mpeg")


# ---------------------------------------------------------------- Admin
@app.get("/api/bookings")
async def bookings():
    return calendar_store.list_bookings()


@app.get("/api/slots/{date}")
async def slots(date: str):
    return calendar_store.available_slots(date)


@app.post("/api/kb/reload")
async def reload_kb():
    kb.reload()
    return {"chunks": len(kb.chunks)}


@app.get("/")
async def index():
    return FileResponse(config.ROOT / "static" / "index.html")
