import asyncio
import logging
import uuid
from xml.sax.saxutils import escape

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel

from . import calendar_store, config
from .agent import end_session, get_session
from .kb import kb

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
log = logging.getLogger("main")

app = FastAPI(title="AI Call Agent")


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


# ---------------------------------------------------------------- Browser test calls (100% free)
class WebMsg(BaseModel):
    session_id: str
    text: str


@app.post("/api/web/start")
async def web_start():
    sid = "web-" + uuid.uuid4().hex[:8]
    session = get_session(sid, "browser")
    return {"session_id": sid, "reply": session.greeting(), "action": None}


@app.post("/api/web/message")
async def web_message(msg: WebMsg):
    session = get_session(msg.session_id, "browser")
    reply = await session.respond(msg.text)
    action = session.action
    if action:
        end_session(session.id)
    return {"reply": reply, "action": action, "human_phone": config.HUMAN_PHONE if action == "transfer" else None}


# ---------------------------------------------------------------- Natural voice (free Edge neural TTS)
_tts_cache: dict[str, bytes] = {}


@app.on_event("startup")
async def _warm_tts():
    """Pre-generate the greeting audio so every call starts speaking instantly."""
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
    asyncio.create_task(warm())


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
