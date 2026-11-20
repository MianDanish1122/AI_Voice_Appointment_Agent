"""The call agent: one CallSession per call/chat. It keeps the conversation, owns the shared BookingState
(fed by voice, typed chat and the editable web form), answers from the knowledge base and books slots.

Design rule: the LLM talks and extracts; *code* decides. Dates, times and emails are also extracted
deterministically from what the caller said, booking needs name + email + purpose + a free slot + an explicit
"yes" to a read-back summary, and a ready-made spoken answer exists for every dead end, so a weak or
rate-limited model cannot produce a wrong booking.
"""
import asyncio
import datetime as dt
import json
import logging
import re
import time

from . import calendar_store, config, gcal, llm
from .booking import (FIELDS, BookingState, check_slot, clear_slot_cache, extract_email, next_question,
                      normalize_email, ordinal, pretty_when, spoken_email)
from .kb import kb

log = logging.getLogger("agent")

TOOLS = [
    {"type": "function", "function": {
        "name": "check_available_slots",
        "description": "Free meeting times for a date. Use when the context has no SLOTS CHECKED for that date.",
        "parameters": {"type": "object", "properties": {"date": {"type": "string", "description": "YYYY-MM-DD"}},
                       "required": ["date"]}}},
    {"type": "function", "function": {
        "name": "update_booking_details",
        "description": "Save what the caller just said or changed. Pass ONLY the fields they stated in their latest "
                       "message. Call it again whenever they correct something.",
        "parameters": {"type": "object", "properties": {
            "date": {"type": "string", "description": "YYYY-MM-DD"},
            "time": {"type": "string", "description": "HH:MM 24-hour"},
            "name": {"type": "string", "description": "caller's full name"},
            "email": {"type": "string", "description": "email, e.g. ahmed.khan@gmail.com"},
            "purpose": {"type": "string", "description": "what the meeting is about, in the caller's words"}}}}},
    {"type": "function", "function": {
        "name": "book_appointment",
        "description": "Finalize once everything is known and the caller confirmed. No arguments needed.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "transfer_to_human",
        "description": "Transfer to a person. ONLY when the caller asks for a human/manager, is clearly upset, or said yes to "
                       "your offer of a human. If you just don't know something, OFFER a human first instead of calling this.",
        "parameters": {"type": "object", "properties": {"reason": {"type": "string"}}, "required": ["reason"]}}},
    {"type": "function", "function": {
        "name": "end_call", "description": "End the call after the caller clearly says goodbye.",
        "parameters": {"type": "object", "properties": {}}}},
]


def _system_prompt(caller: str | None) -> str:
    today = calendar_store.now()
    days = ", ".join(
        f"{(today + dt.timedelta(days=i)).strftime('%a %Y-%m-%d')}" + (" (today)" if i == 0 else "") for i in range(8))
    open_days = ", ".join(dt.date(2024, 1, 1 + d).strftime("%A") for d in config.WORK_DAYS)
    return f"""You are {config.AGENT_NAME}, the receptionist at {config.BUSINESS_NAME}, on a live voice call (also shown as chat).
Speak like a warm, natural person: 1-3 short sentences, contractions, no lists/markdown/emojis/URLs, English only.
Say dates and times naturally ("Friday, November 20th at 3 PM"). End every reply with exactly ONE question.
Facts: answer only from the KNOWLEDGE BASE; if it is not there, say so and offer a human.

Booking ({open_days}, {config.OPEN_TIME}-{config.CLOSE_TIME} {config.TIMEZONE}, {config.SLOT_MINUTES}-minute slots). Collect: date, time, full name, email, purpose.
1. BOOKING STATE (in the context) is the single source of truth and is shown to the caller in an editable form. Never ask for anything already filled; if the caller changed a value in the form, use it.
2. Whenever the caller states or CHANGES date, time, name, email or purpose, call update_booking_details with only what they said in their latest message. Never invent values or use placeholders. A bare hour like "at 5" means business hours (5 = 17:00).
3. For a date they name, rely on "SLOTS CHECKED" / "DATE HINT" in the context; otherwise call check_available_slots. Never guess weekdays or availability.
4. If they change their mind ("actually make it..."), acknowledge in one short sentence ("Sure, I'll move it to ...") and update; availability is re-checked for you.
5. Ask only for the next missing item, in this order: date, time, name, email, purpose. Never repeat a question you just asked.
6. When all five are known the system reads a summary and asks the caller to confirm. Never say a meeting is booked; the system says so after the calendar confirms.
Today is {today.strftime('%A %Y-%m-%d %H:%M')}. Dates: {days}. Use {today.year} or {today.year + 1}, never earlier. Caller phone: {caller or 'unknown'}.
If the caller asks for a human/manager or is upset, call transfer_to_human. Call end_call only after a clear goodbye."""


# ------------------------------------------------------------------ small text helpers
_FILLER = re.compile(r"(still on the line|are you still there)", re.I)
_ABBREV = {"mr", "mrs", "ms", "dr", "st", "vs", "prof", "no"}


def _clean_sentence(s: str) -> str:
    s = re.sub(r"([?!])[.…]+", r"\1", s)
    s = re.sub(r"\.{2,}|…", ".", s)
    return _humanize_times(re.sub(r"[*_#`]+", "", s)).strip()


def _split_sentences(text: str) -> list[str]:
    text = re.sub(r"([.!?])(?=[A-Z])", r"\1 ", text)
    out, buf = [], ""
    for piece in re.split(r"(?<=[.!?])\s+", text.strip()):
        buf = f"{buf} {piece}".strip() if buf else piece
        last = re.sub(r"\W", "", buf.split()[-1]).lower() if buf.split() else ""
        if last in _ABBREV and not buf.endswith(("?", "!")):
            continue                                   # "Mr." is not the end of a sentence
        out.append(buf)
        buf = ""
    if buf:
        out.append(buf)
    return out


def _clean(text: str, max_sentences: int = 3) -> str:
    """Make a reply phone-friendly: drop markdown, duplicate/filler sentences, cap length, stop at the first question."""
    out, seen = [], set()
    for s in _split_sentences(_clean_sentence(text)):
        key = re.sub(r"\W+", " ", s.lower()).strip()
        if not key or key in seen or (out and _FILLER.search(s)):
            continue
        seen.add(key)
        out.append(s.strip())
        if s.endswith("?") or len(out) >= max_sentences:
            break
    return " ".join(out)


class _SentenceStream:
    """Turns streamed LLM tokens into clean, speakable sentences (same rules as _clean)."""

    def __init__(self, emit, t0: float, timings: dict):
        self.emit, self.t0, self.timings = emit, t0, timings
        self.buf, self.sent, self.done, self.seen = "", [], False, set()

    def feed(self, delta: str):
        if self.done:
            return
        self.buf += delta
        while True:
            self.buf = re.sub(r"([.!?])(?=[A-Z])", r"\1 ", self.buf)       # "address?And" -> "address? And"
            m = re.search(r"(?<=[.!?])\s+(?=\S)", self.buf)
            if not m:
                return
            head, self.buf = self.buf[:m.start()], self.buf[m.end():]
            words = head.split()
            if words and re.sub(r"\W", "", words[-1]).lower() in _ABBREV and not head.endswith(("?", "!")):
                self.buf = head + " " + self.buf        # "Mr." - keep waiting
                return
            self._push(head)
            if self.done:
                return

    def flush(self):
        if not self.done and self.buf.strip():
            self._push(self.buf)
        self.buf = ""

    def _push(self, raw: str):
        s = _clean_sentence(raw)
        key = re.sub(r"\W+", " ", s.lower()).strip()
        if not key or key in self.seen or (self.sent and _FILLER.search(s)):
            return
        self.seen.add(key)
        self.sent.append(s)
        self.timings.setdefault("first_sentence_ms", int((time.perf_counter() - self.t0) * 1000))
        self.emit(s)
        if s.endswith("?") or len(self.sent) >= 3:
            self.done = True


_GOODBYE = re.compile(r"\b(bye|goodbye|good bye|thanks|thank you|that'?s all|that is all|nothing else|no thanks|"
                      r"that'?s it|i'?m done|end the call|hang up|allah hafiz|khuda hafiz)\b", re.I)
_YES = re.compile(r"\b(yes|yeah|yep|yup|sure|correct|right|ok|okay|confirm|confirmed|go ahead|please do|book it|"
                  r"sounds good|perfect|absolutely|definitely|haan|ji)\b", re.I)
_NO = re.compile(r"\b(no|nope|nah|not|wrong|incorrect|wait|change|instead|actually)\b", re.I)
_HUMAN_NOUN = r"(?:human|real person|live person|person|manager|representative|team member|agent|someone|operator|customer service)"


def _wants_human(text: str) -> bool:
    """'can I talk to a real person' / 'connect me to a human' / 'agent please' - but NOT 'book a meeting with your manager'."""
    t = text.lower()
    if re.search(rf"\b(?:talk|speak|chat|connect|transfer|put me through)\b[^.?!]{{0,40}}\b{_HUMAN_NOUN}\b", t):
        return True
    if (re.search(r"\b(?:human|real person|live agent|live person|operator)\b", t)
            and re.search(r"\b(?:want|need|get|give|let|can|could|please|prefer)\b", t)
            and not re.search(r"\b(?:book|meeting|appointment|schedule)\b", t)):
        return True
    return len(t.split()) <= 3 and bool(re.search(rf"\b{_HUMAN_NOUN}\b", t))


_QUESTION = re.compile(r"\b(what|how|why|which|who|where|when|do you|can you tell|price|cost|how much)\b", re.I)
_PLAIN_NO = re.compile(r"\b(no|nope|nah|wrong|incorrect|not correct|not right)\b", re.I)
_WANTS_HUMAN = re.compile(r"\b(human|person|someone|agent|manager|representative|staff|team member|real)\b", re.I)
_BOOKING_WORDS = re.compile(r"\b(book|booking|meeting|appointment|schedule|slot|available|availability|free|reserve|"
                            r"set up|fix|call me|consultation|tomorrow)\b", re.I)


# ------------------------------------------------------------------ date / time understanding (in code)
_MONTHS = r"jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec"
_DAY_ONLY = re.compile(
    rf"(?:\bon\s+(?:the\s+)?(\d{{1,2}})(?:st|nd|rd|th)?\b|\b(?:the\s+)?(\d{{1,2}})(?:st|nd|rd|th)\b|\b(\d{{1,2}})\s+(?:date|tareekh|tarikh)\b)"
    rf"(?!\s*(?::\d|am\b|pm\b|a\.m|p\.m|o'clock|(?:of\s+)?(?:{_MONTHS})))", re.I)
_MONTH_NUM = {m: i + 1 for i, m in enumerate("jan feb mar apr may jun jul aug sep oct nov dec".split())}
_MON = r"(jan\w*|feb\w*|mar\w*|apr\w*|may|jun\w*|jul\w*|aug\w*|sep\w*|oct\w*|nov\w*|dec\w*)"
_MD = re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s*(?:of\s+)?{_MON}\b", re.I)      # 20 November / 20th of Nov
_DM = re.compile(rf"\b{_MON}\s+(\d{{1,2}})(?:st|nd|rd|th)?\b", re.I)                # November 20
_TIME = re.compile(r"\b(\d{1,2})(?::([0-5]\d))?\s*(a\.?m\.?|p\.?m\.?)", re.I)
_REL = re.compile(r"\b(day after tomorrow|tomorrow|today|tonight)\b", re.I)
_WEEKDAY = re.compile(r"\b(?:(next|this|coming)\s+)?(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", re.I)
_WD = {n: i for i, n in enumerate("monday tuesday wednesday thursday friday saturday sunday".split())}


def _extract_date(text: str, relative: bool = True) -> dt.date | None:
    """The date the caller named; when they changed their mind, the LAST mention wins.
    Understands '20 November', 'Dec 3rd', 'the 20th', 'tomorrow', 'day after tomorrow', 'next Monday'."""
    today = calendar_store.now().date()
    cands: list[tuple[int, dt.date]] = []
    for rx, order in ((_MD, (1, 2)), (_DM, (2, 1))):
        for m in rx.finditer(text):
            day, mon = int(m.group(order[0])), _MONTH_NUM[m.group(order[1])[:3].lower()]
            try:
                d = dt.date(today.year, mon, day)
                if d < today and (today - d).days > 30:   # long past this year -> they mean next year
                    d = dt.date(today.year + 1, mon, day)
                cands.append((m.start(), d))              # a date that passed within 30 days stays "this year"
            except ValueError:
                pass
    for m in _DAY_ONLY.finditer(text):
        n = int(next(g for g in m.groups() if g))
        d = calendar_store.next_day_of_month(n) if 1 <= n <= 31 else None
        if d:
            cands.append((m.start(), d))
    if relative:
        for m in _REL.finditer(text):
            w = m.group(1).lower()
            cands.append((m.start(), today + dt.timedelta(days=2 if w.startswith("day after") else 0 if w in ("today", "tonight") else 1)))
        for m in _WEEKDAY.finditer(text):
            ahead = (_WD[m.group(2).lower()] - today.weekday()) % 7 or 7
            cands.append((m.start(), today + dt.timedelta(days=ahead)))
    return max(cands, key=lambda c: c[0])[1] if cands else None


def _extract_time(text: str) -> dt.time | None:
    """Last explicit time ('3 PM', '11:30 am') in the text."""
    found = None
    for m in _TIME.finditer(text):
        h, minute = int(m.group(1)), int(m.group(2) or 0)
        if 1 <= h <= 12:
            found = dt.time(h % 12 + (12 if m.group(3).lower().startswith("p") else 0), minute)
    return found



# ------------------------------------------------------------------ answers to the question we just asked (code, not LLM)
_NAME_STOP = {"and", "my", "email", "e-mail", "i", "im", "want", "would", "like", "need", "calling", "speaking", "from",
              "at", "here", "with", "to", "for", "book", "meeting", "please", "thanks", "thank", "is", "the", "a", "an",
              "can", "could", "you", "yes", "no", "okay", "ok", "hello", "hi", "hey", "so", "just", "also", "phone",
              "number", "tomorrow", "today", "why", "what", "how", "who", "where", "when", "which", "do", "does", "did",
              "are", "will", "should", "not", "sorry", "wait", "actually", "um", "uh", "well"}
_NUM_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
              "ten": 10, "eleven": 11, "twelve": 12}


def _extract_name(text: str, asked: bool) -> str | None:
    """'my name is Ahmed Khan and ...' -> 'Ahmed Khan'. Bare answers ('Ahmed Khan', "it's Ahmed Khan") count only when
    the agent has just asked for the name."""
    if "?" in text:
        return None
    m = re.search(r"\b(my (?:full )?name is|my name's|name is|call me|this is|i am|i'm|it's|it is)\s+(.+)", text, re.I)
    chunk = None
    if m:
        lead = m.group(1).lower()
        if asked or lead.startswith(("my ", "name is", "call me")):
            chunk = m.group(2)
    elif asked:
        chunk = text
    if not chunk:
        return None
    words = []
    for w in re.findall(r"[A-Za-z][A-Za-z'.-]*", chunk):
        if w.lower() in _NAME_STOP:
            break
        words.append(w)
        if len(words) == 3:
            break
    return " ".join(words) or None


def _extract_purpose(text: str, asked: bool) -> str | None:
    """'I want to discuss pricing' -> 'pricing'. A bare answer counts only right after we asked what the meeting is about."""
    if "?" in text or _QUESTION.search(text):
        return None
    m = re.search(r"\b(?:to discuss|discuss(?:ing)?|regarding|talk(?:ing)? about|speak about)\s+([^.!]+)", text, re.I)
    cand = m.group(1) if m else None
    if not cand and asked:
        cand = re.sub(r"^(?:(?:yes|yeah|ok|okay|sure|well|um|so)[,\s]+)*"
                      r"(?:(?:i|we) (?:want|would like|need|wanna|'d like) to |it'?s |it is |that'?s |just )?"
                      r"(?:about |for |regarding )?", "", text.strip(), flags=re.I)
    # keep just the topic: cut at the next clause ("..., book on the 26th", "... and my email is ...")
    cand = re.split(r"[,;]|\s+(?:and|but)\s+(?:i|my|the|we|please|book|email)\b|\s+(?:book|email)\b", cand or "", 1, flags=re.I)[0]
    cand = re.sub(r"\s+", " ", cand).strip(" ,.;")
    if len(re.findall(r"[A-Za-z]", cand)) < 3 or _YES.fullmatch(cand.lower()) or _PLAIN_NO.fullmatch(cand.lower()):
        return None
    return cand[:120]


def _bare_time(text: str) -> dt.time | None:
    """'5', 'at 5', "5 o'clock", 'five thirty', '3:30' as the answer to 'what time?': business hours decide am/pm."""
    t = text.lower().strip(" .!")
    m = re.fullmatch(r"(?:at |around |make it |let's do |lets do |how about )?(\d{1,2})(?::([0-5]\d))?(?: o'?clock)?", t)
    if m:
        h, minute = int(m.group(1)), int(m.group(2) or 0)
    else:
        m = re.fullmatch(r"(?:at |around |make it )?(" + "|".join(_NUM_WORDS) + r")(?: o'?clock)?(?: (thirty|fifteen|forty five))?", t)
        if not m:
            return None
        h, minute = _NUM_WORDS[m.group(1)], {"thirty": 30, "fifteen": 15, "forty five": 45}.get(m.group(2), 0)
    if not 1 <= h <= 12:
        return None
    return dt.time(h + 12 if h < 8 else h % 12 if h == 12 else h, minute) if h != 12 else dt.time(12, minute)


def _asked_field_from(text: str) -> str | None:
    t = text.lower()
    if re.search(r"full name|your name|last name|first name", t):
        return "name"
    if "email" in t:
        return "email"
    if re.search(r"discuss|purpose|meeting about|reason for|what.*about", t):
        return "purpose"
    if re.search(r"what time|which time|time works|what time", t):
        return "time"
    if re.search(r"which day|what date|which date|what day|day would", t):
        return "date"
    return None


def spoken_time(t: dt.time) -> str:
    """15:00 -> '3 PM', 11:30 -> '11:30 AM' (how people say it)."""
    h = t.hour % 12 or 12
    ampm = "AM" if t.hour < 12 else "PM"
    return f"{h} {ampm}" if t.minute == 0 else f"{h}:{t.minute:02d} {ampm}"


_HHMM = re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)\b(?!\s*(?:[AaPp]\.?[Mm]\b))")


def _humanize_times(text: str) -> str:
    """'We have 10:00, 10:30 or 15:00' -> 'We have 10 AM, 10:30 AM or 3 PM'."""
    return _HHMM.sub(lambda m: spoken_time(dt.time(int(m.group(1)), int(m.group(2)))), text)


def _date_label(d: dt.date) -> str:
    return f"{d.strftime('%A, %B')} {ordinal(d.day)}"


def _free_spoken(free: list[str], n: int = 3) -> str:
    return ", ".join(spoken_time(dt.time.fromisoformat(t)) for t in free[:n])


async def _slots_note(user_text: str, relative: bool) -> tuple[str, str]:
    """When the caller names a date, look the slots up in code. Returns (note for the model, a ready-made
    spoken answer used if every model fails), so date questions always get a correct reply."""
    d = _extract_date(user_text, relative)
    if not d:
        return "", ""
    res = await asyncio.to_thread(calendar_store.available_slots, d.isoformat())
    short = _date_label(d)
    label = f"{short}, {d.year}"
    if "error" in res:
        return (f"\nSLOTS CHECKED for {label}: {res['error']} Tell the caller this and ask for another date.",
                f"Sorry, I can't book {short}. Could you pick another date?")
    free = res["available"]
    if not free:
        return (f"\nSLOTS CHECKED for {label}: no free times ({res.get('note', 'fully booked')}). "
                "Tell the caller and suggest another day.",
                f"Sorry, we have no openings on {short}. Would another day work for you?")
    spoken = [spoken_time(dt.time.fromisoformat(t)) for t in free]
    note = f"\nSLOTS CHECKED for {label} (date {d.isoformat()}): free times: {', '.join(spoken[:8])}" + \
           (f" and {len(free) - 8} more" if len(free) > 8 else "") + "."
    want = _extract_time(user_text)
    if want:
        ok = want.strftime("%H:%M") in free
        note += f" The caller asked for {spoken_time(want)}: it is {'FREE' if ok else 'NOT available'}."
        fallback = (f"{spoken_time(want)} is free on {short}." if ok else
                    f"Sorry, {spoken_time(want)} isn't available on {short}. I have {', '.join(spoken[:3])}. Which works for you?")
    else:
        fallback = f"On {short} I have {', '.join(spoken[:3])} open. Which time works for you?"
    note += " Use this directly; no need to call check_available_slots again for this date. Never ask for month or year."
    return note, fallback


# ------------------------------------------------------------------ the session
class CallSession:
    def __init__(self, session_id: str, caller: str | None = None):
        self.id = session_id
        self.caller = caller
        self.action: str | None = None            # None | "transfer" | "hangup"
        self.turn = 0                              # number of caller messages so far
        self.tools_this_turn = 0
        self.booking = BookingState()
        self.manual_text: list[str] = []           # what the caller typed into the form (counts as "said by the caller")
        self.awaiting: str | None = None           # 'email' | 'email_suggestion': a yes/no to our email read-back
        self.pending_key: str | None = None        # key of the summary the caller was asked to confirm
        self.pending_turn = -1
        self.forced_reply: str | None = None
        self.date_fallback = ""
        self.last_user_text = ""
        self.via = "voice"
        self.call_active = False
        self.on_state = None                       # callback: push the booking state to the UI
        self._spoken_prefix = ""                   # part of the reply already delivered (see _say)
        # Vapi path: Vapi owns the conversation, we only see it through webhooks (see app/vapi.py)
        self.external = False
        self.external_user_texts: list[str] = []
        self.external_last_agent = ""
        self.vapi_processed = 0
        self.last_active = time.time()
        self.messages: list[dict] = [{"role": "system", "content": _system_prompt(caller)}]

    # ---------------------------------------------------------------- basics
    def greeting(self) -> str:
        text = f"Hello, thank you for calling {config.BUSINESS_NAME}. This is {config.AGENT_NAME}. How can I help you today?"
        self.messages.append({"role": "assistant", "content": text})
        self.call_active = True
        return text

    def _phone(self) -> str:
        return "" if self.caller in (None, "browser") else self.caller

    def _asked_field(self) -> str | None:
        """Which field was the agent's last question about? (so a bare 'Ahmed Khan' or 'five' is understood)"""
        if self.external:
            return _asked_field_from(self.external_last_agent) if self.external_last_agent else None
        for m in reversed(self.messages[:-1]):
            if m["role"] == "assistant" and m.get("content") and not m.get("tool_calls"):
                return _asked_field_from(m["content"])
        return None

    def _caller_text(self) -> str:
        user = [m["content"] for m in self.messages if m["role"] == "user" and m.get("content")]
        return " ".join(user + self.manual_text + self.external_user_texts)

    def _notify(self):
        if self.on_state:
            try:
                self.on_state()
            except Exception:
                log.exception("on_state failed")

    def _trim_history(self):
        if len(self.messages) > 18:
            tail = self.messages[-14:]
            while tail and tail[0]["role"] != "user":   # never start with an orphaned tool result
                tail = tail[1:]
            self.messages = [self.messages[0]] + tail

    def _state_note(self) -> str:
        st, v = self.booking, self.booking.values

        def show(f):
            if not v[f]:
                return "-"
            tag = {"manual": " [typed in form]", "typed": " [typed]"}.get(st.source.get(f), "")
            if f == "date":
                return f"{_date_label(dt.date.fromisoformat(v[f]))}, {v[f]}{tag}"
            if f == "time":
                return f"{spoken_time(dt.time.fromisoformat(v[f]))}{tag}"
            if f == "email":
                return f"{v[f]} ({'confirmed' if st.email_status == 'confirmed' else 'not yet confirmed'}){tag}"
            return v[f] + tag

        miss = st.missing()
        errs = "; ".join(f"{k}: {e}" for k, e in st.errors.items())
        return ("\nBOOKING STATE: " + " | ".join(f"{f}={show(f)}" for f in FIELDS)
                + f"\nStill missing: {', '.join(miss) if miss else 'nothing'}." + (f" Form errors: {errs}." if errs else ""))

    # ---------------------------------------------------------------- shared state helpers
    async def public_state(self, fresh: bool = False) -> dict:
        st = self.booking
        slot = None
        if st.values["date"] and not st.booked:      # once booked there is nothing left to check (saves a Google call)
            slot = await asyncio.to_thread(check_slot, st.values["date"], st.values["time"], fresh)
        d = st.public(slot)
        d["call_active"] = self.call_active
        d["pending_confirm"] = bool(self.pending_key and self.pending_key == st.key())   # a read-back is waiting for "yes"
        d["email_check"] = self.awaiting                                                   # 'email' | 'email_suggestion' | None
        return d

    def _summary_or_none(self) -> str | None:
        """All five details known? -> verify the email if shaky, re-check the slot, then read everything back."""
        st = self.booking
        if st.missing():
            return None
        v = st.values
        if st.email_status == "unconfirmed" and st.email_conf == "low":
            return self._ask_email_check()
        chk = check_slot(v["date"], v["time"])
        if not chk["ok"]:
            return self._slot_problem(chk)
        self.pending_key, self.pending_turn = st.key(), self.turn
        return f"Just to confirm: a meeting on {st.summary()} Shall I book it?"

    def _slot_problem(self, chk: dict) -> str:
        v = self.booking.values
        when = pretty_when(v["date"], v["time"]) if v["date"] and v["time"] else "that time"
        if chk.get("code") == "calendar_down":
            return "I'm having trouble reaching the calendar right now, so I can't check availability. Could we try again in a moment?"
        if chk.get("free"):
            return f"Sorry, {when.split(' at ')[-1]} isn't available that day. I have {_free_spoken(chk['free'])}. Which works for you?"
        return f"{chk.get('error', 'Sorry, that slot is not available.')} Would another day work for you?"

    def _ask_email_check(self) -> str:
        st = self.booking
        email = st.values["email"]
        if st.email_suggestion:
            self.awaiting = "email_suggestion"
            return (f"I heard {spoken_email(email)}. Did you mean {spoken_email(st.email_suggestion)}?")
        self.awaiting = "email"
        return (f"I want to make sure I have your email correctly. I heard {spoken_email(email)}. Is that correct?")

    def _result_reply(self, res: dict) -> str:
        if res.get("success"):
            when = pretty_when(res["date"], res["time"])
            inv = " I've sent a calendar invite to your email." if res.get("invite_sent") else ""
            pre = "That meeting is already on the calendar. " if res.get("already_booked") else "Done! "
            return f"{pre}Your meeting is booked for {when}.{inv} Is there anything else I can help you with?"
        err = res.get("error", "")
        alts = res.get("alternatives") or res.get("free") or []
        if res.get("code") == "busy" or "already" in err.lower():
            return ("Sorry, that time was just taken. " +
                    (f"I have {_free_spoken(alts)}. Which works for you?" if alts else "Could you pick another time?"))
        if "calendar" in err.lower():
            return ("I'm having trouble reaching the calendar right now, so nothing has been booked. "
                    "Would you like to try again in a moment?")
        return f"Sorry, I couldn't book that. {err} Would you like to try a different time?"

    def _finalize_sync(self) -> dict:
        """The only place that books. Re-validates everything; success is reported only after the calendar API confirmed."""
        st, v = self.booking, self.booking.values
        miss = st.missing()
        if miss:
            return {"success": False, "code": "incomplete", "error": f"Missing or invalid: {', '.join(miss)}."}
        if st.booked and st.booked.get("key") == st.key():
            return {**st.booked, "success": True, "already_booked": True}
        st.email_status = "confirmed"            # the caller confirmed a summary that showed this email
        res = calendar_store.book(v["date"], v["time"], v["name"], self._phone(), v["purpose"], v["email"])
        if res.get("success"):
            st.booked = {k: res.get(k) for k in ("date", "time", "weekday", "name", "invite_sent", "event_link",
                                                 "meet_link", "calendar")}
            st.booked["key"] = st.key()
            st.booked["email"] = v["email"]
            st.booked["purpose"] = v["purpose"]
            st.booked["when"] = pretty_when(res["date"], res["time"])
            clear_slot_cache()
            self.pending_key = self.awaiting = None
        return res

    # ---------------------------------------------------------------- updates from voice / typing (LLM tool + code)
    def _mentioned(self, field: str, value: str, text: str) -> bool:
        low = text.lower()
        if field == "email":
            return value in normalize_email(text)
        if field == "name":
            return all(p in low for p in value.lower().split())
        return any(w in low for w in re.findall(r"[a-z]{4,}", value.lower()))

    def _update(self, args: dict) -> dict:
        st = self.booking
        src = "typed" if self.via == "typed" else "voice"
        corpus, last = self._caller_text(), self.last_user_text
        saved, errors = {}, {}
        relative = self._booking_context(last)

        for f in FIELDS:
            val = str(args.get(f) or "").strip()
            if f == "date":
                explicit = _extract_date(last, relative)
                if explicit:
                    val = explicit.isoformat()
                elif st.values["date"]:
                    continue                    # the caller named no date this turn: never overwrite with a stale one
            elif f == "time":
                explicit = _extract_time(last)
                if explicit:
                    val = explicit.strftime("%H:%M")
                elif st.values["time"] and not re.search(r"\b(at|for|make it|around)\s+\d{1,2}(:\d\d)?\b", last, re.I):
                    continue
            elif f == "email":
                found = extract_email(last)
                if found:
                    val = found
                elif val:
                    val = normalize_email(val)
                    if val not in normalize_email(corpus):          # the model must not invent an email
                        errors["email"] = "The caller has not said that email yet - ask for it."
                        continue
            if not val:
                continue
            if f == "name" and not self._mentioned("name", val, corpus):
                errors["name"] = "The caller has not told you that name - ask for their full name."
                continue
            if f == "purpose" and not self._mentioned("purpose", val, corpus):
                errors["purpose"] = "The caller has not said that - ask what the meeting is about."
                continue
            # a value the caller typed in the form is not overwritten by a stale voice guess
            if (f in ("name", "email", "purpose") and st.source.get(f) == "manual" and st.values[f]
                    and val.lower() != st.values[f].lower() and not self._mentioned(f, val, last)):
                continue
            res = st.set(f, val, src)
            if res["ok"]:
                saved[f] = st.values[f]
            else:
                errors[f] = res["error"]
        self._notify()
        out = {"saved": saved, "errors": errors, "missing": st.missing()}
        if saved:
            summary = self._summary_or_none()
            if summary:
                self.forced_reply = summary
        return out

    def _booking_context(self, text: str) -> bool:
        """Is the caller talking about booking? (gates relative dates like 'today' so chit-chat doesn't trigger lookups)"""
        v = self.booking.values
        return bool(_BOOKING_WORDS.search(text) or _TIME.search(text) or v["date"] or v["time"] or v["name"])

    _ACKS = ("Got it.", "Perfect.", "Great.", "Sure thing.", "Thanks.")

    def _acknowledge(self, newly: list[str], prev: dict) -> str:
        """One natural sentence for what we just learned + the next question. Availability is real (looked up)."""
        st, v = self.booking, self.booking.values
        bits = []
        revised = [f for f in newly if prev.get(f) and f in ("date", "time")]
        if revised:
            bits.append("Sure, I'll update that.")
        if v["date"] and ("date" in newly or "time" in newly):
            label = _date_label(dt.date.fromisoformat(v["date"]))
            if v["time"]:
                chk = check_slot(v["date"], v["time"])
                if not chk["ok"]:
                    return self._slot_problem(chk)
                bits.append(f"{spoken_time(dt.time.fromisoformat(v['time']))} on {label} is free.")
            else:
                info = calendar_store.available_slots(v["date"])
                free = info.get("available") or []
                if "error" in info or not free:
                    return (f"Sorry, we have no openings on {label}. Would another day work for you?"
                            if "error" not in info else f"Sorry, I can't book {label}. Could you pick another date?")
                return " ".join(bits + [f"On {label} I have {_free_spoken(free)} open. Which time works for you?"])
        elif "name" in newly and v["name"]:
            bits.append(f"Thanks, {v['name'].split()[0]}.")
        else:
            bits.append(self._ACKS[self.turn % len(self._ACKS)])
        return " ".join(bits + [next_question(st)])

    # ---------------------------------------------------------------- tools
    def _run_tool(self, name: str, raw_args: str) -> dict:
        try:
            args = json.loads(raw_args or "{}")
        except json.JSONDecodeError:
            args = {}
        log.info("[%s] tool %s %s", self.id, name, args)
        self.tools_this_turn += 1
        if name == "check_available_slots":
            res = calendar_store.available_slots(args.get("date", ""))
            if "available" in res:
                free = res.pop("available")
                res["free_times"] = [spoken_time(dt.time.fromisoformat(t)) for t in free[:10]]
                res["more"] = max(0, len(free) - 10)
            return res
        if name == "update_booking_details":
            return self._update(args)
        if name == "book_appointment":
            if args:
                self._update(args)
            st = self.booking
            if st.booked and st.booked.get("key") == st.key():   # already done (a yes was handled by the rules)
                self.forced_reply = self._result_reply({**st.booked, "success": True, "already_booked": True})
                return {**st.booked, "success": True, "already_booked": True}
            if self.forced_reply:                       # e.g. the system is verifying the email
                return {"success": False, "instruction": "NOT booked. The system is asking the caller a question."}
            miss = st.missing()
            if miss:
                return {"success": False, "missing": miss,
                        "error": f"Still missing: {', '.join(miss)}. Ask for {miss[0]} next. Do NOT say it is booked."}
            if (self.pending_key == st.key() and self.pending_turn < self.turn
                    and _YES.search(self.last_user_text) and not _NO.search(self.last_user_text)):
                res = self._finalize_sync()             # the caller said yes to the read-back
                self.forced_reply = self._result_reply(res)
                return res
            self.forced_reply = self._summary_or_none()  # read the details back and wait for "yes"
            return {"success": False, "needs_confirmation": True,
                    "instruction": "NOT booked yet. Waiting for the caller to say yes."}
        if name == "transfer_to_human":
            if _GOODBYE.search(self.last_user_text) and not _WANTS_HUMAN.search(self.last_user_text):
                self.action = "hangup"          # a plain goodbye, not a request for a person
                return {"success": True, "note": "Caller said goodbye; the call will end. Say a short goodbye."}
            if not config.HUMAN_PHONE:
                return {"success": False, "error": "No human available right now. Offer to book a callback meeting."}
            self.action = "transfer"
            return {"success": True, "note": "Call will be transferred after your next sentence."}
        if name == "end_call":
            if self.tools_this_turn > 1 or not _GOODBYE.search(self.last_user_text):
                return {"success": False, "error": "Caller has not said goodbye. Do NOT end the call. "
                                                   "Help them or ask if there is anything else you can do."}
            self.action = "hangup"
            return {"success": True}
        return {"error": f"unknown tool {name}"}

    # ---------------------------------------------------------------- manual edits from the web form
    async def apply_manual(self, fields: dict) -> dict:
        """The caller edited the form: update the shared state (manual input wins) and, if a call is on, let the
        agent continue from there instead of asking again."""
        self.last_active = time.time()
        st = self.booking
        before_missing = st.missing()
        results = {}
        for f, val in fields.items():
            if f not in FIELDS:
                continue
            results[f] = st.set(f, str(val or ""), "manual")
            if val:
                self.manual_text.append(f"{f}: {val}")
            if f == "email":
                self.awaiting = None
        self.pending_key = None                  # anything edited needs a fresh confirmation
        state = await self.public_state(fresh=True)
        followup = ""
        if self.call_active and results:
            slot = state["slot"] or {}
            if st.values["date"] and st.values["time"] and not slot.get("ok") \
                    and slot.get("code") in ("busy", "closed", "date_not_allowed"):
                followup = self._slot_problem(slot)
            else:
                still = st.missing()
                filled = [f for f in before_missing if f not in still]
                if filled and not still:
                    summary = await asyncio.to_thread(self._summary_or_none)
                    followup = ("Thanks, I have everything now. " + summary) if summary else ""
                elif filled and before_missing[0] in filled:
                    followup = "Got it. " + next_question(st)
            if followup:
                self.messages.append({"role": "assistant", "content": followup})
        return {"state": state, "followup": followup, "results": results}

    async def confirm_from_ui(self, key: str) -> dict:
        """'Confirm booking' button: only valid for exactly the summary the caller saw."""
        st = self.booking
        if st.booked and st.booked.get("key") == st.key() == key:     # double click: report it, never rebook
            return {**st.booked, "success": True, "already_booked": True}
        if key != st.key():
            return {"success": False, "code": "stale", "error": "The details changed since you reviewed them. Please review again."}
        if st.missing():
            return {"success": False, "code": "incomplete", "error": f"Please complete: {', '.join(st.missing())}."}
        chk = await asyncio.to_thread(check_slot, st.values["date"], st.values["time"])
        if not chk["ok"]:
            return {"success": False, **{k: chk[k] for k in ("code", "error", "free") if k in chk}}
        res = await asyncio.to_thread(self._finalize_sync)
        if res.get("success"):
            self.messages.append({"role": "assistant", "content": self._result_reply(res)})
        return res

    # ---------------------------------------------------------------- the conversation turn
    async def respond(self, user_text: str, via: str = "voice", emit=None, on_state=None, timings: dict | None = None) -> str:
        """Handle one caller message. `emit(sentence)` (optional) receives the reply sentence by sentence as soon as
        each is ready; `on_state()` is called when the booking state changed; `timings` receives latency numbers."""
        t0 = time.perf_counter()
        timings = timings if timings is not None else {}
        self.last_active = time.time()
        self.via, self.on_state = via, on_state
        self.messages.append({"role": "user", "content": user_text})
        self.turn += 1
        self.tools_this_turn = 0
        self.forced_reply = None
        self.date_fallback = ""
        self.last_user_text = user_text
        self._trim_history()

        def early(sentence: str):                 # speak now, finish the slow part (Google) afterwards
            timings.setdefault("first_sentence_ms", int((time.perf_counter() - t0) * 1000))
            emit(sentence)

        self._spoken_prefix = ""
        direct = await self._deterministic(user_text, early if emit else None)
        if direct is not None:
            timings.update(total_ms=int((time.perf_counter() - t0) * 1000), model="rules (no LLM)", llm_ms=0)
            return self._say(direct, emit, timings, t0)

        context = kb.search(user_text + " " + " ".join(
            m["content"] for m in self.messages[-4:-1] if m["role"] == "user" and m.get("content")))
        slots_note, self.date_fallback = await _slots_note(user_text, self._booking_context(user_text))
        kb_msg = {"role": "system", "content": f"KNOWLEDGE BASE:\n{context}" + self._state_note() + slots_note}

        stream = _SentenceStream(emit, t0, timings) if emit else None
        llm_ms, stats = 0, {}
        try:
            for _ in range(5):
                msgs = [self.messages[0], kb_msg] + self.messages[1:]
                stats = {}
                msg = await llm.chat(msgs, TOOLS, on_text=stream.feed if stream else None, stats=stats)
                llm_ms += stats.get("total_ms", 0)
                timings["model"] = stats.get("model")
                tool_calls = msg.get("tool_calls")
                if tool_calls:
                    self.messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": tool_calls})
                    tt = time.perf_counter()
                    for tc in tool_calls:
                        result = await asyncio.to_thread(
                            self._run_tool, tc["function"]["name"], tc["function"].get("arguments", "{}"))
                        self.messages.append({"role": "tool", "tool_call_id": tc["id"], "content": json.dumps(result)})
                    timings["tools_ms"] = timings.get("tools_ms", 0) + int((time.perf_counter() - tt) * 1000)
                    if self.forced_reply:
                        timings.update(total_ms=int((time.perf_counter() - t0) * 1000), llm_ms=llm_ms)
                        return self._say(self.forced_reply, emit, timings, t0)
                    continue
                if stream:
                    stream.flush()
                text = " ".join(stream.sent) if stream and stream.sent else _clean(msg.get("content") or "")
                if self.action == "hangup" and text.endswith("?"):
                    log.info("[%s] ignoring end_call: agent is still asking a question", self.id)
                    self.action = None
                if text:
                    timings.update(total_ms=int((time.perf_counter() - t0) * 1000), llm_ms=llm_ms)
                    if stream and stream.sent:
                        self.messages.append({"role": "assistant", "content": text})
                        return text
                    return self._say(text, emit, timings, t0)
                break
        except Exception:
            log.exception("LLM failure")
            timings.update(total_ms=int((time.perf_counter() - t0) * 1000), llm_ms=llm_ms, model="fallback")
            if self.date_fallback:   # models are down, but we can still answer a date question correctly
                return self._say(self.date_fallback, emit, timings, t0)
            if config.HUMAN_PHONE:
                self.action = "transfer"
                return self._say("Sorry, I am having a technical issue. Let me connect you to a team member.", emit, timings, t0)
            return self._say("Sorry, I am having a technical issue. Could you please say that again?", emit, timings, t0)

        timings.update(total_ms=int((time.perf_counter() - t0) * 1000), llm_ms=llm_ms)
        if self.action == "transfer":
            return self._say("Please hold, I am connecting you to a team member.", emit, timings, t0)
        if self.action == "hangup":
            return self._say("Thank you for calling. Goodbye!", emit, timings, t0)
        if self.date_fallback:
            return self._say(self.date_fallback, emit, timings, t0)
        return self._say("Sorry, could you please repeat that?", emit, timings, t0)

    def _say(self, text: str, emit, timings: dict, t0: float) -> str:
        """Record a ready-made reply and deliver it sentence by sentence."""
        self.messages.append({"role": "assistant", "content": text})
        if emit:
            rest = text[len(self._spoken_prefix):].strip() if self._spoken_prefix and text.startswith(self._spoken_prefix) else text
            for s in _split_sentences(rest):
                timings.setdefault("first_sentence_ms", int((time.perf_counter() - t0) * 1000))
                emit(s)
        self._spoken_prefix = ""
        return text

    async def _deterministic(self, text: str, early=None) -> str | None:
        """Everything that needs no LLM: yes/no to our read-backs, and dates/times/emails spoken or typed.
        Faster (no model round trip) and exact (no model guessing)."""
        st = self.booking
        yes = bool(_YES.search(text)) and not _NO.search(text)
        plain_no = bool(_PLAIN_NO.search(text))
        found_email = extract_email(text)

        # 0) "let me talk to a person" -> hand over right away (a plain, instant, reliable answer)
        if _wants_human(text) and not found_email:
            self.pending_key = self.awaiting = None
            if config.HUMAN_PHONE:
                self.action = "transfer"
                return "Of course. Connecting you to a team member now."
            return "I'm sorry, nobody is available right now. I can book a meeting so a team member calls you back. Would you like that?"

        # 1) "Is that your email?" -> yes / no
        if self.awaiting and (yes or plain_no) and (not found_email or found_email == st.values["email"]):
            aw, self.awaiting = self.awaiting, None
            if yes:
                if aw == "email_suggestion" and st.email_suggestion:
                    st.set("email", st.email_suggestion, "typed")
                else:
                    st.email_status = "confirmed"
                self._notify()
                summary = await asyncio.to_thread(self._summary_or_none)
                return "Thank you. " + (summary or next_question(st))
            st.set("email", "", "voice")
            self._notify()
            return ("No problem. You can spell it letter by letter, or type it into the email box. "
                    "What's your email address?")

        # 2) "Shall I book it?" -> yes
        if self.pending_key and yes and self.pending_key == st.key() and self.pending_turn < self.turn:
            filler = ""
            if early and gcal.enabled():          # the Google write takes ~1-2 s: start talking while it runs
                filler = "One moment while I book that."
                early(filler)
                self._spoken_prefix = filler
            res = await asyncio.to_thread(self._finalize_sync)
            self._notify()
            reply = self._result_reply(res)
            return f"{filler} {reply}".strip()
        if self.pending_key:
            self.pending_key = None
            if plain_no and len(text.split()) <= 4:
                return "No problem. What would you like to change?"

        # 3) dates / times / emails / names / purpose the caller just said or typed -> shared state (code, not LLM)
        asked = self._asked_field()
        src = "typed" if self.via == "typed" else "voice"
        relative = self._booking_context(text) or asked in ("date", "time")
        prev, newly = dict(st.values), []

        def put(field: str, value: str):
            if st.set(field, value, src)["changed"]:
                newly.append(field)

        d = _extract_date(text, relative)
        if d:
            put("date", d.isoformat())
        t = _extract_time(text) or (_bare_time(text) if asked == "time" else None)
        if t:
            put("time", t.strftime("%H:%M"))
        if found_email:
            put("email", found_email)
        name = _extract_name(text, asked == "name")
        if name and not (st.source.get("name") == "manual" and st.values["name"] and name.lower() != st.values["name"].lower()):
            put("name", name)
        purpose = _extract_purpose(text, asked == "purpose")
        if purpose and not (st.source.get("purpose") == "manual" and st.values["purpose"]):
            put("purpose", purpose)
        date_err = st.errors.get("date") if d else None      # e.g. "We're closed on Saturdays" / "already passed"
        if newly or date_err:
            self._notify()
            if "?" in text or _QUESTION.search(text):
                return None                         # they also asked something: state is saved, let the LLM answer
            if _GOODBYE.search(text) or _WANTS_HUMAN.search(text):
                return None
            if date_err:                            # say it ourselves: exact, instant, and works if the model is down
                return "Sorry, " + date_err[0].lower() + date_err[1:]
            # an email we are not sure about is verified before anything else
            if found_email and st.email_status == "unconfirmed" and st.email_conf == "low":
                return self._ask_email_check()
            if not st.missing():
                return await asyncio.to_thread(self._summary_or_none)
            # the caller only gave data: acknowledge it and ask for the next item - no LLM round trip needed
            return await asyncio.to_thread(self._acknowledge, newly, prev)
        return None


SESSIONS: dict[str, CallSession] = {}


def get_session(session_id: str, caller: str | None = None) -> CallSession:
    if session_id not in SESSIONS:
        cutoff = time.time() - 6 * 3600                      # forget calls nobody touched for 6 hours
        for k in [k for k, s in SESSIONS.items() if s.last_active < cutoff]:
            SESSIONS.pop(k, None)
        SESSIONS[session_id] = CallSession(session_id, caller)
    return SESSIONS[session_id]


def end_session(session_id: str):
    SESSIONS.pop(session_id, None)
