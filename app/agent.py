"""The call agent: keeps conversation state, answers from KB, books slots, transfers."""
import asyncio
import datetime as dt
import json
import logging
import re

from . import calendar_store, config, llm
from .kb import kb

log = logging.getLogger("agent")

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "check_available_slots",
            "description": "Get free meeting slots for a date. Always call this before offering or booking times.",
            "parameters": {
                "type": "object",
                "properties": {"date": {"type": "string", "description": "YYYY-MM-DD"}},
                "required": ["date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "book_appointment",
            "description": "Book a meeting slot once you have date, time, full name, email and purpose.",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "YYYY-MM-DD"},
                    "time": {"type": "string", "description": "HH:MM 24-hour"},
                    "name": {"type": "string", "description": "Caller's full name"},
                    "email": {"type": "string", "description": "Caller's email in written form, e.g. ahmed.khan@gmail.com"},
                    "phone": {"type": "string", "description": "Caller phone if known"},
                    "purpose": {"type": "string", "description": "What the meeting is about, from what the caller said"},
                },
                "required": ["date", "time", "name", "email", "purpose"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "transfer_to_human",
            "description": "Transfer the call to a human. Use when the caller asks for a human, is upset, "
            "or the question is not covered by the knowledge base and needs a person.",
            "parameters": {
                "type": "object",
                "properties": {"reason": {"type": "string"}},
                "required": ["reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "end_call",
            "description": "End the call after the caller says goodbye / has nothing else.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


def _system_prompt(caller: str | None) -> str:
    today = calendar_store.now()
    days = "\n".join(
        f"- {(today + dt.timedelta(days=i)).strftime('%A %Y-%m-%d')}" + (" (today)" if i == 0 else "")
        for i in range(8)
    )
    open_days = ", ".join(dt.date(2024, 1, 1 + d).strftime("%A") for d in config.WORK_DAYS)
    return f"""You are {config.AGENT_NAME}, a warm, friendly receptionist for {config.BUSINESS_NAME}, talking on a live voice call.
How you speak:
- Sound like a real person: short natural sentences, contractions (I'll, you're, that's), a little warmth
  ("Sure!", "Perfect.", "No problem."). 1-3 short sentences per reply. No lists, markdown, emojis or URLs.
- ALWAYS reply in English only, even if the caller uses another language.
- Ask one question at a time, and end your reply with that question.
- Say dates and times the way people do: "Friday, November 20th at 3 PM" (never "15:00" or "2026-11-20").
- If speech recognition garbles something (an email, a name), politely ask the caller to repeat or spell it.

Answering questions:
- Answer ONLY using the KNOWLEDGE BASE below. Never invent prices, policies or facts.
- If the answer is not in the knowledge base, say you are not sure and offer to transfer to a team member.

Booking meetings:
- Business hours: {open_days}, {config.OPEN_TIME}-{config.CLOSE_TIME} ({config.TIMEZONE}), {config.SLOT_MINUTES}-minute slots.
- NEVER work out weekdays, past dates or availability yourself. For ANY date the caller mentions, ALWAYS call
  check_available_slots first and use its answer (it tells you if a date is past, closed or too far ahead).
- Understand relative dates ("tomorrow", "day after tomorrow", "next Monday") using the date list below.
  If the caller gives only a day number ("the 20th") the system adds a DATE HINT - use it; never ask for month or year.
  Dates are always in the current or next year ({today.year}/{today.year + 1}), never earlier.
  Times like "3 PM" or "evening 5" mean 15:00 / 17:00.
- Step 1: call check_available_slots for the requested date. If the caller already said a time and it is free,
  say so briefly ("3 PM works!") and ask for their full name; otherwise offer 2-3 free times and STOP to wait.
  Every reply must end with a question.
- Step 2: collect these three things, ONE question at a time, in this order:
  (a) the caller's FULL NAME, (b) their EMAIL ADDRESS, (c) the PURPOSE of the meeting ("What would you like to discuss?").
  Skip any the caller already told you. NEVER call book_appointment before the caller has actually said all
  three in this call. Never use example or placeholder values (John Doe, example.com, [Name], "Free consultation").
  Speech-to-text writes emails like "ahmed at gmail dot com" - convert to ahmed@gmail.com. If the email sounds
  incomplete or unclear, ask them to spell the part before the @ letter by letter.
- Step 3: call book_appointment. The system will read the details back and ask the caller to confirm.
- Step 4: if the caller says yes, call book_appointment again with the same details to finalize.
- If the caller changes their mind about date or time at any point, forget the old slot and start again from step 1.
- If the caller corrects their name or email, use the corrected value.
- Only say a meeting is booked if book_appointment returned success. Then ask if they need anything else.
- Never speak for the caller and never write more than one reply at a time. Never use placeholders like [Name].
- Current time: {today.strftime('%A %Y-%m-%d %H:%M')}. Upcoming dates:
{days}
- Caller phone number: {caller or 'unknown'}.

Human transfer: if the caller asks for a human/agent/manager, or is angry, call transfer_to_human and tell them you are connecting them.
Ending: call end_call ONLY when the caller clearly says goodbye or that they need nothing else. Never end the call after booking without asking if they need anything else.
"""


_FILLER = re.compile(r"(still on the line|are you still there)", re.I)


def _clean(text: str, max_sentences: int = 3) -> str:
    """Make the reply phone-friendly: drop markdown, duplicate/filler sentences, cap length.
    Small models sometimes glue several replies together ("...help?Would you like...")."""
    text = re.sub(r"([?!])[.…]+", r"\1", text)      # "address?......" -> "address?"
    text = re.sub(r"\.{2,}|…", ".", text)
    text = _humanize_times(re.sub(r"[*_#`]+", "", text).strip())
    text = re.sub(r"([.!?])(?=[A-Z])", r"\1 ", text)
    sentences = re.split(r"(?<=[.!?])\s+", text)
    out, seen = [], set()
    for s in sentences:
        key = re.sub(r"\W+", " ", s.lower()).strip()
        if not key or key in seen or (out and _FILLER.search(s)):
            continue
        seen.add(key)
        out.append(s.strip())
    # stop at the first question so the caller knows what to answer (drops trailing junk like "...? No.")
    result = []
    for s in out:
        result.append(s)
        if s.endswith("?") or len(result) >= max_sentences:
            break
    return " ".join(result)


_GOODBYE = re.compile(r"\b(bye|goodbye|good bye|thanks|thank you|that'?s all|that is all|nothing else|no thanks|"
                      r"that'?s it|i'?m done|end the call|hang up|allah hafiz|khuda hafiz)\b", re.I)


_EMAIL = re.compile(r"[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}")


_MONTHS = r"jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec"
_DAY_ONLY = re.compile(
    rf"(?:\bon\s+(?:the\s+)?(\d{{1,2}})(?:st|nd|rd|th)?\b|\b(?:the\s+)?(\d{{1,2}})(?:st|nd|rd|th)\b|\b(\d{{1,2}})\s+(?:date|tareekh|tarikh)\b)"
    rf"(?!\s*(?::\d|am\b|pm\b|a\.m|p\.m|o'clock|(?:of\s+)?(?:{_MONTHS})))", re.I)


def _date_hint(user_text: str) -> str:
    """The caller said only a day number ('the 20th'): work out the date in code so the model never has to
    guess the month or year (and never asks the caller for them)."""
    m = _DAY_ONLY.search(user_text)
    if not m:
        return ""
    n = int(next(g for g in m.groups() if g))
    d = calendar_store.next_day_of_month(n) if 1 <= n <= 31 else None
    if not d:
        return ""
    return (f"\nDATE HINT: the caller said only the day number {n}. That means {d.strftime('%A, %B')} {_ordinal(d.day)}, "
            f"{d.year} ({d.isoformat()}). Use this date with check_available_slots. NEVER ask the caller for the month or year.")


_MONTH_NUM = {m: i + 1 for i, m in enumerate("jan feb mar apr may jun jul aug sep oct nov dec".split())}
_MON = r"(jan\w*|feb\w*|mar\w*|apr\w*|may|jun\w*|jul\w*|aug\w*|sep\w*|oct\w*|nov\w*|dec\w*)"
_MD = re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s*(?:of\s+)?{_MON}\b", re.I)      # 20 November / 20th of Nov
_DM = re.compile(rf"\b{_MON}\s+(\d{{1,2}})(?:st|nd|rd|th)?\b", re.I)                # November 20
_TIME = re.compile(r"\b(\d{1,2})(?::([0-5]\d))?\s*(a\.?m\.?|p\.?m\.?)", re.I)


def _extract_date(text: str) -> dt.date | None:
    """Date the caller named: '20 November', 'Dec 3rd' or only a day number ('the 20th')."""
    today = calendar_store.now().date()
    m = _MD.search(text)
    month = day = None
    if m:
        day, month = int(m.group(1)), _MONTH_NUM[m.group(2)[:3].lower()]
    else:
        m = _DM.search(text)
        if m:
            month, day = _MONTH_NUM[m.group(1)[:3].lower()], int(m.group(2))
    if month and day:
        try:
            d = dt.date(today.year, month, day)
            if d < today and (today - d).days > 30:   # long past this year -> they mean next year
                d = dt.date(today.year + 1, month, day)
        except ValueError:
            return None
        return d   # a date that passed within the last 30 days stays "this year" -> "that date has passed"
    m = _DAY_ONLY.search(text)
    if m:
        n = int(next(g for g in m.groups() if g))
        return calendar_store.next_day_of_month(n) if 1 <= n <= 31 else None
    return None


def _extract_time(text: str) -> dt.time | None:
    m = _TIME.search(text)
    if not m:
        return None
    h, minute = int(m.group(1)), int(m.group(2) or 0)
    pm = m.group(3).lower().startswith("p")
    if not 1 <= h <= 12:
        return None
    return dt.time(h % 12 + (12 if pm else 0), minute)


async def _slots_note(user_text: str) -> tuple[str, str]:
    """When the caller names a date, look the slots up in code. Returns (note for the model, a ready-made
    spoken answer used if every model fails), so date questions always get a correct reply."""
    d = _extract_date(user_text)
    if not d:
        return "", ""
    res = await asyncio.to_thread(calendar_store.available_slots, d.isoformat())
    short = f"{d.strftime('%A, %B')} {_ordinal(d.day)}"
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
        fallback = (f"{spoken_time(want)} is free on {short}. May I have your full name?" if ok else
                    f"Sorry, {spoken_time(want)} isn't available on {short}. I have {', '.join(spoken[:3])}. Which works for you?")
    else:
        fallback = f"On {short} I have {', '.join(spoken[:3])} open. Which time works for you?"
    note += " Use this directly; you do not need to call check_available_slots again for this date. Never ask for month or year."
    return note, fallback


_YES = re.compile(r"\b(yes|yeah|yep|yup|sure|correct|right|ok|okay|confirm|confirmed|go ahead|please do|book it|"
                  r"sounds good|perfect|absolutely|definitely|haan|ji)\b", re.I)
_NO = re.compile(r"\b(no|not|wrong|wait|change|instead|actually)\b", re.I)
_WANTS_HUMAN = re.compile(r"\b(human|person|someone|agent|manager|representative|staff|team member|real)\b", re.I)


def _normalize_email(raw: str) -> str:
    """Accept both 'ahmed@gmail.com' and speech-to-text style 'Ahmed at gmail dot com'."""
    e = " " + raw.lower().strip() + " "
    e = re.sub(r"\s+(at the rate|at)\s+", "@", e)
    e = re.sub(r"\s+(dot|period)\s+", ".", e)
    e = re.sub(r"\s+(underscore)\s+", "_", e)
    e = re.sub(r"\s+(dash|hyphen)\s+", "-", e)
    return re.sub(r"\s+", "", e).strip(".,")


def _spoken_email(email: str) -> str:
    """'ahmed.k@gmail.com' -> 'ahmed dot k at gmail dot com' so text-to-speech reads it clearly."""
    return email.replace("@", " at ").replace(".", " dot ")


def _ordinal(n: int) -> str:
    return f"{n}{'th' if 11 <= n % 100 <= 13 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def spoken_time(t: dt.time) -> str:
    """15:00 -> '3 PM', 11:30 -> '11:30 AM' (how people say it)."""
    h = t.hour % 12 or 12
    ampm = "AM" if t.hour < 12 else "PM"
    return f"{h} {ampm}" if t.minute == 0 else f"{h}:{t.minute:02d} {ampm}"


def pretty_slot(date: str, time: str) -> str:
    d = dt.date.fromisoformat(date)
    return f"{d.strftime('%A, %B')} {_ordinal(d.day)} at {spoken_time(dt.time.fromisoformat(time))}"


_HHMM = re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)\b(?!\s*(?:[AaPp]\.?[Mm]\b))")


def _humanize_times(text: str) -> str:
    """'We have 10:00, 10:30 or 15:00' -> 'We have 10 AM, 10:30 AM or 3 PM'."""
    return _HHMM.sub(lambda m: spoken_time(dt.time(int(m.group(1)), int(m.group(2)))), text)


class CallSession:
    def __init__(self, session_id: str, caller: str | None = None):
        self.id = session_id
        self.caller = caller
        self.action: str | None = None  # None | "transfer" | "hangup"
        self.turn = 0  # number of caller messages so far
        self.tools_this_turn = 0
        self.pending: tuple[str, str] | None = None  # (date, time) waiting for caller's confirmation
        self.pending_turn = -1
        self.forced_reply: str | None = None
        self.date_fallback = ""
        self.last_user_text = ""
        self.messages: list[dict] = [{"role": "system", "content": _system_prompt(caller)}]

    def greeting(self) -> str:
        text = f"Hello, thank you for calling {config.BUSINESS_NAME}. This is {config.AGENT_NAME}. How can I help you today?"
        self.messages.append({"role": "assistant", "content": text})
        return text

    _BAD_PURPOSE = {"", "meeting", "appointment", "n/a", "none", "unknown", "purpose"}
    _BAD_NAMES = {"", "name", "caller", "unknown", "customer", "user", "client", "n/a", "none"}

    def _caller_text(self) -> str:
        return " ".join(m["content"] for m in self.messages if m["role"] == "user" and m.get("content"))

    def _last_agent_text(self) -> str:
        """The agent's last spoken reply before the caller's current message."""
        for m in reversed(self.messages[:-1]):
            if m["role"] == "assistant" and m.get("content") and not m.get("tool_calls"):
                return m["content"]
        return ""

    def _confirmed(self, key: tuple) -> bool:
        """True when the caller said yes (in a later turn) to a read-back of exactly these details."""
        if not _YES.search(self.last_user_text) or _NO.search(self.last_user_text):
            return False  # the caller's current message must be a clear "yes"
        if self.pending == key and self.pending_turn < self.turn:
            return True  # yes to the system's own read-back
        # Otherwise the agent may have read the details back in its own words (e.g. after the caller
        # corrected the email). Accept only if that reply contained this exact date, time and email.
        date, time, name, email = key
        last = self._last_agent_text().lower()
        d = dt.date.fromisoformat(date)
        t = dt.time.fromisoformat(time)
        date_ok = (d.strftime("%B").lower() in last and str(d.day) in last) or d.strftime("%A").lower() in last
        time_ok = time in last or f"{t.hour % 12 or 12}:{t.minute:02d}" in last or f"{t.hour % 12 or 12} " in last
        email_ok = email in _normalize_email(last) or email in last
        return "?" in last and date_ok and time_ok and email_ok

    def _book(self, args: dict) -> dict:
        """Two-step booking enforced in code: the model must read back the slot and get the
        caller's 'yes' in a later turn before the slot is actually booked."""
        name = (args.get("name") or "").strip()
        if name.lower() in self._BAD_NAMES or any(ch in name for ch in "[]{}<>"):
            return {"success": False, "error": "Caller's name is unknown. Ask the caller for their name. Do NOT say it is booked."}
        email = _normalize_email(args.get("email") or "")
        caller_said = self._caller_text()
        if not _EMAIL.fullmatch(email) or email not in _normalize_email(caller_said):
            # the model must not invent an email: it has to appear in what the caller actually said
            return {"success": False, "error": "You do not have the caller's email yet (never guess it). Ask the caller "
                                               "for their email address. Do NOT say it is booked."}
        if not all(part in caller_said.lower() for part in name.lower().split()):
            return {"success": False, "error": "The caller has not told you this name. Ask for their full name. "
                                               "Do NOT say it is booked."}
        purpose = (args.get("purpose") or "").strip()
        pwords = re.findall(r"[a-z]{4,}", purpose.lower())
        if (not purpose or purpose.lower() in self._BAD_PURPOSE
                or not any(w in caller_said.lower() for w in pwords)):
            return {"success": False, "error": "You do not know the purpose of the meeting yet. Ask the caller what "
                                               "the meeting is about. Do NOT say it is booked."}
        date, time = (args.get("date") or "").strip(), (args.get("time") or "").strip()
        try:
            date = calendar_store._parse_date(date).isoformat()   # fixes a stale year the model may have written
        except ValueError:
            return {"success": False, "error": "Invalid date. Call check_available_slots with a YYYY-MM-DD date first."}
        avail = calendar_store.available_slots(date)
        if "error" in avail:
            return {"success": False, "error": avail["error"]}
        if time not in avail["available"]:
            return {"success": False, "error": f"{time} is not available.", "available": avail["available"][:6]}
        key = (date, time, name.lower(), email)
        when = pretty_slot(date, time)
        if not self._confirmed(key):
            # Ask for confirmation with a fixed sentence (dates come from the calendar, not the model).
            self.pending, self.pending_turn = key, self.turn
            self.forced_reply = (f"Just to confirm: a meeting on {when} for {name}, about {purpose.rstrip('.')}. "
                                 f"Your email is {_spoken_email(email)}. Shall I book it?")
            return {"success": False, "needs_confirmation": True,
                    "instruction": "NOT booked yet. Waiting for the caller to say yes."}
        self.pending = None
        res = calendar_store.book(date, time, name, args.get("phone") or self.caller or "",
                                  purpose, email)
        if res.get("success"):
            invite = " I've sent a calendar invite to your email." if res.get("invite_sent") else ""
            self.forced_reply = f"Done! Your meeting is booked for {when}.{invite} Is there anything else I can help you with?"
        return res

    def _run_tool(self, name: str, raw_args: str) -> dict:
        try:
            args = json.loads(raw_args or "{}")
        except json.JSONDecodeError:
            args = {}
        log.info("[%s] tool %s %s", self.id, name, args)
        self.tools_this_turn += 1
        if name == "check_available_slots":
            return calendar_store.available_slots(args.get("date", ""))
        if name == "book_appointment":
            return self._book(args)
        if name == "transfer_to_human":
            if _GOODBYE.search(self.last_user_text) and not _WANTS_HUMAN.search(self.last_user_text):
                # caller is saying goodbye, not asking for a person -> end the call instead
                self.action = "hangup"
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

    async def respond(self, user_text: str) -> str:
        self.messages.append({"role": "user", "content": user_text})
        self.turn += 1
        self.tools_this_turn = 0
        self.forced_reply = None
        self.last_user_text = user_text
        context = kb.search(user_text + " " + " ".join(
            m["content"] for m in self.messages[-4:-1] if m["role"] == "user" and m.get("content")))
        slots_note, self.date_fallback = await _slots_note(user_text)
        kb_msg = {"role": "system", "content": f"KNOWLEDGE BASE:\n{context}" + _date_hint(user_text) + slots_note}

        try:
            for _ in range(5):
                msgs = [self.messages[0], kb_msg] + self.messages[1:]
                msg = await llm.chat(msgs, TOOLS)
                tool_calls = msg.get("tool_calls")
                if tool_calls:
                    self.messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": tool_calls})
                    for tc in tool_calls:
                        result = await asyncio.to_thread(
                            self._run_tool, tc["function"]["name"], tc["function"].get("arguments", "{}"))
                        self.messages.append({"role": "tool", "tool_call_id": tc["id"], "content": json.dumps(result)})
                    if self.forced_reply:
                        self.messages.append({"role": "assistant", "content": self.forced_reply})
                        return self.forced_reply
                    continue
                text = _clean(msg.get("content") or "")
                if self.action == "hangup" and text.endswith("?"):
                    log.info("[%s] ignoring end_call: agent is still asking a question", self.id)
                    self.action = None
                if text:
                    self.messages.append({"role": "assistant", "content": text})
                    return text
                break
        except Exception:
            log.exception("LLM failure")
            if self.date_fallback:   # models are down, but we can still answer a date question correctly
                self.messages.append({"role": "assistant", "content": self.date_fallback})
                return self.date_fallback
            if config.HUMAN_PHONE:
                self.action = "transfer"
                return "Sorry, I am having a technical issue. Let me connect you to a team member."
            return "Sorry, I am having a technical issue. Could you please say that again?"

        if self.action == "transfer":
            return "Please hold, I am connecting you to a team member."
        if self.action == "hangup":
            return "Thank you for calling. Goodbye!"
        if self.date_fallback:
            self.messages.append({"role": "assistant", "content": self.date_fallback})
            return self.date_fallback
        return "Sorry, could you please repeat that?"


SESSIONS: dict[str, CallSession] = {}


def get_session(session_id: str, caller: str | None = None) -> CallSession:
    if session_id not in SESSIONS:
        SESSIONS[session_id] = CallSession(session_id, caller)
    return SESSIONS[session_id]


def end_session(session_id: str):
    SESSIONS.pop(session_id, None)
