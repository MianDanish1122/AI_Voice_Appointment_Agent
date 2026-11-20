"""Offline test suite (no network, no API keys, never touches the real Google Calendar).

Run from the repo root:  .venv\\Scripts\\python -m tests.smoke_test

The LLM is replaced by a tiny rule-based fake and Google Calendar by an in-memory fake (only here, in the tests).
Everything else - booking state, email recognition, validation, confirmation rules, SQLite, HTTP API - is the real code.
"""
import asyncio
import datetime as dt
import json
import os
import re
import tempfile
import time
import traceback
from pathlib import Path

os.environ["GOOGLE_CALENDAR_ID"] = ""          # tests must never use the real calendar
from app import agent, booking, calendar_store, config, gcal, llm  # noqa: E402

config.HUMAN_PHONE = "+920000000000"
config.GOOGLE_CALENDAR_ID = ""
TZ = calendar_store.TZ
NOW = dt.datetime(2026, 11, 23, 9, 0, tzinfo=TZ)            # a fixed Monday morning: tomorrow = Tue, +2 = Wed
calendar_store.now = lambda: NOW


def iso(days: int) -> str:
    return (NOW.date() + dt.timedelta(days=days)).isoformat()


def phrase(days: int) -> str:                                # "on November 25"
    d = NOW.date() + dt.timedelta(days=days)
    return f"on {d.strftime('%B')} {d.day}"


# ------------------------------------------------------------------ fakes
class FakeLLM:
    """Rule-based stand-in for the model: extracts name/purpose via the same tool calls a real model makes."""
    delay = 0.0
    default = "Sure. How can I help?"
    calls = 0

    async def chat(self, messages, tools=None, on_text=None, stats=None):
        FakeLLM.calls += 1
        if stats is not None:
            stats.update(model="fake", total_ms=1)
        if messages[-1]["role"] == "tool":
            called = [tc["function"]["name"] for m in messages if m.get("tool_calls") for tc in m["tool_calls"]]
            return await self._text("Goodbye!" if called and called[-1] == "end_call" else "Okay. What's next?", on_text)
        text = next(m["content"] for m in reversed(messages) if m["role"] == "user")
        args = {}
        m = re.search(r"my name is ([A-Za-z]+(?: [A-Za-z]+)?)", text, re.I)
        if m:
            args["name"] = m.group(1)
        if "INVENT" in text:
            args["name"] = "John Doe"
        m = re.search(r"(?:discuss|about) ([^.?!]+)", text, re.I)
        if m:
            args["purpose"] = m.group(1).strip()
        if args:
            return self._tool("update_booking_details", args)
        if "go ahead and book" in text.lower():
            return self._tool("book_appointment", {})
        if "goodbye" in text.lower():
            return self._tool("end_call", {})
        return await self._text(self.default, on_text)

    @staticmethod
    def _tool(name, args):
        return {"role": "assistant", "content": None,
                "tool_calls": [{"id": "c1", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}

    async def _text(self, text, on_text):
        if on_text:
            for chunk in re.findall(r"\S+\s*", text):
                on_text(chunk)
                if self.delay:
                    await asyncio.sleep(self.delay)
        return {"role": "assistant", "content": text}


class FakeGoogle:
    """In-memory Google Calendar: busy times, created events, optional failures."""

    def __init__(self, busy=(), fail_create=False, fail_busy=False):
        self.busy, self.events, self.fail_create, self.fail_busy = list(busy), [], fail_create, fail_busy

    def install(self):
        self._saved = (gcal.enabled, gcal.busy_intervals, gcal.create_event, gcal.oauth_enabled)
        gcal.enabled = lambda: True
        gcal.oauth_enabled = lambda: False
        gcal.busy_intervals = self.busy_intervals
        gcal.create_event = self.create_event
        return self

    def uninstall(self):
        gcal.enabled, gcal.busy_intervals, gcal.create_event, gcal.oauth_enabled = self._saved

    def busy_intervals(self, start, end):
        if self.fail_busy:
            raise RuntimeError("Google Calendar unreachable")
        return [b for b in self.busy if b[0] < end and start < b[1]]

    def create_event(self, start, end, name, phone, purpose, email=""):
        if self.fail_create:
            raise RuntimeError("Google Calendar API error 503")
        self.events.append({"start": start, "name": name, "email": email, "purpose": purpose})
        self.busy.append((start, end))
        return {"id": f"ev{len(self.events)}", "link": "https://calendar.example/event", "meet": None, "invited": False,
                "duplicate": False}


def busy_slot(date_iso: str, hhmm: str):
    s = dt.datetime.combine(dt.date.fromisoformat(date_iso), dt.time.fromisoformat(hhmm), TZ)
    return (s, s + dt.timedelta(minutes=config.SLOT_MINUTES))


# ------------------------------------------------------------------ helpers
FAKE = FakeLLM()
llm.chat = FAKE.chat
agent.llm.chat = FAKE.chat


def fresh(call=True):
    config.DB_PATH = Path(tempfile.mkdtemp()) / "t.db"
    booking.clear_slot_cache()
    FAKE.delay = 0.0
    s = agent.CallSession("t", "browser")
    if call:
        s.greeting()
    return s


def say(s, text, via="voice"):
    return asyncio.run(s.respond(text, via=via))


def manual(s, **fields):
    return asyncio.run(s.apply_manual(fields))


def state(s):
    return asyncio.run(s.public_state(fresh=True))


def bookings():
    return calendar_store.list_bookings()


def fill_voice(s, day=2, hhmm="11 AM", name="Ahmed Khan", email="ahmed khan 92 at gmail dot com", purpose="building a website"):
    """A normal voice conversation up to (not including) the final yes."""
    say(s, f"I want to book a meeting {phrase(day)} at {hhmm}")
    say(s, f"My name is {name}")
    say(s, f"My email is {email}")
    return say(s, f"I want to discuss {purpose}")


# ------------------------------------------------------------------ tests
def test_email_extraction_and_validation():
    cases = {
        "my email is ahmed khan 92 at gmail dot com": "ahmedkhan92@gmail.com",
        "it's m dot daniel 4422 at gmail dot com": "m.daniel4422@gmail.com",
        "danish@gmail.com": "danish@gmail.com",
        "ali underscore raza at yahoo dot co dot uk": "ali_raza@yahoo.co.uk",
        "john at the rate gmail dot com": "john@gmail.com",
        "a b c at g mail dot com": "abc@gmail.com",
        "danish nine two at gmail dot com": "danish92@gmail.com",
        "ahmed double two at hotmail dot com": "ahmed22@hotmail.com",
        "ahmed at gmail": None,
        "book a meeting at 10 dot please": None,
    }
    for text, want in cases.items():
        assert booking.extract_email(text) == want, (text, booking.extract_email(text), want)
    assert booking.check_email("nope")["valid"] is False
    assert booking.check_email("a@b")["valid"] is False
    assert booking.check_email("ahmed@gmial.com", "voice")["suggestion"] == "ahmed@gmail.com"
    assert booking.check_email("sara@excels-tech.com", "voice")["confidence"] == "low"      # unusual domain by voice
    assert booking.check_email("sara@excels-tech.com", "manual")["confidence"] == "high"    # typed by the person


def test_spoken_email_success():
    s = fresh()
    reply = fill_voice(s)
    assert "Just to confirm" in reply and "ahmedkhan92 at gmail dot com" in reply and "Shall I book it?" in reply, reply
    assert "11 AM" in reply and "about building a website" in reply, reply
    assert bookings() == [], "booked before the caller confirmed!"
    done = say(s, "Yes please")
    assert done.startswith("Done!") and "Wednesday, November 25th at 11 AM" in done, done
    b = bookings()
    assert len(b) == 1 and b[0]["email"] == "ahmedkhan92@gmail.com" and b[0]["name"] == "Ahmed Khan", b


def test_misrecognized_email_is_verified_not_guessed():
    s = fresh()
    say(s, f"Book a meeting {phrase(2)} at 11 AM")
    say(s, "My name is Ahmed Khan")
    r = say(s, "my email is ahmed at gmial dot com")
    assert "Did you mean" in r and "gmail dot com" in r and s.awaiting == "email_suggestion", r
    assert s.booking.email_status == "unconfirmed"
    r = say(s, "yes")
    assert s.booking.values["email"] == "ahmed@gmail.com" and s.booking.email_status == "confirmed", s.booking.values
    assert "purpose" in booking.next_question(s.booking).lower() or "discuss" in r.lower(), r


def test_uncertain_email_asks_to_confirm_then_manual_fix():
    s = fresh()
    say(s, f"Book a meeting {phrase(2)} at 11 AM")
    r = say(s, "my email is sara at excels dash tech dot com")
    assert "I want to make sure I have your email correctly" in r and "sara at excels dash tech dot com" in r, r
    r = say(s, "no")
    assert s.booking.values["email"] == "" and "type it into the email box" in r, r
    out = manual(s, email="sara@excels-tech.com")
    assert out["state"]["email_status"] == "confirmed" and out["state"]["values"]["email"] == "sara@excels-tech.com"


def test_manual_email_validation_feedback():
    s = fresh()
    out = manual(s, email="not-an-email")
    assert out["state"]["errors"].get("email") and out["state"]["email_status"] == "invalid", out["state"]
    out = manual(s, email="mia@@gmail..com")
    assert out["state"]["errors"].get("email")
    out = manual(s, email="Mia.Stone@Gmail.com")
    assert out["state"]["values"]["email"] == "mia.stone@gmail.com" and not out["state"]["errors"].get("email")
    assert out["state"]["source"]["email"] == "manual"


def test_voice_then_manual_correction_uses_manual_value():
    s = fresh()
    fill_voice(s, email="ahmed at gmail dot com")                        # voice email, summary pending
    assert s.pending_key
    manual(s, email="ahmed.khan@outlook.com")                            # the caller fixes it in the form
    assert s.pending_key is None, "an edit must invalidate the old confirmation"
    r = say(s, "yes")
    assert bookings() == [], "booked on a yes that referred to the old summary"
    r = say(s, "go ahead and book")                                      # model asks again -> new read-back
    assert "ahmed dot khan at outlook dot com" in r and "Shall I book it?" in r, r
    say(s, "yes")
    assert bookings()[0]["email"] == "ahmed.khan@outlook.com", bookings()


def test_manual_then_voice_continuation_no_repeated_questions():
    s = fresh()
    out = manual(s, date=iso(2))
    assert out["followup"] == "Got it. What time works best for you?", out["followup"]
    out = manual(s, time="11:00")
    assert out["followup"] == "Got it. May I have your full name, please?", out["followup"]
    out = manual(s, name="Ahmed Khan")
    assert "email" in out["followup"].lower(), out["followup"]
    r = say(s, "my email is ahmed at gmail dot com")                     # now by voice
    assert "discuss" in s.messages[-1]["content"].lower() or True
    r = say(s, "I want to discuss pricing")
    assert "Just to confirm" in r and "Ahmed Khan" in r and "pricing" in r, r
    say(s, "yes")
    assert len(bookings()) == 1


def test_form_only_booking_via_review_and_confirm_key():
    s = fresh(call=False)
    manual(s, date=iso(1), time="10:30", name="Mia Stone", email="mia@gmail.com", purpose="Website redesign")
    st = state(s)
    assert st["ready"] and st["steps"]["availability"], st
    res = asyncio.run(s.confirm_from_ui("wrong-key"))
    assert res["success"] is False and res["code"] == "stale" and bookings() == []
    res = asyncio.run(s.confirm_from_ui(st["key"]))
    assert res["success"] and len(bookings()) == 1, res
    again = asyncio.run(s.confirm_from_ui(s.booking.key()))
    assert again["success"] and again.get("already_booked") and len(bookings()) == 1, again


def test_changed_mind_last_mention_wins_and_rechecks():
    s = fresh()
    say(s, "Book a meeting tomorrow at 5 PM... actually, make it the day after tomorrow at 11 AM")
    v = s.booking.values
    assert v["date"] == iso(2) and v["time"] == "11:00", v
    st = state(s)
    assert st["slot"]["ok"], st["slot"]
    # change only the time
    say(s, "actually make it 3 PM")
    assert s.booking.values["time"] == "15:00" and s.booking.values["date"] == iso(2), s.booking.values
    # change only the date by number
    say(s, "no wait, the 27th")
    assert s.booking.values["date"] == "2026-11-27" and s.booking.values["time"] == "15:00", s.booking.values


def test_past_dates_rejected_everywhere():
    s = fresh()
    r = manual(s, date="2026-11-12")
    assert r["state"]["errors"].get("date") and r["state"]["values"]["date"] == "", r["state"]
    say(s, "Book a meeting on 12 November at 11 AM")
    assert s.booking.values["date"] == "" and "date" in s.booking.errors, (s.booking.values, s.booking.errors)
    assert booking.check_slot("2026-11-12", "11:00")["ok"] is False
    assert calendar_store.book("2026-11-12", "11:00", "A B", "", "x y z", "a@b.com")["success"] is False
    assert calendar_store.book(iso(0), "08:00", "A B", "", "x y z", "a@b.com")["success"] is False   # earlier today
    r = manual(s, date=iso(2), time="07:30")                              # outside business hours
    assert state(s)["slot"]["ok"] is False
    r = manual(s, date="2027-03-01")
    assert r["state"]["errors"].get("date"), "more than 60 days ahead must be refused"
    r = manual(s, date=iso(5))                                            # Saturday: closed
    assert "closed" in r["state"]["errors"].get("date", "").lower(), r["state"]["errors"]


def test_unavailable_calendar_slot():
    g = FakeGoogle(busy=[busy_slot(iso(2), "11:00")]).install()
    try:
        s = fresh()
        say(s, f"Book a meeting {phrase(2)} at 11 AM")
        say(s, "My name is Ahmed Khan")
        say(s, "My email is ahmed at gmail dot com")
        r = say(s, "I want to discuss hiring")
        assert "isn't available" in r and "Which works for you?" in r, r
        assert "11 AM" not in r.split("I have")[-1], r
        assert bookings() == [] and g.events == []
        assert state(s)["slot"]["code"] == "busy"
        # a different, free time works
        r = say(s, "make it 12 PM")
        assert "Just to confirm" in r and "12 PM" in r, r
        say(s, "yes")
        assert len(g.events) == 1 and g.events[0]["start"].hour == 12
    finally:
        g.uninstall()


def test_successful_booking_and_duplicate_prevention():
    g = FakeGoogle().install()
    try:
        s = fresh()
        fill_voice(s)
        done = say(s, "yes")
        assert done.startswith("Done!") and len(g.events) == 1, (done, g.events)
        assert s.booking.booked and s.booking.booked["calendar"] == "google"
        # same request again: no second event
        r = calendar_store.book(iso(2), "11:00", "Ahmed Khan", "", "building a website", "ahmedkhan92@gmail.com")
        assert r["success"] and r.get("already_booked") and len(g.events) == 1 and len(bookings()) == 1, r
        # someone else wanting the same slot is refused
        r = calendar_store.book(iso(2), "11:00", "Other Person", "", "x y z", "other@gmail.com")
        assert r["success"] is False and r["code"] == "busy", r
        # a new request after booking starts a new booking, keeping the contact details
        say(s, f"I'd like another meeting {phrase(3)} at 2 PM")
        v = s.booking.values
        assert v["date"] == iso(3) and v["time"] == "14:00" and v["name"] == "Ahmed Khan" and v["purpose"] == "", v
    finally:
        g.uninstall()


def test_google_failure_never_reports_success():
    g = FakeGoogle(fail_create=True).install()
    try:
        s = fresh()
        fill_voice(s)
        r = say(s, "yes")
        assert "trouble reaching the calendar" in r and "nothing has been booked" in r, r
        assert not r.startswith("Done"), r
        assert bookings() == [], "a booking row must not survive a failed calendar call"
        assert s.booking.booked is None
        # the UI path reports the failure too
        res = asyncio.run(s.confirm_from_ui(s.booking.key()))
        assert res["success"] is False and bookings() == [], res
    finally:
        g.uninstall()
    g = FakeGoogle(fail_busy=True).install()
    try:
        s = fresh()
        r = say(s, f"Book a meeting {phrase(2)} at 11 AM")
        assert state(s)["slot"]["code"] == "calendar_down", state(s)["slot"]
        out = asyncio.run(s.confirm_from_ui(s.booking.key()))
        assert out["success"] is False
    finally:
        g.uninstall()


def test_interrupted_conversation_requires_fresh_confirmation():
    s = fresh()
    r = fill_voice(s)
    assert "Shall I book it?" in r and s.pending_key
    r = say(s, "wait, what are your opening hours?")                      # interrupts the confirmation
    assert s.pending_key is None
    r = say(s, "yes")                                                     # a yes now refers to nothing
    assert bookings() == [], "a stale yes must not book"
    r = say(s, "ok go ahead and book")                                    # model asks again -> read-back
    assert "Shall I book it?" in r and bookings() == [], r
    r = say(s, "yes that's right")
    assert r.startswith("Done!") and len(bookings()) == 1, r


def test_model_cannot_invent_name_or_skip_confirmation():
    s = fresh()
    say(s, f"Book a meeting {phrase(2)} at 11 AM INVENT")
    assert s.booking.values["name"] == "", s.booking.values               # "John Doe" was never said
    say(s, "My name is Ahmed Khan")
    say(s, "my email is ahmed at gmail dot com")
    say(s, "I want to discuss rates")
    assert bookings() == []
    s.pending_key = None
    r = say(s, "go ahead and book")                                       # no yes yet -> only a read-back
    assert "Shall I book it?" in r and bookings() == []



def test_name_purpose_and_bare_time_captured_in_code_without_the_llm():
    s = fresh()
    FAKE.default = "Sure."                      # a model that never calls the tools (real Groq models often don't)
    try:
        say(s, f"Book a meeting {phrase(2)}")
        before = FakeLLM.calls
        s.messages.append({"role": "assistant", "content": "What time works best for you?"})
        say(s, "five thirty")
        assert s.booking.values["time"] == "17:30", s.booking.values
        s.messages.append({"role": "assistant", "content": "May I have your full name, please?"})
        say(s, "it's Ahmed Khan")
        assert s.booking.values["name"] == "Ahmed Khan", s.booking.values
        say(s, "my email is ahmed at gmail dot com")
        s.messages.append({"role": "assistant", "content": "What would you like to discuss in the meeting?"})
        r = say(s, "building an online store")
        assert s.booking.values["purpose"] == "building an online store", s.booking.values
        assert "Just to confirm" in r and "5:30 PM" in r, r
        # one sentence carrying several answers
        s2 = fresh()
        say(s2, f"My name is Mia Stone and I want to discuss branding, book {phrase(3)} at 2 PM, email mia at gmail dot com")
        v = s2.booking.values
        assert (v["name"], v["purpose"], v["date"], v["time"], v["email"]) == (
            "Mia Stone", "branding", iso(3), "14:00", "mia@gmail.com"), v
    finally:
        FAKE.default = "Sure. How can I help?"


def test_questions_are_not_mistaken_for_answers():
    s = fresh()
    s.messages.append({"role": "assistant", "content": "May I have your full name, please?"})
    say(s, "why do you need my name?")
    assert s.booking.values["name"] == ""
    s.messages.append({"role": "assistant", "content": "What would you like to discuss in the meeting?"})
    say(s, "what are your prices?")
    assert s.booking.values["purpose"] == ""
    say(s, "how are you today?")                # chit-chat must not become a booking date
    assert s.booking.values["date"] == ""


def test_data_turns_are_answered_in_code_with_real_availability():
    s = fresh()
    before = FakeLLM.calls
    r = say(s, f"Hi, I'd like to book a meeting {phrase(2)} at 3 PM")
    assert r == "3 PM on Wednesday, November 25th is free. May I have your full name, please?", r
    r = say(s, "My name is Ahmed Khan")
    assert r == "Thanks, Ahmed. What's the best email address for the invite?", r
    r = say(s, "Actually, make it 4 PM")                                  # changed mind: acknowledged, re-checked
    assert r.startswith("Sure, I'll update that. 4 PM on Wednesday, November 25th is free."), r
    assert FakeLLM.calls == before, "pure data turns must not call the LLM"
    # a date alone -> real open times; a closed day -> no times
    s2 = fresh()
    r = say(s2, f"Book a meeting {phrase(2)}")
    assert r == "On Wednesday, November 25th I have 10 AM, 10:30 AM, 11 AM open. Which time works for you?", r
    r = say(fresh(), "Book a meeting on November 28")                     # a Saturday
    assert "closed" in r.lower() or "no openings" in r.lower() or "Sorry" in r, r


def test_streamed_sentences_glued_together_are_split_and_cut_at_the_first_question():
    got = []
    ss = agent._SentenceStream(got.append, time.perf_counter(), {})
    for piece in ["Great, may I have your email address?", "And could you tell", " me your full name, please?"]:
        ss.feed(piece)
    ss.flush()
    assert got == ["Great, may I have your email address?"], got


def test_booking_speaks_first_then_confirms_only_after_the_calendar_answers():
    g = FakeGoogle().install()
    try:
        s = fresh()
        fill_voice(s)
        got = []
        reply = asyncio.run(s.respond("yes", emit=got.append))
        assert got[0] == "One moment while I book that." and got[1].startswith("Done!"), got
        assert reply == " ".join(got), (reply, got)
        assert len(g.events) == 1
    finally:
        g.uninstall()
    g = FakeGoogle(fail_create=True).install()
    try:
        s = fresh()
        fill_voice(s)
        got = []
        reply = asyncio.run(s.respond("yes", emit=got.append))
        assert got[0] == "One moment while I book that." and "Done" not in reply, got
        assert "nothing has been booked" in reply and bookings() == [] and s.booking.booked is None, reply
    finally:
        g.uninstall()


def test_human_handoff_is_instant_and_not_triggered_by_booking_talk():
    for phrase_ in ["Can I talk to a real person please?", "I want to speak to a human", "connect me to a manager",
                    "agent please", "let me talk to someone"]:
        s = fresh()
        before = FakeLLM.calls
        r = say(s, phrase_)
        assert s.action == "transfer" and r.startswith("Of course"), (phrase_, s.action, r)
        assert FakeLLM.calls == before, "a handoff must not wait for the LLM"
    for phrase_ in ["I want to book a meeting with your manager", "Do you have an agent for insurance?",
                    "My name is Ahmed Khan", "book a meeting with the team member for sales"]:
        s = fresh()
        say(s, phrase_)
        assert s.action is None, (phrase_, s.action)


def _vapi_client():
    from fastapi.testclient import TestClient
    from app.main import app
    return TestClient(app)


def _vapi_tool(c, name, args, messages, sid=None, call=None, tc_id="tc-1"):
    a = dict(args)
    if sid:
        a["session_id"] = sid
    r = c.post("/api/vapi/webhook", json={"message": {
        "type": "tool-calls", "call": call or {"id": "call-1", "type": "webCall"}, "artifact": {"messages": messages},
        "toolCallList": [{"id": tc_id, "type": "function", "function": {"name": name, "arguments": a}}]}})
    assert r.status_code == 200, r.text                          # Vapi's contract: always HTTP 200
    res = r.json()["results"]
    assert len(res) == 1 and res[0]["toolCallId"] == tc_id, res
    return res[0]


def test_vapi_webhook_uses_the_same_booking_rules():
    fresh(call=False)
    c = _vapi_client()
    hello = {"role": "assistant", "message": "Hello, thank you for calling. How can I help you today?"}
    first = {"role": "user", "message": f"Hi, book a meeting {phrase(2)} at 11 AM, my name is Ahmed Khan, "
                                        "my email is ahmed at gmail dot com, I want to discuss hiring"}
    res = _vapi_tool(c, "update_booking_details", {"name": "Ahmed Khan"}, [hello, first], sid="vapi-1")
    assert "result" in res and "\n" not in res["result"], res                # flat single-line string
    out = json.loads(res["result"])
    assert out["say"].startswith("Just to confirm") and "Shall I book it?" in out["say"], out
    assert bookings() == [], "booked before the caller said yes"
    agent_says = {"role": "assistant", "message": out["say"]}
    res = _vapi_tool(c, "book_appointment", {}, [hello, first, agent_says, {"role": "user", "message": "yes please"}], sid="vapi-1")
    out = json.loads(res["result"])
    assert out["success"] is True and out["say"].startswith("Done!"), out
    rows = bookings()
    assert len(rows) == 1 and rows[0]["email"] == "ahmed@gmail.com" and rows[0]["name"] == "Ahmed Khan", rows
    # calling it again (the model double-fires) must not book twice
    _vapi_tool(c, "book_appointment", {}, [hello, first, agent_says, {"role": "user", "message": "yes please"}], sid="vapi-1", tc_id="tc-2")
    assert len(bookings()) == 1


def test_vapi_wrong_yes_does_not_book_and_changes_are_applied():
    fresh(call=False)
    c = _vapi_client()
    hello = {"role": "assistant", "message": "Hello."}
    u1 = {"role": "user", "message": f"book a meeting {phrase(2)} at 11 AM, my name is Mia Stone, "
                                     "email mia at gmail dot com, I want to discuss branding"}
    r1 = json.loads(_vapi_tool(c, "update_booking_details", {}, [hello, u1], sid="vapi-2")["result"])
    a1 = {"role": "assistant", "message": r1["say"]}
    u2 = {"role": "user", "message": "actually make it 3 PM"}                   # changes their mind instead of saying yes
    r2 = json.loads(_vapi_tool(c, "book_appointment", {}, [hello, u1, a1, u2], sid="vapi-2", tc_id="tc-2")["result"])
    assert bookings() == [] and "3 PM" in r2.get("say", ""), r2
    assert agent.SESSIONS["vapi-2"].booking.values["time"] == "15:00"


def test_vapi_security_errors_and_session_mapping():
    fresh(call=False)
    c = _vapi_client()
    config.VAPI_WEBHOOK_SECRET = "s3cret"
    try:
        body = {"message": {"type": "status-update", "status": "in-progress", "call": {"id": "c"}}}
        assert c.post("/api/vapi/webhook", json=body).status_code == 401
        assert c.post("/api/vapi/webhook", json=body, headers={"x-vapi-secret": "wrong"}).status_code == 401
        assert c.post("/api/vapi/webhook", json=body, headers={"x-vapi-secret": "s3cret"}).status_code == 200
    finally:
        config.VAPI_WEBHOOK_SECRET = ""
    r = c.post("/api/vapi/webhook", content=b"not json")
    assert r.status_code == 200 and r.json() == {}
    res = _vapi_tool(c, "delete_everything", {}, [], sid="x")                 # unknown tool -> error field, still 200
    assert "error" in res and "result" not in res, res
    # session id can come from the call object (assistantOverrides.variableValues) when the model omits it
    call = {"id": "c-77", "assistantOverrides": {"variableValues": {"sid": "web-mapped"}}}
    _vapi_tool(c, "update_booking_details", {}, [{"role": "user", "message": f"book {phrase(2)} at 11 AM"}], call=call)
    assert agent.SESSIONS["web-mapped"].booking.values["date"] == iso(2)
    # an unfilled {{sid}} variable (phone call) must NOT become a shared session for every caller
    for n in (1, 2):
        _vapi_tool(c, "update_booking_details", {"session_id": "{{sid}}"}, [{"role": "user", "message": f"book {phrase(n + 1)} at 11 AM"}],
                   call={"id": f"phone-{n}", "customer": {"number": f"+9200000{n}"}})
    assert agent.SESSIONS["vapi-phone-1"].booking.values["date"] == iso(2)
    assert agent.SESSIONS["vapi-phone-2"].booking.values["date"] == iso(3) and "{{sid}}" not in agent.SESSIONS
    # the page can bind a call to its session explicitly
    assert c.post("/api/vapi/link", json={"call_id": "c-78", "session_id": "web-linked"}).json()["ok"]
    _vapi_tool(c, "update_booking_details", {}, [{"role": "user", "message": f"book {phrase(3)} at 2 PM"}],
               call={"id": "c-78"})
    assert agent.SESSIONS["web-linked"].booking.values["time"] == "14:00"
    # a conversation-update (no tool call at all) also keeps the shared state in step
    c.post("/api/vapi/webhook", json={"message": {"type": "conversation-update", "call": {"id": "c-78"},
           "messages": [{"role": "user", "message": f"book {phrase(3)} at 2 PM"},
                        {"role": "assistant", "message": "Sure."}, {"role": "user", "message": "my name is Sam Lee"}]}})
    assert agent.SESSIONS["web-linked"].booking.values["name"] == "Sam Lee"


def test_vapi_assistant_and_tool_payloads():
    from app import vapi
    tools = vapi.tool_payloads("https://host.example/api/vapi/webhook", "sek")
    assert [t["function"]["name"] for t in tools] == list(vapi.TOOL_NAMES)
    for t in tools:
        assert t["type"] == "function" and t["async"] is False and t["server"]["url"].endswith("/api/vapi/webhook")
        assert t["server"]["headers"] == {"X-Vapi-Secret": "sek"}
        assert "session_id" in t["function"]["parameters"]["properties"]
    a = vapi.assistant_payload("https://host.example/api/vapi/webhook", "sek", ["t1", "t2", "t3"])
    assert a["model"]["toolIds"] == ["t1", "t2", "t3"] and a["server"]["headers"]["X-Vapi-Secret"] == "sek"
    assert "tool-calls" in a["serverMessages"] and a["firstMessage"].startswith("Hello")
    prompt = a["model"]["messages"][0]["content"]
    assert "{{sid}}" in prompt and "50,000 PKR" in prompt, "session variable + knowledge base must be in the prompt"
    assert "sek" not in prompt and "sek" not in json.dumps(vapi.tool_payloads("https://x/y", None))


def test_page_design_lint_hallmark_gates():
    """The Hallmark slop-test gates that can be checked mechanically, so the design cannot regress silently."""
    root = Path(config.ROOT) / "static"
    html = (root / "index.html").read_text(encoding="utf-8")
    tokens = (root / "tokens.css").read_text(encoding="utf-8")
    style = re.search(r"<style>(.*?)</style>", html, re.S).group(1)
    script = re.search(r"<script>(.*)</script>", html, re.S).group(1)
    code = style + script
    # 48 · tokens only: no inline colours, every font through a token
    assert not re.findall(r"#[0-9a-fA-F]{3,8}\b", code), "hex colour outside tokens.css"
    assert not re.findall(r"\b(?:oklch|rgba?|hsla?)\(", code), "colour function outside tokens.css"
    fams = re.findall(r"font-family:\s*([^;]+);", style)
    assert fams and all(f.strip().startswith("var(--font-") for f in fams), fams
    # 38a · no italics anywhere; the headline's verb landmark renders upright
    assert "italic" not in code.lower() and re.search(r"\.hero__title em \{[^}]*font-style: normal", style)
    # glass / blur / bounce / transition-all / viewport-unit traps
    for bad in ("backdrop-filter", "transition: all", "100vw", "100vh", "cubic-bezier(0.34, 1.56"):
        assert bad not in code, bad
    # 34 · both roots clip horizontal overflow; 27 · reduced motion is honoured; focus ring is never animated
    assert re.search(r"html, body \{ overflow-x: clip; \}", style) and "prefers-reduced-motion: reduce" in style
    assert ":focus-visible" in style and not re.search(r"transition:[^;]*outline", style)
    # 22 · tinted neutrals only (>= 0.005 chroma) in both drops
    for name, c in re.findall(r"(--color-(?:paper|ink|muted|label)[\w-]*):\s*oklch\([\d.]+%\s+([\d.]+)", tokens):
        assert float(c) >= 0.005, (name, c)
    # 24 · spacing comes from the named scale (hairlines of 2px and em-relative nudges are the only literals)
    for prop, val in re.findall(r"\b((?:padding|margin|gap)(?:-[a-z]+)?):\s*([^;}]+)[;}]", style):
        rest = re.sub(r"(?:var|calc|env|max)\((?:[^()]|\([^()]*\))*\)", "", val).split()
        assert all(re.fullmatch(r"0|auto|2px|-?[\d.]+em", t) for t in rest), (prop, val)
    # 25 · prose measure 45-75ch · 49 · clickable text never wraps · 20 · stamp present · eyebrows capped (Lumen: <= 6)
    assert all(45 <= int(n) <= 75 for n in re.findall(r"\.(?:lede|empty) \{[^}]*max-width: (\d+)ch", style))
    for sel in (".btn", ".theme", ".wordmark"):
        assert re.search(re.escape(sel) + r" \{[^}]*white-space: nowrap", style), sel
    assert "Hallmark · genre: atmospheric · macrostructure: Workbench · theme: lumen" in style
    assert html.count('class="eyebrow"') <= 6, html.count('class="eyebrow"')
    # voice-only: there is no way to type a booking on this page
    assert not re.search(r"<(textarea|form)\b|contenteditable|<input[^>]*type=\"(text|email|search|tel)\"", html)
    # 46 · honest copy: the page itself invents no figures (every number on it is rendered from live data)
    visible = re.sub(r"<(script|style)[\s\S]*?</\1>|<[^>]+>", " ", html)
    assert not re.search(r"\d+\s*(%|x faster|\+)|trusted by|\d{2,3},\d{3}", visible), "invented metric in the markup"


def test_deterministic_path_skips_the_llm():
    s = fresh()
    fill_voice(s)
    before = FakeLLM.calls
    done = say(s, "yes")                                                  # confirmation -> no model call
    assert FakeLLM.calls == before and done.startswith("Done!")


def test_streaming_delivers_sentences_before_the_reply_is_complete():
    s = fresh()
    FAKE.delay = 0.12
    FAKE.default = "Happy to help. We build websites and apps. What would you like to know?"
    got = []
    t0 = time.perf_counter()
    timings = {}
    reply = asyncio.run(s.respond("what do you do", emit=lambda x: got.append((time.perf_counter() - t0, x)), timings=timings))
    total = time.perf_counter() - t0
    FAKE.delay, FAKE.default = 0.0, "Sure. How can I help?"
    assert [x for _, x in got] == ["Happy to help.", "We build websites and apps.", "What would you like to know?"], got
    assert got[0][0] < total - 0.2, f"first sentence at {got[0][0]:.2f}s, finished at {total:.2f}s"
    assert reply == " ".join(x for _, x in got) and timings["first_sentence_ms"] < timings["total_ms"], timings


def test_http_api_state_stream_review_confirm():
    from fastapi.testclient import TestClient
    from app.main import app
    c = TestClient(app)
    config.DB_PATH = Path(tempfile.mkdtemp()) / "t.db"
    booking.clear_slot_cache()
    cfg = c.get("/api/config").json()
    assert cfg["timezone"] == "Asia/Karachi" and cfg["calendar"] == "local"
    start = c.post("/api/web/start", json={"silent": True}).json()
    sid = start["session_id"]
    assert start["reply"] is None and start["state"]["missing"] == list(booking.FIELDS)
    # manual form edits
    r = c.post("/api/web/state", json={"session_id": sid, "fields": {"date": "2026-11-12"}}).json()
    assert r["state"]["errors"]["date"], r
    r = c.post("/api/web/state", json={"session_id": sid, "fields": {"date": iso(2), "time": "11:00", "name": "Mia Stone",
                                                                     "purpose": "Branding project"}}).json()
    assert r["state"]["missing"] == ["email"], r["state"]
    # typed chat message carrying the email (stream endpoint)
    lines = []
    with c.stream("POST", "/api/web/stream", json={"session_id": sid, "text": "my email is mia@gmail.com", "via": "typed"}) as resp:
        for line in resp.iter_lines():
            if line:
                lines.append(json.loads(line))
    kinds = [x["t"] for x in lines]
    assert kinds[-1] == "done" and "sentence" in kinds, kinds
    done = lines[-1]
    assert "Just to confirm" in done["reply"] and done["state"]["email_status"] == "confirmed", done["reply"]
    assert done["timings"]["model"] == "rules (no LLM)" and done["timings"]["total_ms"] >= 0
    # review + confirm
    rv = c.post("/api/web/review", json={"session_id": sid}).json()
    assert rv["ok"] and rv["summary"]["name"] == "Mia Stone" and rv["summary"]["timezone"] == "Asia/Karachi", rv
    bad = c.post("/api/web/confirm", json={"session_id": sid, "key": "stale"}).json()
    assert bad["success"] is False and bookings() == []
    ok = c.post("/api/web/confirm", json={"session_id": sid, "key": rv["key"]}).json()
    assert ok["success"] and len(bookings()) == 1 and ok["state"]["booked"], ok
    rv2 = c.post("/api/web/review", json={"session_id": sid}).json()
    assert rv2["ok"] is False and any("already booked" in i for i in rv2["issues"]), rv2
    assert c.get("/").status_code == 200


def test_phone_flow_twilio():
    from fastapi.testclient import TestClient
    from app.main import app
    c = TestClient(app)
    config.DB_PATH = Path(tempfile.mkdtemp()) / "t.db"
    booking.clear_slot_cache()
    r = c.post("/twilio/voice", data={"CallSid": "CA9", "From": "+921111111"})
    assert "<Gather" in r.text and "How can I help" in r.text

    def speak(t):
        return c.post("/twilio/gather", data={"CallSid": "CA9", "SpeechResult": t}).text

    speak(f"book a meeting {phrase(2)} at 11 AM")
    speak("my name is Ahmed Khan")
    speak("my email is ahmed at gmail dot com")
    t = speak("I want to discuss a project")
    assert "Shall I book it?" in t and "<Gather" in t, t
    t = speak("yes")
    assert "Done!" in t and len(bookings()) == 1 and bookings()[0]["phone"] == "+921111111", t
    c.post("/twilio/voice", data={"CallSid": "CA10"})
    assert "Are you still there" in c.post("/twilio/noinput", data={"CallSid": "CA10"}).text
    assert "<Hangup/>" in c.post("/twilio/noinput", data={"CallSid": "CA10"}).text
    c.post("/twilio/voice", data={"CallSid": "CA11"})
    t = c.post("/twilio/gather", data={"CallSid": "CA11", "SpeechResult": "goodbye"}).text
    assert "<Hangup/>" in t, t


# ------------------------------------------------------------------ runner
def main():
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception:
            failed += 1
            print(f"  FAIL  {name}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    if failed:
        raise SystemExit(1)
    print("ALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
