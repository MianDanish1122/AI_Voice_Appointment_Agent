"""Offline smoke test with a fake LLM (no API key needed). Run: .venv\\Scripts\\python -m tests.smoke_test"""
import asyncio
import datetime as dt
import json
import os
import tempfile
from pathlib import Path

from app import calendar_store, config

config.DB_PATH = Path(tempfile.mkdtemp()) / "test.db"
config.HUMAN_PHONE = "+920000000000"
config.GOOGLE_CALENDAR_ID = ""  # never touch the real Google Calendar from this offline test

from fastapi.testclient import TestClient  # noqa: E402

from app import agent, llm  # noqa: E402
from app.main import app  # noqa: E402


def next_workday() -> str:
    d = calendar_store.now().date() + dt.timedelta(days=1)
    while d.weekday() not in config.WORK_DAYS:
        d += dt.timedelta(days=1)
    return d.isoformat()


DAY = next_workday()
script = []  # queue of fake LLM responses


def tc(name, args):
    return {"tool_calls": [{"id": f"c{len(script)}", "type": "function",
                            "function": {"name": name, "arguments": json.dumps(args)}}], "content": None}


async def fake_chat(messages, tools=None):
    return script.pop(0)

llm.chat = fake_chat
agent.llm.chat = fake_chat

c = TestClient(app)

# KB
from app.kb import kb  # noqa: E402
assert "50,000" in kb.search("website price"), "KB search failed"

# Calendar
s = calendar_store.available_slots(DAY)
assert s["available"], s
assert calendar_store.book(DAY, "11:00", "Ali")["success"]
assert not calendar_store.book(DAY, "11:00", "Bilal")["success"], "double booking allowed!"
assert "11:00" not in calendar_store.available_slots(DAY)["available"]

# Twilio flow: answer -> book -> transfer
r = c.post("/twilio/voice", data={"CallSid": "CA1", "From": "+920000"})
assert "<Gather" in r.text and "How can I help" in r.text

script += [tc("check_available_slots", {"date": DAY}), {"content": "We have 10:00 and 10:30 free. Which one?"}]
r = c.post("/twilio/gather", data={"CallSid": "CA1", "SpeechResult": "book a meeting tomorrow"})
assert "10 AM" in r.text and "<Gather" in r.text

# placeholder name must be rejected; first real call only asks for confirmation (nothing booked yet)
script += [tc("book_appointment", {"date": DAY, "time": "10:30", "name": "[Name]", "email": "danish@test.com", "purpose": "website design"}),
           tc("book_appointment", {"date": DAY, "time": "10:30", "name": "Danish", "email": "danish at test dot com", "purpose": "website design"})]
r = c.post("/twilio/gather", data={"CallSid": "CA1", "SpeechResult": "10:30, my name is Danish, email danish at test dot com, it is about website design"})
assert "Shall I book it?" in r.text and "10:30 AM" in r.text, r.text
assert not any(b["name"] in ("Danish", "[Name]") for b in c.get("/api/bookings").json()), "booked without confirmation!"
script += [tc("book_appointment", {"date": DAY, "time": "10:30", "name": "Danish", "email": "danish at test dot com", "purpose": "website design"})]
r = c.post("/twilio/gather", data={"CallSid": "CA1", "SpeechResult": "yes"})
assert "is booked for" in r.text, r.text
assert not script, "unused fake LLM replies left"

# past dates are refused
assert "past" in calendar_store.available_slots("2026-01-05")["error"]
assert any(b["name"] == "Danish" and b["phone"] == "+920000" for b in c.get("/api/bookings").json())

script += [tc("transfer_to_human", {"reason": "asked"}), {"content": "Connecting you now."}]
r = c.post("/twilio/gather", data={"CallSid": "CA1", "SpeechResult": "talk to a human"})
assert "<Dial>" in r.text, r.text

# Silence handling
c.post("/twilio/voice", data={"CallSid": "CA2"})
assert "Are you still there" in c.post("/twilio/noinput", data={"CallSid": "CA2"}).text
assert "<Hangup/>" in c.post("/twilio/noinput", data={"CallSid": "CA2"}).text

# Browser flow: model wrongly calls end_call right after checking slots -> call must stay open
sid = c.post("/api/web/start").json()["session_id"]
script += [{"tool_calls": [{"id": "a", "type": "function", "function": {"name": "check_available_slots", "arguments": json.dumps({"date": DAY})}},
                           {"id": "b", "type": "function", "function": {"name": "end_call", "arguments": "{}"}}], "content": None},
           {"content": "We have 10:00 and 11:30. Which time works for you?"}]
r = c.post("/api/web/message", json={"session_id": sid, "text": "20 Nov to discuss business"}).json()
assert r["action"] is None, r
script += [tc("end_call", {}), {"content": "Which time works for you?"}]
r = c.post("/api/web/message", json={"session_id": sid, "text": "hmm"}).json()
assert r["action"] is None, r
script += [tc("end_call", {}), {"content": "Goodbye!"}]
r = c.post("/api/web/message", json={"session_id": sid, "text": "bye"}).json()
assert r["action"] == "hangup"

assert c.get("/").status_code == 200

# Google Calendar path (faked): busy 14:00-15:00 must be hidden; booking creates an event
from app import gcal  # noqa: E402
TZ = calendar_store.TZ
d = dt.date.fromisoformat(DAY)
busy = [(dt.datetime.combine(d, dt.time(14, 0), TZ), dt.datetime.combine(d, dt.time(15, 0), TZ))]
events = []
gcal.enabled = lambda: True
gcal.busy_intervals = lambda s, e: [b for b in busy if b[0] < e and s < b[1]]
gcal.create_event = lambda s, e, n, p, pu, em="": events.append((s, n)) or {"id": "x", "link": "http://cal"}
free = calendar_store.available_slots(DAY)["available"]
assert "14:00" not in free and "14:30" not in free and "15:00" in free, free
assert not calendar_store.book(DAY, "14:30", "Zara")["success"]
assert calendar_store.book(DAY, "15:00", "Zara")["success"] and events[-1][1] == "Zara"
assert not any(b["name"] == "Zara" and "14:30" in b["start"] for b in calendar_store.list_bookings())
print("ALL SMOKE TESTS PASSED")
