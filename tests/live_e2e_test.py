"""LIVE end-to-end check: real LLM (Groq/Gemini) + real Google Calendar through the real HTTP API.
Creates ONE test event, verifies it in Google Calendar, then deletes it. Uses a temporary booking DB.

Run:  .venv\\Scripts\\python -m tests.live_e2e_test
Burns a few free-tier LLM requests - don't run it in a loop.
"""
import datetime as dt
import json
import tempfile
import time
from pathlib import Path

from app import booking, calendar_store, config, gcal

config.DB_PATH = Path(tempfile.mkdtemp()) / "live_e2e.db"
from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

assert gcal.enabled(), "Google Calendar is not configured (.env GOOGLE_CALENDAR_ID + service_account.json)"
NAME = "Upgrade Check"


def pick_slots():
    """Two free slots on the next working day, so the 'changed my mind' step is real."""
    d = calendar_store.now().date() + dt.timedelta(days=1)
    for _ in range(10):
        if d.weekday() in config.WORK_DAYS:
            free = calendar_store.available_slots(d.isoformat()).get("available", [])
            late = [t for t in free if t >= "15:00"]
            if len(late) >= 2:
                return d, late[0], late[1]
        d += dt.timedelta(days=1)
    raise SystemExit("no free slots found in the next 10 days")


def spoken(t):
    h, m = map(int, t.split(":"))
    return f"{h % 12 or 12}{':%02d' % m if m else ''} {'AM' if h < 12 else 'PM'}"


def turn(c, sid, text, via="typed"):
    t0 = time.perf_counter()
    first, lines = None, []
    with c.stream("POST", "/api/web/stream", json={"session_id": sid, "text": text, "via": via}) as r:
        for line in r.iter_lines():
            if not line:
                continue
            ev = json.loads(line)
            lines.append(ev)
            if ev["t"] == "sentence" and first is None:
                first = int((time.perf_counter() - t0) * 1000)
    done = lines[-1]
    assert done["t"] == "done", lines
    tm = done["timings"]
    print(f"\nYOU  : {text}\nAGENT: {done['reply']}\n       [{tm.get('model')} | server total {tm.get('total_ms')} ms | "
          f"first sentence {tm.get('first_sentence_ms')} ms | llm {tm.get('llm_ms')} ms | http first byte->sentence {first} ms]")
    return done


def main():
    c = TestClient(app)
    day, t1, t2 = pick_slots()
    phrase = f"{day.strftime('%B')} {day.day}"
    sid = c.post("/api/web/start", json={"silent": False}).json()["session_id"]
    event_ids = []
    try:
        turn(c, sid, f"Hi, I'd like to book a meeting on {phrase} at {spoken(t1)}")
        d = turn(c, sid, f"Actually, make it {spoken(t2)} instead")
        assert d["state"]["values"]["time"] == t2 and d["state"]["values"]["date"] == day.isoformat(), d["state"]["values"]
        turn(c, sid, f"My name is {NAME}")
        d = turn(c, sid, "my email is upgrade check 77 at gmail dot com")
        assert d["state"]["values"]["email"] == "upgradecheck77@gmail.com" and d["state"]["email_status"] == "confirmed"
        d = turn(c, sid, "I want to discuss testing the new upgrade")
        assert "Just to confirm" in d["reply"] and spoken(t2) in d["reply"], d["reply"]
        assert calendar_store.list_bookings() == [], "booked before the caller confirmed!"
        d = turn(c, sid, "yes")
        assert d["reply"].startswith("One moment while I book that.") and "Done!" in d["reply"], d["reply"]
        st = d["state"]
        assert st["booked"] and st["booked"]["calendar"] == "google", st["booked"]
        print("\nbooked:", st["booked"]["when"], "|", st["booked"].get("event_link"))

        # verify in the REAL Google Calendar
        start = dt.datetime.combine(day, dt.time.fromisoformat(t2), calendar_store.TZ)
        r = gcal._http.get(f"{gcal.API}/calendars/{config.GOOGLE_CALENDAR_ID}/events", headers=gcal._headers(), params={
            "timeMin": start.isoformat(), "timeMax": (start + dt.timedelta(minutes=config.SLOT_MINUTES)).isoformat(),
            "q": NAME, "singleEvents": "true"})
        items = r.json().get("items", [])
        event_ids = [i["id"] for i in items]
        assert len(items) == 1, items
        ev = items[0]
        assert ev["start"]["timeZone"] == config.TIMEZONE and "upgradecheck77@gmail.com" in ev["description"], ev
        print("Google event:", ev["summary"], ev["start"]["dateTime"], ev["start"]["timeZone"])

        # a second 'yes' / double-click must not create another event
        again = c.post("/api/web/confirm", json={"session_id": sid, "key": st["key"]}).json()
        assert again["success"] and again["result"].get("already_booked"), again
        r = gcal._http.get(f"{gcal.API}/calendars/{config.GOOGLE_CALENDAR_ID}/events", headers=gcal._headers(), params={
            "timeMin": start.isoformat(), "timeMax": (start + dt.timedelta(minutes=config.SLOT_MINUTES)).isoformat(), "q": NAME})
        assert len(r.json().get("items", [])) == 1, "duplicate event created!"
        print("duplicate confirm -> still exactly 1 event")

        # the slot is now taken: a new session must not be offered it
        sid2 = c.post("/api/web/start", json={"silent": True}).json()["session_id"]
        r = c.post("/api/web/state", json={"session_id": sid2, "fields": {"date": day.isoformat(), "time": t2}}).json()
        assert r["state"]["slot"]["ok"] is False and r["state"]["slot"]["code"] == "busy", r["state"]["slot"]
        print("taken slot is refused for the next caller:", r["state"]["slot"]["error"])
        print("\nLIVE E2E PASSED")
    finally:
        # always clean up the real calendar - by searching for the test name, so it works even if an assert failed above
        lo = dt.datetime.combine(day, dt.time(0, 0), calendar_store.TZ)
        r = gcal._http.get(f"{gcal.API}/calendars/{config.GOOGLE_CALENDAR_ID}/events", headers=gcal._headers(), params={
            "timeMin": lo.isoformat(), "timeMax": (lo + dt.timedelta(days=1)).isoformat(), "q": NAME, "singleEvents": "true"})
        ids = {i["id"] for i in r.json().get("items", [])} | set(event_ids)
        for eid in ids:
            gcal._http.delete(f"{gcal.API}/calendars/{config.GOOGLE_CALENDAR_ID}/events/{eid}", headers=gcal._headers())
        print(f"cleanup: deleted {len(ids)} test event(s) from Google Calendar")


if __name__ == "__main__":
    main()
