"""Live Google Calendar test: read busy times, book a test slot through the real booking code,
verify the event exists in Google Calendar, then delete it. Run: .venv\\Scripts\\python -m tests.gcal_live_test"""
import datetime as dt
import tempfile
from pathlib import Path

import httpx

from app import calendar_store, config, gcal

config.DB_PATH = Path(tempfile.mkdtemp()) / "gcal_test.db"
cal = config.GOOGLE_CALENDAR_ID
print("Calendar:", cal, "| enabled:", gcal.enabled())

# 1) access check
r = httpx.get(f"{gcal.API}/calendars/{cal}", headers=gcal._headers(), timeout=20)
print("1) Calendar access:", r.status_code, r.json().get("summary") if r.status_code == 200 else r.text[:300])
if r.status_code != 200:
    raise SystemExit("-> Calendar not shared with the service account yet.")

# 2) free slots (uses Google freeBusy)
d = calendar_store.now().date() + dt.timedelta(days=1)
while d.weekday() not in config.WORK_DAYS:
    d += dt.timedelta(days=1)
slots = calendar_store.available_slots(d.isoformat())
print("2) Free slots", d, ":", slots.get("available", slots))
slot = slots["available"][-1]

# 3) book through the real booking code -> creates Google event
res = calendar_store.book(d.isoformat(), slot, "TEST Ahmed (AI agent test)", "+923000000000", "test booking")
print("3) Booking result:", res)

# 4) verify event exists and that slot now shows busy
start = dt.datetime.combine(d, dt.time.fromisoformat(slot), calendar_store.TZ)
end = start + dt.timedelta(minutes=config.SLOT_MINUTES)
ev = httpx.get(f"{gcal.API}/calendars/{cal}/events", headers=gcal._headers(), timeout=20,
               params={"timeMin": start.isoformat(), "timeMax": end.isoformat(), "q": "TEST Ahmed"}).json()
items = ev.get("items", [])
print("4) Event in Google Calendar:", [(i["summary"], i["start"]["dateTime"]) for i in items])
print("   Busy from Google now:", gcal.busy_intervals(start, end))

# 5) cleanup
for i in items:
    httpx.delete(f"{gcal.API}/calendars/{cal}/events/{i['id']}", headers=gcal._headers(), timeout=20).raise_for_status()
print("5) Test event deleted:", len(items))
print("GOOGLE CALENDAR TEST PASSED" if res.get("success") and items else "GOOGLE CALENDAR TEST FAILED")
