"""Appointment slots stored in a local SQLite file (data/bookings.db)."""
import datetime as dt
import logging
import sqlite3
from zoneinfo import ZoneInfo

from . import config, gcal

TZ = ZoneInfo(config.TIMEZONE)
log = logging.getLogger("calendar")


def _conn():
    c = sqlite3.connect(config.DB_PATH)
    c.row_factory = sqlite3.Row
    c.execute(
        """CREATE TABLE IF NOT EXISTS bookings(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            start TEXT UNIQUE NOT NULL,
            name TEXT, phone TEXT, purpose TEXT,
            created_at TEXT, email TEXT)"""
    )
    cols = {r["name"] for r in c.execute("PRAGMA table_info(bookings)")}
    if "email" not in cols:  # upgrade databases created before the email field existed
        c.execute("ALTER TABLE bookings ADD COLUMN email TEXT")
    return c


def now() -> dt.datetime:
    return dt.datetime.now(TZ)


def _parse_date(s: str) -> dt.date:
    return dt.date.fromisoformat(s.strip())


def _parse_time(s: str) -> dt.time:
    s = s.strip().upper().replace(".", "")
    for fmt in ("%H:%M", "%I:%M %p", "%I %p", "%I:%M%p", "%I%p", "%H"):
        try:
            return dt.datetime.strptime(s, fmt).time()
        except ValueError:
            pass
    raise ValueError(f"Could not understand time '{s}'. Use HH:MM (24h).")


def _all_slots(day: dt.date) -> list[dt.datetime]:
    if day.weekday() not in config.WORK_DAYS:
        return []
    start = dt.datetime.combine(day, _parse_time(config.OPEN_TIME), TZ)
    end = dt.datetime.combine(day, _parse_time(config.CLOSE_TIME), TZ)
    step = dt.timedelta(minutes=config.SLOT_MINUTES)
    slots, t = [], start
    while t + step <= end:
        slots.append(t)
        t += step
    return slots


def _booked(day: dt.date) -> set[str]:
    with _conn() as c:
        rows = c.execute("SELECT start FROM bookings WHERE start LIKE ?", (f"{day.isoformat()}%",))
        return {r["start"] for r in rows}


def available_slots(date: str) -> dict:
    try:
        day = _parse_date(date)
    except ValueError:
        return {"error": "Invalid date. Use YYYY-MM-DD."}
    spoken = day.strftime("%A, %B ") + str(day.day)
    today = now().date()
    if day < today:
        return {"error": f"{spoken}, {day.year} is in the past. Today is {today.strftime('%A, %B ')}{today.day}. "
                         "Ask the caller for a future date."}
    if day > today + dt.timedelta(days=60):
        return {"error": f"{spoken}, {day.year} is too far ahead. Bookings are only possible within 60 days."}
    slots = _all_slots(day)
    if not slots:
        return {"date": date, "date_spoken": spoken, "available": [], "note": "Closed on this day. Suggest another day."}
    booked = _booked(day)
    step = dt.timedelta(minutes=config.SLOT_MINUTES)
    busy = []
    if gcal.enabled():
        try:
            busy = gcal.busy_intervals(slots[0], slots[-1] + step)
        except Exception as e:
            log.exception("Google freeBusy failed")
            return {"error": f"Calendar unavailable right now: {e}"}
    current = now()
    free = [
        s.strftime("%H:%M") for s in slots
        if s > current and s.isoformat() not in booked
        and not any(b0 < s + step and s < b1 for b0, b1 in busy)
    ]
    return {"date": date, "date_spoken": spoken, "available": free}


def book(date: str, time: str, name: str, phone: str = "", purpose: str = "", email: str = "") -> dict:
    try:
        day = _parse_date(date)
        t = _parse_time(time)
    except ValueError as e:
        return {"success": False, "error": str(e)}
    start = dt.datetime.combine(day, t, TZ)
    valid = {s.isoformat() for s in _all_slots(day)}
    if start.isoformat() not in valid:
        return {"success": False, "error": "That time is not a valid slot (outside working hours or not on slot boundary)."}
    if start <= now():
        return {"success": False, "error": "That time is in the past."}
    try:
        with _conn() as c:
            c.execute(
                "INSERT INTO bookings(start, name, phone, purpose, created_at, email) VALUES(?,?,?,?,?,?)",
                (start.isoformat(), name, phone, purpose, now().isoformat(), email),
            )
    except sqlite3.IntegrityError:
        return {"success": False, "error": "Slot already booked.", "alternatives": available_slots(date).get("available", [])[:5]}

    invite_sent = False
    if gcal.enabled():
        end = start + dt.timedelta(minutes=config.SLOT_MINUTES)
        try:
            if gcal.busy_intervals(start, end):
                _delete(start)
                return {"success": False, "error": "Slot already busy in calendar.",
                        "alternatives": available_slots(date).get("available", [])[:5]}
            ev = gcal.create_event(start, end, name, phone, purpose, email)
            log.info("Google Calendar event created: %s (invite sent: %s)", ev.get("link"), ev.get("invited"))
            invite_sent = bool(ev.get("invited"))
        except Exception as e:
            log.exception("Google Calendar booking failed")
            _delete(start)
            return {"success": False, "error": f"Calendar error, booking not saved: {e}"}
    return {"success": True, "date": date, "time": start.strftime("%H:%M"), "weekday": day.strftime("%A"), "name": name,
            "invite_sent": invite_sent}


def _delete(start: dt.datetime):
    with _conn() as c:
        c.execute("DELETE FROM bookings WHERE start = ?", (start.isoformat(),))


def list_bookings() -> list[dict]:
    with _conn() as c:
        return [dict(r) for r in c.execute("SELECT * FROM bookings ORDER BY start")]
