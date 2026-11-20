"""Call history: every conversation (who said what, and how it ended) kept in SQLite next to the bookings.

Text transcripts only - no audio is stored. The web page asks only for the call ids it created itself (kept in that
browser), so on a public deployment one visitor never sees another visitor's calls."""
import datetime as dt
import json
import logging
import sqlite3

from . import config

log = logging.getLogger("callog")


def _conn():
    c = sqlite3.connect(config.DB_PATH)
    c.row_factory = sqlite3.Row
    c.execute(
        """CREATE TABLE IF NOT EXISTS calls(
            sid TEXT PRIMARY KEY, channel TEXT, caller TEXT,
            started TEXT, updated TEXT,
            name TEXT, email TEXT, outcome TEXT, turns TEXT)""")
    return c


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def outcome_of(session) -> str:
    if session.booking.booked:
        return "booked"
    if session.action == "transfer":
        return "transferred to a human"
    if session.action == "hangup":
        return "ended by the agent"
    return "no booking"


def add(session, *turns: tuple[str, str]) -> None:
    """Append (role, text) turns to this call and refresh its summary. Never raises: history must not break a call."""
    turns = [(r, t.strip()) for r, t in turns if t and t.strip()]
    if not turns:
        return
    try:
        stamp = _now()
        with _conn() as c:
            row = c.execute("SELECT turns FROM calls WHERE sid=?", (session.id,)).fetchone()
            old = json.loads(row["turns"]) if row else []
            old += [{"role": r, "text": t, "at": stamp} for r, t in turns]
            v = session.booking.values
            channel = "phone" if str(session.id).startswith("CA") or (session.caller not in (None, "browser")) else "web"
            if row:
                c.execute("UPDATE calls SET updated=?, name=?, email=?, outcome=?, turns=? WHERE sid=?",
                          (stamp, v["name"], v["email"], outcome_of(session), json.dumps(old), session.id))
            else:
                c.execute("INSERT INTO calls(sid, channel, caller, started, updated, name, email, outcome, turns)"
                          " VALUES(?,?,?,?,?,?,?,?,?)",
                          (session.id, channel, session.caller if channel == "phone" else "", stamp, stamp,
                           v["name"], v["email"], outcome_of(session), json.dumps(old)))
    except Exception:
        log.exception("call history write failed")


def _summary(r: sqlite3.Row) -> dict:
    turns = json.loads(r["turns"] or "[]")
    first_user = next((t["text"] for t in turns if t["role"] == "user"), "")
    start = dt.datetime.fromisoformat(r["started"])
    end = dt.datetime.fromisoformat(r["updated"])
    return {"sid": r["sid"], "channel": r["channel"], "started": r["started"], "seconds": int((end - start).total_seconds()),
            "name": r["name"] or "", "email": r["email"] or "", "outcome": r["outcome"], "turns": len(turns),
            "preview": first_user[:90]}


def list_calls(sids: list[str], limit: int = 30) -> list[dict]:
    sids = [s for s in sids if s][:100]
    if not sids:
        return []
    with _conn() as c:
        q = ",".join("?" * len(sids))
        rows = c.execute(f"SELECT * FROM calls WHERE sid IN ({q}) ORDER BY started DESC LIMIT ?", (*sids, limit)).fetchall()
    return [_summary(r) for r in rows]


def get_call(sid: str) -> dict | None:
    with _conn() as c:
        r = c.execute("SELECT * FROM calls WHERE sid=?", (sid,)).fetchone()
    if not r:
        return None
    return {**_summary(r), "transcript": json.loads(r["turns"] or "[]")}
