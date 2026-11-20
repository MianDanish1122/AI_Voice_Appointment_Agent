"""Google Calendar via a (free) service account.

Setup: create a service account in Google Cloud, enable "Google Calendar API", download its JSON key,
then in Google Calendar -> your calendar -> Settings -> "Share with specific people" add the service
account email with "Make changes to events".
"""
import base64
import datetime as dt
import hashlib
import json
import logging
import threading
import time

import httpx

from . import config

log = logging.getLogger("gcal")
SCOPES = ["https://www.googleapis.com/auth/calendar"]
API = "https://www.googleapis.com/calendar/v3"
_creds = None
_creds_lock = threading.Lock()
# One shared connection pool: avoids a fresh TLS handshake on every availability check / booking.
_http = httpx.Client(timeout=httpx.Timeout(15.0, connect=6.0), limits=httpx.Limits(max_keepalive_connections=4, keepalive_expiry=60))
_oauth_cache: tuple[float, dict | None] = (0.0, None)


def enabled() -> bool:
    return bool(config.GOOGLE_CALENDAR_ID and (config.GOOGLE_SERVICE_ACCOUNT_JSON or config.GOOGLE_SERVICE_ACCOUNT_FILE
                                               or oauth_enabled()))


def _oauth_info() -> dict | None:
    global _oauth_cache
    if time.monotonic() - _oauth_cache[0] < 30:
        return _oauth_cache[1]
    info = None
    if config.GOOGLE_OAUTH_TOKEN_JSON:
        info = json.loads(config.GOOGLE_OAUTH_TOKEN_JSON)
    else:
        path = config.ROOT / config.GOOGLE_OAUTH_TOKEN_FILE
        if path.exists():
            info = json.loads(path.read_text())
    _oauth_cache = (time.monotonic(), info)
    return info


def oauth_enabled() -> bool:
    """Signed in as the calendar owner (scripts/google_login.py) -> we can send real invites + Meet links.
    A service account cannot invite attendees on a personal Gmail calendar."""
    return _oauth_info() is not None


def _token() -> str:
    from google.auth.transport.requests import Request
    from google.oauth2 import service_account
    from google.oauth2.credentials import Credentials

    global _creds
    with _creds_lock:
        if _creds is None:
            info = _oauth_info()
            if info:
                _creds = Credentials.from_authorized_user_info(info, scopes=SCOPES)
            elif config.GOOGLE_SERVICE_ACCOUNT_JSON:
                info = json.loads(config.GOOGLE_SERVICE_ACCOUNT_JSON)
                _creds = service_account.Credentials.from_service_account_info(info, scopes=SCOPES)
            else:
                path = config.ROOT / config.GOOGLE_SERVICE_ACCOUNT_FILE
                _creds = service_account.Credentials.from_service_account_file(str(path), scopes=SCOPES)
        if not _creds.valid:
            _creds.refresh(Request())
        return _creds.token


def _headers() -> dict:
    return {"Authorization": f"Bearer {_token()}"}


def warm():
    """Fetch the access token and open the connection ahead of the first caller (saves ~1 s on the first turn)."""
    if not enabled():
        return
    try:
        now = dt.datetime.now(dt.timezone.utc)
        busy_intervals(now, now + dt.timedelta(hours=1))
        log.info("Google Calendar connection warmed up")
    except Exception as e:
        log.warning("Google Calendar warm-up failed: %r", e)


def busy_intervals(start: dt.datetime, end: dt.datetime) -> list[tuple[dt.datetime, dt.datetime]]:
    body = {"timeMin": start.isoformat(), "timeMax": end.isoformat(), "timeZone": config.TIMEZONE,
            "items": [{"id": config.GOOGLE_CALENDAR_ID}]}
    # read-only call, safe to retry on flaky connections
    for attempt in range(3):
        try:
            r = _http.post(f"{API}/freeBusy", json=body, headers=_headers())
            break
        except Exception:
            if attempt == 2:
                raise
    r.raise_for_status()
    cal = r.json()["calendars"][config.GOOGLE_CALENDAR_ID]
    if cal.get("errors"):
        raise RuntimeError(f"Google Calendar error: {cal['errors']} (is the calendar shared with the service account?)")
    return [(dt.datetime.fromisoformat(b["start"].replace("Z", "+00:00")),
             dt.datetime.fromisoformat(b["end"].replace("Z", "+00:00"))) for b in cal["busy"]]


def _event_id(start: dt.datetime, name: str, email: str) -> str:
    """Deterministic Google event id (base32hex, as the API requires): creating the same appointment twice
    returns HTTP 409 instead of a second event, which makes booking idempotent."""
    raw = f"{start.isoformat()}|{email.lower()}|{name.lower()}".encode()
    return base64.b32hexencode(hashlib.sha1(raw).digest()).decode().lower().rstrip("=")


def create_event(start: dt.datetime, end: dt.datetime, name: str, phone: str, purpose: str, email: str = "") -> dict:
    ev_id = _event_id(start, name, email)
    body = {
        "id": ev_id,
        "summary": f"Meeting with {name}" + (f" – {purpose}" if purpose else ""),
        "description": (f"Booked by AI voice agent.\nName: {name}\nEmail: {email or '-'}\n"
                        f"Phone: {phone or 'unknown'}\nPurpose: {purpose or '-'}"),
        "start": {"dateTime": start.isoformat(), "timeZone": config.TIMEZONE},
        "end": {"dateTime": end.isoformat(), "timeZone": config.TIMEZONE},
    }
    params = {}
    if oauth_enabled():
        # real calendar invite to the caller + a Google Meet link
        if email:
            body["attendees"] = [{"email": email, "displayName": name}]
            params["sendUpdates"] = "all"
        body["conferenceData"] = {"createRequest": {"requestId": ev_id,
                                                    "conferenceSolutionKey": {"type": "hangoutsMeet"}}}
        params["conferenceDataVersion"] = 1
    url = f"{API}/calendars/{config.GOOGLE_CALENDAR_ID}/events"
    r = _http.post(url, json=body, params=params, headers=_headers())
    duplicate = False
    if r.status_code == 409:                      # same appointment already exists -> reuse it, don't create another
        duplicate = True
        r = _http.get(f"{url}/{ev_id}", headers=_headers())
        r.raise_for_status()
        if r.json().get("status") == "cancelled":   # an earlier identical event was deleted -> restore it
            duplicate = False
            r = _http.put(f"{url}/{ev_id}", json={**body, "status": "confirmed"}, params=params, headers=_headers())
    r.raise_for_status()
    ev = r.json()
    return {"id": ev.get("id"), "link": ev.get("htmlLink"), "meet": ev.get("hangoutLink"),
            "invited": bool(params.get("sendUpdates")), "duplicate": duplicate}
