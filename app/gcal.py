"""Google Calendar via a (free) service account.

Setup: create a service account in Google Cloud, enable "Google Calendar API", download its JSON key,
then in Google Calendar -> your calendar -> Settings -> "Share with specific people" add the service
account email with "Make changes to events".
"""
import datetime as dt
import json

import httpx

from . import config

SCOPES = ["https://www.googleapis.com/auth/calendar"]
API = "https://www.googleapis.com/calendar/v3"
_creds = None


def enabled() -> bool:
    return bool(config.GOOGLE_CALENDAR_ID and (config.GOOGLE_SERVICE_ACCOUNT_JSON or config.GOOGLE_SERVICE_ACCOUNT_FILE
                                               or oauth_enabled()))


def _oauth_info() -> dict | None:
    if config.GOOGLE_OAUTH_TOKEN_JSON:
        return json.loads(config.GOOGLE_OAUTH_TOKEN_JSON)
    path = config.ROOT / config.GOOGLE_OAUTH_TOKEN_FILE
    if path.exists():
        return json.loads(path.read_text())
    return None


def oauth_enabled() -> bool:
    """Signed in as the calendar owner (scripts/google_login.py) -> we can send real invites + Meet links.
    A service account cannot invite attendees on a personal Gmail calendar."""
    return _oauth_info() is not None


def _token() -> str:
    global _creds
    from google.auth.transport.requests import Request
    from google.oauth2 import service_account
    from google.oauth2.credentials import Credentials

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


def busy_intervals(start: dt.datetime, end: dt.datetime) -> list[tuple[dt.datetime, dt.datetime]]:
    body = {"timeMin": start.isoformat(), "timeMax": end.isoformat(), "timeZone": config.TIMEZONE,
            "items": [{"id": config.GOOGLE_CALENDAR_ID}]}
    # read-only call, safe to retry on flaky connections
    for attempt in range(3):
        try:
            r = httpx.post(f"{API}/freeBusy", json=body, headers=_headers(), timeout=15)
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


def create_event(start: dt.datetime, end: dt.datetime, name: str, phone: str, purpose: str, email: str = "") -> dict:
    body = {
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
        body["conferenceData"] = {"createRequest": {"requestId": f"ai-{int(start.timestamp())}-{abs(hash(email)) % 10**6}",
                                                    "conferenceSolutionKey": {"type": "hangoutsMeet"}}}
        params["conferenceDataVersion"] = 1
    r = httpx.post(f"{API}/calendars/{config.GOOGLE_CALENDAR_ID}/events", json=body, params=params,
                   headers=_headers(), timeout=15)
    r.raise_for_status()
    ev = r.json()
    return {"id": ev.get("id"), "link": ev.get("htmlLink"), "meet": ev.get("hangoutLink"), "invited": bool(params.get("sendUpdates"))}
