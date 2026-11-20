"""Shared booking state: the single source of truth for voice, typed chat and the editable form.

Everything the caller says, types or edits ends up in one BookingState per session. The agent reads it on
every turn (so it never re-asks for what is known), the web form renders it, and booking is only ever
done from it after validation + explicit confirmation.
"""
from __future__ import annotations

import datetime as dt
import difflib
import hashlib
import re
import time as _time
from dataclasses import dataclass, field

from . import calendar_store, config

FIELDS = ("date", "time", "name", "email", "purpose")  # the order the agent asks in

COMMON_DOMAINS = {
    "gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "icloud.com", "live.com", "proton.me",
    "protonmail.com", "aol.com", "msn.com", "yahoo.co.uk", "me.com", "googlemail.com",
}
_TLD_TYPOS = {".con": ".com", ".cmo": ".com", ".vom": ".com", ".coom": ".com", ".cim": ".com", ".om": ".com"}
EMAIL_RE = re.compile(r"^[a-z0-9][a-z0-9._%+-]*@[a-z0-9-]+(?:\.[a-z0-9-]+)*\.[a-z]{2,}$")

_BAD_NAMES = {"name", "caller", "unknown", "customer", "user", "client", "n/a", "none", "john doe", "jane doe",
              "full name", "your name", "test user"}
_BAD_PURPOSE = {"meeting", "appointment", "n/a", "none", "unknown", "purpose", "free consultation"}


# ------------------------------------------------------------------ email
_DIGIT_WORDS = {"zero": "0", "oh": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
                "six": "6", "seven": "7", "eight": "8", "nine": "9"}
_TLDS = {"com", "net", "org", "edu", "gov", "io", "co", "ai", "app", "dev", "info", "biz", "tech", "xyz", "online",
         "store", "site", "pro", "cloud", "agency", "digital", "email", "live", "me", "us", "uk", "pk"}
_SEPARATORS = {"dot": ".", "period": ".", "underscore": "_", "dash": "-", "hyphen": "-", "minus": "-", "plus": "+"}
_STOP = {"is", "its", "it's", "my", "email", "e-mail", "mail", "address", "id", "the", "yes", "yeah", "okay", "ok",
         "sure", "and", "please", "so", "actually", "no", "now", "um", "uh", "i", "me", "name", "phone", "number",
         "send", "it", "that", "this", "here", "use", "make", "want", "have", "give", "can", "you", "to", "for",
         "at", "of", "on", "in", "an"}   # single letters ("a", "i") are NOT stop words: callers spell emails


def normalize_email(raw: str) -> str:
    """Accept both 'ahmed@gmail.com' and speech-to-text style 'Ahmed at gmail dot com'."""
    e = " " + (raw or "").lower().strip() + " "
    e = re.sub(r"\s+(at the rate( of)?|at)\s+", "@", e)
    e = re.sub(r"\s+(dot|period)\s+", ".", e)
    e = re.sub(r"\s+(underscore)\s+", "_", e)
    e = re.sub(r"\s+(dash|hyphen)\s+", "-", e)
    return re.sub(r"\s+", "", e).strip(".,")


def spoken_email(email: str) -> str:
    """'ahmed.k@gmail.com' -> 'ahmed dot k at gmail dot com' so text-to-speech reads it clearly."""
    return email.replace("@", " at ").replace(".", " dot ").replace("_", " underscore ").replace("-", " dash ")


def spelled_email(email: str) -> str:
    """Letter-by-letter reading of the part before the @ ('a h m e d')."""
    local = email.split("@")[0]
    return " ".join(local)


def extract_email(text: str) -> str | None:
    """Find an email address in what the caller said or typed. Handles real addresses ('a@b.com') and
    speech-to-text style ('ahmed khan 92 at gmail dot com', 'm dot danish at g mail dot com',
    'ali underscore raza at yahoo dot co dot uk', 'nine two')."""
    t = (text or "").lower().replace("at the rate of", " at ").replace("at the rate", " at ")
    lit = re.search(r"[a-z0-9][a-z0-9._%+-]*@[a-z0-9-]+(?:\.[a-z0-9-]+)+", t)
    if lit:
        return lit.group(0).strip(".")
    chunks = re.split(r"[,;?!]|\.\s", t)           # sentence-ish pieces; keep dots inside words
    for chunk in chunks:
        toks = re.findall(r"[a-z0-9']+|[@._+-]", chunk)
        for i in range(len(toks) - 1, -1, -1):
            if toks[i] not in ("at", "@"):
                continue
            dom = _read_domain(toks, i + 1)
            if not dom:
                continue
            local = _read_local(toks, i)
            if local:
                cand = f"{local}@{dom}"
                if EMAIL_RE.match(cand):
                    return cand
    return None


def _read_domain(toks: list[str], j: int) -> str | None:
    labels, cur, seen_dot = [], "", False
    n = len(toks)
    while j < n:
        tk = toks[j]
        if tk in _SEPARATORS and _SEPARATORS[tk] == "." or tk == ".":
            if not cur:
                return None
            labels.append(cur)
            cur, seen_dot = "", True
            j += 1
            # exactly one token for each label after the first dot
            if j < n and re.fullmatch(r"[a-z0-9]+", toks[j]) and toks[j] not in _SEPARATORS:
                cur = toks[j]
                j += 1
                if j < n and (toks[j] == "." or toks[j] in ("dot", "period")):
                    continue
                break
            return None
        if tk in _SEPARATORS and _SEPARATORS[tk] == "-":
            cur += "-"
            j += 1
            continue
        if re.fullmatch(r"[a-z0-9]+", tk) and not seen_dot:
            cur += tk                                   # "g mail" -> "gmail"
            j += 1
            if j - 1 > 0 and len(cur) > 25:
                return None
            continue
        break
    if cur:
        labels.append(cur)
    if not seen_dot or len(labels) < 2 or not labels[-1].isalpha():
        return None
    if labels[-1] not in _TLDS and len(labels[-1]) != 2:      # "10 dot please" is not a domain
        return None
    return ".".join(labels)


def _read_local(toks: list[str], at_idx: int) -> str:
    parts: list[str] = []
    k, count = at_idx - 1, 0
    while k >= 0 and count < 14:
        tk = toks[k]
        if tk in _STOP:
            break
        if tk in _SEPARATORS or tk in "._+-":
            parts.append(_SEPARATORS.get(tk, tk))
        elif tk in ("double", "triple"):
            if parts:
                prev = parts.pop()
                parts.append(prev * (2 if tk == "double" else 3))
        else:
            parts.append(_DIGIT_WORDS.get(tk, tk))
        k -= 1
        count += 1
    local = "".join(reversed(parts)).strip("._+-")
    return local if re.fullmatch(r"[a-z0-9][a-z0-9._%+-]*", local or "") else ""


def check_email(email: str, source: str = "voice") -> dict:
    """Validate an email and rate how sure we are that it was captured correctly.
    source: 'voice' (speech-to-text, error-prone) | 'typed' | 'manual' (the person typed it themselves)."""
    e = (email or "").strip().lower()
    if not e:
        return {"valid": False, "empty": True}
    if not EMAIL_RE.match(e) or ".." in e or e.split("@")[0].endswith("."):
        return {"valid": False, "error": "That doesn't look like a valid email address (example: name@gmail.com)."}
    local, domain = e.split("@")
    suggestion = None
    for bad, good in _TLD_TYPOS.items():
        if domain.endswith(bad):
            suggestion = f"{local}@{domain[:-len(bad)]}{good}"
    if domain not in COMMON_DOMAINS and not suggestion:
        close = difflib.get_close_matches(domain, COMMON_DOMAINS, n=1, cutoff=0.82)
        if close:
            suggestion = f"{local}@{close[0]}"
    if source == "voice":
        low = bool(suggestion) or len(local) < 3 or domain not in COMMON_DOMAINS
    else:
        low = False        # the person typed it; they can see it
    return {"valid": True, "email": e, "confidence": "low" if low else "high", "suggestion": suggestion}


# ------------------------------------------------------------------ other validators
def check_name(name: str) -> tuple[str, str | None]:
    n = re.sub(r"\s+", " ", (name or "").strip())
    if n.lower() in _BAD_NAMES or any(ch in n for ch in "[]{}<>@/\\0123456789"):
        return "", "Please enter your real full name."
    parts = [p for p in re.split(r"\s+", n) if p]
    if len(parts) < 2 or sum(ch.isalpha() for ch in n) < 3 or not all(any(c.isalpha() for c in p) for p in parts):
        return n, "Please enter your full name (first and last name)."
    if len(n) > 80:
        return n[:80], "That name is too long."
    return (n.title() if n.islower() or n.isupper() else n), None


def check_purpose(purpose: str) -> tuple[str, str | None]:
    p = re.sub(r"\s+", " ", (purpose or "").strip())
    if p.lower() in _BAD_PURPOSE or len(p) < 3:
        return p, "Tell us briefly what the meeting is about."
    return p[:200], None


def check_slot(date: str, time: str, fresh: bool = True) -> dict:
    """Server-side validation of a date + time, independent of anything the UI restricts.
    Returns {ok, code, error, free(list of HH:MM on that date)}."""
    try:
        d = calendar_store._parse_date(date)
    except (ValueError, TypeError):
        return {"ok": False, "code": "bad_date", "error": "Please choose a valid date."}
    av = calendar_store.available_slots(d.isoformat()) if fresh else cached_slots(d.isoformat())
    if "error" in av:
        code = "calendar_down" if "Calendar unavailable" in av["error"] else "date_not_allowed"
        msg = av["error"].split(" Ask the caller")[0]
        if code == "date_not_allowed" and "in the past" in msg:
            msg = f"{d.strftime('%A, %B')} {d.day} has already passed. Please choose a future date."
        elif code == "date_not_allowed":
            msg = f"{d.strftime('%A, %B')} {d.day} is too far ahead - bookings open up to 60 days."
        return {"ok": False, "code": code, "error": msg, "free": []}
    free = av["available"]
    if not free:
        return {"ok": False, "code": "closed", "free": [],
                "error": f"We have no openings on {d.strftime('%A, %B')} {d.day} (closed or fully booked)."}
    if not time:
        return {"ok": False, "code": "no_time", "free": free}
    try:
        t = calendar_store._parse_time(time).strftime("%H:%M")
    except (ValueError, TypeError, AttributeError):
        return {"ok": False, "code": "bad_time", "error": "Please choose a valid time.", "free": free}
    if t not in free:
        return {"ok": False, "code": "busy", "free": free,
                "error": f"{_spoken_t(t)} is not available on {d.strftime('%A, %B')} {d.day}."}
    return {"ok": True, "code": "free", "free": free}


def _spoken_t(hhmm: str) -> str:
    t = dt.time.fromisoformat(hhmm)
    h = t.hour % 12 or 12
    suffix = "AM" if t.hour < 12 else "PM"
    return f"{h} {suffix}" if t.minute == 0 else f"{h}:{t.minute:02d} {suffix}"


def ordinal(n: int) -> str:
    return f"{n}{'th' if 11 <= n % 100 <= 13 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def pretty_when(date: str, time: str) -> str:
    d = dt.date.fromisoformat(date)
    return f"{d.strftime('%A, %B')} {ordinal(d.day)} at {_spoken_t(time)}"


# ------------------------------------------------------------------ state
@dataclass
class BookingState:
    values: dict = field(default_factory=lambda: {f: "" for f in FIELDS})
    source: dict = field(default_factory=dict)      # field -> voice | typed | manual
    errors: dict = field(default_factory=dict)      # field -> message shown in the form
    email_status: str = "empty"                     # empty | unconfirmed | confirmed | invalid
    email_conf: str = "high"
    email_suggestion: str | None = None
    booked: dict | None = None
    version: int = 0
    heard: dict = field(default_factory=dict)       # field -> what the caller said but could not be accepted yet (shown dimmed)

    # -- keys
    def key(self) -> str:
        v = self.values
        raw = "|".join([v["date"], v["time"], v["name"].lower(), v["email"], v["purpose"].lower()])
        return hashlib.sha1(raw.encode()).hexdigest()[:16]

    # -- writing
    def set(self, name: str, value: str, source: str = "voice") -> dict:
        """Validate and store one field. Returns {ok, error, changed}."""
        value = (value or "").strip()
        old = self.values.get(name, "")
        if self.booked and value and self.booked.get("key") == self.key():
            self.new_booking(keep_contact=True)          # a new request after a finished booking
            old = self.values.get(name, "")
        err = None
        stored = ""
        if name == "date":
            if value:
                try:
                    d = calendar_store._parse_date(value)
                    stored = d.isoformat()
                    if d < calendar_store.now().date():
                        err = f"{d.strftime('%A, %B')} {d.day} has already passed. Please choose a future date."
                        stored = ""
                    elif d > calendar_store.now().date() + dt.timedelta(days=60):
                        err, stored = "Bookings open up to 60 days ahead.", ""
                    elif d.weekday() not in config.WORK_DAYS:
                        err = f"We're closed on {d.strftime('%A')}s. Please pick another day."
                        stored = ""
                except (ValueError, TypeError):
                    err = "Please choose a valid date."
        elif name == "time":
            if value:
                try:
                    stored = calendar_store._parse_time(value).strftime("%H:%M")
                except (ValueError, TypeError):
                    err = "Please choose a valid time."
        elif name == "name":
            if value:
                stored, err = check_name(value)
                if err:
                    stored = ""
        elif name == "purpose":
            if value:
                stored, err = check_purpose(value)
                if err:
                    stored = ""
        elif name == "email":
            return self._set_email(value, source)
        else:
            return {"ok": False, "error": f"unknown field {name}", "changed": False}

        self.errors.pop(name, None)
        if err:
            self.errors[name] = err
            if value and not stored:
                self.heard[name] = value
        if stored or not value:
            self.heard.pop(name, None)
        self.values[name] = stored
        if stored or not value:
            self.source[name] = source
        changed = stored != old
        if changed:
            self.version += 1
        return {"ok": not err, "error": err, "changed": changed}

    def _set_email(self, value: str, source: str) -> dict:
        old = self.values["email"]
        self.errors.pop("email", None)
        if not value:
            self.values["email"], self.email_status, self.email_suggestion = "", "empty", None
            self.heard.pop("email", None)
            self.source["email"] = source
            changed = bool(old)
            self.version += changed
            return {"ok": True, "error": None, "changed": changed}
        res = check_email(normalize_email(value) if " " in value or " at " in value.lower() else value, source)
        if not res["valid"]:
            self.values["email"], self.email_status, self.email_suggestion = "", "invalid", None
            self.errors["email"] = res.get("error", "Invalid email.")
            self.heard["email"] = value
            self.source["email"] = source
            return {"ok": False, "error": self.errors["email"], "changed": bool(old)}
        email = res["email"]
        self.heard.pop("email", None)
        same = email == old
        self.values["email"] = email
        self.source["email"] = source
        self.email_conf = res["confidence"]
        self.email_suggestion = res["suggestion"]
        if source in ("typed", "manual"):
            self.email_status = "confirmed"
        elif not (same and self.email_status == "confirmed"):
            self.email_status = "unconfirmed"
        if not same:
            self.version += 1
        return {"ok": True, "error": None, "changed": not same, "confidence": res["confidence"],
                "suggestion": res["suggestion"]}

    def new_booking(self, keep_contact: bool = True):
        keep = {k: self.values[k] for k in ("name", "email")} if keep_contact else {}
        status = self.email_status if keep_contact else "empty"
        self.values = {f: "" for f in FIELDS}
        self.values.update(keep)
        self.errors, self.booked, self.heard = {}, None, {}
        self.email_status = status if keep.get("email") else "empty"
        self.version += 1

    # -- reading
    def missing(self) -> list[str]:
        out = [f for f in FIELDS if not self.values[f] or f in self.errors]
        if self.values["email"] and self.email_status == "invalid" and "email" not in out:
            out.append("email")
        return out

    def summary(self) -> str:
        v = self.values
        return (f"{pretty_when(v['date'], v['time'])} for {v['name']}, about {v['purpose'].rstrip('.')}. "
                f"Your email is {spoken_email(v['email'])}.")

    def public(self, slot: dict | None = None) -> dict:
        missing = self.missing()
        steps = {
            "details": not missing,
            "availability": bool(slot and slot.get("ok")),
            "review": bool(not missing and slot and slot.get("ok")),
            "booked": bool(self.booked),
        }
        if self.booked:                      # once booked, every step is done (the slot itself is now taken by us)
            steps = {k: True for k in steps}
        return {
            "values": dict(self.values), "heard": dict(self.heard), "source": dict(self.source), "errors": dict(self.errors),
            "email_status": self.email_status, "email_confidence": self.email_conf,
            "email_suggestion": self.email_suggestion, "missing": missing,
            "slot": slot, "steps": steps, "key": self.key(), "version": self.version,
            "booked": self.booked, "ready": steps["review"] and not self.booked,
        }


# ------------------------------------------------------------------ prompts that need no LLM
_ASK = {
    "date": "Which day would you like?",
    "time": "What time works best for you?",
    "name": "May I have your full name, please?",
    "email": "What's the best email address for the invite?",
    "purpose": "What would you like to discuss in the meeting?",
}


def next_question(state: BookingState) -> str:
    miss = state.missing()
    return _ASK[miss[0]] if miss else ""


# ------------------------------------------------------------------ tiny TTL cache for availability lookups
_SLOT_CACHE: dict[str, tuple[float, dict]] = {}


def cached_slots(date: str, ttl: float = 8.0) -> dict:
    hit = _SLOT_CACHE.get(date)
    if hit and _time.monotonic() - hit[0] < ttl:
        return hit[1]
    res = calendar_store.available_slots(date)
    _SLOT_CACHE[date] = (_time.monotonic(), res)
    return res


def clear_slot_cache():
    _SLOT_CACHE.clear()
