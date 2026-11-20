"""Booking confirmation email to the caller (free: any SMTP account, e.g. Gmail with an App Password).

Configure SMTP_HOST / SMTP_PORT / SMTP_USER / SMTP_PASSWORD (+ optional MAIL_FROM). Without them nothing is sent and the
agent never claims an email was sent. The mail carries the details, an "add to Google Calendar" link and an .ics file."""
import datetime as dt
import logging
import smtplib
import ssl
import urllib.parse
import uuid
from email.message import EmailMessage

from . import config

log = logging.getLogger("mailer")


def enabled() -> bool:
    return bool(config.SMTP_HOST and config.SMTP_USER and config.SMTP_PASSWORD)


def _utc(d: dt.datetime) -> str:
    return d.astimezone(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def calendar_link(title: str, start: dt.datetime, end: dt.datetime, details: str) -> str:
    q = urllib.parse.urlencode({"action": "TEMPLATE", "text": title, "dates": f"{_utc(start)}/{_utc(end)}", "details": details})
    return "https://calendar.google.com/calendar/render?" + q


def _ics(title: str, start: dt.datetime, end: dt.datetime, details: str, organizer: str) -> str:
    esc = lambda s: s.replace("\\", "\\\\").replace(";", "\\;").replace(",", "\\,").replace("\n", "\\n")
    return "\r\n".join([
        "BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//Voice Appointment Agent//EN", "METHOD:PUBLISH", "BEGIN:VEVENT",
        f"UID:{uuid.uuid4()}@voice-agent", f"DTSTAMP:{_utc(dt.datetime.now(dt.timezone.utc))}",
        f"DTSTART:{_utc(start)}", f"DTEND:{_utc(end)}", f"SUMMARY:{esc(title)}", f"DESCRIPTION:{esc(details)}",
        f"ORGANIZER;CN={esc(config.BUSINESS_NAME)}:mailto:{organizer}", "END:VEVENT", "END:VCALENDAR", ""])


def build(to: str, name: str, when_text: str, purpose: str, start: dt.datetime, end: dt.datetime,
          meet_link: str | None = None) -> EmailMessage:
    sender = config.MAIL_FROM or config.SMTP_USER
    title = f"Meeting with {config.BUSINESS_NAME}"
    details = f"{purpose}\nBooked by voice with {config.AGENT_NAME}."
    add = calendar_link(title, start, end, details)
    tz = config.TIMEZONE
    lines = [f"Hi {name.split()[0]},", "", f"Your meeting with {config.BUSINESS_NAME} is booked.", "",
             f"When:   {when_text} ({tz})", f"Topic:  {purpose}", f"Name:   {name}", f"Email:  {to}"]
    if meet_link:
        lines.append(f"Join:   {meet_link}")
    lines += ["", f"Add it to your Google Calendar: {add}", "(the attached .ics file works with any calendar app)", "",
              f"Need to change it? Just call again and tell {config.AGENT_NAME}.", "", f"{config.BUSINESS_NAME}"]
    msg = EmailMessage()
    msg["Subject"] = f"Booking confirmed: {when_text}"
    msg["From"] = f"{config.BUSINESS_NAME} <{sender}>"
    msg["To"] = to
    msg.set_content("\n".join(lines))
    msg.add_attachment(_ics(title, start, end, details, sender).encode("utf-8"), maintype="text", subtype="calendar",
                       filename="meeting.ics")
    return msg


def send(msg: EmailMessage) -> bool:
    """Send via SMTP; True only if the server accepted it. Never raises (a mail problem must not undo a booking)."""
    if not enabled():
        return False
    try:
        if config.SMTP_PORT == 465:
            with smtplib.SMTP_SSL(config.SMTP_HOST, 465, timeout=12, context=ssl.create_default_context()) as s:
                s.login(config.SMTP_USER, config.SMTP_PASSWORD)
                s.send_message(msg)
        else:
            with smtplib.SMTP(config.SMTP_HOST, config.SMTP_PORT, timeout=12) as s:
                s.starttls(context=ssl.create_default_context())
                s.login(config.SMTP_USER, config.SMTP_PASSWORD)
                s.send_message(msg)
        log.info("confirmation email sent")
        return True
    except Exception:
        log.exception("confirmation email failed")
        return False
