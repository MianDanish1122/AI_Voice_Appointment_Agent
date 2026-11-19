"""Real conversation test against the configured LLM (uses a temp DB, Google Calendar off).
Run: .venv\\Scripts\\python -m tests.live_chat_test"""
import asyncio
import datetime as dt
import tempfile
import time
from pathlib import Path

from app import calendar_store, config

config.DB_PATH = Path(tempfile.mkdtemp()) / "live.db"
config.GOOGLE_CALENDAR_ID = ""

from app.agent import CallSession  # noqa: E402


def next_workday_name() -> str:
    d = calendar_store.now().date() + dt.timedelta(days=1)
    while d.weekday() not in config.WORK_DAYS:
        d += dt.timedelta(days=1)
    return d.strftime("%A")


async def main():
    day = next_workday_name()
    s = CallSession("live-test", "+923331112222")
    print("AGENT:", s.greeting())
    turns = [
        "Hi, I'd like to book a meeting for tomorrow at 3 PM.",
        "No no wait, make it Monday at 11 in the morning instead.",
        "My name is Ahmed Khan.",
        "It's ahmed khan 92 at gmail dot com",
        "I want to discuss building an e-commerce website.",
        "Yes, that's correct.",
        "No, that's all, thank you. Bye!",
    ]
    for t in turns:
        print("\nCALLER:", t)
        t0 = time.time()
        n = len(s.messages)
        reply = await s.respond(t)
        for m in s.messages[n:]:
            for tc in m.get("tool_calls") or []:
                print("   TOOL:", tc["function"]["name"], tc["function"]["arguments"])
        print(f"AGENT ({time.time() - t0:.1f}s):", reply, f"[action={s.action}]" if s.action else "")
        if s.action:
            break
    print("\nBOOKINGS:", calendar_store.list_bookings())
    tools = [m["tool_calls"][0]["function"]["name"] for m in s.messages if m.get("tool_calls")]
    print("TOOLS CALLED:", tools)


asyncio.run(main())
