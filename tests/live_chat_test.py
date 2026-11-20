"""Live check of the LLM-answered turns (questions, out-of-scope, unclear dates) with the real model, through the real
streaming API. Google Calendar is switched off here. Run: .venv\\Scripts\\python -m tests.live_chat_test
Burns a few free-tier LLM requests - don't loop it."""
import json
import tempfile
import time
from pathlib import Path

from app import config

config.DB_PATH = Path(tempfile.mkdtemp()) / "live_chat.db"
config.GOOGLE_CALENDAR_ID = ""
from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

QUESTIONS = [
    "What services do you offer and how much does an e-commerce website cost?",
    "Do you offer refunds if I cancel?",
    "Can you build me a quantum computer?",
    "Is Sunday available for a meeting?",
    "Can I talk to a real person please?",
]


def main():
    c = TestClient(app)
    sid = c.post("/api/web/start", json={"silent": False}).json()["session_id"]
    for q in QUESTIONS:
        t0, first = time.perf_counter(), None
        with c.stream("POST", "/api/web/stream", json={"session_id": sid, "text": q, "via": "typed"}) as r:
            lines = []
            for line in r.iter_lines():
                if line:
                    ev = json.loads(line)
                    lines.append(ev)
                    if ev["t"] == "sentence" and first is None:
                        first = int((time.perf_counter() - t0) * 1000)
        d = lines[-1]
        tm = d.get("timings", {})
        print(f"\nYOU  : {q}\nAGENT: {d['reply']}\n       [{tm.get('model')} | first sentence {first} ms | total {tm.get('total_ms')} ms"
              f"{' | action=' + d['action'] if d.get('action') else ''}]")
        if d.get("action"):
            break


if __name__ == "__main__":
    main()
