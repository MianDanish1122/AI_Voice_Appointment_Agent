# CLAUDE.md — handoff for any Claude session working on this repo

The owner speaks Roman Urdu / Hindi mixed with English; reply in the same style, short and practical.
**Never commit or print secrets** (`.env`, `service_account.json`, `client_secret.json`, `google_token.json`). The repo is PUBLIC.

## What this is
A **$0 (free-tier only) AI voice appointment agent**, built for a 48-hour job assessment:
- Caller talks to a receptionist ("Danish Solution" at "Danish Digital Solutions") by **web call** (Chrome mic) or **phone** (Twilio trial webhook).
- Agent answers from `knowledge/` (company FAQ), checks free slots, collects **full name + email + purpose**, reads details back, and on "yes" books a **Google Calendar** event. It can transfer to a human (`HUMAN_PHONE`) or end the call.
- Assessment deliverables: **live demo link / number**, **2–3 min screen recording** (call + event appearing in Google Calendar + architecture walkthrough), **3–4 bullet architecture summary** (README has it). Evaluators will test with tricky phrases ("tomorrow 5 PM… no, day after at 11", Sunday, 10 PM, spelled emails).

## Run it
```powershell
python -m venv .venv ; .\.venv\Scripts\pip install -r requirements.txt
copy .env.example .env      # then fill keys (see below)
.\.venv\Scripts\python -m uvicorn app.main:app --port 8000     # open http://localhost:8000 in Chrome/Edge
```
Tests (run from repo root):
- `python -m tests.smoke_test` — offline, fake LLM, **never touches real Google** (it blanks `GOOGLE_CALENDAR_ID`). Must pass before every commit.
- `python -m tests.live_chat_test` — real LLM conversation (burns free-tier tokens; Google off).
- `python -m tests.gcal_live_test` — real Google Calendar: creates a test event then deletes it.
Do NOT run several live tests back-to-back: Groq/Gemini free limits are hit quickly and fallbacks answer worse.

## Config (`.env`, see `.env.example`)
`LLM_BASE_URL/LLM_API_KEY/LLM_MODEL` (Groq, default `openai/gpt-oss-120b`; old llama-3.3 is gone), `GEMINI_API_KEY` (extra free provider),
`GOOGLE_CALENDAR_ID` + `service_account.json` (service account shared on the calendar with "Make changes to events"),
`BUSINESS_NAME`, `AGENT_NAME`, `TIMEZONE=Asia/Karachi`, `WORK_DAYS`, `OPEN_TIME/CLOSE_TIME/SLOT_MINUTES`, `HUMAN_PHONE`,
`TTS_VOICE` (Twilio), `WEB_TTS_VOICE` (edge-tts), optional `GOOGLE_OAUTH_TOKEN_*` (invites).
Keys were pasted into a chat earlier → **rotate Groq + Gemini keys after submission**.

## Code map
- `app/main.py` — FastAPI: Twilio webhooks (`/twilio/voice|gather|noinput|status`), browser API (`/api/web/start|message`), `/api/tts` (edge-tts, cached, greeting pre-warmed), `/api/bookings`, `/api/slots/{date}`, `/api/kb/reload`, serves `static/index.html`.
- `app/agent.py` — the brain: system prompt, tools (`check_available_slots`, `book_appointment`, `transfer_to_human`, `end_call`), per-call `CallSession`, and **guard rails in code** (see below).
- `app/calendar_store.py` — slots + bookings (SQLite `data/bookings.db`), merges Google `freeBusy`; refuses past / closed / >60-day dates; fixes stale model years.
- `app/gcal.py` — Google Calendar REST (service account, or OAuth token if present → real invites + Meet link).
- `app/llm.py` — OpenAI-compatible client; provider chain **Groq main → Gemini → Groq small models**; 429 puts a model on cooldown; an empty answer counts as failure.
- `app/kb.py` — loads `knowledge/*.md|txt`; whole KB is sent if small, else BM25.
- `static/index.html` — two-column UI (orb + call controls + voice choice left; transcript + bookings right). Browser STT (continuous, 1.3 s silence), echo filter, barge-in, neural voice with speed probe → falls back to browser voice.
- `scripts/google_login.py` — one-time OAuth sign-in (needs `client_secret.json`) so the agent can email real calendar invites.

## Guard rails implemented in code (don't weaken; the fallback models need them)
- Booking needs **name, email, purpose that the caller actually said** (checked against caller text); placeholders ([Name], John Doe, example.com) rejected.
- **Two-step confirm**: first `book_appointment` call only makes the system read back date/time/name/purpose/email; booking happens only after a clear "yes" next turn (`_confirmed`).
- `end_call` only if caller said goodbye; `transfer_to_human` turns into hang-up on a plain goodbye.
- Spoken emails ("ahmed at gmail dot com") normalised; times spoken as "3 PM"; replies cleaned for TTS (`_clean`).
- Date handling in code: `_extract_date` ("the 20th", "20 November", "Dec 3rd") → `_slots_note` looks up slots **before** the LLM and injects them; `date_fallback` is a ready-made spoken answer used if every model fails/returns empty.

## Status (end of last session)
Working & tested: web call flow, booking → real Google Calendar event (tested live), email/purpose rules, past/closed-day handling, LLM fallback chain, two-column UI, GitHub repo.
**Not yet verified / open items**
1. **No public live link yet** (most important). Plan: deploy this repo (Dockerfile present) to Hugging Face Spaces / Render / Vercel with env secrets; `GOOGLE_SERVICE_ACCOUNT_JSON` holds the key text. Then point Twilio webhook to `<url>/twilio/voice`. Twilio trial can only call verified numbers → web link is the safe deliverable. A competitor's demo uses **Vapi** (managed voice platform, free trial credits) — allowed by the assessment; consider it only if this stack's voice quality is not good enough.
2. Last change set (date prefetch + empty-answer fallback in `app/agent.py`, `app/llm.py`) passed `smoke_test` but the **real-LLM re-test of "meeting on the 20th" was interrupted** — re-run `tests.live_chat_test`/a few date phrasings once rate limits reset.
3. Browser behaviour (echo filter, barge-in, voice quality) is unverified by automated tests — needs a human with headphones in Chrome/Edge. Neural edge-tts takes 5–13 s on the owner's network, so the page auto-selects the browser voice (Edge's "Natural/Online" voices are best).
4. Calendar **invites to the caller's email** need the OAuth step (`client_secret.json` → `python -m scripts.google_login`); until then the email is only written in the event description.
5. Groq/Gemini free limits: a burst of calls can exhaust them (Groq main model was rate-limited for minutes after testing). `gemini-2.5-flash-lite` sometimes returns empty after a tool call (handled by falling through). Consider shrinking the prompt/tool tokens if this bites.
6. Screen recording + final hosting are the owner's tasks; keep README's architecture summary accurate.

## Conventions
- Keep replies of the agent to 1–3 short sentences, English only, ending with a question.
- Commit messages end with the Co-Authored-By line your harness specifies; push only when the owner asks. Remote: `https://github.com/MianDanish1122/AI_Voice_Appointment_Agent` (owner's GitHub account is `MianDanish1122`; the machine's default git login is a different account, so pushes use `https://MianDanish1122@github.com/...`).
