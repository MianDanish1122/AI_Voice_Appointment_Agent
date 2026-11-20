# CLAUDE.md — handoff for any Claude session working on this repo

The owner speaks Roman Urdu / Hindi mixed with English; reply in the same style, short and practical.
**Never commit or print secrets** (`.env`, `service_account.json`, `client_secret.json`, `google_token.json`). The repo is PUBLIC.
Push to GitHub only when the owner asks. Remote: `https://github.com/MianDanish1122/AI_Voice_Appointment_Agent` — the machine's
default git login is a different account, so push with `https://MianDanish1122@github.com/...`.

## What this is
A **$0 (free-tier only) AI voice appointment agent** for a 48-hour job assessment. Caller talks (web call in Chrome/Edge, or Twilio phone)
or types, or fills a form — all three edit ONE shared booking state. The agent collects date, time, full name, email, purpose,
reads them back, and on an explicit "yes" (or the Confirm button) creates a real **Google Calendar** event. Deliverables the owner still
has to produce: a **live public link**, a **2–3 min screen recording**, a 3–4 bullet architecture summary (README has it).
Evaluators will try tricky phrases ("tomorrow 5 PM… no, day after at 11", Sunday, 10 PM, spelled emails).

## Run / test
```powershell
python -m venv .venv ; .\.venv\Scripts\pip install -r requirements.txt ; copy .env.example .env   # fill keys
.\.venv\Scripts\python -m uvicorn app.main:app --port 8000      # http://localhost:8000 in Chrome/Edge
.\.venv\Scripts\python -m tests.smoke_test      # 30 offline tests, fake LLM + fake Google + design lint. MUST pass before every commit
.\.venv\Scripts\python -m tests.ui_test         # 13 real-Chrome tests of the VOICE-ONLY page (incl. fit-to-screen on 8 window sizes and a full call on 1366x650) (UI_ONLY=<name part> runs a subset; needs: pip install -r requirements-dev.txt; uses installed Chrome)
.\.venv\Scripts\python -m tests.live_e2e_test   # real LLM + REAL Google Calendar; creates a test event, verifies, deletes it
.\.venv\Scripts\python -m tests.live_chat_test  # real LLM answers
```
Live tests burn free-tier tokens and touch the real calendar: do not loop them. If a live test dies mid-way, look for leftover
events named "Upgrade Check" / "Profile Probe" in the owner's calendar and delete them. Offline tests blank `GOOGLE_CALENDAR_ID`.

## Architecture (see README for the diagram)
- `app/booking.py` — shared `BookingState` (per session), email extraction/validation/confidence (`extract_email`, `check_email`), name/purpose/slot validators (`check_slot`), spoken helpers.
- `app/agent.py` — `CallSession.respond()`: (1) `_deterministic()` handles in code, with no LLM: yes/no to read-backs, human handoff, dates/times/emails/names/purpose from the caller's words (incl. answers to the question just asked), real availability, acknowledge-and-ask-next; (2) otherwise the LLM (tools: `check_available_slots`, `update_booking_details`, `book_appointment`, `transfer_to_human`, `end_call`) with the BOOKING STATE + KB + "SLOTS CHECKED" injected; streamed sentence by sentence via `_SentenceStream`. `apply_manual()` = form edits. `_finalize_sync()` is the ONLY place that books.
- `app/main.py` — web API `/api/web/{start,stream,message,state,review,confirm,reset,end}`, `/api/config`, `/api/tts`, `/api/bookings`, phone `/twilio/*`. Use `spawn()` for fire-and-forget tasks (strong refs!).
- `app/llm.py` — provider chain Groq `openai/gpt-oss-120b` → Gemini 2.5 Flash → smaller Groq models; 429 cooldown; empty answer = failure; SSE streaming; shared HTTP client + warm-up.
- `app/gcal.py` — Google REST over a shared `httpx.Client`; deterministic event id ⇒ idempotent create (409 ⇒ reuse; cancelled ⇒ restore); service account, or OAuth token ⇒ real invites + Meet.
- `app/calendar_store.py` — slots + bookings in SQLite (`UNIQUE(start)`), merges Google freeBusy, idempotent for the same name+email.
- `app/vapi.py` + `scripts/vapi_setup.py` — OPTIONAL Vapi engine (Vapi owns STT/LLM/TTS/phone and calls our webhook `POST /api/vapi/webhook` for tools `update_booking_details` / `check_available_slots` / `book_appointment`). The caller's words arrive in `artifact.messages` / `conversation-update` and go through the same `CallSession._deterministic()` rules; tool results are flat JSON strings with a `say` field the Vapi LLM must read verbatim; the page passes its session id as `variableValues.sid` (+ `/api/vapi/link`). Keys in `.env`: `VAPI_API_KEY` (private, setup script only), `VAPI_PUBLIC_KEY`, `VAPI_ASSISTANT_ID`, `VAPI_WEBHOOK_SECRET`. NOT yet run against a live Vapi call (needs the public URL) and the assistant/tools are NOT created in the owner's Vapi account yet (`python -m scripts.vapi_setup --server-url https://<host>` after deploying). Unverified against Vapi's live API: transcriber/voice ids, tool PATCH shape, `{{"now" | date: ...}}` liquid in the prompt, web SDK `start(id, overrides)` signature and returned call id.
- `static/index.html` + `static/tokens.css` — the whole UI, **voice-only by the owner's request** (no text box, no form, no typing surface; `test_voice_only_there_is_no_typing_surface` enforces it). Built with the Hallmark skill: theme **Lumen** (Night default / Day via `data-drop`), Workbench layout, N9 nav, Ft5 footer, hand-built SVG voice dial (never an orb), read-only live booking sheet, transcript, recent bookings. All colours/fonts come from `tokens.css` and `test_page_design_lint_hallmark_gates` mechanically enforces the Hallmark gates (tokens-only, no italics/blur/100vw, tinted neutrals, spacing scale, nowrap affordances, eyebrow cap, no invented metrics); `.hallmark/log.json` records the pick. The manual/typed backend APIs (`/api/web/state|review|confirm|message`) still exist and are covered by the offline tests, but the page does not use them. `window.__app` is exposed for tests.

## Rules that matter (don't weaken them — weak/rate-limited models rely on them)
- Booking needs all five fields valid, a free slot (fresh check), and a clear "yes" to a read-back **in a later turn** for exactly the same state key (any edit resets it). The UI confirm path checks the same key. Success is reported only after the Google call returned.
- The LLM may not invent name/email/purpose: values must appear in what the caller said/typed (`_mentioned`). Date/time come from code when the caller named them (last mention wins). Manual (form) edits are not overwritten by stale voice guesses.
- Uncertain voice emails (unusual domain, typo suggestion, very short) are verified out loud first; typed/form emails count as verified.
- Date errors (closed day, past, >60 days) are spoken by code, not left to the model.
- Free-tier models often skip tool calls (measured), so everything critical also exists as code.

## Design rules for the page (Hallmark/Lumen)
- Prose is lowercase, mono labels are the only UPPERCASE; headings are roman (never italic); one accent-coloured verb in the headline; no orb, no glass/blur, no gradient text; every number on screen is measured or live; the whole call must fit the screen on desktop-sized windows (app shell: page does not scroll, transcript scrolls inside itself; every stage block has an explicit grid row; short <=736px screens get a compact variant) while small windows/phones scroll normally; the 'start call' button must stay above the fold at 1280x800; mobile floor 320/375/414/768 px (no horizontal scroll, clickable text never wraps). Re-run `tests.ui_test` and read the screenshots in `tests/screens/` after any visual change.

## Learned the hard way
- **Do not write Python patch scripts through a bash heredoc if they contain backslashes** (`\b`, `\s`, `\n`): the harness mangles them (we got real backspace characters in a regex). Write the script/file with the Write/Edit tools. After scripted edits run a control-character scan.
- `asyncio.create_task()` without a stored reference can be garbage-collected mid-flight (a stream never finished) → use `spawn()`.
- Headless Chrome has no microphone/speech: `tests/ui_test.py` mocks the speech APIs; the mock must wait for the page to be in "Listening" like a real caller.
- Google `events.insert` takes ~1.4 s by itself, so the agent says "One moment while I book that." first.

## Status / open items
Done & verified (offline 29/29, real-Chrome 9/9, live E2E with real Groq/Gemini + real Google Calendar): everything in README "What it does".
Still open:
0. **Vapi** (optional): after deploying run `scripts/vapi_setup`, set the 3 env vars on the host, then make ONE real test call and fix whatever the live API disagrees with (see the unverified list above). Vapi credits are the owner's money: don't loop test calls.
1. **No public live link yet** (most important). Deploy this repo (Dockerfile present) to Hugging Face Spaces / Render with the `.env` values as secrets (`GOOGLE_SERVICE_ACCOUNT_JSON` = key text). Point the Twilio webhook at `<url>/twilio/voice` (trial: verified numbers only; the web link is the safe deliverable).
2. Real microphone behaviour (echo filter, barge-in, email n-best re-ranking, neural voice quality) is only verified with a mocked recognizer — needs a human with headphones in Chrome/Edge.
3. Calendar **invites to the caller's email** need the OAuth step (`client_secret.json` → `python -m scripts.google_login`); until then the email is only in the event description.
4. Groq/Gemini free limits can still bite during a burst of tests/calls (fallbacks + code paths keep the booking flow correct, but LLM-answered turns get slower/weaker).
5. Screen recording + hosting are the owner's tasks. Rotate the Groq and Gemini keys (they were pasted in chat) after submission.
6. Competitor demo (https://ai-voice-appointment-agent.vercel.app) uses Vapi (managed voice platform with free trial credits; allowed by the assessment). Only consider it if this stack's voice quality isn't good enough; never probe its backend.

## Conventions
- Agent replies: 1–3 short sentences, English only, end with one question, times spoken as "3 PM".
- Commit messages end with the Co-Authored-By line the harness specifies.
