# AI Voice Appointment Agent ($0, free-tier tools only)

A voice agent that answers a call (web call link or phone number), talks naturally, collects
**full name, email, date and time**, confirms the details, and books the meeting in **Google Calendar**
(checking availability first, so it never double-books).

## Architecture

```
Caller ──web call (Chrome mic)──┐                         ┌─► Groq LLM (gpt-oss-120b, free) ─► fallbacks:
       ──phone (Twilio trial)───┤                         │     gpt-oss-20b / qwen / Gemini (free)
                                ▼                         │
                     FastAPI server (this repo) ──────────┤─► knowledge/ (company FAQ, local BM25 search)
                     speech ⇄ text, agent logic           │
                                                          └─► Google Calendar API
                                                                freeBusy (availability) + events.insert
```

- **Voice in/out:** web call uses the browser's speech recognition (continuous, waits for the caller to finish,
  supports interrupting the agent) and a free Microsoft neural voice (edge-tts) for replies. Phone calls use
  Twilio `<Gather input="speech">` + neural `<Say>`.
- **Brain:** each caller sentence goes to the LLM with tools `check_available_slots`, `book_appointment`,
  `transfer_to_human`, `end_call`. Relative dates ("tomorrow at 3 PM", "next Monday") are resolved against
  today's date in the configured time zone; date changes mid-call restart the slot check.
- **Guard rails in code (not just the prompt):** past/closed/too-far dates refused; name and email must have
  been said by the caller (no invented data); spoken emails ("ahmed at gmail dot com") normalised; booking only
  after the system reads back date, time, name and email and the caller says yes; call ends only on goodbye.
- **Calendar:** `freeBusy` hides busy slots, then `events.insert` creates the event (Asia/Karachi time zone)
  with name, email, phone and purpose. Optional owner sign-in (`scripts/google_login.py`) sends a real
  invite to the caller's email with a Google Meet link.

Below are the setup notes (Roman Urdu).

## 1. Setup (ek dafa)

```powershell
cd "d:\AI Voice"
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
copy .env.example .env
```

**Free LLM key:** https://console.groq.com/keys par sign up karo → key banao → `.env` mein `LLM_API_KEY=` par paste karo.
(Ya Gemini free key: https://aistudio.google.com/apikey — ya Ollama bilkul local, `.env.example` mein options hain.)

`.env` mein apna business name, timings, `HUMAN_PHONE` set karo.

## 2. Knowledge base

`knowledge/` folder mein `.md` ya `.txt` files daalo (services, prices, FAQ, policies).
Sample file `knowledge/company_info.md` ko apni real info se replace karo.
Server chalte hue file badli ho to: `POST http://localhost:8000/api/kb/reload` (ya server restart).

## 3. Run

```powershell
.\.venv\Scripts\python -m uvicorn app.main:app --port 8000
```

## 4. Browser mein test (bilkul free, bina phone ke)

Chrome / Edge mein kholo: http://localhost:8000 → **Start call** → mic allow karo → baat karo.
Try karo: *"What services do you offer?"*, *"Book a meeting tomorrow at 3 PM"*, *"I want to talk to a human"*.
Bookings neeche table mein dikhti hain.

## 5. Real phone call (Twilio free trial)

1. https://www.twilio.com/try-twilio par free account (trial credit milta hai, card nahi chahiye) → ek free trial number lo.
2. Server ko internet par expose karo (free, account nahi chahiye):
   - Cloudflared download: https://github.com/cloudflare/cloudflared/releases (`cloudflared-windows-amd64.exe`)
   - `cloudflared-windows-amd64.exe tunnel --url http://localhost:8000`
   - Ye ek URL dega jaise `https://xyz.trycloudflare.com`
3. Twilio Console → Phone Numbers → apna number → **Voice Configuration**:
   - *A call comes in*: Webhook → `https://xyz.trycloudflare.com/twilio/voice` (HTTP POST)
   - *Call status changes*: `https://xyz.trycloudflare.com/twilio/status`
4. Ab us number par call karo. 🎉

**Trial ki limits:** sirf *verified* numbers call kar sakte hain aur transfer bhi sirf verified number par hoga
(Twilio Console → Verified Caller IDs mein apna aur `HUMAN_PHONE` add karo). Call ke shuru mein Twilio ka chhota sa trial message bajta hai.

**Urdu/Hindi:** `.env` mein `STT_LANGUAGE=ur-PK` (ya `hi-IN`) aur `TTS_VOICE=Polly.Aditi`, `TTS_LANGUAGE=hi-IN` try karo.

## 6. Google Calendar (free)

1. https://console.cloud.google.com → New project.
2. "APIs & Services" → Library → **Google Calendar API** → Enable.
3. "IAM & Admin" → Service Accounts → Create → Keys → Add key → JSON → download.
   File ko project folder mein `service_account.json` naam se rakho.
4. Google Calendar (calendar.google.com) → apna calendar → Settings and sharing →
   *Share with specific people* → service account ka email (`...@...iam.gserviceaccount.com`) add karo → **Make changes to events**.
5. Usi page par *Integrate calendar* → **Calendar ID** copy karke `.env` mein `GOOGLE_CALENDAR_ID=` par lagao.

Ab agent Google Calendar ke busy time ko skip karega aur booking par event create karega.
Agar `GOOGLE_CALENDAR_ID` khali ho to sirf local SQLite use hota hai.

## 7. Live link (free hosting) – Hugging Face Spaces

1. https://huggingface.co → New Space → SDK: **Docker** → Public.
2. Is folder ki files upload karo (`.env`, `service_account.json`, `.venv`, `data`, `tools` **mat** upload karo).
3. Space → Settings → **Secrets** mein `.env` wali values daalo
   (`LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`, `GOOGLE_CALENDAR_ID`, `HUMAN_PHONE`, `BUSINESS_NAME` ...)
   aur `GOOGLE_SERVICE_ACCOUNT_JSON` mein service_account.json ka **pura text** paste karo.
4. Live link: `https://<username>-<space>.hf.space` — yehi submit karo.
   Twilio webhook bhi isi URL par lagao: `https://<username>-<space>.hf.space/twilio/voice` (tunnel ki zarurat nahi).

Note: hosted version mein Ollama nahi chalega — Groq (ya Gemini) key use karo.

## Useful URLs

| URL | Kaam |
|---|---|
| `GET /api/bookings` | sari bookings |
| `GET /api/slots/2026-11-20` | us din ke free slots |
| `POST /api/kb/reload` | knowledge base reload |

## Files

- `app/agent.py` — AI ka dimagh: prompt, tools (slot check, book, transfer, end call)
- `app/calendar_store.py` — slots + bookings (SQLite)
- `app/kb.py` — knowledge base search
- `app/llm.py` — free LLM client (OpenAI-compatible)
- `app/main.py` — Twilio webhooks + browser test API
- `static/index.html` — browser test page
