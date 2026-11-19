import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def env(key: str, default: str = "") -> str:
    return os.getenv(key, default).strip()


LLM_BASE_URL = env("LLM_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/")
LLM_API_KEY = env("LLM_API_KEY")
LLM_MODEL = env("LLM_MODEL", "openai/gpt-oss-120b")
LLM_TIMEOUT = float(env("LLM_TIMEOUT", "25"))
# Tried in order when the main model hits a free-tier rate limit.
LLM_FALLBACK_MODELS = [m.strip() for m in env("LLM_FALLBACK_MODELS", "openai/gpt-oss-20b,qwen/qwen3.8-27b").split(",") if m.strip()]
# Optional extra free provider: Google Gemini (https://aistudio.google.com/apikey)
GEMINI_API_KEY = env("GEMINI_API_KEY")
GEMINI_MODELS = [m.strip() for m in env("GEMINI_MODELS", "gemini-2.5-flash,gemini-2.5-flash-lite").split(",") if m.strip()]

# Natural voice for the web call (free Microsoft Edge neural voices)
WEB_TTS_VOICE = env("WEB_TTS_VOICE", "en-US-AvaMultilingualNeural")

BUSINESS_NAME = env("BUSINESS_NAME", "Acme Digital Solutions")
AGENT_NAME = env("AGENT_NAME", "Sara")
TIMEZONE = env("TIMEZONE", "Asia/Karachi")
WORK_DAYS = [int(d) for d in env("WORK_DAYS", "0,1,2,3,4").split(",") if d.strip()]
OPEN_TIME = env("OPEN_TIME", "10:00")
CLOSE_TIME = env("CLOSE_TIME", "18:00")
SLOT_MINUTES = int(env("SLOT_MINUTES", "30"))

HUMAN_PHONE = env("HUMAN_PHONE")

# Google Calendar (optional; if empty, bookings are stored only in local SQLite)
GOOGLE_CALENDAR_ID = env("GOOGLE_CALENDAR_ID")
GOOGLE_SERVICE_ACCOUNT_FILE = env("GOOGLE_SERVICE_ACCOUNT_FILE", "service_account.json")
GOOGLE_SERVICE_ACCOUNT_JSON = env("GOOGLE_SERVICE_ACCOUNT_JSON")  # full JSON text (handy on cloud hosting)
# Optional: sign in as the calendar owner (python -m scripts.google_login) to send real invites to callers.
GOOGLE_OAUTH_TOKEN_FILE = env("GOOGLE_OAUTH_TOKEN_FILE", "google_token.json")
GOOGLE_OAUTH_TOKEN_JSON = env("GOOGLE_OAUTH_TOKEN_JSON")

STT_LANGUAGE = env("STT_LANGUAGE", "en-US")
TTS_VOICE = env("TTS_VOICE", "Polly.Joanna-Neural")
TTS_LANGUAGE = env("TTS_LANGUAGE", "en-US")

KNOWLEDGE_DIR = ROOT / "knowledge"
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)
DB_PATH = DATA_DIR / "bookings.db"
