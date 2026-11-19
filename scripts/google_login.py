"""One-time Google sign-in so the agent can send real calendar invites (with a Google Meet link) to callers.

Needs `client_secret.json` (Google Cloud -> APIs & Services -> Credentials -> Create credentials ->
OAuth client ID -> Application type: Desktop app -> Download JSON) in the project folder.

Run:  .venv\\Scripts\\python -m scripts.google_login
A browser opens, you sign in with the calendar owner's Google account, and google_token.json is saved.
"""
import http.server
import json
import secrets
import sys
import threading
import urllib.parse
import webbrowser
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
SCOPE = "https://www.googleapis.com/auth/calendar"


def main():
    secret_file = ROOT / "client_secret.json"
    if not secret_file.exists():
        sys.exit(f"Missing {secret_file}. Download the OAuth 'Desktop app' client JSON and save it with that name.")
    raw = json.loads(secret_file.read_text())
    cfg = raw.get("installed") or raw.get("web")
    client_id, client_secret = cfg["client_id"], cfg["client_secret"]

    result: dict = {}
    state = secrets.token_urlsafe(16)

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            if q.get("state", [""])[0] == state:
                result.update({k: v[0] for k, v in q.items()})
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            msg = "Signed in! You can close this tab." if "code" in result else f"Sign-in failed: {result.get('error', 'unknown')}"
            self.wfile.write(f"<h2>{msg}</h2>".encode())

        def log_message(self, *a):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    redirect = f"http://127.0.0.1:{server.server_port}"
    url = "https://accounts.google.com/o/oauth2/v2/auth?" + urllib.parse.urlencode({
        "client_id": client_id, "redirect_uri": redirect, "response_type": "code", "scope": SCOPE,
        "access_type": "offline", "prompt": "consent", "state": state,
    })
    print("Opening browser for Google sign-in. If it doesn't open, visit:\n" + url)
    webbrowser.open(url)
    t = threading.Thread(target=server.handle_request)
    t.start()
    t.join(timeout=300)
    server.server_close()
    if "code" not in result:
        sys.exit(f"Sign-in not completed: {result.get('error', 'timed out')}")

    r = httpx.post("https://oauth2.googleapis.com/token", data={
        "code": result["code"], "client_id": client_id, "client_secret": client_secret,
        "redirect_uri": redirect, "grant_type": "authorization_code",
    }, timeout=60)
    r.raise_for_status()
    tok = r.json()
    if "refresh_token" not in tok:
        sys.exit("Google did not return a refresh token. Remove the app's access at "
                 "https://myaccount.google.com/permissions and run this again.")
    out = {"refresh_token": tok["refresh_token"], "client_id": client_id, "client_secret": client_secret,
           "token_uri": "https://oauth2.googleapis.com/token", "scopes": [SCOPE]}
    (ROOT / "google_token.json").write_text(json.dumps(out, indent=2))
    print("Saved google_token.json - the agent will now send calendar invites to callers.")


if __name__ == "__main__":
    main()
