"""Create or update the Vapi assistant + tools for this app (idempotent: matched by name, never duplicates).

Vapi must be able to reach this server, so run it with the PUBLIC https address of the deployed app:

    .venv\\Scripts\\python -m scripts.vapi_setup --server-url https://your-app.hf.space --dry-run   # show what would be sent
    .venv\\Scripts\\python -m scripts.vapi_setup --server-url https://your-app.hf.space             # do it

It uses VAPI_API_KEY (the PRIVATE key) from .env, generates VAPI_WEBHOOK_SECRET if you have none, and writes
VAPI_ASSISTANT_ID / VAPI_WEBHOOK_SECRET back into .env. Put VAPI_PUBLIC_KEY, VAPI_ASSISTANT_ID and VAPI_WEBHOOK_SECRET into
the host's environment too. Re-run it whenever the knowledge base (knowledge/*.md) changes - the prompt embeds it.
Your other Vapi assistants are not touched.
"""
import argparse
import json
import re
import secrets
import sys
from pathlib import Path

import httpx

from app import config, vapi

API = "https://api.vapi.ai"
ENV = Path(config.ROOT) / ".env"


def set_env(key: str, value: str):
    text = ENV.read_text(encoding="utf-8") if ENV.exists() else ""
    line = f"{key}={value}"
    if re.search(rf"^{key}=.*$", text, flags=re.M):
        text = re.sub(rf"^{key}=.*$", lambda m: line, text, flags=re.M)
    else:
        text += ("" if text.endswith("\n") or not text else "\n") + line + "\n"
    ENV.write_text(text, encoding="utf-8")


def mask(obj, secret: str):
    return json.loads(json.dumps(obj).replace(secret, "***"))


def call(client: httpx.Client, method: str, path: str, **kw):
    r = client.request(method, API + path, **kw)
    if r.status_code >= 400:
        sys.exit(f"Vapi {method} {path} -> HTTP {r.status_code}: {r.text[:600]}")
    return r.json() if r.content else {}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server-url", required=True, help="public https base URL of this app, e.g. https://you.hf.space")
    ap.add_argument("--voice", default="Elliot", help="Vapi voice id (default Elliot)")
    ap.add_argument("--model", default="gpt-4o-mini", help="LLM used by Vapi (billed through your Vapi credits)")
    ap.add_argument("--dry-run", action="store_true", help="print the payloads and exit; nothing is sent")
    a = ap.parse_args()

    base = a.server_url.rstrip("/")
    if not base.startswith("https://") and not a.dry_run:
        sys.exit("Vapi needs a public https URL (a localhost address cannot be reached by Vapi).")
    url = base + "/api/vapi/webhook"
    secret = config.VAPI_WEBHOOK_SECRET or secrets.token_urlsafe(24)
    tools = vapi.tool_payloads(url, secret)

    if a.dry_run:
        print(json.dumps({"tools": mask(tools, secret),
                          "assistant": mask(vapi.assistant_payload(url, secret, ["<tool ids>"], a.voice, a.model), secret)},
                         indent=2)[:6000])
        return
    if not config.VAPI_API_KEY:
        sys.exit("VAPI_API_KEY (the private key) is missing in .env")

    with httpx.Client(headers={"Authorization": f"Bearer {config.VAPI_API_KEY}"}, timeout=40) as c:
        existing = {(t.get("function") or {}).get("name"): t for t in call(c, "GET", "/tool")}
        tool_ids = []
        for t in tools:
            name = t["function"]["name"]
            if name in existing and existing[name].get("server", {}).get("url", "").rstrip("/").endswith("/api/vapi/webhook"):
                body = {k: v for k, v in t.items() if k != "type"}            # a tool's type cannot change
                tid = call(c, "PATCH", f"/tool/{existing[name]['id']}", json=body)["id"]
                print("updated tool", name)
            else:
                tid = call(c, "POST", "/tool", json=t)["id"]
                print("created tool", name)
            tool_ids.append(tid)

        payload = vapi.assistant_payload(url, secret, tool_ids, a.voice, a.model)
        mine = next((x for x in call(c, "GET", "/assistant") if x.get("name") == payload["name"]), None)
        if mine:
            aid = call(c, "PATCH", f"/assistant/{mine['id']}", json=payload)["id"]
            print("updated assistant", payload["name"])
        else:
            aid = call(c, "POST", "/assistant", json=payload)["id"]
            print("created assistant", payload["name"])

    set_env("VAPI_ASSISTANT_ID", aid)
    set_env("VAPI_WEBHOOK_SECRET", secret)
    print(f"\nassistant id: {aid}\nwebhook     : {url}\nSaved VAPI_ASSISTANT_ID and VAPI_WEBHOOK_SECRET to .env.")
    print("On the host set: VAPI_PUBLIC_KEY, VAPI_ASSISTANT_ID, VAPI_WEBHOOK_SECRET (same values).")


if __name__ == "__main__":
    main()
