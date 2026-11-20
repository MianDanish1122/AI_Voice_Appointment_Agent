"""Browser tests: the real voice-only page in real Chrome (Playwright) against the real backend, in-process.
Only the LLM and Google Calendar are faked (same fakes as tests/smoke_test.py), and the browser's speech APIs are
mocked because headless Chrome has no microphone / speech service.

One-time dev setup:  .venv\\Scripts\\pip install playwright      (uses your installed Chrome; nothing to download)
Run:                 .venv\\Scripts\\python -m tests.ui_test        (UI_ONLY=<part of a test name> runs a subset)
Screenshots land in tests/screens/ .
"""
import json
import os
import re
import tempfile
import threading
import time
from pathlib import Path

from tests import smoke_test as T                       # installs the fake LLM + fixed clock (Mon 2026-11-23 09:00)
from app import booking, config                          # noqa: E402

import uvicorn                                           # noqa: E402
from playwright.sync_api import expect, sync_playwright  # noqa: E402

LAST = {}
CONTEXTS = []                       # every browser context a test opens; closed after the test (memory on small machines)
PORT = 8765
URL = f"http://127.0.0.1:{PORT}/"
SHOTS = Path(__file__).parent / "screens"
SHOTS.mkdir(exist_ok=True)

INIT = """
(() => {
  class MockSR { constructor(){ this.continuous=false; this.interimResults=false; this.maxAlternatives=1; window.__sr=this; window.__srCount=(window.__srCount||0)+1; }
                 start(){} stop(){} abort(){} }
  window.SpeechRecognition = MockSR; window.webkitSpeechRecognition = MockSR;
  window.__say = (text, final=true, alts=null) => { const sr=window.__sr; if(!sr||!sr.onresult) return false;
      const r=(alts||[text]).map(t=>({transcript:t})); r.isFinal=final; sr.onresult({resultIndex:0, results:[r]}); return true; };
  window.__speakMs = 60;
  const synth = { speaking:false, onvoiceschanged:null, getVoices(){ return []; }, cancel(){ clearTimeout(this._t); },
                  speak(u){ this._t=setTimeout(()=>{ u.onend && u.onend(); }, window.__speakMs); } };
  Object.defineProperty(window,'speechSynthesis',{value:synth, configurable:true});
  window.SpeechSynthesisUtterance = function(t){ this.text=t; };
})();
"""

FAKE_VAPI_SDK = """
export default class Vapi {
  constructor(key){ this.key=key; this.h={}; window.__vapi=this; }
  on(e,f){ (this.h[e]=this.h[e]||[]).push(f); return this; }
  emit(e,...a){ (this.h[e]||[]).forEach(f=>f(...a)); }
  async start(id, ov){ window.__vapiStart={key:this.key,id,ov}; setTimeout(()=>this.emit('call-start'),20); return {id:'call-xyz'}; }
  stop(){ window.__vapiStopped=true; this.emit('call-end'); }
}
"""


def start_server():
    config.VAPI_PUBLIC_KEY, config.VAPI_ASSISTANT_ID = "pk_test", "asst_test"     # makes the Vapi engine option appear
    config.DB_PATH = Path(tempfile.mkdtemp()) / "ui.db"
    from app.main import app
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=PORT, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    while not server.started:
        time.sleep(0.1)
    return server


def new_page(browser, width=1440, height=900, scheme="dark"):
    ctx = browser.new_context(viewport={"width": width, "height": height}, color_scheme=scheme, permissions=["microphone"])
    CONTEXTS.append(ctx)
    ctx.add_init_script(INIT)
    page = ctx.new_page()
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" and "Failed to load resource" not in m.text else None)
    page.route(re.compile(r".*/api/tts.*"), lambda r: r.fulfill(status=503))        # force the browser voice
    page.route(re.compile(r"https://fonts\.(googleapis|gstatic)\.com/.*"), lambda r: r.abort())
    page.errors = errors
    LAST["page"] = page
    booking.clear_slot_cache()
    page.goto(URL)
    expect(page.locator("#wordmark")).to_contain_text(re.compile("danish digital solutions", re.I))
    return page


def say(page, text, final=True, wait_listening=True):
    """Pretend the caller spoke (the page's own silence timer then sends it). Like a real caller, wait until the
    agent has finished and the page is listening - speech during 'thinking' is (correctly) ignored."""
    if wait_listening:
        expect(page.locator("#phase")).to_contain_text("listening")
    before = page.locator(".line.agent").count()
    for _ in range(80):
        if page.evaluate("(t) => window.__say(t, %s)" % ("true" if final else "false"), text):
            break
        page.wait_for_timeout(100)
    else:
        raise AssertionError("recognizer never became ready")
    if wait_listening:        # a real caller waits for the answer before speaking again
        expect(page.locator(".line.agent")).not_to_have_count(before)
        expect(page.locator("#phase")).to_contain_text(re.compile("listening|ended"))


def last_agent(page):
    return page.locator(".line.agent .text").last


def start_call(page):
    page.click("#callBtn")
    expect(page.locator(".line.agent .text").first).to_contain_text("Hello, thank you for calling")
    expect(page.locator("#callBtnText")).to_have_text("end call")


I = re.I


def test_voice_call_books_end_to_end(browser):
    config.DB_PATH = Path(tempfile.mkdtemp()) / "a.db"
    page = new_page(browser)
    start_call(page)
    expect(page.locator("#apparatus")).to_have_attribute("data-live", "1")
    say(page, "Book a meeting on November 25 at 11 AM")
    expect(page.locator("#vDate")).to_have_text(re.compile("wednesday, november 25", I))      # the sheet fills as you speak
    expect(page.locator("#vTime")).to_have_text(re.compile("11 am", I))
    expect(page.locator("#slotLine")).to_contain_text(re.compile("available", I))
    say(page, "My name is Ahmed Khan")
    expect(page.locator("#vName")).to_have_text(re.compile("ahmed khan", I))
    say(page, "my email is ahmed khan 92 at gmail dot com")
    expect(page.locator("#vEmail")).to_have_text("ahmedkhan92@gmail.com")
    expect(page.locator("#tagEmail")).to_contain_text(re.compile("say yes to confirm", I))
    page.screenshot(path=str(SHOTS / "lumen-night-call.png"))
    say(page, "I want to discuss pricing for a new website")
    expect(last_agent(page)).to_contain_text("Just to confirm")
    expect(page.locator("#cue")).to_contain_text(re.compile("say .yes. to book", I))          # voice-only: the cue tells you to say yes
    assert T.bookings() == [], "nothing may be booked before the caller says yes"
    say(page, "yes")
    expect(page.locator("#success")).to_be_visible()
    expect(last_agent(page)).to_contain_text("Done!")
    expect(page.locator("#successDl")).to_contain_text(re.compile("wednesday, november 25th at 11 am", I))
    expect(page.locator("#vPurpose")).to_have_text(re.compile("pricing for a new website", I))
    expect(page.locator(".step.is-done")).to_have_count(4)                                    # the progress track completes
    rows = T.bookings()
    assert len(rows) == 1 and rows[0]["email"] == "ahmedkhan92@gmail.com", rows
    expect(page.locator("#statBookings")).to_have_text("1")
    expect(page.locator("#statFirst")).to_contain_text("s")                                    # a real measured number, not a placeholder
    expect(page.locator("#bookings")).to_contain_text(re.compile("ahmed khan", I))
    page.screenshot(path=str(SHOTS / "lumen-night-booked.png"))
    assert not page.errors, page.errors


def test_voice_only_there_is_no_typing_surface(browser):
    page = new_page(browser)
    assert page.locator("input[type=text], input[type=email], input[type=search], textarea, form, [contenteditable]").count() == 0
    assert page.locator("#text, #composer, .try, #fName, #fEmail, #fDate, #fTime, #fPurpose, #reviewBtn").count() == 0
    # what remains are settings controls, not a way to type the booking
    assert page.locator("select").count() >= 1 and page.locator("#callBtn").count() == 1
    page.keyboard.type("hello")                                  # typing on the page does nothing
    assert page.locator(".line").count() == 0


def test_email_is_verified_by_voice(browser):
    config.DB_PATH = Path(tempfile.mkdtemp()) / "b.db"
    page = new_page(browser)
    start_call(page)
    say(page, "Book a meeting on November 25 at 11 AM")
    say(page, "My name is Ahmed Khan")
    say(page, "my email is ahmed at gmial dot com")                              # mis-heard domain
    expect(last_agent(page)).to_contain_text("Did you mean")
    expect(page.locator("#tagEmail")).to_contain_text(re.compile("say yes for ahmed@gmail.com", I))
    expect(page.locator("#cue")).to_contain_text(re.compile("say .yes. if the email is right", I))
    say(page, "yes")                                                             # the corrected spelling is confirmed by voice
    expect(page.locator("#vEmail")).to_have_text("ahmed@gmail.com")
    expect(page.locator("#tagEmail")).to_have_text(re.compile("verified", I))
    assert not page.errors, page.errors


def test_changed_mind_updates_the_sheet_live(browser):
    config.DB_PATH = Path(tempfile.mkdtemp()) / "c.db"
    page = new_page(browser)
    start_call(page)
    say(page, "Book a meeting tomorrow at 5 PM... actually, make it the day after tomorrow at 11 AM")
    expect(page.locator("#vDate")).to_have_text(re.compile("wednesday, november 25", I))
    expect(page.locator("#vTime")).to_have_text(re.compile("11 am", I))
    say(page, "actually make it 3 PM")
    expect(page.locator("#vTime")).to_have_text(re.compile("3 pm", I))
    expect(last_agent(page)).to_contain_text("Sure, I'll update that")
    say(page, "Is Saturday okay?")
    page.wait_for_timeout(300)
    assert not page.errors, page.errors


def test_unavailable_slot_shows_alternatives(browser):
    g = T.FakeGoogle(busy=[T.busy_slot("2026-11-25", "11:00")]).install()
    try:
        config.DB_PATH = Path(tempfile.mkdtemp()) / "d.db"
        page = new_page(browser)
        start_call(page)
        say(page, "Book a meeting on November 25 at 11 AM")
        expect(last_agent(page)).to_contain_text("isn't available")
        expect(page.locator("#slotLine")).to_contain_text(re.compile("not available", I))
        expect(page.locator("#slotLine")).to_contain_text(re.compile("try .* am", I))      # open times offered, to be said out loud
        assert T.bookings() == [] and g.events == []
    finally:
        g.uninstall()


def test_google_down_is_reported_not_hidden(browser):
    g = T.FakeGoogle(fail_create=True).install()
    try:
        config.DB_PATH = Path(tempfile.mkdtemp()) / "e.db"
        page = new_page(browser)
        start_call(page)
        say(page, "Book a meeting on November 25 at 11 AM")
        say(page, "My name is Ahmed Khan")
        say(page, "my email is ahmed at gmail dot com")
        say(page, "I want to discuss hiring")
        expect(last_agent(page)).to_contain_text("Just to confirm")
        say(page, "yes")
        expect(last_agent(page)).to_contain_text("nothing has been booked")
        expect(page.locator("#success")).to_be_hidden()
        assert T.bookings() == []
    finally:
        g.uninstall()


def test_interruption(browser):
    config.DB_PATH = Path(tempfile.mkdtemp()) / "f.db"
    page = new_page(browser)
    page.evaluate("window.__speakMs = 6000")                                      # a long-winded agent
    page.click("#callBtn")
    expect(page.locator(".line.agent .text").first).to_contain_text("Hello")
    expect(page.locator("#phase")).to_contain_text("speaking")
    page.wait_for_timeout(400)
    say(page, "wait what are your opening hours please", final=False, wait_listening=False)   # talks over the agent
    expect(page.locator(".note", has_text="interrupted")).to_be_visible()
    expect(page.locator("#phase")).to_contain_text("listening")


def test_vapi_engine_wires_the_call_and_the_sheet_follows(browser):
    """Our side of the Vapi integration, end to end: the page starts a Vapi call (SDK mocked) with our session id, Vapi's
    webhook (posted here exactly as Vapi would) drives the same booking rules, and the sheet + success card follow live."""
    import httpx
    config.DB_PATH = Path(tempfile.mkdtemp()) / "v.db"
    page = new_page(browser)
    page.route(re.compile(r"https://esm\.sh/@vapi-ai/web.*"),
               lambda r: r.fulfill(status=200, content_type="application/javascript", body=FAKE_VAPI_SDK))
    page.click(".settings summary")
    expect(page.locator("#engineRow")).to_be_visible()
    page.select_option("#engine", "vapi")
    expect(page.locator("#voiceRow")).to_be_hidden()
    expect(page.locator("#coEngine")).to_have_text("vapi")
    page.click("#callBtn")
    page.wait_for_function("window.__vapiStart")
    start = page.evaluate("window.__vapiStart")
    assert start["key"] == "pk_test" and start["id"] == "asst_test", start
    sid = start["ov"]["variableValues"]["sid"]
    assert sid.startswith("web-"), start
    expect(page.locator("#phase")).to_contain_text("connected")

    def vapi_tool(name, messages, args=None):
        body = {"message": {"type": "tool-calls", "call": {"id": "call-xyz", "type": "webCall"}, "artifact": {"messages": messages},
                            "toolCallList": [{"id": "tc", "type": "function",
                                              "function": {"name": name, "arguments": {"session_id": sid, **(args or {})}}}]}}
        r = httpx.post(URL + "api/vapi/webhook", json=body, timeout=30)
        assert r.status_code == 200, r.text
        return json.loads(r.json()["results"][0]["result"])

    hello = {"role": "assistant", "message": "Hello, how can I help?"}
    u1 = {"role": "user", "message": "Book a meeting on November 25 at 11 AM, my name is Ahmed Khan, "
                                     "my email is ahmed at gmail dot com, I want to discuss hiring"}
    out = vapi_tool("update_booking_details", [hello, u1])
    assert out["say"].startswith("Just to confirm"), out
    expect(page.locator("#vDate")).to_have_text(re.compile("wednesday, november 25", I))      # the webhook filled the sheet...
    expect(page.locator("#vTime")).to_have_text(re.compile("11 am", I))
    expect(page.locator("#vName")).to_have_text(re.compile("ahmed khan", I))
    expect(page.locator("#vEmail")).to_have_text("ahmed@gmail.com")
    expect(page.locator("#vPurpose")).to_have_text(re.compile("hiring", I))                    # ...and the page showed it without a reload
    expect(page.locator("#cue")).to_contain_text(re.compile("say .yes. to book", I))
    page.evaluate("window.__vapi.emit('message', {type:'transcript', role:'user', transcriptType:'final', transcript:'book it please'})")
    expect(page.locator(".line.user .text")).to_contain_text("book it please")
    assert T.bookings() == [], "nothing may be booked before the caller says yes"
    out = vapi_tool("book_appointment", [hello, u1, {"role": "assistant", "message": out["say"]}, {"role": "user", "message": "yes"}])
    assert out["success"] is True and out["say"].startswith("Done!"), out
    expect(page.locator("#success")).to_be_visible()
    assert len(T.bookings()) == 1
    page.click("#callBtn")                                                       # End call
    page.wait_for_function("window.__vapiStopped")
    expect(page.locator("#callBtnText")).to_have_text("start call")
    assert not page.errors, page.errors


def test_primary_action_is_above_the_fold_on_a_laptop(browser):
    """Hallmark gate 44: on a 13-inch laptop (1280x800) the headline and the 'start call' button are visible without scrolling."""
    page = new_page(browser, 1280, 800, "dark")
    for sel in ("#h1", "#callBtn"):
        box = page.locator(sel).bounding_box()
        assert box and box["y"] + box["height"] <= 800, f"{sel} is below the fold: {box}"
    page.screenshot(path=str(SHOTS / "lumen-night-laptop-fold.png"))


def test_night_and_day_drops_and_responsive_floor(browser):
    config.DB_PATH = Path(tempfile.mkdtemp()) / "g.db"
    # night is the default for a dark OS, day for a light one; the toggle flips and remembers
    night = new_page(browser, 1440, 900, "dark")
    assert night.evaluate("document.documentElement.dataset.drop") == "night"
    night.screenshot(path=str(SHOTS / "lumen-night-desktop.png"))
    night.click("#themeBtn")
    assert night.evaluate("document.documentElement.dataset.drop") == "day"
    day = new_page(browser, 1440, 900, "light")
    assert day.evaluate("document.documentElement.dataset.drop") == "day"
    day.wait_for_timeout(500)
    day.screenshot(path=str(SHOTS / "lumen-day-desktop.png"))
    # the mobile floor: 320 / 375 / 414 / 768 - no horizontal scroll, no clickable text wrapping to two lines
    for w, h in ((320, 640), (375, 812), (414, 896), (768, 1024)):
        p = new_page(browser, w, h, "dark")
        p.wait_for_timeout(200)
        scroll = p.evaluate("[document.documentElement.scrollWidth, window.innerWidth, document.body.scrollWidth]")
        assert scroll[0] <= scroll[1] + 1 and scroll[2] <= scroll[1] + 1, f"horizontal scroll at {w}px: {scroll}"
        lines = p.evaluate("""() => ['#callBtnText', '#micBtnText', '#themeLabel', '.settings summary', '#wordmark'].map(s => {
            const e = document.querySelector(s), r = document.createRange(); r.selectNodeContents(e);
            return [s, new Set([...r.getClientRects()].map(x => Math.round(x.top))).size];
        })""")
        assert all(n <= 1 for _, n in lines), f"clickable text wraps at {w}px: {lines}"
        if w == 375:
            p.screenshot(path=str(SHOTS / "lumen-night-phone.png"), full_page=True)
    assert not night.errors, night.errors


def main():
    server = start_server()
    expect.set_options(timeout=15000)       # generous: CI boxes and old laptops are slow with Chrome + server together
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    only = os.environ.get("UI_ONLY")
    if only:
        tests = [(n, f) for n, f in tests if only in n]
    failed = 0
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True,
                                    args=["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream"])
        for name, fn in tests:
            try:
                fn(browser)
                print(f"  PASS  {name}")
            except Exception as e:
                failed += 1
                print(f"  FAIL  {name}: {str(e)[:400]}")
                try:                                    # what the user (and the server) saw when it failed
                    pg = LAST["page"]
                    print("        transcript:", " | ".join(pg.inner_text("#log").splitlines())[-600:])
                    print("        phase:", pg.inner_text("#phase"), "| sheet:", pg.evaluate(
                        "JSON.stringify(window.__app.S.state && window.__app.S.state.values)"))
                except Exception:
                    pass
            for ctx in CONTEXTS:                         # free the pages of this test before the next one starts
                try:
                    ctx.close()
                except Exception:
                    pass
            CONTEXTS.clear()
        browser.close()
    server.should_exit = True
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
