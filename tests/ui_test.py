"""Browser tests: the real page in real Chrome (Playwright) against the real backend, in-process.
Only the LLM and Google Calendar are faked (same fakes as tests/smoke_test.py), and the browser's speech APIs are
mocked because headless Chrome has no microphone / speech service.

One-time dev setup:  .venv\\Scripts\\pip install playwright      (uses your installed Chrome; nothing to download)
Run:                 .venv\\Scripts\\python -m tests.ui_test
Screenshots land in tests/screens/ .
"""
import json
import re
import tempfile
import threading
import time
from pathlib import Path

from tests import smoke_test as T                       # installs the fake LLM + fixed clock (Mon 2026-11-23 09:00)
from app import booking, config, gcal                    # noqa: E402

import uvicorn                                           # noqa: E402
from playwright.sync_api import expect, sync_playwright  # noqa: E402

LAST = {}
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


def start_server():
    config.DB_PATH = Path(tempfile.mkdtemp()) / "ui.db"
    from app.main import app
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=PORT, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    while not server.started:
        time.sleep(0.1)
    return server


def new_page(browser, width=1440, height=900, scheme="light"):
    ctx = browser.new_context(viewport={"width": width, "height": height}, color_scheme=scheme, permissions=["microphone"])
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
    expect(page.locator("#bizline")).to_contain_text("Danish Digital Solutions")
    return page


def say(page, text, final=True, wait_listening=True):
    """Pretend the caller spoke (the page's own silence timer then sends it). Like a real caller, wait until the
    agent has finished and the page is listening - speech during 'thinking' is (correctly) ignored."""
    if wait_listening:
        expect(page.locator("#phase")).to_contain_text("Listening")
    before = page.locator(".msg.agent").count()
    for _ in range(80):
        if page.evaluate("(t) => window.__say(t, %s)" % ("true" if final else "false"), text):
            break
        page.wait_for_timeout(100)
    else:
        raise AssertionError("recognizer never became ready")
    if wait_listening:        # a real caller waits for the answer before speaking again
        expect(page.locator(".msg.agent")).not_to_have_count(before)
        expect(page.locator("#phase")).to_contain_text(re.compile("Listening|ended"))


def last_agent(page):
    return page.locator(".msg.agent .body").last


def test_form_only_flow(browser):
    T.fresh(call=False)
    config.DB_PATH = Path(tempfile.mkdtemp()) / "a.db"
    page = new_page(browser)
    assert page.get_attribute("#fDate", "min") == "2026-11-23", "date picker must block past dates"
    # email validation feedback
    page.fill("#fEmail", "not-an-email"); page.press("#fEmail", "Enter")
    expect(page.locator("#msg-email")).to_contain_text("valid email")
    # past date refused by the server too
    page.fill("#fDate", "2026-11-12"); page.press("#fDate", "Tab")
    expect(page.locator("#msg-date")).to_contain_text("passed")
    # fill everything by hand
    page.fill("#fName", "Mia Stone"); page.press("#fName", "Enter")
    page.fill("#fEmail", "Mia@Gmail.com"); page.press("#fEmail", "Enter")
    expect(page.locator("#msg-email")).to_contain_text("verified")
    page.fill("#fDate", "2026-11-25"); page.press("#fDate", "Tab")
    expect(page.locator("#fTime option[value='11:00']")).to_have_count(1)
    page.select_option("#fTime", "11:00")
    expect(page.locator("#slotLine")).to_contain_text("is available")
    page.fill("#fPurpose", "Website redesign"); page.press("#fPurpose", "Tab")
    expect(page.locator("#reviewBtn")).to_be_enabled()
    page.screenshot(path=str(SHOTS / "form-ready-desktop-light.png"))
    page.click("#reviewBtn")
    expect(page.locator("#reviewDlg")).to_be_visible()
    expect(page.locator("#dlgDl")).to_contain_text("mia@gmail.com")
    page.click("#dlgConfirm")
    expect(page.locator("#success")).to_be_visible()
    expect(page.locator("#successDl")).to_contain_text("Wednesday, November 25th at 11 AM")
    rows = T.bookings()
    assert len(rows) == 1 and rows[0]["email"] == "mia@gmail.com" and rows[0]["name"] == "Mia Stone", rows
    expect(page.locator("#bookings")).to_contain_text("Mia Stone")
    page.screenshot(path=str(SHOTS / "booked-desktop-light.png"))
    assert not page.errors, page.errors


def test_voice_call_with_manual_edit_in_the_middle(browser):
    config.DB_PATH = Path(tempfile.mkdtemp()) / "b.db"
    page = new_page(browser)
    page.click("#callBtn")
    expect(page.locator(".msg.agent .body").first).to_contain_text("Hello, thank you for calling")
    expect(page.locator("#callBtnText")).to_have_text("End call")
    say(page, "Book a meeting on November 25 at 11 AM")
    expect(page.locator("#fDate")).to_have_value("2026-11-25")                 # voice fills the form
    expect(page.locator("#fTime")).to_have_value("11:00")
    say(page, "My name is Ahmed Khan")
    expect(page.locator("#fName")).to_have_value("Ahmed Khan")
    say(page, "my email is ahmed khan 92 at gmail dot com")
    expect(page.locator("#fEmail")).to_have_value("ahmedkhan92@gmail.com")
    expect(page.locator("#msg-email")).to_contain_text("Heard by voice")
    page.screenshot(path=str(SHOTS / "call-voice-filling-desktop-light.png"))
    # the caller types the last missing item instead of speaking it
    page.fill("#fPurpose", "Pricing for a new website"); page.press("#fPurpose", "Tab")
    expect(last_agent(page)).to_contain_text("Just to confirm", timeout=8000)
    expect(last_agent(page)).to_contain_text("Pricing for a new website")
    assert T.bookings() == [], "nothing may be booked before the caller says yes"
    say(page, "yes")
    expect(page.locator("#success")).to_be_visible(timeout=8000)
    expect(last_agent(page)).to_contain_text("Done!")
    rows = T.bookings()
    assert len(rows) == 1 and rows[0]["email"] == "ahmedkhan92@gmail.com", rows
    expect(page.locator("#fName")).to_be_hidden()                                # form swapped for the success card
    assert not page.errors, page.errors


def test_voice_email_then_manual_correction(browser):
    config.DB_PATH = Path(tempfile.mkdtemp()) / "c.db"
    page = new_page(browser)
    page.click("#callBtn")
    expect(page.locator(".msg.agent .body").first).to_contain_text("Hello")
    say(page, "Book a meeting on November 25 at 11 AM")
    say(page, "My name is Ahmed Khan")
    say(page, "my email is ahmed at gmial dot com")                              # mis-heard domain
    expect(last_agent(page)).to_contain_text("Did you mean")
    expect(page.locator("#emailBar")).to_contain_text("ahmed@gmail.com")
    page.fill("#fEmail", "ahmed.khan@outlook.com"); page.press("#fEmail", "Enter")   # the caller fixes it by hand
    expect(page.locator("#msg-email")).to_contain_text("verified")
    expect(page.locator("#fEmail")).to_have_value("ahmed.khan@outlook.com")
    say(page, "I want to discuss hiring")
    expect(last_agent(page)).to_contain_text("ahmed dot khan at outlook dot com")      # the agent uses the corrected email
    say(page, "yes")
    expect(page.locator("#success")).to_be_visible(timeout=8000)
    assert T.bookings()[0]["email"] == "ahmed.khan@outlook.com"
    assert not page.errors, page.errors


def test_interruption(browser):
    config.DB_PATH = Path(tempfile.mkdtemp()) / "d.db"
    page = new_page(browser)
    page.evaluate("window.__speakMs = 6000")                                      # a long-winded agent
    page.click("#callBtn")
    expect(page.locator(".msg.agent .body").first).to_contain_text("Hello")
    expect(page.locator("#phase")).to_contain_text("speaking")
    page.wait_for_timeout(400)
    say(page, "wait what are your opening hours please", final=False, wait_listening=False)   # talks over the agent
    expect(page.locator(".note", has_text="interrupted")).to_be_visible()
    expect(page.locator("#phase")).to_contain_text("Listening")


def test_typed_chat_without_a_call(browser):
    config.DB_PATH = Path(tempfile.mkdtemp()) / "e.db"
    page = new_page(browser)
    page.fill("#text", "Book a meeting on November 25 at 2 PM"); page.press("#text", "Enter")
    expect(page.locator("#fDate")).to_have_value("2026-11-25")
    expect(page.locator("#fTime")).to_have_value("14:00")
    expect(page.locator(".msg.user .body")).to_contain_text("2 PM")
    page.fill("#text", "my email is sara@gmail.com"); page.press("#text", "Enter")
    expect(page.locator("#fEmail")).to_have_value("sara@gmail.com")
    expect(page.locator("#msg-email")).to_contain_text("verified")                  # typed = visible = verified
    expect(page.locator("#latency")).to_contain_text("Last reply")
    assert not page.errors, page.errors


def test_unavailable_slot_offers_alternatives(browser):
    g = T.FakeGoogle(busy=[T.busy_slot("2026-11-25", "11:00")]).install()
    try:
        config.DB_PATH = Path(tempfile.mkdtemp()) / "f.db"
        page = new_page(browser)
        page.fill("#fDate", "2026-11-25"); page.press("#fDate", "Tab")
        expect(page.locator("#fTime option[value='11:00']")).to_have_count(0)
        page.fill("#text", "make it 11 AM please"); page.press("#text", "Enter")
        expect(page.locator("#slotLine")).to_contain_text("not available")
        expect(page.locator("#alts .pill-btn").first).to_be_visible()
        page.locator("#alts .pill-btn").first.click()                               # one click picks an open time
        expect(page.locator("#slotLine")).not_to_contain_text("not available")
    finally:
        g.uninstall()


def test_google_down_is_reported_not_hidden(browser):
    g = T.FakeGoogle(fail_create=True).install()
    try:
        config.DB_PATH = Path(tempfile.mkdtemp()) / "g.db"
        page = new_page(browser)
        page.fill("#fName", "Mia Stone"); page.press("#fName", "Enter")
        page.fill("#fEmail", "mia@gmail.com"); page.press("#fEmail", "Enter")
        page.fill("#fDate", "2026-11-25"); page.press("#fDate", "Tab")
        expect(page.locator("#fTime option[value='11:00']")).to_have_count(1)
        page.select_option("#fTime", "11:00")
        page.fill("#fPurpose", "Branding"); page.press("#fPurpose", "Tab")
        expect(page.locator("#reviewBtn")).to_be_enabled()
        page.click("#reviewBtn"); page.click("#dlgConfirm")
        expect(page.locator(".note.bad")).to_contain_text("Not booked")
        expect(page.locator("#success")).to_be_hidden()
        assert T.bookings() == []
    finally:
        g.uninstall()


def test_looks_right_on_phone_and_dark(browser):
    config.DB_PATH = Path(tempfile.mkdtemp()) / "h.db"
    page = new_page(browser, 390, 844, "dark")
    page.fill("#text", "Book a meeting on November 25 at 11 AM"); page.press("#text", "Enter")
    expect(page.locator("#fDate")).to_have_value("2026-11-25")
    width = page.evaluate("document.documentElement.scrollWidth")
    assert width <= 392, f"horizontal scroll on a phone: {width}px"
    page.screenshot(path=str(SHOTS / "phone-dark.png"), full_page=True)
    page2 = new_page(browser, 1440, 900, "dark")
    page2.fill("#text", "What services do you offer?"); page2.press("#text", "Enter")
    expect(page2.locator(".msg.agent")).to_have_count(1)
    page2.screenshot(path=str(SHOTS / "desktop-dark.png"))


def main():
    server = start_server()
    expect.set_options(timeout=15000)       # generous: CI boxes and old laptops are slow with Chrome + server together
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
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
                print(f"  FAIL  {name}: {str(e)[:300]}")
                try:                                    # what the user (and the server) saw when it failed
                    pg = LAST["page"]
                    print("        chat:", " | ".join(pg.inner_text("#log").splitlines())[-700:])
                    print("        phase:", pg.inner_text("#phase"), "| fields:", pg.evaluate(
                        "JSON.stringify(window.__app.S.state && window.__app.S.state.values)"))
                    from app.agent import SESSIONS
                    for sid, sess in SESSIONS.items():
                        print("        server", sid, "pending:", bool(sess.pending_key), "awaiting:", sess.awaiting,
                              "booked:", bool(sess.booking.booked), "email_status:", sess.booking.email_status)
                except Exception:
                    pass
        browser.close()
    server.should_exit = True
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
