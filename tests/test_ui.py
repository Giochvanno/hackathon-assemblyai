"""
The page, in a real browser.

Everything else in this suite tests the server. This file drives web/index.html
in headless Chromium against a real, local server — with the model replaced,
so nothing leaves the machine — and checks two things: that the page does
what it says, and that the layout holds when it fills up. A glossary that
works with five terms and pushes the buttons off screen at sixty is broken.

Needs Playwright and its Chromium:

    pip install playwright
    python -m playwright install chromium

Without them these tests are skipped, not failed: the rest of the suite does
not depend on a browser.
"""

import socket
import threading
import time

import pytest

playwright_api = pytest.importorskip("playwright.sync_api")

import terms as terms_module            # noqa: E402


# --- a real server on a free port --------------------------------------------

def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def base_url(tmp_path_factory):
    import uvicorn
    import server

    data = tmp_path_factory.mktemp("ui-data")
    saved = (server.DATA_DIR, server.SEED_DIR, terms_module.TermJudge._ask)
    server.DATA_DIR = data / "glossary"
    server.SEED_DIR = data / "seed"
    # The judge answers from here: every word is a term. No gateway, no key.
    terms_module.TermJudge._ask = lambda self, words, contexts=None: {
        w.lower(): "a term in this subject" for w in words
    }

    port = _free_port()
    srv = uvicorn.Server(uvicorn.Config(server.app, host="127.0.0.1", port=port,
                                        log_level="warning"))
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    deadline = time.time() + 10
    while not srv.started and time.time() < deadline:
        time.sleep(0.05)
    assert srv.started, "the test server did not start"

    yield f"http://127.0.0.1:{port}"

    srv.should_exit = True
    thread.join(timeout=5)
    server.DATA_DIR, server.SEED_DIR, terms_module.TermJudge._ask = saved


@pytest.fixture(scope="module")
def browser():
    with playwright_api.sync_playwright() as p:
        try:
            b = p.chromium.launch(args=[
                "--use-fake-ui-for-media-stream",
                "--use-fake-device-for-media-stream",
            ])
        except Exception as exc:                       # no Chromium installed
            pytest.skip(f"Chromium not available: {exc}")
        yield b
        b.close()


@pytest.fixture
def page(browser, base_url):
    """A fresh page, with every JavaScript error collected."""
    ctx = browser.new_context(viewport={"width": 1440, "height": 900})
    pg = ctx.new_page()
    pg.errors = []
    pg.on("pageerror", lambda e: pg.errors.append(str(e)))
    # The token is the one call that would reach AssemblyAI. Refuse it here.
    pg.route("**/api/token*", lambda route: route.fulfill(status=500, body="no token in tests"))
    pg.goto(base_url)
    yield pg
    assert pg.errors == [], f"JavaScript errors: {pg.errors}"
    ctx.close()


def box(pg, selector):
    return pg.eval_on_selector(selector, "e => { const r = e.getBoundingClientRect();"
                                         " return {x: r.x, y: r.y, w: r.width, h: r.height,"
                                         " right: r.right, bottom: r.bottom}; }")


# --- before anything happens --------------------------------------------------

def test_the_page_explains_itself_before_the_lecture_starts(page):
    assert page.is_visible("#empty")
    assert "Start listening" in page.inner_text("#empty")
    assert page.get_attribute("#state", "data-state") == "idle"


def test_the_header_no_longer_shows_the_course_id(page):
    assert page.query_selector("#course") is None
    assert "c-programming" not in page.inner_text("header")


# --- the glossary -------------------------------------------------------------

def test_a_new_term_arrives_closed_and_opens_on_click(page):
    page.evaluate("addEntry('malloc', 'allocates memory on the heap', true)")

    card = page.locator("#glossary .card").first
    assert card.locator("summary").inner_text() == "malloc"
    assert not card.locator("p").is_visible(), "the definition waits to be asked for"

    card.locator("summary").click()
    assert card.locator("p").inner_text() == "allocates memory on the heap"
    assert page.inner_text("#new-count") == "1"
    assert not page.is_visible("#new-empty")


def test_the_newest_term_is_on_top(page):
    page.evaluate("addEntry('malloc', 'a', true); addEntry('free', 'b', true)")
    names = page.locator("#glossary .card summary").all_inner_texts()
    assert names == ["free", "malloc"]


def test_a_term_the_course_already_knew_goes_to_the_folded_list(page):
    page.evaluate("addEntry('printf', '', false)")
    assert page.locator("#glossary .card").count() == 0
    assert page.inner_text("#known-count") == "1"
    assert not page.is_visible("#known-list span"), "folded until opened"
    page.click("#known-box summary")
    assert page.inner_text("#known-list") == "printf"


def test_the_same_term_is_listed_once(page):
    page.evaluate("addEntry('Malloc', 'a', true); addEntry('malloc', 'a', true);"
                  "addEntry('malloc', '', false)")
    assert page.locator("#glossary .card").count() == 1
    assert page.inner_text("#known-count") == "0"


def test_a_term_without_a_definition_says_so(page):
    page.evaluate("addEntry('scanf', '', true)")
    page.click("#glossary .card summary")
    text = page.locator("#glossary .card p")
    assert "No definition yet" in text.inner_text()
    assert "none" in text.get_attribute("class")


def test_open_all_is_a_mode_that_later_cards_follow(page):
    page.evaluate("addEntry('malloc', 'a', true)")
    page.click("#expand")
    assert page.inner_text("#expand") == "close all"
    page.evaluate("addEntry('free', 'b', true)")
    opened = page.eval_on_selector_all("#glossary .card", "cs => cs.map(c => c.open)")
    assert opened == [True, True], "a card arriving in open-all mode opens too"

    page.click("#expand")
    opened = page.eval_on_selector_all("#glossary .card", "cs => cs.map(c => c.open)")
    assert opened == [False, False]
    assert page.inner_text("#expand") == "open all"


# --- status and log -----------------------------------------------------------

@pytest.mark.parametrize("message, state", [
    ("listening", "live"),
    ("reconnecting in 4s…", "warn"),
    ("disconnected", "warn"),
    ("could not connect — press Start to try again", "err"),
    ("error — see log", "err"),
    ("closed (1000) · 2962 frames sent", "idle"),
    ("requesting microphone…", "idle"),
])
def test_the_status_dot_follows_the_message(page, message, state):
    page.evaluate("m => say(m)", message)
    assert page.inner_text("#status") == message
    assert page.get_attribute("#state", "data-state") == state


def test_the_log_is_hidden_until_asked_for_and_an_error_flags_the_button(page):
    assert not page.is_visible("#log")
    page.evaluate("log('all fine', 'ok')")
    assert "alert" not in page.get_attribute("#log-toggle", "class")
    page.evaluate("log('socket error', 'err')")
    assert "alert" in page.get_attribute("#log-toggle", "class")

    page.click("#log-toggle")
    assert page.is_visible("#log")
    assert "socket error" in page.inner_text("#log")
    assert "alert" not in page.get_attribute("#log-toggle", "class")
    assert page.inner_text("#log-toggle") == "hide log"


# --- the real paths, against the real server -----------------------------------

def test_a_finished_turn_reaches_the_server_and_its_terms_come_back(page):
    page.evaluate("document.getElementById('empty')?.remove()")
    page.evaluate("finalise('Then we call malloc on the heap.')")
    assert page.locator(".turn").count() == 1
    # The server queued the words; the stand-in judge defines them within its
    # two-second cycle; the page collects them the way it does every 1.5 s.
    deadline = time.time() + 15
    names = []
    while "malloc" not in names and time.time() < deadline:
        page.evaluate("poll()")
        names = page.locator("#glossary .card summary").all_inner_texts()
        time.sleep(0.5)
    assert "malloc" in names
    assert page.locator(".turn .term.fresh").count() >= 1, "and it is highlighted in the text"


def test_a_failed_start_is_reported_and_leaves_the_page_usable(page):
    page.click("#go")
    page.wait_for_function("document.getElementById('state').dataset.state === 'err'",
                           timeout=10000)
    assert not page.is_visible("#empty"), "start clears the instructions"
    assert "could not" in page.inner_text("#status")
    page.click("#log-toggle")
    assert "/api/token 500" in page.inner_text("#log")
    assert page.inner_text("#go") == "Start listening", "the student can try again"


# --- the layout under load ----------------------------------------------------

# No hyphens or spaces: nothing for the browser to break the line at.
LONG = "__builtin_object_size_of_a_pointer_to_function_returning_struct"


def _fill(pg):
    pg.evaluate("document.getElementById('empty')?.remove()")
    pg.evaluate("""long => {
        for (let i = 0; i < 40; i++) addEntry('known' + i, '', false);
        for (let i = 0; i < 60; i++) addEntry('term' + i, 'a one-line definition of the term, at the limit of ninety chars ok', true);
        addEntry(long, 'a name longer than the sidebar is wide', true);
        for (let i = 0; i < 40; i++) {
          const el = document.createElement('div'); el.className = 'turn';
          el.textContent = 'The lecturer keeps talking about memory and pointers ' .repeat(6);
          transcript.append(el);
          if (i % 5 === 0) showGloss(el, 'term' + i, 'a one-line definition of the term');
        }
        say('could not connect — press Start to try again after checking the network');
        document.getElementById('lat').textContent = '615 ms';
        document.getElementById('count').textContent = '1234';
        showWhy({terms: Array.from({length: 12}, (_, i) => ({term: 'term' + i})),
                 density: {in_window: 12, window_minutes: 3, per_minute: 4, session_per_minute: 2.5},
                 why: 'An explanation of the thread. '.repeat(25)});
        document.getElementById('log-toggle').click();
        for (let i = 0; i < 50; i++) log('a diagnostic line number ' + i);
        document.querySelectorAll('#glossary .card').forEach(c => c.open = true);
    }""", LONG)


@pytest.mark.parametrize("width, height", [
    (1920, 1080), (1440, 900), (1366, 768), (1024, 768), (800, 700),
    (1024, 560),                  # a short window: the panels must give way
])
def test_the_layout_holds_with_everything_open(page, width, height):
    page.set_viewport_size({"width": width, "height": height})
    _fill(page)

    # Nothing pushes the page sideways.
    assert page.evaluate("document.documentElement.scrollWidth") <= width

    # Both buttons stay whole and on screen, clear of the status text.
    for sel in ("#lost", "#go"):
        b = box(page, sel)
        assert b["x"] >= 0 and b["right"] <= width - 330, f"{sel} pushed off"
    assert box(page, "#state")["right"] <= box(page, "#lost")["x"], "status runs under the buttons"
    assert box(page, "#state")["h"] <= 24, "a long status stays on one line"

    # The sidebar keeps its width however long a term's name is.
    aside = box(page, "aside")
    assert abs(aside["w"] - 330) < 1
    spill = page.eval_on_selector_all(
        "#glossary .card summary, #glossary .card p",
        "els => els.filter(e => e.scrollWidth > e.clientWidth + 1).map(e => e.textContent)")
    assert spill == [], f"text runs out of its card: {spill[:2]}"

    # The transcript keeps room to read even with the panel and the log open.
    assert box(page, "#transcript")["h"] >= 139

    # And nothing below the fold: the bottom bar is on screen.
    assert box(page, "#bar")["bottom"] <= height + 1


def test_on_a_phone_the_header_fits_and_the_sidebar_steps_aside(page):
    page.set_viewport_size({"width": 390, "height": 844})
    _fill(page)
    assert page.evaluate("document.documentElement.scrollWidth") <= 390
    for sel in ("#lost", "#go"):
        b = box(page, sel)
        assert b["x"] >= 0 and b["right"] <= 390, f"{sel} pushed off"
    assert not page.is_visible("aside")
    assert box(page, "#bar")["h"] <= 44, "the bottom bar stays one line (two lines are ~60 px)"


def test_on_a_phone_the_instructions_do_not_point_at_a_missing_sidebar(page):
    page.set_viewport_size({"width": 390, "height": 844})
    assert page.is_visible("#empty")
    assert "collects on the right" not in page.inner_text("#empty")


# --- the glossary on a phone ----------------------------------------------------

def test_the_glossary_button_is_for_phones_only(page):
    assert not page.is_visible("#gloss-btn"), "a desktop has the sidebar"
    page.set_viewport_size({"width": 390, "height": 844})
    assert page.is_visible("#gloss-btn")


def test_on_a_phone_the_glossary_opens_as_a_sheet_and_closes_three_ways(page):
    page.set_viewport_size({"width": 390, "height": 844})
    page.evaluate("addEntry('malloc', 'allocates memory on the heap', true);"
                  "addEntry('free', 'gives the memory back', true)")
    assert page.inner_text("#gloss-n") == "2", "the button counts today's terms"
    assert not page.is_visible("aside"), "closed until asked for"

    for close in ("#gloss-close", "#sheet-bg", "Escape"):
        page.evaluate("document.querySelectorAll('#glossary .card').forEach(c => c.open = false)")
        page.click("#gloss-btn")
        assert page.is_visible("aside")
        page.locator("#glossary .card summary").first.click()
        assert page.locator("#glossary .card p").first.is_visible()
        if close == "Escape":
            page.keyboard.press("Escape")
        else:
            page.click(close, position={"x": 20, "y": 20})
        assert not page.is_visible("aside"), f"{close} did not close the sheet"


def test_the_sheet_fits_the_phone_screen_when_full(page):
    page.set_viewport_size({"width": 390, "height": 844})
    _fill(page)
    page.evaluate("document.getElementById('why').classList.remove('open')")
    page.click("#gloss-btn")
    page.wait_for_timeout(300)            # let the slide-up finish before measuring
    sheet = box(page, "aside")
    assert sheet["x"] >= 0 and sheet["right"] <= 390
    assert sheet["bottom"] <= 844 + 1 and sheet["y"] > 0
    assert page.evaluate("document.documentElement.scrollWidth") <= 390
    spill = page.eval_on_selector_all(
        "#glossary .card summary, #glossary .card p",
        "els => els.filter(e => e.scrollWidth > e.clientWidth + 1).map(e => e.textContent)")
    assert spill == []
    # the sheet scrolls inside itself rather than stretching the page
    assert page.evaluate("document.querySelector('aside').scrollHeight >"
                         " document.querySelector('aside').clientHeight")


def test_on_a_phone_all_three_buttons_fit_before_and_during_a_lecture(page):
    page.set_viewport_size({"width": 390, "height": 844})
    for label in ("Start listening", "Stop"):
        page.evaluate("t => document.getElementById('go').textContent = t", label)
        for sel in ("#gloss-btn", "#lost", "#go"):
            b = box(page, sel)
            assert b["x"] >= 0 and b["right"] <= 390, f"{sel} off screen with '{label}'"
            assert b["h"] <= 44, f"{sel} wrapped onto two lines with '{label}'"


def test_the_settled_terms_are_explained(page):
    page.click("#known-box summary")
    assert "heard at least twice" in page.inner_text("#known-box")
