"""
The "I'm lost" button: why the lecture stopped making sense.

It answers a different question from a summary, and the difference is the
point. A summary says what was said — but the student heard it. This says
which terms landed on them in the last few minutes and what ties those
together, which is something no transcript alone can work out. It needs the
course memory to know which words were new to this student on this course.
"""

import time

from fastapi.testclient import TestClient
from lost import recent_transcript


def press(srv, **kw):
    """Press the button. No lifespan needed — this endpoint starts no worker."""
    body = {"course": "c", "subject": "C programming", "turns": ["…"], **kw}
    r = TestClient(srv.app).post("/api/lost", json=body)
    r.raise_for_status()
    return r.json()


def introduce(srv, *terms, ago=0.0):
    """Put terms into the record as if they had been met `ago` seconds back."""
    now = time.monotonic()
    srv._introduced.setdefault("c", []).extend(
        {"term": t, "definition": f"what {t} means", "at": now - ago} for t in terms
    )


# --- what comes back ---------------------------------------------------------

def test_terms_met_inside_the_window_are_the_answer(srv, explainer):
    explainer()
    introduce(srv, "malloc", "heap", ago=30)

    body = press(srv, window_s=180)

    assert [t["term"] for t in body["terms"]] == ["malloc", "heap"]
    assert body["why"] == "Here is the thread."


def test_terms_met_before_the_window_are_left_out(srv, explainer):
    """The student is lost now, not twenty minutes ago."""
    explainer()
    introduce(srv, "printf", ago=600)
    introduce(srv, "malloc", ago=20)

    body = press(srv, window_s=180)

    assert [t["term"] for t in body["terms"]] == ["malloc"]


def test_a_stretch_with_nothing_new_says_so_without_asking_the_model(srv, explainer):
    """
    Inventing an explanation here would be worse than admitting there is none:
    if nothing new arrived, the gap is somewhere we are not looking, and the
    student deserves to know that rather than a paragraph of filler.
    """
    calls = explainer()
    introduce(srv, "printf", ago=600)

    body = press(srv, window_s=180)

    assert body["terms"] == []
    assert body["why"] == ""
    assert body["note"]
    assert calls == [], "no terms, no question"


# --- what the model is asked -------------------------------------------------

def test_the_model_gets_both_the_words_and_the_terms(srv, explainer):
    calls = explainer()
    introduce(srv, "malloc", ago=10)

    press(srv, turns=["we allocate with malloc"], window_s=180)

    prompt = calls[0]
    assert "we allocate with malloc" in prompt
    assert "malloc — what malloc means" in prompt
    assert "C programming" in prompt


def test_a_failing_gateway_still_returns_the_terms(srv, explainer):
    """The term list is the finding. The paragraph is the polish."""
    explainer(fails=True)
    introduce(srv, "malloc", "heap", ago=10)

    body = press(srv, window_s=180)

    assert [t["term"] for t in body["terms"]] == ["malloc", "heap"]
    assert body["why"] == ""
    assert body["note"]


# --- the number for the report -----------------------------------------------

def test_density_counts_the_terms_per_minute_of_the_window(srv, explainer):
    explainer()
    introduce(srv, "malloc", "heap", "pointer", ago=10)

    body = press(srv, window_s=120)

    assert body["density"]["in_window"] == 3
    assert body["density"]["per_minute"] == 1.5


def test_too_little_material_reports_no_baseline_rather_than_a_wrong_one(srv, explainer):
    """A rate over ten seconds of lecture is noise wearing a number's clothes."""
    explainer()
    introduce(srv, "malloc", ago=5)

    body = press(srv, window_s=180)

    assert body["density"]["session_per_minute"] is None


def test_the_baseline_is_the_lecture_s_own_average(srv, explainer):
    explainer()
    introduce(srv, "printf", "scanf", ago=120)     # two terms, two minutes ago
    introduce(srv, "malloc", "heap", ago=10)

    body = press(srv, window_s=60)

    assert body["density"]["in_window"] == 2
    assert body["density"]["per_minute"] == 2.0
    assert body["density"]["session_per_minute"] == 2.0     # 4 terms over 2 minutes


# --- the transcript we send --------------------------------------------------

def test_a_long_transcript_is_trimmed_from_the_front(srv):
    """If something has to go it is the oldest part: the confusion is at the end."""
    turns = ["x" * 500 for _ in range(10)] + ["the last thing said"]

    text = recent_transcript(turns)

    assert text.endswith("the last thing said")
    assert len(text) <= 2001
    assert text.startswith("…")


def test_blank_turns_do_not_reach_the_model(srv):
    assert recent_transcript(["", "  ", "real speech"]) == "real speech"


def test_markdown_from_the_model_does_not_reach_the_student():
    """On CS50 the panel showed **notation** with the asterisks."""
    from lost import plain
    assert plain("the **notation** called `hexadecimal`,\n  base 16") == \
        "the notation called hexadecimal, base 16"
