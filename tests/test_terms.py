"""
The judge: what happens to a word between being heard and being explained.

Every test here is named after a bug that actually shipped. Three of them are
the same bug in three places — "judged and rejected" and "never judged" look
identical from the outside, and every time we let them collapse into one, the
product silently lost terms.
"""

import json

import pytest
import terms as terms_module
from terms import PROMPT_FINGERPRINT, TermJudge, mark, parse_reply


# --- the recurring bug -------------------------------------------------------

def test_a_word_the_reply_omits_is_left_unjudged(tmp_path, model):
    """
    The model answered about two words and ignored the third. The third is not
    rejected — nobody judged it. Caching it as rejected buries it forever,
    because a cached word is never asked again.
    """
    model(omits=["putchar"])
    j = TermJudge("C programming", cache_dir=str(tmp_path), course="c")
    j.api_key = "k"

    kept = j.judge(["scanf", "putchar"])

    assert "scanf" in kept
    assert "putchar" not in kept              # no definition to show yet
    assert "putchar" not in j.cache           # and no verdict recorded either


def test_an_omitted_word_is_asked_again_next_time(tmp_path, model):
    fake = model(omits=["putchar"])
    j = TermJudge("C programming", cache_dir=str(tmp_path), course="c")
    j.api_key = "k"
    j.judge(["scanf", "putchar"])

    fake2 = model()                            # this time the model answers
    j2 = TermJudge("C programming", cache_dir=str(tmp_path), course="c")
    j2.api_key = "k"
    kept = j2.judge(["scanf", "putchar"])

    assert fake2.asked == [["putchar"]], "only the unjudged word should be re-asked"
    assert "putchar" in kept


def test_a_rejection_is_remembered_and_never_asked_twice(tmp_path, model):
    fake = model(rejects=["please"])
    j = TermJudge("C programming", cache_dir=str(tmp_path), course="c")
    j.api_key = "k"
    j.judge(["please"])
    assert j.cache["please"] is None

    fake2 = model()
    j2 = TermJudge("C programming", cache_dir=str(tmp_path), course="c")
    j2.api_key = "k"
    assert j2.judge(["please"]) == {}
    assert fake2.asked == [], "a settled verdict costs nothing to reuse"


def test_cached_reports_three_outcomes_not_two(judge):
    judge.cache = {"scanf": "reads formatted input", "please": None}

    defined, rejected, unknown = judge.cached(["scanf", "please", "malloc"])

    assert defined == {"scanf": "reads formatted input"}
    assert rejected == ["please"]
    assert unknown == ["malloc"]


# --- failure that is not rejection -------------------------------------------

def test_an_unreachable_model_raises_instead_of_returning_nothing(tmp_path, model):
    """
    "Could not ask" and "asked, nothing qualified" are both an empty result.
    The caller deletes rejected words from the course memory, so confusing the
    two wiped the whole glossary the first time this happened.
    """
    model(fails=True)
    j = TermJudge("C programming", cache_dir=str(tmp_path), course="c")
    j.api_key = "k"

    with pytest.raises(RuntimeError):
        j.judge(["scanf", "malloc"])


# --- the cache file ----------------------------------------------------------

def test_cache_is_named_after_the_course_not_the_subject(tmp_path, judge):
    judge.cache = {"scanf": "reads input"}
    judge._save()

    assert (tmp_path / "c-programming.definitions.json").exists()
    assert not (tmp_path / "C programming.definitions.json").exists()


def test_cache_written_under_a_different_prompt_is_discarded(tmp_path, judge, monkeypatch):
    judge.cache = {"program": "a sequence of instructions"}
    judge._save()

    monkeypatch.setattr(terms_module, "PROMPT_FINGERPRINT", "0000deadbeef")
    reopened = TermJudge("C programming", cache_dir=str(tmp_path), course="c-programming")

    assert reopened.cache == {}, "old verdicts must not survive a new question"


def test_cache_survives_when_the_prompt_is_unchanged(tmp_path, judge):
    judge.cache = {"scanf": "reads formatted input"}
    judge._save()

    reopened = TermJudge("C programming", cache_dir=str(tmp_path), course="c-programming")
    assert reopened.cache == {"scanf": "reads formatted input"}

    stored = json.loads((tmp_path / "c-programming.definitions.json").read_text())
    assert stored["prompt"] == PROMPT_FINGERPRINT


# --- reading what the model said ---------------------------------------------

@pytest.mark.parametrize("line", [
    "scanf = reads formatted input",
    "scanf — reads formatted input",
    "scanf: reads formatted input",
    "scanf | reads formatted input",
    "1. scanf = reads formatted input",
    "- scanf = reads formatted input",
    "`scanf` = reads formatted input",
    "**scanf** = reads formatted input",
])
def test_parse_reply_survives_the_shapes_models_actually_use(line):
    assert parse_reply(line, ["scanf"]) == {"scanf": "reads formatted input"}


@pytest.mark.parametrize("answer", ["no", "none", "n/a", "not a term", "-"])
def test_parse_reply_reads_every_way_of_saying_no(answer):
    assert parse_reply(f"please = {answer}", ["please"]) == {"please": None}


def test_parse_reply_strips_our_own_everyday_marker():
    """We add "(everyday)" to the question; the model echoes it in the answer."""
    reply = "class (everyday) = a template for objects\nprogram (everyday) = no"
    assert parse_reply(reply, ["class", "program"]) == {
        "class": "a template for objects",
        "program": None,
    }


def test_parse_reply_ignores_lines_about_words_we_did_not_ask_about():
    reply = "Here are the definitions:\nmalloc = allocates memory\nscanf = reads input"
    assert parse_reply(reply, ["scanf"]) == {"scanf": "reads input"}


def test_a_definition_too_long_to_read_at_a_glance_is_cut():
    long = "x" * 200
    out = parse_reply(f"scanf = {long}", ["scanf"])["scanf"]
    assert len(out) <= terms_module.MAX_DEFINITION_CHARS
    assert out.endswith("…")


# --- the frequency marking ---------------------------------------------------

def test_everyday_words_are_marked_and_rare_ones_are_not():
    assert mark("program") == "program (everyday)"
    assert mark("scanf") == "scanf"


def test_marking_is_case_insensitive():
    assert mark("Program") == "Program (everyday)"