"""
The server: what the browser gets back, and when.

The rule the whole design turns on is that /api/observe never touches the
network. It answers from what is already known; anything unknown goes into a
queue and a background worker asks the model in batches. Judging inside the
request put the last definition of a 35-second test 36 seconds behind the
speaker — which is the exact failure the product exists to prevent.
"""

import pytest
from fastapi.testclient import TestClient


def client(srv):
    return TestClient(srv.app)


def observe(c, text, course="c", subject="C programming"):
    r = c.post("/api/observe", json={"course": course, "subject": subject, "text": text})
    r.raise_for_status()
    return r.json()


# --- the fast path -----------------------------------------------------------

def test_an_unknown_word_is_queued_rather_than_judged_in_the_request(srv, model):
    fake = model()
    with client(srv) as c:
        body = observe(c, "we call malloc here")

    assert "malloc" in body["pending"]
    assert body["new_terms"] == {}
    assert fake.asked == [], "the request must not wait for the model"


def test_a_word_judged_on_an_earlier_lecture_comes_back_immediately(srv, model):
    model()
    with client(srv) as c:
        srv.judge_for("c", "C programming").cache["malloc"] = "allocates memory"

        body = observe(c, "we call malloc here")

    assert body["new_terms"] == {"malloc": "allocates memory"}
    assert body["pending"] == []


def test_a_word_already_judged_ordinary_is_dropped_without_asking_again(srv, model):
    """
    "returns" gets past the cheap word-list filter but the model has already
    called it ordinary. It must not cost a second question, and it must not
    linger in the memory where it would go out as a keyterm.
    """
    model()
    with client(srv) as c:
        srv.judge_for("c", "C programming").cache["returns"] = None

        body = observe(c, "malloc returns a pointer")

    assert "returns" not in body["pending"]
    assert srv.glossary_for("c").is_new("returns") is True


def test_the_same_word_twice_in_one_breath_is_queued_once(srv, model):
    model()
    with client(srv) as c:
        body = observe(c, "malloc and malloc again with malloc")

    assert body["pending"].count("malloc") == 1


# --- the worker --------------------------------------------------------------

def test_a_word_the_model_never_mentioned_stays_in_the_course_memory(srv, model):
    """
    Today's bug. terms.py deliberately leaves an unanswered word uncached so it
    gets asked again — and flush() deleted it from the memory anyway, so there
    was nothing left to ask about. This is how getchar and putchar vanished.
    """
    model(omits=["putchar"])
    with client(srv) as c:
        observe(c, "the stdio header gives you putchar")
        srv.flush("c")

    assert srv.glossary_for("c").is_new("putchar") is False


def test_a_word_the_model_rejected_is_removed_from_the_course_memory(srv, model):
    """Otherwise "finish" and "raise" go out as keyterms and hurt recognition."""
    model(rejects=["returns"])
    with client(srv) as c:
        observe(c, "malloc returns a pointer")
        srv.flush("c")

    g = srv.glossary_for("c")
    assert g.is_new("returns") is True, "the model called it ordinary"
    assert g.is_new("malloc") is False, "and the real term stayed"


def test_the_worker_batches_several_turns_into_one_question(srv, model):
    """One call per turn outran the gateway's rate limit within half a minute."""
    fake = model()
    with client(srv) as c:
        observe(c, "we call malloc here")
        observe(c, "the heap grows upward")
        observe(c, "a pointer holds an address")
        srv.flush("c")

    assert len(fake.asked) == 1
    assert {"malloc", "heap", "pointer"} <= set(fake.asked[0])


def test_verdicts_reach_the_browser_through_updates(srv, model):
    model()
    with client(srv) as c:
        observe(c, "we call malloc here")
        srv.flush("c")

        body = c.get("/api/updates", params={"course": "c"}).json()

    assert [t["term"] for t in body["terms"]] == ["malloc"]
    assert body["terms"][0]["definition"]
    assert body["waiting"] == 0


def test_each_verdict_is_handed_over_exactly_once(srv, model):
    """
    /api/updates drains rather than accumulates, so the browser never has to
    work out which verdicts it has already drawn.
    """
    model()
    with client(srv) as c:
        observe(c, "we call malloc here")
        srv.flush("c")

        first = c.get("/api/updates", params={"course": "c"}).json()
        second = c.get("/api/updates", params={"course": "c"}).json()

    assert len(first["terms"]) == 1
    assert second["terms"] == []


def test_a_blip_costs_a_retry_not_the_term(srv, model):
    """
    A word already in the course memory never comes back as "first time", so
    nothing re-queues it by itself. Dropping it on one failed call meant a
    thirty-second network blip permanently blanked every term spoken during
    it. So a failure puts the batch back.
    """
    model(fails=True)
    with client(srv) as c:
        observe(c, "we call malloc here")
        srv.flush("c")

        body = c.get("/api/updates", params={"course": "c"}).json()

    assert srv.glossary_for("c").is_new("malloc") is False, "the term survives"
    assert body["terms"] == [], "nothing is reported yet — it will be asked again"
    assert "malloc" in srv._pending["c"], "and it is back in the queue"


def test_the_retry_gives_up_rather_than_looping_all_lecture(srv, model):
    """
    An unreachable gateway would otherwise have the worker asking about the
    same twenty words every two seconds until the lecture ends, never reaching
    the ones spoken since.
    """
    fake = model(fails=True)
    with client(srv) as c:
        observe(c, "we call malloc here")
        for _ in range(srv.MAX_ATTEMPTS + 2):
            srv.flush("c")

        body = c.get("/api/updates", params={"course": "c"}).json()

    assert len(fake.asked) == srv.MAX_ATTEMPTS
    assert "malloc" not in srv._pending.get("c", {})
    assert body["terms"][0]["term"] == "malloc"
    assert body["terms"][0]["note"], "the browser is told why there is no definition"
    assert srv.glossary_for("c").is_new("malloc") is False, "the term still survives"


def test_a_word_with_no_verdict_is_asked_again(srv, model):
    """
    The reply omitted it. terms.py leaves it unjudged on purpose; the queue has
    to be the thing that brings it back, because the glossary never will.
    """
    model(omits=["putchar"])
    with client(srv) as c:
        observe(c, "the stdio header gives you putchar")
        srv.flush("c")

    assert "putchar" in srv._pending["c"]
    assert srv.glossary_for("c").is_new("putchar") is False


# --- the shipped course ------------------------------------------------------

def test_the_seed_fills_an_empty_disk(srv, model):
    srv.SEED_DIR.mkdir(parents=True)
    (srv.SEED_DIR / "c.json").write_text(
        '{"course": "c", "terms": {"scanf": {"surface": "scanf", "count": 3, '
        '"first_seen": "2026-09-10"}}}'
    )

    srv.plant_seed()

    assert srv.glossary_for("c").is_new("scanf") is False


def test_the_seed_never_overwrites_what_this_instance_learned(srv, model):
    """
    plant_seed runs on every boot, so it has to be a floor under the memory
    rather than a reset of it.
    """
    srv.SEED_DIR.mkdir(parents=True)
    (srv.SEED_DIR / "c.json").write_text(
        '{"course": "c", "terms": {"scanf": {"surface": "scanf", "count": 3, '
        '"first_seen": "2026-09-10"}}}'
    )
    srv.plant_seed()

    g = srv.glossary_for("c")
    g.observe("malloc")
    g.save()
    srv._courses.clear()

    srv.plant_seed()                       # a restart that did not wipe the disk

    assert srv.glossary_for("c").is_new("malloc") is False


def test_a_missing_seed_folder_is_not_an_error(srv):
    srv.plant_seed()


# --- the token ---------------------------------------------------------------

def test_the_token_endpoint_refuses_without_a_key(srv, monkeypatch):
    monkeypatch.setattr(srv, "API_KEY", None)
    with client(srv) as c:
        assert c.get("/api/token").status_code == 500


# --- one course must not leak into another -----------------------------------

def test_a_term_learned_on_one_course_is_still_new_on_another(srv, model):
    """
    The queues, the glossaries and the caches are all keyed by course. If any
    of them were shared, a student on their first algorithms lecture would be
    told nothing is new because someone else's C course had heard the word.
    """
    model()
    with client(srv) as c:
        observe(c, "we call malloc here", course="c")
        srv.flush("c")

        body = observe(c, "we call malloc here", course="algorithms")

    assert "malloc" in body["pending"], "new to this course, whatever the other one knows"


def test_verdicts_are_delivered_to_the_course_that_asked(srv, model):
    model()
    with client(srv) as c:
        observe(c, "we call malloc here", course="c")
        observe(c, "recursion needs a base case", course="algorithms")
        srv.flush("c")
        srv.flush("algorithms")

        for_c = c.get("/api/updates", params={"course": "c"}).json()
        for_algo = c.get("/api/updates", params={"course": "algorithms"}).json()

    # Set membership, not equality: the cheap filter deliberately lets extra
    # words through, and pinning the exact list would make this test fail for
    # a reason that has nothing to do with what it is checking.
    in_c = {t["term"] for t in for_c["terms"]}
    in_algo = {t["term"] for t in for_algo["terms"]}

    assert "malloc" in in_c and "malloc" not in in_algo
    assert "recursion" in in_algo and "recursion" not in in_c


def test_each_course_is_judged_in_its_own_subject(srv, model):
    """
    The subject goes into the prompt: "stack" means one thing in a C lecture
    and another in an algorithms one, and the judge is told which it is.
    """
    model()
    with client(srv) as c:
        observe(c, "the stack grows downward", course="c", subject="C programming")
        observe(c, "a stack has push and pop", course="algorithms",
                subject="Data structures")

    assert srv.judge_for("c").subject == "C programming"
    assert srv.judge_for("algorithms").subject == "Data structures"

# --- a course name is a filename ---------------------------------------------

@pytest.mark.parametrize("name", [
    "../../etc/passwd", "a/b", "..", "", "x" * 200, "has space", "nul\x00byte",
])
def test_a_course_name_that_is_not_a_filename_is_refused(srv, name):
    """
    The name goes straight into a path, and /api/observe writes a file on every
    single turn. Refused rather than sanitised: quietly rewriting the name
    would open a second, empty course and say nothing.
    """
    r = TestClient(srv.app).post("/api/observe", json={"course": name, "text": "malloc"})
    assert r.status_code == 400


def test_ordinary_course_names_still_work(srv, model):
    model()
    for name in ["c-programming", "CSE305", "algorithms_2", "intro.to.ai"]:
        r = TestClient(srv.app).post("/api/observe", json={"course": name, "text": "malloc"})
        assert r.status_code == 200, name


def test_an_absurd_window_is_clamped_rather_than_honoured(srv, explainer):
    explainer()
    r = TestClient(srv.app).post("/api/lost",
                                 json={"course": "c", "window_s": 10 ** 9}).json()
    assert r["density"]["per_minute"] == 0.0     # no terms, and no division blow-up


# --- everyday English is decided by rule, not by the model --------------------

def test_an_everyday_word_never_reaches_the_judge(srv, model):
    """
    On a real lecture the model accepted "super", "actual" and "store" and
    rejected "function" and "return": wrong in both directions in this band.
    So the band is not asked about at all, and costs no gateway call.
    """
    fake = model()
    with client(srv) as c:
        body = observe(c, "this is super obvious, malloc gives you memory")
        srv.flush("c")

    assert "super" not in body["pending"]
    assert "malloc" in body["pending"], "rare words still go to the judge"
    assert all("super" not in batch for batch in fake.asked)
    assert srv.glossary_for("c").is_new("super") is True, "and it leaves the memory"


def test_a_word_the_course_declares_is_judged_after_all(srv, model):
    srv.SEED_DIR.mkdir(parents=True, exist_ok=True)
    (srv.SEED_DIR / "c.terms.txt").write_text("# comment\nfree   # the function\n")
    model()
    with client(srv) as c:
        body = observe(c, "you must free the block")

    assert "free" in body["pending"]


def test_a_declared_word_reaches_the_judge_untagged(srv, model, monkeypatch):
    """Tagging "free" as everyday would get it rejected — the tag exists for that."""
    srv.SEED_DIR.mkdir(parents=True, exist_ok=True)
    (srv.SEED_DIR / "c.terms.txt").write_text("free\n")
    import terms
    sent = []
    monkeypatch.setattr(terms, "ask_gateway",
                        lambda prompt, key, max_tokens=900: sent.append(prompt) or "free = releases memory")
    j = srv.judge_for("c", "C programming")
    j.api_key = "k"
    j._ask(["free"])

    assert "free (everyday)" not in sent[0]
    assert "\nfree" in sent[0]


def test_a_course_without_a_terms_file_declares_nothing(srv):
    assert srv.declared_for("c") == frozenset()


# --- the density line in the first minutes ------------------------------------

def test_a_young_session_is_not_divided_by_a_full_window(srv, explainer):
    """
    Two minutes into a lecture, a three-minute window used to divide by three,
    so the burst the student pressed the button for read as calmer than
    average.
    """
    import time
    explainer()
    now = time.monotonic()
    srv._introduced["c"] = [{"term": t, "definition": "", "at": now - 30}
                            for t in ("malloc", "heap", "pointer", "free")]

    body = TestClient(srv.app).post("/api/lost", json={
        "course": "c", "turns": ["…"], "window_s": 180, "session_s": 120}).json()

    assert body["density"]["window_minutes"] == 2.0
    assert body["density"]["per_minute"] == 2.0          # 4 terms over 2 minutes, not 3
    assert body["density"]["session_per_minute"] == 2.0


def test_an_earlier_sitting_does_not_count_towards_this_one(srv, explainer):
    import time
    explainer()
    now = time.monotonic()
    srv._introduced["c"] = [
        {"term": "printf", "definition": "", "at": now - 600},   # last session
        {"term": "malloc", "definition": "", "at": now - 20},
    ]

    body = TestClient(srv.app).post("/api/lost", json={
        "course": "c", "turns": ["…"], "window_s": 180, "session_s": 60}).json()

    assert [t["term"] for t in body["terms"]] == ["malloc"]
    assert body["density"]["session_per_minute"] == 1.0


# --- the sentence travels with the word --------------------------------------

def test_a_queued_word_reaches_the_judge_with_its_sentence(srv, model):
    fake = model()
    with client(srv) as c:
        observe(c, "So there's this dichotomy. Then we call malloc on it.")
        srv.flush("c")

    assert fake.contexts["dichotomy"] == "So there's this dichotomy."
    assert fake.contexts["malloc"] == "Then we call malloc on it."


def test_the_first_sentence_a_word_was_heard_in_is_the_one_judged(srv, model):
    fake = model()
    with client(srv) as c:
        observe(c, "We call malloc here.")
        observe(c, "Again malloc, said differently.")
        srv.flush("c")

    assert fake.contexts["malloc"] == "We call malloc here."


def test_a_retry_keeps_the_sentence(srv, model):
    """A retry judged on the bare word is the old judge, the one that let noise through."""
    model(fails=True)
    with client(srv) as c:
        observe(c, "We call malloc here.")
        srv.flush("c")

    assert srv._pending["c"]["malloc"] == "We call malloc here."
