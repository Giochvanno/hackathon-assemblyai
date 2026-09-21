"""
The course memory: the one idea the whole project rests on.

A word is new when this course has not used it before. That has nothing to do
with whether the model transcribed it correctly — a perfectly recognised word
can still be the first time the student has met it.
"""

from glossary import CourseGlossary, is_termlike


def course(tmp_path, name="c-programming"):
    return CourseGlossary(name, directory=str(tmp_path))


# --- the core question -------------------------------------------------------

def test_first_appearance_is_new_and_the_second_is_not(tmp_path):
    g = course(tmp_path)
    assert g.observe("malloc") is True
    assert g.observe("malloc") is False


def test_a_turn_reports_only_the_words_met_for_the_first_time(tmp_path):
    g = course(tmp_path)
    g.observe_turn("we call malloc here")

    fresh = g.observe_turn("malloc returns a pointer")
    assert "malloc" not in fresh, "already heard on this course"
    assert "pointer" in fresh


def test_this_filter_is_permissive_on_purpose(tmp_path):
    """
    "returns" is not a term, and it gets through — deliberately. This filter
    is a cheap first pass; the model is the one that decides. A false positive
    costs one question to the model, a false negative means the student meets
    a word with no help at all, so the errors are not symmetric and we lean
    towards offering.
    """
    g = course(tmp_path)
    assert "returns" in g.observe_turn("malloc returns a pointer")


def test_memory_survives_being_reopened(tmp_path):
    g = course(tmp_path)
    g.observe_turn("malloc allocates memory")
    g.save()

    assert course(tmp_path).is_new("malloc") is False


# --- what may become a term at all -------------------------------------------

def test_ordinary_short_words_never_enter(tmp_path):
    g = course(tmp_path)
    assert g.observe_turn("we are going to use it") == []


def test_a_contraction_is_not_a_term():
    """"That's" and "What's" once sat at the top of the candidate list."""
    assert is_termlike("That's") is False
    assert is_termlike("doesn't") is False


def test_a_course_code_with_digits_is_a_term():
    """CSE305 was excluded by a regex that only allowed letters."""
    assert is_termlike("CSE305") is True


def test_punctuation_does_not_create_a_second_term(tmp_path):
    g = course(tmp_path)
    g.observe("malloc")
    assert g.observe("malloc.") is False
    assert g.observe("(malloc)") is False


def test_a_plural_is_the_same_term_as_its_singular(tmp_path):
    g = course(tmp_path)
    g.observe("pointer")
    assert g.observe("pointers") is False


# --- forgetting --------------------------------------------------------------

def test_forget_removes_a_word_however_it_was_punctuated(tmp_path):
    g = course(tmp_path)
    g.observe_turn("do not forget the ampersand.")
    g.forget("ampersand")
    assert g.is_new("ampersand") is True


def test_forgetting_a_word_that_was_never_there_is_harmless(tmp_path):
    course(tmp_path).forget("nonexistent")


# --- what goes to AssemblyAI as keyterms -------------------------------------

def test_a_word_heard_once_is_not_sent_as_a_keyterm(tmp_path):
    """One appearance is as likely to be a mishearing as a real term."""
    g = course(tmp_path)
    g.observe("malloc")
    assert g.keyterms() == []

    g.observe("malloc")
    assert g.keyterms() == ["malloc"]


def test_keyterms_come_back_most_used_first(tmp_path):
    g = course(tmp_path)
    for _ in range(5):
        g.observe("scanf")
    for _ in range(2):
        g.observe("printf")

    assert g.keyterms() == ["scanf", "printf"]


def test_keyterms_respect_the_api_limit_of_a_hundred(tmp_path):
    g = course(tmp_path)
    for i in range(150):
        g.observe(f"termword{i}")
        g.observe(f"termword{i}")

    assert len(g.keyterms()) == 100


def test_a_term_too_long_for_the_api_is_left_out(tmp_path):
    g = course(tmp_path)
    monster = "a" * 60          # the API refuses keyterms over 50 characters
    g.observe(monster)
    g.observe(monster)

    assert g.keyterms() == []


def test_the_lowercase_form_wins_over_a_sentence_initial_capital(tmp_path):
    g = course(tmp_path)
    g.observe("Malloc")        # first word of a sentence
    g.observe("malloc")
    g.observe("malloc")

    assert g.keyterms() == ["malloc"]


# --- one memory per course ---------------------------------------------------

def test_two_courses_do_not_share_a_memory(tmp_path):
    """
    "New" means new to THIS course. A word the C course has worn out can still
    be the first time an algorithms student meets it, and that is the whole
    product — not "new to the model", not "rare in English".
    """
    c = course(tmp_path, "c-programming")
    algo = course(tmp_path, "algorithms")

    c.observe_turn("we call malloc here")
    c.save()

    assert algo.is_new("malloc") is True


def test_each_course_keeps_its_own_file(tmp_path):
    course(tmp_path, "c-programming").save()
    course(tmp_path, "algorithms").save()

    assert {p.name for p in tmp_path.glob("*.json")} == {
        "c-programming.json", "algorithms.json"
    }


def test_reopening_one_course_does_not_disturb_another(tmp_path):
    c = course(tmp_path, "c-programming")
    c.observe("malloc")
    c.observe("malloc")
    c.save()

    algo = course(tmp_path, "algorithms")
    algo.observe("recursion")
    algo.observe("recursion")
    algo.save()

    assert course(tmp_path, "c-programming").keyterms() == ["malloc"]
    assert course(tmp_path, "algorithms").keyterms() == ["recursion"]