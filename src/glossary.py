"""
The course's memory: which terms this course has already used.

This is the mechanism the whole project rests on. A word is "new to the
student" when it has not appeared in any earlier lecture of this course —
which has nothing to do with whether the model transcribed it correctly.
A perfectly recognised word can still be the first time you have met it.

    from glossary import CourseGlossary
    g = CourseGlossary("c-programming")
    g.observe_turn("we allocate memory with malloc")   # -> ["malloc"] is new
    g.save()

Stored as one JSON file per course, so the memory survives between lectures.
"""

import json
import os
import re
import threading
from datetime import date
from pathlib import Path

from candidates import COMMON, stem

# Same shape as candidates.py: letters first, digits allowed after, so
# course codes and identifiers survive.
WORD_RE = re.compile(r"^[A-Za-z][A-Za-z0-9\-'.]*$")

MIN_LEN = 4


def is_termlike(word: str) -> bool:
    """
    Could this word be a course term at all?

    Deliberately permissive: a false positive costs one glance at a definition,
    a false negative means the student meets a word with no help. Errors are
    not symmetric, so we lean towards offering.
    """
    raw = word.strip(".,;:!?()\"'")
    if not raw or not WORD_RE.match(raw):
        return False
    key = stem(raw)
    if len(key) < MIN_LEN or key in COMMON:
        return False
    return True


class CourseGlossary:
    def __init__(self, course: str, directory: str = "glossary") -> None:
        self.course = course
        self.path = Path(directory) / f"{course}.json"
        self.terms: dict[str, dict] = {}
        # A request thread and the judging worker both reach this object. Without
        # a lock, json.dumps iterating self.terms while the other thread inserts
        # a word raises "dictionary changed size during iteration" — which, on
        # the worker path, discards a batch of definitions already paid for.
        # Reentrant because observe_turn calls observe, and save reads terms.
        self._guard = threading.RLock()
        self.load()

    # --- persistence ---------------------------------------------------------
    def load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            # A half-written file used to raise here, out of the constructor,
            # out of every endpoint for this course, forever. One bad shutdown
            # would have taken the course down permanently. Step aside instead:
            # the damaged file is kept for inspection, the course starts from
            # the seed, and the lecture goes on.
            broken = self.path.with_suffix(".json.broken")
            print(f"[glossary] {self.path.name} is unreadable ({exc}); moved to {broken.name}")
            try:
                os.replace(self.path, broken)
            except OSError:
                pass
            return
        self.terms = data.get("terms", {})

    def save(self) -> None:
        """
        Write the memory out, atomically.

        write_text truncates the file and then streams into it, so two threads
        saving at once — the request on every turn, the worker on every flush —
        leave a file that is neither version and parses as neither. Serialise
        under the lock, then swap the finished file into place in one step, so
        a reader only ever sees a whole file.
        """
        with self._guard:
            body = json.dumps(
                {"course": self.course, "terms": self.terms},
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(f".json.{os.getpid()}.tmp")
            tmp.write_text(body, encoding="utf-8")
            os.replace(tmp, self.path)

    # --- the core question ---------------------------------------------------
    def is_new(self, word: str) -> bool:
        return is_termlike(word) and stem(word.strip(".,;:!?()\"'")) not in self.terms

    def observe(self, word: str) -> bool:
        """Record one word. Returns True if this was its first appearance."""
        raw = word.strip(".,;:!?()\"'")
        if not is_termlike(raw):
            return False

        key = stem(raw)
        with self._guard:
            entry = self.terms.get(key)
            if entry is None:
                self.terms[key] = {
                    "surface": raw,
                    "count": 1,
                    "first_seen": date.today().isoformat(),
                }
                return True

            entry["count"] += 1
            # keep the lowercase form: a leading capital is usually sentence start
            if entry["surface"][:1].isupper() and not raw[:1].isupper():
                entry["surface"] = raw
            return False

    def forget(self, word: str) -> None:
        """
        Drop a word the judge rejected.

        Needed because observe() admits first and asks later: the cheap filter
        cannot tell "malloc" from "finish", so both enter, and this removes the
        one that turned out to be ordinary. Without it the course memory fills
        with noise and that noise goes out as keyterms.
        """
        with self._guard:
            self.terms.pop(stem(word.strip(".,;:!?()\"'")), None)

    def observe_turn(self, text: str) -> list[str]:
        """Feed a finished turn. Returns the terms met for the first time."""
        fresh = []
        for token in text.split():
            if self.observe(token):
                fresh.append(token.strip(".,;:!?()\"'"))
        return fresh

    # --- what we send back to the API ---------------------------------------
    def keyterms(self, limit: int = 100) -> list[str]:
        """
        The established vocabulary of this course, most-used first.

        API limits: at most 100 terms, each 50 characters or fewer. Terms seen
        once are left out — one appearance is as likely to be a mishearing as
        a real term.
        """
        with self._guard:
            settled = [
                (k, v) for k, v in self.terms.items() if v["count"] >= 2 and len(k) <= 50
            ]
        settled.sort(key=lambda kv: -kv[1]["count"])
        return [v["surface"] for _, v in settled[:limit]]

    def stats(self) -> str:
        total = len(self.terms)
        settled = sum(1 for v in self.terms.values() if v["count"] >= 2)
        return f"{self.course}: {total} terms known, {settled} settled"