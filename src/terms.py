"""
Decide which candidate words are really course terms, and define them.

The word-list filter in glossary.py is too blunt for live speech: on a real
lecture it called `finish`, `raise` and `progress` terms. No list fixes that —
judging a word needs to know what the course is about.

So one call does both jobs at once:
  * throw out the ordinary words
  * return a one-line definition for the ones that survive

One line is a hard rule, not a style preference. The explanation appears while
the lecturer is still talking; a paragraph costs the student the next thirty
seconds, which is exactly what the product is supposed to save.

Results are cached — a term is judged once per course, ever.
"""

import hashlib
import json
import os
import re
import threading
import time
from pathlib import Path

import requests

# Shared across every course: the rate limit is per account, not per glossary.
# Guarded, because the pacing is a check-then-sleep and there are now two kinds
# of caller — the judging worker and the "I'm lost" button on a request thread.
# Unguarded, both read the same stale timestamp, both decide the gap is wide
# enough, and both fire in the same millisecond: the 429 storm this exists to
# prevent.
_last_call = {"at": 0.0}
_pace = threading.Lock()

# Some servers answer 429 with an hour. Sleeping that long on a request thread
# is indistinguishable from a hang, so we cap it and let the retry budget run
# out instead.
MAX_RETRY_AFTER_S = 30.0

GATEWAY = "https://llm-gateway.assemblyai.com/v1/chat/completions"
MODEL = "qwen3.5-4b-32k-fast"   # fast and cheap; swap for a larger one if needed
BATCH = 20                       # terms per request
MAX_DEFINITION_CHARS = 90

# A lecture produces a finished turn every few seconds, and calling the gateway
# on each one trips its rate limit within half a minute. Space the calls out and
# retry the ones that still bounce — the transcript is already on screen by then,
# so a second's wait for a definition costs nothing.
MIN_INTERVAL_S = 1.2
RETRIES = 3
BACKOFF_S = 1.5

# Everyday English, by corpus frequency rather than by hand. A hand-written
# stop list was letting "program", "crash" and "output" through, and no list
# written by one person at one sitting is ever finished.
#
# It marks words, it does not reject them: "class", "stack" and "table" are
# everyday words AND real terms, and the difference is not how common the word
# is but whether this subject gives it a second meaning. That question the
# model can answer — but only if it is asked about the right words, which is
# what the marking is for.
EVERYDAY = {
    w.strip()
    for w in (Path(__file__).resolve().parent.parent / "data" / "everyday-english.txt")
    .read_text(encoding="utf-8")
    .splitlines()
    if w.strip()
}

# The memory learns from the transcript, so a judge that accepts a mishearing
# does more than show one wrong definition: the word settles into the course,
# goes out as a keyterm, and pushes recognition TOWARDS the mistake. A soak
# test on a synthetic voice heard "the stack" as "ZStack" every time, and the
# old rule ("rare words are terms unless plainly not technical") accepted it —
# a brand name is rare and sounds technical. Rare is not the same as ours.
#
# Line format, not JSON. A small model asked for JSON copies the example's keys
# verbatim — it answered {"term": "scanf"} instead of {"scanf": "..."} — while
# judging the words correctly. One word per line, one separator, nothing to
# nest and nothing to balance.
#
# RETIRED. This is the judge that saw only the bare word. Kept because the
# benchmarks compare against it — it is the "before" in every table.
#
# Why it went: on real speech, the ordinary rare words ("dichotomy", "hereafter",
# "prefix") sit at exactly the same corpus frequency as the terms ("decimal",
# "syntax", "heap") — zipf 2.4–4.3 against 1.4–4.2 — so neither a frequency
# rule nor a stricter sentence in this prompt can tell them apart. On the
# 35-word real-speech benchmark it let through 16.3 of 19 noise words.
PROMPT_WORDS = """Subject: {subject}

For each word below, write exactly one line:

    word = short definition
    word = no

Words marked (everyday) are ordinary English. Define one only when this
subject gives it a different meaning from the one it has outside the lecture
hall — a class or a stack in programming is not the everyday thing, so those
get a definition. A program, a crash, an error or an output mean exactly what
they always mean, so those get "no". If you are not sure the meaning is
different, answer "no" — an ordinary word highlighted as a term costs the
student more than a term left for the sidebar.

Unmarked words are rare in everyday English, but rare does not make them
terms. Define an unmarked word only if it is a term, a name or a tool of
{subject} — something a student of {subject} would need explained. An
ordinary word the lecturer happens to use gets "no", however rare it is.
These words also come from live speech recognition, so a word that does not
fit {subject} at all — a product or brand name, a word from some other field —
is usually a mishearing: answer "no".

Definitions must be under {limit} characters — one line a student can read at a
glance while the lecturer keeps talking.

Example for a lecture on databases:

    rollback = undoes every change made since the transaction began
    index (everyday) = a lookup structure that makes queries faster
    problem (everyday) = no
    Zenbook = no

Now do these words. Nothing else, no headings, no numbering:

{words}"""


# The judge in production: each word with the sentence it was heard in.
#
# A word alone cannot say whether it is a term; the sentence can. "dichotomy"
# and "decimal" are equally rare in English, but "so there's this dichotomy"
# and "not using decimal but letters" are not equally about C.
#
# Measured before it shipped (src/bench_judge.py, 35 real-speech words, three
# batch orders each):
#
#                     terms lost (of 14)   noise let through (of 19)
#     bare word              0.0                   16.3
#     with sentence          0.3                    3.0
#
# The rule fixed in advance said: ship only if it loses no more terms. It lost
# one — "decimal", in one order of three — so by that rule it should not have
# shipped. We shipped it anyway, knowingly: in a live lecture, thirteen false
# highlights per nineteen cost the student more attention than one missed term
# in forty-two verdicts, and the missed term is still in the transcript.
# This text is the one that was measured; changing it means measuring again.
PROMPT = """Subject: {subject}

Each word below was heard in a live lecture, with the sentence it was said in.
For each word, write exactly one line:

    word = short definition, as the word is used in {subject}
    word = no

Give a definition only if, in that sentence, the word is used as a term of
{subject} — something a student of {subject} would need explained. If the
sentence uses it in its ordinary sense, if it belongs to some other field, or
if it looks like a speech-recognition mistake, answer "no".

Definitions must be under {limit} characters — one line a student can read at a
glance while the lecturer keeps talking.

Example, for a lecture on databases. The words:

    rollback
      heard in: "if the transfer fails, a rollback undoes every step it took"
    table
      heard in: "just put your laptop on the table for now"

The answer:

    rollback = undoes every change made since the transaction began
    table = no

Now these words. Answer with one line per word and nothing else:

{words}"""

# Long enough for a lecturer's sentence, short enough that twenty of them do
# not crowd out the question. Turns on real speech ran 60-80 seconds, so
# "the turn" is far too much context; the sentence is the unit.
MAX_CONTEXT_CHARS = 240

# A sentence ends at . ! or ? FOLLOWED BY SPACE. A bare period is not enough:
# "stdio.h", "0x1F" and "./addresses" are all said in a C lecture, and cutting
# there hands the judge half a sentence.
_SENTENCE_END = re.compile(r"[.!?](?=\s|$)")


def sentence_with(word: str, text: str) -> str | None:
    """
    The sentence of `text` that `word` was heard in, or None if it isn't there.
    """
    i = text.find(word)
    if i < 0:
        m = re.search(re.escape(word), text, re.IGNORECASE)
        if not m:
            return None
        i = m.start()

    # Search the whole text and keep the ends before the word. Stopping the
    # search AT the word would make "$" match there, and "stdio.h" would end
    # a sentence just because the word began after its dot.
    start = 0
    for m in _SENTENCE_END.finditer(text):
        if m.end() > i:
            break
        start = m.end()
    m = _SENTENCE_END.search(text, i + len(word))
    end = m.end() if m else len(text)

    s = " ".join(text[start:end].split())
    if len(s) > MAX_CONTEXT_CHARS:
        # A run-on sentence: keep a window around the word, not the head of it.
        at = s.lower().find(word.lower())
        lo = max(0, at - MAX_CONTEXT_CHARS // 2)
        s = "…" + s[lo:lo + MAX_CONTEXT_CHARS].strip() + "…"
    # It goes inside double quotes in the prompt.
    return s.replace('"', "'")


def build_prompt(subject: str, words: list[str],
                 contexts: dict[str, str | None] | None = None) -> str:
    """
    The production question. A word with no sentence goes in bare.

    The sentence sits on its own line after a label with a colon, so that if
    the model echoes it back, parse_reply reads "heard in" as the key, finds it
    is not a word we asked about, and skips it — instead of taking the
    sentence for a definition.
    """
    contexts = contexts or {}
    lines = [
        f'{w}\n  heard in: "{contexts[w]}"' if contexts.get(w) else w
        for w in words
    ]
    return PROMPT.format(subject=subject, limit=MAX_DEFINITION_CHARS,
                         words="\n".join(lines))


def is_everyday(word: str) -> bool:
    """
    Checked as spoken AND by stem. The list holds "screen" but not always
    "screens", and a rule that lets the plural through while stopping the
    singular is a rule with a hole exactly where lecturers talk in plurals.
    """
    from candidates import stem          # local: candidates imports nothing of ours
    w = word.lower().strip(".,;:!?()\"'")
    return w in EVERYDAY or stem(w) in EVERYDAY


def mark(word: str, declared: frozenset[str] = frozenset()) -> str:
    """
    Tag an everyday word for the judge — unless the course has declared it.

    A declared word is one the course says is a term here even though it is
    ordinary English ("free", "address"). It goes to the judge untagged, as a
    candidate term, because telling the judge it is everyday would get it
    rejected: that is what the tag is for.
    """
    if word.lower() in declared:
        return word
    return f"{word} (everyday)" if is_everyday(word) else word


# Short on purpose: it goes in every cache file and only has to detect change,
# not prove anything. The model name is in it because the same question put to
# a different model is a different question.
PROMPT_FINGERPRINT = hashlib.sha256(
    (PROMPT + MODEL + str(len(EVERYDAY))).encode()
).hexdigest()[:12]


def ask_gateway(prompt: str, api_key: str | None, max_tokens: int = 900) -> str:
    """
    One question to the LLM Gateway, with the pacing that keeps it answering.

    Module-level rather than a method because the rate limit belongs to the
    ACCOUNT, not to any one judge or course. Two callers spacing their own
    requests independently would still trip it together, which is how we
    collected 429s the first time.
    """
    if not api_key:
        raise RuntimeError("ASSEMBLYAI_API_KEY is not set")

    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }

    last: Exception | None = None
    for attempt in range(RETRIES):
        # Claim the slot and sleep while holding it, so the next caller waits
        # for us rather than racing us.
        with _pace:
            gap = time.monotonic() - _last_call["at"]
            if gap < MIN_INTERVAL_S:
                time.sleep(MIN_INTERVAL_S - gap)
            _last_call["at"] = time.monotonic()

        try:
            r = requests.post(
                GATEWAY,
                headers={"authorization": api_key, "content-type": "application/json"},
                json=body,
                timeout=30,
            )
            with _pace:
                _last_call["at"] = time.monotonic()

            if r.status_code == 429:
                # Honour the server's own advice when it gives any — but the
                # header is allowed to be an HTTP date, and float() on that
                # raises ValueError, which is not a RequestException and so
                # escaped the retry loop entirely.
                wait = BACKOFF_S * (attempt + 1)
                try:
                    wait = float(r.headers["retry-after"])
                except (KeyError, ValueError):
                    pass
                wait = min(wait, MAX_RETRY_AFTER_S)
                print(f"[terms] rate limited, waiting {wait:.1f}s")
                time.sleep(wait)
                last = RuntimeError("rate limited")
                continue

            r.raise_for_status()
            try:
                return r.json()["choices"][0]["message"]["content"]
            except (ValueError, KeyError, IndexError, TypeError) as exc:
                # A 200 carrying an error envelope, or any change in shape.
                # None of these are RequestException, so without this they
                # skipped the retry and surfaced as a whole failed batch.
                raise RuntimeError(f"unreadable gateway reply: {exc}") from exc

        except (requests.RequestException, RuntimeError) as exc:
            with _pace:
                _last_call["at"] = time.monotonic()
            last = exc
            if attempt < RETRIES - 1:
                time.sleep(BACKOFF_S * (attempt + 1))

    raise last or RuntimeError("gateway did not answer")


class TermJudge:
    def __init__(self, subject: str, cache_dir: str = "glossary",
                 course: str | None = None,
                 declared: frozenset[str] = frozenset()) -> None:
        self.subject = subject
        self.declared = frozenset(w.lower() for w in declared)
        # Named after the COURSE, not the subject. The subject is prose that
        # goes into the prompt ("C programming"); the course is the identity
        # ("c-programming"). Naming the file after the subject put a space in
        # it and, worse, split one course's cache across two files whenever a
        # request arrived without a subject — silently, and only after a
        # restart, which is the hardest kind of bug to see.
        self.cache_path = Path(cache_dir) / f"{course or subject}.definitions.json"
        self.cache: dict[str, str | None] = {}
        self.api_key = os.environ.get("ASSEMBLYAI_API_KEY")
        self._load()

    def _load(self) -> None:
        if not self.cache_path.exists():
            return

        try:
            data = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            # Losing the cache costs one re-judging pass. Refusing to start
            # costs the lecture.
            print(f"[terms] {self.cache_path.name} is unreadable ({exc}); starting fresh")
            return
        # A cached verdict is never re-asked — that is the whole point of a
        # cache, and it is also how a change to the prompt goes unnoticed:
        # every word judged under the old rules keeps its old answer forever.
        # The fingerprint makes the cache expire with the question that
        # produced it, so changing PROMPT is a code change and nothing else.
        if data.get("prompt") != self.fingerprint():
            print(f"[terms] prompt changed — re-judging {self.subject} from scratch")
            return
        # A cache written before is_rejection knew every phrasing may hold
        # "not a C programming term" as a definition. Read it the way the
        # parser now would, so the fix reaches words judged before it.
        self.cache = {
            k: (None if is_rejection(v) else v)
            for k, v in data.get("terms", {}).items()
        }

    def _save(self) -> None:
        # Same reasoning as CourseGlossary.save: truncate-then-stream leaves a
        # half file behind if anything else writes at the same moment.
        body = json.dumps(
            {"prompt": self.fingerprint(), "terms": dict(self.cache)},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.cache_path.with_suffix(f".json.{os.getpid()}.tmp")
        tmp.write_text(body, encoding="utf-8")
        os.replace(tmp, self.cache_path)

    def fingerprint(self) -> str:
        """
        What the cached verdicts were answers to.

        The course's declared words change the question: "free" asked while
        tagged everyday was rejected and cached, and a cached word is never
        asked again — so declaring it later would change nothing. Folding the
        list into the fingerprint expires the cache when the list changes.
        With nothing declared it is exactly the prompt's own fingerprint.
        """
        if not self.declared:
            return PROMPT_FINGERPRINT
        return hashlib.sha256(
            (PROMPT_FINGERPRINT + "|" + ",".join(sorted(self.declared))).encode()
        ).hexdigest()[:12]

    def cached(self, words: list[str]) -> tuple[dict[str, str], list[str], list[str]]:
        """
        Split words by what we already know, without touching the network.

        Three outcomes, not two. "Judged a term", "judged and rejected" and
        "never asked" need different handling by the caller: the first goes on
        screen now, the second is dropped from the glossary now, and only the
        third is worth a call to the model. Answering the first two instantly
        is what keeps a repeated term from ever waiting again.
        """
        defined: dict[str, str] = {}
        rejected: list[str] = []
        unknown: list[str] = []

        for w in words:
            key = w.lower()
            if key not in self.cache:
                unknown.append(w)
            elif self.cache[key]:
                defined[w] = self.cache[key]
            else:
                rejected.append(w)

        return defined, rejected, unknown

    # --- the call ------------------------------------------------------------
    def _ask(self, words: list[str],
             contexts: dict[str, str | None] | None = None) -> dict[str, str | None]:
        # No "(everyday)" tag any more: everyday words never reach the judge
        # (server.observe drops them by rule), and the declared ones that do
        # are meant to be judged as candidate terms — the sentence decides.
        prompt = build_prompt(self.subject, words, contexts)
        return parse_reply(ask_gateway(prompt, self.api_key), words)

    def judge(self, words: list[str],
              contexts: dict[str, str | None] | None = None) -> dict[str, str]:
        """
        Returns {term: definition} for the words that are real terms.

        Raises if the model could not be reached at all. That distinction is
        the point: "judged and rejected" and "could not be judged" look
        identical from the outside — both are an absent key — and treating the
        second like the first silently throws away every term in the lecture.
        The caller must be able to tell them apart.
        """
        unseen = [w for w in words if w.lower() not in self.cache]
        failures: list[str] = []

        for i in range(0, len(unseen), BATCH):
            chunk = unseen[i : i + BATCH]
            try:
                # The verdict is cached per word, so the sentence it was FIRST
                # heard in decides it for the course. The lecturer introduces
                # a term when it first comes up, so that is the right sentence.
                verdicts = self._ask(chunk, contexts)
            except Exception as exc:
                failures.append(f"{type(exc).__name__}: {exc}")
                continue
            for w in chunk:
                key = w.lower()
                if key not in verdicts:
                    # The reply never mentioned this word. That is a parsing
                    # failure, not a verdict — caching it as "rejected" would
                    # bury the word permanently and silently, and no later run
                    # would ever ask about it again.
                    print(f"[terms] no verdict for {w!r}, leaving it unjudged")
                    continue
                self.cache[key] = verdicts[key]

        if unseen and failures and len(failures) * BATCH >= len(unseen):
            raise RuntimeError(f"LLM Gateway unreachable — {failures[0]}")

        if unseen:
            self._save()

        return {w: self.cache[w.lower()] for w in words if self.cache.get(w.lower())}


SEPARATORS = ("=", "—", ":", "|", " - ")
REJECTIONS = {"no", "-", "none", "null", "n/a", "not a term", "ordinary"}

# The model does not always say "no" in one word. On CS50 it answered
# "zoom = not a C programming term" — five times in one lecture — and the
# parser, finding text after the "=", showed it as a definition: the model had
# rejected the word and the student saw it highlighted anyway. The word "term"
# is required, so a real definition that happens to start the same way
# ("NULL = not a valid address") is still a definition.
_REFUSAL = re.compile(r"^not (a|an)\b.*\bterm\b|^no\s*[,;:(—]", re.IGNORECASE)


def is_rejection(value: str | None) -> bool:
    """True when the model's answer says "not a term", in any of its phrasings."""
    if not value:
        return True
    v = value.strip().strip("`\"'").rstrip(".").strip()
    return not v or v.lower() in REJECTIONS or bool(_REFUSAL.search(v))


def parse_reply(content: str, asked: list[str]) -> dict[str, str | None]:
    """
    Read "word = definition" lines.

    Tolerant on purpose: models add headings, numbering, bullets and stray
    backticks, and they pick whichever separator they feel like. Lines whose
    left side isn't a word we asked about are skipped rather than guessed at.
    """
    wanted = {w.lower().strip(".,;:!?()\"'"): w for w in asked}
    out: dict[str, str | None] = {}

    for raw_line in content.splitlines():
        line = raw_line.strip().strip("`").lstrip("-*•").strip()
        if not line:
            continue
        # drop "1." / "2)" numbering
        line = re.sub(r"^\d+[.)]\s*", "", line)

        for sep in SEPARATORS:
            if sep not in line:
                continue
            left, right = line.split(sep, 1)
            key = left.strip().strip("`\"'*").lower()
            key = re.sub(r"\s*\(.*?\)\s*$", "", key)   # drop our own (everyday) marker
            key = key.strip(".,;:!?()")
            if key not in wanted:
                break  # a line about something else; don't try other separators

            value = right.strip().strip("`\"'").rstrip(".")
            if is_rejection(value):
                out[key] = None
            else:
                text = " ".join(value.split())
                if len(text) > MAX_DEFINITION_CHARS:
                    text = text[: MAX_DEFINITION_CHARS - 1].rstrip() + "…"
                out[key] = text
            break

    return out