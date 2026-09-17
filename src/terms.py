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
import time
from pathlib import Path

import requests

# Shared across every course: the rate limit is per account, not per glossary.
_last_call = {"at": 0.0}

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

# Line format, not JSON. A small model asked for JSON copies the example's keys
# verbatim — it answered {"term": "scanf"} instead of {"scanf": "..."} — while
# judging the words correctly. One word per line, one separator, nothing to
# nest and nothing to balance.
PROMPT = """Subject: {subject}

For each word below, write exactly one line:

    word = short definition
    word = no

Words marked (everyday) are ordinary English. Define one only when this
subject gives it a different meaning from the one it has outside the lecture
hall — a class or a stack in programming is not the everyday thing, so those
get a definition. A program, a crash, an error or an output mean exactly what
they always mean, so those get "no".

Unmarked words are rare outside this subject. Define them, unless they are
plainly not technical.

Definitions must be under {limit} characters — one line a student can read at a
glance while the lecturer keeps talking.

Example for a lecture on databases:

    rollback = undoes every change made since the transaction began
    index (everyday) = a lookup structure that makes queries faster
    problem (everyday) = no

Now do these words. Nothing else, no headings, no numbering:

{words}"""


def mark(word: str) -> str:
    return f"{word} (everyday)" if word.lower() in EVERYDAY else word


# Short on purpose: it goes in every cache file and only has to detect change,
# not prove anything. The model name is in it because the same question put to
# a different model is a different question.
PROMPT_FINGERPRINT = hashlib.sha256(
    (PROMPT + MODEL + str(len(EVERYDAY))).encode()
).hexdigest()[:12]


class TermJudge:
    def __init__(self, subject: str, cache_dir: str = "glossary") -> None:
        self.subject = subject
        self.cache_path = Path(cache_dir) / f"{subject}.definitions.json"
        self.cache: dict[str, str | None] = {}
        self.api_key = os.environ.get("ASSEMBLYAI_API_KEY")
        self._load()

    def _load(self) -> None:
        if not self.cache_path.exists():
            return

        data = json.loads(self.cache_path.read_text(encoding="utf-8"))
        # A cached verdict is never re-asked — that is the whole point of a
        # cache, and it is also how a change to the prompt goes unnoticed:
        # every word judged under the old rules keeps its old answer forever.
        # The fingerprint makes the cache expire with the question that
        # produced it, so changing PROMPT is a code change and nothing else.
        if data.get("prompt") != PROMPT_FINGERPRINT:
            print(f"[terms] prompt changed — re-judging {self.subject} from scratch")
            return
        self.cache = data.get("terms", {})

    def _save(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(
            json.dumps(
                {"prompt": PROMPT_FINGERPRINT, "terms": self.cache},
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

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
    def _ask(self, words: list[str]) -> dict[str, str | None]:
        if not self.api_key:
            raise RuntimeError("ASSEMBLYAI_API_KEY is not set")

        prompt = PROMPT.format(
            subject=self.subject,
            limit=MAX_DEFINITION_CHARS,
            words="\n".join(mark(w) for w in words),   # one per line, as the answer
        )
        body = {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 900,
            "temperature": 0,
        }

        last: Exception | None = None
        for attempt in range(RETRIES):
            # Keep our own calls apart before the gateway has to push back.
            gap = time.monotonic() - _last_call["at"]
            if gap < MIN_INTERVAL_S:
                time.sleep(MIN_INTERVAL_S - gap)

            try:
                r = requests.post(
                    GATEWAY,
                    headers={
                        "authorization": self.api_key,
                        "content-type": "application/json",
                    },
                    json=body,
                    timeout=30,
                )
                _last_call["at"] = time.monotonic()

                if r.status_code == 429:
                    # Honour the server's own advice when it gives any.
                    wait = float(r.headers.get("retry-after") or BACKOFF_S * (attempt + 1))
                    print(f"[terms] rate limited, waiting {wait:.1f}s")
                    time.sleep(wait)
                    last = RuntimeError("rate limited")
                    continue

                r.raise_for_status()
                return parse_reply(r.json()["choices"][0]["message"]["content"], words)

            except requests.RequestException as exc:
                _last_call["at"] = time.monotonic()
                last = exc
                if attempt < RETRIES - 1:
                    time.sleep(BACKOFF_S * (attempt + 1))

        raise last or RuntimeError("gateway did not answer")

    def judge(self, words: list[str]) -> dict[str, str]:
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
                verdicts = self._ask(chunk)
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
            if not value or value.lower() in REJECTIONS:
                out[key] = None
            else:
                text = " ".join(value.split())
                if len(text) > MAX_DEFINITION_CHARS:
                    text = text[: MAX_DEFINITION_CHARS - 1].rstrip() + "…"
                out[key] = text
            break

    return out