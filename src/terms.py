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

import json
import os
from pathlib import Path

import requests

GATEWAY = "https://llm-gateway.assemblyai.com/v1/chat/completions"
MODEL = "qwen3.5-4b-32k-fast"   # fast and cheap; swap for a larger one if needed
BATCH = 20                       # terms per request
MAX_DEFINITION_CHARS = 90

PROMPT = """You are helping a student follow a lecture in {subject}.

Below are words the transcript picked up. For each, decide whether it is a
technical term, proper noun, identifier, or piece of jargon that a student new
to this subject would not already understand.

Ordinary English words used in their ordinary sense are NOT terms, even when
they appear often: finish, raise, progress, output, enter, error.

Return JSON only, no prose:
{{"term": "definition under {limit} characters", "other": null}}

Use null for anything that is not a term. Definitions must be one short line a
student can read at a glance without losing the thread of the lecture.

Words: {words}"""


class TermJudge:
    def __init__(self, subject: str, cache_dir: str = "glossary") -> None:
        self.subject = subject
        self.cache_path = Path(cache_dir) / f"{subject}.definitions.json"
        self.cache: dict[str, str | None] = {}
        self.api_key = os.environ.get("ASSEMBLYAI_API_KEY")
        self._load()

    def _load(self) -> None:
        if self.cache_path.exists():
            self.cache = json.loads(self.cache_path.read_text(encoding="utf-8"))

    def _save(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(
            json.dumps(self.cache, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    # --- the call ------------------------------------------------------------
    def _ask(self, words: list[str]) -> dict[str, str | None]:
        if not self.api_key:
            raise RuntimeError("ASSEMBLYAI_API_KEY is not set")

        prompt = PROMPT.format(
            subject=self.subject, limit=MAX_DEFINITION_CHARS, words=", ".join(words)
        )
        r = requests.post(
            GATEWAY,
            headers={"authorization": self.api_key, "content-type": "application/json"},
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 900,
                "temperature": 0,
            },
            timeout=30,
        )
        r.raise_for_status()
        return parse_reply(r.json()["choices"][0]["message"]["content"], words)

    def judge(self, words: list[str]) -> dict[str, str]:
        """
        Returns {term: definition} for the words that are real terms.
        Anything judged ordinary is dropped — and remembered as dropped, so we
        never spend a call on it again.
        """
        unseen = [w for w in words if w.lower() not in self.cache]

        for i in range(0, len(unseen), BATCH):
            chunk = unseen[i : i + BATCH]
            try:
                verdicts = self._ask(chunk)
            except Exception as exc:
                # A failed judgement must not take the transcript down with it.
                print(f"[terms] judging failed, keeping words unjudged: {exc}")
                continue
            for w in chunk:
                self.cache[w.lower()] = verdicts.get(w.lower())

        if unseen:
            self._save()

        return {
            w: self.cache[w.lower()]
            for w in words
            if self.cache.get(w.lower())
        }


def parse_reply(content: str, asked: list[str]) -> dict[str, str | None]:
    """
    Pull the JSON object out of the reply.

    Models like to wrap JSON in ```json fences or add a sentence before it, so
    take the outermost braces rather than trusting the whole string to parse.
    """
    start, end = content.find("{"), content.rfind("}")
    if start == -1 or end == -1:
        return {}

    try:
        raw = json.loads(content[start : end + 1])
    except json.JSONDecodeError:
        return {}

    asked_lower = {w.lower(): w for w in asked}
    out: dict[str, str | None] = {}
    for key, value in raw.items():
        k = key.lower().strip()
        if k not in asked_lower:
            continue
        if not isinstance(value, str) or not value.strip():
            out[k] = None
            continue
        text = " ".join(value.split())
        if len(text) > MAX_DEFINITION_CHARS:
            text = text[: MAX_DEFINITION_CHARS - 1].rstrip() + "…"
        out[k] = text
    return out