"""
Put the same words to several versions of the judge's prompt, side by side.

    python src/compare_prompt.py

Every change to the prompt so far was made from a single screenshot, and each
one fixed what that screenshot showed. None was ever compared with the
version before it on the same words — which is how you "improve" one thing
and quietly break another. This asks every version the same question in the
same order, so the only thing that differs between the columns is the prompt.

Temperature is 0, so each column is what that prompt will actually do with
this batch, not a lucky draw. It is still ONE batch: live speech puts words
into different company, and a small model's answer depends on the company.
Treat a tie as a tie.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from terms import MAX_DEFINITION_CHARS, ask_gateway, mark, parse_reply
# The bare-word judge, retired from production (see terms.py); every version
# compared here is a bare-word prompt, so this is the one they are built on.
from terms import PROMPT_WORDS as PROMPT

SUBJECT = "C programming"

# (word, what a good judge says). "term" means it deserves a definition.
CASES = [
    # core terms: the product is useless if these are lost
    ("scanf", "term"), ("printf", "term"), ("malloc", "term"), ("pointer", "term"),
    ("heap", "term"), ("stack", "term"), ("array", "term"), ("segmentation", "term"),
    ("ampersand", "term"), ("union", "term"),
    # real names of the subject's world: must survive a mishearing rule
    ("Valgrind", "term"), ("POSIX", "term"),
    # ordinary words: a highlight here is noise on the student's screen
    ("program", "no"), ("crash", "no"), ("output", "no"), ("please", "no"),
    ("written", "no"), ("depend", "no"), ("treat", "no"), ("memory", "no"),
    # mishearings from the soak test: accepting one feeds it back as a keyterm
    ("ZStack", "no"), ("Zephyrus", "no"),
    # everyday English words that C gives a meaning of its own. These are the
    # price of any rule that pushes everyday words towards "no" — and the first
    # version of this table did not ask about them, so it could not see that
    # price being paid. "function" going unhighlighted on a soak run found it.
    ("function", "term"), ("string", "term"), ("return", "term"), ("loop", "term"),
    # genuinely ambiguous in C — reported, not scored
    ("class", "either"), ("block", "either"), ("local", "either"),
]

# As committed on 17 September (commit 248a614 carries it).
V1 = '''Subject: {subject}

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

{words}'''

# The first mishearing fix: "rare does not mean it belongs here".
_V2_FROM = """Unmarked words are rare outside this subject. Define them, unless they are
plainly not technical."""
assert _V2_FROM in V1, "V1 text drifted from what v2 was built on"
V2 = V1.replace(
    '''Unmarked words are rare outside this subject. Define them, unless they are
plainly not technical.''',
    '''Unmarked words are rare in everyday English, but rare does not mean they
belong here. These words come from live speech recognition, and a rare word
that does not fit {subject} — a product or brand name, a word from some other
field — is usually a mishearing of something ordinary. Define an unmarked word
only if a {subject} lecturer would really say it; otherwise answer "no".''',
).replace(
    "    problem (everyday) = no\n",
    "    problem (everyday) = no\n    Zenbook = no\n",
)

# v3 without its one line for everyday words ("if unsure, answer no"). v3
# changed two things at once; this separates them, so a loss can be pinned on
# the line that caused it instead of on the version as a whole.
_UNSURE = """ If you are not sure the meaning is
different, answer "no" — an ordinary word highlighted as a term costs the
student more than a term left for the sidebar."""
assert _UNSURE in PROMPT, "v3 text drifted from what v4 is built on"
V4 = PROMPT.replace(_UNSURE, "")

VERSIONS = [
    ("v1 (17 Sep)", V1),
    ("v2 (ZStack)", V2),
    ("v3 (retired)", PROMPT),
    ("v3 - unsure", V4),
]


def verdict(parsed: dict, word: str) -> str:
    """Three answers, not two — "the reply never mentioned it" is its own case."""
    key = word.lower()
    if key not in parsed:
        return "absent"
    return "term" if parsed[key] else "no"


def main() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    except ImportError:
        pass
    key = os.environ.get("ASSEMBLYAI_API_KEY")
    if not key:
        sys.exit("No ASSEMBLYAI_API_KEY")

    words = [w for w, _ in CASES]
    results = {}
    for name, template in VERSIONS:
        prompt = template.format(
            subject=SUBJECT,
            limit=MAX_DEFINITION_CHARS,
            words="\n".join(mark(w) for w in words),
        )
        print(f"asking {name}…", flush=True)
        reply = ask_gateway(prompt, key)
        parsed = parse_reply(reply, words)
        results[name] = {w: verdict(parsed, w) for w in words}

    names = [n for n, _ in VERSIONS]
    print()
    print(f"{'word':<14}{'should be':<11}" + "".join(f"{n:<15}" for n in names))
    print("-" * (25 + 15 * len(names)))
    score = {n: 0 for n in names}
    scored = 0
    for word, want in CASES:
        row = f"{word:<14}{want:<11}"
        for n in names:
            got = results[n][word]
            ok = want == "either" or got == want
            if want != "either" and ok:
                score[n] += 1
            mark_ = "  " if want == "either" else ("✓ " if ok else "✗ ")
            row += f"{mark_}{got:<13}"
        if want != "either":
            scored += 1
        print(row)

    print("-" * (25 + 15 * len(names)))
    print(f"{'right':<25}" + "".join(f"{score[n]}/{scored:<13}" for n in names))

    # Which way each version errs matters more than the total: losing a core
    # term and highlighting an ordinary word are not the same kind of mistake.
    print()
    for n in names:
        lost = [w for w, want in CASES if want == "term" and results[n][w] != "term"]
        noise = [w for w, want in CASES if want == "no" and results[n][w] == "term"]
        print(f"{n}:  terms lost {len(lost)} {lost or ''}  ·  noise let through {len(noise)} {noise or ''}")


if __name__ == "__main__":
    main()
