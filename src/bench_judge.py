"""
Does the judge get better if it sees the sentence a word was said in?

    python src/bench_judge.py

Every word here was either heard on a real lecture (CS50, week 4, streamed
through Lecture Lens) or comes from our own lecture on C, and each carries the
sentence it was spoken in and what a good judge should say. Two judges answer
the same words:

    words     the retired judge: the bare word (terms.PROMPT_WORDS)
    context   the production judge: the word and the sentence it was heard in
              (terms.build_prompt — the same function the server calls)

Each is asked in three different orders. A small model's verdict depends on
which other words share the batch — on the old 29-word table, adding seven
words flipped three verdicts at temperature 0 — so a single order proves
nothing, and the averages are what count.

The decision rule was fixed before the first run and is still printed with the
numbers: context goes in only if, on average, it loses no more real terms AND
lets at least 30% less noise through. On the first run it cut noise from 16.3
to 3.0 but lost "decimal" in one order of three, so the rule said no. It
shipped anyway, as a deliberate trade — terms.py says why. The rule stays
printed so that every later run is read against the same bar.
"""

import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from terms import (MAX_DEFINITION_CHARS, PROMPT_WORDS, ask_gateway, build_prompt,
                   mark, parse_reply)

SUBJECT = "C programming"
ORDERS = 3                 # shuffles per judge
BATCH = 18                 # close to production's batch of 20

# (word, sentence it was heard in, what a good judge says)
CASES = [
    # -- noise, heard on CS50: every one was highlighted on the live run --------
    ("semester", "for fun, at the beginning of the semester, we have a staff training with all of the teaching fellows", "no"),
    ("spreadsheet", "we gave them all this Google spreadsheet, and we sort of resized all of the rows and columns", "no"),
    ("resized", "we sort of resized all of the rows and columns to just be squares instead of the default rectangles", "no"),
    ("canvas", "a team who in a few minutes made a Super Mario World, a bigger canvas, of course, than this here easel", "no"),
    ("Scratch", "here we have a pixel-based version of Scratch", "no"),
    ("screenshot", "here's a screenshot of Photoshop, if you've never used it before", "no"),
    ("picker", "this is like the color picker that you can pull up just to pick any number of millions of colors", "no"),
    ("slider", "we've picked the color by moving the slider all the way down here to the bottom left-hand corner", "no"),
    ("arbitrary", "like 72, 73, 33 was the arbitrary example we used in week 0 for the color yellow", "no"),
    ("equivalently", "at the end of the day it's just bits, and equivalently, it's just numbers", "no"),
    ("uppercase", "it's conventional to use uppercase or lowercase, so long as you're generally consistent", "no"),
    ("capitalization", "and the capitalization actually doesn't matter", "no"),
    ("mathematical", "otherwise known as base 16 for mathematical reasons that go back to our discussion in week 0", "no"),
    ("identical", "so it's identical up until this point to our world of decimal", "no"),
    ("reasoning", "so here's some of that same reasoning from week 0", "no"),
    ("smiley", "it's a smiley face, how do you see that", "no"),
    ("treating", "creating images using Google Spreadsheets by treating each of the cells as just a dot on the screen", "no"),
    ("Mario", "a team who in a few minutes made a Super Mario World", "no"),
    ("graphics", "we've seen that already in sort of black and white graphics", "no"),
    # -- real terms, heard on CS50 --------------------------------------------------
    ("hexadecimal", "hexadecimal, or what we might call base 16", "term"),
    ("binary", "so in hexadecimal and in binary and in decimal, it's the same way to represent the number", "term"),
    ("decimal", "the way you write it conventionally is not using decimal but using letters of the alphabet", "term"),
    ("bits", "at the end of the day it's, of course, just bits", "term"),
    ("RGB", "we introduced RGB, which stands for red, green, blue", "term"),
    # -- real terms, from our own lecture on C ---------------------------------------
    ("malloc", "every call to malloc must be matched by a call to free", "term"),
    ("pointer", "a pointer holds an address, and dereferencing it reaches the data", "term"),
    ("heap", "you pass free the same pointer malloc gave you, and the block goes back to the heap", "term"),
    ("dereference", "to dereference a pointer means to follow it, to read or write the value it points at", "term"),
    ("segmentation", "the operating system stops you, and the program dies with a segmentation fault", "term"),
    ("sizeof", "the sizeof operator gives you the size of a type in bytes", "term"),
    ("struct", "a struct lets you group related values under one name", "term"),
    ("linker", "finally the linker takes all the object files, plus the libraries, and fills in those holes", "term"),
    ("preprocessor", "first the preprocessor runs; it handles every line that begins with a hash", "term"),
    # -- genuinely ambiguous here: reported, not scored ------------------------------
    ("notation", "so hexadecimal notation here, otherwise known as base 16", "either"),
    ("pixel", "if you actually have a larger grid, you can do even more with pixel art", "either"),
]



def ask_words(batch, key):
    """What production does today: the bare words."""
    prompt = PROMPT_WORDS.format(subject=SUBJECT, limit=MAX_DEFINITION_CHARS,
                           words="\n".join(mark(w) for w, _, _ in batch))
    return parse_reply(ask_gateway(prompt, key), [w for w, _, _ in batch])


def ask_context(batch, key):
    """Production: each word with the sentence it was heard in."""
    prompt = build_prompt(SUBJECT, [w for w, _, _ in batch], {w: s for w, s, _ in batch})
    return parse_reply(ask_gateway(prompt, key), [w for w, _, _ in batch])


def verdict(parsed, word):
    key = word.lower()
    if key not in parsed:
        return "absent"
    return "term" if parsed[key] else "no"


def run(judge, key, seed):
    cases = CASES[:]
    random.Random(seed).shuffle(cases)
    got = {}
    for i in range(0, len(cases), BATCH):
        batch = cases[i:i + BATCH]
        parsed = judge(batch, key)
        for w, _, _ in batch:
            got[w] = verdict(parsed, w)
    lost = [w for w, _, want in CASES if want == "term" and got[w] != "term"]
    noise = [w for w, _, want in CASES if want == "no" and got[w] == "term"]
    return got, lost, noise


def main() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    except ImportError:
        pass
    key = os.environ.get("ASSEMBLYAI_API_KEY")
    if not key:
        sys.exit("No ASSEMBLYAI_API_KEY")

    n_terms = sum(1 for *_, want in CASES if want == "term")
    n_noise = sum(1 for *_, want in CASES if want == "no")
    print(f"{len(CASES)} words: {n_terms} real terms, {n_noise} noise, "
          f"{len(CASES) - n_terms - n_noise} ambiguous (not scored)")
    print(f"{ORDERS} orders per judge, batches of {BATCH}. "
          f"The gateway rate-limits, so expect a few pauses.\n")

    results = {}
    for name, judge in (("words", ask_words), ("context", ask_context)):
        runs = []
        for seed in range(ORDERS):
            print(f"  {name}, order {seed + 1}…", flush=True)
            runs.append(run(judge, key, seed))
        results[name] = runs

    print()
    for name, runs in results.items():
        print(f"== {name}")
        for i, (_, lost, noise) in enumerate(runs, 1):
            print(f"   order {i}: lost {len(lost):>2} {lost or ''}")
            print(f"            noise {len(noise):>2} {noise or ''}")
    print()

    avg = {name: (sum(len(r[1]) for r in runs) / ORDERS,
                  sum(len(r[2]) for r in runs) / ORDERS)
           for name, runs in results.items()}
    for name, (lost, noise) in avg.items():
        print(f"{name:<8} average: terms lost {lost:4.1f} / {n_terms}   "
              f"noise let through {noise:4.1f} / {n_noise}")

    (lw, nw), (lc, nc) = avg["words"], avg["context"]
    no_worse = lc <= lw
    clearly_cleaner = nc <= 0.7 * nw
    print("\nRule, fixed in advance: ship context only if it loses no more terms")
    print("AND lets at least 30% less noise through.")
    print(f"  loses no more terms:      {'yes' if no_worse else 'NO'}  ({lc:.1f} vs {lw:.1f})")
    print(f"  at least 30% less noise:  {'yes' if clearly_cleaner else 'NO'}  ({nc:.1f} vs {nw:.1f})")
    print(f"\nVERDICT: {'ship context' if no_worse and clearly_cleaner else 'keep the current judge'}")

    print("\nambiguous words, for the record:")
    for w, s, want in CASES:
        if want == "either":
            print(f"  {w:<10} words: {[r[0][w] for r in results['words']]}"
                  f"   context: {[r[0][w] for r in results['context']]}")


if __name__ == "__main__":
    main()
