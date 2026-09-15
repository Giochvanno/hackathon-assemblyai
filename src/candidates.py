"""
Find glossary candidates in a saved transcript.

Answers the question the whole project rests on: does this lecture contain
vocabulary the model actually struggles with?

    python src/candidates.py transcripts/lecture-20260910-143000.json

Four signals, combined. No single one is trustworthy on its own:

  repetition   a lecture says its key terms over and over; noise appears once
  confidence   the model reports per-word certainty, and doubt clusters on
               unfamiliar vocabulary
  rarity       common English words are not course terms, however often the
               lecturer says "the"
  shape        mid-sentence capitals and hyphenated compounds mark proper
               nouns and coined terms

A word can be recognised confidently and still be wrong ("Kwon" heard as
"con" is ordinary English), which is why rarity and shape carry weight even
when confidence looks fine.
"""

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

# The most frequent English words. Anything here is structural, never a term.
COMMON = set(
    """
a about above across actually add after again against ago all almost alone along
already also although always am among amount an and another answer any anyone
anything are around as ask asked asking at away back bad be became because become
been before began begin behind being believe below best better between big both
bring but buy by call called came can cannot case certain change check class
clear close come coming common complete could course cover day days deal did
different do does doing done door down draw during each early easy either else
end enough entire even ever every everyone everything exactly example except
far fast feel few find fine first five follow following for form found four
from full further gave general get give given go goes going gone good got great
group had half hand happen hard has have having he head hear held help her here
high him his hold holds home hope hour how however idea if important in include
indeed inside instead into is it its itself just keep kept kind knew know known
large last late later learn least leave left less let letter level life like
likely line list little live long look looking lot love low made main make
making man many may maybe me mean means meant meet might mind mine minute miss
moment money month more morning most move much must my name near need needed
never new next nice night no none nor not note nothing now number of off often
okay old on once one only open or order other others our out outside over own
page part particular past people perhaps person pick place plan play please
point possible present probably problem put question questions quick quite
rather reach read ready real really reason receive remember report rest result
results right room run said same saw say saying says second see seem seen send
sense sent serious set several shall she short should show side simple since
sir six small so some someone something sometimes soon sort sound speak special
spend stand start state stay step still stop story such sure take taken talk
talking tell ten than thank thanks that the their them themselves then there
these they thing things think this those though thought three through time to
today together told too took top toward true try turn turned two under
understand until up upon us use used using usually very want was watch water
way we week well went were what when where whether which while who whole why
will wish with within without woman word words work working world would write
wrong year years yes yesterday yet you young your yourself
""".split()
)

# Digits are allowed after the first letter: course codes (CSE305, 18.06) and
# model names (GPT4, K2) are exactly the vocabulary a general model has never met.
WORD_RE = re.compile(r"^[A-Za-z][A-Za-z0-9\-'.]*$")


def load(path: str) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    words: list[dict] = []
    for turn in data.get("turns", []):
        words.extend(turn.get("words") or [])
    return words


CONTRACTION = re.compile(r"'(s|t|re|ve|ll|d|m)$", re.I)

# Latin and Greek plurals, which the -s rules below cannot reach. Short list:
# these are the ones that actually turn up in technical lectures.
IRREGULAR = {
    "matrices": "matrix", "indices": "index", "vertices": "vertex",
    "appendices": "appendix", "axes": "axis", "analyses": "analysis",
    "hypotheses": "hypothesis", "criteria": "criterion", "phenomena": "phenomenon",
    "radii": "radius", "formulae": "formula", "schemata": "schema",
}


def stem(word: str) -> str:
    """
    "That's" -> "that". Without this the common-word list misses every
    contraction, and they flood the top of the ranking.
    """
    w = word.strip(".,;:!?()\"'").lower()
    # "n't" first: stripping the generic "'t" would leave "doesn"
    if w.endswith("n't"):
        w = w[:-3]
    else:
        w = CONTRACTION.sub("", w)

    if w in IRREGULAR:
        return IRREGULAR[w]

    # Fold plurals so "pointer" and "pointers" are one term. Conservative:
    # only endings that are almost always inflection, never part of a stem.
    if len(w) > 4:
        if w.endswith("ies"):
            return w[:-3] + "y"
        if w.endswith(("ses", "xes", "zes", "ches", "shes")):
            return w[:-2]
        if w.endswith("s") and not w.endswith(("ss", "us", "is")):
            return w[:-1]
    return w


def analyse(words: list[dict], min_count: int = 2) -> list[dict]:
    seen: dict[str, list[dict]] = defaultdict(list)
    surface: dict[str, str] = {}

    for w in words:
        raw = (w.get("text") or "").strip(".,;:!?()\"'")
        key = stem(raw)
        if not raw or not WORD_RE.match(raw) or len(key) < 4:
            continue
        seen[key].append(w)
        # prefer the lowercase spelling: a leading capital is usually just the
        # start of a sentence, not a proper noun
        if key not in surface or (surface[key][:1].isupper() and not raw[:1].isupper()):
            surface[key] = raw

    out = []
    for key, hits in seen.items():
        count = len(hits)
        if count < min_count:
            continue

        confs = [h["confidence"] for h in hits if h.get("confidence") is not None]
        mean_conf = statistics.mean(confs) if confs else None
        word = surface[key]

        rare = key not in COMMON
        # A capital at position 0 means "sentence started here" far more often
        # than "proper noun", so only interior capitals, digits and hyphens count.
        shaped = (
            bool(re.search(r"[A-Z]", word[1:]))
            or bool(re.search(r"\d", word))
            or "-" in word
        )

        # Score: repetition matters, doubt matters more, commonness kills it.
        score = 0.0
        if rare:
            score += 1.0
        if shaped:
            score += 0.6
        score += min(count / 6, 1.2)
        if mean_conf is not None:
            score += (1 - mean_conf) * 2.5

        if not rare:
            score *= 0.15  # keep it visible but at the bottom

        out.append(
            {
                "term": word,
                "count": count,
                "confidence": round(mean_conf, 3) if mean_conf is not None else None,
                "min_confidence": round(min(confs), 3) if confs else None,
                "score": round(score, 2),
            }
        )

    return sorted(out, key=lambda r: -r["score"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("transcript", help="a .json file written by stream.py")
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--min-count", type=int, default=2)
    ap.add_argument("--write", help="save the top terms, one per line, for --keyterms-file")
    args = ap.parse_args()

    words = load(args.transcript)
    if not words:
        raise SystemExit(
            "No word-level data in that file.\n"
            "Word timings ride on unformatted turns — if this run has none, "
            "the transcript was saved before that fix."
        )

    rows = analyse(words, args.min_count)[: args.top]

    print(f"{len(words)} words analysed, {len(rows)} candidates\n")
    print(f"{'term':<24}{'seen':>6}{'conf':>8}{'worst':>8}{'score':>8}")
    print("-" * 54)
    for r in rows:
        c = f"{r['confidence']:.2f}" if r["confidence"] is not None else "  -"
        m = f"{r['min_confidence']:.2f}" if r["min_confidence"] is not None else "  -"
        print(f"{r['term']:<24}{r['count']:>6}{c:>8}{m:>8}{r['score']:>8.2f}")

    shaky = [r for r in rows if r["confidence"] is not None and r["confidence"] < 0.7]
    print()
    if shaky:
        print(f"{len(shaky)} repeated terms below 0.70 confidence — the glossary has work to do:")
        print("  " + ", ".join(r["term"] for r in shaky[:12]))
    else:
        print("Nothing repeated sits below 0.70. The model is coping with this")
        print("vocabulary; the glossary will earn its keep on course-specific")
        print("names rather than on general terminology.")

    if args.write:
        # keyterms API limits: 100 terms, 50 chars each
        terms = [r["term"] for r in rows if 5 <= len(r["term"]) <= 50][:100]
        Path(args.write).write_text("\n".join(terms) + "\n", encoding="utf-8")
        print(f"\nwrote {len(terms)} terms -> {args.write}")


if __name__ == "__main__":
    main()