"""
Feed saved transcripts through the course memory, in order.

Shows what the glossary panel would actually fill with on real speech, before
the browser is working. Also the way to build a course history from recordings
you already have.

    python src/seed.py c-programming transcripts/lec1.json transcripts/lec2.json

Pass them oldest first — the point is to see the second lecture recognise what
the first one taught.
"""

import argparse
import json
from pathlib import Path

from glossary import CourseGlossary


def turns_of(path: Path) -> list[str]:
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        return [t["text"] for t in data.get("turns", []) if t.get("text")]
    return [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("course")
    ap.add_argument("transcripts", nargs="+", help="oldest first")
    ap.add_argument("--reset", action="store_true", help="start from an empty memory")
    ap.add_argument("--show", type=int, default=25, help="new terms to print per file")
    args = ap.parse_args()

    g = CourseGlossary(args.course)
    if args.reset:
        g.terms = {}

    for path in map(Path, args.transcripts):
        if not path.exists():
            print(f"missing: {path}")
            continue

        before = len(g.terms)
        fresh: list[str] = []
        lines = turns_of(path)
        for line in lines:
            fresh += g.observe_turn(line)

        print(f"\n=== {path.name}")
        print(f"    {len(lines)} turns, {len(fresh)} first-time terms, "
              f"memory {before} -> {len(g.terms)}")
        if fresh:
            shown = fresh[: args.show]
            print("    " + ", ".join(shown) + ("  …" if len(fresh) > args.show else ""))

    g.save()
    print(f"\n{g.stats()}")
    print(f"saved -> {g.path}")

    top = g.keyterms(20)
    if top:
        print("\ntop of the glossary (what gets sent as keyterms):")
        for t in top:
            print(f"    {t}  ×{g.terms[t.lower()]['count'] if t.lower() in g.terms else '?'}")


if __name__ == "__main__":
    main()