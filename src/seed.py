"""
Teach a course its first lecture, and snapshot it into the repository.

    python src/seed.py c-programming --subject "C programming" --text "In C, printf writes to standard output."
    python src/seed.py c-programming --subject "C programming" transcripts/lec1.json
    python src/seed.py c-programming --snapshot

Everything goes through a RUNNING server's /api/observe rather than straight
into CourseGlossary. That is the whole point of this rewrite: the glossary's own
filter is a word list, and on real speech it admits "program", "crash" and
"output". Only the server runs the second filter — the model — and only the
server drops what that filter rejects. A seed built without it becomes the face
of production with the junk baked in.

No microphone needed. Typed text goes through exactly the same path as speech,
minus the transcription errors, which is what you want in a seed anyway.
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import requests

# Import rather than recompute: DATA_DIR is defined once, in server.py, and a
# second copy of that rule here is how the snapshot ends up reading the wrong
# folder six weeks from now.
from server import DATA_DIR, SEED_DIR

DEFAULT_SERVER = "http://localhost:8000"
SETTLE_S = 1.5          # between turns, so the worker batches them as it would live
PATIENCE_S = 90         # how long to wait for the queue to drain


def lines_of(path: Path) -> list[str]:
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        return [t["text"] for t in data.get("turns", []) if t.get("text")]
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def snapshot() -> None:
    """Copy what the course has learned into the repository."""
    files = sorted(DATA_DIR.glob("*.json"))
    if not files:
        sys.exit(f"nothing to snapshot: {DATA_DIR} is empty")

    SEED_DIR.mkdir(parents=True, exist_ok=True)
    for src in files:
        shutil.copy2(src, SEED_DIR / src.name)
        print(f"  {src.name}  ({src.stat().st_size} bytes)")
    print(f"\nsnapshot -> {SEED_DIR}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("course", nargs="?", help="course name, e.g. c-programming")
    ap.add_argument("sources", nargs="*", type=Path, help="transcript files, oldest first")
    ap.add_argument("--text", action="append", default=[], help="a sentence, repeatable")
    ap.add_argument("--subject", help="what the course is about, for the judge")
    ap.add_argument("--server", default=DEFAULT_SERVER)
    ap.add_argument("--snapshot", action="store_true",
                    help="copy the course memory into seed/ and exit")
    args = ap.parse_args()

    if args.snapshot:
        snapshot()
        return

    if not args.course:
        ap.error("course is required unless --snapshot")

    turns = list(args.text)
    for path in args.sources:
        if not path.exists():
            sys.exit(f"missing: {path}")
        turns += lines_of(path)
    if not turns:
        ap.error("nothing to feed: pass --text or a transcript file")

    try:
        requests.get(f"{args.server}/", timeout=5)
    except requests.RequestException:
        sys.exit(f"no server at {args.server} — start it first:\n"
                 f"    uvicorn server:app --reload --port 8000   (from src/)")

    queued = 0
    for i, text in enumerate(turns, 1):
        r = requests.post(
            f"{args.server}/api/observe",
            json={"course": args.course, "subject": args.subject, "text": text},
            timeout=30,
        )
        r.raise_for_status()
        body = r.json()
        pending = body.get("pending", [])
        queued += len(pending)
        print(f"{i:>3}. {text[:64]}{'…' if len(text) > 64 else ''}")
        if pending:
            print(f"     queued: {', '.join(pending)}")
        time.sleep(SETTLE_S)

    # The verdicts come back through the same endpoint the browser polls, so
    # this waits exactly as long as a student would.
    print(f"\nwaiting for {queued} word(s) to be judged…")
    kept: dict[str, str] = {}
    deadline = time.monotonic() + PATIENCE_S

    while time.monotonic() < deadline:
        time.sleep(2)
        u = requests.get(f"{args.server}/api/updates",
                         params={"course": args.course}, timeout=15).json()
        for t in u.get("terms", []):
            kept[t["term"]] = t.get("definition", "")
            if t.get("note"):
                print(f"  ! {t['term']}: {t['note']}")
        if not u.get("waiting"):
            break
    else:
        print("  (gave up waiting; the rest will land in the glossary anyway)")

    print(f"\n{len(kept)} term(s) kept, {queued - len(kept)} rejected as ordinary\n")
    for term in sorted(kept):
        print(f"  {term:<16} {kept[term]}")

    final = requests.post(f"{args.server}/api/keyterms",
                          json={"course": args.course}, timeout=15).json()
    print(f"\ncourse memory: {final['known_count']} terms, "
          f"{len(final['keyterms'])} settled enough to send as keyterms")
    print(f"stored in {DATA_DIR}")
    print("\nHappy with it? Snapshot it into the repository:")
    print("    python src/seed.py --snapshot")


if __name__ == "__main__":
    main()