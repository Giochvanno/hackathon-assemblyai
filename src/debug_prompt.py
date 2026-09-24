"""
Show the prompt we send and the raw reply we get back.

    python src/debug_prompt.py

The filter rejects every word and reports no error, which means the model
answers and we fail to read it. The only way forward is to look at what it
actually said.
"""

import json
import os
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from terms import GATEWAY, MODEL, build_prompt, parse_reply

# A deliberate mix: rare terms, ordinary words, and the hard middle —
# "class" and "stack" are everyday words that this subject redefines,
# "program" and "crash" are everyday words that it does not.
WORDS = ["scanf", "printf", "malloc", "ampersand", "segmentation",
         "class", "stack", "program", "crash", "output", "please",
         # Mishearings the soak test produced — these must be "no" —
         # and real proper nouns of the subject, which must NOT be caught by
         # the same rule. A fix that rejects Valgrind has broken something.
         "ZStack", "Zephyrus", "Valgrind", "POSIX",
         # What leaked on the soak test: ordinary words that must be "no" —
         # and the core terms that the stricter rule must still let through.
         "written", "depend", "treat", "memory",
         "pointer", "heap", "array", "union"]


def main() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    except ImportError:
        pass

    key = os.environ.get("ASSEMBLYAI_API_KEY")
    if not key:
        sys.exit("No ASSEMBLYAI_API_KEY")

    # The production prompt. No sentences here: this tool is for seeing the
    # raw reply and the parse, and a bare word is the harder case.
    prompt = build_prompt("C programming", WORDS)

    print("=" * 70)
    print("PROMPT SENT")
    print("=" * 70)
    print(prompt)

    r = requests.post(
        GATEWAY,
        headers={"authorization": key, "content-type": "application/json"},
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 900,
            "temperature": 0,
        },
        timeout=60,
    )

    print()
    print("=" * 70)
    print(f"HTTP {r.status_code}")
    print("=" * 70)

    if r.status_code != 200:
        print(r.text[:1000])
        return

    body = r.json()
    content = body["choices"][0]["message"]["content"]

    print("RAW REPLY")
    print("-" * 70)
    print(repr(content))
    print("-" * 70)
    print(content)

    print()
    print("=" * 70)
    print("WHAT OUR PARSER MAKES OF IT")
    print("=" * 70)
    parsed = parse_reply(content, WORDS)
    if not parsed:
        print("  nothing — this is why every word is rejected")
    for w in WORDS:
        v = parsed.get(w.lower(), "<absent>")
        print(f"  {w:<15} {v!r}")

    usage = body.get("usage")
    if usage:
        print(f"\ntokens: {usage}")


if __name__ == "__main__":
    main()