"""
One request to the LLM Gateway, and a plain reading of what came back.

    python src/check_gateway.py

Answers the only question that matters right now: is this account allowed to
use the gateway at all, or are we being turned away for some other reason?
A 429 can mean "too fast" or "no quota", and those need different responses.
"""

import json
import os
import sys
import time
from pathlib import Path

import requests

GATEWAY = "https://llm-gateway.assemblyai.com/v1/chat/completions"
MODELS = ["qwen3.5-4b-32k-fast", "gpt-5-nano", "gemini-2.5-flash-lite"]

MEANING = {
    200: "works — the account can use the gateway",
    401: "the key was rejected. Wrong key, or it has been revoked",
    403: "the key is valid but this account has no access to the gateway",
    404: "wrong endpoint or unknown model name",
    429: "turned away: either too many requests, or no quota left for this service",
}


def main() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    except ImportError:
        pass

    key = os.environ.get("ASSEMBLYAI_API_KEY")
    if not key:
        sys.exit("No ASSEMBLYAI_API_KEY. Put it in .env next to requirements.txt")

    print(f"key: {key[:6]}…{key[-4:]}  ({len(key)} chars)\n")

    for i, model in enumerate(MODELS):
        if i:
            time.sleep(2)  # don't create the very problem we are diagnosing

        try:
            r = requests.post(
                GATEWAY,
                headers={"authorization": key, "content-type": "application/json"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": "Reply with the word OK."}],
                    "max_tokens": 5,
                },
                timeout=30,
            )
        except requests.RequestException as exc:
            print(f"{model:<24} network error: {exc}")
            continue

        note = MEANING.get(r.status_code, "unexpected status")
        print(f"{model:<24} HTTP {r.status_code} — {note}")

        if r.status_code == 200:
            try:
                said = r.json()["choices"][0]["message"]["content"].strip()
                print(f"{'':<24} model said: {said!r}")
            except Exception:
                print(f"{'':<24} odd body: {r.text[:200]}")
        else:
            body = r.text.strip()
            if body:
                print(f"{'':<24} {body[:300]}")
            for h in ("retry-after", "x-ratelimit-remaining", "x-ratelimit-reset"):
                if h in r.headers:
                    print(f"{'':<24} {h}: {r.headers[h]}")

    print("\n--- for comparison, the transcription API on the same key ---")
    try:
        t = requests.get(
            "https://streaming.assemblyai.com/v3/token",
            headers={"authorization": key},
            params={"expires_in_seconds": 60},
            timeout=15,
        )
        print(f"streaming token          HTTP {t.status_code}"
              f"{' — fine' if t.status_code == 200 else ' — ' + t.text[:200]}")
    except requests.RequestException as exc:
        print(f"streaming token          network error: {exc}")

    print(
        "\nIf streaming works and the gateway does not, the key is fine and the\n"
        "gateway is the thing that is unavailable."
    )


if __name__ == "__main__":
    main()