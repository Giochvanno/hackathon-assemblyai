"""
The slow loop: token minting and course memory.

Audio never passes through here. The browser opens its own socket to
AssemblyAI with a short-lived token this server signs, so the transcript
comes back in a quarter of a second instead of taking a detour.

    uvicorn server:app --reload --port 8000   (run from src/)
    open http://localhost:8000

Judging terms is deliberately NOT done inside the request that reports them.
A lecture finishes a turn every three or four seconds; one gateway call per
turn outruns the rate limit within half a minute, and every retry pushes the
queue further behind the speaker. Measured on a 35-second test: the last turn
was answered 36 seconds after it was spoken — long after the lecturer had
moved on, which is the exact failure this product exists to prevent.

So the request answers instantly with what is already known, and a single
background worker asks the model about the rest in batches. Answers come back
through /api/updates a second or two later, still while the sentence is on
screen.
"""

import asyncio
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

import lost
from glossary import CourseGlossary
from terms import BATCH, TermJudge

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

ROOT = Path(__file__).resolve().parent.parent

API_KEY = os.environ.get("ASSEMBLYAI_API_KEY")
TOKEN_URL = "https://streaming.assemblyai.com/v3/token"
WEB_DIR = ROOT / "web"

# Where the course memory lives. Anchored to the project, not to the working
# directory: locally the server starts in src/ and in production somewhere else
# entirely, and a relative path would quietly point at a different, empty
# glossary in each — no error, no terms, no way to tell why. DATA_DIR lets a
# host mount real storage over it.
DATA_DIR = Path(os.environ.get("DATA_DIR") or ROOT / "glossary")

# A course the project ships with, committed to the repository.
#
# Free hosting gives a service a filesystem that is wiped on every restart, and
# a product whose whole claim is "this course already taught you that word"
# cannot open with an empty memory. So the first lecture lives in the repository
# and is copied into place on boot. What the course learns after that survives
# until the next restart and no longer — an honest limitation, not a hidden one.
SEED_DIR = ROOT / "seed"

# How often the worker empties the queue. Short enough that a definition still
# lands while its sentence is on screen, long enough that several turns share
# one call instead of each buying its own rate-limit penalty.
FLUSH_INTERVAL_S = 2.0

# One glossary and one judge per course, kept in memory, written through to disk.
_courses: dict[str, CourseGlossary] = {}
_judges: dict[str, TermJudge] = {}

# Words waiting to be judged, and verdicts waiting to be collected.
# dict, not list: it keeps insertion order and drops duplicates for free, so a
# word repeated three times in one breath costs one question, not three.
_pending: dict[str, dict[str, None]] = {}
_resolved: dict[str, list[dict]] = {}

# Every term this run has introduced, with the moment it happened. The glossary
# knows THAT a term is known; this knows WHEN the student first met it, which
# is the only way to answer "what landed on me in the last three minutes".
# Kept in memory on purpose: it describes one sitting, not the course.
_introduced: dict[str, list[dict]] = {}

# Requests run in FastAPI's threadpool and the worker runs in its own thread;
# both read and write the structures above. One lock for all of them is enough
# at this scale and leaves no room for a half-applied update.
_lock = threading.Lock()


def record_introduction(course: str, term: str, definition: str) -> None:
    """Note that the student has just met this term for the first time."""
    with _lock:
        _introduced.setdefault(course, []).append(
            {"term": term, "definition": definition, "at": time.monotonic()}
        )


def glossary_for(name: str) -> CourseGlossary:
    if name not in _courses:
        _courses[name] = CourseGlossary(name, directory=str(DATA_DIR))
    return _courses[name]


def judge_for(name: str, subject: str | None = None) -> TermJudge:
    if name not in _judges:
        _judges[name] = TermJudge(subject or name, cache_dir=str(DATA_DIR), course=name)
    return _judges[name]


# --- the background worker ---------------------------------------------------

def flush(name: str) -> None:
    """
    Ask the model about one batch of queued words. Runs off the request path.
    """
    with _lock:
        queue = _pending.get(name)
        if not queue:
            return
        words = list(queue)[:BATCH]
        for w in words:
            queue.pop(w, None)

    g = glossary_for(name)
    j = judge_for(name)

    try:
        defined = j.judge(words)
    except Exception as exc:
        # The model could not be reached. Show the words undefined rather than
        # dropping them: "we could not explain this" is still worth more to the
        # student than silence, and the term stays in the course memory.
        print(f"[flush] {exc}")
        with _lock:
            _resolved.setdefault(name, []).extend(
                {"term": w, "definition": "", "note": str(exc)} for w in words
            )
        return

    # Three outcomes again, and the middle one is the trap. A word the reply
    # never mentioned is NOT a rejection — terms.py deliberately leaves it
    # uncached so it gets asked again — but treating it as one here deleted it
    # from the course memory anyway, quietly, before it ever got a second
    # chance. Ask the cache what actually happened instead of inferring it
    # from an absent key.
    _, rejected, unjudged = j.cached(words)

    out = []
    for w in words:
        if w in defined:
            out.append({"term": w, "definition": defined[w], "note": ""})
            record_introduction(name, w, defined[w])
        elif w in rejected:
            # Judged and rejected: drop it, or "finish" and "raise" sit in the
            # course memory forever and go out as keyterms. AssemblyAI warns
            # that common words in that list make recognition worse, so the
            # filter is protecting the transcript, not just the sidebar.
            g.forget(w)

    if unjudged:
        print(f"[flush] no verdict yet for {', '.join(unjudged)} — keeping them")

    g.save()
    with _lock:
        if out:
            _resolved.setdefault(name, []).extend(out)


async def worker() -> None:
    while True:
        await asyncio.sleep(FLUSH_INTERVAL_S)
        with _lock:
            waiting = [n for n, q in _pending.items() if q]
        for name in waiting:
            try:
                await asyncio.to_thread(flush, name)
            except Exception as exc:              # never let the loop die
                print(f"[worker] {type(exc).__name__}: {exc}")


def plant_seed() -> None:
    """
    Copy the shipped course memory into place, once, for whatever is not there.

    Never overwrites: a file already in DATA_DIR is what this instance has
    learned, and it outranks the snapshot in the repository. So this is a floor
    under the memory, not a reset of it, and it is safe to run on every boot.
    """
    if not SEED_DIR.is_dir():
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for src in sorted(SEED_DIR.glob("*.json")):
        dst = DATA_DIR / src.name
        if dst.exists():
            continue
        dst.write_bytes(src.read_bytes())
        print(f"[seed] planted {src.name}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    plant_seed()
    task = asyncio.create_task(worker())
    yield
    task.cancel()


app = FastAPI(title="Lecture Lens", lifespan=lifespan)


class TurnIn(BaseModel):
    course: str = "default"
    text: str
    # What the course is about, so the judge knows what counts as a term here.
    subject: str | None = None


class TermsIn(BaseModel):
    course: str = "default"


class LostIn(BaseModel):
    course: str = "default"
    subject: str | None = None
    # The tail of the transcript, oldest first. The browser keeps it; this
    # server never stores a lecture, which is somebody else's speech.
    turns: list[str] = []
    window_s: float = 180.0


@app.get("/api/token")
def token(expires_in_seconds: int = 120) -> dict:
    """
    Mint a one-time streaming token.

    The API key stays on this side. The browser gets something that is useless
    a couple of minutes from now and only opens a single session.
    """
    if not API_KEY:
        raise HTTPException(500, "ASSEMBLYAI_API_KEY is not set on the server")

    r = requests.get(
        TOKEN_URL,
        headers={"authorization": API_KEY},
        params={"expires_in_seconds": max(1, min(600, expires_in_seconds))},
        timeout=10,
    )
    if r.status_code != 200:
        raise HTTPException(r.status_code, f"AssemblyAI refused: {r.text[:200]}")
    return r.json()


@app.post("/api/observe")
def observe(turn: TurnIn) -> dict:
    """
    Feed one finished turn. Returns, immediately:

      * new_terms — first-time terms this course has already judged before,
        with their definitions, ready to go on screen now
      * pending  — first-time words nobody has judged yet; their definitions
        arrive through /api/updates within a couple of seconds

    Nothing here touches the network, so the answer is as fast as the disk.
    """
    g = glossary_for(turn.course)
    j = judge_for(turn.course, turn.subject)

    fresh = g.observe_turn(turn.text)
    defined, rejected, unknown = j.cached(fresh)

    for w in defined:
        record_introduction(turn.course, w, defined[w])

    for w in rejected:
        g.forget(w)

    if unknown:
        with _lock:
            queue = _pending.setdefault(turn.course, {})
            for w in unknown:
                queue[w] = None

    g.save()
    return {
        "new_terms": defined,
        "pending": unknown,
        "known_count": len(g.terms),
        "keyterms": g.keyterms(),
        "note": "",
    }


@app.get("/api/updates")
def updates(course: str = "default") -> dict:
    """
    Collect the verdicts that came back since the last call, and forget them.

    Draining rather than accumulating keeps this endpoint honest: each verdict
    is handed over exactly once, so the browser never has to work out which of
    them it has already drawn.
    """
    g = glossary_for(course)
    with _lock:
        terms = _resolved.pop(course, [])
        waiting = len(_pending.get(course, {}))
    return {
        "terms": terms,
        "waiting": waiting,
        "known_count": len(g.terms),
        "keyterms": g.keyterms(),
    }


@app.post("/api/lost")
def im_lost(body: LostIn) -> dict:
    """
    The student pressed the button. Answer why, not what.

    The terms come from this server's own record of what the student has met
    for the first time; the words come from the browser, which is the only
    place a lecture is ever held. So the two halves of the answer are assembled
    from two different memories, and neither of them is a stored transcript.
    """
    now = time.monotonic()
    with _lock:
        events = list(_introduced.get(body.course, []))

    window = [e for e in events if now - e["at"] <= body.window_s]
    minutes = body.window_s / 60.0

    density = {
        "in_window": len(window),
        "per_minute": round(len(window) / minutes, 2) if minutes else 0.0,
        "session_per_minute": None,
    }
    # A rate is meaningless without something to compare it to. The lecture's
    # own average so far is the fairest baseline we have: it needs no extra
    # data and it adapts to how fast this particular lecturer introduces
    # things. Below half a minute of material it says nothing, so we say so.
    if events:
        span_min = (now - events[0]["at"]) / 60.0
        if span_min >= 0.5:
            density["session_per_minute"] = round(len(events) / span_min, 2)

    if not window:
        return {
            "terms": [],
            "why": "",
            "density": density,
            "note": "no new terms in this stretch — the gap is probably earlier",
        }

    terms = [{"term": e["term"], "definition": e["definition"]} for e in window]

    try:
        why = lost.explain(
            subject=body.subject or body.course,
            turns=body.turns,
            terms=terms,
            minutes=minutes,
            api_key=API_KEY,
        )
        note = ""
    except Exception as exc:
        # The list of terms is the finding; the paragraph is the polish. Losing
        # the second is no reason to withhold the first.
        print(f"[lost] {exc}")
        why, note = "", str(exc)

    return {"terms": terms, "why": why, "density": density, "note": note}


@app.post("/api/keyterms")
def keyterms(body: TermsIn) -> dict:
    """What the course already knows — sent to the socket as it opens."""
    g = glossary_for(body.course)
    return {"keyterms": g.keyterms(), "known_count": len(g.terms)}


@app.get("/")
def index() -> FileResponse:
    page = WEB_DIR / "index.html"
    if not page.exists():
        raise HTTPException(404, "web/index.html is missing")
    return FileResponse(page)