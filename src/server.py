"""
The slow loop: token minting and course memory.

Audio never passes through here. The browser opens its own socket to
AssemblyAI with a short-lived token this server signs, so the transcript
comes back in a quarter of a second instead of taking a detour.

    uvicorn server:app --reload --port 8000   (run from src/)
    open http://localhost:8000
"""

import os
from pathlib import Path

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from glossary import CourseGlossary
from terms import TermJudge

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

API_KEY = os.environ.get("ASSEMBLYAI_API_KEY")
TOKEN_URL = "https://streaming.assemblyai.com/v3/token"
WEB_DIR = Path(__file__).resolve().parent.parent / "web"

app = FastAPI(title="Lecture Lens")

# One glossary and one judge per course, kept in memory, written through to disk.
_courses: dict[str, CourseGlossary] = {}
_judges: dict[str, TermJudge] = {}


def course(name: str) -> CourseGlossary:
    if name not in _courses:
        _courses[name] = CourseGlossary(name)
    return _courses[name]


def judge(name: str, subject: str | None = None) -> TermJudge:
    if name not in _judges:
        _judges[name] = TermJudge(subject or name)
    return _judges[name]


class TurnIn(BaseModel):
    course: str = "default"
    text: str
    # What the course is about, so the judge knows what counts as a term here.
    subject: str | None = None


class TermsIn(BaseModel):
    course: str = "default"


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
    Feed one finished turn. Returns the terms met for the first time in this
    course, with a one-line definition each — the whole product in one endpoint.

    Two filters in sequence, and the order matters:

      1. the word list rejects obvious noise, cheaply and instantly
      2. the model decides whether what survives is really a term

    Words the model rejects are FORGOTTEN, not just hidden. Otherwise "finish"
    and "raise" would sit in the course memory forever and, worse, get sent as
    keyterms — and AssemblyAI warns that common words in that list make
    recognition worse, not better. The filter protects the transcript, not only
    the sidebar.
    """
    g = course(turn.course)
    fresh = g.observe_turn(turn.text)

    defined: dict[str, str] = {}
    if fresh:
        try:
            defined = judge(turn.course, turn.subject).judge(fresh)
        except Exception as exc:
            print(f"[observe] judging unavailable: {exc}")
            defined = {t: "" for t in fresh}  # degrade to no definitions

        for term in fresh:
            if term not in defined:
                g.forget(term)

    g.save()
    return {
        "new_terms": defined,
        "known_count": len(g.terms),
        "keyterms": g.keyterms(),
    }


@app.post("/api/keyterms")
def keyterms(body: TermsIn) -> dict:
    """What the course already knows — sent to the socket as it opens."""
    g = course(body.course)
    return {"keyterms": g.keyterms(), "known_count": len(g.terms)}


@app.get("/")
def index() -> FileResponse:
    page = WEB_DIR / "index.html"
    if not page.exists():
        raise HTTPException(404, "web/index.html is missing")
    return FileResponse(page)