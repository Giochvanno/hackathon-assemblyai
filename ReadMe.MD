# Lecture Lens

A voice agent that follows a lecture with you and tells you what you don't know,
while the lecturer is still talking.

Built on AssemblyAI for the Voice Agent Hackathon, September 2026.

## The problem

You are studying in a second language. The lecturer says a word you have never
met. By the time you have worked out whether it mattered, three sentences have
passed, and the rest of the lecture builds on the part you missed.

Subtitles don't help: you heard the words. Translation doesn't help: the sentence
wasn't the problem. Recording doesn't help: it's already too late.

The gap is that you don't know which word you don't know until you are already
behind.

## What it does

Lecture Lens keeps a record of every term your course has used. When a word
arrives that isn't in that record, it is new to you, and a one-line definition
appears next to it immediately.

The definition is one line by design. Anything longer costs you the next thirty
seconds of the lecture, which is what the tool is supposed to save.

The record grows from lecture to lecture, and it is also sent back to AssemblyAI
as keyterms, so recognition of your course's own vocabulary improves over time.

## Measured results

Real audio: a C programming lecture recorded on a phone in a Korean university
lecture hall.

**Latency**, speech to transcript on screen:

| | |
|---|---|
| median | 262 ms |
| 95th percentile | 463 ms |
| fastest | 166 ms |

Measured against the audio timeline, not the send time — see the comment in
`src/stream.py` for why the obvious method gives a wrong answer.

**Recognition**, same audio file, run twice, the only difference being the
course glossary:

```
without glossary:  Still many of you does not finish the scanner.
with glossary:     Still many of you does not finish the scanf.
```

`scanf` was transcribed as "scanner", "scan app" and "scanned" seven times
across the lecture. With the glossary it was recognised correctly every time.
`printf` improved as well.

## How it works

Audio never passes through our server. The browser gets a short-lived token and
opens its own socket to AssemblyAI, which is why the transcript comes back in a
quarter of a second. Everything slower runs beside that loop.

```
browser ──audio──► AssemblyAI ──transcript──► browser
   │                                              │
   └──────────► server ◄──────────────────────────┘
                  token · course memory · term filter
```

A finished turn goes through two filters:

1. A word list rejects ordinary speech. Cheap, instant, and too blunt on its
   own — on real audio it called `finish`, `raise` and `progress` terms.
2. AssemblyAI's LLM Gateway decides whether what survived is really a term for
   this subject, and returns the definition in the same call.

Words the second filter rejects are removed from the course memory, not just
hidden. They would otherwise be sent as keyterms, and common words in that list
make recognition worse rather than better.

AssemblyAI features used: Universal-Streaming, keyterms prompting with
mid-session `UpdateConfiguration`, and the LLM Gateway.

## Setup

Requires Python 3.11+.

```bash
git clone <repo>
cd lecture-lens
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env        # then put your key in it
```

Get a key at [assemblyai.com/dashboard/api-keys](https://www.assemblyai.com/dashboard/api-keys).
The free tier covers 333 hours of streaming, which is more than enough.

## Running

```bash
cd src
uvicorn server:app --port 8000
```

Open http://localhost:8000 and press "Start listening".

Microphone access needs either localhost or HTTPS — browsers refuse it on plain
HTTP.

### Working from a recording

```bash
python src/convert.py lecture.m4a --start 00:20:00 --duration 10:00
python src/stream.py --file samples/lecture-slice.wav
python src/candidates.py transcripts/lecture-slice-*.json
```

`convert.py` finds ffmpeg on PATH, or falls back to the binary bundled with
`imageio-ffmpeg`, so no separate install is needed.

Replay runs at real speed on purpose. Sending the file at once would let the
server work faster than a person speaks, and the latency figures would be
fiction.

## Layout

```
src/
  server.py      token minting, course memory over HTTP
  glossary.py    what this course has already taught
  terms.py       term filter and definitions, via the LLM Gateway
  stream.py      terminal client: streaming, latency, transcript capture
  convert.py     any recording to 16 kHz mono WAV
  candidates.py  transcript analysis: which words the model struggled with
  seed.py        build a course memory from existing transcripts
web/
  index.html     the browser client
```

## Limitations

**Streaming is English-first.** Universal-3.5 Pro Streaming covers 18 languages;
Korean and Russian are not among them. Recorded audio can use Universal-2, which
covers 99, but not in real time.

**Some errors no glossary can fix.** "answer the question" came out as "menu and
serve question" — ordinary words, mangled by accent and distance. Keyterms only
help where the problem is vocabulary.

**Multi-word terms are handled as separate words.** "segmentation fault" enters
the glossary as two entries.

**Term detection is permissive.** A word that is unusual but not technical can
still reach the model for judgement. Wrong in the cheap direction: a needless
definition costs a glance, a missing one costs the lecture.

## License

MIT