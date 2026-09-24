"""
Answer the question the glossary cannot: why did this stop making sense?

The sidebar tells a student what a word means. It does not tell them why five
words arrived in three minutes, which is the actual reason a lecture slips
away — not one unknown term, but a burst of them landing faster than anyone
can absorb.

So "I'm lost" does not summarise the passage. Summarising is what any
transcript tool can do, and it answers the wrong question: the student heard
the words, they just lost the thread between them. This takes the terms that
this course had never used before, and asks what ties them together in what
was actually said.

Everything here rests on the course memory. Without it there is no way to tell
a term the student met for the first time from one the lecturer has used every
week since March, and the whole answer collapses back into a summary.
"""

from terms import ask_gateway

# Four sentences, because the student is reading this while the lecturer keeps
# talking. The same rule as the one-line definitions, only slightly relaxed:
# the point is to get them back on the thread, not to re-teach the passage.
MAX_TOKENS = 320

# How much of the transcript to send. Long enough to hold the thread, short
# enough that the answer arrives while the passage is still on screen.
MAX_TRANSCRIPT_CHARS = 2000

PROMPT = """Subject: {subject}

A student is following this lecture with a live glossary beside it. They have
just pressed "I am lost". This is what the lecturer said in the last
{minutes:.0f} minutes:

{transcript}

These are the terms in that passage that this course had never used before,
with the one-line definitions the student has already been shown:

{terms}

Write at most four sentences, addressed to the student as "you".

Explain how those terms fit together in what was just said, and name the one
step that most likely lost them. Do not define the terms again. Do not
summarise the passage — they heard it. Give them the thread, not the content.
"""


def recent_transcript(turns: list[str]) -> str:
    """
    The tail of what was said, newest kept.

    Trimmed from the front: if something has to go, it should be the oldest
    part, because the confusion is at the end — that is where the button was
    pressed.
    """
    text = "\n".join(t.strip() for t in turns if t and t.strip())
    if len(text) <= MAX_TRANSCRIPT_CHARS:
        return text
    return "…" + text[-MAX_TRANSCRIPT_CHARS:]


def explain(subject: str, turns: list[str], terms: list[dict],
            minutes: float, api_key: str | None) -> str:
    """
    Ask the model for the thread. Raises if the gateway cannot be reached —
    the caller still has the term list, which is worth showing on its own.
    """
    listed = "\n".join(
        f"  {t['term']} — {t['definition']}" if t.get("definition") else f"  {t['term']}"
        for t in terms
    )
    prompt = PROMPT.format(
        subject=subject,
        minutes=max(1.0, minutes),
        transcript=recent_transcript(turns),
        terms=listed,
    )
    return plain(ask_gateway(prompt, api_key, max_tokens=MAX_TOKENS))


def plain(text: str) -> str:
    """
    The model answers in markdown — **bold** terms, `code` — and the panel
    shows text, so on CS50 the student read literal asterisks. Keep the words,
    drop the markup.
    """
    for mark_ in ("**", "__", "`"):
        text = text.replace(mark_, "")
    return " ".join(text.split())
