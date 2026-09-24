"""
Shared setup: put src/ on the path, and give every test a server with a
predictable model behind it.

The real judge calls a language model over the network. A test that did that
would be slow, would cost money, and — worst — would fail for reasons that have
nothing to do with the code under test. So the model is replaced by a stub that
answers exactly as instructed, and the tests check what OUR code does with the
answer, which is where every bug in this project has actually lived.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import terms as terms_module          # noqa: E402


class FakeModel:
    """
    Stands in for the gateway. Three behaviours, because three things happen in
    practice and each one broke the product at least once:

        defines  — the word is a term, here is its definition
        rejects  — the word is ordinary
        omits    — the reply simply does not mention the word at all

    The third is not hypothetical. It is how `getchar` and `putchar` vanished.
    """

    def __init__(self, rejects=(), omits=(), fails=False):
        self.rejects = {w.lower() for w in rejects}
        self.omits = {w.lower() for w in omits}
        self.fails = fails
        self.asked: list[list[str]] = []
        self.contexts: dict[str, str | None] = {}   # the sentence each word was judged by

    def __call__(self, judge, words, contexts=None):
        self.asked.append(list(words))
        for w in words:
            self.contexts[w] = (contexts or {}).get(w)
        if self.fails:
            raise RuntimeError("gateway did not answer")
        lines = [
            f"{w} = no" if w.lower() in self.rejects else f"{w} = a term in this subject"
            for w in words
            if w.lower() not in self.omits
        ]
        return terms_module.parse_reply("\n".join(lines), words)


@pytest.fixture
def model(monkeypatch):
    """A model that defines everything. Tests narrow it by rebuilding it."""
    def install(**kwargs):
        fake = FakeModel(**kwargs)
        monkeypatch.setattr(terms_module.TermJudge, "_ask",
                            lambda self, words, contexts=None: fake(self, words, contexts))
        return fake
    return install


@pytest.fixture
def judge(tmp_path, model):
    """A TermJudge writing its cache into a throwaway directory."""
    model()
    j = terms_module.TermJudge("C programming", cache_dir=str(tmp_path),
                               course="c-programming")
    j.api_key = "test-key"
    return j


@pytest.fixture
def srv(tmp_path, monkeypatch):
    """
    The server module with its global state emptied and its data directory
    pointed somewhere disposable.

    server.py keeps courses, judges and queues in module-level dicts, so tests
    would otherwise leak into each other in whatever order pytest picked.
    """
    import server

    monkeypatch.setattr(server, "DATA_DIR", tmp_path / "glossary")
    monkeypatch.setattr(server, "SEED_DIR", tmp_path / "seed")
    for store in (server._courses, server._judges, server._pending,
                  server._resolved, server._introduced, server._attempts,
                  server._declared):
        store.clear()

    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "test-key")
    yield server

    for store in (server._courses, server._judges, server._pending,
                  server._resolved, server._introduced, server._attempts,
                  server._declared):
        store.clear()

@pytest.fixture
def explainer(monkeypatch):
    """
    Stands in for the model behind the "I'm lost" button.

    lost.py binds ask_gateway at import time, so the patch goes on lost's own
    name. Patching terms.ask_gateway would leave lost still holding the real
    one — quietly, and only in this one place.
    """
    import lost

    def install(reply="Here is the thread.", fails=False):
        calls = []

        def fake(prompt, api_key, max_tokens=900):
            calls.append(prompt)
            if fails:
                raise RuntimeError("gateway did not answer")
            return reply

        monkeypatch.setattr(lost, "ask_gateway", fake)
        return calls

    return install
