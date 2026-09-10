"""
Brick 1 — streaming transcription with latency measurement.

Run:
    export ASSEMBLYAI_API_KEY=...      # never hardcode the key
    python src/stream.py               # microphone
    python src/stream.py --file a.wav  # replay a file (repeatable tests)
"""

import argparse
import json
import os
import queue
import shutil
import statistics
import sys
import threading
import time
import urllib.parse
import wave

import websocket  # pip install websocket-client

ENDPOINT = "wss://streaming.assemblyai.com/v3/ws"
SAMPLE_RATE = 16_000
FRAME_MS = 50
FRAME_BYTES = int(SAMPLE_RATE * 2 * FRAME_MS / 1000)  # 16-bit mono PCM


def build_url(keyterms: list[str] | None = None) -> str:
    """Connection URL. keyterms_prompt must be a JSON-stringified array on raw WS."""
    params = {
        "speech_model": "universal-3-5-pro",
        "sample_rate": SAMPLE_RATE,
        "encoding": "pcm_s16le",
        "format_turns": "true",  # required for keyterm prompting
    }
    if keyterms:
        # API limits: max 100 terms, each <= 50 chars.
        cleaned = [t for t in keyterms if 1 <= len(t) <= 50][:100]
        params["keyterms_prompt"] = json.dumps(cleaned)
    return f"{ENDPOINT}?{urllib.parse.urlencode(params)}"


class LatencyTracker:
    """
    Time between a word being SPOKEN and its transcript arriving.

    The naive version — "now minus last audio send" — is wrong: audio streams
    continuously, so the last send is never more than one frame ago and you
    measure the frame size, not the model.

    Instead compare two clocks:
      * audio timeline — `words[].end`, ms since the session's first sample
      * wall clock     — ms since we sent that first sample
    Their difference is the real lag.
    """

    def __init__(self) -> None:
        self.samples_ms: list[float] = []
        self.stream_start: float | None = None
        self.audio_ms_sent = 0.0

    def mark_audio(self, chunk_bytes: int) -> None:
        if self.stream_start is None:
            self.stream_start = time.monotonic()
        # bytes -> ms, for 16-bit mono at SAMPLE_RATE
        self.audio_ms_sent += chunk_bytes / 2 / SAMPLE_RATE * 1000

    def record(self, word_end_ms: float) -> float | None:
        if self.stream_start is None:
            return None
        elapsed_ms = (time.monotonic() - self.stream_start) * 1000
        ms = elapsed_ms - word_end_ms
        if ms < 0:  # file replay can outrun the wall clock
            return None
        self.samples_ms.append(ms)
        return ms

    def report(self) -> str:
        if not self.samples_ms:
            return "no samples"
        s = sorted(self.samples_ms)
        p50 = statistics.median(s)
        p95 = s[min(len(s) - 1, int(len(s) * 0.95))]
        return f"n={len(s)}  p50={p50:.0f}ms  p95={p95:.0f}ms  max={s[-1]:.0f}ms"


class Session:
    def __init__(self, api_key: str, keyterms: list[str] | None = None) -> None:
        self.api_key = api_key
        self.keyterms = keyterms or []
        self.latency = LatencyTracker()
        self.audio_q: queue.Queue[bytes | None] = queue.Queue()
        self.ws: websocket.WebSocketApp | None = None
        self.finals: list[str] = []
        self._line_len = 0  # width of the partial currently on screen
        self._last_word_end: float | None = None  # audio-timeline ms, for latency

    def _overwrite(self, text: str, newline: bool) -> None:
        """
        Redraw the current terminal line.

        Two traps here:
        1. `\\r` moves the cursor home but does not erase, so a short line
           landing on a long one leaves the tail behind ("diagnosis.ust").
           Fixed by padding to the previous width.
        2. `\\r` only rewinds ONE visual line. A partial wider than the
           terminal wraps, and the earlier rows stay on screen forever.
           Fixed by truncating partials to the terminal width.
        """
        if not newline:
            width = shutil.get_terminal_size((100, 24)).columns - 1
            if len(text) > width:
                text = text[: width - 1] + "…"

        pad = max(0, self._line_len - len(text))
        print("\r" + text + " " * pad, end="\n" if newline else "", flush=True)
        self._line_len = 0 if newline else len(text)

    # --- websocket callbacks -------------------------------------------------
    def on_open(self, ws: websocket.WebSocketApp) -> None:
        print("[session] connected", flush=True)
        threading.Thread(target=self._pump_audio, args=(ws,), daemon=True).start()

    def on_message(self, ws: websocket.WebSocketApp, raw: str) -> None:
        msg = json.loads(raw)
        kind = msg.get("type")

        if kind == "Begin":
            print(f"[session] id={msg.get('id')}", flush=True)

        elif kind == "Turn":
            text = msg.get("transcript", "")

            # Word timings ride on the UNformatted turns; the formatted final
            # that follows has none. Remember the latest we saw so the final
            # still gets a number.
            words = msg.get("words") or []
            if words and words[-1].get("end") is not None:
                self._last_word_end = words[-1]["end"]

            if not text:
                return

            if msg.get("end_of_turn"):
                self.finals.append(text)
                ms = (
                    self.latency.record(self._last_word_end)
                    if self._last_word_end is not None
                    else None
                )
                stamp = f"[{ms:4.0f}ms]" if ms is not None else "[  --ms]"
                self._overwrite(f"{stamp} {text}", newline=True)
                self._last_word_end = None
            else:
                self._overwrite(f"        … {text}", newline=False)

        elif kind == "Termination":
            print(
                f"\n[session] audio={msg.get('audio_duration_seconds')}s "
                f"billed={msg.get('session_duration_seconds')}s",
                flush=True,
            )

    def on_error(self, ws: websocket.WebSocketApp, err: Exception) -> None:
        print(f"[error] {err}", file=sys.stderr, flush=True)

    def on_close(self, ws: websocket.WebSocketApp, code, reason) -> None:
        print(f"\n[session] closed {code} {reason}", flush=True)
        print(f"[latency] {self.latency.report()}", flush=True)

    # --- audio ---------------------------------------------------------------
    def _pump_audio(self, ws: websocket.WebSocketApp) -> None:
        while True:
            chunk = self.audio_q.get()
            if chunk is None:
                break
            try:
                self.latency.mark_audio(len(chunk))
                ws.send(chunk, websocket.ABNF.OPCODE_BINARY)
            except Exception as exc:  # socket already closing
                print(f"[audio] stopped: {exc}", file=sys.stderr)
                break
        try:
            ws.send(json.dumps({"type": "Terminate"}))
        except Exception:
            pass

    def update_keyterms(self, terms: list[str]) -> None:
        """Glossary grows mid-lecture — no reconnect needed."""
        if not self.ws:
            return
        cleaned = [t for t in terms if 1 <= len(t) <= 50][:100]
        self.ws.send(
            json.dumps({"type": "UpdateConfiguration", "keyterms_prompt": cleaned})
        )
        print(f"[glossary] {len(cleaned)} terms active", flush=True)

    def run(self) -> None:
        self.ws = websocket.WebSocketApp(
            build_url(self.keyterms),
            header={"Authorization": self.api_key},
            on_open=self.on_open,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close,
        )
        self.ws.run_forever()


# --- audio sources -----------------------------------------------------------
def feed_microphone(session: Session) -> None:
    import sounddevice as sd  # pip install sounddevice

    def callback(indata, frames, time_info, status):
        if status:
            print(f"[mic] {status}", file=sys.stderr)
        session.audio_q.put(bytes(indata))

    with sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        blocksize=FRAME_BYTES // 2,
        dtype="int16",
        channels=1,
        callback=callback,
    ):
        print("[mic] listening — Ctrl+C to stop", flush=True)
        threading.Event().wait()


def feed_file(session: Session, path: str) -> None:
    """Replay a 16kHz mono WAV in real time. Same input every run = comparable numbers."""
    with wave.open(path, "rb") as wav:
        assert wav.getframerate() == SAMPLE_RATE, "resample to 16kHz mono first"
        while chunk := wav.readframes(FRAME_BYTES // 2):
            session.audio_q.put(chunk)
            time.sleep(FRAME_MS / 1000)
    session.audio_q.put(None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", help="16kHz mono WAV to replay instead of the mic")
    ap.add_argument("--keyterms", help="comma-separated starting glossary")
    args = ap.parse_args()

    # Read .env from the project root if it exists, so the key never lives in code.
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass  # env var may still be set by hand

    api_key = os.environ.get("ASSEMBLYAI_API_KEY")
    if not api_key:
        sys.exit(
            "No API key found.\n"
            "  Put ASSEMBLYAI_API_KEY=<key> in a .env file next to this project,\n"
            "  or set it in your shell. Never paste it into the source."
        )

    terms = [t.strip() for t in args.keyterms.split(",")] if args.keyterms else []
    session = Session(api_key, terms)

    source = threading.Thread(
        target=feed_file if args.file else feed_microphone,
        args=(session, args.file) if args.file else (session,),
        daemon=True,
    )

    threading.Thread(target=source.start, daemon=True).start()
    try:
        session.run()
    except KeyboardInterrupt:
        session.audio_q.put(None)
        time.sleep(0.5)


if __name__ == "__main__":
    main()