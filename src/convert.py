"""
Convert any recording to the 16 kHz mono WAV the streaming API expects.

    python src/convert.py "C:\\Users\\me\\Recordings\\lecture.m4a"
    python src/convert.py lecture.mp4 -o samples/lecture.wav

Finds ffmpeg in this order:
  1. on PATH
  2. the binary bundled with imageio-ffmpeg  (pip install imageio-ffmpeg)

The second path exists so a Windows install never has to touch PATH.
"""

import argparse
import shutil
import subprocess
import sys
import wave
from pathlib import Path

SAMPLE_RATE = 16_000


def find_ffmpeg() -> str:
    on_path = shutil.which("ffmpeg")
    if on_path:
        return on_path

    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass

    raise SystemExit(
        "ffmpeg not found. Either:\n"
        "  winget install Gyan.FFmpeg      (then reopen the terminal)\n"
        "  pip install imageio-ffmpeg      (no PATH changes needed)"
    )


def convert(src: Path, dst: Path, start: str | None = None, dur: str | None = None) -> None:
    exe = find_ffmpeg()
    dst.parent.mkdir(parents=True, exist_ok=True)

    cmd = [exe, "-y"]
    # -ss before -i seeks without decoding everything first: instant on a 90-min file
    if start:
        cmd += ["-ss", start]
    cmd += ["-i", str(src)]
    if dur:
        cmd += ["-t", dur]
    cmd += [
        "-ar", str(SAMPLE_RATE),   # 16 kHz
        "-ac", "1",                # mono
        "-c:a", "pcm_s16le",       # 16-bit PCM, what the API wants
        "-vn",                     # drop video if the source is a recording
        str(dst),
    ]
    print(f"[ffmpeg] {exe}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        tail = "\n".join(result.stderr.strip().splitlines()[-6:])
        raise SystemExit(f"ffmpeg failed:\n{tail}")


def describe(path: Path) -> None:
    with wave.open(str(path), "rb") as w:
        secs = w.getnframes() / w.getframerate()
        print(
            f"[ok] {path}\n"
            f"     {w.getframerate()} Hz, {w.getnchannels()} ch, "
            f"{w.getsampwidth() * 8}-bit, {secs / 60:.1f} min, "
            f"{path.stat().st_size / 1e6:.1f} MB"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", help="any audio or video file")
    ap.add_argument("-o", "--out", help="output .wav (default: samples/<name>.wav)")
    ap.add_argument("--start", help="skip to this point, e.g. 00:10:00")
    ap.add_argument(
        "--duration",
        help="how much to keep, e.g. 10:00. Replay runs at 1x, so a slice "
        "costs its own length in wall clock and in streaming quota.",
    )
    args = ap.parse_args()

    src = Path(args.source).expanduser()
    if not src.exists():
        raise SystemExit(f"no such file: {src}")

    stem = src.stem
    if args.duration:
        stem += "-slice"
    dst = Path(args.out) if args.out else Path("samples") / f"{stem}.wav"
    convert(src, dst, args.start, args.duration)
    describe(dst)
    print(f"\nnext:\n  python src/stream.py --file {dst}")


if __name__ == "__main__":
    main()