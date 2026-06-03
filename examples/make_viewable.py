#!/usr/bin/env python3
"""
Turn a linear-master recording (from record.py) into a viewable MP4.

The master written by record.py is *scene-linear* 10-bit HEVC — correct for
ingest into Resolve/Nuke/processing, but it looks dark in normal players
because nothing tone-maps it. This script bakes in a tone curve so the result
plays correctly anywhere (QuickTime, Preview, browsers).

The tone curve is applied by the SAME code the live GUI uses
(``hteng_camera.convert.to_display``), so a file converted with
``--curve log --param 120`` looks pixel-for-pixel like the GUI did at those
settings — we don't re-derive the curve in ffmpeg's expression language.

Pipeline:
    linear MP4 --ffmpeg decode--> rgb48le frames (16-bit linear)
              --convert.to_display(curve,param)--> 8-bit RGB (tone-mapped)
              --ffmpeg encode--> 8-bit H.264, tagged bt709 (plays anywhere)

Usage:
    python make_viewable.py master.mp4 viewable.mp4
    python make_viewable.py --curve log --param 120 master.mp4 out.mp4
    python make_viewable.py --curve gamma --param 2.2 --ev 1.5 in.mp4 out.mp4
"""

import argparse
import subprocess
import sys

import numpy as np

from hteng_camera import convert


def _probe_size(path: str) -> tuple[int, int]:
    """Return (width, height) of the first video stream via ffprobe."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height",
         "-of", "csv=p=0:s=x", path],
        capture_output=True, text=True,
    )
    if out.returncode != 0 or "x" not in out.stdout:
        sys.exit(f"[error] could not probe {path}: {out.stderr.strip()}")
    w, h = out.stdout.strip().split("x")
    return int(w), int(h)


def _probe_fps(path: str) -> str:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=avg_frame_rate", "-of", "csv=p=0", path],
        capture_output=True, text=True,
    )
    fps = out.stdout.strip()
    return fps if fps and fps != "0/0" else "30"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Bake a tone curve into a linear-master recording.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("input", help="Linear master .mp4 (from record.py)")
    ap.add_argument("output", help="Output viewable .mp4")
    ap.add_argument("--curve", choices=["gamma", "log", "reinhard"],
                    default="log", help="Tone curve (matches the GUI)")
    ap.add_argument("--param", type=float, default=120.0,
                    help="Curve strength (gamma exponent / log a / reinhard k)")
    ap.add_argument("--ev", type=float, default=1.0,
                    help="Exposure multiplier applied before the curve")
    ap.add_argument("--black", type=float, default=0.0, help="Black level (0..0.5)")
    ap.add_argument("--white", type=float, default=1.0, help="White level (0.5..1)")
    ap.add_argument("--crf", type=int, default=18,
                    help="H.264 quality (lower=better; 18 visually lossless)")
    args = ap.parse_args()

    w, h = _probe_size(args.input)
    fps = _probe_fps(args.input)
    frame_bytes = w * h * 3 * 2          # rgb48le: 3 ch * uint16

    # Decoder: linear master -> raw 16-bit linear RGB on stdout. We do NOT let
    # ffmpeg apply any transfer; the file's 'linear' tag means these bytes are
    # the scene-linear signal, exactly what to_display expects.
    dec = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", args.input,
         "-pix_fmt", "rgb48le", "-f", "rawvideo", "pipe:1"],
        stdout=subprocess.PIPE,
    )

    # Encoder: tone-mapped 8-bit RGB -> H.264, tagged bt709 (a normal display
    # video). +faststart for progressive playback.
    enc = subprocess.Popen(
        ["ffmpeg", "-y", "-v", "error",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", fps,
         "-i", "pipe:0",
         "-c:v", "libx264", "-crf", str(args.crf), "-pix_fmt", "yuv420p",
         "-colorspace", "bt709", "-color_primaries", "bt709",
         "-color_trc", "bt709", "-color_range", "tv",
         "-movflags", "+faststart", args.output],
        stdin=subprocess.PIPE,
    )

    n = 0
    try:
        while True:
            buf = dec.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break                    # EOF (or partial trailing read)
            linear = np.frombuffer(buf, dtype="<u2").reshape(h, w, 3)
            # max_in=65535: the master fills the full 16-bit range (record.py
            # wrote align_to_16bit frames, 12-bit data left-shifted by 4).
            disp = convert.to_display(
                linear, curve=args.curve, param=args.param, exposure=args.ev,
                black=args.black, white=args.white, max_in=65535.0)
            enc.stdin.write(np.ascontiguousarray(disp).tobytes())
            n += 1
    finally:
        try:
            enc.stdin.close()
        except OSError:
            pass
        dec.wait()
        enc.wait()

    print(f"[info] wrote {n} frames -> {args.output} "
          f"(curve={args.curve} param={args.param} ev={args.ev})")


if __name__ == "__main__":
    main()
