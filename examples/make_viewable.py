#!/usr/bin/env python3
"""
Turn a recording (from record.py) into a viewable, plays-anywhere MP4.

record.py writes one of two masters, tagged in the stream's transfer VUI:

  bt709 master  (record.py default) — already BT.709 gamma-encoded 10-bit HEVC.
                It is a normal SDR video that already looks correct; it just
                isn't maximally compatible (10-bit, full range). So "making it
                viewable" is a pure *format* transcode: 10-bit -> 8-bit, full
                -> limited range, HEVC -> H.264. No tone curve, no colour
                scaling — the gamma is left exactly as recorded. ffmpeg does the
                whole thing; there is no per-frame Python.

  linear master (record.py --transfer linear) — scene-linear 10-bit HEVC. It
                looks dark in normal players because nothing tone-maps it, so
                here we DO bake in a tone curve, using the same code the live
                GUI uses (``hteng_camera.convert.tonemap_linear``) — a file
                converted with ``--curve log --param 120`` looks pixel-for-pixel
                like the GUI did at those settings.

The input's ``color_trc`` tag selects the path automatically (override with
``--mode``).

Pipelines:
    bt709:  HEVC 10-bit full-range --ffmpeg--> H.264 8-bit limited-range (bt709)
    linear: HEVC 10-bit linear --decode--> rgb48le
            --convert.tonemap_linear(curve,param)--> 8-bit RGB
            --encode--> H.264 8-bit (bt709)

The output path defaults to the input with an ``_8bit`` suffix
(``master.mp4`` -> ``master_8bit.mp4``); override it with ``-o``.

Usage:
    python make_viewable.py master.mp4                 # -> master_8bit.mp4
    python make_viewable.py --crf 20 master.mp4
    python make_viewable.py master.mp4 -o viewable.mp4
    # linear master (or force the tone-map path):
    python make_viewable.py --mode tonemap --curve log --param 120 in.mp4
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

from hteng_camera import convert


def _probe(path: str, entries: str) -> str:
    """Return one or more ffprobe stream entries (raw stdout), or '' on error."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", f"stream={entries}", "-of", "csv=p=0:s=x", path],
        capture_output=True, text=True,
    )
    return out.stdout.strip() if out.returncode == 0 else ""


def _probe_size(path: str) -> tuple[int, int]:
    s = _probe(path, "width,height")
    if "x" not in s:
        sys.exit(f"[error] could not probe dimensions of {path}")
    w, h = s.split("x")
    return int(w), int(h)


def _probe_fps(path: str) -> str:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=avg_frame_rate", "-of", "csv=p=0", path],
        capture_output=True, text=True,
    )
    fps = out.stdout.strip()
    return fps if fps and fps != "0/0" else "30"


def _probe_transfer(path: str) -> str:
    """Return the stream's transfer tag ('bt709', 'linear', ...) or '' if unset."""
    return _probe(path, "color_transfer")


# ---------------------------------------------------------------------------
# Path 1: bt709 master -> pure ffmpeg format transcode (no tone curve)
# ---------------------------------------------------------------------------

def _transcode_bt709(inp: str, out: str, crf: int) -> int:
    """Transcode an already-gamma-encoded master to a compatible 8-bit H.264.

    The master is BT.709 gamma, 10-bit, full range. Players want 8-bit and
    (conventionally) limited range, so the only conversions are bit depth and
    range — the transfer is bt709 in and bt709 out, so the *look* is untouched.
    scale=in_range=full:out_range=tv remaps the levels (not just retags them);
    -color_* stamp the output VUI to match.
    """
    cmd = [
        "ffmpeg", "-y", "-v", "error", "-i", inp,
        # Bit depth happens via -pix_fmt; this only remaps full->limited levels.
        "-vf", "scale=in_range=full:out_range=tv",
        "-c:v", "libx264", "-crf", str(crf), "-pix_fmt", "yuv420p",
        "-colorspace", "bt709", "-color_primaries", "bt709",
        "-color_trc", "bt709", "-color_range", "tv",
        "-movflags", "+faststart", out,
    ]
    return subprocess.run(cmd).returncode


# ---------------------------------------------------------------------------
# Path 2: linear master -> tone-mapped 8-bit H.264 (bakes in a display curve)
# ---------------------------------------------------------------------------

def _tonemap_linear_master(inp: str, out: str, w: int, h: int, fps: str,
                           crf: int, curve: str, param: float, ev: float,
                           black: float, white: float) -> int:
    """Decode a linear master, bake a tone curve per frame, re-encode to H.264."""
    frame_bytes = w * h * 3 * 2          # rgb48le: 3 ch * uint16

    # Decoder: linear master -> raw 16-bit linear RGB on stdout. We do NOT let
    # ffmpeg apply any transfer; the 'linear' tag means these bytes are the
    # scene-linear signal, exactly what tonemap_linear expects.
    dec = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", inp,
         "-pix_fmt", "rgb48le", "-f", "rawvideo", "pipe:1"],
        stdout=subprocess.PIPE,
    )

    # Encoder: tone-mapped 8-bit RGB -> H.264, tagged bt709 (a normal display
    # video). +faststart for progressive playback.
    enc = subprocess.Popen(
        ["ffmpeg", "-y", "-v", "error",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", fps,
         "-i", "pipe:0",
         "-c:v", "libx264", "-crf", str(crf), "-pix_fmt", "yuv420p",
         "-colorspace", "bt709", "-color_primaries", "bt709",
         "-color_trc", "bt709", "-color_range", "tv",
         "-movflags", "+faststart", out],
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
            disp = convert.tonemap_linear(
                linear, curve=curve, param=param, exposure=ev,
                black=black, white=white, max_in=65535.0)
            enc.stdin.write(np.ascontiguousarray(disp).tobytes())
            n += 1
    finally:
        try:
            enc.stdin.close()
        except OSError:
            pass
        dec.wait()
        enc.wait()
    return n


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Make a record.py master viewable (transcode or tone-map).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("input", help="Master .mp4 (from record.py)")
    ap.add_argument("-o", "--output", default=None,
                    help="Output path (default: input with an '_8bit' suffix)")
    ap.add_argument(
        "--mode", choices=["auto", "transcode", "tonemap"], default="auto",
        help=(
            "auto: pick by the input's transfer tag (bt709 -> transcode, "
            "linear -> tonemap). transcode: 10->8-bit format-only (gamma "
            "master). tonemap: bake a display curve (linear master)."
        ),
    )
    ap.add_argument("--crf", type=int, default=18,
                    help="H.264 quality (lower=better; 18 visually lossless)")
    # Tone-map (linear-master) options — ignored by the transcode path.
    ap.add_argument("--curve", choices=["gamma", "log", "reinhard"],
                    default="log", help="[tonemap] Tone curve (matches the GUI)")
    ap.add_argument("--param", type=float, default=120.0,
                    help="[tonemap] Curve strength (gamma exp / log a / reinhard k)")
    ap.add_argument("--ev", type=float, default=1.0,
                    help="[tonemap] Exposure multiplier applied before the curve")
    ap.add_argument("--black", type=float, default=0.0,
                    help="[tonemap] Black level (0..0.5)")
    ap.add_argument("--white", type=float, default=1.0,
                    help="[tonemap] White level (0.5..1)")
    args = ap.parse_args()

    # Default output: input filename with an '_8bit' suffix (keep the directory
    # and extension), e.g. /data/master.mp4 -> /data/master_8bit.mp4.
    if args.output:
        output = args.output
    else:
        p = Path(args.input)
        output = str(p.with_name(f"{p.stem}_8bit{p.suffix}"))

    # Resolve mode: 'auto' reads the input transfer tag. Anything not explicitly
    # 'linear' is treated as already display-encoded (bt709) -> transcode.
    mode = args.mode
    if mode == "auto":
        trc = _probe_transfer(args.input)
        mode = "tonemap" if trc == "linear" else "transcode"
        print(f"[info] input transfer={trc or 'unset'} -> mode {mode}")

    if mode == "transcode":
        rc = _transcode_bt709(args.input, output, args.crf)
        if rc != 0:
            sys.exit(f"[error] ffmpeg transcode failed (code {rc}).")
        print(f"[info] transcoded (10->8-bit, gamma untouched) -> {output}")
    else:
        w, h = _probe_size(args.input)
        fps = _probe_fps(args.input)
        n = _tonemap_linear_master(
            args.input, output, w, h, fps, args.crf,
            args.curve, args.param, args.ev, args.black, args.white)
        print(f"[info] wrote {n} frames -> {output} "
              f"(curve={args.curve} param={args.param} ev={args.ev})")


if __name__ == "__main__":
    main()
