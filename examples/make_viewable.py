#!/usr/bin/env python3
"""
Turn a recording (from record.py) into a viewable, plays-anywhere MP4.

record.py writes one of two masters, tagged in the stream's transfer VUI:

  bt709 master  (record.py default) — already BT.709 gamma-encoded 10-bit HEVC.
                It is a normal SDR video that already looks correct; it just
                isn't maximally compatible (10-bit, full range). So "making it
                viewable" is a pure *format* transcode: 10-bit -> 8-bit, full
                -> limited range, re-encoded as 8-bit HEVC (hvc1). No tone curve,
                no colour scaling — the gamma is left exactly as recorded.
                ffmpeg does the whole thing; there is no per-frame Python.

  linear master (record.py --transfer linear) — scene-linear 10-bit HEVC. It
                looks dark in normal players because nothing tone-maps it, so
                here we DO bake in a tone curve, using the same code the live
                GUI uses (``hteng_camera.convert.tonemap_linear``) — a file
                converted with ``--curve log --param 120`` looks pixel-for-pixel
                like the GUI did at those settings.

The input's ``color_trc`` tag selects the path automatically (override with
``--mode``).

Pipelines:
    bt709:  HEVC 10-bit full-range --ffmpeg--> HEVC 8-bit limited-range (bt709)
    linear: HEVC 10-bit linear --decode--> rgb48le
            --convert.tonemap_linear(curve,param)--> 8-bit RGB
            --encode--> HEVC 8-bit (bt709)

The output path defaults to the input with an ``_8bit`` suffix
(``master.mp4`` -> ``master_8bit.mp4``); override it with ``-o``.

Usage:
    python make_viewable.py master.mp4                 # -> master_8bit.mp4
    python make_viewable.py --crf 20 master.mp4
    python make_viewable.py master.mp4 -o viewable.mp4
    # linear master (or force the tone-map path):
    python make_viewable.py --mode tonemap --curve log --param 120 in.mp4
    # stereo pair (tone-mapped or bt709, both treated the same way):
    python make_viewable.py left.mp4 right.mp4
    python make_viewable.py left.mp4 right.mp4 -o stereo.mp4
"""

import argparse
import platform
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np

from hteng_camera import convert

def _video_enc(crf: int) -> list[str]:
    """Return ffmpeg HEVC encoder + quality flags. VideoToolbox on macOS, x265 elsewhere.

    HEVC (vs H.264) lets us encode the full-resolution side-by-side stereo frame
    — h264_videotoolbox caps each axis at 4096, which the hstacked pair exceeds;
    HEVC's limit is 8192. The hvc1 tag makes the .mp4 play in QuickTime/Apple.

    hevc_videotoolbox drops transfer/primaries from the bitstream VUI (it only
    writes them when the *input* already carried them), so a raw-frame encode
    ends up tagged 'unknown'. We always output bt709, so stamp it back into the
    VUI with the hevc_metadata bitstream filter (1 == bt709 for all three).
    """
    bsf = ["-bsf:v", "hevc_metadata=transfer_characteristics=1:"
           "colour_primaries=1:matrix_coefficients=1"]
    if platform.system() == "Darwin":
        # VideoToolbox -q:v is 1..100, higher = better (the OPPOSITE of x26x crf,
        # where 0=lossless..51=worst). Map crf onto it; crf 18 -> ~q 65.
        q = max(1, min(100, round((1 - crf / 51) * 100)))
        return ["-c:v", "hevc_videotoolbox", "-q:v", str(q), "-tag:v", "hvc1", *bsf]
    return ["-c:v", "libx265", "-crf", str(crf), "-tag:v", "hvc1", *bsf]


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
    """Transcode an already-gamma-encoded master to a compatible 8-bit HEVC.

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
        *_video_enc(crf), "-pix_fmt", "yuv420p",
        "-colorspace", "bt709", "-color_primaries", "bt709",
        "-color_trc", "bt709", "-color_range", "tv",
        "-movflags", "+faststart", out,
    ]
    return subprocess.run(cmd).returncode


# ---------------------------------------------------------------------------
# Path 2: linear master -> tone-mapped 8-bit HEVC (bakes in a display curve)
# ---------------------------------------------------------------------------

def _tonemap_linear_master(inp: str, out: str, w: int, h: int, fps: str,
                           crf: int, curve: str, param: float, ev: float,
                           black: float, white: float) -> int:
    """Decode a linear master, bake a tone curve per frame, re-encode to HEVC."""
    frame_bytes = w * h * 3 * 2          # rgb48le: 3 ch * uint16

    # Decoder: linear master -> raw 16-bit linear RGB on stdout. We do NOT let
    # ffmpeg apply any transfer; the 'linear' tag means these bytes are the
    # scene-linear signal, exactly what tonemap_linear expects.
    dec = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", inp,
         "-pix_fmt", "rgb48le", "-f", "rawvideo", "pipe:1"],
        stdout=subprocess.PIPE,
    )

    # Encoder: tone-mapped 8-bit RGB -> HEVC, tagged bt709 (a normal display
    # video). +faststart for progressive playback.
    enc = subprocess.Popen(
        ["ffmpeg", "-y", "-v", "error",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", fps,
         "-i", "pipe:0",
         *_video_enc(crf), "-pix_fmt", "yuv420p",
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


def _decode_to_rgb24(inp: str, w: int, h: int, fps: str,
                     mode: str, curve: str, param: float,
                     ev: float, black: float, white: float) -> subprocess.Popen:
    """Return an encoder Popen whose stdin receives rgb24 frames.

    For the transcode path, returns a ffmpeg decoder piping yuv->rgb24 directly.
    For the tonemap path, returns a Popen whose stdin we feed tone-mapped frames.
    This function is only used by the stereo path; single-file paths are unchanged.
    """
    if mode == "transcode":
        # bt709 master: decode to rgb24, with full->limited level conversion
        proc = subprocess.Popen(
            ["ffmpeg", "-v", "error", "-i", inp,
             "-vf", "scale=in_range=full:out_range=tv",
             "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"],
            stdout=subprocess.PIPE,
        )
        return proc, None           # (reader, tonemap_thread) -- no thread needed

    # tonemap path: decode linear -> rgb48le, tone-map in Python, write rgb24
    import io
    dec = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", inp,
         "-pix_fmt", "rgb48le", "-f", "rawvideo", "pipe:1"],
        stdout=subprocess.PIPE,
    )
    frame_bytes_in  = w * h * 3 * 2
    frame_bytes_out = w * h * 3

    # We can't easily pipe through a Python transform and present a simple
    # stdout to the caller, so we use a thread + os.pipe.
    import os
    rfd, wfd = os.pipe()
    reader = os.fdopen(rfd, "rb")
    writer = os.fdopen(wfd, "wb")

    def _pump():
        try:
            while True:
                buf = dec.stdout.read(frame_bytes_in)
                if len(buf) < frame_bytes_in:
                    break
                linear = np.frombuffer(buf, dtype="<u2").reshape(h, w, 3)
                disp = convert.tonemap_linear(
                    linear, curve=curve, param=param, exposure=ev,
                    black=black, white=white, max_in=65535.0)
                writer.write(np.ascontiguousarray(disp).tobytes())
        finally:
            writer.close()
            dec.wait()

    t = threading.Thread(target=_pump, daemon=True)
    t.start()

    # Wrap the raw read fd in a Popen-like object so the caller can use .stdout
    class _PipeSource:
        def __init__(self, r, thread, dec_proc):
            self.stdout = r
            self._thread = thread
            self._dec = dec_proc
        def wait(self):
            self._thread.join()
            return self._dec.returncode

    return _PipeSource(reader, t, dec), t


def _transcode_stereo(inp_l: str, inp_r: str, out: str,
                      mode_l: str, mode_r: str,
                      w_l: int, h_l: int, fps_l: str,
                      w_r: int, h_r: int,
                      crf: int, curve: str, param: float,
                      ev: float, black: float, white: float) -> int:
    """Decode both videos to rgb24 frame streams, hstack per-frame, encode.

    The full-resolution hstacked frame (e.g. 4896 wide for two 2448-px eyes) is
    encoded as HEVC, which has no trouble with the width — h264_videotoolbox
    would refuse anything over 4096 px.
    """
    # Both sides must have the same height; width can differ (hstack handles it).
    if h_l != h_r:
        sys.exit(f"[error] stereo videos have different heights ({h_l} vs {h_r}); "
                 "cannot hstack")

    w_out = w_l + w_r
    h_out = h_l

    src_l, _ = _decode_to_rgb24(inp_l, w_l, h_l, fps_l,
                                 mode_l, curve, param, ev, black, white)
    src_r, _ = _decode_to_rgb24(inp_r, w_r, h_r, fps_l,
                                 mode_r, curve, param, ev, black, white)

    enc = subprocess.Popen(
        ["ffmpeg", "-y", "-v", "error",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w_out}x{h_out}",
         "-r", fps_l, "-i", "pipe:0",
         *_video_enc(crf), "-pix_fmt", "yuv420p",
         "-colorspace", "bt709", "-color_primaries", "bt709",
         "-color_trc", "bt709", "-color_range", "tv",
         "-movflags", "+faststart", out],
        stdin=subprocess.PIPE,
    )

    frame_l = w_l * h_l * 3
    frame_r = w_r * h_r * 3
    n = 0
    try:
        while True:
            buf_l = src_l.stdout.read(frame_l)
            buf_r = src_r.stdout.read(frame_r)
            if len(buf_l) < frame_l or len(buf_r) < frame_r:
                break
            left  = np.frombuffer(buf_l, dtype=np.uint8).reshape(h_l, w_l, 3)
            right = np.frombuffer(buf_r, dtype=np.uint8).reshape(h_r, w_r, 3)
            enc.stdin.write(np.ascontiguousarray(np.hstack([left, right])).tobytes())
            n += 1
    finally:
        try:
            enc.stdin.close()
        except OSError:
            pass
        src_l.wait()
        src_r.wait()
        enc.wait()
    return n


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Make a record.py master viewable (transcode or tone-map).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("input", help="Master .mp4 (from record.py)")
    ap.add_argument("input2", nargs="?", default=None,
                    help="Optional second .mp4 for a side-by-side stereo pair")
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
                    help="HEVC quality (lower=better; 18 visually lossless)")
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

    # Default output suffix.
    if args.output:
        output = args.output
    else:
        p = Path(args.input)
        suffix = "_stereo_8bit" if args.input2 else "_8bit"
        output = str(p.with_name(f"{p.stem}{suffix}{p.suffix}"))

    # Resolve mode for input 1 (and input2 if present).
    def _resolve_mode(path: str, label: str) -> str:
        if args.mode != "auto":
            return args.mode
        trc = _probe_transfer(path)
        m = "tonemap" if trc == "linear" else "transcode"
        print(f"[info] {label} transfer={trc or 'unset'} -> mode {m}")
        return m

    mode = _resolve_mode(args.input, "input")

    if args.input2:
        mode2 = _resolve_mode(args.input2, "input2")
        w_l, h_l = _probe_size(args.input)
        w_r, h_r = _probe_size(args.input2)
        fps = _probe_fps(args.input)
        n = _transcode_stereo(
            args.input, args.input2, output,
            mode, mode2,
            w_l, h_l, fps,
            w_r, h_r,
            args.crf, args.curve, args.param, args.ev, args.black, args.white)
        print(f"[info] wrote {n} stereo frames -> {output}")
    elif mode == "transcode":
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
