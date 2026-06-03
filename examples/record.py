#!/usr/bin/env python3
"""
Record an HTENG camera to a 10-bit HEVC MP4.

Pixel path
----------
Camera (12-bit linear Bayer)
  -> demosaic (cv2 bilinear SIMD)
  -> uint16 RGB, values 0..65520   (12-bit data left-shifted to fill 16-bit)
  -> pipe as rgb48le to ffmpeg
  -> scale (software BT.709 RGB->YUV)
  -> 10-bit YUV 4:2:0, full range
  -> HEVC encoder (NVENC / VideoToolbox / x265)
  -> MP4

Encoders
--------
Pick with --encoder; default 'auto' maps by platform (no probing/guessing):
  nvenc         hevc_nvenc       NVIDIA GPU (Maxwell+, driver >=418). Linux auto.
  videotoolbox  hevc_videotoolbox  Apple Silicon hardware HEVC. macOS auto.
  x265          libx265          software, universal, slower. Manual fallback.

Colour metadata
---------------
The encoded stream is tagged:
  color_trc = linear      No OETF applied; signal is scene-linear light.
  color_primaries = bt709 Rec.709 primaries (matches most CMOS sensors).
  colorspace = bt709      BT.709 YUV matrix coefficients.
  color_range = pc        Full (0-1023) luma/chroma range, no clip.

These tags let DaVinci Resolve, FFmpeg vf colorspace, etc. apply the
correct tone map on ingest — the recorded file is the "before tone-map"
master.

Requirements
------------
  pip install hteng_camera
  FFmpeg with the chosen encoder (nvenc on Linux, videotoolbox on macOS, or
  libx265 anywhere).

Usage
-----
  python record.py output.mp4
  python record.py --duration 10 --exposure-ms 25 --quality 20 clip.mp4
  python record.py --serial 044162023020 --fps 60 fast.mp4
  python record.py --encoder x265 software.mp4
"""

import argparse
import platform
import queue
import shutil
import subprocess
import sys
import threading
import time
from typing import Optional

import numpy as np

from hteng_camera import HTCamera, enums


# ---------------------------------------------------------------------------
# Encoder registry — codec name, output pixel format, and the quality flag each
# encoder uses. RGB->YUV conversion, colour metadata, and container are shared.
# ---------------------------------------------------------------------------

# quality_flag: the ffmpeg flag controlling constant-quality rate control, and a
#   builder turning our 0..51-style --quality into that encoder's args.
_ENCODERS = {
    "nvenc": {
        "codec": "hevc_nvenc",
        "pix_fmt": "p010le",          # 10-bit 4:2:0 semi-planar
        "extra": ["-profile:v", "main10", "-preset", "p4"],
        # constant-QP archival mode; -b:v 0 lets QP govern.
        "quality": lambda q: ["-rc", "constqp", "-qp", str(q), "-b:v", "0"],
    },
    "videotoolbox": {
        "codec": "hevc_videotoolbox",
        "pix_fmt": "p010le",          # Apple HW also speaks p010le
        "extra": ["-profile:v", "main10"],
        # VideoToolbox has no QP mode; -q:v is a 0..100 quality (higher=better),
        # so map QP (lower=better) onto it roughly.
        "quality": lambda q: ["-q:v", str(max(1, min(100, 100 - 2 * q)))],
    },
    "x265": {
        "codec": "libx265",
        "pix_fmt": "yuv420p10le",     # x265 uses planar 10-bit, not p010le
        "extra": ["-preset", "medium"],
        # x265's CRF is already a 0..51 quality scale (lower=better) — use --quality directly.
        "quality": lambda q: ["-crf", str(q)],
    },
}

# Platform default for --encoder auto. A deterministic lookup, not a capability
# probe — mirrors how the library picks its shared lib by platform.
_AUTO_ENCODER = {"Linux": "nvenc", "Darwin": "videotoolbox"}


def _resolve_encoder(name: str) -> str:
    """Map 'auto' to the platform default, else pass an explicit name through."""
    if name != "auto":
        return name
    enc = _AUTO_ENCODER.get(platform.system())
    if enc is None:
        sys.exit(
            f"[error] no default encoder for platform {platform.system()!r}; "
            f"pass --encoder explicitly ({'/'.join(_ENCODERS)})."
        )
    return enc


# ---------------------------------------------------------------------------
# Preflight: make sure ffmpeg exists and has the chosen encoder
# ---------------------------------------------------------------------------

def _preflight_ffmpeg(encoder: str) -> None:
    """Fail early with an actionable message if ffmpeg / the encoder isn't usable.

    Without this, a missing ffmpeg surfaces as a raw FileNotFoundError and a
    missing encoder only shows up as a late nonzero exit code with the real
    error buried in ffmpeg's stderr.
    """
    if shutil.which("ffmpeg") is None:
        sys.exit(
            "[error] ffmpeg not found on PATH.\n"
            "        Install it (e.g. `sudo apt install ffmpeg`, "
            "`brew install ffmpeg`)."
        )
    codec = _ENCODERS[encoder]["codec"]
    try:
        out = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=10,
        ).stdout
    except (subprocess.SubprocessError, OSError) as e:
        sys.exit(f"[error] could not query ffmpeg encoders: {e}")
    if codec not in out:
        hint = {
            "nvenc": "Needs an NVIDIA GPU (Maxwell+), driver >=418, and ffmpeg "
                     "built with --enable-nvenc.",
            "videotoolbox": "Needs macOS with VideoToolbox (Apple Silicon or "
                            "recent Intel Macs).",
            "x265": "Needs ffmpeg built with --enable-libx265.",
        }[encoder]
        sys.exit(
            f"[error] this ffmpeg has no {codec} encoder (for --encoder "
            f"{encoder}).\n        {hint}\n"
            f"        Check with: ffmpeg -encoders | grep {codec}"
        )


# ---------------------------------------------------------------------------
# FFmpeg command builder
# ---------------------------------------------------------------------------

def _ffmpeg_cmd(encoder: str, w: int, h: int, fps: int, quality: int,
                output: str) -> list[str]:
    """Build the ffmpeg invocation for 10-bit HEVC encoding with ``encoder``.

    Input pixel format rgb48le (3 x uint16-LE channels, 0-65535 per channel)
    is the natural container for our 16-bit-aligned demosaiced frames.

    The software scale filter does the RGB->YUV colour-matrix conversion on the
    CPU; full range (in_range=full:out_range=full) avoids luma/chroma clipping
    and preserves the entire 0-65520 dynamic range we captured. Only the codec,
    output pixel format, and quality flags vary by encoder (see _ENCODERS); the
    input, colour conversion, metadata, and container are identical.
    """
    enc = _ENCODERS[encoder]
    return [
        "ffmpeg", "-y",

        # ── input: raw 16-bit linear RGB from stdin ──────────────────────
        "-f", "rawvideo",
        "-pix_fmt", "rgb48le",       # 3 x uint16-LE per pixel, packed
        "-s", f"{w}x{h}",
        "-r", str(fps),
        "-i", "pipe:0",

        # ── RGB→YUV conversion (CPU) ─────────────────────────────────────
        # Explicit BT.709 matrix + full range so coefficients are
        # unambiguous and no signal is clipped during colorspace conversion.
        "-vf", (
            "scale="
            "in_color_matrix=bt709:out_color_matrix=bt709:"
            "in_range=full:out_range=full"
        ),

        # ── encoder (codec + 10-bit pixel format + quality) ──────────────
        "-c:v", enc["codec"],
        *enc["extra"],
        "-pix_fmt", enc["pix_fmt"],
        # --quality is a QP/CRF-style number (lower = better; ~18 visually
        # lossless, 24-28 ~half size). Each encoder maps it to its own knob.
        *enc["quality"](quality),

        # ── colour metadata ───────────────────────────────────────────────
        # The -color_* output flags alone DON'T reach the HEVC bitstream for
        # these encoders (nvenc/videotoolbox/x265 each drop transfer/primaries),
        # so a decoder would read "unknown" and apply the wrong tone map. The
        # hevc_metadata bitstream filter stamps the VUI into the stream itself,
        # encoder-agnostically. H.265 enum values:
        #   transfer_characteristics=8  -> Linear (scene-linear, no OETF)
        #   colour_primaries=1          -> BT.709
        #   matrix_coefficients=1       -> BT.709
        #   video_full_range_flag=1     -> full range (matches out_range=full)
        "-bsf:v", (
            "hevc_metadata="
            "transfer_characteristics=8:colour_primaries=1:"
            "matrix_coefficients=1:video_full_range_flag=1"
        ),
        # Also pass the high-level tags (harmless; helps tools that read
        # container-level metadata rather than the bitstream VUI).
        "-colorspace", "bt709",
        "-color_primaries", "bt709",
        "-color_trc", "linear",      # scene-linear light, no OETF baked in
        "-color_range", "pc",        # full range matches out_range=full above

        # ── MP4 container ─────────────────────────────────────────────────
        "-movflags", "+faststart",   # moov atom at front for progressive play
        output,
    ]


# ---------------------------------------------------------------------------
# Producer: camera capture thread
# ---------------------------------------------------------------------------

def _capture_loop(
    cam: HTCamera,
    q: "queue.Queue[np.ndarray]",
    stop: threading.Event,
) -> None:
    """Grab frames as fast as the hardware allows and push them into the queue.

    align_to_16bit=True shifts the 12-bit (0-4095) demosaiced values left by
    4 to fill the 16-bit container (0-65520), which maps them to the full
    dynamic range expected by the rgb48le ffmpeg input format.

    When the queue is full the oldest frame is dropped, so the queue always
    holds the freshest available footage and memory stays bounded regardless
    of temporary encoder hiccups.
    """
    while not stop.is_set():
        frame, _info = cam.grab(align_to_16bit=True)
        if frame is None:
            continue                 # timeout — retry immediately
        if q.full():
            try:
                q.get_nowait()       # drop stale frame
            except queue.Empty:
                pass
        q.put(frame)


# ---------------------------------------------------------------------------
# Consumer: fixed-rate encoder thread
# ---------------------------------------------------------------------------

def _encode_loop(
    proc: subprocess.Popen,
    q: "queue.Queue[np.ndarray]",
    stop: threading.Event,
    fps: int,
) -> None:
    """Pull the freshest available frame at each 1/fps deadline and pipe it
    to ffmpeg's stdin.

    Timing model
    ~~~~~~~~~~~~
    A monotonic deadline is advanced by exactly one frame interval on each
    tick, so the output PTS stream is perfectly uniform at the target fps
    regardless of capture jitter.  If the camera delivers faster than the
    target rate, intermediate frames are dropped (always taking the freshest).
    If it delivers slower (e.g. long exposure), the last frame is duplicated —
    the encoded video never stalls.

    The inner sleep uses coarse_sleep + 1 ms spin to hit the deadline without
    burning a full CPU core on a tight-loop.
    """
    interval = 1.0 / fps
    last_frame: Optional[np.ndarray] = None
    deadline = time.monotonic() + interval

    while not (stop.is_set() and q.empty()):
        now = time.monotonic()
        remaining = deadline - now

        if remaining > 0.001:
            # Coarse sleep for most of the interval, then spin the last ms.
            time.sleep(remaining - 0.001)
            continue

        # Snap the deadline forward if we fell more than one frame behind
        # (e.g. after a long GIL pause).
        if remaining < -interval:
            deadline = now + interval
        else:
            deadline += interval

        # Drain the queue to get the freshest available frame.
        frame: Optional[np.ndarray] = None
        while True:
            try:
                frame = q.get_nowait()
            except queue.Empty:
                break

        if frame is None:
            frame = last_frame   # duplicate on camera underrun
        if frame is None:
            continue             # no data yet; wait for the first frame

        last_frame = frame

        # Write frame as raw bytes.  view(uint8) reinterprets the uint16 data
        # as bytes without a copy (assuming C-contiguous layout, which cv2's
        # demosaic always returns).
        try:
            proc.stdin.write(np.ascontiguousarray(frame).view(np.uint8))
        except (BrokenPipeError, OSError):
            break

    try:
        proc.stdin.close()
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Record HTENG camera to a 10-bit HEVC MP4",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("output", help="Output .mp4 path")
    ap.add_argument(
        "--encoder", choices=["auto", *_ENCODERS], default="auto",
        help=(
            "HEVC encoder. 'auto' maps by platform (Linux->nvenc, "
            "macOS->videotoolbox); or force nvenc / videotoolbox / x265."
        ),
    )
    ap.add_argument(
        "--fps", type=int, default=30,
        help="Target output frame rate",
    )
    ap.add_argument(
        "--duration", type=float, default=None,
        help="Recording duration in seconds (omit to record until Ctrl+C)",
    )
    ap.add_argument(
        "--serial", default=None,
        help="Camera serial number (default: first found)",
    )
    ap.add_argument(
        "--exposure-ms", type=float, default=25.0,
        help=(
            "Exposure time in ms.  Must be < 1000/fps (e.g. < 33 ms at 30 fps) "
            "or the camera will run below the target rate and frames will be "
            "duplicated.  Auto-exposure is disabled for stable fps."
        ),
    )
    ap.add_argument(
        "--gain", type=float, default=1.0,
        help="Analog gain multiplier",
    )
    ap.add_argument(
        "--quality", type=int, default=18,
        help=(
            "Constant-quality level, QP/CRF-style (0=best, 51=worst). "
            "18 is visually lossless; 24-28 gives roughly half the file size. "
            "Mapped to each encoder's native knob (nvenc QP / x265 CRF / "
            "videotoolbox quality)."
        ),
    )
    args = ap.parse_args()

    encoder = _resolve_encoder(args.encoder)

    # Fail fast if ffmpeg / the chosen encoder isn't usable, before the camera.
    _preflight_ffmpeg(encoder)

    frame_interval_ms = 1000.0 / args.fps
    if args.exposure_ms >= frame_interval_ms:
        print(
            f"[warn] exposure {args.exposure_ms:.1f} ms >= frame interval "
            f"{frame_interval_ms:.1f} ms — camera will run below {args.fps} fps "
            "and output frames will be duplicated."
        )

    # ── Open camera ──────────────────────────────────────────────────────────
    print("[info] Opening camera…")
    cam = HTCamera(serial=args.serial)
    cam.set_ae(False)                        # fixed exposure → stable fps
    cam.set_frame_speed(enums.FRAME_SPEED_HIGH)  # max USB throughput; avoid underrun
    cam.set_exposure_ms(args.exposure_ms)
    cam.set_analog_gain(args.gain)

    # Discard the first grab: the oldest buffered frame may pre-date our
    # exposure setting.
    cam.grab()
    test_frame, _ = cam.grab(align_to_16bit=True)
    if test_frame is None:
        raise RuntimeError("Camera returned no frame — check connection.")
    h, w = test_frame.shape[:2]
    print(
        f"[info] {w}x{h} @ {args.fps} fps, "
        f"exposure {args.exposure_ms:.1f} ms, gain {args.gain:.2f}x, "
        f"encoder {encoder} ({_ENCODERS[encoder]['codec']}), quality {args.quality}"
    )

    # ── Start ffmpeg ─────────────────────────────────────────────────────────
    cmd = _ffmpeg_cmd(encoder, w, h, args.fps, args.quality, args.output)
    print(f"[info] ffmpeg command:\n  {' '.join(cmd)}\n")
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    # ── Start capture + encode threads ───────────────────────────────────────
    stop = threading.Event()
    # Queue depth of 60 frames: ~2 s of buffering at 30 fps, which is enough
    # to absorb any OS scheduling jitter without unbounded memory use.
    q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=60)

    cap_thread = threading.Thread(
        target=_capture_loop, args=(cam, q, stop), daemon=True,
        name="hteng-capture",
    )
    enc_thread = threading.Thread(
        target=_encode_loop, args=(proc, q, stop, args.fps),
        name="hteng-encode",
    )
    cap_thread.start()
    enc_thread.start()

    # ── Wait for duration / Ctrl+C ───────────────────────────────────────────
    # Sleep in short slices (rather than one long sleep / blocking join) so a
    # Ctrl+C is noticed promptly and the encoder thread can't keep us hanging.
    try:
        print("[info] Recording… press Ctrl+C to stop.")
        t_end = None if args.duration is None else time.monotonic() + args.duration
        while enc_thread.is_alive():
            if t_end is not None and time.monotonic() >= t_end:
                break
            if proc.poll() is not None:    # ffmpeg died (e.g. NVENC error)
                print("[warn] ffmpeg exited early — stopping.")
                break
            time.sleep(0.1)
    except KeyboardInterrupt:
        print()                            # newline after ^C

    # ── Tear down ─────────────────────────────────────────────────────────────
    # Order matters: stop+join the CAPTURE producer first so it stops feeding the
    # queue, otherwise the encoder's "stop and queue empty" exit can never be
    # reached (the daemon keeps refilling it). Then the encoder drains and exits.
    print("[info] Stopping…")
    stop.set()
    cap_thread.join(timeout=5)
    enc_thread.join(timeout=10)

    try:
        ret = proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        print("[warn] ffmpeg didn't exit in time — killing it.")
        proc.kill()
        ret = proc.wait()
    cam.close()

    if ret == 0:
        print(f"[info] Saved → {args.output}")
    else:
        print(f"[warn] ffmpeg exited with code {ret} — output may be incomplete.")


if __name__ == "__main__":
    main()
