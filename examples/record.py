#!/usr/bin/env python3
"""
Record an HTENG camera to a 10-bit HEVC MP4.

Pixel path
----------
Camera (12-bit packed Bayer)
  -> capture thread: unpack to a uint16 Bayer plane (native kernel, ~0.5 ms)
  -> latest-frame mailbox (one slot — always the freshest plane)
  -> encode thread, at each 1/fps deadline:
       demosaic (cv2 SIMD; edge-aware by default, see --demosaic)
       [bt709]  tonemap_linear: BT.709 OETF, 0..4095 -> 0..65535 (native LUT)
       [linear] left-shift by 4 to fill the 16-bit container (0..65520)
  -> pipe as rgb48le to ffmpeg
  -> scale: BT.709 RGB->YUV matrix conversion
  -> 10-bit YUV 4:2:0, full range
  -> HEVC encoder (NVENC / VideoToolbox / x265)
  -> MP4

Demosaic and the transfer encode run only for frames actually written: when the
camera outpaces --fps, a dropped frame costs just the cheap unpack.

Encoders
--------
Pick with --encoder; default 'auto' maps by platform (no probing/guessing):
  nvenc         hevc_nvenc       NVIDIA GPU (Maxwell+, driver >=418). Linux auto.
  videotoolbox  hevc_videotoolbox  Apple Silicon hardware HEVC. macOS auto.
  x265          libx265          software, universal, slower. Manual fallback.

Transfer function (--transfer)
-------------------------------
  bt709   BT.709 gamma applied before encoding (default). H.265's perceptual
          quantisation models are designed for gamma-encoded content, so this
          gives ~20-30% smaller files vs linear at the same visual quality.
          The gamma is applied in Python (convert.tonemap_linear, native LUT) —
          no dependency on an ffmpeg built with libzimg/zscale. The encoded file
          is gamma-encoded; recover linear with:
            ffmpeg -vf zscale=transferin=bt709:transfer=linear ...   (if zscale)
          or invert the BT.709 OETF in your colour tool of choice.
  linear  No OETF applied; signal is scene-linear light. Largest files but
          the recorded data is a direct linear-light master.

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
  python record.py --transfer linear linear_master.mp4
"""

import argparse
import platform
import shutil
import signal
import subprocess
import sys
import threading
import time
from typing import Optional

import numpy as np

from hteng_camera import HTCamera, convert, enums


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
                output: str, transfer: str = "bt709") -> list[str]:
    """Build the ffmpeg invocation for 10-bit HEVC encoding with ``encoder``.

    Input pixel format rgb48le (3 x uint16-LE channels, 0-65535 per channel)
    is the natural container for our 16-bit-aligned demosaiced frames.

    The BT.709 OETF (gamma) for transfer='bt709' is applied *in Python* by
    convert.tonemap_linear before the frame reaches ffmpeg (see _encode_loop) —
    this avoids depending on an ffmpeg built with libzimg/zscale, which the
    stock Homebrew build lacks. So ffmpeg's only job here is the RGB->YUV matrix
    conversion; the -vf is a plain scale regardless of transfer, and ``transfer``
    only selects the VUI transfer_characteristics tag stamped into the stream.

    transfer='bt709': frames arrive already gamma-encoded; tag the stream BT.709
    so a decoder knows to invert it. ~20-30% smaller than linear at equal CRF/QP.

    transfer='linear': frames arrive as scene-linear light; tag the stream
    Linear. Largest files, but a direct linear-light master.

    Only the codec, output pixel format, and quality flags vary by encoder
    (see _ENCODERS); the input, colour conversion, metadata, and container are
    identical.
    """
    enc = _ENCODERS[encoder]

    # H.265 VUI transfer_characteristics enum values used in hevc_metadata:
    #   1  -> BT.709 (gamma-encoded)
    #   8  -> Linear (scene-linear, no OETF)
    _TRANSFER_META = {"bt709": "1", "linear": "8"}
    _TRANSFER_TRC  = {"bt709": "bt709", "linear": "linear"}

    return [
        "ffmpeg", "-y",

        # ── input: raw 16-bit RGB from stdin (already transfer-encoded) ───
        "-f", "rawvideo",
        "-pix_fmt", "rgb48le",       # 3 x uint16-LE per pixel, packed
        "-s", f"{w}x{h}",
        "-r", str(fps),
        "-i", "pipe:0",

        # ── RGB→YUV conversion (CPU) ─────────────────────────────────────
        # The transfer curve is already baked in upstream (Python), so this is
        # just the BT.709 colour-matrix conversion at full range (no clipping).
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
        #   colour_primaries=1          -> BT.709
        #   matrix_coefficients=1       -> BT.709
        #   video_full_range_flag=1     -> full range (matches out_range=full)
        "-bsf:v", (
            f"hevc_metadata="
            f"transfer_characteristics={_TRANSFER_META[transfer]}:"
            f"colour_primaries=1:matrix_coefficients=1:video_full_range_flag=1"
        ),
        # Also pass the high-level tags (harmless; helps tools that read
        # container-level metadata rather than the bitstream VUI).
        "-colorspace", "bt709",
        "-color_primaries", "bt709",
        "-color_trc", _TRANSFER_TRC[transfer],
        "-color_range", "pc",        # full range matches out_range=full above

        # ── MP4 container ─────────────────────────────────────────────────
        "-movflags", "+faststart",   # moov atom at front for progressive play
        output,
    ]


# ---------------------------------------------------------------------------
# Producer: camera capture thread
# ---------------------------------------------------------------------------

class _Mailbox:
    """Single-slot latest-frame handoff between capture and encode threads.

    The encoder only ever wants the freshest frame, so anything deeper than one
    slot just buffers frames destined to be dropped (and at ~15 MB per 5 MP
    Bayer plane, depth is real memory). put() overwrites; take() empties.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None

    def put(self, frame: np.ndarray) -> None:
        with self._lock:
            self._frame = frame

    def take(self) -> Optional[np.ndarray]:
        with self._lock:
            frame, self._frame = self._frame, None
            return frame


def _capture_loop(cam: HTCamera, box: _Mailbox, stop: threading.Event) -> None:
    """Grab raw Bayer planes as fast as the hardware allows.

    Only the cheap unpack (~0.5 ms native) runs here; demosaic and the transfer
    encode happen in the encode thread, and only for frames actually written —
    when the camera outpaces --fps, overwritten frames cost nothing more.
    """
    while not stop.is_set():
        bayer, _info = cam.grab_bayer12()
        if bayer is None:
            continue                 # timeout — retry immediately
        box.put(bayer)


# ---------------------------------------------------------------------------
# Consumer: fixed-rate encoder thread
# ---------------------------------------------------------------------------

def _encode_loop(
    proc: subprocess.Popen,
    box: _Mailbox,
    stop: threading.Event,
    fps: int,
    cv_code: int,
    transfer: str = "bt709",
    wb_gains=None,
) -> None:
    """At each 1/fps deadline, demosaic + transfer-encode the freshest Bayer
    plane and pipe it to ffmpeg's stdin.

    Pixel work
    ~~~~~~~~~~
    Demosaic (cv2 SIMD) and the transfer encode run here, per *written* frame:
      bt709   convert.tonemap_linear's native uint16->uint16 LUT maps the
              demosaiced 0..4095 linear signal through the BT.709 OETF to the
              full 0..65535 range (so peak sensor white is peak rgb48le white).
              Gamma-encoded input compresses ~20-30% better under HEVC.
              wb_gains (the camera's calibrated white balance, if any) ride the
              same LUT — zero extra per-frame cost.
      linear  left-shift by 4 to fill the 16-bit container (0..65520). No WB —
              the linear master stays raw sensor data for later processing.
    A duplicated frame (camera underrun) rewrites the previous *encoded* bytes,
    so duplicates cost no pixel work at all.

    Timing model
    ~~~~~~~~~~~~
    A monotonic deadline is advanced by exactly one frame interval on each
    tick, so the output PTS stream is perfectly uniform at the target fps
    regardless of capture jitter.  If the camera delivers faster than the
    target rate, intermediate frames are dropped (the mailbox keeps only the
    freshest). If it delivers slower (e.g. long exposure), the last frame is
    duplicated — the encoded video never stalls.

    The inner sleep uses coarse_sleep + 1 ms spin to hit the deadline without
    burning a full CPU core on a tight-loop.
    """
    interval = 1.0 / fps
    last_out: Optional[np.ndarray] = None
    deadline = time.monotonic() + interval

    while not stop.is_set():
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

        bayer = box.take()
        if bayer is not None:
            rgb = convert.demosaic(bayer, cv_code)        # linear, 0..4095
            if transfer == "linear":
                out = rgb << 4
            else:
                out = convert.tonemap_linear(
                    rgb, curve=transfer, max_in=4095.0, out_dtype=np.uint16,
                    wb_gains=wb_gains,
                )
            last_out = out
        elif last_out is not None:
            out = last_out           # duplicate on camera underrun
        else:
            continue                 # no data yet; wait for the first frame

        # Write frame as raw bytes.  view(uint8) reinterprets the uint16 data
        # as bytes without a copy (the LUT apply / demosaic both return
        # C-contiguous arrays).
        try:
            proc.stdin.write(out.view(np.uint8))
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
        "--demosaic", choices=list(enums.DEMOSAIC_QUALITY), default="ea",
        help=(
            "Demosaic algorithm. 'ea' (edge-aware, default) suppresses the "
            "zipper/false-colour fringing that 'bilinear' shows on edges."
        ),
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
    ap.add_argument(
        "--no-wb", action="store_true",
        help=(
            "Don't apply the camera's calibrated white-balance gains. By "
            "default, gains from calibrations/calib_<serial>.json (if present) "
            "are folded into the bt709 transfer LUT — zero per-frame cost. "
            "Linear masters are never white-balanced (they stay raw)."
        ),
    )
    ap.add_argument(
        "--transfer", choices=["bt709", "linear"], default="bt709",
        help=(
            "Transfer function baked into the encoded file. "
            "'bt709' (default) applies BT.709 gamma before encoding — "
            "20-30%% smaller files because H.265 quantisation is designed for "
            "gamma content. Invert on decode with: "
            "ffmpeg -vf zscale=transferin=bt709:transfer=linear ... "
            "'linear' stores scene-linear light with no OETF applied."
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
    cam = HTCamera(serial=args.serial, demosaic_quality=args.demosaic)
    cam.set_ae(False)                        # fixed exposure → stable fps
    cam.set_frame_speed(enums.FRAME_SPEED_HIGH)  # max USB throughput; avoid underrun
    cam.set_exposure_ms(args.exposure_ms)
    cam.set_analog_gain(args.gain)

    # Discard the first grab: the oldest buffered frame may pre-date our
    # exposure setting.
    cam.grab_bayer12(timeout_ms=2000)
    test_frame, _ = cam.grab_bayer12(timeout_ms=2000)
    if test_frame is None:
        raise RuntimeError("Camera returned no frame — check connection.")
    h, w = test_frame.shape[:2]

    wb = None
    if args.transfer != "linear" and not args.no_wb:
        wb = cam.wb_gains
    wb_tag = ("off (linear master)" if args.transfer == "linear" else
              "off (--no-wb)" if args.no_wb else
              f"{np.round(wb, 3).tolist()}" if wb is not None else
              "none calibrated")
    print(
        f"[info] {w}x{h} @ {args.fps} fps, "
        f"exposure {args.exposure_ms:.1f} ms, gain {args.gain:.2f}x, "
        f"encoder {encoder} ({_ENCODERS[encoder]['codec']}), quality {args.quality}, "
        f"transfer {args.transfer}, wb {wb_tag}"
    )

    # ── Start ffmpeg ─────────────────────────────────────────────────────────
    cmd = _ffmpeg_cmd(encoder, w, h, args.fps, args.quality, args.output, args.transfer)
    print(f"[info] ffmpeg command:\n  {' '.join(cmd)}\n")
    # bufsize=0: each frame is one big write; buffering would just memcpy
    # ~30 MB through a BufferedWriter first.
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, bufsize=0)

    # ── Start capture + encode threads ───────────────────────────────────────
    stop = threading.Event()
    box = _Mailbox()

    cap_thread = threading.Thread(
        target=_capture_loop, args=(cam, box, stop), daemon=True,
        name="hteng-capture",
    )
    enc_thread = threading.Thread(
        target=_encode_loop,
        args=(proc, box, stop, args.fps, cam._cv_code, args.transfer, wb),
        name="hteng-encode",
    )
    cap_thread.start()
    enc_thread.start()

    # ── Wait for duration / Ctrl+C ───────────────────────────────────────────
    # Sleep in short slices (rather than one long sleep / blocking join) so a
    # Ctrl+C is noticed promptly and the encoder thread can't keep us hanging.
    stopped_intentionally = False
    try:
        print("[info] Recording… press Ctrl+C to stop.")
        t_end = None if args.duration is None else time.monotonic() + args.duration
        while enc_thread.is_alive():
            if t_end is not None and time.monotonic() >= t_end:
                stopped_intentionally = True
                break
            if proc.poll() is not None:    # ffmpeg died (e.g. NVENC error)
                print("[warn] ffmpeg exited early — stopping.")
                break
            time.sleep(0.1)
    except KeyboardInterrupt:
        stopped_intentionally = True
        print()                            # newline after ^C

    # ── Tear down ─────────────────────────────────────────────────────────────
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

    # ffmpeg exits with 255 (or -SIGINT on some platforms) when it handles a
    # SIGINT gracefully — that's expected and the file is complete.
    clean_exit = ret == 0 or (stopped_intentionally and ret in (255, -signal.SIGINT))
    if clean_exit:
        print(f"[info] Saved → {args.output}")
    else:
        print(f"[warn] ffmpeg exited with code {ret} — output may be incomplete.")


if __name__ == "__main__":
    main()
