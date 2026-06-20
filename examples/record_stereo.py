#!/usr/bin/env python3
"""
Record a synchronized HTENG stereo pair to two frame-matched 10-bit HEVC MP4s.

This is the two-camera sibling of ``record.py``. The per-frame pixel path
(unpack -> demosaic -> BT.709/linear transfer -> rgb48le -> ffmpeg) is identical;
see ``record.py`` for the full rationale on encoders, transfer functions, and
colour metadata. The only new thing here is *synchronization*.

Frame matching (timestamp-paired, left = master)
-------------------------------------------------
Both cameras free-run at max USB throughput, and we call ``reset_timestamp()``
on the pair back-to-back at startup so their hardware frame timestamps share a
common time base (to within the gap between the two calls, ~50-200 µs).

The LEFT camera is the master clock: it streams into a single-slot mailbox
(newest frame wins, carrying its timestamp). The RIGHT camera streams into a
small ring buffer of its most recent frames (also timestamped).

A single encode loop ticks at the 1/fps deadline. On each tick it takes the
freshest LEFT frame as the reference, then picks the RIGHT frame from the ring
whose timestamp is *closest* to the left frame's — the best temporal match
available, rather than just "the newest right frame". Because the reference left
frame is already a few ms behind the tick instant, the ring holds right frames
both before and after it, so the nearest-neighbour search genuinely brackets the
left timestamp (no explicit lag needed). The measured skew (t_right - t_left)
for every emitted frame is logged to sync_log.csv.

Both files are fed one frame per tick, so frame N of left and frame N of right
are the same tick — the MP4s stay identical in length and aligned by frame
index. A tick only writes once both cameras have produced a frame (so the
streams start together); if the master underruns, the last pair is duplicated
rather than desyncing.

This nearest-timestamp pairing (vs. plain newest-newest) removes the fixed phase
offset between two free-running sensors, the slow drift between their unlocked
clocks, and one-off USB jitter — bounding skew to ~half a camera frame interval.
For true simultaneous exposure, wire a hardware trigger.

Output
------
Everything lands in one folder (the positional argument):

    <out_dir>/
      left.mp4                       left camera, frame-matched to right.mp4
      right.mp4                      right camera
      sync_log.csv                   per-frame t_left/t_right/skew (µs)
      recording.json                 manifest: fps, exposure, per-camera ROI
                                     (with the intrinsics offset note)
      calib_<serial_left>.json       copy of each camera's calibration (if found)
      calib_<serial_right>.json
      stereo_<L>_<R>.json            copy of the cam2cam transform (if found)

The calibration files are copied verbatim from the calibration search path
(HTENG_CALIB_DIR, ./calibrations, .) so the recording is self-describing.

Usage
-----
  python record_stereo.py take01/                     # first two cameras found
  python record_stereo.py --left 046060323003 --right 046060323006 take01/
  python record_stereo.py --duration 10 --fps 60 --exposure-ms 12 fast/
  python record_stereo.py --encoder x265 --transfer linear master/
"""

import argparse
import atexit
import csv
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np

from hteng_camera import HTCamera, calibration, convert, enums, list_cameras


# ---------------------------------------------------------------------------
# Encoder registry — identical to record.py (codec, output pixel format, and the
# quality flag each encoder uses). See record.py for the full commentary.
# ---------------------------------------------------------------------------

_ENCODERS = {
    "nvenc": {
        "codec": "hevc_nvenc",
        "pix_fmt": "p010le",
        "extra": ["-profile:v", "main10", "-preset", "p4"],
        "quality": lambda q: ["-rc", "constqp", "-qp", str(q), "-b:v", "0"],
    },
    "videotoolbox": {
        "codec": "hevc_videotoolbox",
        "pix_fmt": "p010le",
        "extra": ["-profile:v", "main10"],
        "quality": lambda q: ["-q:v", str(max(1, min(100, 100 - 2 * q)))],
    },
    "x265": {
        "codec": "libx265",
        "pix_fmt": "yuv420p10le",
        "extra": ["-preset", "medium"],
        "quality": lambda q: ["-crf", str(q)],
    },
}

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
    """Fail early with an actionable message if ffmpeg / the encoder isn't usable."""
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
# FFmpeg command builder — identical to record.py (one process per camera)
# ---------------------------------------------------------------------------

def _ffmpeg_cmd(encoder: str, w: int, h: int, fps: int, quality: int,
                output: str, transfer: str = "bt709") -> list[str]:
    """Build the ffmpeg invocation for 10-bit HEVC encoding with ``encoder``.

    See record.py for the full rationale; the transfer curve is baked in upstream
    (Python), so ffmpeg only does the BT.709 RGB->YUV matrix conversion and stamps
    the colour metadata into the HEVC bitstream.
    """
    enc = _ENCODERS[encoder]

    _TRANSFER_META = {"bt709": "1", "linear": "8"}
    _TRANSFER_TRC = {"bt709": "bt709", "linear": "linear"}

    return [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-pix_fmt", "rgb48le",
        "-s", f"{w}x{h}",
        "-r", str(fps),
        "-i", "pipe:0",
        "-vf", (
            "scale="
            "in_color_matrix=bt709:out_color_matrix=bt709:"
            "in_range=full:out_range=full"
        ),
        "-c:v", enc["codec"],
        *enc["extra"],
        "-pix_fmt", enc["pix_fmt"],
        *enc["quality"](quality),
        "-bsf:v", (
            f"hevc_metadata="
            f"transfer_characteristics={_TRANSFER_META[transfer]}:"
            f"colour_primaries=1:matrix_coefficients=1:video_full_range_flag=1"
        ),
        "-colorspace", "bt709",
        "-color_primaries", "bt709",
        "-color_trc", _TRANSFER_TRC[transfer],
        "-color_range", "pc",
        "-movflags", "+faststart",
        output,
    ]


# ---------------------------------------------------------------------------
# Producers: master mailbox (left) + timestamped ring buffer (right)
# ---------------------------------------------------------------------------

class _Mailbox:
    """Single-slot latest-frame handoff for the MASTER (left) camera.

    Holds the newest ``(timestamp_us, bayer)`` only — the encoder always wants
    the freshest reference frame, and at ~10 MB per plane, depth is real memory.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._item: Optional[tuple[int, np.ndarray]] = None

    def put(self, ts: int, frame: np.ndarray) -> None:
        with self._lock:
            self._item = (ts, frame)

    def take(self) -> Optional[tuple[int, np.ndarray]]:
        with self._lock:
            item, self._item = self._item, None
            return item


class _Ring:
    """Bounded timestamped frame history for the SLAVE (right) camera.

    Keeps the last ``depth`` ``(timestamp_us, bayer)`` planes so the encoder can
    pick the one nearest a given left timestamp. Memory is ``depth`` planes
    (~10 MB each), the price of timestamp pairing over a 1-slot mailbox.
    """

    def __init__(self, depth: int) -> None:
        self._lock = threading.Lock()
        self._buf: deque[tuple[int, np.ndarray]] = deque(maxlen=depth)

    def put(self, ts: int, frame: np.ndarray) -> None:
        with self._lock:
            self._buf.append((ts, frame))

    def nearest(self, ts: int) -> Optional[tuple[int, np.ndarray]]:
        """The buffered frame whose timestamp is closest to ``ts`` (or None)."""
        with self._lock:
            if not self._buf:
                return None
            return min(self._buf, key=lambda it: abs(it[0] - ts))


def _capture_master(cam: HTCamera, box: _Mailbox, stop: threading.Event) -> None:
    """Grab left (master) Bayer planes, keeping only the newest in the mailbox."""
    while not stop.is_set():
        bayer, info = cam.grab_bayer12()
        if bayer is None:
            continue
        box.put(info["time"], bayer)


def _capture_slave(cam: HTCamera, ring: _Ring, stop: threading.Event) -> None:
    """Grab right (slave) Bayer planes into the ring so the encoder can pick the
    one nearest each master timestamp."""
    while not stop.is_set():
        bayer, info = cam.grab_bayer12()
        if bayer is None:
            continue
        ring.put(info["time"], bayer)


# ---------------------------------------------------------------------------
# Consumer: fixed-rate encoder loop, left as master, right matched by timestamp
# ---------------------------------------------------------------------------

def _encode(bayer: np.ndarray, cv_code: int, transfer: str, wb) -> np.ndarray:
    """Demosaic one Bayer plane and apply the transfer encode (see record.py)."""
    rgb = convert.demosaic(bayer, cv_code)               # linear, 0..4095
    if transfer == "linear":
        return rgb << 4
    return convert.tonemap_linear(
        rgb, curve=transfer, max_in=4095.0, out_dtype=np.uint16, wb_gains=wb,
    )


def _encode_loop_stereo(
    left: dict,
    right: dict,
    stop: threading.Event,
    fps: int,
    transfer: str,
    sync_csv: Path,
) -> None:
    """At each 1/fps deadline, encode the freshest LEFT frame and the RIGHT frame
    whose timestamp is nearest to it, writing one frame to each ffmpeg pipe.

    ``left``/``right`` are per-camera dicts with keys ``proc``, ``cv_code``,
    ``wb``; ``left`` also has ``box`` (a :class:`_Mailbox`) and ``right`` has
    ``ring`` (a :class:`_Ring`).

    Frame matching
    ~~~~~~~~~~~~~~
    The left camera is the master clock. Each tick takes the newest left frame as
    the reference and selects the right frame from the ring with the closest
    timestamp — the best temporal pair available. Both pipes get exactly one
    frame per tick, so the files stay equal-length and index-aligned. A tick
    writes nothing until both a left reference and a right match exist (streams
    start together); if the master underruns, the previous encoded pair is
    rewritten rather than desyncing. The per-frame skew (t_right - t_left) is
    written to ``sync_csv``.

    The timing model (monotonic deadline advanced one interval per tick,
    coarse-sleep + 1 ms spin) is exactly record.py's.
    """
    interval = 1.0 / fps
    deadline = time.monotonic() + interval
    last_pair: Optional[tuple[np.ndarray, np.ndarray]] = None
    frame_idx = 0

    with open(sync_csv, "w", newline="") as f:
        log = csv.writer(f)
        log.writerow(["frame", "t_left_us", "t_right_us", "skew_us", "fresh"])

        while not stop.is_set():
            now = time.monotonic()
            remaining = deadline - now

            if remaining > 0.001:
                time.sleep(remaining - 0.001)
                continue

            # Snap forward if we fell more than one frame behind (e.g. GIL pause).
            if remaining < -interval:
                deadline = now + interval
            else:
                deadline += interval

            ref = left["box"].take()
            if ref is not None:
                t_left, bayer_left = ref
                match = right["ring"].nearest(t_left)
                if match is None:
                    continue                     # right not warmed up yet
                t_right, bayer_right = match
                out_left = _encode(bayer_left, left["cv_code"], transfer, left["wb"])
                out_right = _encode(bayer_right, right["cv_code"], transfer, right["wb"])
                last_pair = (out_left, out_right)
                log.writerow([frame_idx, t_left, t_right, t_right - t_left, 1])
            elif last_pair is not None:
                out_left, out_right = last_pair  # master underrun → duplicate pair
                log.writerow([frame_idx, "", "", "", 0])
            else:
                continue                         # nothing to write yet

            try:
                left["proc"].stdin.write(out_left.view(np.uint8))
                right["proc"].stdin.write(out_right.view(np.uint8))
            except (BrokenPipeError, OSError):
                stop.set()                       # a dead pipe stops the pair
                break
            frame_idx += 1

    for s in (left, right):
        try:
            s["proc"].stdin.close()
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Calibration copy: snapshot the calib JSONs alongside the videos
# ---------------------------------------------------------------------------

def _copy_calibration(out_dir: Path, serial_left: str, serial_right: str) -> None:
    """Copy each camera's calib_<serial>.json and the stereo transform into
    ``out_dir`` so the recording is self-describing.

    Files are located on the same search path the library uses to auto-load them
    (HTENG_CALIB_DIR, ./calibrations, .). Missing files are reported, not fatal —
    you can record before the rig is calibrated.
    """
    dirs = calibration.search_dirs()

    def _locate(*names: Path) -> Optional[Path]:
        """First of the given filenames that exists on the search path."""
        for d in dirs:
            for name in names:
                p = d / name.name
                if p.exists():
                    return p
        return None

    targets = {
        f"calib_{serial_left}.json":
            (calibration.CameraCalibration.default_path(serial_left),),
        f"calib_{serial_right}.json":
            (calibration.CameraCalibration.default_path(serial_right),),
        # Stereo file naming is order-sensitive; accept either L/R ordering.
        f"stereo_{serial_left}_{serial_right}.json": (
            calibration.StereoCalibration.default_path(serial_left, serial_right),
            calibration.StereoCalibration.default_path(serial_right, serial_left),
        ),
    }

    copied, missing = [], []
    for label, candidates in targets.items():
        src = _locate(*candidates)
        if src is None:
            missing.append(label)
            continue
        shutil.copy2(src, out_dir / src.name)
        copied.append(src.name)

    if copied:
        print(f"[info] copied calibration: {', '.join(copied)}")
    if missing:
        print(f"[warn] calibration not found (skipped): {', '.join(missing)}")


def _find_stereo_for(serials: list[str]):
    """Load the StereoCalibration whose (left, right) serials are both present,
    so the recording can adopt the calibration's L/R convention. Returns a
    StereoCalibration or None. Tries both orderings of the present serials."""
    for a in serials:
        for b in serials:
            if a == b:
                continue
            for d in calibration.search_dirs():
                p = calibration.StereoCalibration.default_path(a, b, d)
                if p.exists():
                    try:
                        return calibration.StereoCalibration.load(p)
                    except (OSError, ValueError, KeyError, json.JSONDecodeError):
                        pass
    return None


def _side_meta(cam: HTCamera, serial: str, full_wh: tuple[int, int],
               exposure_ms: float, gain_x: float) -> dict:
    """Per-camera manifest entry: the applied ROI (read back from the camera)
    plus the full sensor size, so the full-frame intrinsics can be corrected."""
    res = cam.current_resolution()
    return {
        "serial": serial,
        "full_sensor": {"width": int(full_wh[0]), "height": int(full_wh[1])},
        "roi": {
            "x_offset": int(res.iHOffsetFOV),
            "y_offset": int(res.iVOffsetFOV),
            "width": int(res.iWidthFOV),
            "height": int(res.iHeightFOV),
        },
        "exposure_ms": round(float(exposure_ms), 3),
        "gain_x": round(float(gain_x), 3),
        "calibration_file": f"calib_{serial}.json",
    }


def _write_manifest(out_dir: Path, args, encoder: str, left: dict, right: dict) -> None:
    """Write recording.json describing how the videos were captured, including
    the per-camera ROI needed to adjust the copied full-frame intrinsics.

    The copied calib_<serial>.json holds FULL-SENSOR intrinsics. A centered ROI
    crop leaves fx, fy and distortion unchanged but moves the principal point:
        cx_roi = cx_full - roi.x_offset
        cy_roi = cy_full - roi.y_offset
    and the image size becomes roi.width x roi.height. This note + the ROI are
    recorded so that step is unambiguous later.
    """
    manifest = {
        "format": "hteng-camera-stereo-recording/1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "fps": args.fps,
        "encoder": encoder,
        "quality": args.quality,
        "transfer": args.transfer,
        "demosaic": args.demosaic,
        "roi_height_fraction": args.roi_height,
        "files": {
            "left": "left.mp4",
            "right": "right.mp4",
            "sync_log": "sync_log.csv",
            "stereo_transform":
                f"stereo_{left['serial']}_{right['serial']}.json",
        },
        "left": _side_meta(left["cam"], left["serial"], left["full"],
                           left["exp"], left["gain"]),
        "right": _side_meta(right["cam"], right["serial"], right["full"],
                            right["exp"], right["gain"]),
        "intrinsics_note": (
            "calib_<serial>.json intrinsics are FULL-SENSOR. For this recording's "
            "ROI, shift the principal point: cx_roi = cx_full - roi.x_offset, "
            "cy_roi = cy_full - roi.y_offset. fx, fy and distortion are unchanged; "
            "image size is roi.width x roi.height. The stereo R|t is ROI-invariant."
        ),
    }
    path = out_dir / "recording.json"
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[info] wrote manifest → {path}")


def _report_sync(sync_csv: Path) -> None:
    """Print a one-line summary of the measured left/right skew from the log."""
    try:
        with open(sync_csv, newline="") as f:
            skews = [abs(int(row["skew_us"]))
                     for row in csv.DictReader(f)
                     if row["fresh"] == "1" and row["skew_us"] != ""]
    except (OSError, ValueError):
        return
    if not skews:
        return
    skews.sort()
    mean = sum(skews) / len(skews)
    median = skews[len(skews) // 2]
    print(
        f"[info] frame sync skew |t_right - t_left|: "
        f"median {median / 1000:.2f} ms, mean {mean / 1000:.2f} ms, "
        f"max {max(skews) / 1000:.2f} ms over {len(skews)} matched frames"
    )


# ---------------------------------------------------------------------------
# Camera setup helper
# ---------------------------------------------------------------------------

def _open_and_configure(serial: str, args, tag: str
                        ) -> tuple[HTCamera, float, float, tuple[int, int], tuple[int, int]]:
    """Open one camera, apply the fixed-exposure / max-throughput config, and
    *verify it latched*. Returns
    ``(cam, exposure_ms, gain_x, full_sensor_wh, applied_hw)`` — the values the
    camera actually reports, the uncropped sensor size (for the manifest /
    intrinsics offset), and the (h, w) of a real delivered frame (for sizing the
    encoder). We don't declare success until a frame actually arrives, so a cold
    or wedged stream keeps retrying here instead of failing downstream.

    Why the readback loop: control writes go through HTCamera._ctrl, which only
    retries on a non-zero SDK status. This hardware can return 0 (success) from a
    write that didn't actually take — most often right after the wedged-channel
    (-52) reconnect that can happen while opening the second camera. The classic
    symptom is set_ae(False) not sticking on one camera: auto-exposure stays on,
    the sensor exposes long in a dim scene, its frame interval balloons, and the
    stereo skew blows up. So we set, start streaming (some values only latch once
    playing), read back, and re-apply until exposure and gain match — or warn.
    """
    cam = HTCamera(serial=serial, demosaic_quality=args.demosaic)
    cam.set_frame_speed(enums.FRAME_SPEED_HIGH)  # set first: changing speed can reset exposure

    # Full (uncropped) sensor size, read before any ROI — needed to offset the
    # full-frame calibration intrinsics for a cropped recording.
    res0 = cam.current_resolution()
    full_w, full_h = int(res0.iWidthFOV), int(res0.iHeightFOV)

    # Optional centered height crop. Sensor readout/USB is row-paced (~constant
    # µs/row), so cropping ROWS — not columns — is what raises fps and shrinks the
    # frame interval (and thus the stereo skew bound). Full width is kept.
    if args.roi_height < 1.0:
        probe, _ = cam.grab_bayer12(timeout_ms=2000)
        if probe is None:
            raise RuntimeError(f"{tag} camera: no frame to probe ROI resolution.")
        full_h, full_w = probe.shape[:2]            # ground-truth full dims
        keep_h = max(16, int(full_h * args.roi_height) // 16 * 16)  # 16-px aligned
        oy = ((full_h - keep_h) // 2) // 16 * 16                    # centered
        try:
            cam.set_roi(full_w, keep_h, 0, oy)
        except RuntimeError as e:
            raise RuntimeError(
                f"{tag} camera rejected ROI {full_w}x{keep_h}@(0,{oy}): {e}")
        print(f"[info] {tag} ROI: {full_w}x{full_h} -> {full_w}x{keep_h} "
              f"(keep {args.roi_height:.0%} of rows, centered)")

    want_exp, want_gain = args.exposure_ms, args.gain
    exp_tol = max(0.5, 0.1 * want_exp)           # SDK quantises exposure to line-time
    gain_tol = max(0.2, 0.1 * want_gain)         # and gain to a discrete step
    got_exp = got_gain = None
    applied_hw = None                            # (h, w) of a real delivered frame
    last_err = None
    for attempt in range(8):
        # The setters route through HTCamera._ctrl, which heals a wedged (-52)
        # control channel via CameraReConnect. The readback GETTERS don't — they
        # call the SDK directly — so a -52 on readback (or a setter that exhausts
        # its own retries) is transient: swallow it and loop, and the next setter
        # pass reconnects the channel. Don't let it abort the whole recording.
        try:
            cam.set_ae(False)                    # fixed exposure -> stable fps
            cam.set_exposure_ms(want_exp)
            cam.set_analog_gain(want_gain)
            frame, _ = cam.grab_bayer12(timeout_ms=2000)   # also starts streaming
            if frame is not None:
                applied_hw = frame.shape[:2]
            got_exp = cam.get_exposure_ms()
            got_gain = cam.get_analog_gain()
        except Exception as e:                   # SdkError (-52 wedged channel), etc.
            last_err = e
            time.sleep(0.3)
            continue
        # Require a real frame AND the latched exposure/gain before success.
        if (applied_hw is not None
                and abs(got_exp - want_exp) <= exp_tol
                and abs(got_gain - want_gain) <= gain_tol):
            return cam, got_exp, got_gain, (full_w, full_h), applied_hw
        time.sleep(0.15)

    if applied_hw is None:
        # Opened, but the stream never delivered a frame — a genuinely bad
        # connection / wedged device. Raise so the caller releases the handle.
        raise RuntimeError(
            f"{tag} camera ({serial}) opened but delivered no frames after 8 "
            f"tries (last error: {last_err}). Unplug/replug and retry.")

    if got_exp is None:
        # Frames arrive but the control channel stayed wedged for readback. The
        # set_* commands go through _ctrl and are reliable, so proceed with the
        # requested values rather than crashing — just unverified.
        print(
            f"[warn] {tag} camera ({serial}) control channel stayed wedged; "
            f"proceeding with requested {want_exp:.1f} ms / {want_gain:.2f}x "
            f"unverified (last error: {last_err})."
        )
        return cam, want_exp, want_gain, (full_w, full_h), applied_hw

    print(
        f"[warn] {tag} camera ({serial}) did not accept exposure/gain after 8 "
        f"tries: asked {want_exp:.1f} ms / {want_gain:.2f}x, got "
        f"{got_exp:.1f} ms / {got_gain:.2f}x. Auto-exposure may be stuck on — "
        "the stereo skew will be large. Try unplug/replug, or rerun."
    )
    return cam, got_exp, got_gain, (full_w, full_h), applied_hw


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

class _KeepAwake:
    """Keep a Mac awake for the whole recording — including with the lid CLOSED.

    Two layers, because macOS has two independent sleep paths:

      caffeinate -imsuw <pid>   blocks *idle* sleep (system/display/disk). The
                                -w ties it to our pid so it can't outlive us
                                (it exits the instant this process does, even on
                                a crash). This is the part `caffeinate <cmd>`
                                from the shell already covers.
      pmset -a disablesleep 1   blocks *clamshell* (lid-close) sleep — the path
                                caffeinate cannot touch. Without it, shutting
                                the lid sleeps the Mac, drops USB, and kills the
                                capture regardless of any caffeinate assertion.
                                Needs sudo; we save the prior value and restore
                                it on exit.

    No-op off macOS. Any failure degrades to a warning — the recording still
    runs, it just isn't guaranteed to survive a lid close.
    """

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled and platform.system() == "Darwin"
        self._caffeinate: Optional[subprocess.Popen] = None
        self._pmset_prev: Optional[int] = None   # prior SleepDisabled (0/1) to restore
        self._lid_set = False                    # did WE flip disablesleep -> 1?
        self._prev_handlers: dict = {}           # SIGTERM/SIGHUP handlers we replaced
        self._ka_stop: Optional[threading.Event] = None  # sudo keep-alive control
        self._released = False

    def engage(self) -> None:
        if not self.enabled:
            if platform.system() != "Darwin":
                print("[info] keep-awake: not macOS — relying on the OS / your wrapper.")
            return
        # Layer 1 — idle sleep, tied to our pid so it dies with us even on a crash.
        try:
            self._caffeinate = subprocess.Popen(
                ["caffeinate", "-imsuw", str(os.getpid())])
            print("[info] keep-awake: caffeinate running (blocks idle sleep).")
        except FileNotFoundError:
            print("[warn] keep-awake: caffeinate not found — idle sleep NOT blocked.")

        # Layer 2 — lid-close sleep. Needs sudo; prompt once now, while we're
        # still single-threaded and before any camera is open.
        self._pmset_prev = self._read_sleep_disabled()
        if self._set_disablesleep(True, interactive=True):
            self._lid_set = True
            self._start_sudo_keepalive()   # keep creds warm so restore never prompts
            print("[info] keep-awake: pmset disablesleep=1 — safe to close the lid.")
        else:
            print("[warn] keep-awake: could NOT disable lid-close sleep (needs sudo).\n"
                  "       Closing the lid WILL sleep the Mac and kill the recording.\n"
                  "       Fix: run this script under sudo, or first run\n"
                  "           sudo pmset -a disablesleep 1\n"
                  "       (and `sudo pmset -a disablesleep 0` when you're done).")

        # Restore on EVERY exit path: normal return / sys.exit / unhandled
        # exception are caught by atexit; SIGTERM & SIGHUP would otherwise kill
        # us before atexit runs, so we trap them and release first. (SIGINT is
        # owned by main()'s teardown; SIGKILL can't be trapped — caffeinate still
        # dies via -w, but pmset can't be restored on a -9.)
        atexit.register(self.release)
        self._install_signal_restore()

    def _install_signal_restore(self) -> None:
        for sig in (signal.SIGTERM, signal.SIGHUP):
            try:
                prev = signal.getsignal(sig)
            except (ValueError, OSError):
                continue
            if prev is signal.SIG_IGN:
                continue   # already ignored (e.g. launched under nohup) — leave it
            self._prev_handlers[sig] = prev

            def handler(signum, _frame, _prev=prev):
                self.release()
                # Re-raise through whatever was there before (e.g. the library's
                # camera-uninit handler) and then the default action, so the
                # signal still does its job instead of being swallowed.
                signal.signal(signum, _prev if callable(_prev) else signal.SIG_DFL)
                os.kill(os.getpid(), signum)

            try:
                signal.signal(sig, handler)
            except (ValueError, OSError):
                pass

    def _start_sudo_keepalive(self) -> None:
        # `release` uses `sudo -n` (non-interactive) so it never hangs on a
        # prompt mid-teardown / in a signal handler. But the credential cached
        # at engage expires (~5 min), which would break restore on a long run.
        # Refresh it every 60 s so the timestamp stays valid for the whole
        # recording. No-op when already root.
        if os.geteuid() == 0:
            return
        self._ka_stop = threading.Event()

        def loop():
            while not self._ka_stop.wait(60):
                subprocess.run(["sudo", "-n", "-v"], capture_output=True)

        threading.Thread(target=loop, daemon=True, name="sudo-keepalive").start()

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        if self._ka_stop is not None:
            self._ka_stop.set()                  # stop refreshing sudo creds
        # Restore lid behaviour first — it's the change that persists till reboot.
        # Only undo it if WE set it (don't stomp a value the user chose). If we
        # couldn't read the prior value, fall back to 0 (off) rather than leak a
        # permanent no-sleep state. Warm credential keeps this from blocking.
        if self._lid_set:
            target = self._pmset_prev if self._pmset_prev is not None else 0
            if not self._set_disablesleep(bool(target), interactive=False):
                print("[warn] keep-awake: could not restore pmset disablesleep — "
                      f"run `sudo pmset -a disablesleep {target}` yourself.")
        if self._caffeinate is not None and self._caffeinate.poll() is None:
            self._caffeinate.terminate()
            try:
                self._caffeinate.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._caffeinate.kill()

    @staticmethod
    def _read_sleep_disabled() -> Optional[int]:
        try:
            out = subprocess.check_output(["pmset", "-g"], text=True)
        except Exception:
            return None
        for line in out.splitlines():
            if "SleepDisabled" in line:
                return 1 if line.split()[-1] == "1" else 0
        return 0   # key absent ⇒ feature off

    @staticmethod
    def _set_disablesleep(on: bool, interactive: bool) -> bool:
        cmd = ["pmset", "-a", "disablesleep", "1" if on else "0"]
        if os.geteuid() != 0:
            # -n (non-interactive) unless we're allowed to prompt AND have a tty.
            if interactive and sys.stdin.isatty():
                print("[info] keep-awake: sudo needed to disable lid-close sleep…")
                cmd = ["sudo", *cmd]
            else:
                cmd = ["sudo", "-n", *cmd]
        try:
            return subprocess.run(cmd, capture_output=not interactive).returncode == 0
        except Exception:
            return False


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Record a synchronized HTENG stereo pair to two "
                    "frame-matched 10-bit HEVC MP4s",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "out_dir",
        help="Output folder for left.mp4, right.mp4, and copied calibration JSONs",
    )
    ap.add_argument(
        "--left", default=None,
        help="Serial of the LEFT camera (default: first camera enumerated)",
    )
    ap.add_argument(
        "--right", default=None,
        help="Serial of the RIGHT camera (default: second camera enumerated)",
    )
    ap.add_argument(
        "--encoder", choices=["auto", *_ENCODERS], default="auto",
        help="HEVC encoder. 'auto' maps by platform (Linux->nvenc, "
             "macOS->videotoolbox); or force nvenc / videotoolbox / x265.",
    )
    ap.add_argument("--fps", type=int, default=30, help="Target output frame rate")
    ap.add_argument(
        "--duration", type=float, default=None,
        help="Recording duration in seconds (omit to record until Ctrl+C)",
    )
    ap.add_argument(
        "--exposure-ms", type=float, default=5.0,
        help="Exposure time in ms. Must be < 1000/fps or both cameras run below "
             "the target rate and frames are duplicated. A short exposure keeps "
             "the sensors running fast, which tightens the stereo pairing. AE is "
             "disabled for stable, matched fps.",
    )
    ap.add_argument("--gain", type=float, default=3.0, help="Analog gain multiplier")
    ap.add_argument(
        "--demosaic", choices=list(enums.DEMOSAIC_QUALITY), default="ea",
        help="Demosaic algorithm. 'ea' (edge-aware, default) suppresses the "
             "zipper/false-colour fringing that 'bilinear' shows on edges.",
    )
    ap.add_argument(
        "--quality", type=int, default=18,
        help="Constant-quality level, QP/CRF-style (0=best, 51=worst). 18 is "
             "visually lossless; 24-28 gives roughly half the file size.",
    )
    ap.add_argument(
        "--no-wb", action="store_true",
        help="Don't apply each camera's calibrated white-balance gains. By "
             "default, gains from its calibration are folded into the bt709 "
             "transfer LUT (zero per-frame cost). Linear masters are never "
             "white-balanced (they stay raw).",
    )
    ap.add_argument(
        "--transfer", choices=["bt709", "linear"], default="bt709",
        help="Transfer function baked into the encoded files. 'bt709' (default) "
             "applies BT.709 gamma before encoding (20-30%% smaller files). "
             "'linear' stores scene-linear light with no OETF applied.",
    )
    ap.add_argument(
        "--roi-height", type=float, default=1.0,
        help="Keep this fraction of sensor ROWS, centered, at full width "
             "(e.g. 0.85 = central 85%% of rows). Readout is row-paced, so "
             "cropping height — not width — is what raises fps and tightens the "
             "stereo skew bound. 1.0 = full frame.",
    )
    ap.add_argument(
        "--sync-buffer", type=int, default=8,
        help="Depth of the right-camera ring buffer the master's timestamp is "
             "paired against. More frames give the nearest-timestamp search more "
             "to choose from but cost ~10 MB each. 8 is plenty when the sensors "
             "run faster than --fps.",
    )
    ap.add_argument(
        "--no-keep-awake", action="store_true",
        help="Don't keep the Mac awake. By default the recording blocks idle "
             "sleep (caffeinate) AND lid-close sleep (pmset disablesleep, needs "
             "sudo) so you can shut the laptop and let it run. Both are restored "
             "on exit. No-op off macOS.",
    )
    args = ap.parse_args()

    if args.sync_buffer < 1:
        sys.exit("[error] --sync-buffer must be >= 1.")
    if not 0.0 < args.roi_height <= 1.0:
        sys.exit("[error] --roi-height must be in (0, 1].")

    encoder = _resolve_encoder(args.encoder)
    _preflight_ffmpeg(encoder)

    frame_interval_ms = 1000.0 / args.fps
    if args.exposure_ms >= frame_interval_ms:
        print(
            f"[warn] exposure {args.exposure_ms:.1f} ms >= frame interval "
            f"{frame_interval_ms:.1f} ms — cameras will run below {args.fps} fps "
            "and output frames will be duplicated."
        )

    # ── Resolve which serials are left / right ───────────────────────────────
    cams = list_cameras()
    if len(cams) < 2:
        sys.exit(
            f"[error] need two cameras for a stereo recording; found "
            f"{len(cams)} ({', '.join(c['serial'] for c in cams) or 'none'})."
        )
    # Decide left/right deterministically. USB enumeration order is NOT stable
    # across runs/replug, so it must not silently pick sides. Priority:
    #   1. explicit --left/--right
    #   2. the stereo calibration's serial_left/serial_right (matches the L/R the
    #      rig was calibrated with — verified physical via the baseline sign)
    #   3. enumeration order, with a loud warning
    present = [c["serial"] for c in cams]
    stereo = _find_stereo_for(present)
    if stereo is not None:
        base_left, base_right, side_src = stereo.serial_left, stereo.serial_right, \
            f"stereo calibration (stereo_{stereo.serial_left}_{stereo.serial_right}.json)"
    else:
        base_left, base_right, side_src = present[0], present[1], "USB enumeration order"

    serial_left = args.left if args.left is not None else base_left
    serial_right = args.right if args.right is not None else base_right
    if args.left is not None or args.right is not None:
        side_src = "--left/--right" + (
            "" if (args.left is not None and args.right is not None)
            else f" + {side_src}")

    if serial_left == serial_right:
        sys.exit("[error] left and right resolved to the same serial "
                 f"({serial_left}). Pass distinct --left/--right serials.")
    missing = [s for s in (serial_left, serial_right) if s not in present]
    if missing:
        sys.exit(f"[error] requested serial(s) {missing} not connected. "
                 f"Present: {', '.join(present)}.")
    if stereo is None and args.left is None and args.right is None:
        print("[warn] left/right assigned by USB enumeration order, which is NOT "
              "stable across replug/reboot — sides may swap between runs. Pass "
              "--left/--right, or add a stereo calibration, to pin them.")
    print(f"[info] sides from {side_src}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_left = str(out_dir / "left.mp4")
    out_right = str(out_dir / "right.mp4")

    # Keep the laptop awake (incl. lid closed) for the whole run. Engage here —
    # past all the arg/camera validation, so a trivial error never triggers the
    # sudo prompt, but before we open cameras or start threads so the one-time
    # prompt is clean and single-threaded. release() is idempotent and also
    # atexit/SIGTERM-registered, so every exit path still undoes it.
    awake = _KeepAwake(enabled=not args.no_keep_awake)
    awake.engage()

    # ── Open + configure both cameras ────────────────────────────────────────
    print(f"[info] Opening cameras  left={serial_left}  right={serial_right} …")
    cam_left = cam_right = None
    try:
        cam_left, exp_left, gain_left, full_left, dims_left = \
            _open_and_configure(serial_left, args, "left")
        cam_right, exp_right, gain_right, full_right, dims_right = \
            _open_and_configure(serial_right, args, "right")
    except BaseException:
        # If the second camera (or a Ctrl+C mid-open) fails after the first is up,
        # release the opened handle immediately — a leaked handle comes back as a
        # wedged (-52) or DEVICE_IS_OPENED (-18) on the next run.
        for c in (cam_left, cam_right):
            if c is not None:
                try:
                    c.close()
                except Exception:
                    pass
        raise

    # Shared hardware time base for the pair (soft sync; see module docstring).
    cam_left.reset_timestamp()
    cam_right.reset_timestamp()

    # Encoder dims come from the real frame each camera delivered during open
    # (no extra grab to time out on).
    dims = {"left": dims_left, "right": dims_right}
    if dims["left"] != dims["right"]:
        print(f"[warn] left {dims['left'][::-1]} and right {dims['right'][::-1]} "
              "resolutions differ — each video uses its own camera's size.")

    wb_left = None if (args.transfer == "linear" or args.no_wb) else cam_left.wb_gains
    wb_right = None if (args.transfer == "linear" or args.no_wb) else cam_right.wb_gains
    print(
        f"[info] {dims['left'][1]}x{dims['left'][0]} @ {args.fps} fps, "
        f"encoder {encoder} ({_ENCODERS[encoder]['codec']}), quality {args.quality}, "
        f"transfer {args.transfer}"
    )
    print(
        f"[info] measured  left: {exp_left:.1f} ms / {gain_left:.2f}x   "
        f"right: {exp_right:.1f} ms / {gain_right:.2f}x   "
        f"(asked {args.exposure_ms:.1f} ms / {args.gain:.2f}x)"
    )

    # ── Copy calibration JSONs + write the recording manifest ────────────────
    _copy_calibration(out_dir, serial_left, serial_right)
    _write_manifest(
        out_dir, args, encoder,
        {"cam": cam_left, "serial": serial_left, "full": full_left,
         "exp": exp_left, "gain": gain_left},
        {"cam": cam_right, "serial": serial_right, "full": full_right,
         "exp": exp_right, "gain": gain_right},
    )

    # ── Start one ffmpeg per camera ──────────────────────────────────────────
    procs = []
    for out_path, (tag, cam) in zip((out_left, out_right),
                                    (("left", cam_left), ("right", cam_right))):
        h, w = dims[tag]
        cmd = _ffmpeg_cmd(encoder, w, h, args.fps, args.quality, out_path, args.transfer)
        # bufsize=0: each frame is one big write; buffering just memcpys ~30 MB.
        procs.append(subprocess.Popen(cmd, stdin=subprocess.PIPE, bufsize=0))
    proc_left, proc_right = procs

    # ── Producers (left mailbox / right ring) + one shared encode thread ─────
    stop = threading.Event()
    box_left = _Mailbox()
    ring_right = _Ring(args.sync_buffer)

    left = {"proc": proc_left,  "box": box_left,
            "cv_code": cam_left._cv_code,  "wb": wb_left}
    right = {"proc": proc_right, "ring": ring_right,
             "cv_code": cam_right._cv_code, "wb": wb_right}
    sync_csv = out_dir / "sync_log.csv"

    cap_threads = [
        threading.Thread(target=_capture_master, args=(cam_left, box_left, stop),
                         daemon=True, name="hteng-capture-left"),
        threading.Thread(target=_capture_slave, args=(cam_right, ring_right, stop),
                         daemon=True, name="hteng-capture-right"),
    ]
    enc_thread = threading.Thread(
        target=_encode_loop_stereo,
        args=(left, right, stop, args.fps, args.transfer, sync_csv),
        name="hteng-encode",
    )
    for t in cap_threads:
        t.start()
    enc_thread.start()

    # ── Wait for duration / Ctrl+C ───────────────────────────────────────────
    # Handle SIGINT ourselves so Ctrl+C tears down in the SAME order as the
    # duration path: set `stop`, let the capture/encode threads exit, THEN close
    # the cameras. hteng_camera installs an import-time SIGINT handler that
    # instead uninits the camera handles immediately — while the capture threads
    # are still grabbing on them. That both contends with the in-flight grab and
    # sends the next grab into HTCamera._ctrl's reconnect-retry loop (20 tries x
    # 0.6 s ≈ 12 s, and it never checks `stop`), which is why an interrupted run
    # took far longer to exit than a timed one. Overriding the handler keeps
    # shutdown fast. A second Ctrl+C restores the library's handler, which
    # uninits the camera handles before exiting — NOT SIG_DFL, which would
    # hard-kill the process and leave the cameras open (wedging the next run).
    interrupted = threading.Event()

    def _on_sigint(_signum, _frame):
        interrupted.set()
        stop.set()
        signal.signal(signal.SIGINT, prev_sigint)      # 2nd Ctrl+C: lib cleanup + exit

    prev_sigint = signal.signal(signal.SIGINT, _on_sigint)

    stopped_intentionally = False
    try:
        print("[info] Recording… press Ctrl+C to stop.")
        t_end = None if args.duration is None else time.monotonic() + args.duration
        while enc_thread.is_alive():
            if interrupted.is_set():
                stopped_intentionally = True
                print()                                # newline after ^C
                break
            if t_end is not None and time.monotonic() >= t_end:
                stopped_intentionally = True
                break
            if proc_left.poll() is not None or proc_right.poll() is not None:
                print("[warn] an ffmpeg exited early — stopping.")
                break
            time.sleep(0.1)
    finally:
        signal.signal(signal.SIGINT, prev_sigint)      # restore for teardown

    # ── Tear down ─────────────────────────────────────────────────────────────
    print("[info] Stopping…")
    stop.set()
    for t in cap_threads:
        t.join(timeout=5)
    enc_thread.join(timeout=10)

    rets = []
    for proc in (proc_left, proc_right):
        try:
            rets.append(proc.wait(timeout=15))
        except subprocess.TimeoutExpired:
            print("[warn] ffmpeg didn't exit in time — killing it.")
            proc.kill()
            rets.append(proc.wait())
    cam_left.close()
    cam_right.close()
    awake.release()                     # restore sleep settings; stop caffeinate

    # ffmpeg exits 255 / -SIGINT on a graceful SIGINT — expected, file is complete.
    def _clean(ret: int) -> bool:
        return ret == 0 or (stopped_intentionally and ret in (255, -signal.SIGINT))

    if all(_clean(r) for r in rets):
        print(f"[info] Saved → {out_left}")
        print(f"[info] Saved → {out_right}")
        print(f"[info] Sync log → {sync_csv}")
        _report_sync(sync_csv)
    else:
        print(f"[warn] ffmpeg exited with codes {rets} — output may be incomplete.")


if __name__ == "__main__":
    main()
