#!/usr/bin/env python3
"""
Record a synchronized stereo + IMU egomotion capture: two frame-matched
10-bit HEVC MP4s plus (optionally) a raw IMU log, all sharing one clock.

This is the two-camera sibling of ``record.py``, extended with an optional
IMU stream for feeding a downstream VIO/SLAM pipeline. The per-frame pixel
path (unpack -> demosaic -> BT.709/linear transfer -> rgb48le -> ffmpeg) is
identical to ``record.py``; see that file for the full rationale on encoders,
transfer functions, and colour metadata. The two new things here are
*camera synchronization* and *IMU capture* (see the two sections below).

Frame matching (timestamp-paired, left = master)
-------------------------------------------------
Both cameras free-run at max USB throughput, and we call ``reset_timestamp()``
on the pair back-to-back at startup so their hardware frame timestamps share a
common time base (to within the gap between the two calls, ~50-200 µs).

The LEFT camera is the master clock: it streams into a small bounded queue
(every frame, carrying its timestamp). The RIGHT camera streams into a small
ring buffer of its most recent frames (also timestamped).

EVENT-DRIVEN, NOT WALL-CLOCK PACED. The encode loop emits one output pair for
each LEFT frame the sensor delivers — it does NOT tick on a 1/fps deadline.
Sampling a free-running sensor on an independent wall clock is a resampler: the
sensor runs at its own rate (~40 Hz here) and a 30 Hz tick keeps 3 of every 4
frames, giving a 1,1,2-spaced "3-then-break" cadence (25% of slots dropped; see
git history for the diagnosis). Driving off frame arrival instead makes the
output spacing BE the sensor's spacing — uniform by construction — while true
per-frame timestamps still go to sync_log.csv for the solve. --fps, if set,
decimates by an INTEGER sensor-frame stride (still uniform); the default keeps
every frame.

For each master frame, the RIGHT frame is picked from the ring by *closest*
timestamp — the best temporal match, not just "the newest right frame". Because
the master reference lags real time by the queue/encode latency, the ring
brackets it on both sides, so nearest-neighbour genuinely straddles the left
timestamp (no explicit lag needed). The skew (t_right - t_left) for every
emitted frame is logged to sync_log.csv.

Both files are fed one frame per emitted pair, so frame N of left and frame N of
right are the same instant — the MP4s stay identical in length and aligned by
frame index. A pair is written only once both cameras have produced a frame (so
the streams start together). If encoding can't keep the sensor's pace the queue
fills and the SDK drops frames at the source — surfacing as a larger real-
timestamp gap in sync_log (which the solve already handles via per-frame dt),
never as a silent systematic decimation.

This nearest-timestamp pairing (vs. plain newest-newest) removes the fixed phase
offset between two free-running sensors, the slow drift between their unlocked
clocks, and one-off USB jitter — bounding skew to ~half a camera frame interval.
For true simultaneous exposure, wire a hardware trigger.

IMU synchronization (optional, --imu-port)
-------------------------------------------
The IMU (hteng_camera.imu.YbImu) has no hardware timestamp counter of its
own, so it cannot share the cameras' frame-timestamp reset the way the two
cameras share it with each other. Instead, every host-side clock reading in
this script — the camera-reset anchor and every IMU sample — uses
``time.perf_counter()``, and the two camera clocks are related to that same
perf_counter timeline through one recorded offset:

  cam_reset_perf_counter_us = perf_counter() reading taken at the same instant
  cam_left.reset_timestamp() / cam_right.reset_timestamp() are called.

``sync_log.csv``'s ``t_left_us``/``t_right_us`` are left untouched — they
keep meaning exactly what they mean without an IMU (camera-firmware counter
time, relative to the reset instant). ``imu_log.csv``'s ``host_time_us`` is
the *absolute* perf_counter reading for each sample (no relative-zero
subtraction). To align a camera frame with the IMU stream, add the recorded
offset to the frame's timestamp:

  t_frame_perfcounter_us = cam_reset_perf_counter_us + t_left_us  # or t_right_us

then compare directly against imu_log.csv's host_time_us column. This keeps
each CSV's semantics independent and makes the conversion explicit and
recomputable rather than baked into one side.

Accuracy: the cam_reset_perf_counter_us read lags the actual
CameraRstTimeStamp() calls by call/return overhead (tens of µs, similar order
to the ~50-200 µs left/right reset gap above). Separately — and NOT
corrected for — there is an inherent few-ms of serial/USB transmission
latency between an IMU sample's true physical sampling instant and the host
receiving and timestamping it; the wire protocol gives no way to recover the
true sensor-side sample instant. This is a soft-sync approximation, not a
hardware-timestamped guarantee. IMU report rate is capped at 100 Hz — this is
a verified FIRMWARE ceiling for this board, not just a library-side clamp.

Output
------
Everything lands in one folder (the positional argument):

    <out_dir>/
      left.mp4                       left camera, frame-matched to right.mp4
      right.mp4                      right camera
      sync_log.csv                   per-frame t_left/t_right/skew (µs)
      imu_log.csv                    raw accel/gyro/mag samples (only if --imu-port given)
      audio.mka                      AirPods PCM with absolute AVFoundation PTS
                                     (unless unavailable or --no-audio)
      recording.json                 manifest: fps, exposure, per-camera ROI,
                                     IMU config + clock-alignment offset
                                     (with the intrinsics offset note)
      calib_<serial_left>.json       copy of each camera's calibration (if found)
      calib_<serial_right>.json
      stereo_<L>_<R>.json            copy of the cam2cam transform (if found)

The calibration files are copied verbatim from the calibration search path
(HTENG_CALIB_DIR, ./calibrations, .) so the recording is self-describing.

Usage
-----
  python record_ego.py take01/                     # first two cameras found
  python record_ego.py --left 046060323003 --right 046060323006 take01/
  python record_ego.py --duration 10 --fps 60 --exposure-ms 12 fast/
  python record_ego.py --encoder x265 --transfer linear master/
  python record_ego.py --imu-port /dev/cu.usbserial-1440 take01/     # + raw IMU log
  python record_ego.py --audio-device "AirPods Pro" take01/          # pin by name
"""

import argparse
import atexit
import csv
import json
import os
import platform
import queue
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from hteng_camera import HTCamera, AutoExposure, calibration, convert, enums, list_cameras
from hteng_camera.imu import ImuSample, YbImu


# ---------------------------------------------------------------------------
# Voice notifications (macOS say, no-op elsewhere)
# ---------------------------------------------------------------------------

def _speak(msg: str) -> None:
    """Fire-and-forget TTS via macOS `say`. No-op on other platforms."""
    if platform.system() == "Darwin":
        subprocess.Popen(["say", msg],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _elapsed_str(seconds: float) -> str:
    """'30 seconds', '1 minute', '1 minute 30', '2 minutes', etc."""
    total = int(round(seconds))
    m, s = divmod(total, 60)
    if m == 0:
        return f"{s} seconds"
    label = "minute" if m == 1 else "minutes"
    return f"{m} {label}" if s == 0 else f"{m} {label} {s}"


def _start_announcer(stop: threading.Event, interval: float = 30.0) -> None:
    """Daemon thread that speaks the elapsed time every ``interval`` seconds."""
    t0 = time.perf_counter()

    def loop():
        next_tick = t0 + interval
        while not stop.wait(timeout=max(0.0, next_tick - time.perf_counter())):
            elapsed = time.perf_counter() - t0
            _speak(_elapsed_str(elapsed))
            next_tick += interval

    threading.Thread(target=loop, daemon=True, name="hteng-announcer").start()


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
# AirPods audio capture (macOS AVFoundation -> timestamp-preserving MKA)
# ---------------------------------------------------------------------------

def _avfoundation_audio_devices() -> list[tuple[int, str]]:
    """Return current AVFoundation audio devices as ``(index, name)`` pairs."""
    if platform.system() != "Darwin":
        return []
    proc = subprocess.run(
        [
            "ffmpeg", "-hide_banner",
            "-f", "avfoundation",
            "-list_devices", "true",
            "-i", "",
        ],
        capture_output=True,
        text=True,
    )
    devices: list[tuple[int, str]] = []
    in_audio = False
    for line in proc.stderr.splitlines():
        if "AVFoundation audio devices:" in line:
            in_audio = True
            continue
        if not in_audio:
            continue
        match = re.search(r"\[(\d+)\]\s+(.+)$", line)
        if match:
            devices.append((int(match.group(1)), match.group(2)))
    return devices


def _resolve_airpods_device(requested: Optional[str]) -> str:
    """Resolve a selector once, then return the stable device name."""
    devices = _avfoundation_audio_devices()
    if requested is None:
        matches = [name for _index, name in devices
                   if "airpods" in name.casefold()]
    elif requested.isdigit():
        matches = [name for index, name in devices
                   if index == int(requested)]
    else:
        exact = [name for _index, name in devices
                 if name.casefold() == requested.casefold()]
        matches = exact or [
            name for _index, name in devices
            if requested.casefold() in name.casefold()
        ]
    if len(matches) != 1:
        available = ", ".join(name for _index, name in devices) or "none"
        target = "connected AirPods" if requested is None else repr(requested)
        raise RuntimeError(
            f"expected exactly one audio device matching {target}; "
            f"available: {available}"
        )
    device = matches[0]
    if "airpods" not in device.casefold():
        raise RuntimeError(
            f"refusing non-AirPods audio device {device!r}; "
            "connect/select AirPods instead"
        )
    return device


def _start_audio_capture(device: str, output: Path,
                         sample_rate: int = 16000) -> Optional[subprocess.Popen]:
    """Start direct-to-disk PCM audio and wait until its MKA header exists."""
    output.unlink(missing_ok=True)
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-nostdin", "-loglevel", "warning",
        "-copyts",
        "-thread_queue_size", "128",
        "-f", "avfoundation",
        "-i", f":{device}",
        "-map", "0:a:0",
        "-vn", "-ac", "1", "-ar", str(sample_rate),
        "-c:a", "pcm_s16le",
        "-avoid_negative_ts", "disabled",
        "-cluster_time_limit", "1000",
        "-flush_packets", "1",
        "-f", "matroska",
        str(output),
    ]
    # Keep terminal Ctrl+C in the parent. The parent sends exactly one SIGINT
    # during ordered teardown so ffmpeg can finalize the Matroska trailer.
    proc = subprocess.Popen(cmd, start_new_session=True)
    deadline = time.perf_counter() + 10.0
    while proc.poll() is None and time.perf_counter() < deadline:
        if output.exists() and output.stat().st_size > 0:
            return proc
        time.sleep(0.02)
    if proc.poll() is None:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        print("[warn] AirPods audio did not become ready within 10 seconds.")
    else:
        print(f"[warn] AirPods audio ffmpeg exited during startup "
              f"(code {proc.returncode}).")
    output.unlink(missing_ok=True)
    return None


def _stop_audio_capture(proc: subprocess.Popen, timeout: float = 10.0) -> int:
    """Gracefully finalize an audio MKA, killing only if ffmpeg is stuck."""
    if proc.poll() is None:
        try:
            proc.send_signal(signal.SIGINT)
        except ProcessLookupError:
            pass
    try:
        return proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        return proc.wait()


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

class _MasterQueue:
    """Bounded FIFO of ``(timestamp_us, bayer)`` for the MASTER (left) camera.

    A FIFO, not a latest-wins mailbox: the encoder emits one output pair per
    master frame, so EVERY frame must reach it in order — dropping to "newest
    only" is exactly the wall-clock resampling this design removes. Depth is a
    few frames of slack for encode-time jitter; at ~10 MB/plane it stays small.

    When the encoder can't keep the sensor's pace the queue fills and ``put``
    drops the newest frame (rather than blocking the SDK grab thread, which
    would wedge the camera). A drop is a genuine skipped capture — it shows up
    as a larger real-timestamp gap in sync_log, which the solve handles, and is
    counted so the run can report it.
    """

    def __init__(self, depth: int) -> None:
        self._q: "queue.Queue[tuple[int, np.ndarray]]" = queue.Queue(maxsize=depth)
        self.dropped = 0

    def put(self, ts: int, frame: np.ndarray) -> None:
        try:
            self._q.put_nowait((ts, frame))
        except queue.Full:
            self.dropped += 1

    def take(self, timeout: float) -> Optional[tuple[int, np.ndarray]]:
        """Next frame in arrival order, or None if none arrives within
        ``timeout`` s (so the encode loop can re-check ``stop``)."""
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None


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


# ---------------------------------------------------------------------------
# IMU raw-sample CSV writer (see "IMU synchronization" in the module
# docstring for how host_time_us here relates to sync_log.csv's timestamps)
# ---------------------------------------------------------------------------

class _ImuCsvWriter:
    """Thread-safe per-sample CSV writer, invoked directly from YbImu's own
    receive thread via its ``on_raw_sample`` callback.

    A lock-guarded direct write (no separate queue/writer thread) is enough
    here: the IMU is hardware-capped at 100 Hz, and a CSV append at that rate
    is trivially fast compared to disk I/O bandwidth. If this were ever
    logging something much higher-rate, a bounded queue drained by a
    dedicated writer thread would be worth it to keep a slow disk from ever
    backing up the IMU's receive thread — not needed at this rate.
    """

    _COLUMNS = [
        "sample", "host_time_us",
        "ax_g", "ay_g", "az_g",
        "gx_rad_s", "gy_rad_s", "gz_rad_s",
        "mx_ut", "my_ut", "mz_ut",
    ]

    def __init__(self, path: Path) -> None:
        self._lock = threading.Lock()
        self._file = open(path, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(self._COLUMNS)
        self._count = 0

    def on_sample(self, sample: ImuSample) -> None:
        """Callback for ``YbImu.start(on_raw_sample=...)``. Writes
        ``host_time_us`` as an ABSOLUTE ``perf_counter()`` microsecond
        reading (no relative-zero subtraction) — see the module docstring's
        "IMU synchronization" section for how this lines up with
        ``sync_log.csv``'s camera timestamps via the recorded
        ``cam_reset_perf_counter_us`` offset."""
        with self._lock:
            self._writer.writerow([
                self._count, int(sample.host_time_s * 1e6),
                sample.ax, sample.ay, sample.az,
                sample.gx, sample.gy, sample.gz,
                sample.mx, sample.my, sample.mz,
            ])
            self._count += 1

    def close(self) -> None:
        with self._lock:
            self._file.close()


def _capture_master(cam: HTCamera, box: _MasterQueue, stop: threading.Event) -> None:
    """Grab left (master) Bayer planes and enqueue every one in arrival order."""
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


def _measure_native_hz(cam: HTCamera, n: int = 30) -> Optional[float]:
    """Grab ``n`` frames back-to-back and return the sensor's delivered rate in
    Hz from the median inter-frame gap of their hardware timestamps, or None if
    too few frames arrive. Used to turn --fps into an integer decimation stride
    over the true (free-running) rate rather than resampling on a wall clock.
    Measured on ``info["time"]`` (camera-firmware µs), the same clock sync_log
    logs, so the stride matches what the encode loop actually sees."""
    ts = []
    for _ in range(n + 1):
        bayer, info = cam.grab_bayer12(timeout_ms=2000)
        if bayer is not None:
            ts.append(info["time"])
    if len(ts) < 3:
        return None
    gaps = np.diff(np.asarray(ts, dtype=np.float64))       # µs
    gaps = gaps[gaps > 0]
    if len(gaps) == 0:
        return None
    return 1e6 / float(np.median(gaps))


# ---------------------------------------------------------------------------
# Consumer: event-driven encoder loop, left as master, right matched by timestamp
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
    stride: int,
    transfer: str,
    sync_csv: Path,
    ae: Optional[dict] = None,
) -> None:
    """Emit one encoded pair per (stride-th) LEFT frame the sensor delivers,
    each matched to the RIGHT frame whose timestamp is nearest it. Event-driven
    — there is NO wall-clock deadline (see the module docstring's "EVENT-DRIVEN"
    note for why: pacing a free-running sensor on an independent clock resamples
    it into a non-uniform 1,1,2 cadence).

    ``left``/``right`` are per-camera dicts with keys ``proc``, ``cv_code``,
    ``wb``; ``left`` also has ``box`` (a :class:`_MasterQueue`) and ``right`` has
    ``ring`` (a :class:`_Ring`).

    Frame matching
    ~~~~~~~~~~~~~~
    The left camera is the master. Each master frame is pulled in arrival order;
    the right frame is the ring entry with the closest timestamp — the best
    temporal pair available. Both pipes get exactly one frame per emitted pair,
    so the files stay equal-length and index-aligned. Nothing is emitted until a
    right match exists (streams start together). Per-frame skew (t_right -
    t_left) goes to ``sync_csv``; every logged row is a real capture (fresh=1).

    Decimation
    ~~~~~~~~~~
    ``stride`` keeps every stride-th master frame (1 = keep all). Because the
    master stream is uniform, every stride-th frame of it is also uniform — this
    is honest integer decimation, not the resampling the old deadline loop did.
    ``stride`` is derived in ``main()`` from the measured native rate and --fps.

    Auto-exposure
    ~~~~~~~~~~~~~
    If ``ae`` is given (``{"ctrl": AutoExposure, "cams": [left_cam, right_cam],
    "exp": ms, "gain": x}``), each emitted left frame is metered and the SAME
    exposure/gain applied to BOTH cameras via :meth:`AutoExposure.apply`.
    Metering the master and copying to the slave keeps the pair photometrically
    matched (essential for stereo), and the anti-flicker exposure makes that match
    hold even though the two sensors free-run out of phase — a flicker-free
    exposure integrates whole 120 Hz cycles, so both collect equal light
    regardless of phase. We meter the raw Bayer plane (cheap, pre-demosaic).
    """
    frame_idx = 0        # emitted pairs (== output frame count)
    master_seen = 0      # master frames pulled since streaming began (for stride)

    with open(sync_csv, "w", newline="") as f:
        log = csv.writer(f)
        log.writerow(["frame", "t_left_us", "t_right_us", "skew_us", "fresh"])

        while not stop.is_set():
            ref = left["box"].take(timeout=0.5)
            if ref is None:
                continue                         # no frame yet; re-check stop
            t_left, bayer_left = ref

            match = right["ring"].nearest(t_left)
            if match is None:
                continue                         # right not warmed up; pre-roll
            t_right, bayer_right = match

            # Integer decimation on the uniform master stream. master_seen only
            # advances once the right side is live, so warmup frames don't shift
            # the stride phase.
            keep = (master_seen % stride == 0)
            master_seen += 1
            if not keep:
                continue

            # Software AE: meter the master's raw Bayer, apply the same
            # exposure/gain to both cameras (flash-free, anti-flicker on).
            if ae is not None:
                ctrl = ae["ctrl"]
                new_exp, new_gain, _ = ctrl.update(
                    bayer_left, ae["exp"], ae["gain"])
                try:
                    ae["exp"], ae["gain"] = ctrl.apply(
                        ae["cams"], new_exp, new_gain, ae["exp"], ae["gain"])
                except Exception:
                    pass                         # a wedged control write must not
                                                 # stall the encode/sync loop

            out_left = _encode(bayer_left, left["cv_code"], transfer, left["wb"])
            out_right = _encode(bayer_right, right["cv_code"], transfer, right["wb"])
            log.writerow([frame_idx, t_left, t_right, t_right - t_left, 1])

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


def _write_manifest(out_dir: Path, args, encoder: str, left: dict, right: dict,
                     imu_info: dict, audio_info: dict, rate_info: dict) -> None:
    """Write recording.json describing how the videos were captured, including
    the per-camera ROI needed to adjust the copied full-frame intrinsics.

    The copied calib_<serial>.json holds FULL-SENSOR intrinsics. A centered ROI
    crop leaves fx, fy and distortion unchanged but moves the principal point:
        cx_roi = cx_full - roi.x_offset
        cy_roi = cy_full - roi.y_offset
    and the image size becomes roi.width x roi.height. This note + the ROI are
    recorded so that step is unambiguous later.

    ``imu_info`` is ``{"enabled": False}`` if no IMU was requested/succeeded,
    or the fully-populated dict built in ``main()`` if it did — it reflects
    ACTUAL outcome, not just whether --imu-port was passed, so a manifest
    never claims IMU logging succeeded when it silently didn't. ``audio_info``
    follows the same actual-outcome convention for AirPods capture.
    """
    manifest = {
        "format": "hteng-camera-stereo-recording/1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        # Nominal container rate; the sensor free-runs and frames are decimated by
        # an integer stride (event-driven, not wall-clock resampled). Frame TIMES
        # are non-uniform-but-honest -- always use sync_log.csv's per-frame
        # t_left_us/t_right_us for the solve, not this rate.
        "fps": rate_info["effective_hz"],
        "fps_requested": args.fps or None,
        "sensor_native_hz": rate_info["native_hz"],
        "decimation_stride": rate_info["stride"],
        "encoder": encoder,
        "quality": args.quality,
        "transfer": args.transfer,
        "demosaic": args.demosaic,
        "roi_height_fraction": args.roi_height,
        "auto_exposure": (
            {
                "enabled": True,
                "target": args.ae_target,
                "exposure_min_ms": args.ae_exp_min,
                "exposure_max_ms": max(args.ae_exp_max, args.ae_exp_min),
                "gain_max_x": args.ae_gain_max,
                "flicker_hz": args.ae_flicker_hz,
                "note": "Exposure/gain were driven live (metered on LEFT, applied "
                        "to both). The per-camera exposure_ms/gain_x below are the "
                        "STARTING values, not constant for the whole recording.",
            }
            if args.auto_exposure else {"enabled": False}
        ),
        "files": {
            "left": "left.mp4",
            "right": "right.mp4",
            "sync_log": "sync_log.csv",
            "imu_log": "imu_log.csv" if imu_info.get("enabled") else None,
            "audio": "audio.mka" if audio_info.get("enabled") else None,
            "stereo_transform":
                f"stereo_{left['serial']}_{right['serial']}.json",
        },
        "left": _side_meta(left["cam"], left["serial"], left["full"],
                           left["exp"], left["gain"]),
        "right": _side_meta(right["cam"], right["serial"], right["full"],
                            right["exp"], right["gain"]),
        "imu": imu_info,
        "audio": audio_info,
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
        self._cleanup_callbacks: list[Callable[[], None]] = []
        self._released = False

    def engage(self) -> None:
        if not self.enabled:
            if platform.system() != "Darwin":
                print("[info] keep-awake: not macOS — relying on the OS / your wrapper.")
            # The object also owns ordered SIGTERM/SIGHUP cleanup callbacks.
            # Install those guards even when sleep prevention was disabled.
            atexit.register(self.release)
            self._install_signal_restore()
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

    def add_cleanup(self, callback: Callable[[], None]) -> None:
        """Run ``callback`` once on normal exit, SIGTERM, or SIGHUP."""
        if self._released:
            callback()
            return
        self._cleanup_callbacks.append(callback)

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
        for callback in reversed(self._cleanup_callbacks):
            try:
                callback()
            except Exception as e:
                print(f"[warn] exit cleanup failed: {e}")
        self._cleanup_callbacks.clear()
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
    ap.add_argument(
        "--fps", type=int, default=0,
        help="Target output rate. The sensor free-runs at its own (fixed) rate; "
             "this keeps the nearest INTEGER-stride fraction of frames (uniform), "
             "never a wall-clock resample. 0 (default) keeps every frame at the "
             "native rate. E.g. asking 30 on a ~40 Hz sensor -> stride 1 -> ~40 "
             "fps (closest integer stride); ask 20 for a clean half.")
    ap.add_argument(
        "--duration", type=float, default=None,
        help="Recording duration in seconds (omit to record until Ctrl+C)",
    )
    ap.add_argument(
        "--exposure-ms", type=float, default=5.0,
        help="Exposure time in ms. Must be < 1000/fps or both cameras run below "
             "the target rate and frames are duplicated. A short exposure keeps "
             "the sensors running fast, which tightens the stereo pairing. With "
             "auto-exposure (the default) this is the STARTING exposure; with "
             "--no-auto-exposure it is held fixed.",
    )
    ap.add_argument("--gain", type=float, default=3.0, help="Analog gain multiplier")
    ap.add_argument(
        "--no-auto-exposure", dest="auto_exposure", action="store_false",
        help="Disable software auto-exposure and hold exposure/gain fixed at "
             "--exposure-ms / --gain for the whole recording. By DEFAULT, AE is "
             "on: it meters the LEFT (master) frame and applies the SAME "
             "exposure+gain to BOTH cameras so the pair stays photometrically "
             "matched, gain-first (keeps exposure low for a stable frame rate) "
             "with anti-flicker. --exposure-ms / --gain then set the starting "
             "point and the AE knobs (--ae-*) tune it.",
    )
    ap.set_defaults(auto_exposure=True)
    ap.add_argument(
        "--ae-target", type=float, default=0.35,
        help="Software AE target brightness, 0..1 (centre-weighted, fraction of "
             "full scale). Only used with --auto-exposure.",
    )
    ap.add_argument(
        "--ae-exp-min", type=float, default=5.0,
        help="Software AE exposure floor (ms). The loop prefers this; with "
             "anti-flicker it snaps to flicker-free multiples within "
             "[floor, --ae-exp-max]. Only used with --auto-exposure.",
    )
    ap.add_argument(
        "--ae-exp-max", type=float, default=17.0,
        help="Software AE exposure cap (ms). The loop only lengthens toward this "
             "once gain saturates. Default 17 ms spans the 8.33/16.67 ms "
             "flicker-free steps at 60 Hz. Keep < 1000/fps for a stable frame "
             "rate. Only used with --auto-exposure.",
    )
    ap.add_argument(
        "--ae-gain-max", type=float, default=4.0,
        help="Software AE max analog gain (x). Lower = less noise. Only used "
             "with --auto-exposure.",
    )
    ap.add_argument(
        "--ae-flicker-hz", type=float, default=60.0,
        help="Mains frequency for anti-flicker (60 US / 50 EU), 0 to disable. "
             "Pins exposure to whole light-flicker cycles so indoor lights don't "
             "pulse and the unsynced stereo pair stays matched. Only used with "
             "--auto-exposure.",
    )
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
        help="Depth (frames) of BOTH the right-camera ring the master timestamp "
             "is paired against AND the master FIFO of frames awaiting encode. "
             "More gives the nearest-timestamp search more to choose from and "
             "more encode-jitter slack before frames drop, but costs ~10 MB each. "
             "8 is plenty at the sensor's native rate.",
    )
    ap.add_argument(
        "--no-keep-awake", action="store_true",
        help="Don't keep the Mac awake. By default the recording blocks idle "
             "sleep (caffeinate) AND lid-close sleep (pmset disablesleep, needs "
             "sudo) so you can shut the laptop and let it run. Both are restored "
             "on exit. No-op off macOS.",
    )
    ap.add_argument(
        "--imu-port", default=None,
        help="IMU serial port. Autodetected by default; pass this to pin "
             "an exact port if autodetection picks the wrong device.",
    )
    ap.add_argument(
        "--no-imu", action="store_true",
        help="Disable IMU logging even if one is autodetected.",
    )
    ap.add_argument(
        "--imu-rate", type=int, default=100,
        help="IMU report rate in Hz, [10, 100]. 100 is also a hard "
             "firmware ceiling for this board -- more has no effect.",
    )
    ap.add_argument(
        "--imu-algo", type=int, choices=[6, 9], default=9,
        help="Onboard fusion mode 6 or 9 (unused; this script only logs "
             "raw data). Matters only because 6-axis makes the firmware "
             "stop sampling the magnetometer -- mx/my/mz log as 0.0.",
    )
    ap.add_argument(
        "--no-audio", action="store_true",
        help="Disable AirPods microphone recording. By default, exactly one "
             "connected AVFoundation audio device whose name contains "
             "'AirPods' is autodetected and recorded to audio.mka.",
    )
    ap.add_argument(
        "--audio-device", default=None,
        help="AirPods AVFoundation device name, unique name substring, or "
             "current index. The selector is resolved to a name before capture "
             "and non-AirPods devices are rejected.",
    )
    args = ap.parse_args()

    if args.sync_buffer < 1:
        sys.exit("[error] --sync-buffer must be >= 1.")
    if not 0.0 < args.roi_height <= 1.0:
        sys.exit("[error] --roi-height must be in (0, 1].")
    if not 10 <= args.imu_rate <= 100:
        sys.exit("[error] --imu-rate must be in [10, 100] (firmware hard-caps at 100 anyway).")

    encoder = _resolve_encoder(args.encoder)
    _preflight_ffmpeg(encoder)

    audio_device: Optional[str] = None
    if not args.no_audio:
        if platform.system() != "Darwin":
            print("[info] AirPods audio is macOS-only; recording without audio.")
        else:
            try:
                audio_device = _resolve_airpods_device(args.audio_device)
                print(f"[info] selected AirPods input: {audio_device}")
            except RuntimeError as e:
                sys.exit(f"[error] AirPods audio unavailable: {e}\n"
                         "        Connect exactly one AirPods input, select it "
                         "with --audio-device, or pass --no-audio.")

    # Long exposure caps the sensor's native rate (it can't deliver frames faster
    # than it integrates). The event-driven loop never duplicates — it just emits
    # whatever the sensor delivers — so this is informational, and the actual rate
    # is measured + reported once the cameras are open.
    if args.fps and args.exposure_ms >= 1000.0 / args.fps:
        print(f"[warn] exposure {args.exposure_ms:.1f} ms exceeds the "
              f"{1000.0 / args.fps:.1f} ms interval of the requested {args.fps} "
              "fps — the sensor will deliver below that rate.")

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
    out_audio = out_dir / "audio.mka"

    # Keep the laptop awake (incl. lid closed) for the whole run. Engage here —
    # past all the arg/camera validation, so a trivial error never triggers the
    # sudo prompt, but before we open cameras or start threads so the one-time
    # prompt is clean and single-threaded. release() is idempotent and also
    # atexit/SIGTERM-registered, so every exit path still undoes it.
    awake = _KeepAwake(enabled=not args.no_keep_awake)
    awake.engage()

    # ── Open + configure the IMU (optional, on by default) ───────────────────
    # Without --no-imu, this always runs: an explicit --imu-port opens that
    # exact port, otherwise the IMU is autodetected by probing candidate
    # USB-serial ports with a real version-query handshake (see
    # hteng_camera.imu.YbImu.autodetect) -- no flag needed if one is plugged
    # in. Either way, any failure here is a warning, not fatal -- the stereo
    # recording must not break because of optional add-on hardware.
    # Calibration is deliberately NOT done here -- it's a one-time-per-mount
    # (accel/gyro) or environment-dependent (magnetometer) step, not
    # something to run on every recording; do it separately before capturing.
    imu: Optional[YbImu] = None
    imu_csv: Optional[_ImuCsvWriter] = None
    imu_version: Optional[str] = None
    if not args.no_imu:
        try:
            if args.imu_port:
                imu = YbImu(args.imu_port)
                # YbImu.start() must run before ANY set_*/get_version call --
                # see YbImu's docstring: sending a command first can silently
                # wedge the serial link (found by direct hardware testing
                # while building this integration).
                imu.start()
            else:
                print("[info] IMU: autodetecting (pass --imu-port to pin an "
                      "exact port, --no-imu to disable)...")
                imu = YbImu.autodetect()
                if imu is None:
                    print("[info] IMU: none detected -- recording without IMU logging.")
            if imu is not None:
                imu_csv = _ImuCsvWriter(out_dir / "imu_log.csv")
                imu.on_raw_sample = imu_csv.on_sample
                imu.set_report_rate(args.imu_rate)
                imu.set_algo_type(args.imu_algo)
                imu_version = imu.get_version()
                print(f"[info] IMU: streaming from {imu.port} at {args.imu_rate} Hz, "
                      f"algo={args.imu_algo}, firmware {imu_version or 'unknown'}.")
        except Exception as e:
            print(f"[warn] IMU setup failed: {e}. Continuing WITHOUT IMU logging.")
            if imu_csv is not None:
                imu_csv.close()
            if imu is not None:
                try:
                    imu.close()
                except Exception:
                    pass
            imu = None
            imu_csv = None
            imu_version = None

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

    # Measure the sensor's true delivered rate (before the reset, so these warmup
    # grabs don't consume post-reset timestamps). --fps then decimates by an
    # INTEGER stride over this rate instead of resampling on a wall clock, which
    # is what produced the non-uniform 1,1,2 frame cadence (see module docstring).
    native_hz = _measure_native_hz(cam_left) or float(args.fps or 30)
    if args.fps and args.fps > 0:
        stride = max(1, int(round(native_hz / args.fps)))
    else:
        stride = 1                               # keep every frame (native rate)
    effective_hz = native_hz / stride
    print(f"[info] sensor native ~{native_hz:.2f} Hz; stride {stride} "
          f"-> output ~{effective_hz:.2f} fps"
          + (f" (asked {args.fps})" if args.fps else " (all frames)"))

    # Start Bluetooth/CoreAudio before resetting the camera clocks so device
    # negotiation cannot perturb active capture. MKA packet PTS remain on the
    # absolute AVFoundation host clock, so this short audio pre-roll is harmless.
    audio_proc: Optional[subprocess.Popen] = None
    if audio_device is not None:
        audio_proc = _start_audio_capture(audio_device, out_audio)
        if audio_proc is None:
            for cam in (cam_left, cam_right):
                cam.close()
            if imu is not None:
                imu.close()
            if imu_csv is not None:
                imu_csv.close()
            awake.release()
            sys.exit("[error] AirPods input was selected but audio capture "
                     "failed to start. Reconnect it or pass --no-audio.")
        else:
            print(f"[info] AirPods audio recording -> {out_audio}")
            # Later setup still has several fallible operations. Ensure an
            # exception or early sys.exit finalizes the MKA instead of leaving
            # the audio ffmpeg running after the parent exits.
            atexit.register(_stop_audio_capture, audio_proc)
            awake.add_cleanup(
                lambda proc=audio_proc: _stop_audio_capture(proc, timeout=5.0))

    # Shared hardware time base for the pair (soft sync; see module docstring).
    cam_left.reset_timestamp()
    cam_right.reset_timestamp()
    # Anchor for relating the IMU's absolute perf_counter timestamps to the
    # cameras' reset-relative ones -- see "IMU synchronization" above.
    cam_reset_perf_counter_us = int(time.perf_counter() * 1e6)

    # Encoder dims come from the real frame each camera delivered during open
    # (no extra grab to time out on).
    dims = {"left": dims_left, "right": dims_right}
    if dims["left"] != dims["right"]:
        print(f"[warn] left {dims['left'][::-1]} and right {dims['right'][::-1]} "
              "resolutions differ — each video uses its own camera's size.")

    wb_left = None if (args.transfer == "linear" or args.no_wb) else cam_left.wb_gains
    wb_right = None if (args.transfer == "linear" or args.no_wb) else cam_right.wb_gains
    print(
        f"[info] {dims['left'][1]}x{dims['left'][0]} @ ~{effective_hz:.2f} fps, "
        f"encoder {encoder} ({_ENCODERS[encoder]['codec']}), quality {args.quality}, "
        f"transfer {args.transfer}"
    )
    print(
        f"[info] measured  left: {exp_left:.1f} ms / {gain_left:.2f}x   "
        f"right: {exp_right:.1f} ms / {gain_right:.2f}x   "
        f"(asked {args.exposure_ms:.1f} ms / {args.gain:.2f}x)"
    )

    # ── Software auto-exposure (optional) ────────────────────────────────────
    # One controller, metering the LEFT master and applying the same exposure/gain
    # to BOTH cameras so the stereo pair stays photometrically matched. The loop
    # floors at --ae-exp-min and only lengthens toward --ae-exp-max once gain
    # saturates. The starting operating point is what the cameras latched on open.
    ae = None
    if args.auto_exposure:
        gain_lo, gain_hi, _ = cam_left.gain_range()
        exp_cap = max(args.ae_exp_max, args.ae_exp_min)
        ctrl = AutoExposure(
            gain_lo, min(gain_hi, args.ae_gain_max),
            exp_min=args.ae_exp_min, exp_max=exp_cap,
            target=args.ae_target, flicker_hz=args.ae_flicker_hz,
        )
        steps = ctrl._exposure_steps()
        if args.ae_flicker_hz and len(steps) == 1 and exp_cap < 1000.0 / (2 * args.ae_flicker_hz):
            print(f"[warn] AE: no flicker-free exposure fits [{args.ae_exp_min:.1f}, "
                  f"{exp_cap:.1f}] ms at {args.ae_flicker_hz:g} Hz — raise "
                  f"--exposure-ms to >= {1000.0/(2*args.ae_flicker_hz):.2f} ms to "
                  f"engage anti-flicker. Running with exposure pinned at the floor.")
        ae = {"ctrl": ctrl, "cams": [cam_left, cam_right],
              "exp": exp_left, "gain": gain_left}
        # Start the slave from the same operating point as the master.
        ctrl.apply(cam_right, exp_left, gain_left, exp_right, gain_right)
        print(f"[info] AE on: target {args.ae_target:.2f}, gain ≤ "
              f"{min(gain_hi, args.ae_gain_max):.1f}x, exposure steps {steps} ms "
              f"({args.ae_flicker_hz:g} Hz anti-flicker); metering LEFT → both.")

    # ── Copy calibration JSONs + write the recording manifest ────────────────
    _copy_calibration(out_dir, serial_left, serial_right)
    if imu is not None:
        imu_info = {
            "enabled": True,
            "port": imu.port,
            "report_rate_hz": args.imu_rate,
            "report_rate_hz_effective": 100,
            "algo_type": args.imu_algo,
            "firmware_version": imu_version,
            "cam_reset_perf_counter_us": cam_reset_perf_counter_us,
            "clock_alignment": (
                "imu_log.csv's host_time_us is an ABSOLUTE time.perf_counter() "
                "reading in microseconds (no relative-zero subtraction). To "
                "align a frame from sync_log.csv, compute "
                "cam_reset_perf_counter_us + t_left_us (or t_right_us) and "
                "compare directly against host_time_us. Accuracy: tens to "
                "hundreds of microseconds from the reset_timestamp()/"
                "perf_counter() call gap, plus an UNCORRECTED few-ms of "
                "serial/USB transmission latency per IMU sample -- a "
                "soft-sync approximation, not a hardware-timestamped "
                "guarantee."
            ),
            "logged_fields": (
                "Raw accelerometer (g), gyroscope (rad/s), magnetometer (uT) "
                "per sample -- NOT the onboard fused quaternion/Euler angles. "
                "algo_type above affects more than the (unlogged) onboard "
                "fusion output: in 6-axis mode the firmware stops sampling "
                "the magnetometer, so mx_ut/my_ut/mz_ut in imu_log.csv are a "
                "constant 0.0 regardless of motion. Raw accel/gyro are "
                "unaffected by algo_type either way."
            ),
        }
    else:
        imu_info = {"enabled": False}
    audio_info = (
        {
            "enabled": True,
            "file": "audio.mka",
            "device": audio_device,
            "backend": "avfoundation",
            "codec": "pcm_s16le",
            "sample_rate_hz": 16000,
            "channels": 1,
            "clock_alignment": (
                "Every Matroska audio packet PTS is an ABSOLUTE AVFoundation "
                "host timestamp on the same macOS monotonic clock as Python "
                "time.perf_counter(). PTS are retained with -copyts and "
                "-avoid_negative_ts disabled at 1 ms Matroska precision. "
                "Do not remux the original if absolute timestamps are needed; "
                "many tools rebase media to zero. Transcript-relative time t "
                "maps to host time as audio stream start_time + t."
            ),
        }
        if audio_proc is not None else {"enabled": False}
    )
    _write_manifest(
        out_dir, args, encoder,
        {"cam": cam_left, "serial": serial_left, "full": full_left,
         "exp": exp_left, "gain": gain_left},
        {"cam": cam_right, "serial": serial_right, "full": full_right,
         "exp": exp_right, "gain": gain_right},
        imu_info,
        audio_info,
        {"native_hz": round(native_hz, 3), "stride": stride,
         "effective_hz": round(effective_hz, 3)},
    )

    # ── Start one ffmpeg per camera ──────────────────────────────────────────
    procs = []
    for out_path, (tag, cam) in zip((out_left, out_right),
                                    (("left", cam_left), ("right", cam_right))):
        h, w = dims[tag]
        # Container playback rate = the true emitted rate (native/stride), not the
        # requested --fps, so real-time playback matches wall-clock duration. The
        # solve uses per-frame sync_log timestamps regardless; this only sets the
        # nominal timebase stamped in the MP4.
        cmd = _ffmpeg_cmd(encoder, w, h, max(1, round(effective_hz)),
                          args.quality, out_path, args.transfer)
        # bufsize=0: each frame is one big write; buffering just memcpys ~30 MB.
        procs.append(subprocess.Popen(cmd, stdin=subprocess.PIPE, bufsize=0))
    proc_left, proc_right = procs

    # ── Producers (left FIFO / right ring) + one shared encode thread ────────
    stop = threading.Event()
    box_left = _MasterQueue(args.sync_buffer)
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
        args=(left, right, stop, stride, args.transfer, sync_csv, ae),
        name="hteng-encode",
    )
    for t in cap_threads:
        t.start()
    enc_thread.start()
    _speak("recording")
    _start_announcer(stop)

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
    audio_failed_reported = False
    try:
        print("[info] Recording… press Ctrl+C to stop.")
        t_end = None if args.duration is None else time.perf_counter() + args.duration
        while enc_thread.is_alive():
            if interrupted.is_set():
                stopped_intentionally = True
                print()                                # newline after ^C
                _speak("stopped")
                break
            if t_end is not None and time.perf_counter() >= t_end:
                stopped_intentionally = True
                _speak("done")
                break
            if proc_left.poll() is not None or proc_right.poll() is not None:
                print("[warn] an ffmpeg exited early — stopping.")
                _speak("error")
                break
            if (audio_proc is not None and audio_proc.poll() is not None
                    and not audio_failed_reported):
                audio_failed_reported = True
                print(f"[warn] audio ffmpeg exited early with code "
                      f"{audio_proc.returncode}; camera recording continues.")
            time.sleep(0.1)
    finally:
        signal.signal(signal.SIGINT, prev_sigint)      # restore for teardown

    # ── Tear down ─────────────────────────────────────────────────────────────
    print("[info] Stopping…")
    stop.set()
    # Audio runs in its own process session, so terminal Ctrl+C reaches only the
    # parent. Send exactly one deliberate SIGINT for a valid Matroska trailer.
    if audio_proc is not None and audio_proc.poll() is None:
        try:
            audio_proc.send_signal(signal.SIGINT)
        except ProcessLookupError:
            pass
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
    audio_ret = None
    if audio_proc is not None:
        if audio_proc.poll() is None:
            print("[info] waiting for audio ffmpeg to finalize...")
        audio_ret = _stop_audio_capture(audio_proc)
    cam_left.close()
    cam_right.close()
    if imu is not None:
        imu.close()                     # stop the receive thread before closing the sink
    if imu_csv is not None:
        imu_csv.close()
    awake.release()                     # restore sleep settings; stop caffeinate

    # ffmpeg exits 255 / -SIGINT on a graceful SIGINT — expected, file is complete.
    def _clean(ret: int) -> bool:
        return ret == 0 or (stopped_intentionally and ret in (255, -signal.SIGINT))

    if all(_clean(r) for r in rets):
        print(f"[info] Saved → {out_left}")
        print(f"[info] Saved → {out_right}")
        print(f"[info] Sync log → {sync_csv}")
        if imu_info.get("enabled"):
            print(f"[info] IMU log → {out_dir / 'imu_log.csv'}")
        if audio_info.get("enabled"):
            if (not audio_failed_reported
                    and audio_ret in (0, 255, -signal.SIGINT)):
                print(f"[info] AirPods audio → {out_audio}")
            else:
                print(f"[warn] audio ffmpeg exited with code {audio_ret}; "
                      f"{out_audio} may be incomplete.")
        _report_sync(sync_csv)
        if box_left.dropped:
            print(f"[warn] dropped {box_left.dropped} master frames at the encoder "
                  "(couldn't keep the sensor's pace) — they show as larger dt gaps "
                  "in sync_log, not silent decimation. Lower --fps / resolution or "
                  "raise --sync-buffer if this is high.")
        _speak("saved")
    else:
        print(f"[warn] ffmpeg exited with codes {rets} — output may be incomplete.")
        _speak("save failed")


if __name__ == "__main__":
    main()
