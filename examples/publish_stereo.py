"""Publish a calibrated HTENG stereo pair as one self-describing ZMQ packet.

The publisher owns camera discovery, timestamp reset/mapping, startup phase
alignment, frame pairing, software auto-exposure, color conversion, and rig
metadata. Consumers only need to decode ``eyeball-stereo-pair/1`` from
``ipc:///tmp/stereo_pair``.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
from pathlib import Path
import queue
import signal
import struct
import threading
import time
import uuid
from ctypes import POINTER, byref, c_ubyte

import cv2
import numpy as np
from hteng_camera import (
    _sdk,
    AutoExposure,
    HTCamera,
    calibration,
    convert,
    enums,
    list_cameras,
)
from hteng_camera._sdk import tSdkFrameHead
from eye.transport import transport_classes


PAIR_TOPIC = "ipc:///tmp/stereo_pair"
STREAM_FORMAT = "eyeball-stereo-pair/1"
RIG_FORMAT = "eyeball-stereo-rig/2"
_MAGIC = b"ESTPAIR1"
# Only bounds how long a queue read blocks so the loop can notice SIGINT — a
# pair publishes the moment both sides have delivered, never on a timer.
_STOP_POLL_S = 0.5
_PREFIX = struct.Struct("<8sIQQ")
_MJ_TO_CV = np.diag([1.0, -1.0, -1.0])
_HEAD_FROM_CANONICAL_MJ = np.array(
    [[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    dtype=np.float64,
)
_BAYER8_TILE_BY_ORDINAL = {
    0x08: "GR",
    0x09: "RG",
    0x0A: "GB",
    0x0B: "BG",
}


def _encode_pair(metadata: dict, left: np.ndarray, right: np.ndarray) -> bytes:
    header = dict(metadata)
    header["format"] = STREAM_FORMAT
    encoded = json.dumps(
        header, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    left_view = memoryview(left).cast("B")
    right_view = memoryview(right).cast("B")
    return b"".join(
        (
            _PREFIX.pack(_MAGIC, len(encoded), len(left_view), len(right_view)),
            encoded,
            left_view,
            right_view,
        )
    )


def _select_bayer8(camera: HTCamera) -> None:
    media = next(
        (item for item in camera.media_types() if item["bits"] == 8),
        None,
    )
    if media is None:
        raise RuntimeError(f"{camera.serial}: no 8-bit Bayer media type")
    tile = _BAYER8_TILE_BY_ORDINAL.get(media["code"] & 0xFF)
    if tile is None:
        raise RuntimeError(
            f"{camera.serial}: unsupported 8-bit media {media['code']:#010x}"
        )
    cv_pattern = {"GR": "GB", "RG": "BG", "GB": "GR", "BG": "RG"}[tile]
    camera.pause()
    camera._ctrl(
        lambda: _sdk.CameraSetMediaType(camera.h, media["index"]),
        "CameraSetMediaType",
    )
    camera._media_index = media["index"]
    camera._media_code = media["code"]
    camera._cv_code = getattr(cv2, f"COLOR_Bayer{cv_pattern}2RGB_EA")


def _grab_bayer(
    camera: HTCamera,
    capture_bits: int,
    timeout_ms: int,
    priority: int = enums.GET_NEWEST,
):
    if capture_bits == 12:
        return camera.grab_bayer12(
            timeout_ms=timeout_ms, priority=priority
        )

    camera.play()
    head = tSdkFrameHead()
    raw_ptr = POINTER(c_ubyte)()
    status = _sdk.CameraGetImageBufferPriority(
        camera.h,
        byref(head),
        byref(raw_ptr),
        timeout_ms,
        priority,
    )
    if status != 0:
        return None, {}
    try:
        size = int(head.iWidth) * int(head.iHeight)
        if int(head.uBytes) != size:
            raise RuntimeError(
                f"expected 8-bit Bayer (1 B/px), got "
                f"{int(head.uBytes) / size:.2f} B/px"
            )
        bayer = np.frombuffer(
            (c_ubyte * size).from_address(
                ctypes.addressof(raw_ptr.contents)
            ),
            dtype=np.uint8,
            count=size,
        ).reshape(int(head.iHeight), int(head.iWidth)).copy()
        info = {"time": int(head.uiTimeStamp) * 100}
    finally:
        _sdk.CameraReleaseImageBuffer(camera.h, raw_ptr)
    return bayer, info


def _bt709_lut8(wb_gains) -> np.ndarray:
    gains = (1.0, 1.0, 1.0) if wb_gains is None else wb_gains
    linear = (
        np.arange(256, dtype=np.float32)[:, None]
        * np.asarray(gains, dtype=np.float32)[None, :]
        / 255.0
    )
    encoded = np.where(
        linear < 0.018,
        4.5 * linear,
        1.099 * np.power(linear, 0.45) - 0.099,
    )
    return np.clip(encoded * 255.0 + 0.5, 0, 255).astype(
        np.uint8
    ).reshape(256, 1, 3)


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def _se3_half(transform: np.ndarray) -> np.ndarray:
    """Principal SE(3) square root via log/exp."""
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    omega, _ = cv2.Rodrigues(rotation)
    omega = omega.reshape(3)
    theta = float(np.linalg.norm(omega))
    omega_hat = _skew(omega)
    if theta < 1e-8:
        V = np.eye(3) + 0.5 * omega_hat + omega_hat @ omega_hat / 6.0
    else:
        V = (
            np.eye(3)
            + (1.0 - math.cos(theta)) / theta**2 * omega_hat
            + (theta - math.sin(theta)) / theta**3
            * (omega_hat @ omega_hat)
        )
    tangent_translation = np.linalg.solve(V, translation)
    half_omega = omega / 2.0
    half_theta = theta / 2.0
    half_hat = _skew(half_omega)
    if half_theta < 1e-8:
        V_half = np.eye(3) + 0.5 * half_hat + half_hat @ half_hat / 6.0
    else:
        V_half = (
            np.eye(3)
            + (1.0 - math.cos(half_theta)) / half_theta**2 * half_hat
            + (half_theta - math.sin(half_theta)) / half_theta**3
            * (half_hat @ half_hat)
        )
    result = np.eye(4)
    result[:3, :3], _ = cv2.Rodrigues(half_omega)
    result[:3, 3] = V_half @ (tangent_translation / 2.0)
    return result


def _find_stereo_calibration(serials: set[str]):
    matches = {}
    for directory in calibration.search_dirs():
        for path in Path(directory).glob("stereo_*.json"):
            try:
                stereo = calibration.StereoCalibration.load(path)
            except Exception:
                continue
            pair = (stereo.serial_left, stereo.serial_right)
            if set(pair).issubset(serials):
                matches.setdefault(pair, (stereo, path))
    if len(matches) != 1:
        found = [f"{left}+{right}" for left, right in matches]
        raise RuntimeError(
            "Expected exactly one connected calibrated stereo pair; "
            f"found {len(matches)} ({', '.join(found) or 'none'})"
        )
    return next(iter(matches.values()))


def _load_head_pose(
    stereo,
    explicit_path: Path | None,
    allow_identity: bool,
) -> tuple[np.ndarray, np.ndarray, bool, str]:
    filename = (
        f"head_pose_{stereo.serial_left}_{stereo.serial_right}.json"
    )
    candidates = [explicit_path] if explicit_path is not None else [
        Path(directory) / filename for directory in calibration.search_dirs()
    ]
    for path in candidates:
        if path is None or not path.exists():
            continue
        with open(path) as file:
            value = json.load(file)
        fmt = value.get("format")
        if fmt not in (
            "hteng-camera-head-pose/1",
            "hteng-camera-head-pose/2",
        ):
            raise ValueError(f"{path}: unsupported head-pose format")
        if (
            value.get("serial_left") != stereo.serial_left
            or value.get("serial_right") != stereo.serial_right
        ):
            raise ValueError(f"{path}: head-pose serials do not match stereo rig")
        base_from_head = np.asarray(
            value["base_from_head"], dtype=np.float64
        )
        if base_from_head.shape != (4, 4):
            raise ValueError(f"{path}: base_from_head must be 4x4")
        if fmt == "hteng-camera-head-pose/2":
            head_from_stereo_midpoint = np.asarray(
                value["head_from_stereo_midpoint"], dtype=np.float64
            )
            if head_from_stereo_midpoint.shape != (4, 4):
                raise ValueError(
                    f"{path}: head_from_stereo_midpoint must be 4x4"
                )
        else:
            # v1 defined the head frame as the physical stereo midpoint.
            head_from_stereo_midpoint = np.eye(4)
        return (
            base_from_head,
            head_from_stereo_midpoint,
            True,
            str(path),
        )
    if not allow_identity:
        raise FileNotFoundError(
            f"No {filename} found. Calibrate the mount or pass "
            "--allow-identity-head-pose for non-robot bench testing."
        )
    return np.eye(4), np.eye(4), False, "identity override"


def _build_rig(
    stereo,
    left: HTCamera,
    right: HTCamera,
    base_from_head,
    head_from_stereo_midpoint,
):
    left_intr = left.calibration.intrinsics if left.calibration else None
    right_intr = right.calibration.intrinsics if right.calibration else None
    if left_intr is None or right_intr is None:
        raise RuntimeError("Both connected cameras require intrinsics calibration")

    # Convert OpenCV relative coordinates into the renderer's MuJoCo-style
    # camera coordinates, then split the full transform symmetrically.
    right_from_left_cv = np.eye(4)
    right_from_left_cv[:3, :3] = stereo.R
    right_from_left_cv[:3, 3] = stereo.t
    basis = np.eye(4)
    basis[:3, :3] = _MJ_TO_CV
    right_from_left = basis @ right_from_left_cv @ basis
    half = _se3_half(right_from_left)
    canonical = np.eye(4)
    canonical[:3, :3] = _HEAD_FROM_CANONICAL_MJ
    head_from_left = head_from_stereo_midpoint @ canonical @ half
    head_from_right = (
        head_from_stereo_midpoint @ canonical @ np.linalg.inv(half)
    )

    def projection(cam: HTCamera, intr) -> dict:
        resolution = cam.current_resolution()
        width = int(resolution.iWidthFOV)
        height = int(resolution.iHeightFOV)
        roi_x = int(resolution.iHOffsetFOV)
        roi_y = int(resolution.iVOffsetFOV)
        full_width, full_height = map(int, intr.image_size)
        if (
            roi_x < 0
            or roi_y < 0
            or roi_x + width > full_width
            or roi_y + height > full_height
        ):
            raise RuntimeError(
                f"{cam.serial}: ROI {width}x{height}+{roi_x}+{roi_y} falls "
                f"outside calibrated sensor {full_width}x{full_height}"
            )
        K = np.asarray(intr.K, dtype=np.float64).copy()
        K[0, 2] -= roi_x
        K[1, 2] -= roi_y
        return {
            "serial": cam.serial,
            "width": width,
            "height": height,
            "model": intr.model,
            "K": K.tolist(),
            "dist": intr.dist.tolist(),
            "fov_deg": 180.0,
            "encoding": "rgb8",
            "transfer": "bt709",
        }

    rig = {
        "format": RIG_FORMAT,
        "calibration_id": (
            f"hteng:{stereo.serial_left}:{stereo.serial_right}:"
            f"{stereo.saved_at or 'unspecified'}"
        ),
        "left": projection(left, left_intr),
        "right": projection(right, right_intr),
        "right_from_left": right_from_left.tolist(),
        "head_from_left_camera": head_from_left.tolist(),
        "head_from_right_camera": head_from_right.tolist(),
        "base_from_head": base_from_head.tolist(),
        "base_from_head_explicit": True,
    }
    canonical_json = json.dumps(
        rig, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    import hashlib
    return rig, hashlib.sha256(canonical_json).hexdigest()


class _DropOldestQueue:
    def __init__(self, depth: int):
        self.queue = queue.Queue(maxsize=depth)

    def put(self, value) -> None:
        try:
            self.queue.put_nowait(value)
        except queue.Full:
            try:
                self.queue.get_nowait()
            except queue.Empty:
                pass
            self.queue.put_nowait(value)

    def get(self, timeout: float):
        return self.queue.get(timeout=timeout)


def _reset_anchor(camera: HTCamera) -> float:
    before = time.perf_counter()
    camera.reset_timestamp()
    after = time.perf_counter()
    return (before + after) / 2.0


def _stamp_stream(
    camera: HTCamera,
    anchor: float,
    count: int,
    output: list,
    errors: list,
    capture_bits: int,
) -> None:
    try:
        while len(output) < count:
            bayer, info = _grab_bayer(camera, capture_bits, timeout_ms=500)
            if bayer is None:
                continue
            stamp = anchor + int(info["time"]) * 1e-6
            # Discard SDK-buffered frames from before reset_timestamp().
            if stamp > time.perf_counter() + 1.0:
                continue
            output.append(stamp)
    except Exception as exc:
        errors.append(exc)


def _measure_skew(
    left: HTCamera,
    right: HTCamera,
    left_anchor: float,
    right_anchor: float,
    capture_bits: int,
    frames: int = 8,
) -> tuple[float, float]:
    """Return ``(skew_s, period_s)`` for the pair on the shared time base.

    ``skew_s`` is how far the right sensor's exposures sit from the left's,
    wrapped into ±period/2. Repeatable to a few tens of microseconds — the
    hardware counters are quantised at 100 µs and do not drift apart.
    """
    left_stamps: list[float] = []
    right_stamps: list[float] = []
    errors: list = []
    threads = [
        threading.Thread(
            target=_stamp_stream,
            args=(
                camera,
                anchor,
                frames,
                stamps,
                errors,
                capture_bits,
            ),
            daemon=True,
        )
        for camera, anchor, stamps in (
            (left, left_anchor, left_stamps),
            (right, right_anchor, right_stamps),
        )
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5.0)
    if errors:
        raise RuntimeError(f"skew measurement failed: {errors[0]}")
    if len(left_stamps) < 3 or len(right_stamps) < 3:
        raise RuntimeError("skew measurement did not see enough frames")
    period = float(np.median(np.diff(left_stamps)))
    right_arr = np.asarray(right_stamps)
    deltas = []
    for stamp in left_stamps:
        offsets = right_arr - stamp
        deltas.append(offsets[np.argmin(np.abs(offsets))])
    # Nearest-neighbour deltas alias across ±period/2, so average them on the
    # circle instead of linearly.
    phase = np.angle(
        np.mean(np.exp(2j * np.pi * np.asarray(deltas) / period))
    )
    return float(phase * period / (2.0 * math.pi)), period


def _align_phase(
    left: HTCamera,
    right: HTCamera,
    exposure_ms: float,
    gain: float,
    tolerance_s: float,
    max_attempts: int,
    capture_bits: int,
    vertical_trim: float,
) -> tuple[HTCamera, float, float, float, float]:
    """Bring the two free-running sensors within ``tolerance_s`` of each other.

    Each sensor free-runs at an arbitrary phase, and nothing short of a full
    re-open moves it: pause/play, ``reset_timestamp`` and ROI/frame-speed
    bounces all leave the offset exactly where it was (the last two only snap
    it onto a coarse grid, never near zero). A re-open lands the right sensor
    on a fresh phase — measured on this rig it steps a few ms around the frame
    period each time — so measure and re-open until the phase falls inside the
    tolerance. Once aligned the offset holds: it moved by 0.012 ms over five
    minutes of bench measurement, so this only has to run at startup.

    Returns ``(right, left_anchor, right_anchor, skew_s, period_s)``.
    """
    left_anchor = _reset_anchor(left)
    right_anchor = _reset_anchor(right)
    serial = right.serial
    started = time.perf_counter()
    print(
        f"[stereo] phase-aligning sensors to within "
        f"{tolerance_s * 1e3:.2f} ms (re-opening {serial} until the random "
        "phase lands in range)"
    )
    try:
        for attempt in range(1, max_attempts + 1):
            skew, period = _measure_skew(
                left,
                right,
                left_anchor,
                right_anchor,
                capture_bits,
            )
            elapsed = time.perf_counter() - started
            if abs(skew) <= tolerance_s:
                print(
                    f"[stereo] sensors aligned to {skew * 1e3:+.3f} ms "
                    f"(period {period * 1e3:.2f} ms) after {attempt} "
                    f"attempt(s), {elapsed:.1f} s"
                )
                return right, left_anchor, right_anchor, skew, period
            expected = period / (2.0 * tolerance_s)
            print(
                f"[stereo] align attempt {attempt}/{max_attempts} "
                f"({elapsed:.1f} s, ~{expected:.0f} expected): skew "
                f"{skew * 1e3:+.3f} ms exceeds {tolerance_s * 1e3:.2f} ms; "
                f"re-opening {serial}"
            )
            right.close()
            right = _open_camera(
                serial,
                exposure_ms,
                gain,
                capture_bits,
                vertical_trim,
            )
            right_anchor = _reset_anchor(right)
    except BaseException:
        # main() still holds the handle we were given; only a replacement it
        # has never seen needs closing here.
        right.close()
        raise
    right.close()
    raise RuntimeError(
        f"could not align the sensors within {tolerance_s * 1e3:.3f} ms in "
        f"{max_attempts} attempts; raise --align-tolerance-ms or "
        "--align-max-attempts"
    )


def _report_profile(samples: list) -> None:
    """Print median/p95 for each publish-loop stage, in milliseconds."""
    names = (
        "capture->SDK return",
        "pixel conversion",
        "conversion->paired",
        "auto-exposure",
        "encode",
        "send",
        "POST-CAPTURE",
        "TOTAL capture->wire",
    )
    columns = list(zip(*samples))
    print(f"[profile] {len(samples)} pairs")
    for name, column in zip(names, columns):
        ordered = sorted(column)
        median = ordered[len(ordered) // 2]
        p95 = ordered[int(len(ordered) * 0.95)]
        print(
            f"[profile]   {name:<22} med {median * 1e3:7.2f}  "
            f"p95 {p95 * 1e3:7.2f} ms"
        )


def _next_pair(
    left_queue: _DropOldestQueue,
    right_queue: _DropOldestQueue,
    match_window: float,
    timeout: float,
):
    """Pop one frame from each side, dropping whichever side is behind until
    the two land within ``match_window``.

    With the sensors phase-aligned at startup this is a straight pop from each
    queue; the drop loop only runs after a frame goes missing.
    """
    left_frame = left_queue.get(timeout=timeout)
    right_frame = right_queue.get(timeout=timeout)
    while abs(right_frame[1] - left_frame[1]) > match_window:
        if left_frame[1] < right_frame[1]:
            left_frame = left_queue.get(timeout=timeout)
        else:
            right_frame = right_queue.get(timeout=timeout)
    return left_frame, right_frame


def _capture(
    camera: HTCamera,
    anchor: float,
    output: _DropOldestQueue,
    stop: threading.Event,
    errors: list,
    priority: int,
    capture_bits: int,
) -> None:
    """Grab, demosaic and tonemap one camera, and queue the finished RGB.

    Doing the pixel work here rather than in the publish loop is what keeps it
    off the critical path: each camera has a full frame period of slack, so its
    conversion overlaps the *other* camera's USB transfer and its own next
    exposure. The publish loop then only has to pair, encode and send.
    """
    cv_code = camera._cv_code
    wb_gains = camera.wb_gains
    lut8 = _bt709_lut8(wb_gains) if capture_bits == 8 else None
    try:
        while not stop.is_set():
            bayer, info = _grab_bayer(
                camera,
                capture_bits,
                timeout_ms=500,
                priority=priority,
            )
            if bayer is None:
                continue
            grabbed_perf = time.perf_counter()
            hardware_us = int(info["time"])
            capture_perf = anchor + hardware_us * 1e-6
            # Discard SDK-buffered frames from before reset_timestamp().
            if capture_perf > time.perf_counter() + 1.0:
                continue
            linear = convert.demosaic(bayer, cv_code)
            if capture_bits == 8:
                rgb = cv2.LUT(linear, lut8)
            else:
                rgb = convert.tonemap_linear(
                    linear,
                    curve="bt709",
                    out_dtype=np.uint8,
                    wb_gains=wb_gains,
                )
            output.put(
                (
                    hardware_us,
                    capture_perf,
                    grabbed_perf,
                    time.perf_counter(),
                    bayer,
                    rgb,
                )
            )
    except Exception as exc:
        errors.append(exc)
        stop.set()


def _open_camera(
    serial: str,
    exposure_ms: float,
    gain: float,
    capture_bits: int = 12,
    vertical_trim: float = 0.0,
) -> HTCamera:
    camera = HTCamera(serial=serial, demosaic_quality="ea")
    camera.set_frame_speed(enums.FRAME_SPEED_HIGH)
    resolution = camera.current_resolution()
    full_width = int(resolution.iWidthFOV)
    full_height = int(resolution.iHeightFOV)
    trim = int(round(full_height * vertical_trim)) & ~1
    height = full_height - 2 * trim
    if height < 2:
        camera.close()
        raise ValueError("--vertical-trim removes the entire sensor image")
    camera.set_roi(
        full_width,
        height,
        0,
        trim,
    )
    if capture_bits == 8:
        _select_bayer8(camera)
    camera.set_ae(False)
    camera.set_exposure_ms(exposure_ms)
    camera.set_analog_gain(gain)
    for _ in range(3):
        frame, _ = _grab_bayer(
            camera,
            capture_bits,
            timeout_ms=2000,
        )
        if frame is not None:
            return camera
    camera.close()
    raise RuntimeError(f"{serial}: camera opened but delivered no frames")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head-pose", type=Path)
    parser.add_argument("--allow-identity-head-pose", action="store_true")
    parser.add_argument("--exposure-ms", type=float, default=5.0)
    parser.add_argument("--gain", type=float, default=3.0)
    parser.add_argument("--ae-target", type=float, default=0.35)
    parser.add_argument("--ae-exp-min", type=float, default=5.0)
    parser.add_argument("--ae-exp-max", type=float, default=17.0)
    parser.add_argument("--ae-gain-max", type=float, default=4.0)
    parser.add_argument("--ae-flicker-hz", type=float, default=60.0)
    parser.add_argument(
        "--capture-bits",
        type=int,
        choices=(8, 12),
        default=8,
        help="raw Bayer transport depth; 8 substantially reduces USB latency",
    )
    parser.add_argument(
        "--vertical-trim",
        type=float,
        default=0.10,
        metavar="FRACTION",
        help="fraction cropped from each of the sensor's top and bottom edges",
    )
    # Each re-open costs ~1.3 s and lands on a fresh phase, so the startup
    # cost is roughly (period / 2 / tolerance) attempts: ~4 s at 5 ms, ~27 s
    # at 1 ms.
    parser.add_argument("--align-tolerance-ms", type=float, default=5.0)
    parser.add_argument("--align-max-attempts", type=int, default=60)
    parser.add_argument(
        "--profile",
        action="store_true",
        help="print publish-loop stage timings every 100 pairs",
    )
    # The library default is 4, which leaves this 8-core box idle: the two
    # conversions run concurrently in the capture threads, and 8 measured
    # ~2.3 ms/pair faster end to end. Past 8 it flattens out.
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument(
        "--grab-priority",
        choices=("newest", "next"),
        default="newest",
        help="SDK frame-buffer policy (next may reduce latency on some cameras)",
    )
    parser.add_argument(
        "--transport",
        choices=("iceoryx2", "zmq"),
        default="iceoryx2",
    )
    args = parser.parse_args()
    if not 0.0 <= args.vertical_trim < 0.5:
        parser.error("--vertical-trim must be in [0, 0.5)")
    convert.set_num_threads(args.threads)

    present = {camera["serial"] for camera in list_cameras()}
    stereo, stereo_path = _find_stereo_calibration(present)
    (
        base_from_head,
        head_from_stereo_midpoint,
        pose_explicit,
        pose_source,
    ) = _load_head_pose(
        stereo, args.head_pose, args.allow_identity_head_pose
    )
    print(
        f"[stereo] calibration {stereo_path}; "
        f"left={stereo.serial_left} right={stereo.serial_right}"
    )
    print(f"[stereo] head pose: {pose_source}")

    left = right = None
    stop = threading.Event()
    Publisher, _ = transport_classes(args.transport)
    policy_publisher = None
    errors = []
    try:
        left = _open_camera(
            stereo.serial_left,
            args.exposure_ms,
            args.gain,
            args.capture_bits,
            args.vertical_trim,
        )
        right = _open_camera(
            stereo.serial_right,
            args.exposure_ms,
            args.gain,
            args.capture_bits,
            args.vertical_trim,
        )
        # Phase-align before anything else binds to the right camera object:
        # alignment re-opens it, which invalidates the previous handle.
        (
            right,
            left_anchor,
            right_anchor,
            aligned_skew,
            period,
        ) = _align_phase(
            left,
            right,
            args.exposure_ms,
            args.gain,
            args.align_tolerance_ms * 1e-3,
            args.align_max_attempts,
            args.capture_bits,
            args.vertical_trim,
        )
        rig, calibration_hash = _build_rig(
            stereo,
            left,
            right,
            base_from_head,
            head_from_stereo_midpoint,
        )
        rig["base_from_head_explicit"] = pose_explicit
        left_projection = rig["left"]
        right_projection = rig["right"]
        print(
            f"[stereo] capture {args.capture_bits}-bit; "
            f"ROI {left_projection['width']}x{left_projection['height']}"
            f"; "
            f"principal points "
            f"L=({left_projection['K'][0][2]:.3f}, "
            f"{left_projection['K'][1][2]:.3f}) "
            f"R=({right_projection['K'][0][2]:.3f}, "
            f"{right_projection['K'][1][2]:.3f})"
        )
        import hashlib
        calibration_hash = hashlib.sha256(
            json.dumps(
                rig, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()

        gain_min, gain_max, _ = left.gain_range()
        ae = AutoExposure(
            gain_min,
            min(gain_max, args.ae_gain_max),
            exp_min=args.ae_exp_min,
            exp_max=args.ae_exp_max,
            target=args.ae_target,
            flicker_hz=args.ae_flicker_hz,
            max_val=255.0 if args.capture_bits == 8 else 4095.0,
        )
        exposure, gain = args.exposure_ms, args.gain
        ae.apply(right, exposure, gain, exposure, gain)

        left_queue = _DropOldestQueue(4)
        right_queue = _DropOldestQueue(4)
        grab_priority = {
            "newest": enums.GET_NEWEST,
            "next": enums.GET_NEXT,
        }[args.grab_priority]
        threads = [
            threading.Thread(
                target=_capture,
                args=(
                    left,
                    left_anchor,
                    left_queue,
                    stop,
                    errors,
                    grab_priority,
                    args.capture_bits,
                ),
                daemon=True,
            ),
            threading.Thread(
                target=_capture,
                args=(
                    right,
                    right_anchor,
                    right_queue,
                    stop,
                    errors,
                    grab_priority,
                    args.capture_bits,
                ),
                daemon=True,
            ),
        ]
        for thread in threads:
            thread.start()

        policy_publisher = Publisher(
            PAIR_TOPIC,
            conflate=False,
            max_buffer_size=30,
            max_message_size=32 * 1024 * 1024,
        )
        stream_id = str(uuid.uuid4())
        sequence = 0
        profile = [] if args.profile else None
        match_window = period / 2.0
        drift_limit = max(args.align_tolerance_ms * 1e-3 * 4.0, 2e-3)
        last_drift_warning = 0.0

        def handle_signal(_signum, _frame):
            stop.set()

        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)

        while not stop.is_set():
            try:
                left_frame, right_frame = _next_pair(
                    left_queue, right_queue, match_window, _STOP_POLL_S
                )
            except queue.Empty:
                continue
            (
                left_hw,
                left_perf,
                left_grabbed,
                left_converted,
                left_bayer,
                left_rgb,
            ) = left_frame
            (
                right_hw,
                right_perf,
                right_grabbed,
                right_converted,
                _right_bayer,
                right_rgb,
            ) = right_frame
            skew = right_perf - left_perf
            if abs(skew) > drift_limit:
                now = time.perf_counter()
                if now - last_drift_warning > 5.0:
                    last_drift_warning = now
                    print(
                        f"[stereo] sensor skew {skew * 1e3:+.3f} ms has drifted "
                        f"past {drift_limit * 1e3:.3f} ms (aligned at "
                        f"{aligned_skew * 1e3:+.3f} ms); restart to re-align"
                    )

            t_popped = time.perf_counter()
            new_exp, new_gain, _ = ae.update(left_bayer, exposure, gain)
            exposure, gain = ae.apply(
                (left, right), new_exp, new_gain, exposure, gain
            )
            t_ae = time.perf_counter()
            pair_perf = (left_perf + right_perf) / 2.0
            metadata = {
                "stream_id": stream_id,
                "sequence": sequence,
                "calibration_hash": calibration_hash,
                "pair_capture_perf_s": pair_perf,
                "left_capture_perf_s": left_perf,
                "right_capture_perf_s": right_perf,
                "left_hardware_timestamp_us": left_hw,
                "right_hardware_timestamp_us": right_hw,
                "sensor_skew_s": skew,
                "publish_perf_s": time.perf_counter(),
                "exposure_ms": exposure,
                "gain_x": gain,
                "base_from_head": base_from_head.tolist(),
                "rig": rig,
            }
            packet = _encode_pair(metadata, left_rgb, right_rgb)
            t_encoded = time.perf_counter()
            policy_publisher.send_bytes(packet)
            sequence += 1

            if profile is not None:
                t_sent = time.perf_counter()
                profile.append(
                    (
                        max(left_grabbed - left_perf,
                            right_grabbed - right_perf),
                        max(left_converted - left_grabbed,
                            right_converted - right_grabbed),
                        t_popped - max(left_converted, right_converted),
                        t_ae - t_popped,        # auto-exposure
                        t_encoded - t_ae,       # encode/copy
                        t_sent - t_encoded,     # send
                        t_sent - t_popped,      # post-capture critical path
                        t_sent - pair_perf,     # total, capture -> on wire
                    )
                )
                if len(profile) >= 100:
                    _report_profile(profile)
                    profile.clear()

        for thread in threads:
            thread.join(timeout=2.0)
        if errors:
            raise RuntimeError(f"capture thread failed: {errors[0]}")
    finally:
        stop.set()
        if policy_publisher is not None:
            policy_publisher.close()
        for camera in (left, right):
            if camera is not None:
                camera.close()


if __name__ == "__main__":
    main()
