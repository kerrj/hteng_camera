"""Publish a calibrated HTENG stereo pair as one self-describing ZMQ packet.

The publisher owns camera discovery, timestamp reset/mapping, frame pairing,
software auto-exposure, color conversion, and rig metadata. Consumers only need
to decode ``eyeball-stereo-pair/1`` from ``ipc:///tmp/stereo_pair``.
"""

from __future__ import annotations

import argparse
from collections import deque
import json
import math
from pathlib import Path
import queue
import signal
import struct
import threading
import time
import uuid

import cv2
import numpy as np
from hteng_camera import (
    AutoExposure,
    HTCamera,
    calibration,
    convert,
    enums,
    list_cameras,
)
from eye.transport import transport_classes


PAIR_TOPIC = "ipc:///tmp/stereo_pair"
RECORD_TOPIC = "ipc:///tmp/stereo_pair_record"
STREAM_FORMAT = "eyeball-stereo-pair/1"
RIG_FORMAT = "eyeball-stereo-rig/2"
_MAGIC = b"ESTPAIR1"
_PREFIX = struct.Struct("<8sIQQ")
_MJ_TO_CV = np.diag([1.0, -1.0, -1.0])
_HEAD_FROM_CANONICAL_MJ = np.array(
    [[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    dtype=np.float64,
)


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
        if (width, height) != tuple(intr.image_size):
            raise RuntimeError(
                f"{cam.serial}: full ROI {width}x{height} does not match "
                f"intrinsics {intr.image_size}; refusing an implicit rescale"
            )
        return {
            "serial": cam.serial,
            "width": width,
            "height": height,
            "model": intr.model,
            "K": intr.K.tolist(),
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


def _play_together(cameras: tuple[HTCamera, HTCamera]) -> None:
    barrier = threading.Barrier(len(cameras) + 1)
    errors = []

    def play(camera: HTCamera) -> None:
        try:
            barrier.wait()
            camera.play()
        except Exception as exc:
            errors.append(exc)

    threads = [
        threading.Thread(target=play, args=(camera,), daemon=True)
        for camera in cameras
    ]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join()
    if errors:
        raise RuntimeError(f"parallel camera start failed: {errors[0]}")


def _capture(
    camera: HTCamera,
    anchor: float,
    output: _DropOldestQueue,
    stop: threading.Event,
    errors: list,
) -> None:
    try:
        while not stop.is_set():
            bayer, info = camera.grab_bayer12(timeout_ms=500)
            if bayer is None:
                continue
            hardware_us = int(info["time"])
            capture_perf = anchor + hardware_us * 1e-6
            # Discard SDK-buffered frames from before reset_timestamp().
            if capture_perf > time.perf_counter() + 1.0:
                continue
            output.put((hardware_us, capture_perf, bayer))
    except Exception as exc:
        errors.append(exc)
        stop.set()


def _open_camera(serial: str, exposure_ms: float, gain: float) -> HTCamera:
    camera = HTCamera(serial=serial, demosaic_quality="ea")
    camera.set_frame_speed(enums.FRAME_SPEED_HIGH)
    resolution = camera.current_resolution()
    camera.set_roi(
        int(resolution.iWidthFOV), int(resolution.iHeightFOV), 0, 0
    )
    camera.set_ae(False)
    camera.set_exposure_ms(exposure_ms)
    camera.set_analog_gain(gain)
    for _ in range(3):
        frame, _ = camera.grab_bayer12(timeout_ms=2000)
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
    parser.add_argument("--pair-wait-ms", type=float, default=30.0)
    parser.add_argument(
        "--transport",
        choices=("iceoryx2", "zmq"),
        default="iceoryx2",
    )
    args = parser.parse_args()

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
    policy_publisher = record_publisher = None
    errors = []
    try:
        left = _open_camera(stereo.serial_left, args.exposure_ms, args.gain)
        right = _open_camera(stereo.serial_right, args.exposure_ms, args.gain)
        rig, calibration_hash = _build_rig(
            stereo,
            left,
            right,
            base_from_head,
            head_from_stereo_midpoint,
        )
        rig["base_from_head_explicit"] = pose_explicit
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
        )
        exposure, gain = args.exposure_ms, args.gain
        ae.apply(right, exposure, gain, exposure, gain)

        # Opening and warming the cameras sequentially leaves their free-running
        # exposure cycles at an arbitrary phase. Restart them together before
        # anchoring their counters so nearest-time pairs are also close in phase.
        left.pause()
        right.pause()
        _play_together((left, right))
        left_anchor = _reset_anchor(left)
        right_anchor = _reset_anchor(right)
        left_queue = _DropOldestQueue(4)
        right_queue = _DropOldestQueue(8)
        threads = [
            threading.Thread(
                target=_capture,
                args=(left, left_anchor, left_queue, stop, errors),
                daemon=True,
            ),
            threading.Thread(
                target=_capture,
                args=(right, right_anchor, right_queue, stop, errors),
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
        if args.transport == "zmq":
            record_publisher = Publisher(
                RECORD_TOPIC,
                conflate=False,
                max_buffer_size=30,
                max_message_size=32 * 1024 * 1024,
            )
        stream_id = str(uuid.uuid4())
        sequence = 0
        right_buffer = deque(maxlen=8)
        right_intervals = deque(maxlen=16)
        last_right_perf = None
        pairing_primed = False

        def handle_signal(_signum, _frame):
            stop.set()

        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)

        while not stop.is_set():
            try:
                left_hw, left_perf, left_bayer = left_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            deadline = time.perf_counter() + args.pair_wait_ms * 1e-3
            while True:
                if right_buffer:
                    latest_gap = left_perf - right_buffer[-1][1]
                    period = (
                        float(np.median(right_intervals))
                        if right_intervals else None
                    )
                    if latest_gap <= 0:
                        break
                    if period is not None and latest_gap < period / 2.0:
                        break
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                try:
                    frame = right_queue.get(timeout=remaining)
                    right_perf = frame[1]
                    if last_right_perf is not None:
                        interval = right_perf - last_right_perf
                        if 0.0 < interval < 0.2:
                            right_intervals.append(interval)
                    last_right_perf = right_perf
                    right_buffer.append(frame)
                except queue.Empty:
                    break
            if not right_buffer:
                continue
            # The first left frame can precede every buffered right frame. Keep
            # those right frames and start with the next left frame so nearest
            # matching can choose either side of its timestamp instead of
            # locking the whole stream to the later phase.
            if not pairing_primed:
                pairing_primed = True
                continue
            right_frame = min(
                right_buffer, key=lambda frame: abs(frame[1] - left_perf)
            )
            right_hw, right_perf, right_bayer = right_frame
            while right_buffer and right_buffer[0][1] <= right_perf:
                right_buffer.popleft()

            new_exp, new_gain, _ = ae.update(left_bayer, exposure, gain)
            exposure, gain = ae.apply(
                (left, right), new_exp, new_gain, exposure, gain
            )
            left_rgb = convert.tonemap_linear(
                convert.demosaic(left_bayer, left._cv_code),
                curve="bt709",
                out_dtype=np.uint8,
                wb_gains=left.wb_gains,
            )
            right_rgb = convert.tonemap_linear(
                convert.demosaic(right_bayer, right._cv_code),
                curve="bt709",
                out_dtype=np.uint8,
                wb_gains=right.wb_gains,
            )
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
                "sensor_skew_s": right_perf - left_perf,
                "publish_perf_s": time.perf_counter(),
                "exposure_ms": exposure,
                "gain_x": gain,
                "base_from_head": base_from_head.tolist(),
                "rig": rig,
            }
            packet = _encode_pair(metadata, left_rgb, right_rgb)
            policy_publisher.send_bytes(packet)
            if record_publisher is not None:
                record_publisher.send_bytes(packet)
            sequence += 1

        for thread in threads:
            thread.join(timeout=2.0)
        if errors:
            raise RuntimeError(f"capture thread failed: {errors[0]}")
    finally:
        stop.set()
        for publisher in (policy_publisher, record_publisher):
            if publisher is not None:
                publisher.close()
        for camera in (left, right):
            if camera is not None:
                camera.close()


if __name__ == "__main__":
    main()
