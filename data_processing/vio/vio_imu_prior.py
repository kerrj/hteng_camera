"""VIO pipeline stage 4: turn raw IMU samples into per-frame priors for stage 5
bundle adjustment.

Two separate quantities, not fused together:

1. Relative rotation (frame i -> i+1): strapdown gyro integration over the
   ~33ms inter-frame window, gyro only -- no Madgwick/Mahony fusion, since
   that blends in an accel-based tilt correction that assumes accel==gravity,
   false during real translational acceleration (this recording: |accel|
   0.70g-1.28g). 33ms is too short for gyro drift to matter anyway.

2. Per-frame gravity direction + confidence weight: anchors roll/pitch
   (relative rotation alone leaves them free to drift), from a windowed
   raw-accel average, weighted by closeness to 1g (downweighted, not
   discarded, when real acceleration corrupts the signal).

IMU->camera extrinsic rotation R_CI is a fixed CAD-derived constant (IMU
+x/+y/+z = forward/left/up -> cv2 camera +z/+x/+y = forward/right/down), not
calibrated. Stationary accel reads ~+1g AWAY from ground (normal force, not
gravity itself), so normalize(mean accel) gives "up" directly, no sign flip.

Frame timestamps: sync_log.csv's t_left_us + recording.json's
imu.cam_reset_perf_counter_us, landing in imu_log.csv's host_time_us clock.
One frame's t_left_us can be pre-reset garbage -- detected generically via
timestamp non-monotonicity, not a hardcoded index.

Run (from data_processing/vio/):
    python vio_imu_prior.py ../../testimu --out ../../testimu/imu_relative.npz
"""
import argparse
import bisect
import csv
import json
import os

import numpy as np

R_CI = np.array([
    [0.0, -1.0, 0.0],
    [0.0, 0.0, -1.0],
    [1.0, 0.0, 0.0],
], dtype=np.float64)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("recording")
    p.add_argument("--imu-log", default=None, help="default: <recording>/imu_log.csv")
    p.add_argument("--sync-log", default=None, help="default: <recording>/sync_log.csv")
    p.add_argument("--recording-json", default=None,
                    help="default: <recording>/recording.json")
    p.add_argument("--out", default=None, help="default: <recording>/derived/imu_relative.npz")
    p.add_argument("--gravity-half-window-ms", type=float, default=50.0,
                    help="accel samples within +-this many ms of a frame's timestamp "
                         "are averaged to estimate that frame's gravity direction")
    p.add_argument("--gravity-max-accel-dev", type=float, default=0.15,
                    help="gravity confidence weight decays linearly to 0 as the "
                         "window's mean |accel| deviates from 1g by up to this much "
                         "(in g) -- real translational acceleration corrupts the "
                         "gravity signal, so a window during fast motion should "
                         "contribute little to nothing")
    return p.parse_args()


def load_frame_times_us(sync_log_path, cam_reset_perf_counter_us):
    """Returns (frame_idx, frame_time_us, valid). valid[i] is False if
    frame_time_us doesn't strictly increase from the previous entry."""
    idx, times = [], []
    with open(sync_log_path) as f:
        for row in csv.DictReader(f):
            if row["t_left_us"] == "":
                continue
            idx.append(int(row["frame"]))
            times.append(cam_reset_perf_counter_us + int(row["t_left_us"]))
    idx = np.asarray(idx, dtype=np.int64)
    times = np.asarray(times, dtype=np.int64)
    valid = np.ones(len(times), dtype=bool)
    valid[:-1][np.diff(times) <= 0] = False
    return idx, times, valid


def load_imu_log(imu_log_path):
    """Returns dict of (N,) float64 arrays: host_time_us, gx, gy, gz, ax, ay, az."""
    cols = {"host_time_us": [], "gx": [], "gy": [], "gz": [], "ax": [], "ay": [], "az": []}
    with open(imu_log_path) as f:
        for row in csv.DictReader(f):
            cols["host_time_us"].append(int(row["host_time_us"]))
            cols["gx"].append(float(row["gx_rad_s"]))
            cols["gy"].append(float(row["gy_rad_s"]))
            cols["gz"].append(float(row["gz_rad_s"]))
            cols["ax"].append(float(row["ax_g"]))
            cols["ay"].append(float(row["ay_g"]))
            cols["az"].append(float(row["az_g"]))
    return {k: np.asarray(v, dtype=np.float64) for k, v in cols.items()}


def interp_at(t, ts, vals):
    """Linear interpolation of vals (N,3) at time t, given ts (N,) sorted.
    Clamps at the ends rather than extrapolating."""
    i = bisect.bisect_left(ts, t)
    if i <= 0:
        return vals[0]
    if i >= len(ts):
        return vals[-1]
    t0, t1 = ts[i - 1], ts[i]
    frac = (t - t0) / (t1 - t0)
    return vals[i - 1] + (vals[i] - vals[i - 1]) * frac


def axis_angle_to_quat(v):
    """v: (3,) rotation vector (axis * angle, radians). Returns wxyz quat."""
    angle = np.linalg.norm(v)
    if angle < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = v / angle
    half = angle / 2.0
    return np.concatenate([[np.cos(half)], axis * np.sin(half)])


def quat_mult(q1, q2):
    """Hamilton product, wxyz convention."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def integrate_gyro_window(t_a, t_b, i_ts, gyro):
    """Strapdown quaternion integration of body-frame angular velocity `gyro`
    (N,3) sampled at `i_ts` (N,) over [t_a, t_b] (host_time_us, same clock).
    Builds the exact sample sequence within the window (interpolating the two
    endpoints so the integration covers precisely [t_a, t_b], not whatever
    samples happen to land inside it) then composes one small rotation per
    sub-interval via trapezoidal-averaged angular velocity. Returns wxyz quat
    (IMU frame, frame_a -> frame_b)."""
    lo = bisect.bisect_right(i_ts, t_a)
    hi = bisect.bisect_left(i_ts, t_b)
    times = np.concatenate([[t_a], i_ts[lo:hi], [t_b]])
    vals = np.concatenate([[interp_at(t_a, i_ts, gyro)], gyro[lo:hi],
                            [interp_at(t_b, i_ts, gyro)]], axis=0)
    q = np.array([1.0, 0.0, 0.0, 0.0])
    for k in range(len(times) - 1):
        dt = (times[k + 1] - times[k]) / 1e6
        omega = (vals[k] + vals[k + 1]) / 2.0
        q = quat_mult(q, axis_angle_to_quat(omega * dt))
    return q


def frame_gravity(t, i_ts, accel, half_window_us, max_dev_g):
    """Mean accel (IMU frame) over samples within `half_window_us` of `t`,
    normalized to unit "up", plus its norm and a legacy [0,1] confidence
    weight. Returns (up_imu (3,), accel_norm_g, weight)."""
    lo = bisect.bisect_left(i_ts, t - half_window_us)
    hi = bisect.bisect_right(i_ts, t + half_window_us)
    if hi <= lo:
        i = min(bisect.bisect_left(i_ts, t), len(i_ts) - 1)
        window = accel[i:i + 1]
    else:
        window = accel[lo:hi]
    mean_accel = window.mean(axis=0)
    mag = np.linalg.norm(mean_accel)
    up_imu = mean_accel / max(mag, 1e-9)
    dev = abs(mag - 1.0)
    weight = max(0.0, 1.0 - dev / max_dev_g)
    return up_imu, mag, weight


def main():
    args = parse_args()
    imu_log_path = args.imu_log or os.path.join(args.recording, "imu_log.csv")
    sync_log_path = args.sync_log or os.path.join(args.recording, "sync_log.csv")
    rec_json_path = args.recording_json or os.path.join(args.recording, "recording.json")
    out_path = args.out or os.path.join(args.recording, "derived", "imu_relative.npz")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    rec = json.load(open(rec_json_path))
    if not rec.get("imu", {}).get("enabled"):
        raise SystemExit(f"{rec_json_path}: imu.enabled is false -- no IMU data "
                          f"was captured for this recording")
    cam_reset_us = rec["imu"]["cam_reset_perf_counter_us"]

    frame_idx, frame_time_us, frame_valid = load_frame_times_us(sync_log_path, cam_reset_us)
    n_bad = (~frame_valid).sum()
    print(f"{len(frame_idx)} frames with a timestamp, {n_bad} invalid "
          f"(non-monotonic -- see module docstring)")

    imu = load_imu_log(imu_log_path)
    i_ts = imu["host_time_us"]
    gyro = np.stack([imu["gx"], imu["gy"], imu["gz"]], axis=1)
    accel = np.stack([imu["ax"], imu["ay"], imu["az"]], axis=1)
    valid_times = frame_time_us[frame_valid]
    print(f"{len(i_ts)} imu samples, span {(i_ts[-1] - i_ts[0]) / 1e6:.2f}s, "
          f"frames span {(valid_times[-1] - valid_times[0]) / 1e6:.2f}s "
          f"(over valid frames only)")
    if i_ts[0] > valid_times[0] or i_ts[-1] < valid_times[-1]:
        print("WARNING: imu_log.csv does not fully bracket the frame timestamps -- "
              "boundary frames will use clamped (not interpolated) accel/gyro values")

    half_window_us = args.gravity_half_window_ms * 1000.0
    n = len(frame_idx)
    gravity_cam = np.zeros((n, 3), dtype=np.float64)
    gravity_accel_norm_g = np.zeros(n, dtype=np.float64)
    gravity_weight = np.zeros(n, dtype=np.float64)
    for k in range(n):
        up_imu, accel_norm_g, w = frame_gravity(
            frame_time_us[k],
            i_ts,
            accel,
            half_window_us,
            args.gravity_max_accel_dev,
        )
        # gravity_cam stores the DOWN direction (toward Earth) -- frame_gravity
        # returns "up" (see its docstring on the accel sign convention), so
        # flip here to match this field's name/what downstream consumers
        # (vio_visualize_imu_prior.py, the BA gravity prior) both expect.
        gravity_cam[k] = -(R_CI @ up_imu)
        gravity_accel_norm_g[k] = accel_norm_g
        gravity_weight[k] = w if frame_valid[k] else 0.0

    rel_quat = np.zeros((n - 1, 4), dtype=np.float64)
    rel_valid = np.zeros(n - 1, dtype=bool)
    for k in range(n - 1):
        if not (frame_valid[k] and frame_valid[k + 1]):
            rel_quat[k] = [1.0, 0.0, 0.0, 0.0]
            continue
        q_imu = integrate_gyro_window(frame_time_us[k], frame_time_us[k + 1], i_ts, gyro)
        # rotate the relative-rotation quaternion into the camera frame:
        # R_cam = R_CI @ R_imu @ R_CI.T, expressed here directly on the quat's
        # equivalent rotation matrix (simpler than deriving a quat conjugation
        # identity by hand, and this runs once per frame pair, not hot code).
        w, x, y, z = q_imu
        R_imu = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ])
        R_cam = R_CI @ R_imu @ R_CI.T
        trace = np.trace(R_cam)
        qw = np.sqrt(max(trace + 1.0, 0.0)) / 2.0
        if qw > 1e-6:
            qx = (R_cam[2, 1] - R_cam[1, 2]) / (4 * qw)
            qy = (R_cam[0, 2] - R_cam[2, 0]) / (4 * qw)
            qz = (R_cam[1, 0] - R_cam[0, 1]) / (4 * qw)
        else:
            # near-180-degree rotation -- fall back to a numerically safer
            # extraction (won't happen for a single ~33ms step in practice,
            # kept only so this never silently divides by ~0).
            qx = np.sqrt(max((R_cam[0, 0] + 1) / 2.0, 0.0))
            qy = np.sqrt(max((R_cam[1, 1] + 1) / 2.0, 0.0))
            qz = np.sqrt(max((R_cam[2, 2] + 1) / 2.0, 0.0))
        rel_quat[k] = [qw, qx, qy, qz]
        rel_valid[k] = True

    n_grav_gated = int((gravity_weight < 0.5).sum())
    print(f"gravity: {n_grav_gated}/{n} frames with weight<0.5 (fast-motion windows)")
    print(f"relative rotation: {rel_valid.sum()}/{n - 1} edges valid")

    np.savez(out_path,
              frame_idx=frame_idx, frame_time_us=frame_time_us, frame_valid=frame_valid,
              rel_quat=rel_quat, rel_valid=rel_valid,
              gravity_cam=gravity_cam,
              gravity_accel_norm_g=gravity_accel_norm_g,
              gravity_weight=gravity_weight)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
