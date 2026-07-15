"""Fuse stereo VGGT-Omega windows and IMU priors in a JAXLS pose graph.

Inputs are the camera measurements emitted by ``vio_vggt_window_infer.py``.
The graph has one world-to-left-camera SE(3) variable per retained video frame
and one log-scale variable per Omega window. Visual relative-pose factors share
the window scale. Stereo baseline factors make those scales metric. IMU
relative-rotation factors are composed between retained frames, and gravity
anchors roll/pitch without imposing global yaw.

The output follows this repository's trajectory NPZ contract and can be opened
with ``data_processing/visualize_data.py --no-color``.
"""

import argparse
import glob
import json
import os
import time

import jax
import jax.numpy as jnp
import jaxlie
import jaxls
import numpy as np


class WindowLogScaleVar(
        jaxls.Var[jax.Array], default_factory=lambda: jnp.asarray(0.0)):
    """Metric scale applied to one foundation-model inference window."""


def quat_to_matrix(quat):
    quat = np.asarray(quat)
    w, x, y, z = np.moveaxis(quat, -1, 0)
    return np.stack([
        np.stack([
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
        ], axis=-1),
        np.stack([
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
        ], axis=-1),
        np.stack([
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ], axis=-1),
    ], axis=-2)


def matrix_to_quat(rotation):
    """Stable rotation matrix to wxyz quaternion conversion."""
    return np.asarray(jaxlie.SO3.from_matrix(
        jnp.asarray(rotation)).wxyz, dtype=np.float64)


def matrix_to_pose(matrix):
    return np.concatenate([
        matrix_to_quat(matrix[:3, :3]),
        np.asarray(matrix[:3, 3], np.float64),
    ])


def pose_to_matrix(pose):
    output = np.eye(4)
    output[:3, :3] = quat_to_matrix(pose[:4])
    output[:3, 3] = pose[4:]
    return output


def compose_stereo_poses(left_poses, R_stereo, t_stereo):
    """Compose calibrated left-to-right extrinsics with world-to-left poses."""
    from scipy.spatial.transform import Rotation

    left_poses = np.asarray(left_poses, np.float64)
    stereo = np.eye(4, dtype=np.float64)
    stereo[:3, :3] = np.asarray(R_stereo, np.float64)
    stereo[:3, 3] = np.asarray(t_stereo, np.float64).reshape(3)
    right_matrices = np.einsum(
        "ij,njk->nik",
        stereo,
        np.asarray([pose_to_matrix(pose) for pose in left_poses]),
    )
    quat_xyzw = Rotation.from_matrix(
        right_matrices[:, :3, :3]).as_quat()
    return np.concatenate([
        quat_xyzw[:, [3, 0, 1, 2]],
        right_matrices[:, :3, 3],
    ], axis=1)


def interpolate_poses(frame_idx, poses, native_fps, target_frame_idx=None):
    """Interpolate world-to-camera poses at every native video frame."""
    from scipy.interpolate import CubicSpline
    from scipy.spatial.transform import Rotation, RotationSpline

    frame_idx = np.asarray(frame_idx, np.int64)
    poses = np.asarray(poses, np.float64)
    if len(frame_idx) < 2:
        return frame_idx.copy(), poses.copy()
    if np.any(np.diff(frame_idx) <= 0):
        raise ValueError("pose frame indices must be strictly increasing")

    key_times = frame_idx.astype(np.float64) / float(native_fps)
    if target_frame_idx is None:
        full_frame_idx = np.arange(
            frame_idx[0], frame_idx[-1] + 1, dtype=np.int64)
    else:
        full_frame_idx = np.asarray(target_frame_idx, np.int64)
        if np.any(np.diff(full_frame_idx) <= 0):
            raise ValueError("target frame indices must be strictly increasing")
    full_times = full_frame_idx.astype(np.float64) / float(native_fps)

    rotation_wc = Rotation.from_quat(poses[:, [1, 2, 3, 0]])
    rotation_cw = rotation_wc.inv()
    centers = -rotation_cw.apply(poses[:, 4:])

    full_centers = CubicSpline(
        key_times, centers, axis=0)(full_times)
    full_rotation_wc = RotationSpline(
        key_times, rotation_cw)(full_times).inv()
    full_translation = -full_rotation_wc.apply(full_centers)
    full_quat_xyzw = full_rotation_wc.as_quat()
    full_poses = np.concatenate([
        full_quat_xyzw[:, [3, 0, 1, 2]],
        full_translation,
    ], axis=1)
    return full_frame_idx, full_poses


def extrinsic_matrices(extrinsics):
    count = len(extrinsics)
    output = np.tile(np.eye(4), (count, 1, 1))
    output[:, :3, :4] = extrinsics
    return output


def relative_pose(extrinsic_i, extrinsic_j):
    """Camera-i to camera-j transform from world-to-camera extrinsics."""
    return extrinsic_j @ np.linalg.inv(extrinsic_i)


def visual_relative_cost(vals, pose_i, pose_j, scale_var, measured,
                         translation_weight, rotation_weight):
    actual = vals[pose_j] @ vals[pose_i].inverse()
    target = jaxlie.SE3(measured)
    scale = jnp.exp(vals[scale_var])
    translation = (
        actual.translation() - scale * target.translation()
    ) * translation_weight
    rotation = (
        target.rotation().inverse() @ actual.rotation()
    ).log() * rotation_weight
    return jnp.concatenate([translation, rotation])


def baseline_cost(vals, scale_var, predicted_baseline, known_baseline,
                  weight):
    # A dimensionless log ratio gives this factor comparable leverage at any
    # physical baseline, unlike a residual measured in meters.
    residual = (
        vals[scale_var]
        + jnp.log(predicted_baseline)
        - jnp.log(known_baseline)
    )
    return residual[None] * weight


def imu_rotation_cost(vals, pose_i, pose_j, delta, weight):
    R_i = vals[pose_i].rotation()
    R_j = vals[pose_j].rotation()
    return (
        jaxlie.SO3(delta).inverse() @ (R_i @ R_j.inverse())
    ).log() * weight


def gravity_cost(vals, pose_var, measured_down, weight):
    predicted = vals[pose_var].rotation() @ jnp.array([0.0, 0.0, -1.0])
    return (predicted - measured_down) * weight


def pose_anchor_cost(vals, pose_var, target, weight):
    return (jaxlie.SE3(target).inverse() @ vals[pose_var]).log() * weight


def constant_velocity_cost(vals, pose_prev, pose, pose_next,
                           dt_prev, dt_next, weight):
    def center(var):
        transform = vals[var]
        return -(transform.rotation().inverse() @ transform.translation())

    velocity_prev = (center(pose) - center(pose_prev)) / dt_prev
    velocity_next = (center(pose_next) - center(pose)) / dt_next
    return (velocity_next - velocity_prev) * weight


def quat_multiply(q1, q2):
    w1, x1, y1, z1 = np.moveaxis(np.asarray(q1), -1, 0)
    w2, x2, y2, z2 = np.moveaxis(np.asarray(q2), -1, 0)
    return np.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], axis=-1)


def compose_imu_rotation(imu, frame_a, frame_b):
    """Compose saved consecutive-frame IMU rotations over [frame_a, frame_b]."""
    frame_idx = imu["frame_idx"].astype(np.int64)
    positions = {int(frame): index for index, frame in enumerate(frame_idx)}
    if frame_a not in positions or frame_b not in positions:
        return None
    start, stop = positions[frame_a], positions[frame_b]
    if start >= stop:
        return None
    rel_valid = imu["rel_valid"][start:stop]
    if not np.all(rel_valid):
        return None
    output = np.array([1.0, 0.0, 0.0, 0.0])
    for delta in imu["rel_quat"][start:stop]:
        output = quat_multiply(output, delta)
        output /= np.linalg.norm(output)
    return output


def load_windows(windows_dir):
    windows = []
    for path in sorted(glob.glob(os.path.join(windows_dir, "*.npz"))):
        data = np.load(path)
        extrinsics = extrinsic_matrices(data["extrinsics"])
        if "image_frame_idx" in data and "image_eye" in data:
            image_frame_idx = data["image_frame_idx"].astype(np.int64)
            image_eye = data["image_eye"].astype(np.int8)
            if "image_segment" in data:
                image_segment = data["image_segment"].astype(np.int8)
            else:
                segment_by_frame = dict(zip(
                    data["frame_idx"].astype(np.int64),
                    data["segment"].astype(np.int8),
                ))
                image_segment = np.asarray([
                    segment_by_frame[int(frame)] for frame in image_frame_idx
                ], np.int8)
        else:
            frame_idx = data["frame_idx"].astype(np.int64)
            image_frame_idx = np.repeat(frame_idx, 2)
            image_eye = np.tile(np.asarray([0, 1], np.int8), len(frame_idx))
            image_segment = np.repeat(
                data["segment"].astype(np.int8), 2)
        left_mask = image_eye == 0
        windows.append({
            "path": path,
            "kind": str(data["kind"]),
            "frame_idx": image_frame_idx[left_mask],
            "segment": image_segment[left_mask],
            "extrinsics": extrinsics,
            "image_frame_idx": image_frame_idx,
            "image_eye": image_eye,
        })
    if not windows:
        raise ValueError(f"no window NPZ files found in {windows_dir}")
    return windows


def edge_pairs(window):
    """Sparse, well-conditioned relative edges from one inference window."""
    if window["kind"] == "loop":
        first = np.flatnonzero(window["segment"] == 0)
        second = np.flatnonzero(window["segment"] == 1)
        if not len(first) or not len(second):
            return []
        return [(
            int(first[len(first) // 2]),
            int(second[len(second) // 2]),
            "loop",
        )]

    count = len(window["frame_idx"])
    pairs = {
        (index, index + 1): "adjacent"
        for index in range(count - 1)
        if window["segment"][index] == window["segment"][index + 1]
    }
    for segment in np.unique(window["segment"]):
        ids = np.flatnonzero(window["segment"] == segment)
        if len(ids):
            anchor = int(ids[len(ids) // 2])
            for index in ids:
                if index == anchor:
                    continue
                pair = (min(anchor, int(index)), max(anchor, int(index)))
                pairs.setdefault(pair, "window_anchor")
    return sorted(
        (pair[0], pair[1], edge_type)
        for pair, edge_type in pairs.items()
        if pair[0] != pair[1]
    )


def build_measurements(windows, baseline_m):
    frames = sorted({
        int(frame)
        for window in windows
        for frame in window["frame_idx"]
        if window["kind"] == "track"
    })
    frame_to_pose = {frame: index for index, frame in enumerate(frames)}
    edges = []
    predicted_baselines = []
    initial_log_scales = []
    window_quality = []
    for window_id, window in enumerate(windows):
        if "image_eye" in window:
            left_mask = window["image_eye"] == 0
            right_mask = window["image_eye"] == 1
            left = window["extrinsics"][left_mask]
            left_frames = window["image_frame_idx"][left_mask]
            right_by_frame = {
                int(frame): extrinsic
                for frame, extrinsic in zip(
                    window["image_frame_idx"][right_mask],
                    window["extrinsics"][right_mask],
                )
            }
            stereo_pairs = [
                (left[index], right_by_frame[int(frame)])
                for index, frame in enumerate(left_frames)
                if int(frame) in right_by_frame
            ]
        else:
            left = window["extrinsics"][0::2]
            right = window["extrinsics"][1::2]
            stereo_pairs = list(zip(left, right))
        if not stereo_pairs:
            raise ValueError(f"{window['path']}: no stereo anchor pairs")
        stereo_relative = np.asarray([
            relative_pose(left_extrinsic, right_extrinsic)
            for left_extrinsic, right_extrinsic in stereo_pairs
        ])
        baselines = np.linalg.norm(stereo_relative[:, :3, 3], axis=1)
        finite = np.isfinite(baselines) & (baselines > 1e-6)
        if not np.any(finite):
            raise ValueError(f"{window['path']}: no valid predicted baselines")
        median_baseline = float(np.median(baselines[finite]))
        dispersion = float(
            np.median(np.abs(baselines[finite] - median_baseline))
            / median_baseline)
        predicted_baselines.append(median_baseline)
        initial_log_scales.append(np.log(baseline_m / median_baseline))
        window_quality.append(1.0 / (1.0 + 10.0 * dispersion))

        for local_i, local_j, edge_type in edge_pairs(window):
            frame_i = int(window["frame_idx"][local_i])
            frame_j = int(window["frame_idx"][local_j])
            if frame_i not in frame_to_pose or frame_j not in frame_to_pose:
                continue
            measurement = matrix_to_pose(relative_pose(
                left[local_i], left[local_j]))
            edges.append((
                frame_to_pose[frame_i],
                frame_to_pose[frame_j],
                window_id,
                measurement,
                window_quality[-1],
                window["kind"],
                edge_type,
            ))
    return {
        "frames": np.asarray(frames, np.int64),
        "edges": edges,
        "predicted_baselines": np.asarray(predicted_baselines, np.float32),
        "initial_log_scales": np.asarray(initial_log_scales, np.float32),
        "window_quality": np.asarray(window_quality, np.float32),
    }


def initialize_poses(measurements):
    """Chain metric-scaled track edges to initialize the global trajectory."""
    frames = measurements["frames"]
    edge_map = {}
    for edge in measurements["edges"]:
        pose_i, pose_j, window_id, measured, quality, kind = edge[:6]
        if kind != "track" or pose_j != pose_i + 1:
            continue
        score = edge_map.get((pose_i, pose_j), (None, -np.inf))[1]
        if quality > score:
            edge_map[(pose_i, pose_j)] = (
                (window_id, measured), quality)

    poses = np.zeros((len(frames), 7), np.float64)
    poses[0, 0] = 1.0
    current = np.eye(4)
    for index in range(len(frames) - 1):
        key = (index, index + 1)
        if key not in edge_map:
            raise ValueError(
                f"no sequential visual edge for frames "
                f"{frames[index]}->{frames[index + 1]}")
        window_id, measured = edge_map[key][0]
        transform = pose_to_matrix(measured)
        transform[:3, 3] *= np.exp(
            measurements["initial_log_scales"][window_id])
        current = transform @ current
        poses[index + 1] = matrix_to_pose(current)
    return poses


def solve_graph(args, windows, manifest, imu):
    baseline_m = float(manifest["baseline_m"])
    measurements = build_measurements(windows, baseline_m)
    frames = measurements["frames"]
    pose_init = initialize_poses(measurements)
    n_poses = len(frames)
    n_windows = len(windows)
    print(
        f"graph: {n_poses} poses, {n_windows} window scales, "
        f"{len(measurements['edges'])} visual edges")

    edge_i = np.asarray([edge[0] for edge in measurements["edges"]], np.int32)
    edge_j = np.asarray([edge[1] for edge in measurements["edges"]], np.int32)
    edge_window = np.asarray(
        [edge[2] for edge in measurements["edges"]], np.int32)
    edge_pose = np.asarray(
        [edge[3] for edge in measurements["edges"]], np.float32)
    edge_quality = np.asarray(
        [edge[4] for edge in measurements["edges"]], np.float32)
    edge_pair_weight = np.asarray([
        args.window_anchor_weight if edge[6] == "window_anchor" else 1.0
        for edge in measurements["edges"]
    ], np.float32)

    imu_i, imu_j, imu_delta = [], [], []
    for index in range(n_poses - 1):
        delta = compose_imu_rotation(
            imu, int(frames[index]), int(frames[index + 1]))
        if delta is not None:
            imu_i.append(index)
            imu_j.append(index + 1)
            imu_delta.append(delta)
    imu_i = np.asarray(imu_i, np.int32)
    imu_j = np.asarray(imu_j, np.int32)
    imu_delta = np.asarray(imu_delta, np.float32)
    print(f"graph: {len(imu_i)} composed IMU rotation edges")

    imu_frame_to_index = {
        int(frame): index for index, frame in enumerate(imu["frame_idx"])}
    gravity = np.asarray([
        imu["gravity_cam"][imu_frame_to_index[int(frame)]]
        for frame in frames
    ], np.float32)
    imu_indices = np.asarray([
        imu_frame_to_index[int(frame)] for frame in frames
    ], np.int64)
    if "gravity_accel_norm_g" in imu:
        accel_norm_g = np.asarray(
            imu["gravity_accel_norm_g"][imu_indices], np.float32)
        gravity_confidence = np.exp(
            -0.5 * (
                (accel_norm_g - 1.0) / args.gravity_accel_sigma_g
            ) ** 2
        )
    else:
        # Older stage-4 artifacts only contain a linearly tapered confidence.
        gravity_confidence = np.asarray(
            imu["gravity_weight"][imu_indices], np.float32) ** 2
    gravity_residual_weight = (
        args.gravity_weight * np.sqrt(gravity_confidence)
    ).astype(np.float32)
    print(
        "graph: gravity confidence "
        f"median={np.median(gravity_confidence):.3f}, "
        f"<0.1={np.mean(gravity_confidence < 0.1):.1%}")

    poses = jaxls.SE3Var(id=jnp.arange(n_poses))
    scales = WindowLogScaleVar(id=jnp.arange(n_windows))
    costs = [
        jaxls.Cost(
            visual_relative_cost,
            (
                jaxls.SE3Var(id=jnp.asarray(edge_i)),
                jaxls.SE3Var(id=jnp.asarray(edge_j)),
                WindowLogScaleVar(id=jnp.asarray(edge_window)),
                jnp.asarray(edge_pose),
                jnp.asarray(
                    args.visual_translation_weight
                    * edge_quality
                    * edge_pair_weight),
                jnp.asarray(
                    args.visual_rotation_weight
                    * edge_quality
                    * edge_pair_weight),
            ),
        ),
        jaxls.Cost(
            baseline_cost,
            (
                WindowLogScaleVar(id=jnp.arange(n_windows)),
                jnp.asarray(measurements["predicted_baselines"]),
                jnp.asarray(baseline_m),
                jnp.asarray(args.baseline_weight),
            ),
        ),
        jaxls.Cost(
            gravity_cost,
            (
                jaxls.SE3Var(id=jnp.arange(n_poses)),
                jnp.asarray(gravity),
                jnp.asarray(gravity_residual_weight[:, None]),
            ),
        ),
        jaxls.Cost(
            pose_anchor_cost,
            (
                jaxls.SE3Var(id=jnp.asarray(0)),
                jnp.asarray(pose_init[0], np.float32),
                jnp.asarray(args.anchor_weight),
            ),
        ),
    ]
    if len(imu_i):
        costs.append(jaxls.Cost(
            imu_rotation_cost,
            (
                jaxls.SE3Var(id=jnp.asarray(imu_i)),
                jaxls.SE3Var(id=jnp.asarray(imu_j)),
                jnp.asarray(imu_delta),
                jnp.asarray(args.imu_rotation_weight),
            ),
        ))
    if args.constant_velocity_weight > 0 and n_poses >= 3:
        dt = np.diff(frames).astype(np.float32) / manifest["native_fps"]
        costs.append(jaxls.Cost(
            constant_velocity_cost,
            (
                jaxls.SE3Var(id=jnp.arange(n_poses - 2)),
                jaxls.SE3Var(id=jnp.arange(1, n_poses - 1)),
                jaxls.SE3Var(id=jnp.arange(2, n_poses)),
                jnp.asarray(dt[:-1]),
                jnp.asarray(dt[1:]),
                jnp.asarray(args.constant_velocity_weight),
            ),
        ))

    values = jaxls.VarValues.make([
        poses.with_value(jaxlie.SE3(jnp.asarray(pose_init, jnp.float32))),
        scales.with_value(jnp.asarray(
            measurements["initial_log_scales"], jnp.float32)),
    ])
    t0 = time.time()
    problem = jaxls.LeastSquaresProblem(
        costs, [poses, scales]).analyze(schur_elimination="off")
    print(f"analyze: {time.time() - t0:.2f}s")
    initial_cost = float(
        jnp.sum(problem.compute_residual_vector(values) ** 2))
    t0 = time.time()
    solution, summary = problem.solve(
        values,
        linear_solver=args.linear_solver,
        trust_region=jaxls.TrustRegionConfig(),
        termination=jaxls.TerminationConfig(
            max_iterations=args.iterations,
            early_termination=False,
        ),
        return_summary=True,
        verbose=False,
    )
    jax.block_until_ready(solution[jaxls.SE3Var])
    elapsed = time.time() - t0
    final_cost = float(
        jnp.sum(problem.compute_residual_vector(solution) ** 2))
    print(
        f"solve: {elapsed:.2f}s, cost {initial_cost:.6g} -> "
        f"{final_cost:.6g}")
    solved_log_scale = np.asarray(solution[WindowLogScaleVar])
    fitted_baseline = (
        np.exp(solved_log_scale) * measurements["predicted_baselines"])
    baseline_relative_error = fitted_baseline / baseline_m - 1.0
    print(
        "baseline: "
        f"median={np.median(fitted_baseline) * 1000.0:.3f} mm, "
        f"mean_abs_error={np.mean(np.abs(baseline_relative_error)):.4%}, "
        f"max_abs_error={np.max(np.abs(baseline_relative_error)):.4%}")
    return {
        "frame_idx": frames,
        "poses": np.asarray(solution[jaxls.SE3Var].wxyz_xyz),
        "window_log_scale": solved_log_scale,
        "window_log_scale_init": measurements["initial_log_scales"],
        "predicted_baseline": measurements["predicted_baselines"],
        "fitted_baseline": fitted_baseline,
        "baseline_relative_error": baseline_relative_error,
        "initial_cost": initial_cost,
        "final_cost": final_cost,
        "cost_history": np.asarray(summary.cost_history),
        "wall_seconds": elapsed,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording")
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--imu-relative", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--track-only",
        action="store_true",
        help="exclude cached loop windows for the preliminary graph",
    )
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument(
        "--linear-solver",
        choices=("conjugate_gradient", "dense_cholesky", "cholmod"),
        default="conjugate_gradient",
    )
    parser.add_argument("--visual-translation-weight", type=float, default=10.0)
    parser.add_argument("--visual-rotation-weight", type=float, default=10.0)
    parser.add_argument(
        "--window-anchor-weight",
        type=float,
        default=1.0,
        help="relative weight for center-to-frame edges; 0 is adjacent-only",
    )
    parser.add_argument("--baseline-weight", type=float, default=100.0)
    parser.add_argument("--imu-rotation-weight", type=float, default=10.0)
    parser.add_argument("--gravity-weight", type=float, default=1.0)
    parser.add_argument(
        "--gravity-accel-sigma-g",
        type=float,
        default=0.05,
        help="Gaussian taper width for |mean accel|-1g gravity confidence",
    )
    parser.add_argument("--constant-velocity-weight", type=float, default=0.1)
    parser.add_argument("--anchor-weight", type=float, default=1000.0)
    return parser.parse_args()


def main():
    args = parse_args()
    recording = os.path.abspath(args.recording)
    input_dir = args.input_dir or os.path.join(
        recording, "derived", "vggt_omega")
    manifest = json.load(open(os.path.join(input_dir, "manifest.json")))
    windows = load_windows(os.path.join(input_dir, "windows"))
    if args.track_only:
        windows = [window for window in windows if window["kind"] == "track"]
    imu_path = args.imu_relative or os.path.join(
        recording, "derived", "imu_relative.npz")
    if not os.path.exists(imu_path):
        imu_path = os.path.join(recording, "imu_relative.npz")
    imu = np.load(imu_path)
    result = solve_graph(args, windows, manifest, imu)
    native_frame_count = max(
        int(manifest.get("native_frame_count", 0)),
        int(np.max(imu["frame_idx"])) + 1,
    )
    frame_idx, poses = interpolate_poses(
        result["frame_idx"],
        result["poses"],
        manifest["native_fps"],
        target_frame_idx=np.arange(native_frame_count, dtype=np.int64),
    )
    R_stereo = np.asarray(manifest["R_stereo"], np.float64)
    t_stereo = np.asarray(manifest["t_stereo"], np.float64)
    right_poses = compose_stereo_poses(poses, R_stereo, t_stereo)
    keyframe_right_poses = compose_stereo_poses(
        result["poses"], R_stereo, t_stereo)

    out_path = args.out or os.path.join(
        recording, "derived", "trajectory_vggt_omega.npz")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(
        out_path,
        frame_idx=frame_idx,
        camera_names=np.asarray(["left", "right"]),
        pose_wxyz_xyz=poses,
        pose_wxyz_xyz_left=poses,
        pose_wxyz_xyz_right=right_poses,
        keyframe_frame_idx=result["frame_idx"],
        keyframe_pose_wxyz_xyz=result["poses"],
        keyframe_pose_wxyz_xyz_left=result["poses"],
        keyframe_pose_wxyz_xyz_right=keyframe_right_poses,
        R_stereo=R_stereo,
        t_stereo=t_stereo,
        baseline_m=np.asarray(manifest["baseline_m"]),
        points=np.zeros((0, 3), np.float32),
        point_alive=np.zeros(0, bool),
        window_log_scale=result["window_log_scale"],
        window_log_scale_init=result["window_log_scale_init"],
        predicted_baseline=result["predicted_baseline"],
        fitted_baseline=result["fitted_baseline"],
        baseline_relative_error=result["baseline_relative_error"],
        graph_initial_cost=np.asarray(result["initial_cost"]),
        graph_final_cost=np.asarray(result["final_cost"]),
        graph_cost_history=result["cost_history"],
        wall_seconds=np.asarray(result["wall_seconds"]),
    )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
