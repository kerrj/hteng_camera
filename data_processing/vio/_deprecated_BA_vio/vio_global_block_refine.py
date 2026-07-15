"""Persistent global block-coordinate VIO refinement and random initialization.

Unlike window stitching, one pose and landmark state persists for the entire
recording. Each update jointly optimizes a contiguous pose block and a sampled
landmark block while every other variable remains fixed. Factors crossing the
block boundary are retained:

  * selected landmarks contribute observations from fixed poses outside the
    pose block;
  * local poses observe only robustly triangulated fixed global landmarks;
  * IMU rotation and translation-smoothness factors cross block boundaries.

Landmarks are first reduced to the union of sliding per-frame active sets.
Selection balances temporal continuity, track length, stereo support, and
round-robin image-grid coverage. Stereo observations are sampled as atomic
left/right pairs. Random initialization first supports a frozen-rotation
positioning phase, then optional free-SE3 refinement. Epochs move temporal
block boundaries, retriangulate retained landmarks, and save checkpoints.
"""

import argparse
import json
import os
import time

import jax
import jax_dataclasses as jdc
import jax.numpy as jnp
import jaxlie
import jaxls
import numpy as np

from vio_bundle_adjust import (
    CamCenterVar,
    Point3Var,
    load_intrinsics,
    load_stereo,
    load_tracks,
    unproject_all,
)


def recording_product(recording, filename):
    derived = os.path.join(recording, "derived", filename)
    return derived if os.path.exists(derived) else os.path.join(
        recording, filename)


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


def poses_to_centers(poses):
    rotation = quat_to_matrix(poses[:, :4])
    return -np.einsum("nji,nj->ni", rotation, poses[:, 4:])


def centers_to_poses(quats, centers):
    rotation = quat_to_matrix(quats)
    translation = -np.einsum("nij,nj->ni", rotation, centers)
    return np.concatenate([quats, translation], axis=1)


def interpolate_poses(poses, sample_pos, output_size):
    """Interpolate sparse world-to-camera poses onto the native frame grid."""
    sample_pos = np.asarray(sample_pos)
    if len(sample_pos) == output_size:
        return poses
    centers = poses_to_centers(poses)
    output_pos = np.arange(output_size)
    center_out = np.stack([
        np.interp(output_pos, sample_pos, centers[:, axis])
        for axis in range(3)
    ], axis=1)

    quats = np.asarray(poses[:, :4], dtype=np.float64).copy()
    quats /= np.maximum(np.linalg.norm(quats, axis=1, keepdims=True), 1e-12)
    for index in range(1, len(quats)):
        if np.dot(quats[index - 1], quats[index]) < 0:
            quats[index] *= -1

    segment = np.searchsorted(sample_pos, output_pos, side="right") - 1
    segment = np.clip(segment, 0, len(sample_pos) - 2)
    span = sample_pos[segment + 1] - sample_pos[segment]
    alpha = (output_pos - sample_pos[segment]) / np.maximum(span, 1)
    q0, q1 = quats[segment], quats[segment + 1]
    dot = np.clip(np.sum(q0 * q1, axis=1), -1.0, 1.0)
    angle = np.arccos(dot)
    sin_angle = np.sin(angle)
    near = np.abs(sin_angle) < 1e-7
    denominator = np.where(near, 1.0, sin_angle)
    weight0 = np.where(
        near, 1.0 - alpha,
        np.sin((1.0 - alpha) * angle) / denominator)
    weight1 = np.where(
        near, alpha, np.sin(alpha * angle) / denominator)
    quat_out = weight0[:, None] * q0 + weight1[:, None] * q1
    quat_out /= np.maximum(
        np.linalg.norm(quat_out, axis=1, keepdims=True), 1e-12)
    return centers_to_poses(quat_out, center_out)


def time_spread_rows(rows, pose_ids, is_right, cap):
    """Select temporally spread pose groups without splitting stereo pairs."""
    rows = np.asarray(rows, dtype=np.int64)
    if len(rows) <= cap:
        return rows
    groups = []
    for pose in np.unique(pose_ids[rows]):
        pose_rows = rows[pose_ids[rows] == pose]
        left = pose_rows[~is_right[pose_rows]]
        right = pose_rows[is_right[pose_rows]]
        group = []
        if len(left):
            group.append(int(left[0]))
        if len(right):
            group.append(int(right[0]))
        if group:
            groups.append(np.asarray(group, np.int64))

    def spread(group_indices, count):
        group_indices = np.asarray(group_indices, np.int64)
        if count <= 0:
            return []
        if len(group_indices) <= count:
            return group_indices.tolist()
        frames = np.asarray([
            pose_ids[groups[index][0]] for index in group_indices])
        chosen = [0, len(group_indices) - 1]
        while len(chosen) < count:
            distance = np.min(
                np.abs(frames[:, None] - frames[np.asarray(chosen)][None]),
                axis=1,
            )
            distance[np.asarray(chosen)] = -1
            chosen.append(int(np.argmax(distance)))
        return group_indices[np.sort(chosen[:count])].tolist()

    stereo = [index for index, group in enumerate(groups) if len(group) == 2]
    mono = [index for index, group in enumerate(groups) if len(group) == 1]
    selected = spread(stereo, min(len(stereo), cap // 2))
    used = 2 * len(selected)
    selected.extend(spread(mono, min(len(mono), cap - used)))
    selected = sorted(selected, key=lambda index: pose_ids[groups[index][0]])
    return np.concatenate([groups[index] for index in selected])


def _group_bounds(ids, size):
    order = np.argsort(ids, kind="stable")
    bounds = np.searchsorted(ids[order], np.arange(size + 1))
    return order, bounds


def _track_statistics(point_ids, pose_ids, is_right, n_points):
    order, bounds = _group_bounds(point_ids, n_points)
    frames = []
    length = np.zeros(n_points, np.int32)
    span = np.zeros(n_points, np.int32)
    stereo = np.zeros(n_points, np.int32)
    for point in range(n_points):
        rows = order[bounds[point]:bounds[point + 1]]
        unique = np.unique(pose_ids[rows])
        frames.append(unique)
        length[point] = len(unique)
        if len(unique):
            span[point] = int(unique[-1] - unique[0] + 1)
        for pose in unique:
            eyes = is_right[rows[pose_ids[rows] == pose]]
            stereo[point] += bool(np.any(eyes) and np.any(~eyes))
    return {
        "frames": frames,
        "length": length,
        "span": span,
        "stereo": stereo,
    }


def _representative_rows(rows, point_ids, is_right):
    """One row per point, preferring the left-eye observation."""
    by_point = {}
    for row in rows:
        point = int(point_ids[row])
        if point not in by_point or (
                is_right[by_point[point]] and not is_right[row]):
            by_point[point] = int(row)
    points = np.fromiter(by_point.keys(), dtype=np.int64)
    representative = np.fromiter(by_point.values(), dtype=np.int64)
    return points, representative


def select_active_track_union(pose_ids, point_ids, px, is_right, n_poses,
                              n_points, active_size, grid_shape,
                              temporal_radius, min_track_frames,
                              quality_fraction):
    """Construct sliding spatially balanced active sets and return their union."""
    stats = _track_statistics(point_ids, pose_ids, is_right, n_points)
    pose_lo = np.searchsorted(pose_ids, np.arange(n_poses + 1))
    width = max(float(np.max(px[:, 0]) + 1), 1.0)
    height = max(float(np.max(px[:, 1]) + 1), 1.0)
    grid_x, grid_y = grid_shape
    selected_union = set()
    previous = set()
    active_counts = np.zeros(n_poses, np.int32)

    for pose in range(n_poses):
        rows = np.arange(pose_lo[pose], pose_lo[pose + 1])
        candidates, representative = _representative_rows(
            rows, point_ids, is_right)
        eligible = stats["length"][candidates] >= min_track_frames
        candidates = candidates[eligible]
        representative = representative[eligible]
        if not len(candidates):
            previous = set()
            continue

        gx = np.clip(
            (grid_x * px[representative, 0] / width).astype(int),
            0, grid_x - 1)
        gy = np.clip(
            (grid_y * px[representative, 1] / height).astype(int),
            0, grid_y - 1)
        cell = gx + grid_x * gy
        score_by_point = {}
        for cell_id in range(grid_x * grid_y):
            local = candidates[cell == cell_id]

            def score(point):
                track_frames = stats["frames"][point]
                insertion = np.searchsorted(track_frames, pose)
                has_prev = (
                    insertion > 0
                    and track_frames[insertion - 1] == pose - 1)
                has_next = (
                    insertion + 1 < len(track_frames)
                    and track_frames[insertion + 1] == pose + 1)
                lo = np.searchsorted(
                    track_frames, pose - temporal_radius, side="left")
                hi = np.searchsorted(
                    track_frames, pose + temporal_radius, side="right")
                return (
                    point in previous,
                    has_prev + has_next,
                    hi - lo,
                    int(stats["length"][point]),
                    int(stats["span"][point]),
                    int(stats["stereo"][point]),
                    -int(point),
                )

            for point in local:
                score_by_point[int(point)] = score(int(point))

        # Most slots preserve persistent, high-quality tracks. The remainder
        # greedily fills underrepresented cells. A strict equal quota per cell
        # causes severe churn when sparse peripheral cells have only weak,
        # short-lived tracks.
        quality_slots = int(round(active_size * quality_fraction))
        quality_slots = np.clip(quality_slots, 0, active_size)
        quality_order = sorted(
            candidates.tolist(),
            key=lambda point: score_by_point[int(point)],
            reverse=True,
        )
        active = quality_order[:quality_slots]
        active_set = set(active)
        cell_by_point = {
            int(point): int(cell_id)
            for point, cell_id in zip(candidates, cell)
        }
        cell_count = np.zeros(grid_x * grid_y, np.int32)
        for point in active:
            cell_count[cell_by_point[int(point)]] += 1
        remaining_by_cell = [[] for _ in range(grid_x * grid_y)]
        for point in quality_order:
            point = int(point)
            if point not in active_set:
                remaining_by_cell[cell_by_point[point]].append(point)
        cell_offset = np.zeros(grid_x * grid_y, np.int32)
        while len(active) < active_size:
            available_cells = [
                cell_id for cell_id, group in enumerate(remaining_by_cell)
                if cell_offset[cell_id] < len(group)
            ]
            if not available_cells:
                break
            best_cell = max(
                available_cells,
                key=lambda cell_id: (
                    -int(cell_count[cell_id]),
                    score_by_point[
                        remaining_by_cell[cell_id][cell_offset[cell_id]]],
                ),
            )
            point = remaining_by_cell[best_cell][cell_offset[best_cell]]
            cell_offset[best_cell] += 1
            active.append(point)
            active_set.add(point)
            cell_count[best_cell] += 1
        previous = set(active)
        selected_union.update(active)
        active_counts[pose] = len(active)

    selected = np.asarray(sorted(selected_union), dtype=np.int64)
    selected_mask = np.zeros(n_points, bool)
    selected_mask[selected] = True
    selected_per_pose = np.zeros(n_poses, np.int32)
    for pose in range(n_poses):
        rows = slice(pose_lo[pose], pose_lo[pose + 1])
        selected_per_pose[pose] = np.unique(
            point_ids[rows][selected_mask[point_ids[rows]]]).size
    adjacent = np.zeros(max(n_poses - 1, 0), np.int32)
    selected_frames = [
        np.unique(point_ids[pose_lo[pose]:pose_lo[pose + 1]][
            selected_mask[
                point_ids[pose_lo[pose]:pose_lo[pose + 1]]]])
        for pose in range(n_poses)
    ]
    for pose in range(n_poses - 1):
        adjacent[pose] = np.intersect1d(
            selected_frames[pose], selected_frames[pose + 1],
            assume_unique=True).size
    print(
        f"active-set union: {len(selected)}/{n_points} tracks; "
        f"active/frame p10/p50/min="
        f"{np.percentile(active_counts, 10):.0f}/"
        f"{np.median(active_counts):.0f}/{active_counts.min()}; "
        f"retained/frame p10/p50={np.percentile(selected_per_pose, 10):.0f}/"
        f"{np.median(selected_per_pose):.0f}")
    if len(adjacent):
        weak = int(np.argmin(adjacent))
        print(
            f"selected adjacent support p10/p50/min="
            f"{np.percentile(adjacent, 10):.0f}/{np.median(adjacent):.0f}/"
            f"{adjacent[weak]} at solver edge {weak}->{weak + 1}")
    return selected, stats, active_counts, selected_per_pose, adjacent


def load_solver_observations(recording, solver_frames, device, active_size,
                             grid_shape, temporal_radius, min_track_frames,
                             quality_fraction, tracks_path=None):
    """Load solver-grid observations, select active tracks, and unproject."""
    frame_to_pose = {
        int(frame): pose for pose, frame in enumerate(solver_frames)}
    tracks = load_tracks(
        tracks_path or recording_product(recording, "tracks.jsonl"),
        int(solver_frames[-1]),
    )
    pose_id, source_track_id, px, right = [], [], [], []
    for track_id, observations in enumerate(tracks):
        for eye, frame, pixel in observations:
            pose = frame_to_pose.get(int(frame))
            if pose is None:
                continue
            pose_id.append(pose)
            source_track_id.append(track_id)
            px.append(pixel)
            right.append(eye == "right")
    pose_id = np.asarray(pose_id, np.int32)
    source_track_id = np.asarray(source_track_id, np.int32)
    px = np.asarray(px, np.float32)
    right = np.asarray(right, bool)
    order = np.argsort(pose_id, kind="stable")
    pose_id, source_track_id, px, right = (
        pose_id[order], source_track_id[order], px[order], right[order])
    print(
        f"solver-grid candidates: {len(tracks)} tracks, "
        f"{len(pose_id)} observations")

    selected, source_stats, active_counts, retained_counts, adjacent = (
        select_active_track_union(
            pose_id, source_track_id, px, right, len(solver_frames),
            len(tracks), active_size, grid_shape, temporal_radius,
            min_track_frames, quality_fraction))
    source_to_point = np.full(len(tracks), -1, np.int32)
    source_to_point[selected] = np.arange(len(selected), dtype=np.int32)
    keep = source_to_point[source_track_id] >= 0
    pose_id, px, right = pose_id[keep], px[keep], right[keep]
    point_id = source_to_point[source_track_id[keep]]
    source_track_id = source_track_id[keep]

    import h5py
    with h5py.File(
            recording_product(recording, "features.h5"), "r") as features:
        left_serial = features.attrs["left_serial"]
        right_serial = features.attrs["right_serial"]
    left_cam = load_intrinsics(recording, left_serial)
    right_cam = load_intrinsics(recording, right_serial)
    stereo_rotation, stereo_translation = load_stereo(
        recording, left_serial, right_serial)
    rays = unproject_all(
        {"left": px[~right], "right": px[right]},
        {"left": left_cam, "right": right_cam},
        device,
    )
    ray_cam = np.zeros((len(pose_id), 3), np.float32)
    ray_cam[~right] = rays["left"]
    ray_cam[right] = rays["right"]
    rel_left = np.asarray(jaxlie.SE3.identity().wxyz_xyz, np.float32)
    rel_right = np.asarray(jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3.from_matrix(stereo_rotation),
        stereo_translation,
    ).wxyz_xyz, np.float32)
    rel = np.where(right[:, None], rel_right[None], rel_left[None])

    point_order, point_bounds = _group_bounds(point_id, len(selected))
    pose_lo = np.searchsorted(pose_id, np.arange(len(solver_frames) + 1))
    point_pose_min = np.zeros(len(selected), np.int32)
    point_pose_max = np.zeros(len(selected), np.int32)
    track_length = np.zeros(len(selected), np.int32)
    track_span = np.zeros(len(selected), np.int32)
    track_stereo = np.zeros(len(selected), np.int32)
    track_stereo_parallax = np.zeros(len(selected), np.float32)
    for point, source_track in enumerate(selected):
        rows = point_order[point_bounds[point]:point_bounds[point + 1]]
        frames = np.unique(pose_id[rows])
        point_pose_min[point] = frames[0]
        point_pose_max[point] = frames[-1]
        track_length[point] = source_stats["length"][source_track]
        track_span[point] = source_stats["span"][source_track]
        track_stereo[point] = source_stats["stereo"][source_track]
        parallax = []
        for pose in frames:
            pose_rows = rows[pose_id[rows] == pose]
            left_rows = pose_rows[~right[pose_rows]]
            right_rows = pose_rows[right[pose_rows]]
            if not len(left_rows) or not len(right_rows):
                continue
            left_ray = ray_cam[left_rows[0]]
            right_ray_left = stereo_rotation.T @ ray_cam[right_rows[0]]
            cosine = np.clip(left_ray @ right_ray_left, -1.0, 1.0)
            parallax.append(np.degrees(np.arccos(cosine)))
        if parallax:
            track_stereo_parallax[point] = np.percentile(parallax, 75)

    # Deterministic 20% validation split. Tracks with too few training rows
    # keep all observations so every retained point remains triangulatable.
    validation = (
        (source_track_id.astype(np.int64) * 73856093
         + pose_id.astype(np.int64) * 19349663
         + right.astype(np.int64) * 83492791) % 5 == 0)
    for point in range(len(selected)):
        rows = point_order[point_bounds[point]:point_bounds[point + 1]]
        if np.sum(~validation[rows]) < 2:
            validation[rows] = False
    return {
        "pose_id": pose_id,
        "point_id": point_id,
        "source_track_id": source_track_id,
        "px": px,
        "right": right,
        "ray_cam": ray_cam,
        "rel": rel,
        "point_order": point_order,
        "point_bounds": point_bounds,
        "pose_lo": pose_lo,
        "point_pose_min": point_pose_min,
        "point_pose_max": point_pose_max,
        "track_length": track_length,
        "track_span": track_span,
        "track_stereo": track_stereo,
        "track_stereo_parallax_deg": track_stereo_parallax,
        "validation": validation,
        "selected_source_track_id": selected,
        "active_counts": active_counts,
        "retained_counts": retained_counts,
        "adjacent_support": adjacent,
    }


def load_solver_imu(recording, solver_frames):
    """Compose native IMU edges into relative rotations on the solver grid."""
    raw = np.load(recording_product(recording, "imu_relative.npz"))
    raw_frames = raw["frame_idx"]
    raw_lookup = {int(frame): index for index, frame in enumerate(raw_frames)}
    source = np.asarray([raw_lookup[int(frame)] for frame in solver_frames])

    identity = np.array([1.0, 0.0, 0.0, 0.0], np.float64)
    chain = [jaxlie.SO3(jnp.asarray(identity))]
    for edge in range(len(raw_frames) - 1):
        delta = (
            raw["rel_quat"][edge]
            if raw["rel_valid"][edge]
            else identity)
        chain.append(jaxlie.SO3(jnp.asarray(delta)).inverse() @ chain[-1])
    chain_quat = np.asarray([np.asarray(rotation.wxyz) for rotation in chain])
    solver_chain = chain_quat[source]
    delta_prev = np.tile(identity, (len(solver_frames), 1))
    if len(solver_frames) > 1:
        delta_prev[1:] = np.asarray(jax.vmap(
            lambda rotation_i, rotation_j: (
                jaxlie.SO3(rotation_i)
                @ jaxlie.SO3(rotation_j).inverse()).wxyz
        )(jnp.asarray(solver_chain[:-1]), jnp.asarray(solver_chain[1:])))
    return {
        "delta_prev": delta_prev.astype(np.float32),
        "gravity_cam": raw["gravity_cam"][source].astype(np.float32),
        "gravity_weight": raw["gravity_weight"][source].astype(np.float32),
    }


def imu_initial_quats(imu):
    measured_down = np.asarray(imu["gravity_cam"][0], np.float64)
    measured_down /= np.linalg.norm(measured_down) + 1e-12
    world_down = np.array([0.0, 0.0, -1.0])
    quat0 = np.array([
        1.0 + world_down @ measured_down,
        *np.cross(world_down, measured_down),
    ])
    if np.linalg.norm(quat0) < 1e-6:
        quat0 = np.array([0.0, 1.0, 0.0, 0.0])
    rotations = [jaxlie.SO3(jnp.asarray(
        quat0 / np.linalg.norm(quat0), jnp.float32))]
    for delta in imu["delta_prev"][1:]:
        rotations.append(jaxlie.SO3(jnp.asarray(delta)).inverse() @ rotations[-1])
    return np.asarray([np.asarray(rotation.wxyz) for rotation in rotations])


def rotation_imu_cost(vals, rotation_i, rotation_j, delta, weight):
    actual = vals[rotation_i] @ vals[rotation_j].inverse()
    return (jaxlie.SO3(delta).inverse() @ actual).log() * weight


def rotation_gravity_cost(vals, rotation, measured_down, weight):
    predicted = vals[rotation] @ jnp.array([0.0, 0.0, -1.0])
    return (predicted - measured_down) * weight


def global_rotation_bootstrap(quats, imu, imu_weight, gravity_weight,
                              iterations):
    """Correct the integrated IMU chain using all per-frame gravity factors."""
    n_poses = len(quats)
    rotation_vars = jaxls.SO3Var(id=jnp.arange(n_poses))
    problem = jaxls.LeastSquaresProblem(
        [
            jaxls.Cost(
                rotation_imu_cost,
                (
                    jaxls.SO3Var(id=jnp.arange(n_poses - 1)),
                    jaxls.SO3Var(id=jnp.arange(1, n_poses)),
                    jnp.asarray(imu["delta_prev"][1:]),
                    jnp.asarray(imu_weight, jnp.float32),
                ),
            ),
            jaxls.Cost(
                rotation_gravity_cost,
                (
                    jaxls.SO3Var(id=jnp.arange(n_poses)),
                    jnp.asarray(imu["gravity_cam"]),
                    jnp.asarray(
                        gravity_weight * imu["gravity_weight"][:, None],
                        jnp.float32,
                    ),
                ),
            ),
        ],
        [rotation_vars],
    ).analyze(schur_elimination="off")
    values = jaxls.VarValues.make([
        rotation_vars.with_value(jaxlie.SO3(jnp.asarray(
            quats, jnp.float32))),
    ])
    initial = jnp.sum(problem.compute_residual_vector(values) ** 2)
    cg = jaxls.ConjugateGradientConfig(
        tolerance_min=1e-6,
        tolerance_max=1e-2,
        preconditioner="block_jacobi",
    )
    solve_jit = jax.jit(lambda problem, values: problem.solve(
        values,
        linear_solver=cg,
        sparse_mode="blockrow",
        trust_region=jaxls.TrustRegionConfig(),
        termination=jaxls.TerminationConfig(
            max_iterations=iterations,
            early_termination=True,
        ),
        verbose=False,
        return_summary=True,
    ))
    solve_start = time.time()
    solution, summary = solve_jit(problem, values)
    jax.block_until_ready(solution[jaxls.SO3Var])
    final = jnp.sum(problem.compute_residual_vector(solution) ** 2)
    initial, final = float(initial), float(final)
    accepted = np.isfinite(final) and final <= initial
    print(
        f"rotation bootstrap: accepted={accepted}, "
        f"cost ratio={final / max(initial, 1e-12):.6f}, "
        f"LM iterations={int(summary.iterations) + 1}, "
        f"wall={time.time() - solve_start:.1f}s")
    output = (
        np.asarray(solution[jaxls.SO3Var].wxyz)
        if accepted else quats)
    return output, np.asarray(
        [initial, final, int(summary.iterations) + 1], np.float64)


def linear_stereo_bootstrap(quats, obs, constant_velocity_weight,
                            robust_scale, irls_iterations, lsmr_max_iterations,
                            lsmr_tolerance, initial_centers=None):
    """Initialize all centers and points from one sparse linear stereo solve.

    Once camera rotations are fixed by the IMU chain, each bearing contributes
    two linear equations in its left-camera center and landmark. Stereo camera
    offsets provide metric scale. LSMR selects a minimum-norm representative
    of the remaining global-translation gauge; the result is recentered later.
    """
    from scipy import sparse
    from scipy.sparse.linalg import lsmr

    training_rows = np.flatnonzero(~obs["validation"])
    pose_id = obs["pose_id"][training_rows]
    point_id = obs["point_id"][training_rows]
    ray_cam = obs["ray_cam"][training_rows].astype(np.float64)
    rel = obs["rel"][training_rows].astype(np.float64)

    left_rotation = quat_to_matrix(quats).astype(np.float64)
    rel_rotation = quat_to_matrix(rel[:, :4])
    camera_rotation = np.einsum(
        "nij,njk->nik", rel_rotation, left_rotation[pose_id])
    ray_world = np.einsum("nji,nj->ni", camera_rotation, ray_cam)
    ray_world /= np.maximum(
        np.linalg.norm(ray_world, axis=1, keepdims=True), 1e-12)
    eye_offset = -np.einsum(
        "nji,nj->ni", camera_rotation, rel[:, 4:])

    # Build a stable orthonormal tangent basis for each measured ray.
    axis_id = np.argmin(np.abs(ray_world), axis=1)
    axis = np.eye(3)[axis_id]
    basis0 = np.cross(ray_world, axis)
    basis0 /= np.maximum(
        np.linalg.norm(basis0, axis=1, keepdims=True), 1e-12)
    basis1 = np.cross(ray_world, basis0)
    basis = np.stack([basis0, basis1], axis=1)

    n_poses = len(quats)
    n_points = len(obs["selected_source_track_id"])
    n_obs = len(training_rows)
    n_visual_rows = 2 * n_obs
    n_velocity_rows = 3 * max(n_poses - 2, 0)
    n_variables = 3 * (n_poses + n_points)

    visual_row = np.repeat(np.arange(n_visual_rows), 3)
    dimension = np.tile(np.arange(3), n_visual_rows)
    expanded_pose = np.repeat(np.repeat(pose_id, 2), 3)
    expanded_point = np.repeat(np.repeat(point_id, 2), 3)
    center_col = 3 * expanded_pose + dimension
    point_col = 3 * n_poses + 3 * expanded_point + dimension
    basis_flat = basis.reshape(-1)
    visual_rhs = np.einsum("nai,ni->na", basis, eye_offset).reshape(-1)

    velocity_row = np.arange(n_velocity_rows) + n_visual_rows
    if n_velocity_rows:
        velocity_pose = np.repeat(np.arange(1, n_poses - 1), 3)
        velocity_dimension = np.tile(np.arange(3), n_poses - 2)
        velocity_cols = [
            3 * (velocity_pose + offset) + velocity_dimension
            for offset in (-1, 0, 1)
        ]
    else:
        velocity_cols = [np.zeros(0, np.int64)] * 3

    row = np.concatenate([
        visual_row,
        visual_row,
        velocity_row,
        velocity_row,
        velocity_row,
    ])
    col = np.concatenate([
        center_col,
        point_col,
        *velocity_cols,
    ])
    rhs = np.concatenate([
        visual_rhs,
        np.zeros(n_velocity_rows, np.float64),
    ])
    solution = np.zeros(n_variables, np.float64)
    diagnostics = []

    def residual_and_weight(centers, points):
        vector = points[point_id] - centers[pose_id] - eye_offset
        depth = np.einsum("ni,ni->n", vector, ray_world)
        perpendicular = np.linalg.norm(
            np.einsum("nai,ni->na", basis, vector), axis=1)
        angular_residual = perpendicular / np.maximum(np.abs(depth), 0.1)
        robust_weight = 1.0 / (
            1.0 + (angular_residual / robust_scale) ** 2)
        weight = (
            np.sqrt(robust_weight)
            / np.clip(np.abs(depth), 0.25, 20.0))
        return angular_residual, weight

    if initial_centers is None:
        observation_weight = np.ones(n_obs, np.float64)
    else:
        initial_centers = np.asarray(initial_centers, np.float64)
        if initial_centers.shape != (n_poses, 3):
            raise ValueError(
                "linear bootstrap initial centers do not match solver grid")

        # Triangulate each merged track independently against the trusted
        # center trajectory, then robustify before the first global solve.
        projection = np.einsum("nai,naj->nij", basis, basis)
        camera_center = initial_centers[pose_id] + eye_offset
        point_normal = np.zeros((n_points, 3, 3), np.float64)
        point_rhs = np.zeros((n_points, 3), np.float64)
        np.add.at(point_normal, point_id, projection)
        np.add.at(
            point_rhs,
            point_id,
            np.einsum("nij,nj->ni", projection, camera_center),
        )
        point_normal += 1e-9 * np.eye(3)[None]
        initial_points = np.linalg.solve(
            point_normal, point_rhs[..., None])[..., 0]

        # A loop edge can merge two otherwise-clean tracks. The all-row
        # least-squares point then falls between both surfaces and gives IRLS
        # no useful inlier basin. Seed each point from stereo depth hypotheses
        # and retain the candidate supported by the most observations.
        right = obs["right"][training_rows]
        point_order, point_bounds = _group_bounds(point_id, n_points)
        n_hypothesis_points = 0
        for point in range(n_points):
            rows = point_order[
                point_bounds[point]:point_bounds[point + 1]]
            if len(rows) < 2:
                continue
            stereo_pairs = []
            for pose in np.unique(pose_id[rows]):
                pose_rows = rows[pose_id[rows] == pose]
                left_rows = pose_rows[~right[pose_rows]]
                right_rows = pose_rows[right[pose_rows]]
                if len(left_rows) and len(right_rows):
                    stereo_pairs.append(
                        (int(left_rows[0]), int(right_rows[0])))
            if not stereo_pairs:
                continue
            if len(stereo_pairs) > 24:
                keep = np.rint(np.linspace(
                    0, len(stereo_pairs) - 1, 24)).astype(np.int64)
                stereo_pairs = [stereo_pairs[index] for index in keep]

            candidates = [initial_points[point]]
            for left_row, right_row in stereo_pairs:
                ray_left = ray_world[left_row]
                ray_right = ray_world[right_row]
                cosine = float(ray_left @ ray_right)
                denominator = 1.0 - cosine * cosine
                if denominator < 1e-8:
                    continue
                baseline = (
                    camera_center[right_row] - camera_center[left_row])
                left_rhs = float(ray_left @ baseline)
                right_rhs = float(ray_right @ baseline)
                left_depth = (
                    left_rhs - cosine * right_rhs) / denominator
                right_depth = (
                    cosine * left_rhs - right_rhs) / denominator
                if left_depth <= 0.0 or right_depth <= 0.0:
                    continue
                candidates.append(0.5 * (
                    camera_center[left_row] + left_depth * ray_left
                    + camera_center[right_row] + right_depth * ray_right))
            if len(candidates) == 1:
                continue

            score_rows = rows
            if len(score_rows) > 96:
                keep = np.rint(np.linspace(
                    0, len(score_rows) - 1, 96)).astype(np.int64)
                score_rows = score_rows[keep]
            candidates = np.asarray(candidates)
            direction = (
                candidates[:, None, :] - camera_center[score_rows][None])
            direction /= np.maximum(
                np.linalg.norm(direction, axis=2, keepdims=True), 1e-12)
            forward = np.einsum(
                "cni,ni->cn", direction, ray_world[score_rows])
            residual = np.where(
                forward > 0.0,
                np.sqrt(np.maximum(1.0 - forward * forward, 0.0)),
                1.0,
            )
            inliers = np.sum(residual < 2.0 * robust_scale, axis=1)
            clipped_cost = np.sum(np.minimum(
                (residual / robust_scale) ** 2, 4.0), axis=1)
            best = np.lexsort((clipped_cost, -inliers))[0]
            initial_points[point] = candidates[best]
            n_hypothesis_points += 1
        print(
            "linear bootstrap stereo hypotheses: "
            f"{n_hypothesis_points}/{n_points} points seeded")

        solution[:3 * n_poses] = initial_centers.reshape(-1)
        solution[3 * n_poses:] = initial_points.reshape(-1)
        angular_residual, observation_weight = residual_and_weight(
            initial_centers, initial_points)
        print(
            "linear bootstrap robust warm start: angular p50/p90="
            f"{np.degrees(np.median(angular_residual)):.3f}/"
            f"{np.degrees(np.percentile(angular_residual, 90)):.3f} deg")

    for iteration in range(irls_iterations):
        visual_weight = np.repeat(observation_weight, 2)
        data = np.concatenate([
            -basis_flat * np.repeat(visual_weight, 3),
            basis_flat * np.repeat(visual_weight, 3),
            np.full(
                n_velocity_rows, constant_velocity_weight, np.float64),
            np.full(
                n_velocity_rows, -2.0 * constant_velocity_weight, np.float64),
            np.full(
                n_velocity_rows, constant_velocity_weight, np.float64),
        ])
        weighted_rhs = rhs.copy()
        weighted_rhs[:n_visual_rows] *= visual_weight
        column_norm = np.sqrt(np.maximum(
            np.bincount(
                col, weights=data * data, minlength=n_variables),
            1e-12,
        ))
        column_scale = 1.0 / column_norm
        matrix = sparse.coo_matrix(
            (data * column_scale[col], (row, col)),
            shape=(n_visual_rows + n_velocity_rows, n_variables),
        ).tocsr()
        result = lsmr(
            matrix,
            weighted_rhs,
            atol=lsmr_tolerance,
            btol=lsmr_tolerance,
            maxiter=lsmr_max_iterations,
            x0=solution / column_scale,
        )
        solution = column_scale * result[0]
        centers = solution[:3 * n_poses].reshape(n_poses, 3)
        points = solution[3 * n_poses:].reshape(n_points, 3)

        angular_residual, observation_weight = residual_and_weight(
            centers, points)
        diagnostics.append([
            result[1],
            result[2],
            result[3],
            result[6],
            np.median(angular_residual),
            np.percentile(angular_residual, 90),
        ])
        print(
            f"linear bootstrap {iteration + 1}/{irls_iterations}: "
            f"lsmr stop={result[1]} iters={result[2]} "
            f"residual={result[3]:.4g} cond={result[6]:.3g}; "
            f"angular p50/p90="
            f"{np.degrees(np.median(angular_residual)):.3f}/"
            f"{np.degrees(np.percentile(angular_residual, 90)):.3f} deg")

    poses = centers_to_poses(quats, centers)
    return poses, points, np.asarray(diagnostics, np.float64)


def world_rays(poses, obs):
    left_rotation = quat_to_matrix(poses[:, :4])
    left_centers = poses_to_centers(poses)
    rel_rotation = quat_to_matrix(obs["rel"][:, :4])
    camera_rotation = np.einsum(
        "nij,njk->nik", rel_rotation, left_rotation[obs["pose_id"]])
    centers = (
        left_centers[obs["pose_id"]]
        - np.einsum("nji,nj->ni", camera_rotation, obs["rel"][:, 4:]))
    rays = np.einsum("nji,nj->ni", camera_rotation, obs["ray_cam"])
    rays /= np.maximum(np.linalg.norm(rays, axis=1, keepdims=True), 1e-12)
    return centers, rays


def retriangulate(poses, points, obs, robust_scale, max_obs_per_point,
                  max_med_angle, min_positive_fraction):
    """Robust independent landmark update conditioned on all current poses."""
    centers, rays = world_rays(poses, obs)
    output = points.copy()
    median_angle = np.full(len(points), np.inf)
    positive_fraction = np.zeros(len(points))
    conditioned = np.zeros(len(points), bool)
    identity = np.eye(3)
    training = ~obs["validation"]

    for point in range(len(points)):
        rows = obs["point_order"][
            obs["point_bounds"][point]:obs["point_bounds"][point + 1]]
        rows = rows[training[rows]]
        if len(rows) < 2:
            continue
        rows = time_spread_rows(
            rows, obs["pose_id"], obs["right"], max_obs_per_point)
        point_centers, point_rays = centers[rows], rays[rows]
        projection = (
            identity[None]
            - point_rays[:, :, None] * point_rays[:, None, :])
        normal = np.sum(projection, axis=0)
        rhs = np.einsum("nij,nj->i", projection, point_centers)
        eigenvalues = np.linalg.eigvalsh(normal)
        if eigenvalues[-1] <= 1e-9 or (
                eigenvalues[0] / eigenvalues[-1] <= 1e-6):
            continue
        x = np.linalg.solve(normal, rhs)

        valid_system = False
        for _ in range(8):
            vector = x[None] - point_centers
            distance = np.maximum(np.linalg.norm(vector, axis=1), 1e-6)
            predicted = vector / distance[:, None]
            forward = np.einsum("ni,ni->n", predicted, point_rays)
            residual = np.sqrt(np.maximum(1.0 - forward ** 2, 0.0))
            residual = np.where(forward > 0, residual, 1.0)
            robust = 1.0 / (1.0 + (residual / robust_scale) ** 2)
            weight = robust / np.clip(distance, 0.1, 20.0) ** 2
            normal = np.einsum("n,nij->ij", weight, projection)
            rhs = np.einsum(
                "n,nij,nj->i", weight, projection, point_centers)
            eigenvalues = np.linalg.eigvalsh(normal)
            valid_system = (
                eigenvalues[-1] > 1e-9
                and eigenvalues[0] / eigenvalues[-1] > 1e-5)
            if not valid_system:
                break
            updated = np.linalg.solve(normal, rhs)
            if np.linalg.norm(updated - x) < 1e-6:
                x = updated
                break
            x = updated

        all_rows = obs["point_order"][
            obs["point_bounds"][point]:obs["point_bounds"][point + 1]]
        vector = x[None] - centers[all_rows]
        distance = np.maximum(np.linalg.norm(vector, axis=1), 1e-12)
        predicted = vector / distance[:, None]
        cosine = np.einsum("ni,ni->n", predicted, rays[all_rows])
        angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
        median_angle[point] = np.median(angle)
        positive_fraction[point] = np.mean(cosine > 0)
        conditioned[point] = valid_system
        if valid_system:
            output[point] = x

    alive = (
        conditioned
        & (median_angle <= max_med_angle)
        & (positive_fraction >= min_positive_fraction))
    finite_angles = median_angle[np.isfinite(median_angle)]
    print(
        f"retriangulation: {alive.sum()}/{len(points)} pass; "
        f"median angle "
        f"{np.median(finite_angles) if len(finite_angles) else np.inf:.3f} deg")
    return output, alive, median_angle, positive_fraction


def angular_metrics(poses, points, obs, rows):
    centers, rays = world_rays(poses, obs)
    vector = points[obs["point_id"][rows]] - centers[rows]
    vector /= np.maximum(np.linalg.norm(vector, axis=1, keepdims=True), 1e-12)
    cosine = np.einsum("ni,ni->n", vector, rays[rows])
    angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    return {
        "median": float(np.median(angle)),
        "p90": float(np.percentile(angle, 90)),
        "positive": float(np.mean(cosine > 0)),
    }


def _robust_bearing(ray_world, direction, robust_scale):
    direction = direction / (jnp.linalg.norm(direction) + 1e-9)
    forward = jnp.maximum(jnp.dot(ray_world, direction), 0.0)
    residual = ray_world - forward * direction
    norm = jnp.linalg.norm(residual) + 1e-9
    weight = jax.lax.stop_gradient(
        1.0 / (1.0 + (norm / robust_scale) ** 2))
    return residual * jnp.sqrt(weight)


def bearing_variable_pose_point(vals, pose_var, point_var, ray, rel, weight,
                                robust_scale):
    left_pose = vals[pose_var]
    left_rotation = left_pose.rotation()
    left_center = -(left_rotation.inverse() @ left_pose.translation())
    relative = jaxlie.SE3(rel)
    camera_rotation = relative.rotation() @ left_rotation
    camera_center = (
        left_center - camera_rotation.inverse() @ relative.translation())
    ray_world = camera_rotation.inverse() @ ray
    residual = _robust_bearing(
        ray_world, vals[point_var] - camera_center, robust_scale)
    return residual * weight


def bearing_fixed_pose_point(vals, fixed_pose, point_var, ray, rel, weight,
                             robust_scale):
    left_pose = jaxlie.SE3(fixed_pose)
    left_rotation = left_pose.rotation()
    left_center = -(left_rotation.inverse() @ left_pose.translation())
    relative = jaxlie.SE3(rel)
    camera_rotation = relative.rotation() @ left_rotation
    camera_center = (
        left_center - camera_rotation.inverse() @ relative.translation())
    ray_world = camera_rotation.inverse() @ ray
    residual = _robust_bearing(
        ray_world, vals[point_var] - camera_center, robust_scale)
    return residual * weight


def bearing_pose_fixed_point(vals, pose_var, fixed_point, ray, rel, weight,
                             robust_scale):
    left_pose = vals[pose_var]
    left_rotation = left_pose.rotation()
    left_center = -(left_rotation.inverse() @ left_pose.translation())
    relative = jaxlie.SE3(rel)
    camera_rotation = relative.rotation() @ left_rotation
    camera_center = (
        left_center - camera_rotation.inverse() @ relative.translation())
    ray_world = camera_rotation.inverse() @ ray
    residual = _robust_bearing(
        ray_world, fixed_point - camera_center, robust_scale)
    return residual * weight


def _fixed_rotation_camera(fixed_quat, left_center, ray, rel):
    relative = jaxlie.SE3(rel)
    camera_rotation = relative.rotation() @ jaxlie.SO3(fixed_quat)
    camera_center = (
        left_center - camera_rotation.inverse() @ relative.translation())
    ray_world = camera_rotation.inverse() @ ray
    return camera_center, ray_world


def bearing_variable_center_point(vals, center_var, point_var, fixed_quat,
                                  ray, rel, weight, robust_scale):
    camera_center, ray_world = _fixed_rotation_camera(
        fixed_quat, vals[center_var], ray, rel)
    return _robust_bearing(
        ray_world, vals[point_var] - camera_center, robust_scale) * weight


def bearing_fixed_center_point(vals, fixed_center, point_var, fixed_quat,
                               ray, rel, weight, robust_scale):
    camera_center, ray_world = _fixed_rotation_camera(
        fixed_quat, fixed_center, ray, rel)
    return _robust_bearing(
        ray_world, vals[point_var] - camera_center, robust_scale) * weight


def bearing_center_fixed_point(vals, center_var, fixed_point, fixed_quat,
                               ray, rel, weight, robust_scale):
    camera_center, ray_world = _fixed_rotation_camera(
        fixed_quat, vals[center_var], ray, rel)
    return _robust_bearing(
        ray_world, fixed_point - camera_center, robust_scale) * weight


def depth_variable_pose_point(vals, pose_var, point_var, ray, rel, min_depth,
                              softness, weight):
    left_pose = vals[pose_var]
    left_rotation = left_pose.rotation()
    left_center = -(left_rotation.inverse() @ left_pose.translation())
    relative = jaxlie.SE3(rel)
    camera_rotation = relative.rotation() @ left_rotation
    camera_center = (
        left_center - camera_rotation.inverse() @ relative.translation())
    ray_world = camera_rotation.inverse() @ ray
    depth = jnp.dot(ray_world, vals[point_var] - camera_center)
    violation = softness * jax.nn.softplus((min_depth - depth) / softness)
    return violation[None] * weight


def depth_fixed_pose_point(vals, fixed_pose, point_var, ray, rel, min_depth,
                           softness, weight):
    left_pose = jaxlie.SE3(fixed_pose)
    left_rotation = left_pose.rotation()
    left_center = -(left_rotation.inverse() @ left_pose.translation())
    relative = jaxlie.SE3(rel)
    camera_rotation = relative.rotation() @ left_rotation
    camera_center = (
        left_center - camera_rotation.inverse() @ relative.translation())
    ray_world = camera_rotation.inverse() @ ray
    depth = jnp.dot(ray_world, vals[point_var] - camera_center)
    violation = softness * jax.nn.softplus((min_depth - depth) / softness)
    return violation[None] * weight


def depth_pose_fixed_point(vals, pose_var, fixed_point, ray, rel, min_depth,
                           softness, weight):
    left_pose = vals[pose_var]
    left_rotation = left_pose.rotation()
    left_center = -(left_rotation.inverse() @ left_pose.translation())
    relative = jaxlie.SE3(rel)
    camera_rotation = relative.rotation() @ left_rotation
    camera_center = (
        left_center - camera_rotation.inverse() @ relative.translation())
    ray_world = camera_rotation.inverse() @ ray
    depth = jnp.dot(ray_world, fixed_point - camera_center)
    violation = softness * jax.nn.softplus((min_depth - depth) / softness)
    return violation[None] * weight


def depth_variable_center_point(vals, center_var, point_var, fixed_quat, ray,
                                rel, min_depth, softness, weight):
    camera_center, ray_world = _fixed_rotation_camera(
        fixed_quat, vals[center_var], ray, rel)
    depth = jnp.dot(ray_world, vals[point_var] - camera_center)
    violation = softness * jax.nn.softplus((min_depth - depth) / softness)
    return violation[None] * weight


def depth_fixed_center_point(vals, fixed_center, point_var, fixed_quat, ray,
                             rel, min_depth, softness, weight):
    camera_center, ray_world = _fixed_rotation_camera(
        fixed_quat, fixed_center, ray, rel)
    depth = jnp.dot(ray_world, vals[point_var] - camera_center)
    violation = softness * jax.nn.softplus((min_depth - depth) / softness)
    return violation[None] * weight


def depth_center_fixed_point(vals, center_var, fixed_point, fixed_quat, ray,
                             rel, min_depth, softness, weight):
    camera_center, ray_world = _fixed_rotation_camera(
        fixed_quat, vals[center_var], ray, rel)
    depth = jnp.dot(ray_world, fixed_point - camera_center)
    violation = softness * jax.nn.softplus((min_depth - depth) / softness)
    return violation[None] * weight


def imu_cost(vals, pose_i, pose_j, delta, weight):
    actual = vals[pose_i].rotation() @ vals[pose_j].rotation().inverse()
    return (jaxlie.SO3(delta).inverse() @ actual).log() * weight


def imu_left_boundary(vals, pose_j, fixed_i, delta, weight):
    actual = jaxlie.SO3(fixed_i) @ vals[pose_j].rotation().inverse()
    return (jaxlie.SO3(delta).inverse() @ actual).log() * weight


def imu_right_boundary(vals, pose_i, fixed_j, delta, weight):
    actual = vals[pose_i].rotation() @ jaxlie.SO3(fixed_j).inverse()
    return (jaxlie.SO3(delta).inverse() @ actual).log() * weight


def gravity_cost(vals, pose_var, measured_down, weight):
    predicted = vals[pose_var].rotation() @ jnp.array([0.0, 0.0, -1.0])
    return (predicted - measured_down) * weight


def constant_velocity_cost(vals, pose_prev, pose, pose_next, weight):
    def center(variable):
        value = vals[variable]
        return -(value.rotation().inverse() @ value.translation())

    return (
        center(pose_prev) - 2.0 * center(pose) + center(pose_next)
    ) * weight


def constant_velocity_left_boundary(vals, fixed_prev, pose, pose_next,
                                    weight):
    fixed_prev = jaxlie.SE3(fixed_prev)
    fixed_center = -(
        fixed_prev.rotation().inverse() @ fixed_prev.translation())
    pose_value = vals[pose]
    next_value = vals[pose_next]
    center = -(
        pose_value.rotation().inverse() @ pose_value.translation())
    next_center = -(
        next_value.rotation().inverse() @ next_value.translation())
    return (fixed_center - 2.0 * center + next_center) * weight


def constant_velocity_right_boundary(vals, pose_prev, pose, fixed_next,
                                     weight):
    prev_value = vals[pose_prev]
    pose_value = vals[pose]
    fixed_next = jaxlie.SE3(fixed_next)
    prev_center = -(
        prev_value.rotation().inverse() @ prev_value.translation())
    center = -(
        pose_value.rotation().inverse() @ pose_value.translation())
    fixed_center = -(
        fixed_next.rotation().inverse() @ fixed_next.translation())
    return (prev_center - 2.0 * center + fixed_center) * weight


def center_constant_velocity(vals, center_prev, center, center_next, weight):
    return (
        vals[center_prev] - 2.0 * vals[center] + vals[center_next]
    ) * weight


def center_velocity_left_boundary(vals, fixed_prev, center, center_next,
                                  weight):
    return (
        fixed_prev - 2.0 * vals[center] + vals[center_next]
    ) * weight


def center_velocity_right_boundary(vals, center_prev, center, fixed_next,
                                   weight):
    return (
        vals[center_prev] - 2.0 * vals[center] + fixed_next
    ) * weight


def point_padding_cost(vals, point_var, weight):
    return vals[point_var] * weight


def _pad_rows(rows, cap):
    rows = np.asarray(rows, np.int64)
    if len(rows) > cap:
        rows = rows[:cap]
    if not len(rows):
        return np.zeros(cap, np.int64), np.zeros(cap, np.float32), 0
    padded = np.pad(rows, (0, cap - len(rows)), mode="edge")
    weight = np.zeros(cap, np.float32)
    weight[:len(rows)] = 1.0
    return padded, weight, len(rows)


def _pad_pose_point_groups(rows, obs, cap, rng):
    """Shuffle/cap observations without splitting a stereo pose-point group."""
    grouped = {}
    for row in np.asarray(rows, np.int64):
        key = (int(obs["pose_id"][row]), int(obs["point_id"][row]))
        grouped.setdefault(key, []).append(int(row))
    groups = []
    for group in grouped.values():
        group = np.asarray(group, np.int64)
        order = np.argsort(obs["right"][group], kind="stable")
        groups.append(group[order])
    rng.shuffle(groups)
    selected = []
    count = 0
    for group in groups:
        if count + len(group) > cap:
            continue
        selected.append(group)
        count += len(group)
        if count == cap:
            break
    rows = (
        np.concatenate(selected)
        if selected else np.zeros(0, np.int64))
    return _pad_rows(rows, cap)


def _stereo_pair_count(rows, obs):
    eyes = {}
    for row in np.asarray(rows, np.int64):
        key = (int(obs["pose_id"][row]), int(obs["point_id"][row]))
        eyes.setdefault(key, set()).add(bool(obs["right"][row]))
    return sum(len(group_eyes) == 2 for group_eyes in eyes.values())


def select_fixed_pose_rows(obs, alive, variable_points, start, length,
                           per_frame, grid_shape):
    selected = []
    variable_mask = np.zeros(len(alive), bool)
    variable_mask[variable_points] = True
    grid_x, grid_y = grid_shape
    width = max(float(np.max(obs["px"][:, 0]) + 1), 1.0)
    height = max(float(np.max(obs["px"][:, 1]) + 1), 1.0)
    for pose in range(start, start + length):
        rows = np.arange(obs["pose_lo"][pose], obs["pose_lo"][pose + 1])
        keep = alive[obs["point_id"][rows]]
        keep &= ~variable_mask[obs["point_id"][rows]]
        keep &= ~obs["validation"][rows]
        rows = rows[keep]
        candidates, representative = _representative_rows(
            rows, obs["point_id"], obs["right"])
        if len(candidates) <= per_frame:
            kept_points = candidates
        else:
            gx = np.clip(
                (grid_x * obs["px"][representative, 0] / width).astype(int),
                0, grid_x - 1)
            gy = np.clip(
                (grid_y * obs["px"][representative, 1] / height).astype(int),
                0, grid_y - 1)
            cells = gx + grid_x * gy
            groups = []
            for cell in range(grid_x * grid_y):
                group = candidates[cells == cell]
                crossing = (
                    (obs["point_pose_min"][group] < start)
                    | (obs["point_pose_max"][group] >= start + length))
                groups.append(np.concatenate(
                    [group[crossing], group[~crossing]]))
            kept = []
            depth = 0
            while len(kept) < per_frame:
                added = False
                for group in groups:
                    if depth < len(group):
                        kept.append(int(group[depth]))
                        added = True
                        if len(kept) == per_frame:
                            break
                if not added:
                    break
                depth += 1
            kept_points = np.asarray(kept, np.int64)

        for point in kept_points:
            point_rows = rows[obs["point_id"][rows] == point]
            left = point_rows[~obs["right"][point_rows]]
            right = point_rows[obs["right"][point_rows]]
            if len(left):
                selected.append(int(left[0]))
            if len(right):
                selected.append(int(right[0]))
    return np.asarray(selected, np.int64)


def sample_block_points(obs, alive, start, length, landmark_count, rng,
                        random_fraction, stereo_parallax_weight):
    local_rows = np.arange(
        obs["pose_lo"][start], obs["pose_lo"][start + length])
    candidates = np.unique(obs["point_id"][local_rows])
    candidates = candidates[alive[candidates]]
    if len(candidates) <= landmark_count:
        return candidates
    crossing = (
        (obs["point_pose_min"][candidates] < start)
        | (obs["point_pose_max"][candidates] >= start + length))
    quality = (
        obs["track_length"][candidates].astype(np.float64)
        + 0.25 * obs["track_span"][candidates]
        + 2.0 * obs["track_stereo"][candidates]
        + stereo_parallax_weight
        * obs["track_stereo_parallax_deg"][candidates])
    deterministic_count = int(round(
        landmark_count * (1.0 - random_fraction)))
    deterministic_count = np.clip(deterministic_count, 0, landmark_count)
    deterministic_order = np.lexsort((-quality, ~crossing))
    fixed = candidates[deterministic_order[:deterministic_count]]
    remaining = np.setdiff1d(candidates, fixed, assume_unique=False)
    take = landmark_count - len(fixed)
    if take:
        remaining_quality = (
            obs["track_length"][remaining].astype(np.float64)
            + 0.25 * obs["track_span"][remaining]
            + 2.0 * obs["track_stereo"][remaining]
            + stereo_parallax_weight
            * obs["track_stereo_parallax_deg"][remaining])
        probability = remaining_quality / np.sum(remaining_quality)
        sampled = rng.choice(
            remaining, size=take, replace=False, p=probability)
        return np.concatenate([fixed, sampled])
    return fixed


def make_block_problem(poses, points, alive, obs, imu, start, args, rng,
                       analyze=True, fixed_alive=None):
    if fixed_alive is None:
        fixed_alive = alive
    length = args.block_frames
    variable_points = sample_block_points(
        obs, alive, start, length, args.landmarks_per_block, rng,
        args.landmark_random_fraction, args.stereo_parallax_weight)
    n_variable_points = len(variable_points)
    point_to_local = np.full(len(points), -1, np.int32)
    point_to_local[variable_points] = np.arange(
        n_variable_points, dtype=np.int32)

    variable_rows = []
    for point in variable_points:
        rows = obs["point_order"][
            obs["point_bounds"][point]:obs["point_bounds"][point + 1]]
        rows = rows[~obs["validation"][rows]]
        variable_rows.append(time_spread_rows(
            rows, obs["pose_id"], obs["right"], args.obs_per_landmark))
    variable_rows = (
        np.concatenate(variable_rows)
        if variable_rows else np.zeros(0, np.int64))
    inside_mask = (
        (obs["pose_id"][variable_rows] >= start)
        & (obs["pose_id"][variable_rows] < start + length))
    inside_rows = variable_rows[inside_mask]
    outside_rows = variable_rows[~inside_mask]
    inside_pad, inside_weight, n_inside = _pad_pose_point_groups(
        inside_rows, obs, args.inside_obs_cap, rng)
    outside_pad, outside_weight, n_outside = _pad_pose_point_groups(
        outside_rows, obs, args.outside_obs_cap, rng)

    fixed_rows = select_fixed_pose_rows(
        obs, fixed_alive, variable_points, start, length,
        args.fixed_obs_per_frame, tuple(args.spatial_grid))
    fixed_cap = args.fixed_obs_cap
    fixed_pad, fixed_weight, n_fixed = _pad_rows(fixed_rows, fixed_cap)

    pose_vars = jaxls.SE3Var(id=jnp.arange(length))
    point_vars = Point3Var(id=jnp.arange(args.landmarks_per_block))
    inside_local_pose = np.clip(
        obs["pose_id"][inside_pad] - start, 0, length - 1)
    fixed_local_pose = np.clip(
        obs["pose_id"][fixed_pad] - start, 0, length - 1)
    inside_local_point = np.maximum(
        point_to_local[obs["point_id"][inside_pad]], 0)
    outside_local_point = np.maximum(
        point_to_local[obs["point_id"][outside_pad]], 0)
    costs = [
        jaxls.Cost(
            bearing_variable_pose_point,
            (
                jaxls.SE3Var(id=jnp.asarray(
                    inside_local_pose)),
                Point3Var(id=jnp.asarray(inside_local_point)),
                jnp.asarray(obs["ray_cam"][inside_pad]),
                jnp.asarray(obs["rel"][inside_pad]),
                jnp.asarray(inside_weight),
                jnp.asarray(args.robust_scale, jnp.float32),
            ),
        ),
        jaxls.Cost(
            bearing_fixed_pose_point,
            (
                jnp.asarray(poses[obs["pose_id"][outside_pad]], jnp.float32),
                Point3Var(id=jnp.asarray(outside_local_point)),
                jnp.asarray(obs["ray_cam"][outside_pad]),
                jnp.asarray(obs["rel"][outside_pad]),
                jnp.asarray(outside_weight),
                jnp.asarray(args.robust_scale, jnp.float32),
            ),
        ),
        jaxls.Cost(
            bearing_pose_fixed_point,
            (
                jaxls.SE3Var(id=jnp.asarray(
                    fixed_local_pose)),
                jnp.asarray(points[obs["point_id"][fixed_pad]], jnp.float32),
                jnp.asarray(obs["ray_cam"][fixed_pad]),
                jnp.asarray(obs["rel"][fixed_pad]),
                jnp.asarray(fixed_weight),
                jnp.asarray(args.robust_scale, jnp.float32),
            ),
        ),
        jaxls.Cost(
            depth_variable_pose_point,
            (
                jaxls.SE3Var(id=jnp.asarray(
                    inside_local_pose)),
                Point3Var(id=jnp.asarray(inside_local_point)),
                jnp.asarray(obs["ray_cam"][inside_pad]),
                jnp.asarray(obs["rel"][inside_pad]),
                jnp.asarray(args.positive_depth_min, jnp.float32),
                jnp.asarray(args.positive_depth_softness, jnp.float32),
                jnp.asarray(
                    inside_weight * args.positive_depth_weight,
                    jnp.float32),
            ),
        ),
        jaxls.Cost(
            depth_fixed_pose_point,
            (
                jnp.asarray(poses[obs["pose_id"][outside_pad]], jnp.float32),
                Point3Var(id=jnp.asarray(outside_local_point)),
                jnp.asarray(obs["ray_cam"][outside_pad]),
                jnp.asarray(obs["rel"][outside_pad]),
                jnp.asarray(args.positive_depth_min, jnp.float32),
                jnp.asarray(args.positive_depth_softness, jnp.float32),
                jnp.asarray(
                    outside_weight * args.positive_depth_weight,
                    jnp.float32),
            ),
        ),
        jaxls.Cost(
            depth_pose_fixed_point,
            (
                jaxls.SE3Var(id=jnp.asarray(
                    fixed_local_pose)),
                jnp.asarray(points[obs["point_id"][fixed_pad]], jnp.float32),
                jnp.asarray(obs["ray_cam"][fixed_pad]),
                jnp.asarray(obs["rel"][fixed_pad]),
                jnp.asarray(args.positive_depth_min, jnp.float32),
                jnp.asarray(args.positive_depth_softness, jnp.float32),
                jnp.asarray(
                    fixed_weight * args.positive_depth_weight,
                    jnp.float32),
            ),
        ),
        jaxls.Cost(
            imu_cost,
            (
                jaxls.SE3Var(id=jnp.arange(length - 1)),
                jaxls.SE3Var(id=jnp.arange(1, length)),
                jnp.asarray(imu["delta_prev"][start + 1:start + length]),
                jnp.asarray(args.imu_rot_weight, jnp.float32),
            ),
        ),
        jaxls.Cost(
            gravity_cost,
            (
                jaxls.SE3Var(id=jnp.arange(length)),
                jnp.asarray(imu["gravity_cam"][start:start + length]),
                jnp.asarray(
                    args.gravity_weight
                    * imu["gravity_weight"][start:start + length, None],
                    jnp.float32),
            ),
        ),
        jaxls.Cost(
            constant_velocity_cost,
            (
                jaxls.SE3Var(id=jnp.arange(length - 2)),
                jaxls.SE3Var(id=jnp.arange(1, length - 1)),
                jaxls.SE3Var(id=jnp.arange(2, length)),
                jnp.asarray(args.constant_velocity_weight, jnp.float32),
            ),
        ),
        jaxls.Cost(
            constant_velocity_left_boundary,
            (
                jnp.asarray(poses[max(start - 1, 0)], jnp.float32),
                jaxls.SE3Var(id=jnp.asarray(0)),
                jaxls.SE3Var(id=jnp.asarray(1)),
                jnp.asarray(
                    args.constant_velocity_weight if start > 0 else 0.0,
                    jnp.float32),
            ),
        ),
        jaxls.Cost(
            constant_velocity_right_boundary,
            (
                jaxls.SE3Var(id=jnp.asarray(length - 2)),
                jaxls.SE3Var(id=jnp.asarray(length - 1)),
                jnp.asarray(
                    poses[min(start + length, len(poses) - 1)],
                    jnp.float32),
                jnp.asarray(
                    args.constant_velocity_weight
                    if start + length < len(poses) else 0.0,
                    jnp.float32),
            ),
        ),
        jaxls.Cost(
            imu_left_boundary,
            (
                jaxls.SE3Var(id=jnp.asarray(0)),
                jnp.asarray(
                    poses[max(start - 1, 0), :4], jnp.float32),
                jnp.asarray(
                    imu["delta_prev"][start]
                    if start > 0 else np.array([1.0, 0.0, 0.0, 0.0]),
                    jnp.float32),
                jnp.asarray(
                    args.imu_rot_weight if start > 0 else 0.0,
                    jnp.float32),
            ),
        ),
        jaxls.Cost(
            imu_right_boundary,
            (
                jaxls.SE3Var(id=jnp.asarray(length - 1)),
                jnp.asarray(
                    poses[min(start + length, len(poses) - 1), :4],
                    jnp.float32),
                jnp.asarray(
                    imu["delta_prev"][start + length]
                    if start + length < len(poses)
                    else np.array([1.0, 0.0, 0.0, 0.0]),
                    jnp.float32),
                jnp.asarray(
                    args.imu_rot_weight
                    if start + length < len(poses) else 0.0,
                    jnp.float32),
            ),
        ),
    ]
    padding_weight = (
        np.arange(args.landmarks_per_block) >= n_variable_points
    ).astype(np.float32)
    costs.append(jaxls.Cost(
        point_padding_cost,
        (
            Point3Var(id=jnp.arange(args.landmarks_per_block)),
            jnp.asarray(padding_weight[:, None]),
        ),
    ))
    problem = jaxls.LeastSquaresProblem(costs, [pose_vars, point_vars])
    if analyze:
        problem = problem.analyze(schur_elimination="off")
    point_init = np.zeros((args.landmarks_per_block, 3), np.float32)
    point_init[:n_variable_points] = points[variable_points]
    values = jaxls.VarValues.make([
        pose_vars.with_value(jaxlie.SE3(jnp.asarray(
            poses[start:start + length], jnp.float32))),
        point_vars.with_value(jnp.asarray(point_init)),
    ])
    counts = {
        "inside": n_inside,
        "outside": n_outside,
        "fixed": n_fixed,
        "points": n_variable_points,
        "inside_stereo": _stereo_pair_count(inside_pad[:n_inside], obs),
        "outside_stereo": _stereo_pair_count(outside_pad[:n_outside], obs),
        "fixed_stereo": _stereo_pair_count(fixed_pad[:n_fixed], obs),
        "parallax_p50": float(np.median(
            obs["track_stereo_parallax_deg"][variable_points]))
        if len(variable_points) else 0.0,
    }
    spec = {
        "start": start,
        "variable_points": variable_points,
        "outside_rows": outside_pad,
        "fixed_rows": fixed_pad,
        "counts": counts,
    }
    return problem, values, variable_points, counts, spec


def make_center_block_problem(poses, points, alive, obs, imu, start, args, rng,
                              analyze=True, fixed_alive=None):
    """Build a sampled positioning block with rotations held fixed."""
    sampled, full_values, variable_points, counts, spec = make_block_problem(
        poses, points, alive, obs, imu, start, args, rng, analyze=False,
        fixed_alive=fixed_alive)
    raw = {cost._get_name(): cost for cost in sampled.costs}
    length = args.block_frames

    inside = raw["bearing_variable_pose_point"].args
    outside = raw["bearing_fixed_pose_point"].args
    fixed = raw["bearing_pose_fixed_point"].args
    inside_depth = raw["depth_variable_pose_point"].args
    outside_depth = raw["depth_fixed_pose_point"].args
    fixed_depth = raw["depth_pose_fixed_point"].args
    inside_pose_id = np.asarray(inside[0].id)
    fixed_pose_id = np.asarray(fixed[0].id)
    outside_pose = np.asarray(outside[0])

    center_vars = CamCenterVar(id=jnp.arange(length))
    point_vars = Point3Var(id=jnp.arange(args.landmarks_per_block))
    costs = [
        jaxls.Cost(
            bearing_variable_center_point,
            (
                CamCenterVar(id=jnp.asarray(inside_pose_id)),
                inside[1],
                jnp.asarray(poses[start + inside_pose_id, :4], jnp.float32),
                inside[2],
                inside[3],
                inside[4],
                inside[5],
            ),
        ),
        jaxls.Cost(
            bearing_fixed_center_point,
            (
                jnp.asarray(poses_to_centers(outside_pose), jnp.float32),
                outside[1],
                jnp.asarray(outside_pose[:, :4], jnp.float32),
                outside[2],
                outside[3],
                outside[4],
                outside[5],
            ),
        ),
        jaxls.Cost(
            bearing_center_fixed_point,
            (
                CamCenterVar(id=jnp.asarray(fixed_pose_id)),
                fixed[1],
                jnp.asarray(poses[start + fixed_pose_id, :4], jnp.float32),
                fixed[2],
                fixed[3],
                fixed[4],
                fixed[5],
            ),
        ),
        jaxls.Cost(
            depth_variable_center_point,
            (
                CamCenterVar(id=jnp.asarray(inside_pose_id)),
                inside_depth[1],
                jnp.asarray(poses[start + inside_pose_id, :4], jnp.float32),
                inside_depth[2],
                inside_depth[3],
                inside_depth[4],
                inside_depth[5],
                inside_depth[6],
            ),
        ),
        jaxls.Cost(
            depth_fixed_center_point,
            (
                jnp.asarray(poses_to_centers(outside_pose), jnp.float32),
                outside_depth[1],
                jnp.asarray(outside_pose[:, :4], jnp.float32),
                outside_depth[2],
                outside_depth[3],
                outside_depth[4],
                outside_depth[5],
                outside_depth[6],
            ),
        ),
        jaxls.Cost(
            depth_center_fixed_point,
            (
                CamCenterVar(id=jnp.asarray(fixed_pose_id)),
                fixed_depth[1],
                jnp.asarray(poses[start + fixed_pose_id, :4], jnp.float32),
                fixed_depth[2],
                fixed_depth[3],
                fixed_depth[4],
                fixed_depth[5],
                fixed_depth[6],
            ),
        ),
        jaxls.Cost(
            center_constant_velocity,
            (
                CamCenterVar(id=jnp.arange(length - 2)),
                CamCenterVar(id=jnp.arange(1, length - 1)),
                CamCenterVar(id=jnp.arange(2, length)),
                jnp.asarray(args.constant_velocity_weight, jnp.float32),
            ),
        ),
        jaxls.Cost(
            center_velocity_left_boundary,
            (
                jnp.asarray(
                    poses_to_centers(poses[max(start - 1, 0)][None])[0],
                    jnp.float32),
                CamCenterVar(id=jnp.asarray(0)),
                CamCenterVar(id=jnp.asarray(1)),
                jnp.asarray(
                    args.constant_velocity_weight if start > 0 else 0.0,
                    jnp.float32),
            ),
        ),
        jaxls.Cost(
            center_velocity_right_boundary,
            (
                CamCenterVar(id=jnp.asarray(length - 2)),
                CamCenterVar(id=jnp.asarray(length - 1)),
                jnp.asarray(
                    poses_to_centers(
                        poses[min(start + length, len(poses) - 1)][None])[0],
                    jnp.float32),
                jnp.asarray(
                    args.constant_velocity_weight
                    if start + length < len(poses) else 0.0,
                    jnp.float32),
            ),
        ),
        jaxls.Cost(point_padding_cost, raw["point_padding_cost"].args),
    ]
    problem = jaxls.LeastSquaresProblem(costs, [center_vars, point_vars])
    if analyze:
        problem = problem.analyze(schur_elimination="off")
    values = jaxls.VarValues.make([
        center_vars.with_value(jnp.asarray(
            poses_to_centers(poses[start:start + length]), jnp.float32)),
        point_vars.with_value(full_values[Point3Var]),
    ])
    return problem, values, variable_points, counts, spec


def refresh_dense_template(template, sampled_problem):
    """Refresh fixed-shape factors and their local variable-ID mappings.

    Dense Cholesky consumes jaxls's block-row Jacobian, whose start columns
    come from each analyzed cost. Rebuilding those fixed-shape analyzed costs
    updates the sampled factor incidence without repeating whole-problem
    sparsity analysis or changing the compiled solve shape.
    """
    sampled_by_name = {
        cost._get_name(): cost for cost in sampled_problem.costs}
    if len(sampled_by_name) != len(sampled_problem.costs):
        raise ValueError("block problem contains duplicate cost names")

    updated_costs = []
    for template_cost in template._stacked_costs:
        sampled_cost = sampled_by_name.pop(template_cost._get_name())
        sampled_cost = sampled_cost._broadcast_batch_axes()
        if len(sampled_cost._get_batch_axes()) == 0:
            sampled_cost = jax.tree.map(
                lambda leaf: jnp.asarray(leaf)[None], sampled_cost)
        updated_cost = jax.vmap(type(template_cost)._make)(sampled_cost)
        if jax.tree.structure(updated_cost) != jax.tree.structure(
                template_cost):
            raise ValueError(
                f"cost structure changed for {template_cost._get_name()}")
        updated_costs.append(updated_cost)
    if sampled_by_name:
        raise ValueError(
            f"unmatched sampled costs: {sorted(sampled_by_name)}")

    with jdc.copy_and_mutate(template) as updated_problem:
        updated_problem._stacked_costs = tuple(updated_costs)
    return updated_problem


def solve_block(problem, values, iterations, termination_cost_tol,
                termination_gradient_tol, termination_gradient_start,
                termination_parameter_tol):
    initial = jnp.sum(problem.compute_residual_vector(values) ** 2)
    solution, summary = problem.solve(
        values,
        linear_solver="dense_cholesky",
        sparse_mode="blockrow",
        trust_region=jaxls.TrustRegionConfig(),
        termination=jaxls.TerminationConfig(
            max_iterations=iterations,
            early_termination=True,
            cost_tolerance=termination_cost_tol,
            gradient_tolerance=termination_gradient_tol,
            gradient_tolerance_start_step=termination_gradient_start,
            parameter_tolerance=termination_parameter_tol),
        verbose=False,
        return_summary=True,
    )
    final = jnp.sum(problem.compute_residual_vector(solution) ** 2)
    pose_delta = (
        values[jaxls.SE3Var].inverse() @ solution[jaxls.SE3Var]
    ).log()
    point_delta = solution[Point3Var] - values[Point3Var]
    max_translation = jnp.max(
        jnp.linalg.norm(pose_delta[:, :3], axis=1))
    max_rotation = jnp.max(
        jnp.linalg.norm(pose_delta[:, 3:], axis=1))
    max_point = jnp.max(jnp.linalg.norm(point_delta, axis=1))
    cost_improved = jnp.isfinite(final) & (final <= initial)
    improved = cost_improved
    pose_output = jnp.where(
        improved,
        solution[jaxls.SE3Var].wxyz_xyz,
        values[jaxls.SE3Var].wxyz_xyz,
    )
    point_output = jnp.where(
        improved, solution[Point3Var], values[Point3Var])
    return (
        pose_output,
        point_output,
        initial,
        jnp.where(improved, final, initial),
        improved,
        summary.iterations + 1,
        final,
        max_translation,
        max_rotation,
        max_point,
        cost_improved,
    )


def solve_center_block(problem, values, iterations, termination_cost_tol,
                       termination_gradient_tol, termination_gradient_start,
                       termination_parameter_tol):
    initial = jnp.sum(problem.compute_residual_vector(values) ** 2)
    solution, summary = problem.solve(
        values,
        linear_solver="dense_cholesky",
        sparse_mode="blockrow",
        trust_region=jaxls.TrustRegionConfig(),
        termination=jaxls.TerminationConfig(
            max_iterations=iterations,
            early_termination=True,
            cost_tolerance=termination_cost_tol,
            gradient_tolerance=termination_gradient_tol,
            gradient_tolerance_start_step=termination_gradient_start,
            parameter_tolerance=termination_parameter_tol),
        verbose=False,
        return_summary=True,
    )
    final = jnp.sum(problem.compute_residual_vector(solution) ** 2)
    center_delta = solution[CamCenterVar] - values[CamCenterVar]
    point_delta = solution[Point3Var] - values[Point3Var]
    improved = jnp.isfinite(final) & (final <= initial)
    return (
        jnp.where(improved, solution[CamCenterVar], values[CamCenterVar]),
        jnp.where(improved, solution[Point3Var], values[Point3Var]),
        initial,
        jnp.where(improved, final, initial),
        improved,
        summary.iterations + 1,
        jnp.max(jnp.linalg.norm(center_delta, axis=1)),
        jnp.max(jnp.linalg.norm(point_delta, axis=1)),
    )


def block_starts(n_poses, length, stride):
    starts = list(range(0, max(n_poses - length, 0) + 1, stride))
    if not starts:
        return [0]
    if starts[-1] + length < n_poses:
        starts.append(n_poses - length)
    return starts


def build_dense_template(poses, points, eligible, fixed_alive, obs, imu, start,
                         args):
    """Analyze one representative fixed-shape block problem."""
    template_rng = np.random.default_rng(0)
    template, _, variable_points, counts, _ = make_block_problem(
        poses, points, eligible, obs, imu, start, args, template_rng,
        fixed_alive=fixed_alive)
    print(
        f"  analyzed one dense template: {args.block_frames} poses, "
        f"{args.landmarks_per_block} landmark slots, "
        f"{sum((args.inside_obs_cap, args.outside_obs_cap, args.fixed_obs_cap))} "
        f"visual cost slots; initial sample={len(variable_points)} points, "
        f"obs={counts['inside']}+{counts['outside']}+{counts['fixed']}, "
        f"stereo pairs={counts['inside_stereo']}+"
        f"{counts['outside_stereo']}+{counts['fixed_stereo']}, "
        f"parallax p50={counts['parallax_p50']:.3f}deg")
    return template


def build_center_template(poses, points, eligible, fixed_alive, obs, imu,
                          start, args):
    template_rng = np.random.default_rng(0)
    template, _, variable_points, counts, _ = make_center_block_problem(
        poses, points, eligible, obs, imu, start, args, template_rng,
        fixed_alive=fixed_alive)
    print(
        f"  analyzed one positioning template: {args.block_frames} centers, "
        f"{args.landmarks_per_block} landmark slots; "
        f"initial sample={len(variable_points)} points, "
        f"obs={counts['inside']}+{counts['outside']}+{counts['fixed']}, "
        f"stereo pairs={counts['inside_stereo']}+"
        f"{counts['outside_stereo']}+{counts['fixed_stereo']}, "
        f"parallax p50={counts['parallax_p50']:.3f}deg")
    return template


def run_center_epoch(poses, points, eligible, fixed_alive, obs, imu, starts,
                     template, args, rng, solve_jit):
    output_poses = poses.copy()
    output_points = points.copy()
    order = rng.permutation(len(starts))
    ratios = []
    iterations = []
    accepted_count = 0
    for update_index, start_index in enumerate(order):
        start = starts[start_index]
        sampled_problem, values, variable_points, counts, _ = (
            make_center_block_problem(
                output_poses, output_points, eligible, obs, imu, start, args,
                rng, analyze=False, fixed_alive=fixed_alive))
        problem = refresh_dense_template(template, sampled_problem)
        solve_start = time.time()
        (block_centers, block_points, initial, final, accepted,
         lm_iterations, max_center, max_point) = solve_jit(problem, values)
        jax.block_until_ready(block_centers)
        accepted = bool(accepted)
        ratio = float(final / np.maximum(initial, 1e-12))
        lm_iterations = int(lm_iterations)
        ratios.append(ratio)
        iterations.append(lm_iterations)
        if accepted:
            block_slice = slice(start, start + args.block_frames)
            output_poses[block_slice] = centers_to_poses(
                output_poses[block_slice, :4], np.asarray(block_centers))
            output_points[variable_points] = np.asarray(
                block_points)[:len(variable_points)]
            accepted_count += 1
        sample_id = int(np.sum(
            (variable_points.astype(np.uint64) + 1)
            * np.arange(1, len(variable_points) + 1, dtype=np.uint64)
        ) % np.uint64(1_000_000_007))
        print(
            f"  position block {update_index + 1}/{len(starts)} start={start} "
            f"sample={sample_id:09d} points={counts['points']} obs="
            f"{counts['inside']}+{counts['outside']}+{counts['fixed']} "
            f"stereo_pairs={counts['inside_stereo']}+"
            f"{counts['outside_stereo']}+{counts['fixed_stereo']} "
            f"parallax_p50={counts['parallax_p50']:.3f}deg "
            f"accepted={accepted} ratio={ratio:.4f} "
            f"step_c/p={float(max_center):.3f}/{float(max_point):.3f} "
            f"lm={lm_iterations} time={time.time() - solve_start:.2f}s")
    print(
        f"position epoch blocks: accepted {accepted_count}/{len(starts)}, "
        f"median ratio={np.median(ratios):.4f}, "
        f"LM iterations p50/max={np.median(iterations):.0f}/"
        f"{np.max(iterations)}")
    return (
        output_poses,
        output_points,
        np.asarray(ratios),
        np.asarray(iterations),
        accepted_count,
    )


def run_epoch(poses, points, eligible, fixed_alive, obs, imu, starts, template,
              args, rng, solve_jit):
    output_poses = poses.copy()
    output_points = points.copy()
    order = rng.permutation(len(starts))
    ratios = []
    iterations = []
    accepted_count = 0
    for update_index, start_index in enumerate(order):
        start = starts[start_index]
        sampled_problem, values, variable_points, counts, _ = (
            make_block_problem(
                output_poses, output_points, eligible, obs, imu, start, args,
                rng, analyze=False, fixed_alive=fixed_alive))
        problem = refresh_dense_template(template, sampled_problem)
        solve_start = time.time()
        (block_pose, block_point, initial, final, accepted,
         lm_iterations, raw_final, max_translation, max_rotation, max_point,
         cost_improved) = solve_jit(problem, values)
        jax.block_until_ready(block_pose)
        accepted = bool(accepted)
        ratio = float(final / np.maximum(initial, 1e-12))
        lm_iterations = int(lm_iterations)
        ratios.append(ratio)
        iterations.append(lm_iterations)
        if accepted:
            output_poses[start:start + args.block_frames] = np.asarray(
                block_pose)
            output_points[variable_points] = np.asarray(
                block_point)[:len(variable_points)]
            accepted_count += 1
        sample_id = int(np.sum(
            (variable_points.astype(np.uint64) + 1)
            * np.arange(1, len(variable_points) + 1, dtype=np.uint64)
        ) % np.uint64(1_000_000_007))
        print(
            f"  block {update_index + 1}/{len(starts)} start={start} "
            f"sample={sample_id:09d} points={counts['points']} obs="
            f"{counts['inside']}+{counts['outside']}+{counts['fixed']} "
            f"stereo_pairs={counts['inside_stereo']}+"
            f"{counts['outside_stereo']}+{counts['fixed_stereo']} "
            f"parallax_p50={counts['parallax_p50']:.3f}deg "
            f"accepted={accepted} raw_ratio="
            f"{float(raw_final / np.maximum(initial, 1e-12)):.4f} "
            f"cost_ok={bool(cost_improved)} "
            f"step_t/r/p={float(max_translation):.3f}/"
            f"{np.degrees(float(max_rotation)):.2f}deg/"
            f"{float(max_point):.3f} "
            f"lm={lm_iterations} "
            f"time={time.time() - solve_start:.2f}s")
    print(
        f"epoch blocks: accepted {accepted_count}/{len(starts)}, "
        f"median ratio={np.median(ratios):.4f}, "
        f"LM iterations p50/max={np.median(iterations):.0f}/"
        f"{np.max(iterations)}")
    return (
        output_poses,
        output_points,
        np.asarray(ratios),
        np.asarray(iterations),
        accepted_count,
    )


def output_metadata(obs):
    n_points = len(obs["selected_source_track_id"])
    first_frame = np.zeros(n_points, np.int64)
    first_right = np.zeros(n_points, bool)
    first_px = np.zeros((n_points, 2), np.float32)
    for point in range(n_points):
        rows = obs["point_order"][
            obs["point_bounds"][point]:obs["point_bounds"][point + 1]]
        first = rows[np.argmin(obs["pose_id"][rows])]
        first_frame[point] = obs["pose_id"][first]
        first_right[point] = obs["right"][first]
        first_px[point] = obs["px"][first]
    return first_frame, first_right, first_px


def write_output(path, source, solver_frames, sample_pos, solver_poses, points,
                 alive, median_angle, positive_fraction, obs, metrics,
                 epoch_ratios, config):
    native_poses = interpolate_poses(
        solver_poses, sample_pos, len(source["frame_idx"]))
    first_pose, first_right, first_px = output_metadata(obs)
    payload = {
        "frame_idx": source["frame_idx"],
        "pose_wxyz_xyz": native_poses,
        "solver_frame_idx": solver_frames,
        "global_solver_pose_wxyz_xyz": solver_poses,
        "points": points,
        "point_alive": alive,
        "point_med_ang": median_angle,
        "point_positive_depth_frac": positive_fraction,
        "point_track_id": obs["selected_source_track_id"],
        "point_first_frame": solver_frames[first_pose],
        "point_first_is_right": first_right,
        "point_first_px": first_px,
        "active_count_per_solver_frame": obs["active_counts"],
        "retained_count_per_solver_frame": obs["retained_counts"],
        "active_adjacent_support": obs["adjacent_support"],
        "epoch_block_cost_ratio": np.asarray(epoch_ratios),
        **{f"metric_{key}": np.asarray(value) for key, value in metrics.items()},
        **{f"config_{key}": np.asarray(value) for key, value in config.items()},
    }
    np.savez(path, **payload)
    print(f"wrote {path}")


def write_probe_output(path, frame_idx, poses, points, alive, median_angle,
                       positive_fraction, obs, extra):
    first_pose, first_right, first_px = output_metadata(obs)
    np.savez(
        path,
        frame_idx=frame_idx,
        pose_wxyz_xyz=poses,
        solver_frame_idx=frame_idx,
        global_solver_pose_wxyz_xyz=poses,
        points=points,
        point_alive=alive,
        point_med_ang=median_angle,
        point_positive_depth_frac=positive_fraction,
        point_track_id=obs["selected_source_track_id"],
        point_first_frame=frame_idx[first_pose],
        point_first_is_right=first_right,
        point_first_px=first_px,
        active_count_per_solver_frame=obs["active_counts"],
        retained_count_per_solver_frame=obs["retained_counts"],
        active_adjacent_support=obs["adjacent_support"],
        **{key: np.asarray(value) for key, value in extra.items()},
    )
    print(f"wrote {path}")


def run_random_init_probe(args, source):
    """Solve only the first block from random centers/points and IMU rotations."""
    frame_idx = source["solver_frame_idx"][:args.block_frames]
    obs = load_solver_observations(
        args.recording, frame_idx, args.device, args.active_tracks,
        tuple(args.spatial_grid), args.active_temporal_radius,
        args.min_track_frames, args.active_quality_fraction, args.tracks)
    imu = load_solver_imu(args.recording, frame_idx)
    rng = np.random.default_rng(args.seed)
    quats = imu_initial_quats(imu)
    rotation_diagnostics = np.zeros(3, np.float64)
    if args.rotation_bootstrap:
        print("running global IMU + gravity rotation bootstrap")
        quats, rotation_diagnostics = global_rotation_bootstrap(
            quats,
            imu,
            args.imu_rot_weight,
            args.gravity_weight,
            args.rotation_bootstrap_iters,
        )
    centers = rng.normal(0.0, 0.1, (len(frame_idx), 3))
    poses = centers_to_poses(quats, centers)
    points = rng.normal(
        0.0, 0.5, (len(obs["selected_source_track_id"]), 3))
    points[:, 2] += 1.0
    alive = np.ones(len(points), bool)

    # A standalone random block has no solved global landmark complement yet.
    # Keep the fixed-factor padding shape, but give those rows zero weight.
    fixed_obs_per_frame = args.fixed_obs_per_frame
    args.fixed_obs_per_frame = 0
    problem, values, variable_points, counts, _ = make_block_problem(
        poses, points, alive, obs, imu, 0, args, rng)
    args.fixed_obs_per_frame = fixed_obs_per_frame
    print(
        f"random probe: {len(poses)} poses, {len(points)} retained tracks, "
        f"{len(variable_points)} variable landmarks, obs="
        f"{counts['inside']}+{counts['outside']}+{counts['fixed']}")
    initial_path = os.path.splitext(args.out)[0] + "_init.npz"
    write_probe_output(
        initial_path, frame_idx, poses, points, alive,
        np.full(len(points), np.inf), np.zeros(len(points)), obs,
        {"probe_stage": "random_init"})

    solve_jit = jax.jit(lambda problem, values: solve_block(
        problem,
        values,
        args.block_iters,
        args.termination_cost_tol,
        args.termination_gradient_tol,
        args.termination_gradient_start,
        args.termination_parameter_tol,
    ))
    solve_start = time.time()
    (solved_poses, solved_points, initial_cost, final_cost, accepted,
     lm_iterations, *_) = solve_jit(problem, values)
    jax.block_until_ready(solved_poses)
    accepted = bool(accepted)
    if accepted:
        poses = np.asarray(solved_poses)
        points[variable_points] = np.asarray(
            solved_points)[:len(variable_points)]
    ratio = float(final_cost / np.maximum(initial_cost, 1e-12))
    print(
        f"random probe solve: accepted={accepted}, ratio={ratio:.5f}, "
        f"lm={int(lm_iterations)}, time={time.time() - solve_start:.2f}s")
    points, alive, median_angle, positive_fraction = retriangulate(
        poses, points, obs, args.robust_scale, args.obs_per_landmark,
        args.max_point_med_ang, args.min_positive_depth_frac)
    train = angular_metrics(
        poses, points, obs, np.flatnonzero(~obs["validation"]))
    validation = angular_metrics(
        poses, points, obs, np.flatnonzero(obs["validation"]))
    print(
        f"random probe angular: train p50/p90 "
        f"{train['median']:.3f}/{train['p90']:.3f} deg; validation "
        f"{validation['median']:.3f}/{validation['p90']:.3f} deg")
    write_probe_output(
        args.out, frame_idx, poses, points, alive, median_angle,
        positive_fraction, obs, {
            "probe_stage": "solved",
            "probe_accepted": accepted,
            "probe_cost_ratio": ratio,
            "metric_train_median_deg": train["median"],
            "metric_train_p90_deg": train["p90"],
            "metric_validation_median_deg": validation["median"],
            "metric_validation_p90_deg": validation["p90"],
        })


def evenly_spaced_block_starts(n_poses, length, count):
    if length > n_poses:
        raise ValueError("--block-frames exceeds the available pose count")
    if count <= 1 or length == n_poses:
        return [0]
    starts = np.rint(np.linspace(0, n_poses - length, count)).astype(np.int64)
    return np.unique(starts).tolist()


def stratified_random_block_starts(n_poses, length, count, rng):
    """Sample moving block boundaries while preserving endpoint coverage."""
    max_start = n_poses - length
    if max_start < 0:
        raise ValueError("--block-frames exceeds the available pose count")
    if count <= 1 or max_start == 0:
        return [0]
    if count == 2:
        return [0, max_start]
    edges = np.linspace(0, max_start + 1, count + 1)
    starts = [0]
    for index in range(1, count - 1):
        lo = int(np.ceil(edges[index]))
        hi = int(np.floor(edges[index + 1]))
        hi = max(hi, lo + 1)
        starts.append(int(rng.integers(lo, hi)))
    starts.append(max_start)
    return starts


def recenter_state(poses, points):
    centers = poses_to_centers(poses)
    origin = centers[0].copy()
    return (
        centers_to_poses(poses[:, :4], centers - origin),
        points - origin,
    )


def random_init_frames(recording, source):
    if source is not None and "solver_frame_idx" in source.files:
        return source["solver_frame_idx"]
    imu = np.load(recording_product(recording, "imu_relative.npz"))
    return imu["frame_idx"][imu["frame_valid"]]


def run_random_global(args, source):
    """Run shuffled global block-coordinate epochs from random geometry."""
    frame_idx = random_init_frames(args.recording, source)
    obs = load_solver_observations(
        args.recording, frame_idx, args.device, args.active_tracks,
        tuple(args.spatial_grid), args.active_temporal_radius,
        args.min_track_frames, args.active_quality_fraction, args.tracks)
    imu = load_solver_imu(args.recording, frame_idx)
    rng = np.random.default_rng(args.seed)
    quats = imu_initial_quats(imu)
    rotation_diagnostics = np.zeros(3, np.float64)
    if args.rotation_bootstrap:
        print("running global IMU + gravity rotation bootstrap")
        quats, rotation_diagnostics = global_rotation_bootstrap(
            quats,
            imu,
            args.imu_rot_weight,
            args.gravity_weight,
            args.rotation_bootstrap_iters,
        )
    if args.initial_state:
        initial_state = np.load(args.initial_state)
        if (
                len(initial_state["frame_idx"]) != len(frame_idx)
                or not np.array_equal(initial_state["frame_idx"], frame_idx)):
            raise ValueError(
                "--initial-state frame grid does not match the random run")
        poses = initial_state["pose_wxyz_xyz"].copy()
        points = np.zeros(
            (len(obs["selected_source_track_id"]), 3), np.float64)
        initial_lookup = {
            int(track): point for point, track in enumerate(
                initial_state["point_track_id"])
        }
        matched = np.zeros(len(points), bool)
        for point, track in enumerate(obs["selected_source_track_id"]):
            initial_point = initial_lookup.get(int(track))
            if initial_point is not None:
                points[point] = initial_state["points"][initial_point]
                matched[point] = True
        if not np.all(matched):
            points[~matched] = rng.normal(
                0.0, 0.5, (np.sum(~matched), 3))
            points[~matched, 2] += 1.0
        print(
            f"loaded initial state {args.initial_state}: "
            f"{matched.sum()}/{len(points)} landmarks matched")
    else:
        centers = rng.normal(0.0, 0.1, (len(frame_idx), 3))
        poses = centers_to_poses(quats, centers)
        points = rng.normal(
            0.0, 0.5, (len(obs["selected_source_track_id"]), 3))
        points[:, 2] += 1.0
    eligible = np.ones(len(points), bool)
    if args.initial_state:
        points, alive, median_angle, positive_fraction = retriangulate(
            poses, points, obs, args.robust_scale, args.obs_per_landmark,
            args.max_point_med_ang, args.min_positive_depth_frac)
    else:
        alive = np.zeros(len(points), bool)
        median_angle = np.full(len(points), np.inf)
        positive_fraction = np.zeros(len(points))

    starts = evenly_spaced_block_starts(
        len(poses), args.block_frames, args.blocks_per_epoch)
    if args.max_blocks > 0:
        starts = starts[:args.max_blocks]
    print(
        f"random global state: {len(poses)} poses, {len(points)} landmarks; "
        f"{len(starts)} blocks/epoch, block={args.block_frames} poses, "
        f"landmarks/block={args.landmarks_per_block}")
    solve_jit = jax.jit(lambda problem, values: solve_block(
        problem,
        values,
        args.block_iters,
        args.termination_cost_tol,
        args.termination_gradient_tol,
        args.termination_gradient_start,
        args.termination_parameter_tol,
    ))
    center_solve_jit = jax.jit(lambda problem, values: solve_center_block(
        problem,
        values,
        args.positioning_iters,
        args.termination_cost_tol,
        args.termination_gradient_tol,
        args.termination_gradient_start,
        args.termination_parameter_tol,
    ))
    train_rows = np.flatnonzero(~obs["validation"])
    validation_rows = np.flatnonzero(obs["validation"])
    metric_history = {
        "train_median_deg": [],
        "train_p90_deg": [],
        "validation_median_deg": [],
        "validation_p90_deg": [],
        "accepted_blocks": [],
    }
    ratio_history = []
    iteration_history = []
    stage_history = []
    starts_history = []
    config = {
        "block_frames": args.block_frames,
        "blocks_per_epoch": len(starts),
        "landmarks_per_block": args.landmarks_per_block,
        "landmark_random_fraction": args.landmark_random_fraction,
        "stereo_parallax_weight": args.stereo_parallax_weight,
        "inside_obs_cap": args.inside_obs_cap,
        "outside_obs_cap": args.outside_obs_cap,
        "fixed_obs_cap": args.fixed_obs_cap,
        "fixed_obs_per_frame": args.fixed_obs_per_frame,
        "positioning_epochs": args.positioning_epochs,
        "positioning_iters": args.positioning_iters,
        "se3_epochs": args.epochs,
        "se3_iters": args.block_iters,
        "imu_rotation_weight": args.imu_rot_weight,
        "gravity_weight": args.gravity_weight,
        "constant_velocity_weight": args.constant_velocity_weight,
        "robust_scale": args.robust_scale,
        "linear_bootstrap": args.linear_bootstrap,
        "linear_bootstrap_irls": args.linear_bootstrap_irls,
        "linear_bootstrap_max_iterations":
            args.linear_bootstrap_max_iterations,
        "linear_bootstrap_tolerance": args.linear_bootstrap_tolerance,
        "linear_bootstrap_initial_trajectory":
            args.linear_bootstrap_initial_trajectory or "",
        "rotation_bootstrap": args.rotation_bootstrap,
        "rotation_bootstrap_iters": args.rotation_bootstrap_iters,
        "initial_state": args.initial_state or "",
        "seed": args.seed,
    }

    def record(accepted):
        train = angular_metrics(poses, points, obs, train_rows)
        validation = angular_metrics(
            poses, points, obs, validation_rows)
        metric_history["train_median_deg"].append(train["median"])
        metric_history["train_p90_deg"].append(train["p90"])
        metric_history["validation_median_deg"].append(validation["median"])
        metric_history["validation_p90_deg"].append(validation["p90"])
        metric_history["accepted_blocks"].append(accepted)
        print(
            f"angular metrics: train p50/p90 "
            f"{train['median']:.3f}/{train['p90']:.3f} deg; validation "
            f"{validation['median']:.3f}/{validation['p90']:.3f} deg")

    record(0)
    init_poses, init_points = recenter_state(poses, points)
    initial_path = os.path.splitext(args.out)[0] + "_init.npz"
    write_probe_output(
        initial_path, frame_idx, init_poses, init_points, alive,
        median_angle, positive_fraction, obs, {
            "random_init": True,
            "block_starts": starts,
            **{f"config_{key}": value for key, value in config.items()},
        })

    if args.linear_bootstrap:
        print("running sparse global fixed-rotation stereo bootstrap")
        linear_initial_centers = None
        if args.linear_bootstrap_initial_trajectory:
            linear_initial = np.load(
                args.linear_bootstrap_initial_trajectory)
            if (
                    len(linear_initial["frame_idx"]) != len(frame_idx)
                    or not np.array_equal(
                        linear_initial["frame_idx"], frame_idx)):
                raise ValueError(
                    "--linear-bootstrap-initial-trajectory frame grid "
                    "does not match the random run")
            linear_initial_centers = poses_to_centers(
                linear_initial["pose_wxyz_xyz"])
            print(
                "using linear bootstrap centers from "
                f"{args.linear_bootstrap_initial_trajectory}")
        poses, points, bootstrap_diagnostics = linear_stereo_bootstrap(
            quats,
            obs,
            args.constant_velocity_weight,
            args.robust_scale,
            args.linear_bootstrap_irls,
            args.linear_bootstrap_max_iterations,
            args.linear_bootstrap_tolerance,
            linear_initial_centers,
        )
        points, alive, median_angle, positive_fraction = retriangulate(
            poses, points, obs, args.robust_scale, args.obs_per_landmark,
            args.max_point_med_ang, args.min_positive_depth_frac)
        record(0)
        output_poses, output_points = recenter_state(poses, points)
        bootstrap_path = os.path.splitext(args.out)[0] + "_linear_init.npz"
        write_probe_output(
            bootstrap_path, frame_idx, output_poses, output_points, alive,
            median_angle, positive_fraction, obs, {
                "random_init": True,
                "linear_bootstrap_diagnostics": bootstrap_diagnostics,
                "rotation_bootstrap_diagnostics": rotation_diagnostics,
                "block_starts": starts,
                **{f"config_{key}": value for key, value in config.items()},
                **{f"metric_{key}": value
                   for key, value in metric_history.items()},
            })

    output_root, output_ext = os.path.splitext(args.out)
    output_ext = output_ext or ".npz"
    if args.positioning_epochs:
        print("analyzing one reusable frozen-rotation positioning template")
        center_template = build_center_template(
            poses, points, eligible, alive, obs, imu, starts[0], args)
        for epoch in range(args.positioning_epochs):
            epoch_starts = stratified_random_block_starts(
                len(poses), args.block_frames, len(starts), rng)
            print(
                f"positioning epoch {epoch + 1}/"
                f"{args.positioning_epochs}; starts={epoch_starts}")
            (poses, points, ratios, iterations,
             accepted) = run_center_epoch(
                 poses, points, eligible, alive, obs, imu, epoch_starts,
                 center_template, args, rng, center_solve_jit)
            ratio_history.append(ratios)
            iteration_history.append(iterations)
            stage_history.append("positioning")
            starts_history.append(epoch_starts)
            points, alive, median_angle, positive_fraction = retriangulate(
                poses, points, obs, args.robust_scale,
                args.obs_per_landmark, args.max_point_med_ang,
                args.min_positive_depth_frac)
            record(accepted)
            if (
                    (epoch + 1) % args.checkpoint_every == 0
                    or epoch + 1 == args.positioning_epochs):
                output_poses, output_points = recenter_state(poses, points)
                checkpoint = (
                    f"{output_root}_position_epoch{epoch + 1}{output_ext}")
                write_probe_output(
                    checkpoint, frame_idx, output_poses, output_points, alive,
                    median_angle, positive_fraction, obs, {
                        "random_init": True,
                        "block_starts": starts,
                        "epoch_block_starts": starts_history,
                        "epoch_stage": stage_history,
                        "epoch_block_cost_ratio": ratio_history,
                        "epoch_block_lm_iterations": iteration_history,
                        **{f"config_{key}": value
                           for key, value in config.items()},
                        **{f"metric_{key}": value
                           for key, value in metric_history.items()},
                    })

    if args.epochs:
        print("analyzing one reusable dense SE3 block template")
        template = build_dense_template(
            poses, points, eligible, alive, obs, imu, starts[0], args)
        for epoch in range(args.epochs):
            epoch_starts = stratified_random_block_starts(
                len(poses), args.block_frames, len(starts), rng)
            print(
                f"random epoch {epoch + 1}/{args.epochs}; "
                f"starts={epoch_starts}")
            (poses, points, ratios, iterations,
             accepted) = run_epoch(
                 poses, points, eligible, alive, obs, imu, epoch_starts,
                 template, args, rng, solve_jit)
            ratio_history.append(ratios)
            iteration_history.append(iterations)
            stage_history.append("se3")
            starts_history.append(epoch_starts)
            points, alive, median_angle, positive_fraction = retriangulate(
                poses, points, obs, args.robust_scale, args.obs_per_landmark,
                args.max_point_med_ang, args.min_positive_depth_frac)
            record(accepted)
            if (
                    (epoch + 1) % args.checkpoint_every == 0
                    or epoch + 1 == args.epochs):
                output_poses, output_points = recenter_state(poses, points)
                checkpoint = f"{output_root}_epoch{epoch + 1}{output_ext}"
                write_probe_output(
                    checkpoint, frame_idx, output_poses, output_points, alive,
                    median_angle, positive_fraction, obs, {
                        "random_init": True,
                        "block_starts": starts,
                        "epoch_block_starts": starts_history,
                        "epoch_stage": stage_history,
                        "epoch_block_cost_ratio": ratio_history,
                        "epoch_block_lm_iterations": iteration_history,
                        **{f"config_{key}": value
                           for key, value in config.items()},
                        **{f"metric_{key}": value
                           for key, value in metric_history.items()},
                    })
    output_poses, output_points = recenter_state(poses, points)
    write_probe_output(
        args.out, frame_idx, output_poses, output_points, alive,
        median_angle, positive_fraction, obs, {
            "random_init": True,
            "block_starts": starts,
            "epoch_block_starts": starts_history,
            "epoch_stage": stage_history,
            "epoch_block_cost_ratio": ratio_history,
            "epoch_block_lm_iterations": iteration_history,
            **{f"config_{key}": value for key, value in config.items()},
            **{f"metric_{key}": value
               for key, value in metric_history.items()},
        })


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("recording")
    parser.add_argument("--trajectory", default=None,
                        help="required for refinement; optional frame-grid "
                             "source for --random-init")
    parser.add_argument("--tracks", default=None,
                        help="default: recording's derived/tracks.jsonl")
    parser.add_argument("--out", required=True)
    parser.add_argument("--active-tracks", type=int, default=100)
    parser.add_argument("--spatial-grid", type=int, nargs=2, default=(8, 6))
    parser.add_argument("--active-temporal-radius", type=int, default=15)
    parser.add_argument("--min-track-frames", type=int, default=3)
    parser.add_argument("--active-quality-fraction", type=float, default=0.9,
                        help="fraction of each active set filled strictly by "
                             "persistent track quality before spatial fill")
    parser.add_argument("--block-frames", type=int, default=240)
    parser.add_argument("--block-stride", type=int, default=120)
    parser.add_argument("--landmarks-per-block", type=int, default=300)
    parser.add_argument("--landmark-random-fraction", type=float, default=0.5)
    parser.add_argument("--stereo-parallax-weight", type=float, default=0.0,
                        help="landmark sampling score per degree of measured "
                             "left-right ray parallax")
    parser.add_argument("--inside-obs-cap", type=int, default=7200)
    parser.add_argument("--outside-obs-cap", type=int, default=2480)
    parser.add_argument("--fixed-obs-cap", type=int, default=4560)
    parser.add_argument("--fixed-obs-per-frame", type=int, default=19)
    parser.add_argument("--obs-per-landmark", type=int, default=16)
    parser.add_argument("--block-iters", type=int, default=20)
    parser.add_argument("--positioning-epochs", type=int, default=0,
                        help="frozen-rotation center/landmark epochs before "
                             "free-SE3 refinement")
    parser.add_argument("--positioning-iters", type=int, default=5,
                        help="LM iterations per frozen-rotation block")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--blocks-per-epoch", type=int, default=10)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--max-blocks", type=int, default=0,
                        help="limit blocks per epoch for smoke tests; 0=all")
    parser.add_argument("--random-init-probe", action="store_true",
                        help="solve and export only the first block from random "
                             "centers/points and IMU-chain rotations")
    parser.add_argument("--random-init", action="store_true",
                        help="initialize one global state randomly and update "
                             "shuffled fixed-shape blocks over full recording")
    parser.add_argument("--initial-state", default=None,
                        help="optional global-block checkpoint to continue "
                             "instead of regenerating random geometry")
    parser.add_argument("--linear-bootstrap", action="store_true",
                        help="initialize random geometry with a global sparse "
                             "fixed-rotation stereo solve before block epochs")
    parser.add_argument("--rotation-bootstrap", action="store_true",
                        help="globally correct the IMU rotation chain with "
                             "per-frame gravity before linear stereo")
    parser.add_argument("--rotation-bootstrap-iters", type=int, default=20)
    parser.add_argument("--linear-bootstrap-irls", type=int, default=3)
    parser.add_argument(
        "--linear-bootstrap-max-iterations", type=int, default=500)
    parser.add_argument(
        "--linear-bootstrap-tolerance", type=float, default=1e-6)
    parser.add_argument(
        "--linear-bootstrap-initial-trajectory", default=None,
        help="optional trusted trajectory whose centers initialize robust "
             "weights before the first global linear solve")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--robust-scale", type=float, default=0.05)
    parser.add_argument("--max-point-med-ang", type=float, default=2.0)
    parser.add_argument("--min-positive-depth-frac", type=float, default=0.75)
    parser.add_argument("--imu-rot-weight", type=float, default=100.0)
    parser.add_argument("--gravity-weight", type=float, default=10.0)
    parser.add_argument("--constant-velocity-weight", type=float, default=1.0)
    parser.add_argument("--positive-depth-weight", type=float, default=0.1)
    parser.add_argument("--positive-depth-min", type=float, default=0.05)
    parser.add_argument("--positive-depth-softness", type=float, default=0.1)
    parser.add_argument("--termination-cost-tol", type=float, default=1e-7)
    parser.add_argument("--termination-gradient-tol", type=float, default=1e-6)
    parser.add_argument("--termination-gradient-start", type=int, default=15)
    parser.add_argument(
        "--termination-parameter-tol", type=float, default=1e-8)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    wall_start = time.time()
    source = np.load(args.trajectory) if args.trajectory else None
    if not 0 < args.block_stride <= args.block_frames:
        raise ValueError("--block-stride must be in (0, --block-frames]")
    if not 0.0 <= args.landmark_random_fraction <= 1.0:
        raise ValueError("--landmark-random-fraction must be in [0, 1]")
    if args.stereo_parallax_weight < 0.0:
        raise ValueError("--stereo-parallax-weight must be nonnegative")
    if not 0.0 <= args.active_quality_fraction <= 1.0:
        raise ValueError("--active-quality-fraction must be in [0, 1]")
    if args.checkpoint_every < 1:
        raise ValueError("--checkpoint-every must be positive")
    if args.positioning_epochs < 0 or args.positioning_iters < 1:
        raise ValueError(
            "--positioning-epochs must be nonnegative and "
            "--positioning-iters must be positive")
    if (
            args.linear_bootstrap_irls < 1
            or args.linear_bootstrap_max_iterations < 1
            or args.linear_bootstrap_tolerance <= 0.0
            or args.rotation_bootstrap_iters < 1):
        raise ValueError("linear bootstrap controls must be positive")
    total_visual_obs = (
        args.inside_obs_cap + args.outside_obs_cap + args.fixed_obs_cap)
    if total_visual_obs != 14240:
        raise ValueError(
            "fixed visual observation caps must sum to the baseline problem "
            f"size of 14240, got {total_visual_obs}")
    if args.fixed_obs_per_frame * args.block_frames > args.fixed_obs_cap:
        raise ValueError(
            "--fixed-obs-per-frame * --block-frames exceeds "
            "--fixed-obs-cap")
    if args.random_init:
        run_random_global(args, source)
        print(f"wall total: {time.time() - wall_start:.1f}s")
        return
    if source is None:
        raise ValueError("--trajectory is required unless --random-init is used")

    required = {
        "frame_idx", "pose_wxyz_xyz", "solver_frame_idx",
        "point_track_id", "points",
    }
    missing = required - set(source.files)
    if missing:
        raise ValueError(f"trajectory missing fields: {sorted(missing)}")

    native_frames = source["frame_idx"]
    solver_frames = source["solver_frame_idx"]
    native_lookup = {
        int(frame): index for index, frame in enumerate(native_frames)}
    sample_pos = np.asarray(
        [native_lookup[int(frame)] for frame in solver_frames], np.int64)
    poses = source["pose_wxyz_xyz"][sample_pos].copy()
    if args.block_frames > len(poses):
        raise ValueError("--block-frames exceeds the solver pose count")
    if args.random_init_probe:
        run_random_init_probe(args, source)
        return

    obs = load_solver_observations(
        args.recording, solver_frames, args.device, args.active_tracks,
        tuple(args.spatial_grid), args.active_temporal_radius,
        args.min_track_frames, args.active_quality_fraction, args.tracks)
    imu = load_solver_imu(args.recording, solver_frames)

    points = np.zeros(
        (len(obs["selected_source_track_id"]), 3), np.float64)
    source_point_lookup = {
        int(track): point
        for point, track in enumerate(source["point_track_id"])}
    for point, track in enumerate(obs["selected_source_track_id"]):
        source_point = source_point_lookup.get(int(track))
        if source_point is not None:
            points[point] = source["points"][source_point]
    points, alive, median_angle, positive_fraction = retriangulate(
        poses, points, obs, args.robust_scale, args.obs_per_landmark,
        args.max_point_med_ang, args.min_positive_depth_frac)
    print(
        f"state: {len(poses)} poses, {len(points)} active-union landmarks, "
        f"{len(obs['pose_id'])} observations")

    starts = block_starts(
        len(poses), args.block_frames, args.block_stride)
    if args.max_blocks > 0:
        starts = starts[:args.max_blocks]
    print(
        f"{len(starts)} overlapping blocks of {args.block_frames} poses; "
        f"stride={args.block_stride}")
    rng = np.random.default_rng(args.seed)
    solve_jit = jax.jit(
        lambda problem, values: solve_block(
            problem,
            values,
            args.block_iters,
            args.termination_cost_tol,
            args.termination_gradient_tol,
            args.termination_gradient_start,
            args.termination_parameter_tol,
        )
    )
    print("analyzing one reusable dense block template")
    template = build_dense_template(
        poses, points, alive, alive, obs, imu, starts[0], args)
    train_rows = np.flatnonzero(~obs["validation"])
    validation_rows = np.flatnonzero(obs["validation"])
    metrics = {
        "train_median_deg": [],
        "train_p90_deg": [],
        "validation_median_deg": [],
        "validation_p90_deg": [],
        "validation_positive_frac": [],
        "accepted_blocks": [],
        "wall_seconds": [],
    }
    epoch_ratios = []

    def record_metrics(accepted):
        train = angular_metrics(poses, points, obs, train_rows)
        validation = angular_metrics(
            poses, points, obs, validation_rows)
        metrics["train_median_deg"].append(train["median"])
        metrics["train_p90_deg"].append(train["p90"])
        metrics["validation_median_deg"].append(validation["median"])
        metrics["validation_p90_deg"].append(validation["p90"])
        metrics["validation_positive_frac"].append(validation["positive"])
        metrics["accepted_blocks"].append(accepted)
        metrics["wall_seconds"].append(time.time() - wall_start)
        print(
            f"angular metrics: train p50/p90 "
            f"{train['median']:.3f}/{train['p90']:.3f} deg; "
            f"validation p50/p90 "
            f"{validation['median']:.3f}/{validation['p90']:.3f} deg")

    record_metrics(0)
    config = {
        "active_tracks": args.active_tracks,
        "active_quality_fraction": args.active_quality_fraction,
        "block_frames": args.block_frames,
        "block_stride": args.block_stride,
        "landmarks_per_block": args.landmarks_per_block,
        "landmark_random_fraction": args.landmark_random_fraction,
        "fixed_obs_per_frame": args.fixed_obs_per_frame,
        "inside_obs_cap": args.inside_obs_cap,
        "outside_obs_cap": args.outside_obs_cap,
        "fixed_obs_cap": args.fixed_obs_cap,
        "obs_per_landmark": args.obs_per_landmark,
        "block_iters": args.block_iters,
        "robust_scale": args.robust_scale,
        "positive_depth_weight": args.positive_depth_weight,
        "seed": args.seed,
    }
    output_root, output_ext = os.path.splitext(args.out)
    output_ext = output_ext or ".npz"
    for epoch in range(args.epochs):
        print(f"epoch {epoch + 1}/{args.epochs}")
        poses, points, ratios, _, accepted = run_epoch(
            poses, points, alive, alive, obs, imu, starts, template, args,
            rng, solve_jit)
        epoch_ratios.append(ratios)
        points, alive, median_angle, positive_fraction = retriangulate(
            poses, points, obs, args.robust_scale, args.obs_per_landmark,
            args.max_point_med_ang, args.min_positive_depth_frac)
        record_metrics(accepted)
        checkpoint = f"{output_root}_epoch{epoch + 1}{output_ext}"
        write_output(
            checkpoint, source, solver_frames, sample_pos, poses, points,
            alive, median_angle, positive_fraction, obs, metrics,
            epoch_ratios, config)

    write_output(
        args.out, source, solver_frames, sample_pos, poses, points, alive,
        median_angle, positive_fraction, obs, metrics, epoch_ratios, config)
    print(f"wall total: {time.time() - wall_start:.1f}s")


if __name__ == "__main__":
    main()
