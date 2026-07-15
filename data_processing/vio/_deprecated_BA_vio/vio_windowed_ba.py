"""Windowed VIO solve: independent short-window solves, chained by refined pose.

Rationale: the global stage-5 solve is superlinear in trajectory length (CG
iteration count blows up as LM damping decays on a long, gauge-free problem --
observed: ~50min/iter at 11k frames), but the downstream use only needs LOW
DRIFT OVER ~10s, not global consistency. So: solve fixed-size windows (default
4s, overlapping by 30 frames) independently -- each is small and identical in
shape so jaxls JIT-compiles once per stage -- then chain them.

Per window, both stages of the global solver (both with the analytic-scale
positioning residual, no per-obs ScaleVars):
  1. frozen-rotation positioning (rotations sliced from ONE global IMU chain,
     gravity-aligned at frame 0)
  2. short SE3 refine (default 20 iters): rotations free, tethered by the IMU
     relative-rotation + gravity costs -- same recipe as vio_bundle_adjust
     stage 2. The full-solve A/B showed refine roughly halves 10s drift.

Stitching: consecutive windows share --overlap-frames (default 30) frames. The
gauge transform G aligning window k+1 into the stitched world is fixed by
requiring its refined pose at the last shared frame to equal the stitched
pose there (T_wc_stitched = T_wc_win @ G). This uses the window's own refined
relative motion and anchors where the output switches to the new window.

Fixed problem shape = fixed (n_obs_pad, n_points_pad) per window: observations
come from the longest, widest-span tracks and are time-subsampled/padded.
--max-landmarks directly controls dense-Cholesky size; --max-obs bounds the
dense Jacobian. Only selected observations are unprojected. Landmarks are
re-triangulated per window; only poses are stitched.

Output follows trajectory.npz: duplicate source tracks are robustly merged
across windows, angular/depth outliers are marked in point_alive, and first
video observations are retained for pixel coloring in visualize_data.py.

Run:
    python vio_windowed_ba.py ../../testimu --tracks ../../testimu/tracks.jsonl \
        --out /tmp/traj_windowed.npz
"""
import argparse
import json
import os
import time

import jax
import jax.numpy as jnp
import jaxlie
import numpy as np

import jaxls

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vio_bundle_adjust import (Point3Var, CamCenterVar,
                               load_intrinsics, load_stereo,
                               load_tracks, unproject_all)


def _stack_pytrees(items):
    return jax.tree.map(lambda *xs: jnp.stack(xs), *items)


def _solve_in_chunks(pairs, solve_one, chunk_size, stage_name):
    """Solve equal-shaped problems sequentially or in fixed-size vmap chunks."""
    n = len(pairs)
    chunk_size = n if chunk_size == 0 else min(chunk_size, n)
    solve_one_jit = jax.jit(solve_one)
    solve_chunk_jit = jax.jit(jax.vmap(solve_one)) if chunk_size > 1 else None
    outputs = None
    for start in range(0, n, chunk_size):
        real_count = min(chunk_size, n - start)
        chunk = list(pairs[start:start + real_count])
        chunk.extend([chunk[-1]] * (chunk_size - real_count))
        t0 = time.time()
        if chunk_size == 1:
            out = solve_one_jit(*chunk[0])
            out = jax.tree.map(lambda x: x[None], out)
        else:
            probs = _stack_pytrees([x[0] for x in chunk])
            vals = _stack_pytrees([x[1] for x in chunk])
            out = solve_chunk_jit(probs, vals)
        jax.block_until_ready(jax.tree.leaves(out)[0])
        out_np = jax.tree.map(np.asarray, out)
        if outputs is None:
            outputs = [[] for _ in jax.tree.leaves(out_np)]
        for bucket, leaf in zip(outputs, jax.tree.leaves(out_np)):
            bucket.append(leaf[:real_count])
        print(f"  {stage_name} chunk {start // chunk_size + 1}/"
              f"{-(-n // chunk_size)} ({real_count} windows): "
              f"{time.time() - t0:.2f}s")
    assert outputs is not None
    return tuple(np.concatenate(parts, axis=0) for parts in outputs)


def _sample_track_rows(track_rows, pose_ids, obs_right, take):
    """Spread samples over time while retaining both stereo eyes when present."""
    if len(track_rows) <= take:
        return track_rows

    frames = pose_ids[track_rows]
    chosen = [0, len(track_rows) - 1]
    midpoint = 0.5 * (frames[0] + frames[-1])
    for is_right in (False, True):
        eye = np.flatnonzero(obs_right[track_rows] == is_right)
        if len(eye):
            chosen.append(int(eye[np.argmin(np.abs(frames[eye] - midpoint))]))
    chosen = list(dict.fromkeys(chosen))

    # Greedy max-min frame spacing fills the remaining slots without clustering
    # all retained observations in the densest part of a long track.
    while len(chosen) < take:
        distance = np.min(
            np.abs(frames[:, None] - frames[np.asarray(chosen)][None]), axis=1)
        distance[np.asarray(chosen)] = -1
        best = int(np.argmax(distance))
        if distance[best] <= 0:
            unused = np.setdiff1d(
                np.arange(len(track_rows)), np.asarray(chosen),
                assume_unique=False)
            if not len(unused):
                break
            best = int(unused[len(unused) // 2])
        chosen.append(best)
    return track_rows[np.sort(chosen[:take])]


def _balanced_window_rows(rows, pose_ids, point_ids, obs_right, max_obs,
                          obs_per_track, min_track_frames, ray_world=None,
                          obs_px=None, spatial_grid=None, max_landmarks=0):
    """Keep well-constrained tracks and spread observations over their spans."""
    if len(rows) <= max_obs and max_landmarks == 0:
        return rows

    local_points = point_ids[rows]
    order = np.argsort(local_points, kind="stable")
    rows_sorted = rows[order]
    points_sorted = local_points[order]
    split = np.flatnonzero(np.r_[True, points_sorted[1:] != points_sorted[:-1]])
    groups = np.split(rows_sorted, split[1:])

    candidates = []
    for track_rows in groups:
        frames = pose_ids[track_rows]
        distinct_frames = np.unique(frames).size
        if distinct_frames < min_track_frames:
            continue
        span = int(frames.max() - frames.min())
        stereo_frames = 0
        for frame in np.unique(frames):
            eyes = obs_right[track_rows[frames == frame]]
            stereo_frames += bool(np.any(eyes) and np.any(~eyes))
        parallax = 0.0
        if ray_world is not None:
            thirds = max(1, len(track_rows) // 3)
            ray_a = np.mean(ray_world[track_rows[:thirds]], axis=0)
            ray_b = np.mean(ray_world[track_rows[-thirds:]], axis=0)
            ray_a /= np.linalg.norm(ray_a) + 1e-12
            ray_b /= np.linalg.norm(ray_b) + 1e-12
            parallax = np.degrees(np.arccos(np.clip(ray_a @ ray_b, -1, 1)))
        sort_key = (-distinct_frames, -span, -stereo_frames, -parallax,
                    int(point_ids[track_rows[0]]))
        cell = None
        if spatial_grid is not None:
            px_rows = track_rows[~obs_right[track_rows]]
            if not len(px_rows):
                px_rows = track_rows
            px = np.median(obs_px[px_rows], axis=0)
            gx, gy, width, height = spatial_grid
            cell = (
                min(gx - 1, max(0, int(gx * px[0] / width))),
                min(gy - 1, max(0, int(gy * px[1] / height))),
            )
        candidates.append((sort_key, cell, track_rows))

    if spatial_grid is None:
        candidates.sort(key=lambda x: x[0])
    else:
        by_cell = {}
        for candidate in candidates:
            by_cell.setdefault(candidate[1], []).append(candidate)
        for group in by_cell.values():
            group.sort(key=lambda x: x[0])
        candidates = []
        depth = 0
        while True:
            added = False
            for cell in sorted(by_cell):
                if depth < len(by_cell[cell]):
                    candidates.append(by_cell[cell][depth])
                    added = True
            if not added:
                break
            depth += 1

    selected = []
    used = 0
    for _, _, track_rows in candidates:
        if max_landmarks > 0 and len(selected) >= max_landmarks:
            break
        take = min(len(track_rows), obs_per_track, max_obs - used)
        if take < min_track_frames:
            continue
        selected.append(
            _sample_track_rows(track_rows, pose_ids, obs_right, take))
        used += take
        if max_obs - used < min_track_frames:
            break
    if not selected:
        raise ValueError("Balanced observation sampling found no usable tracks")
    return np.sort(np.concatenate(selected))


def _track_length_window_rows(rows, pose_ids, point_ids, max_obs,
                              obs_per_track, max_landmarks=0):
    """Original deterministic track-length sampler retained as a control."""
    local_points = point_ids[rows]
    order = np.argsort(local_points, kind="stable")
    rows_sorted = rows[order]
    points_sorted = local_points[order]
    split = np.flatnonzero(np.r_[True, points_sorted[1:] != points_sorted[:-1]])
    groups = np.split(rows_sorted, split[1:])
    candidates = []
    for track_rows in groups:
        frames = pose_ids[track_rows]
        distinct_frames = np.unique(frames).size
        if distinct_frames >= 2:
            candidates.append(
                (-distinct_frames, -int(frames.max() - frames.min()),
                 int(point_ids[track_rows[0]]), track_rows))
    candidates.sort(key=lambda x: x[:3])
    selected = []
    used = 0
    for _, _, _, track_rows in candidates:
        if max_landmarks > 0 and len(selected) >= max_landmarks:
            break
        take = min(len(track_rows), obs_per_track, max_obs - used)
        if take < 2:
            continue
        idx = np.linspace(0, len(track_rows) - 1, take, dtype=np.int64)
        selected.append(track_rows[idx])
        used += take
        if max_obs - used < 2:
            break
    if not selected:
        raise ValueError("Track-length sampling found no usable tracks")
    return np.sort(np.concatenate(selected))


def _complete_track_window_rows(rows, pose_ids, point_ids, obs_right, max_obs,
                                min_track_frames, min_pair_tracks,
                                window_start, window_size, max_landmarks=0):
    """Retain complete tracks while guaranteeing adjacent-frame support."""
    local_points = point_ids[rows]
    order = np.argsort(local_points, kind="stable")
    rows_sorted = rows[order]
    points_sorted = local_points[order]
    split = np.flatnonzero(np.r_[True, points_sorted[1:] != points_sorted[:-1]])
    groups = np.split(rows_sorted, split[1:])
    candidates = []
    for track_rows in groups:
        frames = pose_ids[track_rows]
        distinct_frames = np.unique(frames).size
        if distinct_frames >= min_track_frames:
            stereo_frames = 0
            for frame in np.unique(frames):
                eyes = obs_right[track_rows[frames == frame]]
                stereo_frames += bool(np.any(eyes) and np.any(~eyes))
            observed = np.zeros(window_size, dtype=bool)
            observed[np.unique(frames) - window_start] = True
            adjacent = observed[:-1] & observed[1:]
            candidates.append(
                (-distinct_frames, -int(frames.max() - frames.min()),
                 -stereo_frames, int(point_ids[track_rows[0]]),
                 track_rows, adjacent))
    candidates.sort(key=lambda x: x[:4])

    selected = []
    selected_ids = set()
    used = 0
    required_ids = set()
    for edge in range(window_size - 1):
        strongest = [
            candidate_id for candidate_id, candidate in enumerate(candidates)
            if candidate[5][edge]
        ][:min_pair_tracks]
        required_ids.update(strongest)
        if len(strongest) < min_pair_tracks:
            print(f"warning: frame pair {window_start + edge}->"
                  f"{window_start + edge + 1} has only {len(strongest)} "
                  "eligible tracks")

    required_obs = sum(len(candidates[i][4]) for i in required_ids)
    if required_obs > max_obs:
        raise ValueError(
            f"Per-pair top-{min_pair_tracks} complete tracks require "
            f"{required_obs} observations, above --max-obs={max_obs}")
    if max_landmarks > 0 and len(required_ids) > max_landmarks:
        raise ValueError(
            f"Per-pair top-{min_pair_tracks} requires {len(required_ids)} "
            f"landmarks, above --max-landmarks={max_landmarks}")

    pair_support = np.zeros(window_size - 1, dtype=np.int64)
    for candidate_id in sorted(required_ids):
        track_rows, adjacent = (
            candidates[candidate_id][4], candidates[candidate_id][5])
        selected.append(track_rows)
        selected_ids.add(candidate_id)
        used += len(track_rows)
        pair_support += adjacent

    for candidate_id, (_, _, _, _, track_rows, _) in enumerate(candidates):
        if candidate_id in selected_ids:
            continue
        if max_landmarks > 0 and len(selected) >= max_landmarks:
            break
        if used + len(track_rows) > max_obs:
            continue
        selected.append(track_rows)
        used += len(track_rows)
    if not selected:
        raise ValueError("Complete-track sampling found no usable tracks")
    return np.sort(np.concatenate(selected))


def _pair_coverage_window_rows(rows, pose_ids, point_ids, obs_right, max_obs,
                               obs_per_track, min_track_frames,
                               min_pair_tracks, window_start, window_size,
                               max_landmarks=0):
    """Reserve each frame pair's strongest tracks, then fill by track quality."""
    local_points = point_ids[rows]
    order = np.argsort(local_points, kind="stable")
    rows_sorted = rows[order]
    points_sorted = local_points[order]
    split = np.flatnonzero(np.r_[True, points_sorted[1:] != points_sorted[:-1]])
    groups = np.split(rows_sorted, split[1:])
    candidates = []
    for track_rows in groups:
        frames = pose_ids[track_rows]
        unique_frames = np.unique(frames)
        if len(unique_frames) < min_track_frames:
            continue
        stereo_frames = 0
        for frame in unique_frames:
            eyes = obs_right[track_rows[frames == frame]]
            stereo_frames += bool(np.any(eyes) and np.any(~eyes))
        observed = np.zeros(window_size, dtype=bool)
        observed[unique_frames - window_start] = True
        candidates.append((
            -len(unique_frames),
            -int(unique_frames[-1] - unique_frames[0]),
            -stereo_frames,
            int(point_ids[track_rows[0]]),
            track_rows,
            observed[:-1] & observed[1:],
        ))
    candidates.sort(key=lambda x: x[:4])

    reserved = {}
    for edge in range(window_size - 1):
        strongest = [
            candidate_id for candidate_id, candidate in enumerate(candidates)
            if candidate[5][edge]
        ][:min_pair_tracks]
        if len(strongest) < min_pair_tracks:
            print(f"warning: frame pair {window_start + edge}->"
                  f"{window_start + edge + 1} has only {len(strongest)} "
                  "eligible tracks")
        for candidate_id in strongest:
            track_rows = candidates[candidate_id][4]
            frames = pose_ids[track_rows]
            pair_rows = track_rows[
                (frames == window_start + edge)
                | (frames == window_start + edge + 1)]
            reserved.setdefault(candidate_id, []).append(pair_rows)

    chosen = {}
    for candidate_id, parts in reserved.items():
        track_rows = candidates[candidate_id][4]
        base = np.unique(np.concatenate(parts))
        spread = _sample_track_rows(
            track_rows, pose_ids, obs_right,
            min(obs_per_track, len(track_rows)))
        chosen[candidate_id] = np.union1d(base, spread)

    used = sum(len(track_rows) for track_rows in chosen.values())
    if used > max_obs:
        raise ValueError(
            f"Per-pair top-{min_pair_tracks} observations require {used} "
            f"rows, above --max-obs={max_obs}")
    if max_landmarks > 0 and len(chosen) > max_landmarks:
        raise ValueError(
            f"Per-pair top-{min_pair_tracks} requires {len(chosen)} "
            f"landmarks, above --max-landmarks={max_landmarks}")

    for candidate_id, candidate in enumerate(candidates):
        if candidate_id in chosen:
            continue
        if max_landmarks > 0 and len(chosen) >= max_landmarks:
            break
        track_rows = candidate[4]
        sampled = _sample_track_rows(
            track_rows, pose_ids, obs_right,
            min(obs_per_track, len(track_rows)))
        if used + len(sampled) > max_obs:
            continue
        chosen[candidate_id] = sampled
        used += len(sampled)
    if not chosen:
        raise ValueError("Pair-coverage sampling found no usable tracks")
    return np.sort(np.concatenate(list(chosen.values())))


def _quat_to_matrix(q):
    q = np.asarray(q)
    q = q / max(np.linalg.norm(q), 1e-12)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ])


def _quat_to_matrix_batch(q):
    q = np.asarray(q)
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack([
        np.stack([1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)], -1),
        np.stack([2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)], -1),
        np.stack([2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)], -1),
    ], axis=-2)


def _point_diagnostics(q, centers, points, pose_id, point_id, rays, rel, npts):
    """Median bearing error and positive-depth fraction per local landmark."""
    R_wl = _quat_to_matrix_batch(q[pose_id])
    p_left = np.einsum(
        "nij,nj->ni", R_wl, points[point_id] - centers[pose_id])
    R_rel = _quat_to_matrix_batch(rel[:, :4])
    p_eye = np.einsum("nij,nj->ni", R_rel, p_left) + rel[:, 4:]
    direction = p_eye / np.maximum(np.linalg.norm(p_eye, axis=1, keepdims=True),
                                   1e-12)
    angle = np.degrees(np.arccos(np.clip(
        np.einsum("ni,ni->n", direction, rays), -1.0, 1.0)))
    positive = p_eye[:, 2] > 0

    med_angle = np.full(npts, np.inf)
    positive_frac = np.zeros(npts)
    order = np.argsort(point_id, kind="stable")
    point_sorted = point_id[order]
    angle_sorted = angle[order]
    positive_sorted = positive[order]
    bounds = np.searchsorted(point_sorted, np.arange(npts + 1))
    for pid in range(npts):
        sl = slice(bounds[pid], bounds[pid + 1])
        med_angle[pid] = np.median(angle_sorted[sl])
        positive_frac[pid] = np.mean(positive_sorted[sl])
    return med_angle, positive_frac


def _matrix_to_quat(R):
    q = np.asarray(jaxlie.SO3.from_matrix(jnp.asarray(R)).wxyz)
    return q / max(np.linalg.norm(q), 1e-12)


def _pose_matrix(q, c):
    R = _quat_to_matrix(q)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = -(R @ c)
    return T


def _mean_gauge(gauges):
    """Chordal SE(3) mean; gauges are close over one short overlap."""
    M = np.mean([G[:3, :3] for G in gauges], axis=0)
    U, _, Vt = np.linalg.svd(M)
    S = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    out = np.eye(4)
    out[:3, :3] = U @ S @ Vt
    out[:3, 3] = np.mean([G[:3, 3] for G in gauges], axis=0)
    return out


def _rotation_error_deg(Ra, Rb):
    R = Ra @ Rb.T
    return np.degrees(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)))


def _interpolate_pose_samples(centers, quats, sample_pos, output_size):
    """Interpolate sparse camera poses onto the native valid-frame timeline."""
    sample_pos = np.asarray(sample_pos)
    if len(sample_pos) == output_size:
        return centers, quats

    output_pos = np.arange(output_size)
    centers_out = np.stack([
        np.interp(output_pos, sample_pos, centers[:, axis])
        for axis in range(3)
    ], axis=1)

    quats = np.asarray(quats, dtype=np.float64).copy()
    quats /= np.maximum(np.linalg.norm(quats, axis=1, keepdims=True), 1e-12)
    for i in range(1, len(quats)):
        if np.dot(quats[i - 1], quats[i]) < 0:
            quats[i] *= -1

    seg = np.searchsorted(sample_pos, output_pos, side="right") - 1
    seg = np.clip(seg, 0, len(sample_pos) - 2)
    span = sample_pos[seg + 1] - sample_pos[seg]
    alpha = (output_pos - sample_pos[seg]) / np.maximum(span, 1)
    q0, q1 = quats[seg], quats[seg + 1]
    dot = np.clip(np.sum(q0 * q1, axis=1), -1.0, 1.0)
    angle = np.arccos(dot)
    sin_angle = np.sin(angle)
    near = np.abs(sin_angle) < 1e-7
    w0 = np.where(
        near, 1.0 - alpha,
        np.sin((1.0 - alpha) * angle) / np.where(near, 1.0, sin_angle))
    w1 = np.where(
        near, alpha,
        np.sin(alpha * angle) / np.where(near, 1.0, sin_angle))
    quats_out = w0[:, None] * q0 + w1[:, None] * q1
    quats_out /= np.maximum(
        np.linalg.norm(quats_out, axis=1, keepdims=True), 1e-12)
    return centers_out, quats_out


def _stitch_windows(starts, q_all, c_all, p_all, npts_all, n_frames, mode):
    """Stitch window gauges and report disagreement over each overlap."""
    W = q_all.shape[1]
    centers_glob = np.zeros((n_frames, 3))
    quats_glob = np.zeros((n_frames, 4))
    have = np.zeros(n_frames, bool)
    merged_pts = []
    seam_pos, seam_rot, overlap_pos_rms, overlap_rot_rms = [], [], [], []

    for wi, s in enumerate(starts):
        q_win = q_all[wi]
        c_win = c_all[wi]
        p_win = p_all[wi]
        if wi == 0:
            centers_glob[s:s + W] = c_win
            quats_glob[s:s + W] = q_win
            have[s:s + W] = True
            merged_pts.append(p_win[:npts_all[wi]])
            continue

        overlap = np.arange(s, s + W)[have[s:s + W]]
        local = overlap - s
        gauges = [
            np.linalg.inv(_pose_matrix(q_win[j], c_win[j]))
            @ _pose_matrix(quats_glob[f], centers_glob[f])
            for f, j in zip(overlap, local)
        ]
        if mode == "first":
            G = gauges[0]
        elif mode == "last":
            G = gauges[-1]
        elif mode == "mean":
            G = _mean_gauge(gauges)
        elif mode == "gravity":
            # Every window's local world is independently gravity-aligned.
            # Preserve that shared z axis instead of accumulating small
            # roll/pitch errors from unconstrained SE(3) stitch rotations.
            G_full = gauges[-1]
            A_full = np.linalg.inv(G_full)[:3, :3]
            yaw = np.arctan2(
                A_full[1, 0] - A_full[0, 1],
                A_full[0, 0] + A_full[1, 1])
            cy, sy = np.cos(yaw), np.sin(yaw)
            A_yaw = np.array([
                [cy, -sy, 0.0],
                [sy, cy, 0.0],
                [0.0, 0.0, 1.0],
            ])
            anchor_local = local[-1]
            anchor_global = overlap[-1]
            Ginv = np.eye(4)
            Ginv[:3, :3] = A_yaw
            Ginv[:3, 3] = (
                centers_glob[anchor_global] - A_yaw @ c_win[anchor_local])
            G = np.linalg.inv(Ginv)
        else:
            raise ValueError(mode)
        if mode != "gravity":
            Ginv = np.linalg.inv(G)

        transformed_centers = c_win @ Ginv[:3, :3].T + Ginv[:3, 3]
        transformed_quats = np.stack([
            _matrix_to_quat(_pose_matrix(q, c)[:3, :3] @ G[:3, :3])
            for q, c in zip(q_win, c_win)
        ])
        pos_err = np.linalg.norm(
            transformed_centers[local] - centers_glob[overlap], axis=1)
        rot_err = np.array([
            _rotation_error_deg(
                _quat_to_matrix(transformed_quats[j]),
                _quat_to_matrix(quats_glob[f]),
            )
            for f, j in zip(overlap, local)
        ])
        seam_pos.append(pos_err[-1])
        seam_rot.append(rot_err[-1])
        overlap_pos_rms.append(np.sqrt(np.mean(pos_err ** 2)))
        overlap_rot_rms.append(np.sqrt(np.mean(rot_err ** 2)))

        new = np.arange(s, s + W)[~have[s:s + W]]
        centers_glob[new] = transformed_centers[new - s]
        quats_glob[new] = transformed_quats[new - s]
        have[new] = True
        merged_pts.append(
            p_win[:npts_all[wi]] @ Ginv[:3, :3].T + Ginv[:3, 3])

    assert have.all()
    metrics = {
        "seam_pos": np.asarray(seam_pos),
        "seam_rot": np.asarray(seam_rot),
        "overlap_pos_rms": np.asarray(overlap_pos_rms),
        "overlap_rot_rms": np.asarray(overlap_rot_rms),
    }
    return centers_glob, quats_glob, merged_pts, metrics


def _win_residual(vals, center, point_var, pose_quat, ray, rel, w,
                  robust_scale):
    """Positioning residual with the per-observation scale ANALYTICALLY
    eliminated: for fixed X, c the optimal GLOMAP d* = <ray, X-c>/|X-c|^2, and
    substituting it leaves the component of the ray PERPENDICULAR to (X-c) --
    a pure angular (sin-theta) error. Same optimum as the ScaleVar form, but
    the solver never sees the 50k+ per-obs scale variables that dominated the
    window solve (and forced Schur, whose static elimination plan blocked
    vmapping across windows). Frozen rotations; per-obs weight w=0 pads.
    MODULE-LEVEL on purpose: jaxls hashes the residual fn by identity in the
    jit cache key, so a per-window closure would retrace per window."""
    T_rel = jaxlie.SE3(rel)
    R_wl = jaxlie.SO3(pose_quat)
    R_wc = T_rel.rotation() @ R_wl
    cam_pos = vals[center] - (R_wc.inverse() @ T_rel.translation())
    ray_w = R_wc.inverse() @ ray
    v = vals[point_var] - cam_pos
    v_dir = v / (jnp.linalg.norm(v) + 1e-9)
    # d* clamped >= 0 (see vio_bundle_adjust.positioning_cost): without the
    # clamp a behind-camera landmark costs ~0 and the solve diverges.
    d_star = jnp.maximum(jnp.dot(ray_w, v_dir), 0.0)
    r = ray_w - d_star * v_dir
    abs_r = jnp.linalg.norm(r) + 1e-9
    wr = jax.lax.stop_gradient(1.0 / (1.0 + (abs_r / robust_scale) ** 2))
    return r * (w * jnp.sqrt(wr))


def _refine_positioning(vals, pose_var, point_var, ray, rel, w, robust_scale):
    """SE3-refine version of _win_residual: pose is a free SE3Var instead of a
    frozen quat + CamCenterVar. Same analytic-scale (perpendicular component)
    residual with the d*>=0 cheirality clamp."""
    T_wl = vals[pose_var]
    R_wl = T_wl.rotation()
    c_left = -(R_wl.inverse() @ T_wl.translation())
    T_rel = jaxlie.SE3(rel)
    R_wc = T_rel.rotation() @ R_wl
    cam_pos = c_left - (R_wc.inverse() @ T_rel.translation())
    ray_w = R_wc.inverse() @ ray
    v = vals[point_var] - cam_pos
    v_dir = v / (jnp.linalg.norm(v) + 1e-9)
    d_star = jnp.maximum(jnp.dot(ray_w, v_dir), 0.0)
    r = ray_w - d_star * v_dir
    abs_r = jnp.linalg.norm(r) + 1e-9
    wr = jax.lax.stop_gradient(1.0 / (1.0 + (abs_r / robust_scale) ** 2))
    return r * (w * jnp.sqrt(wr))


def _center_positive_depth(vals, center, point_var, pose_quat, ray, rel,
                           min_depth, softness, weight):
    """Soft cheirality barrier with a nonzero gradient behind the camera."""
    T_rel = jaxlie.SE3(rel)
    R_wc = T_rel.rotation() @ jaxlie.SO3(pose_quat)
    cam_pos = vals[center] - (R_wc.inverse() @ T_rel.translation())
    ray_w = R_wc.inverse() @ ray
    depth = jnp.dot(ray_w, vals[point_var] - cam_pos)
    violation = softness * jax.nn.softplus((min_depth - depth) / softness)
    return violation[None] * weight


def _pose_positive_depth(vals, pose_var, point_var, ray, rel, min_depth,
                         softness, weight):
    """SE(3)-refinement version of the soft cheirality barrier."""
    T_wl = vals[pose_var]
    T_rel = jaxlie.SE3(rel)
    R_wc = T_rel.rotation() @ T_wl.rotation()
    c_left = -(T_wl.rotation().inverse() @ T_wl.translation())
    cam_pos = c_left - (R_wc.inverse() @ T_rel.translation())
    ray_w = R_wc.inverse() @ ray
    depth = jnp.dot(ray_w, vals[point_var] - cam_pos)
    violation = softness * jax.nn.softplus((min_depth - depth) / softness)
    return violation[None] * weight


def _point_padding_anchor(vals, point_var, weight):
    """Condition padded-only point columns; real-point weights are zero."""
    return vals[point_var] * weight


def _refine_imu_rot(vals, pose_i, pose_j, delta_wxyz, weight):
    """IMU relative-rotation tether between consecutive window poses (same
    residual as vio_bundle_adjust.imu_rotation_cost)."""
    R_i = vals[pose_i].rotation()
    R_j = vals[pose_j].rotation()
    return (jaxlie.SO3(delta_wxyz).inverse() @ (R_i @ R_j.inverse())).log() * weight


def _refine_gravity(vals, pose_var, measured_down_cam, weight):
    """Roll/pitch prior: world is +z up; each pose should map world-down onto
    its IMU-measured down (yaw free)."""
    down_pred = vals[pose_var].rotation() @ jnp.array([0.0, 0.0, -1.0])
    return (down_pred - measured_down_cam) * weight


def _center_gauge(vals, center, weight):
    """Fix the translation gauge of one independent positioning window."""
    return vals[center] * weight


def _center_constant_velocity(vals, center_prev, center, center_next,
                              dt_prev, dt_next, time_scale, weight):
    """Time-aware velocity-change prior that does not fix the global gauge."""
    v_prev = (vals[center] - vals[center_prev]) / dt_prev
    v_next = (vals[center_next] - vals[center]) / dt_next
    return (v_next - v_prev) * (time_scale * weight)


def _pose_constant_velocity(vals, pose_prev, pose, pose_next,
                            dt_prev, dt_next, time_scale, weight):
    """Constant-velocity residual on camera centers from world-to-cam poses."""
    def camera_center(T):
        return -(T.rotation().inverse() @ T.translation())

    v_prev = (
        camera_center(vals[pose]) - camera_center(vals[pose_prev])
    ) / dt_prev
    v_next = (
        camera_center(vals[pose_next]) - camera_center(vals[pose])
    ) / dt_next
    return (v_next - v_prev) * (time_scale * weight)


def _pose_gauge(vals, pose_var, target_wxyz_xyz, weight):
    """Fix the global SE(3) gauge to the stage-1 first pose."""
    target = jaxlie.SE3(target_wxyz_xyz)
    return (target.inverse() @ vals[pose_var]).log() * weight


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("recording")
    p.add_argument("--tracks", default=None, help="default: <recording>/derived/tracks.jsonl")
    p.add_argument("--imu-relative", default=None,
                    help="default: <recording>/derived/imu_relative.npz")
    p.add_argument("--out", default=None,
                    help="default: <recording>/derived/trajectory_windowed.npz")
    p.add_argument("--window-s", type=float, default=4.0)
    p.add_argument("--target-fps", type=float, default=0.0,
                   help="optimize approximately uniform temporal keyframes at "
                        "this rate, then interpolate poses back to every native "
                        "frame (0 uses every valid frame)")
    p.add_argument("--window-index", type=int, default=None,
                   help="solve and export only this zero-based window; useful "
                        "for dense local diagnostics without solving the full "
                        "recording")
    p.add_argument("--init-seed", type=int, default=None,
                   help="fixed random initialization seed for every window; "
                        "by default each window uses its global window index, "
                        "including in --window-index diagnostic runs")
    p.add_argument("--overlap-frames", type=int, default=30,
                    help="native-rate frames shared between consecutive windows "
                         "(scaled when --target-fps is used). The stitch "
                         "uses the REFINED relative pose across the shared "
                         "frame, so 2 suffices (vs 30 for the old Procrustes-"
                         "on-centers stitch)")
    p.add_argument("--iters", type=int, default=30,
                    help="LM iterations for the positioning stage per window")
    p.add_argument("--refine-iters", type=int, default=20,
                   help="LM iterations for the per-window SE3 refine (IMU "
                        "rotation + gravity tethered); 0 skips refine and "
                        "stitches on IMU-chain rotations")
    p.add_argument("--early-termination", action="store_true",
                   help="allow strict convergence checks to stop LM before "
                        "--iters/--refine-iters (disabled by default)")
    p.add_argument("--termination-cost-tol", type=float, default=1e-7)
    p.add_argument("--termination-gradient-tol", type=float, default=1e-6)
    p.add_argument("--termination-gradient-start", type=int, default=15)
    p.add_argument("--termination-parameter-tol", type=float, default=1e-8)
    p.add_argument("--imu-rot-weight", type=float, default=100.0)
    p.add_argument("--gravity-weight", type=float, default=1.0)
    p.add_argument("--robust-scale", type=float, default=0.05)
    p.add_argument("--constant-velocity-weight", type=float, default=1.0,
                   help="weight on camera-center second differences in both "
                        "positioning and SE3 refinement (0 disables)")
    p.add_argument("--positive-depth-weight", type=float, default=0.1,
                   help="weight on a soft positive signed-depth residual for "
                        "every observation (0 disables)")
    p.add_argument("--positive-depth-min", type=float, default=0.05,
                   help="minimum preferred signed ray depth in meters")
    p.add_argument("--positive-depth-softness", type=float, default=0.1,
                   help="softplus transition width for positive-depth cost")
    p.add_argument("--gauge-weight", type=float, default=0.0,
                   help="exactly-neutral per-window origin/pose gauge anchor "
                        "used to condition dense normal equations; 0 disables")
    p.add_argument("--pad-quantile", type=float, default=50.0,
                    help="percentile of per-window obs counts used as the padded "
                         "problem size; windows above it are subsampled")
    p.add_argument("--max-obs", type=int, default=16000,
                    help="hard cap on (padded) observations per window. The "
                         "window has only a few thousand unknowns, so 16k "
                         "3-dim residuals is still over-determined; dense_cholesky "
                         "materializes the dense Jacobian (obs*3 x vars), which "
                         "OOMs at 56k obs x 15 windows (~108GB). 0 = no cap")
    p.add_argument("--max-landmarks", type=int, default=1500,
                   help="direct cap on selected landmarks per window; this is "
                        "the primary dense-Cholesky complexity control (0=no cap)")
    p.add_argument("--obs-sampling",
                   choices=("spatial", "quality", "coverage", "track_length",
                            "balanced", "complete_track", "pair_coverage",
                            "random"),
                   default="pair_coverage",
                   help="when capped, pair_coverage reserves the strongest "
                        "tracks crossing each adjacent frame pair, then fills "
                        "the remaining budget by track quality; quality adds "
                        "rotation-compensated parallax; track_length is the "
                        "original deterministic sampler")
    p.add_argument("--obs-per-track", type=int, default=8,
                   help="maximum time-spread observations retained per track")
    p.add_argument("--min-track-frames", type=int, default=3,
                   help="minimum distinct in-window frames for a sampled track")
    p.add_argument("--min-pair-tracks", type=int, default=20,
                   help="with pair_coverage or complete_track sampling, "
                        "minimum retained landmarks observed in every adjacent "
                        "frame pair")
    p.add_argument("--spatial-grid", type=int, nargs=2, default=(8, 6),
                   metavar=("COLS", "ROWS"),
                   help="image grid used to distribute --obs-sampling spatial")
    p.add_argument("--solve-chunk-size", type=int, default=1,
                   help="number of equal-shaped windows solved in parallel; "
                        "1 is sequential, 0 vmaps all windows")
    p.add_argument("--stitch-mode",
                   choices=("first", "last", "mean", "gravity"),
                   default="last",
                   help="window gauge from the first shared pose, last shared "
                        "pose (the actual output seam), an SE(3) overlap mean, "
                        "or gravity-preserving yaw plus last-center alignment")
    p.add_argument("--max-point-med-ang", type=float, default=2.0,
                   help="hide window-local landmark estimates whose median "
                        "bearing residual exceeds this many degrees")
    p.add_argument("--min-positive-depth-frac", type=float, default=0.75,
                   help="hide landmark estimates seen behind an observing "
                        "camera more often than this permits")
    p.add_argument("--linear-solver",
                    choices=("conjugate_gradient", "dense_cholesky"),
                    default="dense_cholesky",
                    help="windows are tiny (~1.7k vars), so a dense factorization "
                         "beats CG: no data-dependent inner-iteration count, so "
                         "per-window cost is uniform (CG varied 2-90s/window on "
                         "ill-conditioned fast-motion windows) and vmap lockstep "
                         "isn't dragged down by the slowest window")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    rec = args.recording
    tracks_path = args.tracks or os.path.join(rec, "derived", "tracks.jsonl")
    imu_path = args.imu_relative or os.path.join(rec, "derived", "imu_relative.npz")
    out_path = args.out or os.path.join(rec, "derived", "trajectory_windowed.npz")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    t_wall0 = time.time()

    native_fps = float(
        json.load(open(os.path.join(rec, "recording.json"))).get("fps", 30))
    if args.target_fps < 0 or args.target_fps > native_fps:
        raise ValueError(
            f"--target-fps must be in [0, native fps={native_fps:g}]")
    solver_fps = args.target_fps if args.target_fps > 0 else native_fps
    W = max(8, int(round(args.window_s * solver_fps)))
    V = max(2, int(round(args.overlap_frames * solver_fps / native_fps)))
    if V >= W:
        raise ValueError(
            f"scaled overlap ({V} poses) must be smaller than window ({W})")
    stride = W - V

    imu = np.load(imu_path)
    frame_idx_all = imu["frame_idx"]
    frame_valid = imu["frame_valid"]
    rel_quat_all = imu["rel_quat"]
    rel_valid_all = imu["rel_valid"]
    gravity_cam_all = imu["gravity_cam"]

    # Drop invalid-timestamp frames (same policy as the global solver); the
    # relative chain across a dropped frame gets an identity fill (one frame of
    # gyro is ~nothing).
    keep = frame_valid
    frame_idx_native = frame_idx_all[keep]
    gravity_cam_native = gravity_cam_all[keep]
    gravity_weight_native = imu["gravity_weight"][keep]
    n_frames_native = len(frame_idx_native)
    # delta_prev[i] rotates pose i-1 -> pose i (identity where edge invalid).
    delta_native = np.tile(
        np.array([1.0, 0, 0, 0]), (n_frames_native, 1))
    new_pos = -np.ones(len(frame_idx_all), np.int64)
    new_pos[keep] = np.arange(n_frames_native)
    for e in range(len(rel_quat_all)):
        a, b = new_pos[e], new_pos[e + 1]
        if a >= 0 and b == a + 1 and rel_valid_all[e]:
            delta_native[b] = rel_quat_all[e]

    # ONE global IMU rotation chain, gravity-aligned at frame 0 (same recipe as
    # vio_bundle_adjust). Windows slice it: gyro drift over any 3s window is
    # negligible, and this avoids re-seeding each window from its own gravity
    # estimate (noisy during fast motion -> was causing 10-15 deg window errors).
    g0 = gravity_cam_native[0] / (
        np.linalg.norm(gravity_cam_native[0]) + 1e-12)
    a_down = np.array([0.0, 0.0, -1.0])
    q0 = np.array([1.0 + a_down @ g0, *np.cross(a_down, g0)])
    if np.linalg.norm(q0) < 1e-6:
        q0 = np.array([0.0, 1.0, 0.0, 0.0])
    q0 = q0 / np.linalg.norm(q0)
    def _chain_step(R_prev_wxyz, delta_wxyz):
        R_k = jaxlie.SO3(delta_wxyz).inverse() @ jaxlie.SO3(R_prev_wxyz)
        return R_k.wxyz, R_k.wxyz
    q0_j = jnp.asarray(q0, dtype=jnp.float32)
    _, rot_rest = jax.lax.scan(
        _chain_step, q0_j, jnp.asarray(delta_native[1:], dtype=jnp.float32))
    rot_chain_native = np.asarray(
        jnp.concatenate([q0_j[None], rot_rest], axis=0))

    if args.target_fps > 0:
        sample_step = native_fps / solver_fps
        sample_pos = np.unique(np.rint(
            np.arange(0, n_frames_native - 1, sample_step)).astype(np.int64))
        sample_pos = np.unique(np.concatenate(
            [sample_pos, [n_frames_native - 1]]))
    else:
        sample_pos = np.arange(n_frames_native)

    frame_idx = frame_idx_native[sample_pos]
    gravity_cam = gravity_cam_native[sample_pos]
    gravity_weight_arr = gravity_weight_native[sample_pos]
    rot_chain = rot_chain_native[sample_pos]
    n_frames = len(frame_idx)
    solver_time = (
        frame_idx.astype(np.float64) - float(frame_idx[0])) / native_fps
    delta_prev = np.tile(np.array([1.0, 0, 0, 0]), (n_frames, 1))
    if n_frames > 1:
        delta_prev[1:] = np.asarray(jax.vmap(
            lambda qi, qj: (
                jaxlie.SO3(qi) @ jaxlie.SO3(qj).inverse()).wxyz
        )(jnp.asarray(rot_chain[:-1]), jnp.asarray(rot_chain[1:])))

    frame_to_pose = {int(f): i for i, f in enumerate(frame_idx)}
    max_frame = int(frame_idx_native[-1])

    import h5py
    with h5py.File(os.path.join(rec, "derived", "features.h5"), "r") as f:
        ls, rs = f.attrs["left_serial"], f.attrs["right_serial"]
    Kl, Dl = load_intrinsics(rec, ls)
    Kr, Dr = load_intrinsics(rec, rs)
    R_st, t_st = load_stereo(rec, ls, rs)
    rel_left = np.asarray(jaxlie.SE3.identity().wxyz_xyz)
    rel_right = np.asarray(jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3.from_matrix(R_st), t_st).wxyz_xyz)

    tracks = load_tracks(tracks_path, max_frame)
    actual_solver_fps = (
        (n_frames - 1) * native_fps
        / max(float(frame_idx[-1] - frame_idx[0]), 1.0))
    print(f"{n_frames} solver frames ({n_frames_native} native, "
          f"{actual_solver_fps:.2f} fps), {len(tracks)} tracks, "
          f"window={W}f overlap={V}f")

    # Flat observation table, sorted by pose index for fast window slicing.
    pose_ids, point_ids, obs_px, obs_right = [], [], [], []
    for k, obs in enumerate(tracks):
        for eye, fr, px in obs:
            i = frame_to_pose.get(int(fr))
            if i is None:
                continue
            pose_ids.append(i); point_ids.append(k)
            obs_px.append(px); obs_right.append(eye == "right")
    pose_ids = np.array(pose_ids); point_ids = np.array(point_ids)
    obs_px = np.stack(obs_px); obs_right = np.array(obs_right, bool)
    order = np.argsort(pose_ids, kind="stable")
    pose_ids, point_ids, obs_px, obs_right = (
        pose_ids[order], point_ids[order], obs_px[order], obs_right[order])

    rel_all = np.where(obs_right[:, None], rel_right[None], rel_left[None])

    def unproject_rows(rows):
        right = obs_right[rows]
        rays = unproject_all(
            {"left": obs_px[rows[~right]], "right": obs_px[rows[right]]},
            {"left": (Kl, Dl), "right": (Kr, Dr)}, args.device)
        out = np.zeros((len(rows), 3))
        out[~right] = rays["left"]
        out[right] = rays["right"]
        return out

    ray_cam = None
    ray_world_init = None
    if args.obs_sampling == "quality":
        all_rows = np.arange(len(pose_ids))
        ray_cam = unproject_rows(all_rows)
        R_wl_obs = _quat_to_matrix_batch(rot_chain[pose_ids])
        R_rel_obs = _quat_to_matrix_batch(rel_all[:, :4])
        R_wc_obs = np.einsum("nij,njk->nik", R_rel_obs, R_wl_obs)
        ray_world_init = np.einsum("nji,nj->ni", R_wc_obs, ray_cam)

    # Window starts; last window snapped to cover the tail.
    all_starts = list(range(0, max(n_frames - W, 0) + 1, stride))
    if all_starts[-1] + W < n_frames:
        all_starts.append(n_frames - W)
    if args.window_index is not None:
        if not 0 <= args.window_index < len(all_starts):
            raise ValueError(
                f"--window-index must be in [0, {len(all_starts) - 1}]")
        starts = [all_starts[args.window_index]]
        window_ids = [args.window_index]
        print(f"diagnostic window {args.window_index}/{len(all_starts) - 1}: "
              f"pose rows [{starts[0]}, {starts[0] + W})")
    else:
        starts = all_starts
        window_ids = list(range(len(starts)))
    row_lo = np.searchsorted(pose_ids, np.array(starts))
    row_hi = np.searchsorted(pose_ids, np.array(starts) + W)

    # Fixed padded sizes across windows -> one JIT compile.
    counts = row_hi - row_lo
    obs_budget = int(np.percentile(counts, args.pad_quantile))
    if args.max_obs > 0:
        obs_budget = min(obs_budget, args.max_obs)
    # points per window bounded by obs (each point >=2 obs)
    n_pts_pad = 0
    win_data = []
    rng = np.random.default_rng(0)
    for s, lo, hi in zip(starts, row_lo, row_hi):
        rows = np.arange(lo, hi)
        if len(rows) > obs_budget or args.max_landmarks > 0:
            if args.obs_sampling == "pair_coverage":
                rows = _pair_coverage_window_rows(
                    rows, pose_ids, point_ids, obs_right, obs_budget,
                    args.obs_per_track, args.min_track_frames,
                    args.min_pair_tracks, s, W, args.max_landmarks)
            elif args.obs_sampling == "complete_track":
                rows = _complete_track_window_rows(
                    rows, pose_ids, point_ids, obs_right, obs_budget,
                    args.min_track_frames, args.min_pair_tracks, s, W,
                    args.max_landmarks)
            elif args.obs_sampling in ("track_length", "balanced"):
                rows = _track_length_window_rows(
                    rows, pose_ids, point_ids, obs_budget, args.obs_per_track,
                    args.max_landmarks)
            elif args.obs_sampling in ("spatial", "quality", "coverage"):
                spatial_grid = None
                if args.obs_sampling == "spatial":
                    spatial_grid = (
                        *args.spatial_grid,
                        float(np.max(obs_px[:, 0]) + 1),
                        float(np.max(obs_px[:, 1]) + 1),
                    )
                rows = _balanced_window_rows(
                    rows, pose_ids, point_ids, obs_right, obs_budget,
                    args.obs_per_track, args.min_track_frames,
                    ray_world_init if args.obs_sampling == "quality" else None,
                    obs_px, spatial_grid, args.max_landmarks)
            else:
                rows = np.sort(rng.choice(rows, obs_budget, replace=False))
        # local point reindex
        upts, local_pid = np.unique(point_ids[rows], return_inverse=True)
        n_pts_pad = max(n_pts_pad, len(upts))
        win_data.append((s, rows, local_pid, len(upts), upts))
    n_obs_pad = max(len(x[1]) for x in win_data)

    # Track-based samplers need no calibrated rays. Unproject only observations
    # that survived selection, then scatter them back into flat-row indexing.
    if ray_cam is None:
        needed_rows = np.unique(np.concatenate([x[1] for x in win_data]))
        ray_cam = np.zeros((len(pose_ids), 3))
        ray_cam[needed_rows] = unproject_rows(needed_rows)
        print(f"unprojected {len(needed_rows)}/{len(pose_ids)} observations "
              f"after track selection")
    print(f"{len(starts)} windows; obs/window pad={n_obs_pad} "
          f"(counts p50={int(np.median(counts))} max={counts.max()}), "
          f"pts pad={n_pts_pad}")

    # Assemble padded arrays per window. Padding rows: weight 0 via robust_scale
    # trick is messy; instead point pad rows at pose 0 / point 0 with ZERO ray,
    # which yields residual = -d*(X-c) .. not zero. Cleaner: repeat a real row
    # and give the pad rows a per-row weight multiplier baked into the ray
    # (scaling a residual by 0 = scaling its ray+point contribution by 0 only
    # works if the whole residual is multiplied). We add an explicit per-obs
    # weight argument to the cost instead.
    N = len(win_data)
    P_pose = np.zeros((N, n_obs_pad), np.int32)
    P_point = np.zeros((N, n_obs_pad), np.int32)
    P_ray = np.zeros((N, n_obs_pad, 3), np.float32)
    P_rel = np.tile(rel_left.astype(np.float32), (N, n_obs_pad, 1))
    P_w = np.zeros((N, n_obs_pad), np.float32)
    R_init = np.zeros((N, W, 4), np.float32)
    for wi, (s, rows, local_pid, npts, _) in enumerate(win_data):
        m = len(rows)
        P_pose[wi, :m] = pose_ids[rows] - s
        P_point[wi, :m] = local_pid
        P_ray[wi, :m] = ray_cam[rows]
        P_rel[wi, :m] = rel_all[rows]
        P_w[wi, :m] = 1.0
        if m < n_obs_pad:  # pad by repeating row 0 with weight 0
            P_pose[wi, m:] = P_pose[wi, 0]; P_point[wi, m:] = P_point[wi, 0]
            P_ray[wi, m:] = P_ray[wi, 0]; P_rel[wi, m:] = P_rel[wi, 0]
        # Rotations: slice of ONE global IMU chain (built below) -- per-window
        # gravity re-seeding was noisy during fast motion (low gravity
        # confidence) and produced 10-15 deg window rotation errors.
        R_init[wi] = rot_chain[s:s + W]

    # Every window has the same (W, n_pts_pad, n_obs_pad) shape. Problems can
    # therefore share one compiled solve and optionally run in vmap chunks.
    t0 = time.time()
    quat_per_obs = np.take_along_axis(R_init, P_pose[..., None], axis=1)  # (N,obs,4)

    def make_problem(wi):
        centers = CamCenterVar(id=jnp.arange(W))
        points = Point3Var(id=jnp.arange(n_pts_pad))
        s = win_data[wi][0]
        npts = win_data[wi][3]
        costs = [jaxls.Cost(
            _win_residual,
            (CamCenterVar(id=jnp.asarray(P_pose[wi])),
             Point3Var(id=jnp.asarray(P_point[wi])),
             jnp.asarray(quat_per_obs[wi]),
             jnp.asarray(P_ray[wi]),
             jnp.asarray(P_rel[wi]),
             jnp.asarray(P_w[wi]),
             jnp.asarray(args.robust_scale, dtype=jnp.float32)),
        )]
        if args.positive_depth_weight > 0:
            costs.append(jaxls.Cost(
                _center_positive_depth,
                (CamCenterVar(id=jnp.asarray(P_pose[wi])),
                 Point3Var(id=jnp.asarray(P_point[wi])),
                 jnp.asarray(quat_per_obs[wi]),
                 jnp.asarray(P_ray[wi]),
                 jnp.asarray(P_rel[wi]),
                 jnp.asarray(args.positive_depth_min, dtype=jnp.float32),
                 jnp.asarray(args.positive_depth_softness, dtype=jnp.float32),
                 jnp.asarray(
                     P_w[wi, :, None] * args.positive_depth_weight,
                     dtype=jnp.float32)),
            ))
        pad_weight = (np.arange(n_pts_pad) >= npts).astype(np.float32)[:, None]
        costs.append(jaxls.Cost(
            _point_padding_anchor,
            (Point3Var(id=jnp.arange(n_pts_pad)),
             jnp.asarray(pad_weight)),
        ))
        if args.constant_velocity_weight > 0:
            dt = np.diff(solver_time[s:s + W]).astype(np.float32)
            dt_prev, dt_next = dt[:-1], dt[1:]
            dt_ref = np.float32(1.0 / native_fps)
            time_scale = dt_ref * np.sqrt(
                dt_ref / (0.5 * (dt_prev + dt_next)))
            costs.append(jaxls.Cost(
                _center_constant_velocity,
                (CamCenterVar(id=jnp.arange(W - 2)),
                 CamCenterVar(id=jnp.arange(1, W - 1)),
                 CamCenterVar(id=jnp.arange(2, W)),
                 jnp.asarray(dt_prev),
                 jnp.asarray(dt_next),
                 jnp.asarray(time_scale),
                 jnp.asarray(args.constant_velocity_weight,
                             dtype=jnp.float32)),
            ))
        if args.gauge_weight > 0:
            costs.append(jaxls.Cost(
                _center_gauge,
                (CamCenterVar(id=jnp.asarray(0)),
                 jnp.asarray(args.gauge_weight, dtype=jnp.float32)),
            ))
        problem = jaxls.LeastSquaresProblem(
            costs, [centers, points]).analyze(schur_elimination="off")
        seed = window_ids[wi] if args.init_seed is None else args.init_seed
        key = jax.random.PRNGKey(seed)
        k1, k2 = jax.random.split(key)
        point_init = (
            jax.random.normal(k2, (n_pts_pad, 3)) * 0.5
            + jnp.array([0, 0, 1.0])
        )
        point_init = point_init.at[npts:].set(0.0)
        vals0 = jaxls.VarValues.make([
            centers.with_value(jax.random.normal(k1, (W, 3)) * 0.1),
            points.with_value(point_init),
        ])
        return problem, vals0

    pairs = [make_problem(wi) for wi in range(N)]
    print(f"[timing] build+analyze {N} windows: {time.time() - t0:.2f}s")

    def _solve_positioning(prob, v0):
        initial_cost = jnp.sum(prob.compute_residual_vector(v0) ** 2)
        sol, summary = prob.solve(
            v0, linear_solver=args.linear_solver,
            trust_region=jaxls.TrustRegionConfig(),
            termination=jaxls.TerminationConfig(
                max_iterations=args.iters,
                early_termination=args.early_termination,
                cost_tolerance=args.termination_cost_tol,
                gradient_tolerance=args.termination_gradient_tol,
                gradient_tolerance_start_step=(
                    args.termination_gradient_start),
                parameter_tolerance=args.termination_parameter_tol),
            verbose=False, return_summary=True)
        final_cost = jnp.sum(prob.compute_residual_vector(sol) ** 2)
        improved = jnp.isfinite(final_cost) & (final_cost <= initial_cost)
        centers = jnp.where(improved, sol[CamCenterVar], v0[CamCenterVar])
        points = jnp.where(improved, sol[Point3Var], v0[Point3Var])
        final_cost = jnp.where(improved, final_cost, initial_cost)
        return (centers, points, initial_cost, final_cost,
                summary.cost_history, summary.iterations,
                summary.termination_criteria, summary.termination_deltas)

    def make_refine(wi, centers_np, points_np):
        """Per-window SE3 refine: rotations free, tethered by IMU relative
        rotation + gravity (same recipe as vio_bundle_adjust stage 2). Same
        module-level residuals + schur off + fixed shapes -> compile once."""
        poses = jaxls.SE3Var(id=jnp.arange(W))
        points = Point3Var(id=jnp.arange(n_pts_pad))
        s = win_data[wi][0]
        npts = win_data[wi][3]
        pose_init = jax.vmap(
            lambda q, c: jaxlie.SE3.from_rotation_and_translation(
                jaxlie.SO3(q), -(jaxlie.SO3(q) @ c))
        )(jnp.asarray(R_init[wi]), jnp.asarray(centers_np))
        costs = [
            jaxls.Cost(_refine_positioning,
                       (jaxls.SE3Var(id=jnp.asarray(P_pose[wi])),
                        Point3Var(id=jnp.asarray(P_point[wi])),
                        jnp.asarray(P_ray[wi]),
                        jnp.asarray(P_rel[wi]),
                        jnp.asarray(P_w[wi]),
                        jnp.asarray(args.robust_scale, dtype=jnp.float32))),
            jaxls.Cost(_refine_imu_rot,
                       (jaxls.SE3Var(id=jnp.arange(W - 1)),
                        jaxls.SE3Var(id=jnp.arange(1, W)),
                        jnp.asarray(delta_prev[s + 1:s + W], dtype=jnp.float32),
                        jnp.asarray(args.imu_rot_weight, dtype=jnp.float32))),
            jaxls.Cost(_refine_gravity,
                       (jaxls.SE3Var(id=jnp.arange(W)),
                        jnp.asarray(gravity_cam[s:s + W], dtype=jnp.float32),
                        jnp.asarray(args.gravity_weight * gravity_weight_arr[s:s + W],
                                    dtype=jnp.float32)[:, None])),
        ]
        if args.positive_depth_weight > 0:
            costs.append(jaxls.Cost(
                _pose_positive_depth,
                (jaxls.SE3Var(id=jnp.asarray(P_pose[wi])),
                 Point3Var(id=jnp.asarray(P_point[wi])),
                 jnp.asarray(P_ray[wi]),
                 jnp.asarray(P_rel[wi]),
                 jnp.asarray(args.positive_depth_min, dtype=jnp.float32),
                 jnp.asarray(args.positive_depth_softness, dtype=jnp.float32),
                 jnp.asarray(
                     P_w[wi, :, None] * args.positive_depth_weight,
                     dtype=jnp.float32)),
            ))
        pad_weight = (np.arange(n_pts_pad) >= npts).astype(np.float32)[:, None]
        costs.append(jaxls.Cost(
            _point_padding_anchor,
            (Point3Var(id=jnp.arange(n_pts_pad)),
             jnp.asarray(pad_weight)),
        ))
        if args.constant_velocity_weight > 0:
            dt = np.diff(solver_time[s:s + W]).astype(np.float32)
            dt_prev, dt_next = dt[:-1], dt[1:]
            dt_ref = np.float32(1.0 / native_fps)
            time_scale = dt_ref * np.sqrt(
                dt_ref / (0.5 * (dt_prev + dt_next)))
            costs.append(jaxls.Cost(
                _pose_constant_velocity,
                (jaxls.SE3Var(id=jnp.arange(W - 2)),
                 jaxls.SE3Var(id=jnp.arange(1, W - 1)),
                 jaxls.SE3Var(id=jnp.arange(2, W)),
                 jnp.asarray(dt_prev),
                 jnp.asarray(dt_next),
                 jnp.asarray(time_scale),
                 jnp.asarray(args.constant_velocity_weight,
                             dtype=jnp.float32)),
            ))
        if args.gauge_weight > 0:
            costs.append(jaxls.Cost(
                _pose_gauge,
                (jaxls.SE3Var(id=jnp.asarray(0)),
                 pose_init.wxyz_xyz[0],
                 jnp.asarray(args.gauge_weight, dtype=jnp.float32)),
            ))
        problem = jaxls.LeastSquaresProblem(
            costs, [poses, points]).analyze(schur_elimination="off")
        vals0 = jaxls.VarValues.make([
            poses.with_value(pose_init),
            points.with_value(jnp.asarray(points_np)),
        ])
        return problem, vals0

    def _solve_refine(prob, v0):
        initial_cost = jnp.sum(prob.compute_residual_vector(v0) ** 2)
        sol, summary = prob.solve(
            v0, linear_solver=args.linear_solver,
            trust_region=jaxls.TrustRegionConfig(),
            termination=jaxls.TerminationConfig(
                max_iterations=args.refine_iters,
                early_termination=args.early_termination,
                cost_tolerance=args.termination_cost_tol,
                gradient_tolerance=args.termination_gradient_tol,
                gradient_tolerance_start_step=(
                    args.termination_gradient_start),
                parameter_tolerance=args.termination_parameter_tol),
            verbose=False, return_summary=True)
        final_cost = jnp.sum(prob.compute_residual_vector(sol) ** 2)
        improved = jnp.isfinite(final_cost) & (final_cost <= initial_cost)
        poses = jnp.where(
            improved, sol[jaxls.SE3Var].wxyz_xyz, v0[jaxls.SE3Var].wxyz_xyz)
        points = jnp.where(improved, sol[Point3Var], v0[Point3Var])
        final_cost = jnp.where(improved, final_cost, initial_cost)
        return (poses, points, initial_cost, final_cost,
                summary.cost_history, summary.iterations,
                summary.termination_criteria, summary.termination_deltas)

    (c_all, p_all, pos_initial, pos_final, pos_history, pos_iterations,
     pos_termination_criteria, pos_termination_deltas) = _solve_in_chunks(
         pairs, _solve_positioning, args.solve_chunk_size, "positioning")
    pos_ratio = pos_final / np.maximum(pos_initial, 1e-12)
    print("positioning convergence: median final/initial "
          f"{np.median(pos_ratio):.4f}, worst {np.max(pos_ratio):.4f}")

    if args.refine_iters > 0:
        t0 = time.time()
        refine_pairs = [
            make_refine(wi, c_all[wi], p_all[wi]) for wi in range(N)
        ]
        print(f"[timing] build+analyze refine: {time.time() - t0:.2f}s")
        (pose_all, p_all, ref_initial, ref_final, ref_history, ref_iterations,
         ref_termination_criteria,
         ref_termination_deltas) = _solve_in_chunks(
             refine_pairs, _solve_refine, args.solve_chunk_size, "refine")
        q_all = pose_all[..., :4]
        q_all /= np.maximum(np.linalg.norm(q_all, axis=-1, keepdims=True),
                            1e-12)
        t_all = pose_all[..., 4:]
        R_all = np.asarray(jax.vmap(jax.vmap(
            lambda q: jaxlie.SO3(q).as_matrix()))(jnp.asarray(q_all)))
        c_all = -np.einsum("nwji,nwj->nwi", R_all, t_all)
        ref_ratio = ref_final / np.maximum(ref_initial, 1e-12)
        print("refine convergence: median final/initial "
              f"{np.median(ref_ratio):.4f}, worst {np.max(ref_ratio):.4f}")
    else:
        q_all = R_init
        ref_initial = ref_final = np.zeros(N)
        ref_history = np.zeros((N, 0))
        ref_iterations = np.zeros(N, dtype=np.int32)
        ref_termination_criteria = np.zeros((N, 3), dtype=bool)
        ref_termination_deltas = np.zeros((N, 3))

    point_med_ang_all = []
    point_positive_depth_frac_all = []
    for wi, (_, rows, local_pid, npts, _) in enumerate(win_data):
        med_angle, positive_frac = _point_diagnostics(
            q_all[wi], c_all[wi], p_all[wi],
            P_pose[wi, :len(rows)], local_pid,
            P_ray[wi, :len(rows)], P_rel[wi, :len(rows)], npts)
        point_med_ang_all.append(med_angle)
        point_positive_depth_frac_all.append(positive_frac)

    # A diagnostic single-window export uses local indexing while retaining
    # the original frame IDs and the global IMU-chain initialization.
    if args.window_index is None:
        stitch_starts = starts
        output_n_frames = n_frames
        output_frame_idx = frame_idx
    else:
        stitch_starts = [0]
        output_n_frames = W
        output_frame_idx = frame_idx[starts[0]:starts[0] + W]

    # Compute every stitch from the same window solutions for a free A/B.
    stitched = {}
    npts_all = np.asarray([x[3] for x in win_data])
    for mode in ("first", "last", "mean", "gravity"):
        stitched[mode] = _stitch_windows(
            stitch_starts, q_all, c_all, p_all, npts_all,
            output_n_frames, mode)
        metrics = stitched[mode][3]
        if len(metrics["seam_pos"]):
            print(f"stitch {mode}: seam pos p50/max "
                  f"{1e3*np.median(metrics['seam_pos']):.1f}/"
                  f"{1e3*np.max(metrics['seam_pos']):.1f} mm, "
                  f"rot p50/max {np.median(metrics['seam_rot']):.2f}/"
                  f"{np.max(metrics['seam_rot']):.2f} deg")

    centers_glob, R_glob, merged_pts, stitch_metrics = stitched[args.stitch_mode]
    print(f"[timing] wall total {time.time() - t_wall0:.1f}s")

    interpolate_output = (
        args.window_index is None and n_frames < n_frames_native)
    if interpolate_output:
        centers_glob, R_glob = _interpolate_pose_samples(
            centers_glob, R_glob, sample_pos, n_frames_native)
        output_frame_idx = frame_idx_native

    # Keep poses in the stitched world frame used by merged_pts. Recentered
    # poses with unrecentered points render with a constant trajectory offset.
    t_out = np.asarray(jax.vmap(
        lambda q, c: -(jaxlie.SO3(q) @ c)
    )(jnp.asarray(R_glob), jnp.asarray(centers_glob)))
    poses = np.concatenate([R_glob, t_out], axis=1)
    copy_pts = np.concatenate(merged_pts, 0)

    extra_poses = {}
    for mode, (centers_mode, quats_mode, _, _) in stitched.items():
        if interpolate_output:
            centers_mode, quats_mode = _interpolate_pose_samples(
                centers_mode, quats_mode, sample_pos, n_frames_native)
        t_mode = np.asarray(jax.vmap(
            lambda q, c: -(jaxlie.SO3(q) @ c)
        )(jnp.asarray(quats_mode), jnp.asarray(centers_mode)))
        extra_poses[f"pose_wxyz_xyz_{mode}"] = np.concatenate(
            [quats_mode, t_mode], axis=1)

    # Attach diagnostics and video observations to each window-local estimate,
    # then robustly merge repeated estimates of the same source track.
    copy_track_id = []
    copy_first_frame = []
    copy_first_is_right = []
    copy_first_px = []
    copy_alive = []
    for (_, rows, local_pid, npts, track_ids), med_angle, positive_frac in zip(
            win_data, point_med_ang_all, point_positive_depth_frac_all):
        first = np.full(npts, -1, dtype=np.int64)
        for local_row, pid in enumerate(local_pid):
            if first[pid] < 0:
                first[pid] = rows[local_row]
        assert np.all(first >= 0)
        copy_track_id.append(track_ids)
        copy_first_frame.append(frame_idx[pose_ids[first]])
        copy_first_is_right.append(obs_right[first])
        copy_first_px.append(obs_px[first])
        copy_alive.append(
            (np.bincount(local_pid, minlength=npts) >= 2)
            & (med_angle <= args.max_point_med_ang)
            & (positive_frac >= args.min_positive_depth_frac))

    copy_track_id = np.concatenate(copy_track_id)
    copy_first_frame = np.concatenate(copy_first_frame)
    copy_first_is_right = np.concatenate(copy_first_is_right)
    copy_first_px = np.concatenate(copy_first_px)
    copy_med_ang = np.concatenate(point_med_ang_all)
    copy_positive_depth_frac = np.concatenate(point_positive_depth_frac_all)
    copy_alive = np.concatenate(copy_alive)

    point_track_id, inverse = np.unique(copy_track_id, return_inverse=True)
    n_merged = len(point_track_id)
    pts = np.zeros((n_merged, 3))
    point_first_frame = np.zeros(n_merged, dtype=frame_idx.dtype)
    point_first_is_right = np.zeros(n_merged, bool)
    point_first_px = np.zeros((n_merged, 2), dtype=obs_px.dtype)
    point_alive = np.zeros(n_merged, bool)
    point_med_ang = np.zeros(n_merged)
    point_positive_depth_frac = np.zeros(n_merged)
    point_window_count = np.bincount(inverse, minlength=n_merged)
    copy_order = np.argsort(inverse, kind="stable")
    inverse_sorted = inverse[copy_order]
    copy_bounds = np.searchsorted(inverse_sorted, np.arange(n_merged + 1))
    for pid in range(n_merged):
        copies = copy_order[copy_bounds[pid]:copy_bounds[pid + 1]]
        good = copies[copy_alive[copies]]
        used = good if len(good) else copies
        pts[pid] = np.median(copy_pts[used], axis=0)
        point_alive[pid] = len(good) > 0
        point_med_ang[pid] = np.median(copy_med_ang[used])
        point_positive_depth_frac[pid] = np.median(
            copy_positive_depth_frac[used])
        first_copy = copies[np.argmin(copy_first_frame[copies])]
        point_first_frame[pid] = copy_first_frame[first_copy]
        point_first_is_right[pid] = copy_first_is_right[first_copy]
        point_first_px[pid] = copy_first_px[first_copy]
    print(f"point cloud: {len(copy_pts)} window-local estimates -> "
          f"{n_merged} unique tracks; {point_alive.sum()} pass "
          f"{args.max_point_med_ang:g} deg/depth filtering")

    np.savez(
        out_path,
        frame_idx=output_frame_idx,
        pose_wxyz_xyz=poses,
        points=pts,
        point_first_frame=point_first_frame,
        point_first_is_right=point_first_is_right,
        point_first_px=point_first_px,
        point_alive=point_alive,
        point_med_ang=point_med_ang,
        point_positive_depth_frac=point_positive_depth_frac,
        point_track_id=point_track_id,
        point_window_count=point_window_count,
        window_starts=np.asarray(starts),
        window_start_frame_idx=frame_idx[np.asarray(starts)],
        solver_frame_idx=frame_idx,
        window_quat_wxyz=q_all.astype(np.float32),
        window_centers=c_all.astype(np.float32),
        window_points=p_all.astype(np.float32),
        window_point_count=npts_all,
        window_observation_count=np.asarray(
            [len(window[1]) for window in win_data]),
        positioning_initial_cost=pos_initial,
        positioning_final_cost=pos_final,
        positioning_cost_history=pos_history,
        positioning_iterations=pos_iterations,
        positioning_termination_criteria=pos_termination_criteria,
        positioning_termination_deltas=pos_termination_deltas,
        refine_initial_cost=ref_initial,
        refine_final_cost=ref_final,
        refine_cost_history=ref_history,
        refine_iterations=ref_iterations,
        refine_termination_criteria=ref_termination_criteria,
        refine_termination_deltas=ref_termination_deltas,
        config_positive_depth_weight=np.asarray(args.positive_depth_weight),
        config_constant_velocity_weight=np.asarray(
            args.constant_velocity_weight),
        config_robust_scale=np.asarray(args.robust_scale),
        config_native_fps=np.asarray(native_fps),
        config_target_fps=np.asarray(args.target_fps),
        **{f"stitch_{k}": v for k, v in stitch_metrics.items()},
        **extra_poses,
    )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
