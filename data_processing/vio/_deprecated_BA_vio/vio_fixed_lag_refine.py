"""Alternating cross-window landmark fusion and fixed-lag pose refinement.

This is a second pass over a stitched windowed-BA trajectory:
  1. robustly retriangulate each unique landmark from observations spanning
     all chunk boundaries, with camera poses fixed;
  2. refine overlapping pose blocks against those fixed landmarks, with IMU
     factors to the frozen pose on either side;
  3. retriangulate and run a reverse pose pass.

Only fixed-size local pose problems are given to jaxls, so dense Cholesky and
JIT compilation remain independent of recording length.
"""
import argparse
import json
import os
import time

import jax
import jax.numpy as jnp
import jaxlie
import jaxls
import numpy as np

from vio_bundle_adjust import (
    load_intrinsics,
    load_stereo,
    load_tracks,
    unproject_all,
)


def quat_to_matrix(q):
    q = np.asarray(q)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack([
        np.stack([1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)], -1),
        np.stack([2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)], -1),
        np.stack([2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)], -1),
    ], axis=-2)


def poses_to_centers(poses):
    R = quat_to_matrix(poses[:, :4])
    return -np.einsum("nji,nj->ni", R, poses[:, 4:])


def centers_to_poses(quats, centers):
    R = quat_to_matrix(quats)
    t = -np.einsum("nij,nj->ni", R, centers)
    return np.concatenate([quats, t], axis=1)


def time_spread_rows(rows, frame_ids, is_right, cap):
    if len(rows) <= cap:
        return rows
    frames = frame_ids[rows]
    chosen = [0, len(rows) - 1]
    midpoint = 0.5 * (frames[0] + frames[-1])
    for right in (False, True):
        eye = np.flatnonzero(is_right[rows] == right)
        if len(eye):
            chosen.append(int(eye[np.argmin(np.abs(frames[eye] - midpoint))]))
    chosen = list(dict.fromkeys(chosen))
    while len(chosen) < cap:
        distance = np.min(
            np.abs(frames[:, None] - frames[np.asarray(chosen)][None]), axis=1)
        distance[np.asarray(chosen)] = -1
        best = int(np.argmax(distance))
        if distance[best] <= 0:
            unused = np.setdiff1d(np.arange(len(rows)), np.asarray(chosen))
            if not len(unused):
                break
            best = int(unused[len(unused) // 2])
        chosen.append(best)
    return rows[np.sort(chosen[:cap])]


def build_observations(recording, trajectory, device, starts, length,
                       point_alive, max_obs_per_point, obs_per_frame):
    frame_idx = trajectory["frame_idx"]
    frame_to_pose = {int(frame): i for i, frame in enumerate(frame_idx)}
    point_track_id = trajectory["point_track_id"].astype(np.int64)
    track_to_point = {int(track): i for i, track in enumerate(point_track_id)}

    import h5py
    with h5py.File(os.path.join(recording, "derived", "features.h5"), "r") as f:
        left_serial, right_serial = f.attrs["left_serial"], f.attrs["right_serial"]
    Kl, Dl = load_intrinsics(recording, left_serial)
    Kr, Dr = load_intrinsics(recording, right_serial)
    R_st, t_st = load_stereo(recording, left_serial, right_serial)

    tracks = load_tracks(
        os.path.join(recording, "derived", "tracks.jsonl"), int(frame_idx[-1]))
    pose_id, point_id, px, right = [], [], [], []
    for track_id, output_id in track_to_point.items():
        for eye, frame, pixel in tracks[track_id]:
            p = frame_to_pose.get(int(frame))
            if p is None:
                continue
            pose_id.append(p)
            point_id.append(output_id)
            px.append(pixel)
            right.append(eye == "right")
    pose_id = np.asarray(pose_id, np.int32)
    point_id = np.asarray(point_id, np.int32)
    px = np.asarray(px, np.float32)
    right = np.asarray(right, bool)
    order = np.argsort(pose_id, kind="stable")
    pose_id, point_id, px, right = (
        pose_id[order], point_id[order], px[order], right[order])

    point_order = np.argsort(point_id, kind="stable")
    point_sorted = point_id[point_order]
    point_bounds = np.searchsorted(
        point_sorted, np.arange(len(point_track_id) + 1))
    point_pose_min = np.full(len(point_track_id), len(frame_idx), np.int32)
    point_pose_max = np.full(len(point_track_id), -1, np.int32)
    for pid in range(len(point_track_id)):
        rows = point_order[point_bounds[pid]:point_bounds[pid + 1]]
        if len(rows):
            point_pose_min[pid] = np.min(pose_id[rows])
            point_pose_max[pid] = np.max(pose_id[rows])
    pose_lo = np.searchsorted(pose_id, np.arange(len(frame_idx) + 1))
    metadata = {
        "pose_id": pose_id,
        "point_id": point_id,
        "px": px,
        "right": right,
        "point_order": point_order,
        "point_bounds": point_bounds,
        "point_pose_min": point_pose_min,
        "point_pose_max": point_pose_max,
        "pose_lo": pose_lo,
    }

    needed = []
    for pid in range(len(point_track_id)):
        if not point_alive[pid]:
            continue
        lo, hi = point_bounds[pid:pid + 2]
        rows = point_order[lo:hi]
        if len(rows):
            needed.append(time_spread_rows(
                rows, pose_id, right, max_obs_per_point))
    for start in starts:
        needed.append(select_pose_rows(
            metadata, point_alive, start, length, obs_per_frame))
    needed = np.unique(np.concatenate(needed))
    print(f"retained {len(needed)}/{len(pose_id)} cross-boundary "
          "observations before unprojection")
    pose_id, point_id, px, right = (
        pose_id[needed], point_id[needed], px[needed], right[needed])

    rays = unproject_all(
        {"left": px[~right], "right": px[right]},
        {"left": (Kl, Dl), "right": (Kr, Dr)}, device)
    ray_cam = np.zeros((len(pose_id), 3), np.float32)
    ray_cam[~right] = rays["left"]
    ray_cam[right] = rays["right"]

    rel_left = np.asarray(jaxlie.SE3.identity().wxyz_xyz)
    rel_right = np.asarray(jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3.from_matrix(R_st), t_st).wxyz_xyz)
    rel = np.where(right[:, None], rel_right[None], rel_left[None])
    point_order = np.argsort(point_id, kind="stable")
    point_sorted = point_id[point_order]
    point_bounds = np.searchsorted(
        point_sorted, np.arange(len(point_track_id) + 1))
    pose_lo = np.searchsorted(pose_id, np.arange(len(frame_idx) + 1))
    return {
        "pose_id": pose_id,
        "point_id": point_id,
        "px": px,
        "right": right,
        "ray_cam": ray_cam,
        "rel": rel,
        "point_order": point_order,
        "point_bounds": point_bounds,
        "point_pose_min": point_pose_min,
        "point_pose_max": point_pose_max,
        "pose_lo": pose_lo,
    }


def world_rays(poses, obs):
    R_wl = quat_to_matrix(poses[:, :4])
    centers_left = poses_to_centers(poses)
    R_rel = quat_to_matrix(obs["rel"][:, :4])
    R_wc = np.einsum("nij,njk->nik", R_rel, R_wl[obs["pose_id"]])
    centers = (
        centers_left[obs["pose_id"]]
        - np.einsum("nji,nj->ni", R_wc, obs["rel"][:, 4:]))
    rays = np.einsum("nji,nj->ni", R_wc, obs["ray_cam"])
    rays /= np.maximum(np.linalg.norm(rays, axis=1, keepdims=True), 1e-12)
    return centers, rays


def retriangulate(poses, points, point_alive, obs, robust_scale,
                  max_obs_per_point, max_med_angle):
    centers, rays = world_rays(poses, obs)
    output = points.copy()
    med_angle = np.full(len(points), np.inf)
    positive_frac = np.zeros(len(points))
    conditioned = np.zeros(len(points), bool)
    I = np.eye(3)

    for pid in range(len(points)):
        lo, hi = obs["point_bounds"][pid:pid + 2]
        rows = obs["point_order"][lo:hi]
        if len(rows) < 2:
            continue
        rows = time_spread_rows(
            rows, obs["pose_id"], obs["right"], max_obs_per_point)
        c, d = centers[rows], rays[rows]
        A = I[None] - d[:, :, None] * d[:, None, :]
        x = points[pid].copy()
        valid_system = False
        for _ in range(8):
            v = x[None] - c
            distance = np.maximum(np.linalg.norm(v, axis=1), 1e-6)
            pred = v / distance[:, None]
            forward = np.einsum("ni,ni->n", pred, d)
            residual = np.sqrt(np.maximum(1.0 - forward ** 2, 0.0))
            residual = np.where(forward > 0, residual, 1.0)
            robust = 1.0 / (1.0 + (residual / robust_scale) ** 2)
            weight = robust / np.clip(distance, 0.1, 20.0) ** 2
            H = np.einsum("n,nij->ij", weight, A)
            b = np.einsum("n,nij,nj->i", weight, A, c)
            eig = np.linalg.eigvalsh(H)
            valid_system = eig[-1] > 1e-9 and eig[0] / eig[-1] > 1e-5
            if not valid_system:
                break
            x_new = np.linalg.solve(H, b)
            if np.linalg.norm(x_new - x) < 1e-6:
                x = x_new
                break
            x = x_new

        v = x[None] - c
        distance = np.maximum(np.linalg.norm(v, axis=1), 1e-12)
        pred = v / distance[:, None]
        cosine = np.einsum("ni,ni->n", pred, d)
        angle = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
        med_angle[pid] = np.median(angle)
        positive_frac[pid] = np.mean(cosine > 0)
        conditioned[pid] = valid_system
        if valid_system:
            output[pid] = x

    alive = (
        point_alive & conditioned & (med_angle <= max_med_angle)
        & (positive_frac >= 0.75))
    print(f"retriangulation: {alive.sum()}/{len(points)} landmarks pass; "
          f"median angle {np.median(med_angle[alive]):.3f} deg")
    return output, alive, med_angle, positive_frac


def select_pose_rows(obs, point_alive, start, length, per_frame):
    selected = []
    point_id = obs["point_id"]
    px = obs["px"]
    crosses = (
        (obs["point_pose_min"] < start)
        | (obs["point_pose_max"] >= start + length))
    for pose in range(start, start + length):
        lo, hi = obs["pose_lo"][pose:pose + 2]
        rows = np.arange(lo, hi)
        rows = rows[point_alive[point_id[rows]]]
        if len(rows) <= per_frame:
            selected.append(rows)
            continue

        # Round-robin image cells keeps each pose constrained across the FOV.
        width = max(float(np.max(px[rows, 0]) + 1), 1.0)
        height = max(float(np.max(px[rows, 1]) + 1), 1.0)
        gx = np.clip((8 * px[rows, 0] / width).astype(int), 0, 7)
        gy = np.clip((6 * px[rows, 1] / height).astype(int), 0, 5)
        cells = gx + 8 * gy
        groups = []
        for cell in range(48):
            group = rows[cells == cell]
            group_crosses = crosses[point_id[group]]
            groups.append(np.concatenate([
                group[group_crosses], group[~group_crosses]]))
        keep = []
        depth = 0
        while len(keep) < per_frame:
            added = False
            for group in groups:
                if depth < len(group):
                    keep.append(group[depth])
                    added = True
                    if len(keep) == per_frame:
                        break
            if not added:
                break
            depth += 1
        selected.append(np.asarray(keep, np.int64))
    return np.sort(np.concatenate(selected))


def bearing_cost(vals, pose_var, point, ray, rel, weight, robust_scale):
    T_wl = vals[pose_var]
    R_wl = T_wl.rotation()
    c_left = -(R_wl.inverse() @ T_wl.translation())
    T_rel = jaxlie.SE3(rel)
    R_wc = T_rel.rotation() @ R_wl
    cam_pos = c_left - (R_wc.inverse() @ T_rel.translation())
    ray_world = R_wc.inverse() @ ray
    direction = point - cam_pos
    direction /= jnp.linalg.norm(direction) + 1e-9
    forward = jnp.maximum(jnp.dot(ray_world, direction), 0.0)
    residual = ray_world - forward * direction
    norm = jnp.linalg.norm(residual) + 1e-9
    robust = jax.lax.stop_gradient(
        1.0 / (1.0 + (norm / robust_scale) ** 2))
    return residual * (weight * jnp.sqrt(robust))


def imu_cost(vals, pose_i, pose_j, delta, weight):
    R_i = vals[pose_i].rotation()
    R_j = vals[pose_j].rotation()
    return (
        jaxlie.SO3(delta).inverse() @ (R_i @ R_j.inverse())
    ).log() * weight


def imu_left_boundary(vals, pose_j, fixed_i, delta, weight):
    R_i = jaxlie.SO3(fixed_i)
    R_j = vals[pose_j].rotation()
    return (
        jaxlie.SO3(delta).inverse() @ (R_i @ R_j.inverse())
    ).log() * weight


def imu_right_boundary(vals, pose_i, fixed_j, delta, weight):
    R_i = vals[pose_i].rotation()
    R_j = jaxlie.SO3(fixed_j)
    return (
        jaxlie.SO3(delta).inverse() @ (R_i @ R_j.inverse())
    ).log() * weight


def gravity_cost(vals, pose_var, measured_down, weight):
    predicted = vals[pose_var].rotation() @ jnp.array([0.0, 0.0, -1.0])
    return (predicted - measured_down) * weight


def pose_anchor_cost(vals, pose_var, target, weight):
    return (jaxlie.SE3(target).inverse() @ vals[pose_var]).log() * weight


def make_pose_problem(poses, points, point_alive, obs, imu, start, length,
                      obs_per_frame, robust_scale, imu_weight, gravity_weight,
                      direction, overlap, anchor_weight):
    rows = select_pose_rows(obs, point_alive, start, length, obs_per_frame)
    n_obs_pad = length * obs_per_frame
    if not len(rows):
        raise ValueError(f"no usable observations in pose window at {start}")
    rows = rows[:n_obs_pad]
    m = len(rows)
    padded = np.pad(rows, (0, n_obs_pad - m), mode="edge")
    obs_weight = np.zeros(n_obs_pad, np.float32)
    obs_weight[:m] = 1.0

    pose_vars = jaxls.SE3Var(id=jnp.arange(length))
    if direction == "forward":
        anchor_ids = np.arange(overlap)
    else:
        anchor_ids = np.arange(length - overlap, length)
    costs = [
        jaxls.Cost(
            bearing_cost,
            (
                jaxls.SE3Var(id=jnp.asarray(
                    obs["pose_id"][padded] - start)),
                jnp.asarray(points[obs["point_id"][padded]], jnp.float32),
                jnp.asarray(obs["ray_cam"][padded], jnp.float32),
                jnp.asarray(obs["rel"][padded], jnp.float32),
                jnp.asarray(obs_weight),
                jnp.asarray(robust_scale, jnp.float32),
            ),
        ),
        jaxls.Cost(
            imu_cost,
            (
                jaxls.SE3Var(id=jnp.arange(length - 1)),
                jaxls.SE3Var(id=jnp.arange(1, length)),
                jnp.asarray(imu["delta_prev"][start + 1:start + length]),
                jnp.asarray(imu_weight, jnp.float32),
            ),
        ),
        jaxls.Cost(
            gravity_cost,
            (
                jaxls.SE3Var(id=jnp.arange(length)),
                jnp.asarray(imu["gravity_cam"][start:start + length]),
                jnp.asarray(
                    gravity_weight
                    * imu["gravity_weight"][start:start + length, None],
                    jnp.float32),
            ),
        ),
        jaxls.Cost(
            pose_anchor_cost,
            (
                jaxls.SE3Var(id=jnp.asarray(anchor_ids)),
                jnp.asarray(poses[start + anchor_ids], jnp.float32),
                jnp.asarray(anchor_weight, jnp.float32),
            ),
        ),
        jaxls.Cost(
            imu_left_boundary,
            (
                jaxls.SE3Var(id=jnp.asarray(0)),
                jnp.asarray(poses[max(start - 1, 0), :4], jnp.float32),
                jnp.asarray(
                    imu["delta_prev"][start] if start > 0
                    else np.array([1.0, 0.0, 0.0, 0.0]),
                    jnp.float32),
                jnp.asarray(imu_weight if start > 0 else 0.0, jnp.float32),
            ),
        ),
        jaxls.Cost(
            imu_right_boundary,
            (
                jaxls.SE3Var(id=jnp.asarray(length - 1)),
                jnp.asarray(
                    poses[min(start + length, len(poses) - 1), :4], jnp.float32),
                jnp.asarray(
                    imu["delta_prev"][start + length]
                    if start + length < len(poses)
                    else np.array([1.0, 0.0, 0.0, 0.0]),
                    jnp.float32),
                jnp.asarray(
                    imu_weight if start + length < len(poses) else 0.0,
                    jnp.float32),
            ),
        ),
    ]
    problem = jaxls.LeastSquaresProblem(
        costs, [pose_vars]).analyze(schur_elimination="off")
    values = jaxls.VarValues.make([
        pose_vars.with_value(jaxlie.SE3(jnp.asarray(
            poses[start:start + length], jnp.float32)))
    ])
    n_crossing = int(np.sum(
        (obs["point_pose_min"][obs["point_id"][rows]] < start)
        | (obs["point_pose_max"][obs["point_id"][rows]]
           >= start + length)))
    return problem, values, m, n_crossing


def solve_pose_problem(problem, values, iterations, max_translation_step,
                       max_rotation_step):
    initial = jnp.sum(problem.compute_residual_vector(values) ** 2)
    solution = problem.solve(
        values,
        linear_solver="dense_cholesky",
        trust_region=jaxls.TrustRegionConfig(),
        termination=jaxls.TerminationConfig(
            max_iterations=iterations, early_termination=False),
        verbose=False,
    )
    final = jnp.sum(problem.compute_residual_vector(solution) ** 2)
    delta = (
        values[jaxls.SE3Var].inverse() @ solution[jaxls.SE3Var]
    ).log()
    bounded = (
        (jnp.max(jnp.linalg.norm(delta[:, :3], axis=1))
         <= max_translation_step)
        & (jnp.max(jnp.linalg.norm(delta[:, 3:], axis=1))
           <= max_rotation_step))
    improved = jnp.isfinite(final) & (final <= initial) & bounded
    output = jnp.where(
        improved,
        solution[jaxls.SE3Var].wxyz_xyz,
        values[jaxls.SE3Var].wxyz_xyz)
    return output, initial, jnp.where(improved, final, initial), improved


def pose_pass(poses, points, point_alive, obs, imu, starts, length, direction,
              args, solve_jit):
    output = poses.copy()
    ordered_starts = starts if direction == "forward" else starts[::-1]
    ratios = []
    for index, start in enumerate(ordered_starts):
        problem, values, n_obs, n_crossing = make_pose_problem(
            output, points, point_alive, obs, imu, start, length,
            args.obs_per_frame, args.robust_scale,
            args.imu_rot_weight, args.gravity_weight, direction,
            args.overlap_frames, args.anchor_weight)
        t0 = time.time()
        block, initial, final, accepted = solve_jit(problem, values)
        jax.block_until_ready(block)
        output[start:start + length] = np.asarray(block)
        ratios.append(float(final / np.maximum(initial, 1e-12)))
        print(f"  {direction} {index + 1}/{len(starts)} start={start} "
              f"obs={n_obs} crossing={n_crossing} "
              f"accepted={bool(accepted)} ratio={ratios[-1]:.4f} "
              f"time={time.time() - t0:.2f}s")
    print(f"{direction} pose pass: median ratio={np.median(ratios):.4f}, "
          f"worst={np.max(ratios):.4f}")
    return output, np.asarray(ratios)


def load_imu(recording, frame_idx):
    raw = np.load(os.path.join(recording, "derived", "imu_relative.npz"))
    all_frames = raw["frame_idx"]
    lookup = {int(frame): i for i, frame in enumerate(all_frames)}
    source = np.asarray([lookup[int(frame)] for frame in frame_idx])
    gravity_cam = raw["gravity_cam"][source]
    gravity_weight = raw["gravity_weight"][source]
    delta_prev = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (len(frame_idx), 1))
    rel_quat = raw["rel_quat"]
    rel_valid = raw["rel_valid"]
    for i in range(1, len(frame_idx)):
        a, b = source[i - 1], source[i]
        if b == a + 1 and rel_valid[a]:
            delta_prev[i] = rel_quat[a]
    return {
        "gravity_cam": gravity_cam,
        "gravity_weight": gravity_weight,
        "delta_prev": delta_prev,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("recording")
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--window-s", type=float, default=4.0)
    parser.add_argument("--overlap-frames", type=int, default=30)
    parser.add_argument("--pose-iters", type=int, default=10)
    parser.add_argument("--passes", type=int, default=1,
                        help="number of forward+backward alternating passes")
    parser.add_argument("--obs-per-frame", type=int, default=160)
    parser.add_argument("--max-obs-per-point", type=int, default=48)
    parser.add_argument("--robust-scale", type=float, default=0.025)
    parser.add_argument("--max-point-med-ang", type=float, default=2.0)
    parser.add_argument("--imu-rot-weight", type=float, default=100.0)
    parser.add_argument("--gravity-weight", type=float, default=1.0)
    parser.add_argument("--anchor-weight", type=float, default=100.0,
                        help="SE(3) weight fixing the directional overlap")
    parser.add_argument("--max-translation-step", type=float, default=0.5,
                        help="reject a whole pose block if any step exceeds m")
    parser.add_argument("--max-rotation-step-deg", type=float, default=15.0,
                        help="reject a whole block if any pose step exceeds deg")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    t0 = time.time()
    source = np.load(args.trajectory)
    required = {"point_track_id", "point_alive"}
    missing = required - set(source.files)
    if missing:
        raise ValueError(f"trajectory missing fields: {sorted(missing)}")
    frame_idx = source["frame_idx"]
    poses = source["pose_wxyz_xyz"].copy()
    points = source["points"].copy()
    point_alive = source["point_alive"].astype(bool)

    fps = json.load(open(
        os.path.join(args.recording, "recording.json"))).get("fps", 30)
    length = max(8, int(round(args.window_s * fps)))
    stride = length - args.overlap_frames
    starts = list(range(0, max(len(frame_idx) - length, 0) + 1, stride))
    if starts[-1] + length < len(frame_idx):
        starts.append(len(frame_idx) - length)

    print("loading cross-boundary observations")
    obs = build_observations(
        args.recording, source, args.device, starts, length, point_alive,
        args.max_obs_per_point, args.obs_per_frame)
    imu = load_imu(args.recording, frame_idx)
    print(f"{len(frame_idx)} poses, {len(points)} landmarks, "
          f"{len(obs['pose_id'])} observations, {len(starts)} windows")

    pass_ratios = {}
    solve_jit = jax.jit(
        lambda problem, values: solve_pose_problem(
            problem, values, args.pose_iters, args.max_translation_step,
            np.radians(args.max_rotation_step_deg)))
    for pass_index in range(args.passes):
        points, point_alive, med_angle, positive_frac = retriangulate(
            poses, points, point_alive, obs, args.robust_scale,
            args.max_obs_per_point, args.max_point_med_ang)
        poses, ratios = pose_pass(
            poses, points, point_alive, obs, imu, starts, length,
            "forward", args, solve_jit)
        pass_ratios[f"forward_ratio_{pass_index}"] = ratios
        points, point_alive, med_angle, positive_frac = retriangulate(
            poses, points, point_alive, obs, args.robust_scale,
            args.max_obs_per_point, args.max_point_med_ang)
        poses, ratios = pose_pass(
            poses, points, point_alive, obs, imu, starts, length,
            "backward", args, solve_jit)
        pass_ratios[f"backward_ratio_{pass_index}"] = ratios

    points, point_alive, med_angle, positive_frac = retriangulate(
        poses, points, point_alive, obs, args.robust_scale,
        args.max_obs_per_point, args.max_point_med_ang)
    payload = {key: source[key] for key in source.files}
    payload.update({
        "pose_wxyz_xyz": poses,
        "points": points,
        "point_alive": point_alive,
        "point_med_ang": med_angle,
        "point_positive_depth_frac": positive_frac,
        **pass_ratios,
    })
    np.savez(args.out, **payload)
    print(f"wrote {args.out}; wall={time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
