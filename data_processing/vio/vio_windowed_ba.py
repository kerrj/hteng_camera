"""Windowed VIO solve: independent short-window positioning solves, stitched.

Rationale: the global stage-5 solve is superlinear in trajectory length (CG
iteration count blows up as LM damping decays on a long, gauge-free problem --
observed: ~50min/iter at 11k frames), but the downstream use only needs LOW
DRIFT OVER ~10s, not global consistency. So: solve fixed-size windows (default
3s) independently -- each is tiny, well-conditioned, and identical in shape so
jaxls JIT-compiles ONCE -- then chain windows by SE3-aligning each onto the
previous over their overlap (default 1s).

Per window it runs only the frozen-rotation positioning stage (rotations from
the IMU relative chain, gravity-aligned at the window start; GLOMAP bounded
positioning cost). No SE3 refine: rotations stay IMU-integrated, which the
full-solve validation showed tracks truth to ~1 deg; gyro drift over one 3s
window is negligible.

Fixed problem shape = fixed (n_obs_pad, n_points_pad) per window: observations
are subsampled/padded to the p95 window's size, extra slots get zero-weight
dummy costs. Landmarks are per-window variables (re-triangulated each window);
only poses are stitched.

Output: same trajectory.npz schema as vio_bundle_adjust.py (minus per-landmark
visualizer fields -- points are per-window and not globally consistent, so a
merged cloud is written for rough viz only).

Run:
    python vio_windowed_ba.py ../../testimu --tracks ../../testimu/tracks.jsonl \
        --out /tmp/traj_windowed.npz --window-s 3 --overlap-s 1
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

from vio_bundle_adjust import (Point3Var, CamCenterVar, ScaleVar,
                               positioning_cost, load_intrinsics, load_stereo,
                               load_tracks, unproject_all)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("recording")
    p.add_argument("--tracks", default=None, help="default: <recording>/derived/tracks.jsonl")
    p.add_argument("--imu-relative", default=None,
                    help="default: <recording>/derived/imu_relative.npz")
    p.add_argument("--out", default=None,
                    help="default: <recording>/derived/trajectory_windowed.npz")
    p.add_argument("--window-s", type=float, default=3.0)
    p.add_argument("--overlap-s", type=float, default=1.0)
    p.add_argument("--iters", type=int, default=20,
                    help="LM iterations per window (small window converges fast)")
    p.add_argument("--robust-scale", type=float, default=0.05)
    p.add_argument("--pad-quantile", type=float, default=50.0,
                    help="percentile of per-window obs counts used as the padded "
                         "problem size; windows above it are subsampled")
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

    fps = json.load(open(os.path.join(rec, "recording.json"))).get("fps", 30)
    W = max(8, int(round(args.window_s * fps)))
    V = max(2, int(round(args.overlap_s * fps)))
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
    frame_idx = frame_idx_all[keep]
    gravity_cam = gravity_cam_all[keep]
    n_frames = len(frame_idx)
    # delta_prev[i] rotates pose i-1 -> pose i (identity where edge invalid).
    delta_prev = np.tile(np.array([1.0, 0, 0, 0]), (n_frames, 1))
    new_pos = -np.ones(len(frame_idx_all), np.int64)
    new_pos[keep] = np.arange(n_frames)
    for e in range(len(rel_quat_all)):
        a, b = new_pos[e], new_pos[e + 1]
        if a >= 0 and b == a + 1 and rel_valid_all[e]:
            delta_prev[b] = rel_quat_all[e]

    frame_to_pose = {int(f): i for i, f in enumerate(frame_idx)}
    max_frame = int(frame_idx[-1])

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
    print(f"{n_frames} frames, {len(tracks)} tracks, window={W}f overlap={V}f")

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

    rays = unproject_all({"left": obs_px[~obs_right], "right": obs_px[obs_right]},
                          {"left": (Kl, Dl), "right": (Kr, Dr)}, args.device)
    ray_cam = np.zeros((len(pose_ids), 3))
    ray_cam[~obs_right] = rays["left"]; ray_cam[obs_right] = rays["right"]
    rel_all = np.where(obs_right[:, None], rel_right[None], rel_left[None])

    # Window starts; last window snapped to cover the tail.
    starts = list(range(0, max(n_frames - W, 0) + 1, stride))
    if starts[-1] + W < n_frames:
        starts.append(n_frames - W)
    row_lo = np.searchsorted(pose_ids, np.array(starts))
    row_hi = np.searchsorted(pose_ids, np.array(starts) + W)

    # Fixed padded sizes across windows -> one JIT compile.
    counts = row_hi - row_lo
    n_obs_pad = int(np.percentile(counts, args.pad_quantile))
    # points per window bounded by obs (each point >=2 obs)
    n_pts_pad = 0
    win_data = []
    rng = np.random.default_rng(0)
    for s, lo, hi in zip(starts, row_lo, row_hi):
        rows = np.arange(lo, hi)
        if len(rows) > n_obs_pad:
            rows = rng.choice(rows, n_obs_pad, replace=False)
        # local point reindex
        upts, local_pid = np.unique(point_ids[rows], return_inverse=True)
        n_pts_pad = max(n_pts_pad, len(upts))
        win_data.append((s, rows, local_pid, len(upts)))
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
    for wi, (s, rows, local_pid, npts) in enumerate(win_data):
        m = len(rows)
        P_pose[wi, :m] = pose_ids[rows] - s
        P_point[wi, :m] = local_pid
        P_ray[wi, :m] = ray_cam[rows]
        P_rel[wi, :m] = rel_all[rows]
        P_w[wi, :m] = 1.0
        if m < n_obs_pad:  # pad by repeating row 0 with weight 0
            P_pose[wi, m:] = P_pose[wi, 0]; P_point[wi, m:] = P_point[wi, 0]
            P_ray[wi, m:] = P_ray[wi, 0]; P_rel[wi, m:] = P_rel[wi, 0]
        # IMU rotation chain seeded gravity-aligned at the window start.
        g0 = gravity_cam[s] / (np.linalg.norm(gravity_cam[s]) + 1e-12)
        a = np.array([0.0, 0, -1])
        q0 = np.array([1.0 + a @ g0, *np.cross(a, g0)])
        if np.linalg.norm(q0) < 1e-6:
            q0 = np.array([0.0, 1, 0, 0])
        q = q0 / np.linalg.norm(q0)
        R_init[wi, 0] = q
        Rk = jaxlie.SO3(jnp.asarray(q))
        for j in range(1, W):
            Rk = jaxlie.SO3(jnp.asarray(delta_prev[s + j])).inverse() @ Rk
            R_init[wi, j] = np.asarray(Rk.wxyz)

    # --- per-window solve, vmapped-free but jit-cached (same shapes) ---------
    @jaxls.Cost.factory
    def win_positioning_cost(vals, center, point_var, scale_var, pose_quat,
                             ray, rel, w, robust_scale):
        T_rel = jaxlie.SE3(rel)
        R_wl = jaxlie.SO3(pose_quat)
        R_wc = T_rel.rotation() @ R_wl
        cam_pos = vals[center] - (R_wc.inverse() @ T_rel.translation())
        ray_w = R_wc.inverse() @ ray
        d = jnp.exp(vals[scale_var])
        r = ray_w - d * (vals[point_var] - cam_pos)
        abs_r = jnp.linalg.norm(r) + 1e-9
        wr = jax.lax.stop_gradient(1.0 / (1.0 + (abs_r / robust_scale) ** 2))
        return r * (w * jnp.sqrt(wr))

    # ALL windows in ONE block-diagonal problem: window wi's variables live at
    # id offsets wi*W / wi*n_pts_pad / wi*n_obs_pad. Windows share no variables
    # (block-diagonal), so this is exactly N independent solves -- but with ONE
    # analyze, ONE compile, and the GPU working on all windows concurrently.
    t0 = time.time()
    off_pose = (np.arange(N)[:, None] * W + P_pose).reshape(-1)
    off_point = (np.arange(N)[:, None] * n_pts_pad + P_point).reshape(-1)
    off_scale = np.arange(N * n_obs_pad)
    quat_flat = np.take_along_axis(R_init, P_pose[..., None], axis=1).reshape(-1, 4)

    centers = CamCenterVar(id=jnp.arange(N * W))
    points = Point3Var(id=jnp.arange(N * n_pts_pad))
    scales = ScaleVar(id=jnp.arange(N * n_obs_pad))
    costs = [win_positioning_cost(
        CamCenterVar(id=jnp.asarray(off_pose)),
        Point3Var(id=jnp.asarray(off_point)),
        ScaleVar(id=jnp.asarray(off_scale)),
        jnp.asarray(quat_flat),
        jnp.asarray(P_ray.reshape(-1, 3)),
        jnp.asarray(P_rel.reshape(-1, 7)),
        jnp.asarray(P_w.reshape(-1)),
        jnp.asarray(args.robust_scale),
    )]
    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)
    vals0 = jaxls.VarValues.make([
        centers.with_value(jax.random.normal(k1, (N * W, 3)) * 0.1),
        points.with_value(jax.random.normal(k2, (N * n_pts_pad, 3)) * 0.5
                          + jnp.array([0, 0, 1.0])),
        scales.with_value(jnp.zeros(N * n_obs_pad)),
    ])
    problem = jaxls.LeastSquaresProblem(costs, [centers, points, scales]).analyze()
    print(f"[timing] analyze (once, all windows): {time.time() - t0:.2f}s")
    t0 = time.time()
    sol = problem.solve(
        vals0, linear_solver="conjugate_gradient",
        trust_region=jaxls.TrustRegionConfig(),
        termination=jaxls.TerminationConfig(
            max_iterations=args.iters, early_termination=False),
        verbose=False)
    c_all = np.asarray(sol[CamCenterVar]).reshape(N, W, 3)
    p_all = np.asarray(sol[Point3Var]).reshape(N, n_pts_pad, 3)
    print(f"[timing] solve (once, all {N} windows): {time.time() - t0:.2f}s")

    # --- stitch ---------------------------------------------------------------
    centers_glob = np.zeros((n_frames, 3))
    R_glob = np.zeros((n_frames, 4))
    have = np.zeros(n_frames, bool)
    merged_pts = []
    for wi, (s, rows, local_pid, npts) in enumerate(win_data):
        c_win, p_win = c_all[wi], p_all[wi]
        q_win = R_init[wi]  # rotations frozen per window

        if wi == 0:
            centers_glob[s:s + W] = c_win
            R_glob[s:s + W] = q_win
            have[s:s + W] = True
        else:
            # SE3-align this window onto the stitched trajectory over overlap.
            ov = np.arange(s, s + W)[have[s:s + W]]
            loc = ov - s
            Aw, bw = _rigid(c_win[loc], centers_glob[ov])
            c_al = c_win @ Aw.T + bw
            # rotate window rotations: R_cam->world' = Aw @ R_cam->world
            # stored as WORLD->CAM quats: R_wl' = R_wl @ Aw^T
            R_align = jaxlie.SO3.from_matrix(jnp.asarray(Aw))
            q_al = np.asarray(jax.vmap(
                lambda q: (jaxlie.SO3(q) @ R_align.inverse()).wxyz
            )(jnp.asarray(q_win)))
            new = np.arange(s, s + W)[~have[s:s + W]]
            centers_glob[new] = c_al[new - s]
            R_glob[new] = q_al[new - s]
            have[new] = True
            p_win = p_win @ Aw.T + bw
        merged_pts.append(p_win[:npts])

    assert have.all()
    print(f"[timing] wall total {time.time() - t_wall0:.1f}s")

    # Recenter cam0 at origin; convert to WORLD->CAM t = -R c.
    centers_glob -= centers_glob[0]
    t_out = np.asarray(jax.vmap(
        lambda q, c: -(jaxlie.SO3(q) @ c)
    )(jnp.asarray(R_glob), jnp.asarray(centers_glob)))
    poses = np.concatenate([R_glob, t_out], axis=1)
    pts = np.concatenate(merged_pts, 0) - 0  # rough merged cloud (per-window dup)

    np.savez(out_path, frame_idx=frame_idx, pose_wxyz_xyz=poses, points=pts)
    print(f"wrote {out_path}")


def _rigid(A, B):
    ca, cb = A.mean(0), B.mean(0)
    H = (A - ca).T @ (B - cb)
    U, _, Vt = np.linalg.svd(H)
    S = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ S @ U.T
    return R, cb - R @ ca


if __name__ == "__main__":
    main()
