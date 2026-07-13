"""Windowed VIO solve: independent short-window solves, chained by refined pose.

Rationale: the global stage-5 solve is superlinear in trajectory length (CG
iteration count blows up as LM damping decays on a long, gauge-free problem --
observed: ~50min/iter at 11k frames), but the downstream use only needs LOW
DRIFT OVER ~10s, not global consistency. So: solve fixed-size windows (default
3s, overlapping by 2 frames) independently -- each is tiny, well-conditioned,
and identical in shape so jaxls JIT-compiles ONCE per stage -- then chain them.

Per window, both stages of the global solver (both with the analytic-scale
positioning residual, no per-obs ScaleVars):
  1. frozen-rotation positioning (rotations sliced from ONE global IMU chain,
     gravity-aligned at frame 0)
  2. short SE3 refine (default 3 iters): rotations free, tethered by the IMU
     relative-rotation + gravity costs -- same recipe as vio_bundle_adjust
     stage 2. The full-solve A/B showed refine roughly halves 10s drift.

Stitching: consecutive windows share --overlap-frames (default 2) frames. The
gauge transform G aligning window k+1 into the stitched world is fixed by
requiring its REFINED pose at the first shared frame to equal the stitched
pose there (T_wc_stitched = T_wc_win @ G). This uses the window's own refined
relative motion -- its strongest information -- instead of the old Procrustes
fit over a 30-frame overlap of noisy positions.

Fixed problem shape = fixed (n_obs_pad, n_points_pad) per window: observations
are subsampled/padded (--max-obs caps the dense-Jacobian memory of
dense_cholesky; measured accuracy-neutral). Landmarks are per-window variables
(re-triangulated each window); only poses are stitched.

Output: same trajectory.npz schema as vio_bundle_adjust.py (minus per-landmark
visualizer fields -- points are per-window and not globally consistent, so a
merged cloud is written for rough viz only).

Run:
    python vio_windowed_ba.py ../../testimu --tracks ../../testimu/tracks.jsonl \
        --out /tmp/traj_windowed.npz --window-s 3
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
    p.add_argument("--overlap-frames", type=int, default=2,
                    help="frames shared between consecutive windows. The stitch "
                         "uses the REFINED relative pose across the shared "
                         "frame, so 2 suffices (vs 30 for the old Procrustes-"
                         "on-centers stitch)")
    p.add_argument("--iters", type=int, default=20,
                    help="LM iterations for the positioning stage per window")
    p.add_argument("--refine-iters", type=int, default=3,
                    help="LM iterations for the per-window SE3 refine (IMU "
                         "rotation + gravity tethered); 0 skips refine and "
                         "stitches on IMU-chain rotations")
    p.add_argument("--imu-rot-weight", type=float, default=100.0)
    p.add_argument("--gravity-weight", type=float, default=1.0)
    p.add_argument("--robust-scale", type=float, default=0.05)
    p.add_argument("--pad-quantile", type=float, default=50.0,
                    help="percentile of per-window obs counts used as the padded "
                         "problem size; windows above it are subsampled")
    p.add_argument("--max-obs", type=int, default=12000,
                    help="hard cap on (padded) observations per window. The "
                         "window has only ~5k unknowns, so 12k 3-dim residuals "
                         "is still ~7x over-determined -- and dense_cholesky "
                         "materializes the dense Jacobian (obs*3 x vars), which "
                         "OOMs at 56k obs x 15 windows (~108GB). 0 = no cap")
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

    fps = json.load(open(os.path.join(rec, "recording.json"))).get("fps", 30)
    W = max(8, int(round(args.window_s * fps)))
    V = max(2, args.overlap_frames)
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
    gravity_weight_arr = imu["gravity_weight"][keep]
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

    # ONE global IMU rotation chain, gravity-aligned at frame 0 (same recipe as
    # vio_bundle_adjust). Windows slice it: gyro drift over any 3s window is
    # negligible, and this avoids re-seeding each window from its own gravity
    # estimate (noisy during fast motion -> was causing 10-15 deg window errors).
    g0 = gravity_cam[0] / (np.linalg.norm(gravity_cam[0]) + 1e-12)
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
        _chain_step, q0_j, jnp.asarray(delta_prev[1:], dtype=jnp.float32))
    rot_chain = np.asarray(jnp.concatenate([q0_j[None], rot_rest], axis=0))

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
    if args.max_obs > 0:
        n_obs_pad = min(n_obs_pad, args.max_obs)
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
        # Rotations: slice of ONE global IMU chain (built below) -- per-window
        # gravity re-seeding was noisy during fast motion (low gravity
        # confidence) and produced 10-15 deg window rotation errors.
        R_init[wi] = rot_chain[s:s + W]

    # --- vmapped solve over all windows ---------------------------------------
    # Every window is padded to the same (W, n_pts_pad, n_obs_pad) shape, the
    # residual fn is MODULE-LEVEL (stable function identity -> stable jit key;
    # a @Cost.factory closure here would retrace per window), and analyze runs
    # with schur_elimination="off" so the id arrays are TRACED leaves (Schur's
    # elimination plan is static and would bake per-window connectivity into
    # the jit key). All N per-window problems then share one treedef: stack
    # their leaves and vmap a single compiled solve across the window axis.
    t0 = time.time()
    quat_per_obs = np.take_along_axis(R_init, P_pose[..., None], axis=1)  # (N,obs,4)

    def make_problem(wi):
        centers = CamCenterVar(id=jnp.arange(W))
        points = Point3Var(id=jnp.arange(n_pts_pad))
        cost = jaxls.Cost(
            _win_residual,
            (CamCenterVar(id=jnp.asarray(P_pose[wi])),
             Point3Var(id=jnp.asarray(P_point[wi])),
             jnp.asarray(quat_per_obs[wi]),
             jnp.asarray(P_ray[wi]),
             jnp.asarray(P_rel[wi]),
             jnp.asarray(P_w[wi]),
             jnp.asarray(args.robust_scale, dtype=jnp.float32)),
        )
        problem = jaxls.LeastSquaresProblem(
            [cost], [centers, points]).analyze(schur_elimination="off")
        key = jax.random.PRNGKey(wi)
        k1, k2 = jax.random.split(key)
        vals0 = jaxls.VarValues.make([
            centers.with_value(jax.random.normal(k1, (W, 3)) * 0.1),
            points.with_value(jax.random.normal(k2, (n_pts_pad, 3)) * 0.5
                              + jnp.array([0, 0, 1.0])),
        ])
        return problem, vals0

    pairs = [make_problem(wi) for wi in range(N)]
    print(f"[timing] build+analyze {N} windows: {time.time() - t0:.2f}s")

    def _solve_positioning(prob, v0):
        sol = prob.solve(
            v0, linear_solver=args.linear_solver,
            trust_region=jaxls.TrustRegionConfig(),
            termination=jaxls.TerminationConfig(
                max_iterations=args.iters, early_termination=False),
            verbose=False)
        return sol[CamCenterVar], sol[Point3Var]

    def make_refine(wi, centers_np, points_np):
        """Per-window SE3 refine: rotations free, tethered by IMU relative
        rotation + gravity (same recipe as vio_bundle_adjust stage 2). Same
        module-level residuals + schur off + fixed shapes -> compile once."""
        poses = jaxls.SE3Var(id=jnp.arange(W))
        points = Point3Var(id=jnp.arange(n_pts_pad))
        s = win_data[wi][0]
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
        problem = jaxls.LeastSquaresProblem(
            costs, [poses, points]).analyze(schur_elimination="off")
        pose_init = jax.vmap(
            lambda q, c: jaxlie.SE3.from_rotation_and_translation(
                jaxlie.SO3(q), -(jaxlie.SO3(q) @ c))
        )(jnp.asarray(R_init[wi]), jnp.asarray(centers_np))
        vals0 = jaxls.VarValues.make([
            poses.with_value(pose_init),
            points.with_value(jnp.asarray(points_np)),
        ])
        return problem, vals0

    def _solve_refine(prob, v0):
        sol = prob.solve(
            v0, linear_solver=args.linear_solver,
            trust_region=jaxls.TrustRegionConfig(),
            termination=jaxls.TerminationConfig(
                max_iterations=args.refine_iters, early_termination=False),
            verbose=False)
        return sol[jaxls.SE3Var].wxyz_xyz, sol[Point3Var]

    # Positioning then (optionally) refine, window by window. Compile is paid
    # once per stage; later windows are cached.
    q_all = np.zeros((N, W, 4))
    c_all = np.zeros((N, W, 3))
    p_all = np.zeros((N, n_pts_pad, 3))
    for wi, (prob, v0) in enumerate(pairs):
        t0 = time.time()
        c, p = _solve_positioning(prob, v0)
        jax.block_until_ready(c)
        t_pos = time.time() - t0
        if args.refine_iters > 0:
            t0 = time.time()
            rprob, rv0 = make_refine(wi, np.asarray(c), np.asarray(p))
            pose_out, p = _solve_refine(rprob, rv0)
            jax.block_until_ready(pose_out)
            t_ref = time.time() - t0
            pose_out = np.asarray(pose_out)
            q_all[wi] = pose_out[:, :4]
            # centers from refined SE3: c = -R^T t
            for j in range(W):
                Rj = np.asarray(jaxlie.SO3(jnp.asarray(pose_out[j, :4])).as_matrix())
                c_all[wi, j] = -(Rj.T @ pose_out[j, 4:])
        else:
            t_ref = 0.0
            q_all[wi] = R_init[wi]
            c_all[wi] = np.asarray(c)
        p_all[wi] = np.asarray(p)
        if wi < 3 or wi % 20 == 0:
            print(f"  window {wi + 1}/{N}: pos {t_pos:.2f}s + refine {t_ref:.2f}s")

    # --- stitch via the refined relative pose at the shared frames ------------
    # Window wi and wi+1 share V frames. Both have a (refined) pose for the
    # first shared frame f0. Chain: find the SE3 gauge transform G mapping
    # window wi+1's world into the stitched world so that its pose at f0
    # equals the stitched pose at f0, then apply G to the whole window and
    # keep its poses for all frames after f0. This uses the window's own
    # refined relative motion (its strongest information) instead of a
    # Procrustes fit over a long overlap.
    def pose_mat(q, c):
        R = np.asarray(jaxlie.SO3(jnp.asarray(q)).as_matrix())  # world->cam
        T = np.eye(4); T[:3, :3] = R; T[:3, 3] = -(R @ c)
        return T  # T_wc: world->cam

    centers_glob = np.zeros((n_frames, 3))
    R_glob = np.zeros((n_frames, 4))
    have = np.zeros(n_frames, bool)
    merged_pts = []
    for wi, (s, rows, local_pid, npts) in enumerate(win_data):
        if wi == 0:
            centers_glob[s:s + W] = c_all[wi]
            R_glob[s:s + W] = q_all[wi]
            have[s:s + W] = True
            merged_pts.append(p_all[wi][:npts])
            continue
        # First shared frame: stitched pose T_stitched, window pose T_win.
        f0 = s  # window start (already covered by previous window)
        T_st = pose_mat(R_glob[f0], centers_glob[f0])
        T_wn = pose_mat(q_all[wi][0], c_all[wi][0])
        # Gauge: for any frame, T_wc_stitched = T_wc_win @ G  with
        # G = T_wn^-1 @ T_st (world-side transform). Then cam centers/rots:
        G = np.linalg.inv(T_wn) @ T_st
        Ginv = np.linalg.inv(G)
        new = np.arange(s, s + W)[~have[s:s + W]]
        for f in new:
            T = pose_mat(q_all[wi][f - s], c_all[wi][f - s]) @ G
            Rf = T[:3, :3]
            centers_glob[f] = -(Rf.T @ T[:3, 3])
            R_glob[f] = np.asarray(jaxlie.SO3.from_matrix(jnp.asarray(Rf)).wxyz)
        have[new] = True
        # carry points into the stitched world: X' = Ginv[:3,:3] X + Ginv trans
        merged_pts.append(p_all[wi][:npts] @ Ginv[:3, :3].T + Ginv[:3, 3])

    assert have.all()
    print(f"[timing] wall total {time.time() - t_wall0:.1f}s")

    # Recenter cam0 at origin; convert to WORLD->CAM t = -R c.
    centers_glob -= centers_glob[0]
    t_out = np.asarray(jax.vmap(
        lambda q, c: -(jaxlie.SO3(q) @ c)
    )(jnp.asarray(R_glob), jnp.asarray(centers_glob)))
    poses = np.concatenate([R_glob, t_out], axis=1)
    pts = np.concatenate(merged_pts, 0)  # rough merged cloud (per-window dup)

    np.savez(out_path, frame_idx=frame_idx, pose_wxyz_xyz=poses, points=pts)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
