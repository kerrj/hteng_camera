"""VIO pipeline stage 5: global BA over camera poses + 3D landmarks, with IMU
relative-rotation and gravity priors.

GLOMAP-style bounded positioning cost (Pan et al. 2024 / BATA) instead of
reprojection error, with the per-observation scale analytically eliminated:

    residual_ik = v_ik - <v_ik, u_ik> u_ik,   u_ik = (X_k - c_i)/|X_k - c_i|

i.e. the component of the observed ray v_ik perpendicular to the predicted
direction u_ik -- same optimum as GLOMAP's explicit d_ik variables
(substitute the optimal d* = <v,X-c>/|X-c|^2), but without one extra variable
per observation. Squared norm = sin^2(theta) capped at 1, bounded (reprojection
error diverges under the random init this needs). Metric scale comes from the
stereo baseline.

Two-stage solve (mirrors GLOMAP: rotation averaging -> positioning with
rotations frozen -> full BA):
1. Positioning: rotations are constants from the integrated IMU relative-
   rotation chain; variables are camera centers (CamCenterVar), points, scales.
2. Refine: full SE3 BA from stage 1, with IMU relative-rotation + gravity costs.

Gauge: no anchor cost; the translation null space is left to LM damping and the
result recentered post-hoc to cam0 = origin (no rescale -- stereo baseline sets
scale). Roll/pitch fixed by the gravity prior (world +z up); yaw is free.

Outliers: robust positioning loss (--robust-loss, default Cauchy) down-weights
in-solve via IRLS (residual * sqrt of a stop_gradient'd weight); one solve over
all observations. Per-landmark median residuals saved for the visualizer.

Solver: trust-region LM (Gauss-Newton diverges here).

Run (from data_processing/vio/):
    python vio_bundle_adjust.py ../../testimu --tracks ../../testimu/tracks.jsonl \
        --imu-relative ../../testimu/imu_relative.npz --out ../../testimu/trajectory.npz
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
import fisheye_pinhole as FP


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("recording")
    p.add_argument("--tracks", default=None, help="default: <recording>/derived/tracks.jsonl")
    p.add_argument("--imu-relative", default=None,
                    help="default: <recording>/derived/imu_relative.npz")
    p.add_argument("--out", default=None, help="default: <recording>/derived/trajectory.npz")
    p.add_argument("--n-frames", type=int, default=None,
                    help="cap the number of (left-eye) frames included")
    p.add_argument("--start-frame", type=int, default=None,
                    help="first video frame of the solve window; with --n-frames "
                         "this solves frames [start, start+n] (windowed BA); "
                         "default: the recording's first frame")
    p.add_argument("--profile-compile", action="store_true",
                    help="time JIT compile vs. cached per-iteration solve separately")
    p.add_argument("--robust-loss", choices=("huber", "cauchy", "geman_mcclure"),
                    default="cauchy",
                    help="M-estimator on the positioning residual; escalating "
                         "outlier suppression huber < cauchy < geman_mcclure")
    p.add_argument("--robust-scale", "--huber-delta", dest="robust_scale",
                    type=float, default=0.05,
                    help="robust loss scale (weight=0.5 at residual=this). On the "
                         "bounded sin-theta residual (inliers ~0.02-0.05 = 1-3deg), "
                         "NOT pixels")
    p.add_argument("--imu-rot-weight", type=float, default=100.0,
                    help="weight on the IMU relative-rotation residual")
    p.add_argument("--gravity-weight", type=float, default=1.0,
                    help="weight on the gravity-direction prior (times per-frame "
                         "confidence)")
    p.add_argument("--translation-smoothness-weight", type=float, default=0.0,
                    help="weight on a 2nd-derivative prior over camera position "
                         "across consecutive pose triples; 0 disables")
    p.add_argument("--pose-init-noise", type=float, default=1.0,
                    help="std (m) of the random camera-center init")
    p.add_argument("--landmark-init-noise", type=float, default=1.0,
                    help="std (m) of random landmark init, centered at --init-depth")
    p.add_argument("--init-depth", type=float, default=1.0,
                    help="landmark init ball center depth (m)")
    p.add_argument("--pose-init-seed", type=int, default=0)
    p.add_argument("--init-trajectory", default=None,
                    help="trajectory npz (frame_idx, pose_wxyz_xyz) to take "
                         "stage-1 frozen rotations + camera-center inits from, "
                         "instead of the IMU gyro chain + random centers. Used "
                         "by vio_windowed_ba.py's global refine: the stitched "
                         "windowed solution is a near-correct init everywhere, "
                         "where the raw gyro chain drifts over minutes.")
    p.add_argument("--max-iterations", type=int, default=15,
                    help="iteration cap for stage 1 (frozen-rotation positioning)")
    p.add_argument("--refine-iterations", type=int, default=3,
                    help="iteration cap for stage 2 (full SE3 refine); each iter is "
                         "~15x costlier than stage 1. 0 skips refine")
    p.add_argument("--linear-solver", choices=("conjugate_gradient", "dense_cholesky", "cholmod"),
                    default="conjugate_gradient",
                    help="CG is fastest here; dense_cholesky can't materialize the "
                         "reduced system, cholmod is CPU-bound and slower")
    p.add_argument("--gauss-newton", action="store_true",
                    help="trust_region=None -- diverges here, not recommended")
    p.add_argument("--schur", choices=("auto", "off"), default="auto",
                    help="jaxls variable elimination. With ScaleVars gone the "
                         "landmark block is the only elimination candidate; "
                         "'off' skips the (host-side, slow) elimination-plan "
                         "construction in analyze() at the cost of a bigger "
                         "CG system")
    p.add_argument("--early-termination", action="store_true",
                    help="stop on jaxls's tolerances instead of running all "
                         "iterations. OFF by default: it false-positives on this "
                         "problem, halting at ~2 iters and leaving positions "
                         "under-solved (scrunched trajectory)")
    p.add_argument("--loss-plot", default=None,
                    help="PNG path for the cost curve (always written)")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def load_intrinsics(recording_dir, serial):
    calib = json.load(open(os.path.join(recording_dir, f"calib_{serial}.json")))["intrinsics"]
    return np.array(calib["K"], np.float64), np.array(calib["dist"], np.float64)


def load_stereo(recording_dir, ls, rs):
    st = json.load(open(os.path.join(recording_dir, f"stereo_{ls}_{rs}.json")))
    return np.array(st["R"], np.float64), np.array(st["t"], np.float64).reshape(3)


def load_tracks(tracks_path, min_frame, max_frame):
    """Returns list of tracks; each track is a list of (eye, frame, px (2,))
    observations, restricted to min_frame <= frame <= max_frame (either bound
    None = unbounded)."""
    tracks = []
    with open(tracks_path) as f:
        for line in f:
            r = json.loads(line)
            obs = [o for o in r["observations"]
                   if (min_frame is None or o["frame"] >= min_frame)
                   and (max_frame is None or o["frame"] <= max_frame)]
            if len(obs) < 2:
                continue
            tracks.append([(o["eye"], o["frame"], np.array(o["px"], dtype=np.float64))
                            for o in obs])
    return tracks


def unproject_all(pts_by_eye, cams, device):
    """pts_by_eye: dict eye -> (N,2) pixel array. Returns dict eye -> (N,3)
    unit rays, camera frame."""
    import torch
    rays_by_eye = {}
    for eye, pts in pts_by_eye.items():
        if pts.shape[0] == 0:
            rays_by_eye[eye] = np.zeros((0, 3))
            continue
        K, D = cams[eye]
        pts_t = torch.from_numpy(pts.astype(np.float32)).to(device)
        K_t = torch.tensor(K, dtype=torch.float32, device=device)
        D_t = torch.tensor(D, dtype=torch.float32, device=device)
        rays = FP.fisheye_unproject(pts_t[:, 0], pts_t[:, 1], K_t, D_t)
        rays_by_eye[eye] = rays.cpu().numpy().astype(np.float64)
    return rays_by_eye


class Point3Var(jaxls.Var[jax.Array], default_factory=lambda: jnp.zeros(3)):
    """3D landmark position, world frame."""


class CamCenterVar(jaxls.Var[jax.Array], default_factory=lambda: jnp.zeros(3)):
    """Stage-1 variable: left camera center in world frame (rotations frozen)."""




def _robust_weight(residual, robust_scale, robust_kind):
    """IRLS robust reweight (residual * sqrt of a stop_gradient'd weight).
    robust_kind: 0=Huber, 1=Cauchy, 2=Geman-McClure."""
    abs_r = jnp.linalg.norm(residual) + 1e-9
    x2 = (abs_r / robust_scale) ** 2
    w_huber = jnp.where(abs_r > robust_scale, robust_scale / abs_r, 1.0)
    w_cauchy = 1.0 / (1.0 + x2)
    w_gm = 1.0 / (1.0 + x2) ** 2
    return jax.lax.stop_gradient(jnp.stack([w_huber, w_cauchy, w_gm])[robust_kind])


@jaxls.Cost.factory
def positioning_cost(vals, pose, center, point_var, ray_cam,
                     rel_wxyz_xyz, robust_scale, robust_kind):
    """Positioning residual with GLOMAP's per-observation scale ANALYTICALLY
    eliminated: substituting the optimal d* = <ray, X-c>/|X-c|^2 leaves the
    component of the observed ray perpendicular to the predicted direction --
    a bounded sin-theta heading error with the same optimum, minus one
    ScaleVar per observation (16M of them on an 11k-frame recording; they
    dominated both compile and solve). Metric scale still comes from the
    stereo baseline in rel_wxyz_xyz. Shared by both stages via var-or-constant
    factory args (a Var arg is optimized, a plain array is baked in constant):
      stage 1: pose = constant IMU-chain quats, center = CamCenterVar optimized.
      stage 2: pose = free SE3Var, center = unused dummy constant.
    T_wc = T_rel @ T_wl (T_rel = identity for left eye, stereo transform for
    right, so the baseline is never dropped)."""
    T_rel = jaxlie.SE3(rel_wxyz_xyz)
    if isinstance(pose, jaxls.Var):
        T_wl = vals[pose]
        R_wl = T_wl.rotation()
        c_left = -(R_wl.inverse() @ T_wl.translation())
    else:
        R_wl = jaxlie.SO3(pose)
        c_left = vals[center]
    R_wc = T_rel.rotation() @ R_wl
    cam_pos_world = c_left - (R_wc.inverse() @ T_rel.translation())
    ray_world = R_wc.inverse() @ ray_cam
    v = vals[point_var] - cam_pos_world
    v_dir = v / (jnp.linalg.norm(v) + 1e-9)
    # d* clamped >= 0: GLOMAP's d=exp(s)>0 forbids behind-camera landmarks;
    # the raw perpendicular component is sign-invariant in v (a landmark
    # BEHIND the camera would cost ~0 and the solve diverges -- observed).
    # Clamping makes a behind-camera landmark cost the full ray (norm 1, the
    # bounded max), restoring the constraint.
    d_star = jnp.maximum(jnp.dot(ray_world, v_dir), 0.0)
    residual = ray_world - d_star * v_dir
    return residual * jnp.sqrt(_robust_weight(residual, robust_scale, robust_kind))


@jaxls.Cost.factory
def imu_rotation_cost(vals, pose_i, pose_j, delta_R_wxyz, weight):
    """Residual vs IMU-integrated relative rotation. SE3Var is WORLD->CAMERA,
    so the actual relative rotation is R_i @ R_j.inverse()."""
    R_i = vals[pose_i].rotation()
    R_j = vals[pose_j].rotation()
    delta_R = jaxlie.SO3(delta_R_wxyz)
    actual_rel = R_i @ R_j.inverse()
    err = (delta_R.inverse() @ actual_rel).log()
    return err * weight


@jaxls.Cost.factory
def gravity_cost(vals, pose_var, world_gravity_dir, measured_down_cam, weight):
    """Roll/pitch prior (not yaw): world is +z up, so world-down is -z; each
    frame's rotation should map world-down onto its IMU-measured down. Applies
    to all frames; yaw about gravity is left unconstrained."""
    R_wl = vals[pose_var].rotation()
    down_pred_cam = R_wl @ world_gravity_dir
    return (down_pred_cam - measured_down_cam) * weight


@jaxls.Cost.factory
def translation_smoothness_cost(vals, pose_a, pose_b, pose_c, weight):
    """Second-derivative (acceleration) prior on camera position across three
    consecutive poses -- damps position zigzag without fighting steady motion."""
    def cam_pos(pose_var):
        if isinstance(pose_var, CamCenterVar):  # stage 1: already a center
            return vals[pose_var]
        T = vals[pose_var]
        return -(T.rotation().inverse() @ T.translation())
    accel = cam_pos(pose_a) - 2 * cam_pos(pose_b) + cam_pos(pose_c)
    return accel * weight


def main():
    args = parse_args()
    tracks_path = args.tracks or os.path.join(args.recording, "derived", "tracks.jsonl")
    imu_path = args.imu_relative or os.path.join(args.recording, "derived", "imu_relative.npz")
    out_path = args.out or os.path.join(args.recording, "derived", "trajectory.npz")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    imu = np.load(imu_path)
    frame_idx = imu["frame_idx"]
    frame_valid = imu["frame_valid"]
    rel_quat = imu["rel_quat"]
    rel_valid = imu["rel_valid"]
    gravity_cam = imu["gravity_cam"]
    gravity_weight = imu["gravity_weight"]

    if args.n_frames is not None or args.start_frame is not None:
        lo = args.start_frame if args.start_frame is not None else int(frame_idx[0])
        hi = lo + args.n_frames if args.n_frames is not None else int(frame_idx[-1])
        keep = (frame_idx >= lo) & (frame_idx <= hi)
        a = int(np.argmax(keep))
        b = a + int(keep.sum())
        assert keep.any() and keep[a:b].all(), \
            f"frame window [{lo},{hi}] empty or non-contiguous in frame_idx"
        frame_idx = frame_idx[a:b]
        frame_valid = frame_valid[a:b]
        rel_quat = rel_quat[a:b - 1]
        rel_valid = rel_valid[a:b - 1]
        gravity_cam = gravity_cam[a:b]
        gravity_weight = gravity_weight[a:b]

    # Drop invalid-timestamp frames entirely (they have no IMU tie to neighbors).
    n_dropped = int((~frame_valid).sum())
    if n_dropped:
        print(f"dropping {n_dropped} frame(s) with invalid timestamps "
              f"(frame idx {frame_idx[~frame_valid].tolist()}) from the pose list")
    orig_frame_idx = frame_idx
    keep_mask = frame_valid
    frame_idx = frame_idx[keep_mask]
    gravity_cam = gravity_cam[keep_mask]
    gravity_weight = gravity_weight[keep_mask]

    # Re-index rel_quat/rel_valid to the surviving (possibly shorter) pose list.
    new_pos_of_orig = -np.ones(len(orig_frame_idx), dtype=np.int64)
    new_pos_of_orig[keep_mask] = np.arange(keep_mask.sum())
    orig_edge_a = np.arange(len(rel_valid))
    orig_edge_b = orig_edge_a + 1
    edge_survives = keep_mask[orig_edge_a] & keep_mask[orig_edge_b]
    new_rel_quat = rel_quat[edge_survives]
    new_edge_a = new_pos_of_orig[orig_edge_a[edge_survives]]
    new_edge_b = new_pos_of_orig[orig_edge_b[edge_survives]]
    rel_valid_orig = rel_valid[edge_survives]

    n_frames = len(frame_idx)
    frame_to_pose_idx = {int(f): i for i, f in enumerate(frame_idx)}
    max_frame = int(frame_idx[-1])
    print(f"{n_frames} pose frames (idx {frame_idx[0]}..{max_frame})")

    import h5py
    features_path = os.path.join(args.recording, "derived", "features.h5")
    with h5py.File(features_path, "r") as f:
        ls_serial = f.attrs["left_serial"]
        rs_serial = f.attrs["right_serial"]
    Kl, Dl = load_intrinsics(args.recording, ls_serial)
    Kr, Dr = load_intrinsics(args.recording, rs_serial)
    R_st, t_st = load_stereo(args.recording, ls_serial, rs_serial)
    cams = {"left": (Kl, Dl), "right": (Kr, Dr)}
    rel_wxyz_xyz_left = np.asarray(jaxlie.SE3.identity().wxyz_xyz)
    rel_wxyz_xyz_right = np.asarray(jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3.from_matrix(R_st), t_st).wxyz_xyz)

    tracks = load_tracks(tracks_path, int(frame_idx[0]), max_frame)
    print(f"{len(tracks)} tracks (>=2 obs) within the frame range")

    pose_ids, point_ids, obs_px, obs_is_right = [], [], [], []
    for pt_idx, obs in enumerate(tracks):
        for eye, frame, px in obs:
            if frame not in frame_to_pose_idx:
                continue
            pose_ids.append(frame_to_pose_idx[frame])
            point_ids.append(pt_idx)
            obs_px.append(px)
            obs_is_right.append(eye == "right")

    pose_ids = np.array(pose_ids, dtype=np.int64)
    point_ids = np.array(point_ids, dtype=np.int64)
    obs_px = np.stack(obs_px, axis=0)
    obs_is_right = np.array(obs_is_right, dtype=bool)
    n_obs = len(pose_ids)
    n_points = len(tracks)
    print(f"{n_obs} total observations across {n_points} tracks")

    pts_by_eye = {"left": obs_px[~obs_is_right], "right": obs_px[obs_is_right]}
    rays_by_eye = unproject_all(pts_by_eye, cams, args.device)
    ray_cam = np.zeros((n_obs, 3))
    ray_cam[~obs_is_right] = rays_by_eye["left"]
    ray_cam[obs_is_right] = rays_by_eye["right"]

    rel_wxyz_xyz = np.where(
        obs_is_right[:, None], rel_wxyz_xyz_right[None, :], rel_wxyz_xyz_left[None, :])

    # Frozen stage-1 rotations: integrate the IMU relative-rotation chain from a
    # gravity-aligned cam-0 seed. SE3Var is WORLD->CAMERA, so R_b = delta^-1 @ R_a
    # for a consecutive edge a->b. Build a per-pose "delta from previous" quat
    # (identity where a valid consecutive edge is missing), then left-fold it.
    delta_prev = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n_frames, 1))
    consec = (new_edge_b == new_edge_a + 1) & rel_valid_orig
    delta_prev[new_edge_b[consec]] = new_rel_quat[consec]

    def _chain_step(R_prev_wxyz, delta_wxyz):
        R_k = jaxlie.SO3(delta_wxyz).inverse() @ jaxlie.SO3(R_prev_wxyz)
        return R_k.wxyz, R_k.wxyz

    # Seed cam-0 so world-down (-z) maps to its measured down (shortest-arc,
    # yaw 0). Half-arc quat from a=[0,0,-1] to b=g0 is [1 + a.b, a x b].
    g0 = np.asarray(gravity_cam[0], dtype=np.float64)
    g0 = g0 / (np.linalg.norm(g0) + 1e-12)
    a_down = np.array([0.0, 0.0, -1.0])
    q0 = np.array([1.0 + a_down @ g0, *np.cross(a_down, g0)])
    if np.linalg.norm(q0) < 1e-6:  # g0 ~ +z (antiparallel to world-down)
        q0 = np.array([0.0, 1.0, 0.0, 0.0])
    r0_wxyz = jnp.asarray(q0 / np.linalg.norm(q0), dtype=jnp.float32)
    _, rot_rest = jax.lax.scan(
        _chain_step, r0_wxyz, jnp.asarray(delta_prev[1:], dtype=jnp.float32))
    rot_init_wxyz = jnp.concatenate([r0_wxyz[None, :], rot_rest], axis=0)

    # Random init for camera centers and landmarks (no anchor; recentered later).
    key = jax.random.PRNGKey(args.pose_init_seed)
    key, sub = jax.random.split(key)
    center_init = jax.random.normal(sub, (n_frames, 3)) * args.pose_init_noise
    key, sub = jax.random.split(key)
    point_init = jax.random.normal(sub, (n_points, 3)) * args.landmark_init_noise
    point_init = point_init.at[:, 2].add(args.init_depth)

    if args.init_trajectory:
        init = np.load(args.init_trajectory)
        init_fi = init["frame_idx"]
        pos = np.searchsorted(init_fi, frame_idx)
        if pos.max() >= len(init_fi) or not np.array_equal(init_fi[pos], frame_idx):
            raise ValueError(f"--init-trajectory {args.init_trajectory} does not "
                             f"cover the solve's frames "
                             f"({frame_idx[0]}..{frame_idx[-1]})")
        init_poses = np.asarray(init["pose_wxyz_xyz"])[pos]
        rot_init_wxyz = jnp.asarray(init_poses[:, :4], dtype=jnp.float32)
        R_init = np.asarray(jax.vmap(lambda q: jaxlie.SO3(q).as_matrix())(
            jnp.asarray(init_poses[:, :4])))
        center_init = jnp.asarray(
            -np.einsum("nji,nj->ni", R_init, init_poses[:, 4:]), dtype=jnp.float32)
        print(f"stage-1 rotations + centers initialized from "
              f"{args.init_trajectory} ({len(frame_idx)} frames)")

    keep_edge = rel_valid_orig
    print(f"{int(keep_edge.sum())} IMU relative-rotation edges")
    world_gravity_dir = jnp.array([[0.0, 0.0, -1.0]], dtype=jnp.float32)  # +z up
    per_frame_weight = args.gravity_weight * gravity_weight

    robust_kind = {"huber": 0, "cauchy": 1, "geman_mcclure": 2}[args.robust_loss]

    def solve_problem(costs, var_groups, initial_vals, max_iterations):
        """Analyze + solve one least-squares problem, with timing/cost prints."""
        t0 = time.time()
        problem = jaxls.LeastSquaresProblem(costs, var_groups).analyze(
            schur_elimination=args.schur)
        print(f"[timing] analyze(): {time.time() - t0:.2f}s")

        if args.profile_compile:
            # Two calls at identical max_iterations (a static jit key): first
            # pays compile, second is cached.
            probe_iters = min(max_iterations, 5)
            probe_term = jaxls.TerminationConfig(max_iterations=probe_iters, early_termination=False)
            probe_trust_region = None if args.gauss_newton else jaxls.TrustRegionConfig()
            t0 = time.time()
            problem.solve(initial_vals, linear_solver=args.linear_solver,
                           trust_region=probe_trust_region, termination=probe_term)
            t_warmup = time.time() - t0
            t0 = time.time()
            problem.solve(initial_vals, linear_solver=args.linear_solver,
                           trust_region=probe_trust_region, termination=probe_term)
            t_cached = time.time() - t0
            print(f"[timing] first call ({probe_iters} iters, pays compile): {t_warmup:.2f}s")
            print(f"[timing] second call ({probe_iters} iters, cached): "
                  f"{t_cached:.2f}s ({1000*t_cached/probe_iters:.1f}ms/iter steady-state)")

        t0 = time.time()
        solution, summary = problem.solve(
            initial_vals,
            linear_solver=args.linear_solver,
            trust_region=None if args.gauss_newton else jaxls.TrustRegionConfig(),
            termination=jaxls.TerminationConfig(
                max_iterations=max_iterations,
                early_termination=args.early_termination,
            ),
            return_summary=True,
        )
        t_solve = time.time() - t0
        n_iters = int(summary.iterations) + 1
        print(f"[timing] solve(): {t_solve:.2f}s ({n_iters} iters, "
              f"{1000*t_solve/max(n_iters,1):.1f}ms/iter)")
        cost_history = np.asarray(summary.cost_history)[:n_iters]
        print(f"cost history ({len(cost_history)} steps): "
              f"{cost_history[0]:.4g} -> {cost_history[-1]:.4g}")
        # Convergence trace: cost + relative drop at ~10 points.
        stride = max(1, len(cost_history) // 10)
        for k in range(0, len(cost_history), stride):
            rel = (0.0 if k == 0 else
                   (cost_history[k - stride] - cost_history[k])
                   / max(cost_history[k - stride], 1e-12))
            print(f"  iter {k:3d}: cost {cost_history[k]:.6g}"
                  + ("" if k == 0 else f"  (drop {100 * rel:.2f}%/{stride}it)"))
        if len(cost_history) >= 2:
            final_rel = (cost_history[-2] - cost_history[-1]) / max(cost_history[-2], 1e-12)
            print(f"  final iter: cost {cost_history[-1]:.6g}  "
                  f"(last-step drop {100 * final_rel:.3f}%)")
        return solution, cost_history

    def positioning_and_smoothness_costs(pose_arg, center_arg, cam_pos_vars):
        """Costs shared by both stages. pose_arg/center_arg = the var-or-constant
        pair for positioning_cost; cam_pos_vars = per-frame var type for the
        smoothness triples (CamCenterVar or SE3Var)."""
        costs = [positioning_cost(
            pose_arg,
            center_arg,
            Point3Var(id=jnp.asarray(point_ids)),
            jnp.asarray(ray_cam),
            jnp.asarray(rel_wxyz_xyz),
            jnp.asarray(args.robust_scale),
            robust_kind,
        )]
        if args.translation_smoothness_weight > 0 and n_frames >= 3:
            costs.append(translation_smoothness_cost(
                cam_pos_vars(jnp.arange(n_frames - 2)),
                cam_pos_vars(jnp.arange(1, n_frames - 1)),
                cam_pos_vars(jnp.arange(2, n_frames)),
                jnp.asarray(args.translation_smoothness_weight),
            ))
        return costs

    # Stage 1: global positioning, rotations frozen as constants. The rotation-
    # dependent costs (imu_rotation, gravity) are constant here so are absent.
    print("=== stage 1: frozen-rotation global positioning ===")
    center_vars = CamCenterVar(id=jnp.arange(n_frames))
    point_vars = Point3Var(id=jnp.arange(n_points))
    stage1_costs = positioning_and_smoothness_costs(
        np.asarray(rot_init_wxyz)[pose_ids],      # constant per-obs quats
        CamCenterVar(id=jnp.asarray(pose_ids)),
        lambda ids: CamCenterVar(id=ids),
    )
    stage1_vals = jaxls.VarValues.make([
        center_vars.with_value(center_init),
        point_vars.with_value(point_init),
    ])
    solution1, cost_history = solve_problem(
        stage1_costs, [center_vars, point_vars],
        stage1_vals, args.max_iterations)

    # Convert stage-1 centers to SE3 poses (t = -R c) for stage 2 / output.
    centers1 = solution1[CamCenterVar]
    pose_from_stage1 = jax.vmap(
        lambda q, c: jaxlie.SE3.from_rotation_and_translation(
            jaxlie.SO3(q), -(jaxlie.SO3(q) @ c))
    )(rot_init_wxyz, centers1)

    # Stage 2: full SE3 refine, rotations now free and tethered by the IMU
    # relative-rotation + gravity costs.
    refine_iters = (args.max_iterations if args.refine_iterations is None
                    else args.refine_iterations)
    if refine_iters > 0:
        print("=== stage 2: full SE3 refine ===")
        camera_vars = jaxls.SE3Var(id=jnp.arange(n_frames))
        refine_costs = positioning_and_smoothness_costs(
            jaxls.SE3Var(id=jnp.asarray(pose_ids)),
            jnp.zeros((n_obs, 3)),                # unused dummy constant
            lambda ids: jaxls.SE3Var(id=ids),
        )
        if keep_edge.sum() > 0:
            refine_costs.append(imu_rotation_cost(
                jaxls.SE3Var(id=jnp.asarray(new_edge_a[keep_edge])),
                jaxls.SE3Var(id=jnp.asarray(new_edge_b[keep_edge])),
                jnp.asarray(new_rel_quat[keep_edge]),
                jnp.asarray(args.imu_rot_weight),
            ))
        refine_costs.append(gravity_cost(
            jaxls.SE3Var(id=jnp.arange(n_frames)),
            world_gravity_dir,
            jnp.asarray(gravity_cam),
            jnp.asarray(per_frame_weight)[:, None],
        ))
        stage2_vals = jaxls.VarValues.make([
            camera_vars.with_value(pose_from_stage1),
            point_vars.with_value(solution1[Point3Var]),
        ])
        solution, cost_history2 = solve_problem(
            refine_costs, [camera_vars, point_vars],
            stage2_vals, refine_iters)
        cost_history = np.concatenate([cost_history, cost_history2])
        poses_out = solution[jaxls.SE3Var]
        points_out = solution[Point3Var]
    else:
        solution = solution1
        poses_out = pose_from_stage1
        points_out = solution1[Point3Var]

    # Recenter cam0's center to the world origin (no in-solve anchor). For
    # WORLD->CAM poses, shifting the world by -c0 changes t to t + R c0.
    poses_np = np.asarray(poses_out.wxyz_xyz)
    points_np = np.asarray(points_out)
    R0 = np.asarray(jaxlie.SO3(poses_np[0, :4]).as_matrix())
    c0 = -(R0.T @ poses_np[0, 4:])
    Rall = np.asarray(jax.vmap(lambda q: jaxlie.SO3(q).as_matrix())(poses_np[:, :4]))
    poses_np = poses_np.copy()
    poses_np[:, 4:] += np.einsum("nij,j->ni", Rall, c0)
    points_np = points_np - c0
    print(f"recentered: cam0 center moved from {np.round(c0, 3)} to origin")

    def angular_residuals(poses, pts, obs_subset=None):
        """Per-observation angle (deg) between the observed ray and the
        direction to the optimized point, in numpy. Also returns depth."""
        m = np.ones(n_obs, dtype=bool) if obs_subset is None else obs_subset
        q = np.asarray(poses)[pose_ids[m]]
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        R_wl = np.stack([
            np.stack([1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)], -1),
            np.stack([2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)], -1),
            np.stack([2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)], -1),
        ], axis=1)
        t = q_t = poses[pose_ids[m], 4:]
        right = obs_is_right[m]
        R_st_ = np.asarray(jaxlie.SE3(rel_wxyz_xyz_right).rotation().as_matrix())
        t_st_ = np.asarray(jaxlie.SE3(rel_wxyz_xyz_right).translation())
        R = np.where(right[:, None, None], np.einsum("ij,njk->nik", R_st_, R_wl), R_wl)
        t = np.where(right[:, None], np.einsum("ij,nj->ni", R_st_, q_t) + t_st_, q_t)
        p_cam = np.einsum("nij,nj->ni", R, pts[point_ids[m]]) + t
        dir_cam = p_cam / (np.linalg.norm(p_cam, axis=1, keepdims=True) + 1e-12)
        cos = np.einsum("ni,ni->n", dir_cam, ray_cam[m])
        return np.degrees(np.arccos(np.clip(cos, -1, 1))), p_cam[:, 2]

    # Per-landmark median angular residual for the visualizer; point_alive flags
    # landmarks with >=2 observations.
    ang, _ = angular_residuals(poses_np, points_np)
    point_alive = np.bincount(point_ids, minlength=n_points) >= 2
    point_med_ang = np.full(n_points, np.nan)
    order = np.argsort(point_ids, kind="stable")
    ids_sorted = point_ids[order]
    ang_sorted = ang[order]
    bounds = np.searchsorted(ids_sorted, np.arange(n_points + 1))
    for k in range(n_points):
        if bounds[k + 1] > bounds[k]:
            point_med_ang[k] = np.median(ang_sorted[bounds[k]:bounds[k + 1]])
    print(f"{int(point_alive.sum())}/{n_points} landmarks alive (>=2 obs); "
          f"median angular residual "
          f"{np.nanmedian(point_med_ang[point_alive]):.3f} deg")

    loss_plot_path = args.loss_plot or os.path.join(args.recording, "derived", "loss_curve.png")
    if loss_plot_path:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(cost_history, marker="o", markersize=3)
        ax.set_yscale("log")
        ax.set_xlabel("iteration")
        ax.set_ylabel("total cost")
        ax.set_title(f"vio_bundle_adjust cost history ({args.recording})")
        fig.savefig(loss_plot_path)
        print(f"wrote {loss_plot_path}")

    # Each landmark's first observation, for visualizer pixel-color sampling.
    first_obs_row = np.full(n_points, -1, dtype=np.int64)
    for row, pt_idx in enumerate(point_ids):
        if first_obs_row[pt_idx] < 0:
            first_obs_row[pt_idx] = row
    point_first_frame = frame_idx[pose_ids[first_obs_row]]
    point_first_is_right = obs_is_right[first_obs_row]
    point_first_px = obs_px[first_obs_row]

    np.savez(out_path,
              frame_idx=frame_idx,
              pose_wxyz_xyz=poses_np,
              points=points_np,
              point_first_frame=point_first_frame,
              point_first_is_right=point_first_is_right,
              point_first_px=point_first_px,
              point_alive=point_alive,
              point_med_ang=point_med_ang,
              cost_history=cost_history)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
