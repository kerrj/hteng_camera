"""Stereo MANO bundle adjustment over a hand track, with jaxls.

Keypoints-only (no images, no ViT) — so a dense, temporally-coupled solve over
the whole video is cheap (seconds on CPU). For each frame we have WiLoR's
per-eye 2D keypoints in two *rectified* pinhole crops (left/right) that share a
virtual orientation with x along the stereo baseline, so the right camera is the
left translated by ``-baseline`` along x. We optimize, per frame:

    pose       (16,4) per-joint unit quats (wxyz) on the SO(3) manifold;
                      joint 0 = global_orient, 1..15 = finger joints. The
                      optimizer retracts tangent steps via SO3.exp (egoallo style).
    betas      (10,) shape — FROZEN to WiLoR's per-frame estimate
    t          (3,)  MANO-root translation in the rectified LEFT-crop frame

against three factor types:
  - left reprojection:  project MANO joints (at t) into the left pinhole
  - right reprojection: project (joints at t, shifted -baseline x) into right
  - temporal:           smoothness on (pose, t) between consecutive frames

Reprojection is weighted PER KEYPOINT GROUP (wrist/mcp/pip/tip): the wrist is
root-relative (0,0,0) so it constrains only t (letting stereo disparity fix
metric depth), while fingertips — single skinned verts, noisiest + most
pose-sensitive — are down-weighted. There is NO pose prior: stereo disparity
gives strictly better depth/pose than WiLoR's monocular regression, so pulling
back toward it would only fight the signal we trust. betas stay frozen to WiLoR.

This resolves WiLoR's monocular depth ambiguity (the right-eye term pins down t)
and MANO's anatomical model regularizes per-keypoint disparity noise.

Run:  python stereo_optimize.py --jsonl out/pinhole_stereo/hands.jsonl \
          --mano /tmp/mano_jax.npz --out out/pinhole_stereo/hands3d.jsonl
"""
import argparse
import json

import jax
import jax.numpy as jnp
import jaxlie
import numpy as np

import jaxls
import mano_jax as MJ


# ---- variables -------------------------------------------------------------
# Pose is carried as 16 per-joint quaternions (wxyz) on the SO(3) manifold, à la
# egoallo: the optimizer's tangent step is an SO(3).exp retraction, so updates
# stay on the rotation manifold (no axis-angle singularity / non-uniform metric).
# tangent_dim = 16*3, the variable value is (16,4) unit quats.
class PoseVar(
    jaxls.Var[jax.Array],
    default_factory=lambda: jnp.tile(jnp.array([1.0, 0.0, 0.0, 0.0]), (16, 1)),
    retract_fn=lambda val, delta: (
        jaxlie.SO3(val) @ jaxlie.SO3.exp(delta.reshape(16, 3))
    ).wxyz,
    tangent_dim=16 * 3,
):
    """16 per-joint rotations (global + 15 fingers) as wxyz quats, left frame."""


class TransVar(jaxls.Var[jax.Array], default_factory=lambda: jnp.array([0., 0., 0.5])):
    """MANO-root translation in the rectified-left-crop camera frame (metres)."""


def project(joints_cam, f_px, out_size):
    """Pinhole project (N,3) camera-frame points → (N,2) crop pixels."""
    c = (out_size - 1) / 2.0
    x = f_px * joints_cam[:, 0] / joints_cam[:, 2] + c
    y = f_px * joints_cam[:, 1] / joints_cam[:, 2] + c
    return jnp.stack([x, y], axis=-1)


# OpenPose-21 keypoint groups (after joint_map remap). Used to build the
# per-keypoint reprojection weight vector. Wrist is root-relative (0,0,0) in the
# MANO frame, so its reproj depends ONLY on translation t — weighting it up is
# how we let stereo disparity pin down metric depth. Tips are single skinned
# verts at the chain ends: noisiest in detection + most pose-sensitive, so we
# weight them DOWN, not up.
KP_GROUPS = {
    "wrist": [0],
    "mcp":   [1, 5, 9, 13, 17],          # finger base joints
    "pip":   [2, 3, 6, 7, 10, 11, 14, 15, 18, 19],  # intermediate (PIP/DIP)
    "tip":   [4, 8, 12, 16, 20],         # fingertips (skinned verts)
}


def kp_weights(w_wrist, w_mcp, w_pip, w_tip):
    """Build a (21,) per-keypoint weight vector from the four group weights."""
    w = np.ones(21, np.float32)
    for name, val in (("wrist", w_wrist), ("mcp", w_mcp),
                      ("pip", w_pip), ("tip", w_tip)):
        w[KP_GROUPS[name]] = val
    return jnp.asarray(w)


def make_costs(M, data, w_temporal, huber_px):
    """Build batched jaxls costs from the per-frame stacked arrays in ``data``."""
    n = data["pose0"].shape[0]
    fids = jnp.arange(n)

    # Residuals are normalized to image fractions by dividing pixel error by
    # max(w,h) = out_size. This keeps the reprojection residual O(0.01-0.1)
    # instead of O(10-100) px, so squared costs stay O(1) (numerically stable)
    # and sit on the same scale as the pose prior (radians). The Huber knee is
    # given in px and normalized to match.
    norm = float(data["out_size"])
    huber_n = huber_px / norm

    def huber_w(res):
        a = jnp.abs(res) + 1e-8
        return jax.lax.stop_gradient(jnp.where(a > huber_n, huber_n / a, 1.0))

    # beta (MANO shape) is FROZEN to WiLoR's per-frame estimate — passed as a
    # constant into the projection, not a variable. Hand shape shouldn't vary
    # per frame and isn't useful to fit at this stage, so dropping it removes
    # 10 vars/frame and the competing beta prior.
    # forward-mode AD: the residual has few inputs (51 tangent dims) relative to
    # outputs, and jacfwd benchmarked ~2x faster than jacrev on the MANO chain.
    @jaxls.Cost.factory(jac_mode="forward")
    def reproj(vals, pose_v, t_v, beta_fixed, obs2d, valid, f_px, baseline_x, conf):
        R = jaxlie.SO3(vals[pose_v]).as_matrix()           # (16,4)quat -> (16,3,3)
        joints = MJ.mano_forward_R(M, R, beta_fixed)
        cam = joints + vals[t_v][None, :] - jnp.array([baseline_x, 0.0, 0.0])[None, :]
        proj = project(cam, f_px, data["out_size"])
        # conf carries the per-keypoint group weight (wrist/mcp/pip/tip).
        res = (proj - obs2d) / norm * valid[:, None] * conf[:, None]
        return (res * jnp.sqrt(huber_w(res))).ravel()

    @jaxls.Cost.factory
    def temporal_pose(vals, a, b, w):
        # geodesic difference between consecutive frames' per-joint rotations
        res = (jaxlie.SO3(vals[a]).inverse() @ jaxlie.SO3(vals[b])).log()
        return w * res.reshape(-1)

    @jaxls.Cost.factory
    def temporal(vals, a, b, w):
        return w * (vals[a] - vals[b])

    costs = []
    # left eye: baseline_x = 0 (left camera is the reference frame)
    costs.append(reproj(PoseVar(fids), TransVar(fids), data["beta0"],
                        data["kpL"], data["validL"], data["f_px"],
                        jnp.zeros(n), data["confL"]))
    # right eye: shift by +baseline along x (point moves to right-cam frame)
    costs.append(reproj(PoseVar(fids), TransVar(fids), data["beta0"],
                        data["kpR"], data["validR"], data["f_px"],
                        data["baseline"], data["confR"]))
    # temporal on consecutive frames of the SAME track (only when enabled).
    # NB: couples adjacent KEPT frames; if frames were dropped these aren't
    # adjacent in real time — fine while w_temporal is small, revisit later.
    if w_temporal > 0:
        a, b = jnp.arange(n - 1), jnp.arange(1, n)
        costs.append(temporal_pose(PoseVar(a), PoseVar(b), w_temporal))
        costs.append(temporal(TransVar(a), TransVar(b), w_temporal * 5.0))
    return costs, fids


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--mano", default="/tmp/mano_jax.npz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--hand", choices=["left", "right"], default="right",
                    help="which hand track (is_right) to optimize")
    ap.add_argument("--w-temporal", type=float, default=10.0)
    # per-keypoint reprojection group weights (residual-space; cost ∝ w²).
    # wrist high: it's root-relative (0,0,0) so it only constrains translation
    # → lets stereo disparity fix metric depth. tips low: noisy + pose-sensitive.
    ap.add_argument("--w-wrist", type=float, default=2.0)
    ap.add_argument("--w-mcp", type=float, default=1.0)
    ap.add_argument("--w-pip", type=float, default=1.0)
    ap.add_argument("--w-tip", type=float, default=0.5)
    ap.add_argument("--huber-px", type=float, default=10.0)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--frame-min", type=int, default=None)
    ap.add_argument("--frame-max", type=int, default=None)
    ap.add_argument("--linear", default="conjugate_gradient",
                    help="jaxls linear solver: conjugate_gradient | dense_cholesky | cholmod")
    ap.add_argument("--dy-thresh", type=float, default=8.0,
                    help="max |y_left-y_right| (px) for a keypoint to be an inlier")
    ap.add_argument("--min-inliers", type=int, default=12,
                    help="drop a hand if fewer than this many keypoints survive")
    args = ap.parse_args()

    M = MJ.load_mano(args.mano)
    want_right = 1 if args.hand == "right" else 0

    # Gather the single-best hand of the requested handedness per frame, and
    # build robustness masks. The crops are stereo-RECTIFIED, so rows are
    # epipolar lines: a keypoint whose left/right y disagree by more than
    # --dy-thresh px is a bad cross-eye match → mask it out. Frames with fewer
    # than --min-inliers surviving keypoints are dropped entirely (one eye's
    # detection failed). This is the key fix for outliers blowing up the solve.
    rows = [json.loads(l) for l in open(args.jsonl)]
    frames, pose0, beta0, kpL, kpR, fpx, valid = ([] for _ in range(7))
    n_dropped_frame = 0
    n_masked_kp = 0
    n_kp_total = 0
    for d in rows:
        if args.frame_min is not None and d["frame"] < args.frame_min:
            continue
        if args.frame_max is not None and d["frame"] > args.frame_max:
            continue
        cand = [h for h in d["hands"] if h["is_right"] == want_right]
        if not cand:
            continue
        h = max(cand, key=lambda x: (x["bbox"][2] - x["bbox"][0]))  # largest bbox
        kL = np.array(h["kp_left"], np.float32)
        kR = np.array(h["kp_right"], np.float32)
        dy = np.abs(kL[:, 1] - kR[:, 1])           # epipolar deviation per kp
        disp = kL[:, 0] - kR[:, 0]                 # must be positive (left of right)
        v = (dy < args.dy_thresh) & (disp > 0.5)   # per-keypoint inlier mask
        n_kp_total += 21
        n_masked_kp += int(21 - v.sum())
        if v.sum() < args.min_inliers:             # whole-hand detection failure
            n_dropped_frame += 1
            continue
        frames.append(d["frame"])
        pose0.append(np.array(h["global_orient"] + h["hand_pose"], np.float32))
        beta0.append(np.array(h["betas"], np.float32))
        kpL.append(kL)
        kpR.append(kR)
        fpx.append(h["f_px"])
        valid.append(v.astype(np.float32))
    n = len(frames)
    assert n > 0, "no frames survived robust filtering"
    # per-keypoint group weight vector (wrist/mcp/pip/tip), same for both eyes.
    kpw = kp_weights(args.w_wrist, args.w_mcp, args.w_pip, args.w_tip)
    confL = confR = np.broadcast_to(np.array(kpw), (n, 21)).copy()
    print(f"hand={args.hand}: {n} frames kept; dropped {n_dropped_frame} "
          f"(too few inliers); masked {n_masked_kp}/{n_kp_total} keypoints "
          f"(|dy|>{args.dy_thresh}px)")
    baseline = json.loads(open(args.jsonl).readline())["baseline"]
    out_size = next((h["out_size"] for d in rows for h in d["hands"]), 256)
    print(f"hand={args.hand}: {n} frames with a detection")

    pose0_arr = jnp.asarray(np.stack(pose0))                     # (n,48) axis-angle
    # convert WiLoR axis-angle init -> per-joint quats (n,16,4) wxyz
    quat0 = jax.vmap(lambda p: jaxlie.SO3.exp(p.reshape(16, 3)).wxyz)(pose0_arr)
    data = {
        "pose0": pose0_arr,
        "quat0": quat0,
        "beta0": jnp.asarray(np.stack(beta0)),
        "kpL": jnp.asarray(np.stack(kpL)),
        "kpR": jnp.asarray(np.stack(kpR)),
        "validL": jnp.asarray(np.stack(valid)),
        "validR": jnp.asarray(np.stack(valid)),
        "confL": jnp.asarray(confL),
        "confR": jnp.asarray(confR),
        "f_px": jnp.asarray(np.array(fpx, np.float32)),
        "baseline": jnp.full(n, baseline, jnp.float32),
        "out_size": out_size,
    }

    costs, fids = make_costs(M, data, args.w_temporal, args.huber_px)

    # init: WiLoR pose/betas; translation from per-frame triangulation over the
    # INLIER keypoints only (masked disparities skew the median otherwise).
    disp = jnp.clip(data["kpL"][:, :, 0] - data["kpR"][:, :, 0], 1.0, None)
    z_per_kp = data["f_px"][:, None] * baseline / disp           # (n,21)
    vmask = data["validL"] > 0.5
    z_masked = jnp.where(vmask, z_per_kp, jnp.nan)
    z0 = jnp.nanmedian(z_masked, axis=1)                        # (n,)
    z0 = jnp.nan_to_num(z0, nan=0.5)
    # back-project the crop centre at z0 → x,y (centre ray is ~optical axis)
    t_init = jnp.stack([jnp.zeros(n), jnp.zeros(n), z0], axis=1)

    init = jaxls.VarValues.make([
        PoseVar(fids).with_value(data["quat0"]),
        TransVar(fids).with_value(t_init),
    ])
    problem = jaxls.LeastSquaresProblem(
        costs, [PoseVar(fids), TransVar(fids)]).analyze()
    import time
    t = time.time()
    # LM + dense Cholesky: plain Gauss-Newton (trust_region=None) diverges to
    # NaN here, and the default CG linear solver fits poorly; LM + direct solve
    # converges to ~2px reprojection.
    sol = problem.solve(init, trust_region=jaxls.TrustRegionConfig(),
                        linear_solver=args.linear,
                        termination=jaxls.TerminationConfig(max_iterations=args.iters),
                        verbose=True)
    print(f"solved {n} frames in {time.time()-t:.1f}s")

    quat = np.array(sol[PoseVar]); trans = np.array(sol[TransVar])  # quat (n,16,4)
    quat0_np = np.array(data["quat0"])
    beta = np.array(data["beta0"])  # frozen to WiLoR's estimate
    valid_np = np.array(data["validL"])  # (n,21) inlier mask
    kpL_np = np.array(data["kpL"]); kpR_np = np.array(data["kpR"]); fpx_np = np.array(fpx)

    def fk_R(quat_i, beta_i):
        R = np.array(jaxlie.SO3(jnp.asarray(quat_i)).as_matrix())   # (16,3,3)
        return np.array(MJ.mano_forward_R(M, jnp.asarray(R), jnp.asarray(beta_i)))

    def reproj_px(quat_i, beta_i, t_i, fpx_i, bx):
        j = fk_R(quat_i, beta_i)
        cam = j + t_i[None] - np.array([bx, 0, 0])[None]
        c = (out_size - 1) / 2.0
        return np.stack([fpx_i * cam[:, 0] / cam[:, 2] + c,
                         fpx_i * cam[:, 1] / cam[:, 2] + c], 1)

    # --- detailed diagnostics --------------------------------------------
    # per-keypoint reproj error, split inlier vs outlier, both eyes
    perkp_in, perkp_out = [], []
    frame_med_in = []
    for i in range(n):
        pL = reproj_px(quat[i], beta[i], trans[i], fpx_np[i], 0.0)
        pR = reproj_px(quat[i], beta[i], trans[i], fpx_np[i], baseline)
        eL = np.linalg.norm(pL - kpL_np[i], axis=1)
        eR = np.linalg.norm(pR - kpR_np[i], axis=1)
        m = valid_np[i] > 0.5
        perkp_in += eL[m].tolist() + eR[m].tolist()
        perkp_out += eL[~m].tolist() + eR[~m].tolist()
        if m.any():
            frame_med_in.append(np.concatenate([eL[m], eR[m]]).mean())
    perkp_in = np.array(perkp_in); perkp_out = np.array(perkp_out)
    fmi = np.array(frame_med_in)
    d_arr = trans[:, 2]
    # geodesic drift from WiLoR init: per-frame RMS of per-joint rotation angles
    rel = jax.vmap(lambda a, b: (jaxlie.SO3(a).inverse() @ jaxlie.SO3(b)).log())(
        jnp.asarray(quat0_np), jnp.asarray(quat))         # (n,16,3)
    dpose = np.array(jnp.linalg.norm(rel.reshape(rel.shape[0], -1), axis=1))  # rad

    def pct(a, ps=(50, 90, 99)):
        return "  ".join(f"p{p}={np.percentile(a,p):.2f}" for p in ps)
    print("\n================ OPTIMIZATION DIAGNOSTICS ================")
    print(f"frames solved: {n}   iters: {args.iters}   linear: {args.linear}")
    print(f"weights: temporal={args.w_temporal} huber_px={args.huber_px}  "
          f"kp[wrist={args.w_wrist} mcp={args.w_mcp} pip={args.w_pip} "
          f"tip={args.w_tip}]  (NO pose prior; beta FROZEN to WiLoR)")
    print(f"\nREPROJ ERR inliers (px, {len(perkp_in)} kp): "
          f"mean={perkp_in.mean():.2f}  {pct(perkp_in)}")
    if len(perkp_out):
        print(f"REPROJ ERR outliers(px, {len(perkp_out)} kp): "
              f"mean={perkp_out.mean():.2f}  {pct(perkp_out)}  (masked, not fit)")
    print(f"per-frame mean inlier err: mean={fmi.mean():.2f}  {pct(fmi)}")
    print(f"\nDEPTH (m): median={np.median(d_arr):.3f} mean={d_arr.mean():.3f} "
          f"std={d_arr.std():.3f}  range=[{d_arr.min():.3f},{d_arr.max():.3f}]")
    print(f"  depth percentiles: {pct(d_arr,(5,50,95))}")
    print(f"\nPARAM DRIFT from WiLoR init:")
    print(f"  |Δpose| (rad): mean={dpose.mean():.3f} {pct(dpose)}  (15 joints+global)")
    print(f"  trans xy spread (m): x[{trans[:,0].min():.2f},{trans[:,0].max():.2f}] "
          f"y[{trans[:,1].min():.2f},{trans[:,1].max():.2f}]")
    # temporal jitter (consecutive KEPT frames; gap-agnostic)
    if n > 1:
        dj = np.abs(np.diff(d_arr)) * 1000
        print(f"\nTEMPORAL (consecutive kept frames):")
        print(f"  depth jump (mm): median={np.median(dj):.1f} {pct(dj,(50,90,99)).replace('p','p')}")
    print("==========================================================\n")

    with open(args.out, "w") as f:
        for i, fr in enumerate(frames):
            joints = fk_R(quat[i], beta[i])
            j_world = joints + trans[i][None]
            f.write(json.dumps({
                "frame": int(fr), "is_right": want_right,
                "trans": trans[i].tolist(), "depth_m": float(trans[i][2]),
                "joints_3d_cam": j_world.tolist(),  # rectified-left-crop frame
            }) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
