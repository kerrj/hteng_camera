"""Stereo MANO bundle adjustment over a hand track, with jaxls.

Keypoints-only (no images, no ViT) — so a dense, temporally-coupled solve over
the whole video is cheap (seconds on CPU). For each frame we have WiLoR's
per-eye 2D keypoints in two baseline-aligned pinhole crops that VERGE on the
hand (both virtual cameras aimed at the triangulated point P; rows stay epipolar
but the optical axes are NOT parallel). We optimize, per frame:

    pose       (16,4) per-joint unit quats (wxyz) on the SO(3) manifold;
                      joint 0 = global_orient, 1..15 = finger joints. The
                      optimizer retracts tangent steps via SO3.exp (egoallo style).
    betas      (10,) shape — FROZEN to WiLoR's per-frame estimate
    t          (3,)  MANO-root translation in the LEFT-VIRTUAL crop frame

against three factor types:
  - left reprojection:  project MANO joints (at t) into the left virtual cam
  - right reprojection: project (R_lr @ (joints+t) + t_lr) into the right cam,
                        where R_lr = Rv_r^T Rs Rv_l, t_lr = Rv_r^T ts encode the
                        VERGED stereo geometry exactly (no baseline/disparity
                        shortcut — that only held for parallel axes).
  - temporal:           smoothness on (pose, t) between consecutive frames

Reprojection is weighted PER KEYPOINT GROUP (wrist/mcp/pip/tip; default uniform).
There is NO pose prior: stereo gives strictly better depth/pose than WiLoR's
monocular regression. betas stay frozen to WiLoR.

Results are converted from the per-frame left-virtual frame back to the common
LEFT-FISHEYE (optical-axis) frame before saving (joints_3d_cam, trans, depth_m,
global_orient_R), so they're comparable across frames.

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


def make_costs(M, data, w_temporal, huber_px, w_shape):
    """Build batched jaxls costs from the per-frame stacked arrays in ``data``."""
    n = data["pose0"].shape[0]
    fids = jnp.arange(n)

    # Residuals are normalized to image fractions by dividing pixel error by
    # max(w,h) = out_size. This keeps the reprojection residual O(0.01-0.1)
    # instead of O(10-100) px, so squared costs stay O(1) (numerically stable)
    # and on the same radians-scale as the shape prior. The Huber knee is given
    # in px and normalized by the SAME factor, so |res|>huber_n is exactly
    # "pixel error > huber_px" — the normalization cancels in the threshold.
    norm = float(data["out_size"])
    huber_n = huber_px / norm

    def huber_w(res_px_norm):
        # IRLS weight: returning res*sqrt(w) makes jaxls' squared cost = w*res^2,
        # i.e. linear (robust) beyond the knee, quadratic within. Keyed on the
        # RAW normalized pixel error only — NOT scaled by valid/conf, so a
        # down-weighted keypoint keeps the same px knee (decoupled robustness).
        a = jnp.abs(res_px_norm) + 1e-8
        return jax.lax.stop_gradient(jnp.where(a > huber_n, huber_n / a, 1.0))

    # beta (MANO shape) is FROZEN to WiLoR's per-frame estimate — passed as a
    # constant into the projection, not a variable. Hand shape shouldn't vary
    # per frame and isn't useful to fit at this stage, so dropping it removes
    # 10 vars/frame and the competing beta prior.
    # forward-mode AD: the residual has few inputs (51 tangent dims) relative to
    # outputs, and jacfwd benchmarked ~2x faster than jacrev on the MANO chain.
    # mirror: +1 for right hand, -1 for left. mano_jax is MANO_RIGHT; WiLoR
    # mirrors left hands by negating joint x (and axis-angle comps 1,2), so its
    # stored left-hand pose only matches the keypoints after we x-negate the
    # forward-kinematics output — the canonical MANO left-from-right trick.
    mirror = float(data["mirror"])

    # Pose + translation live in the LEFT-VIRTUAL crop frame. The two virtual
    # cameras VERGE on the hand (they do NOT share orientation), so each eye
    # projects through its OWN rigid transform — NOT a baseline x-shift:
    #     left  eye:  x_l = x                          (R=I, t=0)
    #     right eye:  x_r = R_lr @ x + t_lr
    #         R_lr = Rv_r^T Rs Rv_l,  t_lr = Rv_r^T ts  (precomputed per frame)
    # Both eyes share f_px and principal point = crop centre. Verified to machine
    # precision in test_projection_math.py (P reprojects to both crop centres).
    out_size = data["out_size"]

    def eye_residual(x, obs2d, conf, f_px, R_e, t_e):
        """Reproj residual of placed joints `x` (left-virtual) into one eye."""
        cam = x @ R_e.T + t_e[None, :]                     # → this eye's frame
        proj = project(cam, f_px, out_size)
        res_px = (proj - obs2d) / norm                     # px-true (drives Huber)
        # conf is the per-keypoint group weight; applied AFTER the Huber weight.
        return res_px * conf[:, None] * jnp.sqrt(huber_w(res_px))

    # BOTH eyes in ONE cost: the MANO forward kinematics (the expensive part, and
    # what forward-mode AD differentiates through) runs ONCE per frame; the
    # shared joints are then projected into each eye. Two separate per-eye costs
    # would run FK + its Jacobian twice.
    @jaxls.Cost.factory(jac_mode="forward")
    def reproj(vals, pose_v, t_v, beta_fixed,
               obsL, fL, RL, tL, obsR, fR, RR, tR, conf):
        R = jaxlie.SO3(vals[pose_v]).as_matrix()           # (16,4)quat -> (16,3,3)
        joints = MJ.mano_forward_R(M, R, beta_fixed)
        joints = joints.at[:, 0].multiply(mirror)          # left-hand x-mirror
        x = joints + vals[t_v][None, :]                    # left-virtual frame
        rL = eye_residual(x, obsL, conf, fL, RL, tL)
        rR = eye_residual(x, obsR, conf, fR, RR, tR)
        return jnp.concatenate([rL.ravel(), rR.ravel()])

    # SHAPE prior: pull the 15 INTERNAL finger-joint rotations toward the
    # geodesic mean of the two eyes' WiLoR estimates. Targets exactly the
    # rank-deficient finger DOF (under-constrained along the view ray); leaves
    # global orient (joint 0) and translation/depth untouched — depth stays
    # purely stereo-driven. Residual is geodesic (SO3 log), in radians.
    @jaxls.Cost.factory
    def shape_prior(vals, pose_v, mean15):
        R = jaxlie.SO3(vals[pose_v][1:])                      # joints 1..15
        res = (jaxlie.SO3(mean15).inverse() @ R).log()        # (15,3)
        return (w_shape * res).reshape(-1)

    @jaxls.Cost.factory
    def temporal_pose(vals, a, b, w):
        # geodesic difference between consecutive frames' per-joint rotations
        res = (jaxlie.SO3(vals[a]).inverse() @ jaxlie.SO3(vals[b])).log()
        return w * res.reshape(-1)

    @jaxls.Cost.factory
    def temporal(vals, a, b, w):
        return w * (vals[a] - vals[b])

    costs = []
    eye_I = jnp.broadcast_to(jnp.eye(3), (n, 3, 3))     # left-virtual cam = identity
    eye_0 = jnp.zeros((n, 3))
    # one cost, both eyes: left = (I,0) (pose+t already in left-virtual frame),
    # right = (R_lr, t_lr) (verged rigid transform into the right-virtual frame).
    costs.append(reproj(PoseVar(fids), TransVar(fids), data["beta0"],
                        data["kpL"], data["f_px"], eye_I, eye_0,
                        data["kpR"], data["f_px"], data["R_lr"], data["t_lr"],
                        data["confL"]))
    # shape prior: internal finger pose toward the two-eye geodesic mean
    if w_shape > 0:
        costs.append(shape_prior(PoseVar(fids), data["shape_mean"]))
    # temporal on consecutive frames of the SAME track (only when enabled).
    # NB: couples adjacent KEPT frames; if frames were dropped these aren't
    # adjacent in real time — fine while w_temporal is small, revisit later.
    if w_temporal > 0:
        a, b = jnp.arange(n - 1), jnp.arange(1, n)
        costs.append(temporal_pose(PoseVar(a), PoseVar(b), w_temporal))
        costs.append(temporal(TransVar(a), TransVar(b), w_temporal * 5.0))
    return costs, fids


def build_data(jsonl, calib_dir, left_serial, right_serial, hand,
               frame_min=None, frame_max=None,
               w_wrist=1.0, w_mcp=1.0, w_pip=1.0, w_tip=1.0):
    """Gather per-frame arrays + precompute the optimizer's ``data`` dict.

    Returns (data, t_init, frames, Rvl_arr, want_right). No outlier rejection —
    every detected hand's 21 keypoints in both eyes are kept.
    """
    want_right = 1 if hand == "right" else 0
    # stereo extrinsics (X_right = Rs @ X_left + ts), for the verged per-eye
    # projection transforms R_lr, t_lr.
    st = json.load(open(f"{calib_dir}/stereo_{left_serial}_{right_serial}.json"))
    Rs = np.array(st["R"], np.float32)
    ts = np.array(st["t"], np.float32).reshape(3)

    rows = [json.loads(l) for l in open(jsonl)]
    frames, pose0, beta0, kpL, kpR, fpx = ([] for _ in range(6))
    Rvl, Parr = [], []
    R_lr, t_lr = [], []
    hp_left, hp_right = [], []     # per-eye 15-joint internal pose (axis-angle)
    for d in rows:
        if frame_min is not None and d["frame"] < frame_min:
            continue
        if frame_max is not None and d["frame"] > frame_max:
            continue
        cand = [h for h in d["hands"] if h["is_right"] == want_right]
        if not cand:
            continue
        h = max(cand, key=lambda x: (x["bbox"][2] - x["bbox"][0]))  # largest bbox
        Rv_l = np.array(h["Rv_l"], np.float32)
        Rv_r = np.array(h["Rv_r"], np.float32)
        frames.append(d["frame"])
        pose0.append(np.array(h["global_orient"] + h["hand_pose"], np.float32))
        beta0.append(np.array(h["betas"], np.float32))
        kpL.append(np.array(h["kp_left"], np.float32))
        kpR.append(np.array(h["kp_right"], np.float32))
        fpx.append(h["f_px"])
        Rvl.append(Rv_l)
        Parr.append(np.array(h["P"], np.float32))
        R_lr.append(Rv_r.T @ Rs @ Rv_l)            # left-virtual → right-virtual
        t_lr.append(Rv_r.T @ ts)
        hp_left.append(np.array(h["hand_pose"], np.float32))        # (45,)
        hp_right.append(np.array(h["hand_pose_right"], np.float32))  # (45,)
    n = len(frames)
    assert n > 0, "no frames with a detection of the requested hand"
    # all keypoints valid (no rejection); per-keypoint group weight (uniform).
    valid = [np.ones(21, np.float32)] * n
    kpw = kp_weights(w_wrist, w_mcp, w_pip, w_tip)
    confL = confR = np.broadcast_to(np.array(kpw), (n, 21)).copy()
    out_size = next((h["out_size"] for d in rows for h in d["hands"]), 256)
    print(f"hand={hand}: {n} frames with a detection (no filtering)")

    pose0_arr = np.stack(pose0).reshape(n, 16, 3)                # (n,16,3) axis-angle
    # WiLoR stores left-hand pose in a MIRRORED convention (negate axis-angle
    # comps 1,2 — a reflection conjugation R->S R S). mano_jax is MANO_RIGHT, so
    # un-mirror the init back to right-hand rotation space; the FK output is then
    # x-negated (the `mirror` term) to place the left hand. Both halves of the
    # reflection are needed: comps 1,2 here + output-x in the cost.
    if not want_right:
        pose0_arr[:, :, 1:3] *= -1.0
    pose0_arr = jnp.asarray(pose0_arr.reshape(n, 48))
    # convert WiLoR axis-angle init -> per-joint quats (n,16,4) wxyz
    quat0 = jax.vmap(lambda p: jaxlie.SO3.exp(p.reshape(16, 3)).wxyz)(pose0_arr)

    # SHAPE TARGET: the geodesic mean of the left & right eyes' INTERNAL pose
    # (15 finger joints). hand_pose is parent-relative, hence viewpoint-invariant,
    # so the two eyes are independent noisy measurements of the same hand shape;
    # their mean denoises the rank-deficient finger DOF without touching global
    # orient or translation/depth. Un-mirror both eyes for left hands (same
    # reflection as the init) so the target lives in MANO_RIGHT quat space.
    hpL = np.stack(hp_left).reshape(n, 15, 3)
    hpR = np.stack(hp_right).reshape(n, 15, 3)
    if not want_right:
        hpL[:, :, 1:3] *= -1.0
        hpR[:, :, 1:3] *= -1.0
    qL = jax.vmap(lambda p: jaxlie.SO3.exp(p).wxyz)(jnp.asarray(hpL))   # (n,15,4)
    qR = jax.vmap(lambda p: jaxlie.SO3.exp(p).wxyz)(jnp.asarray(hpR))
    # R_mean = R_l ⊕ ½·log(R_l^T R_r)  (geodesic midpoint on SO3)
    rel = jax.vmap(jax.vmap(lambda a, b:
        (jaxlie.SO3(a).inverse() @ jaxlie.SO3(b)).log()))(qL, qR)      # (n,15,3)
    shape_mean = jax.vmap(jax.vmap(lambda a, d:
        (jaxlie.SO3(a) @ jaxlie.SO3.exp(0.5 * d)).wxyz))(qL, rel)      # (n,15,4)

    Rvl_arr = np.stack(Rvl)                                       # (n,3,3)
    P_arr = np.stack(Parr)                                        # (n,3) left-fisheye
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
        "R_lr": jnp.asarray(np.stack(R_lr)),         # (n,3,3) left-virt→right-virt
        "t_lr": jnp.asarray(np.stack(t_lr)),         # (n,3)
        "shape_mean": shape_mean,                    # (n,15,4) geodesic-mean internal pose
        "out_size": out_size,
        "mirror": 1.0 if want_right else -1.0,   # left-hand x-mirror (MANO_RIGHT)
    }

    # init: WiLoR pose; translation from the triangulated point P (left-fisheye
    # frame) expressed in the LEFT-VIRTUAL frame: t0 = Rv_l^T @ P. The MANO root
    # is wrist-centred and root-relative, so placing the root at P (≈ hand centre)
    # is a good start. Far cleaner than the old z-from-disparity init.
    t_init = jnp.asarray(np.einsum("nij,nj->ni", Rvl_arr.transpose(0, 2, 1), P_arr))
    return data, t_init, frames, Rvl_arr, want_right


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--mano", default="/tmp/mano_jax.npz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--calib-dir", default="../long-test1")
    ap.add_argument("--left-serial", default="046060323008")
    ap.add_argument("--right-serial", default="046060323001")
    ap.add_argument("--hand", choices=["left", "right"], default="right",
                    help="which hand track (is_right) to optimize")
    ap.add_argument("--w-temporal", type=float, default=10.0)
    ap.add_argument("--w-shape", type=float, default=0.1,
                    help="weight pulling internal finger pose toward the two-eye "
                         "geodesic-mean shape (radians-scale geodesic residual). "
                         "Sweep (2026-06-29): 0->54px reproj p99 & 6rad pose drift; "
                         "0.1 -> 1.9px p50 / 10px p99 / 0.3rad drift; 1.0 over-"
                         "constrains. 0.1 is the knee.")
    # per-keypoint reprojection group weights (residual-space; cost ∝ w²).
    # wrist high: it's root-relative (0,0,0) so it only constrains translation
    # → lets stereo disparity fix metric depth. tips low: noisy + pose-sensitive.
    # Default uniform: a sweep (2026-06-29) showed asymmetric group weights only
    # hurt — down-weighting tips let fingers curl into bad minima (p50 2.9→5.6px)
    # and up-weighting the wrist (root-relative → only constrains t) did nothing.
    # Kept tunable as the hook for future per-frame detector confidence.
    ap.add_argument("--w-wrist", type=float, default=1.0)
    ap.add_argument("--w-mcp", type=float, default=1.0)
    ap.add_argument("--w-pip", type=float, default=1.0)
    ap.add_argument("--w-tip", type=float, default=1.0)
    ap.add_argument("--huber-px", type=float, default=5.0,
                    help="Huber knee in PIXELS (cost is quadratic within, linear "
                         "beyond). With the shape prior, inlier reproj is p90~4.5px, "
                         "so 5px robustifies the noisy mid-tail; was 10px (barely "
                         "active). Normalized internally to match residual scale.")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--frame-min", type=int, default=None)
    ap.add_argument("--frame-max", type=int, default=None)
    ap.add_argument("--linear", default="dense_cholesky",
                    help="jaxls linear solver for the Schur-reduced system: "
                         "dense_cholesky (default; fast+deterministic) | "
                         "conjugate_gradient | cholmod")
    args = ap.parse_args()

    M = MJ.load_mano(args.mano)
    data, t_init, frames, Rvl_arr, want_right = build_data(
        args.jsonl, args.calib_dir, args.left_serial, args.right_serial,
        args.hand, args.frame_min, args.frame_max,
        args.w_wrist, args.w_mcp, args.w_pip, args.w_tip)
    n = len(frames)
    out_size = data["out_size"]

    import time
    # ONE big solve over all n frames. jaxls's Schur elimination (analyze's
    # default "auto") eliminates the per-frame PoseVar block — for the
    # block-diagonal (no-temporal) system the reduced system is just the small
    # TransVar block, factored directly by dense_cholesky. This is ~16x faster
    # than the old chunked CG path (189s -> ~12s for 4400 frames) AND
    # deterministic (CG's shared-lambda + Eisenstat-Walker gave 5-80s variance).
    # It also transfers cleanly to temporal smoothing: a banded coupled system
    # Schur-eliminates the same way, whereas chunking would sever smoothness
    # edges at boundaries.
    fids = jnp.arange(n)
    costs, _ = make_costs(M, data, args.w_temporal, args.huber_px, args.w_shape)
    init = jaxls.VarValues.make([
        PoseVar(fids).with_value(data["quat0"]),
        TransVar(fids).with_value(t_init),
    ])
    prob = jaxls.LeastSquaresProblem(
        costs, [PoseVar(fids), TransVar(fids)]).analyze()
    t = time.time()
    # LM trust region required (plain Gauss-Newton diverges to NaN here).
    sol = prob.solve(init, trust_region=jaxls.TrustRegionConfig(),
                     linear_solver=args.linear,
                     termination=jaxls.TerminationConfig(max_iterations=args.iters),
                     verbose=True)
    quat = np.array(sol[PoseVar])
    trans = np.array(sol[TransVar])
    print(f"solved {n} frames in {time.time()-t:.1f}s")

    quat0_np = np.array(data["quat0"])
    beta = np.array(data["beta0"])  # frozen to WiLoR's estimate
    valid_np = np.array(data["validL"])  # (n,21) inlier mask
    kpL_np = np.array(data["kpL"]); kpR_np = np.array(data["kpR"]); fpx_np = np.array(fpx)
    R_lr_np = np.array(data["R_lr"]); t_lr_np = np.array(data["t_lr"])

    mirror = 1.0 if want_right else -1.0
    def fk_R(quat_i, beta_i):
        R = np.array(jaxlie.SO3(jnp.asarray(quat_i)).as_matrix())   # (16,3,3)
        j = np.array(MJ.mano_forward_R(M, jnp.asarray(R), jnp.asarray(beta_i)))
        j[:, 0] *= mirror                                            # left-hand x-mirror
        return j

    def reproj_px(quat_i, beta_i, t_i, fpx_i, R_e, t_e):
        x = fk_R(quat_i, beta_i) + t_i[None]            # left-virtual frame
        cam = x @ R_e.T + t_e[None]                     # → this eye's frame
        c = (out_size - 1) / 2.0
        return np.stack([fpx_i * cam[:, 0] / cam[:, 2] + c,
                         fpx_i * cam[:, 1] / cam[:, 2] + c], 1)

    # --- detailed diagnostics --------------------------------------------
    # per-keypoint reproj error, split inlier vs outlier, both eyes
    I3 = np.eye(3); z3 = np.zeros(3)
    perkp_in, perkp_out = [], []
    frame_med_in = []
    for i in range(n):
        pL = reproj_px(quat[i], beta[i], trans[i], fpx_np[i], I3, z3)
        pR = reproj_px(quat[i], beta[i], trans[i], fpx_np[i], R_lr_np[i], t_lr_np[i])
        eL = np.linalg.norm(pL - kpL_np[i], axis=1)
        eR = np.linalg.norm(pR - kpR_np[i], axis=1)
        m = valid_np[i] > 0.5
        perkp_in += eL[m].tolist() + eR[m].tolist()
        perkp_out += eL[~m].tolist() + eR[~m].tolist()
        if m.any():
            frame_med_in.append(np.concatenate([eL[m], eR[m]]).mean())
    perkp_in = np.array(perkp_in); perkp_out = np.array(perkp_out)
    fmi = np.array(frame_med_in)
    # report depth in the LEFT-FISHEYE frame (optical-axis z), not the per-frame
    # left-virtual z whose axis is tilted toward the hand. j_fish below uses the
    # same Rv_l transform.
    root_virtual = trans                                         # (n,3) left-virt
    root_fish = np.einsum("nij,nj->ni", Rvl_arr, root_virtual)   # → left-fisheye
    d_arr = root_fish[:, 2]
    # geodesic drift from WiLoR init: per-frame RMS of per-joint rotation angles
    rel = jax.vmap(lambda a, b: (jaxlie.SO3(a).inverse() @ jaxlie.SO3(b)).log())(
        jnp.asarray(quat0_np), jnp.asarray(quat))         # (n,16,3)
    dpose = np.array(jnp.linalg.norm(rel.reshape(rel.shape[0], -1), axis=1))  # rad

    def pct(a, ps=(50, 90, 99)):
        return "  ".join(f"p{p}={np.percentile(a,p):.2f}" for p in ps)
    print("\n================ OPTIMIZATION DIAGNOSTICS ================")
    print(f"frames solved: {n}   iters: {args.iters}   linear: {args.linear}")
    print(f"weights: temporal={args.w_temporal} shape={args.w_shape} "
          f"huber_px={args.huber_px}  kp[wrist={args.w_wrist} mcp={args.w_mcp} "
          f"pip={args.w_pip} tip={args.w_tip}]  (shape prior = two-eye mean; "
          f"NO global/depth prior; beta FROZEN to WiLoR)")
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

    # Convert results from the per-frame LEFT-VIRTUAL frame back to the common
    # LEFT-FISHEYE (optical-axis) frame so they're comparable across frames and
    # ready for temporal/world lifting:
    #   joints_fish = Rv_l @ (joints_virtual + t)
    #   global_orient_fish = Rv_l @ R_global   (rotate the wrist rotation in too)
    with open(args.out, "w") as f:
        for i, fr in enumerate(frames):
            joints_v = fk_R(quat[i], beta[i]) + trans[i][None]      # left-virtual
            j_fish = joints_v @ Rvl_arr[i].T                        # → left-fisheye
            # wrist (global) rotation carried into the fisheye frame
            R_glob_v = np.array(jaxlie.SO3(jnp.asarray(quat[i][0])).as_matrix())
            R_glob_fish = Rvl_arr[i] @ R_glob_v
            f.write(json.dumps({
                "frame": int(fr), "is_right": want_right,
                "trans": root_fish[i].tolist(),          # root in left-fisheye frame
                "depth_m": float(root_fish[i][2]),       # optical-axis depth
                "joints_3d_cam": j_fish.tolist(),        # LEFT-FISHEYE frame
                "Rv_l": Rvl_arr[i].tolist(),             # virtual→fisheye (for ref)
                "global_orient_R": R_glob_fish.tolist(),  # wrist rot in fisheye frame
            }) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
