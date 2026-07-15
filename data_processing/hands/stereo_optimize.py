"""Robust stereo MANO optimization for one hand track.

Input is the per-eye WiLoR keypoints and virtual-camera geometry produced by
``wilor_hands_pinhole.py``. The solver runs in two stages:

1. Fit each frame independently, with a separate LM trust region per frame.
2. Reject invalid geometry and poor stereo fits, then jointly optimize accepted
   consecutive frames with a VIO-world minimum-acceleration prior.

MANO shape is fixed to the track's mean WiLoR estimate. Pose uses 16 SO(3)
variables (wrist plus 15 finger joints); translation is represented in each
frame's left virtual camera. Results are converted to the common left-fisheye
camera frame. Enclosed short gaps can be interpolated after the final quality
gate.

Typical run:
    python stereo_optimize.py --calib-dir ../../long-test2 --hand right
"""
import argparse
import json
import os

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


# OpenPose-21 keypoint groups, used for optional reprojection weighting.
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


def make_costs(M, data, beta, huber_px, w_shape,
               w_accel_pose, w_accel_pose_global, w_accel_trans,
               temporal=True):
    """Build batched jaxls costs from the per-frame stacked arrays in ``data``.

    ``beta`` is fixed for the track. ``temporal=False`` creates the independent
    phase-1 problem; ``temporal=True`` adds acceleration factors for phase 2.
    """
    n = data["kpL"].shape[0]
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

    # Shape is fixed to the track mean. Forward-mode AD is faster here because
    # each frame has 51 tangent inputs and 84 reprojection residuals.
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

    # Both eyes share one cost so MANO forward kinematics and its Jacobian run
    # once per frame.
    beta = jnp.asarray(beta)

    @jaxls.Cost.factory(jac_mode="forward")
    def reproj(vals, pose_v, t_v,
               obsL, fL, RL, tL, obsR, fR, RR, tR, conf):
        R = jaxlie.SO3(vals[pose_v]).as_matrix()
        joints = MJ.mano_forward_R(M, R, beta)
        joints = joints.at[:, 0].multiply(mirror)
        x = joints + vals[t_v][None, :]
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

    # ACCELERATION smoother on consecutive frame-index triples. Each displacement
    # is divided by its observed frame interval, then multiplied by the median
    # interval so equal-cadence triples exactly recover the old second difference.
    # This keeps the tuned weights while correctly handling capture gaps.
    #
    @jaxls.Cost.factory
    def accel_pose(vals, a, b, c, prev_scale, next_scale,
                   R_cw_a, R_cw_b, R_cw_c, R_vl_a, R_vl_b, R_vl_c):
        Ra = jaxlie.SO3(vals[a])
        Rb = jaxlie.SO3(vals[b])
        Rc = jaxlie.SO3(vals[c])

        # Internal rotations are parent-relative and already share a common
        # representation. Express both increments in the middle pose's tangent.
        Ra_internal = jaxlie.SO3(vals[a][1:])
        Rb_internal = jaxlie.SO3(vals[b][1:])
        Rc_internal = jaxlie.SO3(vals[c][1:])
        internal = (
            prev_scale * (Rb_internal.inverse() @ Ra_internal).log()
            + next_scale * (Rb_internal.inverse() @ Rc_internal).log()
        )

        # The wrist rotation is stored in a different virtual crop frame each
        # frame. Lift it through left-fisheye camera coordinates into the fixed
        # VIO world before taking its second difference.
        Rw_a = jaxlie.SO3.from_matrix(R_cw_a @ R_vl_a @ Ra.as_matrix()[0])
        Rw_b = jaxlie.SO3.from_matrix(R_cw_b @ R_vl_b @ Rb.as_matrix()[0])
        Rw_c = jaxlie.SO3.from_matrix(R_cw_c @ R_vl_c @ Rc.as_matrix()[0])
        global_accel = (
            prev_scale * (Rw_b.inverse() @ Rw_a).log()
            + next_scale * (Rw_b.inverse() @ Rw_c).log()
        )
        return jnp.concatenate([
            w_accel_pose_global * global_accel,
            (w_accel_pose * internal).reshape(-1),
        ])

    @jaxls.Cost.factory
    def accel_trans(vals, a, b, c, prev_scale, next_scale,
                    R_cw_a, R_cw_b, R_cw_c,
                    t_cw_a, t_cw_b, t_cw_c,
                    R_vl_a, R_vl_b, R_vl_c):
        p_a = R_cw_a @ R_vl_a @ vals[a] + t_cw_a
        p_b = R_cw_b @ R_vl_b @ vals[b] + t_cw_b
        p_c = R_cw_c @ R_vl_c @ vals[c] + t_cw_c
        velocity_delta = (
            prev_scale * (p_a - p_b) + next_scale * (p_c - p_b))
        return w_accel_trans * velocity_delta

    costs = []
    eye_I = jnp.broadcast_to(jnp.eye(3), (n, 3, 3))     # left-virtual cam = identity
    eye_0 = jnp.zeros((n, 3))
    # one cost, both eyes: left = (I,0) (pose+t already in left-virtual frame),
    # right = (R_lr, t_lr) (verged rigid transform into the right-virtual frame).
    reproj_args = (data["kpL"], data["f_px"], eye_I, eye_0,
                   data["kpR"], data["f_px"], data["R_lr"], data["t_lr"],
                   data["kp_weight"])
    costs.append(reproj(PoseVar(fids), TransVar(fids), *reproj_args))
    # shape prior: internal finger pose toward the two-eye geodesic mean
    if w_shape > 0:
        costs.append(shape_prior(PoseVar(fids), data["shape_mean"]))
    # Acceleration factors only connect consecutive accepted frame triples.
    i0, i1, i2 = data["accel_i0"], data["accel_i1"], data["accel_i2"]
    if temporal and (w_accel_pose > 0 or w_accel_pose_global > 0
                     or w_accel_trans > 0) and i0.shape[0] > 0:
        if w_accel_pose > 0 or w_accel_pose_global > 0:
            costs.append(accel_pose(
                PoseVar(i0), PoseVar(i1), PoseVar(i2),
                data["accel_prev_scale"], data["accel_next_scale"],
                data["R_cw"][i0], data["R_cw"][i1], data["R_cw"][i2],
                data["R_vl"][i0], data["R_vl"][i1], data["R_vl"][i2],
            ))
        if w_accel_trans > 0:
            costs.append(accel_trans(
                TransVar(i0), TransVar(i1), TransVar(i2),
                data["accel_prev_scale"], data["accel_next_scale"],
                data["R_cw"][i0], data["R_cw"][i1], data["R_cw"][i2],
                data["t_cw"][i0], data["t_cw"][i1], data["t_cw"][i2],
                data["R_vl"][i0], data["R_vl"][i1], data["R_vl"][i2],
            ))
    return costs


def build_data(jsonl, calib_dir, left_serial, right_serial, hand,
               frame_min=None, frame_max=None,
               w_wrist=1.0, w_mcp=1.0, w_pip=1.0, w_tip=1.0):
    """Gather per-frame arrays + precompute the optimizer's ``data`` dict.

    Returns all candidate observations. Quality filtering happens after the
    independent phase-1 fits.
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
    kpw = kp_weights(w_wrist, w_mcp, w_pip, w_tip)
    kp_weight = np.broadcast_to(np.array(kpw), (n, 21)).copy()
    out_size = next((h["out_size"] for d in rows for h in d["hands"]), 256)
    print(f"hand={hand}: {n} candidate frames")

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
        "quat0": quat0,
        "beta0": jnp.asarray(np.stack(beta0)),
        "kpL": jnp.asarray(np.stack(kpL)),
        "kpR": jnp.asarray(np.stack(kpR)),
        "kp_weight": jnp.asarray(kp_weight),
        "f_px": jnp.asarray(np.array(fpx, np.float32)),
        "R_lr": jnp.asarray(np.stack(R_lr)),         # (n,3,3) left-virt→right-virt
        "t_lr": jnp.asarray(np.stack(t_lr)),         # (n,3)
        "shape_mean": shape_mean,                    # (n,15,4) geodesic-mean internal pose
        "R_vl": jnp.asarray(Rvl_arr),                # left-virtual -> left-fisheye
        "out_size": out_size,
        "mirror": 1.0 if want_right else -1.0,   # left-hand x-mirror (MANO_RIGHT)
    }

    _set_accel_indices(data, frames)

    # init: WiLoR pose; translation from the triangulated point P (left-fisheye
    # frame) expressed in the LEFT-VIRTUAL frame: t0 = Rv_l^T @ P. The MANO root
    # is wrist-centred and root-relative, so placing the root at P (≈ hand centre)
    # is a good start. Far cleaner than the old z-from-disparity init.
    t_init = jnp.asarray(np.einsum("nij,nj->ni", Rvl_arr.transpose(0, 2, 1), P_arr))
    return data, t_init, frames, Rvl_arr, want_right


def _set_accel_indices(data, frames):
    """Rebuild timestamp-aware acceleration triples after frame filtering."""
    fr = np.asarray(frames)
    times = np.asarray(data.get("frame_time_s", fr), np.float64)
    if times.shape != fr.shape or np.any(np.diff(times) <= 0):
        raise ValueError("hand frame times must be matching and increasing")
    reference_dt = float(data.get("frame_time_reference_s", 1.0))
    if reference_dt <= 0:
        raise ValueError("frame_time_reference_s must be positive")
    mid = np.arange(1, len(fr) - 1)
    adjacent = ((fr[mid] - fr[mid - 1] == 1)
                & (fr[mid + 1] - fr[mid] == 1))
    tri_mid = mid[adjacent]
    data["accel_i0"] = jnp.asarray(tri_mid - 1)
    data["accel_i1"] = jnp.asarray(tri_mid)
    data["accel_i2"] = jnp.asarray(tri_mid + 1)
    data["accel_prev_scale"] = jnp.asarray(
        reference_dt / (times[tri_mid] - times[tri_mid - 1]),
        dtype=jnp.float32)
    data["accel_next_scale"] = jnp.asarray(
        reference_dt / (times[tri_mid + 1] - times[tri_mid]),
        dtype=jnp.float32)


def subset_data(data, keep, frames):
    """Select accepted observations and rebuild their temporal topology."""
    keep = np.asarray(keep)
    n = len(keep)
    out = {}
    for key, value in data.items():
        if key.startswith("accel_"):
            continue
        if hasattr(value, "shape") and getattr(value, "ndim", 0) > 0 \
                and value.shape[0] == n:
            out[key] = value[keep]
        else:
            out[key] = value
    _set_accel_indices(out, frames)
    return out


def solve_independent_frames(M, data, t_init, beta, huber_px, w_shape,
                             iters, batch_size, linear_solver):
    """Solve one jaxls problem per frame, vmapped in fixed-size batches.

    Each problem has its own LM trust-region state. This is intentionally
    different from a block-diagonal multi-frame problem, whose single damping
    value lets one singular frame destabilize every independent fit.
    """
    out_size = data["out_size"]
    mirror = data["mirror"]

    def solve_one(kpL, kpR, f_px, R_lr, t_lr, kp_weight,
                  shape_mean, quat0, trans0):
        one = {
            "kpL": kpL[None],
            "kpR": kpR[None],
            "f_px": f_px[None],
            "R_lr": R_lr[None],
            "t_lr": t_lr[None],
            "kp_weight": kp_weight[None],
            "shape_mean": shape_mean[None],
            "out_size": out_size,
            "mirror": mirror,
            "accel_i0": jnp.zeros((0,), dtype=jnp.int32),
            "accel_i1": jnp.zeros((0,), dtype=jnp.int32),
            "accel_i2": jnp.zeros((0,), dtype=jnp.int32),
            "accel_prev_scale": jnp.zeros((0,), dtype=jnp.float32),
            "accel_next_scale": jnp.zeros((0,), dtype=jnp.float32),
        }
        costs = make_costs(
            M, one, beta, huber_px, w_shape, 0.0, 0.0, 0.0,
            temporal=False)
        ids = jnp.arange(1)
        variables = [PoseVar(ids), TransVar(ids)]
        init = jaxls.VarValues.make([
            PoseVar(ids).with_value(quat0[None]),
            TransVar(ids).with_value(trans0[None]),
        ])
        problem = jaxls.LeastSquaresProblem(costs, variables).analyze()
        solution = problem.solve(
            init,
            trust_region=jaxls.TrustRegionConfig(),
            linear_solver=linear_solver,
            termination=jaxls.TerminationConfig(max_iterations=iters),
            verbose=False)
        return solution[PoseVar][0], solution[TransVar][0]

    solve_batch = jax.jit(jax.vmap(solve_one))
    inputs = (
        data["kpL"], data["kpR"], data["f_px"], data["R_lr"], data["t_lr"],
        data["kp_weight"], data["shape_mean"], data["quat0"], t_init,
    )
    n = len(t_init)
    batch_size = min(max(1, batch_size), n)
    quat_parts, trans_parts = [], []
    for start in range(0, n, batch_size):
        count = min(batch_size, n - start)
        batch = [x[start:start + count] for x in inputs]
        if count < batch_size:
            batch = [
                jnp.concatenate([x, jnp.repeat(x[-1:], batch_size - count, axis=0)])
                for x in batch
            ]
        quat, trans = solve_batch(*batch)
        quat_parts.append(np.asarray(quat[:count]))
        trans_parts.append(np.asarray(trans[:count]))
        if start == 0 or start + count == n or (start + count) % 512 == 0:
            print(f"[phase1/fit] {start + count}/{n}", flush=True)
    return np.concatenate(quat_parts), np.concatenate(trans_parts)


def evaluate_fits(M, data, quat, trans, beta, Rvl_arr):
    """Compute phase-1 geometry and reprojection metrics for quality gating."""
    mirror = float(data["mirror"])
    out_size = data["out_size"]

    R = jax.vmap(lambda q: jaxlie.SO3(q).as_matrix())(jnp.asarray(quat))
    joints = jax.vmap(lambda Ri: MJ.mano_forward_R(M, Ri, beta))(R)
    joints = joints.at[:, :, 0].multiply(mirror)
    cam_l = joints + jnp.asarray(trans)[:, None, :]
    cam_r = (jnp.einsum("nkc,ndc->nkd", cam_l, data["R_lr"])
             + data["t_lr"][:, None, :])

    c = (out_size - 1) / 2.0

    def proj(cam):
        f = data["f_px"][:, None]
        return jnp.stack([
            f * cam[:, :, 0] / cam[:, :, 2] + c,
            f * cam[:, :, 1] / cam[:, :, 2] + c,
        ], axis=-1)

    p_l, p_r = proj(cam_l), proj(cam_r)
    err = jnp.concatenate([
        jnp.linalg.norm(p_l - data["kpL"], axis=2),
        jnp.linalg.norm(p_r - data["kpR"], axis=2),
    ], axis=1)
    root_fish = np.einsum("nij,nj->ni", Rvl_arr, np.asarray(trans))
    return {
        "mean_reproj_px": np.asarray(jnp.mean(err, axis=1)),
        "p90_reproj_px": np.asarray(jnp.percentile(err, 90, axis=1)),
        "median_epipolar_px": np.median(
            np.abs(np.asarray(data["kpL"])[:, :, 1]
                   - np.asarray(data["kpR"])[:, :, 1]), axis=1),
        "min_joint_z_m": np.asarray(
            jnp.minimum(jnp.min(cam_l[:, :, 2], axis=1),
                        jnp.min(cam_r[:, :, 2], axis=1))),
        "depth_m": root_fish[:, 2],
        "root_fish": root_fish,
    }


def quality_mask(metrics, min_depth, max_depth, min_joint_z,
                 max_reproj_px, max_epipolar_px):
    """Return accepted phase-1 fits and a per-reason rejection summary."""
    arrays = list(metrics.values())
    finite = np.ones(len(arrays[0]), dtype=bool)
    for value in arrays:
        finite &= np.all(np.isfinite(value).reshape(len(finite), -1), axis=1)
    checks = {
        "nonfinite": finite,
        "depth": ((metrics["depth_m"] >= min_depth)
                  & (metrics["depth_m"] <= max_depth)),
        "joint_z": metrics["min_joint_z_m"] >= min_joint_z,
        "reprojection": metrics["mean_reproj_px"] <= max_reproj_px,
        "epipolar": metrics["median_epipolar_px"] <= max_epipolar_px,
    }
    keep = np.ones(len(finite), dtype=bool)
    rejected = {}
    for name, passed in checks.items():
        rejected[name] = int(np.sum(keep & ~passed))
        keep &= passed
    return keep, rejected


def quat_slerp(q0, q1, alpha):
    """Shortest-path interpolation of wxyz unit quaternions."""
    q0 = np.asarray(q0, np.float64)
    q1 = np.asarray(q1, np.float64)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = np.clip(dot, -1.0, 1.0)
    if dot > 0.9995:
        q = q0 + alpha * (q1 - q0)
        return (q / np.linalg.norm(q)).astype(np.float32)
    theta = np.arccos(dot)
    q = (np.sin((1.0 - alpha) * theta) * q0
         + np.sin(alpha * theta) * q1) / np.sin(theta)
    return q.astype(np.float32)


def quat_to_matrix(q):
    w, x, y, z = np.asarray(q)
    return np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ], np.float32)


def matrix_to_quat(R):
    """Convert a proper rotation matrix to a wxyz unit quaternion."""
    R = np.asarray(R, np.float64)
    candidates = np.array([
        1 + np.trace(R),
        1 + R[0, 0] - R[1, 1] - R[2, 2],
        1 - R[0, 0] + R[1, 1] - R[2, 2],
        1 - R[0, 0] - R[1, 1] + R[2, 2],
    ])
    i = int(np.argmax(candidates))
    if i == 0:
        q = np.array([candidates[0], R[2, 1] - R[1, 2],
                      R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    elif i == 1:
        q = np.array([R[2, 1] - R[1, 2], candidates[1],
                      R[0, 1] + R[1, 0], R[0, 2] + R[2, 0]])
    elif i == 2:
        q = np.array([R[0, 2] - R[2, 0], R[0, 1] + R[1, 0],
                      candidates[2], R[1, 2] + R[2, 1]])
    else:
        q = np.array([R[1, 0] - R[0, 1], R[0, 2] + R[2, 0],
                      R[1, 2] + R[2, 1], candidates[3]])
    return (q / np.linalg.norm(q)).astype(np.float32)


def load_vio_world_transforms(path, frames):
    """Load camera-to-world transforms for the requested video frames."""
    with np.load(path) as trajectory:
        required = {"frame_idx", "pose_wxyz_xyz"}
        missing_keys = required - set(trajectory.files)
        if missing_keys:
            raise ValueError(
                f"{path} is missing trajectory arrays: {sorted(missing_keys)}")
        frame_idx = np.asarray(trajectory["frame_idx"], np.int64)
        poses = np.asarray(trajectory["pose_wxyz_xyz"], np.float64)

    if poses.shape != (len(frame_idx), 7):
        raise ValueError(
            f"{path}: pose_wxyz_xyz must have shape ({len(frame_idx)}, 7), "
            f"got {poses.shape}")
    if len(np.unique(frame_idx)) != len(frame_idx):
        raise ValueError(f"{path}: frame_idx contains duplicates")
    if not np.all(np.isfinite(poses)):
        raise ValueError(f"{path}: trajectory poses contain non-finite values")

    lookup = {int(frame): i for i, frame in enumerate(frame_idx)}
    missing_frames = [int(frame) for frame in frames if int(frame) not in lookup]
    if missing_frames:
        preview = ", ".join(map(str, missing_frames[:10]))
        suffix = "..." if len(missing_frames) > 10 else ""
        raise ValueError(
            f"{path}: no VIO pose for {len(missing_frames)} hand frames "
            f"({preview}{suffix})")

    selected = poses[[lookup[int(frame)] for frame in frames]]
    quat = selected[:, :4]
    quat_norm = np.linalg.norm(quat, axis=1)
    if np.any(quat_norm < 1e-8):
        raise ValueError(f"{path}: trajectory contains a zero quaternion")
    quat = quat / quat_norm[:, None]
    R_wl = np.stack([quat_to_matrix(q) for q in quat])
    R_cw = R_wl.transpose(0, 2, 1)
    t_cw = -np.einsum("nij,nj->ni", R_cw, selected[:, 4:])
    return R_cw.astype(np.float32), t_cw.astype(np.float32)


def load_vio_frame_times(path, frames):
    """Load observed interpolation times for hand frames and nominal cadence."""
    with np.load(path) as trajectory:
        frame_idx = np.asarray(trajectory["frame_idx"], np.int64)
        if "interpolation_frame_time_s" in trajectory:
            frame_time_s = np.asarray(
                trajectory["interpolation_frame_time_s"], np.float64)
        else:
            # Legacy trajectories used frame index as a uniform time axis.
            frame_time_s = frame_idx.astype(np.float64)
    if frame_time_s.shape != frame_idx.shape:
        raise ValueError(f"{path}: interpolation frame times have wrong shape")
    delta = np.diff(frame_time_s)
    if np.any(delta <= 0):
        raise ValueError(f"{path}: interpolation frame times are not increasing")
    lookup = {int(frame): i for i, frame in enumerate(frame_idx)}
    missing = [int(frame) for frame in frames if int(frame) not in lookup]
    if missing:
        raise ValueError(
            f"{path}: no interpolation time for {len(missing)} hand frames")
    selected = frame_time_s[[lookup[int(frame)] for frame in frames]]
    return selected, float(np.median(delta)), dict(zip(frame_idx, frame_time_s))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jsonl", default=None,
                    help="pinhole hands.jsonl (default: <calib-dir>/derived/hands.jsonl)")
    ap.add_argument("--mano", default="/tmp/mano_jax.npz")
    ap.add_argument("--out", default=None,
                    help="output (default: <calib-dir>/derived/hands3d_<hand>.jsonl)")
    ap.add_argument("--calib-dir", default="../../long-test1")
    ap.add_argument("--trajectory", required=True,
                    help="VIO trajectory NPZ; pose_wxyz_xyz must be full-rate "
                         "world-to-left-camera poses")
    ap.add_argument("--left-serial", default="046060323008")
    ap.add_argument("--right-serial", default="046060323001")
    ap.add_argument("--hand", choices=["left", "right"], default="right",
                    help="which hand track (is_right) to optimize")
    # acceleration (Laplacian) smoother weights, on consecutive-in-time triples.
    ap.add_argument("--w-accel-pose", type=float, default=2.0,
                    help="accel penalty on INTERNAL finger joints (1..15); strong "
                         "— articulation changes slowly")
    ap.add_argument("--w-accel-pose-global", type=float, default=0.5,
                    help="accel penalty on GLOBAL wrist rotation (joint 0); weaker "
                         "— the whole hand can rotate quickly")
    ap.add_argument("--w-accel-trans", type=float, default=2.0,
                    help="accel penalty on root translation")
    ap.add_argument("--w-shape", type=float, default=0.1,
                    help="weight pulling internal finger pose toward the two-eye "
                         "geodesic-mean articulation estimate")
    # Per-keypoint reprojection group weights (residual-space; cost scales as w^2).
    ap.add_argument("--w-wrist", type=float, default=1.0)
    ap.add_argument("--w-mcp", type=float, default=1.0)
    ap.add_argument("--w-pip", type=float, default=1.0)
    ap.add_argument("--w-tip", type=float, default=1.0)
    ap.add_argument("--huber-px", type=float, default=5.0,
                    help="Huber reprojection knee in pixels")
    ap.add_argument("--phase1-iters", type=int, default=30,
                    help="LM iterations for each independent per-frame fit")
    ap.add_argument("--phase1-batch", type=int, default=32,
                    help="equal-sized one-frame solves vmapped per GPU batch")
    ap.add_argument("--phase1-linear", default="dense_cholesky",
                    help="jaxls linear solver for each small independent fit")
    ap.add_argument("--min-depth", type=float, default=0.08,
                    help="reject phase-1 roots shallower than this (metres)")
    ap.add_argument("--max-depth", type=float, default=1.5,
                    help="reject phase-1 roots deeper than this (metres)")
    ap.add_argument("--min-joint-z", type=float, default=0.02,
                    help="reject if any joint is behind/too near either camera")
    ap.add_argument("--max-reproj-px", type=float, default=15.0,
                    help="reject phase-1 fits above this two-eye mean error")
    ap.add_argument("--max-epipolar-px", type=float, default=12.0,
                    help="reject detections above this median stereo |dy|")
    ap.add_argument("--interp-max-gap", type=int, default=5,
                    help="linearly/SO3 interpolate enclosed gaps up to N frames; "
                         "0 disables")
    ap.add_argument("--iters", type=int, default=30,
                    help="LM iterations for the filtered temporal solve")
    ap.add_argument("--frame-min", type=int, default=None)
    ap.add_argument("--frame-max", type=int, default=None)
    ap.add_argument("--linear", default="conjugate_gradient",
                    help="jaxls linear solver for the Schur-reduced system: "
                         "conjugate_gradient (default; fastest here once singular "
                         "frames are rejected) | dense_cholesky | cholmod")
    args = ap.parse_args()
    derived = os.path.join(args.calib_dir, "derived")
    if args.jsonl is None:
        args.jsonl = os.path.join(derived, "hands.jsonl")
    if args.out is None:
        args.out = os.path.join(derived, f"hands3d_{args.hand}.jsonl")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    M = MJ.load_mano(args.mano)
    data_all, t_init_all, frames_all, Rvl_all, want_right = build_data(
        args.jsonl, args.calib_dir, args.left_serial, args.right_serial,
        args.hand, args.frame_min, args.frame_max,
        args.w_wrist, args.w_mcp, args.w_pip, args.w_tip)
    R_cw_all, t_cw_all = load_vio_world_transforms(args.trajectory, frames_all)
    frame_time_all, frame_time_reference_s, frame_time_by_index = (
        load_vio_frame_times(args.trajectory, frames_all))
    data_all["R_cw"] = jnp.asarray(R_cw_all)
    data_all["t_cw"] = jnp.asarray(t_cw_all)
    data_all["frame_time_s"] = jnp.asarray(frame_time_all)
    data_all["frame_time_reference_s"] = frame_time_reference_s
    _set_accel_indices(data_all, frames_all)
    print(f"VIO world transforms: {args.trajectory}")
    out_size = data_all["out_size"]

    import time
    beta_fixed = jnp.mean(data_all["beta0"], axis=0)

    # Phase 1: truly independent one-frame problems. A vmapped solve gives every
    # frame its own LM damping state while still compiling/running efficiently.
    t = time.time()
    quat1, trans1 = solve_independent_frames(
        M, data_all, t_init_all, beta_fixed, args.huber_px, args.w_shape,
        args.phase1_iters, args.phase1_batch, args.phase1_linear)
    metrics1 = evaluate_fits(
        M, data_all, quat1, trans1, beta_fixed, Rvl_all)
    keep, rejected = quality_mask(
        metrics1, args.min_depth, args.max_depth, args.min_joint_z,
        args.max_reproj_px, args.max_epipolar_px)
    print(f"phase 1 solved {len(frames_all)} frames in {time.time()-t:.1f}s")
    print(f"phase 1 accepted {int(keep.sum())}/{len(keep)}; "
          + ", ".join(f"{k}={v}" for k, v in rejected.items()))
    if not keep.any():
        raise RuntimeError("phase-1 quality gates rejected every frame")

    frames = list(np.asarray(frames_all)[keep])
    Rvl_arr = Rvl_all[keep]
    data = subset_data(data_all, keep, frames)
    t_init = jnp.asarray(trans1[keep])
    quat_init = jnp.asarray(quat1[keep])
    n = len(frames)

    # Phase 2: only accepted observations enter the coupled min-acceleration
    # solve. Rejected/missing samples cannot contribute a reprojection factor or
    # poison the trust region.
    fids = jnp.arange(n)
    var_list = [PoseVar(fids), TransVar(fids)]
    init_vars = [PoseVar(fids).with_value(quat_init),
                 TransVar(fids).with_value(t_init)]
    init = jaxls.VarValues.make(init_vars)

    costs = make_costs(
        M, data, beta_fixed, args.huber_px, args.w_shape,
        args.w_accel_pose, args.w_accel_pose_global, args.w_accel_trans,
        temporal=True)
    prob = jaxls.LeastSquaresProblem(costs, var_list).analyze()
    # report what Schur elimination chose (analyze() logs it too). With temporal
    # smoothing on this is "no elimination"; printed to make it explicit.
    elim = prob._elimination
    if elim is None:
        print(f"SCHUR: no elimination (full {prob._tangent_dim}-dim banded "
              f"system; expected with the accel smoother on)")
    else:
        print(f"SCHUR: reduced {elim.reduced_dim} of {prob._tangent_dim} dims")
    t = time.time()
    # LM trust region required (plain Gauss-Newton diverges to NaN here).
    sol, summary = prob.solve(
        init, trust_region=jaxls.TrustRegionConfig(),
        linear_solver=args.linear,
        termination=jaxls.TerminationConfig(max_iterations=args.iters),
        verbose=True, return_summary=True)
    # structured convergence trace (return_summary=True) instead of scraping the
    # verbose stdout: iters actually run, per-iter cost + LM lambda.
    n_it = int(summary.iterations)
    ch = np.array(summary.cost_history)[:n_it + 1]
    lh = np.array(summary.lambda_history)[:n_it + 1]
    crit = np.array(summary.termination_criteria)   # [cost, gradient, parameter]
    why = ["cost", "gradient", "parameter"]
    stopped = ", ".join(w for w, c in zip(why, crit) if c) or "max_iters"
    print(f"phase 2 solved {n} accepted frames in {time.time()-t:.1f}s")
    print(f"CONVERGENCE: {n_it} LM iters (max {args.iters}); stopped on: {stopped}")
    print(f"  cost:   {ch[0]:.3e} -> {ch[-1]:.3e}  ({ch[-1]/ch[0]*100:.1f}% of init)")
    print(f"  lambda: {lh[0]:.1e} -> {lh[-1]:.1e}  "
          f"(min {lh.min():.1e}, max {lh.max():.1e})")

    quat = np.array(sol[PoseVar])
    trans = np.array(sol[TransVar])
    beta_solved = np.array(beta_fixed)
    print(f"betas (fixed to WiLoR track mean): {beta_solved.round(2)}")

    quat0_np = np.array(quat_init)
    beta = np.broadcast_to(beta_solved, (n, 10))        # shared shape, per frame
    kpL_np = np.array(data["kpL"]); kpR_np = np.array(data["kpR"]); fpx_np = np.array(data["f_px"])
    R_lr_np = np.array(data["R_lr"]); t_lr_np = np.array(data["t_lr"])

    mirror = 1.0 if want_right else -1.0

    def fk_R_batched(quat_b, beta_b):
        """Vmapped MANO FK over all frames → (n,21,3) left-virtual joints."""
        R = jax.vmap(lambda q: jaxlie.SO3(q).as_matrix())(quat_b)   # (n,16,3,3)
        j = jax.vmap(lambda Ri, bi: MJ.mano_forward_R(M, Ri, bi))(R, beta_b)
        return j.at[:, :, 0].multiply(mirror)                       # left-hand x-mirror

    @jax.jit
    def reproj_all(quat_b, beta_b, trans_b, fpx_b, R_lr_b, t_lr_b):
        """Both-eye reprojected pixels for ALL frames in one call → (n,21,2)x2."""
        joints = fk_R_batched(quat_b, beta_b)                       # (n,21,3)
        x = joints + trans_b[:, None, :]                            # left-virtual
        c = (out_size - 1) / 2.0
        def proj(cam, f):
            return jnp.stack([f[:, None] * cam[:, :, 0] / cam[:, :, 2] + c,
                              f[:, None] * cam[:, :, 1] / cam[:, :, 2] + c], -1)
        pL = proj(x, fpx_b)                                         # left eye (I,0)
        camR = jnp.einsum("nkc,ndc->nkd", x, R_lr_b) + t_lr_b[:, None, :]
        pR = proj(camR, fpx_b)
        return pL, pR

    # --- detailed diagnostics (vectorized) -------------------------------
    # ~9000 un-jitted per-frame JAX calls -> one jitted vmap over all frames.
    pL, pR = reproj_all(jnp.asarray(quat), jnp.asarray(beta), jnp.asarray(trans),
                        jnp.asarray(fpx_np), jnp.asarray(R_lr_np),
                        jnp.asarray(t_lr_np))
    pL = np.array(pL); pR = np.array(pR)
    eL = np.linalg.norm(pL - kpL_np, axis=2)                        # (n,21)
    eR = np.linalg.norm(pR - kpR_np, axis=2)
    perkp = np.concatenate([eL.ravel(), eR.ravel()])
    fmi = np.concatenate([eL, eR], axis=1).mean(axis=1)
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
    print(f"weights: accel[pose={args.w_accel_pose} global={args.w_accel_pose_global} "
          f"trans={args.w_accel_trans}] shape={args.w_shape} "
          f"huber_px={args.huber_px}  ({len(np.asarray(data['accel_i1']))} accel "
          f"triples; observed-time velocities in VIO world; fixed beta)")
    print(f"\nREPROJ ERR (px, {len(perkp)} observations): "
          f"mean={perkp.mean():.2f}  {pct(perkp)}")
    print(f"per-frame mean error: mean={fmi.mean():.2f}  {pct(fmi)}")
    print(f"\nDEPTH (m): median={np.median(d_arr):.3f} mean={d_arr.mean():.3f} "
          f"std={d_arr.std():.3f}  range=[{d_arr.min():.3f},{d_arr.max():.3f}]")
    print(f"  depth percentiles: {pct(d_arr,(5,50,95))}")
    print(f"\nPARAM DRIFT from independent phase-1 fit:")
    print(f"  |Δpose| (rad): mean={dpose.mean():.3f} {pct(dpose)}  (15 joints+global)")
    print(f"  trans xy spread (m): x[{trans[:,0].min():.2f},{trans[:,0].max():.2f}] "
          f"y[{trans[:,1].min():.2f},{trans[:,1].max():.2f}]")
    # Temporal jitter on genuinely adjacent accepted observations.
    if n > 1:
        adjacent = np.diff(frames) == 1
        dj = np.abs(np.diff(d_arr)[adjacent]) * 1000
        if len(dj):
            print(f"\nTEMPORAL (adjacent accepted frames):")
            print(f"  depth jump (mm): median={np.median(dj):.1f} "
                  f"{pct(dj,(50,90,99))}")
    print("==========================================================\n")

    # Convert results from the per-frame LEFT-VIRTUAL frame back to the common
    # LEFT-FISHEYE (optical-axis) frame so they're comparable across frames and
    # ready for temporal/world lifting:
    #   joints_fish = Rv_l @ (joints_virtual + t)
    #   global_orient_fish = Rv_l @ R_global   (rotate the wrist rotation in too)
    # vectorized: FK + frame conversion for ALL frames at once (no per-frame
    # un-jitted JAX calls).
    joints_v_all = np.array(fk_R_batched(jnp.asarray(quat), jnp.asarray(beta)))
    joints_v_all = joints_v_all + trans[:, None, :]                 # (n,21,3) left-virtual
    j_fish_all = np.einsum("nkc,ndc->nkd", joints_v_all, Rvl_arr)   # Rv_l @ j → fisheye
    R_glob_v_all = np.array(jax.vmap(lambda q: jaxlie.SO3(q).as_matrix())(
        jnp.asarray(quat[:, 0])))                                   # (n,3,3)
    R_glob_fish_all = np.einsum("nij,njk->nik", Rvl_arr, R_glob_v_all)
    # Re-apply the geometry gates after smoothing. The temporal solve is warm-
    # started from valid fits, but this guarantees that a failed phase 2 can
    # never emit negative/non-finite depth.
    metrics2 = evaluate_fits(
        M, data, quat, trans, jnp.asarray(beta_solved), Rvl_arr)
    keep2, rejected2 = quality_mask(
        metrics2, args.min_depth, args.max_depth, args.min_joint_z,
        args.max_reproj_px, args.max_epipolar_px)
    if not keep2.all():
        print(f"phase 2 output gate accepted {int(keep2.sum())}/{len(keep2)}; "
              + ", ".join(f"{k}={v}" for k, v in rejected2.items()))

    phase1_kept = {
        key: np.asarray(value)[keep] for key, value in metrics1.items()
        if key != "root_fish"
    }
    records = []
    for i, fr in enumerate(frames):
        if not keep2[i]:
            continue
        records.append({
            "frame": int(fr), "is_right": want_right,
            "trans": root_fish[i].tolist(),
            "depth_m": float(root_fish[i][2]),
            "joints_3d_cam": j_fish_all[i].tolist(),
            "Rv_l": Rvl_arr[i].tolist(),
            "global_orient_R": R_glob_fish_all[i].tolist(),
            "quat": quat[i].tolist(),
            "trans_virtual": trans[i].tolist(),
            "interpolated": False,
            "phase1_mean_reproj_px": float(phase1_kept["mean_reproj_px"][i]),
            "phase1_p90_reproj_px": float(phase1_kept["p90_reproj_px"][i]),
            "phase1_median_epipolar_px": float(
                phase1_kept["median_epipolar_px"][i]),
            "phase1_depth_m": float(phase1_kept["depth_m"][i]),
        })

    # Fill only enclosed short gaps. Position is linear in the common fisheye
    # frame; rotations follow shortest SO(3) arcs. Internal MANO rotations are
    # interpolated independently, then FK regenerates coherent 3D joints.
    interpolated = []
    if args.interp_max_gap > 0:
        for left, right in zip(records[:-1], records[1:]):
            gap = right["frame"] - left["frame"] - 1
            if gap < 1 or gap > args.interp_max_gap:
                continue
            root0, root1 = np.array(left["trans"]), np.array(right["trans"])
            q0, q1 = np.array(left["quat"]), np.array(right["quat"])
            q_rvl0 = matrix_to_quat(left["Rv_l"])
            q_rvl1 = matrix_to_quat(right["Rv_l"])
            q_glob0 = matrix_to_quat(left["global_orient_R"])
            q_glob1 = matrix_to_quat(right["global_orient_R"])
            time0 = frame_time_by_index[left["frame"]]
            time1 = frame_time_by_index[right["frame"]]
            for offset in range(1, gap + 1):
                frame = left["frame"] + offset
                alpha = (
                    (frame_time_by_index[frame] - time0) / (time1 - time0))
                root = (1.0 - alpha) * root0 + alpha * root1
                Rvl_i = quat_to_matrix(quat_slerp(q_rvl0, q_rvl1, alpha))
                Rglob_i = quat_to_matrix(quat_slerp(q_glob0, q_glob1, alpha))
                qi = np.stack([
                    quat_slerp(q0[j], q1[j], alpha) for j in range(16)
                ])
                qi[0] = matrix_to_quat(Rvl_i.T @ Rglob_i)
                trans_i = Rvl_i.T @ root
                interpolated.append({
                    "frame": frame,
                    "is_right": want_right,
                    "trans": root.tolist(),
                    "depth_m": float(root[2]),
                    "Rv_l": Rvl_i.tolist(),
                    "global_orient_R": Rglob_i.tolist(),
                    "quat": qi.tolist(),
                    "trans_virtual": trans_i.tolist(),
                    "interpolated": True,
                    "source_frames": [left["frame"], right["frame"]],
                })

    if interpolated:
        qi = jnp.asarray(np.array([r["quat"] for r in interpolated]))
        ti = np.array([r["trans_virtual"] for r in interpolated])
        Ri = np.array([r["Rv_l"] for r in interpolated])
        bi = jnp.broadcast_to(jnp.asarray(beta_solved), (len(interpolated), 10))
        ji = np.asarray(fk_R_batched(qi, bi)) + ti[:, None, :]
        ji = np.einsum("nkc,ndc->nkd", ji, Ri)
        for record, joints in zip(interpolated, ji):
            record["joints_3d_cam"] = joints.tolist()
        records.extend(interpolated)
        records.sort(key=lambda record: record["frame"])
    print(f"output: {len(records) - len(interpolated)} measured + "
          f"{len(interpolated)} interpolated frames")

    with open(args.out, "w") as f:
        # Keep beta_opt for compatibility with visualize_data.py and older files.
        f.write(json.dumps({
            "meta": True, "is_right": want_right, "mirror": mirror,
            "beta_opt": beta_solved.tolist(),
            "phase1_input_frames": len(frames_all),
            "phase1_accepted_frames": int(keep.sum()),
            "phase2_accepted_frames": int(keep2.sum()),
            "interpolated_frames": len(interpolated),
            "trajectory": os.path.abspath(args.trajectory),
            "temporal_time_source": "trajectory_interpolation_frame_time_s",
            "frame_time_reference_s": frame_time_reference_s,
            "quality_thresholds": {
                "min_depth_m": args.min_depth,
                "max_depth_m": args.max_depth,
                "min_joint_z_m": args.min_joint_z,
                "max_mean_reproj_px": args.max_reproj_px,
                "max_median_epipolar_px": args.max_epipolar_px,
            },
        }) + "\n")
        for record in records:
            f.write(json.dumps(record) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
