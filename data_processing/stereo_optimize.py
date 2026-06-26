"""Stereo MANO bundle adjustment over a hand track, with jaxls.

Keypoints-only (no images, no ViT) — so a dense, temporally-coupled solve over
the whole video is cheap (seconds on CPU). For each frame we have WiLoR's
per-eye 2D keypoints in two *rectified* pinhole crops (left/right) that share a
virtual orientation with x along the stereo baseline, so the right camera is the
left translated by ``-baseline`` along x. We optimize, per frame:

    pose       (48,) axis-angle  global_orient(3) + hand_pose(45)
    betas      (10,) shape
    t          (3,)  MANO-root translation in the rectified LEFT-crop frame

against four factor types:
  - left reprojection:  project MANO joints (at t) into the left pinhole
  - right reprojection: project (joints at t, shifted -baseline x) into right
  - prior:              pose/betas pulled toward WiLoR's per-frame estimate
  - temporal:           smoothness on (pose, t) between consecutive frames

This resolves WiLoR's monocular depth ambiguity (the right-eye term pins down t)
and MANO's anatomical model regularizes per-keypoint disparity noise.

Run:  python stereo_optimize.py --jsonl out/pinhole_stereo/hands.jsonl \
          --mano /tmp/mano_jax.npz --out out/pinhole_stereo/hands3d.jsonl
"""
import argparse
import json

import jax
import jax.numpy as jnp
import numpy as np

import jaxls
import mano_jax as MJ


# ---- variables -------------------------------------------------------------
class PoseVar(jaxls.Var[jax.Array], default_factory=lambda: jnp.zeros(48)):
    """global_orient(3) + hand_pose(45) axis-angle, in rectified-left frame."""


class BetaVar(jaxls.Var[jax.Array], default_factory=lambda: jnp.zeros(10)):
    """MANO shape."""


class TransVar(jaxls.Var[jax.Array], default_factory=lambda: jnp.array([0., 0., 0.5])):
    """MANO-root translation in the rectified-left-crop camera frame (metres)."""


def project(joints_cam, f_px, out_size):
    """Pinhole project (N,3) camera-frame points → (N,2) crop pixels."""
    c = (out_size - 1) / 2.0
    x = f_px * joints_cam[:, 0] / joints_cam[:, 2] + c
    y = f_px * joints_cam[:, 1] / joints_cam[:, 2] + c
    return jnp.stack([x, y], axis=-1)


def make_costs(M, data, w_prior_pose, w_prior_beta, w_temporal, huber_px):
    """Build batched jaxls costs from the per-frame stacked arrays in ``data``."""
    n = data["pose0"].shape[0]
    fids = jnp.arange(n)

    def huber_w(res):
        a = jnp.abs(res) + 1e-8
        return jax.lax.stop_gradient(jnp.where(a > huber_px, huber_px / a, 1.0))

    @jaxls.Cost.factory
    def reproj(vals, pose_v, beta_v, t_v, obs2d, valid, f_px, baseline_x, conf):
        joints = MJ.mano_forward(M, vals[pose_v][:3], vals[pose_v][3:], vals[beta_v])
        cam = joints + vals[t_v][None, :] - jnp.array([baseline_x, 0.0, 0.0])[None, :]
        proj = project(cam, f_px, data["out_size"])
        res = (proj - obs2d) * valid[:, None] * conf[:, None]
        return (res * jnp.sqrt(huber_w(res))).ravel()

    @jaxls.Cost.factory
    def prior_pose(vals, pose_v, pose0):
        return w_prior_pose * (vals[pose_v] - pose0)

    @jaxls.Cost.factory
    def prior_beta(vals, beta_v, beta0):
        return w_prior_beta * (vals[beta_v] - beta0)

    @jaxls.Cost.factory
    def temporal(vals, a, b, w):
        return w * (vals[a] - vals[b])

    costs = []
    # left eye: baseline_x = 0 (left camera is the reference frame)
    costs.append(reproj(PoseVar(fids), BetaVar(fids), TransVar(fids),
                        data["kpL"], data["validL"], data["f_px"],
                        jnp.zeros(n), data["confL"]))
    # right eye: shift by +baseline along x (point moves to right-cam frame)
    costs.append(reproj(PoseVar(fids), BetaVar(fids), TransVar(fids),
                        data["kpR"], data["validR"], data["f_px"],
                        data["baseline"], data["confR"]))
    costs.append(prior_pose(PoseVar(fids), data["pose0"]))
    costs.append(prior_beta(BetaVar(fids), data["beta0"]))
    # temporal on consecutive frames of the SAME track
    a, b = jnp.arange(n - 1), jnp.arange(1, n)
    costs.append(temporal(PoseVar(a), PoseVar(b), w_temporal))
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
    ap.add_argument("--w-prior-pose", type=float, default=2.0)
    ap.add_argument("--w-prior-beta", type=float, default=5.0)
    ap.add_argument("--w-temporal", type=float, default=10.0)
    ap.add_argument("--huber-px", type=float, default=10.0)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--frame-min", type=int, default=None)
    ap.add_argument("--frame-max", type=int, default=None)
    ap.add_argument("--linear", default="conjugate_gradient",
                    help="jaxls linear solver: conjugate_gradient | dense_cholesky | cholmod")
    args = ap.parse_args()

    M = MJ.load_mano(args.mano)
    want_right = 1 if args.hand == "right" else 0

    # Gather the single-best hand of the requested handedness per frame.
    rows = [json.loads(l) for l in open(args.jsonl)]
    frames, pose0, beta0, kpL, kpR, fpx, confL, confR = ([] for _ in range(8))
    for d in rows:
        if args.frame_min is not None and d["frame"] < args.frame_min:
            continue
        if args.frame_max is not None and d["frame"] > args.frame_max:
            continue
        cand = [h for h in d["hands"] if h["is_right"] == want_right]
        if not cand:
            continue
        h = max(cand, key=lambda x: (x["bbox"][2] - x["bbox"][0]))  # largest bbox
        frames.append(d["frame"])
        pose0.append(np.array(h["global_orient"] + h["hand_pose"], np.float32))
        beta0.append(np.array(h["betas"], np.float32))
        kpL.append(np.array(h["kp_left"], np.float32))
        kpR.append(np.array(h["kp_right"], np.float32))
        fpx.append(h["f_px"])
        confL.append(np.ones(21, np.float32))   # TODO: real per-kp confidence
        confR.append(np.ones(21, np.float32))
    n = len(frames)
    assert n > 0, "no frames with a detection in range"
    baseline = json.loads(open(args.jsonl).readline())["baseline"]
    out_size = next((h["out_size"] for d in rows for h in d["hands"]), 256)
    print(f"hand={args.hand}: {n} frames with a detection")

    data = {
        "pose0": jnp.asarray(np.stack(pose0)),
        "beta0": jnp.asarray(np.stack(beta0)),
        "kpL": jnp.asarray(np.stack(kpL)),
        "kpR": jnp.asarray(np.stack(kpR)),
        "validL": jnp.ones((n, 21)),
        "validR": jnp.ones((n, 21)),
        "confL": jnp.asarray(np.stack(confL)),
        "confR": jnp.asarray(np.stack(confR)),
        "f_px": jnp.asarray(np.array(fpx, np.float32)),
        "baseline": jnp.full(n, baseline, jnp.float32),
        "out_size": out_size,
    }

    costs, fids = make_costs(M, data, args.w_prior_pose, args.w_prior_beta,
                             args.w_temporal, args.huber_px)

    # init: WiLoR pose/betas; translation from per-frame median triangulation
    disp = (data["kpL"][:, :, 0] - data["kpR"][:, :, 0])
    disp = jnp.clip(disp, 1.0, None)
    z0 = jnp.median(data["f_px"][:, None] * baseline / disp, axis=1)  # (n,)
    # back-project the crop centre at z0 → x,y (centre ray is ~optical axis)
    t_init = jnp.stack([jnp.zeros(n), jnp.zeros(n), z0], axis=1)

    init = jaxls.VarValues.make([
        PoseVar(fids).with_value(data["pose0"]),
        BetaVar(fids).with_value(data["beta0"]),
        TransVar(fids).with_value(t_init),
    ])
    problem = jaxls.LeastSquaresProblem(
        costs, [PoseVar(fids), BetaVar(fids), TransVar(fids)]).analyze()
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

    pose = np.array(sol[PoseVar]); beta = np.array(sol[BetaVar]); trans = np.array(sol[TransVar])

    # before/after reprojection error against WiLoR's 2D, in BOTH eyes
    def reproj_px(pose_i, beta_i, t_i, fpx_i, bx):
        j = np.array(MJ.mano_forward(M, jnp.asarray(pose_i[:3]),
                                     jnp.asarray(pose_i[3:]), jnp.asarray(beta_i)))
        cam = j + t_i[None] - np.array([bx, 0, 0])[None]
        c = (out_size - 1) / 2.0
        return np.stack([fpx_i * cam[:, 0] / cam[:, 2] + c,
                         fpx_i * cam[:, 1] / cam[:, 2] + c], 1)
    errL = errR = 0.0
    kpL_np = np.array(data["kpL"]); kpR_np = np.array(data["kpR"]); fpx_np = np.array(fpx)
    for i in range(n):
        pL = reproj_px(pose[i], beta[i], trans[i], fpx_np[i], 0.0)
        pR = reproj_px(pose[i], beta[i], trans[i], fpx_np[i], baseline)
        errL += np.linalg.norm(pL - kpL_np[i], axis=1).mean()
        errR += np.linalg.norm(pR - kpR_np[i], axis=1).mean()
    print(f"AFTER reproj err (true f_px): L={errL/n:.2f}px  R={errR/n:.2f}px")

    with open(args.out, "w") as f:
        for i, fr in enumerate(frames):
            joints = np.array(MJ.mano_forward(M, jnp.asarray(pose[i][:3]),
                                              jnp.asarray(pose[i][3:]),
                                              jnp.asarray(beta[i])))
            j_world = joints + trans[i][None]
            f.write(json.dumps({
                "frame": int(fr), "is_right": want_right,
                "trans": trans[i].tolist(), "depth_m": float(trans[i][2]),
                "joints_3d_cam": j_world.tolist(),  # rectified-left-crop frame
            }) + "\n")
    d_arr = trans[:, 2]
    print(f"wrote {args.out}; depth median={np.median(d_arr):.3f}m "
          f"10-90%=[{np.percentile(d_arr,10):.3f},{np.percentile(d_arr,90):.3f}]m")


if __name__ == "__main__":
    main()
