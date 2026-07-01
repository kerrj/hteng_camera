"""Single-frame stereo MANO fit — visualize reprojection before/after.

Renders the actual left+right rectified pinhole crops for one hand, overlays:
  - BEFORE: WiLoR's per-eye 2D keypoints (what each monocular crop predicted)
  - AFTER:  the optimized single 3D MANO hand reprojected into BOTH eyes
A consistent AFTER fit in both views = the stereo optimization worked.
"""
import argparse
import json

import cv2
import jax
import jax.numpy as jnp
import numpy as np
import torch

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import jaxls
import mano_jax as MJ
import fisheye_pinhole as FP
import wilor_hands_batched as W

_BONES = W._BONES


class PoseVar(jaxls.Var[jax.Array], default_factory=lambda: jnp.zeros(48)):
    pass


class BetaVar(jaxls.Var[jax.Array], default_factory=lambda: jnp.zeros(10)):
    pass


class TransVar(jaxls.Var[jax.Array], default_factory=lambda: jnp.array([0., 0., 0.5])):
    pass


def project(j_cam, f_px, out_size):
    c = (out_size - 1) / 2.0
    return jnp.stack([f_px * j_cam[:, 0] / j_cam[:, 2] + c,
                      f_px * j_cam[:, 1] / j_cam[:, 2] + c], axis=-1)


def draw(img, kp, color, thick=1):
    for a, b in _BONES:
        cv2.line(img, tuple(kp[a].astype(int)), tuple(kp[b].astype(int)), color, thick)
    for x, y in kp:
        cv2.circle(img, (int(x), int(y)), 3, color, -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="../../long-test1/left_stereo_8bit.mp4")
    ap.add_argument("--calib-dir", default="../../long-test1")
    ap.add_argument("--mano", default="/tmp/mano_jax.npz")
    ap.add_argument("--frame", type=int, default=6750)
    ap.add_argument("--hand", choices=["left", "right"], default="right")
    ap.add_argument("--out", default="/tmp/oneframe.jpg")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--w-prior-pose", type=float, default=2.0)
    ap.add_argument("--w-prior-beta", type=float, default=5.0)
    ap.add_argument("--out-size", type=int, default=256)
    args = ap.parse_args()
    dev = torch.device("cuda")
    want_right = 1 if args.hand == "right" else 0
    OUT = args.out_size

    M = MJ.load_mano(args.mano)

    # --- calib ---
    ls, rs = "046060323008", "046060323001"
    cl = json.load(open(f"{args.calib_dir}/calib_{ls}.json"))["intrinsics"]
    cr = json.load(open(f"{args.calib_dir}/calib_{rs}.json"))["intrinsics"]
    st = json.load(open(f"{args.calib_dir}/stereo_{ls}_{rs}.json"))
    t = lambda x: torch.tensor(x, device=dev, dtype=torch.float32)
    Kl, Dl, Kr, Dr = t(cl["K"]), t(cl["dist"]), t(cr["K"]), t(cr["dist"])
    Rs, ts = t(st["R"]), t(st["t"])
    baseline = float(np.linalg.norm(st["t"]))
    b_hat = -Rs.T @ ts
    b_hat = b_hat / torch.linalg.norm(b_hat)

    # --- detect + render this frame's crops ---
    pipe, _, _ = W.build_pipeline(False)
    from torchcodec.decoders import VideoDecoder
    dec = VideoDecoder(args.video, device="cuda")
    half = dec.metadata.width // 2
    data = dec.get_frames_in_range(start=args.frame, stop=args.frame + 1).data
    fl = data[0, :, :, :half].float()
    fr = data[0, :, :, half:].float()
    boxes = W.detect_gpu(pipe.hand_detector, data[:, :, :, :half].contiguous(), 0.3)[0]
    cand = [b for b in boxes if int(round(float(b[5]))) == want_right]
    assert cand, f"no {args.hand} hand at frame {args.frame}"
    b = max(cand, key=lambda x: x[2] - x[0])
    bbox = torch.tensor(b[:4], device=dev)
    cL, cR, geo = FP.render_stereo_crop(fl, fr, bbox, Kl, Dl, Kr, Dr, Rs, b_hat, out_size=OUT)
    f_px = float(geo["f_px"])

    # --- WiLoR per-eye keypoints (BEFORE) ---
    def wilor_kp(crop):
        c = torch.flip(crop, dims=[2]) if want_right == 0 else crop
        inp = c.permute(1, 2, 0)[None].to(dev, torch.float16)
        with torch.no_grad():
            o = pipe.wilor_model(inp)
        o = {k: v.detach().cpu().float().numpy() for k, v in o.items()}
        isz = np.array([OUT, OUT])
        oi = W.postprocess({k: v[[0]] for k, v in o.items()}, isz / 2., OUT, isz, want_right, pipe)
        return oi, oi["pred_keypoints_2d"][0]
    oL, kpL = wilor_kp(cL)
    oR, kpR = wilor_kp(cR)
    pose0 = np.concatenate([np.asarray(oL["global_orient"]).ravel(),
                            np.asarray(oL["hand_pose"]).ravel()]).astype(np.float32)
    beta0 = np.asarray(oL["betas"]).ravel().astype(np.float32)

    # --- stereo optimize (single frame) ---
    kpL_j = jnp.asarray(kpL); kpR_j = jnp.asarray(kpR)

    @jaxls.Cost.factory
    def reproj(vals, pv, bv, tv, obs, bx):
        j = MJ.mano_forward(M, vals[pv][:3], vals[pv][3:], vals[bv])
        cam = j + vals[tv][None] - jnp.array([bx, 0., 0.])[None]
        return (project(cam, f_px, OUT) - obs).ravel()

    @jaxls.Cost.factory
    def pr_pose(vals, pv, p0): return args.w_prior_pose * (vals[pv] - p0)

    @jaxls.Cost.factory
    def pr_beta(vals, bv, b0): return args.w_prior_beta * (vals[bv] - b0)

    disp = float(np.clip(np.median(kpL[:, 0] - kpR[:, 0]), 1.0, None))
    z0 = f_px * baseline / disp
    costs = [reproj(PoseVar(0), BetaVar(0), TransVar(0), kpL_j, 0.0),
             reproj(PoseVar(0), BetaVar(0), TransVar(0), kpR_j, baseline),
             pr_pose(PoseVar(0), jnp.asarray(pose0)),
             pr_beta(BetaVar(0), jnp.asarray(beta0))]
    init = jaxls.VarValues.make([
        PoseVar(0).with_value(jnp.asarray(pose0)),
        BetaVar(0).with_value(jnp.asarray(beta0)),
        TransVar(0).with_value(jnp.array([0., 0., z0])),
    ])
    prob = jaxls.LeastSquaresProblem(costs, [PoseVar(0), BetaVar(0), TransVar(0)]).analyze()
    sol = prob.solve(init, termination=jaxls.TerminationConfig(max_iterations=args.iters),
                     verbose=True)
    pose = np.array(sol[PoseVar(0)]); beta = np.array(sol[BetaVar(0)]); trans = np.array(sol[TransVar(0)])
    print(f"init z0={z0:.3f}m  ->  optimized depth={trans[2]:.3f}m")

    # --- reproject optimized hand into both eyes (AFTER) ---
    j_opt = np.array(MJ.mano_forward(M, jnp.asarray(pose[:3]), jnp.asarray(pose[3:]), jnp.asarray(beta)))
    camL = j_opt + trans[None]
    camR = j_opt + trans[None] - np.array([baseline, 0, 0])[None]
    projL = np.array(project(jnp.asarray(camL), f_px, OUT))
    projR = np.array(project(jnp.asarray(camR), f_px, OUT))

    # --- render panel: rows = L/R, cols = BEFORE(WiLoR)/AFTER(stereo) ---
    def to_bgr(crop): return cv2.cvtColor(crop.permute(1, 2, 0).clamp(0, 255).byte().cpu().numpy(), cv2.COLOR_RGB2BGR)
    L = to_bgr(cL); R = to_bgr(cR)
    Lb, La, Rb, Ra = L.copy(), L.copy(), R.copy(), R.copy()
    draw(Lb, kpL, (0, 255, 0)); draw(Rb, kpR, (0, 255, 0))      # BEFORE: WiLoR
    draw(La, projL, (0, 128, 255)); draw(Ra, projR, (0, 128, 255))  # AFTER: stereo
    for img, lab in [(Lb, "L BEFORE (WiLoR)"), (La, "L AFTER (stereo)"),
                     (Rb, "R BEFORE (WiLoR)"), (Ra, "R AFTER (stereo)")]:
        cv2.putText(img, lab, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    panel = np.vstack([np.hstack([Lb, La]), np.hstack([Rb, Ra])])
    cv2.imwrite(args.out, panel)
    # reprojection error before/after
    print(f"reproj err L: before(self)=0  after={np.linalg.norm(projL-kpL,axis=1).mean():.2f}px")
    print(f"reproj err R: before(self)=0  after={np.linalg.norm(projR-kpR,axis=1).mean():.2f}px")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
