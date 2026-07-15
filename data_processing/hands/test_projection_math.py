"""Verify the verged-stereo projection math used by the optimizer.

The renderer aimed BOTH virtual cameras at the triangulated point P, so P must
reproject to the CENTRE of both crops. We check that using only the quantities
the optimizer will use: Rv_l, Rv_r (per eye), f_px, Rs, ts — no disparity/
parallel-axis shortcut.

Geometry (left-fisheye = reference; X_r = Rs X_l + ts):
  pose+t live in the LEFT-VIRTUAL frame: a hand point x = joints + t.
  left  eye:  x_l = x                          (left-virtual cam IS the projector)
  right eye:  x_r = R_lr x + t_lr,  R_lr = Rv_r^T Rs Rv_l,  t_lr = Rv_r^T ts
  project:    u = f_px * X/Z + c,  c = (out_size-1)/2     (both eyes, same f_px)

For P (given in LEFT-FISHEYE frame): its left-virtual coords are x = Rv_l^T P,
which should reproject to centre in BOTH eyes.
"""
import argparse
import json
import os

import numpy as np


def project(x, f_px, out_size):
    c = (out_size - 1) / 2.0
    return np.array([f_px * x[0] / x[2] + c, f_px * x[1] / x[2] + c])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording")
    parser.add_argument("--jsonl", default=None,
                        help="default: <recording>/derived/hands.jsonl")
    parser.add_argument("--left-serial", default="046060323008")
    parser.add_argument("--right-serial", default="046060323001")
    args = parser.parse_args()
    if args.jsonl is None:
        args.jsonl = os.path.join(args.recording, "derived", "hands.jsonl")

    stereo_path = os.path.join(
        args.recording,
        f"stereo_{args.left_serial}_{args.right_serial}.json")
    with open(stereo_path) as handle:
        stereo = json.load(handle)
    Rs = np.asarray(stereo["R"])
    ts = np.asarray(stereo["t"]).reshape(3)

    errs_l, errs_r = [], []
    n = 0
    for line in open(args.jsonl):
        d = json.loads(line)
        for h in d["hands"]:
            Rv_l = np.array(h["Rv_l"]); Rv_r = np.array(h["Rv_r"])
            f_px = h["f_px"]; OUT = h["out_size"]; c = (OUT - 1) / 2.0
            P = np.array(h["P"])                      # left-fisheye frame
            x = Rv_l.T @ P                            # left-virtual coords
            # left eye
            uL = project(x, f_px, OUT)
            # right eye via R_lr, t_lr
            R_lr = Rv_r.T @ Rs @ Rv_l
            t_lr = Rv_r.T @ ts
            xr = R_lr @ x + t_lr
            uR = project(xr, f_px, OUT)
            errs_l.append(np.linalg.norm(uL - c))
            errs_r.append(np.linalg.norm(uR - c))
            n += 1
    if not n:
        raise RuntimeError(f"no hand observations in {args.jsonl}")
    errs_l = np.array(errs_l)
    errs_r = np.array(errs_r)
    print(f"{n} hands.  P should reproject to crop centre ({c:.1f},{c:.1f}).")
    print(f"  LEFT  eye centre err (px): mean={errs_l.mean():.3f} max={errs_l.max():.3f}")
    print(f"  RIGHT eye centre err (px): mean={errs_r.mean():.3f} max={errs_r.max():.3f}")
    ok = errs_l.max() < 1.0 and errs_r.max() < 1.0
    if not ok:
        raise RuntimeError("projection math is off")
    print("PASS")


if __name__ == "__main__":
    main()
