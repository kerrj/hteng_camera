"""Quick static visualization: overlay candidate trajectories on a reference.

Globally SE3-aligns each candidate to the reference (gauge-invariant), then
draws xy / xz projections plus per-frame position error. For eyeballing
coherence of solver experiments without spinning up viser.

Usage:
    python vio_plot_traj.py --ref traj_ref.npz --cand a.npz b.npz \
        --labels liteA windowedB --out compare.png
"""
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vio_eval_drift import load_traj, rigid_align


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--cand", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    labels = args.labels or [f"cand{i}" for i in range(len(args.cand))]

    fr, cr, _ = load_traj(args.ref)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    ax_xy, ax_xz, ax_err = axes
    ax_xy.plot(cr[:, 0], cr[:, 1], "k-", lw=2, label="reference", alpha=0.8)
    ax_xz.plot(cr[:, 0], cr[:, 2], "k-", lw=2, alpha=0.8)

    colors = plt.cm.tab10.colors
    for ci, (path, lab) in enumerate(zip(args.cand, labels)):
        fc, cc, _ = load_traj(path)
        common, ir, ic = np.intersect1d(fr, fc, return_indices=True)
        A, b = rigid_align(cc[ic], cr[ir])
        al = cc[ic] @ A.T + b
        col = colors[ci % 10]
        ax_xy.plot(al[:, 0], al[:, 1], "-", color=col, lw=1, label=lab)
        ax_xz.plot(al[:, 0], al[:, 2], "-", color=col, lw=1)
        err = np.linalg.norm(al - cr[ir], axis=1)
        ax_err.plot(common, 1e3 * err, "-", color=col, lw=1,
                     label=f"{lab} (p50 {1e3*np.median(err):.0f}mm)")

    ax_xy.set_title("top-down (x-y)"); ax_xy.set_aspect("equal"); ax_xy.legend()
    ax_xz.set_title("side (x-z)"); ax_xz.set_aspect("equal")
    ax_err.set_title("position error vs reference (global SE3 align)")
    ax_err.set_xlabel("frame"); ax_err.set_ylabel("mm"); ax_err.legend()
    fig.tight_layout()
    fig.savefig(args.out, dpi=110)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
