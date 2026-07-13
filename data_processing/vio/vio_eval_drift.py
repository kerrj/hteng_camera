"""Compare a candidate trajectory against a reference over short windows.

Metric: for each sliding window of --window-s seconds, rigidly (SE3, no scale)
align the candidate's poses to the reference's over that window, then report
the max position error inside the window and the relative-rotation error.
This measures LOCAL drift -- what the downstream pipeline actually cares
about -- and deliberately ignores global gauge differences (yaw + origin are
free gauges in the solver anyway).

Usage:
    python vio_eval_drift.py --ref trajectory_ref.npz --cand trajectory_cand.npz \
        --fps 30 --window-s 10
"""
import argparse

import numpy as np


def quat_to_R(q):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.stack([
        np.stack([1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)], -1),
        np.stack([2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)], -1),
        np.stack([2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)], -1),
    ], axis=-2)
    return R


def load_traj(path):
    d = np.load(path)
    fidx = d["frame_idx"]
    poses = d["pose_wxyz_xyz"]  # WORLD->CAMERA
    R_wl = quat_to_R(poses[:, :4])
    t_wl = poses[:, 4:]
    # camera center in world + camera rotation (cam->world)
    centers = -np.einsum("nji,nj->ni", R_wl, t_wl)
    R_cw = np.transpose(R_wl, (0, 2, 1))
    return fidx, centers, R_cw


def rigid_align(A, B):
    """SE3 (no scale) aligning A onto B (both (N,3)): returns R, t."""
    ca, cb = A.mean(0), B.mean(0)
    H = (A - ca).T @ (B - cb)
    U, _, Vt = np.linalg.svd(H)
    S = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ S @ U.T
    t = cb - R @ ca
    return R, t


def rot_angle_deg(Ra, Rb):
    """Geodesic angle between rotation matrices, degrees. (N,3,3) each."""
    Rrel = np.einsum("nij,nkj->nik", Ra, Rb)  # Ra @ Rb^T
    tr = np.clip((np.trace(Rrel, axis1=1, axis2=2) - 1) / 2, -1, 1)
    return np.degrees(np.arccos(tr))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--cand", required=True)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--window-s", type=float, default=10.0)
    ap.add_argument("--stride-s", type=float, default=2.0)
    args = ap.parse_args()

    fr, cr, Rr = load_traj(args.ref)
    fc, cc, Rc = load_traj(args.cand)

    # common frames only
    common, ir, ic = np.intersect1d(fr, fc, return_indices=True)
    cr, Rr = cr[ir], Rr[ir]
    cc, Rc = cc[ic], Rc[ic]
    print(f"common frames: {len(common)} "
          f"(ref {len(fr)}, cand {len(fc)})")

    W = int(args.window_s * args.fps)
    S = max(1, int(args.stride_s * args.fps))
    max_pos, end_pos, rot_err = [], [], []
    for s in range(0, len(common) - W, S):
        sl = slice(s, s + W)
        R, t = rigid_align(cc[sl], cr[sl])
        aligned = cc[sl] @ R.T + t
        err = np.linalg.norm(aligned - cr[sl], axis=1)
        max_pos.append(err.max())
        end_pos.append(err[-1])
        # rotation: apply alignment R to candidate cam->world rotations
        Rc_al = np.einsum("ij,njk->nik", R, Rc[sl])
        rot_err.append(np.median(rot_angle_deg(Rc_al, Rr[sl])))

    max_pos, end_pos, rot_err = map(np.array, (max_pos, end_pos, rot_err))
    ext = np.ptp(cr, axis=0)
    print(f"reference extent (m): {np.round(ext, 2)}, "
          f"{len(max_pos)} windows of {args.window_s}s (stride {args.stride_s}s)")
    print(f"win max pos err (mm): p50={1e3*np.median(max_pos):.1f} "
          f"p90={1e3*np.percentile(max_pos,90):.1f} worst={1e3*max_pos.max():.1f}")
    print(f"win end pos err (mm): p50={1e3*np.median(end_pos):.1f} "
          f"p90={1e3*np.percentile(end_pos,90):.1f} worst={1e3*end_pos.max():.1f}")
    print(f"win rot err (deg):    p50={np.median(rot_err):.2f} "
          f"p90={np.percentile(rot_err,90):.2f} worst={rot_err.max():.2f}")


if __name__ == "__main__":
    main()
