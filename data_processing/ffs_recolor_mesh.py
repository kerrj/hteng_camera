"""Re-color a TSDF segment mesh from its SHARPEST frames only.

TSDF color is a running average over every integrated frame -- auto-exposure
swings and residual motion blur wash it out to pastel. This post-pass projects
each mesh vertex into a small set of low-gyro (sharp) frames, checks visibility
against that frame's measured range map (a vertex occluded by a hand or
foreground object simply fails the range test -- no explicit hand handling
needed), and rebuilds the vertex color as a sharpness-weighted average of the
frames that actually saw it. Vertices no sharp frame saw keep their TSDF color.

  python data_processing/ffs_recolor_mesh.py \
      --mesh data_processing/out/lt2_segChi2_tsdf_mesh.ply \
      --prefix data_processing/out/lt2_segChi2 \
      --range-dir data_processing/out/lt2_video_hq \
      --trajectory data_processing/out/lt2_segChi2_trajectory_refined.npz
"""
import argparse
import glob
import json
import os
import re
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import fisheye_pinhole as FP
from ffs_tsdf_segment import quat_to_R


def pick_sharp_frames(frames, dps_of, k, buckets=20):
    """Pick ~k low-gyro frames SPREAD across the segment: the sharpest frames
    cluster in the stillest stretch, which would leave vertices seen only
    elsewhere with no votes. Round-robin the sharpest per temporal bucket."""
    frames = np.asarray(frames)
    if len(frames) <= k:
        return frames
    order_of = np.argsort(np.argsort([dps_of(f) for f in frames]))  # rank
    bounds = np.linspace(0, len(frames), buckets + 1).astype(int)
    per = max(1, int(round(k / buckets)))
    keep = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        if a == b:
            continue
        idx = np.argsort(order_of[a:b])[:per]
        keep.extend(frames[a:b][idx])
    return np.array(sorted(keep))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--prefix", required=True,
                    help="reads <prefix>_segment.json; writes "
                         "<prefix>_tsdf_mesh_recolored.ply unless --out")
    ap.add_argument("--range-dir", default="data_processing/out/lt2_video_hq")
    ap.add_argument("--trajectory", required=True)
    ap.add_argument("--imu", default="long-test2/derived/imu_relative.npz")
    ap.add_argument("--k-frames", type=int, default=90)
    ap.add_argument("--vis-tol", type=float, default=0.025,
                    help="max |measured range - vertex range| (m) to accept a "
                         "frame's color for a vertex")
    ap.add_argument("--max-dps", type=float, default=10.0,
                    help="only frames below this gyro rate are candidates")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seg = json.load(open(f"{args.prefix}_segment.json"))
    s0, e0, fps = seg["start"], seg["end"], seg.get("fps", 30.0)

    meta = json.load(open(f"{args.range_dir}/meta.json"))
    cl = json.load(open(f"{meta['calib_dir']}/calib_{meta['left_serial']}.json"))["intrinsics"]
    Kl = torch.tensor(cl["K"], dtype=torch.float32, device=dev)
    Dl = torch.tensor(cl["dist"], dtype=torch.float32, device=dev)
    scale = meta["scale"]

    stacks = []
    for f in glob.glob(f"{args.range_dir}/range_*_*.npy"):
        s, e = map(int, re.search(r"range_(\d+)_(\d+)\.npy", f).groups())
        stacks.append((s, e, np.load(f, mmap_mode="r")))
    stacks.sort(key=lambda x: x[0])
    Hf, Wf = stacks[0][2].shape[1:]

    def range_of(vf):
        for s, e, st in stacks:
            if s <= vf < e:
                return np.asarray(st[vf - s], np.float32)
        return None

    tr = np.load(args.trajectory)
    pose_of = {int(f): i for i, f in enumerate(tr["frame_idx"])}
    poses = tr["pose_wxyz_xyz"]

    imu = np.load(args.imu)
    gyro_dps = 2 * np.degrees(np.arccos(np.clip(np.abs(imu["rel_quat"][:, 0]), -1, 1))) * fps
    dps_map = dict(zip(imu["frame_idx"][:-1].tolist(), gyro_dps))
    dps_of = lambda f: dps_map.get(f, 1e9)

    cands = [f for f in range(s0, e0)
             if f in pose_of and dps_of(f) < args.max_dps and range_of(f) is not None]
    sel = pick_sharp_frames(cands, dps_of, args.k_frames)
    print(f"recolor from {len(sel)} sharp frames "
          f"(dps p50 {np.median([dps_of(f) for f in sel]):.1f}, "
          f"candidates {len(cands)})", flush=True)

    import open3d as o3d
    mesh = o3d.io.read_triangle_mesh(args.mesh)
    V = torch.tensor(np.asarray(mesh.vertices), dtype=torch.float32, device=dev)
    n = len(V)
    col_sum = torch.zeros((n, 3), dtype=torch.float32, device=dev)
    w_sum = torch.zeros(n, dtype=torch.float32, device=dev)

    cap = cv2.VideoCapture(meta["video"])
    for vf in sel:
        i = pose_of[int(vf)]
        R = torch.tensor(quat_to_R(poses[i, :4]), dtype=torch.float32, device=dev)
        t = torch.tensor(poses[i, 4:], dtype=torch.float32, device=dev)
        v_cam = V @ R.T + t                       # world -> cam (rows)
        rng_v = torch.linalg.norm(v_cam, dim=1)
        ok = v_cam[:, 2] > 0.05
        px = FP.fisheye_project(v_cam, Kl, Dl)    # full-res pixel coords
        u = torch.round(px[:, 0] * scale).long()
        v = torch.round(px[:, 1] * scale).long()
        ok &= (u >= 0) & (u < Wf) & (v >= 0) & (v < Hf)
        rmap = torch.tensor(range_of(int(vf)), device=dev)
        idx = (v.clamp(0, Hf - 1) * Wf + u.clamp(0, Wf - 1))
        r_meas = rmap.reshape(-1)[idx]
        ok &= (r_meas > 0) & ((r_meas - rng_v).abs() < args.vis_tol)
        if not ok.any():
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(vf))
        got, fr = cap.read()
        if not got:
            continue
        rgb = torch.tensor(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB), device=dev)
        uf = torch.round(px[:, 0]).long().clamp(0, rgb.shape[1] - 1)
        vfl = torch.round(px[:, 1]).long().clamp(0, rgb.shape[0] - 1)
        w = 1.0 / (0.5 + dps_of(int(vf)))
        col_sum[ok] += w * rgb[vfl[ok], uf[ok]].float()
        w_sum[ok] += w
    cap.release()

    got = (w_sum > 0)
    new_col = np.asarray(mesh.vertex_colors)
    new_col[got.cpu().numpy()] = (
        (col_sum[got] / w_sum[got, None] / 255.0).cpu().numpy())
    mesh.vertex_colors = o3d.utility.Vector3dVector(new_col)
    out = args.out or f"{args.prefix}_tsdf_mesh_recolored.ply"
    o3d.io.write_triangle_mesh(out, mesh)
    print(f"recolored {int(got.sum()):,}/{n:,} verts -> {out}")


if __name__ == "__main__":
    main()
