"""Fuse per-frame dense FoundationStereo depth into ONE world point cloud, using
VIO camera poses. Piece #3 of the dense world reconstruction:

    world_cloud = accumulate over t:  T_cam_world(t) @ depth_cloud_cam(t)

Each frame's dense range map (ffs_scene_batch output) is unprojected to metric
points in the LEFT-camera frame (fisheye ray * range), transformed to the world
frame by that frame's VIO pose (trajectory.npz, T_wl = world->cam, inverted),
coloured from the video, accumulated, and voxel-downsampled. Quality tracks the
pose quality — a drifty trajectory smears the cloud; a converged one sharpens it.

  python data_processing/ffs_fuse_world.py \
      --range-dir data_processing/out/lt2_video \
      --trajectory long-test2/derived/trajectory.npz \
      --out data_processing/out/lt2_world.ply --frame-stride 10
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


def quat_to_R(q):  # wxyz -> (3,3)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]], np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--range-dir", default="data_processing/out/lt2_video")
    ap.add_argument("--trajectory", required=True)
    ap.add_argument("--out", default="data_processing/out/lt2_world.ply")
    ap.add_argument("--frame-stride", type=int, default=10, help="use every Nth VIO pose")
    ap.add_argument("--start-frame", type=int, default=None,
                    help="first video frame to fuse (segment mode)")
    ap.add_argument("--end-frame", type=int, default=None,
                    help="last video frame to fuse (inclusive)")
    ap.add_argument("--max-points-per-frame", type=int, default=30_000)
    ap.add_argument("--max-range", type=float, default=6.0, help="drop points beyond this (m)")
    ap.add_argument("--voxel", type=float, default=0.01, help="final voxel downsample size (m)")
    ap.add_argument("--clean", action="store_true",
                    help="statistical + radius outlier removal (drops floaters/noise)")
    ap.add_argument("--stat-nb", type=int, default=20)
    ap.add_argument("--stat-std", type=float, default=2.0, help="lower = more aggressive")
    ap.add_argument("--radius-nb", type=int, default=8, help="min neighbors within 3*voxel")
    args = ap.parse_args()

    meta = json.load(open(f"{args.range_dir}/meta.json"))
    cl = json.load(open(f"{meta['calib_dir']}/calib_{meta['left_serial']}.json"))["intrinsics"]
    Kl = torch.tensor(cl["K"], dtype=torch.float32)
    Dl = torch.tensor(cl["dist"], dtype=torch.float32)
    scale = meta["scale"]
    per_eye = meta.get("video_right") is not None

    # range-map stacks, sorted by start frame
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

    # precompute unit rays (left-cam frame) + full-res colour coords, ONCE
    ys, xs = np.meshgrid(np.arange(Hf), np.arange(Wf), indexing="ij")
    fx_full, fy_full = xs / scale, ys / scale
    rays = FP.fisheye_unproject(torch.tensor(fx_full.ravel(), dtype=torch.float32),
                                torch.tensor(fy_full.ravel(), dtype=torch.float32),
                                Kl, Dl).numpy().reshape(Hf, Wf, 3)

    cap = cv2.VideoCapture(meta["video"])
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fish_w = vid_w if per_eye else vid_w // 2
    fish_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    u2 = np.clip(fx_full.astype(int), 0, fish_w - 1)
    v2 = np.clip(fy_full.astype(int), 0, fish_h - 1)
    cur = -1

    def left_rgb(vf):
        nonlocal cur
        if vf != cur + 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, vf)
        ok, fr = cap.read(); cur = vf
        if not ok:
            return None
        eye = fr if per_eye else fr[:, :fish_w]
        return cv2.cvtColor(eye, cv2.COLOR_BGR2RGB)

    tr = np.load(args.trajectory)
    frame_idx = tr["frame_idx"]
    poses = tr["pose_wxyz_xyz"]

    allP, allC = [], []
    used = 0
    for i in range(0, len(frame_idx), args.frame_stride):
        vf = int(frame_idx[i])
        if args.start_frame is not None and vf < args.start_frame:
            continue
        if args.end_frame is not None and vf > args.end_frame:
            continue
        rng = range_of(vf)
        if rng is None:
            continue
        m = (rng > 0) & (rng <= args.max_range)
        if not m.any():
            continue
        Pc = rays[m] * rng[m, None]                       # (N,3) left-cam metric
        rgb = left_rgb(vf)
        if rgb is None:
            continue
        cols = rgb[v2[m], u2[m]]
        if len(Pc) > args.max_points_per_frame:
            sel = np.random.choice(len(Pc), args.max_points_per_frame, replace=False)
            Pc, cols = Pc[sel], cols[sel]
        R_wl = quat_to_R(poses[i, :4]); t_wl = poses[i, 4:]
        R_lw = R_wl.T; c = -(R_lw @ t_wl)                 # cam->world
        Pw = Pc @ R_lw.T + c                              # X_world = R_lw @ X_cam + c
        allP.append(Pw.astype(np.float32)); allC.append(cols)
        used += 1
    cap.release()

    P = np.concatenate(allP); C = np.concatenate(allC)
    print(f"accumulated {len(P):,} pts from {used} frames (stride {args.frame_stride})")

    import open3d as o3d
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(P.astype(np.float64))
    pc.colors = o3d.utility.Vector3dVector(np.clip(C, 0, 255) / 255.0)
    pc = pc.voxel_down_sample(args.voxel)
    print(f"voxel-downsampled to {len(pc.points):,} pts")
    if args.clean:
        pc, _ = pc.remove_statistical_outlier(nb_neighbors=args.stat_nb, std_ratio=args.stat_std)
        print(f"  after statistical outlier removal: {len(pc.points):,} pts")
        pc, _ = pc.remove_radius_outlier(nb_points=args.radius_nb, radius=3 * args.voxel)
        print(f"  after radius outlier removal:      {len(pc.points):,} pts")
    o3d.io.write_point_cloud(args.out, pc)
    print(f"wrote {len(pc.points):,} pts -> {args.out}")


if __name__ == "__main__":
    main()
