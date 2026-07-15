"""Extract the DYNAMIC residual layer of a TSDF task segment.

Static/dynamic decomposition: the TSDF mesh (ffs_tsdf_segment.py) is the
time-averaged static world; the dynamic content of each frame is whatever
the measured depth says is IN FRONT of that static surface. Detection is
reconstruction-vs-observation disagreement -- no hand model, no dilation
heuristics -- so manipulated objects (e.g. a phone) are captured
rigorously alongside the hands.

Per frame: raycast the static mesh from the frame's pose along every
fisheye pixel ray -> static range t_hit; dynamic mask = measured range
in front of static by > tau (or measured close where the static mesh has
a permanent hand-occlusion hole); despeckle; save world-frame points +
RGB into one ragged npz (concatenated arrays + per-frame offsets).

Spec: docs/superpowers/specs/2026-07-13-dynamic-residual-design.md

  python data_processing/ffs_dynamic_residual.py \
      --prefix data_processing/out/lt2_seg \
      --range-dir data_processing/out/lt2_video \
      --trajectory long-test2/derived/trajectory.npz
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


def residual_mask(rng, t_hit, tau, near_orphan):
    """Dynamic = measured AND (in front of static by > tau, OR static miss
    but measured within near_orphan -- permanent hand-occlusion holes)."""
    measured = rng > 0
    static_hit = np.isfinite(t_hit)
    in_front = static_hit & (t_hit - rng > tau)
    orphan = ~static_hit & (rng < near_orphan)
    return measured & (in_front | orphan)


def despeckle(mask, min_area):
    """Morphological open + drop connected components smaller than min_area."""
    m = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN,
                         np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    out = np.zeros_like(m, bool)
    for k in range(1, n):
        if stats[k, cv2.CC_STAT_AREA] >= min_area:
            out[labels == k] = True
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prefix", default="data_processing/out/lt2_seg",
                    help="reads <prefix>_tsdf_mesh.ply + _segment.json, "
                         "writes <prefix>_dynamic.npz")
    ap.add_argument("--range-dir", default="data_processing/out/lt2_video")
    ap.add_argument("--trajectory", default="long-test2/derived/trajectory_vggt_omega_fullrun_20260714.npz")
    ap.add_argument("--tau", type=float, default=0.03,
                    help="min metres in front of static to count as dynamic")
    ap.add_argument("--near-orphan", type=float, default=1.0,
                    help="static-miss pixels closer than this are dynamic")
    ap.add_argument("--min-area", type=int, default=50,
                    help="min connected-component area (half-res px)")
    ap.add_argument("--max-range", type=float, default=2.0)
    ap.add_argument("--frame-stride", type=int, default=1)
    ap.add_argument("--hfov", type=float, default=110.0,
                    help="virtual-pinhole cone of the TSDF integration; residuals "
                         "are only evaluated inside it (outside, the static mesh "
                         "has no coverage and everything would look 'dynamic')")
    ap.add_argument("--img-w", type=int, default=800)
    args = ap.parse_args()

    seg = json.load(open(f"{args.prefix}_segment.json"))
    s0, e0 = seg["start"], seg["end"]

    import open3d as o3d
    mesh = o3d.io.read_triangle_mesh(f"{args.prefix}_tsdf_mesh.ply")
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    meta = json.load(open(f"{args.range_dir}/meta.json"))
    cl = json.load(open(f"{meta['calib_dir']}/calib_{meta['left_serial']}.json"))["intrinsics"]
    Kl = torch.tensor(cl["K"], dtype=torch.float32)
    Dl = torch.tensor(cl["dist"], dtype=torch.float32)
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

    ys, xs = np.meshgrid(np.arange(Hf), np.arange(Wf), indexing="ij")
    rays = FP.fisheye_unproject(torch.tensor((xs / scale).ravel(), dtype=torch.float32),
                                torch.tensor((ys / scale).ravel(), dtype=torch.float32),
                                Kl, Dl).numpy().reshape(Hf, Wf, 3).astype(np.float32)
    ray_norm = np.linalg.norm(rays, axis=-1)  # ||ray||: fisheye rays are unit,
    # but keep explicit so t (euclidean) and range stay comparable

    # static-coverage cone: pixels whose ray lands inside the virtual pinhole
    # image the TSDF was integrated through (same params as ffs_tsdf_segment)
    Wv = args.img_w
    fxv = (Wv / 2) / np.tan(np.radians(args.hfov) / 2)
    Hv = int(round(Wv * 3 / 4 / 2) * 2)
    z = rays[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = fxv * rays[..., 0] / z + Wv / 2
        vv = fxv * rays[..., 1] / z + Hv / 2
    in_cone = (z > 0.05) & (uv >= 0) & (uv < Wv) & (vv >= 0) & (vv < Hv)
    print(f"static-coverage cone: {100 * in_cone.mean():.1f}% of fisheye pixels",
          flush=True)

    cap = cv2.VideoCapture(meta["video"])
    cur = -1

    def left_rgb(vf):
        nonlocal cur
        if vf != cur + 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, vf)
        ok, fr = cap.read()
        cur = vf
        return cv2.cvtColor(fr, cv2.COLOR_BGR2RGB) if ok else None

    all_pts, all_cols, frames, offsets = [], [], [], [0]
    n_pts = 0
    for vf in range(s0, e0, args.frame_stride):
        i = pose_of.get(vf)
        rng = range_of(vf)
        if i is None or rng is None:
            continue
        rng = rng.copy()
        rng[rng > args.max_range] = 0.0
        R_wl = quat_to_R(poses[i, :4])
        c = -(R_wl.T @ poses[i, 4:])
        dirs_w = (rays.reshape(-1, 3) @ R_wl).astype(np.float32)  # cam->world rows
        origins = np.broadcast_to(c.astype(np.float32), dirs_w.shape)
        rc = scene.cast_rays(o3d.core.Tensor(np.concatenate(
            [origins, dirs_w], axis=1).reshape(-1, 6)))
        t_hit = rc["t_hit"].numpy().reshape(Hf, Wf) * ray_norm
        m = despeckle(residual_mask(rng, t_hit, args.tau, args.near_orphan) & in_cone,
                      args.min_area)
        if not m.any():
            frames.append(vf)
            offsets.append(n_pts)
            continue
        rgb = left_rgb(vf)
        if rgb is None:
            continue
        P_cam = rays[m] * rng[m][:, None]
        P_w = P_cam @ R_wl + c                      # rows: R_wl^T P + c
        u2 = np.clip((xs[m] / scale).astype(int), 0, rgb.shape[1] - 1)
        v2 = np.clip((ys[m] / scale).astype(int), 0, rgb.shape[0] - 1)
        all_pts.append(P_w.astype(np.float32))
        all_cols.append(rgb[v2, u2])
        n_pts += len(P_w)
        frames.append(vf)
        offsets.append(n_pts)
    cap.release()

    out = f"{args.prefix}_dynamic.npz"
    np.savez(out,
             points=np.concatenate(all_pts) if all_pts else np.zeros((0, 3), np.float32),
             colors=np.concatenate(all_cols) if all_cols else np.zeros((0, 3), np.uint8),
             frames=np.array(frames, np.int64),
             offsets=np.array(offsets, np.int64))
    per = np.diff(offsets)
    print(f"wrote {out}: {len(frames)} frames, {n_pts:,} dynamic pts "
          f"(per-frame p50 {int(np.median(per))}, max {int(per.max()) if len(per) else 0})")


if __name__ == "__main__":
    main()
