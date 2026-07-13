"""TSDF-fuse ONE short task segment into a clean static workspace mesh.

Pivot from long-trajectory point accumulation (additive noise, floaters) to
the egocentric-manipulation regime: a short, low-translation, hands-busy
segment, fused volumetrically. Per calm frame: hand-masked, range-clamped
fisheye range map -> virtual forward pinhole RGB-D -> Open3D ScalableTSDF
(every frame votes on the surface; noise cancels) -> marching-cubes mesh.
Poses come from the solved trajectory (vio_windowed_ba output); dynamic
hands are masked out via the baked MANO meshes and composited back at view
time by ffs_tsdf_viewer.py.

Spec: docs/superpowers/specs/2026-07-13-tsdf-task-segment-design.md

  python data_processing/ffs_tsdf_segment.py \
      --range-dir data_processing/out/lt2_video \
      --trajectory long-test2/derived/trajectory.npz \
      --imu long-test2/derived/imu_relative.npz \
      --hands-dir data_processing/out/lt2_hands \
      --out-prefix data_processing/out/lt2_seg
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


def pick_segment(centers, frame_idx, gyro_dps, gyro_frames, hand_frames,
                 window, stride, min_hand_frac):
    """Score sliding windows: prefer small camera-center extent + low gyro
    rate, gated on hand presence. Returns (start_frame, end_frame, report).
    Windows are in VIDEO-FRAME numbers over [frame_idx[0], frame_idx[-1]]."""
    hand_set = np.zeros(int(frame_idx[-1]) + 2, bool)
    hand_set[np.asarray(hand_frames, int)] = True
    gyro_of = np.zeros(int(frame_idx[-1]) + 2, np.float64)
    gyro_of[np.asarray(gyro_frames, int)] = gyro_dps

    cands = []
    for s in range(int(frame_idx[0]), int(frame_idx[-1]) - window + 1, stride):
        e = s + window
        m = (frame_idx >= s) & (frame_idx < e)
        if m.sum() < window * 0.9:
            continue
        c = centers[m]
        ext = float(np.linalg.norm(c.max(0) - c.min(0)))
        dps = float(gyro_of[s:e].mean())
        hf = float(hand_set[s:e].mean())
        cands.append((s, e, ext, dps, hf))
    assert cands, "no candidate windows"
    ok = [c for c in cands if c[4] >= min_hand_frac]
    fallback = not ok
    if fallback:  # nothing hand-gated: take the handiest window instead
        best_hf = max(c[4] for c in cands)
        ok = [c for c in cands if c[4] == best_hf]
    exts = np.array([c[2] for c in ok])
    dpss = np.array([c[3] for c in ok])

    def z(v):  # z-score with degenerate-spread guard
        sd = v.std()
        return (v - v.mean()) / sd if sd > 1e-9 else np.zeros_like(v)

    k = int(np.argmin(z(exts) + z(dpss)))
    s, e, ext, dps, hf = ok[k]
    return s, e, {"extent_m": ext, "mean_dps": dps, "hand_frac": hf,
                  "fallback": fallback}


def rasterize_hand_mask(verts_px, H, W, dilate_px):
    """Splat projected MANO verts -> dilated boolean mask (H,W)."""
    m = np.zeros((H, W), np.uint8)
    u = np.clip(np.round(verts_px[:, 0]).astype(int), 0, W - 1)
    v = np.clip(np.round(verts_px[:, 1]).astype(int), 0, H - 1)
    m[v, u] = 1
    k = 2 * dilate_px + 1
    m = cv2.dilate(m, np.ones((k, k), np.uint8))
    return m.astype(bool)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--range-dir", default="data_processing/out/lt2_video")
    ap.add_argument("--trajectory", default="long-test2/derived/trajectory.npz")
    ap.add_argument("--imu", default="long-test2/derived/imu_relative.npz")
    ap.add_argument("--hands-dir", default="data_processing/out/lt2_hands")
    ap.add_argument("--start", type=int, default=None, help="override auto-pick")
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--window-frames", type=int, default=900)
    ap.add_argument("--pick-stride", type=int, default=30)
    ap.add_argument("--min-hand-frac", type=float, default=0.7)
    ap.add_argument("--max-rot-dps", type=float, default=20.0,
                    help="skip motion-blurred frames above this gyro rate")
    ap.add_argument("--max-range", type=float, default=2.0)
    ap.add_argument("--voxel", type=float, default=0.005)
    ap.add_argument("--hfov", type=float, default=110.0)
    ap.add_argument("--img-w", type=int, default=800)
    ap.add_argument("--mask-dilate-px", type=int, default=12)
    ap.add_argument("--frame-stride", type=int, default=1)
    ap.add_argument("--out-prefix", default="data_processing/out/lt2_seg")
    args = ap.parse_args()

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
    frame_idx = tr["frame_idx"]
    poses = tr["pose_wxyz_xyz"]
    Rws = np.stack([quat_to_R(q) for q in poses[:, :4]])
    centers = -np.einsum("nji,nj->ni", Rws, poses[:, 4:])
    pose_of = {int(f): i for i, f in enumerate(frame_idx)}

    # true frame rate from the video (long-test1 is 30 fps, long-test2 is 40!)
    _c = cv2.VideoCapture(meta["video"])
    fps = _c.get(cv2.CAP_PROP_FPS) or 30.0
    _c.release()

    imu = np.load(args.imu)
    gyro_dps = 2 * np.degrees(np.arccos(np.clip(np.abs(imu["rel_quat"][:, 0]), -1, 1))) * fps
    gyro_frames = imu["frame_idx"][:-1]

    hands = {}
    for side in ("left", "right"):
        d = np.load(f"{args.hands_dir}/hand_mesh_{side}.npz")
        hands[side] = (dict(zip(d["frames"].tolist(), d["verts"])), d["faces"])
    hand_frames = np.unique(np.concatenate(
        [np.fromiter(hands[s][0].keys(), int) for s in hands]))

    if args.start is not None:
        s0 = args.start
        e0 = args.end if args.end is not None else args.start + args.window_frames
        rep = {"manual": True}
    else:
        s0, e0, rep = pick_segment(centers, frame_idx, gyro_dps, gyro_frames,
                                   hand_frames, args.window_frames,
                                   args.pick_stride, args.min_hand_frac)
    print(f"segment: frames {s0}..{e0}  (t={s0/fps:.1f}s..{e0/fps:.1f}s @ {fps:g} fps)  "
          f"{rep}", flush=True)

    # virtual forward pinhole
    W = args.img_w
    fx = (W / 2) / np.tan(np.radians(args.hfov) / 2)
    H = int(round(W * 3 / 4 / 2) * 2)
    cx, cy = W / 2, H / 2

    # precompute fisheye rays once (half-res grid, like ffs_fuse_world)
    ys, xs = np.meshgrid(np.arange(Hf), np.arange(Wf), indexing="ij")
    rays = FP.fisheye_unproject(torch.tensor((xs / scale).ravel(), dtype=torch.float32),
                                torch.tensor((ys / scale).ravel(), dtype=torch.float32),
                                Kl, Dl).numpy().reshape(Hf, Wf, 3)

    cap = cv2.VideoCapture(meta["video"])
    cur = -1

    def left_rgb(vf):
        nonlocal cur
        if vf != cur + 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, vf)
        ok, fr = cap.read()
        cur = vf
        return cv2.cvtColor(fr, cv2.COLOR_BGR2RGB) if ok else None

    import open3d as o3d
    vol = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=args.voxel, sdf_trunc=4 * args.voxel,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    intr = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fx, cx, cy)

    gyro_of = np.zeros(int(frame_idx[-1]) + 2, np.float64)
    gyro_of[np.asarray(gyro_frames, int)] = gyro_dps

    used = skipped_blur = missing_hand = 0
    offs = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)]
    for vf in range(s0, e0, args.frame_stride):
        if vf not in pose_of:
            continue
        if gyro_of[vf] > args.max_rot_dps:
            skipped_blur += 1
            continue
        rng = range_of(vf)
        if rng is None:
            continue
        rng = rng.copy()
        rng[rng > args.max_range] = 0.0
        # hand masking
        for side in ("left", "right"):
            verts = hands[side][0].get(vf)
            if verts is None:
                missing_hand += 1
                continue
            px = FP.fisheye_project(torch.tensor(np.asarray(verts, np.float32)),
                                    Kl, Dl).numpy() * scale
            rng[rasterize_hand_mask(px, Hf, Wf, args.mask_dilate_px)] = 0.0
        m = rng > 0
        if not m.any():
            continue
        rgb = left_rgb(vf)
        if rgb is None:
            continue
        P = rays[m] * rng[m][:, None]                       # cam-frame points
        u2 = np.clip((xs[m] / scale).astype(int), 0, rgb.shape[1] - 1)
        v2 = np.clip((ys[m] / scale).astype(int), 0, rgb.shape[0] - 1)
        C = rgb[v2, u2]
        # project into the virtual pinhole; far-to-near painter's ordering
        Z = P[:, 2]
        keep = Z > 0.05
        P, C, Z = P[keep], C[keep], Z[keep]
        u = np.round(fx * P[:, 0] / Z + cx).astype(int)
        v = np.round(fx * P[:, 1] / Z + cy).astype(int)
        order = np.argsort(-Z)
        u, v, Z, C = u[order], v[order], Z[order], C[order]
        depth = np.zeros((H, W), np.float32)
        color = np.zeros((H, W, 3), np.uint8)
        for dx, dy in offs:                                  # 3x3 splat
            uu, vv = u + dx, v + dy
            ok = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
            depth[vv[ok], uu[ok]] = Z[ok]                    # later = nearer wins
            color[vv[ok], uu[ok]] = C[ok]
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(color), o3d.geometry.Image(depth),
            depth_scale=1.0, depth_trunc=args.max_range,
            convert_rgb_to_intensity=False)
        i = pose_of[vf]
        ext = np.eye(4)
        ext[:3, :3] = Rws[i]
        ext[:3, 3] = poses[i, 4:]                            # world->cam = o3d extrinsic
        vol.integrate(rgbd, intr, ext)
        used += 1
    cap.release()
    print(f"integrated {used} frames (skipped {skipped_blur} blurred; "
          f"{missing_hand} frame-sides lacked a hand fit)", flush=True)

    mesh = vol.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    out = f"{args.out_prefix}_tsdf_mesh.ply"
    o3d.io.write_triangle_mesh(out, mesh)
    print(f"wrote {len(mesh.vertices):,} verts / {len(mesh.triangles):,} tris -> {out}")
    with open(f"{args.out_prefix}_segment.json", "w") as f:
        json.dump({"start": int(s0), "end": int(e0), "fps": float(fps), "report": rep,
                   "used": used, "skipped_blur": skipped_blur}, f, indent=2)


if __name__ == "__main__":
    main()
