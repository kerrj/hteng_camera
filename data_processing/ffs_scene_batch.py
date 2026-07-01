"""Full-video wide-FOV metric depth via the FFS tangent-plane mosaic, multi-GPU.

Per frame: render the vertical fan of baseline-aligned pinhole tiles, run FFS on
all tiles in ONE batched forward, back-project on GPU, and fuse into a DENSE
left-fisheye RANGE map (metres, float16). That range map is a complete compact
representation: unproject each fisheye pixel (fisheye_unproject) * range -> the
metric point, and colour is re-sampled from the video frame on demand. No PLYs.

Parallelism: a launcher splits the frame range into contiguous slices, one per
GPU, and spawns itself as a --worker subprocess per slice (CUDA_VISIBLE_DEVICES
pinned). Each worker streams its slice sequentially (fast decode) and writes a
preallocated float16 memmap `range_<start>_<end>.npy` + a shared meta.json.

  # all 8 GPUs, whole video:
  python data_processing/ffs_scene_batch.py --gpus 0 1 2 3 4 5 6 7 \
      --start 0 --end 7006 --out-dir data_processing/out/video

  # single slice (what the launcher runs):
  CUDA_VISIBLE_DEVICES=3 python data_processing/ffs_scene_batch.py --worker \
      --start 3000 --end 3040 --out-dir ...
"""
import argparse
import json
import os
import subprocess
import sys
import time

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import fisheye_pinhole as FP
import ffs_scene_depth as M

PITCHES = [-60, -30, 0, 30, 60]


def build_geom(b_hat, Rs, W, hfov, vfov, dev):
    fx = (W / 2) / np.tan(np.radians(hfov) / 2)
    H = int(round(2 * fx * np.tan(np.radians(vfov) / 2) / 32) * 32)
    dirs = M.rect_dirs(W, H, fx, fx, dev)                    # (H*W,3), z=1
    looks = M.tile_look_dirs(b_hat, PITCHES, dev)
    Rv_ls = [FP.baseline_aligned_R(g, b_hat) for g in looks]
    Rv_rs = [Rs @ Rv for Rv in Rv_ls]
    return fx, H, dirs, Rv_ls, Rv_rs


def ffs_batch(model, tiles_l, tiles_r, dev):
    a = torch.stack([torch.nan_to_num(x) for x in tiles_l]).to(dev)
    b = torch.stack([torch.nan_to_num(x) for x in tiles_r]).to(dev)
    padder = M.InputPadder(a.shape, divis_by=32, force_square=False)
    a, b = padder.pad(a, b)
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=True, dtype=M.AMP_DTYPE):
        d = model.forward(a, b, iters=model.args.valid_iters,
                          test_mode=True, optimize_build_volume="pytorch1")
    return padder.unpad(d.float())                           # (T,1,H,W) or (T,H,W)


def fuse_range_gpu(disps, dirs, Rv_ls, fx, baseline, Kl, Dl,
                   Wf, Hf, scale, min_disp, max_depth, dev):
    """All tiles' disparities -> dense left-fisheye range map (Hf,Wf) float32 on GPU."""
    offs = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)]  # 3x3 radius splat
    rmap = torch.full((Hf * Wf,), float("inf"), device=dev)
    for disp, Rv in zip(disps, Rv_ls):
        disp = disp.reshape(-1)
        depth = fx * baseline / disp.clamp(min=1e-6)
        keep = torch.isfinite(disp) & (disp > min_disp) & (depth > 0) & (depth < max_depth)
        if not keep.any():
            continue
        P = (dirs[keep] * depth[keep, None]) @ Rv.T           # left-frame points
        rng = torch.linalg.norm(P, dim=1)
        px = FP.fisheye_project(P, Kl, Dl) * scale            # -> range-map pixels
        u0 = torch.round(px[:, 0]).long(); v0 = torch.round(px[:, 1]).long()
        for dx, dy in offs:
            u = u0 + dx; v = v0 + dy
            ok = (u >= 0) & (u < Wf) & (v >= 0) & (v < Hf) & (P[:, 2] > 0)
            idx = v[ok] * Wf + u[ok]
            rmap.scatter_reduce_(0, idx, rng[ok], reduce="amin", include_self=True)
    rmap[~torch.isfinite(rmap)] = 0.0
    return rmap.reshape(Hf, Wf)


def worker(args):
    dev = torch.device("cuda")
    Kl, Dl, Kr, Dr, Rs, ts, b_hat, baseline = M.load_calib(
        args.calib_dir, args.left_serial, args.right_serial, dev)
    model = M.load_ffs(args.weights, dev, valid_iters=args.iters)
    fx, H, dirs, Rv_ls, Rv_rs = build_geom(b_hat, Rs, args.tile_w, args.hfov, args.vfov, dev)

    cap = cv2.VideoCapture(args.video)
    fish_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) // 2
    fish_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    Wf, Hf = int(fish_w * args.scale), int(fish_h * args.scale)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)

    n = args.end - args.start
    out = f"{args.out_dir}/range_{args.start}_{args.end}.npy"
    stack = np.lib.format.open_memmap(out, mode="w+", dtype=np.float16, shape=(n, Hf, Wf))

    def split(fr):
        half = fr.shape[1] // 2
        to_t = lambda b: torch.tensor(cv2.cvtColor(b, cv2.COLOR_BGR2RGB),
                                      device=dev, dtype=torch.float32).permute(2, 0, 1)
        return to_t(fr[:, :half]), to_t(fr[:, half:])

    t0 = time.time()
    for i in range(n):
        ok, fr = cap.read()
        if not ok:
            print(f"[gpu slice {args.start}-{args.end}] short read at {args.start+i}", flush=True)
            break
        fL, fR = split(fr)
        tl = [M.render_tile(fL, Rv, dirs, Kl, Dl, args.tile_w, H) for Rv in Rv_ls]
        tr = [M.render_tile(fR, Rv, dirs, Kr, Dr, args.tile_w, H) for Rv in Rv_rs]
        disps = ffs_batch(model, tl, tr, dev).reshape(len(tl), H, args.tile_w)
        rmap = fuse_range_gpu(disps, dirs, Rv_ls, fx, baseline, Kl, Dl,
                              Wf, Hf, args.scale, args.min_disp, args.max_depth, dev)
        stack[i] = rmap.cpu().numpy().astype(np.float16)
        if i and i % 50 == 0:
            fps = (i + 1) / (time.time() - t0)
            print(f"[{args.start}-{args.end}] {i+1}/{n}  {fps:.1f} fps", flush=True)
    stack.flush()
    cap.release()
    print(f"[{args.start}-{args.end}] DONE {n} frames in {(time.time()-t0)/60:.1f} min -> {out}", flush=True)


def launcher(args):
    os.makedirs(args.out_dir, exist_ok=True)
    gpus = args.gpus
    bounds = np.linspace(args.start, args.end, len(gpus) + 1).astype(int)
    slices = list(zip(bounds[:-1], bounds[1:]))
    # shared meta for reconstruction
    with open(f"{args.out_dir}/meta.json", "w") as f:
        json.dump({"video": os.path.abspath(args.video), "calib_dir": os.path.abspath(args.calib_dir),
                   "left_serial": args.left_serial, "right_serial": args.right_serial,
                   "scale": args.scale, "tile_w": args.tile_w, "hfov": args.hfov,
                   "vfov": args.vfov, "pitches": PITCHES, "iters": args.iters,
                   "min_disp": args.min_disp, "max_depth": args.max_depth,
                   "start": args.start, "end": args.end,
                   "slices": [list(map(int, s)) for s in slices],
                   "weights": os.path.abspath(args.weights)}, f, indent=2)
    procs = []
    for gid, (s, e) in zip(gpus, slices):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gid))
        cmd = [sys.executable, os.path.realpath(__file__), "--worker",
               "--start", str(s), "--end", str(e), "--video", args.video,
               "--calib-dir", args.calib_dir, "--left-serial", args.left_serial,
               "--right-serial", args.right_serial, "--weights", args.weights,
               "--out-dir", args.out_dir, "--tile-w", str(args.tile_w),
               "--hfov", str(args.hfov), "--vfov", str(args.vfov),
               "--scale", str(args.scale), "--iters", str(args.iters),
               "--min-disp", str(args.min_disp), "--max-depth", str(args.max_depth)]
        print(f"launch gpu {gid}: frames {s}-{e}", flush=True)
        procs.append(subprocess.Popen(cmd, env=env))
    rc = [p.wait() for p in procs]
    print(f"all workers exited rc={rc}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--gpus", type=int, nargs="+", default=None)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=7006)
    ap.add_argument("--video", default="long-test1/left_stereo_8bit.mp4")
    ap.add_argument("--calib-dir", default="long-test1")
    ap.add_argument("--left-serial", default="046060323008")
    ap.add_argument("--right-serial", default="046060323001")
    ap.add_argument("--weights", default=M.DEFAULT_WEIGHTS)
    ap.add_argument("--out-dir", default="data_processing/out/video")
    ap.add_argument("--tile-w", type=int, default=960)
    ap.add_argument("--hfov", type=float, default=100.0)
    ap.add_argument("--vfov", type=float, default=52.0)
    ap.add_argument("--scale", type=float, default=0.5, help="range-map res vs full fisheye")
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--min-disp", type=float, default=1.0)
    ap.add_argument("--max-depth", type=float, default=20.0)
    args = ap.parse_args()
    if args.worker:
        worker(args)
    else:
        assert args.gpus, "give --gpus or --worker"
        launcher(args)


if __name__ == "__main__":
    main()
