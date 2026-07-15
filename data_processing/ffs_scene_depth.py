"""Wide-FOV metric scene depth from ONE stereo fisheye frame, via FoundationStereo.

A single FoundationStereo (FFS) run needs a *rectified pinhole* pair, which can't
cover our >150deg fisheye field in one shot (tan-blowup at the periphery). So we
mosaic: render several baseline-aligned PARALLEL pinhole stereo tiles that tile
the fisheye FOV, run FFS on each, back-project each tile's metric depth, and fuse
everything in the LEFT-camera frame.

Geometry (see data_processing/CLAUDE.md "Scene depth via Fast-FoundationStereo"):
  - Baseline b_hat is ~horizontal (+x, left->right centre, 70.8 mm). A rectified
    pair (horizontal disparity, depth = f*baseline/disp with the FULL baseline)
    is valid ONLY when the optical axis is PERPENDICULAR to the baseline.
  - Rotating the look dir *in the plane perpendicular to b_hat* keeps it valid ->
    we PITCH about the baseline axis to build a VERTICAL fan of tiles. Each tile
    is made WIDE horizontally; its off-centre columns already look in yawed
    directions (horizontal coverage is free, degrading toward the horizontal
    epipoles at +-x). We do NOT yaw the optical axis (that breaks rectification).
  - Both eyes' virtual cams share ONE world orientation: left uses Rv_l =
    baseline_aligned_R(g, b_hat); right uses Rv_r = Rs @ Rv_l. In the virtual
    frame the two centres differ by a pure +x baseline -> canonical rectified
    pair. (This is the non-verged sibling of fisheye_pinhole.render_stereo_crop.)

Unrecoverable by design: the two thin cones around +-x (horizontal image edges),
where parallax -> 0. Fill those later with temporal motion / a monocular prior.

CLI:
  python data_processing/ffs_scene_depth.py --frame 3000 \
      --video long-test1/left_stereo_8bit.mp4 --calib-dir long-test1 \
      --left-serial 046060323008 --right-serial 046060323001 --out out/scene3000
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch

import fisheye_pinhole as FP

FFS_ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                        "third_party", "Fast-FoundationStereo")
FS_ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                       "third_party", "FoundationStereo")
DEFAULT_WEIGHTS = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                               "weights", "20-26-39", "model_best_bp2_serialize.pth")

# set by load_ffs: extra forward() kwargs differ between model families
FORWARD_EXTRA = {"optimize_build_volume": "pytorch1"}


# --------------------------------------------------------------------------- #
# calibration / model loading
def load_calib(calib_dir, ls, rs, device):
    cl = json.load(open(os.path.join(calib_dir, f"calib_{ls}.json")))["intrinsics"]
    cr = json.load(open(os.path.join(calib_dir, f"calib_{rs}.json")))["intrinsics"]
    st = json.load(open(os.path.join(calib_dir, f"stereo_{ls}_{rs}.json")))
    t = lambda x: torch.tensor(x, device=device, dtype=torch.float32)
    Kl, Dl, Kr, Dr = t(cl["K"]), t(cl["dist"]), t(cr["K"]), t(cr["dist"])
    Rs, ts = t(st["R"]), t(st["t"])
    baseline = float(np.linalg.norm(st["t"]))
    b_hat = -Rs.T @ ts
    b_hat = b_hat / torch.linalg.norm(b_hat)
    return Kl, Dl, Kr, Dr, Rs, ts, b_hat, baseline


def load_ffs(weights, device, valid_iters=8, max_disp=192, family="fast"):
    """family='fast': distilled Fast-FoundationStereo (serialized module).
    family='full': original ViT-L FoundationStereo (cfg.yaml + state dict) --
    much stronger on thin/textureless/shiny surfaces; ~10x slower. The two
    repos both ship packages named `core`/`Utils`, so exactly ONE root goes
    on sys.path per process."""
    global InputPadder, AMP_DTYPE, FORWARD_EXTRA
    if family == "full":
        sys.path.insert(0, FS_ROOT)
        from omegaconf import OmegaConf
        from core.utils.utils import InputPadder
        from core.foundation_stereo import FoundationStereo
        AMP_DTYPE = torch.bfloat16
        FORWARD_EXTRA = {}
        cfg = OmegaConf.load(os.path.join(os.path.dirname(weights), "cfg.yaml"))
        if "vit_size" not in cfg:
            cfg["vit_size"] = "vitl"
        cfg.valid_iters = valid_iters
        cfg.max_disp = max_disp
        model = FoundationStereo(cfg)
        ckpt = torch.load(weights, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        model.args = cfg
        return model.to(device).eval()
    sys.path.insert(0, FFS_ROOT)
    from core.utils.utils import InputPadder
    from Utils import AMP_DTYPE
    FORWARD_EXTRA = {"optimize_build_volume": "pytorch1"}
    model = torch.load(weights, map_location="cpu", weights_only=False)
    model.args.valid_iters = valid_iters
    model.args.max_disp = max_disp
    return model.to(device).eval()


# --------------------------------------------------------------------------- #
# tile geometry
def rot_about(axis, angle):
    """Rodrigues rotation (3,3) about a unit axis by `angle` rad (torch)."""
    a = axis / torch.linalg.norm(axis)
    ax, ay, az = a
    K = torch.tensor([[0, -az, ay], [az, 0, -ax], [-ay, ax, 0]],
                     device=axis.device, dtype=axis.dtype)
    return torch.eye(3, device=axis.device, dtype=axis.dtype) \
        + torch.sin(angle) * K + (1 - torch.cos(angle)) * (K @ K)


def tile_look_dirs(b_hat, pitches_deg, device):
    """Look directions for the pitch fan: forward, pitched about the baseline axis.

    forward = +z projected into the plane perpendicular to b_hat; each tile
    rotates it about b_hat so every g stays perpendicular to the baseline.
    """
    z = torch.tensor([0.0, 0.0, 1.0], device=device)
    f0 = z - (z @ b_hat) * b_hat
    f0 = f0 / torch.linalg.norm(f0)
    dirs = []
    for p in pitches_deg:
        ang = torch.tensor(np.radians(p), device=device, dtype=torch.float32)
        dirs.append(rot_about(b_hat, ang) @ f0)
    return dirs


def rect_dirs(W, H, fx, fy, device):
    """Rectangular pinhole ray dirs (z=1, un-normalized), row-major (H*W,3).

    Centred principal point. z=1 so a point at optical-axis depth Z is Z*dir.
    """
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    ys, xs = torch.meshgrid(torch.arange(H, device=device, dtype=torch.float32),
                            torch.arange(W, device=device, dtype=torch.float32),
                            indexing="ij")
    d = torch.stack([(xs - cx) / fx, (ys - cy) / fy, torch.ones_like(xs)], -1)
    return d.reshape(-1, 3)


def render_tile(frame, Rv, dirs, K, D, W, H, theta_max=2.6):
    """Sample a rectified pinhole tile from a fisheye frame. Returns (3,H,W), NaN off-sensor."""
    rays_cam = dirs @ Rv.T                      # virtual -> that eye's fisheye frame
    col = FP.sample_fisheye(frame, rays_cam, K, D, theta_max)   # (3, H*W), NaN invalid
    return col.reshape(3, H, W)


# --------------------------------------------------------------------------- #
# FFS + back-projection
def ffs_disparity(model, tileL, tileR, device):
    """Run FFS on a (3,H,W) tile pair (0-255 floats). Returns (H,W) disparity numpy."""
    H, W = tileL.shape[1:]
    a = torch.nan_to_num(tileL).unsqueeze(0).to(device)
    b = torch.nan_to_num(tileR).unsqueeze(0).to(device)
    padder = InputPadder(a.shape, divis_by=32, force_square=False)
    a, b = padder.pad(a, b)
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=True, dtype=AMP_DTYPE):
        disp = model.forward(a, b, iters=model.args.valid_iters,
                             test_mode=True, **FORWARD_EXTRA)
    disp = padder.unpad(disp.float()).cpu().numpy().reshape(H, W)
    return disp


def backproject(disp, dirs, Rv_l, fx, baseline, tileL, W, H,
                min_disp=1.0, max_depth=20.0):
    """Tile disparity -> 3D points in the LEFT camera frame + colors + valid mask.

    depth (optical-axis Z) = fx * baseline / disp;  P_virtual = Z * dir (dir.z=1);
    P_left = Rv_l @ P_virtual.
    """
    dirs_np = dirs.cpu().numpy().reshape(H, W, 3)
    Rv = Rv_l.cpu().numpy()
    col = tileL.permute(1, 2, 0).cpu().numpy()             # (H,W,3), NaN off-sensor

    valid = np.isfinite(disp) & (disp > min_disp) & np.isfinite(col).all(-1)
    depth = np.where(disp > min_disp, fx * baseline / np.maximum(disp, 1e-6), 0.0)
    valid &= (depth > 0) & (depth < max_depth)

    P_virt = dirs_np * depth[..., None]                    # (H,W,3), z=depth
    P_left = P_virt @ Rv.T                                 # -> left frame
    return P_left[valid], col[valid], valid, depth


# --------------------------------------------------------------------------- #
def fuse_to_fisheye(points, colors, Kl, Dl, fish_w, fish_h):
    """Z-buffer all fused points into the LEFT fisheye image. Returns range map + color map."""
    P = torch.tensor(points, dtype=torch.float32)
    px = FP.fisheye_project(P, Kl.cpu(), Dl.cpu()).numpy()  # (N,2)
    rng = np.linalg.norm(points, axis=1)
    u = np.round(px[:, 0]).astype(int); v = np.round(px[:, 1]).astype(int)
    on = (u >= 0) & (u < fish_w) & (v >= 0) & (v < fish_h) & (points[:, 2] > 0)
    u, v, rng, cols = u[on], v[on], rng[on], colors[on]
    order = np.argsort(-rng)                                # far first -> near overwrites
    rmap = np.full((fish_h, fish_w), np.inf, np.float32)
    cmap = np.zeros((fish_h, fish_w, 3), np.uint8)
    rmap[v[order], u[order]] = rng[order]
    cmap[v[order], u[order]] = np.clip(cols[order], 0, 255).astype(np.uint8)
    rmap[~np.isfinite(rmap)] = 0
    return rmap, cmap


def save_ply(path, points, colors):
    import open3d as o3d
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pc.colors = o3d.utility.Vector3dVector(np.clip(colors, 0, 255) / 255.0)  # colors are RGB
    o3d.io.write_point_cloud(path, pc)


def turbo(depth):
    m = depth > 0
    if not m.any():
        return np.zeros((*depth.shape, 3), np.uint8)
    lo, hi = np.percentile(depth[m], [2, 98])
    n = np.clip((depth - lo) / max(hi - lo, 1e-6), 0, 1)
    vis = cv2.applyColorMap((n * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    vis[~m] = 0
    return vis


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", type=int, required=True)
    ap.add_argument("--video", default="long-test1/left_stereo_8bit.mp4")
    ap.add_argument("--calib-dir", default="long-test1")
    ap.add_argument("--left-serial", default="046060323008")
    ap.add_argument("--right-serial", default="046060323001")
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--out", default="data_processing/out/scene")
    ap.add_argument("--tile-w", type=int, default=960)
    ap.add_argument("--hfov", type=float, default=100.0, help="horizontal FOV per tile (deg)")
    ap.add_argument("--vfov", type=float, default=52.0, help="vertical FOV per tile (deg)")
    ap.add_argument("--pitches", type=float, nargs="+",
                    default=[-60, -30, 0, 30, 60], help="pitch angles about baseline (deg)")
    ap.add_argument("--max-depth", type=float, default=20.0)
    ap.add_argument("--debug", action="store_true", help="save per-tile color|disparity montage")
    args = ap.parse_args()

    dev = torch.device("cuda")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    Kl, Dl, Kr, Dr, Rs, ts, b_hat, baseline = load_calib(
        args.calib_dir, args.left_serial, args.right_serial, dev)
    print(f"baseline={baseline*1000:.1f}mm  b_hat={b_hat.cpu().numpy().round(3)}")

    # decode the stereo frame (side-by-side 8-bit file: left | right)
    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, fr = cap.read(); cap.release()
    assert ok, f"could not read frame {args.frame}"
    half = fr.shape[1] // 2
    to_t = lambda bgr: torch.tensor(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
                                    device=dev, dtype=torch.float32).permute(2, 0, 1)
    frameL = to_t(fr[:, :half]); frameR = to_t(fr[:, half:])
    fish_h, fish_w = frameL.shape[1:]

    model = load_ffs(args.weights, dev)

    W = args.tile_w
    fx = (W / 2.0) / np.tan(np.radians(args.hfov) / 2.0)
    fy = fx
    H = int(round(2 * fx * np.tan(np.radians(args.vfov) / 2.0) / 32) * 32)
    dirs = rect_dirs(W, H, fx, fy, dev)
    looks = tile_look_dirs(b_hat, args.pitches, dev)
    print(f"tiles: {len(looks)} x {W}x{H}  fx={fx:.1f}  hfov={args.hfov}  vfov~{args.vfov}")

    all_pts, all_cols, panels = [], [], []
    for p, g in zip(args.pitches, looks):
        Rv_l = FP.baseline_aligned_R(g, b_hat)
        Rv_r = Rs @ Rv_l                                    # same world orientation
        tileL = render_tile(frameL, Rv_l, dirs, Kl, Dl, W, H)
        tileR = render_tile(frameR, Rv_r, dirs, Kr, Dr, W, H)
        disp = ffs_disparity(model, tileL, tileR, dev)
        pts, cols, valid, depth = backproject(
            disp, dirs, Rv_l, fx, baseline, tileL, W, H, max_depth=args.max_depth)
        med = np.median(depth[valid]) if valid.any() else float("nan")
        print(f"  pitch {p:+5.0f}: valid {100*valid.mean():4.1f}%  "
              f"disp med {np.median(disp[np.isfinite(disp)]):5.1f}px  depth med {med:.3f}m  pts {len(pts)}")
        all_pts.append(pts); all_cols.append(cols)
        if args.debug:
            rgb = torch.nan_to_num(tileL).permute(1, 2, 0).cpu().numpy()
            bgr = cv2.cvtColor(np.clip(rgb, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            panels.append(np.concatenate([bgr, turbo(depth * valid)], axis=1))
    if args.debug and panels:
        cv2.imwrite(f"{args.out}_tiles.png", np.concatenate(panels[::-1], axis=0))
        print(f"saved: {args.out}_tiles.png  (top=+pitch/up ... bottom=-pitch/down)")

    points = np.concatenate(all_pts); colors = np.concatenate(all_cols)
    rmap, cmap = fuse_to_fisheye(points, colors, Kl, Dl, fish_w, fish_h)
    cov = 100 * (rmap > 0).mean()
    print(f"\nFUSED: {len(points):,} pts  depth {np.percentile(rmap[rmap>0],[5,50,95]).round(3)}m  "
          f"left-fisheye coverage {cov:.1f}%")

    np.save(f"{args.out}_range.npy", rmap)
    save_ply(f"{args.out}_cloud.ply", points, colors)
    cv2.imwrite(f"{args.out}_depth.png", turbo(rmap))
    cv2.imwrite(f"{args.out}_color.png", cmap[:, :, ::-1])   # RGB -> BGR for cv2
    print(f"saved: {args.out}_{{range.npy,cloud.ply,depth.png,color.png}}")


if __name__ == "__main__":
    main()
