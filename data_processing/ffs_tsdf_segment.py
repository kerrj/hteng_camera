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


def R_to_quat(R):  # (3,3) -> wxyz (Shepperd's method, always well-conditioned)
    m00, m01, m02 = R[0]
    m10, m11, m12 = R[1]
    m20, m21, m22 = R[2]
    tr = m00 + m11 + m22
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        q = np.array([0.25 * s, (m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s])
    elif m00 >= m11 and m00 >= m22:
        s = np.sqrt(1.0 + m00 - m11 - m22) * 2
        q = np.array([(m21 - m12) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s])
    elif m11 >= m22:
        s = np.sqrt(1.0 + m11 - m00 - m22) * 2
        q = np.array([(m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s])
    else:
        s = np.sqrt(1.0 + m22 - m00 - m11) * 2
        q = np.array([(m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s])
    return q / np.linalg.norm(q)


def apply_world_correction(ext, dT):
    """ICP found dT aligning this frame's world points onto the model
    (X_model = dT @ X_frame). The camera that observes the corrected points at
    the same pixels is ext' = ext @ inv(dT): ext' (dT X) = ext X."""
    return ext @ np.linalg.inv(dT)


def cam_to_world_correction(ext, dT_c):
    """Conjugate a camera-frame ICP correction (X_model_c = dT_c @ X_meas_c,
    both clouds lifted with identity extrinsic at guess pose ext = world->cam)
    into the equivalent world-frame correction."""
    return np.linalg.inv(ext) @ dT_c @ ext


def rot_angle_deg(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


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


def splat_zbuffer_torch(P, C, fx, cx, cy, W, H):
    """3x3 z-buffer splat of camera-frame points into a virtual pinhole.

    torch (CUDA when the tensors live there); replaces the numpy painter's
    loop, which was 78% of the integration loop's wall clock. Nearest point
    wins per pixel; ties broken by point index (packed key: depth | index).
    P (N,3) float32, C (N,3) uint8 -> depth (H,W) float32, color (H,W,3) uint8.
    """
    assert len(P) < (1 << 22), "point index no longer fits the packed key"
    Z = P[:, 2]
    u0 = torch.round(fx * P[:, 0] / Z + cx).long()
    v0 = torch.round(fx * P[:, 1] / Z + cy).long()
    zq = (Z * 1e4).long().clamp(min=0)                  # 0.1mm depth quanta
    key = (zq << 22) | torch.arange(len(Z), device=Z.device)
    empty = 1 << 62
    best = torch.full((H * W,), empty, dtype=torch.int64, device=Z.device)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            u, v = u0 + dx, v0 + dy
            ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
            best.scatter_reduce_(0, v[ok] * W + u[ok], key[ok],
                                 reduce="amin", include_self=True)
    hit = best < empty
    idx = best[hit] & ((1 << 22) - 1)
    depth = torch.zeros(H * W, dtype=torch.float32, device=Z.device)
    color = torch.zeros((H * W, 3), dtype=torch.uint8, device=Z.device)
    depth[hit] = Z[idx]
    color[hit] = C[idx]
    return depth.reshape(H, W), color.reshape(H, W, 3)


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
    ap.add_argument("--block-count", type=int, default=100_000,
                    help="VBG hashmap capacity (preallocated). Stationary "
                         "workspaces need ~50k at 3mm; large swept volumes "
                         "(kitchen, walking) overflow 100k -> illegal memory "
                         "access at extract. ~82KB GPU per block.")
    ap.add_argument("--hfov", type=float, default=110.0)
    ap.add_argument("--img-w", type=int, default=800)
    ap.add_argument("--mask-dilate-px", type=int, default=12)
    ap.add_argument("--frame-stride", type=int, default=1)
    ap.add_argument("--out-prefix", default="data_processing/out/lt2_seg")
    ap.add_argument("--engine", choices=("vbg", "legacy"), default="vbg",
                    help="vbg = t.geometry.VoxelBlockGrid (CUDA, supports "
                         "--weight-threshold); legacy = ScalableTSDFVolume")
    ap.add_argument("--weight-threshold", type=float, default=3.0,
                    help="vbg only: min integration weight (~frames seen) for a "
                         "voxel to reach the mesh -- kills 1-2-view 'lace'")
    ap.add_argument("--mask-against", default=None,
                    help="pass-2 rebuild: prior mesh ply; pixels whose measured "
                         "depth disagrees (in front by >3 cm, despeckled) are "
                         "masked out of integration IN ADDITION to MANO -- "
                         "catches manipulated objects, blankets, unfit hands")
    ap.add_argument("--refine-poses", action="store_true",
                    help="vbg only: frame-to-model ICP pose polish. Raycast the "
                         "accumulating TSDF from the VIO pose, point-to-plane "
                         "ICP measured-vs-model, integrate at the corrected "
                         "pose. The VIO pose stays the per-frame prior, so "
                         "corrections are bounded (no odometry drift). Writes "
                         "<out-prefix>_trajectory_refined.npz")
    ap.add_argument("--refine-warmup", type=int, default=20,
                    help="integrate this many frames at VIO poses before ICP "
                         "(model too thin to track against earlier)")
    ap.add_argument("--refine-max-corr", type=float, default=0.015,
                    help="ICP max correspondence distance (m)")
    ap.add_argument("--refine-stride", type=int, default=4,
                    help="pixel stride when lifting depth images to ICP clouds")
    ap.add_argument("--refine-max-depth", type=float, default=1.2,
                    help="ICP uses only points nearer than this: stereo depth "
                         "noise grows as Z^2 and passes the correspondence "
                         "cap (~1.5cm) around 1.3m -- beyond that, points "
                         "dilute fitness with pure noise")
    ap.add_argument("--refine-gate-t", type=float, default=0.03,
                    help="reject corrections translating more than this (m)")
    ap.add_argument("--refine-gate-deg", type=float, default=2.0,
                    help="reject corrections rotating more than this (deg)")
    ap.add_argument("--refine-min-fitness", type=float, default=0.35,
                    help="reject ICP results with inlier fraction below this. "
                         "0.35 validated on segment C: accepted corrections "
                         "are temporally smooth (real pose error), and the "
                         "magnitude gates bound any residual ICP noise")
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
    import open3d.core as o3c
    if args.engine == "vbg":
        dev = o3c.Device("CUDA:0") if o3c.cuda.is_available() else o3c.Device("CPU:0")
        vbg = o3d.t.geometry.VoxelBlockGrid(
            attr_names=("tsdf", "weight", "color"),
            attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
            attr_channels=((1), (1), (3)),
            voxel_size=args.voxel, block_resolution=16,
            block_count=args.block_count, device=dev)
        K_t = o3c.Tensor([[fx, 0, cx], [0, fx, cy], [0, 0, 1]], o3c.float64)
    else:
        vol = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=args.voxel, sdf_trunc=4 * args.voxel,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    intr = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fx, cx, cy)

    prior = None
    if args.mask_against:
        from ffs_dynamic_residual import residual_mask, despeckle  # lazy: avoids cycle
        pm = o3d.io.read_triangle_mesh(args.mask_against)
        prior = o3d.t.geometry.RaycastingScene()
        prior.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(pm))
        prior_fns = (residual_mask, despeckle)
        ray_norm = np.linalg.norm(rays, axis=-1)

    # size by BOTH ranges: a segment-only --trajectory (e.g. a refined npz)
    # ends before the recording's gyro frames do
    gyro_of = np.zeros(int(max(frame_idx[-1], gyro_frames.max())) + 2, np.float64)
    gyro_of[np.asarray(gyro_frames, int)] = gyro_dps

    refine = args.refine_poses and args.engine == "vbg"
    if refine:
        import open3d.t.pipelines.registration as treg
        icp_eye = o3c.Tensor(np.eye(4), o3c.float64)
        eye4_d = o3c.Tensor(np.eye(4), o3c.float64).to(dev)
        used_poses = {}                     # vf -> world->cam ext actually used
        corr_t, corr_deg, n_rej = [], [], 0
        rej_fit, rej_gate, rej_pts = 0, 0, 0
        fits = []

    used = skipped_blur = missing_hand = 0
    tdev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        # pass-2: mask disagreement vs the prior mesh (objects, blankets,
        # unfit hands). Only the in-front clause -- the orphan clause would
        # forbid pass 2 from ever filling pass-1 holes with real surface.
        if prior is not None:
            i0 = pose_of[vf]
            R0 = Rws[i0]
            c0 = -(R0.T @ poses[i0, 4:])
            dirs_w = (rays.reshape(-1, 3) @ R0).astype(np.float32)
            org = np.broadcast_to(c0.astype(np.float32), dirs_w.shape)
            rc = prior.cast_rays(o3c.Tensor(np.concatenate(
                [org, dirs_w], axis=1).reshape(-1, 6)))
            t_hit = rc["t_hit"].numpy().reshape(Hf, Wf) * ray_norm
            rmask, dspk = prior_fns
            m2 = dspk(rmask(rng, t_hit, 0.03, 0.0), 50)  # near_orphan=0 disables
            rng[m2] = 0.0
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
        keep = P[:, 2] > 0.05
        P_t = torch.tensor(P[keep], dtype=torch.float32, device=tdev)
        C_t = torch.tensor(C[keep], device=tdev)
        depth_t, color_t = splat_zbuffer_torch(P_t, C_t, fx, cx, cy, W, H)
        i = pose_of[vf]
        ext = np.eye(4)
        ext[:3, :3] = Rws[i]
        ext[:3, 3] = poses[i, 4:]                            # world->cam = o3d extrinsic
        if args.engine == "vbg":
            # NOT dlpack zero-copy: torch's caching allocator recycles the
            # buffer while o3d's async stream may still read it -- illegal
            # memory access surfacing later at extract. ~4ms/frame is cheap.
            d_t = o3d.t.geometry.Image(o3c.Tensor(
                depth_t.cpu().numpy(), device=dev))
            c_t = o3d.t.geometry.Image(o3c.Tensor(
                (color_t.float() / 255.0).cpu().numpy(), device=dev))
            ext_t = o3c.Tensor(ext, o3c.float64)
            blocks = vbg.compute_unique_block_coordinates(
                d_t, K_t, ext_t, 1.0, args.max_range)
            if refine and used >= args.refine_warmup:
                rc = vbg.ray_cast(block_coords=blocks, intrinsic=K_t,
                                  extrinsic=ext_t, width=W, height=H,
                                  render_attributes=["depth", "color"],
                                  depth_scale=1.0,
                                  depth_min=0.05, depth_max=args.max_range,
                                  weight_threshold=1.0)
                model_raw = rc["depth"].reshape((H, W))
                model_d = o3d.t.geometry.Image(model_raw)
                model_c = o3d.t.geometry.Image(
                    rc["color"].reshape((H, W, 3)).to(o3c.float32))
                # Restrict the source to pixels comparable with the model:
                # (a) the raycast hit (surface the model has already seen);
                # (b) measured NOT well in front of it -- that's a dynamic
                # occluder (unfit hand, held object) the static model
                # rightly lacks. Both would dilute fitness with pixels that
                # cannot have correspondences at any pose.
                d_meas = d_t.as_tensor().reshape((H, W))
                comparable = (model_raw > 0) & (d_meas > model_raw - 0.05)
                overlap_d = o3d.t.geometry.Image(
                    d_meas * comparable.to(o3c.float32))
                # Lift both clouds in the GUESS pose's camera frame (identity
                # extrinsic): with_normals=True + non-identity extrinsic hits
                # an Open3D CUDA bug (normal-rotation matmul lands on CPU).
                # ICP's camera-frame dT_c is conjugated back to world below.
                K_d = K_t.to(dev)
                src = o3d.t.geometry.PointCloud.create_from_rgbd_image(
                    o3d.t.geometry.RGBDImage(c_t, overlap_d),
                    K_d, eye4_d, 1.0, args.refine_max_depth,
                    args.refine_stride)
                tgt = o3d.t.geometry.PointCloud.create_from_rgbd_image(
                    o3d.t.geometry.RGBDImage(model_c, model_d),
                    K_d, eye4_d, 1.0, args.refine_max_depth,
                    args.refine_stride, with_normals=True)
                ok_icp = False
                if len(src.point.positions) > 1000 and len(tgt.point.positions) > 1000:
                    # Colored ICP: the near field during motion phases is
                    # often one dominant plane (desk/wall) -- point-to-plane
                    # alone slides in-plane; color gradients pin those DOFs.
                    res = treg.icp(
                        src, tgt, args.refine_max_corr, icp_eye,
                        treg.TransformationEstimationForColoredICP(),
                        treg.ICPConvergenceCriteria(max_iteration=20))
                    dT_c = res.transformation.cpu().numpy()
                    dT = cam_to_world_correction(ext, dT_c)
                    dt_m = float(np.linalg.norm(dT[:3, 3]))
                    dr_deg = rot_angle_deg(dT[:3, :3])
                    fits.append(res.fitness)
                    if res.fitness < args.refine_min_fitness:
                        rej_fit += 1
                    elif dt_m > args.refine_gate_t or dr_deg > args.refine_gate_deg:
                        rej_gate += 1
                    else:
                        ok_icp = True
                else:
                    rej_pts += 1
                if ok_icp:
                    ext = apply_world_correction(ext, dT)
                    ext_t = o3c.Tensor(ext, o3c.float64)
                    blocks = vbg.compute_unique_block_coordinates(
                        d_t, K_t, ext_t, 1.0, args.max_range)
                    corr_t.append(dt_m)
                    corr_deg.append(dr_deg)
                else:
                    n_rej += 1
            if refine:
                used_poses[vf] = ext.copy()
            vbg.integrate(blocks, d_t, c_t, K_t, K_t, ext_t, 1.0, args.max_range)
        else:
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(np.ascontiguousarray(color_t.cpu().numpy())),
                o3d.geometry.Image(np.ascontiguousarray(depth_t.cpu().numpy())),
                depth_scale=1.0, depth_trunc=args.max_range,
                convert_rgb_to_intensity=False)
            vol.integrate(rgbd, intr, ext)
        used += 1
    cap.release()
    print(f"integrated {used} frames (skipped {skipped_blur} blurred; "
          f"{missing_hand} frame-sides lacked a hand fit)", flush=True)
    if args.engine == "vbg":
        nblk = vbg.hashmap().size()
        print(f"VBG blocks: {nblk:,} / {args.block_count:,} capacity", flush=True)
        assert nblk < args.block_count, (
            "block hashmap overflowed -- raise --block-count (mesh would be "
            "corrupt; extraction crashes with illegal memory access)")

    if refine:
        if corr_t:
            print(f"ICP refine: corrected {len(corr_t)} frames "
                  f"(rejected {n_rej}: fitness {rej_fit}, gate {rej_gate}, "
                  f"too-few-pts {rej_pts}); "
                  f"|t| median {1e3*np.median(corr_t):.1f}mm "
                  f"p90 {1e3*np.percentile(corr_t, 90):.1f}mm; "
                  f"rot median {np.median(corr_deg):.3f}deg", flush=True)
        else:
            print(f"ICP refine: 0 corrections accepted ({n_rej} rejected: "
                  f"fitness {rej_fit}, gate {rej_gate}, too-few-pts {rej_pts})",
                  flush=True)
        if fits:
            print(f"ICP fitness p10/50/90: "
                  f"{np.percentile(fits, [10, 50, 90]).round(3)}", flush=True)
        # trajectory-format npz: refined ext where a frame was integrated,
        # the VIO pose otherwise -- downstream (dynamic residual, viewer)
        # can point --trajectory here and see every segment frame
        fr, pq = [], []
        for vf in range(s0, e0):
            if vf in used_poses:
                E = used_poses[vf]
            elif vf in pose_of:
                i = pose_of[vf]
                E = np.eye(4)
                E[:3, :3] = Rws[i]
                E[:3, 3] = poses[i, 4:]
            else:
                continue
            fr.append(vf)
            pq.append(np.concatenate([R_to_quat(E[:3, :3]), E[:3, 3]]))
        rout = f"{args.out_prefix}_trajectory_refined.npz"
        np.savez(rout, frame_idx=np.array(fr, np.int64),
                 pose_wxyz_xyz=np.array(pq, np.float64))
        print(f"wrote {rout} ({len(fr)} frames)", flush=True)

    if args.engine == "vbg":
        mesh = vbg.extract_triangle_mesh(
            weight_threshold=args.weight_threshold).to_legacy()
    else:
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
