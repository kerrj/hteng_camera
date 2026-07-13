# TSDF Task-Segment Reconstruction (PoC) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Clean static workspace mesh (TSDF + MANO hand masking) of one auto-selected ~30 s task segment of long-test2, side-by-side vs naive accumulation, viewable in viser with animated hands.

**Architecture:** One new script `ffs_tsdf_segment.py` (segment selection → per-frame mask/clamp → virtual-pinhole render → Open3D ScalableTSDFVolume → mesh), one new viewer `ffs_tsdf_viewer.py` (static mesh + animated world-frame MANO hands), a two-line segment-bounds edit to `ffs_fuse_world.py`, and a zero-code-change bake run of `precompute_hand_meshes.py` on the long-test2 hand fits.

**Tech Stack:** Python in conda env `eyeball` (torch GPU for splatting, open3d 0.19 ScalableTSDFVolume, trimesh 4.11 for colored viser mesh, cv2, viser). No jax solver work; poses come from the already-refined `long-test2/derived/trajectory.npz` (0.328°).

**Spec:** `docs/superpowers/specs/2026-07-13-tsdf-task-segment-design.md`

## Global Constraints

- Branch `depth_understanding` only; NEVER push to or modify `main`.
- Env: `conda run --no-capture-output -n eyeball python …` for everything. GPU work pins `CUDA_VISIBLE_DEVICES` (any free GPU; check `nvidia-smi` — GPU 3 belongs to another user).
- Recording data gitignored; all outputs under `data_processing/out/`.
- Reuse existing conventions: range maps via `meta.json` + `range_<s>_<e>.npy` memmaps (`scale` 0.5 → half-res fisheye 1224×1024); poses `pose_wxyz_xyz` are WORLD→CAM (`c = −Rᵀt`; world→cam extrinsic is exactly what Open3D `integrate()` expects); fisheye math via `fisheye_pinhole.py` (`FP.fisheye_unproject`, `FP.fisheye_project`).
- Key data facts: refined trajectory covers frames 0..11042 (frame 1 dropped); hand fits `long-test2/derived/hands3d_{left,right}.jsonl` have per-frame `quat (16,4)`, `trans_virtual`, `Rv_l`, meta `beta_opt`/`mirror` — the exact schema `precompute_hand_meshes.py` reads; `data_processing/out/hand_mesh_*.npz` are LONG-TEST1 (do not reuse, do not overwrite); gyro rate per frame-edge = `2·degrees(arccos(|q_w|))·30` dps from `imu_relative.npz` `rel_quat`.
- Tests live in `data_processing/`, run: `cd data_processing && conda run --no-capture-output -n eyeball python -m pytest <file> -v`.
- Commit messages end with:
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`
  `Claude-Session: https://claude.ai/code/session_01BdNbSVuPYhr1KeDeTySjF3`

---

### Task 1: Bake long-test2 hand meshes (run-only, no code)

**Files:**
- Create (outputs, gitignored): `data_processing/out/lt2_hands/hand_mesh_left.npz`, `.../hand_mesh_right.npz`

**Interfaces:**
- Produces: per-hand npz `{frames (N,), verts (N,778,3) float32 in LEFT-FISHEYE camera frame, faces (F,3) int32}` — consumed by Tasks 3 (masking) and 4 (viewer).

- [ ] **Step 1: Run the bake**

```bash
cd /home/smahapatra/hteng_camera
mkdir -p data_processing/out/lt2_hands
conda run --no-capture-output -n eyeball python data_processing/precompute_hand_meshes.py \
    --viz-left  long-test2/derived/hands3d_left.jsonl \
    --viz-right long-test2/derived/hands3d_right.jsonl \
    --mano data_processing/out/mano_jax.npz \
    --out-dir data_processing/out/lt2_hands
```

Expected: two lines like `data_processing/out/lt2_hands/hand_mesh_left.npz: ~8000 frames` (hands3d_left has 8261 records incl. meta). No code changes — the jsonl schema matches the reader.

- [ ] **Step 2: Verify coverage + sane geometry**

```bash
conda run --no-capture-output -n eyeball python - <<'EOF'
import numpy as np
for s in ("left", "right"):
    d = np.load(f"data_processing/out/lt2_hands/hand_mesh_{s}.npz")
    span = np.linalg.norm(d["verts"].reshape(len(d["frames"]), -1, 3).ptp(1), axis=1)
    print(s, "frames", d["frames"].min(), "..", d["frames"].max(), "n", len(d["frames"]),
          "hand-extent median %.3f m" % np.median(span))
    assert d["frames"].max() <= 11042 and d["verts"].shape[1] == 778
EOF
```

Expected: frames within 0..11042, hand extent median ~0.15–0.25 m (a hand-sized mesh). No commit (outputs are gitignored).

---

### Task 2: Segment-bounds args in `ffs_fuse_world.py`

**Files:**
- Modify: `data_processing/ffs_fuse_world.py` (argparse ~line 45; frame loop ~line 108)

**Interfaces:**
- Produces: `--start-frame N --end-frame M` restricting fusion to video frames in `[N, M]`; default `None` = unchanged full-length behavior. Used in Task 4 for the same-segment naive baseline.

- [ ] **Step 1: Implement**

(a) argparse, after `--frame-stride`:

```python
    ap.add_argument("--start-frame", type=int, default=None,
                    help="first video frame to fuse (segment mode)")
    ap.add_argument("--end-frame", type=int, default=None,
                    help="last video frame to fuse (inclusive)")
```

(b) at the top of the frame loop, right after `vf = int(frame_idx[i])`:

```python
        if args.start_frame is not None and vf < args.start_frame:
            continue
        if args.end_frame is not None and vf > args.end_frame:
            continue
```

- [ ] **Step 2: Sanity check (no full run)**

```bash
cd /home/smahapatra/hteng_camera
conda run --no-capture-output -n eyeball python data_processing/ffs_fuse_world.py \
    --range-dir data_processing/out/lt2_video \
    --trajectory long-test2/derived/trajectory.npz \
    --out /tmp/claude-1000845/-home-smahapatra-hteng-camera/87c1bb25-0989-44d9-8102-068a2b853e99/scratchpad/seg_sanity.ply \
    --start-frame 1000 --end-frame 1100 --frame-stride 10 --max-range 2.0
```

Expected: `accumulated … pts from ~10 frames`, writes the ply. (Frames outside [1000,1100] skipped.)

- [ ] **Step 3: Commit**

```bash
git add data_processing/ffs_fuse_world.py
git commit -m "ffs_fuse_world: --start-frame/--end-frame segment bounds"
```

(Append the two Global Constraints trailer lines.)

---

### Task 3: `ffs_tsdf_segment.py` — selection, masking, TSDF

**Files:**
- Create: `data_processing/ffs_tsdf_segment.py`
- Test: `data_processing/test_tsdf_segment.py`

**Interfaces:**
- Consumes: range-dir (`meta.json` + range stacks), `trajectory.npz`, `imu_relative.npz`, Task 1's hand npzs, video for color.
- Produces: CLI below; pure helpers `pick_segment(centers, frame_idx, gyro_dps, gyro_frames, hand_frames, window, stride, min_hand_frac) -> (s, e, report_dict)` and `rasterize_hand_mask(verts_px, H, W, dilate_px) -> bool (H,W)`; outputs `out/<prefix>_tsdf_mesh.ply` + prints chosen segment.
- CLI: `--range-dir out/lt2_video --trajectory long-test2/derived/trajectory.npz --imu long-test2/derived/imu_relative.npz --hands-dir out/lt2_hands --start/--end (override auto-pick) --window-frames 900 --pick-stride 30 --min-hand-frac 0.7 --max-rot-dps 20 --max-range 2.0 --voxel 0.005 --hfov 110 --img-w 800 --mask-dilate-px 12 --frame-stride 1 --out-prefix out/lt2_seg`

- [ ] **Step 1: Write the failing tests**

Create `data_processing/test_tsdf_segment.py`:

```python
import numpy as np

import ffs_tsdf_segment as TS


def test_pick_segment_prefers_calm_handy_window():
    n = 3000
    frame_idx = np.arange(n)
    # camera walks 5 m over frames 0..1000, then stands ~still 1000..2000, walks again
    centers = np.zeros((n, 3))
    centers[:1000, 0] = np.linspace(0, 5, 1000)
    centers[1000:2000, 0] = 5 + 0.01 * np.sin(np.linspace(0, 20, 1000))
    centers[2000:, 0] = np.linspace(5, 10, 1000)
    gyro = np.full(n - 1, 40.0)          # spinning...
    gyro[1000:2000] = 5.0                # ...except while standing
    hands = np.arange(900, 2100)         # hands present around the calm stretch
    s, e, rep = TS.pick_segment(centers, frame_idx, gyro, frame_idx[:-1], hands,
                                window=900, stride=30, min_hand_frac=0.7)
    assert 900 <= s <= 1200 and e == s + 900
    assert rep["hand_frac"] >= 0.7 and rep["mean_dps"] < 10


def test_pick_segment_rejects_handless_windows():
    n = 2000
    frame_idx = np.arange(n)
    centers = np.zeros((n, 3))
    gyro = np.full(n - 1, 5.0)
    hands = np.arange(0, 300)            # hands only at the very start
    s, e, rep = TS.pick_segment(centers, frame_idx, gyro, frame_idx[:-1], hands,
                                window=900, stride=30, min_hand_frac=0.7)
    # nothing satisfies the hand gate -> falls back to best hand_frac window
    assert rep["fallback"] is True and s == 0


def test_rasterize_hand_mask_dilates():
    px = np.array([[50.0, 50.0], [52.0, 50.0]])
    m = TS.rasterize_hand_mask(px, 100, 100, dilate_px=5)
    assert m[50, 50] and m[50, 55] and not m[50, 70]
    assert m.sum() > 50  # dilation grew the two seeds into a blob
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/smahapatra/hteng_camera/data_processing && conda run --no-capture-output -n eyeball python -m pytest test_tsdf_segment.py -v`
Expected: collection error — `ModuleNotFoundError: No module named 'ffs_tsdf_segment'`

- [ ] **Step 3: Implement `ffs_tsdf_segment.py`**

Create `data_processing/ffs_tsdf_segment.py`:

```python
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
    exts = np.array([c[2] for c in ok]); dpss = np.array([c[3] for c in ok])

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

    imu = np.load(args.imu)
    gyro_dps = 2 * np.degrees(np.arccos(np.clip(np.abs(imu["rel_quat"][:, 0]), -1, 1))) * 30
    gyro_frames = imu["frame_idx"][:-1]

    hands = {}
    for side in ("left", "right"):
        d = np.load(f"{args.hands_dir}/hand_mesh_{side}.npz")
        hands[side] = (dict(zip(d["frames"].tolist(), d["verts"])), d["faces"])
    hand_frames = np.unique(np.concatenate(
        [np.fromiter(hands[s][0].keys(), int) for s in hands]))

    if args.start is not None:
        s0, e0 = args.start, args.end if args.end is not None else args.start + args.window_frames
        rep = {"manual": True}
    else:
        s0, e0, rep = pick_segment(centers, frame_idx, gyro_dps, gyro_frames,
                                   hand_frames, args.window_frames,
                                   args.pick_stride, args.min_hand_frac)
    print(f"segment: frames {s0}..{e0}  (t={s0/30:.1f}s..{e0/30:.1f}s)  {rep}", flush=True)

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
        ok, fr = cap.read(); cur = vf
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
        json.dump({"start": int(s0), "end": int(e0), "report": rep,
                   "used": used, "skipped_blur": skipped_blur}, f, indent=2)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/smahapatra/hteng_camera/data_processing && conda run --no-capture-output -n eyeball python -m pytest test_tsdf_segment.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add data_processing/ffs_tsdf_segment.py data_processing/test_tsdf_segment.py
git commit -m "ffs_tsdf_segment: TSDF task-segment fusion with MANO hand masking"
```

---

### Task 4: Viewer (`ffs_tsdf_viewer.py`) — mesh + animated world-frame hands

**Files:**
- Create: `data_processing/ffs_tsdf_viewer.py`

**Interfaces:**
- Consumes: `<prefix>_tsdf_mesh.ply`, `<prefix>_segment.json`, hand npzs (Task 1), `trajectory.npz`.
- Produces: viser server; hands per frame transformed cam→world: `R_wl = quat_to_R(pose[:4]); c = −R_wlᵀ pose[4:]; V_world = V_cam @ R_wl + c` (note: `@ R_wl` on row-vectors = `R_wlᵀ · v`, the cam→world rotation).

- [ ] **Step 1: Implement**

Create `data_processing/ffs_tsdf_viewer.py`:

```python
"""Viser viewer: static TSDF workspace mesh + animated MANO hand surfaces.

The mesh is the hand-masked static world (ffs_tsdf_segment.py); the hands are
re-posed per frame from the baked hand npzs and placed in the world by the
same trajectory the TSDF used -- clean scene + moving hands, one world frame.

  python data_processing/ffs_tsdf_viewer.py --prefix data_processing/out/lt2_seg \
      --hands-dir data_processing/out/lt2_hands \
      --trajectory long-test2/derived/trajectory.npz --share
"""
import argparse
import json
import threading
import time

import numpy as np
import open3d as o3d
import trimesh
import viser

from ffs_tsdf_segment import quat_to_R  # same pose convention


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="data_processing/out/lt2_seg")
    ap.add_argument("--hands-dir", default="data_processing/out/lt2_hands")
    ap.add_argument("--trajectory", default="long-test2/derived/trajectory.npz")
    ap.add_argument("--port", type=int, default=8091)
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--fps", type=float, default=30.0)
    args = ap.parse_args()

    seg = json.load(open(f"{args.prefix}_segment.json"))
    s0, e0 = seg["start"], seg["end"]

    mesh = o3d.io.read_triangle_mesh(f"{args.prefix}_tsdf_mesh.ply")
    tm = trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles),
                         vertex_colors=(np.asarray(mesh.vertex_colors) * 255).astype(np.uint8),
                         process=False)

    tr = np.load(args.trajectory)
    pose_of = {int(f): i for i, f in enumerate(tr["frame_idx"])}
    poses = tr["pose_wxyz_xyz"]

    hands = {}
    for side in ("left", "right"):
        d = np.load(f"{args.hands_dir}/hand_mesh_{side}.npz")
        hands[side] = (dict(zip(d["frames"].tolist(), d["verts"])), d["faces"])

    server = viser.ViserServer(port=args.port)
    if args.share:
        print(f"SHARE URL: {server.request_share_url()}", flush=True)
    server.scene.set_up_direction("+z")
    server.scene.add_mesh_trimesh("/scene", tm)

    hand_h = {}
    for side, color in (("left", (231, 76, 60)), ("right", (46, 204, 113))):
        faces = hands[side][1]
        hand_h[side] = server.scene.add_mesh_simple(
            f"/hands/{side}", vertices=np.zeros((778, 3), np.float32),
            faces=faces, color=color, opacity=0.85, visible=False)

    g_frame = server.gui.add_slider("frame", s0, e0 - 1, 1, s0)
    g_play = server.gui.add_checkbox("play", True)
    lock = threading.Lock()

    def show(vf):
        i = pose_of.get(int(vf))
        if i is None:
            return
        R = quat_to_R(poses[i, :4])
        c = -(R.T @ poses[i, 4:])
        for side in ("left", "right"):
            V = hands[side][0].get(int(vf))
            h = hand_h[side]
            if V is None:
                h.visible = False
                continue
            h.vertices = (np.asarray(V) @ R + c).astype(np.float32)
            h.visible = True

    g_frame.on_update(lambda _: (lock.acquire(), show(g_frame.value), lock.release()))
    print(f"viser on :{args.port}", flush=True)
    while True:
        if g_play.value:
            with lock:
                nxt = g_frame.value + 1
                g_frame.value = s0 if nxt >= e0 else nxt
                show(g_frame.value)
        time.sleep(1.0 / args.fps)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add data_processing/ffs_tsdf_viewer.py
git commit -m "ffs_tsdf_viewer: static TSDF mesh + animated world-frame MANO hands"
```

(Runtime verification happens in Task 5 with real data.)

---

### Task 5: Run the PoC end-to-end + side-by-side + report

**Files:**
- Outputs: `data_processing/out/lt2_seg_tsdf_mesh.ply`, `lt2_seg_segment.json`, `lt2_seg_naive.ply`, `lt2_seg_compare.png`

- [ ] **Step 1: TSDF run (auto-picked segment)**

```bash
cd /home/smahapatra/hteng_camera
conda run --no-capture-output -n eyeball python data_processing/ffs_tsdf_segment.py \
    --range-dir data_processing/out/lt2_video \
    --trajectory long-test2/derived/trajectory.npz \
    --imu long-test2/derived/imu_relative.npz \
    --hands-dir data_processing/out/lt2_hands \
    --out-prefix data_processing/out/lt2_seg
```

Expected: prints the chosen segment (sanity: hand_frac ≥ 0.7, mean_dps low, extent ≲ 1.5 m), integrates several hundred frames, writes a mesh with ≥ 1M vertices. Report the chosen segment to the user.

- [ ] **Step 2: Naive baseline on the SAME segment/frames**

```bash
conda run --no-capture-output -n eyeball python data_processing/ffs_fuse_world.py \
    --range-dir data_processing/out/lt2_video \
    --trajectory long-test2/derived/trajectory.npz \
    --start-frame <S> --end-frame <E> --frame-stride 1 \
    --max-range 2.0 --voxel 0.005 \
    --out data_processing/out/lt2_seg_naive.ply
```

(`<S>`/`<E>` from `lt2_seg_segment.json`.)

- [ ] **Step 3: Side-by-side render**

Density/ortho comparison png (same hist2d pattern as `lt2_world_compare.png`): naive cloud vs TSDF mesh vertices, top-down + side. Save `data_processing/out/lt2_seg_compare.png`, view it, and judge acceptance criteria 2–3 (no hand blobs; visibly cleaner surfaces).

- [ ] **Step 4: Viewer + share link**

```bash
conda run --no-capture-output -n eyeball python data_processing/ffs_tsdf_viewer.py \
    --prefix data_processing/out/lt2_seg --share
```

Verify in the share URL: clean static mesh, hands animating over it. Send the link + comparison png to the user.

- [ ] **Step 5: Docs + final commit**

Append to `data_processing/CLAUDE.md` branch notes: TSDF task-segment PoC (files, chosen segment, results incl. acceptance verdicts). Commit:

```bash
git add data_processing/CLAUDE.md
git commit -m "CLAUDE.md: TSDF task-segment PoC results"
```

---

## Self-Review Notes

- **Spec coverage:** selection → Task 3 (`pick_segment` + tests); masking → Task 3 (`rasterize_hand_mask` + tests) with hands baked in Task 1; virtual pinhole + TSDF → Task 3 main; baseline → Tasks 2 + 5; viewer/composite → Tasks 4 + 5; acceptance criteria → Task 5 steps 1–4.
- **Type consistency:** hand npz schema `{frames, verts (N,778,3), faces}` used identically in Tasks 1/3/4; `quat_to_R` shared via import in Task 4; pose convention (world→cam, `c=−Rᵀt`) consistent with `ffs_fuse_world.py` and Open3D's extrinsic.
- **Known judgment calls:** painter's-order splat (sort by −Z, later writes win) instead of scatter-min — simpler, correct for depth+color together; `add_mesh_simple` per hand (flat color) rather than trimesh for the animated hands — vertices update in place each frame.
