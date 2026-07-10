# data_processing — working notes for Claude

Two parallel, independent pipelines over the same stereo fisheye + IMU
recordings:

- `hands/` — per-frame hand pose (WiLoR + stereo MANO bundle adjustment).
  See `hands/CLAUDE.md`.
- `vio/` — head/camera pose (stereo-inertial bundle adjustment). See
  `vio/CLAUDE.md`.

They're kept separate deliberately: hands are a moving foreground target and
bad SLAM landmarks, while VIO needs static background features. The two are
composed *after* both solve independently: `hand_world(t) = T_cam_world(t) @
hand_cam(t)`.

`fisheye_pinhole.py` (Kannala-Brandt unproject/reproject math) and
`test_projection_math.py`'s sibling live at this top level because both
pipelines import it — everything else is domain-specific and lives in the
subfolder it belongs to.

`visualize_data.py` is the shared viser viewer: VIO camera trajectory +
landmark point cloud (from `vio/vio_bundle_adjust.py`'s `trajectory.npz`),
optionally composed with hand meshes FK'd from `hands/stereo_optimize.py`'s
stereo3d jsonl output (`hand_world(t) = T_cam_world(t) @ hand_cam(t)`, same
composition rule as above) via `--hands-left`/`--hands-right`. Hands are
skipped automatically if those args aren't passed. Lives at this top level
(not under `vio/`) since it now spans both pipelines.

## Data: `long-test1/`

Stereo fisheye recording, at the repo root (`~/hteng_camera/long-test1/` on
remote boxes).
- `left.mp4` / `right.mp4`: 2448×2048, 30 fps, ~7007 frames (~3.9 min).
- OpenCV **fisheye** (Kannala-Brandt, 4 dist coeffs), fx≈775, baseline **70.8 mm**.
- Calib: `calib_<serial>.json` (intrinsics are FULL-SENSOR; shift cx/cy by ROI
  offset — here ROI = full sensor so no shift), `stereo_<L>_<R>.json` gives
  `X_right = R @ X_left + t` (t in metres). left serial `046060323008`,
  right `046060323001`.
- Intrinsics/stereo loaders live in `src/hteng_camera/calibration.py`.
  `Intrinsics.undistort_maps(w, h, balance)` builds fisheye→pinhole remap
  tables (centred principal point, `balance` zooms out FOV) — this is the
  building block for pinhole-crop rendering.
- Newer recordings (e.g. `test42/`) also have `imu_log.csv` (100Hz raw
  accel/gyro/mag, see `src/hteng_camera/imu.py`) and a soft camera↔IMU time
  sync recorded in `recording.json` — see that file's `imu.clock_alignment`
  field for the exact alignment procedure/accuracy caveats.

## Conventions

- Code is authored on the laptop and synced to remote GPU boxes via git.
- Each subpipeline (`hands/`, `vio/`) has its own remote GPU box + conda env —
  see that subfolder's CLAUDE.md for which one and its exact deps. Don't
  assume they share an environment.
- Scripts in `hands/`/`vio/` that need `fisheye_pinhole.py` reach it via a
  `sys.path.insert` bootstrap to this directory (not a package-relative
  import) — see the file-layout discussion in conversation history if this
  ever needs revisiting; the short version is that `data_processing/` is
  deliberately outside the installable `hteng_camera` package, so it can't be
  a normal dotted subpackage import without pulling heavy research deps into
  the pip-installable driver.

## Scene depth via Fast-FoundationStereo (branch `depth_understanding`, 2026-06-30)

**New machine reality:** this work runs on host **sphynx** (2x RTX A6000-48GB,
system CUDA 13.0, driver 580), conda env **`eyeball`** — NOT chungus/eyeball211.
`eyeball` = py3.10, **torch 2.10.0+cu126**, numpy 2.2.6, opencv-python 4.13,
triton 3.6, timm/einops/omegaconf present. `long-test1/` was copied here to
`~/hteng_camera/long-test1/`.

**Goal:** wide-FOV *metric scene* depth (not just hands). A learned *monocular*
model on raw frames can't do it (scale+shift ambiguity → warped cloud; fisheye
is out-of-domain; per-frame scale flicker). The metric anchor must be the
calibrated stereo baseline (70.8 mm). Plan: render a **grid of baseline-aligned
PARALLEL pinhole stereo pairs** (tangent planes tiling the fisheye FOV; reuse
`fisheye_pinhole.baseline_aligned_R` + `Intrinsics.undistort_maps`), run
FoundationStereo per tile, back-project, fuse in the left-camera frame.
Key geometry caveat: baseline is fixed at the physical centres — per look dir the
usable baseline is the component ⟂ the optical axis, so tiles looking along ±x
(toward the epipole) degrade to zero baseline (no depth there). Use the *parallel*
(shared-orientation) rectification per tile, not the *verged* `render_stereo_crop`.

**Fast-FoundationStereo install (DONE, `install_fast_foundationstereo.sh`):**
- Vendored as a **pinned git submodule**: `data_processing/third_party/Fast-FoundationStereo`
  @ `a290ba0` (master, 2026-05-26). It is clone-and-run (no setup.py); add its
  root to `sys.path` and import `core` / `Utils`. Keep the submodule pristine —
  weights live OUTSIDE it.
- Two deps added to `eyeball` (constraints-pinned so torch/numpy didn't move):
  **scikit-image 0.25.2**, **xformers 0.0.35** (the cu126 wheel matches torch
  2.10). Deliberately **skipped opencv-contrib-python** (no aruco/contrib use;
  avoids a 2nd cv2). Everything else FFS needs was already present.
- Weights (~900 MB, gitignored) in `data_processing/weights/`: checkpoints
  `23-36-37` (most accurate) · **`20-26-39` (balanced, our default)** ·
  `20-30-48` (fastest) · `15-44-51`, each `cfg.yaml` + `model_best_bp2_serialize.pth`,
  plus `onnx/` exports. From the repo's Drive folder via `gdown --folder`.
- Fast path: `model.forward(l, r, iters=8, test_mode=True, optimize_build_volume='pytorch1')`
  (Triton GWC kernel — JITs at runtime, needs NO nvcc/TRT build). Inputs are
  `InputPadder(divis_by=32)`-padded float CHW tensors; depth = `fx*baseline/disp`.
- **Smoke test (demo pair, 540x960, A6000):** loads in ~2.6s, forward 6.7s first
  call (Triton compile) then **~70 ms warm**, **1.1 GB** GPU; clean disparity
  (med 95.5px → depth med 0.498 m). Verified visually (desk/keyboard/mug scene).
- GOTCHA: `scripts/run_demo.py` blocks headless — `cv2.imshow`+`waitKey(0)` (line
  106) and an open3d Visualizer window (line ~139). Use the no-GUI forward (see
  the install script's verify block) for batch work.
- The *original* FoundationStereo (CVPR'25) is ALSO pip-installed in `eyeball`
  (`foundation-stereo 1.0.0`, from the kushtimusPrime fork, module
  `foundation_stereo`). Different package, no collision with the Fast submodule;
  leave it unless it gets in the way.

### Scene-depth mosaic — BUILT + validated (`ffs_scene_depth.py`, 2026-07-01)

Single-frame wide-FOV metric depth via the tangent-plane mosaic. Renders a
VERTICAL fan of baseline-aligned PARALLEL pinhole tiles (pitched about the
baseline axis so each stays ⟂ baseline → full 70.8mm baseline, clean horizontal
disparity), runs FFS per tile, back-projects `depth = fx*baseline/disp`, fuses in
the left-cam frame. Reuses `fisheye_pinhole.sample_fisheye` + `baseline_aligned_R`
(the NON-verged sibling of `render_stereo_crop`: right cam = `Rs @ Rv_l`, one
shared world orientation). Horizontal coverage is the tile WIDTH (off-centre
columns = yawed rays), NOT yaw-tiling; we do not yaw (breaks rectification).

Defaults: 5 pitch tiles [-60,-30,0,30,60]°, each 960×384, hfov 100° / vfov 52°,
fx≈403, max_disp 192, weights 20-26-39. Frame decode = cv2 on
`left_stereo_8bit.mp4` (side-by-side; left half=left eye=serial ...008).
Outputs (to `data_processing/out/`, gitignored): `_cloud.ply` (fused, left-cam
metric), `_range.npy`+`_depth.png` (reprojected to left fisheye), `_tiles.png`
(--debug: per-tile color|disp montage).

**Validation (frame 3000, egocentric kitchen scene):** per-tile FFS depth is
clean/sharp on real fisheye (container edges, arms, sink, TV all resolve; smooth
surfaces). Depth medians track pitch sensibly: down −30°→0.61m (hands/counter),
fwd→0.82m, up +30/+60°→2.2/2.9m (room). Fused 1.78M pts, depth p5/50/95 =
0.40/2.05/4.64m. Top-down/side ortho shows coherent room geometry, no seam
doubling. ~1–2s/frame on one A6000.

**Confirmed via `motion_probe.py`:** the rig TRANSLATES (essential>homography,
parallax 0.5–1.3° at low-rotation windows, up to ~4°/s) AND rotates hard/fast
(up to 54°/0.3s, motion blur likely). So temporal fill (phase 2) is viable and
its vertical/forward baselines can fill the horizontal epipole cones — but needs
VO/pose + calm-window selection.

**Known gaps / next (in priority):** (1) fusion is naive z-buffer-nearest — add
confidence-weighted blending (downweight tile edges, near-epipole yaw, low
texture); (2) mask blown-out windows / textureless / far-field (baseline-limited)
noise — biggest visible-quality win; (3) fisheye reprojection is sparse/speckled
(tiles lower-res than the 2448px fisheye) → render clean per-pixel map via
higher-res tiles or radius/mesh splat; (4) coverage 29.7% is part artifact
(sparse splat) + genuine epipole cones; (5) only 1 frame tested — try a calm and
a blurry frame. THEN phase 2 (temporal) / phase 3 (mono fill).


## depth_understanding branch — MANO surface + world-fusion plan (2026-07-05)

- Scene player (`ffs_scene_player.py`) renders the real 778-vertex **MANO
  surface** now: `extract_mano.py` rebuilds the mano npz from `MANO_RIGHT.pkl`;
  `precompute_hand_meshes.py` bakes per-frame verts (from `viz_*.jsonl`) so the
  player stays jax-free; rendering uses PERSISTENT viser handles updated in
  place (after `viz_hands.py`) — the earlier per-frame re-upload crashed the tab.
- Merged `origin/main` (local only, `main` untouched): brought the `vio/`
  camera-pose pipeline + the `hands/`/`vio/` reorg + `<recording>/derived/` outputs.
- NEXT: run `vio/` to get `T_cam_world(t)`, then fuse per-frame scene depth +
  hands into ONE world cloud (`hand_world(t) = T_cam_world(t) @ hand_cam(t)`).
  Newer recording `long-test2/` has a full `derived/` pipeline already.
