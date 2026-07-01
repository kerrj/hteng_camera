# data_processing — working notes for Claude

Pipelines for extracting **hand poses** (and later **head poses**) from the
stereo fisheye recordings. Read `README.md` for the user-facing overview; this
file is the operational context.

## Where things run

- **Code** is authored on the laptop (`~/Documents/hteng_camera`) and synced to
  the GPU box via git (`origin/main`). Do coding here, `git push`, then on the
  remote `git pull`.
- **Remote GPU box**: ssh host `chungus` (`bajcsy.ist.berkeley.edu`), repo at
  `~/hteng_camera`, 8× A100-80GB.
- **Conda env**: `eyeball211` (`source ~/miniconda3/etc/profile.d/conda.sh &&
  conda activate eyeball211`). Bleeding-edge: **py3.13, numpy 2.4,
  torch 2.11+cu130, opencv 4.13**. Don't install into it unless necessary; when
  you must, prefer `--no-deps` so torch/numpy aren't disturbed.

## Data: `long-test1/`

Stereo fisheye recording. On chungus at `~/hteng_camera/long-test1/`.
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

## Hand pose: WiLoR-mini — env setup gotchas

Package: [WiLoR-mini](https://github.com/warmshao/WiLoR-mini), `predict()` API,
auto-downloads weights + MANO from HuggingFace. See `install_wilor.sh`.

**We use an EDITABLE clone** at `~/WiLoR-mini` on chungus (base commit
`ebec42f`), `pip install -e . --no-deps`, with `wilor_mini_speedups.patch`
applied — so we can edit the source. If you `git pull` upstream, reapply the
patch (or re-export it).

### Performance — what was slow and why (investigated 2026-06-25)

**TL;DR — ~20x speedup, 3.5 → ~72 fps** (single eye; both eyes ~55 fps, full
7006-frame video in ~126 s). Each win, in order:

| change | fps | why |
|---|---|---|
| baseline (per-frame, `wilor_hands.py`) | 3.5 | GPU at 0%, fully CPU-bound |
| skimage→cv2 blur drop-in | 11.5 | killed full-frame Gaussian (~135 ms/hand) |
| batched ViT (crops across a 64-frame chunk, B=16-32) | — | 16x faster *per crop* than B=1 |
| GPU crop (torchvision `resize`) vs CPU cv2 warp | 21 | 31 → 0.4 ms/frame |
| GPU-tensor YOLO vs CPU numpy/letterbox | **72** | 30 → 2.8 ms/frame |
| + torchcodec GPU decode on the **8-bit** stereo file | (enables above) | NVDEC, both eyes/decode |

Tried and rejected (don't redo): `torch.compile` (recompiles on varying crop
count → slower), torchcodec on the **10-bit** per-eye files (yuv420p10le isn't
NVDEC-fast, ~55 fps vs 217 on 8-bit). Details below.

Raw pipeline ran at **3.5 fps with the GPU at 0%** — fully CPU-bound. Profiled:

| stage | time | notes |
|---|---|---|
| cv2 decode + cvtColor | ~10 ms | NOT the bottleneck |
| YOLO detector | ~27 ms | GPU |
| **skimage `gaussian` (orig)** | **~135 ms/hand** | full-frame blur on the 5MP image, fired on ~96% of hands |
| WiLoR ViT-Huge backbone | ~53 ms | GPU, compute-dominant |
| GPU<->CPU transfers | ~1.7 ms | negligible |

Root cause: WiLoR-mini anti-alias-blurs the **entire 2448×2048 frame** once per
detected hand before cropping (`downsampling_factor > 1.1` → true for our
large/close hands). **Fix in the patch:** swap skimage's `gaussian` for an
equivalent `cv2.GaussianBlur` drop-in (same separable-Gaussian math, ~4× faster
on a 5MP frame). Result: **3.5 → ~11.5 fps** steady-state (full run ~10 min for
7007 frames).

Things tried that did NOT help (don't redo):
- **`torch.compile(backbone)`**: *slower* (2.6 fps). Per-frame crop batch size
  varies (1 vs 2 hands) → constant recompiles. Left as opt-in kwarg
  `compile_backbone=True` (default off); would need `dynamic=True` or fixed-size
  padded batches to pay off. Marginal anyway since backbone is already fp16.
- **torchcodec GPU decode**: *slower* per frame (18 ms vs cv2's 10.6 ms) for our
  random/sequential access; and decode isn't the limiter (cv2 decodes at 94 fps
  vs the 11.5 fps pipeline). Only worth revisiting if we go decode-only or batch
  many frames' crops through the ViT at once.
- **no_grad**: already handled — `predict`/`predict_with_bboxes` carry
  `@torch.no_grad()`.

TF32 (`set_float32_matmul_precision('high')`) is enabled in the patch (free).

### Batched, fully-on-GPU pipeline — `wilor_hands_batched.py` (~20x total)

`wilor_hands.py` (per-frame) tops out at ~11.5 fps. `wilor_hands_batched.py`
reaches **~72 fps single-eye / ~55 fps both-eyes** (full 7006-frame video, both
eyes, in ~126 s) by keeping everything on the GPU:

- **Decode**: torchcodec `VideoDecoder(device="cuda")` on the **8-bit stereo**
  `left_stereo_8bit.mp4` (HEVC 8-bit decodes on NVDEC at ~217 fps; the 10-bit
  per-eye files do NOT — yuv420p10le falls off the fast path, ~55 fps). One
  decode yields both eyes (slice halves on-GPU).
- **Detect**: YOLO on a **GPU tensor batch** (`detect_gpu`). ultralytics
  letterboxes numpy/list inputs on CPU (~30 ms/frame — was the bottleneck);
  feeding a pre-resized GPU tensor (dims multiple of 32, pad 114/255) is
  ~2.8 ms/frame. Boxes come back in letterbox coords → divide by scale.
- **Crop**: `gpu_crop` — batched `grid_sample`/`TF.resize(antialias=True)`
  instead of WiLoR's per-hand full-frame cv2 blur+warp (~0.4 ms vs ~31 ms).
- **ViT**: gather ALL crops across the 64-frame chunk → run at batch 16-32
  (~16x faster per crop than batch 1).

Per-stage (single eye): decode 4.8 + yolo 5.4 + crop 0.4 + vit ~4 ms.

**Accuracy notes / caveats:**
- GPU crop vs CPU patch: <2 mm 3D-keypoint agreement, except extreme close-ups
  (~9 mm; pinhole crops will supersede these anyway).
- GPU-YOLO vs numpy-YOLO: identical on clear hands; differs only on
  marginal/distorted peripheral hands (visually inspected + user-approved).
- Detector sometimes returns 3-5 "hands" (false positives) though there's one
  person — filter to max-2 / higher conf downstream.

Output: `data_processing/out/stereo/{left,right}/hands.jsonl` (same schema).
Detection coverage on long-test1 left: 64% of frames ≥1 hand, 31% both,
median bbox ~262 px (~11% of frame width).

### Viz — `render_hands_video.py`

Renders an annotated overlay mp4 from a `hands.jsonl` + source video (slices a
stereo half with `--eye`). Output uses mp4v; re-encode with ffmpeg
(`~/miniconda3/envs/eyeball211/bin/ffmpeg -i in.mp4 -c:v libx264 -crf 23 ...`)
before transferring — mp4v is ~2.5x larger than h264.

Installed into `eyeball211` (all `--no-deps`): `wilor_mini` (git), `roma`,
`yacs`, `smplx==0.1.28`, `ultralytics`; plus `dill` (auto-installed by
ultralytics to load the YOLO `detector.pt`).

**The chumpy gotcha (important):** WiLoR's `MANO_RIGHT.pkl` contains
`chumpy.Ch` objects, so `pickle.load` needs chumpy importable. chumpy 0.70 is
dead and does NOT import on numpy≥2 / py≥3.11 (removed `np.bool`,
`inspect.getargspec`; also its setup.py shells out to pip under build
isolation → install with `--no-deps --no-build-isolation`). Fix is a **one-time
de-chumpy conversion** (`mano_dechumpy.py`): shim the removed aliases, load the
pkl, convert chumpy→numpy, resave. After that **nothing imports chumpy at
runtime** (verified: pipeline loads `PIPELINE READY` with chumpy uninstalled).
`chumpy` must stay **uninstalled** in `eyeball211` — only install it transiently
if re-running the conversion. No direct `import chumpy` exists in wilor_mini or
smplx (grep-confirmed); the only coupling was the pickle.

`pipe.predict(image_rgb, hand_conf=0.3)` → list per hand. Useful sibling:
`pipe.predict_with_bboxes(image, bboxes, is_rights)` — feed our own boxes
(needed for pinhole-crop + stereo plans). Output keys documented in README.

**WiLoR assumes a pinhole camera.** On raw fisheye it still detects both hands
and keypoints land reasonably (validated on frames 1500/3000/5000), but 2D
keypoints / `cam_t_full` degrade toward the periphery. The fix is rendering
undistorted pinhole crops toward each hand.

## Roadmap (user's stated intentions)

1. **Pinhole-crop refinement** — **renderer DONE** (`fisheye_pinhole.py`),
   keypoint round-trip validated. Per hand: unproject the YOLO bbox-centre pixel
   to a gaze ray, render a virtual pinhole crop aimed at it, run the WiLoR ViT on
   that (de-warped) crop, map crop keypoints back to fisheye px with
   `crop_px_to_fisheye`. **Stereo-rectified**: left/right virtual cams share a
   world orientation with x along the baseline (`baseline_aligned_R`, after
   lab42 eye/stereo.py) so rows are epipolar lines — disparity is purely
   horizontal (validated |dy|~1-5px vs disparity dx~20-50px). Rejected the
   "identity + off-centre cx,cy" rectification: breaks for off-axis hands (KB/tan
   blows up near 90°). **Quality vs raw crop (visual):** pinhole clearly helps on
   peripheral/distorted hands (e.g. fingers stay separated), neutral-to-slightly
   worse on near-central hands (resampling blur; net trained on natural crops).
   GOTCHA: WiLoR flips LEFT hands to look right-handed — flip the crop before the
   ViT, and `postprocess` already un-flips the output. Do NOT also manually
   un-flip the keypoints (double-flip corrupts left hands only).
   **Batched runner DONE**: `wilor_hands_pinhole.py` — detect on left eye, render
   the rectified L+R pinhole pair per hand, batch BOTH eyes through one ViT,
   ~32 fps (2x ViT for stereo; full video ~4 min). Output `hands.jsonl` adds
   `kp_left`/`kp_right` (rectified crop px), `f_px`, `out_size`, `baseline`.
   Quality vs raw crop for *hand pose*: ~the same (user verdict) — the point of
   pinhole is the rectified views for depth, not better keypoints.
2. **Stereo depth for hands** — match hands across left/right and recover metric
   depth. User's preferred approach: a **mesh/pose optimization** that optimizes
   the hand's 3D position (and pose) to fit *both* sets of 2D keypoints given
   the known stereo disparity/extrinsics — not naive triangulation. The
   rectified pinhole pair from step 1 (`render_stereo_crop` returns both crops +
   `Rv_l`/`Rv_r`/`f_px`/`g` geometry) is the input. **Not yet built.**
3. **Head pose** — separate pipeline, after hands.

## Conventions

- Keep all scripts/files for this work under `data_processing/`.
- Record any new package added to `eyeball211` in `README.md` + here.

### Stereo MANO optimization (step 2) — status + key findings (2026-06-25)

Built: `mano_jax.py` (JAX MANO LBS forward, validated **0.035 mm** vs torch
smplx), `stereo_optimize.py` (full-track jaxls bundle adjustment),
`stereo_optimize_oneframe.py` (single-frame before/after viz). jaxls works on
CPU (jax 0.9.2); toy 2000-var temporal solve in ~1.2 s.

**CAMERA GOTCHA (important):** WiLoR's `pred_keypoints_2d` are produced by
projecting its 3D joints with a *weak-perspective* focal `scaled_focal_length =
FOCAL_LENGTH(5000)/IMAGE_SIZE * img_size.max()`. When we postprocess in CROP
coords (img_size=256) that focal is **5000**, NOT the crop's true pinhole
`f_px` (~550). The 2D *pixel positions* are still valid in the crop, but the
implied depth/scale is WiLoR's monocular guess — do not mix WiLoR's
`pred_cam_t` depth with the true-`f_px` geometry. For the optimizer, project the
metric MANO hand with the TRUE `f_px` and fit WiLoR's 2D pixels.

**Diagnosis of poor one-frame fits — it's detection quality, not geometry.**
Verified per-frame by the epipolar disagreement of WiLoR's own keypoints
(|dy| between the two rectified crops; images are rectified so |dy| should be
~0):
  - frame 3000: |dy| ~3-7 px → depths 0.48/0.56 m (clean, sensible).
  - ORB *image-feature* disparity is always clean (|dy|~7px → 0.89 m), proving
    the rectification/rendering is correct.
  - frames 4850/5500/6750: WiLoR keypoint |dy| = 32-53 px → nonsense depths
    (44/80/99 m). Visual: one eye's crop is well-framed but WiLoR poses it
    collapsed/wrong. Pure monocular ViT pose noise on individual crops.
So the stereo+temporal+Huber optimization is exactly the fix: a single 3D hand
can't reproject as the bad-eye mess, so the joint fit (dominated by the good eye)
+ temporal smoothing should reject these. Plan: validate optimizer on a
good-agreement window (~frame 3000) first; use left/right (or
distance-to-temporal) disagreement as a robust per-frame/per-kp weight.

### Stereo opt — SOLVED (2026-06-25 cont'd)

Single-frame 2-view fit was clean (2px) but the multi-frame solve blew up to
NaN / 25 m depths. **Cause: outlier detections, not the solver.** Bad per-eye
WiLoR poses create huge reprojection residuals that dominate the LS cost.

**Fix (in `stereo_optimize.py`):** the crops are stereo-RECTIFIED, so rows are
epipolar lines — gate each keypoint by `|y_left - y_right|` (`--dy-thresh`,
default 8px); drop a whole hand if `< --min-inliers` (default 12) survive; init
triangulation over inliers only; Huber on the rest. Result on frames 2850-3150
(right hand): 111/256 frames kept (145 dropped, 55% of kps masked — these
detections are genuinely noisy), **inlier reproj ~2px, depth median 0.543 m
[0.47,0.66], median frame-to-frame jump 16.6 mm**.

**Solver:** jaxls `conjugate_gradient` (matrix-free block-sparse, the intended
solver for large problems — see jaxls sparse-matrices design doc). `dense_cholesky`
only for tiny windows; **do NOT pursue `cholmod`/scikit-sparse** — CG handles the
sparsity natively and the sksparse install rabbit hole gutted the env once
(see [[eyeball211-no-conda]]). LM trust region required (plain Gauss-Newton
diverges to NaN). It IS one joint factor-graph solve over all frames (reproj
into both eyes + temporal coupling), which is the efficient design.

TODO: firmer temporal weight to suppress the residual ~276mm spikes; run full
video; lift joints_3d_cam back to a world/head frame; then head pose.

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

### Before/after + outlier viz — `render_stereo_compare.py`

2x2 panel video (L/R eyes x BEFORE/AFTER). BEFORE = WiLoR raw kp (epipolar
outliers, |dy|>thresh, drawn RED); AFTER = optimized 3D MANO reprojected into
both eyes (consistent by construction) with depth label; "REJECTED (n inl)"
banner on dropped frames. Reads pinhole hands.jsonl (geometry+kp) + stereo3d
hands3d.jsonl (optimized joints_3d_cam); re-renders crops from video.
First-30s results: LEFT hand 276/432 kept @0.67m, RIGHT 173/563 kept @0.53m
(right is noisier). Outlier rejection visibly correct — motion-blurred/oof
hands drop to ~6 inliers and are rejected. NOTE the optimizer can run two hands
in one shell but chaining two `nohup ... &` after one `conda activate` fails for
the 2nd (subshell loses activation) — launch separately.

### Stereo opt — conditioning + speed fixes (2026-06-29)

**Normalization was the key stability fix.** Reproj residuals in raw pixels
(O(10-100)) made total cost O(1e6) and badly conditioned vs the radian-scale
pose prior; CG wandered to 38m / 2991-rad garbage. **Fix: divide reproj error by
max(w,h)=out_size** → image-fraction residual O(0.01-0.1), cost O(1). Now the
baseline (NO prior, NO temporal) is stable: reproj inliers p50=2.9px, depth
median 0.52m range[0.24,0.74], depth jump median 15mm. |Δpose| from WiLoR ~3.9
rad (a light pose prior should tame this).

**Beta frozen** to WiLoR (not a variable) — removed 10 vars/frame + beta prior.

**Speed:** MANO forward skinned all 778 verts but uses 5 fingertips (+16 tree
joints, no skinning needed). Now skins only the 5 tips + precomputes the
rest-joint affine in betas → no 778-vertex op. jacfwd ~2x faster than jacrev on
the MANO chain → set `@jaxls.Cost.factory(jac_mode="forward")`. Early LM steps
~0.15s; later steps slow (~2.5s) only because Eisenstat-Walker CG tightens its
inner tolerance near convergence — NOT recompilation (verified). Converges in a
few steps anyway, so high --iters is unnecessary.

Reference: brentyi/egoallo uses jaxls for SMPL-H/MANO exactly like this — FK-only
(never LBS) inside costs, CG solver, one batched Var per timestep with
jnp.arange(T) ids, betas mean-pooled & fixed, `max_iters` as Static[int] so LM
unrolls. Temporal smoothness there is SO3-log of relative quats (1st + 2nd order).
