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
