# data_processing/vio — working notes for Claude

Head/camera pose pipeline: offline stereo-inertial bundle adjustment, in the
same spirit as `../hands/`'s stereo MANO optimizer (jaxls, CG solver, Huber
robust loss) but for static-scene landmarks + the IMU instead of MANO joints.
Not realtime — batch/offline is fine, "fast" just means it shouldn't take
forever to run over a multi-minute recording.

See `../CLAUDE.md` for the shared recording/calibration description.

## Where things run

- **Remote GPU box**: ssh host `sphynx`, repo at `~/hteng_camera`, 8x RTX A6000.
  Different box from the hand pipeline's `chungus`.
- **Conda env**: `jaxgpu` — has a working jax GPU install already; added
  `torch==2.12.1+cu130` (matches the driver's CUDA 13.0 — use the `cu130`
  wheel index, NOT `cu126`) and `lightglue` (`pip install
  git+https://github.com/cvg/LightGlue.git`, installs clean, no custom CUDA
  extension build) into this same env.
- **Run etiquette**: pin jobs to `CUDA_VISIBLE_DEVICES=2` (the free GPU on
  that box). Prototype with a handful of frames before scaling to a full
  video — full runs are slow to iterate on.

## Design decisions (converged 2026-07-01)

- **Separate from hand fitting, composed after.** Hands are a moving
  foreground target — bad SLAM landmarks. Camera pose needs static background
  features. Solve independently, then `hand_world(t) = T_cam_world(t) @
  hand_cam(t)`.
- **No DROID-SLAM / heavyweight learned SLAM.** Rejected specifically for
  install friction (custom CUDA `lietorch` build) clashing with this repo's
  clean-install ethos — not because it'd be less accurate.
- **Stereo extrinsics: fixed transform, not a solved Var.** Left/right are
  rigidly mounted with a well-characterized calibration, so
  `T_right(t) = T_stereo @ T_left(t)` is computed deterministically, not a
  free pose + strong prior factor. Halves the pose variable count. (Rig-flex
  correction via a soft prior is a possible future addition if evidence
  warrants it — no evidence yet.)
- **IMU factor: LOCAL relative rotation between consecutive frames, not a
  global orientation prior.** Integrate gyro (plain strapdown quaternion
  integration — a single inter-frame window is ~33ms, way too short for
  gyro drift to matter, so no Madgwick/AHRS fusion needed for this) from frame
  A's timestamp to frame B's, and add it as a relative-pose factor between
  those two poses — same factor *shape* as the hand optimizer's temporal
  smoothness term (SO3-log of relative quats), just with the "expected"
  relative rotation coming from the IMU instead of a zero-motion assumption.
  Rationale: avoids ever needing a stable *global* attitude reference (no
  shared-reference yaw drift leaking into every frame). This is, not
  coincidentally, the rotation-only component of "real" IMU preintegration
  (Forster et al.) — the simplification is principled, not a shortcut.
  Gyro bias is treated as constant (from `calibrate_imu()`) for v1; only add
  an online-estimated slowly-varying bias Var later if residuals show a
  systematic drift trend.
- **Front end: LightGlue (SuperPoint) for both stereo AND temporal matching**,
  not classical KLT. Chosen for robustness to motion blur / fisheye periphery
  distortion, NOT speed — classical sparse KLT is not actually a compute
  bottleneck (a fundamentally cheap sparse op, unlike the dense per-hand blur
  bug that tanked WiLoR — see `../hands/CLAUDE.md`). Measured on `long-test1`
  (sphynx, RTX A6000, 512 max keypoints): ~38ms/image extraction, ~13-20ms/pair
  matching, both stereo and temporal (sliding window). Cost is linear in video
  length (only ~window_size new pairs per new frame), not quadratic.
- **FOV mask (~130° kept of the 180° fisheye) computed via the calibration
  model, not a raw pixel radius** — Kannala-Brandt `r(θ)` is a distortion
  polynomial, not linear in angle. Reuses `fisheye_unproject` from
  `../fisheye_pinhole.py` (unproject each keypoint to a ray, threshold its
  angle from the optical axis) rather than re-deriving the polynomial.
- **Crop to the FOV mask's bounding box BEFORE running SuperPoint**, not
  after. SuperPoint always resizes its input's long side to a fixed budget
  (1024 by default); cropping out the periphery we're going to mask away
  anyway means more of that resize budget is spent on pixels we actually
  keep. Measured on `long-test1`: 130° mask bbox is ~1785x1785 vs native
  2448x2048 → ~1.37x more effective px/degree in the kept region, for free
  (the crop is also strictly less compute to resize). Verified pixel-correct:
  crop-local keypoints are offset back by (x0,y0) before the FOV mask/storage,
  confirmed by overlaying stage-1 output on the full uncropped frame.
- **Outlier rejection: gate hard, then Huber the rest** — same lesson already
  learned the hard way in the hand optimizer (`../hands/CLAUDE.md`'s "Stereo
  opt — SOLVED" section: plain Gauss-Newton diverged to NaN when bad
  detections dominated; Huber alone wasn't sufficient without a hard
  pre-filter). Both gates work on UNPROJECTED RAYS (fisheye epipolar lines are
  curves in pixel space, not rows — no rectified-rig shortcut applies here).
  Stereo pairs: gated against the KNOWN stereo R,t via a coplanarity residual
  (essential-matrix constraint on ray directions) — no RANSAC needed since the
  geometry is calibrated, not estimated. Temporal pairs: RANSAC essential-matrix
  gate on normalized ray bearings (cv2.findEssentialMat, cameraMatrix=I) — no
  known relative pose between two arbitrary frames. Then track-level
  conflict rejection at stage 3 (see below). Huber loss in the final LM solve
  handles only the residual noise that survives those gates.
- **Track building needs CONFLICT-AWARE union-find, not naive connected
  components** — found by direct testing, not anticipated: a naive
  transitive-closure merge over stage-2's matches produced a "track" with 622
  observations against a theoretical ceiling of 60 (30 frames x 2 eyes).
  Repetitive texture (keyboard keys, monitor text lines) lets a handful of
  spurious matches bridge genuinely different physical points into one
  component; once bridged, transitivity silently pulls in everything
  reachable through it. Fix (`vio_build_tracks.py`): reject any union that
  would place two different keypoints from the same (eye, frame) into one
  track, processing edges most-trustworthy-first (stereo, then temporal by
  ascending gap) so a bad edge gets rejected against an already-good
  component rather than cementing the merge. Re-verified after the fix: max
  track length now exactly matches the frames*eyes ceiling, zero duplicate
  (eye,frame) observations in any track.

## Pipeline stages

Mirrors the hand pipeline's `hands.jsonl → hands3d.jsonl` staged-script style
— each stage independently re-runnable/cacheable, under
`data_processing/vio/out/<recording_name>/` (mirrors `../hands/out/...`'s
actual convention, NOT the `data_processing/out/vio/...` this doc originally
sketched before any code existed).

| Stage | Script | Input | Output | Notes |
|---|---|---|---|---|
| 1. Feature extraction | `vio_extract_features.py` | left/right mp4 | `features.h5` | HDF5, per-eye group with `keypoints`/`scores`/`descriptors` (h5py `vlen` dtype, ragged per frame after FOV mask) + `counts`. Crops to the FOV bbox before extraction (see above), FOV-masked (in full-frame coords) after. Cached separately from matches — extraction is the GPU-heavy, non-reprocessable-cheaply step; matching is cheap so pair sets can change without re-extracting. |
| 2. Pairwise matching + gating | `vio_match_pairs.py` | `features.h5` | `matches.jsonl` | One line per attempted pair (frame ids, eyes, matched index pairs + rejected pairs, geometric-gate pass/fail counts). Temporal gap schedule: dense nearby, sparser further out (`build_temporal_gaps`) — default `[1,2,3,4,6,8,10,15,20,25,30,40,50,60]`, i.e. up to 2s @ 30fps. GOTCHA hit in the full-video run: left/right mp4s can have slightly different frame counts (long-test1: 7007 vs 7006) — bound the frame loop by `min()` of both eyes, not just the left eye's count, or the last frame(s) crash with an out-of-range H5 read. NOT currently batched across pairs — one LightGlue forward call per pair (see Optimization TODO below). |
| 3. Track building | `vio_build_tracks.py` | `matches.jsonl` | `tracks.jsonl` | Conflict-aware union-find (see above) → one line per landmark (list of `{frame, eye, kp_idx, px}` observations). COLMAP-style landmark DB, sorted longest-track-first. 3D triangulation deliberately deferred to stage 5 (needs the triangulation logic anyway as its LM init step). |
| 4. IMU relative factors | `vio_imu_prior.py` | `imu_log.csv`, `sync_log.csv` | `imu_relative.npz` | NOT YET BUILT. Per consecutive-frame-pair relative rotation (strapdown gyro integration between the two frames' soft-synced timestamps), NOT a global orientation stream. |
| 5. Global BA | `vio_bundle_adjust.py` | `tracks.jsonl`, `imu_relative.npz`, stereo calib | `trajectory.npz` | NOT YET BUILT. jaxls: SE3 pose Var per left-frame (right pose derived deterministically from the fixed stereo transform), xyz Var per landmark, reprojection factors, IMU relative-rotation factors, Huber loss. |

Visualizers exist alongside each stage (`vio_visualize_features.py`,
`vio_visualize_matches.py`, `vio_visualize_tracks.py`) — same pattern as the
hand pipeline's render scripts. Tracks are rendered as colored "comet" dots
(golden-ratio hue spacing per track) with fading trajectory tails, capped to
the longest N tracks to stay legible over a multi-second clip.

## Optimization TODO (deliberately deferred until the full-video pass works end to end)

- **Batch LightGlue matching across pairs, not one pair at a time.** Confirmed
  `LightGlue.forward` supports batched `[B x M x 2]` keypoints/descriptors —
  we're just not using it (`vio_match_pairs.py` calls the matcher once per
  pair, batch=1 every time). Per-pair overhead (~15-20ms) is likely dominated
  by Python/CUDA-launch overhead and underused GPU parallelism, not actual
  attention compute — same shape of win as WiLoR's batched-ViT speedup in
  `../hands/CLAUDE.md` (16x/crop). Needs padding each frame's keypoints to a
  common per-batch max count + a validity mask (LightGlue supports this as a
  tensor shape; the padding/masking logic itself doesn't exist yet in
  `vio_extract_features.py`'s storage or the stage-2 pair loop). Decided
  2026-07-01 to debug/validate correctness on a full-video run FIRST, then
  optimize — don't want to debug a new batching path and pipeline correctness
  at the same time.

## Status (2026-07-01)

Stages 1-3 built and validated on `long-test1`. Full-video run (all ~7007
frames) in progress on sphynx — stage 1 complete, stage 2 hit the
frame-count-mismatch bug above partway through (~99% done) and was resumed
from stage 2 after the fix (stage 1's `features.h5` didn't need re-running).
Stages 4 (IMU) and 5 (BA) not yet built. Motion blur robustness still
untested on a real head-motion segment (`testnewcamsblur*` recordings exist
for this) — `long-test1`'s tested portion so far is a mostly-static desk
scene.
