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
  extension build) into this same env. Also `torchcodec` (`pip install
  torchcodec --index-url .../cu130`) + `ffmpeg>=7` from conda-forge (`conda
  install -c conda-forge 'ffmpeg>=7'` — the system ffmpeg was too old and
  lacked NVDEC; verify with `ffmpeg -decoders | grep nvidia`) for stage-1 GPU
  video decode.
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
— each stage independently re-runnable/cacheable. **All products default into
`<recording>/derived/`** (next to the raw `left.mp4`/`imu_log.csv`/`calib_*.json`
inputs, which stay in the recording root) so there's no separate output-tree
bookkeeping — just point every stage at the recording dir and omit `--out`.
Each stage's `--out`/`--features`/`--tracks`/… still override individually.
(Older runs used `data_processing/vio/out/<recording_name>/`; the `derived/`
default replaced it 2026-07-03 — the hand pipeline uses the same convention.)

| Stage | Script | Input | Output | Notes |
|---|---|---|---|---|
| 1. Feature extraction | `vio_extract_features.py` | left/right mp4 | `features.h5` | HDF5, per-eye group with `keypoints`/`scores`/`descriptors` (h5py `vlen` dtype, ragged per frame after FOV mask) + `counts`. Crops to the FOV bbox before extraction (see above), FOV-masked (in full-frame coords) after. Defaults: `--fov-deg 150`, `--max-keypoints 1024`, `--detection-threshold 0.001` (drops the weakest, jitteriest detections), `--resize 1500` (SuperPoint's long-side resize target; the FOV crop is ~2000px so this recovers real resolution over the old 1024). BATCHED (`--batch-size 16`) through `SuperPoint.forward` — crops are same-size so the conv backbone batches cleanly; per-frame fallback if a batch has uneven keypoint counts. **GPU decode: `--decoder torchcodec` (default) decodes via NVDEC + crops/normalizes on-device** — no CPU decode/upload, 2.9ms/frame vs cv2's 24ms (was ~43% of stage-1 wall → ~8%). `--amp` adds fp16 autocast on the forward (~1.5x). Combined ~2.3x. **Caveat: NVDEC ≠ cv2 pixels** (decoder rounding), so torchcodec features differ slightly from a cv2-decoded run — equivalent quality (desc cosine ~0.999) but NOT byte-identical; use `--decoder cv2` for bit-exact reproduction. Needs `ffmpeg>=7` with NVDEC in the conda env (see "Where things run"). Cached separately from matches. |
| 2. Pairwise matching + gating | `vio_match_pairs.py` | `features.h5` | `matches.jsonl` | One line per attempted pair (frame ids, eyes, matched index pairs + rejected pairs, geometric-gate pass/fail counts). Temporal gap schedule: dense nearby, sparser further out (`build_temporal_gaps`) — default dense-to-20 then thinning to a 70-frame tail: `[1..20,21,23,25,27,29,30,35,40,50,60,70]`. Dense-to-20 (vs the old dense-to-4) fixes track flicker (a track dropping out for a frame when its only scheduled gaps happen to fail the gate). GOTCHA: left/right mp4s can differ by a frame (long-test1: 7007 vs 7006) — bound the loop by `min()` of both eyes. BATCHED across pairs (`--batch-size`, LightGlue), with a **GPU-resident FeatureStore** (features uploaded once/frame, fp16 descriptors, top-k selection + batch stacking all on-device) — killed the per-batch numpy-stack + fp16→fp32 + H2D that was 67.6% of stage-2 wall (profiled), for a **3.1x** end-to-end speedup; stage 2 is now GPU-bound (LightGlue forward ~62%, temporal RANSAC ~21%). JSONL writes are on a background thread (`AsyncJsonlWriter`). `--max-keypoints` matters (O(N²) attention). NOTE: the fp16-native store makes runs no longer bit-reproducible run-to-run (fp16 changes LightGlue's match *emission order* → reseeds RANSAC), but raw matches are identical and accept/reject flips are ~0.03% (RANSAC noise). |
| 3. Track building | `vio_build_tracks.py` | `matches.jsonl` (+ optional `loop_matches.jsonl`) | `tracks.jsonl` | Conflict-aware union-find (see above) → one line per landmark (list of `{frame, eye, kp_idx, px}` observations). COLMAP-style landmark DB, sorted longest-track-first. 3D triangulation deliberately deferred to stage 5. `--min-frames 15` (default) drops tracks seen in fewer than 15 DISTINCT frames (eye-independent) — keeps only stable, well-tracked landmarks; distinct from `--min-observations` which counts per-eye observations. `--matches` takes MULTIPLE files (concatenated) — enabling loop closure is just adding `loop_matches.jsonl` as a 2nd `--matches` arg, no cat/rebuild step. |
| 3b. Loop closure (OPTIONAL) | `vio_loop_closure.py` | `features.h5` | `loop_matches.jsonl` | Exhaustive keyframe (stride 30, ~1/s) matching for pairs beyond stage 2's gap horizon (>60 frames), same LightGlue+RANSAC gate as stage 2, records with `pair_type="temporal"` + true gap so they concatenate straight into stage 3. **Off by default** — the stage-5 two-stage solver alone collapsed the ghost on testimu (see below), so this is only for large loops with real inter-visit drift. `--viz-out` writes a montage of accepted pairs to eyeball before trusting them. |
| 4. IMU relative factors | `vio_imu_prior.py` | `imu_log.csv`, `sync_log.csv`, `recording.json` | `imu_relative.npz` | Per consecutive-frame-pair relative rotation (strapdown gyro integration between soft-synced timestamps, NOT a global orientation stream) plus a per-frame gravity direction + confidence weight (windowed raw-accel average, weighted by deviation from 1g). IMU→camera extrinsic is a FIXED CAD-derived rotation constant (`R_CI`), not calibrated. Also flags invalid (bad-timestamp) frames generically via non-monotonicity, not a hardcoded index. |
| 5. Global BA | `vio_bundle_adjust.py` | `tracks.jsonl`, `imu_relative.npz`, stereo calib, `features.h5` | `trajectory.npz` | **Two-stage GLOMAP-faithful solve** (see below). jaxls: per-observation scale Var (GLOMAP's `d_ik`), xyz Var per landmark; GLOMAP bounded positioning cost (Pan et al. 2024 / BATA), not reprojection error. Cauchy robust loss (scale 0.05, on the bounded sin-θ residual) down-weights outliers in-solve — no filter-and-resolve rounds. Solver: jaxls trust-region LM (Gauss-Newton diverges). World is **gravity-aligned** (+z up, fixed by the gravity prior); global yaw + translation origin are free gauges, recentered post-hoc so cam0 = origin (no in-solve anchor cost — GLOMAP avoids it too). Frames with invalid timestamps dropped from the pose list. |

### Stage 5: two-stage solve (GLOMAP-faithful, converged 2026-07-03)

Mirrors GLOMAP's actual pipeline (rotation averaging → global positioning with
rotations FROZEN → full BA), which fixed a catastrophic under-rotation bug and
made the ghost-cluster loop-closure workaround largely unnecessary.

- **Stage 1 — frozen-rotation global positioning.** Rotations come from
  integrating the IMU relative-rotation chain (our rotation-averaging analog,
  seeded gravity-aligned) and enter the residuals as CONSTANTS — the variables
  are camera CENTERS (`CamCenterVar`, 3-DOF), landmarks, and scales. This is
  exactly GLOMAP's `global_positioning.cc` (rotations are baked into the
  observed directions, never parameter blocks). Well-conditioned and fast:
  ~2s/iter, flat by ~iter 8 (default 15).
- **Stage 2 — full SE3 refine.** Rotations now free (SE3Var), tethered by the
  IMU relative-rotation + gravity costs. Starts already in the basin from stage
  1, so it only polishes — flat by ~iter 2 (default 3). ~15× more expensive per
  iter than stage 1, so it dominates wall-clock; keep the count small.
- **Why the staging matters (the bug it fixed):** the old single joint solve
  from random init let early garbage-geometry gradients destroy a correct
  rotation init — a physical 180° turn (frame 424, IMU-confirmed) rendered as
  ~19°. A `--translation-smoothness-weight` position prior made it worse
  (couples into rotation via c = −Rᵀt). Freezing rotations during positioning
  makes them a scaffold the positions organize around; by refine time every
  gradient is computed from near-correct geometry. Two-stage output now tracks
  the IMU rotation truth to ~1° at every checkpoint. `positioning_cost` is one
  factory shared by both stages via jaxls's var-or-constant factory args (a Var
  arg is optimized, a plain array is baked in) — no duplicated residual math.
- **Gauge — no anchor cost.** GLOMAP leaves the translation gauge free (explicit
  "do not set any camera constant for easier convergence" comment) and
  recenters post-hoc. We do the same: 3-DOF translation null space handled by LM
  damping, then recenter cam0→origin at the end (no rescale — the stereo
  baseline fixes metric scale). The old weight-1000 SE3 anchor prior both
  ill-conditioned CG (~2× slowdown) and was gauge-redundant. Roll/pitch fixed by
  the gravity prior; global yaw an intentional free gauge.
- **Loop closure now optional (validated 2026-07-03 on testimu).** With the
  two-stage fix, loop vs no-loop solutions differ by median 23mm cam position
  (0.44% of trajectory extent) and 2.2° rotation after Sim3 alignment — visually
  identical, ghost collapsed either way (centroid_sep 1.28m both, vs 2.90m with
  the OLD solver + no loop). So loop closure is off by default; keep it for long
  trajectories where real translational drift accumulates between distant
  revisits (the IMU chain constrains rotation, nothing constrains slow position
  drift over minutes).

Visualizers: `vio_visualize_features.py`, `vio_visualize_matches.py`,
`vio_visualize_tracks.py` (same pattern as the hand pipeline's render
scripts — see below for tracks specifically), plus `vio_visualize_imu_prior.py`
(gravity arrow + naive cumulative-rotation triad + synced video thumbnail — a
fixed-camera acceptance check since there's no pose chain at that stage yet).
Stage 5's trajectory+point-cloud viewer moved up to
`data_processing/visualize_data.py` (generic across pipelines — see
`../CLAUDE.md`): optimized camera trajectory as left/right stereo frustums,
colored by eye, + landmark point cloud colored by each landmark's ACTUAL
sampled video pixel — much more informative than a synthetic depth colormap
for spotting real-vs-degenerate reconstructions. Since the stage-5 world is
gravity-aligned (+z up), viser's up-direction is just `+z` — no measured-
gravity tilt (that was for the OLD anchor-frame world). Video thumbnail
decodes via torchcodec for fast scrubbing. Static trail frustums are drawn
thin so the bright current-frame frustums pop.

Tracks (stage 3's viz) are rendered as colored "comet" dots (golden-ratio
hue spacing per track) with fading trajectory tails — `--tracks-per-frame 0`
draws every active track every frame, the confirmed-preferred default over
an earlier hard cap.

## Optimization (profiled + shipped 2026-07-03)

Profiled both stages on long-test1 (A6000) and attacked the measured bottlenecks:

- **Stage 2 matching: 3.1x (275s → 89s @ 600 frames).** `match_batch`'s per-batch
  numpy-stack + fp16→fp32 upcast + H2D upload was **67.6%** of wall-clock (the GPU
  starved, forward was only 21%). Fixed with a **GPU-resident FeatureStore**:
  features uploaded once/frame, descriptors kept fp16 (their disk dtype),
  per-frame descending-score order precomputed, and top-k select + batch stack
  done as on-device gathers → assemble dropped to 3.3%. JSONL writes moved to a
  background thread (`AsyncJsonlWriter`). Stage 2 is now GPU-bound (forward ~62%,
  temporal RANSAC ~21%). Correctness: raw matches bit-identical to the old path;
  the fp16 store trades run-to-run reproducibility (fp16 reorders LightGlue's
  match emission → reseeds RANSAC) for the speed, accept/reject flips ~0.03%.
- **Stage 1 extraction: 2.3x (45s → 20s @ 400 frames).** Decode + CPU→GPU upload
  was ~43% of wall. `--decoder torchcodec` (NVDEC, default) decodes + crops
  on-GPU: 2.9ms/frame vs cv2's 24ms. `--amp` (fp16 autocast on the forward) adds
  ~1.5x on the remaining forward cost. See the stage-1 table row for the NVDEC
  pixel-drift caveat.

Remaining ideas, not yet done (roughly by payoff/effort):
- **Multi-GPU sharding of stage 2** — the chunk loop is embarrassingly parallel
  (chunks share only the read-only features.h5 + the deferred temporal accum),
  and sphynx has 8 A6000s. Near-linear speedup for a launcher script; the single
  biggest remaining lever, subject to shared-box run etiquette.
- **Speed up the temporal RANSAC** (now stage 2's #2 cost at ~21%): larger batch,
  or a cheap descriptor MNN pre-filter to skip/replace LightGlue on near-
  covisible short-gap pairs (changes match quality — validate against tracks).
- **Binary match output** (npz/parquet per chunk) if json ever shows up on the
  critical path again (currently hidden behind the writer thread).
- **torchcodec in the two video visualizers** too — same cv2→NVDEC swap,
  visualizer-only so lower priority than the pipeline stages.

## Status (2026-07-03)

Stages 1-3 validated on `long-test1`. All 5 stages run end-to-end on `testimu`
(~897-frame handheld clip with real IMU). Stage 5 reworked into the two-stage
GLOMAP-faithful solve (see "Stage 5" above): tracks the IMU rotation truth to
~1° (was collapsing a 180° turn to ~19°), gravity-aligned world, no anchor
cost, Cauchy robust loss. Loop closure validated as optional and off by
default. Motion blur robustness still untested (`testnewcamsblur*` exist).

Tuned defaults (2026-07-03, validated in viser on testimu): 150° FOV,
1024 keypoints, detection-threshold 0.001, resize 1500, batched extraction;
dense-to-20 / 70-frame-tail match schedule (fixes track flicker); min-frames 15
track filter. End result: ~16k stable landmarks, 0.325° median angular residual.
IMPORTANT: run stage 5 WITHOUT `--early-termination` (now the default) — it
false-positives at ~2 iters and leaves positions under-solved (scrunched
trajectory); stage 1 needs ~15 iters to converge.

Prior open items to re-check against the new solver (were measured on the OLD
one): negative-depth landmarks, per-iteration cost growth over a long run.
Track-quality-weighted costs still flagged as worth trying.
