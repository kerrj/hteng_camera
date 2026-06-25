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

1. **Pinhole-crop refinement** — detect hand on fisheye (or undistorted full
   frame), render an undistorted pinhole view *aimed at the hand* via
   `undistort_maps`, re-run WiLoR there for precise localization. Use
   `predict_with_bboxes`. **TBD / not yet built.**
2. **Stereo depth for hands** — match hands across left/right and recover metric
   depth. User's preferred approach: a **mesh/pose optimization** that optimizes
   the hand's 3D position (and pose) to fit *both* sets of 2D keypoints given
   the known stereo disparity/extrinsics — not naive triangulation. **TBD.**
3. **Head pose** — separate pipeline, after hands.

## Conventions

- Keep all scripts/files for this work under `data_processing/`.
- Record any new package added to `eyeball211` in `README.md` + here.
