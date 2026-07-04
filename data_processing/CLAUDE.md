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
