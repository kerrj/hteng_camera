# calibrations/

Per-sensor calibration, **keyed by serial** and **tracked in git** — one small,
image-free JSON per physical camera, so calibration travels with the repo and is
diffable over time.

| File | Written by | Holds |
|---|---|---|
| `calib_<serial>.json` | `charuco_calibrate.py` (intrinsics), `color_calibrate.py` (color) | Per-sensor intrinsics + color (white balance, eventually CCM) |
| `stereo_<L>_<R>.json` | `charuco_calibrate.py` | Cam2cam pose (R, t) for a pair |
| `head_pose_<L>_<R>.json` | mount calibration | Static robot `base_from_head` pose |

Each tool **merges** into the existing per-sensor file, so running the color tool
after the ChArUco tool keeps both sections.

The live stereo publisher requires a mount-pose sidecar before its output is
used for robot inference:

```json
{
  "format": "hteng-camera-head-pose/2",
  "serial_left": "046060323003",
  "serial_right": "046060323006",
  "base_from_head": [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
  ],
  "head_from_stereo_midpoint": [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
  ]
}
```

The head frame is gravity-aligned: +X robot-forward, +Y left, and +Z up.
`base_from_head` places that frame in robot-base coordinates.
`head_from_stereo_midpoint` rotates the calibrated physical stereo midpoint
into the head frame, encoding camera tilt independently of the head convention.
The example identities are only structural; do not use them as a robot mount
calibration.

## How it's loaded

`HTCamera` auto-loads `calib_<serial>.json` for its serial on open (search order:
`$HTENG_CALIB_DIR`, then `./calibrations`, then `.`). White-balance gains are then
available as `cam.wb_gains` and fold for free into `convert.tonemap_linear`'s LUT.

```python
cam = HTCamera(serial="046060323003")     # auto-loads calibrations/calib_046060323003.json
rgb, _ = cam.grab()
preview = convert.tonemap_linear(rgb, wb_gains=cam.wb_gains)   # white-balanced, free

# Or load directly:
from hteng_camera import calibration
cal = calibration.find("046060323003")
```

## What is *not* tracked

Everything written here is under `calibrations/`, but only the JSONs are tracked.
The raw capture images — ChArUco views in `calib_session_*/` subfolders, and WB
measurement PNGs (`wb_measurement_*`) — are gitignored. They're provenance you can
re-run a solve from, not deliverables. The JSONs are the deliverable.
