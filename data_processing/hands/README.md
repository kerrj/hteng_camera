# data_processing/hands

Pipeline for extracting **hand poses** from the stereo fisheye recordings
(e.g. `../long-test1/`). Head/camera pose is a separate pipeline — see
`../vio/`.

Code is authored on the laptop and synced to the GPU box (`chungus`) via git;
inference runs there in the **`eyeball211`** conda env.

## Hand pose — WiLoR

We use [WiLoR-mini](https://github.com/warmshao/WiLoR-mini), a slimmed,
pip-installable wrapper around [WiLoR](https://github.com/rolpotamias/WiLoR)
(Potamias et al.). It auto-downloads its weights + MANO model from HuggingFace
and exposes a one-call `predict()` API. Per detected hand it returns:

| key (`out["wilor_preds"]`) | shape | meaning |
|---|---|---|
| `pred_keypoints_2d` | (1, 21, 2) | 2D joints in **input-image** pixels |
| `pred_keypoints_3d` | (1, 21, 3) | 3D joints, hand-root frame, metres |
| `pred_vertices`     | (1, 778, 3) | MANO mesh verts, hand-root frame |
| `global_orient`     | (1, 1, 3)  | MANO global orient (axis-angle) |
| `hand_pose`         | (1, 15, 3) | MANO articulation (axis-angle) |
| `betas`             | (1, 10)    | MANO shape |
| `pred_cam_t_full`   | (1, 3)     | hand translation in the full-image pinhole camera |
| `scaled_focal_length` | scalar   | focal (px) of that assumed pinhole camera |

Plus `out["is_right"]` (1.0 = right hand) and `out["hand_bbox"]` (x1,y1,x2,y2).

**Important:** WiLoR assumes a **pinhole** camera. Our footage is wide-angle
fisheye, so `pred_keypoints_2d` / `pred_cam_t_full` are only approximate near
the periphery. Two ways to use it well:
  - keypoints land fine on the raw fisheye for *detection* and rough pose;
  - for precise localization, render an undistorted **pinhole crop** toward the
    hand (`src/hteng_camera/calibration.py::Intrinsics.undistort_maps`) and run
    WiLoR there — TBD, see `wilor_hands.py --pinhole` (planned).

## Environment setup (one-time, on chungus)

See `install_wilor.sh`. The `eyeball211` env is bleeding-edge (py3.13, numpy
2.4, torch 2.11+cu130), so the stock WiLoR pins don't apply. We install
`--no-deps` and add only what's actually missing.

Packages added to `eyeball211` for this:
  - `wilor_mini` (git, --no-deps)
  - `roma`, `yacs`, `smplx==0.1.28`, `ultralytics` (--no-deps)
  - `dill` (auto-installed by ultralytics to load the YOLO detector)

`chumpy` is **not** installed at runtime. It is only needed once to convert the
chumpy-pickled `MANO_RIGHT.pkl` into plain numpy — see `mano_dechumpy.py`.
After that conversion nothing imports chumpy (verified: pipeline loads with
chumpy uninstalled).

## Files

- `install_wilor.sh`  — reproducible env setup on chungus
- `mano_dechumpy.py`  — one-time MANO `.pkl` de-chumpy conversion
- `wilor_hands.py`    — run WiLoR over a video → per-frame hand poses (JSON) + optional viz
