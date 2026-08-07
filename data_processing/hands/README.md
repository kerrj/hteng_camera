# Stereo hand pipeline

This pipeline estimates metric MANO hand poses from synchronized left/right
fisheye videos. **Run VIO before this pipeline.** Hand fitting requires the
completed camera trajectory so acceleration smoothing can operate on hand
motion in the world frame rather than mixing hand and head motion. It has two
hand stages:

1. WiLoR inference on rectified virtual-pinhole crops.
2. Robust stereo MANO fitting, outlier rejection, VIO-world temporal smoothing,
   and interpolation of short enclosed gaps.

Commands below run from the repository root on `sphynx`. Generate the
full-rate trajectory first by following
[`data_processing/vio/README.md`](../vio/README.md); the hand optimizer will
consume that trajectory in stage 2.

Only **stage 2** needs the trajectory. Stage 1 reads just the videos and
calibration, so launch it in the background on a spare GPU the moment VIO
starts rather than waiting for VIO to finish — the two run concurrently and
the pipeline finishes in roughly the time of the slower one:

```bash
REC=take01
CUDA_VISIBLE_DEVICES=6 nohup python \
  data_processing/hands/wilor_hands_pinhole.py \
  "$REC/left.mp4" "$REC/right.mp4" --calib-dir "$REC" \
  > "$REC"_wilor.log 2>&1 & disown
```

Pick GPUs that VIO's `--gpus` list does not claim, and check `nvidia-smi`
first — other jobs are often already on the box. `nohup ... & disown` keeps
both jobs alive across a dropped VPN or SSH session; reconnect and tail the
logs to pick them back up.

## Inputs

A recording directory must contain:

```text
left.mp4
right.mp4
calib_<left-serial>.json
calib_<right-serial>.json
stereo_<left-serial>_<right-serial>.json
derived/trajectory_<name>.npz
```

Outputs are written to `<recording>/derived/` by default.

## One-time setup

Start from a CUDA PyTorch environment with mutually compatible `torch`,
`torchvision`, and `torchcodec` builds. Install the pinned WiLoR package without
its restrictive dependency metadata:

```bash
python -m pip install --no-deps \
  "git+https://github.com/warmshao/WiLoR-mini.git@ebec42f"
```

Install any missing runtime libraries separately so pip does not enforce
WiLoR's `torch<=2.5` or `ultralytics==8.1.34` pins:

```bash
python -m pip install \
  "smplx==0.1.28" timm einops ultralytics opencv-python \
  huggingface-hub scikit-image roma yacs
```

WiLoR's upstream MANO pickle contains obsolete chumpy objects. Install chumpy
only for the conversion, then prepare the model cache:

```bash
python -m pip install --no-deps --no-build-isolation chumpy
python data_processing/hands/prepare_wilor_models.py
```

Removing chumpy afterward is optional. The hand runtime never imports it after
conversion:

```bash
python -m pip uninstall chumpy
```

Models default to `~/.cache/hteng_camera/wilor`. Set `WILOR_MODEL_DIR` or pass
`--model-dir` to both preparation and inference to use another location.
Preparation is idempotent, preserves the original MANO pickle as
`MANO_RIGHT.pkl.chumpy.bak`, does not overwrite existing assets unless
`--force-download` is passed, and prints actionable download/conversion errors.

Export that converted model into the compact bundle used by JAX:

```bash
python data_processing/hands/export_mano_jax.py --out /tmp/mano_jax.npz
```

The exporter reports whether the model is missing, still contains chumpy, or
has an incompatible schema. It uses the same model cache.

The optimizer runs in a separate JAX environment. It does not require WiLoR or
its Torch dependencies.

## Run the pipeline

After VIO has completed, set the recording and its full-rate trajectory:

```bash
REC=long-test2
TRAJ="$REC/derived/trajectory_vggt_omega_fullrun_20260714.npz"
```

### 1. Extract stereo WiLoR observations

```bash
python data_processing/hands/wilor_hands_pinhole.py \
  "$REC/left.mp4" "$REC/right.mp4" \
  --calib-dir "$REC"
```

This writes `$REC/derived/hands.jsonl`. Each frame contains matched left/right
keypoints, MANO initialization, and the two virtual-camera transforms.

For a quick inference smoke test, add `--max-frames 300`.

The YOLO hand detector is TorchInductor-compiled by default
(`fullgraph=True, dynamic=False`, ~1.27x on detection, ~30 s warmup); pass
`--no-compile` to force eager. On one A6000 over 3200 frames of `long-test2`
this moves the whole stage from 12.7 to 13.3 fps with byte-identical
detections. The gain is small because decoding, not inference, dominates:

| stage | share of wall time |
|---|---|
| stereo H.265 decode | ~49% |
| YOLO detection | ~17% |
| verged crop rendering + postprocess | ~23% |
| WiLoR ViT | ~11% |

The WiLoR ViT is deliberately left eager. It measured 1.00-1.01x under every
compile mode tried (static and dynamic, `default` and `reduce-overhead`) because
it is already compute-bound on fp16 SDPA and GEMM kernels. It also cannot reach
`fullgraph=True` at all: `smplx.lbs.batch_rigid_transform` indexes a Python
list with a tensor and `roma.rotmat_to_unitquat` calls a dynamic-shape
operator, giving 18 graph breaks inside third-party code.

Compiling the detector requires patching the bound `forward` instead of
replacing `hand_detector.model`, since ultralytics calls `len()` on the model
and an `OptimizedModule` wrapper does not support it. `reduce-overhead` is not
usable here: its CUDA Graphs reuse output buffers, and `detect_gpu` is called
once per eye while the first eye's boxes are still live.

### 2. Fit each hand track

```bash
conda activate jaxgpu

python data_processing/hands/stereo_optimize.py \
  --calib-dir "$REC" --trajectory "$TRAJ" --hand left

python data_processing/hands/stereo_optimize.py \
  --calib-dir "$REC" --trajectory "$TRAJ" --hand right
```

These commands write:

```text
$REC/derived/hands3d_left.jsonl
$REC/derived/hands3d_right.jsonl
```

The optimizer first fits every candidate frame independently. It rejects
non-finite geometry, invalid depth, joints behind either camera, high stereo
reprojection error, and high epipolar disagreement. A second joint solve adds
minimum-acceleration smoothing only across consecutive accepted observations.
It uses the observed per-frame camera times stored by VIO, including real
capture gaps, rather than assuming uniform spacing. Root translation and global
wrist rotation are smoothed in the VIO world frame; internal finger rotations
remain parent-relative. Finally, enclosed gaps up to five frames are
interpolated at their observed frame times. Leading, trailing, and longer gaps
remain absent.

Useful controls:

```text
--frame-min / --frame-max   Process a range for debugging
--interp-max-gap N          Maximum enclosed gap to fill; 0 disables
--min-depth / --max-depth   Accepted root-depth range (default maximum 1.5 m)
--max-reproj-px             Maximum phase-1 mean reprojection error
--max-epipolar-px           Maximum median left/right vertical disagreement
```

Interpolated rows have `"interpolated": true` and a two-element
`"source_frames"` field. Measured rows include phase-1 quality metrics. The
first JSONL row is file-level metadata and records thresholds and counts.

### 3. Export training data

After both hand tracks finish, consolidate the trajectory, hands, timing, and
calibration into the canonical frame-aligned training file:

```bash
python data_processing/export_training_h5.py "$REC" \
  --trajectory "$TRAJ"
```

This writes `$REC/derived/training.h5`; the left and right MP4s remain separate
for direct video decoding. See
[`data_processing/TRAINING_FORMAT.md`](../TRAINING_FORMAT.md) for the complete
versioned schema, coordinate conventions, validity masks, and efficient loading
guidance.

## Visualize with VIO

Use `data_processing/visualize_data.py` to place hand meshes in the VIO world:

```bash
python data_processing/visualize_data.py "$REC" \
  --trajectory "$REC/derived/trajectory_vggt_omega_fullrun_20260714.npz" \
  --hands-left "$REC/derived/hands3d_left.jsonl" \
  --hands-right "$REC/derived/hands3d_right.jsonl" \
  --mano /tmp/mano_jax.npz \
  --color-mode depth \
  --trail-stride 10 \
  --port 8133
```

Use the appropriate trajectory filename for the recording. When running
remotely, forward the port from the laptop:

```bash
ssh -N -L 8133:localhost:8133 sphynx
```

Then open `http://localhost:8133`.

## Validation

Syntax and focused filtering tests:

```bash
python -m py_compile \
  data_processing/hands/stereo_optimize.py \
  data_processing/hands/export_mano_jax.py \
  data_processing/export_training_h5.py

python data_processing/hands/test_stereo_filtering.py

python data_processing/hands/test_projection_math.py "$REC"

python data_processing/test_training_h5.py
```

## File inventory

- `wilor_hands_pinhole.py`: stereo detection and rectified WiLoR inference.
- `wilor_runtime.py`: shared WiLoR model, detector, and post-processing helpers.
- `stereo_optimize.py`: robust two-stage metric hand fitting.
- `mano_jax.py`: differentiable MANO implementation used by jaxls.
- `export_mano_jax.py`: reproducible MANO-to-NPZ export.
- `prepare_wilor_models.py` and `mano_dechumpy.py`: model download and one-time
  MANO conversion.
- `test_projection_math.py`: verifies stored virtual-camera geometry.
- `test_stereo_filtering.py`: focused filtering, temporal topology, quaternion,
  and VIO-transform tests.
- `../visualize_data.py`: VIO trajectory, camera, landmark, and hand viewer.
- `../export_training_h5.py`: final versioned training-data exporter.
