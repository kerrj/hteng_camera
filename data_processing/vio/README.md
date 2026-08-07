# Offline Stereo-Inertial VIO

This pipeline estimates full-rate left/right camera poses from stereo fisheye
video and IMU data. VGGT-Omega supplies overlapping visual pose windows, JAXLS
fuses them with calibrated stereo scale and IMU priors, and sparse
depth-verified loop windows control long-term drift.

## Requirements

- A recording directory containing `recording.json`, `left.mp4`, `right.mp4`,
  camera/stereo calibration JSON files, `imu_log.csv`, and `sync_log.csv`.
- The `vggtomega` environment for TorchCodec and VGGT-Omega inference.
- The `jaxgpu` environment for JAXLS optimization and pose interpolation.
- A VGGT-Omega checkpoint, currently:
  `/home/jkerr/checkpoints/vggt-omega/vggt_omega_1b_512.pt`.

On `sphynx`, the environments are under `/home/jkerr/miniconda3/envs/`.

## Run

From the repository root, first generate the frame-aligned IMU priors:

```bash
/home/jkerr/miniconda3/envs/jaxgpu/bin/python \
  data_processing/vio/vio_imu_prior.py long-test2
```

Then run the complete visual pipeline and pose graph. Pass one or more idle
physical GPU indices; inference creates one model-owning thread per GPU.

```bash
/home/jkerr/miniconda3/envs/vggtomega/bin/python \
  data_processing/vio/run_vggt_pipeline.py long-test2 \
  --checkpoint /home/jkerr/checkpoints/vggt-omega/vggt_omega_1b_512.pt \
  --gpus 2,3,4,5 \
  --tag vggt_omega_fov100
```

Use `--gpus 2` for a single-GPU run. Reusing the same tag resumes cached
windows after validating their frame identities and inference settings. Use a
new tag when changing the field of view, window geometry, or checkpoint.

Launch this in the background (`nohup ... > vio.log 2>&1 & disown`) and start
the hand pipeline's WiLoR stage concurrently on GPUs outside this `--gpus`
list. That stage needs only the videos and calibration, not this trajectory,
so the two overlap completely; see
[`data_processing/hands/README.md`](../hands/README.md). Only the hand
optimizer (stage 2) has to wait for the trajectory below.

The final trajectory is written to:

```text
<recording>/derived/trajectory_<tag>.npz
```

It contains interpolated poses for every native frame and both cameras. Right
poses are composed from the optimized left trajectory and fixed calibrated
stereo transform, so the exported baseline cannot drift.

Full-rate interpolation uses cubic splines for camera centers and SciPy
`RotationSpline` for orientation. Both are evaluated at the observed per-frame
camera timestamps from `sync_log.csv`, not a uniform frame-index linspace.
Nonpositive timestamp intervals caused by a camera-counter reset are replaced
with the recording's median positive interval; all other observed timing
variation is preserved.

## Defaults

- 15 Hz visual keyframes and 512x512 rectified 100-degree views
- 128-frame windows with 64-frame overlap
- one right-eye metric anchor per second plus window endpoints
- at most eight sparse proximity/view loop regions
- dense pointmap overlap verification for loop windows
- grouped Cauchy loss with scale 2 on all VGGT pose edges
- visual translation/rotation weights 10/10
- window-anchor weight 0.25
- calibrated log-baseline weight 100
- gyro weight 10
- gravity weight 1 with a 0.05 g Gaussian acceleration-norm taper
- constant-velocity weight 0.1
- soft gauge weights 1.0 translation and 0.01 rotation
- TorchInductor compilation of the VGGT-Omega aggregator

Run `run_vggt_pipeline.py --help` for configurable parameters. The complete
configuration and stage timings are stored in
`<recording>/derived/<tag>/pipeline_run.json`; graph-specific robust/gauge
settings are also stored in the trajectory NPZ.

## Inference speed

`Aggregator` holds nearly all of the model's FLOPs and takes a single
fixed-shape tensor, so it is compiled with `fullgraph=True, dynamic=False`.
Measured on one A6000, 32 images per 512-px window:

| | inference | peak memory |
|---|---|---|
| eager | 2.10 s | 6.56 GiB |
| compiled aggregator | 1.54 s | 4.30 GiB |

Warmup (~70 s cold, shared across GPUs through the on-disk Inductor cache) is
paid inside each worker's first window; loop windows compile separately since
their image counts and dense head differ. Results match eager to bf16
kernel-ordering noise (~0.04 deg rotation, ~0.3 px on 512-px intrinsics).

`VGGTOmega.forward` as a whole cannot be captured: it returns a list of
per-layer tensors with `None` in the uncached slots, which Dynamo rejects
(`torch.* op returned non-Tensor`). Pass `--no-compile` to
`vio_vggt_window_infer.py infer` to force eager.

GPU workers are spawned processes, one per device, not threads. TorchDynamo
compilation is not thread-safe: concurrent first calls in one process either
raise `using FX to symbolically trace a dynamo-optimized function` or deadlock
in the Inductor subprocess pool, and serializing warmup instead cost minutes
of idle GPUs per run. Separate processes give each worker private compiler
state, so all devices warm up concurrently while sharing the on-disk Inductor
cache. Nothing CUDA crosses the process boundary: workers decode, infer, and
write their own NPZ files.

The Inductor mode is `default`. The aggregator is compute-bound at these
sizes, so `reduce-overhead`'s CUDA Graphs measured no faster (1.541 s vs
1.546 s at 32x512), and `max-autotune` was also no faster than `default` while
costing far more warmup.

## Visualize

```bash
/home/jkerr/miniconda3/envs/jaxgpu/bin/python \
  data_processing/visualize_data.py long-test2 \
  --trajectory long-test2/derived/trajectory_vggt_omega_fov100.npz \
  --camera-eyes both --trail-stride 30 --no-color --port 8143
```

For a remote run, forward the selected port:

```bash
ssh -L 8143:localhost:8143 sphynx
```

Then open `http://localhost:8143`.

## Tests

With `pytest` installed in the JAX environment:

```bash
cd data_processing/vio
/home/jkerr/miniconda3/envs/jaxgpu/bin/python -m pytest -q \
  test_vggt_pose_graph.py test_vggt_loop_candidates.py
```

The active implementation is:

- `vio_imu_prior.py`
- `vio_vggt_window_infer.py`
- `vio_vggt_pose_graph.py`
- `run_vggt_pipeline.py`

The older feature/landmark BA implementation is available in Git history but
is not a dependency of this pipeline.
