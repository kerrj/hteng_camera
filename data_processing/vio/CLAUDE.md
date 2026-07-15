# Learned Offline VIO

The active camera-pose pipeline uses VGGT-Omega windows, calibrated stereo
scale, IMU rotation/gravity priors, sparse proximity loop closure, and a JAXLS
pose graph. It is offline and optimized for trajectory quality rather than
real-time latency.

Read `LEARNED_VIO_RESEARCH.md` for the selected formulation, hyperparameters,
benchmarks, and validated artifacts.

## Active files

- `vio_imu_prior.py`: consecutive-frame gyro factors and confidence-gated
  gravity observations.
- `vio_vggt_window_infer.py`: direct TorchCodec GPU decode, calibrated fisheye
  remapping, multi-GPU VGGT inference, proximity-loop proposal, and dense depth
  overlap verification.
- `vio_vggt_pose_graph.py`: metric window scales, relative-pose factors, IMU
  priors, interpolation, and calibrated left/right trajectory export.
- `run_vggt_pipeline.py`: resumable end-to-end stage orchestration.
- `test_vggt_pose_graph.py`: pose, scale, interpolation, and stereo convention
  tests.
- `test_vggt_loop_candidates.py`: proximity retrieval and depth-overlap tests.
- `../visualize_data.py`: Viser trajectory inspection.

The previous LightGlue/global-BA implementation is retained only under
`_deprecated_BA_vio/`. Do not add new dependencies on it.

## Environments

- VGGT inference: `vggtomega`
- Pose graph and tests: `jaxgpu`
- Remote GPU host: `sphynx`, repository at `/home/jkerr/hteng_camera`

Use `CUDA_VISIBLE_DEVICES` to select idle GPUs. Multi-GPU inference uses one
model-owning thread per visible device. Eager mode is the practical multi-GPU
default; `reduce-overhead` CUDA Graphs are only thread-safe for one GPU.

## Selected defaults

- 15 Hz keyframes, 512x512 virtual pinhole images
- 128-frame windows with 64-frame overlap
- right-eye anchor every second plus window endpoints
- proximity/view loop candidates with dense depth-overlap verification
- visual translation/rotation weights 10/10
- window-anchor weight 0.25
- gyro weight 10, gravity weight 1, gravity norm sigma 0.05 g
- dimensionless log-baseline weight 100
- constant-velocity weight 0.1
- plain least-squares visual factors

Outputs must contain finite, contiguous native-frame trajectories for both
cameras. Right poses are always composed from the optimized left pose and the
fixed recording calibration.
