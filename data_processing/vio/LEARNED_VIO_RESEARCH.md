# Learned VIO alternatives

Research and prototype status as of 2026-07-14.

## Recommendation

Use [VGGT-Omega](https://github.com/facebookresearch/vggt-omega) as an
offline visual measurement generator, not as a complete VIO system:

1. Rectify each fisheye eye into a common-orientation pinhole view.
2. Interleave left/right images in overlapping temporal windows.
3. Run 8-24 stereo rig frames per window, with 16 and 50% overlap as the
   initial default.
4. Add separate windows around geometrically verified loop candidates.
5. Fuse relative poses in JAXLS with one metric scale per model window,
   calibrated stereo-baseline factors, IMU gyro rotations, and gated
   accelerometer gravity directions.

Omega only accepts images. It does not accept camera, IMU, pose, or intrinsic
priors, so those constraints belong in the downstream graph. Interleaved
stereo gives the model simultaneous cross-eye context; the known baseline
makes each window metric even if Omega's own reconstruction scale drifts.

## Candidate ranking

| Candidate | Stereo | IMU / priors | Release status | Assessment |
|---|---|---|---|---|
| [VGGT-Omega](https://arxiv.org/abs/2605.15195) | Natural multi-image input | No prior inputs | Official code and checkpoint; research license | Best visual backbone to test. Use our graph for metric scale, IMU, and loops. |
| [MapAnything](https://github.com/facebookresearch/map-anything) | Natural multi-image input | Accepts intrinsics, depth, and camera poses; no raw IMU | Official code and Apache model available | Best fallback when inference-time geometric conditioning matters. Pose inputs are conditioning, not uncertainty-weighted IMU factors. |
| [DBA-Fusion](https://github.com/GREAT-WHU/DBA-Fusion) | No; stereo remains unchecked in its README | Tightly coupled GTSAM IMU and other sensors | Released, GPL-3.0, RA-L 2024 | Best released native learned VIO, but old Torch 1.11/custom CUDA/GTSAM stack and monocular-only implementation make integration expensive. |
| [VGGT-SLAM 2.0](https://github.com/MIT-SPARK/VGGT-SLAM) | Visual submaps | No IMU | Released, BSD-2-Clause, RSS 2026 | Reuse inference-only submap, retrieval verification, and factor-graph ideas. It uses an older VGGT fork, so do not replace Omega with its model. |
| [VGGT-Long](https://github.com/DengKaiCQ/VGGT-Long) | Visual chunks | No IMU | Released, ICRA 2026 | Reuse chunk/overlap/loop alignment ideas. Its disk caching and Sim(3)/SE(3) alignment validate the proposed architecture. |
| [DPVO / DPV-SLAM](https://github.com/princeton-vl/DPVO) | Monocular | No IMU | Released, MIT, last substantive feature update 2024 | Fast and mature visual baseline, but does not meet the IMU/stereo goal. |
| [DL-VINS-Factory](https://github.com/limshoonkit/DL-VINS-Factory-ROS2) | VINS-style | IMU | Released, GPL-3.0, 2026 | Learned features plus a conventional VINS backend, focused on ROS2/TensorRT/Jetson. It largely duplicates this repository's current LightGlue/JAXLS direction. |
| [MDE-VIO](https://arxiv.org/abs/2602.11323) | Monocular | VINS-Mono IMU | Paper says code will be released | Interesting learned-depth regularization, but not currently testable and less direct than calibrated stereo. |

BAMF-SLAM is conceptually close (multi-fisheye, IMU, recurrent field
transforms, dense BA), and DVI-SLAM reports stereo/IMU support, but no usable
official implementation was found. They are research references, not current
integration candidates.

## Measured Omega behavior

Measurements used one RTX A6000 on Moggy, Torch 2.11/CUDA 13, 512x512 inputs,
and the released 1.144B-parameter architecture. Large-window measurements used
the full camera+depth model; the production path disables the unused depth
head.

| Rig frames | Input images | Full-model time | Peak allocated |
|---:|---:|---:|---:|
| 16 | 32 | 2.16 s | 6.68 GiB |
| 32 | 64 | 6.42 s | 10.74 GiB |
| 64 | 128 | 20.45 s | 15.56 GiB |
| 128 | 256 | 75.19 s | 25.18 GiB |
| 192 | 384 | 166.45 s | 34.80 GiB |
| 256 | 512 | 293.21 s | 44.41 GiB |

Memory permits very large A6000 chunks, but attention makes them poor value.
Use small overlapping tracking chunks and reserve larger, sparse chunks for
loop ties.

For 32 images:

- Full eager model: 2.158 s steady state.
- Camera-only eager: 2.010 s, 6.56 GiB.
- Camera-only `torch.compile(mode="reduce-overhead")`: 71.8 s first call,
  1.46-1.61 s thereafter.
- Compiling the full depth path fails in TorchInductor `CantSplit` on this
  Torch build.
- Omega already runs its aggregator under BF16 autocast and its heads in
  FP32. Casting the whole model to BF16 fails in the FP32 camera-head
  `LayerNorm`; do not use blanket `model.to(torch.bfloat16)`.

Compile is useful for a fixed-shape long recording, but its startup cost loses
on a short clip. Different final-window shapes can trigger another compile.

## Selected `long-test2` configuration

The current best qualitative result is
`long-test2/derived/trajectory_vggt_omega_w128_loops.npz`:

- 15 Hz Omega keyframes from a 40.323 Hz recording, followed by cubic camera
  center and `RotationSpline` orientation interpolation to all 11,043 frames.
- 512x512, 100 degree pinhole views rendered from the calibrated fisheyes.
- 128-frame tracking windows (8.53 seconds), 64-frame overlap, and 64 windows.
- Dense left-eye frames plus a right-eye metric-scale anchor every second and
  at each window endpoint: 138-139 images per tracking call.
- Camera-only `torch.compile(mode="reduce-overhead")`; Omega's aggregator uses
  BF16 autocast and its camera head remains FP32.
- Eight loop candidates from a 2 Hz trajectory proximity query: 1.5 m maximum
  distance, 15 second minimum temporal gap, 60 degree maximum view-angle
  difference, 2 second vote radius, and 10 second endpoint NMS.
- Each loop call contains two 64-frame neighborhoods. The eager dense head
  verifies symmetric high-confidence depth reprojection with 8 px sampling,
  median confidence cutoff, and 15% relative depth tolerance.
- Exactly one cross-segment relative-pose edge per accepted loop window. All
  eight visually inspected candidates were accepted; symmetric overlap scores
  ranged from 0.116 to 0.722.
- JAXLS: visual translation/rotation weights 10/10, within-window star weight
  0.25, loop edge weight 1, log-ratio stereo baseline weight 100, gyro weight
  10, gravity weight 1 with 0.05 g Gaussian norm taper, constant velocity
  weight 0.1, pose anchor weight 1000, plain L2 visual residuals,
  conjugate-gradient linear solve, and 20 iterations.
- Calibrated stereo baseline: 0.066855 m.

Recorded compute times on one RTX A6000 per inference process:

| Stage | Work | Timed compute | Peak allocated |
|---|---:|---:|---:|
| Fisheye preparation | 4,108 stereo frames | 276.2 s | CPU |
| Tracking Omega | 64 windows | 1,358.8 s | 7.90 GiB |
| Dense loop verification | 8 windows | 395.4 s | 14.88 GiB |
| JAXLS solve | 4,108 keyframe poses, 72 scales | 84.8 s | not recorded |

Tracking calls took 18.42 s steady state. Loop calls averaged 49.42 s. The
recorded core compute is about 30 minutes 45 seconds after preparation, or
35 minutes 20 seconds including preparation; model loading and image
preprocessing overhead are not included in the per-window forward timers.
The final graph has 16,136 visual edges and 4,106 composed IMU rotation edges.
It retained a 117.71 m path while reducing endpoint separation from 1.32 m to
0.12 m.

The current implementation decodes requested raw frames directly to each
worker GPU with TorchCodec, applies calibrated fisheye-to-pinhole sampling with
`grid_sample`, and passes normalized tensors directly to Omega. It does not
write or reload JPEGs. Independent windows are distributed through a thread
pool with one persistent model per visible GPU; `CUDA_VISIBLE_DEVICES` controls
the physical devices and `--num-devices` can cap the worker count. A two-GPU
smoke test processed simultaneous 18/19-image windows in 1.24/1.09 seconds at
5.51/5.59 GiB peak allocated memory per GPU.

The pose graph exports interpolated world-to-camera poses for both eyes at
every native frame. Right poses are composed from the optimized left poses and
the fixed calibrated stereo transform. The dimensionless log-baseline factor
at weight 100 reduced fitted-window baseline error to 0.038% mean and 0.461%
worst case on the selected `long-test2` graph. The updated artifact is
`long-test2/derived/trajectory_vggt_omega_w128_loops_baselinefixed.npz`;
its 11,043 exported stereo pairs reproduce the calibrated 0.0668551596 m
baseline to within 6.1e-15 m.

A representative 138-image tracking window takes 2.98 seconds for direct
decode plus remap (46.4 images/s) and peaks at 2.90 GiB during that stage.
Those stages run independently on every inference GPU. Multi-GPU eager mode is
the practical default. `reduce-overhead` remains useful for one GPU, but its
CUDA Graph runtime is not thread-safe across Python GPU-worker threads; a
multi-GPU request automatically falls back to standard TorchInductor mode.

## Direct-input full-pipeline validation

On 2026-07-14, isolated full runs used idle Sphynx GPUs 2-7. `testimu` ran on
two GPUs in 206 seconds and produced six tracking windows plus one
dense-verified loop. `long-test2` ran on six GPUs in 609 seconds total:
349 seconds for 64 tracking windows, 61 seconds for the preliminary graph and
loop proposal, 137 seconds for eight dense-verified loops, and 62 seconds for
the final graph. All eight long-sequence loops passed with symmetric overlap
scores from 0.118 to 0.683.

The final outputs are:

- `testimu/derived/trajectory_vggt_omega_fullrun_20260714.npz`
- `long-test2/derived/trajectory_vggt_omega_fullrun_20260714.npz`

Both contain finite, contiguous native-frame left/right trajectories and exact
calibrated stereo composition. The long result has 11,043 frames, a 122.08 m
path, 0.087 m endpoint separation, 0.0396% mean latent baseline error, and
0.955% worst latent baseline error.

## Implementation

- `vio_vggt_window_infer.py`: fisheye rectification, stereo/temporal window
  construction, proximity loop proposal, dense overlap verification,
  camera-only Omega inference, and optional fixed-shape TorchInductor
  compilation.
- `vio_vggt_pose_graph.py`: per-frame SE(3), per-window scale, stereo baseline,
  relative visual pose, gyro, sharply norm-gated gravity, loop, and motion
  regularization factors. Omega pose edges use plain least squares.
- `run_vggt_pipeline.py`: resumable prepare, tracking, preliminary graph,
  proximity-loop verification, and final graph orchestration.
- `test_vggt_pose_graph.py` and `test_vggt_loop_candidates.py`: synthetic
  convention, metric-scale, retrieval, and overlap tests.

On `testimu`, rectified stereo matching gave 0.34 px median and 1.56 px p90
vertical error. Real Omega inference processed each 32-image window in about
2.10 s eager. The loop-enabled graph used 149 poses, 53 window scales, 1,484
visual edges, and 148 composed gyro edges.

## Remaining validation

- Compare 100 and 110 degree fields of view using fixed loop pairs and
  deprojected-depth inspection.
- Add confidence-filtered Omega depth points for quantitative map inspection.
- Add external ground truth when a suitable recording becomes available.

## Runner

`run_vggt_pipeline.py` executes and times every stage. Existing window files
are validated by kind and frame identity, so interrupted inference resumes and
changed loop proposals invalidate only stale loop windows.

```bash
conda activate vggtomega
python data_processing/vio/run_vggt_pipeline.py long-test2 \
    --checkpoint /home/jkerr/checkpoints/vggt-omega/vggt_omega_1b_512.pt \
    --gpus 2,3,4,5,6,7 --tag vggt_omega_fov100 --fov-deg 100
```

## 110 degree FOV experiment

The isolated `long-test2` 110 degree run kept every other setting fixed. It
took 346 seconds for tracking, 70 seconds for the preliminary graph/proposal,
and 205 seconds for loop verification plus the final graph. Compared with 100
degrees, mean loop overlap improved from 0.353 to 0.420, graph cost from 396.5
to 287.5, mean latent baseline error from 0.0396% to 0.0251%, and worst
baseline error from 0.955% to 0.240%. However, adaptive loop selection chose a
shorter late-trajectory closure and endpoint separation was 0.671 m rather
than 0.087 m. Visual inspection and a fixed-loop comparison are needed before
changing the 100 degree default.

The direct-input 100 degree run also exposes a localized kink around frame
3032. Overlapping `track_00016` predicts adjacent rotations of 43.5 and 41.0
degrees there, versus about 6 degrees from `track_00017`. The older JPEG-based
window predicted 14.5 and 4.8 degrees, so the same weak region existed but was
less visible. A future hard overlap-consistency gate should reject such a
window-level outlier without reintroducing JPEG preprocessing or robustifying
every visual factor.
