# data_processing

Three pipelines operate on the same calibrated stereo-fisheye, IMU, and audio
recordings:

- **`hands/`** — per-frame hand pose (WiLoR + stereo MANO bundle adjustment).
  See `hands/README.md`.
- **`vio/`** — head/camera pose (offline learned stereo-inertial pose graph).
  See `vio/README.md`.
- **`voice/`** — offline MLX/faster-whisper transcription of `audio.mka`, with
  transcript timestamps mapped onto the recording clock. See `voice/README.md`.

Run VIO first. The hand optimizer consumes its full-rate camera trajectory to
place and smooth hand poses in the world frame. Moving hands are not used as
VIO landmarks. Voice transcription is independent of VIO and hand processing,
but should run before the final training export if voice labels are wanted.

Recording inputs live at the repository root and normally include:

```text
recording.json
left.mp4
right.mp4
calib_<serial>.json
stereo_<left-serial>_<right-serial>.json
imu_log.csv
sync_log.csv
audio.mka
```

Pipeline products are written under `<recording>/derived/`.

After the desired pipelines finish, export the canonical training bundle with
`export_training_h5.py`. Dense camera/hand labels and sparse voice words are
stored in `derived/training.h5`; videos remain as separate MP4 files. The
complete schema, coordinate conventions, and efficient loader pattern are
documented in [`TRAINING_FORMAT.md`](TRAINING_FORMAT.md).

## Shared Tools

- `fisheye_pinhole.py` contains Torch implementations of the OpenCV fisheye
  camera model and virtual-pinhole rendering used by the hand pipeline.
- `visualize_data.py` is the shared Viser viewer for VIO trajectories,
  optional point clouds, loop candidates, and fitted hand meshes.
- `export_training_h5.py` consolidates VIO, hand, timing, and calibration
  outputs into a versioned HDF5 training artifact.

The stereo convention is `X_right = R @ X_left + t`, with translation in
meters. Camera poses exported by VIO are world-to-camera transforms.

Each subfolder README contains its environment setup, commands, defaults,
outputs, and validation steps. Research dependencies stay in those dedicated
environments; `data_processing/` is intentionally outside the installable
camera-driver package.
