# Training data format

Each processed recording can be finalized as:

```text
<recording>/
  left.mp4
  right.mp4
  audio.mka
  derived/voice_transcript.json
  derived/training.h5
```

The MP4 files remain separate so TorchCodec or another video decoder can decode
them directly. The raw MKA also remains separate. `training.h5` contains
frame-aligned numeric labels, sparse voice labels, calibration, timing, and
provenance. Its root attributes include
`format="hteng-camera-training"` and an integer `schema_version`; consumers
must reject unsupported schema versions.

## Export

Run this after VIO and any desired hand and voice processing:

```bash
python data_processing/export_training_h5.py long-test2 \
  --trajectory long-test2/derived/trajectory_vggt_omega_fullrun_20260714.npz
```

The defaults read:

```text
<recording>/derived/imu_relative.npz
<recording>/derived/hands3d_left.jsonl
<recording>/derived/hands3d_right.jsonl
<recording>/derived/voice_transcript.json
```

and write `<recording>/derived/training.h5`. Use `--overwrite` to replace an
existing export. The writer creates a temporary file and atomically renames it
only after a successful export. Hands, IMU, and voice inputs are optional. Pass
`--voice PATH` to override the default transcript.

## Conventions

- Frame indices are zero-based native MP4 frame indices.
- Lengths are meters and timestamps are microseconds.
- Camera coordinates use OpenCV axes: +x right, +y down, +z forward.
- A transform named `A_from_B` maps homogeneous coordinates from frame B into
  frame A.
- Rotations stored as quaternions use `wxyz` component order.
- Hand joints use the 21-joint MANO ordering produced by WiLoR/MANO.
- Invalid dense hand rows contain NaNs and must be selected with `valid`.
- Interpolated hand rows are valid estimates and are additionally marked by
  `interpolated`.

## Schema version 2

All frame-aligned datasets have leading dimension `N`, the number of frames
decodable from each MP4. The exporter requires equal left/right video counts
and one trajectory pose for every encoded frame. A trailing capture present in
the sync log or trajectory but absent from both finalized MP4s is discarded.
Camera axis 1 is ordered `[left, right]`.

```text
/recording_json                              UTF-8 scalar

/frames/index                               int64   [N]
/frames/time_us                             int64   [N]
/frames/time_s                              float64 [N]
/frames/time_valid                          bool    [N]

/cameras/names                              UTF-8   [2]
/cameras/serials                            UTF-8   [2]
/cameras/video_files                        UTF-8   [2]
/cameras/camera_from_world                  float32 [N, 2, 4, 4]
/cameras/world_from_camera                  float32 [N, 2, 4, 4]
/cameras/K                                  float32 [2, 3, 3]
/cameras/distortion                         float32 [2, 4]
/cameras/image_size                         int32   [2, 2]  # width, height
/cameras/right_from_left                    float32 [4, 4]

/hands/{left,right}/valid                   bool    [N]
/hands/{left,right}/interpolated            bool    [N]
/hands/{left,right}/source_frames           int64   [N, 2]
/hands/{left,right}/betas                   float32 [10]
/hands/{left,right}/root_camera_m           float32 [N, 3]
/hands/{left,right}/root_world_m            float32 [N, 3]
/hands/{left,right}/joints_camera_m         float32 [N, 21, 3]
/hands/{left,right}/joints_world_m          float32 [N, 21, 3]
/hands/{left,right}/wrist_rotation_camera   float32 [N, 3, 3]
/hands/{left,right}/wrist_rotation_world    float32 [N, 3, 3]
/hands/{left,right}/joint_quaternion_wxyz   float32 [N, 16, 4]
/hands/{left,right}/translation_virtual_m   float32 [N, 3]
/hands/{left,right}/virtual_to_camera_rotation
                                                float32 [N, 3, 3]
/hands/{left,right}/phase1_mean_reprojection_px
                                                float32 [N]
/hands/{left,right}/phase1_p90_reprojection_px
                                                float32 [N]
/hands/{left,right}/phase1_median_epipolar_px
                                                float32 [N]
/hands/{left,right}/phase1_depth_m           float32 [N]

/voice/transcript                            UTF-8 scalar
/voice/segments/text                         UTF-8   [S]
/voice/segments/id                           int64   [S]
/voice/segments/{start,end}_audio_us         int64   [S]
/voice/segments/{start,end}_pts_us           int64   [S]
/voice/segments/{start,end}_time_us          int64   [S]
/voice/segments/{start,end}_frame_index      int64   [S]

/voice/words/text                            UTF-8   [W]
/voice/words/id                              int64   [W]
/voice/words/probability                     float32 [W]
/voice/words/segment_index                   int64   [W]
/voice/words/{start,end}_audio_us            int64   [W]
/voice/words/{start,end}_pts_us              int64   [W]
/voice/words/{start,end}_time_us             int64   [W]
/voice/words/{start,end}_frame_index         int64   [W]
```

Every dataset has an embedded `description` attribute. Applicable datasets also
have a `units` attribute. The root records source trajectory provenance, and
each hand group preserves the optimizer's JSON metadata and quality thresholds.
If a recording has no accepted poses for one hand, that group remains present
with an all-false validity mask and no `betas` dataset.

The `/voice` group is always present. Its `available` attribute is false and
its tables are empty when `derived/voice_transcript.json` is absent. `S` is the
number of Whisper segments and `W` is the number of timestamped words.
`*_audio_us` is relative to the first decoded audio sample, `*_pts_us` retains
the original MKA timeline, and `*_time_us` is absolute `perf_counter` time.
Absolute values and nearest frame indices are `-1` if clock alignment was not
available. Frame indices identify the native video frame nearest each word or
segment boundary; the original timestamps remain authoritative.

Schema version 1 predates the `/voice` group.

`frames/time_s` is the monotonic relative observed-camera timeline used for VIO
pose interpolation and hand smoothing. It preserves real capture gaps and
repairs only nonpositive camera-counter-reset intervals with the recording's
median positive interval. `frames/time_us` preserves the original host-clock
timestamps; consult `frames/time_valid` before using those absolute values.

## Efficient loading

Frame-aligned numeric datasets are LZF-compressed and chunked in blocks of 256
frames. Slice only the requested range instead of loading the full file:

```python
import h5py

with h5py.File("long-test2/derived/training.h5", "r") as data:
    start, stop = 1024, 1280
    camera_pose = data["cameras/world_from_camera"][start:stop]
    joints = data["hands/right/joints_world_m"][start:stop]
    valid = data["hands/right/valid"][start:stop]

    words = data["voice/words/text"].asstr()[:]
    word_start_frames = data["voice/words/start_frame_index"][:]
```

PyTorch `DataLoader` workers must each open their own HDF5 handle after the
worker starts. Do not create one shared `h5py.File` before forking. For random
training samples, group nearby frame requests by chunk where practical; video
decode, rather than HDF5 labels, will usually dominate input time.
