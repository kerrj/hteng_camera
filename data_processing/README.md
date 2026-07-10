# data_processing

Two independent pipelines over the stereo fisheye + IMU recordings (e.g.
`long-test1/`):

- **`hands/`** — per-frame hand pose (WiLoR + stereo MANO bundle adjustment).
  See `hands/README.md`.
- **`vio/`** — head/camera pose (offline stereo-inertial bundle adjustment).

`fisheye_pinhole.py` (fisheye camera-model math) is shared by both and lives
here at the top level; everything pipeline-specific lives in its subfolder.
