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

A long raw capture that contains several takes can be cut into one recording
directory per take first — see
[Splitting a recording by voice command](#splitting-a-recording-by-voice-command).

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
- `split_recording.py` cuts one raw recording into several smaller recordings
  at spoken voice commands, before any pipeline runs.

The stereo convention is `X_right = R @ X_left + t`, with translation in
meters. Camera poses exported by VIO are world-to-camera transforms.

Each subfolder README contains its environment setup, commands, defaults,
outputs, and validation steps. Research dependencies stay in those dedicated
environments; `data_processing/` is intentionally outside the installable
camera-driver package.

## Splitting a recording by voice command

`split_recording.py` turns one long capture into several sibling recording
directories, cut at phrases spoken during the take. Each chunk has the same
layout and manifest format as its parent, so VIO, hands, voice, and the training
export all run on it with no path or format changes.

Transcribe the parent first — the splitter reads its transcript and refuses to
run without one:

```bash
conda activate hteng-voice
python data_processing/voice/transcribe_audio.py take01
python data_processing/split_recording.py take01 --command "next clip" --dry-run
```

`--dry-run` prints the chunk table (frame range, start, duration, keyframe snap,
matched command) without writing anything. Drop it to write
`take01_000/`, `take01_001/`, … beside the parent, or point `--out` at another
directory to keep them off the capture disk.

Two modes:

- `--mode delimiter --command PHRASE` (default) cuts contiguous chunks between
  consecutive utterances. The span before the first command is discarded unless
  `--keep-head` is passed; the span after the last is kept unless `--drop-tail`
  is.
- `--mode pair --start-command PHRASE --stop-command PHRASE` emits one chunk per
  start/stop pair and discards everything outside a pair. A start that is never
  closed is dropped with a warning, or kept to the end of the recording with
  `--unclosed extend`.

The spoken command itself is never inside a chunk: boundaries sit `--pad`
seconds (default 0.5) clear of it. Matching is exact on normalized words
(lowercased, punctuation removed) and requires every matched word to clear
`--min-probability` (default 0.5), so a low-confidence mistranscription cannot
cut a recording. Repeats chained within `--collapse` seconds of each other
become one boundary spanning the burst. Existing chunk directories are never
overwritten without `--force`.

A command spoken less than `--min-duration` plus twice `--pad` after the
previous one is ignored, and its span merges into the following chunk. The
trigger is dropped rather than the short chunk, so footage between two commands
is never silently discarded.

`--max-no-speech` (default 0.65) rejects matches inside a Whisper segment whose
`no_speech_prob` is higher than that. Whisper can lock onto background noise —
keyboard clatter in a quiet room — and loop a phrase it heard earlier, emitting
repeats with per-word probabilities as high as 0.99 that no confidence gate can
catch. Those segments do carry a high `no_speech_prob` (0.91 versus 0.3–0.5 for
real speech in the recording this was tuned against), which is what this gate
keys on. It is a strong filter, not a guarantee; check `--dry-run` output before
committing a split.

### What the split preserves

Video is stream-copied, so nothing is re-encoded or decoded and a 15-minute
recording splits in seconds. The cost is that a chunk starts on an encoded
keyframe: its first frame is the first keyframe at or after the requested
boundary, currently within 0.3 s. The realized boundary and the snap distance
are recorded in the chunk manifest. Chunk ends are frame-accurate, and left and
right are cut on a shared keyframe so the stereo pair always has equal frame
counts.

`sync_log.csv` frames are renumbered from 0, because `export_training_h5.py`
requires frame indices to be exactly `arange(video_frame_count)`. Nothing else
is rebased: `t_left_us`/`t_right_us`, IMU `host_time_us`, audio packet PTS, and
the manifest's `cam_reset_perf_counter_us` all keep their parent values. A chunk
therefore stays on the original `perf_counter` timeline, remains aligned with
its siblings, and re-transcribing its audio reproduces the parent's absolute
word times. IMU rows are kept with an extra `--imu-margin` second on each side
so frame timestamps stay bracketed for `vio_imu_prior.py`.

The parent's `derived/` products, including its transcript, are not copied into
chunks. Re-run the pipelines — including transcription — on each chunk. Each
chunk manifest gains a `source` block recording the parent path, parent frame
range, realized and requested boundaries, keyframe snap, and the matched command
text and timestamps.

`markers.csv`, from the pre-voice recorder, is not carried into chunks.

Validation:

```bash
cd data_processing/tests
PYTHONPATH=.. python -m unittest test_split_recording
```

## Running on remote GPU hosts

`sphynx` and `moggy` share the same home filesystem, so files written under
`~/hteng_camera/<recording>/derived/` on one host are immediately visible on
the other. Stages with no data dependency between them can be split across
the two hosts to run in parallel:

- WiLoR hand detection (stage 1 of `hands/`, `wilor_hands_pinhole.py`) only
  needs `left.mp4`/`right.mp4` and calibration, not the VIO trajectory. Run it
  on one host while VIO runs on the other to cut wall-clock time roughly in
  half. Stage 2 (`stereo_optimize.py`) does need the finished trajectory, so
  it must wait for VIO.
- Voice transcription is independent of both and can run anywhere, including
  locally on an Apple Silicon laptop via the `hteng-voice` conda env (create
  it once from `voice/environment-mac.yml` if it doesn't exist yet).

Conda envs live under `/home/jkerr/miniconda3/envs/` on both hosts. Check
`ls` there rather than guessing a name — e.g. the VIO/VGGT env is
`vggtomega`, not `vggtenv`. For WiLoR, both `mapanything` and `eyeball211`
have a working `wilor_mini` install.

Always check `nvidia-smi` on both hosts before picking GPUs — other jobs are
often already running on some of the 8 A6000s per box.

Launch long remote jobs with `nohup ... > log 2>&1 & disown` (or equivalent).
A dropped VPN/SSH session does not kill these; reconnect and tail the log or
`ps aux` to pick the job back up with no data loss.

Run `visualize_data.py` on the remote host too, not on the local machine —
recordings and derived products live on `sphynx`/`moggy`, so launch Viser
there and forward the port over SSH (`ssh -L <port>:localhost:<port> sphynx`)
rather than copying data down to view it locally.

`mlx-whisper` transcription is not bit-for-bit deterministic across separate
runs with identical model/settings — segment boundaries and word counts can
drift slightly between runs on the same audio. This is expected decoder
noise, not a pipeline bug.
