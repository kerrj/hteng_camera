# Voice transcription

`transcribe_audio.py` transcribes a recording's `audio.mka` offline and writes
the canonical derived artifact:

```text
<recording>/derived/voice_transcript.json
```

It does not access the microphone or run during capture.

## Apple Silicon setup

Use a dedicated environment:

```bash
conda env create -f data_processing/voice/environment-mac.yml
conda activate hteng-voice
```

The default `large-v3` model is roughly 3 GB and still transcribes faster than
real time on an Apple Silicon laptop. Smaller models are noticeably worse on
isolated spoken commands: `small` both misses real commands and hallucinates
phrases into silence. The environment occupies about 1.2 GB before model weights
because mlx-whisper currently depends on Torch, SciPy, and Numba in addition to
MLX. Model weights are downloaded from Hugging Face on first use and cached.
`ffmpeg` and `ffprobe` must be available on `PATH`.

The checked-in environment pins the MLX 0.32.0 and mlx-whisper 0.4.3 versions
tested on an Apple M4. To update an existing environment to that definition:

```bash
conda env update -n hteng-voice \
  -f data_processing/voice/environment-mac.yml --prune
```

## Run

Pass the recording directory, not merely its MKA, for the standard pipeline:

```bash
conda activate hteng-voice
python data_processing/voice/transcribe_audio.py take01
```

This selects MLX automatically on Apple Silicon and uses the Whisper `large-v3`
model. Override it with `--model` if you need a smaller download, at a real
accuracy cost on command recognition:

```bash
python data_processing/voice/transcribe_audio.py take01 \
  --model mlx-community/whisper-large-v3-turbo
```

Passing a standalone audio file is supported for scratch tests. Its output is
written beside the input as `<stem>.transcript.json` because there is no
recording-level `derived/` directory.

## CUDA

For an NVIDIA machine, install faster-whisper in that machine's environment:

```bash
python -m pip install faster-whisper
python data_processing/voice/transcribe_audio.py take01 \
  --backend faster-whisper --device cuda
```

The CUDA path defaults to FP16. CPU faster-whisper defaults to INT8. These
defaults can be overridden with `--compute-type`. The CUDA adapter is
implemented but has not yet been exercised in this repository.

## Output and timing

The JSON contains normalized full text plus segment and word tables. Segment
and word times include:

- `start_audio_s` / `end_audio_s`: time relative to decoded audio.
- `start_pts_us` / `end_pts_us`: time on the original MKA packet timeline.
- `start_perf_counter_us` / `end_perf_counter_us`: emitted only when the
  recording manifest confirms that AVFoundation PTS uses the same monotonic
  clock as `time.perf_counter()`.

This makes recognized commands directly comparable to camera times using the
recording's existing `cam_reset_perf_counter_us` mapping. Passing a standalone
media file does not claim that its PTS is a host clock; use
`--timestamp-origin perf-counter` only when that relationship is independently
known.

The final training exporter reads this file automatically and stores its
segments and words under `/voice` in `<recording>/derived/training.h5`.
Timestamped words include Whisper confidence, absolute host time, and nearest
native video frame indices. Run the exporter after transcription:

```bash
python data_processing/export_training_h5.py take01 \
  --trajectory take01/derived/trajectory_<tag>.npz
```

## Validation

Run the dependency-free voice tests from this directory:

```bash
cd data_processing/voice
python -m unittest -v test_transcribe_audio.py
```

Run the HDF5 integration test from the repository root:

```bash
python data_processing/test_training_h5.py
```
