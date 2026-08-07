#!/usr/bin/env python3
"""Split a raw recording into sub-recordings at spoken voice commands.

Each chunk is a complete recording directory with the same file layout and
manifest format as its parent, so every downstream pipeline (VIO, hands, voice,
training export) runs on it unchanged.

Video is cut with stream copy, so chunk starts snap to the nearest encoded
keyframe at or after the requested time (0.3 s granularity at the recorder's
current GOP). Nothing is re-encoded and no frame is decoded.

Frame indices are renumbered from 0 because `export_training_h5.py` requires
`sync_log.csv` frames to be exactly `arange(video_frame_count)`. Every clock is
left alone: `t_left_us`/`t_right_us`, IMU `host_time_us`, audio packet PTS, and
the manifest's `cam_reset_perf_counter_us` all keep their parent values, so a
chunk stays aligned with the original `perf_counter` timeline and with its
siblings.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


FORMAT_VERSION = "hteng-camera-recording-split/1"

DEFAULT_PAD_S = 0.5
DEFAULT_MIN_DURATION_S = 2.0
DEFAULT_MIN_PROBABILITY = 0.9
DEFAULT_MAX_NO_SPEECH = 0.6
DEFAULT_COLLAPSE_S = 1.0
DEFAULT_MAX_WORD_GAP_S = 1.0
DEFAULT_IMU_MARGIN_S = 1.0

# A chunk's audio starts on the first packet at or after its first frame. A
# larger gap than this means the parent audio has a hole there, not that the
# copy misaligned.
AUDIO_START_TOLERANCE_US = 1_000_000


# ──────────────────────────────────────────────────────────────────────────────
# Data model


@dataclass(frozen=True)
class Word:
    text: str
    normalized: str
    start_us: int
    end_us: int
    probability: float
    no_speech_prob: float  # of the segment the word belongs to


@dataclass(frozen=True)
class Trigger:
    phrase: str
    role: str  # "split", "start", or "stop"
    text: str
    start_us: int
    end_us: int
    min_probability: float


@dataclass(frozen=True)
class Window:
    """A requested time span, before any frame or keyframe quantization."""

    start_us: int | None  # None: from the first frame
    end_us: int | None  # None: to the last frame
    open_trigger: Trigger | None
    close_trigger: Trigger | None


@dataclass(frozen=True)
class Chunk:
    index: int
    name: str
    start_frame: int
    end_frame: int  # inclusive
    start_time_us: int
    end_time_us: int
    window: Window

    @property
    def frame_count(self) -> int:
        return self.end_frame - self.start_frame + 1


@dataclass(frozen=True)
class VideoIndex:
    """Packet-level index of a stream-copyable video."""

    path: Path
    pts_time_s: list[float]
    sizes: list[int]
    keyframes: list[int]

    @property
    def frame_count(self) -> int:
        return len(self.pts_time_s)

    @property
    def frame_interval_s(self) -> float:
        if len(self.pts_time_s) < 2:
            return 0.0
        return self.pts_time_s[1] - self.pts_time_s[0]


@dataclass(frozen=True)
class FrameTimes:
    """Strictly increasing absolute host times for a subset of native frames."""

    frames: list[int]
    times_us: list[int]


# ──────────────────────────────────────────────────────────────────────────────
# Subprocess helpers


def _run(command: list[str], what: str) -> str:
    try:
        proc = subprocess.run(command, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError(f"{command[0]} is required but was not found on PATH") from exc
    if proc.returncode != 0:
        detail = proc.stderr.strip().splitlines()
        tail = "\n".join(detail[-6:]) if detail else "no stderr"
        raise RuntimeError(f"{what} failed:\n{tail}")
    return proc.stdout


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"could not read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return value


# ──────────────────────────────────────────────────────────────────────────────
# Transcript and command matching


def _normalize(text: str) -> str:
    """Lowercase, drop punctuation, collapse whitespace."""
    cleaned = "".join(c if c.isalnum() else " " for c in text.lower())
    return " ".join(cleaned.split())


def _phrase_tokens(phrase: str) -> list[str]:
    tokens = _normalize(phrase).split()
    if not tokens:
        raise RuntimeError(f"command phrase is empty after normalization: {phrase!r}")
    return tokens


def _load_words(transcript_path: Path) -> list[Word]:
    transcript = _read_json(transcript_path)
    segments = transcript.get("segments")
    if not isinstance(segments, list):
        raise RuntimeError(f"{transcript_path} has no segments list")

    words: list[Word] = []
    missing_clock = False
    for segment in segments:
        no_speech_prob = float(segment.get("no_speech_prob") or 0.0)
        for raw in segment.get("words") or []:
            start = raw.get("start_perf_counter_us")
            end = raw.get("end_perf_counter_us")
            if start is None or end is None:
                missing_clock = True
                continue
            text = str(raw.get("text", ""))
            normalized = _normalize(text)
            if not normalized:
                continue
            probability = raw.get("probability")
            words.append(
                Word(
                    text=text.strip(),
                    normalized=normalized,
                    start_us=int(start),
                    end_us=int(end),
                    probability=1.0 if probability is None else float(probability),
                    no_speech_prob=no_speech_prob,
                )
            )

    if not words:
        if missing_clock:
            raise RuntimeError(
                f"{transcript_path} has no perf_counter word timestamps. Transcribe "
                "a recording directory (not a bare audio file) so the manifest can "
                "confirm clock alignment."
            )
        raise RuntimeError(f"{transcript_path} contains no timestamped words")
    if missing_clock:
        print(
            "[warn] some transcript words lack perf_counter timestamps and were "
            "ignored",
            file=sys.stderr,
        )

    words.sort(key=lambda word: word.start_us)
    return words


def find_triggers(
    words: list[Word],
    phrase: str,
    role: str,
    *,
    min_probability: float,
    max_no_speech: float,
    max_word_gap_us: int,
    collapse_us: int,
) -> list[Trigger]:
    """Locate every utterance of `phrase` in the flattened word stream.

    Matching is exact on normalized tokens. A match must also be contiguous in
    time (no more than `max_word_gap_us` between its words) so that a phrase
    cannot be assembled from words spoken far apart, and every matched word must
    clear `min_probability`.

    A match is also rejected if its segment's `no_speech_prob` exceeds
    `max_no_speech`. Whisper can loop on non-speech noise -- keyboard clatter in
    a quiet room -- and emit confident repeats of a real phrase heard earlier;
    those words carry high per-word probability, so segment no-speech is the
    signal that separates them from speech.

    Repeats chained within `collapse_us` of each other become one trigger
    spanning the whole burst, which absorbs Whisper stuttering an utterance.
    """
    tokens = _phrase_tokens(phrase)
    span = len(tokens)
    triggers: list[Trigger] = []
    index = 0
    while index + span <= len(words):
        candidate = words[index : index + span]
        if [word.normalized for word in candidate] != tokens:
            index += 1
            continue
        gaps = [
            candidate[i + 1].start_us - candidate[i].end_us for i in range(span - 1)
        ]
        if any(gap > max_word_gap_us for gap in gaps):
            index += 1
            continue
        probability = min(word.probability for word in candidate)
        if probability < min_probability:
            print(
                f"[warn] skipping {role} command {phrase!r} at "
                f"{candidate[0].start_us} us: confidence {probability:.2f} < "
                f"{min_probability:.2f}",
                file=sys.stderr,
            )
            index += span
            continue
        no_speech_prob = max(word.no_speech_prob for word in candidate)
        if no_speech_prob > max_no_speech:
            print(
                f"[warn] skipping {role} command {phrase!r} at "
                f"{candidate[0].start_us} us: segment no_speech_prob "
                f"{no_speech_prob:.2f} > {max_no_speech:.2f}",
                file=sys.stderr,
            )
            index += span
            continue
        trigger = Trigger(
            phrase=phrase,
            role=role,
            text=" ".join(word.text for word in candidate),
            start_us=candidate[0].start_us,
            end_us=candidate[-1].end_us,
            min_probability=probability,
        )
        if triggers and trigger.start_us - triggers[-1].end_us < collapse_us:
            # Chained repeat: extend the burst rather than opening a new chunk.
            triggers[-1] = Trigger(
                phrase=triggers[-1].phrase,
                role=triggers[-1].role,
                text=triggers[-1].text,
                start_us=triggers[-1].start_us,
                end_us=max(triggers[-1].end_us, trigger.end_us),
                min_probability=min(triggers[-1].min_probability, probability),
            )
            index += span
            continue
        triggers.append(trigger)
        index += span
    return triggers


# ──────────────────────────────────────────────────────────────────────────────
# Window planning


def _drop_crowded_triggers(triggers: list[Trigger], min_span_us: int) -> list[Trigger]:
    if min_span_us <= 0:
        return triggers
    kept: list[Trigger] = []
    for trigger in triggers:
        if kept and trigger.start_us - kept[-1].end_us < min_span_us:
            print(
                f"[warn] ignoring {trigger.phrase!r} at {trigger.start_us} us: "
                f"only {(trigger.start_us - kept[-1].end_us) / 1e6:.2f} s after "
                "the previous command, so its span merges into the next chunk",
                file=sys.stderr,
            )
            continue
        kept.append(trigger)
    return kept


def plan_delimiter_windows(
    triggers: list[Trigger],
    *,
    keep_head: bool,
    keep_tail: bool,
    min_span_us: int = 0,
) -> list[Window]:
    """One contiguous chunk per delimiter: trigger i to trigger i+1.

    A delimiter that lands too soon after the previous one is treated as
    spurious and dropped, merging its span into the following chunk. Dropping
    the trigger rather than the resulting short chunk means footage is never
    silently discarded between two commands.
    """
    triggers = _drop_crowded_triggers(triggers, min_span_us)
    windows: list[Window] = []
    if keep_head and triggers:
        windows.append(Window(None, triggers[0].start_us, None, triggers[0]))
    for position, trigger in enumerate(triggers):
        following = triggers[position + 1] if position + 1 < len(triggers) else None
        if following is None:
            if not keep_tail:
                continue
            windows.append(Window(trigger.end_us, None, trigger, None))
        else:
            windows.append(
                Window(trigger.end_us, following.start_us, trigger, following)
            )
    return windows


def plan_paired_windows(
    starts: list[Trigger], stops: list[Trigger], *, extend_unclosed: bool
) -> list[Window]:
    """One chunk per start/stop pair; material outside a pair is discarded."""
    events = sorted(starts + stops, key=lambda trigger: trigger.start_us)
    windows: list[Window] = []
    pending: Trigger | None = None
    for event in events:
        if event.role == "start":
            if pending is not None:
                print(
                    f"[warn] start command at {pending.start_us} us was never "
                    f"closed before the next start at {event.start_us} us; "
                    "dropping it",
                    file=sys.stderr,
                )
            pending = event
            continue
        if pending is None:
            print(
                f"[warn] stop command at {event.start_us} us has no open start; "
                "ignoring it",
                file=sys.stderr,
            )
            continue
        windows.append(Window(pending.end_us, event.start_us, pending, event))
        pending = None

    if pending is not None:
        if extend_unclosed:
            windows.append(Window(pending.end_us, None, pending, None))
        else:
            print(
                f"[warn] trailing start command at {pending.start_us} us was never "
                "closed; dropping it (use --unclosed extend to keep it)",
                file=sys.stderr,
            )
    return windows


# ──────────────────────────────────────────────────────────────────────────────
# Recording inputs


def load_frame_times(
    sync_log_path: Path, cam_reset_perf_counter_us: int, frame_count: int
) -> FrameTimes:
    """Absolute host time per native frame, filtered to a strictly increasing run.

    A camera-counter reset can leave one pre-reset `t_left_us` in the log. Those
    entries are dropped rather than repaired: this table is only used to locate
    cut points, and `sync_log.csv` itself is copied through verbatim.
    """
    frames: list[int] = []
    times: list[int] = []
    with sync_log_path.open() as handle:
        for row in csv.DictReader(handle):
            if not row.get("t_left_us"):
                continue
            frame = int(row["frame"])
            if frame >= frame_count:
                continue
            frames.append(frame)
            times.append(cam_reset_perf_counter_us + int(row["t_left_us"]))

    if not frames:
        raise RuntimeError(f"{sync_log_path} has no usable frame timestamps")

    valid = [True] * len(times)
    for i in range(len(times) - 1):
        if times[i + 1] <= times[i]:
            valid[i] = False

    kept_frames: list[int] = []
    kept_times: list[int] = []
    for frame, time_us, is_valid in zip(frames, times, valid):
        if not is_valid:
            continue
        if kept_times and time_us <= kept_times[-1]:
            continue
        kept_frames.append(frame)
        kept_times.append(time_us)

    dropped = len(frames) - len(kept_frames)
    if dropped:
        print(
            f"[warn] ignored {dropped} non-monotonic frame timestamp(s) in "
            f"{sync_log_path.name} when locating cut points",
            file=sys.stderr,
        )
    return FrameTimes(kept_frames, kept_times)


def probe_video_index(path: Path) -> VideoIndex:
    """Per-packet presentation time, size, and keyframe flag for a video file."""
    output = _run(
        [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "packet=pts_time,size,flags",
            "-of", "csv=p=0",
            str(path),
        ],
        f"ffprobe of {path.name}",
    )
    pts_time_s: list[float] = []
    sizes: list[int] = []
    keyframes: list[int] = []
    for line in output.splitlines():
        fields = line.strip().split(",")
        if len(fields) < 3:
            continue
        pts, size, flags = fields[0], fields[1], fields[2]
        if pts == "N/A":
            raise RuntimeError(f"{path} has packets without a presentation time")
        if flags.startswith("K"):
            keyframes.append(len(pts_time_s))
        pts_time_s.append(float(pts))
        sizes.append(int(size))

    if not pts_time_s:
        raise RuntimeError(f"{path} has no video packets")
    if not keyframes:
        raise RuntimeError(f"{path} has no keyframes")
    if pts_time_s != sorted(pts_time_s):
        raise RuntimeError(
            f"{path} has reordered presentation times; stream-copy splitting "
            "assumes packets are in presentation order"
        )
    return VideoIndex(path, pts_time_s, sizes, keyframes)


def probe_audio_pts_us(path: Path) -> tuple[int, int]:
    """First and last packet PTS of the audio stream, in microseconds."""
    output = _run(
        [
            "ffprobe",
            "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "packet=pts_time",
            "-of", "csv=p=0",
            str(path),
        ],
        f"ffprobe of {path.name}",
    )
    values = [
        int(round(float(line.strip().rstrip(",")) * 1e6))
        for line in output.splitlines()
        if line.strip() and not line.startswith("N/A")
    ]
    if not values:
        raise RuntimeError(f"{path} has no audio packets")
    return values[0], values[-1]


# ──────────────────────────────────────────────────────────────────────────────
# Frame resolution


def _first_frame_at_or_after(frame_times: FrameTimes, time_us: int) -> int | None:
    position = bisect.bisect_left(frame_times.times_us, time_us)
    if position >= len(frame_times.frames):
        return None
    return frame_times.frames[position]


def _last_frame_at_or_before(frame_times: FrameTimes, time_us: int) -> int | None:
    position = bisect.bisect_right(frame_times.times_us, time_us) - 1
    if position < 0:
        return None
    return frame_times.frames[position]


def _frame_time_us(frame_times: FrameTimes, frame: int) -> int:
    position = bisect.bisect_left(frame_times.frames, frame)
    if position < len(frame_times.frames) and frame_times.frames[position] == frame:
        return frame_times.times_us[position]
    # The frame's own timestamp was dropped as non-monotonic; report the nearest
    # kept neighbour so provenance still carries a usable time.
    position = min(max(position, 0), len(frame_times.frames) - 1)
    return frame_times.times_us[position]


def resolve_chunks(
    windows: list[Window],
    frame_times: FrameTimes,
    keyframes: list[int],
    frame_count: int,
    *,
    pad_us: int,
    min_duration_us: int,
    name_prefix: str,
) -> list[Chunk]:
    """Turn requested time spans into keyframe-aligned native frame ranges."""
    keyframe_list = sorted(keyframes)
    chunks: list[Chunk] = []
    for window in windows:
        if window.start_us is None:
            start_frame: int | None = frame_times.frames[0]
        else:
            requested = window.start_us + pad_us
            first = _first_frame_at_or_after(frame_times, requested)
            if first is None:
                start_frame = None
            else:
                position = bisect.bisect_left(keyframe_list, first)
                start_frame = (
                    keyframe_list[position]
                    if position < len(keyframe_list)
                    else None
                )

        if window.end_us is None:
            end_frame: int | None = min(frame_times.frames[-1], frame_count - 1)
        else:
            end_frame = _last_frame_at_or_before(frame_times, window.end_us - pad_us)

        label = _window_label(window)
        if start_frame is None or end_frame is None:
            print(f"[warn] dropping chunk {label}: no frames in range", file=sys.stderr)
            continue
        end_frame = min(end_frame, frame_count - 1)
        if end_frame < start_frame:
            print(
                f"[warn] dropping chunk {label}: commands are closer together "
                "than the keyframe interval plus padding",
                file=sys.stderr,
            )
            continue

        start_time_us = _frame_time_us(frame_times, start_frame)
        end_time_us = _frame_time_us(frame_times, end_frame)
        duration_us = end_time_us - start_time_us
        if duration_us < min_duration_us:
            print(
                f"[warn] dropping chunk {label}: {duration_us / 1e6:.2f} s is "
                f"shorter than --min-duration {min_duration_us / 1e6:.2f} s",
                file=sys.stderr,
            )
            continue

        index = len(chunks)
        chunks.append(
            Chunk(
                index=index,
                name=f"{name_prefix}_{index:03d}",
                start_frame=start_frame,
                end_frame=end_frame,
                start_time_us=start_time_us,
                end_time_us=end_time_us,
                window=window,
            )
        )
    return chunks


def _window_label(window: Window) -> str:
    if window.open_trigger is not None:
        return f"after {window.open_trigger.text!r} @ {window.open_trigger.start_us} us"
    if window.close_trigger is not None:
        return f"before {window.close_trigger.text!r} @ {window.close_trigger.start_us} us"
    return "whole recording"


# ──────────────────────────────────────────────────────────────────────────────
# Writers


def cut_video(
    source: VideoIndex, destination: Path, start_frame: int, frame_count: int
) -> None:
    """Stream-copy `frame_count` frames starting at a keyframe.

    The seek target sits just after the keyframe's presentation time so ffmpeg's
    "keyframe at or before" input seek lands on that keyframe and not the
    previous one.
    """
    if start_frame not in set(source.keyframes):
        raise RuntimeError(
            f"{source.path.name}: frame {start_frame} is not a keyframe"
        )
    seek_s = source.pts_time_s[start_frame] + 0.4 * source.frame_interval_s
    command = ["ffmpeg", "-nostdin", "-v", "error", "-y"]
    if start_frame:
        command += ["-ss", f"{seek_s:.6f}"]
    command += [
        "-i", str(source.path),
        "-map", "0",
        "-c", "copy",
        "-frames:v", str(frame_count),
        "-avoid_negative_ts", "make_zero",
        str(destination),
    ]
    _run(command, f"stream copy of {source.path.name}")

    written = probe_video_index(destination)
    expected = source.sizes[start_frame : start_frame + frame_count]
    if written.sizes != expected:
        raise RuntimeError(
            f"{destination.name}: stream copy produced {written.frame_count} "
            f"packets that do not match source frames "
            f"{start_frame}..{start_frame + frame_count - 1}; the input seek "
            "did not land on the intended keyframe"
        )
    if written.keyframes[0] != 0:
        raise RuntimeError(f"{destination.name}: does not start on a keyframe")


def cut_audio(
    source: Path,
    destination: Path,
    audio_start_pts_us: int,
    start_time_us: int,
    end_time_us: int,
) -> tuple[int, int]:
    """Stream-copy the audio span, preserving absolute packet PTS.

    `-copyts` keeps the AVFoundation host timestamps the manifest promises, so a
    chunk's audio stays on the parent's `perf_counter` timeline and can be
    re-transcribed without re-deriving the clock mapping.

    The input seek only reaches the Matroska cluster at or before the requested
    time, which would prepend up to a cluster of audio from before the chunk --
    possibly the tail of the spoken command. The output-side `-ss`/`-to` drops
    those packets exactly, and `-output_ts_offset` restores the absolute values
    that the output seek subtracts.
    """
    relative_start_s = max(0.0, (start_time_us - audio_start_pts_us) / 1e6)
    if end_time_us <= start_time_us:
        raise RuntimeError(f"{destination.name}: non-positive audio duration")
    absolute_start_s = start_time_us / 1e6
    command = [
        "ffmpeg", "-nostdin", "-v", "error", "-y",
        "-copyts",
        "-ss", f"{relative_start_s:.6f}",
        "-i", str(source),
        "-map", "0:a:0",
        "-c", "copy",
        "-ss", f"{absolute_start_s:.6f}",
        "-to", f"{end_time_us / 1e6:.6f}",
        "-output_ts_offset", f"{absolute_start_s:.6f}",
        "-avoid_negative_ts", "disabled",
        str(destination),
    ]
    _run(command, f"stream copy of {source.name}")

    written_start_us, written_end_us = probe_audio_pts_us(destination)
    drift_us = written_start_us - start_time_us
    if drift_us < 0 or drift_us > AUDIO_START_TOLERANCE_US:
        raise RuntimeError(
            f"{destination.name}: first audio packet is at {written_start_us} us "
            f"but the chunk starts at {start_time_us} us; the copied timeline is "
            "not aligned with the parent recording"
        )
    if written_end_us > end_time_us:
        raise RuntimeError(
            f"{destination.name}: audio extends past the chunk's last frame"
        )
    return written_start_us, written_end_us


def _rewrite_csv(
    destination: Path, rows: list[dict[str, str]], fieldnames: list[str]
) -> None:
    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def slice_sync_log(source: Path, destination: Path, chunk: Chunk) -> int:
    """Copy the chunk's rows verbatim, renumbering only the frame column."""
    rows: list[dict[str, str]] = []
    with source.open() as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        for row in reader:
            frame = int(row["frame"])
            if frame < chunk.start_frame or frame > chunk.end_frame:
                continue
            row = dict(row)
            row["frame"] = str(frame - chunk.start_frame)
            rows.append(row)
    if len(rows) != chunk.frame_count:
        raise RuntimeError(
            f"{source.name} has {len(rows)} rows for frames "
            f"{chunk.start_frame}..{chunk.end_frame} but the chunk has "
            f"{chunk.frame_count} frames"
        )
    _rewrite_csv(destination, rows, fieldnames)
    return len(rows)


def slice_imu_log(
    source: Path, destination: Path, start_time_us: int, end_time_us: int
) -> tuple[int, int, int]:
    """Copy IMU samples covering the chunk, renumbering the sample column.

    The window is padded so `vio_imu_prior.py` still finds samples bracketing the
    first and last frame timestamps.
    """
    rows: list[dict[str, str]] = []
    first_us: int | None = None
    last_us = 0
    with source.open() as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        for row in reader:
            host_time_us = int(row["host_time_us"])
            if host_time_us < start_time_us or host_time_us > end_time_us:
                continue
            row = dict(row)
            row["sample"] = str(len(rows))
            rows.append(row)
            if first_us is None:
                first_us = host_time_us
            last_us = host_time_us
    if not rows:
        raise RuntimeError(
            f"{source.name} has no samples between {start_time_us} and "
            f"{end_time_us} us"
        )
    _rewrite_csv(destination, rows, fieldnames)
    return len(rows), int(first_us or 0), last_us


def write_manifest(
    manifest: dict[str, Any],
    destination: Path,
    source_block: dict[str, Any],
    present: set[str],
) -> None:
    """Write the chunk's manifest: the parent's, plus split provenance."""
    chunk_manifest = json.loads(json.dumps(manifest))  # deep copy
    files = dict(chunk_manifest.get("files", {}))
    files.pop("markers", None)  # superseded by voice commands
    for key in ("audio", "imu_log"):
        if key not in present:
            files.pop(key, None)
    chunk_manifest["files"] = files
    # A stream the parent never captured is not enabled for the chunk either.
    for key, block in (("audio", "audio"), ("imu_log", "imu")):
        if key not in present and isinstance(chunk_manifest.get(block), dict):
            chunk_manifest[block]["enabled"] = False
    chunk_manifest.pop("markers", None)
    chunk_manifest["source"] = source_block
    destination.write_text(json.dumps(chunk_manifest, indent=2) + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# Driver


def _resolve_file(recording: Path, manifest: dict[str, Any], key: str, default: str) -> Path:
    files = manifest.get("files") or {}
    name = str(files.get(key, default)) if isinstance(files, dict) else default
    return recording / name


def _describe_trigger(trigger: Trigger | None) -> dict[str, Any] | None:
    if trigger is None:
        return None
    return {
        "role": trigger.role,
        "phrase": trigger.phrase,
        "text": trigger.text,
        "start_perf_counter_us": trigger.start_us,
        "end_perf_counter_us": trigger.end_us,
        "min_word_probability": trigger.min_probability,
    }


def _print_plan(chunks: list[Chunk], frame_times: FrameTimes, pad_us: int) -> None:
    origin_us = frame_times.times_us[0]
    print(f"{'idx':>3}  {'name':<24} {'frames':>18} {'start_s':>9} "
          f"{'dur_s':>8} {'snap_ms':>8}  command")
    for chunk in chunks:
        requested = (
            chunk.window.start_us + pad_us
            if chunk.window.start_us is not None
            else chunk.start_time_us
        )
        snap_ms = (chunk.start_time_us - requested) / 1000.0
        span = f"{chunk.start_frame}-{chunk.end_frame} ({chunk.frame_count})"
        opened = chunk.window.open_trigger
        command = f"{opened.text!r}" if opened else "<recording start>"
        print(
            f"{chunk.index:>3}  {chunk.name:<24} {span:>18} "
            f"{(chunk.start_time_us - origin_us) / 1e6:>9.2f} "
            f"{(chunk.end_time_us - chunk.start_time_us) / 1e6:>8.2f} "
            f"{snap_ms:>8.0f}  {command}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("recording", type=Path, help="raw recording directory")
    parser.add_argument(
        "--mode",
        choices=("delimiter", "pair"),
        default="delimiter",
        help="delimiter: one contiguous chunk between consecutive commands. "
             "pair: one chunk per start/stop command pair (default: delimiter)",
    )
    parser.add_argument(
        "--command",
        help="delimiter mode: the spoken phrase that separates chunks",
    )
    parser.add_argument("--start-command", help="pair mode: phrase that opens a chunk")
    parser.add_argument("--stop-command", help="pair mode: phrase that closes a chunk")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="directory to write chunk recordings into "
             "(default: the recording's parent directory)",
    )
    parser.add_argument(
        "--transcript",
        type=Path,
        default=None,
        help="default: <recording>/derived/voice_transcript.json",
    )
    parser.add_argument(
        "--pad",
        type=float,
        default=DEFAULT_PAD_S,
        help=f"seconds of silence to leave between a spoken command and the "
             f"chunk boundary (default: {DEFAULT_PAD_S})",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=DEFAULT_MIN_DURATION_S,
        help=f"discard chunks shorter than this many seconds "
             f"(default: {DEFAULT_MIN_DURATION_S})",
    )
    parser.add_argument(
        "--min-probability",
        type=float,
        default=DEFAULT_MIN_PROBABILITY,
        help=f"reject command matches whose least confident word falls below "
             f"this Whisper probability (default: {DEFAULT_MIN_PROBABILITY})",
    )
    parser.add_argument(
        "--max-no-speech",
        type=float,
        default=DEFAULT_MAX_NO_SPEECH,
        help=f"reject command matches inside a Whisper segment whose "
             f"no_speech_prob exceeds this, which is how confident repeats "
             f"hallucinated from background noise are caught "
             f"(default: {DEFAULT_MAX_NO_SPEECH})",
    )
    parser.add_argument(
        "--collapse",
        type=float,
        default=DEFAULT_COLLAPSE_S,
        help=f"treat repeat matches within this many seconds as one command "
             f"(default: {DEFAULT_COLLAPSE_S})",
    )
    parser.add_argument(
        "--max-word-gap",
        type=float,
        default=DEFAULT_MAX_WORD_GAP_S,
        help=f"maximum silence between words of a multi-word command "
             f"(default: {DEFAULT_MAX_WORD_GAP_S})",
    )
    parser.add_argument(
        "--imu-margin",
        type=float,
        default=DEFAULT_IMU_MARGIN_S,
        help=f"extra IMU seconds kept on each side of a chunk so frame "
             f"timestamps stay bracketed (default: {DEFAULT_IMU_MARGIN_S})",
    )
    parser.add_argument(
        "--keep-head",
        action="store_true",
        help="delimiter mode: also emit the span before the first command",
    )
    parser.add_argument(
        "--drop-tail",
        action="store_true",
        help="delimiter mode: discard the span after the last command",
    )
    parser.add_argument(
        "--unclosed",
        choices=("drop", "extend"),
        default="drop",
        help="pair mode: what to do with a start command that is never closed "
             "(default: drop)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the chunk plan without writing anything",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite existing chunk directories",
    )
    args = parser.parse_args(argv)

    try:
        return _split(args)
    except RuntimeError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1


def _split(args: argparse.Namespace) -> int:
    recording = args.recording.expanduser().resolve()
    if not recording.is_dir():
        raise RuntimeError(f"not a directory: {recording}")

    manifest_path = recording / "recording.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"missing manifest: {manifest_path}")
    manifest = _read_json(manifest_path)

    imu = manifest.get("imu") or {}
    cam_reset_us = imu.get("cam_reset_perf_counter_us")
    if cam_reset_us is None:
        raise RuntimeError(
            "recording.json has no imu.cam_reset_perf_counter_us, so camera "
            "frames cannot be placed on the transcript's perf_counter timeline"
        )
    cam_reset_us = int(cam_reset_us)

    transcript_path = args.transcript or (recording / "derived" / "voice_transcript.json")
    if not transcript_path.is_file():
        raise RuntimeError(
            f"missing transcript: {transcript_path}\n"
            "        Run data_processing/voice/transcribe_audio.py on this "
            "recording first."
        )

    left_path = _resolve_file(recording, manifest, "left", "left.mp4")
    right_path = _resolve_file(recording, manifest, "right", "right.mp4")
    sync_log_path = _resolve_file(recording, manifest, "sync_log", "sync_log.csv")
    imu_log_path = _resolve_file(recording, manifest, "imu_log", "imu_log.csv")
    audio_path = _resolve_file(recording, manifest, "audio", "audio.mka")
    for required in (left_path, right_path, sync_log_path):
        if not required.is_file():
            raise RuntimeError(f"missing required input: {required}")

    words = _load_words(transcript_path)
    collapse_us = int(args.collapse * 1e6)
    max_word_gap_us = int(args.max_word_gap * 1e6)

    if args.mode == "delimiter":
        if not args.command:
            raise RuntimeError("--command is required in delimiter mode")
        if args.start_command or args.stop_command:
            raise RuntimeError(
                "--start-command/--stop-command apply to --mode pair"
            )
        triggers = find_triggers(
            words,
            args.command,
            "split",
            min_probability=args.min_probability,
            max_no_speech=args.max_no_speech,
            max_word_gap_us=max_word_gap_us,
            collapse_us=collapse_us,
        )
        if not triggers:
            raise RuntimeError(
                f"no utterance of {args.command!r} found in {transcript_path.name}"
            )
        print(f"[info] found {len(triggers)} command(s) {args.command!r}")
        windows = plan_delimiter_windows(
            triggers,
            keep_head=args.keep_head,
            keep_tail=not args.drop_tail,
            min_span_us=int((args.min_duration + 2 * args.pad) * 1e6),
        )
    else:
        if not args.start_command or not args.stop_command:
            raise RuntimeError(
                "--start-command and --stop-command are required in pair mode"
            )
        if args.command:
            raise RuntimeError("--command applies to --mode delimiter")
        starts = find_triggers(
            words,
            args.start_command,
            "start",
            min_probability=args.min_probability,
            max_no_speech=args.max_no_speech,
            max_word_gap_us=max_word_gap_us,
            collapse_us=collapse_us,
        )
        stops = find_triggers(
            words,
            args.stop_command,
            "stop",
            min_probability=args.min_probability,
            max_no_speech=args.max_no_speech,
            max_word_gap_us=max_word_gap_us,
            collapse_us=collapse_us,
        )
        if not starts:
            raise RuntimeError(
                f"no utterance of {args.start_command!r} found in "
                f"{transcript_path.name}"
            )
        print(
            f"[info] found {len(starts)} start and {len(stops)} stop command(s)"
        )
        windows = plan_paired_windows(
            starts, stops, extend_unclosed=args.unclosed == "extend"
        )

    if not windows:
        raise RuntimeError("no chunks to write")

    print(f"[info] indexing {left_path.name} and {right_path.name}")
    left_index = probe_video_index(left_path)
    right_index = probe_video_index(right_path)
    if left_index.frame_count != right_index.frame_count:
        raise RuntimeError(
            f"stereo videos have different frame counts: "
            f"{left_index.frame_count} != {right_index.frame_count}"
        )
    shared_keyframes = sorted(set(left_index.keyframes) & set(right_index.keyframes))
    if not shared_keyframes:
        raise RuntimeError("left and right videos share no keyframe positions")
    if len(shared_keyframes) < len(left_index.keyframes):
        print(
            f"[warn] only {len(shared_keyframes)} of {len(left_index.keyframes)} "
            "left keyframes are also right keyframes; cutting on the shared set",
            file=sys.stderr,
        )

    frame_times = load_frame_times(sync_log_path, cam_reset_us, left_index.frame_count)
    chunks = resolve_chunks(
        windows,
        frame_times,
        shared_keyframes,
        left_index.frame_count,
        pad_us=int(args.pad * 1e6),
        min_duration_us=int(args.min_duration * 1e6),
        name_prefix=recording.name,
    )
    if not chunks:
        raise RuntimeError("every candidate chunk was dropped")

    _print_plan(chunks, frame_times, int(args.pad * 1e6))
    if args.dry_run:
        print("[info] dry run: nothing written")
        return 0

    out_root = (args.out or recording.parent).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    existing = [c.name for c in chunks if (out_root / c.name).exists()]
    if existing and not args.force:
        raise RuntimeError(
            "chunk directories already exist: " + ", ".join(existing) +
            "\n        Pass --force to overwrite them."
        )

    has_audio = audio_path.is_file()
    has_imu = imu_log_path.is_file()
    audio_start_pts_us = probe_audio_pts_us(audio_path)[0] if has_audio else 0
    imu_margin_us = int(args.imu_margin * 1e6)
    calibrations = sorted(recording.glob("calib_*.json")) + sorted(
        recording.glob("stereo_*.json")
    )

    for chunk in chunks:
        destination = out_root / chunk.name
        if destination.exists():
            shutil.rmtree(destination)
        destination.mkdir(parents=True)
        print(
            f"[info] {chunk.name}: frames {chunk.start_frame}-{chunk.end_frame} "
            f"({chunk.frame_count}), "
            f"{(chunk.end_time_us - chunk.start_time_us) / 1e6:.2f} s"
        )

        cut_video(
            left_index, destination / left_path.name, chunk.start_frame, chunk.frame_count
        )
        cut_video(
            right_index, destination / right_path.name, chunk.start_frame, chunk.frame_count
        )
        slice_sync_log(sync_log_path, destination / sync_log_path.name, chunk)

        present = {"left", "right", "sync_log", "stereo_transform"}
        audio_range: dict[str, int] | None = None
        if has_audio:
            start_pts, end_pts = cut_audio(
                audio_path,
                destination / audio_path.name,
                audio_start_pts_us,
                chunk.start_time_us,
                chunk.end_time_us,
            )
            audio_range = {"start_pts_us": start_pts, "end_pts_us": end_pts}
            present.add("audio")

        imu_range: dict[str, int] | None = None
        if has_imu:
            count, first_us, last_us = slice_imu_log(
                imu_log_path,
                destination / imu_log_path.name,
                chunk.start_time_us - imu_margin_us,
                chunk.end_time_us + imu_margin_us,
            )
            imu_range = {
                "sample_count": count,
                "start_host_time_us": first_us,
                "end_host_time_us": last_us,
            }
            present.add("imu_log")

        for calibration in calibrations:
            shutil.copy2(calibration, destination / calibration.name)

        requested_start_us = (
            chunk.window.start_us + int(args.pad * 1e6)
            if chunk.window.start_us is not None
            else chunk.start_time_us
        )
        source_block = {
            "format": FORMAT_VERSION,
            "tool": "data_processing/split_recording.py",
            "recording": recording.name,
            "recording_path": str(recording),
            "chunk_index": chunk.index,
            "chunk_count": len(chunks),
            "split_mode": args.mode,
            "pad_s": args.pad,
            "imu_margin_s": args.imu_margin,
            "min_word_probability": args.min_probability,
            "max_no_speech_prob": args.max_no_speech,
            "frames": {
                "parent_start": chunk.start_frame,
                "parent_end": chunk.end_frame,
                "count": chunk.frame_count,
            },
            "frame_time_us": {
                "start": chunk.start_time_us,
                "end": chunk.end_time_us,
            },
            "requested_start_perf_counter_us": requested_start_us,
            "keyframe_snap_us": chunk.start_time_us - requested_start_us,
            "triggers": {
                "open": _describe_trigger(chunk.window.open_trigger),
                "close": _describe_trigger(chunk.window.close_trigger),
            },
            "audio": audio_range,
            "imu": imu_range,
            "note": (
                "Video was stream-copied, so this chunk starts on a parent "
                "keyframe and no frame was re-encoded. sync_log.csv frames are "
                "renumbered from 0, but t_left_us/t_right_us, IMU host_time_us, "
                "and audio packet PTS are unchanged from the parent recording. "
                "imu.cam_reset_perf_counter_us above therefore still maps this "
                "chunk's frames onto the original perf_counter timeline, and "
                "sibling chunks remain mutually aligned. derived/ products of "
                "the parent are not carried over; re-run the pipelines, "
                "including voice transcription, on this directory."
            ),
        }
        write_manifest(manifest, destination / "recording.json", source_block, present)

    print(f"[info] wrote {len(chunks)} chunk(s) under {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
