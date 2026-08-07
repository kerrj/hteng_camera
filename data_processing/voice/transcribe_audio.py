#!/usr/bin/env python3
"""Transcribe a recording's audio while preserving its container timeline."""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import Any


FORMAT_VERSION = "hteng-camera-voice-transcript/1"
MLX_DEFAULT_MODEL = "mlx-community/whisper-large-v3-mlx"
FASTER_DEFAULT_MODEL = "large-v3"


@dataclass(frozen=True)
class AudioInput:
    path: Path
    recording_dir: Path | None
    manifest: dict[str, Any] | None


@dataclass(frozen=True)
class AudioProbe:
    codec: str
    sample_rate_hz: int
    channels: int
    start_pts_us: int
    duration_s: float | None


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"could not read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return value


def _resolve_audio_input(value: Path) -> AudioInput:
    value = value.expanduser()
    if value.is_dir():
        recording_dir = value
        manifest_path = recording_dir / "recording.json"
        manifest = _read_json(manifest_path) if manifest_path.exists() else None
        audio_name = "audio.mka"
        if manifest is not None:
            files = manifest.get("files", {})
            if isinstance(files, dict) and files.get("audio"):
                audio_name = str(files["audio"])
        audio_path = recording_dir / audio_name
    else:
        audio_path = value
        manifest_path = audio_path.parent / "recording.json"
        manifest = _read_json(manifest_path) if manifest_path.exists() else None
        recording_dir = audio_path.parent if manifest is not None else None

    if not audio_path.is_file():
        raise RuntimeError(f"audio file does not exist: {audio_path}")
    return AudioInput(
        path=audio_path.resolve(),
        recording_dir=recording_dir.resolve() if recording_dir else None,
        manifest=manifest,
    )


def _decimal_us(value: Any, field: str) -> int:
    try:
        seconds = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise RuntimeError(f"invalid ffprobe {field}: {value!r}") from exc
    return int(
        (seconds * Decimal(1_000_000)).to_integral_value(
            rounding=ROUND_HALF_UP
        )
    )


def _relative_duration_s(value: Any, start_pts_us: int) -> float | None:
    if value in (None, "N/A"):
        return None
    try:
        reported_us = _decimal_us(value, "duration")
    except RuntimeError:
        return None
    # Matroska reports the end of the timeline as its duration when packet PTS
    # starts at an absolute host-clock value.
    if start_pts_us > 0 and reported_us >= start_pts_us:
        reported_us -= start_pts_us
    return reported_us / 1_000_000


def _probe_audio(path: Path) -> AudioProbe:
    command = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "a:0",
        "-show_entries",
        "stream=codec_name,sample_rate,channels,start_time,duration:"
        "format=duration:packet=pts_time",
        "-show_packets",
        "-read_intervals", "%+#1",
        "-of", "json",
        str(path),
    ]
    try:
        proc = subprocess.run(command, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError("ffprobe is required but was not found on PATH") from exc
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "ffprobe failed")
    try:
        data = json.loads(proc.stdout)
        stream = data["streams"][0]
    except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"could not parse ffprobe output for {path}") from exc

    packets = data.get("packets") or []
    start_value = packets[0].get("pts_time") if packets else None
    if start_value is None:
        start_value = stream.get("start_time")
    if start_value in (None, "N/A"):
        raise RuntimeError(f"audio stream in {path} has no start PTS")

    file_format = data.get("format") or {}
    duration_value = stream.get("duration")
    if duration_value in (None, "N/A"):
        duration_value = file_format.get("duration")
    start_pts_us = _decimal_us(start_value, "start PTS")
    duration_s = _relative_duration_s(duration_value, start_pts_us)
    try:
        return AudioProbe(
            codec=str(stream["codec_name"]),
            sample_rate_hz=int(stream["sample_rate"]),
            channels=int(stream["channels"]),
            start_pts_us=start_pts_us,
            duration_s=duration_s,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"incomplete audio stream metadata for {path}") from exc


def _manifest_confirms_perf_counter(
    audio_input: AudioInput,
) -> bool:
    manifest = audio_input.manifest
    if manifest is None:
        return False
    audio = manifest.get("audio")
    if not isinstance(audio, dict) or not audio.get("enabled"):
        return False
    alignment = str(audio.get("clock_alignment", "")).casefold()
    return (
        audio.get("backend") == "avfoundation"
        and "perf_counter" in alignment
        and audio_input.path.name == str(audio.get("file", "audio.mka"))
    )


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _select_backend(requested: str) -> str:
    if requested != "auto":
        return requested
    apple_silicon = (
        platform.system() == "Darwin" and platform.machine() == "arm64"
    )
    if apple_silicon and _has_module("mlx_whisper"):
        return "mlx"
    if _has_module("faster_whisper"):
        return "faster-whisper"
    return "mlx" if apple_silicon else "faster-whisper"


def _transcribe_mlx(
    path: Path,
    model: str,
    language: str | None,
    word_timestamps: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        import mlx_whisper
    except ImportError as exc:
        raise RuntimeError(
            "MLX backend is not installed; run: "
            "python -m pip install mlx-whisper"
        ) from exc

    kwargs: dict[str, Any] = {
        "path_or_hf_repo": model,
        "word_timestamps": word_timestamps,
        "condition_on_previous_text": False,
        "verbose": False,
    }
    if language is not None:
        kwargs["language"] = language
    result = mlx_whisper.transcribe(str(path), **kwargs)
    metadata = {
        "language": result.get("language", language),
        "condition_on_previous_text": False,
    }
    return result, metadata


def _faster_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda"
    except (ImportError, RuntimeError):
        pass
    return "cpu"


def _transcribe_faster(
    path: Path,
    model_name: str,
    language: str | None,
    word_timestamps: bool,
    device: str,
    compute_type: str,
    vad_filter: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError(
            "faster-whisper backend is not installed; run: "
            "python -m pip install faster-whisper"
        ) from exc

    resolved_device = _faster_device(device)
    resolved_compute = compute_type
    if resolved_compute == "auto":
        resolved_compute = "float16" if resolved_device == "cuda" else "int8"
    model = WhisperModel(
        model_name,
        device=resolved_device,
        compute_type=resolved_compute,
    )
    segments_iter, info = model.transcribe(
        str(path),
        language=language,
        word_timestamps=word_timestamps,
        vad_filter=vad_filter,
        condition_on_previous_text=False,
    )
    segments = []
    for segment in segments_iter:
        words = []
        if word_timestamps and segment.words:
            words = [
                {
                    "word": word.word,
                    "start": word.start,
                    "end": word.end,
                    "probability": word.probability,
                }
                for word in segment.words
            ]
        segments.append(
            {
                "id": segment.id,
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
                "avg_logprob": segment.avg_logprob,
                "no_speech_prob": segment.no_speech_prob,
                "words": words,
            }
        )
    result = {
        "text": "".join(segment["text"] for segment in segments).strip(),
        "segments": segments,
    }
    metadata = {
        "language": getattr(info, "language", language),
        "language_probability": getattr(info, "language_probability", None),
        "duration_s": getattr(info, "duration", None),
        "duration_after_vad_s": getattr(info, "duration_after_vad", None),
        "device": resolved_device,
        "compute_type": resolved_compute,
        "vad_filter": vad_filter,
        "condition_on_previous_text": False,
    }
    return result, metadata


def _seconds_to_us(value: Any) -> int:
    return _decimal_us(value, "transcript timestamp")


def _normalize_segments(
    raw_segments: list[dict[str, Any]],
    audio_start_pts_us: int,
    perf_counter_aligned: bool,
    include_words: bool,
) -> list[dict[str, Any]]:
    normalized = []
    for fallback_id, raw in enumerate(raw_segments):
        start_us = _seconds_to_us(raw["start"])
        end_us = _seconds_to_us(raw["end"])
        segment: dict[str, Any] = {
            "id": int(raw.get("id", fallback_id)),
            "text": str(raw.get("text", "")).strip(),
            "start_audio_s": start_us / 1_000_000,
            "end_audio_s": end_us / 1_000_000,
            "start_pts_us": audio_start_pts_us + start_us,
            "end_pts_us": audio_start_pts_us + end_us,
        }
        if perf_counter_aligned:
            segment["start_perf_counter_us"] = segment["start_pts_us"]
            segment["end_perf_counter_us"] = segment["end_pts_us"]
        for field in ("avg_logprob", "no_speech_prob"):
            if raw.get(field) is not None:
                segment[field] = float(raw[field])

        if include_words:
            words = []
            for raw_word in raw.get("words") or []:
                if raw_word.get("start") is None or raw_word.get("end") is None:
                    continue
                word_start_us = _seconds_to_us(raw_word["start"])
                word_end_us = _seconds_to_us(raw_word["end"])
                word: dict[str, Any] = {
                    "text": str(raw_word.get("word", "")).strip(),
                    "start_audio_s": word_start_us / 1_000_000,
                    "end_audio_s": word_end_us / 1_000_000,
                    "start_pts_us": audio_start_pts_us + word_start_us,
                    "end_pts_us": audio_start_pts_us + word_end_us,
                }
                if perf_counter_aligned:
                    word["start_perf_counter_us"] = word["start_pts_us"]
                    word["end_perf_counter_us"] = word["end_pts_us"]
                if raw_word.get("probability") is not None:
                    word["probability"] = float(raw_word["probability"])
                words.append(word)
            segment["words"] = words
        normalized.append(segment)
    return normalized


def _default_output(audio_input: AudioInput) -> Path:
    if audio_input.recording_dir is not None:
        return audio_input.recording_dir / "derived" / "voice_transcript.json"
    return audio_input.path.with_suffix(".transcript.json")


def _source_name(audio_input: AudioInput) -> str:
    if audio_input.recording_dir is not None:
        try:
            return str(audio_input.path.relative_to(audio_input.recording_dir))
        except ValueError:
            pass
    return str(audio_input.path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Transcribe audio.mka with MLX Whisper on Apple Silicon or "
            "faster-whisper on CUDA/CPU."
        )
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Recording directory or an audio file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSON (default: <recording>/derived/voice_transcript.json).",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "mlx", "faster-whisper"),
        default="auto",
    )
    parser.add_argument(
        "--model",
        help="Backend model name/repository (default: Whisper small).",
    )
    parser.add_argument(
        "--language",
        default="en",
        help="Language code, or 'auto' for detection (default: en).",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="faster-whisper device (default: CUDA when available, else CPU).",
    )
    parser.add_argument(
        "--compute-type",
        default="auto",
        help="faster-whisper compute type (default: float16 CUDA, int8 CPU).",
    )
    parser.add_argument(
        "--no-vad",
        action="store_true",
        help="Disable faster-whisper silence filtering.",
    )
    parser.add_argument(
        "--no-word-timestamps",
        action="store_true",
        help="Store segment timestamps only.",
    )
    parser.add_argument(
        "--timestamp-origin",
        choices=("auto", "perf-counter", "container"),
        default="auto",
        help=(
            "Label packet PTS as perf_counter time when confirmed by the "
            "recording manifest, forced, or never (default: auto)."
        ),
    )
    args = parser.parse_args(argv)

    try:
        audio_input = _resolve_audio_input(args.input)
        probe = _probe_audio(audio_input.path)
    except RuntimeError as exc:
        parser.error(str(exc))

    manifest_aligned = _manifest_confirms_perf_counter(audio_input)
    perf_counter_aligned = (
        args.timestamp_origin == "perf-counter"
        or (args.timestamp_origin == "auto" and manifest_aligned)
    )
    backend = _select_backend(args.backend)
    model = args.model or (
        MLX_DEFAULT_MODEL if backend == "mlx" else FASTER_DEFAULT_MODEL
    )
    language = None if args.language.casefold() == "auto" else args.language
    word_timestamps = not args.no_word_timestamps

    if backend == "mlx" and (
        args.device != "auto" or args.compute_type != "auto"
    ):
        parser.error("--device and --compute-type only apply to faster-whisper")

    print(f"[info] audio: {audio_input.path}")
    print(f"[info] backend: {backend}; model: {model}")
    try:
        if backend == "mlx":
            raw, backend_metadata = _transcribe_mlx(
                audio_input.path, model, language, word_timestamps
            )
        else:
            raw, backend_metadata = _transcribe_faster(
                audio_input.path,
                model,
                language,
                word_timestamps,
                args.device,
                args.compute_type,
                not args.no_vad,
            )
    except RuntimeError as exc:
        parser.error(str(exc))

    segments = _normalize_segments(
        raw.get("segments") or [],
        probe.start_pts_us,
        perf_counter_aligned,
        word_timestamps,
    )
    output_data = {
        "format": FORMAT_VERSION,
        "source": {
            "audio_file": _source_name(audio_input),
            "codec": probe.codec,
            "sample_rate_hz": probe.sample_rate_hz,
            "channels": probe.channels,
            "duration_s": probe.duration_s,
            "start_pts_us": probe.start_pts_us,
            "pts_clock": (
                "perf_counter" if perf_counter_aligned else "container"
            ),
        },
        "transcription": {
            "backend": backend,
            "model": model,
            "requested_language": language,
            "word_timestamps": word_timestamps,
            **backend_metadata,
        },
        "text": str(raw.get("text", "")).strip(),
        "segments": segments,
    }

    output_path = (args.output or _default_output(audio_input)).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    temporary_path.write_text(json.dumps(output_data, indent=2) + "\n")
    temporary_path.replace(output_path)
    print(f"[ok] wrote {len(segments)} segments to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
