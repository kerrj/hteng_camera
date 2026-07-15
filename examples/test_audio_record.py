#!/usr/bin/env python3
"""Probe low-overhead, timestamp-preserving microphone capture with ffmpeg.

Examples:
  python examples/test_audio_record.py --list
  python examples/test_audio_record.py --device "Justin's AirPods Pro"
  python examples/test_audio_record.py --device 0 --duration 30
  python examples/test_audio_record.py --synthetic --duration 10

The Matroska audio file contains mono 16 kHz PCM plus the absolute
AVFoundation packet timestamps. On macOS those timestamps use the same
monotonic host clock as time.perf_counter().
"""

import argparse
import json
import platform
import re
import resource
import signal
import subprocess
import sys
import time
from pathlib import Path


def _avfoundation_audio_devices() -> tuple[str, list[tuple[int, str]]]:
    if platform.system() != "Darwin":
        raise RuntimeError("AVFoundation audio capture is only available on macOS")
    proc = subprocess.run(
        [
            "ffmpeg", "-hide_banner",
            "-f", "avfoundation",
            "-list_devices", "true",
            "-i", "",
        ],
        capture_output=True,
        text=True,
    )
    devices: list[tuple[int, str]] = []
    in_audio = False
    for line in proc.stderr.splitlines():
        if "AVFoundation audio devices:" in line:
            in_audio = True
            continue
        if not in_audio:
            continue
        match = re.search(r"\[(\d+)\]\s+(.+)$", line)
        if match:
            devices.append((int(match.group(1)), match.group(2)))
    return proc.stderr.rstrip(), devices


def _list_avfoundation_devices() -> int:
    try:
        listing, _ = _avfoundation_audio_devices()
    except RuntimeError as exc:
        print(f"[error] {exc}")
        return 2
    print(listing)
    return 0


def _resolve_airpods_device(requested: str | None) -> str:
    _listing, devices = _avfoundation_audio_devices()
    if requested is None:
        matches = [name for _index, name in devices
                   if "airpods" in name.casefold()]
    elif requested.isdigit():
        matches = [name for index, name in devices
                   if index == int(requested)]
    else:
        exact = [name for _index, name in devices
                 if name.casefold() == requested.casefold()]
        matches = exact or [
            name for _index, name in devices
            if requested.casefold() in name.casefold()
        ]

    if len(matches) != 1:
        available = ", ".join(name for _index, name in devices) or "none"
        target = "connected AirPods" if requested is None else repr(requested)
        raise RuntimeError(
            f"expected exactly one audio device matching {target}; "
            f"available: {available}"
        )
    device = matches[0]
    if "airpods" not in device.casefold():
        raise RuntimeError(
            f"refusing non-AirPods audio device {device!r}; "
            "connect/select AirPods instead"
        )
    return device


def _child_cpu_seconds() -> float:
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return usage.ru_utime + usage.ru_stime


def _probe(path: Path) -> dict:
    proc = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "stream=codec_name,sample_rate,channels,start_time",
            "-show_packets", "-read_intervals", "%+#1",
            "-show_entries", "packet=pts_time,duration_time",
            "-of", "json",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "ffprobe failed")
    return json.loads(proc.stdout)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Record timestamp-preserving MKA audio in a separate process."
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List macOS AVFoundation devices and exit.",
    )
    parser.add_argument(
        "--device",
        help="AVFoundation audio device name or index, as shown by --list.",
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Use ffmpeg's real-time silent source instead of a microphone.",
    )
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument(
        "--output", type=Path, default=Path("/tmp/hteng_audio_test.mka")
    )
    args = parser.parse_args()

    if args.list:
        return _list_avfoundation_devices()
    if args.duration <= 0:
        parser.error("--duration must be positive")
    if not args.synthetic and platform.system() != "Darwin":
        parser.error("microphone capture currently uses macOS AVFoundation")

    device = None
    if not args.synthetic:
        try:
            device = _resolve_airpods_device(args.device)
        except RuntimeError as exc:
            parser.error(str(exc))
        print(f"[info] selected AirPods input: {device}")

    source = (
        ["-re", "-f", "lavfi",
         "-i", "anullsrc=channel_layout=mono:sample_rate=16000"]
        if args.synthetic else
        ["-thread_queue_size", "128", "-f", "avfoundation",
         "-i", f":{device}"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.unlink(missing_ok=True)
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-nostdin", "-loglevel", "warning",
        "-copyts",
        *source,
        "-map", "0:a:0",
        "-vn", "-ac", "1", "-ar", "16000",
        "-c:a", "pcm_s16le",
        "-avoid_negative_ts", "disabled",
        "-cluster_time_limit", "1000",
        "-flush_packets", "1",
        "-f", "matroska",
        str(args.output),
    ]

    cpu_before = _child_cpu_seconds()
    before_us = time.perf_counter_ns() // 1000
    # Keep terminal Ctrl+C in the parent. The parent sends exactly one SIGINT
    # so ffmpeg can write the Matroska trailer instead of seeing two signals.
    proc = subprocess.Popen(cmd, start_new_session=True)
    after_us = time.perf_counter_ns() // 1000

    # Start the requested duration after ffmpeg has opened the input and emitted
    # the Matroska header, rather than charging Bluetooth negotiation against it.
    ready_deadline = time.perf_counter() + 15.0
    while (proc.poll() is None
           and (not args.output.exists() or args.output.stat().st_size == 0)
           and time.perf_counter() < ready_deadline):
        time.sleep(0.02)

    stopped_intentionally = False
    try:
        if proc.poll() is None and args.output.exists():
            print(
                f"[info] recording started; speak now "
                f"({args.duration:g} seconds)...",
                flush=True,
            )
            deadline = time.perf_counter() + args.duration
            while proc.poll() is None and time.perf_counter() < deadline:
                time.sleep(0.02)
            if proc.poll() is None:
                stopped_intentionally = True
                proc.send_signal(signal.SIGINT)
        returncode = proc.wait()
    except KeyboardInterrupt:
        stopped_intentionally = True
        proc.send_signal(signal.SIGINT)
        returncode = proc.wait()

    end_us = time.perf_counter_ns() // 1000
    child_cpu_s = _child_cpu_seconds() - cpu_before
    clean_codes = {0, 255, -signal.SIGINT} if stopped_intentionally else {0}
    if returncode not in clean_codes:
        print(f"[error] ffmpeg exited with status {returncode}.")
        return returncode

    try:
        probe = _probe(args.output)
        stream = probe["streams"][0]
        packet = probe["packets"][0]
    except (OSError, KeyError, IndexError, ValueError, RuntimeError) as exc:
        print(f"[error] could not validate {args.output}: {exc}")
        return 1

    wall_s = (end_us - before_us) / 1e6
    print(f"[ok] {args.output} ({args.output.stat().st_size / 1024:.1f} KiB)")
    print(
        f"[info] {stream['channels']} channel, {stream['sample_rate']} Hz, "
        f"{stream['codec_name']}; absolute first PTS {packet['pts_time']} s"
    )
    print(
        f"[info] ffmpeg CPU {child_cpu_s:.3f} s over {wall_s:.3f} s "
        f"({100.0 * child_cpu_s / max(wall_s, 1e-9):.2f}% of one core)"
    )
    print(
        f"[info] Popen bracket: before={before_us} us, after={after_us} us, "
        f"launch call={(after_us - before_us) / 1000:.1f} ms"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
