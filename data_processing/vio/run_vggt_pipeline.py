"""Run the complete offline VGGT-Omega VIO pipeline.

Inference windows are individually cached, so rerunning this command resumes
tracking and loop verification. Use a new tag when changing geometry or window
settings.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def environment_python(name):
    prefix = Path(os.environ.get("CONDA_PREFIX", sys.prefix)).resolve()
    if prefix.parent.name == "envs":
        envs_dir = prefix.parent
    else:
        envs_dir = prefix / "envs"
    candidate = envs_dir / name / "bin" / "python"
    return str(candidate if candidate.exists() else sys.executable)


def run_stage(name, command, env):
    print(f"\n=== {name} ===", flush=True)
    start = time.time()
    subprocess.run(command, check=True, env=env)
    elapsed = time.time() - start
    print(f"=== {name}: {elapsed:.1f}s ===", flush=True)
    return elapsed


def validate_manifest(path, recording, args):
    with open(path) as f:
        manifest = json.load(f)
    expected = {
        "recording": str(recording),
        "target_fps": args.target_fps,
        "image_size": args.image_size,
        "fov_deg": args.fov_deg,
    }
    mismatches = {
        key: (manifest.get(key), value)
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        details = ", ".join(
            f"{key}: cached={old!r}, requested={new!r}"
            for key, (old, new) in mismatches.items()
        )
        raise ValueError(
            f"{path} is incompatible ({details}); choose a new --tag")


def validate_window_config(path, args):
    with open(path) as f:
        config = json.load(f)
    expected = {
        "window_frames": args.window_frames,
        "overlap_frames": args.overlap_frames,
        "stereo_interval_seconds": args.stereo_interval_seconds,
        "checkpoint": str(Path(args.checkpoint).resolve()),
    }
    mismatches = {
        key: (config.get(key), value)
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        details = ", ".join(
            f"{key}: cached={old!r}, requested={new!r}"
            for key, (old, new) in mismatches.items()
        )
        raise ValueError(
            f"{path} is incompatible ({details}); choose a new --tag")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--gpus", default=None,
                        help="physical GPU list, e.g. 2,3,4,5")
    parser.add_argument("--omega-python", default=environment_python(
        "vggtomega"))
    parser.add_argument("--jax-python", default=environment_python("jaxgpu"))
    parser.add_argument("--target-fps", type=float, default=15.0)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--fov-deg", type=float, default=100.0)
    parser.add_argument("--window-frames", type=int, default=128)
    parser.add_argument("--overlap-frames", type=int, default=64)
    parser.add_argument("--stereo-interval-seconds", type=float, default=1.0)
    parser.add_argument("--visual-cauchy-scale", type=float, default=2.0)
    parser.add_argument(
        "--anchor-translation-weight", type=float, default=1.0)
    parser.add_argument(
        "--anchor-rotation-weight", type=float, default=0.01)
    parser.add_argument("--max-loop-windows", type=int, default=8)
    parser.add_argument("--min-loop-overlap", type=float, default=0.03)
    return parser.parse_args()


def main():
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    recording = Path(args.recording).resolve()
    tag = args.tag or f"vggt_omega_fov{args.fov_deg:g}"
    output_dir = recording / "derived" / tag
    preloop_trajectory = (
        recording / "derived" / f"trajectory_{tag}_preloop.npz")
    final_trajectory = recording / "derived" / f"trajectory_{tag}.npz"
    loop_candidates = output_dir / "proximity_loop_candidates.jsonl"

    gpu_text = args.gpus or os.environ.get("CUDA_VISIBLE_DEVICES")
    if not gpu_text:
        raise ValueError(
            "specify --gpus or set CUDA_VISIBLE_DEVICES explicitly")
    gpu_ids = [item.strip() for item in gpu_text.split(",") if item.strip()]
    if not gpu_ids:
        raise ValueError("no GPU IDs were provided")

    omega_env = os.environ.copy()
    omega_env["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
    jax_env = os.environ.copy()
    jax_env["CUDA_VISIBLE_DEVICES"] = gpu_ids[0]
    jax_env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    infer_script = str(script_dir / "vio_vggt_window_infer.py")
    graph_script = str(script_dir / "vio_vggt_pose_graph.py")
    common_infer = [
        args.omega_python,
        infer_script,
        "infer",
        str(recording),
        "--out-dir",
        str(output_dir),
        "--checkpoint",
        str(Path(args.checkpoint).resolve()),
        "--window-frames",
        str(args.window_frames),
        "--overlap-frames",
        str(args.overlap_frames),
        "--stereo-interval-seconds",
        str(args.stereo_interval_seconds),
        "--num-devices",
        str(len(gpu_ids)),
    ]
    common_graph = [
        args.jax_python,
        graph_script,
        str(recording),
        "--input-dir",
        str(output_dir),
        "--window-anchor-weight",
        "0.25",
        "--visual-cauchy-scale",
        str(args.visual_cauchy_scale),
        "--anchor-translation-weight",
        str(args.anchor_translation_weight),
        "--anchor-rotation-weight",
        str(args.anchor_rotation_weight),
        "--baseline-weight",
        "100",
        "--imu-rotation-weight",
        "10",
        "--gravity-weight",
        "1",
        "--gravity-accel-sigma-g",
        "0.05",
        "--constant-velocity-weight",
        "0.1",
        "--linear-solver",
        "conjugate_gradient",
        "--iterations",
        "20",
    ]

    timings = {}
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        validate_manifest(manifest_path, recording, args)
        windows_path = output_dir / "windows.json"
        if windows_path.exists():
            validate_window_config(windows_path, args)
        print(f"using cached {manifest_path}")
    else:
        timings["prepare"] = run_stage("prepare", [
            args.omega_python,
            infer_script,
            "prepare",
            str(recording),
            "--out-dir",
            str(output_dir),
            "--target-fps",
            str(args.target_fps),
            "--image-size",
            str(args.image_size),
            "--fov-deg",
            str(args.fov_deg),
        ], omega_env)

    timings["tracking"] = run_stage(
        "tracking", common_infer, omega_env)
    timings["preloop_graph"] = run_stage("preloop graph", [
        *common_graph,
        "--track-only",
        "--out",
        str(preloop_trajectory),
    ], jax_env)
    timings["loop_proposal"] = run_stage("loop proposal", [
        args.omega_python,
        infer_script,
        "propose-proximity-loops",
        str(recording),
        "--out-dir",
        str(output_dir),
        "--trajectory",
        str(preloop_trajectory),
        "--max-distance",
        "1.5",
        "--sample-seconds",
        "0.5",
        "--min-gap-seconds",
        "15",
        "--max-view-angle-deg",
        "60",
        "--cluster-radius-seconds",
        "2",
        "--nms-seconds",
        "10",
        "--min-cluster-votes",
        "3",
        "--max-candidates",
        str(args.max_loop_windows),
    ], omega_env)

    with open(loop_candidates) as f:
        loop_count = sum(bool(line.strip()) for line in f)
    timings["loop_verification"] = run_stage("loop verification", [
        *common_infer,
        "--loop-matches",
        str(loop_candidates),
        "--loop-window-frames",
        str(args.window_frames),
        "--max-loop-windows",
        str(args.max_loop_windows),
        "--verify-loop-overlap",
        "--min-loop-overlap",
        str(args.min_loop_overlap),
    ], omega_env)

    timings["final_graph"] = run_stage("final graph", [
        *common_graph,
        "--out",
        str(final_trajectory),
    ], jax_env)
    print(f"\nwrote {final_trajectory}")
    print("stage timings: " + ", ".join(
        f"{name}={elapsed:.1f}s" for name, elapsed in timings.items()))
    summary = {
        "recording": str(recording),
        "output_dir": str(output_dir),
        "trajectory": str(final_trajectory),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "gpus": gpu_ids,
        "target_fps": args.target_fps,
        "image_size": args.image_size,
        "fov_deg": args.fov_deg,
        "window_frames": args.window_frames,
        "overlap_frames": args.overlap_frames,
        "stereo_interval_seconds": args.stereo_interval_seconds,
        "visual_cauchy_scale": args.visual_cauchy_scale,
        "anchor_translation_weight": args.anchor_translation_weight,
        "anchor_rotation_weight": args.anchor_rotation_weight,
        "loop_candidates": loop_count,
        "timings_seconds": timings,
    }
    with open(output_dir / "pipeline_run.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
