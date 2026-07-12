"""VIO stage 5w: windowed bundle adjustment for LONG recordings.

The full-length two-stage solve (vio_bundle_adjust.py) converges on short
spans but can plateau on multi-minute recordings: stage 1 freezes rotations
from the IMU gyro chain seeded at frame 0, and gyro drift over minutes puts
that scaffold far from truth. This stage resets the drift by windowing:

  1. solve overlapping ~30 s windows independently (each window's chain
     re-seeds gravity-aligned at its own first frame) -- parallel across GPUs
  2. stitch: consecutive windows share ~overlap solved frames; both gauges
     are gravity-aligned + metric, so they differ by exactly yaw+translation
     (closed form, vio_stitch.py), chained into window-0's world
  3. blend poses across overlaps (center lerp + quat nlerp)
  4. global refine: full-length vio_bundle_adjust.py --init-trajectory
     <stitched>, so stage 1's frozen rotations are near-correct everywhere;
     stage 2 irons out seams. --refine-tracks tracks_loop.jsonl folds in
     loop closure.

Design doc: docs/superpowers/specs/2026-07-12-windowed-ba-design.md

GOTCHA: BA must run in the `vio` conda env (jax 0.10.2 + current jaxls;
clone of jkerr's jaxgpu) -- older jax/jaxls (e.g. the eyeball env) silently
under-solves LM (testimu full: 9.79 vs 0.325 deg). Pass that interpreter as
--ba-python if this orchestrator itself runs elsewhere.

Run (from data_processing/vio/):
    python vio_windowed_ba.py ../../long-test2 --gpus 0 1 2 3 4 5 6 7 \
        --ba-python ~/miniconda3/envs/vio/bin/python \
        --refine-tracks ../../long-test2/derived/tracks_loop.jsonl
"""
import argparse
import os
import subprocess
import sys
import time

import numpy as np

import vio_stitch as ST

BA_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "vio_bundle_adjust.py")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("recording")
    p.add_argument("--tracks", default=None,
                    help="default: <recording>/derived/tracks.jsonl")
    p.add_argument("--imu-relative", default=None,
                    help="default: <recording>/derived/imu_relative.npz")
    p.add_argument("--out", default=None,
                    help="final refined trajectory; default: "
                         "<recording>/derived/trajectory.npz")
    p.add_argument("--gpus", type=int, nargs="+", required=True)
    p.add_argument("--ba-python", default=sys.executable,
                    help="python interpreter for the BA subprocesses (must be "
                         "an env whose jax/jaxls solves well -- see GOTCHA in "
                         "the module docstring); default: this interpreter")
    p.add_argument("--window-frames", type=int, default=900,
                    help="window length in video frames (~30 s @ 30 fps: short "
                         "enough that the per-window gyro-chain init is accurate)")
    p.add_argument("--overlap-frames", type=int, default=300,
                    help="shared frames between consecutive windows (the stitch "
                         "estimates 4 DOF from these; must be <= window/2)")
    p.add_argument("--min-overlap", type=int, default=30,
                    help="min shared SOLVED frames per seam; fail loudly below")
    p.add_argument("--no-refine", action="store_true",
                    help="stop after stitching (trajectory_stitched.npz only; "
                         "does NOT write --out)")
    p.add_argument("--refine-tracks", default=None,
                    help="tracks file for the global refine, e.g. "
                         "tracks_loop.jsonl to fold in loop closure; "
                         "default: same as --tracks")
    p.add_argument("--resume", action="store_true",
                    help="skip window solves whose output npz already exists")
    return p.parse_args()


def window_ranges(first, last, window, overlap):
    """Inclusive [s, e] video-frame windows covering [first, last]: stride =
    window - overlap; the tail (< stride frames) folds into the final window
    rather than becoming a runt. overlap <= window/2 guarantees only
    CONSECUTIVE windows share frames (blend + stitch assume pairwise seams)."""
    assert window > overlap >= 0, "need window > overlap >= 0"
    assert overlap * 2 <= window, "overlap > window/2 would triple-overlap frames"
    if last - first <= window:
        return [(first, last)]
    stride = window - overlap
    starts = list(range(first, last - window + 1, stride))
    ranges = [[s, s + window] for s in starts]
    ranges[-1][1] = last
    return [tuple(r) for r in ranges]


def solve_windows(args, ranges, win_dir, tracks, imu_rel):
    """Fan window solves out across GPUs, one subprocess per window, one
    window per GPU at a time. Blocks until all succeed; raises on any rc!=0."""
    os.makedirs(win_dir, exist_ok=True)
    jobs = []
    for s, e in ranges:
        out = os.path.join(win_dir, f"window_{s}_{e}.npz")
        if args.resume and os.path.exists(out):
            print(f"[resume] window {s}-{e} exists, skipping", flush=True)
            continue
        jobs.append((s, e, out))
    free = list(args.gpus)
    running = []  # (proc, gpu, s, e, log_path)
    while jobs or running:
        while jobs and free:
            s, e, out = jobs.pop(0)
            g = free.pop(0)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g),
                       XLA_PYTHON_CLIENT_PREALLOCATE="false")
            cmd = [args.ba_python, BA_SCRIPT, args.recording,
                   "--tracks", tracks, "--imu-relative", imu_rel,
                   "--start-frame", str(s), "--n-frames", str(e - s),
                   "--out", out, "--loss-plot", out[:-4] + "_loss.png"]
            log_path = out[:-4] + ".log"
            log_f = open(log_path, "w")
            print(f"[gpu {g}] window {s}-{e} -> {out}", flush=True)
            proc = subprocess.Popen(cmd, env=env, stdout=log_f,
                                    stderr=subprocess.STDOUT)
            log_f.close()  # child holds the fd
            running.append((proc, g, s, e, log_path))
        still = []
        for proc, g, s, e, log_path in running:
            if proc.poll() is None:
                still.append((proc, g, s, e, log_path))
                continue
            free.append(g)
            if proc.returncode != 0:
                raise RuntimeError(f"window {s}-{e} failed "
                                   f"(rc {proc.returncode}), see {log_path}")
            print(f"[gpu {g}] window {s}-{e} done", flush=True)
        running = still
        if running:
            time.sleep(3)


def load_and_stitch(ranges, win_dir, min_overlap):
    """Load window npzs, chain 4-DOF seam alignments into window-0's world.
    Returns placed = [(frame_idx, poses_world0 (N,7) f64, points_world0, npz)]."""
    wins = []
    for s, e in ranges:
        d = np.load(os.path.join(win_dir, f"window_{s}_{e}.npz"))
        med = float(np.nanmedian(d["point_med_ang"][d["point_alive"]]))
        flag = "  <-- WARNING: > 1 deg, check this window" if med > 1.0 else ""
        print(f"window {s}-{e}: {len(d['frame_idx'])} frames, "
              f"median residual {med:.3f} deg{flag}", flush=True)
        wins.append(d)
    theta_cum, t_cum = 0.0, np.zeros(3)
    placed = []
    for k, win in enumerate(wins):
        if k > 0:
            prev = wins[k - 1]
            shared, ia, ib = np.intersect1d(prev["frame_idx"], win["frame_idx"],
                                             return_indices=True)
            if len(shared) < min_overlap:
                raise RuntimeError(f"seam {k}: only {len(shared)} shared frames "
                                   f"(< {min_overlap}) -- windows too disjoint")
            th, tt, diag = ST.fit_yaw_translation(
                np.asarray(prev["pose_wxyz_xyz"], np.float64)[ia],
                np.asarray(win["pose_wxyz_xyz"], np.float64)[ib])
            print(f"seam {k}: yaw {diag['yaw_deg']:+8.3f} deg "
                  f"(vote spread {diag['yaw_spread_deg']:.3f} deg), "
                  f"center rms {diag['center_rms_m'] * 100:.2f} cm, "
                  f"{diag['n_shared']} shared frames", flush=True)
            theta_cum, t_cum = ST.compose_yaw_translation(theta_cum, t_cum, th, tt)
        poses = ST.apply_yaw_translation(
            np.asarray(win["pose_wxyz_xyz"], np.float64), theta_cum, t_cum)
        points = np.asarray(win["points"], np.float64) @ ST.yaw_R(theta_cum).T + t_cum
        placed.append((np.asarray(win["frame_idx"]), poses, points, win))
    return placed


def merge_blend(placed):
    """Merge placed windows into one pose-per-frame trajectory; overlap frames
    blend with a linear ramp (0 -> earlier window, 1 -> later window)."""
    fi = placed[0][0].copy()
    poses = placed[0][1].copy()
    for k in range(1, len(placed)):
        fi_b, pb = placed[k][0], placed[k][1]
        shared, ia, ib = np.intersect1d(fi, fi_b, return_indices=True)
        w = np.linspace(0.0, 1.0, len(shared) + 2)[1:-1]
        poses[ia] = ST.blend_poses(poses[ia], pb[ib], w)
        new = ~np.isin(fi_b, shared)
        fi = np.concatenate([fi, fi_b[new]])
        poses = np.concatenate([poses, pb[new]])
        order = np.argsort(fi)
        fi, poses = fi[order], poses[order]
    return fi, poses


def save_stitched(path, fi, poses, placed):
    """trajectory.npz-compatible stitched output. Landmarks are the per-window
    clouds transformed to world-0, concatenated (overlap duplicates are fine
    for visualization; the refine re-solves landmarks from scratch anyway)."""
    np.savez(
        path, frame_idx=fi, pose_wxyz_xyz=poses,
        points=np.concatenate([p[2] for p in placed]),
        point_first_frame=np.concatenate([p[3]["point_first_frame"] for p in placed]),
        point_first_is_right=np.concatenate([p[3]["point_first_is_right"] for p in placed]),
        point_first_px=np.concatenate([p[3]["point_first_px"] for p in placed]),
        point_alive=np.concatenate([p[3]["point_alive"] for p in placed]),
        point_med_ang=np.concatenate([p[3]["point_med_ang"] for p in placed]),
        cost_history=np.zeros(1))


def main():
    args = parse_args()
    rec = args.recording
    tracks = args.tracks or os.path.join(rec, "derived", "tracks.jsonl")
    imu_rel = args.imu_relative or os.path.join(rec, "derived", "imu_relative.npz")
    out = args.out or os.path.join(rec, "derived", "trajectory.npz")
    win_dir = os.path.join(rec, "derived", "windows")

    imu = np.load(imu_rel)
    first, last = int(imu["frame_idx"][0]), int(imu["frame_idx"][-1])
    ranges = window_ranges(first, last, args.window_frames, args.overlap_frames)
    print(f"{len(ranges)} windows over frames {first}..{last} "
          f"(window {args.window_frames}, overlap {args.overlap_frames}, "
          f"gpus {args.gpus})", flush=True)

    solve_windows(args, ranges, win_dir, tracks, imu_rel)
    placed = load_and_stitch(ranges, win_dir, args.min_overlap)
    fi, poses = merge_blend(placed)
    stitched = os.path.join(rec, "derived", "trajectory_stitched.npz")
    save_stitched(stitched, fi, poses, placed)
    print(f"wrote {stitched} ({len(fi)} poses)", flush=True)

    if args.no_refine:
        print("--no-refine: stopping after stitch (final refine skipped)")
        return
    print("=== global refine from stitched init ===", flush=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpus[0]),
               XLA_PYTHON_CLIENT_PREALLOCATE="false")
    subprocess.run([args.ba_python, BA_SCRIPT, rec,
                    "--tracks", args.refine_tracks or tracks,
                    "--imu-relative", imu_rel,
                    "--init-trajectory", stitched,
                    "--out", out], env=env, check=True)
    print(f"final trajectory: {out}")


if __name__ == "__main__":
    main()
