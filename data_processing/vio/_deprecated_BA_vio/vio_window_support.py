"""Report visual-track support inside one windowed-BA window."""

import argparse
import json
import os
import sys

import numpy as np

from vio_bundle_adjust import load_tracks

sys.path.insert(0, "/tmp")
from vio_windowed_ba_dense_diag import (
    _complete_track_window_rows,
    _pair_coverage_window_rows,
    _track_length_window_rows,
)


def _frame_support(pose_ids, point_ids, rows, start, window_size):
    local_pose = pose_ids[rows] - start
    local_point = point_ids[rows]
    obs = np.bincount(local_pose, minlength=window_size)
    tracks = np.zeros(window_size, dtype=np.int64)
    frame_sets = []
    for frame in range(window_size):
        ids = np.unique(local_point[local_pose == frame])
        frame_sets.append(ids)
        tracks[frame] = len(ids)
    adjacent = np.array([
        np.intersect1d(frame_sets[i], frame_sets[i + 1],
                       assume_unique=True).size
        for i in range(window_size - 1)
    ])
    return obs, tracks, adjacent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("recording")
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--window-index", type=int, required=True)
    ap.add_argument("--window-s", type=float, default=4.0)
    ap.add_argument("--overlap-frames", type=int, default=30)
    ap.add_argument("--max-obs", type=int, default=24000)
    ap.add_argument("--max-landmarks", type=int, default=3000)
    ap.add_argument("--obs-per-track", type=int, default=8)
    ap.add_argument("--obs-sampling",
                    choices=("track_length", "complete_track",
                             "pair_coverage"),
                    default="track_length")
    ap.add_argument("--min-pair-tracks", type=int, default=20)
    ap.add_argument("--frame-lo", type=int, default=None)
    ap.add_argument("--frame-hi", type=int, default=None)
    args = ap.parse_args()

    rec = args.recording
    fps = json.load(open(os.path.join(rec, "recording.json"))).get("fps", 30)
    window_size = max(8, int(round(args.window_s * fps)))
    stride = window_size - max(2, args.overlap_frames)

    imu = np.load(os.path.join(rec, "derived", "imu_relative.npz"))
    frame_idx = imu["frame_idx"][imu["frame_valid"]]
    frame_to_pose = {int(frame): i for i, frame in enumerate(frame_idx)}
    n_frames = len(frame_idx)
    starts = list(range(0, max(n_frames - window_size, 0) + 1, stride))
    if starts[-1] + window_size < n_frames:
        starts.append(n_frames - window_size)
    start = starts[args.window_index]

    tracks = load_tracks(args.tracks, int(frame_idx[-1]))
    pose_ids = []
    point_ids = []
    obs_right = []
    for point_id, observations in enumerate(tracks):
        for eye, frame, _ in observations:
            pose_id = frame_to_pose.get(int(frame))
            if pose_id is not None and start <= pose_id < start + window_size:
                pose_ids.append(pose_id)
                point_ids.append(point_id)
                obs_right.append(eye == "right")
    pose_ids = np.asarray(pose_ids)
    point_ids = np.asarray(point_ids)
    obs_right = np.asarray(obs_right)
    order = np.argsort(pose_ids, kind="stable")
    pose_ids = pose_ids[order]
    point_ids = point_ids[order]
    obs_right = obs_right[order]
    raw_rows = np.arange(len(pose_ids))
    if args.obs_sampling == "pair_coverage":
        selected_rows = _pair_coverage_window_rows(
            raw_rows, pose_ids, point_ids, obs_right, args.max_obs,
            args.obs_per_track, 3, args.min_pair_tracks, start, window_size,
            args.max_landmarks)
    elif args.obs_sampling == "complete_track":
        selected_rows = _complete_track_window_rows(
            raw_rows, pose_ids, point_ids, obs_right, args.max_obs, 3,
            args.min_pair_tracks, start, window_size,
            args.max_landmarks)
    else:
        selected_rows = _track_length_window_rows(
            raw_rows, pose_ids, point_ids, args.max_obs, args.obs_per_track,
            args.max_landmarks)

    raw = _frame_support(
        pose_ids, point_ids, raw_rows, start, window_size)
    selected = _frame_support(
        pose_ids, point_ids, selected_rows, start, window_size)
    print(f"window {args.window_index}: poses [{start}, "
          f"{start + window_size}), video frames "
          f"[{frame_idx[start]}, {frame_idx[start + window_size - 1]}]")
    print(f"raw {len(raw_rows)} obs/{len(np.unique(point_ids))} tracks; "
          f"selected {len(selected_rows)} obs/"
          f"{len(np.unique(point_ids[selected_rows]))} tracks")
    print("frame raw_obs raw_tracks selected_obs selected_tracks "
          "raw_adjacent selected_adjacent")
    lo = args.frame_lo if args.frame_lo is not None else int(frame_idx[start])
    hi = args.frame_hi if args.frame_hi is not None else int(
        frame_idx[start + window_size - 1])
    for local, frame in enumerate(frame_idx[start:start + window_size]):
        if not lo <= frame <= hi:
            continue
        raw_adj = raw[2][local] if local < window_size - 1 else -1
        selected_adj = selected[2][local] if local < window_size - 1 else -1
        print(f"{int(frame)} {raw[0][local]} {raw[1][local]} "
              f"{selected[0][local]} {selected[1][local]} "
              f"{raw_adj} {selected_adj}")


if __name__ == "__main__":
    main()
