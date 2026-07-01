"""Visualize stage 2 (vio_match_pairs.py): for one pair type (stereo, or a
specific temporal gap), render side-by-side frames with matched keypoints
joined by lines -- green for matches that survived the geometric gate, red
for matches that had good LightGlue confidence but got rejected by it (so you
can see what the gate is actually filtering, not just the end result).

Run (from data_processing/vio/):
    python vio_visualize_matches.py ../../long-test1 --matches ../../long-test1/matches.jsonl \
        --pair-type stereo --out ../../long-test1/matches_viz_stereo.mp4
    python vio_visualize_matches.py ../../long-test1 --matches ../../long-test1/matches.jsonl \
        --pair-type temporal --gap 1 --eye left --out ../../long-test1/matches_viz_t1.mp4
"""
import argparse
import json
import os

import cv2
import h5py
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("recording", help="recording dir with left.mp4/right.mp4")
    p.add_argument("--matches", default=None, help="default: <recording>/matches.jsonl")
    p.add_argument("--features", default=None, help="default: <recording>/features.h5 (for fps)")
    p.add_argument("--pair-type", choices=("stereo", "temporal"), default="stereo")
    p.add_argument("--gap", type=int, default=None, help="required for --pair-type temporal")
    p.add_argument("--eye", choices=("left", "right"), default="left",
                    help="which eye's temporal pairs to show (ignored for stereo)")
    p.add_argument("--out", default=None)
    p.add_argument("--max-pairs", type=int, default=None)
    return p.parse_args()


def load_records(matches_path, pair_type, gap, eye):
    recs = []
    with open(matches_path) as f:
        for line in f:
            r = json.loads(line)
            if r["pair_type"] != pair_type:
                continue
            if pair_type == "temporal" and (r["gap"] != gap or r["eye_a"] != eye):
                continue
            recs.append(r)
    recs.sort(key=lambda r: r["frame_a"])
    return recs


def read_frames_up_to(path, max_frame):
    """Read frames [0, max_frame] only -- NOT the whole video. A viz run only
    ever touches a handful of early frames' worth of pairs, but this source
    video can be a multi-minute recording; reading it in full would blow up
    memory for no reason."""
    cap = cv2.VideoCapture(path)
    frames = []
    for _ in range(max_frame + 1):
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames


def draw_pair(frame_a, frame_b, kp_a, kp_b, idx_a, idx_b, rej_a, rej_b, label):
    h, w = frame_a.shape[:2]
    canvas = np.hstack([frame_a, frame_b])
    for k in range(len(idx_a)):
        pa = tuple(np.round(kp_a[idx_a[k]]).astype(int))
        pb = tuple(np.round(kp_b[idx_b[k]]).astype(int) + np.array([w, 0]))
        cv2.line(canvas, pa, pb, (0, 255, 0), 1, cv2.LINE_AA)
    for k in range(len(rej_a)):
        pa = tuple(np.round(kp_a[rej_a[k]]).astype(int))
        pb = tuple(np.round(kp_b[rej_b[k]]).astype(int) + np.array([w, 0]))
        cv2.line(canvas, pa, pb, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 0), 2, cv2.LINE_AA)
    return canvas


def main():
    args = parse_args()
    matches_path = args.matches or os.path.join(args.recording, "matches.jsonl")
    features_path = args.features or os.path.join(args.recording, "features.h5")
    if args.pair_type == "temporal" and args.gap is None:
        raise SystemExit("--gap is required for --pair-type temporal")

    recs = load_records(matches_path, args.pair_type, args.gap, args.eye)
    if args.max_pairs:
        recs = recs[:args.max_pairs]
    if not recs:
        raise SystemExit(f"no matches found for pair_type={args.pair_type} gap={args.gap} eye={args.eye}")
    print(f"{len(recs)} pairs to render")

    with h5py.File(features_path, "r") as f:
        fps = float(f.attrs["fps"])
        kp_cache = {}

        def kp(eye, i):
            key = (eye, i)
            if key not in kp_cache:
                n = int(f[eye]["counts"][i])
                kp_cache[key] = f[eye]["keypoints"][i].reshape(n, 2)
            return kp_cache[key]

        eye_a_video = "left" if args.pair_type == "stereo" else args.eye
        eye_b_video = "right" if args.pair_type == "stereo" else args.eye
        max_frame = max(max(r["frame_a"] for r in recs), max(r["frame_b"] for r in recs))
        frames_a = read_frames_up_to(os.path.join(args.recording, f"{eye_a_video}.mp4"), max_frame)
        frames_b = (frames_a if eye_a_video == eye_b_video
                    else read_frames_up_to(os.path.join(args.recording, f"{eye_b_video}.mp4"), max_frame))

        out_path = args.out or os.path.join(
            args.recording, f"matches_viz_{args.pair_type}"
            + (f"_gap{args.gap}_{args.eye}" if args.pair_type == "temporal" else "") + ".mp4")
        h, w = frames_a[0].shape[:2]
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w * 2, h))

        for n, r in enumerate(recs):
            i, j = r["frame_a"], r["frame_b"]
            if i >= len(frames_a) or j >= len(frames_b):
                continue
            kp_a, kp_b = kp(r["eye_a"], i), kp(r["eye_b"], j)
            label = (f"{args.pair_type} frame {i}->{j}  "
                     f"accepted={r['n_geom']} rejected={r['n_raw']-r['n_geom']}")
            canvas = draw_pair(frames_a[i].copy(), frames_b[j].copy(), kp_a, kp_b,
                                r["idx_a"], r["idx_b"], r["rejected_idx_a"], r["rejected_idx_b"], label)
            writer.write(canvas)
            if n % 20 == 0:
                print(f"{n}/{len(recs)}")

        writer.release()

    print(f"wrote {out_path} (mp4v -- re-encode with ffmpeg to h264 before transferring)")


if __name__ == "__main__":
    main()
