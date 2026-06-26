"""Render an annotated video from a hands.jsonl + the source video.

Draws the 21-joint skeleton + bbox for every detected hand, overlaid on the
frames. Reads results produced by wilor_hands.py or wilor_hands_batched.py.

The source can be a per-eye video (left.mp4) or the side-by-side stereo video
(left_stereo_8bit.mp4) — for the latter pass --eye left|right to slice the half
that matches the jsonl.

Example:
    python render_hands_video.py \
        --jsonl data_processing/out/stereo/left/hands.jsonl \
        --video long-test1/left_stereo_8bit.mp4 --eye left \
        --out data_processing/out/stereo/left_overlay.mp4 --downscale 2
"""
import argparse
import json

import cv2
import numpy as np

_BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
          (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15),
          (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--jsonl", required=True)
    p.add_argument("--video", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--eye", choices=["left", "right"], default=None,
                   help="if video is side-by-side stereo, which half to slice")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--downscale", type=int, default=1,
                   help="shrink output by this factor (file size / speed)")
    p.add_argument("--max-frames", type=int, default=None)
    return p.parse_args()


def draw(bgr, hands):
    for h in hands:
        x1, y1, x2, y2 = (int(v) for v in h["bbox"])
        color = (0, 0, 255) if h["is_right"] else (255, 128, 0)
        cv2.rectangle(bgr, (x1, y1), (x2, y2), color, 2)
        kp = np.asarray(h["keypoints_2d"])
        for a, b in _BONES:
            cv2.line(bgr, tuple(kp[a].astype(int)), tuple(kp[b].astype(int)),
                     (0, 255, 0), 2)
        for x, y in kp:
            cv2.circle(bgr, (int(x), int(y)), 3, (0, 255, 255), -1)
    return bgr


def main():
    args = parse_args()
    # index results by frame
    by_frame = {}
    with open(args.jsonl) as f:
        for line in f:
            d = json.loads(line)
            by_frame[d["frame"]] = d["hands"]

    cap = cv2.VideoCapture(args.video)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if args.eye:
        half = W // 2
        W = half
    outW, outH = W // args.downscale, H // args.downscale
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"),
                             args.fps, (outW, outH))

    fi = 0
    written = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if args.eye == "left":
            bgr = bgr[:, :W]
        elif args.eye == "right":
            bgr = bgr[:, bgr.shape[1] // 2:]
        draw(bgr, by_frame.get(fi, []))
        if args.downscale > 1:
            bgr = cv2.resize(bgr, (outW, outH))
        writer.write(bgr)
        written += 1
        if args.max_frames and written >= args.max_frames:
            break
        fi += 1
        if fi % 500 == 0:
            print(f"{fi} frames", flush=True)
    writer.release()
    cap.release()
    print(f"done: {written} frames -> {args.out}")


if __name__ == "__main__":
    main()
