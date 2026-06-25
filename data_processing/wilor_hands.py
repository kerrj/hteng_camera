"""Run WiLoR-mini hand-pose estimation over a video → per-frame JSON (+ viz).

Runs on chungus in the `eyeball211` env. WiLoR assumes a pinhole camera, so on
our wide fisheye footage the 2D keypoints / camera translation are approximate
near the periphery (good enough for detection + rough pose). A pinhole-crop
refinement mode is planned (see README).

Examples
--------
    # smoke test: a few frames, with annotated jpgs
    python wilor_hands.py long-test1/left.mp4 --out out/left \
        --frames 1500,3000,5000 --viz

    # full run, every 3rd frame, json only
    python wilor_hands.py long-test1/left.mp4 --out out/left --stride 3

Output
------
    <out>/hands.jsonl   one JSON object per processed frame:
        {"frame": int, "width": w, "height": h,
         "hands": [ {"is_right": 0|1, "bbox": [x1,y1,x2,y2],
                     "keypoints_2d": [[x,y]*21],
                     "keypoints_3d": [[x,y,z]*21],
                     "global_orient": [...], "hand_pose": [...], "betas": [...],
                     "cam_t_full": [x,y,z], "focal_px": f}, ... ]}
    <out>/viz/frame_XXXXXX.jpg   (only with --viz)
"""
import argparse
import json
import os

import cv2
import numpy as np
import torch

# MANO/OpenPose 21-joint skeleton (WiLoR joint order)
_BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
          (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15),
          (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--frames", default=None,
                   help="comma-separated explicit frame indices (overrides stride)")
    p.add_argument("--stride", type=int, default=1, help="process every Nth frame")
    p.add_argument("--max-frames", type=int, default=None,
                   help="stop after this many processed frames")
    p.add_argument("--conf", type=float, default=0.3, help="detector confidence")
    p.add_argument("--viz", action="store_true", help="also write annotated jpgs")
    p.add_argument("--fp32", action="store_true", help="use float32 (default fp16)")
    return p.parse_args()


def draw(bgr, hands):
    for h in hands:
        x1, y1, x2, y2 = (int(v) for v in h["bbox"])
        color = (0, 0, 255) if h["is_right"] else (255, 128, 0)
        cv2.rectangle(bgr, (x1, y1), (x2, y2), color, 2)
        kp = np.asarray(h["keypoints_2d"])
        for a, b in _BONES:
            pa, pb = kp[a].astype(int), kp[b].astype(int)
            cv2.line(bgr, tuple(pa), tuple(pb), (0, 255, 0), 1)
        for x, y in kp:
            cv2.circle(bgr, (int(x), int(y)), 3, (0, 255, 255), -1)
    return bgr


def to_record(out):
    wp = out["wilor_preds"]
    return {
        "is_right": int(round(float(out["is_right"]))),
        "bbox": np.asarray(out["hand_bbox"]).ravel().tolist(),
        "keypoints_2d": np.asarray(wp["pred_keypoints_2d"])[0].tolist(),
        "keypoints_3d": np.asarray(wp["pred_keypoints_3d"])[0].tolist(),
        "global_orient": np.asarray(wp["global_orient"]).ravel().tolist(),
        "hand_pose": np.asarray(wp["hand_pose"]).reshape(-1).tolist(),
        "betas": np.asarray(wp["betas"]).ravel().tolist(),
        "cam_t_full": np.asarray(wp["pred_cam_t_full"]).ravel().tolist(),
        "focal_px": float(np.asarray(wp["scaled_focal_length"])),
    }


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.viz:
        os.makedirs(os.path.join(args.out, "viz"), exist_ok=True)

    from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import (
        WiLorHandPose3dEstimationPipeline)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32 if args.fp32 else torch.float16
    pipe = WiLorHandPose3dEstimationPipeline(device=dev, dtype=dtype, verbose=False)

    cap = cv2.VideoCapture(args.video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if args.frames:
        idxs = [int(x) for x in args.frames.split(",")]
    else:
        idxs = list(range(0, total, args.stride))
    if args.max_frames:
        idxs = idxs[:args.max_frames]
    print(f"{args.video}: {w}x{h}, {total} frames; processing {len(idxs)}")

    jsonl = open(os.path.join(args.out, "hands.jsonl"), "w")
    seq_read = (args.stride == 1 and not args.frames)  # sequential = faster reads
    nxt = 0
    for n, fi in enumerate(idxs):
        if seq_read:
            ok, bgr = cap.read()
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, bgr = cap.read()
        if not ok:
            print(f"frame {fi}: read failed"); continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        outs = pipe.predict(rgb, hand_conf=args.conf)
        hands = [to_record(o) for o in outs]
        jsonl.write(json.dumps({"frame": fi, "width": w, "height": h,
                                "hands": hands}) + "\n")
        if args.viz:
            cv2.imwrite(os.path.join(args.out, "viz", f"frame_{fi:06d}.jpg"),
                        draw(bgr, hands))
        if n % 50 == 0:
            print(f"[{n}/{len(idxs)}] frame {fi}: {len(hands)} hand(s)", flush=True)
    jsonl.close()
    cap.release()
    print(f"done → {os.path.join(args.out, 'hands.jsonl')}")


if __name__ == "__main__":
    main()
