"""Render before/after stereo-triangulation comparison video.

For each frame (one hand track), shows the rectified left+right pinhole crops
with:
  - BEFORE (green): WiLoR's raw per-eye 2D keypoints. Keypoints flagged as
    epipolar outliers (|y_left-y_right| > dy_thresh) are drawn RED.
  - AFTER (cyan): the single optimized 3D MANO hand reprojected into both eyes
    (consistent across L/R by construction).
  - A red "REJECTED (n inliers)" banner on frames the optimizer dropped.

Reads the pinhole `hands.jsonl` (crop geometry + WiLoR kp) and the stereo3d
`hands3d.jsonl` (optimized joints_3d_cam in the rectified-left frame). Re-renders
the crops from the video so the overlays sit on real imagery.

    python render_stereo_compare.py --video long-test1/left_stereo_8bit.mp4 \
        --calib-dir long-test1 --pinhole out/pinhole_stereo/hands.jsonl \
        --stereo3d out/hands3d_30s_right.jsonl --hand right \
        --frame-max 900 --out out/stereo_cmp_right.mp4
"""
import argparse
import json

import cv2
import numpy as np
import torch

import fisheye_pinhole as FP
import wilor_hands_batched as W

_BONES = W._BONES


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--calib-dir", default="long-test1")
    ap.add_argument("--pinhole", required=True, help="pinhole hands.jsonl")
    ap.add_argument("--stereo3d", required=True, help="optimized hands3d.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--hand", choices=["left", "right"], default="right")
    ap.add_argument("--left-serial", default="046060323008")
    ap.add_argument("--right-serial", default="046060323001")
    ap.add_argument("--frame-min", type=int, default=0)
    ap.add_argument("--frame-max", type=int, default=900)
    ap.add_argument("--dy-thresh", type=float, default=8.0)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--out-size", type=int, default=256)
    return ap.parse_args()


def load_calib(calib_dir, ls, rs, dev):
    cl = json.load(open(f"{calib_dir}/calib_{ls}.json"))["intrinsics"]
    cr = json.load(open(f"{calib_dir}/calib_{rs}.json"))["intrinsics"]
    st = json.load(open(f"{calib_dir}/stereo_{ls}_{rs}.json"))
    t = lambda x: torch.tensor(x, device=dev, dtype=torch.float32)
    Rs, ts = t(st["R"]), t(st["t"])
    b_hat = -Rs.T @ ts
    b_hat = b_hat / torch.linalg.norm(b_hat)
    return (t(cl["K"]), t(cl["dist"]), t(cr["K"]), t(cr["dist"]), Rs, b_hat,
            float(np.linalg.norm(st["t"])))


def project(j_cam, f_px, out_size, baseline_x=0.0):
    c = (out_size - 1) / 2.0
    cam = j_cam - np.array([baseline_x, 0, 0])[None]
    return np.stack([f_px * cam[:, 0] / cam[:, 2] + c,
                     f_px * cam[:, 1] / cam[:, 2] + c], 1)


def draw_skel(img, kp, color, outlier=None):
    for a, b in _BONES:
        cv2.line(img, tuple(kp[a].astype(int)), tuple(kp[b].astype(int)), color, 1)
    for i, (x, y) in enumerate(kp):
        col = (0, 0, 255) if (outlier is not None and outlier[i]) else color
        cv2.circle(img, (int(x), int(y)), 3, col, -1)


def main():
    args = parse_args()
    dev = torch.device("cuda")
    OUT = args.out_size
    want_right = 1 if args.hand == "right" else 0
    Kl, Dl, Kr, Dr, Rs, b_hat, baseline = load_calib(
        args.calib_dir, args.left_serial, args.right_serial, dev)

    # index inputs by frame
    pin = {}
    for line in open(args.pinhole):
        d = json.loads(line)
        pin[d["frame"]] = d
    opt = {}
    for line in open(args.stereo3d):
        d = json.loads(line)
        opt[d["frame"]] = d   # has joints_3d_cam (world = rectified-left frame), trans

    from torchcodec.decoders import VideoDecoder
    dec = VideoDecoder(args.video, device="cuda")
    half = dec.metadata.width // 2

    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"),
                             args.fps, (OUT * 2, OUT * 2))
    n_written = 0
    for fr in range(args.frame_min, args.frame_max + 1):
        d = pin.get(fr)
        cand = [h for h in d["hands"] if h["is_right"] == want_right] if d else []
        # blank frame if no detection of this hand
        if not cand:
            writer.write(np.zeros((OUT * 2, OUT * 2, 3), np.uint8))
            n_written += 1
            continue
        h = max(cand, key=lambda x: x["bbox"][2] - x["bbox"][0])
        f_px = h["f_px"]
        kpL = np.array(h["kp_left"], np.float32)
        kpR = np.array(h["kp_right"], np.float32)
        dy = np.abs(kpL[:, 1] - kpR[:, 1])
        outlier = (dy >= args.dy_thresh) | ((kpL[:, 0] - kpR[:, 0]) <= 0.5)

        # re-render the rectified crops
        batch = dec.get_frames_in_range(start=fr, stop=fr + 1).data
        fl = batch[0, :, :, :half].float()
        frr = batch[0, :, :, half:].float()
        bbox = torch.tensor(h["bbox"], device=dev)
        cL, cR, _ = FP.render_stereo_crop(fl, frr, bbox, Kl, Dl, Kr, Dr, Rs,
                                          b_hat, out_size=OUT)
        L = cv2.cvtColor(cL.permute(1, 2, 0).clamp(0, 255).byte().cpu().numpy(),
                         cv2.COLOR_RGB2BGR)
        R = cv2.cvtColor(cR.permute(1, 2, 0).clamp(0, 255).byte().cpu().numpy(),
                         cv2.COLOR_RGB2BGR)
        Lb, La, Rb, Ra = L.copy(), L.copy(), R.copy(), R.copy()

        # BEFORE: WiLoR keypoints, outliers in red
        draw_skel(Lb, kpL, (0, 220, 0), outlier)
        draw_skel(Rb, kpR, (0, 220, 0), outlier)

        # AFTER: optimized 3D reprojected into both eyes (if frame was kept)
        o = opt.get(fr)
        if o is not None:
            j = np.array(o["joints_3d_cam"], np.float32)  # rectified-left frame
            draw_skel(La, project(j, f_px, OUT, 0.0), (255, 255, 0))
            draw_skel(Ra, project(j, f_px, OUT, baseline), (255, 255, 0))
            tag = f"AFTER d={o['depth_m']:.2f}m"
        else:
            n_in = int((~outlier).sum())
            for img in (La, Ra):
                cv2.putText(img, f"REJECTED ({n_in} inl)", (10, OUT // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            tag = "AFTER (rejected)"

        cv2.putText(Lb, f"L BEFORE f{fr}", (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(Rb, "R BEFORE", (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(La, "L " + tag, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(Ra, "R AFTER", (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        panel = np.vstack([np.hstack([Lb, La]), np.hstack([Rb, Ra])])
        writer.write(panel)
        n_written += 1
        if n_written % 100 == 0:
            print(f"{n_written} frames", flush=True)
    writer.release()
    print(f"done: {n_written} frames -> {args.out}")


if __name__ == "__main__":
    main()
