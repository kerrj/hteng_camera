"""Render two stereo hand videos: raw detections vs optimized 3D.

Produces TWO mp4s over the same frames, each a 2x2 grid:

      |   LEFT eye        RIGHT eye
  ----+--------------------------------
  R   |  R-hand L crop    R-hand R crop
  L   |  L-hand L crop    L-hand R crop

  - raw  video: WiLoR's per-eye 2D keypoints (green; epipolar outliers red).
  - opt  video: the optimized 3D MANO hand reprojected into both eyes (cyan),
                consistent across L/R by construction, with a depth label.

Each hand is rendered in its OWN bbox-centred rectified pinhole pair (so the two
hands don't share an eye image). Both videos are written in one decode pass.

    python render_hands_stereo.py --video ../../long-test1/left_stereo_8bit.mp4 \
        --calib-dir ../../long-test1 --pinhole out/pinhole_stereo/hands.jsonl \
        --stereo3d-right out/hands3d_full_right.jsonl \
        --stereo3d-left  out/hands3d_full_left.jsonl \
        --raw-out out/hands_raw.mp4 --opt-out out/hands_opt.mp4
"""
import argparse
import json

import cv2
import numpy as np
import torch

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fisheye_pinhole as FP
import wilor_hands_batched as W

_BONES = W._BONES


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--calib-dir", default="../../long-test1")
    ap.add_argument("--pinhole", required=True, help="pinhole hands.jsonl")
    ap.add_argument("--stereo3d-right", required=True)
    ap.add_argument("--stereo3d-left", required=True)
    ap.add_argument("--raw-out", required=True)
    ap.add_argument("--opt-out", required=True)
    ap.add_argument("--left-serial", default="046060323008")
    ap.add_argument("--right-serial", default="046060323001")
    ap.add_argument("--frame-min", type=int, default=0)
    ap.add_argument("--frame-max", type=int, default=7006)
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
    return (t(cl["K"]), t(cl["dist"]), t(cr["K"]), t(cr["dist"]), Rs, ts,
            float(np.linalg.norm(st["t"])))


def project(cam, f_px, out_size):
    """Pinhole project (N,3) virtual-cam points → crop px (centre principal pt)."""
    c = (out_size - 1) / 2.0
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
    Kl, Dl, Kr, Dr, Rs, ts, baseline = load_calib(
        args.calib_dir, args.left_serial, args.right_serial, dev)
    Rs_np = np.array(Rs.cpu()); ts_np = np.array(ts.cpu()).reshape(3)

    pin = {}
    for line in open(args.pinhole):
        d = json.loads(line)
        pin[d["frame"]] = d
    # optimized 3D, indexed by frame, per hand
    opt = {1: {}, 0: {}}
    for want, path in ((1, args.stereo3d_right), (0, args.stereo3d_left)):
        for line in open(path):
            d = json.loads(line)
            opt[want][d["frame"]] = d

    from torchcodec.decoders import VideoDecoder
    dec = VideoDecoder(args.video, device="cuda")
    half = dec.metadata.width // 2

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    sz = (OUT * 2, OUT * 2)
    w_raw = cv2.VideoWriter(args.raw_out, fourcc, args.fps, sz)
    w_opt = cv2.VideoWriter(args.opt_out, fourcc, args.fps, sz)

    def blank():
        return np.zeros((OUT, OUT, 3), np.uint8)

    n_written = 0
    for fr in range(args.frame_min, args.frame_max + 1):
        d = pin.get(fr)
        # default blank crops for both hands/eyes, both modes
        cells_raw = {1: [blank(), blank()], 0: [blank(), blank()]}
        cells_opt = {1: [blank(), blank()], 0: [blank(), blank()]}

        # decode once if any hand of either kind is present
        hands = d["hands"] if d else []
        if hands:
            batch = dec.get_frames_in_range(start=fr, stop=fr + 1).data
            fl = batch[0, :, :, :half].float()
            frr = batch[0, :, :, half:].float()

        for want in (1, 0):                       # right then left
            cand = [h for h in hands if h["is_right"] == want]
            if not cand:
                continue
            h = max(cand, key=lambda x: x["bbox"][2] - x["bbox"][0])
            f_px = h["f_px"]
            kpL = np.array(h["kp_left"], np.float32)
            kpR = np.array(h["kp_right"], np.float32)
            dy = np.abs(kpL[:, 1] - kpR[:, 1])
            outlier = (dy >= args.dy_thresh)   # epipolar gate (matches optimizer)

            # re-render the exact crops the ViT saw, from stored verged geometry
            Rv_l_np = np.array(h["Rv_l"], np.float32)
            Rv_r_np = np.array(h["Rv_r"], np.float32)
            Rv_l = torch.tensor(Rv_l_np, device=dev)
            Rv_r = torch.tensor(Rv_r_np, device=dev)
            cL = FP.render_crop(fl, Rv_l, f_px, Kl, Dl, OUT)
            cR = FP.render_crop(frr, Rv_r, f_px, Kr, Dr, OUT)
            L = cv2.cvtColor(cL.permute(1, 2, 0).clamp(0, 255).byte().cpu().numpy(),
                             cv2.COLOR_RGB2BGR)
            R = cv2.cvtColor(cR.permute(1, 2, 0).clamp(0, 255).byte().cpu().numpy(),
                             cv2.COLOR_RGB2BGR)
            Lr, Rr, Lo, Ro = L.copy(), R.copy(), L.copy(), R.copy()

            # raw: WiLoR keypoints (outliers red)
            draw_skel(Lr, kpL, (0, 220, 0), outlier)
            draw_skel(Rr, kpR, (0, 220, 0), outlier)

            # opt: optimized 3D (LEFT-FISHEYE frame) reprojected into both verged
            # eyes via their own Rv: left = Rv_l^T j; right = Rv_r^T (Rs j + ts).
            o = opt[want].get(fr)
            if o is not None:
                j = np.array(o["joints_3d_cam"], np.float32)        # left-fisheye
                jL = j @ Rv_l_np                                    # Rv_l^T @ j
                jR = (j @ Rs_np.T + ts_np[None]) @ Rv_r_np          # Rv_r^T @ (Rs j+ts)
                draw_skel(Lo, project(jL, f_px, OUT), (255, 255, 0))
                draw_skel(Ro, project(jR, f_px, OUT), (255, 255, 0))
                tag = f"d={o['depth_m']:.2f}m"
            else:
                n_in = int((~outlier).sum())
                for img in (Lo, Ro):
                    cv2.putText(img, f"REJECTED ({n_in} inl)", (8, OUT // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                tag = "rejected"

            cells_raw[want] = [Lr, Rr]
            cells_opt[want] = [Lo, Ro]
            # label the optimized right-eye cell with depth
            cv2.putText(cells_opt[want][1], tag, (5, OUT - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        # labels on the left-eye cells
        for cells, lbl in ((cells_raw, "RAW"), (cells_opt, "OPT")):
            cv2.putText(cells[1][0], f"R-hand L  f{fr}", (5, 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
            cv2.putText(cells[1][1], "R-hand R", (5, 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
            cv2.putText(cells[0][0], "L-hand L", (5, 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
            cv2.putText(cells[0][1], "L-hand R", (5, 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)

        panel_raw = np.vstack([np.hstack(cells_raw[1]), np.hstack(cells_raw[0])])
        panel_opt = np.vstack([np.hstack(cells_opt[1]), np.hstack(cells_opt[0])])
        w_raw.write(panel_raw)
        w_opt.write(panel_opt)
        n_written += 1
        if n_written % 200 == 0:
            print(f"{n_written} frames", flush=True)
    w_raw.release()
    w_opt.release()
    print(f"done: {n_written} frames -> {args.raw_out}, {args.opt_out}")


if __name__ == "__main__":
    main()
