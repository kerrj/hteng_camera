"""Quick visual check: are hands centred in BOTH verged crops?

Renders the stored verged crop pair for the first few frames that have a hand in
both eyes, stacks L|R per hand with a centre crosshair, and writes a contact
sheet. Reuses the pipeline's stored Rv_l/Rv_r geometry via FP.render_crop.
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
from wilor_hands_pinhole import load_calib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="../../long-test1/left_stereo_8bit.mp4")
    ap.add_argument("--calib-dir", default="../../long-test1")
    ap.add_argument("--jsonl", default="out/pinhole_verged_test/hands.jsonl")
    ap.add_argument("--out", default="out/verged_check.png")
    ap.add_argument("--n", type=int, default=8, help="number of hand crops to show")
    ap.add_argument("--out-size", type=int, default=256)
    args = ap.parse_args()
    dev = torch.device("cuda")
    OUT = args.out_size
    Kl, Dl, Kr, Dr, Rs, ts, b_hat, baseline = load_calib(
        args.calib_dir, "046060323008", "046060323001", dev)

    from torchcodec.decoders import VideoDecoder
    dec = VideoDecoder(args.video, device="cuda")
    half = dec.metadata.width // 2

    rows = []
    for line in open(args.jsonl):
        d = json.loads(line)
        if not d["hands"]:
            continue
        batch = dec.get_frames_in_range(start=d["frame"], stop=d["frame"] + 1).data
        fl = batch[0, :, :, :half].float()
        frr = batch[0, :, :, half:].float()
        for h in d["hands"]:
            Rv_l = torch.tensor(h["Rv_l"], device=dev, dtype=torch.float32)
            Rv_r = torch.tensor(h["Rv_r"], device=dev, dtype=torch.float32)
            f_px = h["f_px"]
            cL = FP.render_crop(fl, Rv_l, f_px, Kl, Dl, OUT)
            cR = FP.render_crop(frr, Rv_r, f_px, Kr, Dr, OUT)
            L = cv2.cvtColor(cL.permute(1, 2, 0).clamp(0, 255).byte().cpu().numpy(),
                             cv2.COLOR_RGB2BGR)
            R = cv2.cvtColor(cR.permute(1, 2, 0).clamp(0, 255).byte().cpu().numpy(),
                             cv2.COLOR_RGB2BGR)
            for img, lbl in ((L, f"L f{d['frame']} {'R' if h['is_right'] else 'L'}h"),
                             (R, "R")):
                cv2.line(img, (OUT // 2, 0), (OUT // 2, OUT), (0, 0, 255), 1)
                cv2.line(img, (0, OUT // 2), (OUT, OUT // 2), (0, 0, 255), 1)
                cv2.putText(img, lbl, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (0, 255, 255), 1)
            # depth of triangulated P (left frame z)
            cv2.putText(L, f"Pz={h['P'][2]:.2f}m", (4, OUT - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            rows.append(np.hstack([L, R]))
            if len(rows) >= args.n:
                break
        if len(rows) >= args.n:
            break

    sheet = np.vstack(rows)
    cv2.imwrite(args.out, sheet)
    print(f"wrote {args.out}  ({len(rows)} hand crops)")


if __name__ == "__main__":
    main()
