"""Run BOTH the raw-crop and pinhole-crop WiLoR paths over a video and render a
side-by-side comparison overlay (raw keypoints | pinhole keypoints on fisheye).

Single eye (left). Detection is shared (GPU YOLO); then per hand we run the ViT
twice — once on the raw square crop, once on the de-warped pinhole crop — and
draw both back on the fisheye frame. Output is an mp4 (mp4v; re-encode to h264
to transfer).

    python wilor_pinhole_compare.py ../../long-test1/left_stereo_8bit.mp4 \
        --calib-dir ../../long-test1 --out out/pinhole_cmp.mp4 --downscale 2
"""
import argparse
import json
import os

import cv2
import numpy as np
import torch

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fisheye_pinhole as FP
import wilor_hands_batched as W

_BONES = W._BONES


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video")
    p.add_argument("--calib-dir", default="../../long-test1")
    p.add_argument("--out", required=True)
    p.add_argument("--left-serial", default="046060323008")
    p.add_argument("--right-serial", default="046060323001")
    p.add_argument("--eye", choices=["left", "right"], default="left")
    p.add_argument("--chunk", type=int, default=32)
    p.add_argument("--conf", type=float, default=0.3)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--downscale", type=int, default=2)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--out-size", type=int, default=256)
    return p.parse_args()


def load_calib(calib_dir, ls, rs, device):
    cl = json.load(open(os.path.join(calib_dir, f"calib_{ls}.json")))["intrinsics"]
    cr = json.load(open(os.path.join(calib_dir, f"calib_{rs}.json")))["intrinsics"]
    st = json.load(open(os.path.join(calib_dir, f"stereo_{ls}_{rs}.json")))
    t = lambda x: torch.tensor(x, device=device, dtype=torch.float32)
    Kl, Dl = t(cl["K"]), t(cl["dist"])
    Kr, Dr = t(cr["K"]), t(cr["dist"])
    Rs, ts = t(st["R"]), t(st["t"])
    b_hat = -Rs.T @ ts
    b_hat = b_hat / torch.linalg.norm(b_hat)
    return Kl, Dl, Kr, Dr, Rs, b_hat


def vit_kp_crop(pipe, crop_chw, is_right, out_size, dev):
    """Run WiLoR ViT on a (3,S,S) crop; return (21,2) crop-space keypoints.

    Replicates WiLoR's handedness handling: flip LEFT hands before the ViT;
    postprocess already maps the output back to the (un-flipped) crop frame.
    """
    c = torch.flip(crop_chw, dims=[2]) if is_right == 0 else crop_chw
    inp = c.permute(1, 2, 0)[None].to(dev, torch.float16)
    with torch.no_grad():
        o = pipe.wilor_model(inp)
    o = {k: v.detach().cpu().float().numpy() for k, v in o.items()}
    isz = np.array([out_size, out_size])
    oi = W.postprocess({k: v[[0]] for k, v in o.items()}, isz / 2.0, out_size,
                       isz, is_right, pipe)
    return oi["pred_keypoints_2d"][0]


def raw_kp(pipe, frame_f, bbox, is_right, dev):
    """Current pipeline path: square crop -> ViT -> keypoints in fisheye px."""
    center = (bbox[2:4] + bbox[0:2]) / 2.0
    bbox_size = (2.5 * (bbox[2:4] - bbox[0:2])).max()
    crop = W.gpu_crop(frame_f, center, bbox_size, is_right == 0, pipe.IMAGE_SIZE)
    inp = crop.permute(1, 2, 0)[None].to(dev, torch.float16)
    with torch.no_grad():
        o = pipe.wilor_model(inp)
    o = {k: v.detach().cpu().float().numpy() for k, v in o.items()}
    isz = np.array([frame_f.shape[2], frame_f.shape[1]])
    oi = W.postprocess({k: v[[0]] for k, v in o.items()}, center, bbox_size,
                       isz, is_right, pipe)
    return oi["pred_keypoints_2d"][0]


def pinhole_kp(pipe, frame_f, bbox_t, is_right, calib, dev, out_size):
    """Pinhole path: render de-warped crop -> ViT -> back-map to fisheye px."""
    Kl, Dl, Kr, Dr, Rs, b_hat = calib
    cL, _, geo = FP.render_stereo_crop(frame_f, frame_f, bbox_t, Kl, Dl, Kl, Dl,
                                       Rs, b_hat, out_size=out_size)
    kpc = vit_kp_crop(pipe, cL, is_right, out_size, dev)
    kpf = FP.crop_px_to_fisheye(torch.tensor(kpc, device=dev, dtype=torch.float32),
                                geo["f_px"], out_size, geo["Rv_l"], Kl, Dl)
    return kpf.cpu().numpy()


def draw(canvas, kp, color):
    for a, b in _BONES:
        cv2.line(canvas, tuple(kp[a].astype(int)), tuple(kp[b].astype(int)),
                 color, 2)
    for x, y in kp:
        cv2.circle(canvas, (int(x), int(y)), 3, (0, 255, 255), -1)


def main():
    args = parse_args()
    dev = torch.device("cuda")
    pipe, _, _ = W.build_pipeline(False)
    calib = load_calib(args.calib_dir, args.left_serial, args.right_serial, dev)

    from torchcodec.decoders import VideoDecoder
    dec = VideoDecoder(args.video, device="cuda")
    total = dec.metadata.num_frames
    half = dec.metadata.width // 2
    H = dec.metadata.height
    n_frames = total if args.max_frames is None else min(total, args.max_frames)

    outW = (half * 2) // args.downscale
    outH = H // args.downscale
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"),
                             args.fps, (outW, outH))

    sl = slice(0, half) if args.eye == "left" else slice(half, 2 * half)
    done = 0
    for cs in range(0, n_frames, args.chunk):
        idxs = list(range(cs, min(cs + args.chunk, n_frames)))
        batch = dec.get_frames_in_range(start=idxs[0], stop=idxs[-1] + 1).data
        eye_u8 = batch[:, :, :, sl].contiguous()       # (n,3,H,W) uint8
        dets = W.detect_gpu(pipe.hand_detector, eye_u8, args.conf)
        for li, boxes in enumerate(dets):
            frame_f = eye_u8[li].float()
            rgb = eye_u8[li].permute(1, 2, 0).cpu().numpy()
            c_raw = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
            c_pin = c_raw.copy()
            for b in boxes:
                ir = int(round(float(b[5])))
                c_raw_kp = raw_kp(pipe, frame_f, b[:4], ir, dev)
                draw(c_raw, c_raw_kp, (0, 200, 0))
                bbox_t = torch.tensor(b[:4], device=dev)
                p_kp = pinhole_kp(pipe, frame_f, bbox_t, ir, calib, dev, args.out_size)
                draw(c_pin, p_kp, (0, 128, 255))
            cv2.putText(c_raw, "RAW crop", (40, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 200, 0), 4)
            cv2.putText(c_pin, "PINHOLE crop", (40, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 128, 255), 4)
            combo = np.hstack([c_raw, c_pin])
            combo = cv2.resize(combo, (outW, outH))
            writer.write(combo)
            done += 1
        if cs % (args.chunk * 10) == 0:
            print(f"[{done}/{n_frames}]", flush=True)
    writer.release()
    print(f"done: {done} frames -> {args.out}")


if __name__ == "__main__":
    main()
