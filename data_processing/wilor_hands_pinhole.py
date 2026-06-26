"""Batched WiLoR hand pose on rectified PINHOLE crops, both eyes — sets up depth.

Like wilor_hands_batched.py, but instead of feeding WiLoR a raw square crop we
render an undistorted virtual-pinhole view aimed at each hand (fisheye_pinhole).
Detection runs on the LEFT eye; from each left bbox we render a *baseline-aligned
rectified* left+right pinhole pair (same virtual intrinsics, rows = epipolar
lines). We run the ViT on both eyes' crops (batched together across the whole
chunk) and save both keypoint sets in rectified crop pixels, so per-keypoint
disparity is just ``x_left - x_right`` and depth = f_px * baseline / disparity.

Speed: pinhole rendering is two grid_samples — it *replaces* the raw crop op, so
the only extra cost vs wilor_hands_batched is the 2nd eye's ViT pass (we run
both eyes for stereo). Still gathers all crops across a 64-frame chunk into big
ViT batches.

Output: <out>/hands.jsonl, one JSON object per frame:
    {"frame", "width", "height", "baseline", "hands": [ {
        "is_right": 0|1, "bbox": [x1,y1,x2,y2],
        "f_px": float, "out_size": int,          # rectified virtual pinhole
        "kp_left":  [[x,y]*21],  "kp_right": [[x,y]*21],   # rectified crop px
        "keypoints_2d": [[x,y]*21],              # left kp back-mapped to L fisheye (viz)
        "keypoints_3d": [[x,y,z]*21],            # MANO joints, hand-root frame (left)
    }, ... ]}
"""
import argparse
import json
import os
import time

import numpy as np
import torch

import fisheye_pinhole as FP
import wilor_hands_batched as W


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video", help="8-bit stereo video (left|right side-by-side)")
    p.add_argument("--out", required=True)
    p.add_argument("--calib-dir", default="long-test1")
    p.add_argument("--left-serial", default="046060323008")
    p.add_argument("--right-serial", default="046060323001")
    p.add_argument("--chunk", type=int, default=64)
    p.add_argument("--vit-batch", type=int, default=32)
    p.add_argument("--conf", type=float, default=0.3)
    p.add_argument("--out-size", type=int, default=256)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--fp32", action="store_true")
    return p.parse_args()


def load_calib(calib_dir, ls, rs, device):
    cl = json.load(open(os.path.join(calib_dir, f"calib_{ls}.json")))["intrinsics"]
    cr = json.load(open(os.path.join(calib_dir, f"calib_{rs}.json")))["intrinsics"]
    st = json.load(open(os.path.join(calib_dir, f"stereo_{ls}_{rs}.json")))
    t = lambda x: torch.tensor(x, device=device, dtype=torch.float32)
    Kl, Dl, Kr, Dr = t(cl["K"]), t(cl["dist"]), t(cr["K"]), t(cr["dist"])
    Rs, ts = t(st["R"]), t(st["t"])
    baseline = float(np.linalg.norm(st["t"]))
    b_hat = -Rs.T @ ts
    b_hat = b_hat / torch.linalg.norm(b_hat)
    return Kl, Dl, Kr, Dr, Rs, b_hat, baseline


def postprocess_crop(o_i, is_right, out_size, pipe):
    """ViT output (single hand) -> keypoints in rectified-crop pixels.

    Same as wilor postprocess but with img_size = the crop (out_size square) and
    centre at crop centre. ``o_i`` values are already (1,...) numpy.
    """
    isz = np.array([out_size, out_size])
    o_i = W.postprocess(o_i, isz / 2.0, out_size, isz, is_right, pipe)
    return o_i


def run_chunk(pipe, dtype, calib, frames_l_u8, frames_r_u8, frame_idxs,
              conf, vit_batch, out_size, writer):
    Kl, Dl, Kr, Dr, Rs, b_hat, baseline = calib
    dev = pipe.device
    n = frames_l_u8.shape[0]
    H, Wd = frames_l_u8.shape[2], frames_l_u8.shape[3]
    frames_l_f = frames_l_u8.float()
    frames_r_f = frames_r_u8.float()

    # 1) detect on the left eye (GPU)
    dets = W.detect_gpu(pipe.hand_detector, frames_l_u8, conf)

    # 2) per hand: render rectified L+R pinhole crops + geometry
    crops_l, crops_r, meta = [], [], []  # meta: (li, bbox, is_right, geo)
    per_frame_hands = [[] for _ in range(n)]
    for li, boxes in enumerate(dets):
        for b in boxes:
            is_right = int(round(float(b[5])))
            bbox = torch.tensor(b[:4], device=dev)
            cL, cR, geo = FP.render_stereo_crop(
                frames_l_f[li], frames_r_f[li], bbox, Kl, Dl, Kr, Dr, Rs, b_hat,
                out_size=out_size)
            # WiLoR trains on right hands: flip LEFT-hand crops before the ViT.
            if is_right == 0:
                cL = torch.flip(cL, dims=[2])
                cR = torch.flip(cR, dims=[2])
            crops_l.append(cL)
            crops_r.append(cR)
            meta.append((li, b[:4], is_right, geo))

    if not meta:
        for li, fi in enumerate(frame_idxs):
            writer.write(json.dumps({"frame": int(fi), "width": Wd, "height": H,
                                     "baseline": baseline, "hands": []}) + "\n")
        return

    # 3) batch BOTH eyes' crops through the ViT together
    all_crops = torch.stack(crops_l + crops_r).permute(0, 2, 3, 1).to(dev, dtype)
    outs = {}
    for s in range(0, all_crops.shape[0], vit_batch):
        with torch.no_grad():
            o = pipe.wilor_model(all_crops[s:s + vit_batch])
        o = {k: v.detach().cpu().float().numpy() for k, v in o.items()}
        for k, v in o.items():
            outs.setdefault(k, []).append(v)
    outs = {k: np.concatenate(v, 0) for k, v in outs.items()}
    nL = len(crops_l)

    # 4) per hand: postprocess both eyes in crop coords, back-map left for viz
    for j, (li, bbox, is_right, geo) in enumerate(meta):
        oL = postprocess_crop({k: v[[j]] for k, v in outs.items()},
                              is_right, out_size, pipe)
        oR = postprocess_crop({k: v[[nL + j]] for k, v in outs.items()},
                              is_right, out_size, pipe)
        kpL = oL["pred_keypoints_2d"][0]   # rectified-crop px, left eye
        kpR = oR["pred_keypoints_2d"][0]   # rectified-crop px, right eye
        # left crop kp -> left fisheye px (for overlay / viz)
        kpL_fish = FP.crop_px_to_fisheye(
            torch.tensor(kpL, device=dev, dtype=torch.float32),
            geo["f_px"], out_size, geo["Rv_l"], Kl, Dl).cpu().numpy()
        per_frame_hands[li].append({
            "is_right": is_right,
            "bbox": np.asarray(bbox).ravel().tolist(),
            "f_px": float(geo["f_px"]),
            "out_size": out_size,
            "kp_left": kpL.tolist(),
            "kp_right": kpR.tolist(),
            "keypoints_2d": kpL_fish.tolist(),
            "keypoints_3d": oL["pred_keypoints_3d"][0].tolist(),
        })

    for li, fi in enumerate(frame_idxs):
        writer.write(json.dumps({"frame": int(fi), "width": Wd, "height": H,
                                 "baseline": baseline,
                                 "hands": per_frame_hands[li]}) + "\n")


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    pipe, dev, dtype = W.build_pipeline(args.fp32)
    calib = load_calib(args.calib_dir, args.left_serial, args.right_serial, dev)

    from torchcodec.decoders import VideoDecoder
    dec = VideoDecoder(args.video, device="cuda")
    total = dec.metadata.num_frames
    half = dec.metadata.width // 2
    n_frames = total if args.max_frames is None else min(total, args.max_frames)
    writer = open(os.path.join(args.out, "hands.jsonl"), "w")
    print(f"{args.video}: {dec.metadata.width}x{dec.metadata.height}, "
          f"processing {n_frames} (chunk={args.chunk}, vit_batch={args.vit_batch})")

    t0 = time.time()
    done = 0
    for cs in range(0, n_frames, args.chunk):
        idxs = list(range(cs, min(cs + args.chunk, n_frames)))
        batch = dec.get_frames_in_range(start=idxs[0], stop=idxs[-1] + 1).data
        fl = batch[:, :, :, :half].contiguous()
        fr = batch[:, :, :, half:].contiguous()
        run_chunk(pipe, dtype, calib, fl, fr, idxs, args.conf, args.vit_batch,
                  args.out_size, writer)
        done += len(idxs)
        el = time.time() - t0
        print(f"[{done}/{n_frames}] {done/el:.1f} fps", flush=True)
    writer.close()
    print(f"done in {time.time()-t0:.0f}s -> {args.out}/hands.jsonl")


if __name__ == "__main__":
    main()
