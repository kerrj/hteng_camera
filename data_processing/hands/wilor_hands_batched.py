"""Batched WiLoR hand-pose over the 8-bit stereo video — throughput-optimized.

Why this exists: the per-frame `wilor_hands.py` runs the YOLO detector and the
ViT-Huge backbone at batch 1-2 (one frame's hands), leaving the A100 idle. The
ViT is ~16x faster *per crop* at batch 16 than at batch 1, and YOLO is ~4x
faster on a list of frames. This runner decodes the 8-bit stereo video on the
GPU with torchcodec (~217 fps, both eyes per frame), runs YOLO batched over a
chunk of frames, gathers ALL hand crops across the chunk, and runs the ViT in
big batches.

Input: `left_stereo_8bit.mp4` (HEVC 8-bit, 4896x2048 = left|right side-by-side).
We process whichever eye(s) you ask for; the per-eye crop + post-processing is
byte-for-byte the same math as the upstream pipeline (reuses wilor_mini.utils).

NB: the 10-bit per-eye `left.mp4`/`right.mp4` decode slowly on NVDEC
(yuv420p10le is not a fast hardware path); the 8-bit HEVC stereo file is the
fast input and gives both eyes for the upcoming stereo-depth work.

Output: same JSONL schema as wilor_hands.py, one file per eye:
    <out>/<eye>/hands.jsonl
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as Fn
import torchvision.transforms.v2.functional as TF

# 21-joint skeleton for viz (MANO/OpenPose order)
_BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
          (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15),
          (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video", help="8-bit stereo video (left|right side-by-side)")
    p.add_argument("--out", required=True)
    p.add_argument("--eyes", default="left,right",
                   help="which eyes to process: 'left', 'right', or 'left,right'")
    p.add_argument("--chunk", type=int, default=64,
                   help="frames decoded+detected per chunk")
    p.add_argument("--vit-batch", type=int, default=32,
                   help="max hand crops per ViT forward")
    p.add_argument("--stride", type=int, default=1, help="process every Nth frame")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--conf", type=float, default=0.3)
    p.add_argument("--fp32", action="store_true")
    return p.parse_args()


def build_pipeline(fp32):
    from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import (
        WiLorHandPose3dEstimationPipeline)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32 if fp32 else torch.float16
    return WiLorHandPose3dEstimationPipeline(device=dev, dtype=dtype, verbose=False), dev, dtype


def gpu_crop(frame_chw, center, bbox_size, flip, IMG):
    """Extract one 256x256 hand patch from a GPU frame, on GPU.

    Equivalent to WiLoR's generate_image_patch_cv2 (scale=1, rot=0): take the
    square of side ``bbox_size`` centred at ``center``, optionally h-flip, and
    resize to IMG. Uses torchvision ``resize(antialias=True)`` for the
    downsample anti-aliasing WiLoR did with a Gaussian pre-blur — validated to
    <2mm 3D-keypoint agreement vs the CPU path (one extreme close-up ~9mm, a
    case the pinhole crops will supersede). Returns (3, IMG, IMG) float.
    """
    C, H, W = frame_chw.shape
    s = int(round(float(bbox_size)))
    x0 = int(round(float(center[0]) - s / 2))
    y0 = int(round(float(center[1]) - s / 2))
    pad = s  # pad by a full side so any off-image box still yields a full square
    fp = Fn.pad(frame_chw.unsqueeze(0), (pad, pad, pad, pad)).squeeze(0)
    sub = fp[:, y0 + pad:y0 + pad + s, x0 + pad:x0 + pad + s]
    if flip:
        sub = torch.flip(sub, dims=[2])
    return TF.resize(sub.unsqueeze(0), [IMG, IMG], antialias=True).squeeze(0)


def detect_gpu(detector, frames_gpu, conf, imgsz=640):
    """Run the YOLO hand detector on a GPU uint8 batch, fully on-GPU.

    ultralytics letterboxes numpy/list inputs on the CPU (~25 ms per 5MP frame —
    the dominant cost). Passing a pre-processed GPU tensor (BCHW float 0-1, dims
    multiple of 32) skips that path: ~2.8 ms/frame. We aspect-preserve-resize and
    pad with 114/255 (ultralytics' pad value), then rescale boxes back to the
    original frame. Detections match the numpy path on clear hands; they differ
    only on marginal/distorted peripheral hands (visually inspected).

    frames_gpu: (n, 3, H, W) uint8 cuda. Returns list[ (k,6) np arrays ] of
    [x1,y1,x2,y2,conf,cls] in ORIGINAL-frame pixel coords.
    """
    n, c, H, W = frames_gpu.shape
    s = min(imgsz / H, imgsz / W)
    nh, nw = int(round(H * s)), int(round(W * s))
    r = Fn.interpolate(frames_gpu.float() / 255.0, size=(nh, nw),
                       mode="bilinear", align_corners=False)
    side = ((imgsz + 31) // 32) * 32
    inp = torch.full((n, c, side, side), 114 / 255.0, device=frames_gpu.device)
    inp[:, :, :nh, :nw] = r
    results = detector(inp, conf=conf, verbose=False)
    out = []
    for res in results:
        b = res.boxes.data.detach().cpu().numpy().copy()  # (k,6) in letterbox coords
        if len(b):
            b[:, :4] /= s
        out.append(b)
    return out


def postprocess(out_i, center, bbox_size, img_size, is_right, pipe):
    """Per-hand post-processing (handedness flip, cam-to-full, 2D reprojection).

    Mirrors the tail of WiLoRHandPose3dEstimationPipeline.predict exactly.
    """
    from wilor_mini.utils import utils
    pred_cam = out_i["pred_cam"]
    multiplier = (2 * is_right - 1)
    pred_cam[:, 1] = multiplier * pred_cam[:, 1]
    if is_right == 0:
        out_i["pred_keypoints_3d"][:, :, 0] = -out_i["pred_keypoints_3d"][:, :, 0]
        out_i["pred_vertices"][:, :, 0] = -out_i["pred_vertices"][:, :, 0]
        out_i["global_orient"] = np.concatenate(
            (out_i["global_orient"][:, :, 0:1], -out_i["global_orient"][:, :, 1:3]), axis=-1)
        out_i["hand_pose"] = np.concatenate(
            (out_i["hand_pose"][:, :, 0:1], -out_i["hand_pose"][:, :, 1:3]), axis=-1)
    scaled_focal_length = pipe.FOCAL_LENGTH / pipe.IMAGE_SIZE * img_size.max()
    pred_cam_t_full = utils.cam_crop_to_full(
        pred_cam, center[None], bbox_size, img_size[None], scaled_focal_length)
    out_i["pred_cam_t_full"] = pred_cam_t_full
    out_i["scaled_focal_length"] = scaled_focal_length
    out_i["pred_keypoints_2d"] = utils.perspective_projection(
        out_i["pred_keypoints_3d"], translation=pred_cam_t_full,
        focal_length=np.array([scaled_focal_length] * 2)[None],
        camera_center=img_size[None] / 2)
    return out_i


def record(out_i, bbox, is_right):
    return {
        "is_right": int(is_right),
        "bbox": np.asarray(bbox).ravel().tolist(),
        "keypoints_2d": out_i["pred_keypoints_2d"][0].tolist(),
        "keypoints_3d": out_i["pred_keypoints_3d"][0].tolist(),
        "global_orient": np.asarray(out_i["global_orient"]).ravel().tolist(),
        "hand_pose": np.asarray(out_i["hand_pose"]).reshape(-1).tolist(),
        "betas": np.asarray(out_i["betas"]).ravel().tolist(),
        "cam_t_full": np.asarray(out_i["pred_cam_t_full"]).ravel().tolist(),
        "focal_px": float(np.asarray(out_i["scaled_focal_length"])),
    }


def run_eye_chunk(pipe, dtype, frames_gpu, frame_idxs, conf, vit_batch,
                  writers, eye):
    """Detect + pose a chunk of one eye's frames; write JSONL via writers[eye].

    frames_gpu: (n, 3, H, W) uint8 on cuda. Detection + cropping both on GPU.
    """
    IMG = pipe.IMAGE_SIZE
    n, _, H, W = frames_gpu.shape
    frames_f = frames_gpu.float()  # for cropping/grid_sample
    # 1) batched detection over the whole chunk, fully on GPU
    dets = detect_gpu(pipe.hand_detector, frames_gpu, conf)

    # 2) gather every crop across the chunk — cropping on GPU
    patches, meta = [], []   # meta: (local_idx, bbox, is_right, center, bbox_size, img_size)
    per_frame_hands = [[] for _ in range(n)]
    img_size = np.array([W, H])
    for li, boxes in enumerate(dets):
        for b in boxes:
            bbox = b[:4]
            is_right = int(round(float(b[5])))
            center = (bbox[2:4] + bbox[0:2]) / 2.0
            bbox_size = (2.5 * (bbox[2:4] - bbox[0:2])).max()
            patch = gpu_crop(frames_f[li], center, bbox_size, is_right == 0, IMG)
            patches.append(patch)
            meta.append((li, bbox, is_right, center, bbox_size, img_size))

    # 3) run the ViT in big batches (model wants NHWC)
    if patches:
        crops_t = torch.stack(patches).permute(0, 2, 3, 1).to(pipe.device, dtype)
        outs = {}
        for s in range(0, len(patches), vit_batch):
            with torch.no_grad():
                o = pipe.wilor_model(crops_t[s:s + vit_batch])
            o = {k: v.detach().cpu().float().numpy() for k, v in o.items()}
            for k, v in o.items():
                outs.setdefault(k, []).append(v)
        outs = {k: np.concatenate(v, 0) for k, v in outs.items()}

        # 4) scatter + post-process per hand
        for j, (li, bbox, is_right, center, bbox_size, isz) in enumerate(meta):
            out_i = {k: v[[j]] for k, v in outs.items()}
            out_i = postprocess(out_i, center, bbox_size, isz, is_right, pipe)
            per_frame_hands[li].append(record(out_i, bbox, is_right))

    # 5) emit one JSON line per frame (preserves frame order)
    for li, fi in enumerate(frame_idxs):
        writers[eye].write(json.dumps(
            {"frame": int(fi), "width": W, "height": H,
             "hands": per_frame_hands[li]}) + "\n")


def main():
    args = parse_args()
    eyes = [e.strip() for e in args.eyes.split(",") if e.strip()]
    writers = {}
    for e in eyes:
        os.makedirs(os.path.join(args.out, e), exist_ok=True)
        writers[e] = open(os.path.join(args.out, e, "hands.jsonl"), "w")

    pipe, dev, dtype = build_pipeline(args.fp32)

    from torchcodec.decoders import VideoDecoder
    dec = VideoDecoder(args.video, device="cuda")
    total = dec.metadata.num_frames
    W = dec.metadata.width
    half = W // 2
    idxs = list(range(0, total, args.stride))
    if args.max_frames:
        idxs = idxs[:args.max_frames]
    print(f"{args.video}: {W}x{dec.metadata.height}, {total} frames; "
          f"processing {len(idxs)} (eyes={eyes}, chunk={args.chunk}, vit_batch={args.vit_batch})")

    t0 = time.time()
    done = 0
    for cs in range(0, len(idxs), args.chunk):
        chunk_idxs = idxs[cs:cs + args.chunk]
        # GPU batched decode of the stereo frame, then split eyes on-GPU and
        # move each eye's chunk to CPU numpy HWC RGB (ultralytics + cv2 crop).
        batch = dec.get_frames_in_range(start=chunk_idxs[0],
                                        stop=chunk_idxs[-1] + 1)  # contiguous; stride handled below
        data = batch.data  # (n,3,H,W) uint8 cuda, RGB
        # honor stride within the contiguous decoded range
        sel = [i - chunk_idxs[0] for i in chunk_idxs]
        data = data[sel]
        for e in eyes:
            sub = data[:, :, :, :half] if e == "left" else data[:, :, :, half:]
            frames_gpu = sub.contiguous()                    # (n,3,H,W) uint8 cuda
            run_eye_chunk(pipe, dtype, frames_gpu, chunk_idxs, args.conf,
                          args.vit_batch, writers, e)
        done += len(chunk_idxs)
        el = time.time() - t0
        print(f"[{done}/{len(idxs)}] {done/el:.1f} fps ({el:.0f}s)", flush=True)

    for w in writers.values():
        w.close()
    print(f"done in {time.time()-t0:.0f}s -> {args.out}/<eye>/hands.jsonl")


if __name__ == "__main__":
    main()
