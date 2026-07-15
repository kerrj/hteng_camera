"""Batched WiLoR hand pose on rectified PINHOLE crops, both eyes — sets up depth.

Instead of feeding WiLoR a raw square crop, this renders an undistorted
virtual-pinhole view aimed at each hand (fisheye_pinhole).
Detection runs on BOTH eyes independently; per handedness we match the largest
bbox in each eye, triangulate their centre rays to a common 3D point P (closest
point to the skew rays), and render a *baseline-aligned VERGED* left+right
pinhole pair aimed at P (shared focal). Both crops are centred on the hand and
rows stay epipolar (horizontal), but the optical axes verge — so this is NOT a
parallel-axis rig: depth is recovered by the stereo optimizer from each eye's
SO3 + pinhole projection, NOT f*baseline/disparity. We run the ViT on both eyes'
crops, batched across each chunk, and save both keypoint sets in crop pixels.

Pinhole rendering replaces the raw crop operation. Both eyes are required for
stereo, and all crops from a frame chunk are gathered into large ViT batches.

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

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fisheye_pinhole as FP
import wilor_runtime as W

# Reject a stereo hand whose triangulated depth is below this (metres). A hand
# can't be this close; sub-0.1m (often negative) P comes from near-parallel or
# mismatched left/right bbox-centre rays and would otherwise create a 1/z^2
# projection singularity that destabilises the downstream solve.
MIN_DEPTH_M = 0.1
_REJECT = {"n": 0}   # count of hands rejected for implausible triangulation


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("left", help="left-eye video (torchcodec decodes the 10-bit "
                                "per-eye files fine, ~51fps -- no side-by-side "
                                "8-bit transcode needed)")
    p.add_argument("right", help="right-eye video")
    p.add_argument("--out", default=None,
                   help="output dir for hands.jsonl (default: <calib-dir>/derived)")
    p.add_argument("--calib-dir", default="../../long-test1")
    p.add_argument("--left-serial", default="046060323008")
    p.add_argument("--right-serial", default="046060323001")
    p.add_argument("--chunk", type=int, default=64)
    p.add_argument("--vit-batch", type=int, default=32)
    p.add_argument("--conf", type=float, default=0.3)
    p.add_argument("--out-size", type=int, default=256)
    p.add_argument("--model-dir", default=W.default_model_dir(),
                   help="WiLoR asset cache prepared by prepare_wilor_models.py")
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
    return Kl, Dl, Kr, Dr, Rs, ts, b_hat, baseline


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
    Kl, Dl, Kr, Dr, Rs, ts, b_hat, baseline = calib
    dev = pipe.device
    n = frames_l_u8.shape[0]
    H, Wd = frames_l_u8.shape[2], frames_l_u8.shape[3]
    frames_l_f = frames_l_u8.float()
    frames_r_f = frames_r_u8.float()

    # 1) detect on BOTH eyes independently (GPU)
    dets_l = W.detect_gpu(pipe.hand_detector, frames_l_u8, conf)
    dets_r = W.detect_gpu(pipe.hand_detector, frames_r_u8, conf)

    def largest_per_hand(boxes):
        """{is_right: bbox(4,)} keeping only the largest-bbox box per handedness."""
        best = {}
        for b in boxes:
            ir = int(round(float(b[5])))
            area = (b[2] - b[0]) * (b[3] - b[1])
            if ir not in best or area > best[ir][0]:
                best[ir] = (area, b[:4])
        return {ir: bb for ir, (_, bb) in best.items()}

    # 2) per hand: match L↔R by handedness (largest in each eye), triangulate,
    #    render VERGED rectified crops aimed at the common 3D point.
    crops_l, crops_r, meta = [], [], []  # meta: (li, bbox_l, bbox_r, is_right, geo)
    per_frame_hands = [[] for _ in range(n)]
    for li in range(n):
        bl = largest_per_hand(dets_l[li])
        br = largest_per_hand(dets_r[li])
        for is_right in (bl.keys() & br.keys()):   # need the hand in BOTH eyes
            bbox_l = torch.tensor(bl[is_right], device=dev)
            bbox_r = torch.tensor(br[is_right], device=dev)
            # REJECT implausible triangulation before rendering. Near-parallel or
            # mismatched (left/right boxes on different things) bbox-centre rays
            # give a closest-point P near/behind the baseline (z<0.1m, sometimes
            # negative). Such geometry cannot define a useful virtual crop and
            # is rejected before inference.
            g_l = FP.bbox_center_ray(bbox_l, Kl, Dl)
            g_r = FP.bbox_center_ray(bbox_r, Kr, Dr)
            P_chk = FP.triangulate_rays(g_l, g_r, Rs, ts)
            if float(P_chk[2]) < MIN_DEPTH_M:
                _REJECT["n"] += 1
                continue
            cL, cR, geo = FP.render_stereo_crop(
                frames_l_f[li], frames_r_f[li], bbox_l, bbox_r,
                Kl, Dl, Kr, Dr, Rs, ts, b_hat, out_size=out_size)
            # WiLoR trains on right hands: flip LEFT-hand crops before the ViT.
            if is_right == 0:
                cL = torch.flip(cL, dims=[2])
                cR = torch.flip(cR, dims=[2])
            crops_l.append(cL)
            crops_r.append(cR)
            meta.append((li, bl[is_right], br[is_right], is_right, geo))

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
    for j, (li, bbox_l, bbox_r, is_right, geo) in enumerate(meta):
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
            "bbox": np.asarray(bbox_l).ravel().tolist(),       # left-eye bbox
            "bbox_right": np.asarray(bbox_r).ravel().tolist(),
            "f_px": float(geo["f_px"]),
            "out_size": out_size,
            "kp_left": kpL.tolist(),
            "kp_right": kpR.tolist(),
            "keypoints_2d": kpL_fish.tolist(),
            "keypoints_3d": oL["pred_keypoints_3d"][0].tolist(),
            "P": geo["P"].cpu().numpy().tolist(),              # triangulated pt, left frame
            # MANO params (left-eye estimate) — init for stereo opt.
            # global_orient/hand_pose are axis-angle in the rectified LEFT-crop
            # camera frame (postprocess already undid the left-hand flip).
            "global_orient": np.asarray(oL["global_orient"]).reshape(-1).tolist(),
            "hand_pose": np.asarray(oL["hand_pose"]).reshape(-1).tolist(),
            # right-eye internal pose too: hand_pose is the 15 finger-joint
            # rotations expressed LOCALLY (parent-relative), so it's viewpoint-
            # invariant — both eyes measure the same hand shape. The stereo
            # optimizer averages the two as a shape regularizer.
            "hand_pose_right": np.asarray(oR["hand_pose"]).reshape(-1).tolist(),
            "betas": np.asarray(oL["betas"]).reshape(-1).tolist(),
            # Rv_l/Rv_r: rectified-crop-cam -> {left,right}-fisheye-cam rotations
            # (cols=axes). The crops VERGE on P, so the optimizer must project
            # with each eye's own SO3 + pinhole (no parallel-axis shortcut).
            "Rv_l": geo["Rv_l"].cpu().numpy().tolist(),
            "Rv_r": geo["Rv_r"].cpu().numpy().tolist(),
        })

    for li, fi in enumerate(frame_idxs):
        writer.write(json.dumps({"frame": int(fi), "width": Wd, "height": H,
                                 "baseline": baseline,
                                 "hands": per_frame_hands[li]}) + "\n")


def main():
    args = parse_args()
    if args.out is None:
        args.out = os.path.join(args.calib_dir, "derived")
    os.makedirs(args.out, exist_ok=True)
    pipe, dev, dtype = W.build_pipeline(args.fp32, args.model_dir)
    calib = load_calib(args.calib_dir, args.left_serial, args.right_serial, dev)

    # Two per-eye files: decode each on-GPU, slice nothing. torchcodec decodes
    # the 10-bit HEVC per-eye files at ~51fps (fine for a one-shot run) and
    # skips the side-by-side 8-bit transcode + its disk cost entirely.
    from torchcodec.decoders import VideoDecoder
    dec_l = VideoDecoder(args.left, device="cuda")
    dec_r = VideoDecoder(args.right, device="cuda")
    total = min(dec_l.metadata.num_frames, dec_r.metadata.num_frames)
    n_frames = total if args.max_frames is None else min(total, args.max_frames)
    writer = open(os.path.join(args.out, "hands.jsonl"), "w")
    print(f"{args.left} + {args.right}: "
          f"{dec_l.metadata.width}x{dec_l.metadata.height} per eye, "
          f"processing {n_frames} (chunk={args.chunk}, vit_batch={args.vit_batch})")

    t0 = time.time()
    done = 0
    for cs in range(0, n_frames, args.chunk):
        idxs = list(range(cs, min(cs + args.chunk, n_frames)))
        fl = dec_l.get_frames_in_range(start=idxs[0], stop=idxs[-1] + 1).data
        fr = dec_r.get_frames_in_range(start=idxs[0], stop=idxs[-1] + 1).data
        run_chunk(pipe, dtype, calib, fl, fr, idxs, args.conf, args.vit_batch,
                  args.out_size, writer)
        done += len(idxs)
        el = time.time() - t0
        print(f"[{done}/{n_frames}] {done/el:.1f} fps", flush=True)
    writer.close()
    print(f"done in {time.time()-t0:.0f}s -> {args.out}/hands.jsonl  "
          f"(rejected {_REJECT['n']} hands: triangulated depth < {MIN_DEPTH_M}m)")


if __name__ == "__main__":
    main()
