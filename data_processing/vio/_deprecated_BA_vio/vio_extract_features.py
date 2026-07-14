"""VIO pipeline stage 1: SuperPoint feature extraction with a fisheye FOV mask,
cached to HDF5 for reuse by the (separately re-runnable) matching stage.

Extraction is the GPU-heavy, non-reprocessable-cheaply step, so its output is
cached independently of matches: changing the matching pair set (e.g. a wider
temporal window) later doesn't require re-running this stage.

DECODE (--decoder, default torchcodec): torchcodec/NVDEC decodes + crops +
normalizes on the GPU -- no host decode, no CPU->GPU upload. Profiled 2.9ms/
frame vs cv2's 24ms (long-test1, A6000), collapsing ~43% of stage-1 wall to
~8%. --amp adds fp16 autocast on the forward (~1.5x); combined ~2.3x.
  CAVEAT: NVDEC and cv2/CPU H.264 decode don't produce bit-identical pixels
  (different-but-valid IDCT rounding), so torchcodec features differ slightly
  from a cv2-decoded run -- equivalent quality (descriptor cosine ~0.999) but
  NOT the same bytes; use --decoder cv2 for bit-exact reproduction. fp16 itself
  is clean (torchcodec+amp ~= torchcodec-fp32) -- the drift is all the decoder.

Run (from data_processing/vio/, matching this pipeline's CWD convention):
    python vio_extract_features.py ../../long-test1 --out ../../long-test1/features.h5
"""
import argparse
import json
import os

import cv2
import h5py
import numpy as np
import torch

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fisheye_pinhole as FP

from lightglue import SuperPoint

FORMAT_TAG = "hteng-camera-vio-features/1"
DESCRIPTOR_DIM = 256


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("recording", help="recording dir with left.mp4/right.mp4/recording.json")
    p.add_argument("--out", default=None, help="output .h5 (default: <recording>/derived/features.h5)")
    p.add_argument("--left-serial", default=None, help="default: read from recording.json")
    p.add_argument("--right-serial", default=None)
    p.add_argument("--fov-deg", type=float, default=150.0,
                    help="full FOV kept (keypoints beyond this angle are dropped)")
    p.add_argument("--max-keypoints", type=int, default=1024)
    p.add_argument("--detection-threshold", type=float, default=0.001,
                    help="SuperPoint keypoint score threshold; raising it keeps "
                         "only high-confidence, more repeatable (less jittery) "
                         "keypoints at the cost of count")
    p.add_argument("--resize", type=int, default=1500,
                    help="SuperPoint resizes the crop's long side to this before "
                         "detection; the FOV crop is often larger (130deg ~1785px), "
                         "so raising this recovers real resolution. 0 = no resize")
    p.add_argument("--batch-size", type=int, default=16,
                    help="frames per SuperPoint forward pass (crops are same-size, "
                         "so batching is a large speedup over one-at-a-time)")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--decoder", choices=["torchcodec", "cv2"], default="torchcodec",
                    help="torchcodec: NVDEC GPU decode, frames land in GPU memory "
                         "with no CPU->GPU upload (profiled ~1.9ms/frame vs cv2's "
                         "~24ms/frame decode+upload). cv2: CPU decode fallback "
                         "(needs no ffmpeg-with-nvdec / torchcodec install)")
    p.add_argument("--amp", action="store_true",
                    help="run SuperPoint under fp16 autocast (the conv backbone is "
                         "~half the extraction wall). Off by default pending the "
                         "A/B agreement sweep; same lever as LightGlue's mp=True")
    return p.parse_args()


def load_intrinsics(recording_dir, serial, device):
    calib = json.load(open(os.path.join(recording_dir, f"calib_{serial}.json")))
    intr = calib["intrinsics"]
    t = lambda x: torch.tensor(x, device=device, dtype=torch.float32)
    return t(intr["K"]), t(intr["dist"])


def fov_mask(keypoints_xy, K, dist, theta_max):
    """Boolean mask: True where a keypoint's angle from the optical axis is
    <= theta_max (radians). Reuses fisheye_pinhole's KB inversion (already
    validated for pinhole-crop rendering) rather than re-deriving it."""
    if keypoints_xy.shape[0] == 0:
        return torch.zeros(0, dtype=torch.bool, device=keypoints_xy.device)
    rays = FP.fisheye_unproject(keypoints_xy[:, 0], keypoints_xy[:, 1], K, dist)
    theta = torch.arccos(rays[:, 2].clamp(-1, 1))
    return theta <= theta_max


def fov_crop_box(K, dist, theta_max, img_w, img_h):
    """Pixel bounding box (x0, y0, x1, y1) tight around the FOV mask circle,
    clipped to the frame. Cropping to this BEFORE running SuperPoint (rather
    than resizing the full frame, most of which gets masked out anyway) buys
    real effective resolution in the kept region for free: SuperPoint always
    resizes its input's long side to a fixed budget, so shrinking the input
    to just the region we keep means more of that budget is spent on pixels
    we actually use, and the crop is also strictly cheaper to resize (fewer
    total input pixels)."""
    K_np = K.cpu().numpy().astype(np.float64)
    dist_np = dist.cpu().numpy().astype(np.float64).reshape(4, 1)
    az = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    rays = np.stack([
        np.sin(theta_max) * np.cos(az), np.sin(theta_max) * np.sin(az),
        np.full_like(az, np.cos(theta_max)),
    ], axis=-1).reshape(-1, 1, 3)
    px, _ = cv2.fisheye.projectPoints(rays, np.zeros((3, 1)), np.zeros((3, 1)), K_np, dist_np)
    px = px.reshape(-1, 2)
    cx, cy = K_np[0, 2], K_np[1, 2]
    r = float(np.sqrt(((px - [cx, cy]) ** 2).sum(axis=1)).max())
    x0, y0 = max(0, int(cx - r)), max(0, int(cy - r))
    x1, y1 = min(img_w, int(cx + r) + 1), min(img_h, int(cy + r) + 1)
    return x0, y0, x1, y1


def read_frames_rgb(path, max_frames=None):
    cap = cv2.VideoCapture(path)
    n = 0
    while max_frames is None or n < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        n += 1
    cap.release()


def to_tensor(img, device):
    return torch.from_numpy(img).float().permute(2, 0, 1).to(device) / 255.0


def crop_batches_cv2(video_path, n_frames, box, device, batch_size):
    """CPU-decode path: cv2 decodes + BGR->RGB on the host, then each frame's
    FOV crop is uploaded to the GPU as a CHW float. Yields (base_index, list
    of (3,H,W) GPU float tensors)."""
    x0, y0, x1, y1 = box
    batch, base = [], 0
    for i, img in enumerate(read_frames_rgb(video_path, n_frames)):
        batch.append(to_tensor(img[y0:y1, x0:x1], device))
        if len(batch) == batch_size:
            yield base, batch
            base, batch = i + 1, []
    if batch:
        yield base, batch


def crop_batches_torchcodec(video_path, n_frames, box, device, batch_size):
    """NVDEC GPU-decode path: torchcodec decodes a contiguous frame range
    straight into GPU memory (B,3,H,W uint8), so the FOV crop + uint8->float
    normalize happen on-device with no host decode or CPU->GPU upload (~2ms/
    frame vs cv2's ~24). Same (base_index, list of (3,H,W) GPU float tensors)
    contract as crop_batches_cv2 so extract_eye is decoder-agnostic."""
    from torchcodec.decoders import VideoDecoder
    x0, y0, x1, y1 = box
    dec = VideoDecoder(video_path, device=device)
    for base in range(0, n_frames, batch_size):
        stop = min(base + batch_size, n_frames)
        frames = dec[base:stop]  # (b,3,H,W) uint8 on GPU; tail slice clamps
        crops = frames[:, :, y0:y1, x0:x1].float() / 255.0
        yield base, [crops[j] for j in range(crops.shape[0])]


def extract_batch(extractor, crops, resize, amp=False):
    """Run SuperPoint on a stack of same-size crops (B,3,H,W). All crops share
    one resize scale, so preprocess the batch once and call forward directly
    (extract() is batch=1 only). Returns per-image (kp, score, descriptor).
    amp=True runs the forward under fp16 autocast (see --amp)."""
    from lightglue.utils import ImagePreprocessor
    conf = {} if resize <= 0 else {"resize": resize}
    imgs, scales = ImagePreprocessor(**{**extractor.preprocess_conf, **conf})(crops)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=amp):
        feats = extractor.forward({"image": imgs})
    kps = (feats["keypoints"] + 0.5) / scales[None] - 0.5
    return kps, feats["keypoint_scores"], feats["descriptors"]


def extract_eye(f, eye, video_path, K, D, theta_max, extractor, device, max_frames,
                resize, batch_size, decoder="torchcodec", amp=False):
    cap = cv2.VideoCapture(video_path)
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    n_frames = min(n_total, max_frames) if max_frames else n_total

    x0, y0, x1, y1 = fov_crop_box(K, D, theta_max, img_w, img_h)
    print(f"  {eye}: cropping to ({x0},{y0})-({x1},{y1}) "
          f"[{x1-x0}x{y1-y0}] before extraction (native {img_w}x{img_h}), "
          f"decoder={decoder} amp={amp}")

    grp = f.create_group(eye)
    vlen_f32 = h5py.vlen_dtype(np.dtype("float32"))
    vlen_f16 = h5py.vlen_dtype(np.dtype("float16"))
    kp_ds = grp.create_dataset("keypoints", (n_frames,), dtype=vlen_f32)
    sc_ds = grp.create_dataset("scores", (n_frames,), dtype=vlen_f32)
    de_ds = grp.create_dataset("descriptors", (n_frames,), dtype=vlen_f16)
    n_ds = grp.create_dataset("counts", (n_frames,), dtype=np.int32)
    off = torch.tensor([x0, y0], device=device, dtype=torch.float32)

    def store(i, kp, sc, de):
        kp = kp + off  # crop-local -> full-frame coords
        mask = fov_mask(kp, K, D, theta_max)
        kp, sc, de = kp[mask], sc[mask], de[mask]
        kp_ds[i] = kp.cpu().numpy().astype(np.float32).ravel()
        sc_ds[i] = sc.cpu().numpy().astype(np.float32)
        de_ds[i] = de.cpu().numpy().astype(np.float16).ravel()
        n_ds[i] = kp.shape[0]

    def flush(base, crops):
        try:
            kps, scs, des = extract_batch(extractor, torch.stack(crops), resize, amp)
            for j in range(len(crops)):
                store(base + j, kps[j], scs[j], des[j])
        except RuntimeError:
            # forward() stacks per-frame results, so a batch with uneven
            # keypoint counts (a low-texture frame below the cap) can't stack;
            # fall back to per-frame extract() for this batch.
            r = resize if resize > 0 else None
            for j, crop in enumerate(crops):
                with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                    feat = extractor.extract(crop, resize=r)
                store(base + j, feat["keypoints"][0], feat["keypoint_scores"][0],
                      feat["descriptors"][0])

    box = (x0, y0, x1, y1)
    make_batches = crop_batches_torchcodec if decoder == "torchcodec" else crop_batches_cv2
    for base, crops in make_batches(video_path, n_frames, box, device, batch_size):
        flush(base, crops)
        if base % 200 < batch_size:
            print(f"  {eye} {base}/{n_frames}")

    print(f"{eye}: {n_frames} frames done")


def main():
    args = parse_args()
    device = args.device

    rec = json.load(open(os.path.join(args.recording, "recording.json")))
    ls = args.left_serial or rec["left"]["serial"]
    rs = args.right_serial or rec["right"]["serial"]
    Kl, Dl = load_intrinsics(args.recording, ls, device)
    Kr, Dr = load_intrinsics(args.recording, rs, device)
    theta_max = np.radians(args.fov_deg / 2.0)

    extractor = SuperPoint(max_num_keypoints=args.max_keypoints,
                           detection_threshold=args.detection_threshold).eval().to(device)

    out_path = args.out or os.path.join(args.recording, "derived", "features.h5")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.attrs["format"] = FORMAT_TAG
        f.attrs["fov_deg"] = args.fov_deg
        f.attrs["max_keypoints"] = args.max_keypoints
        f.attrs["descriptor_dim"] = DESCRIPTOR_DIM
        f.attrs["left_serial"] = ls
        f.attrs["right_serial"] = rs
        f.attrs["fps"] = rec.get("fps", 30)

        f.attrs["resize"] = args.resize
        extract_eye(f, "left", os.path.join(args.recording, "left.mp4"),
                    Kl, Dl, theta_max, extractor, device, args.max_frames,
                    args.resize, args.batch_size, args.decoder, args.amp)
        extract_eye(f, "right", os.path.join(args.recording, "right.mp4"),
                    Kr, Dr, theta_max, extractor, device, args.max_frames,
                    args.resize, args.batch_size, args.decoder, args.amp)

    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
