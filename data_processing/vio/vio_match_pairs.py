"""VIO pipeline stage 2: pairwise LightGlue matching + geometric gating.

Pairs matched, per frame i:
  - stereo:   (i, left) <-> (i, right), gated against the KNOWN stereo R,t
              (no RANSAC needed -- the geometry is already calibrated).
  - temporal: (i, eye) <-> (i+gap, eye), same eye, for a schedule of gaps
              covering ~2s (60 frames @ 30fps) with progressively sparser
              stride at longer range (dense nearby, sparse far away, since a
              wide gap only needs occasional coverage for long-baseline
              constraints -- see build_temporal_gaps). Gated via RANSAC
              essential-matrix estimation (no known relative pose exists
              between two arbitrary frames).

Both gates operate on UNPROJECTED RAYS (fisheye_pinhole.fisheye_unproject),
not raw distorted pixels -- fisheye epipolar lines are curves in pixel space,
not the straight "same row" shortcut a rectified rig would give. A pixel
tolerance (--epipolar-px-thresh) is converted to an angular/normalized
tolerance via focal length for both gates, so both use the same effective
strictness.

Two-stage filter before a match reaches a track: (1) cheap LightGlue
match-confidence threshold, (2) the geometric gate above. Matches that pass
(1) but fail (2) are kept in the output too (as "rejected") purely so the
visualizer can show what got filtered and why.

BATCHED matching (not one LightGlue call per pair): frames are grouped into
--chunk-frames windows, pairs within a chunk sorted by keypoint count, then
matched --batch-size at a time. LightGlue requires every sample in a batch to
share the same keypoint count per side, so each batch is truncated to its own
minimum count (keeping the highest-scored points) -- sorting first minimizes
how much that truncation throws away, since it clusters similarly-sized
frames together rather than batching arbitrary/sequential ones. Measured
(this recording, RTX A6000): ~12ms/pair unbatched -> ~2.8ms/pair at
batch_size=16 (~4.3x), with 98.5% match agreement vs unbatched and ~2.8%
completeness loss to truncation. `width_confidence=-1` is REQUIRED for any
batch_size > 1 -- the reference LightGlue implementation's adaptive
point-pruning optimization has a genuine indexing bug for batch>1 (crashes
with the default config; confirmed by direct testing, not a config choice).

Run (from data_processing/vio/):
    python vio_match_pairs.py ../../long-test1 --features ../../long-test1/features.h5 \
        --out ../../long-test1/matches.jsonl --max-frames 30
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

from lightglue import LightGlue


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("recording", help="recording dir with calib_*.json/stereo_*.json")
    p.add_argument("--features", default=None, help="default: <recording>/features.h5")
    p.add_argument("--out", default=None, help="default: <recording>/matches.jsonl")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--match-conf-thresh", type=float, default=0.2)
    p.add_argument("--min-raw-matches", type=int, default=10,
                    help="reject a whole pair if fewer than this many matches survive "
                         "the confidence threshold -- too few for RANSAC to be meaningful")
    p.add_argument("--epipolar-px-thresh", type=float, default=3.0)
    p.add_argument("--dense-max", type=int, default=3)
    p.add_argument("--mid-max", type=int, default=10)
    p.add_argument("--mid-stride", type=int, default=2)
    p.add_argument("--far-max", type=int, default=30)
    p.add_argument("--far-stride", type=int, default=5)
    p.add_argument("--longest-max", type=int, default=60)
    p.add_argument("--longest-stride", type=int, default=10)
    p.add_argument("--chunk-frames", type=int, default=300,
                    help="frames processed per chunk before re-sorting for batching -- "
                         "just bounds memory, all features are already extracted so "
                         "this isn't a streaming-window correctness constraint")
    p.add_argument("--batch-size", type=int, default=16,
                    help="pairs per LightGlue batch call -- 16 captures ~all the "
                         "available speedup (see module docstring sweep); larger "
                         "batches add truncation waste for negligible extra speed")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def build_temporal_gaps(dense_max, mid_max, mid_stride, far_max, far_stride,
                         longest_max, longest_stride):
    """Dense nearby, progressively sparser further out -- covers up to
    longest_max frames (2s @ 30fps by default) without O(longest_max) pairs
    per frame."""
    gaps = set(range(1, dense_max + 1))
    gaps |= set(range(dense_max + 1, mid_max + 1, mid_stride))
    gaps |= set(range(mid_max, far_max + 1, far_stride))
    gaps |= set(range(far_max, longest_max + 1, longest_stride))
    return sorted(gaps)


def load_intrinsics(recording_dir, serial):
    calib = json.load(open(os.path.join(recording_dir, f"calib_{serial}.json")))["intrinsics"]
    return np.array(calib["K"], np.float64), np.array(calib["dist"], np.float64)


def load_stereo(recording_dir, ls, rs):
    st = json.load(open(os.path.join(recording_dir, f"stereo_{ls}_{rs}.json")))
    return np.array(st["R"], np.float64), np.array(st["t"], np.float64).reshape(3)


class FeatureStore:
    """Thin cache over the stage-1 HDF5 so repeated frame lookups (every
    frame appears in ~O(len(gaps)) pairs) don't re-read from disk."""

    def __init__(self, h5_path, device):
        self.f = h5py.File(h5_path, "r")
        self.device = device
        self._cache = {}

    def get(self, eye, i):
        key = (eye, i)
        if key not in self._cache:
            g = self.f[eye]
            n = int(g["counts"][i])
            kp = g["keypoints"][i].reshape(n, 2).astype(np.float32)
            sc = g["scores"][i].astype(np.float32)
            de = g["descriptors"][i].reshape(n, -1).astype(np.float32)
            self._cache[key] = (kp, sc, de)
        return self._cache[key]

    def count(self, eye, i):
        return int(self.f[eye]["counts"][i])

    def n_frames(self, eye):
        return self.f[eye]["counts"].shape[0]

    def close(self):
        self.f.close()


def unproject_batch(pts, K, dist, device):
    """Unproject a big flat (Total,2) pixel array in ONE call. Profiling
    found this was 67% of stage-2's gating time at ~4ms/pair when called
    per-pair-per-side (rays_for, now removed) -- each call was tiny (a few
    hundred points) but paid a GPU round-trip every time. Unprojection has
    NO cross-point interaction (a per-point Newton solve), unlike LightGlue's
    attention, so concatenating points from many pairs into one call is
    exact, not an approximation -- there's nothing to mask or truncate."""
    pts_t = torch.from_numpy(pts.astype(np.float32)).to(device)
    K_t = torch.tensor(K, dtype=torch.float32, device=device)
    d_t = torch.tensor(dist, dtype=torch.float32, device=device)
    return FP.fisheye_unproject(pts_t[:, 0], pts_t[:, 1], K_t, d_t).cpu().numpy()


def gate_stereo(rays_l, rays_r, R, t, theta_tol):
    """Known-geometry coplanarity gate: for a correct match, ray_r must lie
    in the epipolar plane spanned by the baseline t and (R @ ray_l) -- no
    RANSAC needed since R, t are already calibrated, not estimated."""
    t_hat = t / np.linalg.norm(t)
    n = np.cross(t_hat[None, :], rays_l @ R.T)
    norms = np.linalg.norm(n, axis=1, keepdims=True)
    norms = np.where(norms < 1e-9, 1.0, norms)
    n = n / norms
    residual = np.abs(np.sum(rays_r * n, axis=1))
    return residual < np.sin(theta_tol)


def gate_temporal_ransac(rays_a, rays_b, theta_tol):
    """RANSAC essential-matrix inlier gate on normalized ray bearings (no
    known relative pose between two arbitrary frames, unlike the stereo
    pair). cameraMatrix=I because rays are already unit bearings, so
    `threshold` is in the same normalized/angular units as theta_tol."""
    if rays_a.shape[0] < 8:
        return np.zeros(rays_a.shape[0], dtype=bool)
    pts_a = (rays_a[:, :2] / rays_a[:, 2:3]).astype(np.float64)
    pts_b = (rays_b[:, :2] / rays_b[:, 2:3]).astype(np.float64)
    E, mask = cv2.findEssentialMat(pts_a, pts_b, cameraMatrix=np.eye(3),
                                    method=cv2.RANSAC, prob=0.999,
                                    threshold=theta_tol)
    if mask is None:
        return np.zeros(rays_a.shape[0], dtype=bool)
    return mask.ravel().astype(bool)


def top_k_order(scores, k):
    """Indices (into the ORIGINAL 0..len(scores)-1 array) of the k
    highest-scored keypoints, or all of them if there are <= k."""
    if scores.shape[0] <= k:
        return np.arange(scores.shape[0])
    return np.argsort(-scores)[:k]


def match_batch(matcher, store, specs, conf_thresh, device):
    """specs: list of (eye_a, frame_a, eye_b, frame_b). Truncates every pair
    in the batch to this batch's minimum keypoint count per side (keeping the
    highest-scored points), runs ONE LightGlue call, then maps results back
    to indices into the ORIGINAL (untruncated) per-frame keypoint arrays --
    callers never see batch-local indices. Returns a list of (idx (N,2)
    int64, scores (N,) float32), aligned with `specs`."""
    counts_a = [store.count(ea, fa) for ea, fa, _, _ in specs]
    counts_b = [store.count(eb, fb) for _, _, eb, fb in specs]
    M, N = min(counts_a), min(counts_b)

    kp0, sc0, de0, orders_a = [], [], [], []
    kp1, sc1, de1, orders_b = [], [], [], []
    for ea, fa, eb, fb in specs:
        kpa, sca, dea = store.get(ea, fa)
        kpb, scb, deb = store.get(eb, fb)
        oa, ob = top_k_order(sca, M), top_k_order(scb, N)
        kp0.append(kpa[oa]); sc0.append(sca[oa]); de0.append(dea[oa]); orders_a.append(oa)
        kp1.append(kpb[ob]); sc1.append(scb[ob]); de1.append(deb[ob]); orders_b.append(ob)

    t = lambda arr: torch.from_numpy(np.stack(arr)).to(device)
    feats0 = {"keypoints": t(kp0), "keypoint_scores": t(sc0), "descriptors": t(de0)}
    feats1 = {"keypoints": t(kp1), "keypoint_scores": t(sc1), "descriptors": t(de1)}
    with torch.no_grad():
        out = matcher({"image0": feats0, "image1": feats1})

    results = []
    for row in range(len(specs)):
        idx = out["matches"][row].cpu().numpy()
        scores = out["scores"][row].cpu().numpy()
        keep = scores >= conf_thresh
        idx, scores = idx[keep], scores[keep]
        if idx.shape[0] == 0:
            results.append((np.zeros((0, 2), dtype=np.int64), np.zeros((0,), dtype=np.float32)))
            continue
        oa, ob = orders_a[row], orders_b[row]
        idx_orig = np.stack([oa[idx[:, 0]], ob[idx[:, 1]]], axis=1)
        results.append((idx_orig, scores))
    return results


def write_reject_too_few(eye_a, i, eye_b, j, pair_type, gap, idx, out_f):
    # Too few points left for RANSAC to be meaningful: the 5-point
    # essential-matrix algorithm needs 5 points just to fit ONE candidate
    # model, so with barely more than that there's no redundancy left to
    # actually detect an outlier -- confirmed by direct inspection: some
    # long-gap (40-60 frame) pairs during fast head motion had as few as
    # 2-4 raw matches. Reject the whole pair rather than trust a degenerate
    # fit. (Cheap to decide -- no need for this pair's points to go through
    # the batched ray-unprojection step below at all.)
    rec = {
        "eye_a": eye_a, "frame_a": i, "eye_b": eye_b, "frame_b": j,
        "pair_type": pair_type, "gap": gap,
        "n_raw": idx.shape[0], "n_geom": 0,
        "idx_a": [], "idx_b": [],
        "rejected_idx_a": idx[:, 0].tolist(), "rejected_idx_b": idx[:, 1].tolist(),
    }
    out_f.write(json.dumps(rec) + "\n")


def gate_chunk(store, eligible, cams, args, out_f):
    """eligible: list of (eye_a, i, eye_b, j, pair_type, gap, idx, scores),
    already past the min-raw-matches filter. Does ONE ray-unprojection call
    per eye for the WHOLE chunk (not one per pair per side -- see
    unproject_batch), then the existing per-pair RANSAC/stereo gate and
    write, unchanged."""
    # gather every point needing unprojection, grouped by EYE (not by
    # "side a/b" -- unprojection only depends on which eye's K/dist to use,
    # so a pair's side-a and side-b points can land in either eye's group).
    buf_pts = {"left": [], "right": []}
    buf_meta = {"left": [], "right": []}  # (pair_idx, side, n) in append order
    for pair_idx, (ea, i, eb, j, pair_type, gap, idx, scores) in enumerate(eligible):
        kpa, _, _ = store.get(ea, i)
        kpb, _, _ = store.get(eb, j)
        pts_a, pts_b = kpa[idx[:, 0]], kpb[idx[:, 1]]
        buf_pts[ea].append(pts_a); buf_meta[ea].append((pair_idx, "a", pts_a.shape[0]))
        buf_pts[eb].append(pts_b); buf_meta[eb].append((pair_idx, "b", pts_b.shape[0]))

    rays_by_pair_side = {}
    for eye in ("left", "right"):
        if not buf_pts[eye]:
            continue
        all_pts = np.concatenate(buf_pts[eye], axis=0)
        K, D = cams[eye]
        all_rays = unproject_batch(all_pts, K, D, args.device)
        offset = 0
        for pair_idx, side, n in buf_meta[eye]:
            rays_by_pair_side[(pair_idx, side)] = all_rays[offset:offset + n]
            offset += n

    for pair_idx, (eye_a, i, eye_b, j, pair_type, gap, idx, scores) in enumerate(eligible):
        rays_a = rays_by_pair_side[(pair_idx, "a")]
        rays_b = rays_by_pair_side[(pair_idx, "b")]
        Ka, Da = cams[eye_a]
        Kb, Db = cams[eye_b]
        f_avg = (Ka[0, 0] + Ka[1, 1] + Kb[0, 0] + Kb[1, 1]) / 4.0
        theta_tol = args.epipolar_px_thresh / f_avg

        if pair_type == "stereo":
            R, t = cams["stereo"]
            inlier = gate_stereo(rays_a, rays_b, R, t, theta_tol)
        else:
            inlier = gate_temporal_ransac(rays_a, rays_b, theta_tol)

        rec = {
            "eye_a": eye_a, "frame_a": i, "eye_b": eye_b, "frame_b": j,
            "pair_type": pair_type, "gap": gap,
            "n_raw": idx.shape[0], "n_geom": int(inlier.sum()),
            "idx_a": idx[inlier, 0].tolist(), "idx_b": idx[inlier, 1].tolist(),
            "rejected_idx_a": idx[~inlier, 0].tolist(), "rejected_idx_b": idx[~inlier, 1].tolist(),
        }
        out_f.write(json.dumps(rec) + "\n")


def chunk_pair_specs(chunk_start, chunk_end, n_frames, gaps):
    """(eye_a, frame_a, eye_b, frame_b, pair_type, gap) for every frame_a in
    [chunk_start, chunk_end) -- frame_b may extend past chunk_end (up to
    n_frames-1), which is fine since all features are already extracted."""
    specs = []
    for i in range(chunk_start, chunk_end):
        specs.append(("left", i, "right", i, "stereo", 0))
        for eye in ("left", "right"):
            for d in gaps:
                j = i + d
                if j < n_frames:
                    specs.append((eye, i, eye, j, "temporal", d))
    return specs


def main():
    args = parse_args()
    features_path = args.features or os.path.join(args.recording, "features.h5")
    out_path = args.out or os.path.join(args.recording, "matches.jsonl")
    gaps = build_temporal_gaps(args.dense_max, args.mid_max, args.mid_stride,
                                args.far_max, args.far_stride,
                                args.longest_max, args.longest_stride)
    print(f"temporal gaps: {gaps}")

    store = FeatureStore(features_path, args.device)
    ls = store.f.attrs["left_serial"]
    rs = store.f.attrs["right_serial"]
    Kl, Dl = load_intrinsics(args.recording, ls)
    Kr, Dr = load_intrinsics(args.recording, rs)
    R, t = load_stereo(args.recording, ls, rs)
    cams = {"left": (Kl, Dl), "right": (Kr, Dr), "stereo": (R, t)}

    n_frames = min(store.n_frames("left"), store.n_frames("right"))
    if store.n_frames("left") != store.n_frames("right"):
        print(f"WARNING: left has {store.n_frames('left')} frames, right has "
              f"{store.n_frames('right')} -- using the shorter of the two "
              f"({n_frames}) as the bound for both eyes")
    if args.max_frames:
        n_frames = min(n_frames, args.max_frames)

    # width_confidence=-1 is REQUIRED for batch_size > 1: the reference
    # implementation's adaptive point-pruning has a genuine indexing bug for
    # batch>1 (confirmed by direct testing -- crashes with the default
    # config), not a speed/quality tradeoff we're choosing here.
    matcher = LightGlue(features="superpoint", width_confidence=-1).eval().to(args.device)

    n_pairs = 0
    with open(out_path, "w") as out_f:
        for chunk_start in range(0, n_frames, args.chunk_frames):
            chunk_end = min(chunk_start + args.chunk_frames, n_frames)
            specs = chunk_pair_specs(chunk_start, chunk_end, n_frames, gaps)

            # sort by keypoint count so batches group similarly-sized frames
            # together -- minimizes truncation waste vs arbitrary/sequential
            # batching (see module docstring).
            counted = [(s, store.count(s[0], s[1]), store.count(s[2], s[3])) for s in specs]
            counted.sort(key=lambda x: (x[1], x[2]))

            # Collect every batch's match results for the WHOLE chunk before
            # gating -- lets gate_chunk do just 2 ray-unprojection calls
            # (one per eye) for potentially thousands of pairs, instead of
            # 2-per-pair (see gate_chunk / unproject_batch docstrings).
            eligible = []
            for bi in range(0, len(counted), args.batch_size):
                batch = counted[bi:bi + args.batch_size]
                batch_specs = [(ea, fa, eb, fb) for (ea, fa, eb, fb, _, _), _, _ in batch]
                results = match_batch(matcher, store, batch_specs, args.match_conf_thresh, args.device)
                for ((ea, fa, eb, fb, pair_type, gap), _, _), (idx, scores) in zip(batch, results):
                    if idx.shape[0] == 0:
                        continue
                    if idx.shape[0] < args.min_raw_matches:
                        write_reject_too_few(ea, fa, eb, fb, pair_type, gap, idx, out_f)
                    else:
                        eligible.append((ea, fa, eb, fb, pair_type, gap, idx, scores))
                    n_pairs += 1

            gate_chunk(store, eligible, cams, args, out_f)

            print(f"chunk [{chunk_start},{chunk_end}) done, {n_pairs} pairs so far")

    store.close()
    print(f"wrote {n_pairs} pairs to {out_path}")


if __name__ == "__main__":
    main()
