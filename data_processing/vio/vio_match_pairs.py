"""VIO pipeline stage 2: pairwise LightGlue matching + geometric gating.

Pairs matched, per frame i:
  - stereo:   (i, left) <-> (i, right), gated against the KNOWN stereo R,t
              (no RANSAC needed -- the geometry is already calibrated).
  - temporal: (i, eye) <-> (i+gap, eye), same eye, for a schedule of gaps
              covering ~2s (60 frames @ 30fps) with progressively sparser
              stride at longer range (dense nearby, sparse far away, since a
              wide gap only needs occasional coverage for long-baseline
              constraints -- see build_temporal_gaps). Gated via a GPU-batched
              jaxls relative-pose RANSAC (see relpose_5pt_jaxls.py) -- no
              known relative pose exists between two arbitrary frames, so
              this can't reuse the stereo gate's known-geometry shortcut.

Both gates operate on UNPROJECTED RAYS (fisheye_pinhole.fisheye_unproject),
not raw distorted pixels -- fisheye epipolar lines are curves in pixel space,
not the straight "same row" shortcut a rectified rig would give. A pixel
tolerance (--epipolar-px-thresh) is converted to an angular/normalized
tolerance via focal length for both gates (the temporal gate additionally
scales this by --ransac-threshold-scale, see relpose_5pt_jaxls.py's
docstring for why 0.75 is the locked-in default, not 1.0).

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

MATCHING PRECISION/COMPILE (defaults locked in via the bench_lightglue*.py
sweeps on this recording, 300 frames / 8152 pairs, batch_size=16):
  - mp=True (LightGlue's autocast flag): ~1.7x on the matching phase, 99.9%
    match agreement vs fp32. The single dominant lever -- attention was
    already fp16 via the flash path; this extends fp16 to the GNN's
    projections/MLPs. TF32 ("high" matmul precision) was also swept: 1.37x
    at 100% agreement standalone, but it does NOT stack with autocast (both
    accelerate the same GEMMs; once autocast has them in fp16, TF32 has
    nothing left), so mp subsumes it.
  - matcher.compile(mode="reduce-overhead") was evaluated and REJECTED (not
    even offered as a flag): its speed edge over eager+mp was unstable
    across repeated sweeps (+13% to -40%; eager numbers were rock-stable in
    the same sweeps), it pays a ~40s warmup that dominates short runs, and
    an isolated A/B in this script (200 frames, everything else identical)
    showed its pad-and-mask numerics shift ~5% of post-RANSAC gate
    decisions vs eager. Also requires patching a batch>1 mask-broadcast
    bug ([B,N,N] masks fed to SDPA, which needs [B,heads,N,N]-
    broadcastable -- fix is .unsqueeze(1) per mask in masked_forward,
    preserved in bench_lightglue_compile.py if ever revisited).
  - depth_confidence (early-stop) stays at its default: disabling it
    measured slightly SLOWER (batched early-exits do fire; the per-layer
    confidence-head overhead pays for itself).

TEMPORAL-PAIR RANSAC IS BATCHED SEPARATELY FROM LIGHTGLUE, AT A MUCH LARGER,
FULLY DECOUPLED BATCH SIZE (--ransac-batch-size, --ransac-m-pad). There's no
correctness reason for the two to match -- LightGlue's --batch-size is tuned
to minimize its OWN truncation waste (16 already captures ~all the speedup,
see above), while jaxls's RANSAC solve gets STRICTLY faster per pair as batch
size grows (confirmed by direct sweep on this recording: 64->2048 pairs/batch
took cv2-relative speedup from ~3.5x to ~27x, plateauing past ~1024) and pays
its one-time jit-compile cost ONCE per process regardless of how large the
batch is, so there's no reason not to go as large as the eligible temporal-
pair pool allows. Concretely: LightGlue matching still proceeds chunk-by-chunk
as before (unchanged), and stereo pairs are still gated and written
immediately per chunk (cheap, no RANSAC), but every eligible TEMPORAL pair's
rays are instead accumulated across the ENTIRE video into one buffer; only
after the last chunk's matching finishes does the RANSAC gate run, in as few
--ransac-batch-size-sized (fixed-shape, padded) batches as needed to cover
the whole accumulated set -- see gate_temporal_all().

Run (from data_processing/vio/):
    python vio_match_pairs.py ../../long-test1 --features ../../long-test1/features.h5 \
        --out ../../long-test1/matches.jsonl --max-frames 30
"""
import argparse
import json
import os

import h5py
import numpy as np
import torch

import jax
import jax.numpy as jnp

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fisheye_pinhole as FP

from lightglue import LightGlue
import relpose_5pt_jaxls as R5


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
    p.add_argument("--min-gate-inliers", type=int, default=15,
                    help="reject a TEMPORAL pair entirely (all matches moved to "
                         "rejected) if the RANSAC gate's winning model kept fewer than "
                         "this many inliers -- a 5-DOF model that only 'explains' 6-7 "
                         "points is a fit to noise/repetitive texture, not evidence of "
                         "consistent geometry. Found via a real 44s monster track in "
                         "stage 3 welded together by exactly such pairs (n_geom 6-7, "
                         "gaps 30-50) during fast head motion; see "
                         "vio_build_tracks.py's docstring, defense 2. min_raw_matches "
                         "can't catch this: it bounds RAW LightGlue matches, and these "
                         "pairs had plenty -- it's the SURVIVING inlier count that "
                         "exposes the degenerate fit.")
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
    p.add_argument("--ransac-batch-size", type=int, default=2048,
                    help="pairs per jaxls RANSAC batch call, COMPLETELY DECOUPLED "
                         "from --batch-size (LightGlue) -- see module docstring. As "
                         "large as possible: speed strictly improves with batch size "
                         "(confirmed 64->2048 sweep, ~3.5x->~27x vs cv2, plateauing "
                         "past ~1024) and the one-time jit-compile cost is paid once "
                         "regardless of batch size, so there's no downside to going "
                         "large other than memory.")
    p.add_argument("--ransac-n-hyp", type=int, default=100,
                    help="RANSAC hypotheses per pair -- nearly free, see "
                         "relpose_5pt_jaxls.py's docstring sweep")
    p.add_argument("--ransac-max-iters", type=int, default=10,
                    help="LM iterations per hypothesis -- the dominant accuracy "
                         "lever, NOT n_hyp, see relpose_5pt_jaxls.py's docstring sweep")
    p.add_argument("--ransac-m-pad", type=int, default=512,
                    help="fixed point-count every temporal pair is padded/truncated "
                         "to for the RANSAC batch -- keeps jaxls's jit cache hit for "
                         "the WHOLE run (one compile total), see relpose_5pt_jaxls.py")
    p.add_argument("--ransac-threshold-scale", type=float, default=0.75,
                    help="multiplies the epipolar-px-derived threshold for the "
                         "temporal RANSAC gate ONLY (not the stereo gate) -- 0.75 is "
                         "the locked-in false-positive/false-negative balance point, "
                         "see relpose_5pt_jaxls.py's docstring threshold sweep")
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

    def prefetch_range(self, eye, start, end):
        """Bulk-read keypoints/scores/descriptors for frames [start,end) in
        ONE h5py slice call each (3 total), instead of up to (end-start)
        individual per-frame `get()` reads. Confirmed by direct testing this
        matters a lot: a full ~7000-frame run with per-frame reads was still
        stuck partway through its FIRST 300-frame chunk after 2+ hours
        (verified via py-spy: genuinely alternating between real LightGlue
        compute and HDF5 `__getitem__` calls, not hung) -- the earlier fast
        200-frame test (40s) almost certainly benefited from OS page-cache
        locality built up from repeated testing on that same byte range, not
        from anything about the code itself; a full-video run touches far
        more of the 3.3GB file and hits mostly cold reads. Populates the same
        `_cache` dict `get()` reads from, so `get()` itself is unchanged --
        this is purely a batching optimization, same idea as `count()`'s
        fix, just for the (much larger) keypoint/score/descriptor arrays
        instead of the tiny per-frame count."""
        end = min(end, self.n_frames(eye))
        if end <= start:
            return
        g = self.f[eye]
        kp_all = g["keypoints"][start:end]
        sc_all = g["scores"][start:end]
        de_all = g["descriptors"][start:end]
        for offset, i in enumerate(range(start, end)):
            key = (eye, i)
            if key in self._cache:
                continue
            n = self.count(eye, i)
            kp = kp_all[offset].reshape(n, 2).astype(np.float32)
            sc = sc_all[offset].astype(np.float32)
            de = de_all[offset].reshape(n, -1).astype(np.float32)
            self._cache[key] = (kp, sc, de)

    def count(self, eye, i):
        # Cache the WHOLE counts array per eye on first use -- a naive
        # per-call `self.f[eye]["counts"][i]` HDF5 slice read is fine for a
        # handful of calls, but the chunk-building step calls this twice per
        # candidate pair spec (thousands of specs for a multi-hundred-frame
        # chunk), and that many individual small HDF5 reads was confirmed by
        # direct testing to take MULTIPLE MINUTES (a 200-frame run stalled
        # for 20+ min before the very next step printed anything) -- purely
        # from this, not from any actual matching/RANSAC work. The whole
        # array is tiny (one int per frame), so caching it is strictly a win.
        key = f"_counts_{eye}"
        if key not in self._cache:
            self._cache[key] = self.f[eye]["counts"][:]
        return int(self._cache[key][i])

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


def _unproject_pairs(store, pairs, cams, device):
    """Shared helper: pairs is a list of (eye_a, i, eye_b, j, pair_type, gap,
    idx, scores). Does ONE ray-unprojection call per eye for the WHOLE list
    (not one per pair per side -- see unproject_batch), regardless of
    pair_type. Returns rays_by_pair_side keyed by (list-local index, "a"/"b")."""
    buf_pts = {"left": [], "right": []}
    buf_meta = {"left": [], "right": []}  # (pair_idx, side, n) in append order
    for pair_idx, (ea, i, eb, j, pair_type, gap, idx, scores) in enumerate(pairs):
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
        all_rays = unproject_batch(all_pts, K, D, device)
        offset = 0
        for pair_idx, side, n in buf_meta[eye]:
            rays_by_pair_side[(pair_idx, side)] = all_rays[offset:offset + n]
            offset += n
    return rays_by_pair_side


def gate_stereo_chunk(store, eligible_stereo, cams, args, out_f):
    """eligible_stereo: list of (eye_a, i, eye_b, j, "stereo", gap, idx,
    scores) for ONE chunk. Cheap known-geometry gate, no RANSAC -- gated and
    written immediately per chunk, unlike temporal pairs (see
    collect_temporal_pairs / gate_temporal_all)."""
    rays_by_pair_side = _unproject_pairs(store, eligible_stereo, cams, args.device)
    R, t = cams["stereo"]
    for pair_idx, (eye_a, i, eye_b, j, pair_type, gap, idx, scores) in enumerate(eligible_stereo):
        rays_a = rays_by_pair_side[(pair_idx, "a")]
        rays_b = rays_by_pair_side[(pair_idx, "b")]
        Ka, Da = cams[eye_a]
        Kb, Db = cams[eye_b]
        f_avg = (Ka[0, 0] + Ka[1, 1] + Kb[0, 0] + Kb[1, 1]) / 4.0
        theta_tol = args.epipolar_px_thresh / f_avg
        inlier = gate_stereo(rays_a, rays_b, R, t, theta_tol)
        rec = {
            "eye_a": eye_a, "frame_a": i, "eye_b": eye_b, "frame_b": j,
            "pair_type": pair_type, "gap": gap,
            "n_raw": idx.shape[0], "n_geom": int(inlier.sum()),
            "idx_a": idx[inlier, 0].tolist(), "idx_b": idx[inlier, 1].tolist(),
            "rejected_idx_a": idx[~inlier, 0].tolist(), "rejected_idx_b": idx[~inlier, 1].tolist(),
        }
        out_f.write(json.dumps(rec) + "\n")


def collect_temporal_pairs(store, eligible_temporal, cams, args, accum):
    """eligible_temporal: list of (eye_a, i, eye_b, j, "temporal", gap, idx,
    scores) for ONE chunk. Does the same per-chunk ray-unprojection
    consolidation as gate_stereo_chunk (cheap, no reason to defer), but
    APPENDS (eye_a, i, eye_b, j, gap, idx, pts_a2d, pts_b2d) to the GLOBAL
    `accum` list instead of gating immediately -- RANSAC gating is deferred
    until the whole video's temporal pairs are collected, so it can batch as
    aggressively as possible (see gate_temporal_all / module docstring)."""
    rays_by_pair_side = _unproject_pairs(store, eligible_temporal, cams, args.device)
    for pair_idx, (eye_a, i, eye_b, j, pair_type, gap, idx, scores) in enumerate(eligible_temporal):
        rays_a = rays_by_pair_side[(pair_idx, "a")]
        rays_b = rays_by_pair_side[(pair_idx, "b")]
        # (x/z, y/z) normalized bearings -- exactly what relpose_ransac wants,
        # and 1/3 less memory than keeping the full 3D ray across the whole
        # video's worth of accumulated pairs.
        pts_a2d = (rays_a[:, :2] / rays_a[:, 2:3]).astype(np.float32)
        pts_b2d = (rays_b[:, :2] / rays_b[:, 2:3]).astype(np.float32)
        accum.append((eye_a, i, eye_b, j, gap, idx, pts_a2d, pts_b2d))


def gate_temporal_all(accum, theta_tol_ours, args, out_f):
    """accum: list of (eye_a, i, eye_b, j, gap, idx, pts_a2d, pts_b2d) for
    EVERY temporal pair in the WHOLE VIDEO (gathered by collect_temporal_pairs
    across all chunks). Runs the jaxls RANSAC gate in --ransac-batch-size-
    sized fixed-shape batches -- completely decoupled from the LightGlue
    --batch-size/--chunk-frames (see module docstring for why: no correctness
    reason they need to match, and larger RANSAC batches are strictly faster
    per pair). Pads every batch (point count to --ransac-m-pad, pair count to
    --ransac-batch-size for the last partial batch) with a validity mask so
    every call to relpose_ransac uses the IDENTICAL shape -- jaxls/jax jit
    compiles once for the whole run, not once per batch (see
    relpose_5pt_jaxls.py's docstring "CRITICAL gotcha")."""
    if not accum:
        return
    B = args.ransac_batch_size
    Mp = args.ransac_m_pad
    n_hyp = args.ransac_n_hyp
    max_iters = args.ransac_max_iters

    # sort by point count first -- minimizes truncation waste for the (rare)
    # pairs with more than Mp points, same rationale as LightGlue's batching.
    order = sorted(range(len(accum)), key=lambda k: accum[k][6].shape[0])

    n_full_batches = -(-len(order) // B)  # ceil division
    key = jax.random.PRNGKey(0)
    n_written = 0
    for bi in range(0, len(order), B):
        batch_order = order[bi:bi + B]
        pa_list, pb_list, mask_list = [], [], []
        for k in batch_order:
            _, _, _, _, _, _, pts_a2d, pts_b2d = accum[k]
            valid_n = min(pts_a2d.shape[0], Mp)
            pa = pts_a2d[:valid_n]
            pb = pts_b2d[:valid_n]
            if valid_n < Mp:
                pad = np.zeros((Mp - valid_n, 2), dtype=np.float32)
                pa = np.concatenate([pa, pad], axis=0)
                pb = np.concatenate([pb, pad], axis=0)
            mask = np.zeros(Mp, dtype=bool)
            mask[:valid_n] = True
            pa_list.append(pa); pb_list.append(pb); mask_list.append(mask)
        # pad the LAST partial batch up to B with dummy all-invalid entries
        # so every call keeps the exact same (B, Mp) shape.
        n_real = len(batch_order)
        for _ in range(B - n_real):
            pa_list.append(np.zeros((Mp, 2), dtype=np.float32))
            pb_list.append(np.zeros((Mp, 2), dtype=np.float32))
            mask_list.append(np.zeros(Mp, dtype=bool))

        pts_a_t = jnp.asarray(np.stack(pa_list))
        pts_b_t = jnp.asarray(np.stack(pb_list))
        mask_t = jnp.asarray(np.stack(mask_list))
        key, sub = jax.random.split(key)
        inlier, _ = R5.relpose_ransac(pts_a_t, pts_b_t, sub, n_hyp, theta_tol_ours, max_iters,
                                       valid_mask=mask_t)
        inlier_np = np.asarray(inlier)

        for row, k in enumerate(batch_order):
            eye_a, i, eye_b, j, gap, idx, pts_a2d, pts_b2d = accum[k]
            valid_n = min(pts_a2d.shape[0], Mp)
            g = inlier_np[row][:valid_n]
            # idx has idx.shape[0] rows; valid_n == min(idx.shape[0], Mp) --
            # if idx.shape[0] > Mp (truncated), the tail beyond Mp was never
            # scored, so treat it as rejected (consistent with "trust fewer,
            # scored points over an untested tail").
            full_inlier = np.zeros(idx.shape[0], dtype=bool)
            full_inlier[:valid_n] = g
            # too few surviving inliers = degenerate fit, reject the whole
            # pair (see --min-gate-inliers help text; the record is still
            # written, with everything in rejected_*, so the visualizer can
            # show what was filtered -- same convention as min_raw_matches).
            if int(full_inlier.sum()) < args.min_gate_inliers:
                full_inlier[:] = False
            rec = {
                "eye_a": eye_a, "frame_a": i, "eye_b": eye_b, "frame_b": j,
                "pair_type": "temporal", "gap": gap,
                "n_raw": idx.shape[0], "n_geom": int(full_inlier.sum()),
                "idx_a": idx[full_inlier, 0].tolist(), "idx_b": idx[full_inlier, 1].tolist(),
                "rejected_idx_a": idx[~full_inlier, 0].tolist(),
                "rejected_idx_b": idx[~full_inlier, 1].tolist(),
            }
            out_f.write(json.dumps(rec) + "\n")
            n_written += 1
        print(f"  temporal RANSAC batch {bi // B + 1}/{n_full_batches} done "
              f"({n_written}/{len(accum)} pairs written)", flush=True)


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
    # mp=True: see "MATCHING PRECISION/COMPILE" in the module docstring for
    # the sweep numbers behind this default (~1.7x, 99.9% agreement).
    matcher = LightGlue(features="superpoint", width_confidence=-1,
                        mp=True).eval().to(args.device)

    # ONE global threshold for ALL temporal pairs regardless of eye -- matches
    # the exact methodology relpose_5pt_jaxls.py's own sweeps were validated
    # with (a single f_avg combining both eyes' K), and the resulting per-eye
    # error is negligible (the two cameras have near-identical focal lengths)
    # -- see relpose_5pt_jaxls.py's docstring for the threshold-scale sweep
    # this 0.75 default comes from.
    Kl_, Dl_ = cams["left"]
    Kr_, Dr_ = cams["right"]
    f_avg_global = (Kl_[0, 0] + Kl_[1, 1] + Kr_[0, 0] + Kr_[1, 1]) / 4.0
    theta_tol_ours = (args.epipolar_px_thresh / f_avg_global) * args.ransac_threshold_scale
    print(f"temporal RANSAC threshold: {theta_tol_ours:.6f} "
          f"(epipolar-px-thresh {args.epipolar_px_thresh} / f_avg {f_avg_global:.1f} "
          f"x scale {args.ransac_threshold_scale})")

    n_pairs = 0
    temporal_accum = []  # (eye_a, i, eye_b, j, gap, idx, pts_a2d, pts_b2d) -- WHOLE VIDEO
    with open(out_path, "w") as out_f:
        for chunk_start in range(0, n_frames, args.chunk_frames):
            chunk_end = min(chunk_start + args.chunk_frames, n_frames)
            specs = chunk_pair_specs(chunk_start, chunk_end, n_frames, gaps)

            # Bulk-prefetch every frame this chunk's specs can touch BEFORE
            # the per-pair store.get() calls in match_batch -- temporal pairs
            # look up to max(gaps) frames past chunk_end, so the prefetch
            # window extends that far too. See prefetch_range's docstring for
            # why this matters (a LOT, for a full-video run).
            prefetch_end = min(chunk_end + max(gaps), n_frames)
            store.prefetch_range("left", chunk_start, prefetch_end)
            store.prefetch_range("right", chunk_start, prefetch_end)

            # sort by keypoint count so batches group similarly-sized frames
            # together -- minimizes truncation waste vs arbitrary/sequential
            # batching (see module docstring).
            counted = [(s, store.count(s[0], s[1]), store.count(s[2], s[3])) for s in specs]
            counted.sort(key=lambda x: (x[1], x[2]))

            # Collect every batch's match results for the WHOLE chunk before
            # gating -- lets gate_stereo_chunk/collect_temporal_pairs do just
            # 2 ray-unprojection calls (one per eye) for potentially thousands
            # of pairs, instead of 2-per-pair (see unproject_batch docstring).
            eligible_stereo, eligible_temporal = [], []
            for bi in range(0, len(counted), args.batch_size):
                batch = counted[bi:bi + args.batch_size]
                batch_specs = [(ea, fa, eb, fb) for (ea, fa, eb, fb, _, _), _, _ in batch]
                results = match_batch(matcher, store, batch_specs, args.match_conf_thresh, args.device)
                for ((ea, fa, eb, fb, pair_type, gap), _, _), (idx, scores) in zip(batch, results):
                    if idx.shape[0] == 0:
                        continue
                    if idx.shape[0] < args.min_raw_matches:
                        write_reject_too_few(ea, fa, eb, fb, pair_type, gap, idx, out_f)
                    elif pair_type == "stereo":
                        eligible_stereo.append((ea, fa, eb, fb, pair_type, gap, idx, scores))
                    else:
                        eligible_temporal.append((ea, fa, eb, fb, pair_type, gap, idx, scores))
                    n_pairs += 1

            # stereo: cheap known-geometry gate, no reason to defer -- gated
            # and written immediately, same as before.
            gate_stereo_chunk(store, eligible_stereo, cams, args, out_f)
            # temporal: unproject now (cheap, per-chunk consolidation is
            # already efficient) but defer RANSAC gating -- accumulated
            # across ALL chunks so it can batch as aggressively as possible,
            # decoupled from --chunk-frames/--batch-size (see module
            # docstring and gate_temporal_all).
            collect_temporal_pairs(store, eligible_temporal, cams, args, temporal_accum)

            print(f"chunk [{chunk_start},{chunk_end}) done, {n_pairs} pairs so far "
                  f"({len(temporal_accum)} temporal pairs accumulated for RANSAC)")

        print(f"\nrunning batched temporal RANSAC over all {len(temporal_accum)} "
              f"accumulated pairs (batch size {args.ransac_batch_size})...")
        gate_temporal_all(temporal_accum, theta_tol_ours, args, out_f)

    store.close()
    print(f"wrote {n_pairs} pairs to {out_path}")


if __name__ == "__main__":
    main()
