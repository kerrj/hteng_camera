"""VIO loop-closure matching pass: match KEYFRAME pairs that are far apart in
time (beyond stage 2's sequential gap-schedule horizon) so revisits of a place
get tied together, instead of drifting into two ghost copies.

WHY THIS EXISTS (found by direct investigation): stage 2's temporal gap
schedule caps at ~60 frames. When the camera leaves a scene and comes back
hundreds of frames later, NO pair -- and no chain of pairs -- connects the two
visits, so stage 3 builds two disjoint track sets for the same physical points
and stage 5 triangulates them in two drifted world locations (the classic
"ghost cluster" symptom). Confirmed on `testimu`: 0 tracks bridged the frame-0
desk and the frame-685 desk revisit despite both being densely tracked
locally.

This is exactly COLMAP's "sequential matcher + loop detection" split: the
sequential matcher (our gap schedule) handles odometry; a separate loop pass
adds the long-range edges. GLOMAP-style global BA (our stage 5) then uses those
edges automatically -- a global solver treats a loop edge like any other view-
graph edge, no special drift-correction step needed. We skip the vocab-tree
retrieval COLMAP uses only to avoid O(n^2) at large scale: at ~1 keyframe/sec
this clip is ~30 keyframes, so exhaustive all-keyframe-pairs (COLMAP's
"exhaustive matcher") is cheap and strictly more complete than retrieval.

Matching + gating are REUSED verbatim from vio_match_pairs.py (LightGlue batch
matching, then the jaxls RANSAC essential-matrix gate) so loop edges are gated
identically to normal temporal edges -- same --min-gate-inliers degenerate-fit
guard, which matters MORE across long baselines where repetitive texture
(monitor text, keyboard) is a bigger false-positive risk. Output records use
pair_type="temporal" with the true (large) gap, so they concatenate straight
into matches.jsonl and flow through stage 3 unchanged.

Writes <recording>/loop_matches.jsonl (same schema as matches.jsonl) and,
with --viz-out, a montage PNG of the accepted long-range pairs for eyeballing
whether the keyframe matches are real BEFORE they touch tracks/BA.

Run (from data_processing/vio/):
    CUDA_VISIBLE_DEVICES=2 python vio_loop_closure.py ../../testimu \
        --viz-out ../../testimu/loop_matches_viz.png
"""
import argparse
import json
import os

import cv2
import numpy as np

import jax

import vio_match_pairs as MP


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("recording")
    p.add_argument("--features", default=None, help="default: <recording>/features.h5")
    p.add_argument("--out", default=None, help="default: <recording>/loop_matches.jsonl")
    p.add_argument("--keyframe-stride", type=int, default=30,
                    help="use every Nth frame as a loop-closure keyframe (30 = "
                         "~1/sec @ 30fps); matching is cheap so err denser if a "
                         "known loop isn't caught")
    p.add_argument("--min-gap", type=int, default=60,
                    help="only match keyframe pairs at least this far apart -- below "
                         "this the sequential stage-2 schedule already covers them")
    p.add_argument("--eye", choices=("left", "right", "both"), default="left",
                    help="which eye's keyframes to loop-match; 'left' is enough since "
                         "stereo edges link both eyes within each cluster")
    p.add_argument("--max-frames", type=int, default=None)
    # matching / gating knobs -- defaults mirror vio_match_pairs.py exactly
    p.add_argument("--match-conf-thresh", type=float, default=0.2)
    p.add_argument("--min-raw-matches", type=int, default=10)
    p.add_argument("--min-gate-inliers", type=int, default=15)
    p.add_argument("--epipolar-px-thresh", type=float, default=3.0)
    p.add_argument("--ransac-threshold-scale", type=float, default=0.75)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--ransac-batch-size", type=int, default=2048)
    p.add_argument("--ransac-n-hyp", type=int, default=100)
    p.add_argument("--ransac-max-iters", type=int, default=10)
    p.add_argument("--ransac-m-pad", type=int, default=512)
    p.add_argument("--viz-out", default=None,
                    help="PNG montage of accepted loop pairs (best first) for visual "
                         "inspection; default: skip viz")
    p.add_argument("--viz-max-pairs", type=int, default=24)
    p.add_argument("--viz-tile-width", type=int, default=900,
                    help="downscaled width of each side-by-side pair tile")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def build_keyframe_specs(keyframes, min_gap, eye):
    """All (eye_a, i, eye_b, j) keyframe pairs with j-i > min_gap. Same eye on
    both sides (a loop is a revisit of the SAME camera's view); j>i so each
    pair appears once."""
    eyes = ("left", "right") if eye == "both" else (eye,)
    specs = []
    for e in eyes:
        for a in range(len(keyframes)):
            for b in range(a + 1, len(keyframes)):
                i, j = keyframes[a], keyframes[b]
                if j - i > min_gap:
                    specs.append((e, i, e, j))
    return specs


def render_montage(recording, accepted, store, args):
    """accepted: list of (eye_a, fa, eye_b, fb, gap, idx_a, idx_b, n_raw)
    sorted best-first. Reads only the needed frames (sparse keyframes) by
    scanning each video once, then tiles downscaled side-by-side pair images
    with green match lines into one PNG."""
    accepted = accepted[:args.viz_max_pairs]
    if not accepted:
        print("no accepted loop pairs to visualize")
        return
    needed = {"left": set(), "right": set()}
    for ea, fa, eb, fb, *_ in accepted:
        needed[ea].add(fa)
        needed[eb].add(fb)

    frames = {"left": {}, "right": {}}
    for eye in ("left", "right"):
        if not needed[eye]:
            continue
        cap = cv2.VideoCapture(os.path.join(recording, f"{eye}.mp4"))
        want = needed[eye]
        last = max(want)
        idx = 0
        while idx <= last:
            ok, fr = cap.read()
            if not ok:
                break
            if idx in want:
                frames[eye][idx] = fr
            idx += 1
        cap.release()

    tiles = []
    tw = args.viz_tile_width
    for ea, fa, eb, fb, gap, idx_a, idx_b, n_raw in accepted:
        fr_a = frames[ea].get(fa)
        fr_b = frames[eb].get(fb)
        if fr_a is None or fr_b is None:
            continue
        kp_a = store.get(ea, fa)[0]
        kp_b = store.get(eb, fb)[0]
        h, w = fr_a.shape[:2]
        canvas = np.hstack([fr_a.copy(), fr_b.copy()])
        for ka, kb in zip(idx_a, idx_b):
            pa = tuple(np.round(kp_a[ka]).astype(int))
            pb = tuple(np.round(kp_b[kb]).astype(int) + np.array([w, 0]))
            cv2.line(canvas, pa, pb, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.circle(canvas, pa, 3, (0, 200, 255), -1, cv2.LINE_AA)
            cv2.circle(canvas, pb, 3, (0, 200, 255), -1, cv2.LINE_AA)
        label = f"{ea} {fa} -> {fb}  gap {gap}  inliers {len(idx_a)}/{n_raw}"
        cv2.putText(canvas, label, (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.6,
                    (0, 255, 255), 4, cv2.LINE_AA)
        scale = tw / canvas.shape[1]
        tile = cv2.resize(canvas, (tw, int(canvas.shape[0] * scale)))
        tiles.append(tile)

    if not tiles:
        print("no renderable tiles")
        return
    ncols = 2
    th = tiles[0].shape[0]
    rows = []
    for r in range(0, len(tiles), ncols):
        row_tiles = tiles[r:r + ncols]
        while len(row_tiles) < ncols:
            row_tiles.append(np.zeros_like(tiles[0]))
        rows.append(np.hstack(row_tiles))
    montage = np.vstack(rows)
    cv2.imwrite(args.viz_out, montage)
    print(f"wrote {args.viz_out} ({len(tiles)} pairs, best-first)")


def main():
    args = parse_args()
    features_path = args.features or os.path.join(args.recording, "features.h5")
    out_path = args.out or os.path.join(args.recording, "loop_matches.jsonl")

    store = MP.FeatureStore(features_path, args.device)
    ls = store.f.attrs["left_serial"]
    rs = store.f.attrs["right_serial"]
    Kl, Dl = MP.load_intrinsics(args.recording, ls)
    Kr, Dr = MP.load_intrinsics(args.recording, rs)
    R, t = MP.load_stereo(args.recording, ls, rs)
    cams = {"left": (Kl, Dl), "right": (Kr, Dr), "stereo": (R, t)}

    n_frames = min(store.n_frames("left"), store.n_frames("right"))
    if args.max_frames:
        n_frames = min(n_frames, args.max_frames)
    keyframes = list(range(0, n_frames, args.keyframe_stride))
    specs = build_keyframe_specs(keyframes, args.min_gap, args.eye)
    print(f"{len(keyframes)} keyframes (stride {args.keyframe_stride}), "
          f"{len(specs)} long-range pairs (gap > {args.min_gap})")

    f_avg = (Kl[0, 0] + Kl[1, 1] + Kr[0, 0] + Kr[1, 1]) / 4.0
    theta_tol = (args.epipolar_px_thresh / f_avg) * args.ransac_threshold_scale

    matcher = MP.LightGlue(features="superpoint", width_confidence=-1,
                           mp=True).eval().to(args.device)

    # sort by keypoint count so LightGlue batches group similarly-sized frames
    # (minimizes match_batch's per-batch min-count truncation) -- same rationale
    # as vio_match_pairs.py's chunk sort.
    counted = [(s, store.count(s[0], s[1]), store.count(s[2], s[3])) for s in specs]
    counted.sort(key=lambda x: (x[1], x[2]))

    eligible = []  # (ea, fa, eb, fb, "temporal", gap, idx, scores)
    for bi in range(0, len(counted), args.batch_size):
        batch = counted[bi:bi + args.batch_size]
        batch_specs = [s for s, _, _ in batch]
        results = MP.match_batch(matcher, store, batch_specs,
                                 args.match_conf_thresh, args.device)
        for (ea, fa, eb, fb), (idx, scores) in zip(batch_specs, results):
            if idx.shape[0] < args.min_raw_matches:
                continue
            eligible.append((ea, fa, eb, fb, "temporal", fb - fa, idx, scores))
        print(f"  matched {min(bi + args.batch_size, len(counted))}/{len(counted)} "
              f"pairs ({len(eligible)} with >= {args.min_raw_matches} raw)", flush=True)

    # Reuse stage 2's ray-consolidation + batched RANSAC gate verbatim -- writes
    # records (pair_type="temporal", true gap) straight into loop_matches.jsonl.
    accum = []
    MP.collect_temporal_pairs(store, eligible, cams, args, accum)
    with open(out_path, "w") as out_f:
        MP.gate_temporal_all(accum, theta_tol, args, out_f)
    print(f"wrote {out_path}")

    # Re-read what the gate accepted (n_geom >= min_gate_inliers) for viz +
    # summary, best-first.
    accepted = []
    with open(out_path) as fp:
        for line in fp:
            r = json.loads(line)
            if r["n_geom"] <= 0:
                continue
            accepted.append((r["eye_a"], r["frame_a"], r["eye_b"], r["frame_b"],
                             r["gap"], r["idx_a"], r["idx_b"], r["n_raw"]))
    accepted.sort(key=lambda a: len(a[5]), reverse=True)
    n_attempted = len(specs)
    print(f"\n{len(accepted)}/{n_attempted} loop pairs ACCEPTED "
          f"(>= {args.min_gate_inliers} gate inliers)")
    print("top accepted loop pairs (frame_a -> frame_b, gap, inliers/raw):")
    for ea, fa, eb, fb, gap, ia, ib, n_raw in accepted[:20]:
        print(f"  {ea:5s} {fa:5d} -> {fb:5d}  gap {gap:4d}  {len(ia):3d}/{n_raw}")

    if args.viz_out:
        render_montage(args.recording, accepted, store, args)
    store.close()


if __name__ == "__main__":
    main()
