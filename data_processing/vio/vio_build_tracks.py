"""VIO pipeline stage 3: chain stage-2 pairwise matches into multi-frame
tracks (a COLMAP-style landmark observation database) -- the persistent
track IDs LightGlue itself doesn't give you, since it only outputs pairwise
correspondences.

Node = one observation, (eye, frame, keypoint_idx). Edge = one accepted
match from stage 2 (rejected matches are never added as edges).

THREE defenses against a bad edge corrupting a track, all found necessary by
direct investigation (not anticipated up front):

1. CONFLICT-AWARE union-find, not naive connected components: plain
   transitive closure over noisy pairwise matches over-merges. A single
   spurious edge can bridge two otherwise-separate physical points into one
   blob; once that bridge exists, transitivity silently pulls in everything
   reachable through it. Symptom observed directly: a supposed single-point
   track with far more observations than frames*eyes, and >1 different
   keypoint claimed for the same (eye,frame) -- physically impossible for
   one 3D point seen once per camera per instant. Fix: reject any union that
   would create two different node keys sharing the same (eye,frame) in the
   resulting component.

2. MINIMUM POST-GATE INLIER COUNT on temporal pairs: a temporal pair whose
   RANSAC gate kept only a handful of inliers (e.g. 6-7) is not evidence of
   consistent geometry -- a 5-DOF essential-matrix model fit to noise or
   repetitive texture can "explain" that few points by chance. Found by
   direct investigation of a 2007-observation monster track spanning 44s of
   obviously-different physical points (floor -> sink -> cabinet): it was
   4-5 legitimate tracks welded together by ~3 long-gap (30-50 frame)
   bridge edges from pairs with n_geom of just 6-7, out of 21k healthy
   edges. Defense (1) can't catch these: the bridged segments live in
   DISJOINT frame ranges, so no (eye,frame) conflict ever arises. Filter
   the whole pair (all its edges) when n_geom < --min-gate-inliers. Stage 2
   applies the same filter at generation time (its min_raw_matches
   pre-filter only bounds RAW LightGlue matches, not surviving inliers);
   it's applied here too so pre-existing matches files can be re-filtered
   without a re-match, and as defense-in-depth.

3. DISPLACEMENT-RATE gate on temporal edges: (1) alone doesn't catch a
   different failure mode, also observed directly -- a single bad detection
   at one frame gets stitched into an otherwise-perfect track (e.g. a track
   drifting smoothly frame to frame that briefly "teleports" 600+ px away
   for exactly one frame, then continues its original smooth path). This
   isn't a same-frame conflict, so (1) can't catch it, and it isn't
   explained by too few RANSAC points either -- confirmed by inspecting the
   actual matches.jsonl records involved, which had 300+ raw matches, no
   different from any healthy pair. RANSAC only guarantees majority
   consensus, not zero false-positive inliers; a lone bad point can satisfy
   the winning model by chance even with abundant data. Fix: reject a
   temporal edge outright if its implied pixel-displacement RATE (distance /
   frame gap) exceeds what real ego-motion produces -- calibrated against
   this recording's own data: even during fast head motion, per-frame track
   displacement medians ~30-60px/frame, while the observed bad edges implied
   300-600+ px/frame. Stereo edges are exempt (same timestamp -- large
   pixel distance there is normal disparity/parallax, not implausible
   motion).

Edges are processed most-trustworthy-first (stereo, gated against KNOWN
calibration with no estimation uncertainty, then temporal by ascending gap
i.e. least motion first) so a bad edge gets rejected against an
already-good component instead of being the one that cements it.

Tracks shorter than --min-observations are dropped -- they don't
meaningfully constrain a bundle adjustment and just add solve cost.

3D triangulation of each track's landmark position is deliberately NOT done
here -- deferred to the bundle-adjustment stage, which needs the
triangulation logic anyway as its LM initialization step. This stage is
purely the 2D observation graph.

Run (from data_processing/vio/):
    python vio_build_tracks.py ../../long-test1 --matches ../../long-test1/matches.jsonl \
        --features ../../long-test1/features.h5 --out ../../long-test1/tracks.jsonl

    # optional loop closure: add loop_matches.jsonl as a second --matches file
    python vio_build_tracks.py ../../REC \
        --matches ../../REC/matches.jsonl ../../REC/loop_matches.jsonl
"""
import argparse
import json
import math
import os

import h5py


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("recording")
    p.add_argument("--matches", default=None, nargs="+",
                    help="one or more match files, concatenated (default: "
                         "<recording>/matches.jsonl). Add loop_matches.jsonl as a "
                         "2nd file for loop closure")
    p.add_argument("--features", default=None, help="default: <recording>/features.h5")
    p.add_argument("--out", default=None, help="default: <recording>/tracks.jsonl")
    p.add_argument("--min-observations", type=int, default=3)
    p.add_argument("--min-frames", type=int, default=15,
                    help="drop tracks seen in fewer than this many DISTINCT frames "
                         "(eye-independent), keeping only well-tracked stable "
                         "landmarks; 0 disables")
    p.add_argument("--max-px-per-frame", type=float, default=100.0,
                    help="reject a temporal edge if its implied displacement rate "
                         "(px distance / frame gap) exceeds this -- calibrated well "
                         "above real ego-motion (~30-60px/frame even during fast head "
                         "motion in this recording) and well below observed bad edges "
                         "(300-600+ px/frame)")
    p.add_argument("--min-gate-inliers", type=int, default=15,
                    help="drop a temporal pair's edges entirely if its RANSAC gate "
                         "kept fewer than this many inliers (n_geom) -- defense 2 in "
                         "the module docstring. Redundant with stage 2's identical "
                         "post-gate filter for newly-generated matches files, but "
                         "kept here as defense-in-depth and so pre-existing files "
                         "can be re-filtered without re-matching.")
    return p.parse_args()


class ConflictAwareUnionFind:
    """Union-find over integer node ids, where each node also carries an
    (eye, frame) identity key. A union is REJECTED (returns False, no
    mutation) if it would place two different nodes sharing the same
    (eye, frame) into one component -- see module docstring for why this
    matters more than the usual path-compression/union-by-size concerns."""

    def __init__(self):
        self.parent = []
        self.size = []
        self.members = []  # members[root] = {(eye,frame): node_id}, else None

    def add(self, eye, frame):
        node = len(self.parent)
        self.parent.append(node)
        self.size.append(1)
        self.members.append({(eye, frame): node})
        return node

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # path halving
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return True
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        for ef, node in self.members[rb].items():
            existing = self.members[ra].get(ef)
            if existing is not None and existing != node:
                return False  # conflict -- reject this merge entirely
        self.members[ra].update(self.members[rb])
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        self.members[rb] = None
        return True


def prune_spike_outliers(obs_for_eye, max_px_per_frame):
    """Remove an interior observation whose transitions to BOTH neighbors are
    implausible (see module docstring, defense 3) but whose neighbors are
    consistent with EACH OTHER once it's skipped -- the exact signature of a
    lone bad detection wedged into an otherwise-clean track via a multi-hop
    bridge (each hop individually plausible, so the per-edge check in main()
    can't catch it; this is the necessary complement, applied after the
    track's full trajectory is known). obs_for_eye: list of (frame, px),
    already sorted by frame, all the same eye. Mutates and returns it."""
    changed = True
    while changed and len(obs_for_eye) > 2:
        changed = False
        for i in range(1, len(obs_for_eye) - 1):
            f0, p0 = obs_for_eye[i - 1]
            f1, p1 = obs_for_eye[i]
            f2, p2 = obs_for_eye[i + 1]
            dist = lambda pa, pb: math.hypot(pb[0] - pa[0], pb[1] - pa[1])
            rate_in = dist(p0, p1) / max(f1 - f0, 1)
            rate_out = dist(p1, p2) / max(f2 - f1, 1)
            rate_skip = dist(p0, p2) / max(f2 - f0, 1)
            if (rate_in > max_px_per_frame and rate_out > max_px_per_frame
                    and rate_skip <= max_px_per_frame):
                del obs_for_eye[i]
                changed = True
                break
    return obs_for_eye


def edge_priority(rec):
    """Lower = processed first = trusted more. Stereo is gated against KNOWN
    calibration (no estimation uncertainty) so it's as trustworthy as the
    least-motion temporal pairs; among temporal pairs, smaller gap = less
    motion = more reliable."""
    return (0, 0) if rec["pair_type"] == "stereo" else (1, rec["gap"])


def main():
    args = parse_args()
    matches_paths = args.matches or [os.path.join(args.recording, "matches.jsonl")]
    features_path = args.features or os.path.join(args.recording, "features.h5")
    out_path = args.out or os.path.join(args.recording, "tracks.jsonl")

    recs = []
    for mp in matches_paths:
        with open(mp) as fp:
            recs.extend(json.loads(line) for line in fp)
    if len(matches_paths) > 1:
        print(f"concatenated {len(matches_paths)} match files "
              f"({', '.join(os.path.basename(p) for p in matches_paths)}) "
              f"-> {len(recs)} match records")
    recs.sort(key=edge_priority)

    node_id = {}  # (eye, frame, kp_idx) -> int
    uf = ConflictAwareUnionFind()

    def nid(eye, frame, kp_idx):
        key = (eye, frame, kp_idx)
        if key not in node_id:
            node_id[key] = uf.add(eye, frame)
        return node_id[key]

    with h5py.File(features_path, "r") as f:
        kp_cache = {}

        def px(eye, frame, kp_idx):
            key = (eye, frame)
            if key not in kp_cache:
                n_kp = int(f[eye]["counts"][frame])
                kp_cache[key] = f[eye]["keypoints"][frame].reshape(n_kp, 2)
            return kp_cache[key][kp_idx]

        n_edges = 0
        n_rejected_conflict = 0
        n_rejected_implausible = 0
        n_pairs_low_inliers = 0
        for r in recs:
            ea, fa, eb, fb = r["eye_a"], r["frame_a"], r["eye_b"], r["frame_b"]
            is_temporal = r["pair_type"] == "temporal"
            gap = r.get("gap") or 0
            if is_temporal and r["n_geom"] < args.min_gate_inliers:
                n_pairs_low_inliers += 1
                continue
            for ia, ib in zip(r["idx_a"], r["idx_b"]):
                n_edges += 1
                if is_temporal:
                    pa, pb = px(ea, fa, ia), px(eb, fb, ib)
                    dist = math.hypot(pb[0] - pa[0], pb[1] - pa[1])
                    if dist / max(gap, 1) > args.max_px_per_frame:
                        n_rejected_implausible += 1
                        continue
                a, b = nid(ea, fa, ia), nid(eb, fb, ib)
                if not uf.union(a, b):
                    n_rejected_conflict += 1

        print(f"{len(recs)} matched pairs ({n_pairs_low_inliers} temporal pairs dropped "
              f"for < {args.min_gate_inliers} gate inliers), {len(node_id)} distinct "
              f"observations, {n_edges} edges ({n_rejected_implausible} rejected as "
              f"implausible motion, {n_rejected_conflict} rejected as conflicting)")

        by_root = {}
        for key, idx in node_id.items():
            by_root.setdefault(uf.find(idx), []).append(key)
        print(f"{len(by_root)} tracks after conflict-aware merging")

        # Longest tracks first -- downstream consumers (viz, BA compute
        # budgeting) both want to prioritize the best-constrained tracks.
        ordered = sorted(by_root.values(), key=len, reverse=True)

        n_written = 0
        n_dropped_short = 0
        n_dropped_few_frames = 0
        n_pruned_obs = 0
        with open(out_path, "w") as out_f:
            for obs_keys in ordered:
                by_eye = {"left": [], "right": []}
                for eye, frame, kp_idx in obs_keys:
                    by_eye[eye].append((frame, px(eye, frame, kp_idx), kp_idx))

                kept = []
                for eye in ("left", "right"):
                    seq = sorted(by_eye[eye], key=lambda t: t[0])
                    pruned = prune_spike_outliers(
                        [(f, p) for f, p, _ in seq], args.max_px_per_frame)
                    pruned_frames = {f for f, _ in pruned}
                    n_pruned_obs += len(seq) - len(pruned)
                    for f, p, kp_idx in seq:
                        if f in pruned_frames:
                            kept.append({"eye": eye, "frame": f, "kp_idx": int(kp_idx),
                                         "px": p.tolist()})

                if len(kept) < args.min_observations:
                    n_dropped_short += 1
                    continue
                if len({o["frame"] for o in kept}) < args.min_frames:
                    n_dropped_few_frames += 1
                    continue
                kept.sort(key=lambda o: (o["frame"], o["eye"]))
                rec = {"track_id": n_written, "n_obs": len(kept), "observations": kept}
                out_f.write(json.dumps(rec) + "\n")
                n_written += 1

    print(f"wrote {n_written} tracks (dropped {n_dropped_short} with "
          f"< {args.min_observations} observations, {n_dropped_few_frames} with "
          f"< {args.min_frames} distinct frames, pruned {n_pruned_obs} spike-outlier "
          f"observations) to {out_path}")


if __name__ == "__main__":
    main()
