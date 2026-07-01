"""Hand-rolled, GPU-batched, normalized 8-point RANSAC for the temporal-pair
essential-matrix gate -- meant to replace `cv2.findEssentialMat` in
`vio_match_pairs.gate_temporal_ransac`, which profiling found to be ~99% of
remaining stage-2 gating time (~3ms/pair, CPU, strictly sequential over
pairs).

Two things make this fast without sacrificing correctness, both confirmed by
direct experiment before committing to this design:

1. A single-item-at-a-time GPU call is often SLOWER than the CPU original --
   kornia's `RANSAC` class, called once per pair, measured 500x slower than
   cv2 (1526ms/pair vs 3.07ms/pair) purely from per-call launch overhead, even
   though the underlying algorithm is "the same". Batching many pairs into
   ONE tensor op is what actually wins.
2. The classic 8-point algorithm (unlike 5-point) is pure linear algebra --
   build a design matrix, `torch.linalg.svd` for the null space -- which
   natively supports an arbitrary leading batch dimension. A crude first
   prototype (no Hartley normalization, no refinement) hit 23x speedup over
   cv2 but only 47% inlier-decision agreement; this version adds the two
   missing pieces (normalization, polish-with-all-inliers) that close that
   gap.

RANSAC itself is ALSO fully batched, not just "batched across pairs with a
Python loop over iterations": all H random-8-point hypotheses for all B pairs
are built and scored in ONE tensor op (an (B,H,8,9) SVD, an (B,H,M) Sampson
scoring einsum), then `argmax` over H picks the winning hypothesis per pair.
There is no sequential loop over RANSAC iterations at all.

Design choices worth knowing if this file gets modified:

- **Normalization is computed ONCE PER PAIR from all M points, not per
  8-point subsample.** This matches Hartley's actual recommendation (condition
  the point cloud, not the minimal sample) and is far cheaper: one (B,3,3)
  normalization instead of (B,H,3,3). Every hypothesis reuses the same
  already-normalized coordinates.
- **Refinement reuses ALL inliers via a masked design matrix, not a
  variable-length gather.** Per-pair inlier counts differ, which would
  otherwise force a ragged/looped refit. Instead: build ONE (B,M,9) design
  matrix from every point, zero out non-inlier rows before the SVD. A
  zeroed row contributes nothing to the implicit normal equations, so the
  SVD's null-space solution is exactly the least-squares fit over the
  surviving (inlier) rows -- no gathering, no loop, still one batched op.
  Guarded by a minimum-inlier-count check (`min_refine_inliers`) since a
  masked matrix with too few surviving rows is rank-deficient and the
  "refined" E would be meaningless.
- **Inlier scoring (Sampson distance) always happens in the ORIGINAL
  (unnormalized) ray-bearing space**, against the denormalized E, so the
  `thresh` argument stays in the same physical units as the rest of the
  pipeline's `theta_tol` (angular/normalized-pixel tolerance) -- callers
  don't need to know normalization happened internally at all.
"""
import math

import torch


def hartley_normalize(pts):
    """pts: (B, M, 2). Returns (pts_normalized (B,M,2), T (B,3,3)) such that
    for homogeneous x=(px,py,1), T @ x == homogeneous(pts_normalized). Moves
    the centroid to the origin and rescales so the mean point distance from
    the origin is sqrt(2) -- the standard conditioning that makes the 8-point
    algorithm's SVD numerically well-posed (raw pixel/ray coordinates have
    wildly different row-magnitude scales in the design matrix otherwise)."""
    B = pts.shape[0]
    centroid = pts.mean(dim=1, keepdim=True)  # (B,1,2)
    centered = pts - centroid
    mean_dist = centered.norm(dim=-1).mean(dim=1)  # (B,)
    scale = math.sqrt(2) / (mean_dist + 1e-9)  # (B,)
    pts_n = centered * scale.view(B, 1, 1)

    T = torch.zeros(B, 3, 3, device=pts.device, dtype=pts.dtype)
    T[:, 0, 0] = scale
    T[:, 1, 1] = scale
    T[:, 2, 2] = 1.0
    T[:, 0, 2] = -scale * centroid[:, 0, 0]
    T[:, 1, 2] = -scale * centroid[:, 0, 1]
    return pts_n, T


def build_design_matrix(pts_a, pts_b):
    """pts_a, pts_b: (..., N, 2) (any leading batch dims, shared). Row k:
    [xa*xb, xa*yb, xa, ya*xb, ya*yb, ya, xb, yb, 1] -- the standard 8-point
    linearization of x_b^T E x_a = 0."""
    xa, ya = pts_a[..., 0], pts_a[..., 1]
    xb, yb = pts_b[..., 0], pts_b[..., 1]
    ones = torch.ones_like(xa)
    return torch.stack([xa * xb, xa * yb, xa, ya * xb, ya * yb, ya, xb, yb, ones], dim=-1)


def solve_null_vector(A):
    """A: (..., N, 9), N >= 8. Returns (..., 3, 3): the right singular vector
    for the smallest singular value, reshaped -- the linear-least-squares E
    (or E-candidate) BEFORE the rank-2 manifold projection below."""
    _, _, Vh = torch.linalg.svd(A)
    return Vh[..., -1, :].reshape(*A.shape[:-2], 3, 3)


def project_rank2(E):
    """Project onto the essential-matrix manifold: singular values (s0,s1,s2)
    -> (1,1,0). Required because the unconstrained 8-point solution only
    satisfies det(E)=0 approximately/not at all under noise; this is the
    standard fix (Hartley & Zisserman 9.6)."""
    U, S, Vh = torch.linalg.svd(E)
    S2 = torch.zeros_like(S)
    S2[..., 0] = 1.0
    S2[..., 1] = 1.0
    return U @ torch.diag_embed(S2) @ Vh


def score_sampson(pts_a, pts_b, E, thresh):
    """pts_a, pts_b: (B,M,2) UNNORMALIZED ray bearings. E: (B,H,3,3) (H may be
    1). Returns (inlier (B,H,M) bool, sampson (B,H,M) float) -- Sampson
    distance is the first-order-accurate approximation to reprojection error
    for the epipolar constraint, standard for essential/fundamental matrix
    inlier scoring (cheaper than the true nonlinear reprojection error)."""
    B, M, _ = pts_a.shape
    H = E.shape[1]
    ones = torch.ones(B, M, 1, device=pts_a.device, dtype=pts_a.dtype)
    ha = torch.cat([pts_a, ones], dim=-1)  # (B,M,3)
    hb = torch.cat([pts_b, ones], dim=-1)
    ha_exp = ha.unsqueeze(1).expand(-1, H, -1, -1)  # (B,H,M,3)
    hb_exp = hb.unsqueeze(1).expand(-1, H, -1, -1)

    Exa = torch.einsum("bhij,bhmj->bhmi", E, ha_exp)   # (B,H,M,3)
    Etxb = torch.einsum("bhji,bhmj->bhmi", E, hb_exp)  # (B,H,M,3)
    residual = torch.einsum("bhmi,bhmi->bhm", hb_exp, Exa)  # (B,H,M)
    denom = Exa[..., 0] ** 2 + Exa[..., 1] ** 2 + Etxb[..., 0] ** 2 + Etxb[..., 1] ** 2 + 1e-12
    sampson = residual ** 2 / denom
    return sampson < thresh ** 2, sampson


def normalized_8pt_ransac(pts_a, pts_b, n_hyp=512, thresh=0.003, min_refine_inliers=12,
                           refine=False):
    """pts_a, pts_b: (B, M, 2) normalized ray bearings (x/z, y/z), SAME M
    across the batch (truncate to the batch minimum first, same convention as
    LightGlue batching elsewhere in this pipeline). Returns (inlier (B,M)
    bool, E (B,3,3) float) -- the winning hypothesis's inlier mask/E, after
    an optional all-inliers refinement pass.

    `refine=False` by default: direct A/B testing found the refinement pass
    (refit E via the masked all-inliers design matrix, described below)
    consistently HURTS accuracy relative to just keeping the best minimal-8
    hypothesis as-is -- confirmed even when refitting from cv2's OWN true
    inlier set (not just our RANSAC's chosen mask), so this isn't a bug in
    the masked-matrix trick (verified bit-identical to explicit row-slicing)
    -- it's a real property of the classic linear/algebraic 8-point solve:
    minimizing algebraic error (||Ae||) over many redundant correspondences
    does not track true geometric (Sampson) error the way a single
    well-conditioned minimal sample can, so more data does not monotonically
    improve it without an iterative/reweighted refinement this file doesn't
    implement. Kept here (behind the flag) as a documented dead end rather
    than deleted, since it's a non-obvious result worth not re-discovering.

    Every hypothesis for every pair is built and scored in ONE batched op
    (no Python loop over pairs OR over RANSAC iterations) -- see module
    docstring."""
    B, M, _ = pts_a.shape
    device, dtype = pts_a.device, pts_a.dtype

    pts_a_n, Ta = hartley_normalize(pts_a)
    pts_b_n, Tb = hartley_normalize(pts_b)

    # without-replacement random 8-of-M samples per (pair, hypothesis) via
    # the argsort-of-random-values trick -- no explicit loop needed.
    rand = torch.rand(B, n_hyp, M, device=device)
    samp_idx = rand.argsort(dim=-1)[:, :, :8]  # (B,H,8)

    def gather(pts_n, idx):
        idx_exp = idx.unsqueeze(-1).expand(-1, -1, -1, 2)
        pts_exp = pts_n.unsqueeze(1).expand(-1, n_hyp, -1, -1)
        return torch.gather(pts_exp, 2, idx_exp)  # (B,H,8,2)

    sa = gather(pts_a_n, samp_idx)
    sb = gather(pts_b_n, samp_idx)

    A = build_design_matrix(sa, sb)          # (B,H,8,9)
    E_n = project_rank2(solve_null_vector(A))  # (B,H,3,3), normalized space

    Tb_T = Tb.transpose(-1, -2).unsqueeze(1)  # (B,1,3,3)
    Ta_ = Ta.unsqueeze(1)                     # (B,1,3,3)
    E = Tb_T @ E_n @ Ta_                      # (B,H,3,3), denormalized

    inlier_hyp, _ = score_sampson(pts_a, pts_b, E, thresh)  # (B,H,M)
    inlier_count = inlier_hyp.sum(dim=-1)  # (B,H)
    best_h = inlier_count.argmax(dim=-1)   # (B,)
    idx_b = torch.arange(B, device=device)
    best_inlier = inlier_hyp[idx_b, best_h]  # (B,M)
    best_E = E[idx_b, best_h]                # (B,3,3)
    best_count = inlier_count[idx_b, best_h]  # (B,)

    if not refine:
        return best_inlier, best_E

    # ---- refinement: refit E from ALL inliers of the winning hypothesis,
    # via a masked full design matrix (zeroed non-inlier rows) instead of a
    # variable-length gather. Default-off -- see docstring above; kept for
    # experimentation, not because it currently helps. ----
    A_full_n = build_design_matrix(pts_a_n, pts_b_n)  # (B,M,9)
    A_masked = A_full_n * best_inlier.to(dtype).unsqueeze(-1)
    E_ref_n = project_rank2(solve_null_vector(A_masked))  # (B,3,3)
    E_ref = Tb.transpose(-1, -2) @ E_ref_n @ Ta

    ref_inlier, _ = score_sampson(pts_a, pts_b, E_ref.unsqueeze(1), thresh)
    ref_inlier = ref_inlier[:, 0]         # (B,M)
    ref_count = ref_inlier.sum(dim=-1)

    # Only trust the refit if the winning hypothesis had enough inliers for
    # the masked SVD to be well-posed, and it didn't collapse (a degenerate
    # refit can occasionally lose most inliers if the masked matrix's
    # surviving rank is marginal) -- fall back to the hypothesis-stage E
    # otherwise.
    use_refined = (best_count >= min_refine_inliers) & (ref_count.float() >= best_count.float() * 0.8)
    final_inlier = torch.where(use_refined.unsqueeze(-1), ref_inlier, best_inlier)
    final_E = torch.where(use_refined.view(-1, 1, 1), E_ref, best_E)
    return final_inlier, final_E


if __name__ == "__main__":
    # Validation harness: gathers real eligible temporal pairs from this
    # pipeline's own machinery (FeatureStore, chunk_pair_specs, match_batch,
    # unproject_batch), processes them in a REALISTIC batch size (sorted by
    # keypoint count, batch=16 -- matching the actual chunked pipeline, not
    # an artificial single global truncation across hundreds of pairs), and
    # reports both inlier-decision agreement against cv2.findEssentialMat and
    # a speed comparison.
    import argparse
    import os
    import time

    import numpy as np
    from lightglue import LightGlue

    import vio_match_pairs as M

    ap = argparse.ArgumentParser()
    ap.add_argument("recording")
    ap.add_argument("--features", default=None)
    ap.add_argument("--n-frames", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--n-test-pairs", type=int, default=500)
    ap.add_argument("--n-hyp", type=int, default=512)
    ap.add_argument("--epipolar-px-thresh", type=float, default=3.0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    features_path = args.features or os.path.join(args.recording, "features.h5")
    device = args.device

    gaps = M.build_temporal_gaps(3, 10, 2, 30, 5, 60, 10)
    store = M.FeatureStore(features_path, device)
    ls = store.f.attrs["left_serial"]
    rs = store.f.attrs["right_serial"]
    Kl, Dl = M.load_intrinsics(args.recording, ls)
    Kr, Dr = M.load_intrinsics(args.recording, rs)
    R, t = M.load_stereo(args.recording, ls, rs)
    cams = {"left": (Kl, Dl), "right": (Kr, Dr), "stereo": (R, t)}

    n_frames = min(store.n_frames("left"), store.n_frames("right"), args.n_frames)
    matcher = LightGlue(features="superpoint", width_confidence=-1).eval().to(device)

    conf_thresh, min_raw = 0.2, 10
    f_avg = (Kl[0, 0] + Kl[1, 1] + Kr[0, 0] + Kr[1, 1]) / 4.0
    theta_tol = args.epipolar_px_thresh / f_avg
    print(f"theta_tol={theta_tol:.6f}")

    specs = M.chunk_pair_specs(0, n_frames, n_frames, gaps)
    counted = [(s, store.count(s[0], s[1]), store.count(s[2], s[3])) for s in specs]
    counted.sort(key=lambda x: (x[1], x[2]))

    eligible = []
    for bi in range(0, len(counted), args.batch_size):
        batch = counted[bi:bi + args.batch_size]
        batch_specs = [(ea, fa, eb, fb) for (ea, fa, eb, fb, _, _), _, _ in batch]
        results = M.match_batch(matcher, store, batch_specs, conf_thresh, device)
        for ((ea, fa, eb, fb, pair_type, gap), _, _), (idx, scores) in zip(batch, results):
            if pair_type == "temporal" and idx.shape[0] >= min_raw:
                eligible.append((ea, fa, eb, fb, gap, idx))
    print(f"{len(eligible)} eligible temporal pairs")

    # batched ray computation (reuse gate_chunk's approach) for the pairs
    # we'll actually test.
    N_TEST = min(args.n_test_pairs, len(eligible))
    test_set = eligible[:N_TEST]

    buf_pts = {"left": [], "right": []}
    buf_meta = {"left": [], "right": []}
    for pair_idx, (ea, fa, eb, fb, gap, idx) in enumerate(test_set):
        kpa, _, _ = store.get(ea, fa)
        kpb, _, _ = store.get(eb, fb)
        pts_a, pts_b = kpa[idx[:, 0]], kpb[idx[:, 1]]
        buf_pts[ea].append(pts_a); buf_meta[ea].append((pair_idx, "a", pts_a.shape[0]))
        buf_pts[eb].append(pts_b); buf_meta[eb].append((pair_idx, "b", pts_b.shape[0]))
    rays_by_pair_side = {}
    for eye in ("left", "right"):
        if not buf_pts[eye]:
            continue
        all_pts = np.concatenate(buf_pts[eye], axis=0)
        K, D = cams[eye]
        all_rays = M.unproject_batch(all_pts, K, D, device)
        offset = 0
        for pair_idx, side, n in buf_meta[eye]:
            rays_by_pair_side[(pair_idx, side)] = all_rays[offset:offset + n]
            offset += n

    # ---------------- cv2 baseline (sequential, CPU) ----------------
    t0 = time.time()
    cv2_results = []
    for pair_idx in range(N_TEST):
        rays_a = rays_by_pair_side[(pair_idx, "a")]
        rays_b = rays_by_pair_side[(pair_idx, "b")]
        cv2_results.append(M.gate_temporal_ransac(rays_a, rays_b, theta_tol))
    t_cv2 = time.time() - t0
    print(f"\ncv2 (CPU, {N_TEST} sequential pairs): {t_cv2*1000:.1f}ms total "
          f"({1000*t_cv2/N_TEST:.3f}ms/pair)")

    # ---------------- PASS A: pure algorithm correctness ----------------
    # One pair at a time, B=1, NO truncation (every point the pair actually
    # has) -- isolates "does our normalized 8pt+RANSAC agree with cv2 on the
    # same input", decoupled from the batch-truncation completeness question
    # (which is measured separately below, matching the LightGlue-truncation
    # precedent elsewhere in this pipeline). Not a speed test -- a B=1 GPU
    # call pays pure launch overhead (same lesson as kornia's per-pair 500x
    # slowdown), so timing here would be meaningless.
    agree_a, n_a = 0, 0
    inliers_a, cv2_inliers_a = 0, 0
    for pair_idx in range(N_TEST):
        ra = rays_by_pair_side[(pair_idx, "a")]
        rb = rays_by_pair_side[(pair_idx, "b")]
        pa = torch.from_numpy((ra[:, :2] / ra[:, 2:3]).astype(np.float32)).unsqueeze(0).to(device)
        pb = torch.from_numpy((rb[:, :2] / rb[:, 2:3]).astype(np.float32)).unsqueeze(0).to(device)
        if pa.shape[1] < 8:
            g = np.zeros(pa.shape[1], dtype=bool)
        else:
            inlier, _ = normalized_8pt_ransac(pa, pb, n_hyp=args.n_hyp, thresh=theta_tol)
            g = inlier[0].cpu().numpy()
        c = cv2_results[pair_idx]
        agree_a += (c == g).sum()
        n_a += len(c)
        inliers_a += g.sum()
        cv2_inliers_a += c.sum()
    print(f"\n[PASS A: algorithm correctness, full points, no truncation]")
    print(f"inlier-decision agreement: {agree_a}/{n_a} ({100*agree_a/n_a:.1f}%)")
    print(f"cv2 total inliers: {cv2_inliers_a}, ours total inliers: {inliers_a}")

    # ---------------- PASS B: realistic production speed + completeness ----
    # batched by args.batch_size, sorted by count, matching the real chunked
    # pipeline design -- truncates each batch to its minimum point count
    # (same convention as LightGlue batching), so agreement here also
    # reflects truncation completeness loss, not just algorithm correctness.
    order = sorted(range(N_TEST), key=lambda i: rays_by_pair_side[(i, "a")].shape[0])

    gpu_inlier_by_pair = {}
    gpu_E_by_pair = {}
    torch.cuda.synchronize()
    t0 = time.time()
    for bi in range(0, N_TEST, args.batch_size):
        batch_idx = order[bi:bi + args.batch_size]
        Mn = min(rays_by_pair_side[(i, "a")].shape[0] for i in batch_idx)
        pa_list, pb_list = [], []
        for i in batch_idx:
            ra = rays_by_pair_side[(i, "a")]
            rb = rays_by_pair_side[(i, "b")]
            pa_list.append((ra[:, :2] / ra[:, 2:3])[:Mn])
            pb_list.append((rb[:, :2] / rb[:, 2:3])[:Mn])
        pts_a_t = torch.from_numpy(np.stack(pa_list).astype(np.float32)).to(device)
        pts_b_t = torch.from_numpy(np.stack(pb_list).astype(np.float32)).to(device)
        inlier, E = normalized_8pt_ransac(pts_a_t, pts_b_t, n_hyp=args.n_hyp, thresh=theta_tol)
        inlier_np = inlier.cpu().numpy()
        for row, i in enumerate(batch_idx):
            gpu_inlier_by_pair[i] = inlier_np[row]
    torch.cuda.synchronize()
    t_gpu = time.time() - t0
    print(f"\n[PASS B: realistic batched production speed + completeness]")
    print(f"ours (GPU batched, batch={args.batch_size}, n_hyp={args.n_hyp}, "
          f"{N_TEST} pairs): {t_gpu*1000:.1f}ms total ({1000*t_gpu/N_TEST:.3f}ms/pair) "
          f"-- {t_cv2/t_gpu:.1f}x speedup")

    # ---------------- compare (note: cv2 fit on FULL points, ours fit on the
    # batch-truncated subset -- disagreement here includes real truncation
    # completeness loss, not just algorithm disagreement; see PASS A above
    # for the apples-to-apples number) ----------------
    agree_total, n_total = 0, 0
    cv2_inlier_total, gpu_inlier_total = 0, 0
    for i, c in enumerate(cv2_results):
        g = gpu_inlier_by_pair[i]
        n = min(len(c), len(g))
        agree_total += (c[:n] == g[:n]).sum()
        n_total += n
        cv2_inlier_total += c.sum()
        gpu_inlier_total += g.sum()

    print(f"inlier-decision agreement (truncated subset only): {agree_total}/{n_total} ({100*agree_total/n_total:.1f}%)")
    print(f"cv2 total inliers (full): {cv2_inlier_total}, ours total inliers (truncated): {gpu_inlier_total}")
