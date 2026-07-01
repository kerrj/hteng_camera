"""GPU-batched relative-pose RANSAC via jaxls nonlinear least squares, meant to
replace both `ransac_8pt.py` (which hit a real accuracy ceiling -- see below)
and eventually `cv2.findEssentialMat` in `vio_match_pairs.gate_temporal_ransac`.

Why this exists (see `ransac_8pt.py`'s docstring for the full investigation):
the classic linear/algebraic 8-point algorithm, even correctly
Hartley-normalized, was found to have a REAL accuracy ceiling on this data --
confirmed by direct experiment, not a bug: feeding cv2's own verified true
inlier set through a linear least-squares 8-point refit still recovered far
fewer inliers than cv2's actual result (69/286 vs 248/286 on one real pair),
and sweeping 2000 random *genuinely all-inlier* 8-point samples never got
close either (best 162/286). Root cause: the linear 8-point solve minimizes
ALGEBRAIC error (||A e||), which is not the same objective as the true
geometric (Sampson) error -- normalization only fixes numerical conditioning,
not this bias. This gets systematically worse for larger temporal gaps (more
general motion): 85.7% cv2-agreement at gap=1, down to ~35% by gap=40-60.

Fix: solve the CORRECT nonlinear objective (5-DOF relative pose: SO3 rotation
+ translation DIRECTION, minimizing Sampson error) directly via jaxls
Levenberg-Marquardt, instead of a biased linear proxy. This sidesteps
implementing the classical 5-point algorithm's polynomial/resultant root-
finding (which the earlier session discussion rejected as too complex/
bug-prone) while still getting a geometrically-correct minimal-sample
estimator -- jaxls just optimizes the actual manifold-constrained problem.

Batching design (the key insight that makes this fast): jaxls ALREADY
provides exactly the batching mechanism we need, the same way the hand
pipeline's stereo MANO optimizer solves thousands of per-frame pose Vars in
ONE call (see `../hands/stereo_optimize.py`) -- instantiate `RotVar(ids)` /
`TransDirVar(ids)` with `ids = jnp.arange(N)` where N = (num pairs) x
(num RANSAC hypotheses per pair), completely independent per id (no shared
Vars, no coupling costs between hypotheses), so the resulting Hessian is
block-diagonal and jaxls's own Schur/CG machinery solves ALL N tiny 5-point
problems in a single LM solve -- no explicit `jax.vmap` needed, we're just
handing jaxls a big batch of independent minimal problems exactly like it
already does for per-frame hand poses.

**N must be a FIXED constant across calls** to avoid `jaxls`/JAX recompiling
on every call (jit/analyze() cache keys off array shapes, which includes the
Var id count) -- so the real integration should always process a fixed
(pairs-per-chunk x hypotheses-per-pair) batch size, padding rather than
letting N vary call to call, the same "fixed shape" discipline already used
elsewhere in this pipeline (LightGlue's batch truncation-to-minimum).

The RANSAC selection itself is unchanged from `ransac_8pt.py`: every
hypothesis is solved in that ONE batched call (no Python loop over pairs or
iterations), then scored against the FULL point set via Sampson distance and
picked via `argmax` over hypotheses per pair.
"""
import jax
import jax.numpy as jnp
import jaxlie
import numpy as np

import jaxls


class RotVar(
    jaxls.Var[jax.Array],
    default_factory=lambda: jnp.array([1.0, 0.0, 0.0, 0.0]),
    retract_fn=lambda val, delta: (jaxlie.SO3(val) @ jaxlie.SO3.exp(delta)).wxyz,
    tangent_dim=3,
):
    """Relative rotation R (ray frame a -> ray frame b), wxyz quat. One
    instance per (pair, hypothesis) -- completely independent of every other
    instance (no shared/coupling costs), so the Hessian is block-diagonal."""


class TransDirVar(
    jaxls.Var[jax.Array],
    default_factory=lambda: jnp.array([1.0, 0.0, 0.0, 0.0]),
    retract_fn=lambda val, delta: (jaxlie.SO3(val) @ jaxlie.SO3.exp(delta)).wxyz,
    tangent_dim=3,
):
    """Translation DIRECTION, represented as an SO3 rotating the fixed base
    vector e_z=[0,0,1] -- reuses the same SO3 retract machinery as RotVar
    rather than a bespoke S2 manifold type. 3-DOF manifold but only 2 DOF are
    physically meaningful (rotation about the resulting direction is a free
    gauge symmetry) -- harmless, since translation SCALE is unobservable from
    2D correspondences anyway (only direction matters for the essential
    matrix / epipolar constraint)."""


_BASE_Z = jnp.array([0.0, 0.0, 1.0])


def skew(v):
    z = jnp.zeros_like(v[..., 0])
    return jnp.stack([
        jnp.stack([z, -v[..., 2], v[..., 1]], axis=-1),
        jnp.stack([v[..., 2], z, -v[..., 0]], axis=-1),
        jnp.stack([-v[..., 1], v[..., 0], z], axis=-1),
    ], axis=-2)


def sampson_residual(R, t_hat, pts_a, pts_b):
    """R: (...,3,3), t_hat: (...,3), pts_a/pts_b: (...,K,2) (K correspondences
    sharing the same R,t_hat -- either K=5 for the minimal-sample cost below,
    or K=M for final scoring against the full point set). Returns (...,K):
    the SIGNED-SQRT Sampson residual (squaring it gives the usual Sampson
    distance) -- used directly as the jaxls cost residual so the LM solve
    minimizes true Sampson error, not raw algebraic error."""
    E = skew(t_hat) @ R  # (...,3,3)
    ones = jnp.ones(pts_a.shape[:-1] + (1,), dtype=pts_a.dtype)
    ha = jnp.concatenate([pts_a, ones], axis=-1)  # (...,K,3)
    hb = jnp.concatenate([pts_b, ones], axis=-1)
    Exa = jnp.einsum("...ij,...kj->...ki", E, ha)   # (...,K,3)
    Etxb = jnp.einsum("...ji,...kj->...ki", E, hb)  # (...,K,3)
    residual = jnp.einsum("...ki,...ki->...k", hb, Exa)  # (...,K)
    denom = Exa[..., 0] ** 2 + Exa[..., 1] ** 2 + Etxb[..., 0] ** 2 + Etxb[..., 1] ** 2 + 1e-12
    return residual / jnp.sqrt(denom)


@jaxls.Cost.factory
def epipolar_cost(vals, rot_v, trans_v, pts_a5, pts_b5):
    """Per-hypothesis cost: 5 Sampson residuals from that hypothesis's 5
    sampled correspondences. jaxls auto-vmaps this single-instance function
    body over the batch of (pair, hypothesis) ids."""
    R = jaxlie.SO3(vals[rot_v]).as_matrix()
    t_hat = jaxlie.SO3(vals[trans_v]).as_matrix() @ _BASE_Z
    return sampson_residual(R, t_hat, pts_a5, pts_b5)


from functools import partial


@partial(jax.jit, static_argnames=("max_iters",))
def solve_5pt_batch(pts_a5, pts_b5, max_iters=10):
    """pts_a5, pts_b5: (N,5,2) jax arrays, ONE 5-point minimal sample per row.
    N must stay FIXED across calls to avoid retracing (see module docstring).

    jax.jit is REQUIRED here, not optional: `LeastSquaresProblem.analyze()` +
    `.solve()` do all their sparsity-structure/type bookkeeping in eager
    Python (confirmed by reading jaxls/_problem.py -- no internal jax.jit,
    only static shape/type branching, and `LeastSquaresProblem` /
    `AnalyzedLeastSquaresProblem` are `@jdc.pytree_dataclass`, i.e. designed
    to be traced). Without wrapping this function, EVERY call re-runs that
    ~4s Python-level analysis from scratch (confirmed by direct testing: the
    same 16-hypothesis problem re-logged "Building optimization problem" on
    every single call, no caching, 2000ms/pair vs cv2's 0.76ms/pair). With
    jax.jit, that analysis happens once per unique (N, max_iters) shape and
    is replayed as compiled XLA on every subsequent call.

    Returns (R (N,3,3), t_hat (N,3)) -- the LM-converged relative pose for
    each of the N independent 5-point problems, solved in ONE call."""
    n = pts_a5.shape[0]
    ids = jnp.arange(n)
    costs = [epipolar_cost(RotVar(ids), TransDirVar(ids), pts_a5, pts_b5)]
    init = jaxls.VarValues.make([RotVar(ids), TransDirVar(ids)])
    prob = jaxls.LeastSquaresProblem(costs, [RotVar(ids), TransDirVar(ids)]).analyze()
    sol = prob.solve(init, trust_region=jaxls.TrustRegionConfig(),
                      termination=jaxls.TerminationConfig(max_iterations=max_iters),
                      verbose=False)
    R = jaxlie.SO3(sol[RotVar]).as_matrix()
    t_hat = jaxlie.SO3(sol[TransDirVar]).as_matrix() @ _BASE_Z
    return R, t_hat


def sample_minimal_indices(rng, n_pairs, n_hyp, m_points):
    """Without-replacement random 5-of-M samples for every (pair, hypothesis)
    -- same argsort-of-random-values trick as ransac_8pt.py. Returns int32
    (n_pairs, n_hyp, 5)."""
    rand = jax.random.uniform(rng, (n_pairs, n_hyp, m_points))
    return jnp.argsort(rand, axis=-1)[:, :, :5]


def score_sampson_full(pts_a, pts_b, E, thresh):
    """pts_a, pts_b: (B,M,2). E: (B,H,3,3). Returns (inlier (B,H,M) bool,
    inlier_count (B,H)) -- identical convention/threshold semantics to
    ransac_8pt.score_sampson (kept jax-native here so this module has no
    torch dependency)."""
    B, M, _ = pts_a.shape
    H = E.shape[1]
    ones = jnp.ones((B, M, 1), dtype=pts_a.dtype)
    ha = jnp.concatenate([pts_a, ones], axis=-1)
    hb = jnp.concatenate([pts_b, ones], axis=-1)
    ha_exp = jnp.broadcast_to(ha[:, None], (B, H, M, 3))
    hb_exp = jnp.broadcast_to(hb[:, None], (B, H, M, 3))
    Exa = jnp.einsum("bhij,bhmj->bhmi", E, ha_exp)
    Etxb = jnp.einsum("bhji,bhmj->bhmi", E, hb_exp)
    residual = jnp.einsum("bhmi,bhmi->bhm", hb_exp, Exa)
    denom = Exa[..., 0] ** 2 + Exa[..., 1] ** 2 + Etxb[..., 0] ** 2 + Etxb[..., 1] ** 2 + 1e-12
    sampson = residual ** 2 / denom
    inlier = sampson < thresh ** 2
    return inlier, inlier.sum(axis=-1)


@partial(jax.jit, static_argnames=("n_hyp", "max_iters"))
def relpose_ransac(pts_a, pts_b, rng, n_hyp, thresh, max_iters=10):
    """pts_a, pts_b: (B, M, 2) normalized ray bearings, SAME M across the
    batch (same truncate-to-batch-minimum convention as the rest of this
    pipeline). B and n_hyp (hence N=B*n_hyp) should stay fixed across calls
    -- see module docstring. Returns (inlier (B,M) bool, E (B,3,3))."""
    B, M, _ = pts_a.shape
    samp_idx = sample_minimal_indices(rng, B, n_hyp, M)  # (B,H,5)

    pair_idx = jnp.arange(B)[:, None, None]
    pts_a5 = pts_a[pair_idx, samp_idx]  # (B,H,5,2)
    pts_b5 = pts_b[pair_idx, samp_idx]

    N = B * n_hyp
    R, t_hat = solve_5pt_batch(pts_a5.reshape(N, 5, 2), pts_b5.reshape(N, 5, 2), max_iters)
    R = R.reshape(B, n_hyp, 3, 3)
    t_hat = t_hat.reshape(B, n_hyp, 3)
    E = skew(t_hat) @ R  # (B,H,3,3)

    inlier_hyp, inlier_count = score_sampson_full(pts_a, pts_b, E, thresh)  # (B,H,M),(B,H)
    best_h = jnp.argmax(inlier_count, axis=-1)  # (B,)
    idx_b = jnp.arange(B)
    best_inlier = inlier_hyp[idx_b, best_h]
    best_E = E[idx_b, best_h]
    return best_inlier, best_E


if __name__ == "__main__":
    # Validation harness: same structure/methodology as ransac_8pt.py's
    # `__main__` -- gather real eligible temporal pairs, PASS A (algorithm
    # correctness: one pair at a time, full points, no truncation, compared
    # against cv2.findEssentialMat) and PASS B (realistic batched production
    # speed, fixed B/n_hyp to respect jaxls's no-recompile constraint).
    import argparse
    import os
    import time
    import sys

    import cv2

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import vio_match_pairs as M
    from lightglue import LightGlue
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("recording")
    ap.add_argument("--features", default=None)
    ap.add_argument("--n-frames", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--n-test-pairs", type=int, default=100)
    ap.add_argument("--n-hyp", type=int, default=64)
    ap.add_argument("--max-iters", type=int, default=10)
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
    cams = {"left": (Kl, Dl), "right": (Kr, Dr)}

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

    # ---------------- cv2 baseline ----------------
    t0 = time.time()
    cv2_results = []
    for pair_idx in range(N_TEST):
        rays_a = rays_by_pair_side[(pair_idx, "a")]
        rays_b = rays_by_pair_side[(pair_idx, "b")]
        cv2_results.append(M.gate_temporal_ransac(rays_a, rays_b, theta_tol))
    t_cv2 = time.time() - t0
    print(f"\ncv2 (CPU, {N_TEST} sequential pairs): {t_cv2*1000:.1f}ms total "
          f"({1000*t_cv2/N_TEST:.3f}ms/pair)")

    # ---------------- PASS A: algorithm correctness (full points, B=1) ------
    key = jax.random.PRNGKey(0)
    agree_a, n_a = 0, 0
    inliers_a, cv2_inliers_a = 0, 0
    t_jax_total = 0.0
    for pair_idx in range(N_TEST):
        ra = rays_by_pair_side[(pair_idx, "a")]
        rb = rays_by_pair_side[(pair_idx, "b")]
        pa = jnp.asarray((ra[:, :2] / ra[:, 2:3]).astype(np.float32))[None]
        pb = jnp.asarray((rb[:, :2] / rb[:, 2:3]).astype(np.float32))[None]
        n = pa.shape[1]
        c = cv2_results[pair_idx]
        if n < 8:
            g = np.zeros(n, dtype=bool)
        else:
            key, sub = jax.random.split(key)
            t0 = time.time()
            inlier, _ = relpose_ransac(pa, pb, sub, args.n_hyp, theta_tol, args.max_iters)
            inlier = jax.block_until_ready(inlier)
            t_jax_total += time.time() - t0
            g = np.asarray(inlier[0])
        agree_a += (c == g).sum()
        n_a += len(c)
        inliers_a += g.sum()
        cv2_inliers_a += c.sum()
    print(f"\n[PASS A: algorithm correctness, full points, no truncation, B=1]")
    print(f"(NOTE: B=1 timing includes per-call jaxls compile overhead -- not "
          f"representative of batched speed; accuracy is the point of this pass)")
    print(f"inlier-decision agreement: {agree_a}/{n_a} ({100*agree_a/n_a:.1f}%)")
    print(f"cv2 total inliers: {cv2_inliers_a}, ours total inliers: {inliers_a}")

    # ---------------- PASS B: realistic batched production speed -----------
    # fixed (batch_size, n_hyp) across all calls -- required for jaxls to hit
    # its compilation cache instead of re-tracing every batch.
    order = sorted(range(N_TEST), key=lambda i: rays_by_pair_side[(i, "a")].shape[0])
    B = args.batch_size

    # build all fixed-M batches first (M can vary batch-to-batch depending on
    # each batch's minimum point count, which -- since M is a traced shape,
    # not a static_argname -- triggers its OWN separate jit trace; grouping
    # here just lets us report per-batch timing/M so a stray recompile is
    # visible rather than silently absorbed into "first batch excluded").
    batches = []
    for bi in range(0, N_TEST, B):
        batch_idx = order[bi:bi + B]
        if len(batch_idx) < B:
            break  # drop the ragged last batch -- fixed shape only
        Mn = min(rays_by_pair_side[(i, "a")].shape[0] for i in batch_idx)
        if Mn < 8:
            continue
        pa_list, pb_list = [], []
        for i in batch_idx:
            ra = rays_by_pair_side[(i, "a")]
            rb = rays_by_pair_side[(i, "b")]
            pa_list.append((ra[:, :2] / ra[:, 2:3])[:Mn])
            pb_list.append((rb[:, :2] / rb[:, 2:3])[:Mn])
        pts_a_t = jnp.asarray(np.stack(pa_list).astype(np.float32))
        pts_b_t = jnp.asarray(np.stack(pb_list).astype(np.float32))
        batches.append((batch_idx, Mn, pts_a_t, pts_b_t))

    # ---- EXPLICIT warmup: trace/compile using the FIRST batch's exact shape
    # before any timing starts, rather than folding compile cost into
    # "skip batch 1" (which silently breaks if a later batch has a different
    # M and triggers its own untimed recompile). ----
    key = jax.random.PRNGKey(1)
    if batches:
        _, _, warm_a, warm_b = batches[0]
        key, sub = jax.random.split(key)
        warm_out, _ = relpose_ransac(warm_a, warm_b, sub, args.n_hyp, theta_tol, args.max_iters)
        jax.block_until_ready(warm_out)
        print(f"\n[warmup done: M={batches[0][1]}, shape {warm_a.shape}]")

    gpu_inlier_by_pair = {}
    t_gpu_total = 0.0
    n_timed = 0
    last_m = batches[0][1] if batches else None
    for batch_idx, Mn, pts_a_t, pts_b_t in batches:
        key, sub = jax.random.split(key)
        t0 = time.time()
        inlier, E = relpose_ransac(pts_a_t, pts_b_t, sub, args.n_hyp, theta_tol, args.max_iters)
        inlier = jax.block_until_ready(inlier)
        dt = time.time() - t0
        flag = "" if Mn == last_m else "  <- M changed, this batch pays a fresh compile"
        print(f"  batch M={Mn:4d} n_pairs={len(batch_idx):3d}: {dt*1000:8.1f}ms "
              f"({1000*dt/len(batch_idx):.3f}ms/pair){flag}")
        last_m = Mn
        t_gpu_total += dt
        n_timed += len(batch_idx)
        inlier_np = np.asarray(inlier)
        for row, i in enumerate(batch_idx):
            gpu_inlier_by_pair[i] = inlier_np[row]

    print(f"\n[PASS B: realistic batched production speed (fixed batch={B}, n_hyp={args.n_hyp})]")
    if n_timed > 0:
        print(f"ours (GPU batched, post-warmup, {n_timed} pairs): "
              f"{t_gpu_total*1000:.1f}ms total ({1000*t_gpu_total/n_timed:.3f}ms/pair)")

    agree_total, n_total = 0, 0
    cv2_inlier_total, gpu_inlier_total = 0, 0
    for i, g in gpu_inlier_by_pair.items():
        c = cv2_results[i]
        n = min(len(c), len(g))
        agree_total += (c[:n] == g[:n]).sum()
        n_total += n
        cv2_inlier_total += c.sum()
        gpu_inlier_total += g.sum()
    print(f"inlier-decision agreement (truncated subset only): {agree_total}/{n_total} "
          f"({100*agree_total/max(n_total,1):.1f}%)")
    print(f"cv2 total inliers (full): {cv2_inlier_total}, ours total inliers (truncated): {gpu_inlier_total}")
