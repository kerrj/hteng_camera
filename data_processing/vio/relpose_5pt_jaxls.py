"""GPU-batched relative-pose RANSAC via jaxls nonlinear least squares, meant to
replace `cv2.findEssentialMat` in `vio_match_pairs.gate_temporal_ransac`.

Why this exists: an earlier prototype (a hand-rolled, GPU-batched, Hartley-
normalized linear 8-point RANSAC -- since deleted, its findings are fully
summarized here) hit a REAL accuracy ceiling on this data --
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

**CRITICAL gotcha found by direct measurement, not anticipated:** the LM
solve (`solve_5pt_batch`) must be its OWN top-level `@jax.jit` function,
called directly -- NEVER nested inside another `@jax.jit`'d function that
also touches `M` (the full per-pair point count used for final scoring,
which is NOT the same as `N` and DOES vary batch to batch after
truncation-to-batch-minimum). First implementation wrapped sampling + solve
+ scoring in ONE outer `jax.jit`; since `M` changes almost every batch (363,
376, 385, 392, ... across just 4 real batches), that outer function retraced
constantly, and because a nested `jax.jit` call gets INLINED into its
enclosing trace (not independently cache-hit), every retrace dragged
`solve_5pt_batch`'s jaxls `analyze()` step along with it too -- even though
`solve_5pt_batch`'s own inputs (shape `(N,5,2)`, N fixed) never changed.
Measured cost of this bug: 259ms/pair -> 112ms/pair as batch size grew,
looking like slow-but-improving GPU throughput, when the real signature was
"every batch pays a ~2.3s recompile" (confirmed by logging per-batch time:
the ONE batch matching the warmup's exact M ran at 1.04ms/pair, competitive
with cv2's 0.66ms/pair; every other batch, with a different M, paid the
full ~2.3s analyze() cost again). Fix: `sample_minimal_points` and
`select_best` (both M-dependent) are their own SEPARATE `@jax.jit` functions;
`solve_5pt_batch` is called directly between them by a plain (non-jitted)
Python orchestration function (`relpose_ransac`) so its cache -- keyed only
on `N = B * n_hyp`, fixed for the life of the process -- is actually hit on
every call, while only the cheap sampling/scoring steps pay for `M` varying.

**Locked-in defaults, confirmed by direct sweeps on the full-gap-range
(200-frame, gaps 1-60) real dataset:**
- `n_hyp=100`: nearly free (batched, not sequential) -- 32->512 barely moved
  speed in isolation.
- `max_iters=10`: the dominant accuracy lever, NOT n_hyp -- 3->10 iterations
  moved cv2-agreement 84.4%->90.9% (every hypothesis cold-starts from the
  same identity/e_z init, so large-gap/large-motion pairs need more LM steps
  to converge that far regardless of which 5 points were sampled). Plateaus
  past 10 (10->20 only +0.8% for ~2x the cost).
- **Threshold: use 0.75x the epipolar-px-derived `theta_tol`, not 1.0x.**
  "Aggregate agreement with cv2" is a MISLEADING tuning signal here -- it's
  dominated by the "both agree it's an inlier" majority class (cv2 marks
  ~86.5% of points inlier) and stays ~flat (91.0%->91.5%) as the threshold is
  loosened from 1.0x to 1.5x, while the false-positive rate (fraction of
  cv2's REJECTED points that we admit -- the dangerous error, since this
  pipeline's whole design is "gate hard, then Huber the rest", see
  vio/CLAUDE.md) more than DOUBLES over that same range (35.4%->50.2%).
  Measured false-negative/false-positive breakdown by threshold scale (batch
  256, m_pad 512, n_hyp 100, max_iters 10):
      scale  thresh    agree%   FN%    FP%
      0.25   0.00097   54.5%   51.1%   9.0%
      0.50   0.00194   79.1%   21.4%  17.4%
      0.75   0.00291   88.2%    9.5%  26.5%
      1.00   0.00387   91.0%    4.9%  35.4%
      1.25   0.00484   91.5%    3.0%  43.5%
      1.50   0.00581   91.5%    2.0%  50.2%
      2.00   0.00775   90.9%    1.2%  60.1%
  0.75x chosen as the balance point: meaningfully lower false-positive rate
  than 1.0x (26.5% vs 35.4%) without the false-negative rate blowing up the
  way it does at 0.5x and below (9.5% vs 21.4%). Callers (e.g. the eventual
  vio_match_pairs.py integration) should multiply `theta_tol` by 0.75 before
  passing it to `relpose_ransac`/`score_sampson_full` -- NOT change the
  physical epipolar-px-threshold definition itself, which stays shared with
  `gate_stereo` and cv2's convention.
"""
from functools import partial

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


@jaxls.Cost.factory
def masked_epipolar_cost(vals, rot_v, trans_v, pts_a, pts_b, mask):
    """Polish-step cost: Sampson residual over the FULL (padded) per-pair
    point set, with non-inlier/padded rows zeroed out -- exactly the
    "masked design matrix" trick from ransac_8pt.py's refinement attempt,
    except here it's safe: we're minimizing the actual Sampson objective via
    LM, not an algebraic-error linear proxy, so adding more (masked) inlier
    data should only improve the fit, not bias it (that bias was specific to
    the linear 8-point's algebraic-vs-geometric-error mismatch)."""
    R = jaxlie.SO3(vals[rot_v]).as_matrix()
    t_hat = jaxlie.SO3(vals[trans_v]).as_matrix() @ _BASE_Z
    res = sampson_residual(R, t_hat, pts_a, pts_b)  # (M,)
    return jnp.where(mask, res, 0.0)


def quat_from_two_vectors(v_from, v_to):
    """Shortest-arc rotation quaternion (wxyz) mapping unit vector `v_from`
    to unit vector `v_to`, batched over leading dims. Used to warm-start
    TransDirVar's polish-step initial value from a winning hypothesis's
    t_hat (recall TransDirVar's value is an SO3 that rotates the fixed
    e_z=[0,0,1] to the translation direction -- any quat achieving that
    works as an init; the "shortest arc" choice is just a convenient one,
    since the rotation-about-t_hat gauge freedom doesn't affect the cost)."""
    dot = jnp.sum(v_from * v_to, axis=-1, keepdims=True)
    cross = jnp.cross(v_from, v_to)
    q = jnp.concatenate([1.0 + dot, cross], axis=-1)
    return q / (jnp.linalg.norm(q, axis=-1, keepdims=True) + 1e-12)


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
    each of the N independent 5-point problems, solved in ONE call.

    NOTE on jaxls's "variable IDs are traced; solves will not use
    elimination" log message: this problem IS fully block-diagonal (every
    hypothesis independent, no shared Vars), which should hit jaxls's
    cheapest path (exact per-block inversion, no CG/Cholesky iteration at
    all). Tried building `ids` with plain `numpy.arange` + `analyze(use_onp=
    True)` to get analyze() to see concrete (non-traced) ids and enable
    elimination -- this CRASHED (`TracerArrayConversionError`), because
    `analyze(use_onp=True)` appears to assume it's called fully eagerly
    (outside any `jax.jit` trace), which conflicts with our requirement that
    `analyze()` run INSIDE the jit trace for repeated-call caching to work at
    all (see the big comment above this function). Reverted to plain
    `jnp.arange` -- functionally correct (CG/dense_cholesky solves the full
    system instead of the ideal exact block inverse), just leaves some
    possible speed on the table. Worth revisiting if this ever becomes the
    bottleneck again, but not blocking today."""
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


def sample_minimal_indices(rng, valid_mask, n_hyp):
    """Without-replacement random 5-of-M samples for every (pair, hypothesis)
    -- same argsort-of-random-values trick as ransac_8pt.py, extended with a
    validity mask so PADDED points (see module docstring on fixed-M batching)
    are never selected: invalid slots get their random key pushed to +inf
    before the argsort, so they always sort last and are never among the
    first 5 -- exact as long as each pair has >=5 valid (non-padded) points,
    always true here since min-raw-matches elsewhere in this pipeline is 10.
    valid_mask: (n_pairs, M) bool. Returns int32 (n_pairs, n_hyp, 5)."""
    n_pairs, m_points = valid_mask.shape
    rand = jax.random.uniform(rng, (n_pairs, n_hyp, m_points))
    rand = jnp.where(valid_mask[:, None, :], rand, jnp.inf)
    return jnp.argsort(rand, axis=-1)[:, :, :5]


@partial(jax.jit, static_argnames=("n_hyp",))
def sample_minimal_points(pts_a, pts_b, valid_mask, rng, n_hyp):
    """pts_a, pts_b: (B,M,2), valid_mask: (B,M) bool -- M-DEPENDENT shape, so
    this jit boundary retraces whenever M changes. Kept SEPARATE from
    solve_5pt_batch (see module docstring) so that retrace only costs a cheap
    argsort+gather, not jaxls's ~seconds-long problem analysis. With FIXED-M
    batch padding (M padded to a constant bucket size, valid_mask marking the
    real points), this never retraces either -- see module docstring on
    padding. Returns (B,H,5,2) x2 sampled minimal subsets (not yet reshaped
    to (B*H,5,2) -- that reshape is free/metadata-only, done by the caller
    right before solve_5pt_batch)."""
    B, M, _ = pts_a.shape
    samp_idx = sample_minimal_indices(rng, valid_mask, n_hyp)  # (B,H,5)
    pair_idx = jnp.arange(B)[:, None, None]
    return pts_a[pair_idx, samp_idx], pts_b[pair_idx, samp_idx]


@jax.jit
def score_sampson_full(pts_a, pts_b, valid_mask, E, thresh):
    """pts_a, pts_b: (B,M,2). valid_mask: (B,M) bool. E: (B,H,3,3). Returns
    (inlier (B,H,M) bool, inlier_count (B,H)) -- identical convention/
    threshold semantics to ransac_8pt.score_sampson (kept jax-native here so
    this module has no torch dependency). M-DEPENDENT shape -- see
    sample_minimal_points. Padded (invalid) points are masked out of both
    the returned inlier array and the count, so they never contribute to
    hypothesis selection or leak into the caller's result."""
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
    inlier = (sampson < thresh ** 2) & valid_mask[:, None, :]
    return inlier, inlier.sum(axis=-1)


@jax.jit
def select_best(pts_a, pts_b, valid_mask, E, R, t_hat, thresh):
    """M-dependent (cheap) jit boundary: scores every hypothesis then argmaxes
    to pick the winner per pair. Kept separate from solve_5pt_batch for the
    same reason as sample_minimal_points. Also returns the winning R/t_hat
    (not just E) so the polish step can warm-start from them."""
    B = pts_a.shape[0]
    inlier_hyp, inlier_count = score_sampson_full(pts_a, pts_b, valid_mask, E, thresh)  # (B,H,M),(B,H)
    best_h = jnp.argmax(inlier_count, axis=-1)  # (B,)
    idx_b = jnp.arange(B)
    return (inlier_hyp[idx_b, best_h], E[idx_b, best_h],
            R[idx_b, best_h], t_hat[idx_b, best_h])


@partial(jax.jit, static_argnames=("max_iters",))
def polish_batch(pts_a, pts_b, mask, R_init, t_hat_init, max_iters=5):
    """Refit (R, t_hat) per pair using ALL of that pair's inlier points (not
    just the winning hypothesis's 5), warm-started from the hypothesis
    itself rather than cold-starting at identity. pts_a, pts_b: (B,M,2).
    mask: (B,M) bool (RANSAC inlier decision AND-ed with the padding-validity
    mask already, from select_best/score_sampson_full). R_init: (B,3,3),
    t_hat_init: (B,3). B and M must stay FIXED across calls, same discipline
    as solve_5pt_batch/sample_minimal_points -- this is its own top-level
    jit boundary so its compilation cache (keyed on (B,M,max_iters)) is
    actually reused, not inlined into some other function's trace.

    Unlike ransac_8pt.py's refinement attempt (which made things WORSE: it
    minimizes algebraic error, biased relative to true geometric error), this
    refit directly continues minimizing the same Sampson objective the
    hypothesis stage used, just with more (masked) data and a warm start --
    should only improve on the hypothesis, never regress it (mathematically:
    the hypothesis's own 5-point solution is a strict subset of the residual
    set here, so it's not an unrelated objective being swapped in)."""
    B = pts_a.shape[0]
    ids = jnp.arange(B)
    rot_init_quat = jaxlie.SO3.from_matrix(R_init).wxyz
    trans_init_quat = quat_from_two_vectors(jnp.broadcast_to(_BASE_Z, t_hat_init.shape), t_hat_init)
    costs = [masked_epipolar_cost(RotVar(ids), TransDirVar(ids), pts_a, pts_b, mask)]
    init = jaxls.VarValues.make([
        RotVar(ids).with_value(rot_init_quat),
        TransDirVar(ids).with_value(trans_init_quat),
    ])
    prob = jaxls.LeastSquaresProblem(costs, [RotVar(ids), TransDirVar(ids)]).analyze()
    sol = prob.solve(init, trust_region=jaxls.TrustRegionConfig(),
                      termination=jaxls.TerminationConfig(max_iterations=max_iters),
                      verbose=False)
    R = jaxlie.SO3(sol[RotVar]).as_matrix()
    t_hat = jaxlie.SO3(sol[TransDirVar]).as_matrix() @ _BASE_Z
    return R, t_hat


def relpose_ransac(pts_a, pts_b, rng, n_hyp, thresh, max_iters=10, valid_mask=None,
                    polish=True, polish_iters=5):
    """pts_a, pts_b: (B, M, 2) normalized ray bearings. M can either be the
    real (truncate-to-batch-minimum) count, SAME across the batch (pass
    valid_mask=None, an implicit all-True mask), or a FIXED padded bucket
    size shared across every batch for the whole run, with `valid_mask`
    (B,M) bool marking which entries are real vs. padding -- the latter is
    what avoids retracing sample_minimal_points/select_best on every batch
    (see module docstring). Returns (inlier (B,M) bool, E (B,3,3)).

    Deliberately NOT itself @jax.jit-decorated: it's plain Python
    orchestration over independently-jitted stages with DIFFERENT shape
    dependencies --
      1. sample_minimal_points (M-dependent, cheap to retrace)
      2. solve_5pt_batch       (N=B*n_hyp-dependent ONLY -- never touches M,
                                 so as long as B and n_hyp stay fixed across
                                 calls this NEVER retraces regardless of how
                                 much M varies batch-to-batch)
      3. select_best           (M-dependent, cheap to retrace)
      4. polish_batch          (B,M-dependent -- fixed as long as batch size
                                 and the M padding bucket stay constant, same
                                 as steps 1/3)
    Wrapping this orchestration function itself in @jax.jit would inline all
    stages into ONE trace keyed on the union of their shapes, which would
    silently reintroduce the exact bug this split fixes: an M change would
    then force the expensive fixed-shape LM solve to retrace too, even
    though nothing about ITS inputs changed. See module docstring.

    `polish=True` (default) adds one more small solve per pair (not per
    hypothesis) that refits (R, t_hat) from ALL of the winning hypothesis's
    inlier points, warm-started from that hypothesis -- see polish_batch's
    docstring for why this is safe (unlike ransac_8pt.py's refinement, which
    made things worse)."""
    B, M, _ = pts_a.shape
    if valid_mask is None:
        valid_mask = jnp.ones((B, M), dtype=bool)
    pts_a5, pts_b5 = sample_minimal_points(pts_a, pts_b, valid_mask, rng, n_hyp)  # (B,H,5,2)
    N = B * n_hyp
    R, t_hat = solve_5pt_batch(pts_a5.reshape(N, 5, 2), pts_b5.reshape(N, 5, 2), max_iters)
    R = R.reshape(B, n_hyp, 3, 3)
    t_hat = t_hat.reshape(B, n_hyp, 3)
    E = skew(t_hat) @ R  # (B,H,3,3)
    inlier, E_best, R_best, t_hat_best = select_best(pts_a, pts_b, valid_mask, E, R, t_hat, thresh)
    if not polish:
        return inlier, E_best
    R_ref, t_hat_ref = polish_batch(pts_a, pts_b, inlier, R_best, t_hat_best, polish_iters)
    E_ref = skew(t_hat_ref) @ R_ref  # (B,3,3)
    inlier_ref, _ = score_sampson_full(pts_a, pts_b, valid_mask, E_ref[:, None], thresh)
    return inlier_ref[:, 0], E_ref


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

    ap = argparse.ArgumentParser()
    ap.add_argument("recording")
    ap.add_argument("--features", default=None)
    ap.add_argument("--n-frames", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--n-test-pairs", type=int, default=100)
    ap.add_argument("--pass-a-pairs", type=int, default=20,
                    help="cap on PASS A's pair count, independent of --n-test-pairs "
                         "(PASS A's varying-M per-pair recompiles get very slow with "
                         "polish=True -- see inline comment)")
    ap.add_argument("--n-hyp", type=int, default=100,
                    help="RANSAC hypotheses per pair -- n_hyp is nearly free (batched, "
                         "not sequential), confirmed by direct sweep: 32->512 barely "
                         "moved speed. max_iters matters far more for accuracy on hard "
                         "(large-gap/large-motion) pairs -- see --max-iters.")
    ap.add_argument("--max-iters", type=int, default=10,
                    help="LM iterations per hypothesis -- confirmed by direct sweep "
                         "this is the dominant accuracy lever, not n_hyp: 3->10 iters "
                         "moved cv2-agreement 84.4%%->90.9%% on a hard (200-frame, all "
                         "gaps 1-60) test, since every hypothesis cold-starts from the "
                         "same identity/e_z init and large-gap pairs need more LM steps "
                         "to converge that far regardless of which 5 points were "
                         "sampled. Diminishing returns past 10 (10->20 iters only "
                         "+0.8%% for ~2x the cost).")
    ap.add_argument("--m-pad", type=int, default=512,
                    help="fixed point-count PASS B pads/truncates every batch to, "
                         "with a validity mask -- keeps sample_minimal_points/"
                         "select_best's shape constant across ALL batches so they "
                         "compile exactly once for the whole run, not once per batch")
    ap.add_argument("--epipolar-px-thresh", type=float, default=3.0)
    ap.add_argument("--threshold-scale", type=float, default=0.75,
                    help="multiplies theta_tol before it's used by relpose_ransac "
                         "(NOT by cv2, which stays at the raw epipolar-px threshold "
                         "as the fixed reference) -- 0.75 is the locked-in balance "
                         "point between false-negative and false-positive rate, see "
                         "module docstring's threshold-sweep table")
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
    theta_tol = args.epipolar_px_thresh / f_avg  # cv2's reference threshold, unscaled
    theta_tol_ours = theta_tol * args.threshold_scale  # ours: scaled per the locked-in finding
    print(f"theta_tol={theta_tol:.6f} (cv2), {theta_tol_ours:.6f} (ours, x{args.threshold_scale})")

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
    # capped to --pass-a-pairs (independent of N_TEST/PASS B's size): every
    # pair here has a DIFFERENT full point count M, and with polish=True
    # (default) each DISTINCT M pays its own separate compile of BOTH
    # sample_minimal_points/select_best AND polish_batch. Confirmed by direct
    # testing this pileup, not a slow polish_batch itself, is the culprit:
    # polish_batch benchmarked in isolation (fixed shape, called twice)
    # compiles in ~2.2s and runs in ~3ms regardless of (B,M) -- fast. But
    # PASS A calls it with a FRESH (B=1, M=that pair's count) on every single
    # iteration, so 64 real pairs (64 different M values) meant 64 separate
    # ~2s-ish compiles stacking up -- ran 25+ min and was killed; 20 pairs
    # finished in ~3 min. PASS B (fixed m_pad, ONE compile for the whole run)
    # is the only practical way to validate polish at any real scale --
    # confirmed: batch=64/n_hyp=100/m_pad=512 with polish hit 0.379ms/pair
    # (vs 0.353ms/pair without polish -- polish is nearly free once compiled)
    # at 96.4% cv2-agreement (vs 95.6% without polish) on the n-frames=40
    # test. PASS A here is just a quick correctness spot-check on a handful
    # of pairs, deliberately kept small -- not where polish's benefit or its
    # true (fixed-shape) speed should be judged.
    key = jax.random.PRNGKey(0)
    agree_a, n_a = 0, 0
    inliers_a, cv2_inliers_a = 0, 0
    n_pass_a = min(args.pass_a_pairs, N_TEST)
    for pair_idx in range(n_pass_a):
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
            inlier, _ = relpose_ransac(pa, pb, sub, args.n_hyp, theta_tol_ours, args.max_iters)
            inlier = jax.block_until_ready(inlier)
            g = np.asarray(inlier[0])
        agree_a += (c == g).sum()
        n_a += len(c)
        inliers_a += g.sum()
        cv2_inliers_a += c.sum()
    print(f"\n[PASS A: algorithm correctness, full points, no truncation, B=1]")
    print(f"(NOTE: B=1 timing includes per-call jaxls compile overhead -- not "
          f"representative of batched speed; accuracy is the point of this pass)")
    if n_a > 0:
        print(f"inlier-decision agreement: {agree_a}/{n_a} ({100*agree_a/n_a:.1f}%)")
        print(f"cv2 total inliers: {cv2_inliers_a}, ours total inliers: {inliers_a}")
    else:
        print("(skipped: --pass-a-pairs 0)")

    # ---------------- PASS B: realistic batched production speed -----------
    # fixed (batch_size, n_hyp, m_pad) across ALL batches for the whole run --
    # required for sample_minimal_points/select_best to hit their compilation
    # cache too (solve_5pt_batch already never retraces since it only depends
    # on N=B*n_hyp). Every batch is padded/truncated to the SAME m_pad, with
    # a validity mask marking real vs. padded points, instead of the earlier
    # truncate-to-batch-minimum scheme (which let M drift batch to batch and
    # forced sample_minimal_points/select_best to retrace every time -- see
    # module docstring's "CRITICAL gotcha").
    order = sorted(range(N_TEST), key=lambda i: rays_by_pair_side[(i, "a")].shape[0])
    B = args.batch_size
    Mp = args.m_pad

    batches = []          # (batch_idx, valid_counts, pts_a, pts_b, mask)
    for bi in range(0, N_TEST, B):
        batch_idx = order[bi:bi + B]
        if len(batch_idx) < B:
            break  # drop the ragged last batch -- fixed shape only
        pa_list, pb_list, mask_list, valid_counts = [], [], [], []
        skip = False
        for i in batch_idx:
            ra = rays_by_pair_side[(i, "a")]
            rb = rays_by_pair_side[(i, "b")]
            if ra.shape[0] < 8:
                skip = True
                break
            pa = (ra[:, :2] / ra[:, 2:3])[:Mp].astype(np.float32)
            pb = (rb[:, :2] / rb[:, 2:3])[:Mp].astype(np.float32)
            valid_n = pa.shape[0]
            if valid_n < Mp:
                pad = np.zeros((Mp - valid_n, 2), dtype=np.float32)
                pa = np.concatenate([pa, pad], axis=0)
                pb = np.concatenate([pb, pad], axis=0)
            mask = np.zeros(Mp, dtype=bool)
            mask[:valid_n] = True
            pa_list.append(pa); pb_list.append(pb); mask_list.append(mask)
            valid_counts.append(valid_n)
        if skip:
            continue
        pts_a_t = jnp.asarray(np.stack(pa_list))
        pts_b_t = jnp.asarray(np.stack(pb_list))
        mask_t = jnp.asarray(np.stack(mask_list))
        batches.append((batch_idx, valid_counts, pts_a_t, pts_b_t, mask_t))

    # ---- EXPLICIT warmup: trace/compile using the FIRST batch's exact
    # (fixed, from now on ALWAYS this) shape before any timing starts. ----
    key = jax.random.PRNGKey(1)
    if batches:
        _, _, warm_a, warm_b, warm_mask = batches[0]
        key, sub = jax.random.split(key)
        warm_out, _ = relpose_ransac(warm_a, warm_b, sub, args.n_hyp, theta_tol_ours,
                                      args.max_iters, valid_mask=warm_mask)
        jax.block_until_ready(warm_out)
        print(f"\n[warmup done: m_pad={Mp}, shape {warm_a.shape} -- EVERY batch "
              f"from here on uses this exact shape, so this is the ONLY compile]")

    gpu_inlier_by_pair = {}
    t_gpu_total = 0.0
    n_timed = 0
    for batch_idx, valid_counts, pts_a_t, pts_b_t, mask_t in batches:
        key, sub = jax.random.split(key)
        t0 = time.time()
        inlier, E = relpose_ransac(pts_a_t, pts_b_t, sub, args.n_hyp, theta_tol_ours,
                                    args.max_iters, valid_mask=mask_t)
        inlier = jax.block_until_ready(inlier)
        dt = time.time() - t0
        print(f"  batch n_pairs={len(batch_idx):3d}: {dt*1000:8.1f}ms "
              f"({1000*dt/len(batch_idx):.3f}ms/pair)")
        t_gpu_total += dt
        n_timed += len(batch_idx)
        inlier_np = np.asarray(inlier)
        for row, i in enumerate(batch_idx):
            gpu_inlier_by_pair[i] = (inlier_np[row], valid_counts[row])

    print(f"\n[PASS B: realistic batched production speed (fixed batch={B}, "
          f"n_hyp={args.n_hyp}, m_pad={Mp})]")
    if n_timed > 0:
        print(f"ours (GPU batched, post-warmup, {n_timed} pairs): "
              f"{t_gpu_total*1000:.1f}ms total ({1000*t_gpu_total/n_timed:.3f}ms/pair)")

    agree_total, n_total = 0, 0
    cv2_inlier_total, gpu_inlier_total = 0, 0
    for i, (g, valid_n) in gpu_inlier_by_pair.items():
        c = cv2_results[i]
        n = min(len(c), valid_n)  # only compare over the REAL (non-padded) points
        agree_total += (c[:n] == g[:n]).sum()
        n_total += n
        cv2_inlier_total += c.sum()
        gpu_inlier_total += g[:n].sum()
    print(f"inlier-decision agreement (real points only, padding excluded): "
          f"{agree_total}/{n_total} ({100*agree_total/max(n_total,1):.1f}%)")
    print(f"cv2 total inliers (full): {cv2_inlier_total}, ours total inliers: {gpu_inlier_total}")
