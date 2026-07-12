# Windowed Bundle Adjustment for Long Recordings — Design

**Date:** 2026-07-12
**Branch:** `depth_understanding`
**Status:** Approved (design), pending implementation

## Problem

`vio/vio_bundle_adjust.py` converges to 0.2–0.6° median angular residual on
short spans (testimu 896 frames → 0.23°, long-test2 300-frame window → 0.37°,
30 s window → 0.61°) but plateaus at 8–9° on full-length long-test2 (11042
frames, ~6 min). Loop closure barely helps (9.35° → 8.15°).

Root cause (diagnosed, not speculative): stage 1 freezes rotations from the
integrated IMU gyro chain seeded at frame 0. Gyro drift over 6 minutes puts
those frozen rotations far enough from truth that stage 1 positions organize
around a wrong scaffold, and stage 2's local refine cannot escape the basin.
The solver is fine; the *init* is the problem, and its error grows with
recording length.

## Goal

A reusable pipeline stage that produces a converged full-length
`trajectory.npz` for arbitrarily long recordings, byte-compatible with the
existing schema so `visualize_data.py` and `ffs_fuse_world.py` work unchanged.
Immediate consumer: dense world fusion of long-test2 (+ hands overlay).

## Approach (chosen: A — windowed BA → 4-DOF stitch → global refine)

Considered:

- **A. Windowed BA + stitch + global refine (chosen).** Reuses the validated
  solver; windows parallelize across GPUs; refine restores global consistency
  and composes with loop closure; degrades gracefully (stitch-only is a
  usable trajectory).
- **B. Windowed + stitch only.** Subset of A (`--no-refine`); stitch errors
  compound over ~18 seams and nothing enforces loop consistency.
- **C. Vision rotation averaging (GLOMAP-style) to replace the IMU-chain
  init.** Most principled fix of the root cause, but a substantial new
  component (two-view relative rotations on fisheye rays + robust averaging);
  future option if A's refine proves insufficient.

Key insight making A cheap: each window solve re-seeds the IMU rotation chain
gravity-aligned at the window's first frame, so gyro drift resets every
window — putting every window in the regime where the solver is proven to
converge. And per-window gauge freedom is exactly **yaw + translation**
(gravity fixes roll/pitch, stereo baseline fixes scale), so stitching is a
4-DOF closed-form alignment, not an optimization.

## Architecture

Two files touched, one new:

### `vio/vio_bundle_adjust.py` (modify — two new args, nothing else)

- `--start-frame S`: restrict the solve to frames in `[S, S + n_frames]`
  (composes with existing `--n-frames`; bounds are video frame numbers
  matched against `frame_idx`). Implementation: extend the existing
  `--n-frames` keep-mask; `load_tracks` gains a `min_frame` bound alongside
  the existing max; tracks with <2 in-window observations drop (same rule as
  today). The IMU chain seeds gravity-aligned at index 0 of whatever range
  survives — already the code's behavior, no change needed.
- `--init-trajectory <npz>`: take stage-1's frozen rotations and center
  inits from a prior trajectory file instead of the gyro chain + random
  centers. Landmarks/scales stay randomly initialized (proven to converge
  under good rotations). Frame alignment: assert the npz's `frame_idx`
  matches the solve's surviving frame list exactly.

Rationale for flags-on-the-existing-script over an importable-core refactor:
the solver stays the single source of truth, jax gets one process per GPU,
and the change surface on the solver is minimal.

### `vio/vio_windowed_ba.py` (new — the stage/orchestrator)

Pipeline conventions: positional `recording` arg; inputs default from
`<recording>/derived/` (`tracks.jsonl`, `imu_relative.npz`, `features.h5`);
final output `derived/trajectory.npz`.

1. **Window**: defaults `--window-frames 900 --overlap-frames 300` (stride
   600; long-test2 → 18 windows). Bounds in video frame numbers. Last window
   extends to the recording end (no runt shorter than the overlap).
2. **Fan out**: one subprocess per window,
   `vio_bundle_adjust.py --start-frame … --out derived/windows/window_<s>_<e>.npz`,
   `CUDA_VISIBLE_DEVICES` pinned, one window per GPU at a time, waves until
   done (same launcher pattern as `ffs_scene_batch.py`). Subprocess env gets
   `XLA_PYTHON_CLIENT_PREALLOCATE=false`.
3. **Stitch** (see below) → `derived/trajectory_stitched.npz`.
4. **Global refine** (unless `--no-refine`): full-length
   `vio_bundle_adjust.py --init-trajectory derived/trajectory_stitched.npz`
   → final `derived/trajectory.npz`.

## Stitching (4-DOF closed form) & blending

Consecutive windows A, B share ~300 solved frames. `X_A = Rz(θ) X_B + t`.

- **Yaw**: per shared frame f, `M_f = (R_f^A)ᵀ R_f^B` (poses are world→cam;
  same camera, two world gauges) ≈ pure z-rotation. Vote
  `θ_f = atan2(M₁₀−M₀₁, M₀₀+M₁₁)`; θ = circular mean over the overlap.
- **Translation**: `t = mean_f (c_f^A − Rz c_f^B)` over camera centers
  `c = −Rᵀ t_pose`.
- **Chain**: `T₀ = I`, `T_k = T_{k−1} ∘ T_{k−1→k}` — all windows into
  window-0's world.
- **Blend**: in each overlap, linear ramp weight across the overlap;
  position lerp + quaternion nlerp (sign-aligned). Post-alignment
  disagreement should be mm-scale.
- **Per-seam diagnostics (printed)**: yaw-vote spread (deg) and RMS center
  disagreement (m) after alignment. Assert ≥ 30 shared surviving frames per
  seam, else fail loudly.

## Global refine

The existing two-stage solve over all frames, initialized from the stitched
trajectory via `--init-trajectory`:

- Stage 1 identical except frozen rotations + center init come from the
  stitched npz (not the 6-min gyro chain / random).
- Stage 2 polishes full SE3 with IMU relative-rotation + gravity costs —
  irons out stitch seams, distributes error globally.
- Loop closure composes for free: pass `--tracks derived/tracks_loop.jsonl`
  to the refine run (already built for long-test2).

## Outputs

- `derived/windows/window_<s>_<e>.npz` — per-window trajectories (existing
  schema), kept for debugging.
- `derived/trajectory_stitched.npz` — blended full-length poses + all
  windows' landmarks transformed into world-0 (viewable in
  `visualize_data.py` even with `--no-refine`).
- `derived/trajectory.npz` — final refined output, existing schema.

## Acceptance criteria

1. Every window ≤ ~1° median angular residual (runner flags violators).
2. Seam diagnostics sub-degree yaw spread / sub-cm center RMS.
3. Final full-length median residual ≪ the 8–9° baseline (expect < 1°).
4. The real test: `ffs_fuse_world.py` on the refined long-test2 trajectory
   produces a recognizable dense world in viser (quality comparable to the
   testimu result), with hands overlayable.

## Edge cases

- Invalid-timestamp frames: already dropped inside the solver; overlap
  intersection operates on surviving `frame_idx` values.
- Thin overlaps (dropped frames): assert ≥ 30 shared frames per seam.
- Blurry/degenerate window: per-window residual report flags it loudly
  rather than silently poisoning the stitch.
- jax GPU memory: one solve process per GPU; preallocation disabled.

## Constraints

- All work stays on `depth_understanding`; nothing is pushed to or merged
  into `main`.
- Recording data (`long-test1/`, `long-test2/`) stays gitignored.
