# Dynamic Residual Layer (TSDF PoC extension) — Design

**Date:** 2026-07-13 · **Branch:** `depth_understanding` · **Status:** approved in conversation

## Idea (user's, confirmed well-formed)

Static/dynamic decomposition: the TSDF mesh is the time-averaged static
world; the DYNAMIC content per frame is whatever the measured depth says is
**in front of** that static surface. Detection by reconstruction-vs-
observation disagreement — no hand model, no heuristics — so the
manipulated object (the phone) is captured rigorously, not by mask
dilation luck.

## Method

Per frame of the segment:
1. Raycast the static mesh (`o3d.t.geometry.RaycastingScene`) from that
   frame's pose along every fisheye pixel ray → static range `t_hit`.
2. Dynamic mask = measured range `rng` valid AND
   (`t_hit − rng > tau` (default 3 cm) OR (`t_hit` = miss AND `rng <
   near_orphan` (default 1.0 m))). The second clause catches hands in
   front of permanent static holes (regions occluded by hands in every
   frame never got reconstructed).
3. Despeckle: morphological open + drop connected components < min-area px.
4. Save per frame: world-frame points + RGB (ragged npz: concatenated
   arrays + per-frame offsets).

Viewer (`ffs_tsdf_viewer.py`) gains `--dynamic <npz>`: a persistent
point-cloud handle updated per frame, plus GUI toggles for MANO hands vs
residual points (dream composite: residual for objects, MANO for hands).

## Files

- New: `data_processing/ffs_dynamic_residual.py` (mesh + range maps +
  trajectory → `<prefix>_dynamic.npz`), pure helpers unit-tested:
  `residual_mask(rng, t_hit, tau, near_orphan)`, `despeckle(mask, min_area)`.
- Modify: `data_processing/ffs_tsdf_viewer.py` (`--dynamic`, toggles).

## Acceptance

1. Playback shows hands AND the phone moving over the clean static mesh.
2. Static surfaces contribute ~no residual points on typical frames
   (leakage = specks, not sheets).
3. Known caveat accepted: per-frame depth noise (~3–7 mm at hand range),
   shimmer, holes on the glossy phone screen.

## Out of scope (noted follow-ups)

Pass-2 static rebuild using the residual mask instead of MANO dilation;
temporal smoothing of the dynamic layer; object segmentation/tracking.
