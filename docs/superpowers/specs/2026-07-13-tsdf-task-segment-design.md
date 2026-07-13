# TSDF Task-Segment Reconstruction (PoC) — Design

**Date:** 2026-07-13
**Branch:** `depth_understanding`
**Status:** Approved (design), pending implementation

## Problem / motivation

The accumulated point-cloud worlds (ffs_fuse_world.py) contain "points in
empty space": per-frame stereo depth noise is additive under naive point
accumulation, far-field depth (70.8 mm baseline) is fuzzy, and dynamic
content (hands) smears into the static geometry. For egocentric
manipulation the target regime is a SHORT, LOW-TRANSLATION, task-focused
segment — near-field (stereo's sweet spot), single-window pose quality,
and small enough to fuse volumetrically.

## Goal (PoC)

On one auto-selected ~30 s task segment of long-test2, produce a clean
static workspace **mesh** via TSDF fusion with hand masking, compared
side-by-side against the naive point accumulation of the same inputs, and
viewable in viser with the animated MANO hand surfaces composited on top.

## Approach (chosen: A — TSDF on existing artifacts)

Reuse: refined trajectory (`long-test2/derived/trajectory.npz`, 0.328°),
range maps (`data_processing/out/lt2_video`), MANO fits
(`long-test2/derived/hands3d_{left,right}.jsonl` → per-frame 778-vert
meshes via the existing precompute path). Only the fusion method is new —
isolating the variable under test. (Rejected for the PoC: B — re-solve
poses/depth at max quality first — conflates fusion gains with input
gains; it is the follow-up if the mesh is visibly depth-limited.)

## Components

New file `data_processing/ffs_tsdf_segment.py` with subcommands/phases:

1. **Segment selection**: slide a 900-frame window over the refined
   trajectory; score = camera-center extent (prefer < ~1 m), mean gyro
   rate (imu_relative.npz rel_quat), hand-fit presence fraction
   (hands3d jsonls). Pick the calmest hand-active window; print chosen
   frame range + timestamp for user sanity check. Overridable via
   `--start/--end`.
2. **Per-frame depth prep** (calm frames only, gyro rate below
   `--max-rot-dps`, default ~20°/s): load range map → clamp to
   `--max-range` (default 2 m) → mask hands: project both MANO meshes
   into the fisheye (FP.fisheye_project), splat verts + dilate
   (`--mask-dilate-px`), zero masked range pixels.
3. **Virtual pinhole rendering**: Open3D TSDF ingests pinhole RGBD only;
   our range maps are fisheye. Per frame, min-splat the masked points
   into a forward-facing virtual pinhole depth image (~110° hfov,
   ~800 px) + matching color image sampled from the video at the same
   fisheye pixels. Virtual cam shares the left-cam pose (identity
   relative rotation) so extrinsic = T_wl.
4. **TSDF integration**: `o3d.pipelines.integration.ScalableTSDFVolume`,
   `--voxel` default 5 mm, sdf_trunc ~4× voxel; integrate every prepared
   frame; `extract_triangle_mesh()` → `out/<segment>_tsdf_mesh.ply`.
5. **Baseline**: `ffs_fuse_world.py` gains `--start-frame/--end-frame`
   (segment bounds, same stride/frames) → same-segment naive cloud for
   the side-by-side.
6. **Viewer**: viser scene = static TSDF mesh + animated per-frame MANO
   hand surfaces (persistent-handle pattern from ffs_scene_player) in the
   world frame; `--share` link.

## Acceptance criteria

1. Chosen segment is visibly a manipulation moment (hands present most
   frames, low camera motion).
2. Hands absent from the static mesh (no hand-shaped blobs).
3. TSDF mesh visibly cleaner than the naive same-segment cloud
   (side-by-side render + viser).
4. Mesh + animated hands viewable together in viser via share link.

## Constraints

- All work on `depth_understanding`; nothing to main.
- Runs entirely in the `eyeball` env (no BA; poses already solved).
- Recording data stays gitignored; outputs under `data_processing/out/`.
