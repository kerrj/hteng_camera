"""Synthetic tests for proximity loop proposal and depth verification."""

import numpy as np

import vio_vggt_window_infer as infer


def make_pose(center):
    pose = np.zeros(7, np.float64)
    pose[0] = 1.0
    pose[4:] = -np.asarray(center)
    return pose


def test_proximity_view_pairs_form_loop_region():
    frames = np.asarray([0, 10, 20, 100, 110, 120], np.int64)
    poses = np.stack([
        make_pose([0.00, 0.0, 0.0]),
        make_pose([0.10, 0.0, 0.0]),
        make_pose([0.20, 0.0, 0.0]),
        make_pose([0.02, 0.0, 0.0]),
        make_pose([0.12, 0.0, 0.0]),
        make_pose([0.22, 0.0, 0.0]),
    ])
    pairs = infer.proximity_loop_pairs(
        frames,
        poses,
        max_distance=0.15,
        min_gap=50,
        max_view_angle_deg=30.0,
    )
    assert pairs
    assert all(b - a >= 50 for a, b, _ in pairs)
    peaks = infer.rank_loop_pair_peaks(
        pairs, radius=15, min_votes=3, nms_radius=30, max_candidates=5)
    assert len(peaks) == 1
    assert peaks[0]["n_votes"] >= 3


def test_depth_reprojection_overlap():
    height, width = 32, 32
    depth = np.full((height, width), 2.0, np.float32)
    confidence = np.ones((height, width), np.float32)
    intrinsics = np.array([
        [20.0, 0.0, 15.5],
        [0.0, 20.0, 15.5],
        [0.0, 0.0, 1.0],
    ])
    extrinsic = np.eye(4)[:3]
    overlap = infer.depth_reprojection_fraction(
        depth, confidence, intrinsics, extrinsic,
        depth, confidence, intrinsics, extrinsic,
        pixel_stride=4,
        confidence_quantile=0.5,
        relative_tolerance=0.05,
    )
    assert overlap == 1.0

    inconsistent = infer.depth_reprojection_fraction(
        depth, confidence, intrinsics, extrinsic,
        depth * 2.0, confidence, intrinsics, extrinsic,
        pixel_stride=4,
        confidence_quantile=0.5,
        relative_tolerance=0.05,
    )
    assert inconsistent == 0.0
