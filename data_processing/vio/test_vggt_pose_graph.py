"""Synthetic convention and metric-scale test for vio_vggt_pose_graph."""

from types import SimpleNamespace

import numpy as np

import vio_vggt_window_infer as infer
import vio_vggt_pose_graph as graph


def make_extrinsic(center):
    output = np.eye(4, dtype=np.float64)
    output[:3, 3] = -np.asarray(center)
    return output


def make_window(frames, segment, model_units_per_meter, baseline):
    extrinsics = []
    for frame in frames:
        center_left = np.array([0.2 * frame, 0.0, 0.0])
        center_right = center_left + np.array([baseline, 0.0, 0.0])
        extrinsics.extend([
            make_extrinsic(center_left * model_units_per_meter),
            make_extrinsic(center_right * model_units_per_meter),
        ])
    return {
        "path": "synthetic",
        "kind": "loop" if len(set(segment)) > 1 else "track",
        "frame_idx": np.asarray(frames, np.int64),
        "segment": np.asarray(segment, np.int8),
        "extrinsics": np.asarray(extrinsics),
    }


def make_sparse_window(frames, model_units_per_meter, baseline, right_frames):
    extrinsics = []
    image_frame_idx = []
    image_eye = []
    for frame in frames:
        center_left = np.array([0.2 * frame, 0.0, 0.0])
        extrinsics.append(make_extrinsic(
            center_left * model_units_per_meter))
        image_frame_idx.append(frame)
        image_eye.append(0)
        if frame in right_frames:
            center_right = center_left + np.array([baseline, 0.0, 0.0])
            extrinsics.append(make_extrinsic(
                center_right * model_units_per_meter))
            image_frame_idx.append(frame)
            image_eye.append(1)
    return {
        "path": "sparse-synthetic",
        "kind": "track",
        "frame_idx": np.asarray(frames, np.int64),
        "segment": np.zeros(len(frames), np.int8),
        "extrinsics": np.asarray(extrinsics),
        "image_frame_idx": np.asarray(image_frame_idx, np.int64),
        "image_eye": np.asarray(image_eye, np.int8),
    }


def test_fractional_frame_selection(tmp_path):
    selected = infer.choose_frames(
        str(tmp_path),
        n_frames=30,
        native_fps=30.0,
        target_fps=12.0,
        max_frames=None,
    )
    np.testing.assert_array_equal(
        selected, [0, 2, 5, 8, 10, 12, 15, 18, 20, 22, 25, 28])


def test_sparse_stereo_inputs_include_segment_endpoints():
    manifest = {
        "target_fps": 15.0,
        "frames": [
            {"frame": frame, "left": f"l{frame}", "right": f"r{frame}"}
            for frame in range(20)
        ],
    }
    spec = {
        "frames": manifest["frames"][:5] + manifest["frames"][10:15],
        "segments": [0] * 5 + [1] * 5,
    }
    frames, eyes, segments = infer.make_window_image_inputs(
        spec, manifest, stereo_interval_seconds=1.0)

    right_mask = eyes == 1
    np.testing.assert_array_equal(frames[right_mask], [0, 4, 10, 14])
    np.testing.assert_array_equal(segments[right_mask], [0, 0, 1, 1])
    assert len(frames) == 14


def test_window_cache_validates_frame_identity(tmp_path):
    spec = {
        "name": "loop_00000",
        "kind": "loop",
        "frames": [{"frame": 10}, {"frame": 20}],
    }
    out_path = tmp_path / "loop_00000.npz"
    reject_path = tmp_path / "loop_00000.rejected.json"
    np.savez(out_path, kind=np.asarray("loop"), frame_idx=np.asarray([10, 20]))
    assert infer.window_cache_matches(out_path, reject_path, spec)

    changed = {**spec, "frames": [{"frame": 11}, {"frame": 21}]}
    assert not infer.window_cache_matches(out_path, reject_path, changed)


def test_loop_cache_prunes_candidates_no_longer_selected(tmp_path):
    keep = tmp_path / "loop_00000.npz"
    stale = tmp_path / "loop_00001.npz"
    rejected = tmp_path / "loop_00002.rejected.json"
    for path in (keep, stale, rejected):
        path.touch()
    specs = [{"name": "loop_00000", "kind": "loop"}]

    removed = infer.prune_stale_loop_cache(tmp_path, specs)

    assert keep.exists()
    assert {path.name for path in removed} == {
        stale.name, rejected.name}


def test_sparse_stereo_baseline_recovery():
    baseline = 0.07
    window = make_sparse_window(
        frames=[0, 1, 2, 3],
        model_units_per_meter=2.0,
        baseline=baseline,
        right_frames={0, 3},
    )
    measurements = graph.build_measurements([window], baseline)
    np.testing.assert_allclose(measurements["predicted_baselines"], [0.14])
    np.testing.assert_allclose(
        measurements["initial_log_scales"], [np.log(0.5)])
    assert len(measurements["edges"]) == 4


def test_loop_window_adds_one_cross_edge_only():
    window = make_window(
        frames=[0, 1, 10, 11],
        segment=[0, 0, 1, 1],
        model_units_per_meter=1.0,
        baseline=0.07,
    )
    assert graph.edge_pairs(window) == [(1, 3, "loop")]


def test_full_rate_pose_interpolation_preserves_keyframes():
    frame_idx = np.asarray([0, 2, 4, 6], np.int64)
    angles = np.deg2rad([0.0, 20.0, 40.0, 60.0])
    poses = np.zeros((len(frame_idx), 7), np.float64)
    poses[:, 0] = np.cos(angles / 2.0)
    poses[:, 3] = np.sin(angles / 2.0)
    centers = np.stack([
        0.1 * frame_idx,
        0.01 * frame_idx ** 2,
        np.zeros_like(frame_idx),
    ], axis=1)
    poses[:, 4:] = -np.einsum(
        "nij,nj->ni", graph.quat_to_matrix(poses[:, :4]), centers)

    target_frames = np.arange(8)
    full_frames, full_poses = graph.interpolate_poses(
        frame_idx,
        poses,
        native_fps=30.0,
        target_frame_idx=target_frames,
    )

    np.testing.assert_array_equal(full_frames, target_frames)
    np.testing.assert_allclose(full_poses[frame_idx], poses, atol=1e-10)
    full_centers = -np.einsum(
        "nji,nj->ni",
        graph.quat_to_matrix(full_poses[:, :4]),
        full_poses[:, 4:],
    )
    np.testing.assert_allclose(
        full_centers[:, 0], 0.1 * full_frames, atol=1e-10)
    np.testing.assert_allclose(
        full_centers[:, 1], 0.01 * full_frames ** 2, atol=1e-10)


def test_pose_interpolation_uses_observed_frame_times():
    frame_idx = np.asarray([0, 2, 4], np.int64)
    target_frames = np.arange(5)
    target_times = np.asarray([0.0, 0.04, 0.11, 0.15, 0.30])
    key_times = target_times[frame_idx]
    poses = np.zeros((3, 7), np.float64)
    poses[:, 0] = 1.0
    poses[:, 4] = -key_times

    _, full_poses = graph.interpolate_poses(
        frame_idx,
        poses,
        target_frame_idx=target_frames,
        frame_times=key_times,
        target_frame_times=target_times,
    )

    np.testing.assert_allclose(full_poses[:, 4], -target_times, atol=1e-10)


def test_repaired_frame_times_preserve_gaps_and_repair_reset():
    frames = np.arange(5)
    raw_time_us = np.asarray([1000, 26000, 100, 50100, 75100])
    times = graph.repaired_frame_times(frames, raw_time_us)

    # Positive observed intervals are 25, 50, and 25 ms; the reset interval
    # is replaced by their 25 ms median.
    np.testing.assert_allclose(times, [0.0, 0.025, 0.050, 0.100, 0.125])
    np.testing.assert_allclose(
        graph.frame_times_at(frames, times, [0, 2, 4]),
        [0.0, 0.050, 0.125],
    )


def test_composed_stereo_poses_preserve_calibrated_baseline():
    angles = np.deg2rad([0.0, 30.0, 75.0])
    left = np.zeros((len(angles), 7), np.float64)
    left[:, 0] = np.cos(angles / 2.0)
    left[:, 2] = np.sin(angles / 2.0)
    centers_left = np.asarray([
        [0.0, 0.0, 0.0],
        [1.0, 2.0, 3.0],
        [-4.0, 1.0, 0.5],
    ])
    left[:, 4:] = -np.einsum(
        "nij,nj->ni", graph.quat_to_matrix(left[:, :4]), centers_left)
    stereo_angle = np.deg2rad(2.0)
    R_stereo = np.asarray([
        [np.cos(stereo_angle), 0.0, np.sin(stereo_angle)],
        [0.0, 1.0, 0.0],
        [-np.sin(stereo_angle), 0.0, np.cos(stereo_angle)],
    ])
    t_stereo = np.asarray([-0.066, 0.002, 0.001])

    right = graph.compose_stereo_poses(left, R_stereo, t_stereo)
    centers_right = -np.einsum(
        "nji,nj->ni",
        graph.quat_to_matrix(right[:, :4]),
        right[:, 4:],
    )

    np.testing.assert_allclose(
        np.linalg.norm(centers_right - centers_left, axis=1),
        np.linalg.norm(t_stereo),
        atol=1e-12,
    )
    for left_pose, right_pose in zip(left, right):
        np.testing.assert_allclose(
            graph.pose_to_matrix(right_pose),
            np.block([
                [R_stereo, t_stereo[:, None]],
                [np.zeros((1, 3)), np.ones((1, 1))],
            ]) @ graph.pose_to_matrix(left_pose),
            atol=1e-12,
        )


def test_stereo_window_scales_and_pose_graph():
    baseline = 0.07
    windows = [
        make_window([0, 1, 2, 3], [0, 0, 0, 0], 2.0, baseline),
        make_window([2, 3, 4, 5], [0, 0, 0, 0], 0.5, baseline),
        make_window([0, 1, 4, 5], [0, 0, 1, 1], 1.25, baseline),
    ]
    imu = {
        "frame_idx": np.arange(6, dtype=np.int64),
        "frame_valid": np.ones(6, bool),
        "rel_valid": np.ones(5, bool),
        "rel_quat": np.tile(
            np.array([1.0, 0.0, 0.0, 0.0]), (5, 1)),
        "gravity_cam": np.tile(
            np.array([0.0, 0.0, -1.0]), (6, 1)),
        "gravity_weight": np.ones(6),
    }
    args = SimpleNamespace(
        visual_translation_weight=10.0,
        visual_rotation_weight=10.0,
        visual_cauchy_scale=2.0,
        window_anchor_weight=1.0,
        baseline_weight=100.0,
        imu_rotation_weight=10.0,
        gravity_weight=1.0,
        gravity_accel_sigma_g=0.05,
        constant_velocity_weight=0.0,
        anchor_translation_weight=1.0,
        anchor_rotation_weight=0.01,
        linear_solver="dense_cholesky",
        iterations=8,
    )
    result = graph.solve_graph(
        args,
        windows,
        {"baseline_m": baseline, "native_fps": 10.0},
        imu,
    )
    scales = np.exp(result["window_log_scale"])
    np.testing.assert_allclose(scales, [0.5, 2.0, 0.8], rtol=2e-3)

    poses = result["poses"]
    centers = -np.einsum(
        "nji,nj->ni", graph.quat_to_matrix(poses[:, :4]), poses[:, 4:])
    np.testing.assert_allclose(
        centers[:, 0], 0.2 * np.arange(6), atol=2e-3)
    np.testing.assert_allclose(centers[:, 1:], 0.0, atol=2e-3)
