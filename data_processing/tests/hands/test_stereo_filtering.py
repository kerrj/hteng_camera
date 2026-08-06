import numpy as np
import os
import tempfile

import stereo_optimize as SO


def test_quality_mask_rejects_invalid_geometry_and_bad_fit():
    metrics = {
        "mean_reproj_px": np.array([2.0, 2.0, 20.0, 2.0]),
        "p90_reproj_px": np.array([3.0, 3.0, 30.0, 3.0]),
        "median_epipolar_px": np.array([2.0, 2.0, 2.0, 20.0]),
        "min_joint_z_m": np.array([0.2, -0.1, 0.2, 0.2]),
        "depth_m": np.array([0.3, 0.3, 0.3, 0.3]),
        "root_fish": np.array([
            [0.0, 0.0, 0.3],
            [0.0, 0.0, 0.3],
            [0.0, 0.0, 0.3],
            [0.0, 0.0, 0.3],
        ]),
    }
    keep, rejected = SO.quality_mask(
        metrics, min_depth=0.08, max_depth=2.0, min_joint_z=0.02,
        max_reproj_px=15.0, max_epipolar_px=12.0)

    np.testing.assert_array_equal(keep, [True, False, False, False])
    assert rejected["joint_z"] == 1
    assert rejected["reprojection"] == 1
    assert rejected["epipolar"] == 1


def test_quaternion_matrix_round_trip_and_slerp():
    q0 = np.array([1.0, 0.0, 0.0, 0.0], np.float32)
    q1 = np.array([np.sqrt(0.5), 0.0, np.sqrt(0.5), 0.0], np.float32)

    recovered = SO.matrix_to_quat(SO.quat_to_matrix(q1))
    assert np.isclose(abs(np.dot(recovered, q1)), 1.0, atol=1e-6)
    np.testing.assert_allclose(SO.quat_slerp(q0, q1, 0.0), q0, atol=1e-6)
    np.testing.assert_allclose(SO.quat_slerp(q0, q1, 1.0), q1, atol=1e-6)

    midpoint = SO.quat_slerp(q0, q1, 0.5)
    expected = np.array([
        np.cos(np.pi / 8), 0.0, np.sin(np.pi / 8), 0.0
    ], np.float32)
    np.testing.assert_allclose(midpoint, expected, atol=1e-6)


def test_subset_data_rebuilds_only_consecutive_acceleration_triples():
    data = {
        "pose0": np.zeros((6, 48), np.float32),
        "constant": 256,
        "accel_i0": np.array([0, 1, 2, 3]),
        "accel_i1": np.array([1, 2, 3, 4]),
        "accel_i2": np.array([2, 3, 4, 5]),
    }
    keep = np.array([True, True, False, True, True, True])
    frames = [10, 11, 13, 14, 15]

    out = SO.subset_data(data, keep, frames)

    assert out["pose0"].shape == (5, 48)
    np.testing.assert_array_equal(out["accel_i0"], [2])
    np.testing.assert_array_equal(out["accel_i1"], [3])
    np.testing.assert_array_equal(out["accel_i2"], [4])


def test_acceleration_scales_use_observed_frame_intervals():
    data = {
        "frame_time_s": np.array([0.0, 0.025, 0.075]),
        "frame_time_reference_s": 0.025,
    }
    SO._set_accel_indices(data, [0, 1, 2])

    np.testing.assert_allclose(data["accel_prev_scale"], [1.0])
    np.testing.assert_allclose(data["accel_next_scale"], [0.5])


def test_load_vio_world_transforms_matches_frames_and_inverts_poses():
    angle = np.pi / 2
    poses = np.array([
        [1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0],
        [np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2),
         0.5, -0.25, 1.0],
    ])
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "trajectory.npz")
        np.savez(path, frame_idx=np.array([10, 20]), pose_wxyz_xyz=poses)
        R_cw, t_cw = SO.load_vio_world_transforms(path, [20, 10])

    R_wl_20 = SO.quat_to_matrix(poses[1, :4])
    np.testing.assert_allclose(R_cw[0], R_wl_20.T, atol=1e-6)
    np.testing.assert_allclose(t_cw[0], -R_wl_20.T @ poses[1, 4:], atol=1e-6)
    np.testing.assert_allclose(R_cw[1], np.eye(3), atol=1e-6)
    np.testing.assert_allclose(t_cw[1], -poses[0, 4:], atol=1e-6)


if __name__ == "__main__":
    test_quality_mask_rejects_invalid_geometry_and_bad_fit()
    test_quaternion_matrix_round_trip_and_slerp()
    test_subset_data_rebuilds_only_consecutive_acceleration_triples()
    test_acceleration_scales_use_observed_frame_intervals()
    test_load_vio_world_transforms_matches_frames_and_inverts_poses()
    print("5 stereo filtering tests passed")
