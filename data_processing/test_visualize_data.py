import numpy as np

import visualize_data as viewer


def horizontal_forward_rotation():
    # Camera +z points world +x; matrix columns are camera axes in world.
    return np.array([
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ])


def test_follow_camera_is_behind_and_gravity_aligned():
    center = np.array([1.0, 2.0, 3.0])
    position, look_at, up = viewer.presentation_camera_target(
        center, horizontal_forward_rotation(), "Follow")

    np.testing.assert_allclose(position, [0.25, 2.0, 3.24])
    np.testing.assert_allclose(look_at, [1.55, 2.0, 3.04])
    np.testing.assert_allclose(up, [0.0, 0.0, 1.0])


def test_ego_camera_looks_along_optical_axis():
    center = np.array([1.0, 2.0, 3.0])
    position, look_at, up = viewer.presentation_camera_target(
        center, horizontal_forward_rotation(), "Ego")

    np.testing.assert_allclose(position, [0.975, 2.0, 3.0])
    np.testing.assert_allclose(look_at, [2.0, 2.0, 3.0])
    np.testing.assert_allclose(up, [0.0, 0.0, 1.0])


def test_overview_camera_frames_nonzero_trajectory_extent():
    centers = np.array([
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 0.2],
        [1.0, 0.0, 0.4],
    ])
    position, look_at, up = viewer.overview_camera_target(centers)

    assert np.linalg.norm(position - look_at) >= 2.0
    np.testing.assert_allclose(look_at, np.median(centers, axis=0))
    np.testing.assert_allclose(up, [0.0, 0.0, 1.0])


def test_camera_smoothing_is_time_constant_based():
    previous = (
        np.zeros(3),
        np.zeros(3),
        np.array([0.0, 0.0, 1.0]),
    )
    target = (
        np.ones(3),
        np.full(3, 2.0),
        np.array([0.0, 0.0, 1.0]),
    )
    position, look_at, up = viewer.smooth_camera_target(
        previous, target, dt=0.22, time_constant=0.22)
    alpha = 1.0 - np.exp(-1.0)

    np.testing.assert_allclose(position, alpha)
    np.testing.assert_allclose(look_at, 2.0 * alpha)
    np.testing.assert_allclose(up, [0.0, 0.0, 1.0])


def test_presentation_path_smoothing_reduces_pose_jitter():
    times = np.arange(20, dtype=np.float64) * 0.025
    centers = np.zeros((20, 3), np.float64)
    centers[:, 0] = np.linspace(0.0, 1.0, 20)
    centers[:, 1] = 0.04 * (-1.0) ** np.arange(20)
    rotations = np.broadcast_to(
        horizontal_forward_rotation(), (20, 3, 3)).copy()
    rotations[:, 1, 2] += 0.08 * (-1.0) ** np.arange(20)

    smooth_centers, smooth_rotations = viewer.smooth_presentation_trajectory(
        centers, rotations, times, time_constant=0.2)

    assert np.std(smooth_centers[:, 1]) < np.std(centers[:, 1]) * 0.3
    assert np.std(smooth_rotations[:, 1, 2]) < 0.025
    np.testing.assert_allclose(
        np.linalg.norm(smooth_rotations[:, :, 2], axis=1), 1.0)


if __name__ == "__main__":
    test_follow_camera_is_behind_and_gravity_aligned()
    test_ego_camera_looks_along_optical_axis()
    test_overview_camera_frames_nonzero_trajectory_extent()
    test_camera_smoothing_is_time_constant_based()
    test_presentation_path_smoothing_reduces_pose_jitter()
    print("5 presentation viewer tests passed")
