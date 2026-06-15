import numpy as np
import cv2
import pytest

from vr_passthrough import kb_project


def _sample_dirs():
    # Rays spread across a wide fisheye field, all with positive z (in front).
    rng = np.random.default_rng(0)
    az = rng.uniform(-np.pi, np.pi, 200)
    theta = rng.uniform(0.0, np.deg2rad(85), 200)  # up to 85 deg off-axis
    x = np.sin(theta) * np.cos(az)
    y = np.sin(theta) * np.sin(az)
    z = np.cos(theta)
    return np.stack([x, y, z], axis=1).astype(np.float64)


def test_kb_project_matches_opencv():
    K = np.array([[800.0, 0, 960.0], [0, 800.0, 540.0], [0, 0, 1.0]])
    dist = np.array([-0.02, 0.004, -0.0008, 0.0001])  # k1..k4
    dirs = _sample_dirs()

    # OpenCV reference: project unit-depth 3D points through the fisheye model.
    obj = dirs.reshape(-1, 1, 3)
    ref, _ = cv2.fisheye.projectPoints(
        obj, np.zeros(3), np.zeros(3), K, dist.reshape(4, 1))
    ref = ref.reshape(-1, 2)

    ours = kb_project(dirs, K, dist)

    assert ours.shape == ref.shape
    assert np.allclose(ours, ref, atol=1e-3), np.abs(ours - ref).max()
