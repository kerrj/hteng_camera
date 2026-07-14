import numpy as np

import ffs_tsdf_segment as TS


def test_pick_segment_prefers_calm_handy_window():
    n = 3000
    frame_idx = np.arange(n)
    # camera walks 5 m over frames 0..1000, then stands ~still 1000..2000, walks again
    centers = np.zeros((n, 3))
    centers[:1000, 0] = np.linspace(0, 5, 1000)
    centers[1000:2000, 0] = 5 + 0.01 * np.sin(np.linspace(0, 20, 1000))
    centers[2000:, 0] = np.linspace(5, 10, 1000)
    gyro = np.full(n - 1, 40.0)          # spinning...
    gyro[1000:2000] = 5.0                # ...except while standing
    hands = np.arange(900, 2100)         # hands present around the calm stretch
    s, e, rep = TS.pick_segment(centers, frame_idx, gyro, frame_idx[:-1], hands,
                                window=900, stride=30, min_hand_frac=0.7)
    assert 900 <= s <= 1200 and e == s + 900
    assert rep["hand_frac"] >= 0.7 and rep["mean_dps"] < 10


def test_pick_segment_rejects_handless_windows():
    n = 2000
    frame_idx = np.arange(n)
    centers = np.zeros((n, 3))
    gyro = np.full(n - 1, 5.0)
    hands = np.arange(0, 300)            # hands only at the very start
    s, e, rep = TS.pick_segment(centers, frame_idx, gyro, frame_idx[:-1], hands,
                                window=900, stride=30, min_hand_frac=0.7)
    # nothing satisfies the hand gate -> falls back to best hand_frac window
    assert rep["fallback"] is True and s == 0


def test_rasterize_hand_mask_dilates():
    px = np.array([[50.0, 50.0], [52.0, 50.0]])
    m = TS.rasterize_hand_mask(px, 100, 100, dilate_px=5)
    assert m[50, 50] and m[50, 55] and not m[50, 70]
    assert m.sum() > 50  # dilation grew the two seeds into a blob


def test_splat_zbuffer_nearest_wins():
    import torch
    fx, cx, cy, W, H = 100.0, 32, 32, 64, 64
    # two points hitting the same pixel: the nearer one must win
    P = torch.tensor([[0.0, 0.0, 2.0], [0.0, 0.0, 1.0]])
    C = torch.tensor([[255, 0, 0], [0, 255, 0]], dtype=torch.uint8)
    d, c = TS.splat_zbuffer_torch(P, C, fx, cx, cy, W, H)
    assert d[32, 32] == 1.0 and tuple(c[32, 32].tolist()) == (0, 255, 0)
    assert d[33, 33] == 1.0          # inside the 3x3 footprint
    assert d[30, 30] == 0.0          # outside it


def _random_pose(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    E = np.eye(4)
    E[:3, :3] = TS.quat_to_R(q)
    E[:3, 3] = rng.normal(size=3)
    return E


def test_R_to_quat_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(50):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        q2 = TS.R_to_quat(TS.quat_to_R(q))
        assert abs(np.dot(q, q2)) > 1 - 1e-9  # equal up to sign


def test_apply_world_correction_reobserves_moved_points():
    # ICP says the frame's world points X really sit at dT @ X. The corrected
    # camera must see dT @ X at the SAME camera coords the guess saw X at.
    rng = np.random.default_rng(1)
    for _ in range(20):
        ext, dT = _random_pose(rng), _random_pose(rng)
        X = np.append(rng.normal(size=3), 1.0)
        cam_before = ext @ X
        cam_after = TS.apply_world_correction(ext, dT) @ (dT @ X)
        assert np.allclose(cam_before, cam_after, atol=1e-10)


def test_cam_to_world_correction_matches_world_icp():
    # An ICP run in the guess camera's frame and one run in world frame
    # describe the same alignment: conjugation must map one to the other.
    rng = np.random.default_rng(2)
    for _ in range(20):
        ext, dT_c = _random_pose(rng), _random_pose(rng)
        X_w = np.append(rng.normal(size=3), 1.0)
        # camera-frame ICP moved the measured cam-frame point onto the model
        model_c = dT_c @ (ext @ X_w)
        # the world-frame correction must land on the same model point
        dT_w = TS.cam_to_world_correction(ext, dT_c)
        model_w = np.linalg.inv(ext) @ model_c
        assert np.allclose(dT_w @ X_w, model_w, atol=1e-10)
