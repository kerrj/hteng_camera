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
