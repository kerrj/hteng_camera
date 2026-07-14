import numpy as np

from ffs_recolor_mesh import pick_sharp_frames


def test_pick_sharp_frames_spreads_over_segment():
    frames = np.arange(1000, 1900)
    # sharpest frames all cluster at the start; dps ramps up over the segment
    dps_of = lambda f: (f - 1000) * 0.05
    sel = pick_sharp_frames(frames, dps_of, k=60, buckets=20)
    # must NOT all come from the sharp start: every fifth of the segment
    # contributes at least one frame
    for lo in range(1000, 1900, 180):
        assert ((sel >= lo) & (sel < lo + 180)).any()
    assert len(sel) <= 66


def test_pick_sharp_frames_prefers_sharp_within_bucket():
    frames = np.arange(0, 100)
    dps = np.full(100, 50.0)
    dps[7] = 1.0                       # one sharp frame in the first bucket
    dps_of = lambda f: dps[f]
    sel = pick_sharp_frames(frames, dps_of, k=10, buckets=10)
    assert 7 in sel


def test_pick_sharp_frames_short_input_passthrough():
    frames = np.arange(5)
    sel = pick_sharp_frames(frames, lambda f: 1.0, k=60)
    assert np.array_equal(sel, frames)
