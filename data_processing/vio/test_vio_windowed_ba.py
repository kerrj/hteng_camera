import numpy as np
import pytest

import vio_stitch as ST
import vio_windowed_ba as WBA


def test_window_ranges_long_test2_shape():
    r = WBA.window_ranges(0, 11042, 900, 300)
    assert r[0][0] == 0 and r[-1][1] == 11042
    for (s0, e0), (s1, e1) in zip(r, r[1:]):
        assert s1 == s0 + 600          # stride = window - overlap
        assert e0 - s1 >= 300          # shared span >= overlap
    lens = [e - s for s, e in r]
    assert lens[:-1] == [900] * (len(r) - 1)
    assert 900 <= lens[-1] <= 900 + 600  # tail folded into final window
    assert len(r) == 17


def test_window_ranges_short_recording_single_window():
    assert WBA.window_ranges(0, 500, 900, 300) == [(0, 500)]


def test_window_ranges_rejects_bad_overlap():
    with pytest.raises(AssertionError):
        WBA.window_ranges(0, 5000, 900, 500)   # overlap > window/2


def _rand_poses(n, seed):
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return np.concatenate([q, rng.normal(size=(n, 3))], axis=1)


def test_merge_blend_agreeing_windows_pass_through():
    fi_a = np.arange(0, 10)
    pa = _rand_poses(10, 0)
    fi_b = np.arange(5, 15)
    pb = np.concatenate([pa[5:], _rand_poses(5, 1)])  # agrees on shared 5..9
    fi, poses = WBA.merge_blend([(fi_a, pa, None, None), (fi_b, pb, None, None)])
    assert np.array_equal(fi, np.arange(15))
    assert np.allclose(ST.quat_to_R(poses[5:10, :4]), ST.quat_to_R(pa[5:10, :4]),
                       atol=1e-9)
    assert np.allclose(ST.centers_of(poses[5:10]), ST.centers_of(pa[5:10]),
                       atol=1e-9)
    assert np.allclose(poses[:5], pa[:5]) and np.allclose(poses[10:], pb[5:])


def test_merge_blend_ramps_between_disagreeing_windows():
    fi_a, fi_b = np.arange(0, 10), np.arange(5, 15)
    pa, pb = _rand_poses(10, 2), _rand_poses(10, 3)
    fi, poses = WBA.merge_blend([(fi_a, pa, None, None), (fi_b, pb, None, None)])
    ca = ST.centers_of(pa[5:10])
    cb = ST.centers_of(pb[:5])
    cm = ST.centers_of(poses[5:10])
    w = np.linspace(0.0, 1.0, 7)[1:-1]
    assert np.allclose(cm, (1 - w[:, None]) * ca + w[:, None] * cb, atol=1e-9)
