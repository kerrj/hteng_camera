import numpy as np

import ffs_dynamic_residual as DR


def test_residual_mask_in_front_of_static():
    rng = np.array([[1.00, 1.00, 0.00],
                    [0.50, 1.00, 1.00]], np.float32)   # measured
    t_hit = np.array([[1.02, 1.20, 1.00],
                      [1.00, np.inf, np.inf]], np.float32)  # static
    m = DR.residual_mask(rng, t_hit, tau=0.03, near_orphan=0.8)
    assert not m[0, 0]        # 2 cm in front < tau -> static noise, not dynamic
    assert m[0, 1]            # 20 cm in front -> dynamic
    assert not m[0, 2]        # no measurement
    assert m[1, 0]            # 50 cm in front -> dynamic
    assert not m[1, 1]        # static miss, measured 1.0 >= near_orphan -> ignore
    assert not m[1, 2]


def test_residual_mask_near_orphan():
    rng = np.array([[0.5]], np.float32)
    t_hit = np.array([[np.inf]], np.float32)
    assert DR.residual_mask(rng, t_hit, 0.03, 0.8)[0, 0]   # close + no static -> dynamic


def test_despeckle_kills_specks_keeps_blobs():
    m = np.zeros((60, 60), bool)
    m[5, 5] = True                     # single-pixel speck
    m[20:40, 20:40] = True             # 400-px blob
    out = DR.despeckle(m, min_area=50)
    assert not out[5, 5]
    assert out[25:35, 25:35].all()
