import numpy as np

import vio_stitch as ST


def rand_poses(n, seed):
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    q[q[:, 0] < 0] *= -1
    return np.concatenate([q, rng.normal(size=(n, 3))], axis=1)


def gauge_b_of(poses_a, theta, t):
    """Re-express gauge-a poses in gauge b, where X_a = Rz(theta) X_b + t.
    The inverse map is X_b = Rz(-theta) X_a - Rz(-theta) t."""
    return ST.apply_yaw_translation(poses_a, -theta, -(ST.yaw_R(-theta) @ t))


def test_quat_mul_matches_matrix_product():
    a, b = rand_poses(8, 0), rand_poses(8, 1)
    R_ab = ST.quat_to_R(ST.quat_mul(a[:, :4], b[:, :4]))
    assert np.allclose(R_ab, ST.quat_to_R(a[:, :4]) @ ST.quat_to_R(b[:, :4]),
                       atol=1e-12)


def test_fit_recovers_known_gauge():
    poses_a = rand_poses(50, 2)
    theta_true, t_true = 1.234, np.array([0.5, -2.0, 0.3])
    poses_b = gauge_b_of(poses_a, theta_true, t_true)
    theta, t, diag = ST.fit_yaw_translation(poses_a, poses_b)
    assert abs(theta - theta_true) < 1e-9
    assert np.allclose(t, t_true, atol=1e-9)
    assert diag["center_rms_m"] < 1e-9
    assert diag["yaw_spread_deg"] < 1e-6
    assert diag["n_shared"] == 50


def test_fit_near_pi_wraparound_with_noise():
    rng = np.random.default_rng(4)
    poses_a = rand_poses(200, 5)
    theta_true, t_true = np.pi - 0.01, np.array([3.0, 0.0, 1.0])
    poses_b = gauge_b_of(poses_a, theta_true, t_true)
    poses_b[:, 4:] += rng.normal(scale=1e-3, size=(200, 3))  # mm noise on t
    theta, t, _ = ST.fit_yaw_translation(poses_a, poses_b)
    assert abs((theta - theta_true + np.pi) % (2 * np.pi) - np.pi) < 1e-3
    assert np.allclose(t, t_true, atol=5e-3)


def test_apply_roundtrip():
    poses_a = rand_poses(20, 3)
    theta, t = -0.7, np.array([1.0, 2.0, -0.5])
    poses_b = gauge_b_of(poses_a, theta, t)
    back = ST.apply_yaw_translation(poses_b, theta, t)
    sign = np.sign((poses_a[:, :4] * back[:, :4]).sum(1, keepdims=True))
    assert np.allclose(poses_a[:, :4], back[:, :4] * sign, atol=1e-9)
    assert np.allclose(poses_a[:, 4:], back[:, 4:], atol=1e-9)


def test_compose_equals_sequential_apply():
    poses = rand_poses(15, 8)
    th1, t1 = 0.4, np.array([1.0, -1.0, 0.2])
    th2, t2 = -1.1, np.array([0.3, 2.0, -0.7])
    seq = ST.apply_yaw_translation(ST.apply_yaw_translation(poses, th2, t2), th1, t1)
    thc, tc = ST.compose_yaw_translation(th1, t1, th2, t2)
    comp = ST.apply_yaw_translation(poses, thc, tc)
    assert np.allclose(ST.centers_of(seq), ST.centers_of(comp), atol=1e-9)
    assert np.allclose(ST.quat_to_R(seq[:, :4]), ST.quat_to_R(comp[:, :4]), atol=1e-9)


def test_blend_endpoints_and_midpoint():
    pa, pb = rand_poses(10, 6), rand_poses(10, 7)
    b0 = ST.blend_poses(pa, pb, np.zeros(10))
    b1 = ST.blend_poses(pa, pb, np.ones(10))
    assert np.allclose(b0, pa, atol=1e-12)
    assert np.allclose(ST.quat_to_R(b1[:, :4]), ST.quat_to_R(pb[:, :4]), atol=1e-12)
    assert np.allclose(ST.centers_of(b1), ST.centers_of(pb), atol=1e-12)
    mid = ST.blend_poses(pa, pb, np.full(10, 0.5))
    assert np.allclose(ST.centers_of(mid),
                       0.5 * (ST.centers_of(pa) + ST.centers_of(pb)), atol=1e-12)
