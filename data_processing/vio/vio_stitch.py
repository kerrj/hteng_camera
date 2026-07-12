"""4-DOF stitching math for windowed VIO trajectories.

Windowed BA solves (vio_bundle_adjust.py over a --start-frame range) share
gravity alignment (+z up: roll/pitch fixed by the gravity prior) and metric
scale (stereo baseline), so two windows' world gauges differ by exactly a yaw
about +z plus a translation:

    X_a = Rz(theta) @ X_b + t

Poses are (N,7) wxyz_xyz WORLD->CAMERA (jaxls SE3Var convention):
X_cam = R @ X_world + t_pose, camera center c = -R^T @ t_pose.
Pure numpy, no GPU.
"""
import numpy as np


def quat_to_R(q):
    """(...,4) wxyz -> (...,3,3)."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], -2)


def quat_mul(a, b):
    """Hamilton product (wxyz), broadcasting: R(quat_mul(a,b)) = R(a) @ R(b)."""
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw], -1)


def yaw_R(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def centers_of(poses):
    """(N,7) -> (N,3) camera centers c = -R^T t."""
    R = quat_to_R(poses[:, :4])
    return -np.einsum("nji,nj->ni", R, poses[:, 4:])


def fit_yaw_translation(poses_a, poses_b):
    """4-DOF gauge alignment from the SAME frames solved in two windows.

    Pose relation: X_cam = R_a X_a + t_a = R_b X_b + t_b with
    X_a = Rz X_b + t  =>  R_b = R_a Rz, so each shared frame votes
    Rz ~ R_a^T R_b (nearly pure z-rotation since both gauges are
    gravity-aligned). theta = circular mean of the votes; then
    t = mean(c_a - Rz c_b) over camera centers.

    Returns (theta, t, diag): diag has yaw_deg, yaw_spread_deg (vote std),
    center_rms_m (post-alignment center disagreement), n_shared.
    """
    Ra = quat_to_R(poses_a[:, :4])
    Rb = quat_to_R(poses_b[:, :4])
    M = np.einsum("nji,njk->nik", Ra, Rb)  # R_a^T R_b
    votes = np.arctan2(M[:, 1, 0] - M[:, 0, 1], M[:, 0, 0] + M[:, 1, 1])
    theta = float(np.arctan2(np.sin(votes).mean(), np.cos(votes).mean()))
    Rz = yaw_R(theta)
    ca, cb = centers_of(poses_a), centers_of(poses_b)
    t = (ca - cb @ Rz.T).mean(axis=0)
    resid = ca - (cb @ Rz.T + t)
    wrap = (votes - theta + np.pi) % (2 * np.pi) - np.pi
    diag = {"yaw_deg": float(np.degrees(theta)),
            "yaw_spread_deg": float(np.degrees(wrap.std())),
            "center_rms_m": float(np.sqrt((resid ** 2).sum(1).mean())),
            "n_shared": len(poses_a)}
    return theta, t, diag


def apply_yaw_translation(poses, theta, t):
    """Re-express WORLD->CAM poses solved in gauge b in gauge a
    (X_a = Rz X_b + t): R' = R Rz^T, c' = Rz c + t, t' = -R' c'."""
    qz_inv = np.array([np.cos(theta / 2), 0.0, 0.0, -np.sin(theta / 2)])
    q_new = quat_mul(poses[:, :4], qz_inv[None, :])
    c_new = centers_of(poses) @ yaw_R(theta).T + t
    t_new = -np.einsum("nij,nj->ni", quat_to_R(q_new), c_new)
    return np.concatenate([q_new, t_new], axis=1)


def compose_yaw_translation(theta1, t1, theta2, t2):
    """Composition (theta1,t1) o (theta2,t2): apply 2 first, then 1.
    X0 = Rz1 (Rz2 X + t2) + t1 = Rz(th1+th2) X + (Rz1 t2 + t1)."""
    return theta1 + theta2, yaw_R(theta1) @ t2 + t1


def blend_poses(poses_a, poses_b, w):
    """Per-frame blend of two ALIGNED pose sets; w in [0,1], 0 -> a, 1 -> b.
    Camera-center lerp + quaternion nlerp (sign-aligned) -- adequate for the
    mm/sub-degree disagreements left after 4-DOF alignment."""
    qa, qb = poses_a[:, :4], poses_b[:, :4].copy()
    qb[(qa * qb).sum(1) < 0] *= -1
    q = (1 - w[:, None]) * qa + w[:, None] * qb
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    c = (1 - w[:, None]) * centers_of(poses_a) + w[:, None] * centers_of(poses_b)
    t = -np.einsum("nij,nj->ni", quat_to_R(q), c)
    return np.concatenate([q, t], axis=1)
