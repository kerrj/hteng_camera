"""HTENG stereo → Quest 3S WebXR passthrough host.

Run:  python examples/vr_passthrough.py
See docs/superpowers/specs/2026-06-15-vr-passthrough-design.md
"""
import numpy as np


def kb_project(dirs, K, dist):
    """Kannala-Brandt forward projection (the exact math the client shader mirrors).

    dirs : (N,3) ray directions in the OpenCV lens frame (x right, y down, z fwd).
    K    : (3,3) camera matrix.  dist : (4,) [k1,k2,k3,k4].
    Returns (N,2) pixel coords. Mirrors cv2.fisheye.projectPoints for z>0 rays.
    """
    dirs = np.asarray(dirs, np.float64).reshape(-1, 3)
    K = np.asarray(K, np.float64).reshape(3, 3)
    k1, k2, k3, k4 = np.asarray(dist, np.float64).ravel()[:4]
    x, y, z = dirs[:, 0], dirs[:, 1], dirs[:, 2]
    rxy = np.hypot(x, y)
    theta = np.arctan2(rxy, z)
    t2 = theta * theta
    thetad = theta * (1.0 + k1 * t2 + k2 * t2**2 + k3 * t2**3 + k4 * t2**4)
    scale = np.where(rxy > 1e-9, thetad / np.maximum(rxy, 1e-12), 1.0)
    u = K[0, 0] * (x * scale) + K[0, 2]
    v = K[1, 1] * (y * scale) + K[1, 2]
    return np.stack([u, v], axis=1)
