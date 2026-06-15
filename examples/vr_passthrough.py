"""HTENG stereo → Quest 3S WebXR passthrough host.

Run:  python examples/vr_passthrough.py
See docs/superpowers/specs/2026-06-15-vr-passthrough-design.md
"""
import threading

import numpy as np

try:
    from turbojpeg import TurboJPEG, TJPF_RGB
    _TJ = TurboJPEG()
except Exception:  # libturbojpeg missing or init failed → cv2 fallback
    _TJ = None


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


class Mailbox:
    """Single-slot, newest-wins handoff between a producer thread and the
    async sender. put() overwrites; get_latest() peeks (never consumes)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._item = None

    def put(self, item):
        with self._lock:
            self._item = item

    def get_latest(self):
        with self._lock:
            return self._item


def encode_jpeg(rgb, quality=85):
    """RGB uint8 (H,W,3) → JPEG bytes. TurboJPEG if available, else cv2."""
    if _TJ is not None:
        return _TJ.encode(np.ascontiguousarray(rgb), quality=quality,
                          pixel_format=TJPF_RGB)
    import cv2
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return buf.tobytes()
