"""Build a solid hand mesh from 21 joints (tapered tubes per bone + joint spheres).

Used to render optimized hand poses as a semi-transparent surface in viser
(nicer than a wireframe, and driven only by the verified joints_3d_cam — no MANO
model / handedness reasoning needed). Returns one combined (V, F) triangle mesh.
"""
import numpy as np

_BONES = np.array([(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
                   (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15),
                   (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)])


def _basis(z):
    z = z / (np.linalg.norm(z) + 1e-9)
    a = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = np.cross(a, z); x /= np.linalg.norm(x) + 1e-9
    y = np.cross(z, x)
    return x, y, z


def _tube(p0, p1, r0, r1, sides=8):
    """Tapered tube (radii r0->r1) from p0 to p1, no end caps. (V,F)."""
    d = p1 - p0
    L = np.linalg.norm(d)
    if L < 1e-6:
        return np.zeros((0, 3)), np.zeros((0, 3), int)
    x, y, _ = _basis(d)
    ang = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    ring = np.cos(ang)[:, None] * x + np.sin(ang)[:, None] * y     # (sides,3)
    V = np.vstack([p0 + r0 * ring, p1 + r1 * ring])                # (2*sides,3)
    F = []
    for i in range(sides):
        j = (i + 1) % sides
        F += [(i, j, j + sides), (i, j + sides, i + sides)]
    return V, np.array(F, int)


def _sphere(c, r, stacks=5, slices=8):
    lat = np.linspace(0, np.pi, stacks + 1)
    lon = np.linspace(0, 2 * np.pi, slices, endpoint=False)
    V = [c + r * np.array([np.sin(a) * np.cos(b), np.sin(a) * np.sin(b), np.cos(a)])
         for a in lat for b in lon]
    V = np.array(V)
    F = []
    for i in range(stacks):
        for j in range(slices):
            j2 = (j + 1) % slices
            a = i * slices + j; b = i * slices + j2
            c2 = (i + 1) * slices + j; d = (i + 1) * slices + j2
            F += [(a, b, d), (a, d, c2)]
    return V, np.array(F, int)


def hand_mesh(joints, bone_r=0.007, tip_r=0.004, joint_r=0.009):
    """joints (21,3) -> combined (V,F). Tubes taper toward fingertips."""
    parts = []
    for a, b in _BONES:
        r1 = tip_r if b in (4, 8, 12, 16, 20) else bone_r      # fingertip bones taper
        parts.append(_tube(joints[a], joints[b], bone_r, r1))
    for k, j in enumerate(joints):
        parts.append(_sphere(j, tip_r if k in (4, 8, 12, 16, 20) else joint_r))
    V = np.zeros((0, 3)); F = np.zeros((0, 3), int)
    for v, f in parts:
        if len(v):
            F = np.vstack([F, f + len(V)]); V = np.vstack([V, v])
    return V.astype(np.float32), F.astype(np.int32)
