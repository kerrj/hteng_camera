"""Render virtual pinhole crops from fisheye frames, on GPU, stereo-rectified.

For precise hand pose on wide fisheye footage we don't feed WiLoR the raw
(distorted) crop — we render an undistorted *pinhole* view aimed at each hand so
the network sees a roughly rectilinear hand, the way it was trained. For stereo
depth we render the SAME hand in both eyes with virtual cameras that share a
world orientation whose x-axis lies along the stereo baseline. Then image rows
are epipolar lines: disparity is purely horizontal and is a clean depth cue
(validated: residual |dy| ~1-5px vs disparity dx ~20-50px).

Design (decided 2026-06-25, see CLAUDE.md):
  - Per hand, unproject the bbox-centre pixel through the LEFT fisheye model to a
    gaze ray ``g`` (left-camera frame).
  - Build a virtual rotation ``Rv`` (cv2 camera frame: z=g forward, x along the
    baseline component perpendicular to g, y=z×x). This is the baseline-aligned
    look-at from lab42 eye/stereo.py::gaze_dir_to_so3(baseline_aligned=True).
  - Left virtual cam uses ``Rv``; right uses ``Rs @ Rv`` (same world orientation,
    expressed in the right frame). Both share focal/FOV → rectified pair.
  - Sample each fisheye frame with the Kannala-Brandt forward projection
    (ported from eye/sim/calibrated_spherical_video.py::_sample_from_calibrated).

Depends only on torch. cv2 convention throughout (forward +Z, image +y = +y),
which differs from the MuJoCo convention in the lab42 source; the projection is
re-derived for cv2 frame here and checked against cv2.fisheye.projectPoints.
"""
import torch
import torch.nn.functional as F


def fisheye_unproject(px, py, K, dist, n_iter=10):
    """Unproject fisheye pixel(s) → unit ray(s), cv2 camera frame (+Z forward).

    Inverts cv2.fisheye (Newton on the theta_d polynomial). px/py tensors or
    scalars; K (3,3), dist (4,) tensors. Returns (...,3) unit rays.
    """
    px = torch.as_tensor(px, dtype=torch.float32, device=K.device)
    py = torch.as_tensor(py, dtype=torch.float32, device=K.device)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    k1, k2, k3, k4 = dist[0], dist[1], dist[2], dist[3]
    a = (px - cx) / fx
    b = (py - cy) / fy
    theta_d = torch.sqrt(a * a + b * b)
    theta = theta_d.clone()
    for _ in range(n_iter):
        t2 = theta * theta
        f = theta * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4)))) - theta_d
        fp = 1 + t2 * (3 * k1 + t2 * (5 * k2 + t2 * (7 * k3 + t2 * 9 * k4)))
        theta = theta - f / fp
    sin_t = torch.sin(theta)
    r = torch.where(theta_d > 1e-9, theta_d, torch.ones_like(theta_d))
    d = torch.stack([sin_t * a / r, sin_t * b / r, torch.cos(theta)], dim=-1)
    return d / torch.linalg.norm(d, dim=-1, keepdim=True)


def triangulate_rays(g_l, g_r, Rs, ts):
    """Closest point to two (generally SKEW) gaze rays, in the LEFT frame.

    The left/right YOLO bbox-centre rays won't actually intersect, so we take the
    midpoint of their common perpendicular (the point closest to both rays).

    Args:
        g_l: (3,) ray direction in the LEFT camera frame.
        g_r: (3,) ray direction in the RIGHT camera frame.
        Rs, ts: stereo extrinsics, X_right = Rs @ X_left + ts (left→right).

    Returns:
        P: (3,) the common point in the LEFT camera frame.
    """
    u = g_l / torch.linalg.norm(g_l)                 # left ray dir, left frame
    cr = -Rs.T @ ts                                  # right cam centre, left frame
    v = Rs.T @ g_r                                   # right ray dir, left frame
    v = v / torch.linalg.norm(v)
    w0 = -cr                                         # O_left(=0) - cr
    a = u @ u; b = u @ v; c = v @ v
    d = u @ w0; e = v @ w0
    denom = a * c - b * b                            # 0 ⇔ rays parallel
    denom = torch.where(denom.abs() < 1e-8, torch.ones_like(denom), denom)
    s = (b * e - c * d) / denom                      # param along left ray
    t = (a * e - b * d) / denom                      # param along right ray
    p_left = u * s                                   # closest pt on left ray
    p_right = cr + v * t                             # closest pt on right ray
    return 0.5 * (p_left + p_right)


def render_crop(frame, Rv, f_px, K, dist, out_size, theta_max=2.6):
    """Sample one virtual pinhole crop (Rv: virtual-cam→fisheye-cam, cols=axes)."""
    rays = pinhole_rays(out_size, f_px, K.device)             # (N,3) virtual cam
    rays_cam = rays @ Rv.T                                    # → fisheye cam frame
    crop = sample_fisheye(frame, rays_cam, K, dist, theta_max)
    return torch.nan_to_num(crop).reshape(3, out_size, out_size)


def baseline_aligned_R(g, b_hat):
    """Virtual-cam rotation (cols = cam x,y,z axes in left frame), cv2 frame.

    z = optical axis = g; x = baseline component ⟂ g (so rows ∥ baseline →
    epipolar); y = z × x (image down). g, b_hat: (3,) unit tensors.
    """
    g = g / torch.linalg.norm(g)
    x = b_hat - (b_hat @ g) * g
    nx = torch.linalg.norm(x)
    if nx < 1e-6:                       # gaze ∥ baseline (degenerate): fall back
        alt = torch.tensor([0.0, 0.0, 1.0], device=g.device, dtype=g.dtype)
        x = alt - (alt @ g) * g
        nx = torch.linalg.norm(x)
    x = x / nx
    y = torch.linalg.cross(g, x)
    return torch.stack([x, y, g], dim=1)


def pinhole_rays(out_size, f_px, device, dtype=torch.float32):
    """Unit rays for an out_size² virtual pinhole, cv2 frame (+Z fwd, +y down).

    Pixel (u,v) → ((u-c)/f, (v-c)/f, 1) normalized. Returns (N,3), row-major.
    """
    c = (out_size - 1) / 2.0
    ys, xs = torch.meshgrid(torch.arange(out_size, device=device, dtype=dtype),
                            torch.arange(out_size, device=device, dtype=dtype),
                            indexing="ij")
    d = torch.stack([(xs - c) / f_px, (ys - c) / f_px, torch.ones_like(xs)], -1)
    d = d.reshape(-1, 3)
    return d / torch.linalg.norm(d, dim=-1, keepdim=True)


def sample_fisheye(frame, rays_cam, K, dist, theta_max):
    """Sample a fisheye frame along camera-frame rays (cv2 KB forward model).

    Ported from calibrated_spherical_video._sample_from_calibrated, cv2 frame.

    Args:
        frame: (C, H, W) tensor on device (any float dtype).
        rays_cam: (N, 3) ray directions in the fisheye camera frame (+Z fwd).
        K (3,3), dist (4,): fisheye intrinsics/distortion tensors.
        theta_max: half-FOV cutoff (rad); rays beyond → NaN (KB fold guard).

    Returns:
        (C, N) sampled colors; off-sensor/behind → NaN.
    """
    C, H, W = frame.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    k1, k2, k3, k4 = dist[0], dist[1], dist[2], dist[3]
    x, y, z = rays_cam[:, 0], rays_cam[:, 1], rays_cam[:, 2]
    xy_radius = torch.sqrt(x * x + y * y)
    theta = torch.atan2(xy_radius, z)
    t2 = theta * theta
    theta_d = theta * (1.0 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))))
    inv_r = torch.where(xy_radius > 1e-9, 1.0 / xy_radius, torch.zeros_like(xy_radius))
    px = fx * theta_d * x * inv_r + cx
    py = fy * theta_d * y * inv_r + cy
    valid = (theta < theta_max) & (px >= 0) & (px < W) & (py >= 0) & (py < H)
    grid_x = 2.0 * (px + 0.5) / W - 1.0
    grid_y = 2.0 * (py + 0.5) / H - 1.0
    grid = torch.stack([grid_x, grid_y], -1).view(1, 1, -1, 2)
    sampled = F.grid_sample(frame.unsqueeze(0), grid.to(frame.dtype),
                            mode="bilinear", padding_mode="zeros",
                            align_corners=False).squeeze(0).squeeze(1)  # (C,N)
    return sampled.masked_fill(~valid.unsqueeze(0), float("nan"))


def fisheye_project(rays_cam, K, dist):
    """Project camera-frame rays → fisheye pixels (cv2 KB forward model).

    rays_cam: (..., 3). Returns (..., 2) pixel coords. (Same math as
    sample_fisheye's px/py, exposed for mapping keypoints back to the frame.)
    """
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    k1, k2, k3, k4 = dist[0], dist[1], dist[2], dist[3]
    x, y, z = rays_cam[..., 0], rays_cam[..., 1], rays_cam[..., 2]
    xy_radius = torch.sqrt(x * x + y * y)
    theta = torch.atan2(xy_radius, z)
    t2 = theta * theta
    theta_d = theta * (1.0 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))))
    inv_r = torch.where(xy_radius > 1e-9, 1.0 / xy_radius, torch.zeros_like(xy_radius))
    px = fx * theta_d * x * inv_r + cx
    py = fy * theta_d * y * inv_r + cy
    return torch.stack([px, py], dim=-1)


def crop_px_to_fisheye(uv, f_px, out_size, Rv, K, dist):
    """Map pinhole-crop pixel(s) → fisheye pixel(s).

    uv: (..., 2) crop pixels. f_px/out_size define the crop pinhole; Rv is the
    virtual-cam→fisheye-cam rotation (cols = axes). Returns (..., 2) fisheye px.
    """
    c = (out_size - 1) / 2.0
    u, v = uv[..., 0], uv[..., 1]
    rays = torch.stack([(u - c) / f_px, (v - c) / f_px, torch.ones_like(u)], -1)
    rays = rays / torch.linalg.norm(rays, dim=-1, keepdim=True)
    rays_cam = rays @ Rv.T
    return fisheye_project(rays_cam, K, dist)


def fov_focal(bbox, g, K, dist, out_size, fov_scale=2.2):
    """Focal (px) so the (expanded) bbox angular extent fills out_size.

    bbox: (4,) [x1,y1,x2,y2] left-image px. g: (3,) centre gaze ray.
    """
    corners = torch.tensor([[bbox[0], bbox[1]], [bbox[2], bbox[1]],
                            [bbox[0], bbox[3]], [bbox[2], bbox[3]]],
                           device=K.device)
    rays = fisheye_unproject(corners[:, 0], corners[:, 1], K, dist)  # (4,3)
    ang = torch.arccos(torch.clamp(rays @ g, -1, 1)).max()
    half_fov = ang * fov_scale / 2.0
    return (out_size / 2.0) / torch.tan(half_fov)


def bbox_center_ray(bbox, K, dist):
    """Unproject a bbox-centre pixel → unit gaze ray in that eye's camera frame."""
    return fisheye_unproject(float((bbox[0] + bbox[2]) / 2),
                             float((bbox[1] + bbox[3]) / 2), K, dist)


def render_stereo_crop(frame_l, frame_r, bbox_l, bbox_r, Kl, Dl, Kr, Dr, Rs, ts,
                       b_hat, out_size=256, fov_scale=2.2, theta_max=2.6):
    """Render a VERGED, baseline-aligned pinhole crop pair for one hand.

    Both eyes detect the hand independently (bbox_l in LEFT image, bbox_r in
    RIGHT). We triangulate the two bbox-centre rays to a common 3D point P
    (closest point to the skew rays), then aim each eye's virtual camera AT P —
    so the hand is centred in BOTH crops (fixing the old bug where the right crop
    just reused the left's orientation and drifted off-centre with disparity).

    Each virtual camera is baseline-aligned (x-axis along the stereo baseline in
    its own frame), so even though the optical axes verge on P, image rows remain
    epipolar lines (horizontal). The cameras are NOT parallel, so depth is NOT
    f*baseline/disparity here — downstream geometry must use Rv_l/Rv_r + the
    pinhole projection only (no parallel-axis / disparity shortcut).

    Returns (crop_l, crop_r) (3,out,out) tensors (NaN→0) plus geometry
    {Rv_l, Rv_r, f_px, g_l, g_r, P} (P in the LEFT camera frame).
    """
    g_l = bbox_center_ray(bbox_l, Kl, Dl)                     # left frame
    g_r = bbox_center_ray(bbox_r, Kr, Dr)                     # right frame
    P = triangulate_rays(g_l, g_r, Rs, ts)                    # left frame
    # re-aim both eyes exactly at P (refines the raw bbox-centre rays)
    look_l = P / torch.linalg.norm(P)
    P_r = Rs @ P + ts                                         # P in right frame
    look_r = P_r / torch.linalg.norm(P_r)
    # shared focal: size the crop from the LEFT bbox's angular extent about P
    f_px = fov_focal(bbox_l, look_l, Kl, Dl, out_size, fov_scale)
    b_hat_r = Rs @ b_hat                                      # baseline dir, right frame
    Rv_l = baseline_aligned_R(look_l, b_hat)
    Rv_r = baseline_aligned_R(look_r, b_hat_r)
    crop_l = render_crop(frame_l, Rv_l, f_px, Kl, Dl, out_size, theta_max)
    crop_r = render_crop(frame_r, Rv_r, f_px, Kr, Dr, out_size, theta_max)
    return crop_l, crop_r, {"Rv_l": Rv_l, "Rv_r": Rv_r, "f_px": f_px,
                            "g_l": g_l, "g_r": g_r, "P": P}
