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


def render_stereo_crop(frame_l, frame_r, bbox_l, Kl, Dl, Kr, Dr, Rs, b_hat,
                       out_size=256, fov_scale=2.2, theta_max=2.6):
    """Render a rectified pinhole crop pair for one hand (bbox in LEFT image).

    Returns (crop_l, crop_r) as (3, out_size, out_size) tensors (NaN→0), plus a
    dict of geometry {Rv_l, Rv_r, f_px, g} for later 3D lifting.
    """
    g = fisheye_unproject(float((bbox_l[0] + bbox_l[2]) / 2),
                          float((bbox_l[1] + bbox_l[3]) / 2), Kl, Dl)
    f_px = fov_focal(bbox_l, g, Kl, Dl, out_size, fov_scale)
    rays = pinhole_rays(out_size, f_px, Kl.device)            # (N,3) virtual cam
    Rv_l = baseline_aligned_R(g, b_hat)
    Rv_r = Rs @ Rv_l                                          # same world orient.
    rays_l = rays @ Rv_l.T                                    # virtual→left frame
    rays_r = rays @ Rv_r.T                                    # virtual→right frame
    crop_l = sample_fisheye(frame_l, rays_l, Kl, Dl, theta_max)
    crop_r = sample_fisheye(frame_r, rays_r, Kr, Dr, theta_max)
    shp = (3, out_size, out_size)
    return (torch.nan_to_num(crop_l).reshape(shp),
            torch.nan_to_num(crop_r).reshape(shp),
            {"Rv_l": Rv_l, "Rv_r": Rv_r, "f_px": f_px, "g": g})
