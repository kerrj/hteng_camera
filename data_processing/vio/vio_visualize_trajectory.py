"""viser replay of stage 5 (vio_bundle_adjust.py) output: optimized camera
trajectory + landmark point cloud, plus a left-eye video thumbnail synced
to the frame slider.

Landmarks colored by their actual video pixel at first observation
(--color-mode pixel, default) or a depth colormap. Left (orange) + right
(cyan) stereo frustums per frame, right pose via T_wr = T_stereo @ T_wl.
World is gravity-aligned (+z up), so viser's up-direction is +z.

Run (from data_processing/vio/):
    python vio_visualize_trajectory.py ../../testimu --trajectory ../../testimu/trajectory.npz
Then forward the printed port to your laptop (if run remotely).
"""
import argparse
import functools
import os
import threading
import time

import numpy as np


def quat_to_matrix(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def matrix_to_quat(R):
    trace = np.trace(R)
    qw = np.sqrt(max(trace + 1.0, 0.0)) / 2.0
    if qw > 1e-6:
        qx = (R[2, 1] - R[1, 2]) / (4 * qw)
        qy = (R[0, 2] - R[2, 0]) / (4 * qw)
        qz = (R[1, 0] - R[0, 1]) / (4 * qw)
    else:
        qx = np.sqrt(max((R[0, 0] + 1) / 2.0, 0.0))
        qy = np.sqrt(max((R[1, 1] + 1) / 2.0, 0.0))
        qz = np.sqrt(max((R[2, 2] + 1) / 2.0, 0.0))
    return np.array([qw, qx, qy, qz])


def depth_to_color(z, lo, hi):
    """z: (N,). Maps [lo,hi] -> a blue(near)->red(far) colormap, clipping
    outside that range to the endpoint colors -- so implausible depth
    outliers land visually at the extreme color, not off-scale/invisible."""
    t = np.clip((z - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    r = t
    b = 1.0 - t
    g = 0.3 * (1.0 - np.abs(2 * t - 1))
    return np.stack([r, g, b], axis=1)


def main():
    import viser

    ap = argparse.ArgumentParser()
    ap.add_argument("recording")
    ap.add_argument("--trajectory", default=None,
                     help="default: <recording>/trajectory.npz")
    ap.add_argument("--video", default=None, help="default: <recording>/left.mp4")
    ap.add_argument("--video-right", default=None,
                     help="default: <recording>/right.mp4 -- only needed for "
                          "--color-mode pixel landmarks whose first observation "
                          "was in the right eye")
    ap.add_argument("--color-mode", choices=("pixel", "depth"), default="pixel",
                     help="pixel: sample each landmark's color from the actual "
                          "video frame/eye/px of its first observation (needs "
                          "trajectory.npz's point_first_* fields, written by "
                          "vio_bundle_adjust.py). depth: percentile colormap.")
    ap.add_argument("--thumb-w", type=int, default=480)
    ap.add_argument("--point-size", type=float, default=0.005)
    ap.add_argument("--robust-scale", type=float, default=0.05,
                    help="Cauchy scale used in the solve -- for the outlier "
                         "heatmap's weight reconstruction (match the BA run)")
    ap.add_argument("--frustum-scale", type=float, default=0.03)
    ap.add_argument("--port", type=int, default=8081)
    args = ap.parse_args()

    traj_path = args.trajectory or os.path.join(args.recording, "trajectory.npz")
    video_path = args.video or os.path.join(args.recording, "left.mp4")
    video_right_path = args.video_right or os.path.join(args.recording, "right.mp4")

    d = np.load(traj_path)
    frame_idx = d["frame_idx"]
    pose_wxyz_xyz = d["pose_wxyz_xyz"]  # (n_frames, 7) T_wl (WORLD->CAMERA)
    points = d["points"]  # (n_points, 3) world frame

    # point_alive/point_med_ang only exist if vio_bundle_adjust.py's outlier
    # filter rounds ran -- fall back to keeping everything for older outputs.
    keep_points = d["point_alive"] if "point_alive" in d else np.ones(len(points), dtype=bool)
    n_dropped = int((~keep_points).sum())
    if n_dropped:
        print(f"hiding {n_dropped}/{len(points)} landmarks dropped by outlier filtering")

    n_frames = len(frame_idx)
    fmin, fmax = 0, n_frames - 1

    # Invert every T_wl to get T_lw (camera's pose IN world frame) for plotting.
    cam_pos_world = np.zeros((n_frames, 3))
    cam_quat_world = np.zeros((n_frames, 4))
    for i in range(n_frames):
        q_wl = pose_wxyz_xyz[i, :4]
        t_wl = pose_wxyz_xyz[i, 4:]
        R_wl = quat_to_matrix(q_wl)
        R_lw = R_wl.T
        t_lw = -(R_lw @ t_wl)
        cam_pos_world[i] = t_lw
        cam_quat_world[i] = matrix_to_quat(R_lw)

    # Derive the RIGHT camera's world pose per frame: T_wr = T_stereo @ T_wl.
    import json
    stereo_file = next(f for f in os.listdir(args.recording) if f.startswith("stereo_"))
    with open(os.path.join(args.recording, stereo_file)) as f:
        stereo = json.load(f)
    R_st = np.array(stereo["R"], np.float64)
    t_st = np.array(stereo["t"], np.float64).reshape(3)

    cam_pos_world_r = np.zeros((n_frames, 3))
    cam_quat_world_r = np.zeros((n_frames, 4))
    for i in range(n_frames):
        R_wl = quat_to_matrix(pose_wxyz_xyz[i, :4])
        t_wl = pose_wxyz_xyz[i, 4:]
        R_wr = R_st @ R_wl
        t_wr = R_st @ t_wl + t_st
        R_rw = R_wr.T
        t_rw = -(R_rw @ t_wr)
        cam_pos_world_r[i] = t_rw
        cam_quat_world_r[i] = matrix_to_quat(R_rw)

    left_frame = None
    if os.path.exists(video_path):
        import cv2
        cap = cv2.VideoCapture(video_path)

        @functools.lru_cache(maxsize=512)
        def _decode(fr):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fr))
            ok, img = cap.read()
            if not ok:
                return np.zeros((10, 10, 3), dtype=np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]
            tw = args.thumb_w
            return cv2.resize(img, (tw, int(h * tw / w)), interpolation=cv2.INTER_AREA)

        left_frame = lambda fr: _decode(int(fr))
    else:
        print(f"[warn] no video at {video_path} -- running without thumbnail panel")

    if args.color_mode == "pixel" and "point_first_frame" in d:
        # Sample each landmark's color from the full-res frame at its exact
        # pixel. Landmarks sorted by frame, decoded SEQUENTIALLY -- random
        # per-landmark seeks were still running after 20+ min on ~20k
        # landmarks; sequential decode is what video codecs are fast at.
        import cv2

        first_frame = d["point_first_frame"]
        first_is_right = d["point_first_is_right"]
        first_px = d["point_first_px"]
        point_colors = np.full((len(points), 3), 0.5)  # gray fallback

        for path, is_right_eye in ((video_path, False), (video_right_path, True)):
            if not os.path.exists(path):
                continue
            eye_mask = (first_is_right == is_right_eye) & keep_points
            eye_indices = np.where(eye_mask)[0]
            if len(eye_indices) == 0:
                continue
            order = eye_indices[np.argsort(first_frame[eye_indices])]

            # Named `color_cap`, not `cap` -- releasing it must not touch
            # the thumbnail closure's own `cap` above.
            color_cap = cv2.VideoCapture(path)
            cur_frame = -1
            img = None
            for k in order:
                target = int(first_frame[k])
                # advance sequentially from cur_frame to target -- a single
                # forward .read() per intervening frame, which is the fast
                # path for any reasonable video codec, instead of seeking.
                while cur_frame < target:
                    ok, raw = color_cap.read()
                    cur_frame += 1
                    if not ok:
                        img = None
                        break
                    img = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
                if img is None:
                    continue
                x, y = int(round(first_px[k, 0])), int(round(first_px[k, 1]))
                h, w = img.shape[:2]
                if 0 <= y < h and 0 <= x < w:
                    point_colors[k] = img[y, x] / 255.0
            color_cap.release()
        print(f"{len(points)} landmarks colored from their first observation's "
              f"actual video pixel")
    else:
        z = points[:, 2]
        lo, hi = np.percentile(z, [2, 98])
        point_colors = depth_to_color(z, lo, hi)
        print(f"{len(points)} landmarks, depth range shown [{lo:.2f}, {hi:.2f}]m "
              f"(2nd-98th percentile; outside this clips to endpoint colors)")

    server = viser.ViserServer(port=args.port)
    server.scene.world_axes.visible = False

    server.scene.set_up_direction("+z")  # stage-5 world is gravity-aligned

    # Outlier heatmap: Cauchy down-weight reconstructed from each landmark's
    # median angular residual, r = sin(med_ang), w = 1/(1+(r/scale)^2).
    heat_colors = None
    if "point_med_ang" in d:
        med_ang = np.asarray(d["point_med_ang"], dtype=np.float64)
        r = np.sin(np.radians(med_ang))
        w = 1.0 / (1.0 + (r / args.robust_scale) ** 2)
        w = np.where(np.isfinite(w), w, 1.0)
        heat_colors = np.zeros((len(points), 3))  # green (inlier) -> red (outlier)
        heat_colors[:, 0] = 1.0 - w
        heat_colors[:, 1] = w
        finite = np.isfinite(med_ang) & keep_points
        if finite.any():
            for thr in (0.75, 0.5, 0.25, 0.1):
                frac = float((w[finite] < thr).mean())
                print(f"  outlier weight < {thr}: {100*frac:5.1f}% of landmarks "
                      f"({int((w[finite] < thr).sum())})")
            print(f"  median Cauchy weight: {np.median(w[finite]):.3f}, "
                  f"median residual: {np.median(med_ang[finite]):.2f} deg")

    pc = server.scene.add_point_cloud(
        "/landmarks", points=points[keep_points], colors=point_colors[keep_points],
        point_size=args.point_size, point_shape="circle")

    if heat_colors is not None:
        gui_outliers = server.gui.add_checkbox("outlier heatmap", False)

        @gui_outliers.on_update
        def _(_=None):
            pc.colors = (heat_colors if gui_outliers.value
                         else point_colors)[keep_points]

    # All frustums faint for the trajectory shape; one bright pair for the
    # current frame (updated below).
    for i in range(n_frames):
        server.scene.add_camera_frustum(
            f"/trail/cam_l_{i}", fov=1.4, aspect=1.2, scale=args.frustum_scale * 0.6,
            color=(0.6, 0.4, 0.1), wxyz=cam_quat_world[i], position=cam_pos_world[i])
        server.scene.add_camera_frustum(
            f"/trail/cam_r_{i}", fov=1.4, aspect=1.2, scale=args.frustum_scale * 0.6,
            color=(0.1, 0.4, 0.5), wxyz=cam_quat_world_r[i], position=cam_pos_world_r[i])
    current_frustum_l = server.scene.add_camera_frustum(
        "/current_left", fov=1.4, aspect=1.2, scale=args.frustum_scale,
        color=(1.0, 0.6, 0.0), wxyz=cam_quat_world[0], position=cam_pos_world[0])
    current_frustum_r = server.scene.add_camera_frustum(
        "/current_right", fov=1.4, aspect=1.2, scale=args.frustum_scale,
        color=(0.0, 0.8, 1.0), wxyz=cam_quat_world_r[0], position=cam_pos_world_r[0])

    gui_play = server.gui.add_checkbox("play", False)
    gui_frame = server.gui.add_slider("frame", fmin, fmax, 1, fmin)
    gui_info = server.gui.add_text("info", "", disabled=True)
    gui_img = server.gui.add_image(left_frame(frame_idx[fmin]), label="left eye",
                                    format="jpeg") if left_frame else None

    _update_lock = threading.Lock()

    def update(_=None):
        with _update_lock:
            fr = int(gui_frame.value)
            img = left_frame(frame_idx[fr]) if left_frame is not None else None
            lines = [
                f"pose slot {fr} (video frame {frame_idx[fr]})",
                f"  cam position (world): {cam_pos_world[fr]}",
            ]
            with server.atomic():
                current_frustum_l.wxyz = cam_quat_world[fr]
                current_frustum_l.position = cam_pos_world[fr]
                current_frustum_r.wxyz = cam_quat_world_r[fr]
                current_frustum_r.position = cam_pos_world_r[fr]
                gui_info.value = "\n".join(lines)
                if img is not None:
                    gui_img.image = img

    gui_frame.on_update(update)
    update()

    print(f"viser running on port {args.port} -- forward it: "
          f"ssh -L {args.port}:localhost:{args.port} <host>")

    while True:
        if gui_play.value:
            nxt = int(gui_frame.value) + 1
            gui_frame.value = fmin if nxt > fmax else nxt
        time.sleep(1.0 / 30.0)


if __name__ == "__main__":
    main()
