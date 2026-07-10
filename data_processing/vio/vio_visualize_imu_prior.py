"""viser replay of stage 4 (vio_imu_prior.py) output -- the acceptance check
for the IMU math BEFORE bundle adjustment exists, since there's no pose chain
yet to sanity-check against. Shows, at a FIXED camera (no trajectory yet):

  - a gravity arrow: -gravity_cam[frame] (i.e. pointing "up", away from
    Earth), so if R_CI and the accel sign convention are right, it should
    point roughly toward viser's world +z (up) whenever the head is roughly
    level, and should visibly destabilize/shrink (low confidence -> short/
    faded arrow) during fast motion.
  - a small rotating axes triad showing the NAIVE cumulative product of
    rel_quat from frame 0 -- NOT a claim about drift-free global attitude
    (that's exactly what the relative-only IMU factor design in
    vio/CLAUDE.md avoids relying on for BA), purely a visual "does this
    rotate the way the video shows the head rotating" check.
  - the left-eye video thumbnail, synced to the same frame slider, so you
    can eyeball rotation/gravity stability against actual head motion.
  - a numeric readout: gravity confidence weight, per-step rotation angle.

Run (from data_processing/vio/):
    python vio_visualize_imu_prior.py ../../testimu --imu-relative ../../testimu/imu_relative.npz
Then forward the printed port to your laptop (if run remotely).
"""
import argparse
import functools
import os
import threading
import time

import numpy as np


def quat_mult(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_to_wxyz_matrix(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def cumulative_quats(rel_quat, rel_valid):
    """q[0] = identity; q[k+1] = q[k] * rel_quat[k] (or unchanged if that
    edge is invalid). Naive chaining for visualization ONLY -- see module
    docstring on why this isn't a drift-free global attitude claim."""
    n = rel_quat.shape[0] + 1
    q = np.zeros((n, 4))
    q[0] = [1.0, 0.0, 0.0, 0.0]
    for k in range(n - 1):
        if rel_valid[k]:
            q[k + 1] = quat_mult(q[k], rel_quat[k])
        else:
            q[k + 1] = q[k]
    return q


# viser world is +z up; cv2 camera frame here is +z forward, +x right, +y
# down (see fisheye_pinhole.py). Rotate camera-frame vectors into a
# world-ish display frame by simply relabeling axes: display_x = cam_x,
# display_y = -cam_z, display_z = -cam_y (so cam "up" = -cam_y maps to
# display +z). This is ONLY a display convenience (no bearing on the
# camera-frame math computed in vio_imu_prior.py) -- makes the render's
# "up" match viser's "up" so the gravity arrow reads naturally.
_DISPLAY_FROM_CAM = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0],
    [0.0, -1.0, 0.0],
])


def main():
    import viser

    ap = argparse.ArgumentParser()
    ap.add_argument("recording")
    ap.add_argument("--imu-relative", default=None,
                     help="default: <recording>/imu_relative.npz")
    ap.add_argument("--video", default=None, help="default: <recording>/left.mp4")
    ap.add_argument("--thumb-w", type=int, default=480)
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()

    npz_path = args.imu_relative or os.path.join(args.recording, "imu_relative.npz")
    video_path = args.video or os.path.join(args.recording, "left.mp4")

    d = np.load(npz_path)
    frame_idx = d["frame_idx"]
    gravity_cam = d["gravity_cam"]
    gravity_weight = d["gravity_weight"]
    rel_quat = d["rel_quat"]
    rel_valid = d["rel_valid"]
    cum_q = cumulative_quats(rel_quat, rel_valid)
    step_angle_deg = np.concatenate([[0.0], np.degrees(
        2 * np.arccos(np.clip(np.abs(rel_quat[:, 0]), -1, 1)))])

    n = len(frame_idx)
    fmin, fmax = 0, n - 1

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

    server = viser.ViserServer(port=args.port)
    server.scene.world_axes.visible = False

    server.scene.add_frame("/camera", axes_length=0.05, axes_radius=0.002)
    arrow_h = server.scene.add_arrows(
        "/camera/gravity",
        points=np.array([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.15]]], dtype=np.float32),
        colors=np.array([[1.0, 0.9, 0.1]], dtype=np.float32),
        shaft_radius=0.004, head_radius=0.012, head_length=0.03)
    orientation_h = server.scene.add_frame(
        "/camera/orientation", axes_length=0.08, axes_radius=0.003)

    gui_play = server.gui.add_checkbox("play", False)
    gui_frame = server.gui.add_slider("frame", fmin, fmax, 1, fmin)
    gui_info = server.gui.add_text("info", "", disabled=True)
    gui_img = server.gui.add_image(left_frame(fmin), label="left eye",
                                    format="jpeg") if left_frame else None

    _update_lock = threading.Lock()

    def update(_=None):
        with _update_lock:
            fr = int(gui_frame.value)
            img = left_frame(fr) if left_frame is not None else None

            up_cam = -gravity_cam[fr]  # gravity_cam points "down" (toward Earth)
            w = float(gravity_weight[fr])
            up_disp = _DISPLAY_FROM_CAM @ up_cam
            mag = np.linalg.norm(up_disp)
            up_disp = up_disp / max(mag, 1e-9)

            R_cam = quat_to_wxyz_matrix(cum_q[fr])
            R_disp = _DISPLAY_FROM_CAM @ R_cam @ _DISPLAY_FROM_CAM.T
            orient_wxyz = viser.transforms.SO3.from_matrix(R_disp).wxyz

            lines = [
                f"frame {fr} (idx {frame_idx[fr]})",
                f"  gravity weight: {w:.2f}  (low = fast motion, gated)",
                f"  step rotation:  {step_angle_deg[fr]:.2f} deg",
            ]
            with server.atomic():
                # length/color both fall off with low confidence so a
                # fast-motion frame's gravity estimate visibly reads as
                # "don't trust this" rather than silently pointing wrong.
                arrow_len = 0.05 + 0.15 * w
                arrow_h.points = np.array(
                    [[[0.0, 0.0, 0.0], (up_disp * arrow_len).tolist()]], dtype=np.float32)
                arrow_h.colors = np.array([[1.0, 0.9 * w + 0.1, 0.1]], dtype=np.float32)
                orientation_h.wxyz = orient_wxyz
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
