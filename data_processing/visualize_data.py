"""Generic viser visualizer for data_processing outputs: the VIO camera
trajectory + landmark point cloud, optionally composed with hand meshes
(FK'd from each hand's stereo-optimized MANO params, placed in world via the
left camera's pose at that frame: hand_world(t) = T_cam_world(t) @
hand_cam(t)). Supersedes vio/vio_visualize_trajectory.py.

Landmarks colored by their actual video pixel at first observation
(--color-mode pixel, default) or a depth colormap. Left (orange) + right
(cyan) stereo frustums per frame, right pose via T_wr = T_stereo @ T_wl.
World is gravity-aligned (+z up), so viser's up-direction is +z. Static
trail frustums are drawn thin so the bright current-frame frustums pop.

Video thumbnail decoded via torchcodec (fast random seek for scrubbing).

All products default into `<recording>/derived/` (trajectory.npz, hands3d_*.jsonl),
next to the raw left.mp4/stereo_*.json inputs in the recording root -- so a bare
`python visualize_data.py ../testimu` finds everything, hands included, with no
path bookkeeping. Hands are auto-loaded if their jsonl exists, skipped otherwise.

Run (from data_processing/):
    python visualize_data.py ../testimu                 # traj + hands (if present)
    python visualize_data.py ../testimu --trajectory /some/other/trajectory.npz
Then forward the printed port to your laptop (if run remotely).
"""
import argparse
import functools
import json
import os
import sys
import threading
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "hands"))


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


def make_frame_loader(video_path, thumb_w):
    """torchcodec-backed random-access frame loader -- much faster than cv2's
    CAP_PROP_POS_FRAMES seek for the scrubber's random-access pattern."""
    import cv2
    from torchcodec.decoders import VideoDecoder

    dec = VideoDecoder(video_path, device="cpu")
    lock = threading.Lock()  # torchcodec decoders are not thread-safe; the
    # play loop and the scrub callback run on different threads.

    @functools.lru_cache(maxsize=512)
    def _decode(fr):
        with lock:
            f = dec.get_frames_in_range(start=fr, stop=fr + 1).data[0]  # (3,H,W)
        img = f.permute(1, 2, 0).numpy().astype(np.uint8)
        h, w = img.shape[:2]
        tw = thumb_w
        return cv2.resize(img, (tw, int(h * tw / w)), interpolation=cv2.INTER_AREA)

    return lambda fr: _decode(int(fr))


def load_hand_track(path):
    """Read a stereo3d hands jsonl (hands/stereo_optimize.py output) ->
    (meta dict, {video_frame: hand dict})."""
    meta, frames = None, {}
    for line in open(path):
        d = json.loads(line)
        if d.get("meta"):
            meta = d
            continue
        frames[d["frame"]] = d
    return meta, frames


def hand_mesh_cam(M, faces, h, beta, mirror):
    """FK one hand's MANO mesh into the LEFT-FISHEYE CAMERA frame. Returns (778,3)."""
    import jax.numpy as jnp
    import jaxlie
    import mano_jax as MJ

    R = jaxlie.SO3(jnp.asarray(np.array(h["quat"]))).as_matrix()  # (16,3,3)
    v = np.array(MJ.mano_mesh_R(M, R, jnp.asarray(beta)))          # (778,3) virtual
    v[:, 0] *= mirror                                              # left-hand x-mirror
    v = v + np.array(h["trans_virtual"])[None, :]                  # place root
    v = v @ np.array(h["Rv_l"]).T                                   # -> left-fisheye cam
    return v


def main():
    import viser

    ap = argparse.ArgumentParser()
    ap.add_argument("recording")
    ap.add_argument("--trajectory", default=None,
                     help="default: <recording>/derived/trajectory.npz")
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
    ap.add_argument("--hands-left", default=None,
                     help="left-hand stereo3d jsonl (hands/stereo_optimize.py "
                          "output; default: <recording>/derived/hands3d_left.jsonl "
                          "if present, else skipped)")
    ap.add_argument("--hands-right", default=None,
                     help="right-hand stereo3d jsonl (default: "
                          "<recording>/derived/hands3d_right.jsonl if present)")
    ap.add_argument("--mano", default="/tmp/mano_jax.npz")
    ap.add_argument("--thumb-w", type=int, default=480)
    ap.add_argument("--point-size", type=float, default=0.005)
    ap.add_argument("--frustum-scale", type=float, default=0.03)
    ap.add_argument("--trail-line-width", type=float, default=0.5,
                     help="static trail frustums drawn thin so the bright "
                          "current-frame frustums pop")
    ap.add_argument("--current-line-width", type=float, default=3.0)
    ap.add_argument("--port", type=int, default=8081)
    ap.add_argument("--share", action="store_true",
                     help="request a public viser share-tunnel URL (no SSH forward needed)")
    ap.add_argument("--trail-stride", type=int, default=50,
                     help="draw a trail frustum every Nth frame (the full path is a "
                          "polyline); avoids 2*n_frames frustum objects choking the browser")
    args = ap.parse_args()

    derived = os.path.join(args.recording, "derived")
    traj_path = args.trajectory or os.path.join(derived, "trajectory.npz")
    # Hands default into derived/ too; auto-loaded below only if the file exists,
    # so a recording with no hand data just skips them.
    if args.hands_left is None:
        args.hands_left = os.path.join(derived, "hands3d_left.jsonl")
    if args.hands_right is None:
        args.hands_right = os.path.join(derived, "hands3d_right.jsonl")
    video_path = args.video or os.path.join(args.recording, "left.mp4")
    video_right_path = args.video_right or os.path.join(args.recording, "right.mp4")

    d = np.load(traj_path)
    frame_idx = d["frame_idx"]
    pose_wxyz_xyz = d["pose_wxyz_xyz"]  # (n_frames, 7) T_wl (WORLD->CAMERA)
    points = d["points"]  # (n_points, 3) world frame
    keep_points = d["point_alive"] if "point_alive" in d else np.ones(len(points), dtype=bool)
    n_dropped = int((~keep_points).sum())
    if n_dropped:
        print(f"hiding {n_dropped}/{len(points)} landmarks dropped by outlier filtering")

    n_frames = len(frame_idx)
    fmin, fmax = 0, n_frames - 1

    # Invert every T_wl to get T_lw (camera's pose IN world frame) for plotting
    # and (if hands are present) for placing hand meshes in world.
    cam_pos_world = np.zeros((n_frames, 3))
    cam_quat_world = np.zeros((n_frames, 4))
    cam_R_world = np.zeros((n_frames, 3, 3))
    for i in range(n_frames):
        q_wl = pose_wxyz_xyz[i, :4]
        t_wl = pose_wxyz_xyz[i, 4:]
        R_wl = quat_to_matrix(q_wl)
        R_lw = R_wl.T
        t_lw = -(R_lw @ t_wl)
        cam_pos_world[i] = t_lw
        cam_quat_world[i] = matrix_to_quat(R_lw)
        cam_R_world[i] = R_lw

    # Derive the RIGHT camera's world pose per frame: T_wr = T_stereo @ T_wl.
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
        left_frame = make_frame_loader(video_path, args.thumb_w)
    else:
        print(f"[warn] no video at {video_path} -- running without thumbnail panel")

    if args.color_mode == "pixel" and "point_first_frame" in d:
        # Sample each landmark's color from the full-res frame at its exact
        # pixel. Landmarks sorted by frame, decoded SEQUENTIALLY -- random
        # per-landmark seeks were still running after 20+ min on ~20k
        # landmarks; sequential decode is what video codecs are fast at (this
        # pass is forward-only, so plain cv2 .read() is already the fast path
        # -- torchcodec's win above is for the scrubber's RANDOM access).
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

            color_cap = cv2.VideoCapture(path)
            cur_frame = -1
            img = None
            for k in order:
                target = int(first_frame[k])
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

    # --- optional hands -------------------------------------------------------
    hand_tracks = {}  # name -> (meta, {video_frame: hand dict}, color)
    if args.hands_left and os.path.exists(args.hands_left):
        hand_tracks["left"] = (*load_hand_track(args.hands_left), (1.0, 0.5, 0.3))
    if args.hands_right and os.path.exists(args.hands_right):
        hand_tracks["right"] = (*load_hand_track(args.hands_right), (0.3, 0.6, 1.0))
    M, faces = None, None
    if hand_tracks:
        import mano_jax as MJ
        M = MJ.load_mano(args.mano)
        faces = np.asarray(M["faces"])
        print(f"hands: {', '.join(hand_tracks)} (FK'd into world via left-cam pose)")

    server = viser.ViserServer(port=args.port)
    if args.share:
        share_url = server.request_share_url()
        print(f"SHARE URL: {share_url}", flush=True)
    server.scene.world_axes.visible = False
    server.scene.set_up_direction("+z")  # stage-5 world is gravity-aligned

    pc = server.scene.add_point_cloud(
        "/landmarks", points=points[keep_points], colors=point_colors[keep_points],
        point_size=args.point_size, point_shape="circle")

    # Full trajectory as ONE polyline through the left-cam centers — NOT a
    # frustum per frame (2*n_frames frustum objects tanks the browser at
    # multi-minute scale; jkerr's original was sized for ~900 frames). Sparse
    # trail frustums (every --trail-stride) give orientation cues along the path.
    seg = np.stack([cam_pos_world[:-1], cam_pos_world[1:]], axis=1)   # (n-1,2,3)
    server.scene.add_line_segments("/trail/path", seg.astype(np.float32),
                                   colors=(150, 100, 30), line_width=1.5)
    for i in range(0, n_frames, max(1, args.trail_stride)):
        server.scene.add_camera_frustum(
            f"/trail/cam_l_{i}", fov=1.4, aspect=1.2, scale=args.frustum_scale * 0.5,
            line_width=args.trail_line_width,
            color=(0.6, 0.4, 0.1), wxyz=cam_quat_world[i], position=cam_pos_world[i])
    current_frustum_l = server.scene.add_camera_frustum(
        "/current_left", fov=1.4, aspect=1.2, scale=args.frustum_scale,
        line_width=args.current_line_width,
        color=(1.0, 0.6, 0.0), wxyz=cam_quat_world[0], position=cam_pos_world[0])
    current_frustum_r = server.scene.add_camera_frustum(
        "/current_right", fov=1.4, aspect=1.2, scale=args.frustum_scale,
        line_width=args.current_line_width,
        color=(0.0, 0.8, 1.0), wxyz=cam_quat_world_r[0], position=cam_pos_world_r[0])

    n_v = int(np.asarray(faces).max()) + 1 if faces is not None else 0
    mesh_h = {name: server.scene.add_mesh_simple(
                  f"/hand_{name}", np.zeros((n_v, 3), np.float32), faces,
                  color=color, flat_shading=False, visible=False)
              for name, (_, _, color) in hand_tracks.items()}

    gui_play = server.gui.add_checkbox("play", False)
    gui_frame = server.gui.add_slider("frame", fmin, fmax, 1, fmin)
    gui_info = server.gui.add_text("info", "", disabled=True)
    gui_img = server.gui.add_image(left_frame(frame_idx[fmin]), label="left eye",
                                    format="jpeg") if left_frame else None

    _update_lock = threading.Lock()

    def update(_=None):
        with _update_lock:
            fr = int(gui_frame.value)
            vframe = int(frame_idx[fr])
            img = left_frame(vframe) if left_frame is not None else None
            lines = [
                f"pose slot {fr} (video frame {vframe})",
                f"  cam position (world): {cam_pos_world[fr]}",
            ]
            hand_verts = {}
            for name, (meta, frames, _) in hand_tracks.items():
                h = frames.get(vframe)
                if h is None:
                    continue
                v_cam = hand_mesh_cam(M, faces, h, meta["beta_opt"], meta["mirror"])
                hand_verts[name] = (v_cam @ cam_R_world[fr].T
                                    + cam_pos_world[fr][None, :]).astype(np.float32)
            with server.atomic():
                current_frustum_l.wxyz = cam_quat_world[fr]
                current_frustum_l.position = cam_pos_world[fr]
                current_frustum_r.wxyz = cam_quat_world_r[fr]
                current_frustum_r.position = cam_pos_world_r[fr]
                for name in hand_tracks:
                    if name in hand_verts:
                        mesh_h[name].vertices = hand_verts[name]
                        mesh_h[name].visible = True
                    else:
                        mesh_h[name].visible = False
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
