"""Generic Viser visualizer for data_processing outputs: the VIO camera
trajectory and optional point cloud, loop candidates, and hand meshes
(FK'd from each hand's stereo-optimized MANO params, placed in world via the
left camera's pose at that frame: hand_world(t) = T_cam_world(t) @
hand_cam(t)).

Optional landmarks are colored by their actual video pixel at first observation
(--color-mode pixel, default) or a depth colormap. Left (orange) + right
(cyan) stereo frustums per frame, right pose via T_wr = T_stereo @ T_wl.
World is gravity-aligned (+z up), so viser's up-direction is +z. Static
trail frustums are drawn thin so the bright current-frame frustums pop.

Video thumbnails are decoded by torchcodec on CUDA when available, then
downsampled on-device with torchvision before being sent to viser.

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


WORLD_UP = np.array([0.0, 0.0, 1.0])


def presentation_camera_target(camera_center, camera_to_world_rotation, mode):
    """Return gravity-stable viewer (position, look_at, up) for one ego pose."""
    center = np.asarray(camera_center, np.float64)
    rotation = np.asarray(camera_to_world_rotation, np.float64)
    forward = rotation[:, 2]
    forward /= max(np.linalg.norm(forward), 1e-12)
    heading = forward - WORLD_UP * np.dot(forward, WORLD_UP)
    heading_norm = np.linalg.norm(heading)
    if heading_norm < 1e-6:
        heading = rotation[:, 0]
        heading -= WORLD_UP * np.dot(heading, WORLD_UP)
        heading_norm = np.linalg.norm(heading)
    heading /= max(heading_norm, 1e-12)

    if mode == "Follow":
        position = center - 0.75 * heading + 0.24 * WORLD_UP
        look_at = center + 0.55 * forward + 0.04 * WORLD_UP
    elif mode == "Ego":
        position = center - 0.025 * forward
        look_at = center + forward
    else:
        raise ValueError(f"unsupported presentation camera mode: {mode}")
    return position, look_at, WORLD_UP.copy()


def overview_camera_target(camera_centers):
    """Return a stable three-quarter overview that frames the trajectory."""
    centers = np.asarray(camera_centers, np.float64)
    center = np.median(centers, axis=0)
    radius = max(float(np.percentile(
        np.linalg.norm(centers - center, axis=1), 95)), 0.75)
    direction = np.array([1.15, -1.15, 0.85])
    direction /= np.linalg.norm(direction)
    position = center + direction * (2.7 * radius)
    return position, center, WORLD_UP.copy()


def smooth_camera_target(previous, target, dt, time_constant=0.22):
    """Exponential camera smoothing independent of playback update rate."""
    if previous is None or dt <= 0.0:
        return tuple(np.asarray(value, np.float64) for value in target)
    alpha = 1.0 - np.exp(-dt / max(time_constant, 1e-6))
    smoothed = tuple(
        np.asarray(old) + alpha * (np.asarray(new) - np.asarray(old))
        for old, new in zip(previous, target)
    )
    up = smoothed[2] / max(np.linalg.norm(smoothed[2]), 1e-12)
    return smoothed[0], smoothed[1], up


def bidirectional_time_smooth(values, times, time_constant):
    """Zero-phase-ish exponential smoothing using actual sample intervals."""
    values = np.asarray(values, np.float64)
    times = np.asarray(times, np.float64)
    if values.shape[0] != len(times):
        raise ValueError("smoothing values and times must have matching lengths")
    if len(times) < 2 or time_constant <= 0.0:
        return values.copy()
    if np.any(np.diff(times) <= 0):
        raise ValueError("smoothing times must be strictly increasing")

    forward = values.copy()
    for index in range(1, len(times)):
        alpha = 1.0 - np.exp(
            -(times[index] - times[index - 1]) / time_constant)
        forward[index] = (
            forward[index - 1] + alpha * (values[index] - forward[index - 1]))

    backward = values.copy()
    for index in range(len(times) - 2, -1, -1):
        alpha = 1.0 - np.exp(
            -(times[index + 1] - times[index]) / time_constant)
        backward[index] = (
            backward[index + 1]
            + alpha * (values[index] - backward[index + 1]))
    return 0.5 * (forward + backward)


def smooth_presentation_trajectory(centers, rotations, times, time_constant):
    """Smooth only the virtual viewer path, preserving the estimated poses."""
    centers = bidirectional_time_smooth(centers, times, time_constant)
    rotations = np.asarray(rotations, np.float64).copy()
    forward = bidirectional_time_smooth(
        rotations[:, :, 2], times, time_constant)
    right = bidirectional_time_smooth(
        rotations[:, :, 0], times, time_constant)
    forward /= np.maximum(
        np.linalg.norm(forward, axis=1, keepdims=True), 1e-12)
    right /= np.maximum(
        np.linalg.norm(right, axis=1, keepdims=True), 1e-12)
    rotations[:, :, 2] = forward
    rotations[:, :, 0] = right
    return centers, rotations


def make_frame_loader(video_path, thumb_w, requested_device="auto"):
    """Build a cached thumbnail loader and return it with the video's FPS.

    torchcodec handles random access and codec buffering. On CUDA, the decoded
    frame stays on the GPU through the resize; only the small thumbnail crosses
    to host memory for viser.
    """
    import torch
    from torchvision.transforms.v2 import functional as tvf
    from torchcodec.decoders import VideoDecoder

    if requested_device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = requested_device
    if device == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA video decode requested but CUDA is unavailable; using CPU")
        device = "cpu"

    def new_decoder(decoder_device):
        return VideoDecoder(video_path, device=decoder_device)

    try:
        dec = new_decoder(device)
    except RuntimeError as exc:
        if device != "cuda":
            raise
        print(f"[warn] CUDA video decoder unavailable ({exc}); using CPU")
        device = "cpu"
        dec = new_decoder(device)

    fps = float(getattr(dec.metadata, "average_fps", 0.0) or 30.0)
    num_frames = getattr(dec.metadata, "num_frames", None)
    print(f"video thumbnails: torchcodec decode + torchvision resize on {device} "
          f"({thumb_w}px wide, {fps:g} fps)")
    lock = threading.Lock()  # torchcodec decoders are not thread-safe; the
    # play loop and the scrub callback run on different threads.
    gpu_dec = dec if device == "cuda" else None
    cpu_dec = dec if device == "cpu" else None
    cuda_decode_failed = False

    @functools.lru_cache(maxsize=128)
    def _decode(fr, random_access):
        nonlocal gpu_dec, cpu_dec, cuda_decode_failed
        if num_frames is not None:
            fr = min(max(fr, 0), int(num_frames) - 1)
        with lock:
            # Rapid nonlocal seeks can poison the NVDEC/FFmpeg decoder state.
            # Keep those seeks on an independent CPU decoder; playback remains
            # on the undisturbed CUDA decoder.
            use_cpu = random_access or device == "cpu" or cuda_decode_failed
            if use_cpu and cpu_dec is None:
                cpu_dec = new_decoder("cpu")
            if not use_cpu and gpu_dec is None:
                gpu_dec = new_decoder("cuda")
            active_dec = cpu_dec if use_cpu else gpu_dec
            try:
                f = active_dec.get_frames_in_range(
                    start=fr, stop=fr + 1).data[0]
            except RuntimeError as exc:
                if use_cpu:
                    cpu_dec = new_decoder("cpu")
                    f = cpu_dec.get_frames_in_range(
                        start=fr, stop=fr + 1).data[0]
                else:
                    print(f"[warn] CUDA seek failed; using CPU for frame {fr}: {exc}")
                    gpu_dec = None
                    cuda_decode_failed = True
                    if cpu_dec is None:
                        cpu_dec = new_decoder("cpu")
                    f = cpu_dec.get_frames_in_range(
                        start=fr, stop=fr + 1).data[0]
            _, h, w = f.shape
            th = max(1, round(h * thumb_w / w))
            thumb = tvf.resize(f, [th, thumb_w], antialias=True)
            img = thumb.permute(1, 2, 0).contiguous().cpu().numpy()
        return img.astype(np.uint8, copy=False)

    return (lambda fr, random_access=False:
            _decode(int(fr), bool(random_access))), fps


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


def hand_mesh_cam(mano_mesh_fn, h, beta, mirror):
    """FK one hand's MANO mesh into the LEFT-FISHEYE CAMERA frame. Returns (778,3)."""
    quaternions = np.asarray(h["quat"], np.float32)
    beta = np.asarray(beta, np.float32)
    v = np.array(mano_mesh_fn(quaternions, beta))                  # (778,3) virtual
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
    ap.add_argument(
        "--loop-candidates",
        default=None,
        help="optional JSONL loop candidates to overlay as paired frustums",
    )
    ap.add_argument("--video", default=None, help="default: <recording>/left.mp4")
    ap.add_argument("--video-right", default=None,
                     help="default: <recording>/right.mp4 -- only needed for "
                          "--color-mode pixel landmarks whose first observation "
                          "was in the right eye")
    ap.add_argument("--color-mode", choices=("pixel", "depth"), default="pixel",
                     help="pixel: sample each landmark's color from the actual "
                          "video frame/eye/px of its first observation (needs "
                          "trajectory point_first_* fields). depth: percentile "
                          "colormap.")
    ap.add_argument("--no-color", action="store_true",
                    help="skip video decoding and render landmarks in one color")
    ap.add_argument("--hands-left", default=None,
                     help="left-hand stereo3d jsonl (hands/stereo_optimize.py "
                          "output; default: <recording>/derived/hands3d_left.jsonl "
                          "if present, else skipped)")
    ap.add_argument("--hands-right", default=None,
                     help="right-hand stereo3d jsonl (default: "
                          "<recording>/derived/hands3d_right.jsonl if present)")
    ap.add_argument("--mano", default="/tmp/mano_jax.npz")
    ap.add_argument("--thumb-w", type=int, default=320,
                    help="thumbnail width sent to viser (default: 320)")
    ap.add_argument("--video-device", choices=("auto", "cuda", "cpu"),
                    default="auto",
                    help="torchcodec decode device; auto uses CUDA when available")
    ap.add_argument("--playback-fps", type=float, default=40.0,
                    help="video timeline rate (default: 40)")
    ap.add_argument("--point-size", type=float, default=0.005)
    ap.add_argument("--frustum-scale", type=float, default=0.04)
    ap.add_argument("--trail-line-width", type=float, default=0.35,
                     help="static trail frustums drawn thin so the bright "
                          "current-frame frustums pop")
    ap.add_argument("--trail-path-width", type=float, default=1.25,
                     help="width of the connected camera path")
    ap.add_argument("--current-line-width", type=float, default=2.5)
    ap.add_argument("--camera-eyes", choices=("left", "both"), default="left",
                    help="initial camera frustums to render (default: left)")
    ap.add_argument("--trail-stride", type=int, default=80,
                    help="render one static trail frustum every N poses")
    ap.add_argument(
        "--view-mode",
        choices=("Overview", "Follow", "Ego"),
        default="Follow",
        help="initial viewer camera mode (default: Follow)",
    )
    ap.add_argument(
        "--follow-smoothing",
        type=float,
        default=0.16,
        help="follow-camera exponential smoothing time in seconds",
    )
    ap.add_argument(
        "--follow-path-smoothing",
        type=float,
        default=0.45,
        help="zero-lag smoothing applied only to the virtual viewer path",
    )
    ap.add_argument("--port", type=int, default=8081)
    args = ap.parse_args()
    if args.trail_stride < 1:
        ap.error("--trail-stride must be at least 1")
    if args.playback_fps <= 0:
        ap.error("--playback-fps must be positive")
    if args.follow_smoothing < 0:
        ap.error("--follow-smoothing must be nonnegative")
    if args.follow_path_smoothing < 0:
        ap.error("--follow-path-smoothing must be nonnegative")
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
    if "interpolation_frame_time_s" in d:
        frame_time_s = np.asarray(d["interpolation_frame_time_s"], np.float64)
        if (frame_time_s.shape != frame_idx.shape
                or np.any(np.diff(frame_time_s) <= 0)):
            raise ValueError("trajectory interpolation frame times are invalid")
        frame_time_s = frame_time_s - frame_time_s[0]
        print("playback timeline: observed camera frame times")
    else:
        frame_time_s = (
            np.asarray(frame_idx, np.float64) - float(frame_idx[0])
        ) / args.playback_fps
        print(f"playback timeline: uniform {args.playback_fps:g} fps fallback")
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

    presentation_pos_world, presentation_R_world = (
        smooth_presentation_trajectory(
            cam_pos_world,
            cam_R_world,
            frame_time_s,
            args.follow_path_smoothing,
        ))

    left_frame = None
    if os.path.exists(video_path):
        left_frame, detected_video_fps = make_frame_loader(
            video_path, args.thumb_w, args.video_device)
        if abs(detected_video_fps - args.playback_fps) > 0.01:
            print(f"playback: {args.playback_fps:g} fps "
                  f"(container reports {detected_video_fps:g} fps)")
    else:
        print(f"[warn] no video at {video_path} -- running without thumbnail panel")

    if len(points) == 0:
        point_colors = np.empty((0, 3), np.float32)
        print("trajectory has no landmarks; showing cameras and hands only")
    elif args.no_color:
        point_colors = np.tile(
            np.array([[0.62, 0.68, 0.72]], np.float32), (len(points), 1))
        print(f"{len(points)} landmarks shown without video colors")
    elif args.color_mode == "pixel" and "point_first_frame" in d:
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
        hand_tracks["left"] = (*load_hand_track(args.hands_left), (0.3, 0.6, 1.0))
    if args.hands_right and os.path.exists(args.hands_right):
        hand_tracks["right"] = (*load_hand_track(args.hands_right), (1.0, 0.4, 0.25))
    M, faces, mano_mesh_fn = None, None, None
    if hand_tracks:
        # The visualization FK is tiny. Keep JAX off the GPUs so its allocator
        # cannot compete with torchcodec/NVDEC for thumbnail decode memory.
        os.environ["JAX_PLATFORMS"] = "cpu"
        import jax
        import jax.numpy as jnp
        import jaxlie
        import mano_jax as MJ

        M = MJ.load_mano(args.mano)
        faces = np.asarray(M["faces"])

        @jax.jit
        def mano_mesh_fn(quaternions, beta):
            rotations = jaxlie.SO3(jnp.asarray(quaternions)).as_matrix()
            return MJ.mano_mesh_R(M, rotations, jnp.asarray(beta))

        # Compile and synchronize once. Both hands have identical input shapes
        # and dtypes and therefore reuse this executable for every frame.
        warm_name = next(iter(hand_tracks))
        warm_meta, warm_frames, _ = hand_tracks[warm_name]
        warm_hand = next(iter(warm_frames.values()))
        hand_mesh_cam(
            mano_mesh_fn,
            warm_hand,
            warm_meta["beta_opt"],
            warm_meta["mirror"],
        )
        print(f"hands: {', '.join(hand_tracks)} (CPU-JIT MANO FK; "
              "compiled once; placed via left-cam pose)")
    server = viser.ViserServer(port=args.port)
    server.scene.world_axes.visible = False
    server.scene.set_up_direction("+z")  # stage-5 world is gravity-aligned

    path_span = np.ptp(cam_pos_world[:, :2], axis=0)
    grid_size = max(6.0, float(np.ceil(np.max(path_span) + 4.0)))
    floor_height = float(np.median(cam_pos_world[:, 2]) - 1.778)
    floor_grid = server.scene.add_grid(
        "/floor",
        width=grid_size,
        height=grid_size,
        plane="xy",
        cell_color=(190, 195, 201),
        cell_thickness=0.35,
        cell_size=0.5,
        section_color=(158, 166, 175),
        section_thickness=0.7,
        section_size=1.0,
        fade_distance=max(grid_size * 0.65, 4.0),
        fade_strength=1.5,
        fade_from="camera",
        shadow_opacity=0.08,
        position=(
            float(np.median(cam_pos_world[:, 0])),
            float(np.median(cam_pos_world[:, 1])),
            floor_height,
        ),
    )

    if keep_points.any():
        server.scene.add_point_cloud(
            "/landmarks", points=points[keep_points],
            colors=point_colors[keep_points],
            point_size=args.point_size, point_shape="circle")

    trail_segments = np.stack(
        [cam_pos_world[:-1], cam_pos_world[1:]], axis=1).astype(np.float32)
    server.scene.add_line_segments(
        "/trail/path",
        points=trail_segments,
        line_width=args.trail_path_width,
        colors=(214, 145, 42),
    )
    if args.loop_candidates:
        frame_to_slot = {
            int(frame): index for index, frame in enumerate(frame_idx)}
        palette = [
            (255, 77, 109),
            (0, 201, 167),
            (89, 161, 255),
            (255, 190, 64),
            (190, 110, 255),
            (0, 190, 230),
            (255, 118, 72),
            (120, 220, 90),
        ]
        with open(args.loop_candidates) as f:
            loop_candidates = [
                json.loads(line) for line in f if line.strip()]
        loop_images = {}
        for candidate_index, candidate in enumerate(loop_candidates):
            slots = []
            for key in ("frame_a", "frame_b"):
                candidate_frame = int(candidate[key])
                if candidate_frame in frame_to_slot:
                    slots.append(frame_to_slot[candidate_frame])
                else:
                    slots.append(int(np.argmin(
                        np.abs(frame_idx - candidate_frame))))
            slot_a, slot_b = slots
            color = palette[candidate_index % len(palette)]
            server.scene.add_line_segments(
                f"/loops/{candidate_index:02d}/chord",
                points=np.asarray([[
                    cam_pos_world[slot_a],
                    cam_pos_world[slot_b],
                ]], np.float32),
                line_width=4.0,
                colors=color,
            )
            for endpoint, slot in (("a", slot_a), ("b", slot_b)):
                video_frame = int(frame_idx[slot])
                if video_frame not in loop_images and left_frame is not None:
                    try:
                        loop_images[video_frame] = left_frame(
                            video_frame, random_access=True)
                    except (RuntimeError, IndexError) as exc:
                        print(
                            f"[warn] loop thumbnail unavailable for frame "
                            f"{video_frame}: {exc}")
                        loop_images[video_frame] = None
                server.scene.add_camera_frustum(
                    f"/loops/{candidate_index:02d}/{endpoint}",
                    fov=1.4,
                    aspect=1.2,
                    scale=args.frustum_scale * 2.5,
                    line_width=3.0,
                    color=np.asarray(color, np.float32) / 255.0,
                    image=loop_images.get(video_frame),
                    format="jpeg",
                    jpeg_quality=85,
                    wxyz=cam_quat_world[slot],
                    position=cam_pos_world[slot],
                )
        print(
            f"{len(loop_candidates)} loop candidates overlaid from "
            f"{args.loop_candidates}")

    # Sparse, small history frustums preserve heading context without competing
    # with the textured current left camera.
    trail_frustums_l = []
    trail_frustums_r = []
    show_stereo_initially = args.camera_eyes == "both"
    for i in range(0, n_frames, args.trail_stride):
        trail_frustums_l.append(server.scene.add_camera_frustum(
            f"/trail/cam_l_{i}", fov=1.4, aspect=1.2,
            scale=args.frustum_scale * 0.28,
            line_width=args.trail_line_width,
            color=(0.42, 0.31, 0.16),
            wxyz=cam_quat_world[i],
            position=cam_pos_world[i],
            cast_shadow=False,
        ))
        trail_frustums_r.append(server.scene.add_camera_frustum(
            f"/trail/cam_r_{i}", fov=1.4, aspect=1.2,
            scale=args.frustum_scale * 0.28,
            line_width=args.trail_line_width,
            color=(0.12, 0.32, 0.38),
            wxyz=cam_quat_world_r[i],
            position=cam_pos_world_r[i],
            visible=show_stereo_initially,
            cast_shadow=False,
        ))

    initial_image = None
    if left_frame is not None:
        try:
            initial_image = left_frame(int(frame_idx[fmin]))
        except (RuntimeError, IndexError) as exc:
            print(f"[warn] initial thumbnail unavailable: {exc}")
    current_frustum_l = server.scene.add_camera_frustum(
        "/current_left", fov=1.4, aspect=1.2,
        scale=args.frustum_scale * 2.25,
        line_width=args.current_line_width,
        color=(1.0, 0.58, 0.08),
        image=initial_image,
        format="jpeg",
        jpeg_quality=85,
        wxyz=cam_quat_world[0],
        position=cam_pos_world[0],
        cast_shadow=False,
    )
    current_frustum_r = server.scene.add_camera_frustum(
        "/current_right", fov=1.4, aspect=1.2, scale=args.frustum_scale * 0.72,
        line_width=args.current_line_width,
        color=(0.0, 0.72, 0.88),
        wxyz=cam_quat_world_r[0],
        position=cam_pos_world_r[0],
        visible=show_stereo_initially,
        cast_shadow=False,
    )

    n_v = int(np.asarray(faces).max()) + 1 if faces is not None else 0
    mesh_h = {name: server.scene.add_mesh_simple(
                  f"/hand_{name}", np.zeros((n_v, 3), np.float32), faces,
                  color=color, flat_shading=False, visible=False,
                  cast_shadow=True, receive_shadow=True)
              for name, (_, _, color) in hand_tracks.items()}

    gui_view = server.gui.add_dropdown(
        "View", ("Overview", "Follow", "Ego"),
        initial_value=args.view_mode)
    gui_play = server.gui.add_checkbox("Play", False)
    gui_speed = server.gui.add_slider(
        "Speed", 0.25, 2.0, 0.25, 1.0)
    gui_frame = server.gui.add_slider("Frame", fmin, fmax, 1, fmin)
    gui_show_stereo = server.gui.add_checkbox(
        "Stereo rig", show_stereo_initially)
    gui_show_history = server.gui.add_checkbox("Camera history", True)
    gui_show_floor = server.gui.add_checkbox("Floor grid", True)
    gui_show_hands = server.gui.add_checkbox(
        "Hands", bool(hand_tracks))
    gui_frame_scene = server.gui.add_button("Frame scene")

    _update_lock = threading.Lock()
    _timeline_lock = threading.Lock()
    _timeline_state = {
        "playhead_s": float(frame_time_s[fmin]),
        "rendered_slot": fmin,
    }
    _camera_states = {}
    overview_target = overview_camera_target(cam_pos_world)

    def set_client_camera(
            client_id, client, target, snap=False, already_atomic=False):
        now = time.monotonic()
        state = _camera_states.get(client_id)
        previous = None if state is None else state["target"]
        dt = 0.0 if state is None else now - state["time"]
        if snap or args.follow_smoothing == 0.0:
            smoothed = target
        else:
            smoothed = smooth_camera_target(
                previous, target, dt, args.follow_smoothing)
        def assign():
            client.camera.position = smoothed[0]
            client.camera.look_at = smoothed[1]
            client.camera.up_direction = smoothed[2]
        if already_atomic:
            assign()
        else:
            with client.atomic():
                assign()
        _camera_states[client_id] = {
            "target": smoothed,
            "time": now,
        }

    def set_all_client_cameras(
            target, snap=False, already_atomic=False):
        for client_id, client in server.get_clients().items():
            set_client_camera(
                client_id,
                client,
                target,
                snap=snap,
                already_atomic=already_atomic,
            )

    def update_presentation_cameras(
            frame_slot, snap=False, already_atomic=False):
        mode = gui_view.value
        if mode == "Overview":
            return
        target = presentation_camera_target(
            presentation_pos_world[frame_slot],
            presentation_R_world[frame_slot],
            mode,
        )
        set_all_client_cameras(
            target,
            snap=snap or mode == "Ego",
            already_atomic=already_atomic,
        )

    def update_layer_visibility():
        history = bool(gui_show_history.value)
        stereo = bool(gui_show_stereo.value)
        ego = gui_view.value == "Ego"
        for handle in trail_frustums_l:
            handle.visible = history
        for handle in trail_frustums_r:
            handle.visible = history and stereo
        current_frustum_l.visible = not ego
        current_frustum_r.visible = stereo and not ego
        floor_grid.visible = bool(gui_show_floor.value)

    def render_frame(frame_slot, random_access=False, publish_slider=False):
        with _update_lock:
            fr = int(frame_slot)
            vframe = int(frame_idx[fr])
            img = None
            if left_frame is not None:
                try:
                    img = left_frame(vframe, random_access=random_access)
                except (RuntimeError, IndexError) as exc:
                    # Keep the previous thumbnail. A bad video seek must not
                    # take down trajectory/hand scrubbing or the play loop.
                    print(f"[warn] thumbnail unavailable for frame {vframe}: {exc}")
            hand_verts = {}
            if gui_show_hands.value:
                for name, (meta, frames, _) in hand_tracks.items():
                    h = frames.get(vframe)
                    if h is None:
                        continue
                    v_cam = hand_mesh_cam(
                        mano_mesh_fn, h, meta["beta_opt"], meta["mirror"])
                    hand_verts[name] = (
                        v_cam @ cam_R_world[fr].T
                        + cam_pos_world[fr][None, :]
                    ).astype(np.float32)
            with server.atomic():
                if publish_slider:
                    gui_frame.value = fr
                current_frustum_l.wxyz = cam_quat_world[fr]
                current_frustum_l.position = cam_pos_world[fr]
                current_frustum_r.wxyz = cam_quat_world_r[fr]
                current_frustum_r.position = cam_pos_world_r[fr]
                for name in hand_tracks:
                    if name in hand_verts and gui_show_hands.value:
                        mesh_h[name].vertices = hand_verts[name]
                        mesh_h[name].visible = True
                    else:
                        mesh_h[name].visible = False
                if img is not None:
                    current_frustum_l.image = img
                update_presentation_cameras(fr, already_atomic=True)

    @gui_frame.on_update
    def _on_frame_scrub(event):
        # Server-side slider publication during playback has no client_id.
        # Only a real client scrub is allowed to seek the central clock.
        if getattr(event, "client_id", None) is None:
            return
        frame_slot = int(gui_frame.value)
        with _timeline_lock:
            _timeline_state["playhead_s"] = float(frame_time_s[frame_slot])
            _timeline_state["rendered_slot"] = frame_slot
        render_frame(frame_slot, random_access=True)

    @gui_play.on_update
    def _on_play_toggle(_event):
        gui_frame.disabled = bool(gui_play.value)

    @gui_view.on_update
    def _(_event):
        _camera_states.clear()
        update_layer_visibility()
        if gui_view.value == "Overview":
            set_all_client_cameras(overview_target, snap=True)
        else:
            with _timeline_lock:
                frame_slot = _timeline_state["rendered_slot"]
            update_presentation_cameras(frame_slot, snap=True)

    @gui_show_stereo.on_update
    def _(_event):
        update_layer_visibility()

    @gui_show_history.on_update
    def _(_event):
        update_layer_visibility()

    @gui_show_floor.on_update
    def _(_event):
        update_layer_visibility()

    @gui_show_hands.on_update
    def _(_event):
        with _timeline_lock:
            frame_slot = _timeline_state["rendered_slot"]
        render_frame(frame_slot, random_access=True)

    @gui_frame_scene.on_click
    def _(_event):
        gui_view.value = "Overview"
        _camera_states.clear()
        set_all_client_cameras(overview_target, snap=True)

    @server.on_client_connect
    def _(client):
        client_id = getattr(client, "client_id", id(client))
        if gui_view.value == "Overview":
            set_client_camera(
                client_id, client, overview_target, snap=True)
        else:
            with _timeline_lock:
                frame_slot = _timeline_state["rendered_slot"]
            target = presentation_camera_target(
                presentation_pos_world[frame_slot],
                presentation_R_world[frame_slot],
                gui_view.value,
            )
            set_client_camera(client_id, client, target, snap=True)

    update_layer_visibility()
    render_frame(fmin)

    print(f"viser running on port {args.port} -- forward it: "
          f"ssh -L {args.port}:localhost:{args.port} <host>")

    last_tick_s = time.monotonic()
    was_playing = False
    while True:
        now_s = time.monotonic()
        playing = bool(gui_play.value)
        target_slot = None
        with _timeline_lock:
            if playing and was_playing:
                _timeline_state["playhead_s"] += (
                    now_s - last_tick_s) * float(gui_speed.value)
                duration_s = float(frame_time_s[fmax] - frame_time_s[fmin])
                if (_timeline_state["playhead_s"]
                        > float(frame_time_s[fmax])):
                    _timeline_state["playhead_s"] = (
                        float(frame_time_s[fmin])
                        + (_timeline_state["playhead_s"]
                           - float(frame_time_s[fmin]))
                        % max(duration_s, 1e-12)
                    )
                candidate_slot = int(np.searchsorted(
                    frame_time_s,
                    _timeline_state["playhead_s"],
                    side="right",
                ) - 1)
                candidate_slot = max(fmin, min(candidate_slot, fmax))
                if candidate_slot != _timeline_state["rendered_slot"]:
                    _timeline_state["rendered_slot"] = candidate_slot
                    target_slot = candidate_slot
        last_tick_s = now_s
        was_playing = playing
        if target_slot is not None:
            # The central clock renders directly. The slider is published in
            # the same transaction and its server-side callback is ignored.
            render_frame(target_slot, publish_slider=True)
        time.sleep(0.005)


if __name__ == "__main__":
    main()
