"""Generate stereo VGGT-Omega window measurements for the VIO pose graph.

This stage intentionally contains no JAX dependencies. Run it in the dedicated
PyTorch environment used for VGGT-Omega; run ``vio_vggt_pose_graph.py`` in the
repository's ``jaxgpu`` environment afterwards.

The raw cameras are fisheye. ``prepare`` renders a common-orientation pinhole
view for each eye: the left virtual camera uses the left camera frame, and the
right virtual camera uses the calibrated left-to-right rotation. This gives
Omega rectilinear images while preserving a calibrated stereo rig.

Each inference call receives dense left images and either dense or periodically
sampled right images. Track windows overlap in time. Optional loop windows
concatenate neighborhoods around two loop matches. Only camera predictions are
cached here; the downstream graph owns metric scale, IMU fusion, and global
consistency.

Examples (on sphynx):

    conda activate vggtomega
    CUDA_VISIBLE_DEVICES=3 python vio_vggt_window_infer.py prepare ../../testimu
    CUDA_VISIBLE_DEVICES=3 python vio_vggt_window_infer.py infer ../../testimu \
        --checkpoint /path/to/vggt_omega_1b_512.pt
"""

import argparse
import concurrent.futures
import json
import math
import os
import queue
import threading
import time
from pathlib import Path

import cv2
import numpy as np


FORMAT_TAG = "hteng-camera-vggt-windows/1"


def recording_product(recording, filename):
    derived = os.path.join(recording, "derived", filename)
    if os.path.exists(derived):
        return derived
    return os.path.join(recording, filename)


def load_recording_calibration(recording):
    rec = json.load(open(os.path.join(recording, "recording.json")))
    left_serial = rec["left"]["serial"]
    right_serial = rec["right"]["serial"]

    def intrinsics(serial):
        path = os.path.join(recording, f"calib_{serial}.json")
        data = json.load(open(path))["intrinsics"]
        return (
            np.asarray(data["K"], np.float64),
            np.asarray(data["dist"], np.float64),
        )

    K_left, D_left = intrinsics(left_serial)
    K_right, D_right = intrinsics(right_serial)
    stereo_path = os.path.join(
        recording, f"stereo_{left_serial}_{right_serial}.json")
    stereo = json.load(open(stereo_path))
    return (
        rec,
        (K_left, D_left),
        (K_right, D_right),
        np.asarray(stereo["R"], np.float64),
        np.asarray(stereo["t"], np.float64).reshape(3),
    )


def make_fisheye_map(K, distortion, virtual_to_camera, size, fov_deg):
    """Map a square virtual pinhole image into a calibrated fisheye image."""
    focal = (size / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    center = (size - 1) / 2.0
    yy, xx = np.meshgrid(
        np.arange(size, dtype=np.float64),
        np.arange(size, dtype=np.float64),
        indexing="ij",
    )
    rays_virtual = np.stack([
        (xx - center) / focal,
        (yy - center) / focal,
        np.ones_like(xx),
    ], axis=-1)
    rays_virtual /= np.linalg.norm(rays_virtual, axis=-1, keepdims=True)
    rays_camera = rays_virtual.reshape(-1, 3) @ virtual_to_camera.T
    pixels, _ = cv2.fisheye.projectPoints(
        rays_camera.reshape(-1, 1, 3),
        np.zeros((3, 1)),
        np.zeros((3, 1)),
        K,
        distortion.reshape(4, 1),
    )
    pixels = pixels.reshape(size, size, 2).astype(np.float32)
    K_virtual = np.array([
        [focal, 0.0, center],
        [0.0, focal, center],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    return pixels[..., 0], pixels[..., 1], K_virtual


def video_metadata(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"failed to open {path}")
    out = {
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": float(cap.get(cv2.CAP_PROP_FPS)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    return out


def choose_frames(recording, n_frames, native_fps, target_fps, max_frames):
    step = max(1.0, native_fps / target_fps)
    selected = np.unique(np.rint(
        np.arange(0.0, n_frames, step)).astype(np.int64))
    selected = selected[selected < n_frames]
    imu_path = recording_product(recording, "imu_relative.npz")
    if os.path.exists(imu_path):
        imu = np.load(imu_path)
        valid = set(imu["frame_idx"][imu["frame_valid"]].astype(int).tolist())
        selected = np.asarray(
            [frame for frame in selected if int(frame) in valid],
            dtype=np.int64,
        )
    if max_frames is not None:
        selected = selected[:max_frames]
    return selected


def fisheye_map_to_grid(map_x, map_y, source_width, source_height, device):
    """Convert OpenCV pixel-coordinate remap arrays to a Torch sampling grid."""
    import torch

    x = torch.as_tensor(map_x, dtype=torch.float32, device=device)
    y = torch.as_tensor(map_y, dtype=torch.float32, device=device)
    x = 2.0 * x / (source_width - 1) - 1.0
    y = 2.0 * y / (source_height - 1) - 1.0
    return torch.stack((x, y), dim=-1).unsqueeze(0)


def remap_fisheye_batch(frames, grid):
    """Remap an NCHW uint8 RGB batch to normalized float images on-device."""
    import torch.nn.functional as F

    return F.grid_sample(
        frames.float().div_(255.0),
        grid.expand(len(frames), -1, -1, -1),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )


class DirectStereoVideoLoader:
    """Decode and rectify requested stereo frames without leaving the GPU."""

    def __init__(self, manifest, device, decode_batch_size):
        import torch
        from torchcodec.decoders import VideoDecoder

        self.torch = torch
        self.device = torch.device(device)
        self.decode_batch_size = decode_batch_size
        recording = manifest["recording"]
        _, left_calib, right_calib, R_stereo, _ = (
            load_recording_calibration(recording))
        left_path = os.path.join(recording, "left.mp4")
        right_path = os.path.join(recording, "right.mp4")
        left_meta = video_metadata(left_path)
        right_meta = video_metadata(right_path)
        image_size = int(manifest["image_size"])
        fov_deg = float(manifest["fov_deg"])
        map_lx, map_ly, _ = make_fisheye_map(
            *left_calib, np.eye(3), image_size, fov_deg)
        map_rx, map_ry, _ = make_fisheye_map(
            *right_calib, R_stereo, image_size, fov_deg)
        self.grids = (
            fisheye_map_to_grid(
                map_lx, map_ly, left_meta["width"], left_meta["height"],
                self.device),
            fisheye_map_to_grid(
                map_rx, map_ry, right_meta["width"], right_meta["height"],
                self.device),
        )
        self.decoders = (
            VideoDecoder(left_path, device=self.device),
            VideoDecoder(right_path, device=self.device),
        )
        self.image_size = image_size

    def load(self, frame_idx, eye, output_size):
        import torch.nn.functional as F

        frame_idx = np.asarray(frame_idx, np.int64)
        eye = np.asarray(eye, np.int8)
        output = self.torch.empty(
            (len(frame_idx), 3, output_size, output_size),
            dtype=self.torch.float32,
            device=self.device,
        )
        for eye_id in (0, 1):
            positions = np.flatnonzero(eye == eye_id)
            for start in range(0, len(positions), self.decode_batch_size):
                selected_positions = positions[
                    start:start + self.decode_batch_size]
                selected_frames = frame_idx[selected_positions].tolist()
                decoded = self.decoders[eye_id].get_frames_at(
                    selected_frames).data
                remapped = remap_fisheye_batch(decoded, self.grids[eye_id])
                if output_size != self.image_size:
                    remapped = F.interpolate(
                        remapped,
                        size=(output_size, output_size),
                        mode="bicubic",
                        align_corners=False,
                        antialias=True,
                    )
                output[self.torch.as_tensor(
                    selected_positions,
                    dtype=self.torch.long,
                    device=self.device,
                )] = remapped
        return output


def prepare(args):
    recording = os.path.abspath(args.recording)
    out_dir = Path(args.out_dir or os.path.join(
        recording, "derived", "vggt_omega"))
    out_dir.mkdir(parents=True, exist_ok=True)

    rec, left_calib, right_calib, R_stereo, t_stereo = (
        load_recording_calibration(recording))
    left_path = os.path.join(recording, "left.mp4")
    right_path = os.path.join(recording, "right.mp4")
    left_meta = video_metadata(left_path)
    right_meta = video_metadata(right_path)
    n_frames = min(left_meta["frames"], right_meta["frames"])
    native_fps = float(rec.get("fps", left_meta["fps"]))
    selected = choose_frames(
        recording, n_frames, native_fps, args.target_fps, args.max_frames)

    _, _, K_virtual = make_fisheye_map(
        *left_calib, np.eye(3), args.image_size, args.fov_deg)

    frame_records = [{"frame": int(frame)} for frame in selected]
    t0 = time.time()
    manifest = {
        "format": FORMAT_TAG,
        "recording": recording,
        "native_frame_count": n_frames,
        "native_fps": native_fps,
        "target_fps": args.target_fps,
        "image_size": args.image_size,
        "fov_deg": args.fov_deg,
        "K_virtual": K_virtual.tolist(),
        "R_stereo": R_stereo.tolist(),
        "t_stereo": t_stereo.tolist(),
        "baseline_m": float(np.linalg.norm(t_stereo)),
        "image_source": "torchcodec_direct",
        "frames": frame_records,
    }
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(
        f"wrote {manifest_path}: {len(frame_records)} frame references in "
        f"{time.time() - t0:.1f}s")


def window_starts(length, window, overlap):
    if window <= overlap:
        raise ValueError("--window-frames must be larger than --overlap-frames")
    if length <= window:
        return [0]
    starts = list(range(0, length - window + 1, window - overlap))
    if starts[-1] + window < length:
        starts.append(length - window)
    return starts


def load_loop_pairs(path, available_frames, min_gap, max_pairs):
    """Select temporally distant frame pairs from loop-candidate JSONL."""
    if path is None or not os.path.exists(path):
        return []
    available = np.asarray(sorted(available_frames), dtype=np.int64)
    candidates = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("pair_type", "temporal") != "temporal":
                continue
            a = record.get("frame_a", record.get("frame0"))
            b = record.get("frame_b", record.get("frame1"))
            if a is None or b is None or abs(int(b) - int(a)) < min_gap:
                continue
            score = record.get(
                "score",
                record.get("n_votes", 0),
            )
            candidates.append((float(score), int(a), int(b)))
    candidates.sort(reverse=True)
    output = []
    used = []
    for _, a, b in candidates:
        a = int(available[np.argmin(np.abs(available - a))])
        b = int(available[np.argmin(np.abs(available - b))])
        if any(abs(a - x) < min_gap // 4 and abs(b - y) < min_gap // 4
               for x, y in used):
            continue
        output.append((a, b))
        used.append((a, b))
        if len(output) >= max_pairs:
            break
    return output


def make_window_specs(manifest, args):
    frames = manifest["frames"]
    specs = []
    for index, start in enumerate(window_starts(
            len(frames), args.window_frames, args.overlap_frames)):
        specs.append({
            "name": f"track_{index:05d}",
            "kind": "track",
            "segments": [0] * min(args.window_frames, len(frames)),
            "frames": frames[start:start + args.window_frames],
        })
    if args.max_track_windows is not None:
        specs = specs[:args.max_track_windows]

    by_frame = {int(item["frame"]): i for i, item in enumerate(frames)}
    loop_pairs = load_loop_pairs(
        args.loop_matches,
        by_frame,
        args.loop_min_gap,
        args.max_loop_windows,
    )
    half = max(2, args.loop_window_frames // 2)
    for index, (frame_a, frame_b) in enumerate(loop_pairs):
        chunks = []
        for frame in (frame_a, frame_b):
            center = by_frame[frame]
            start = max(0, min(center - half // 2, len(frames) - half))
            chunks.append(frames[start:start + half])
        specs.append({
            "name": f"loop_{index:05d}",
            "kind": "loop",
            "segments": [0] * len(chunks[0]) + [1] * len(chunks[1]),
            "frames": chunks[0] + chunks[1],
        })
    return specs


def make_window_image_inputs(spec, manifest, stereo_interval_seconds):
    """Build dense-left, optionally sparse-right model inputs for one window."""
    if stereo_interval_seconds <= 0:
        stereo_stride = 1
    else:
        stereo_stride = max(
            1,
            int(round(manifest["target_fps"] * stereo_interval_seconds)),
        )
    manifest_position = {
        int(item["frame"]): index
        for index, item in enumerate(manifest["frames"])
    }
    segment_endpoints = set()
    segments = np.asarray(spec["segments"], np.int8)
    for segment in np.unique(segments):
        positions = np.flatnonzero(segments == segment)
        segment_endpoints.update((int(positions[0]), int(positions[-1])))

    image_frame_idx = []
    image_eye = []
    image_segment = []
    for local_index, (item, segment) in enumerate(zip(
            spec["frames"], spec["segments"])):
        frame = int(item["frame"])
        image_frame_idx.append(frame)
        image_eye.append(0)
        image_segment.append(segment)

        position = manifest_position[frame]
        include_right = (
            stereo_stride == 1
            or position % stereo_stride == 0
            or local_index in segment_endpoints
        )
        if include_right:
            image_frame_idx.append(frame)
            image_eye.append(1)
            image_segment.append(segment)
    return (
        np.asarray(image_frame_idx, np.int64),
        np.asarray(image_eye, np.int8),
        np.asarray(image_segment, np.int8),
    )


def rank_loop_pair_peaks(pairs, radius, min_votes, nms_radius,
                         max_candidates):
    """Rank localized vote-density peaks without chaining large components."""
    if not pairs:
        return []
    cell_size = max(1, radius)
    grid = {}
    for index, (frame_a, frame_b, _) in enumerate(pairs):
        cell = (frame_a // cell_size, frame_b // cell_size)
        grid.setdefault(cell, []).append(index)

    peaks = []
    for index, (frame_a, frame_b, score) in enumerate(pairs):
        cell = (frame_a // cell_size, frame_b // cell_size)
        neighbors = []
        for delta_a in (-1, 0, 1):
            for delta_b in (-1, 0, 1):
                for other in grid.get(
                        (cell[0] + delta_a, cell[1] + delta_b), []):
                    other_a, other_b, _ = pairs[other]
                    if (abs(frame_a - other_a) <= radius
                            and abs(frame_b - other_b) <= radius):
                        neighbors.append(other)
        if len(neighbors) < min_votes:
            continue
        peaks.append({
            "frame_a": frame_a,
            "frame_b": frame_b,
            "score": score,
            "n_votes": len(neighbors),
            "span_a": [
                min(pairs[other][0] for other in neighbors),
                max(pairs[other][0] for other in neighbors),
            ],
            "span_b": [
                min(pairs[other][1] for other in neighbors),
                max(pairs[other][1] for other in neighbors),
            ],
        })
    peaks.sort(
        key=lambda peak: (peak["n_votes"], peak["score"]), reverse=True)

    selected = []
    for peak in peaks:
        if any(
                abs(peak["frame_a"] - other["frame_a"]) <= nms_radius
                or abs(peak["frame_b"] - other["frame_b"]) <= nms_radius
                for other in selected):
            continue
        selected.append(peak)
        if len(selected) >= max_candidates:
            break
    return selected


def pose_centers_and_forward(poses):
    """Camera centers and +Z viewing directions in world coordinates."""
    poses = np.asarray(poses, np.float64)
    quat = poses[:, :4]
    quat /= np.linalg.norm(quat, axis=1, keepdims=True).clip(1e-12)
    w, x, y, z = quat.T
    rotation = np.stack([
        np.stack([
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
        ], axis=-1),
        np.stack([
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
        ], axis=-1),
        np.stack([
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ], axis=-1),
    ], axis=-2)
    centers = -np.einsum("nji,nj->ni", rotation, poses[:, 4:])
    forward = rotation[:, 2, :]
    return centers, forward


def proximity_loop_pairs(frames, poses, max_distance, min_gap,
                         max_view_angle_deg):
    """Find temporally distant, nearby poses with compatible view direction."""
    from scipy.spatial import cKDTree

    frames = np.asarray(frames, np.int64)
    centers, forward = pose_centers_and_forward(poses)
    raw_pairs = np.asarray(
        list(cKDTree(centers).query_pairs(max_distance)), dtype=np.int64)
    if raw_pairs.size == 0:
        return []
    raw_pairs = raw_pairs.reshape(-1, 2)
    frame_gap = frames[raw_pairs[:, 1]] - frames[raw_pairs[:, 0]]
    distance = np.linalg.norm(
        centers[raw_pairs[:, 1]] - centers[raw_pairs[:, 0]], axis=1)
    view_dot = np.einsum(
        "ij,ij->i", forward[raw_pairs[:, 0]], forward[raw_pairs[:, 1]])
    min_view_dot = math.cos(math.radians(max_view_angle_deg))
    keep = (frame_gap >= min_gap) & (view_dot >= min_view_dot)

    output = []
    for pair, pair_distance, pair_view_dot in zip(
            raw_pairs[keep], distance[keep], view_dot[keep]):
        distance_score = max(0.0, 1.0 - pair_distance / max_distance)
        view_score = (
            (pair_view_dot - min_view_dot)
            / max(1e-6, 1.0 - min_view_dot)
        )
        score = 0.5 * distance_score + 0.5 * view_score
        output.append((
            int(frames[pair[0]]),
            int(frames[pair[1]]),
            float(score),
        ))
    return output


def propose_proximity_loops(args):
    recording = os.path.abspath(args.recording)
    out_dir = Path(args.out_dir or os.path.join(
        recording, "derived", "vggt_omega"))
    manifest = json.load(open(out_dir / "manifest.json"))
    trajectory = np.load(args.trajectory)
    if "keyframe_frame_idx" in trajectory:
        frames = trajectory["keyframe_frame_idx"].astype(np.int64)
        poses = trajectory["keyframe_pose_wxyz_xyz"].astype(np.float64)
    else:
        frames = trajectory["frame_idx"].astype(np.int64)
        poses = trajectory["pose_wxyz_xyz"].astype(np.float64)

    available = {int(item["frame"]) for item in manifest["frames"]}
    keep = np.asarray([int(frame) in available for frame in frames])
    frames, poses = frames[keep], poses[keep]
    native_fps = float(manifest["native_fps"])
    sample_gap = max(1, int(round(args.sample_seconds * native_fps)))
    sample_indices = [0]
    for index in range(1, len(frames)):
        if frames[index] - frames[sample_indices[-1]] >= sample_gap:
            sample_indices.append(index)
    if sample_indices[-1] != len(frames) - 1:
        sample_indices.append(len(frames) - 1)
    frames = frames[sample_indices]
    poses = poses[sample_indices]
    pairs = proximity_loop_pairs(
        frames,
        poses,
        args.max_distance,
        int(round(args.min_gap_seconds * native_fps)),
        args.max_view_angle_deg,
    )
    clusters = rank_loop_pair_peaks(
        pairs,
        radius=int(round(args.cluster_radius_seconds * native_fps)),
        min_votes=args.min_cluster_votes,
        nms_radius=int(round(args.nms_seconds * native_fps)),
        max_candidates=args.max_candidates,
    )

    centers, forward = pose_centers_and_forward(poses)
    by_frame = {int(frame): index for index, frame in enumerate(frames)}
    out_path = Path(
        args.out or out_dir / "proximity_loop_candidates.jsonl")
    with open(out_path, "w") as output:
        for cluster in clusters:
            index_a = by_frame[cluster["frame_a"]]
            index_b = by_frame[cluster["frame_b"]]
            cluster["distance_m"] = float(np.linalg.norm(
                centers[index_a] - centers[index_b]))
            cluster["view_dot"] = float(
                forward[index_a] @ forward[index_b])
            cluster["pair_type"] = "temporal"
            cluster["source"] = "trajectory_proximity_view"
            output.write(json.dumps(cluster) + "\n")
    print(
        f"wrote {out_path}: {len(clusters)} regions from {len(pairs)} "
        f"proximity/view pairs across {len(frames)} poses")


def _as_homogeneous(extrinsic):
    output = np.eye(4, dtype=np.float64)
    output[:3, :4] = np.asarray(extrinsic, np.float64)[:3, :4]
    return output


def depth_reprojection_fraction(
        source_depth, source_conf, source_intrinsics, source_extrinsic,
        target_depth, target_conf, target_intrinsics, target_extrinsic,
        pixel_stride, confidence_quantile, relative_tolerance):
    """Fraction of confident source depth that agrees after target projection."""
    source_depth = np.asarray(source_depth, np.float64).squeeze()
    target_depth = np.asarray(target_depth, np.float64).squeeze()
    source_conf = np.asarray(source_conf, np.float64).squeeze()
    target_conf = np.asarray(target_conf, np.float64).squeeze()
    height, width = source_depth.shape
    yy, xx = np.meshgrid(
        np.arange(pixel_stride // 2, height, pixel_stride),
        np.arange(pixel_stride // 2, width, pixel_stride),
        indexing="ij",
    )
    yy, xx = yy.ravel(), xx.ravel()
    depth = source_depth[yy, xx]
    confidence = source_conf[yy, xx]
    finite_confidence = confidence[np.isfinite(confidence)]
    if not len(finite_confidence):
        return 0.0
    confidence_cutoff = np.quantile(
        finite_confidence, confidence_quantile)
    valid_source = (
        np.isfinite(depth) & (depth > 0)
        & np.isfinite(confidence) & (confidence >= confidence_cutoff)
    )
    if not np.any(valid_source):
        return 0.0
    yy, xx, depth = yy[valid_source], xx[valid_source], depth[valid_source]

    K_source = np.asarray(source_intrinsics, np.float64)
    points_source = np.stack([
        (xx - K_source[0, 2]) / K_source[0, 0] * depth,
        (yy - K_source[1, 2]) / K_source[1, 1] * depth,
        depth,
        np.ones_like(depth),
    ], axis=1)
    source_to_target = (
        _as_homogeneous(target_extrinsic)
        @ np.linalg.inv(_as_homogeneous(source_extrinsic))
    )
    points_target = points_source @ source_to_target.T
    z_target = points_target[:, 2]
    K_target = np.asarray(target_intrinsics, np.float64)
    u_target = np.rint(
        K_target[0, 0] * points_target[:, 0] / z_target
        + K_target[0, 2]).astype(np.int64)
    v_target = np.rint(
        K_target[1, 1] * points_target[:, 1] / z_target
        + K_target[1, 2]).astype(np.int64)
    target_height, target_width = target_depth.shape
    in_view = (
        np.isfinite(z_target) & (z_target > 0)
        & (u_target >= 0) & (u_target < target_width)
        & (v_target >= 0) & (v_target < target_height)
    )
    if not np.any(in_view):
        return 0.0

    u_target = u_target[in_view]
    v_target = v_target[in_view]
    z_target = z_target[in_view]
    measured_depth = target_depth[v_target, u_target]
    measured_confidence = target_conf[v_target, u_target]
    finite_target_confidence = target_conf[np.isfinite(target_conf)]
    if not len(finite_target_confidence):
        return 0.0
    target_confidence_cutoff = np.quantile(
        finite_target_confidence, confidence_quantile)
    consistent_target = (
        np.isfinite(measured_depth) & (measured_depth > 0)
        & np.isfinite(measured_confidence)
        & (measured_confidence >= target_confidence_cutoff)
    )
    tolerance = relative_tolerance * np.minimum(
        z_target, measured_depth).clip(1e-6)
    consistent_depth = (
        np.abs(z_target - measured_depth) <= tolerance)
    return float(np.count_nonzero(
        consistent_target & consistent_depth) / len(depth))


def loop_depth_overlap(predictions, extrinsics, intrinsics, image_eye,
                       image_segment, args):
    center_indices = []
    for segment in (0, 1):
        indices = np.flatnonzero(
            (image_eye == 0) & (image_segment == segment))
        if not len(indices):
            return 0.0, (0.0, 0.0)
        center_indices.append(int(indices[len(indices) // 2]))
    depth = predictions["depth"][0].float().cpu().numpy()
    confidence = predictions["depth_conf"][0].float().cpu().numpy()
    extrinsics = extrinsics[0].float().cpu().numpy()
    intrinsics = intrinsics[0].float().cpu().numpy()
    first, second = center_indices
    common = (
        args.overlap_pixel_stride,
        args.overlap_confidence_quantile,
        args.overlap_relative_tolerance,
    )
    forward = depth_reprojection_fraction(
        depth[first], confidence[first], intrinsics[first], extrinsics[first],
        depth[second], confidence[second], intrinsics[second],
        extrinsics[second], *common)
    backward = depth_reprojection_fraction(
        depth[second], confidence[second], intrinsics[second],
        extrinsics[second],
        depth[first], confidence[first], intrinsics[first],
        extrinsics[first], *common)
    return min(forward, backward), (forward, backward)


def inference_devices(torch, requested_device, max_devices):
    """Resolve worker devices after CUDA_VISIBLE_DEVICES has been applied."""
    requested_device = str(requested_device)
    if not requested_device.startswith("cuda"):
        return [torch.device(requested_device)]
    if requested_device != "cuda":
        return [torch.device(requested_device)]
    count = torch.cuda.device_count()
    if count == 0:
        raise RuntimeError("CUDA inference requested, but no GPUs are visible")
    if max_devices > 0:
        count = min(count, max_devices)
    return [torch.device(f"cuda:{index}") for index in range(count)]


def window_cache_matches(out_path, reject_path, spec):
    expected_frames = np.asarray(
        [item["frame"] for item in spec["frames"]], np.int64)
    if out_path.exists():
        try:
            with np.load(out_path) as data:
                return (
                    str(data["kind"]) == spec["kind"]
                    and np.array_equal(data["frame_idx"], expected_frames)
                )
        except (KeyError, OSError, ValueError):
            return False
    if reject_path.exists():
        try:
            rejected = json.load(open(reject_path))
            return (
                rejected.get("kind") == spec["kind"]
                and rejected.get("frame_idx") == expected_frames.tolist()
            )
        except (OSError, ValueError):
            return False
    return False


def prune_stale_loop_cache(windows_dir, specs):
    expected = {
        spec["name"] for spec in specs if spec["kind"] == "loop"}
    removed = []
    for pattern in ("loop_*.npz", "loop_*.rejected.json"):
        for path in windows_dir.glob(pattern):
            name = path.name.split(".", 1)[0]
            if name not in expected:
                path.unlink()
                removed.append(path)
    return removed


def infer(args):
    import torch
    from vggt_omega.models import VGGTOmega
    from vggt_omega.utils.pose_enc import encoding_to_camera

    recording = os.path.abspath(args.recording)
    out_dir = Path(args.out_dir or os.path.join(
        recording, "derived", "vggt_omega"))
    manifest_path = out_dir / "manifest.json"
    manifest = json.load(open(manifest_path))
    if manifest.get("format") != FORMAT_TAG:
        raise ValueError(f"unsupported manifest format in {manifest_path}")
    windows_dir = out_dir / "windows"
    windows_dir.mkdir(parents=True, exist_ok=True)

    if args.checkpoint is None:
        raise ValueError("--checkpoint is required for inference")
    torch.set_float32_matmul_precision("high")
    if args.compile:
        if args.verify_loop_overlap:
            raise ValueError(
                "--compile is unsupported with --verify-loop-overlap")
        if not str(args.device).startswith("cuda"):
            raise ValueError("--compile currently requires a CUDA device")

    specs = make_window_specs(manifest, args)
    if args.loop_matches is not None:
        for path in prune_stale_loop_cache(windows_dir, specs):
            print(f"removed stale {path.name}")
    devices = inference_devices(torch, args.device, args.num_devices)
    effective_compile_mode = args.compile_mode
    if args.compile and len(devices) > 1:
        thread_safe_modes = {
            "reduce-overhead": "default",
            "max-autotune": "max-autotune-no-cudagraphs",
        }
        effective_compile_mode = thread_safe_modes.get(
            args.compile_mode, args.compile_mode)
        if effective_compile_mode != args.compile_mode:
            print(
                f"compile mode {args.compile_mode!r} uses thread-local CUDA "
                f"Graphs; using {effective_compile_mode!r} for "
                f"{len(devices)} GPU workers")
    print(
        f"inference workers: {', '.join(str(device) for device in devices)}")
    pending = []
    for index, spec in enumerate(specs):
        out_path = windows_dir / f"{spec['name']}.npz"
        reject_path = windows_dir / f"{spec['name']}.rejected.json"
        if (not args.overwrite
                and window_cache_matches(out_path, reject_path, spec)):
            print(f"[{index + 1}/{len(specs)}] cached {out_path.name}")
        else:
            if out_path.exists() or reject_path.exists():
                print(f"[{index + 1}/{len(specs)}] stale {out_path.name}")
            pending.append((index, spec))

    tasks = queue.Queue()
    for task in pending:
        tasks.put(task)
    timings = []
    decode_timings = []
    results_lock = threading.Lock()
    print_lock = threading.Lock()

    def worker(device):
        if device.type == "cuda":
            torch.cuda.set_device(device)
        model = VGGTOmega()
        state = torch.load(
            args.checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        del state
        # Tracking only consumes pose_enc. Loop verification keeps the dense
        # head to reject candidate regions without geometric overlap.
        if not args.verify_loop_overlap:
            model.dense_head = None
        model = model.eval().to(device)
        if args.compile:
            model = torch.compile(
                model, mode=effective_compile_mode, fullgraph=False)
        video_loader = DirectStereoVideoLoader(
            manifest, device, args.decode_batch_size)

        while True:
            try:
                index, spec = tasks.get_nowait()
            except queue.Empty:
                break
            try:
                out_path = windows_dir / f"{spec['name']}.npz"
                reject_path = windows_dir / f"{spec['name']}.rejected.json"
                out_path.unlink(missing_ok=True)
                reject_path.unlink(missing_ok=True)
                image_frame_idx, image_eye, image_segment = (
                    make_window_image_inputs(
                        spec, manifest, args.stereo_interval_seconds))
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                decode_start = time.time()
                images = video_loader.load(
                    image_frame_idx, image_eye, args.resolution)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                decode_elapsed = time.time() - decode_start
                repeat_timings = []
                repeat_peak_memory = []
                predictions = None
                for _ in range(args.benchmark_repeats):
                    if predictions is not None:
                        del predictions
                    if device.type == "cuda":
                        torch.cuda.reset_peak_memory_stats(device)
                        torch.cuda.synchronize(device)
                    t0 = time.time()
                    with torch.inference_mode():
                        predictions = model(images)
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                        peak_memory = torch.cuda.max_memory_allocated(device)
                    else:
                        peak_memory = 0
                    repeat_timings.append(time.time() - t0)
                    repeat_peak_memory.append(peak_memory)
                elapsed = repeat_timings[-1]
                extrinsics, intrinsics = encoding_to_camera(
                    predictions["pose_enc"],
                    predictions["images"].shape[-2:],
                )
                overlap_score = np.nan
                overlap_directional = (np.nan, np.nan)
                if args.verify_loop_overlap and spec["kind"] == "loop":
                    overlap_score, overlap_directional = loop_depth_overlap(
                        predictions,
                        extrinsics,
                        intrinsics,
                        image_eye,
                        image_segment,
                        args,
                    )
                    with print_lock:
                        print(
                            f"[{index + 1}/{len(specs)}] "
                            f"{spec['name']} overlap {overlap_score:.3f} "
                            f"({overlap_directional[0]:.3f}, "
                            f"{overlap_directional[1]:.3f})")
                    if overlap_score < args.min_loop_overlap:
                        with open(reject_path, "w") as f:
                            json.dump({
                                "name": spec["name"],
                                "kind": spec["kind"],
                                "frame_idx": [
                                    int(item["frame"])
                                    for item in spec["frames"]
                                ],
                                "overlap_score": overlap_score,
                                "overlap_directional": overlap_directional,
                                "threshold": args.min_loop_overlap,
                            }, f, indent=2)
                        continue
                frame_idx = np.asarray(
                    [item["frame"] for item in spec["frames"]], np.int64)
                np.savez_compressed(
                    out_path,
                    format=np.asarray(FORMAT_TAG),
                    kind=np.asarray(spec["kind"]),
                    frame_idx=frame_idx,
                    image_frame_idx=image_frame_idx,
                    image_eye=image_eye,
                    image_segment=image_segment,
                    segment=np.asarray(spec["segments"], np.int8),
                    extrinsics=extrinsics[0].float().cpu().numpy(),
                    intrinsics=intrinsics[0].float().cpu().numpy(),
                    loop_overlap_score=np.asarray(overlap_score),
                    loop_overlap_directional=np.asarray(
                        overlap_directional),
                    inference_seconds=np.asarray(elapsed),
                    decode_remap_seconds=np.asarray(decode_elapsed),
                    benchmark_inference_seconds=np.asarray(repeat_timings),
                    peak_memory_bytes=np.asarray(repeat_peak_memory[-1]),
                    benchmark_peak_memory_bytes=np.asarray(
                        repeat_peak_memory),
                    inference_device=np.asarray(str(device)),
                )
                with results_lock:
                    timings.append(elapsed)
                    decode_timings.append(decode_elapsed)
                with print_lock:
                    print(
                        f"[{index + 1}/{len(specs)}] {out_path.name} on "
                        f"{device}: {len(image_frame_idx)} images in "
                        f"{decode_elapsed:.2f}s decode/remap + "
                        f"{elapsed:.2f}s inference, "
                        f"{repeat_peak_memory[-1] / 2**30:.2f} GiB"
                        + (
                            " (all repeats: "
                            + ", ".join(
                                f"{value:.2f}s"
                                for value in repeat_timings)
                            + ")"
                            if len(repeat_timings) > 1 else ""
                        ))
            finally:
                tasks.task_done()

    wall_start = time.time()
    if pending:
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=len(devices)) as executor:
            futures = [
                executor.submit(worker, device) for device in devices
            ]
            for future in futures:
                future.result()
    wall_elapsed = time.time() - wall_start

    summary = {
        "format": FORMAT_TAG,
        "checkpoint": os.path.abspath(args.checkpoint),
        "window_frames": args.window_frames,
        "overlap_frames": args.overlap_frames,
        "loop_window_frames": args.loop_window_frames,
        "stereo_interval_seconds": args.stereo_interval_seconds,
        "camera_only": not args.verify_loop_overlap,
        "torch_compile": args.compile,
        "torch_compile_mode_requested": (
            args.compile_mode if args.compile else None),
        "torch_compile_mode": (
            effective_compile_mode if args.compile else None),
        "inference_devices": [str(device) for device in devices],
        "inference_workers": len(devices),
        "windows": [spec["name"] for spec in specs],
        "new_inference_seconds": float(sum(timings)),
        "new_decode_remap_seconds": float(sum(decode_timings)),
        "new_inference_wall_seconds": wall_elapsed,
    }
    with open(out_dir / "windows.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {out_dir / 'windows.json'}")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("recording")
    prepare_parser.add_argument("--out-dir", default=None)
    prepare_parser.add_argument("--target-fps", type=float, default=5.0)
    prepare_parser.add_argument("--image-size", type=int, default=512)
    prepare_parser.add_argument("--fov-deg", type=float, default=100.0)
    prepare_parser.add_argument("--max-frames", type=int, default=None)
    prepare_parser.set_defaults(func=prepare)

    proximity_parser = subparsers.add_parser("propose-proximity-loops")
    proximity_parser.add_argument("recording")
    proximity_parser.add_argument("--trajectory", required=True)
    proximity_parser.add_argument("--out-dir", default=None)
    proximity_parser.add_argument("--out", default=None)
    proximity_parser.add_argument("--max-distance", type=float, default=1.5)
    proximity_parser.add_argument(
        "--sample-seconds",
        type=float,
        default=0.5,
        help="temporal cadence for the cheap trajectory radius query",
    )
    proximity_parser.add_argument(
        "--min-gap-seconds", type=float, default=15.0)
    proximity_parser.add_argument(
        "--max-view-angle-deg", type=float, default=60.0)
    proximity_parser.add_argument(
        "--cluster-radius-seconds", type=float, default=2.0)
    proximity_parser.add_argument("--nms-seconds", type=float, default=5.0)
    proximity_parser.add_argument("--min-cluster-votes", type=int, default=3)
    proximity_parser.add_argument("--max-candidates", type=int, default=20)
    proximity_parser.set_defaults(func=propose_proximity_loops)

    infer_parser = subparsers.add_parser("infer")
    infer_parser.add_argument("recording")
    infer_parser.add_argument("--out-dir", default=None)
    infer_parser.add_argument("--checkpoint", default=None)
    infer_parser.add_argument("--device", default="cuda")
    infer_parser.add_argument(
        "--num-devices",
        type=int,
        default=0,
        help="maximum visible CUDA devices to use; 0 uses all",
    )
    infer_parser.add_argument("--resolution", type=int, default=512)
    infer_parser.add_argument(
        "--decode-batch-size",
        type=int,
        default=32,
        help="per-eye TorchCodec decode/remap batch size on each GPU",
    )
    infer_parser.add_argument("--window-frames", type=int, default=16)
    infer_parser.add_argument("--overlap-frames", type=int, default=8)
    infer_parser.add_argument(
        "--max-track-windows",
        type=int,
        default=None,
        help="limit tracking windows for pilot runs",
    )
    infer_parser.add_argument(
        "--stereo-interval-seconds",
        type=float,
        default=0.0,
        help="right-eye cadence; 0 includes the right eye at every timestep",
    )
    infer_parser.add_argument("--loop-matches", default=None)
    infer_parser.add_argument("--loop-window-frames", type=int, default=16)
    infer_parser.add_argument("--loop-min-gap", type=int, default=120)
    infer_parser.add_argument("--max-loop-windows", type=int, default=100)
    infer_parser.add_argument(
        "--verify-loop-overlap",
        action="store_true",
        help="run the dense head and reject loop windows with poor overlap",
    )
    infer_parser.add_argument("--min-loop-overlap", type=float, default=0.03)
    infer_parser.add_argument(
        "--overlap-pixel-stride", type=int, default=8)
    infer_parser.add_argument(
        "--overlap-confidence-quantile", type=float, default=0.5)
    infer_parser.add_argument(
        "--overlap-relative-tolerance", type=float, default=0.15)
    infer_parser.add_argument(
        "--compile",
        action="store_true",
        help="compile the fixed-shape camera-only model with TorchInductor",
    )
    infer_parser.add_argument(
        "--compile-mode",
        choices=(
            "default",
            "reduce-overhead",
            "max-autotune",
            "max-autotune-no-cudagraphs",
        ),
        default="reduce-overhead",
    )
    infer_parser.add_argument(
        "--benchmark-repeats",
        type=int,
        default=1,
        help="repeat each model call for timing; benchmark use only",
    )
    infer_parser.add_argument("--overwrite", action="store_true")
    infer_parser.set_defaults(func=infer)
    return parser


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
