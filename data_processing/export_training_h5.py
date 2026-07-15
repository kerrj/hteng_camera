"""Export one recording's derived labels as a frame-aligned training HDF5."""

import argparse
from datetime import datetime, timezone
from itertools import chain
import json
import os

import h5py
import numpy as np


FORMAT = "hteng-camera-training"
SCHEMA_VERSION = 2
FRAME_CHUNK = 256


def quat_to_matrix(quat):
    quat = np.asarray(quat, np.float64)
    quat = quat / np.linalg.norm(quat, axis=-1, keepdims=True)
    w, x, y, z = np.moveaxis(quat, -1, 0)
    return np.stack([
        1 - 2 * (y * y + z * z),
        2 * (x * y - z * w),
        2 * (x * z + y * w),
        2 * (x * y + z * w),
        1 - 2 * (x * x + z * z),
        2 * (y * z - x * w),
        2 * (x * z - y * w),
        2 * (y * z + x * w),
        1 - 2 * (x * x + y * y),
    ], axis=-1).reshape(quat.shape[:-1] + (3, 3))


def poses_to_matrices(poses):
    poses = np.asarray(poses, np.float64)
    if poses.ndim != 2 or poses.shape[1] != 7:
        raise ValueError(f"poses must have shape (N, 7), got {poses.shape}")
    matrices = np.broadcast_to(np.eye(4), (len(poses), 4, 4)).copy()
    matrices[:, :3, :3] = quat_to_matrix(poses[:, :4])
    matrices[:, :3, 3] = poses[:, 4:]
    return matrices


def invert_transforms(transforms):
    inverse = np.broadcast_to(np.eye(4), transforms.shape).copy()
    rotation_t = np.swapaxes(transforms[..., :3, :3], -1, -2)
    inverse[..., :3, :3] = rotation_t
    inverse[..., :3, 3] = -np.einsum(
        "...ij,...j->...i", rotation_t, transforms[..., :3, 3])
    return inverse


def _dataset(group, name, value, description, units=None, frame_aligned=False):
    value = np.asarray(value)
    kwargs = {}
    if frame_aligned and len(value):
        kwargs.update(
            chunks=(min(FRAME_CHUNK, len(value)),) + value.shape[1:],
            compression="lzf",
            shuffle=value.dtype.kind in "fiu",
        )
    dataset = group.create_dataset(name, data=value, **kwargs)
    dataset.attrs["description"] = description
    if units is not None:
        dataset.attrs["units"] = units
    return dataset


def _load_recording_metadata(recording):
    path = os.path.join(recording, "recording.json")
    with open(path) as handle:
        return json.load(handle), path


def _load_calibration(recording, metadata):
    cameras = []
    for eye in ("left", "right"):
        camera_meta = metadata[eye]
        path = os.path.join(recording, camera_meta["calibration_file"])
        with open(path) as handle:
            calibration = json.load(handle)
        intrinsics = calibration["intrinsics"]
        K = np.asarray(intrinsics["K"], np.float64).copy()
        roi = camera_meta["roi"]
        K[0, 2] -= roi["x_offset"]
        K[1, 2] -= roi["y_offset"]
        cameras.append({
            "serial": str(camera_meta["serial"]),
            "K": K,
            "distortion": np.asarray(intrinsics["dist"], np.float64),
            "image_size": np.asarray([roi["width"], roi["height"]], np.int32),
            "model": intrinsics["model"],
            "path": path,
        })

    stereo_path = os.path.join(
        recording, metadata["files"]["stereo_transform"])
    with open(stereo_path) as handle:
        stereo = json.load(handle)
    right_from_left = np.eye(4, dtype=np.float64)
    right_from_left[:3, :3] = np.asarray(stereo["R"], np.float64)
    right_from_left[:3, 3] = np.asarray(stereo["t"], np.float64).reshape(3)
    return cameras, stereo, stereo_path, right_from_left


def _video_frame_count(recording, metadata):
    import cv2

    counts = []
    for eye in ("left", "right"):
        path = os.path.join(recording, metadata["files"][eye])
        video = cv2.VideoCapture(path)
        count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        opened = video.isOpened()
        video.release()
        if not opened or count <= 0:
            raise ValueError(f"could not read frame count from {path}")
        counts.append(count)
    if counts[0] != counts[1]:
        raise ValueError(
            f"stereo videos have different frame counts: "
            f"left={counts[0]}, right={counts[1]}")
    return counts[0]


def _load_trajectory(path, right_from_left, video_frame_count):
    with np.load(path) as trajectory:
        frame_idx = np.asarray(trajectory["frame_idx"], np.int64)
        interpolation_time_s = (
            np.asarray(trajectory["interpolation_frame_time_s"], np.float64)
            if "interpolation_frame_time_s" in trajectory else None)
        left = np.asarray(
            trajectory["pose_wxyz_xyz_left"]
            if "pose_wxyz_xyz_left" in trajectory
            else trajectory["pose_wxyz_xyz"],
            np.float64,
        )
        if "pose_wxyz_xyz_right" in trajectory:
            right = poses_to_matrices(trajectory["pose_wxyz_xyz_right"])
        else:
            right = np.einsum(
                "ij,njk->nik", right_from_left, poses_to_matrices(left))
    if len(frame_idx) != len(left):
        raise ValueError("trajectory frame and pose counts differ")
    if len(np.unique(frame_idx)) != len(frame_idx):
        raise ValueError("trajectory frame_idx contains duplicates")
    if np.any(np.diff(frame_idx) <= 0):
        raise ValueError("trajectory frame_idx must be strictly increasing")
    keep = frame_idx < video_frame_count
    dropped = int((~keep).sum())
    if dropped:
        print(
            f"discarding {dropped} trailing trajectory frame(s) absent from "
            "the encoded stereo videos")
    frame_idx = frame_idx[keep]
    left = left[keep]
    right = right[keep]
    if interpolation_time_s is not None:
        interpolation_time_s = interpolation_time_s[keep]
    expected = np.arange(video_frame_count, dtype=np.int64)
    if not np.array_equal(frame_idx, expected):
        raise ValueError(
            "trajectory must contain exactly one pose for every encoded video "
            f"frame 0..{video_frame_count - 1}")
    camera_from_world = np.stack([poses_to_matrices(left), right], axis=1)
    if not np.all(np.isfinite(camera_from_world)):
        raise ValueError("trajectory contains non-finite poses")
    if (interpolation_time_s is not None
            and (interpolation_time_s.shape != frame_idx.shape
                 or np.any(np.diff(interpolation_time_s) <= 0))):
        raise ValueError("trajectory interpolation times are invalid")
    return frame_idx, camera_from_world, interpolation_time_s


def _load_frame_times(path, frame_idx):
    time_us = np.full(len(frame_idx), -1, np.int64)
    valid = np.zeros(len(frame_idx), bool)
    if path is None or not os.path.exists(path):
        return time_us, valid
    with np.load(path) as imu:
        imu_frame = np.asarray(imu["frame_idx"], np.int64)
        imu_time = np.asarray(imu["frame_time_us"], np.int64)
        imu_valid = np.asarray(imu["frame_valid"], bool)
    lookup = {int(frame): i for i, frame in enumerate(frame_idx)}
    for frame, timestamp, is_valid in zip(imu_frame, imu_time, imu_valid):
        output_index = lookup.get(int(frame))
        if output_index is not None:
            time_us[output_index] = timestamp
            valid[output_index] = is_valid
    return time_us, valid


def _repaired_relative_times(time_us):
    delta_us = np.diff(np.asarray(time_us, np.float64))
    positive = delta_us[delta_us > 0]
    if not len(positive):
        raise ValueError("frame timestamps contain no positive intervals")
    delta_us = np.where(delta_us > 0, delta_us, np.median(positive))
    return np.concatenate([[0.0], np.cumsum(delta_us) / 1e6])


def _nearest_frame_indices(event_time_us, frame_idx, frame_time_us,
                           frame_time_valid):
    event_time_us = np.asarray(event_time_us, np.int64)
    result = np.full(event_time_us.shape, -1, np.int64)
    valid = (
        np.asarray(frame_time_valid, bool)
        & (np.asarray(frame_time_us, np.int64) >= 0)
    )
    if not np.any(valid):
        return result

    times = np.asarray(frame_time_us, np.int64)[valid]
    frames = np.asarray(frame_idx, np.int64)[valid]
    order = np.argsort(times, kind="stable")
    times = times[order]
    frames = frames[order]
    unique = np.concatenate([[True], np.diff(times) > 0])
    times = times[unique]
    frames = frames[unique]

    query_valid = event_time_us >= 0
    query = event_time_us[query_valid]
    insertion = np.searchsorted(times, query)
    right = np.minimum(insertion, len(times) - 1)
    left = np.maximum(insertion - 1, 0)
    choose_right = np.abs(times[right] - query) < np.abs(query - times[left])
    nearest = np.where(choose_right, right, left)
    result[query_valid] = frames[nearest]
    return result


def _empty_voice_table(include_words=False):
    table = {
        "text": np.asarray([], dtype=object),
        "id": np.asarray([], dtype=np.int64),
        "start_audio_us": np.asarray([], dtype=np.int64),
        "end_audio_us": np.asarray([], dtype=np.int64),
        "start_pts_us": np.asarray([], dtype=np.int64),
        "end_pts_us": np.asarray([], dtype=np.int64),
        "start_time_us": np.asarray([], dtype=np.int64),
        "end_time_us": np.asarray([], dtype=np.int64),
        "start_frame_index": np.asarray([], dtype=np.int64),
        "end_frame_index": np.asarray([], dtype=np.int64),
    }
    if include_words:
        table["probability"] = np.asarray([], dtype=np.float32)
        table["segment_index"] = np.asarray([], dtype=np.int64)
    return table


def _voice_timestamp(record, prefix, source_uses_perf_counter):
    try:
        audio_us = int(round(float(record[f"{prefix}_audio_s"]) * 1e6))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"voice transcript row has invalid {prefix}_audio_s") from exc
    pts_us = int(record.get(f"{prefix}_pts_us", -1))
    time_us = int(record.get(f"{prefix}_perf_counter_us", -1))
    if time_us < 0 and source_uses_perf_counter:
        time_us = pts_us
    return audio_us, pts_us, time_us


def _load_voice(path, frame_idx, frame_time_us, frame_time_valid):
    empty = {
        "available": False,
        "text": "",
        "format": "",
        "source": {},
        "transcription": {},
        "segments": _empty_voice_table(),
        "words": _empty_voice_table(include_words=True),
    }
    if path is None or not os.path.exists(path):
        return empty

    with open(path) as handle:
        transcript = json.load(handle)
    if transcript.get("format") != "hteng-camera-voice-transcript/1":
        raise ValueError(
            f"{path}: unsupported voice transcript format "
            f"{transcript.get('format')!r}")
    source = transcript.get("source", {})
    source_uses_perf_counter = source.get("pts_clock") == "perf_counter"

    segment_rows = []
    word_rows = []
    for segment_index, segment in enumerate(transcript.get("segments", [])):
        start_audio, start_pts, start_time = _voice_timestamp(
            segment, "start", source_uses_perf_counter)
        end_audio, end_pts, end_time = _voice_timestamp(
            segment, "end", source_uses_perf_counter)
        if end_audio < start_audio:
            raise ValueError(
                f"{path}: segment {segment_index} ends before it starts")
        segment_rows.append({
            "text": str(segment.get("text", "")),
            "id": int(segment.get("id", segment_index)),
            "start_audio_us": start_audio,
            "end_audio_us": end_audio,
            "start_pts_us": start_pts,
            "end_pts_us": end_pts,
            "start_time_us": start_time,
            "end_time_us": end_time,
        })
        for word_index, word in enumerate(segment.get("words", [])):
            word_start_audio, word_start_pts, word_start_time = (
                _voice_timestamp(word, "start", source_uses_perf_counter))
            word_end_audio, word_end_pts, word_end_time = _voice_timestamp(
                word, "end", source_uses_perf_counter)
            if word_end_audio < word_start_audio:
                raise ValueError(
                    f"{path}: segment {segment_index} word {word_index} "
                    "ends before it starts")
            word_rows.append({
                "text": str(word.get("text", "")),
                "id": len(word_rows),
                "start_audio_us": word_start_audio,
                "end_audio_us": word_end_audio,
                "start_pts_us": word_start_pts,
                "end_pts_us": word_end_pts,
                "start_time_us": word_start_time,
                "end_time_us": word_end_time,
                "probability": float(word.get("probability", np.nan)),
                "segment_index": segment_index,
            })

    def build_table(rows, include_words=False):
        table = _empty_voice_table(include_words=include_words)
        if not rows:
            return table
        for name in table:
            if name in ("start_frame_index", "end_frame_index"):
                continue
            dtype = table[name].dtype
            table[name] = np.asarray(
                [row[name] for row in rows], dtype=dtype)
        table["start_frame_index"] = _nearest_frame_indices(
            table["start_time_us"], frame_idx, frame_time_us, frame_time_valid)
        table["end_frame_index"] = _nearest_frame_indices(
            table["end_time_us"], frame_idx, frame_time_us, frame_time_valid)
        return table

    return {
        "available": True,
        "text": str(transcript.get("text", "")),
        "format": transcript["format"],
        "source": source,
        "transcription": transcript.get("transcription", {}),
        "segments": build_table(segment_rows),
        "words": build_table(word_rows, include_words=True),
    }


def _write_voice_table(group, table, descriptions):
    text_dtype = h5py.string_dtype("utf-8")
    text_dataset = group.create_dataset(
        "text", data=table["text"], dtype=text_dtype)
    text_dataset.attrs["description"] = descriptions["text"]
    for name, value in table.items():
        if name == "text":
            continue
        units = "us" if name.endswith("_us") else None
        _dataset(group, name, value, descriptions[name], units=units)


def _write_voice(group, voice, source_path):
    group.attrs["available"] = bool(voice["available"])
    group.attrs["source_file"] = (
        os.path.basename(source_path)
        if voice["available"] and source_path else "")
    group.attrs["source_format"] = voice["format"]
    group.attrs["source_metadata_json"] = json.dumps(
        voice["source"], separators=(",", ":"))
    group.attrs["transcription_metadata_json"] = json.dumps(
        voice["transcription"], separators=(",", ":"))
    transcript = group.create_dataset(
        "transcript",
        data=voice["text"],
        dtype=h5py.string_dtype("utf-8"),
    )
    transcript.attrs["description"] = "Complete normalized transcript text."

    common = {
        "text": "Normalized recognized text.",
        "id": "Source transcript row identifier.",
        "start_audio_us": "Start relative to the first decoded audio sample.",
        "end_audio_us": "End relative to the first decoded audio sample.",
        "start_pts_us": "Start on the original audio container PTS timeline.",
        "end_pts_us": "End on the original audio container PTS timeline.",
        "start_time_us": (
            "Start on the perf_counter host clock; -1 when unavailable."),
        "end_time_us": (
            "End on the perf_counter host clock; -1 when unavailable."),
        "start_frame_index": (
            "Native video frame nearest start_time_us; -1 when unavailable."),
        "end_frame_index": (
            "Native video frame nearest end_time_us; -1 when unavailable."),
    }
    _write_voice_table(
        group.create_group("segments"), voice["segments"], common)
    _write_voice_table(
        group.create_group("words"),
        voice["words"],
        {
            **common,
            "probability": "Whisper word confidence probability.",
            "segment_index": (
                "Zero-based row index into the voice/segments table."),
        },
    )


def _empty_hand_arrays(frame_count):
    def nan(shape):
        return np.full(shape, np.nan, np.float32)

    return {
        "valid": np.zeros(frame_count, bool),
        "interpolated": np.zeros(frame_count, bool),
        "source_frames": np.full((frame_count, 2), -1, np.int64),
        "root_camera_m": nan((frame_count, 3)),
        "joints_camera_m": nan((frame_count, 21, 3)),
        "wrist_rotation_camera": nan((frame_count, 3, 3)),
        "joint_quaternion_wxyz": nan((frame_count, 16, 4)),
        "translation_virtual_m": nan((frame_count, 3)),
        "virtual_to_camera_rotation": nan((frame_count, 3, 3)),
        "phase1_mean_reprojection_px": nan((frame_count,)),
        "phase1_p90_reprojection_px": nan((frame_count,)),
        "phase1_median_epipolar_px": nan((frame_count,)),
        "phase1_depth_m": nan((frame_count,)),
    }


def _load_hand(path, frame_idx, world_from_left):
    arrays = _empty_hand_arrays(len(frame_idx))
    metadata = {}
    if path is None or not os.path.exists(path):
        return metadata, arrays

    frame_lookup = {int(frame): i for i, frame in enumerate(frame_idx)}
    with open(path) as handle:
        first = json.loads(next(handle))
        rows = handle
        if first.get("meta"):
            metadata = first
        else:
            rows = chain([json.dumps(first)], handle)
        seen = set()
        for line in rows:
            record = json.loads(line)
            frame = int(record["frame"])
            if frame in seen:
                raise ValueError(f"{path}: duplicate hand frame {frame}")
            seen.add(frame)
            index = frame_lookup.get(frame)
            if index is None:
                continue
            arrays["valid"][index] = True
            arrays["interpolated"][index] = bool(
                record.get("interpolated", False))
            if "source_frames" in record:
                arrays["source_frames"][index] = record["source_frames"]
            field_map = {
                "trans": "root_camera_m",
                "joints_3d_cam": "joints_camera_m",
                "global_orient_R": "wrist_rotation_camera",
                "quat": "joint_quaternion_wxyz",
                "trans_virtual": "translation_virtual_m",
                "Rv_l": "virtual_to_camera_rotation",
                "phase1_mean_reproj_px": "phase1_mean_reprojection_px",
                "phase1_p90_reproj_px": "phase1_p90_reprojection_px",
                "phase1_median_epipolar_px":
                    "phase1_median_epipolar_px",
                "phase1_depth_m": "phase1_depth_m",
            }
            for source, destination in field_map.items():
                if source in record:
                    arrays[destination][index] = record[source]

    valid = arrays["valid"]
    rotation = world_from_left[:, :3, :3]
    translation = world_from_left[:, :3, 3]
    arrays["root_world_m"] = np.full_like(arrays["root_camera_m"], np.nan)
    arrays["joints_world_m"] = np.full_like(arrays["joints_camera_m"], np.nan)
    arrays["wrist_rotation_world"] = np.full_like(
        arrays["wrist_rotation_camera"], np.nan)
    arrays["root_world_m"][valid] = (
        np.einsum("nij,nj->ni", rotation[valid],
                  arrays["root_camera_m"][valid]) + translation[valid])
    arrays["joints_world_m"][valid] = (
        np.einsum("nij,nkj->nki", rotation[valid],
                  arrays["joints_camera_m"][valid])
        + translation[valid, None, :])
    arrays["wrist_rotation_world"][valid] = np.einsum(
        "nij,njk->nik", rotation[valid],
        arrays["wrist_rotation_camera"][valid])
    return metadata, arrays


def _write_hand(group, metadata, arrays, source_path, is_right):
    group.attrs["source_file"] = os.path.basename(source_path)
    group.attrs["is_right"] = bool(metadata.get("is_right", is_right))
    group.attrs["mirror"] = float(metadata.get("mirror", np.nan))
    group.attrs["source_metadata_json"] = json.dumps(
        metadata, separators=(",", ":"))
    if "beta_opt" in metadata:
        _dataset(
            group, "betas", np.asarray(metadata["beta_opt"], np.float32),
            "Recording-level MANO shape coefficients.")
    descriptions = {
        "valid": "True where this hand has an accepted measured or interpolated pose.",
        "interpolated": "True where the pose fills a short enclosed detection gap.",
        "source_frames": "Bounding measured frame indices for interpolated poses; -1 otherwise.",
        "root_camera_m": "MANO root position in the left OpenCV camera frame.",
        "root_world_m": "MANO root position in the VIO world frame.",
        "joints_camera_m": "21 MANO joints in the left OpenCV camera frame.",
        "joints_world_m": "21 MANO joints in the VIO world frame.",
        "wrist_rotation_camera": "Hand-local to left-camera wrist rotation matrix.",
        "wrist_rotation_world": "Hand-local to VIO-world wrist rotation matrix.",
        "joint_quaternion_wxyz": "MANO joint-local unit quaternions in wxyz order.",
        "translation_virtual_m": "MANO root translation in the rectified virtual camera.",
        "virtual_to_camera_rotation": "Rectified virtual-camera to left-camera rotation.",
        "phase1_mean_reprojection_px": "Mean stereo reprojection error before temporal smoothing.",
        "phase1_p90_reprojection_px": "90th-percentile reprojection error before smoothing.",
        "phase1_median_epipolar_px": "Median stereo vertical disagreement before smoothing.",
        "phase1_depth_m": "Root depth before temporal smoothing.",
    }
    for name, value in arrays.items():
        units = "m" if name.endswith("_m") else (
            "px" if name.endswith("_px") else None)
        _dataset(
            group, name, value, descriptions[name], units=units,
            frame_aligned=True)


def export_recording(recording, trajectory_path, output_path,
                     hands_left=None, hands_right=None, imu_path=None,
                     voice_path=None, overwrite=False, video_frame_count=None):
    recording = os.path.abspath(recording)
    trajectory_path = os.path.abspath(trajectory_path)
    output_path = os.path.abspath(output_path)
    if os.path.exists(output_path) and not overwrite:
        raise FileExistsError(
            f"{output_path} exists; pass --overwrite to replace it")

    metadata, _ = _load_recording_metadata(recording)
    if video_frame_count is None:
        video_frame_count = _video_frame_count(recording, metadata)
    cameras, stereo, stereo_path, right_from_left = _load_calibration(
        recording, metadata)
    frame_idx, camera_from_world, interpolation_time_s = _load_trajectory(
        trajectory_path, right_from_left, video_frame_count)
    world_from_camera = invert_transforms(camera_from_world)
    frame_time_us, frame_time_valid = _load_frame_times(imu_path, frame_idx)
    if interpolation_time_s is None:
        interpolation_time_s = _repaired_relative_times(frame_time_us)
    voice = _load_voice(
        voice_path, frame_idx, frame_time_us, frame_time_valid)
    left_meta, left_hand = _load_hand(
        hands_left, frame_idx, world_from_camera[:, 0])
    right_meta, right_hand = _load_hand(
        hands_right, frame_idx, world_from_camera[:, 0])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    temporary_path = output_path + ".tmp"
    if os.path.exists(temporary_path):
        os.remove(temporary_path)
    try:
        with h5py.File(temporary_path, "w") as output:
            output.attrs["format"] = FORMAT
            output.attrs["schema_version"] = SCHEMA_VERSION
            output.attrs["created_utc"] = datetime.now(
                timezone.utc).isoformat()
            output.attrs["length_unit"] = "m"
            output.attrs["time_unit"] = "us"
            output.attrs["coordinate_system"] = (
                "OpenCV camera axes: +x right, +y down, +z forward")
            output.attrs["transform_convention"] = (
                "A_from_B maps homogeneous coordinates from frame B to A")
            output.attrs["video_storage"] = "external MP4"
            output.attrs["source_trajectory"] = os.path.relpath(
                trajectory_path, recording)
            output.create_dataset(
                "recording_json",
                data=json.dumps(metadata, separators=(",", ":")),
                dtype=h5py.string_dtype("utf-8"),
            ).attrs["description"] = (
                "Verbatim recording.json metadata at export time.")

            frames = output.create_group("frames")
            _dataset(
                frames, "index", frame_idx,
                "Zero-based native video frame index.", frame_aligned=True)
            _dataset(
                frames, "time_us", frame_time_us,
                "Frame time on imu_log.csv host_time_us clock; -1 if unavailable.",
                units="us", frame_aligned=True)
            _dataset(
                frames, "time_s", interpolation_time_s,
                "Monotonic relative observed-camera timeline used by VIO and hand smoothing.",
                units="s", frame_aligned=True)
            _dataset(
                frames, "time_valid", frame_time_valid,
                "True where time_us is available and strictly monotonic.",
                frame_aligned=True)

            camera_group = output.create_group("cameras")
            text = h5py.string_dtype("utf-8")
            camera_group.create_dataset(
                "names", data=np.asarray(["left", "right"], dtype=object),
                dtype=text)
            camera_group.create_dataset(
                "serials",
                data=np.asarray([camera["serial"] for camera in cameras],
                                dtype=object),
                dtype=text)
            camera_group.create_dataset(
                "video_files",
                data=np.asarray([
                    metadata["files"]["left"], metadata["files"]["right"]
                ], dtype=object),
                dtype=text,
            )
            _dataset(
                camera_group, "camera_from_world",
                camera_from_world.astype(np.float32),
                "World-to-camera homogeneous transforms for [left, right].",
                frame_aligned=True)
            _dataset(
                camera_group, "world_from_camera",
                world_from_camera.astype(np.float32),
                "Camera-to-world homogeneous transforms for [left, right].",
                frame_aligned=True)
            _dataset(
                camera_group, "K",
                np.stack([camera["K"] for camera in cameras]).astype(np.float32),
                "ROI-adjusted OpenCV intrinsic matrices for [left, right].")
            _dataset(
                camera_group, "distortion",
                np.stack([camera["distortion"] for camera in cameras]).astype(
                    np.float32),
                "OpenCV fisheye distortion coefficients for [left, right].")
            _dataset(
                camera_group, "image_size",
                np.stack([camera["image_size"] for camera in cameras]),
                "Image [width, height] for [left, right].", units="px")
            camera_group.attrs["camera_model"] = cameras[0]["model"]
            camera_group.attrs["calibration_files"] = json.dumps([
                os.path.relpath(camera["path"], recording)
                for camera in cameras
            ])
            _dataset(
                camera_group, "right_from_left",
                right_from_left.astype(np.float32),
                "Calibrated left-camera to right-camera homogeneous transform.")
            camera_group["right_from_left"].attrs["source_file"] = (
                os.path.relpath(stereo_path, recording))
            camera_group.attrs["baseline_m"] = float(stereo["baseline"])

            hands = output.create_group("hands")
            _write_hand(
                hands.create_group("left"), left_meta, left_hand,
                hands_left or "", is_right=False)
            _write_hand(
                hands.create_group("right"), right_meta, right_hand,
                hands_right or "", is_right=True)
            _write_voice(output.create_group("voice"), voice, voice_path)
        os.replace(temporary_path, output_path)
    except BaseException:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)
        raise
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording")
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--hands-left", default=None)
    parser.add_argument("--hands-right", default=None)
    parser.add_argument("--imu", default=None)
    parser.add_argument("--voice", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    derived = os.path.join(args.recording, "derived")
    hands_left = args.hands_left or os.path.join(
        derived, "hands3d_left.jsonl")
    hands_right = args.hands_right or os.path.join(
        derived, "hands3d_right.jsonl")
    imu_path = args.imu or os.path.join(derived, "imu_relative.npz")
    voice_path = args.voice or os.path.join(
        derived, "voice_transcript.json")
    output_path = args.out or os.path.join(derived, "training.h5")
    result = export_recording(
        args.recording,
        args.trajectory,
        output_path,
        hands_left=hands_left,
        hands_right=hands_right,
        imu_path=imu_path,
        voice_path=voice_path,
        overwrite=args.overwrite,
    )
    with h5py.File(result, "r") as output:
        frame_count = len(output["frames/index"])
        left_count = int(output["hands/left/valid"][:].sum())
        right_count = int(output["hands/right/valid"][:].sum())
        word_count = len(output["voice/words/text"])
    size_mb = os.path.getsize(result) / 2 ** 20
    print(
        f"wrote {result}: {frame_count} frames, "
        f"{left_count} left-hand poses, {right_count} right-hand poses, "
        f"{word_count} voice words, "
        f"{size_mb:.1f} MiB")


if __name__ == "__main__":
    main()
