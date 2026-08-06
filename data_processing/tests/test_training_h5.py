import json
import os
import tempfile

import h5py
import numpy as np

import export_training_h5 as export


def _write_json(path, value):
    with open(path, "w") as handle:
        json.dump(value, handle)


def test_training_export_is_dense_and_uses_explicit_transforms():
    with tempfile.TemporaryDirectory() as recording:
        derived = os.path.join(recording, "derived")
        os.makedirs(derived)
        _write_json(os.path.join(recording, "recording.json"), {
            "format": "hteng-camera-stereo-recording/1",
            "fps": 30,
            "files": {
                "left": "left.mp4",
                "right": "right.mp4",
                "stereo_transform": "stereo_L_R.json",
            },
            "left": {
                "serial": "L",
                "calibration_file": "calib_L.json",
                "roi": {
                    "x_offset": 10, "y_offset": 20,
                    "width": 100, "height": 80,
                },
            },
            "right": {
                "serial": "R",
                "calibration_file": "calib_R.json",
                "roi": {
                    "x_offset": 0, "y_offset": 0,
                    "width": 100, "height": 80,
                },
            },
        })
        calibration = {
            "intrinsics": {
                "model": "fisheye",
                "K": [[50, 0, 60], [0, 50, 50], [0, 0, 1]],
                "dist": [0.1, 0.2, 0.3, 0.4],
            },
        }
        _write_json(os.path.join(recording, "calib_L.json"), calibration)
        _write_json(os.path.join(recording, "calib_R.json"), calibration)
        _write_json(os.path.join(recording, "stereo_L_R.json"), {
            "R": np.eye(3).tolist(),
            "t": [-0.07, 0, 0],
            "baseline": 0.07,
        })

        frames = np.arange(3, dtype=np.int64)
        left_pose = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 1, 0, 0],
            [1, 0, 0, 0, 2, 0, 0],
        ], np.float64)
        right_pose = left_pose.copy()
        right_pose[:, 4] -= 0.07
        trajectory = os.path.join(derived, "trajectory.npz")
        np.savez(
            trajectory,
            frame_idx=frames,
            pose_wxyz_xyz_left=left_pose,
            pose_wxyz_xyz_right=right_pose,
        )
        imu = os.path.join(derived, "imu_relative.npz")
        np.savez(
            imu,
            frame_idx=frames,
            frame_time_us=np.array([100, 200, 300]),
            frame_valid=np.array([True, True, True]),
        )

        hand_record = {
            "frame": 1,
            "is_right": 0,
            "trans": [0, 0, 1],
            "joints_3d_cam": np.tile([0, 0, 1], (21, 1)).tolist(),
            "Rv_l": np.eye(3).tolist(),
            "global_orient_R": np.eye(3).tolist(),
            "quat": np.tile([1, 0, 0, 0], (16, 1)).tolist(),
            "trans_virtual": [0, 0, 1],
            "interpolated": False,
            "phase1_mean_reproj_px": 2.0,
        }
        left_hands = os.path.join(derived, "hands3d_left.jsonl")
        with open(left_hands, "w") as handle:
            handle.write(json.dumps({
                "meta": True,
                "is_right": 0,
                "mirror": -1,
                "beta_opt": [0] * 10,
            }) + "\n")
            handle.write(json.dumps(hand_record) + "\n")
        right_hands = os.path.join(derived, "hands3d_right.jsonl")
        with open(right_hands, "w") as handle:
            handle.write(json.dumps({
                "meta": True,
                "is_right": 1,
                "mirror": 1,
                "beta_opt": [0] * 10,
            }) + "\n")
        voice = os.path.join(derived, "voice_transcript.json")
        _write_json(voice, {
            "format": "hteng-camera-voice-transcript/1",
            "source": {
                "audio_file": "audio.mka",
                "start_pts_us": 50,
                "pts_clock": "perf_counter",
            },
            "transcription": {
                "backend": "mlx",
                "model": "mlx-community/whisper-base-mlx",
                "language": "en",
            },
            "text": "start demo",
            "segments": [{
                "id": 7,
                "text": "start demo",
                "start_audio_s": 0.00005,
                "end_audio_s": 0.00025,
                "start_pts_us": 100,
                "end_pts_us": 300,
                "start_perf_counter_us": 100,
                "end_perf_counter_us": 300,
                "words": [
                    {
                        "text": "start",
                        "start_audio_s": 0.00005,
                        "end_audio_s": 0.00013,
                        "start_pts_us": 100,
                        "end_pts_us": 180,
                        "start_perf_counter_us": 100,
                        "end_perf_counter_us": 180,
                        "probability": 0.95,
                    },
                    {
                        "text": "demo",
                        "start_audio_s": 0.00017,
                        "end_audio_s": 0.00025,
                        "start_pts_us": 220,
                        "end_pts_us": 300,
                        "start_perf_counter_us": 220,
                        "end_perf_counter_us": 300,
                        "probability": 0.9,
                    },
                ],
            }],
        })

        output = os.path.join(derived, "training.h5")
        export.export_recording(
            recording,
            trajectory,
            output,
            hands_left=left_hands,
            hands_right=right_hands,
            imu_path=imu,
            voice_path=voice,
            video_frame_count=3,
        )

        with h5py.File(output, "r") as data:
            assert data.attrs["format"] == export.FORMAT
            assert data.attrs["schema_version"] == export.SCHEMA_VERSION
            np.testing.assert_array_equal(data["frames/index"], frames)
            np.testing.assert_array_equal(
                data["hands/left/valid"], [False, True, False])
            np.testing.assert_allclose(
                data["frames/time_s"], [0.0, 0.0001, 0.0002])
            assert np.isnan(data["hands/left/root_camera_m"][0]).all()
            np.testing.assert_allclose(
                data["hands/left/root_world_m"][1], [-1, 0, 1])
            np.testing.assert_allclose(
                data["cameras/camera_from_world"][1, 0, :3, 3], [1, 0, 0])
            np.testing.assert_allclose(
                data["cameras/world_from_camera"][1, 0, :3, 3], [-1, 0, 0])
            np.testing.assert_allclose(
                data["cameras/right_from_left"][:3, 3], [-0.07, 0, 0])
            np.testing.assert_allclose(data["cameras/K"][0, :2, 2], [50, 30])
            assert data["cameras/video_files"].asstr()[:].tolist() == [
                "left.mp4", "right.mp4"]
            assert bool(data["voice"].attrs["available"])
            assert data["voice/transcript"].asstr()[()] == "start demo"
            assert data["voice/words/text"].asstr()[:].tolist() == [
                "start", "demo"]
            np.testing.assert_array_equal(
                data["voice/words/start_time_us"], [100, 220])
            np.testing.assert_array_equal(
                data["voice/words/start_frame_index"], [0, 1])
            np.testing.assert_array_equal(
                data["voice/words/end_frame_index"], [1, 2])
            np.testing.assert_allclose(
                data["voice/words/probability"], [0.95, 0.9])
            np.testing.assert_array_equal(
                data["voice/words/segment_index"], [0, 0])

        empty_voice = export._load_voice(
            None,
            frames,
            np.array([100, 200, 300]),
            np.array([True, True, True]),
        )
        empty_output = os.path.join(derived, "empty_voice.h5")
        with h5py.File(empty_output, "w") as data:
            export._write_voice(data.create_group("voice"), empty_voice, None)
        with h5py.File(empty_output, "r") as data:
            assert not bool(data["voice"].attrs["available"])
            assert data["voice/transcript"].asstr()[()] == ""
            assert len(data["voice/segments/text"]) == 0
            assert len(data["voice/words/text"]) == 0


if __name__ == "__main__":
    test_training_export_is_dense_and_uses_explicit_transforms()
    print("training HDF5 export test passed")
