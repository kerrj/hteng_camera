import json
import tempfile
import unittest
from pathlib import Path

from transcribe_audio import (
    AudioInput,
    _default_output,
    _manifest_confirms_perf_counter,
    _normalize_segments,
    _relative_duration_s,
    _resolve_audio_input,
)


class TranscribeAudioTest(unittest.TestCase):
    def test_recording_input_and_manifest_clock(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "audio.mka").touch()
            (root / "recording.json").write_text(
                json.dumps(
                    {
                        "files": {"audio": "audio.mka"},
                        "audio": {
                            "enabled": True,
                            "file": "audio.mka",
                            "backend": "avfoundation",
                            "clock_alignment": (
                                "Packet PTS uses time.perf_counter()."
                            ),
                        },
                    }
                )
            )

            audio_input = _resolve_audio_input(root)

            self.assertEqual(audio_input.path, (root / "audio.mka").resolve())
            self.assertTrue(_manifest_confirms_perf_counter(audio_input))
            self.assertEqual(
                _default_output(audio_input),
                root.resolve() / "derived" / "voice_transcript.json",
            )

    def test_unverified_file_is_not_labeled_perf_counter(self) -> None:
        audio_input = AudioInput(
            path=Path("/tmp/audio.mka"),
            recording_dir=None,
            manifest=None,
        )
        self.assertFalse(_manifest_confirms_perf_counter(audio_input))

    def test_segment_and_word_pts_are_offset_from_audio_start(self) -> None:
        raw = [
            {
                "id": 4,
                "start": 1.25,
                "end": 2.5,
                "text": " start demo ",
                "words": [
                    {
                        "word": " start",
                        "start": 1.25,
                        "end": 1.75,
                        "probability": 0.9,
                    }
                ],
            }
        ]

        segments = _normalize_segments(
            raw,
            audio_start_pts_us=382_330_111_000,
            perf_counter_aligned=True,
            include_words=True,
        )

        segment = segments[0]
        self.assertEqual(segment["text"], "start demo")
        self.assertEqual(segment["start_pts_us"], 382_331_361_000)
        self.assertEqual(
            segment["start_perf_counter_us"], segment["start_pts_us"]
        )
        self.assertEqual(
            segment["words"][0]["start_pts_us"], 382_331_361_000
        )
        self.assertEqual(segment["words"][0]["text"], "start")

    def test_absolute_matroska_duration_is_made_relative(self) -> None:
        duration = _relative_duration_s(
            "382331.111000",
            start_pts_us=382_330_111_000,
        )
        self.assertEqual(duration, 1.0)


if __name__ == "__main__":
    unittest.main()
