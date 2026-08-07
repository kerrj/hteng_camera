import csv
import tempfile
import unittest
from pathlib import Path

from split_recording import (
    Chunk,
    FrameTimes,
    Trigger,
    Window,
    Word,
    find_triggers,
    load_frame_times,
    plan_delimiter_windows,
    plan_paired_windows,
    resolve_chunks,
    slice_imu_log,
    slice_sync_log,
)


SECOND = 1_000_000


def _word(text, start_us, end_us, probability=1.0, no_speech_prob=0.0):
    from split_recording import _normalize

    return Word(
        text, _normalize(text), start_us, end_us, probability, no_speech_prob
    )


def _speak(phrase, start_us, probability=1.0, word_gap_us=100_000,
           no_speech_prob=0.0):
    """Words for `phrase` spoken back to back starting at `start_us`."""
    words = []
    cursor = start_us
    for token in phrase.split():
        words.append(
            _word(token, cursor, cursor + 200_000, probability, no_speech_prob)
        )
        cursor += 200_000 + word_gap_us
    return words


def _find(words, phrase, **overrides):
    options = {
        "min_probability": 0.9,
        "max_no_speech": 0.6,
        "max_word_gap_us": SECOND,
        "collapse_us": SECOND,
    }
    options.update(overrides)
    return find_triggers(words, phrase, "split", **options)


class FindTriggersTest(unittest.TestCase):
    def test_matches_normalized_multiword_phrase(self) -> None:
        words = _speak("Next,", 0) + _speak("Clip.", 500_000)
        triggers = _find(words, "next clip")
        self.assertEqual(len(triggers), 1)
        self.assertEqual(triggers[0].start_us, 0)
        self.assertEqual(triggers[0].end_us, 700_000)

    def test_low_confidence_match_is_rejected(self) -> None:
        words = _speak("next clip", 0, probability=0.62)
        self.assertEqual(_find(words, "next clip"), [])
        self.assertEqual(len(_find(words, "next clip", min_probability=0.5)), 1)

    def test_least_confident_word_gates_the_match(self) -> None:
        words = _word("next", 0, 100_000, 0.99), _word("clip", 200_000, 300_000, 0.4)
        self.assertEqual(_find(list(words), "next clip"), [])

    def test_words_spoken_far_apart_do_not_form_a_phrase(self) -> None:
        words = [_word("next", 0, 100_000), _word("clip", 30 * SECOND, 30 * SECOND)]
        self.assertEqual(_find(words, "next clip"), [])

    def test_repeats_within_the_collapse_window_are_one_command(self) -> None:
        words = _speak("mark", 0) + _speak("mark", 400_000) + _speak("mark", 5 * SECOND)
        triggers = _find(words, "mark")
        self.assertEqual([trigger.start_us for trigger in triggers], [0, 5 * SECOND])

    def test_a_stutter_chain_collapses_into_one_trigger(self) -> None:
        # Each repeat lands within the collapse window of the previous one, so
        # the burst is one boundary spanning all of it -- not four.
        words = []
        for start in (0, 700_000, 1_400_000, 2_100_000):
            words += _speak("mark", start)
        triggers = _find(words, "mark")
        self.assertEqual(len(triggers), 1)
        self.assertEqual(triggers[0].start_us, 0)
        self.assertEqual(triggers[0].end_us, 2_300_000)

    def test_confident_words_in_a_no_speech_segment_are_rejected(self) -> None:
        # Whisper looping on keyboard noise emits high per-word probability
        # inside a segment it flags as non-speech.
        words = _speak("mark", 0, probability=0.99, no_speech_prob=0.91)
        self.assertEqual(_find(words, "mark"), [])
        self.assertEqual(len(_find(words, "mark", max_no_speech=0.95)), 1)

    def test_phrase_is_not_matched_inside_a_longer_utterance(self) -> None:
        words = _speak("the next clipboard", 0)
        self.assertEqual(_find(words, "next clip"), [])


class PlanWindowsTest(unittest.TestCase):
    def _trigger(self, start_us, role="split"):
        return Trigger("mark", role, "mark", start_us, start_us + 500_000, 1.0)

    def test_a_crowded_delimiter_merges_instead_of_losing_footage(self) -> None:
        # The middle command comes too soon after the first, so it is ignored
        # and its span joins the following chunk rather than being discarded.
        triggers = [
            self._trigger(10 * SECOND),
            self._trigger(13 * SECOND),
            self._trigger(60 * SECOND),
        ]
        windows = plan_delimiter_windows(
            triggers, keep_head=False, keep_tail=False, min_span_us=10 * SECOND
        )
        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0].start_us, 10 * SECOND + 500_000)
        self.assertEqual(windows[0].end_us, 60 * SECOND)

    def test_delimiter_windows_are_contiguous_between_commands(self) -> None:
        triggers = [self._trigger(10 * SECOND), self._trigger(20 * SECOND)]
        windows = plan_delimiter_windows(triggers, keep_head=False, keep_tail=True)
        self.assertEqual(len(windows), 2)
        self.assertEqual(windows[0].start_us, 10 * SECOND + 500_000)
        self.assertEqual(windows[0].end_us, 20 * SECOND)
        self.assertEqual(windows[1].start_us, 20 * SECOND + 500_000)
        self.assertIsNone(windows[1].end_us)

    def test_head_and_tail_are_optional(self) -> None:
        triggers = [self._trigger(10 * SECOND)]
        windows = plan_delimiter_windows(triggers, keep_head=True, keep_tail=False)
        self.assertEqual(len(windows), 1)
        self.assertIsNone(windows[0].start_us)
        self.assertEqual(windows[0].end_us, 10 * SECOND)

    def test_paired_windows_span_start_to_stop(self) -> None:
        starts = [self._trigger(10 * SECOND, "start")]
        stops = [self._trigger(20 * SECOND, "stop")]
        windows = plan_paired_windows(starts, stops, extend_unclosed=False)
        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0].start_us, 10 * SECOND + 500_000)
        self.assertEqual(windows[0].end_us, 20 * SECOND)

    def test_unmatched_commands_are_dropped_or_extended(self) -> None:
        starts = [self._trigger(10 * SECOND, "start"), self._trigger(30 * SECOND, "start")]
        stops = [self._trigger(20 * SECOND, "stop"), self._trigger(5 * SECOND, "stop")]
        dropped = plan_paired_windows(starts, stops, extend_unclosed=False)
        self.assertEqual([window.start_us for window in dropped], [10 * SECOND + 500_000])
        extended = plan_paired_windows(starts, stops, extend_unclosed=True)
        self.assertEqual(len(extended), 2)
        self.assertIsNone(extended[1].end_us)

    def test_restarting_without_stopping_drops_the_open_span(self) -> None:
        starts = [self._trigger(10 * SECOND, "start"), self._trigger(15 * SECOND, "start")]
        stops = [self._trigger(20 * SECOND, "stop")]
        windows = plan_paired_windows(starts, stops, extend_unclosed=False)
        self.assertEqual([window.start_us for window in windows], [15 * SECOND + 500_000])


class ResolveChunksTest(unittest.TestCase):
    def setUp(self) -> None:
        # 100 frames at 25 ms, keyframes every 12.
        self.frame_times = FrameTimes(
            frames=list(range(100)),
            times_us=[1_000_000 + 25_000 * i for i in range(100)],
        )
        self.keyframes = list(range(0, 100, 12))

    def _resolve(self, windows, **overrides):
        options = {
            "pad_us": 500_000,
            "min_duration_us": 0,
            "name_prefix": "take",
        }
        options.update(overrides)
        return resolve_chunks(
            windows, self.frame_times, self.keyframes, 100, **options
        )

    def test_start_snaps_forward_to_a_keyframe_and_end_does_not(self) -> None:
        # Command ends at frame 10 (1.25 s); +0.5 s pad lands on frame 30, which
        # snaps up to keyframe 36. The closing command starts at frame 80.
        window = Window(1_250_000, 3_000_000, None, None)
        chunk = self._resolve([window])[0]
        self.assertEqual(chunk.start_frame, 36)
        self.assertIn(chunk.start_frame, self.keyframes)
        self.assertEqual(chunk.end_frame, 60)  # 3.0 s - 0.5 s pad = frame 60
        self.assertEqual(chunk.frame_count, 25)

    def test_open_ended_window_covers_the_whole_recording(self) -> None:
        chunk = self._resolve([Window(None, None, None, None)])[0]
        self.assertEqual((chunk.start_frame, chunk.end_frame), (0, 99))

    def test_short_chunks_are_dropped(self) -> None:
        window = Window(1_250_000, 3_000_000, None, None)
        self.assertEqual(self._resolve([window], min_duration_us=5 * SECOND), [])

    def test_commands_closer_than_the_gop_are_dropped(self) -> None:
        window = Window(1_250_000, 2_000_000, None, None)
        self.assertEqual(self._resolve([window]), [])

    def test_chunks_are_numbered_after_drops(self) -> None:
        windows = [
            Window(1_250_000, 2_000_000, None, None),  # dropped
            Window(1_250_000, 3_000_000, None, None),
        ]
        chunks = self._resolve(windows)
        self.assertEqual([chunk.name for chunk in chunks], ["take_000"])


class SliceLogsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.addCleanup(self.directory.cleanup)

    def _write_csv(self, name, fieldnames, rows):
        path = self.root / name
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def _sync_log(self, count=20):
        return self._write_csv(
            "sync_log.csv",
            ["frame", "t_left_us", "t_right_us", "skew_us", "fresh"],
            [
                {
                    "frame": i,
                    "t_left_us": 25_000 * i,
                    "t_right_us": 25_000 * i - 900,
                    "skew_us": -900,
                    "fresh": 1,
                }
                for i in range(count)
            ],
        )

    def _chunk(self, start_frame, end_frame):
        return Chunk(
            index=0,
            name="take_000",
            start_frame=start_frame,
            end_frame=end_frame,
            start_time_us=0,
            end_time_us=0,
            window=Window(None, None, None, None),
        )

    def test_sync_log_renumbers_frames_and_keeps_timestamps(self) -> None:
        source = self._sync_log()
        destination = self.root / "out_sync.csv"
        written = slice_sync_log(source, destination, self._chunk(5, 9))

        with destination.open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(written, 5)
        self.assertEqual([row["frame"] for row in rows], ["0", "1", "2", "3", "4"])
        self.assertEqual(rows[0]["t_left_us"], str(25_000 * 5))
        self.assertEqual(rows[-1]["t_left_us"], str(25_000 * 9))
        self.assertEqual(rows[0]["t_right_us"], str(25_000 * 5 - 900))

    def test_sync_log_row_count_must_match_the_frame_range(self) -> None:
        source = self._sync_log(count=8)
        with self.assertRaises(RuntimeError):
            slice_sync_log(source, self.root / "out.csv", self._chunk(5, 20))

    def test_imu_log_is_windowed_and_renumbered(self) -> None:
        source = self._write_csv(
            "imu_log.csv",
            ["sample", "host_time_us", "ax_g"],
            [
                {"sample": i, "host_time_us": 1_000_000 + 10_000 * i, "ax_g": 0.5}
                for i in range(100)
            ],
        )
        destination = self.root / "out_imu.csv"
        count, first_us, last_us = slice_imu_log(
            source, destination, 1_100_000, 1_200_000
        )

        with destination.open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(count, 11)
        self.assertEqual(first_us, 1_100_000)
        self.assertEqual(last_us, 1_200_000)
        self.assertEqual([row["sample"] for row in rows], [str(i) for i in range(11)])
        self.assertEqual(rows[0]["ax_g"], "0.5")

    def test_empty_imu_window_is_an_error(self) -> None:
        source = self._write_csv(
            "imu_log.csv",
            ["sample", "host_time_us"],
            [{"sample": 0, "host_time_us": 10}],
        )
        with self.assertRaises(RuntimeError):
            slice_imu_log(source, self.root / "out.csv", 5_000_000, 6_000_000)


class LoadFrameTimesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.addCleanup(self.directory.cleanup)

    def _write(self, rows):
        path = self.root / "sync_log.csv"
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["frame", "t_left_us"])
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_absolute_times_use_the_camera_reset_offset(self) -> None:
        path = self._write([{"frame": i, "t_left_us": 25_000 * i} for i in range(4)])
        times = load_frame_times(path, 1_000, 4)
        self.assertEqual(times.frames, [0, 1, 2, 3])
        self.assertEqual(times.times_us[0], 1_000)
        self.assertEqual(times.times_us[-1], 1_000 + 75_000)

    def test_pre_reset_timestamp_is_dropped(self) -> None:
        rows = [{"frame": 0, "t_left_us": 900_000_000}]
        rows += [{"frame": i, "t_left_us": 25_000 * i} for i in range(1, 4)]
        times = load_frame_times(self._write(rows), 0, 4)
        self.assertEqual(times.frames, [1, 2, 3])
        self.assertEqual(times.times_us, sorted(times.times_us))

    def test_frames_beyond_the_video_are_ignored(self) -> None:
        path = self._write([{"frame": i, "t_left_us": 25_000 * i} for i in range(6)])
        self.assertEqual(load_frame_times(path, 0, 4).frames, [0, 1, 2, 3])


if __name__ == "__main__":
    unittest.main()
