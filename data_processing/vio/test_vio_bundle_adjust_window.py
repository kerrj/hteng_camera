import json

import vio_bundle_adjust as BA


def _write_tracks(path, recs):
    with open(path, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")


def test_load_tracks_min_max_window(tmp_path):
    p = str(tmp_path / "tracks.jsonl")
    _write_tracks(p, [
        # 10-frame track: frames 0..9 survive windowing to [3,7] with 5 obs
        {"observations": [{"eye": "left", "frame": f, "px": [1.0, 2.0]}
                          for f in range(10)]},
        # straddler: only 1 obs inside [3,7] -> dropped (<2 obs rule)
        {"observations": [{"eye": "left", "frame": 0, "px": [0.0, 0.0]},
                          {"eye": "right", "frame": 5, "px": [0.0, 0.0]},
                          {"eye": "left", "frame": 9, "px": [0.0, 0.0]}]},
    ])
    tracks = BA.load_tracks(p, 3, 7)
    assert len(tracks) == 1
    frames = [f for _, f, _ in tracks[0]]
    assert min(frames) == 3 and max(frames) == 7 and len(frames) == 5


def test_load_tracks_none_bounds_keep_everything(tmp_path):
    p = str(tmp_path / "tracks.jsonl")
    _write_tracks(p, [{"observations": [{"eye": "left", "frame": f, "px": [0.0, 0.0]}
                                        for f in range(4)]}])
    assert len(BA.load_tracks(p, None, None)[0]) == 4
