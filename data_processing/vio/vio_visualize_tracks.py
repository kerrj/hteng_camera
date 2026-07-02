"""Visualize stage 3 (vio_build_tracks.py): render tracked keypoints over
video as colored dot-with-fading-tail "comets" -- color encodes track
identity (stable for the whole clip, golden-ratio hue spacing keyed by
track_id), the tail is each track's recent trajectory (older = dimmer), and a
bright dot marks its position in the CURRENT frame specifically (a track can
show a tail with no current dot if it wasn't observed exactly this frame --
a normal gap in an otherwise-live track, not an error).

Track selection is LOCAL PER FRAME, not a single global pick for the whole
video: up to --tracks-per-frame tracks with an observation in the current
tail window are drawn, preferring longer tracks when there's competition for
display slots. A single global "top N longest overall" selection (the first
version of this script) systematically starves segments where tracks are
individually shorter-lived (e.g. motion blur / occlusion) even though those
segments have PLENTY of valid tracks available -- confirmed directly: a
frame that showed 5/300 "active" under the old global selection actually had
416 real tracks with an observation there, just not among the 300 tracks
that happened to be longest-lived across the ENTIRE multi-minute video.

Streams video frames one at a time (cv2.VideoCapture.read() in the render
loop) rather than preloading the whole clip into memory -- a full-video
render otherwise holds the entire source video's raw frames in a Python
list simultaneously (~100GB+ for a multi-minute 5MP recording), which
happened to be safe on a 1TB-RAM box but is not a reasonable general
assumption.

Run (from data_processing/vio/):
    python vio_visualize_tracks.py ../../long-test1 --tracks ../../long-test1/tracks.jsonl \
        --eye left --max-frames 300 --out ../../long-test1/tracks_viz.mp4
"""
import argparse
import bisect
import colorsys
import json
import os
from collections import defaultdict

import cv2
import h5py

GOLDEN_RATIO_CONJ = 0.6180339887498949


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("recording")
    p.add_argument("--tracks", default=None, help="default: <recording>/tracks.jsonl")
    p.add_argument("--features", default=None, help="default: <recording>/features.h5 (for fps)")
    p.add_argument("--eye", choices=("left", "right"), default="left")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--tracks-per-frame", type=int, default=100,
                    help="target number of tracks drawn per frame, selected locally "
                         "from whatever's active in that frame's tail window -- "
                         "not a single fixed set for the whole video. 0 = no cap, "
                         "draw every active track every frame")
    p.add_argument("--tail-length", type=int, default=20)
    p.add_argument("--out", default=None)
    return p.parse_args()


def track_color(track_id):
    hue = (track_id * GOLDEN_RATIO_CONJ) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 1.0)
    return (int(b * 255), int(g * 255), int(r * 255))  # BGR for cv2


def to_pt(px):
    return (int(round(px[0])), int(round(px[1])))


def main():
    args = parse_args()
    tracks_path = args.tracks or os.path.join(args.recording, "tracks.jsonl")
    features_path = args.features or os.path.join(args.recording, "features.h5")

    tracks = []  # {"track_id", "frames": [sorted], "px": [matching]}
    with open(tracks_path) as f:
        for line in f:
            r = json.loads(line)
            obs = [o for o in r["observations"] if o["eye"] == args.eye]
            if args.max_frames is not None:
                obs = [o for o in obs if o["frame"] < args.max_frames]
            if len(obs) < 2:
                continue
            obs.sort(key=lambda o: o["frame"])
            tracks.append({"track_id": r["track_id"],
                            "frames": [o["frame"] for o in obs],
                            "px": [o["px"] for o in obs]})
    if not tracks:
        raise SystemExit(f"no tracks with >=2 observations in eye={args.eye} "
                          f"(max_frames={args.max_frames})")
    print(f"{len(tracks)} candidate tracks with >=2 observations in eye={args.eye}")

    # Inverted index: frame -> [track indices with an EXACT observation there].
    # Used to incrementally maintain the sliding-window active set below
    # (O(total observations) over the whole render, not O(tracks * frames)).
    frame_to_tracks = defaultdict(list)
    max_frame = 0
    for ti, t in enumerate(tracks):
        for fr in t["frames"]:
            frame_to_tracks[fr].append(ti)
            if fr > max_frame:
                max_frame = fr
    if args.max_frames is not None:
        max_frame = min(max_frame, args.max_frames - 1)

    colors = [track_color(t["track_id"]) for t in tracks]

    with h5py.File(features_path, "r") as f:
        fps = float(f.attrs["fps"])

    out_path = args.out or os.path.join(args.recording, f"tracks_viz_{args.eye}.mp4")
    cap = cv2.VideoCapture(os.path.join(args.recording, f"{args.eye}.mp4"))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    # observations-currently-in-window count per track index; a track is
    # "active" at frame i iff its count here is > 0.
    active_count = defaultdict(int)

    i = 0
    while i <= max_frame:
        ok, canvas = cap.read()
        if not ok:
            break

        for ti in frame_to_tracks.get(i, ()):
            active_count[ti] += 1
        expired_frame = i - args.tail_length - 1
        if expired_frame in frame_to_tracks:
            for ti in frame_to_tracks[expired_frame]:
                active_count[ti] -= 1
                if active_count[ti] <= 0:
                    del active_count[ti]

        active_ti = list(active_count.keys())
        n_active_total = len(active_ti)
        if args.tracks_per_frame and len(active_ti) > args.tracks_per_frame:
            active_ti.sort(key=lambda ti: -len(tracks[ti]["frames"]))
            active_ti = active_ti[:args.tracks_per_frame]

        for ti in active_ti:
            t = tracks[ti]
            color = colors[ti]
            lo = bisect.bisect_left(t["frames"], i - args.tail_length)
            hi = bisect.bisect_right(t["frames"], i)
            frs, pxs = t["frames"][lo:hi], t["px"][lo:hi]
            for k in range(1, len(frs)):
                age_frac = (frs[k] - (i - args.tail_length)) / max(args.tail_length, 1)
                faded = tuple(int(c * (0.25 + 0.75 * age_frac)) for c in color)
                cv2.line(canvas, to_pt(pxs[k - 1]), to_pt(pxs[k]), faded, 1, cv2.LINE_AA)
            if frs and frs[-1] == i:
                cv2.circle(canvas, to_pt(pxs[-1]), 4, color, -1, cv2.LINE_AA)

        cv2.putText(canvas, f"frame {i}  drawn={len(active_ti)}  total_active={n_active_total}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
        writer.write(canvas)
        if i % 200 == 0:
            print(f"{i}/{max_frame}")
        i += 1

    cap.release()
    writer.release()
    print(f"wrote {out_path} (mp4v -- re-encode with ffmpeg to h264 before transferring)")


if __name__ == "__main__":
    main()
