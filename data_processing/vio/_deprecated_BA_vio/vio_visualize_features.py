"""Visualize stage 1 (vio_extract_features.py): overlay extracted SuperPoint
keypoints + the FOV mask boundary on the source video, left/right side by side.

Run (from data_processing/vio/):
    python vio_visualize_features.py ../../long-test1 \
        --features ../../long-test1/features.h5 --out ../../long-test1/features_viz.mp4
"""
import argparse
import json
import os

import cv2
import h5py
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("recording", help="recording dir with left.mp4/right.mp4/recording.json")
    p.add_argument("--features", default=None, help="default: <recording>/features.h5")
    p.add_argument("--out", default=None, help="default: <recording>/features_viz.mp4")
    p.add_argument("--eye", choices=("left", "right", "both"), default="both")
    p.add_argument("--max-frames", type=int, default=None)
    return p.parse_args()


def fov_boundary_px(K, dist, theta_max, n=180):
    """(n,2) pixel loop tracing the FOV mask boundary (angle == theta_max from
    the optical axis), via cv2.fisheye's OWN forward projection — not a
    re-derived polynomial, so it's exactly consistent with the inverse
    (fisheye_pinhole.fisheye_unproject) used at extraction time."""
    az = np.linspace(0, 2 * np.pi, n, endpoint=False)
    rays = np.stack([
        np.sin(theta_max) * np.cos(az),
        np.sin(theta_max) * np.sin(az),
        np.full_like(az, np.cos(theta_max)),
    ], axis=-1).astype(np.float64).reshape(-1, 1, 3)
    px, _ = cv2.fisheye.projectPoints(
        rays, np.zeros((3, 1)), np.zeros((3, 1)), K.astype(np.float64),
        dist.reshape(4, 1).astype(np.float64))
    return px.reshape(-1, 2)


def draw_frame(frame, kp, boundary, label):
    for x, y in kp:
        cv2.circle(frame, (int(round(x)), int(round(y))), 3, (0, 255, 0), -1, cv2.LINE_AA)
    cv2.polylines(frame, [boundary.astype(np.int32)], True, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 0), 2, cv2.LINE_AA)
    return frame


def main():
    args = parse_args()
    features_path = args.features or os.path.join(args.recording, "features.h5")
    out_path = args.out or os.path.join(args.recording, "features_viz.mp4")
    eyes = ["left", "right"] if args.eye == "both" else [args.eye]

    with h5py.File(features_path, "r") as f:
        theta_max = np.radians(float(f.attrs["fov_deg"]) / 2.0)
        fps = float(f.attrs["fps"])

        caps, boundaries = {}, {}
        for eye in eyes:
            caps[eye] = cv2.VideoCapture(os.path.join(args.recording, f"{eye}.mp4"))
            serial = f.attrs[f"{eye}_serial"]
            calib = json.load(open(os.path.join(
                args.recording, f"calib_{serial}.json")))["intrinsics"]
            boundaries[eye] = fov_boundary_px(
                np.array(calib["K"]), np.array(calib["dist"]), theta_max)

        n_frames = f[eyes[0]]["counts"].shape[0]
        if args.max_frames:
            n_frames = min(n_frames, args.max_frames)

        w = int(caps[eyes[0]].get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(caps[eyes[0]].get(cv2.CAP_PROP_FRAME_HEIGHT))
        # avc1 (H.264) so the mp4 plays in QuickTime/browsers without re-encoding.
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"avc1"),
                                  fps, (w * len(eyes), h))
        if not writer.isOpened():
            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                      fps, (w * len(eyes), h))

        for i in range(n_frames):
            panels = []
            for eye in eyes:
                ok, frame = caps[eye].read()
                if not ok:
                    break
                kp = f[eye]["keypoints"][i].reshape(-1, 2)
                draw_frame(frame, kp, boundaries[eye], f"{eye} frame {i} n={kp.shape[0]}")
                panels.append(frame)
            if len(panels) != len(eyes):
                break
            writer.write(np.hstack(panels) if len(panels) > 1 else panels[0])
            if i % 200 == 0:
                print(f"{i}/{n_frames}")

        writer.release()
        for cap in caps.values():
            cap.release()

    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
