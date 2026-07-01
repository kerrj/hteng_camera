"""Scrub the FFS scene-depth video in viser: per-frame metric cloud, reconstructed
on the fly from the compact range maps (ffs_scene_batch output) + video colour.

A range map is a complete representation: unproject each fisheye pixel to a unit
ray (constant -> precomputed once) and scale by the stored range; colour is
sampled from the left-eye video frame. So per frame is just a masked multiply +
a colour gather + a decode. GUI: frame slider, play/pause, fps, point size,
max-range clip.

  python data_processing/ffs_scene_player.py --out-dir data_processing/out/video

Selftest (no server; validates reconstruction against whatever frames exist):
  python data_processing/ffs_scene_player.py --out-dir data_processing/out/video --selftest 0 3000 6000
"""
import argparse
import glob
import json
import os
import re
import sys
import time

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import fisheye_pinhole as FP
import hand_mesh as HM

# MANO 21-joint skeleton (matches render_hands_video._BONES)
_BONES = np.array([(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
                   (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15),
                   (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)])


def load_hands(path):
    """frame -> (21,3) joints (left-fisheye frame), or {} if no file."""
    d = {}
    if path and os.path.exists(path):
        for line in open(path):
            r = json.loads(line)
            d[r["frame"]] = np.asarray(r["joints_3d_cam"], np.float32)
    return d


class Scene:
    def __init__(self, out_dir):
        self.meta = json.load(open(f"{out_dir}/meta.json"))
        cl = json.load(open(f"{self.meta['calib_dir']}/calib_{self.meta['left_serial']}.json"))["intrinsics"]
        self.Kl = np.array(cl["K"], np.float64)
        self.Dl = np.array(cl["dist"], np.float64)
        self.scale = self.meta["scale"]
        # range-map stacks, sorted by start frame
        stacks = []
        for f in glob.glob(f"{out_dir}/range_*_*.npy"):
            s, e = map(int, re.search(r"range_(\d+)_(\d+)\.npy", f).groups())
            stacks.append((s, e, np.load(f, mmap_mode="r")))
        self.stacks = sorted(stacks, key=lambda x: x[0])
        self.total = max(e for _, e, _ in self.stacks)
        self.Hf, self.Wf = self.stacks[0][2].shape[1:]

        # precompute unit rays (left-cam frame) + full-res colour coords, ONCE
        ys, xs = np.meshgrid(np.arange(self.Hf), np.arange(self.Wf), indexing="ij")
        fx = xs / self.scale; fy = ys / self.scale                  # -> full-sensor px
        rays = FP.fisheye_unproject(torch.tensor(fx.ravel(), dtype=torch.float32),
                                    torch.tensor(fy.ravel(), dtype=torch.float32),
                                    torch.tensor(self.Kl, dtype=torch.float32),
                                    torch.tensor(self.Dl, dtype=torch.float32))
        self.rays = rays.numpy().reshape(self.Hf, self.Wf, 3)

        self.cap = cv2.VideoCapture(self.meta["video"])
        self.fish_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) // 2
        self.fish_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.u2 = np.clip(fx.astype(int), 0, self.fish_w - 1)
        self.v2 = np.clip(fy.astype(int), 0, self.fish_h - 1)
        self._cur = -1

    def range_of(self, f):
        for s, e, st in self.stacks:
            if s <= f < e:
                return np.asarray(st[f - s], np.float32)
        return np.zeros((self.Hf, self.Wf), np.float32)

    def left_rgb(self, f):
        if f != self._cur + 1:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, fr = self.cap.read()
        self._cur = f
        if not ok:
            return np.zeros((self.fish_h, self.fish_w, 3), np.uint8)
        return cv2.cvtColor(fr[:, :self.fish_w], cv2.COLOR_BGR2RGB)

    def reconstruct(self, f, max_range, max_points):
        rng = self.range_of(f)
        m = (rng > 0) & (rng <= max_range)
        if not m.any():
            return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8)
        pts = self.rays[m] * rng[m, None]
        rgb = self.left_rgb(f)
        cols = rgb[self.v2[m], self.u2[m]]
        if len(pts) > max_points:
            idx = np.random.choice(len(pts), max_points, replace=False)
            pts, cols = pts[idx], cols[idx]
        return pts.astype(np.float32), cols.astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data_processing/out/video")
    ap.add_argument("--port", type=int, default=8092)
    ap.add_argument("--point-size", type=float, default=0.006)
    ap.add_argument("--max-points", type=int, default=700_000)
    # *_fisheye.jsonl: joints rotated from the legacy virtual-crop frame into the
    # left-fisheye frame (fix_hands_frame.py). The raw hands3d_full_*.jsonl are in
    # the wrong frame and project to the image centre — do not use them directly.
    ap.add_argument("--hands-left", default="data_processing/out/hands3d_full_left_fisheye.jsonl")
    ap.add_argument("--hands-right", default="data_processing/out/hands3d_full_right_fisheye.jsonl")
    ap.add_argument("--selftest", type=int, nargs="*", default=None)
    args = ap.parse_args()

    sc = Scene(args.out_dir)
    hands_L, hands_R = load_hands(args.hands_left), load_hands(args.hands_right)
    print(f"scene: {sc.total} frames  range-map {sc.Hf}x{sc.Wf}  {len(sc.stacks)} stacks  "
          f"hands L/R {len(hands_L)}/{len(hands_R)} frames")

    if args.selftest is not None:
        for f in (args.selftest or [0]):
            p, c = sc.reconstruct(f, 20.0, args.max_points)
            if len(p):
                r = np.linalg.norm(p, axis=1)
                print(f"  frame {f}: {len(p):,} pts  range p5/50/95 "
                      f"{np.percentile(r,[5,50,95]).round(2)}m  color mean {c.mean(0).round(0)}")
            else:
                print(f"  frame {f}: empty (not written yet?)")
        return

    import viser
    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("-y")
    server.scene.add_frame("/left_cam", axes_length=0.15, axes_radius=0.005)
    g_frame = server.gui.add_slider("frame", 0, sc.total - 1, 1, 0)
    g_play = server.gui.add_checkbox("play", False)
    g_fps = server.gui.add_slider("fps", 1, 30, 1, 10)
    g_ps = server.gui.add_slider("point size", 0.001, 0.02, 0.001, args.point_size)
    g_mr = server.gui.add_slider("max range (m)", 0.3, 15.0, 0.1, 6.0)
    g_hands = server.gui.add_checkbox("show hands", True)
    g_op = server.gui.add_slider("hand opacity", 0.1, 1.0, 0.05, 0.55)

    def draw_hand(name, J, color):
        if J is None or len(J) == 0:                          # hide when absent
            server.scene.add_mesh_simple(name, np.zeros((3, 3), np.float32),
                                         np.array([[0, 1, 2]], np.int32), visible=False)
            return
        V, F = HM.hand_mesh(J)
        server.scene.add_mesh_simple(name, V, F, color=color, opacity=g_op.value,
                                     side="double", flat_shading=False, cast_shadow=False)

    def show(f):
        f = int(f)
        p, c = sc.reconstruct(f, g_mr.value, args.max_points)
        server.scene.add_point_cloud("/scene", points=p, colors=c,
                                     point_size=g_ps.value, point_shape="rounded")
        draw_hand("/hands/left", hands_L.get(f) if g_hands.value else None, (40, 220, 40))
        draw_hand("/hands/right", hands_R.get(f) if g_hands.value else None, (250, 140, 20))

    # ALL decoding happens in this thread only: cv2.VideoCapture is not
    # thread-safe (concurrent reads abort libavcodec). GUI callbacks run on
    # viser's websocket threads, so they must NOT decode — they only mutate
    # widget state, and this loop re-renders whenever anything changes.
    show(0)
    last = (0, g_ps.value, g_mr.value, g_hands.value, g_op.value)
    print(f"viser player on :{args.port} — ssh -L {args.port}:localhost:{args.port} sphynx")
    while True:
        if g_play.value:
            g_frame.value = (int(g_frame.value) + 1) % sc.total
        key = (int(g_frame.value), g_ps.value, g_mr.value, g_hands.value, g_op.value)
        if key != last:
            show(key[0])
            last = key
        time.sleep(1.0 / g_fps.value)


if __name__ == "__main__":
    main()
