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


def load_hand_mesh(path):
    """Precomputed MANO meshes (precompute_hand_meshes.py): -> ({frame: (778,3)}, faces)."""
    if not path or not os.path.exists(path):
        return {}, None
    d = np.load(path)
    verts, frames, faces = d["verts"], d["frames"], d["faces"]
    return {int(f): verts[i] for i, f in enumerate(frames)}, faces


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
    ap.add_argument("--max-points", type=int, default=400_000)
    # Precomputed MANO hand meshes in the left-fisheye frame (778 verts/frame),
    # from precompute_hand_meshes.py (jkerr's viz_*.jsonl + full-mesh MANO forward).
    ap.add_argument("--hand-mesh-left", default="data_processing/out/hand_mesh_left.npz")
    ap.add_argument("--hand-mesh-right", default="data_processing/out/hand_mesh_right.npz")
    ap.add_argument("--selftest", type=int, nargs="*", default=None)
    args = ap.parse_args()

    sc = Scene(args.out_dir)
    hands_L, facesL = load_hand_mesh(args.hand_mesh_left)
    hands_R, facesR = load_hand_mesh(args.hand_mesh_right)
    FACES = facesL if facesL is not None else facesR
    print(f"scene: {sc.total} frames  range-map {sc.Hf}x{sc.Wf}  {len(sc.stacks)} stacks  "
          f"hand-mesh L/R {len(hands_L)}/{len(hands_R)} frames")

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

    import threading
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

    # PERSISTENT scene handles, created ONCE and updated IN PLACE each frame
    # (after jkerr's viz_hands.py). Re-adding rebuilds + re-uploads the whole
    # object every frame — a ~700k-point cloud re-upload floods the browser and
    # crashes the tab (plus leaks handler threads). Meshes carry the real FACES
    # at creation; only .vertices/.visible/.opacity change.
    pc_h = server.scene.add_point_cloud(
        "/scene", np.zeros((1, 3), np.float32), np.zeros((1, 3), np.uint8),
        point_size=args.point_size, point_shape="rounded")
    tracks = {"left": (hands_L, (40, 220, 40)), "right": (hands_R, (250, 140, 20))}
    mesh_h = {name: server.scene.add_mesh_simple(
                  f"/hand_{name}", np.zeros((778, 3), np.float32), FACES, color=color,
                  opacity=0.55, side="double", flat_shading=False,
                  cast_shadow=False, visible=False)
              for name, (_, color) in tracks.items()}

    # the play loop and the scrub/slider callbacks fire on different threads;
    # serialize the whole render so two updates can't interleave on the shared
    # handles or the (thread-unsafe) cv2 decoder inside reconstruct().
    _lock = threading.Lock()

    def update(_=None):
        with _lock:
            fr = int(g_frame.value)
            p, c = sc.reconstruct(fr, g_mr.value, args.max_points)
            pc_h.points, pc_h.colors, pc_h.point_size = p, c, g_ps.value
            for name, (track, _color) in tracks.items():
                V = track.get(fr)
                if V is None or not g_hands.value:
                    mesh_h[name].visible = False
                else:
                    mesh_h[name].vertices = V.astype(np.float32)
                    mesh_h[name].opacity = g_op.value
                    mesh_h[name].visible = True

    for g in (g_frame, g_ps, g_mr, g_hands, g_op):
        g.on_update(update)
    update()

    print(f"viser player on :{args.port} — ssh -L {args.port}:localhost:{args.port} sphynx")
    while True:
        if g_play.value:
            nxt = int(g_frame.value) + 1
            g_frame.value = 0 if nxt >= sc.total else nxt   # fires update via on_update
        time.sleep(1.0 / g_fps.value)


if __name__ == "__main__":
    main()
