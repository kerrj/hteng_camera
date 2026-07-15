"""Viser viewer: static TSDF workspace mesh + animated MANO hand surfaces.

The mesh is the hand-masked static world (ffs_tsdf_segment.py); the hands are
re-posed per frame from the baked hand npzs and placed in the world by the
same trajectory the TSDF used -- clean scene + moving hands, one world frame.

  python data_processing/ffs_tsdf_viewer.py --prefix data_processing/out/lt2_seg \
      --hands-dir data_processing/out/lt2_hands \
      --trajectory long-test2/derived/trajectory.npz --share
"""
import argparse
import json
import time

import numpy as np
import open3d as o3d
import trimesh
import viser

from ffs_tsdf_segment import quat_to_R, R_to_quat  # same pose convention


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="data_processing/out/lt2_seg")
    ap.add_argument("--hands-dir", default="data_processing/out/lt2_hands")
    ap.add_argument("--trajectory", default="long-test2/derived/trajectory_vggt_omega_fullrun_20260714.npz")
    ap.add_argument("--port", type=int, default=8091)
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--fps", type=float, default=None,
                    help="playback rate; default: the recording fps from the "
                         "segment json (falls back to 30)")
    ap.add_argument("--dynamic", default=None,
                    help="<prefix>_dynamic.npz from ffs_dynamic_residual.py: "
                         "per-frame residual points played over the static mesh")
    ap.add_argument("--dynamic-point-size", type=float, default=0.004)
    ap.add_argument("--dynamic-max-points", type=int, default=25_000,
                    help="per-frame cap (random subsample) -- keeps the "
                         "websocket stream viable at playback rates")
    ap.add_argument("--mesh", default=None,
                    help="mesh ply override (e.g. a decimated _tsdf_view.ply "
                         "for big segments); default <prefix>_tsdf_mesh.ply")
    ap.add_argument("--gamma", type=float, default=2.2,
                    help="linearize sRGB-stored colors before serving: "
                         "three.js treats vertex colors as LINEAR and applies "
                         "an sRGB output transform, so serving sRGB values "
                         "as-is double-encodes and washes everything pastel. "
                         "Set 1.0 to disable.")
    args = ap.parse_args()

    seg = json.load(open(f"{args.prefix}_segment.json"))
    s0, e0 = seg["start"], seg["end"]
    if args.fps is None:
        args.fps = seg.get("fps", 30.0)

    mesh = o3d.io.read_triangle_mesh(args.mesh or f"{args.prefix}_tsdf_mesh.ply")
    vcol = np.power(np.clip(np.asarray(mesh.vertex_colors), 0, 1), args.gamma)
    tm = trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles),
                         vertex_colors=(vcol * 255).astype(np.uint8),
                         process=False)

    tr = np.load(args.trajectory)
    pose_of = {int(f): i for i, f in enumerate(tr["frame_idx"])}
    poses = tr["pose_wxyz_xyz"]

    hands = {}
    for side in ("left", "right"):
        d = np.load(f"{args.hands_dir}/hand_mesh_{side}.npz")
        hands[side] = (dict(zip(d["frames"].tolist(), d["verts"])), d["faces"])

    dyn = None
    if args.dynamic:
        d = np.load(args.dynamic)
        dcol = (np.power(d["colors"] / 255.0, args.gamma) * 255).astype(np.uint8)
        dyn = {"points": d["points"], "colors": dcol,
               "slice_of": {int(f): (int(d["offsets"][k]), int(d["offsets"][k + 1]))
                            for k, f in enumerate(d["frames"])}}

    server = viser.ViserServer(port=args.port)
    if args.share:
        print(f"SHARE URL: {server.request_share_url()}", flush=True)
    server.scene.set_up_direction("+z")
    server.scene.add_mesh_trimesh("/scene", tm)

    hand_h = {}
    for side, color in (("left", (231, 76, 60)), ("right", (46, 204, 113))):
        faces = hands[side][1]
        hand_h[side] = server.scene.add_mesh_simple(
            f"/hands/{side}", vertices=np.zeros((778, 3), np.float32),
            faces=faces, color=color, opacity=0.85, visible=False)

    dyn_h = None
    if dyn is not None:
        dyn_h = server.scene.add_point_cloud(
            "/dynamic", points=np.zeros((1, 3), np.float32),
            colors=np.zeros((1, 3), np.uint8),
            point_size=args.dynamic_point_size, point_shape="rounded")

    # head path: camera centers over the segment (early=blue -> late=red),
    # plus a frustum glyph at the current frame showing the gaze direction
    path_frames = [vf for vf in range(s0, e0) if vf in pose_of]
    centers = np.array([-(quat_to_R(poses[pose_of[vf]][:4]).T
                          @ poses[pose_of[vf]][4:]) for vf in path_frames])
    segs = np.stack([centers[:-1], centers[1:]], axis=1).astype(np.float32)
    tgrad = np.linspace(0, 1, len(segs))[:, None]
    seg_col = np.stack([(np.concatenate([60 + 195 * tgrad, 80 + 0 * tgrad,
                                         255 - 195 * tgrad], 1))] * 2, axis=1)
    path_h = server.scene.add_line_segments(
        "/head/path", points=segs, colors=seg_col.astype(np.uint8),
        line_width=3.0)
    frus_h = server.scene.add_camera_frustum(
        "/head/cam", fov=np.radians(70), aspect=4 / 3, scale=0.1,
        line_width=2.5, color=(255, 200, 40))

    g_frame = server.gui.add_slider("frame", s0, e0 - 1, 1, s0)
    g_play = server.gui.add_checkbox("play", True)
    g_mano = server.gui.add_checkbox("MANO hands", True)
    g_dyn = server.gui.add_checkbox("residual points", dyn is not None)
    g_path = server.gui.add_checkbox("head path", True)

    def on_path_toggle(_):
        path_h.visible = g_path.value
        frus_h.visible = g_path.value
    g_path.on_update(on_path_toggle)
    # NO shared lock: holding one across viser setters deadlocks against viser's
    # internal update lock (ABBA). Single-writer instead: the play loop is the
    # only caller of show() while playing; the scrub callback only acts paused.

    def show(vf):
        i = pose_of.get(int(vf))
        if i is None:
            return
        R = quat_to_R(poses[i, :4])
        c = -(R.T @ poses[i, 4:])
        if g_path.value:
            frus_h.wxyz = R_to_quat(R.T)     # cam->world; viser looks +z
            frus_h.position = c
        for side in ("left", "right"):
            V = hands[side][0].get(int(vf))
            h = hand_h[side]
            if V is None or not g_mano.value:
                h.visible = False
                continue
            h.vertices = (np.asarray(V) @ R + c).astype(np.float32)
            h.visible = True
        if dyn_h is not None:
            sl = dyn["slice_of"].get(int(vf))
            if sl is None or sl[0] == sl[1] or not g_dyn.value:
                dyn_h.visible = False
            else:
                P = dyn["points"][sl[0]:sl[1]]
                C = dyn["colors"][sl[0]:sl[1]]
                if len(P) > args.dynamic_max_points:
                    sel = np.random.default_rng(vf).choice(
                        len(P), args.dynamic_max_points, replace=False)
                    P, C = P[sel], C[sel]
                dyn_h.points = P
                dyn_h.colors = C
                dyn_h.visible = True

    def on_scrub(_):
        if not g_play.value:
            show(g_frame.value)

    g_frame.on_update(on_scrub)
    print(f"viser on :{args.port}", flush=True)
    while True:
        if g_play.value:
            nxt = g_frame.value + 1
            g_frame.value = s0 if nxt >= e0 else nxt
            show(g_frame.value)
        time.sleep(1.0 / args.fps)


if __name__ == "__main__":
    main()
