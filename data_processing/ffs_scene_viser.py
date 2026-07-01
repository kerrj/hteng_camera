"""Serve a fused scene point cloud (ffs_scene_depth output) in viser.

    python data_processing/ffs_scene_viser.py --ply data_processing/out/scene3000_cloud.ply

Opens a web viewer on :8080 (same port as examples/viser_control.py). The cloud
is in the LEFT-camera metric frame: +x right, +y down, +z forward, origin = left
camera (shown as an axis triad). GUI sliders control point size and a max-range
clip (handy for hiding baseline-limited far-field noise).
"""
import argparse
import time

import numpy as np
import open3d as o3d
import viser


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", default="data_processing/out/scene3000_cloud.ply")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--point-size", type=float, default=0.004)
    ap.add_argument("--max-points", type=int, default=2_000_000)
    args = ap.parse_args()

    pc = o3d.io.read_point_cloud(args.ply)
    P = np.asarray(pc.points).astype(np.float32)
    C = (np.clip(np.asarray(pc.colors), 0, 1) * 255).astype(np.uint8)   # RGB
    if len(P) > args.max_points:
        idx = np.random.choice(len(P), args.max_points, replace=False)
        P, C = P[idx], C[idx]
    rng = np.linalg.norm(P, axis=1)
    print(f"{len(P):,} pts  range p50={np.median(rng):.2f}m max={rng.max():.2f}m")

    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("-y")                 # world +y is down
    server.scene.add_frame("/left_cam", axes_length=0.15, axes_radius=0.005)

    ps = server.gui.add_slider("point size", 0.001, 0.02, 0.001, args.point_size)
    mr = server.gui.add_slider("max range (m)", 0.2, float(np.ceil(rng.max())),
                               0.1, float(np.ceil(rng.max())))

    def redraw():
        m = rng <= mr.value
        server.scene.add_point_cloud("/scene", points=P[m], colors=C[m],
                                     point_size=ps.value, point_shape="rounded")

    ps.on_update(lambda _: redraw())
    mr.on_update(lambda _: redraw())
    redraw()

    print(f"viser serving on port {args.port} — open http://<sphynx-host>:{args.port} "
          f"(or ssh -L {args.port}:localhost:{args.port} sphynx)")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
