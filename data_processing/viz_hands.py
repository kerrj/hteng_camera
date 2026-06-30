"""viser 3D visualization of stereo hand-pose results (WIP data viewer).

Shows, in the LEFT-FISHEYE camera frame (the world for now — we assume the head
doesn't move, so the camera is fixed at the origin):
  - the camera frustum at the origin,
  - the MANO hand MESH(es) for the current frame,
  - a frame scrubber + play/pause,
  - a toggle: shape betas = optimized (shared, solved) vs WiLoR-mean (default).

Long-term this will also show head pose; for now the camera is static.

Mesh re-posing (per hand, per frame), from the optimizer's saved params:
  R_i      = SO3(quat_i)                  # (16,3,3) per-joint
  verts_v  = mano_mesh_R(R_i, beta)       # left-virtual frame, root-relative
  verts_v[:,0] *= mirror                  # left-hand x-mirror (MANO_RIGHT)
  verts_v += trans_virtual_i              # place root
  verts_f  = verts_v @ Rv_l_i.T           # → left-fisheye (world) frame

Run (in eyeball211, has viser+smplx):
  python viz_hands.py --right out/smooth_right.jsonl --left out/smooth_left.jsonl \
      --mano /tmp/mano_jax.npz [--video-thumbs out/thumbs]
Then forward the printed port to your laptop.
"""
import argparse
import json

import numpy as np
import jax.numpy as jnp
import jaxlie

import mano_jax as MJ


def load_track(path):
    """Read a stereo3d jsonl → (meta dict, {frame: hand dict})."""
    meta, frames = None, {}
    for line in open(path):
        d = json.loads(line)
        if d.get("meta"):
            meta = d
            continue
        frames[d["frame"]] = d
    return meta, frames


def mesh_verts(M, faces, quat, trans_virtual, Rv_l, beta, mirror):
    """Re-pose one hand's MANO mesh into the LEFT-FISHEYE frame. Returns (778,3)."""
    R = jaxlie.SO3(jnp.asarray(np.array(quat))).as_matrix()       # (16,3,3)
    v = np.array(MJ.mano_mesh_R(M, R, jnp.asarray(beta)))         # (778,3) virtual
    v[:, 0] *= mirror                                             # left-hand x-mirror
    v = v + np.array(trans_virtual)[None, :]                      # place root
    v = v @ np.array(Rv_l).T                                      # → fisheye world
    return v


def main():
    import viser

    ap = argparse.ArgumentParser()
    ap.add_argument("--right", help="right-hand stereo3d jsonl")
    ap.add_argument("--left", help="left-hand stereo3d jsonl")
    ap.add_argument("--mano", default="/tmp/mano_jax.npz")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()

    M = MJ.load_mano(args.mano)
    faces = np.asarray(M["faces"])

    tracks = {}   # name -> (meta, frames, color)
    if args.right:
        tracks["right"] = (*load_track(args.right), (0.3, 0.6, 1.0))
    if args.left:
        tracks["left"] = (*load_track(args.left), (1.0, 0.5, 0.3))
    assert tracks, "need at least one of --right / --left"

    all_frames = sorted(set().union(*[set(fr) for _, fr, _ in tracks.values()]))
    fmin, fmax = all_frames[0], all_frames[-1]

    server = viser.ViserServer(port=args.port)
    server.scene.world_axes.visible = True

    # camera frustum at origin (left-fisheye cam: +z forward, +y down in cv2).
    # viser is +z up by convention; we just draw a marker frame + a small frustum.
    server.scene.add_frame("/camera", axes_length=0.05, axes_radius=0.002)
    server.scene.add_camera_frustum(
        "/camera/frustum", fov=1.4, aspect=1.0, scale=0.06, color=(0.7, 0.7, 0.7))

    # --- GUI -----------------------------------------------------------------
    gui_play = server.gui.add_checkbox("play", False)
    gui_frame = server.gui.add_slider("frame", fmin, fmax, 1, fmin)
    gui_beta = server.gui.add_dropdown("shape betas", ("optimized", "WiLoR-mean"),
                                       "optimized")
    gui_info = server.gui.add_text("info", "", disabled=True)

    def beta_for(meta):
        return (meta["beta_opt"] if gui_beta.value == "optimized"
                else meta["beta_wilor_mean"])

    def update(_=None):
        fr = int(gui_frame.value)
        lines = [f"frame {fr}"]
        for name, (meta, frames, color) in tracks.items():
            h = frames.get(fr)
            handle = f"/hand_{name}"
            if h is None:
                server.scene.add_mesh_simple(handle, np.zeros((3, 3)), faces[:1],
                                             visible=False)
                continue
            v = mesh_verts(M, faces, h["quat"], h["trans_virtual"], h["Rv_l"],
                           beta_for(meta), meta["mirror"])
            server.scene.add_mesh_simple(handle, v, faces, color=color,
                                         flat_shading=False, visible=True)
            lines.append(f"  {name}: depth {h['depth_m']:.2f}m")
        gui_info.value = "\n".join(lines)

    gui_frame.on_update(update)
    gui_beta.on_update(update)
    update()

    print(f"viser running on port {args.port} — forward it: "
          f"ssh -L {args.port}:localhost:{args.port} <host>")

    import time
    while True:
        if gui_play.value:
            nxt = int(gui_frame.value) + 1
            gui_frame.value = fmin if nxt > fmax else nxt
        time.sleep(1.0 / 30.0)


if __name__ == "__main__":
    main()
