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
import functools
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
    ap.add_argument("--video", help="8-bit stereo video; show the LEFT-eye frame")
    ap.add_argument("--thumb-w", type=int, default=256,
                    help="downscaled width for the left-eye frame (decode is heavy)")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()

    # left-eye frame loader (decode on demand, cached). cv2 + torchcodec are in
    # eyeball211; use torchcodec to match the rest of the pipeline.
    left_frame = None
    if args.video:
        import threading
        import cv2
        from torchcodec.decoders import VideoDecoder
        _dec = VideoDecoder(args.video, device="cpu")
        _half = _dec.metadata.width // 2
        _dec_lock = threading.Lock()   # torchcodec decoder is NOT thread-safe:
        # the play loop and the scrub callback run on different threads, and a
        # backward seek racing a forward decode segfaults. Serialize all access.

        @functools.lru_cache(maxsize=512)
        def _decode(fr):
            with _dec_lock:
                f = _dec.get_frames_in_range(start=fr, stop=fr + 1).data[0]  # (3,H,W)
            img = f[:, :, :_half].permute(1, 2, 0).numpy().astype(np.uint8)
            h, w = img.shape[:2]
            tw = args.thumb_w
            return cv2.resize(img, (tw, int(h * tw / w)),
                              interpolation=cv2.INTER_AREA)

        def left_frame(fr):
            return _decode(int(fr))

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
    server.scene.world_axes.visible = False

    # camera frustum at origin (left-fisheye cam: +z forward, +y down in cv2).
    # viser is +z up by convention; we just draw a marker frame + a small frustum.
    server.scene.add_frame("/camera", axes_length=0.05, axes_radius=0.002)
    # PERSISTENT scene handles, created ONCE — we update them in place each frame
    # rather than re-adding (re-adding rebuilds + re-uploads the whole object,
    # which made scrubbing very slow).
    aspect0 = 1.0
    if left_frame is not None:
        aspect0 = left_frame(fmin).shape[1] / left_frame(fmin).shape[0]
    frustum_h = server.scene.add_camera_frustum(
        "/camera/frustum", fov=1.4, aspect=aspect0, scale=0.08,
        color=(0.7, 0.7, 0.7), format="jpeg",
        image=left_frame(fmin) if left_frame else None)
    # create with the FULL faces + correctly-sized dummy verts (778); only
    # .vertices is updated per frame, so faces must be the real topology here.
    n_v = int(np.asarray(M["v_template"]).shape[0])                # 778
    mesh_h = {name: server.scene.add_mesh_simple(
                  f"/hand_{name}", np.zeros((n_v, 3), np.float32), faces,
                  color=color, flat_shading=False, visible=False)
              for name, (_, _, color) in tracks.items()}

    # --- GUI -----------------------------------------------------------------
    gui_play = server.gui.add_checkbox("play", False)
    gui_frame = server.gui.add_slider("frame", fmin, fmax, 1, fmin)
    gui_beta = server.gui.add_dropdown("shape betas", ("optimized", "WiLoR-mean"),
                                       "optimized")
    gui_info = server.gui.add_text("info", "", disabled=True)
    gui_img = server.gui.add_image(left_frame(fmin), label="left eye",
                                   format="jpeg") if left_frame else None

    def beta_for(meta):
        return (meta["beta_opt"] if gui_beta.value == "optimized"
                else meta["beta_wilor_mean"])

    # the play loop and scrub/dropdown callbacks fire on different threads;
    # serialize the whole render so two updates can't interleave on the shared
    # scene handles / decoder.
    import threading
    _update_lock = threading.Lock()

    def update(_=None):
        with _update_lock:
            fr = int(gui_frame.value)
            # decode OUTSIDE atomic() (it's slow / holds the decoder lock); only
            # the scene/gui mutations need to be batched.
            img = left_frame(fr) if left_frame is not None else None
            verts = {}
            lines = [f"frame {fr}"]
            for name, (meta, frames, color) in tracks.items():
                h = frames.get(fr)
                if h is None:
                    continue
                verts[name] = mesh_verts(
                    M, faces, h["quat"], h["trans_virtual"], h["Rv_l"],
                    beta_for(meta), meta["mirror"]).astype(np.float32)
                dist = float(np.linalg.norm(h["trans"]))   # from camera origin
                lines.append(f"  {name}: dist {dist:.2f}m")
            # apply everything in ONE atomic batch so meshes + image move together
            with server.atomic():
                for name in tracks:
                    if name in verts:
                        mesh_h[name].vertices = verts[name]
                        mesh_h[name].visible = True
                    else:
                        mesh_h[name].visible = False
                gui_info.value = "\n".join(lines)
                if img is not None:
                    frustum_h.image = img
                    gui_img.image = img

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
