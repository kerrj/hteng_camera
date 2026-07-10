"""Precompute per-frame MANO hand meshes (778 verts, LEFT-FISHEYE frame) -> npz.

Re-poses jkerr's viz_*.jsonl (quat / trans_virtual / Rv_l + meta beta/mirror)
through the full-mesh MANO forward, so the viser player can render the real hand
surface without importing jax at runtime. One npz per hand:
    frames (N,), verts (N,778,3) float32, faces (F,3) int32.

  python data_processing/precompute_hand_meshes.py \
      --viz-left  data_processing/out/viz_left.jsonl \
      --viz-right data_processing/out/viz_right.jsonl \
      --mano data_processing/out/mano_jax.npz --out-dir data_processing/out
"""
import argparse
import json
import os
import sys

import numpy as np
import jax.numpy as jnp
import jaxlie

# mano_jax lives in data_processing/hands/ after the origin/main reorg
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "hands"))
import mano_jax as MJ


def load_track(path):
    meta, frames = None, {}
    for line in open(path):
        d = json.loads(line)
        (meta := d) if d.get("meta") else frames.__setitem__(d["frame"], d)
    return meta, frames


def build(M, faces, viz_path, out_path):
    meta, frames = load_track(viz_path)
    beta = jnp.asarray(meta["beta_opt"]); mirror = meta["mirror"]
    frs = sorted(frames)
    V = np.zeros((len(frs), 778, 3), np.float32)
    for i, f in enumerate(frs):
        h = frames[f]
        R = jaxlie.SO3(jnp.asarray(np.array(h["quat"]))).as_matrix()      # (16,3,3)
        v = np.array(MJ.mano_mesh_R(M, R, beta))                          # (778,3) virtual
        v[:, 0] *= mirror                                                 # left-hand x-mirror
        v = v + np.array(h["trans_virtual"])[None, :]                     # place root
        v = v @ np.array(h["Rv_l"]).T                                     # -> left-fisheye
        V[i] = v
    np.savez(out_path, frames=np.array(frs, np.int64), verts=V, faces=faces)
    print(f"{out_path}: {len(frs)} frames")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--viz-left", default="data_processing/out/viz_left.jsonl")
    ap.add_argument("--viz-right", default="data_processing/out/viz_right.jsonl")
    ap.add_argument("--mano", default="data_processing/out/mano_jax.npz")
    ap.add_argument("--out-dir", default="data_processing/out")
    args = ap.parse_args()
    M = MJ.load_mano(args.mano)
    faces = np.asarray(M["faces"], np.int32)
    build(M, faces, args.viz_left, f"{args.out_dir}/hand_mesh_left.npz")
    build(M, faces, args.viz_right, f"{args.out_dir}/hand_mesh_right.npz")


if __name__ == "__main__":
    main()
