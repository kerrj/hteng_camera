"""Build the mano_jax .npz bundle from a (de-chumpy'd) MANO_RIGHT.pkl.

mano_jax.load_mano expects an npz with: v_template, shapedirs, posedirs, weights,
J_regressor, faces, parents, fingertips, joint_map. All come from the standard
MANO pkl except `fingertips` (5 tip vertex ids) and `joint_map` (MANO16+5tips ->
OpenPose21). jkerr's original bundle lived in /tmp on chungus; regenerate here.

  python data_processing/extract_mano.py \
      --pkl /home/jkerr/WiLoR-mini/wilor_mini/pretrained_models/MANO_RIGHT.pkl \
      --out data_processing/out/mano_jax.npz
"""
import argparse
import pickle

import numpy as np
import scipy.sparse

# MANO right-hand fingertip vertex ids (thumb,index,middle,ring,pinky) + the
# MANO16+tips -> OpenPose21 remap. Only joint_map[0]=0 matters for the mesh.
FINGERTIPS = np.array([744, 320, 443, 554, 671], np.int64)
JOINT_MAP = np.array([0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18,
                      10, 11, 12, 19, 7, 8, 9, 20], np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default="/home/jkerr/WiLoR-mini/wilor_mini/pretrained_models/MANO_RIGHT.pkl")
    ap.add_argument("--out", default="data_processing/out/mano_jax.npz")
    args = ap.parse_args()

    d = pickle.load(open(args.pkl, "rb"), encoding="latin1")
    v_template = np.asarray(d["v_template"], np.float32).reshape(778, 3)
    shapedirs = np.asarray(d["shapedirs"], np.float32)[:, :, :10].reshape(778, 3, 10)
    posedirs = np.asarray(d["posedirs"], np.float32).reshape(778, 3, -1)         # (778,3,135)
    weights = np.asarray(d["weights"], np.float32).reshape(778, 16)
    Jr = d["J_regressor"]
    Jr = Jr.toarray() if scipy.sparse.issparse(Jr) else np.asarray(Jr)
    J_regressor = np.asarray(Jr, np.float32).reshape(16, 778)
    faces = np.asarray(d["f"], np.int64)
    parents = np.asarray(d["kintree_table"], np.int64)[0].copy()
    parents[0] = -1

    print("posedirs", posedirs.shape, "shapedirs", shapedirs.shape, "faces", faces.shape)
    np.savez(args.out, v_template=v_template, shapedirs=shapedirs, posedirs=posedirs,
             weights=weights, J_regressor=J_regressor, faces=faces, parents=parents,
             fingertips=FINGERTIPS, joint_map=JOINT_MAP)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
