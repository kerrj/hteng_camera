"""Minimal JAX MANO forward (LBS) → 21 OpenPose-ordered joints.

Differentiable through pose (axis-angle) + shape (betas) so jaxls can optimize
MANO params to fit stereo keypoints. Replicates wilor_mini's MANO wrapper:
16 MANO joints (J_regressor) + 5 fingertip vertices, remapped to the 21-joint
OpenPose order. Validated against the torch smplx output (see validate_mano_jax).

Load the tensors bundle with ``load_mano(npz_path)`` (produced by extract_mano).
"""
import jax
import jax.numpy as jnp
import numpy as np


def load_mano(npz_path):
    z = np.load(npz_path)
    return {k: jnp.asarray(z[k]) for k in z.files}


def axis_angle_to_matrix(aa):
    """(..., 3) axis-angle → (..., 3, 3) rotation matrix (Rodrigues)."""
    theta = jnp.linalg.norm(aa, axis=-1, keepdims=True)
    k = aa / jnp.clip(theta, 1e-8)
    kx, ky, kz = k[..., 0], k[..., 1], k[..., 2]
    zeros = jnp.zeros_like(kx)
    K = jnp.stack([zeros, -kz, ky, kz, zeros, -kx, -ky, kx, zeros], axis=-1)
    K = K.reshape(aa.shape[:-1] + (3, 3))
    s = jnp.sin(theta)[..., None]
    c = jnp.cos(theta)[..., None]
    eye = jnp.eye(3)
    return eye + s * K + (1 - c) * (K @ K)


def mano_forward(M, global_orient, hand_pose, betas):
    """MANO LBS forward for ONE hand.

    Args:
        M: dict from load_mano.
        global_orient: (3,) axis-angle, wrist rotation.
        hand_pose: (45,) axis-angle, 15 finger joints.
        betas: (10,) shape coeffs.

    Returns:
        joints21: (21, 3) joints in MANO root frame (root-relative, metres),
            OpenPose order, matching wilor_mini's pred_keypoints_3d convention.
    """
    # 1) shape-blended template + per-shape joint locations
    v_shaped = M["v_template"] + jnp.einsum("vck,k->vc", M["shapedirs"], betas)  # (778,3)
    J = M["J_regressor"] @ v_shaped                                              # (16,3)

    # 2) pose: 16 joint rotations (wrist + 15). pose feature = (R - I) for the
    #    15 non-root joints, flattened (135,), drives posedirs.
    full_pose = jnp.concatenate([global_orient, hand_pose], axis=0).reshape(16, 3)
    R = axis_angle_to_matrix(full_pose)                                         # (16,3,3)
    pose_feat = (R[1:] - jnp.eye(3)).reshape(-1)                                # (135,)
    v_posed = v_shaped + jnp.einsum("vck,k->vc", M["posedirs"], pose_feat)      # (778,3)

    # 3) global rigid transforms per joint along the kinematic tree.
    #    parents indexes a Python list during graph construction, so it must be
    #    static ints (not traced jnp values) — pull to a host list once.
    parents = [int(p) for p in np.asarray(M["parents"])]
    G = []
    # root
    G0 = jnp.eye(4).at[:3, :3].set(R[0]).at[:3, 3].set(J[0])
    G.append(G0)
    for i in range(1, 16):
        local = jnp.eye(4).at[:3, :3].set(R[i]).at[:3, 3].set(J[i] - J[parents[i]])
        G.append(G[parents[i]] @ local)
    G = jnp.stack(G, axis=0)                                                    # (16,4,4)

    # 4) remove the rest-pose offset so identity pose → no transform:
    #    G_rel_i = G_i - [0 | G_i @ [J_i; 0]] (subtract from translation column).
    J0 = jnp.concatenate([J, jnp.zeros((16, 1))], axis=1)                       # (16,4)
    offset = jnp.einsum("iab,ib->ia", G, J0)[:, :3]                            # (16,3)
    G_rel = G.at[:, :3, 3].add(-offset)

    # 5) skin vertices
    T = jnp.einsum("vj,jab->vab", M["weights"], G_rel)                          # (778,4,4)
    v_h = jnp.concatenate([v_posed, jnp.ones((778, 1))], axis=1)                # (778,4)
    v_skinned = jnp.einsum("vab,vb->va", T, v_h)[:, :3]                         # (778,3)

    # 6) joints: the 16 MANO joints are the rest-pose joints J carried by the
    #    LBS transforms G (NOT re-regressed from posed verts); fingertips are
    #    posed vertices. Remap to OpenPose 21.
    J_h = jnp.concatenate([J, jnp.ones((16, 1))], axis=1)                       # (16,4)
    j16 = jnp.einsum("iab,ib->ia", G_rel, J_h)[:, :3]                          # (16,3)
    tips = v_skinned[M["fingertips"]]                                          # (5,3)
    j21 = jnp.concatenate([j16, tips], axis=0)[M["joint_map"]]                  # (21,3)
    # root-relative (wilor pred_keypoints_3d are centred on the wrist joint 0)
    return j21 - j21[0]
