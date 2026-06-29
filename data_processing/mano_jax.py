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
    M = {k: jnp.asarray(z[k]) for k in z.files}
    # Precompute fingertip-only slices so the fast forward skins just the 5
    # fingertip vertices instead of all 778 (we never use the other 773).
    tips = np.asarray(z["fingertips"])                       # (5,)
    M["tip_v_template"] = jnp.asarray(z["v_template"][tips])         # (5,3)
    M["tip_shapedirs"] = jnp.asarray(z["shapedirs"][tips])          # (5,3,10)
    M["tip_posedirs"] = jnp.asarray(z["posedirs"][tips])           # (5,3,135)
    M["tip_weights"] = jnp.asarray(z["weights"][tips])            # (5,16)
    # Rest-pose joints as a tiny affine in betas: J = J_tmpl + J_shapedirs·beta,
    # so we never form all 778 shaped vertices just to regress 16 joints.
    Jr = np.asarray(z["J_regressor"])                                # (16,778)
    M["J_tmpl"] = jnp.asarray(Jr @ np.asarray(z["v_template"]))      # (16,3)
    M["J_shapedirs"] = jnp.asarray(
        np.einsum("jv,vck->jck", Jr, np.asarray(z["shapedirs"])))    # (16,3,10)
    return M


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
    """MANO LBS forward for ONE hand, from axis-angle pose.

    Args:
        M: dict from load_mano.
        global_orient: (3,) axis-angle, wrist rotation.
        hand_pose: (45,) axis-angle, 15 finger joints.
        betas: (10,) shape coeffs.

    Returns:
        joints21: (21, 3) joints in MANO root frame (root-relative, metres),
            OpenPose order, matching wilor_mini's pred_keypoints_3d convention.
    """
    full_pose = jnp.concatenate([global_orient, hand_pose], axis=0).reshape(16, 3)
    R = axis_angle_to_matrix(full_pose)                                         # (16,3,3)
    return mano_forward_R(M, R, betas)


def mano_forward_R(M, R, betas):
    """MANO LBS forward for ONE hand, from rotation matrices.

    Representation-agnostic core: takes the 16 per-joint rotation matrices
    directly (wrist + 15 fingers), so the optimizer can carry pose as SO(3)
    manifold variables (quaternions) instead of axis-angle and convert to R
    here via jaxlie.

    Args:
        M: dict from load_mano.
        R: (16, 3, 3) per-joint rotation matrices, joint 0 = wrist/global.
        betas: (10,) shape coeffs.

    Returns:
        joints21: (21, 3) root-relative joints, OpenPose order.
    """
    # 1) rest-pose joints from shape, as a precomputed affine in betas (no
    #    778-vertex blend). Fingertip verts are shape-blended directly in step 5.
    J = M["J_tmpl"] + jnp.einsum("jck,k->jc", M["J_shapedirs"], betas)           # (16,3)

    # 2) pose feature = (R - I) for the 15 non-root joints, flattened (135,),
    #    drives posedirs.
    pose_feat = (R[1:] - jnp.eye(3)).reshape(-1)                                # (135,)

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

    # 5) skin ONLY the 5 fingertip vertices (not all 778 — we discard the rest).
    tip_posed = (M["tip_v_template"]
                 + jnp.einsum("vck,k->vc", M["tip_shapedirs"], betas)
                 + jnp.einsum("vck,k->vc", M["tip_posedirs"], pose_feat))       # (5,3)
    T = jnp.einsum("vj,jab->vab", M["tip_weights"], G_rel)                      # (5,4,4)
    v_h = jnp.concatenate([tip_posed, jnp.ones((5, 1))], axis=1)                # (5,4)
    tips = jnp.einsum("vab,vb->va", T, v_h)[:, :3]                              # (5,3)

    # 6) joints: the 16 MANO joints are the rest-pose joints J carried by the
    #    LBS transforms G (no skinning); fingertips are the skinned verts above.
    #    Remap to OpenPose 21.
    J_h = jnp.concatenate([J, jnp.ones((16, 1))], axis=1)                       # (16,4)
    j16 = jnp.einsum("iab,ib->ia", G_rel, J_h)[:, :3]                          # (16,3)
    j21 = jnp.concatenate([j16, tips], axis=0)[M["joint_map"]]                  # (21,3)
    # root-relative (wilor pred_keypoints_3d are centred on the wrist joint 0)
    return j21 - j21[0]
