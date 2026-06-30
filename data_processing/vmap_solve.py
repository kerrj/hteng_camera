"""Per-frame vmapped Levenberg-Marquardt solve for the no-temporal stereo fit.

With temporal smoothing off, every frame is an independent 51-DOF problem
(16*3 pose tangent + 3 translation) with ~129 residuals (21 kp x 2 eyes x 2 +
15*3 shape prior). Batching them into ONE jaxls least-squares system makes LM
share a single trust-region lambda across all frames, so one badly-conditioned
frame stalls the whole batch (the 5-80s nondeterministic CG blowup we profiled).

Instead: solve each frame on its own, jax.vmap'd. Each system is tiny and dense,
gets its OWN lambda, and there's no cross-frame coupling. This is the fast path
for w_temporal == 0.

State per frame: (quat (16,4), trans (3,)). LM step works in the 51-dim tangent
(pose via SO3.exp retraction, trans additive).
"""
import jax
import jax.numpy as jnp
import jaxlie

import mano_jax as MJ


def make_residual_fn(M, mirror, norm, huber_n, w_shape):
    """Return r(quat, trans, frame_data) -> (R,) residual for ONE frame."""
    def project(cam, f_px):
        c = (norm - 1) / 2.0
        return jnp.stack([f_px * cam[:, 0] / cam[:, 2] + c,
                          f_px * cam[:, 1] / cam[:, 2] + c], axis=-1)

    def huber_sqrtw(res_px):
        a = jnp.abs(res_px) + 1e-8
        w = jnp.where(a > huber_n, huber_n / a, 1.0)
        return jnp.sqrt(jax.lax.stop_gradient(w))

    def residual(quat, trans, fd):
        R = jaxlie.SO3(quat).as_matrix()                     # (16,3,3)
        joints = MJ.mano_forward_R(M, R, fd["beta"])
        joints = joints.at[:, 0].multiply(mirror)
        x = joints + trans[None, :]                          # left-virtual (21,3)
        # left eye
        rl = (project(x, fd["f_px"]) - fd["kpL"]) / norm
        rl = rl * fd["conf"][:, None] * huber_sqrtw(rl)
        # right eye
        camr = x @ fd["R_lr"].T + fd["t_lr"][None, :]
        rr = (project(camr, fd["f_px"]) - fd["kpR"]) / norm
        rr = rr * fd["conf"][:, None] * huber_sqrtw(rr)
        # shape prior (joints 1..15, geodesic to two-eye mean)
        rs = (jaxlie.SO3(fd["shape_mean"]).inverse()
              @ jaxlie.SO3(quat[1:])).log()                  # (15,3)
        rs = w_shape * rs
        return jnp.concatenate([rl.ravel(), rr.ravel(), rs.ravel()])
    return residual


def make_solver(residual, n_iters=15, lam0=1e-2, lam_up=3.0, lam_down=0.5):
    """Vmappable LM solver for one frame. retract: pose via SO3.exp, trans add."""
    def retract(quat, trans, delta):
        dq = delta[:48].reshape(16, 3)
        quat_new = (jaxlie.SO3(quat) @ jaxlie.SO3.exp(dq)).wxyz
        return quat_new, trans + delta[48:]

    def solve_one(quat0, trans0, fd):
        def r_of_delta(delta, quat, trans):
            q, t = retract(quat, trans, delta)
            return residual(q, t, fd)

        def body(carry, _):
            quat, trans, lam = carry
            z = jnp.zeros(51)
            r = r_of_delta(z, quat, trans)                   # (R,)
            J = jax.jacfwd(r_of_delta)(z, quat, trans)       # (R,51)
            JtJ = J.T @ J
            Jtr = J.T @ r
            A = JtJ + lam * jnp.diag(jnp.diag(JtJ) + 1e-9)   # LM damping
            delta = -jnp.linalg.solve(A, Jtr)
            q_new, t_new = retract(quat, trans, delta)
            cost_old = r @ r
            cost_new = (lambda rr: rr @ rr)(residual(q_new, t_new, fd))
            improved = cost_new < cost_old
            quat = jnp.where(improved, q_new, quat)
            trans = jnp.where(improved, t_new, trans)
            lam = jnp.where(improved, lam * lam_down, lam * lam_up)
            lam = jnp.clip(lam, 1e-8, 1e6)
            return (quat, trans, lam), None

        (quat, trans, _), _ = jax.lax.scan(
            body, (quat0, trans0, jnp.float32(lam0)), None, length=n_iters)
        return quat, trans

    return jax.jit(jax.vmap(solve_one))
