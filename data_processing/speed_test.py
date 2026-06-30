"""Profile the stereo_optimize solve to find where time goes.

For each chunk size, times three solves on DIFFERENT chunks of the SAME shape:
  - solve#1 includes the JAX compile.
  - solve#2/#3 should be fast IF jaxls reuses the compiled solve. If #2/#3 are
    ~as slow as #1, we're recompiling every chunk (the suspected bug).
Also separates build (make_costs) and analyze() from the solve itself.

Run:  python speed_test.py --jsonl out/pinhole_verged_full/hands.jsonl --hand right
"""
import argparse
import time

import jax
import jax.numpy as jnp

import jaxls
import mano_jax as MJ
import stereo_optimize as SO
import vmap_solve as VS


def run_vmap(Mh, data, t_init, n, args):
    """Benchmark + validate the per-frame vmapped LM solver."""
    norm = float(data["out_size"])
    huber_n = args.huber_px / norm
    mirror = float(data["mirror"])
    residual = VS.make_residual_fn(Mh, mirror, norm, huber_n, args.w_shape)
    solver = VS.make_solver(residual, n_iters=args.iters)

    # stack per-frame data dicts
    fd = {
        "beta": data["beta0"], "kpL": data["kpL"], "kpR": data["kpR"],
        "conf": data["confL"], "f_px": data["f_px"],
        "R_lr": data["R_lr"], "t_lr": data["t_lr"], "shape_mean": data["shape_mean"],
    }
    q0 = data["quat0"]; t0 = t_init

    t = time.time()
    quat, trans = solver(q0, t0, fd)
    jax.block_until_ready(quat)
    t_compile = time.time() - t
    # warm runs
    times = []
    for _ in range(3):
        t = time.time()
        quat, trans = solver(q0, t0, fd)
        jax.block_until_ready(quat)
        times.append(time.time() - t)
    t_warm = min(times)

    # reprojection error (inlier-style: all kp) to validate the fit
    rs = jax.vmap(lambda q, tr, d: residual(q, tr, d))(quat, trans, fd)
    # recover px error: residual is normalized & huber-weighted; recompute raw
    import numpy as np
    quat = np.array(quat); trans = np.array(trans)
    print(f"\nVMAP solver: {n} frames, iters={args.iters}")
    print(f"  compile+1st run: {t_compile:.2f}s")
    print(f"  warm full-batch: {t_warm*1000:.1f}ms  ({t_warm/n*1e6:.1f} us/frame)")
    print(f"  -> {n} frames in {t_warm:.3f}s warm "
          f"(vs jaxls ~5-80s per 500-chunk)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--mano", default="/tmp/mano_jax.npz")
    ap.add_argument("--calib-dir", default="../long-test1")
    ap.add_argument("--left-serial", default="046060323008")
    ap.add_argument("--right-serial", default="046060323001")
    ap.add_argument("--hand", default="right")
    ap.add_argument("--iters", type=int, default=15)
    ap.add_argument("--w-shape", type=float, default=0.1)
    ap.add_argument("--huber-px", type=float, default=5.0)
    ap.add_argument("--chunks", default="100,200,300,500")
    ap.add_argument("--linear", default="conjugate_gradient")
    ap.add_argument("--cg-tol", type=float, default=None,
                    help="if set, FIXED CG tolerance (min=max) — disables "
                         "Eisenstat-Walker adaptive tightening")
    ap.add_argument("--vmap", action="store_true",
                    help="benchmark the per-frame vmapped LM solver instead")
    args = ap.parse_args()

    linear = args.linear
    if args.cg_tol is not None:
        linear = jaxls.ConjugateGradientConfig(
            tolerance_min=args.cg_tol, tolerance_max=args.cg_tol)

    Mh = MJ.load_mano(args.mano)
    data, t_init, frames, _, _ = SO.build_data(
        args.jsonl, args.calib_dir, args.left_serial, args.right_serial, args.hand)
    n = len(frames)
    print(f"loaded {n} frames, hand={args.hand}, linear={args.linear}, iters={args.iters}")

    if args.vmap:
        run_vmap(Mh, data, t_init, n, args)
        return

    def one_solve(s, e):
        cn = e - s
        cfids = jnp.arange(cn)
        cdata = {k: (v[s:e] if hasattr(v, "shape") and getattr(v, "ndim", 0) >= 1
                     else v) for k, v in data.items()}
        t0 = time.time()
        ccosts, _ = SO.make_costs(Mh, cdata, 0.0, args.huber_px, args.w_shape)
        t1 = time.time()
        cinit = jaxls.VarValues.make([
            SO.PoseVar(cfids).with_value(cdata["quat0"]),
            SO.TransVar(cfids).with_value(t_init[s:e]),
        ])
        prob = jaxls.LeastSquaresProblem(
            ccosts, [SO.PoseVar(cfids), SO.TransVar(cfids)]).analyze()
        t2 = time.time()
        sol = prob.solve(cinit, trust_region=jaxls.TrustRegionConfig(),
                         linear_solver=linear,
                         termination=jaxls.TerminationConfig(max_iterations=args.iters),
                         verbose=False)
        jax.block_until_ready(sol[SO.PoseVar])
        t3 = time.time()
        return (t1 - t0) * 1e3, (t2 - t1) * 1e3, (t3 - t2) * 1e3

    for chunk in [int(c) for c in args.chunks.split(",")]:
        if chunk > n:
            continue
        print(f"\n===== chunk={chunk} =====")
        for lab, o in zip(["solve#1(compile)", "solve#2", "solve#3"],
                          [0, chunk, 2 * chunk]):
            if o + chunk > n:
                break
            b, a, s = one_solve(o, o + chunk)
            print(f"  {lab:18s} build={b:7.1f}ms  analyze={a:7.1f}ms  solve={s:8.1f}ms")


if __name__ == "__main__":
    main()
