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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--mano", default="/tmp/mano_jax.npz")
    ap.add_argument("--calib-dir", default="../../long-test1")
    ap.add_argument("--left-serial", default="046060323008")
    ap.add_argument("--right-serial", default="046060323001")
    ap.add_argument("--hand", default="right")
    ap.add_argument("--iters", type=int, default=15)
    ap.add_argument("--w-shape", type=float, default=0.1)
    ap.add_argument("--huber-px", type=float, default=5.0)
    ap.add_argument("--chunks", default="100,200,300,500")
    ap.add_argument("--linear", default="conjugate_gradient")
    ap.add_argument("--schur", default="auto", choices=["auto", "off"],
                    help="Schur elimination in analyze(). 'auto' makes the fully "
                         "block-diagonal (no-temporal) system an exact per-frame "
                         "blockwise inverse.")
    args = ap.parse_args()

    linear = args.linear

    Mh = MJ.load_mano(args.mano)
    data, t_init, frames, _, _ = SO.build_data(
        args.jsonl, args.calib_dir, args.left_serial, args.right_serial, args.hand)
    n = len(frames)
    print(f"loaded {n} frames, hand={args.hand}, linear={args.linear}, "
          f"schur={args.schur}, iters={args.iters}")

    def one_solve(s, e):
        cn = e - s
        cfids = jnp.arange(cn)
        cdata = {k: (v[s:e] if hasattr(v, "shape") and getattr(v, "ndim", 0) >= 1
                     else v) for k, v in data.items()}
        t0 = time.time()
        ccosts, _ = SO.make_costs(Mh, cdata, args.huber_px, args.w_shape, 0.0, 0.0, 0.0, 0.5)
        t1 = time.time()
        cinit = jaxls.VarValues.make([
            SO.PoseVar(cfids).with_value(cdata["quat0"]),
            SO.TransVar(cfids).with_value(t_init[s:e]),
        ])
        prob = jaxls.LeastSquaresProblem(
            ccosts, [SO.PoseVar(cfids), SO.TransVar(cfids)]
        ).analyze(schur_elimination=args.schur)
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

    # all-in-one-call: solve ALL n frames in a single jaxls problem. With Schur
    # 'auto' on the block-diagonal (no-temporal) system this is an exact per-frame
    # blockwise inverse, so one call should scale fine and avoids per-chunk
    # recompiles. (Server load makes wall times noisy — take the min.)
    print(f"\n===== ALL {n} frames in ONE call =====")
    b, a, s = one_solve(0, n)
    print(f"  build={b:7.1f}ms  analyze={a:7.1f}ms  solve={s:8.1f}ms "
          f"({s/n*1e3:.2f}ms/frame)")


if __name__ == "__main__":
    main()
