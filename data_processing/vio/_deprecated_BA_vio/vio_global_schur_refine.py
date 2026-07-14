"""Global VIO refinement with landmark Schur elimination and matrix-free CG.

This is a companion to vio_global_block_refine.py. It starts from a global
checkpoint, retains robustly triangulated landmarks, and performs a few LM
iterations over every pose and retained point simultaneously. Landmark Schur
elimination reduces each linear system to the pose variables; jaxls then solves
that reduced system with preconditioned conjugate gradient.
"""

import argparse
import time

import jax
import jax.numpy as jnp
import jaxlie
import jaxls
import numpy as np

from vio_bundle_adjust import CamCenterVar, Point3Var
from vio_global_block_refine import (
    angular_metrics,
    bearing_variable_center_point,
    bearing_variable_pose_point,
    center_constant_velocity,
    centers_to_poses,
    constant_velocity_cost,
    depth_variable_center_point,
    depth_variable_pose_point,
    gravity_cost,
    imu_cost,
    load_solver_imu,
    load_solver_observations,
    retriangulate,
    poses_to_centers,
    time_spread_rows,
    write_probe_output,
)


def select_global_rows(obs, alive, obs_per_landmark):
    """Select temporally spread training observations for reliable points."""
    variable_points = np.flatnonzero(alive)
    point_to_local = np.full(len(alive), -1, np.int32)
    point_to_local[variable_points] = np.arange(
        len(variable_points), dtype=np.int32)
    selected = []
    for point in variable_points:
        rows = obs["point_order"][
            obs["point_bounds"][point]:obs["point_bounds"][point + 1]]
        rows = rows[~obs["validation"][rows]]
        selected.append(time_spread_rows(
            rows,
            obs["pose_id"],
            obs["right"],
            obs_per_landmark,
        ))
    rows = (
        np.concatenate(selected)
        if selected else np.zeros(0, np.int64))
    return rows, variable_points, point_to_local


def make_global_problem(poses, points, alive, obs, imu, args):
    rows, variable_points, point_to_local = select_global_rows(
        obs, alive, args.obs_per_landmark)
    n_poses = len(poses)
    n_points = len(variable_points)
    point_vars = Point3Var(id=jnp.arange(n_points))
    row_point_vars = Point3Var(
        id=jnp.asarray(point_to_local[obs["point_id"][rows]]))
    if args.freeze_rotations:
        pose_vars = CamCenterVar(id=jnp.arange(n_poses))
        row_pose_vars = CamCenterVar(
            id=jnp.asarray(obs["pose_id"][rows]))
        costs = [
            jaxls.Cost(
                bearing_variable_center_point,
                (
                    row_pose_vars,
                    row_point_vars,
                    jnp.asarray(poses[obs["pose_id"][rows], :4]),
                    jnp.asarray(obs["ray_cam"][rows]),
                    jnp.asarray(obs["rel"][rows]),
                    jnp.ones(len(rows), jnp.float32),
                    jnp.asarray(args.robust_scale, jnp.float32),
                ),
            ),
            jaxls.Cost(
                depth_variable_center_point,
                (
                    row_pose_vars,
                    row_point_vars,
                    jnp.asarray(poses[obs["pose_id"][rows], :4]),
                    jnp.asarray(obs["ray_cam"][rows]),
                    jnp.asarray(obs["rel"][rows]),
                    jnp.asarray(args.positive_depth_min, jnp.float32),
                    jnp.asarray(args.positive_depth_softness, jnp.float32),
                    jnp.full(
                        len(rows),
                        args.positive_depth_weight,
                        jnp.float32,
                    ),
                ),
            ),
            jaxls.Cost(
                center_constant_velocity,
                (
                    CamCenterVar(id=jnp.arange(n_poses - 2)),
                    CamCenterVar(id=jnp.arange(1, n_poses - 1)),
                    CamCenterVar(id=jnp.arange(2, n_poses)),
                    jnp.asarray(
                        args.constant_velocity_weight,
                        jnp.float32,
                    ),
                ),
            ),
        ]
        pose_values = pose_vars.with_value(jnp.asarray(
            poses_to_centers(poses), jnp.float32))
    else:
        pose_vars = jaxls.SE3Var(id=jnp.arange(n_poses))
        row_pose_vars = jaxls.SE3Var(
            id=jnp.asarray(obs["pose_id"][rows]))
        costs = [
            jaxls.Cost(
                bearing_variable_pose_point,
                (
                    row_pose_vars,
                    row_point_vars,
                    jnp.asarray(obs["ray_cam"][rows]),
                    jnp.asarray(obs["rel"][rows]),
                    jnp.ones(len(rows), jnp.float32),
                    jnp.asarray(args.robust_scale, jnp.float32),
                ),
            ),
            jaxls.Cost(
                depth_variable_pose_point,
                (
                    row_pose_vars,
                    row_point_vars,
                    jnp.asarray(obs["ray_cam"][rows]),
                    jnp.asarray(obs["rel"][rows]),
                    jnp.asarray(args.positive_depth_min, jnp.float32),
                    jnp.asarray(args.positive_depth_softness, jnp.float32),
                    jnp.full(
                        len(rows),
                        args.positive_depth_weight,
                        jnp.float32,
                    ),
                ),
            ),
            jaxls.Cost(
                imu_cost,
                (
                    jaxls.SE3Var(id=jnp.arange(n_poses - 1)),
                    jaxls.SE3Var(id=jnp.arange(1, n_poses)),
                    jnp.asarray(imu["delta_prev"][1:]),
                    jnp.asarray(args.imu_rot_weight, jnp.float32),
                ),
            ),
            jaxls.Cost(
                gravity_cost,
                (
                    jaxls.SE3Var(id=jnp.arange(n_poses)),
                    jnp.asarray(imu["gravity_cam"]),
                    jnp.asarray(
                        args.gravity_weight
                        * imu["gravity_weight"][:, None],
                        jnp.float32,
                    ),
                ),
            ),
            jaxls.Cost(
                constant_velocity_cost,
                (
                    jaxls.SE3Var(id=jnp.arange(n_poses - 2)),
                    jaxls.SE3Var(id=jnp.arange(1, n_poses - 1)),
                    jaxls.SE3Var(id=jnp.arange(2, n_poses)),
                    jnp.asarray(
                        args.constant_velocity_weight,
                        jnp.float32,
                    ),
                ),
            ),
        ]
        pose_values = pose_vars.with_value(jaxlie.SE3(jnp.asarray(
            poses, jnp.float32)))
    problem = jaxls.LeastSquaresProblem(
        costs,
        [pose_vars, point_vars],
    ).analyze(schur_elimination=(Point3Var,))
    values = jaxls.VarValues.make([
        pose_values,
        point_vars.with_value(jnp.asarray(
            points[variable_points], jnp.float32)),
    ])
    stereo_pairs = len(rows) - len(np.unique(np.stack([
        obs["pose_id"][rows],
        obs["point_id"][rows],
    ], axis=1), axis=0))
    print(
        f"global problem: {n_poses} poses, {n_points} points, "
        f"{len(rows)} visual observations, {stereo_pairs} stereo pairs")
    return problem, values, variable_points


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("recording")
    parser.add_argument("--initial-state", required=True)
    parser.add_argument("--tracks", default=None,
                        help="default: recording's derived/tracks.jsonl")
    parser.add_argument("--out", required=True)
    parser.add_argument("--active-tracks", type=int, default=100)
    parser.add_argument("--spatial-grid", type=int, nargs=2, default=(8, 6))
    parser.add_argument("--active-temporal-radius", type=int, default=15)
    parser.add_argument("--min-track-frames", type=int, default=3)
    parser.add_argument("--active-quality-fraction", type=float, default=0.9)
    parser.add_argument("--obs-per-landmark", type=int, default=16)
    parser.add_argument("--lm-iters", type=int, default=3)
    parser.add_argument("--lambda-initial", type=float, default=0.0005)
    parser.add_argument("--freeze-rotations", action="store_true")
    parser.add_argument("--cg-tolerance-min", type=float, default=1e-6)
    parser.add_argument("--cg-tolerance-max", type=float, default=1e-2)
    parser.add_argument("--robust-scale", type=float, default=0.05)
    parser.add_argument("--max-point-med-ang", type=float, default=2.0)
    parser.add_argument("--min-positive-depth-frac", type=float, default=0.75)
    parser.add_argument("--imu-rot-weight", type=float, default=100.0)
    parser.add_argument("--gravity-weight", type=float, default=10.0)
    parser.add_argument("--constant-velocity-weight", type=float, default=1.0)
    parser.add_argument("--positive-depth-weight", type=float, default=0.1)
    parser.add_argument("--positive-depth-min", type=float, default=0.05)
    parser.add_argument("--positive-depth-softness", type=float, default=0.1)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    wall_start = time.time()
    source = np.load(args.initial_state)
    frame_idx = source["frame_idx"]
    poses = source["pose_wxyz_xyz"].copy()
    obs = load_solver_observations(
        args.recording,
        frame_idx,
        args.device,
        args.active_tracks,
        tuple(args.spatial_grid),
        args.active_temporal_radius,
        args.min_track_frames,
        args.active_quality_fraction,
        args.tracks,
    )
    imu = load_solver_imu(args.recording, frame_idx)

    points = np.zeros(
        (len(obs["selected_source_track_id"]), 3), np.float64)
    source_lookup = {
        int(track): point
        for point, track in enumerate(source["point_track_id"])
    }
    matched = np.zeros(len(points), bool)
    for point, track in enumerate(obs["selected_source_track_id"]):
        source_point = source_lookup.get(int(track))
        if source_point is not None:
            points[point] = source["points"][source_point]
            matched[point] = True
    if not np.all(matched):
        raise ValueError(
            f"initial state matched only {matched.sum()}/{len(points)} points")
    points, alive, median_angle, positive_fraction = retriangulate(
        poses,
        points,
        obs,
        args.robust_scale,
        args.obs_per_landmark,
        args.max_point_med_ang,
        args.min_positive_depth_frac,
    )
    train_rows = np.flatnonzero(~obs["validation"])
    validation_rows = np.flatnonzero(obs["validation"])
    train_before = angular_metrics(poses, points, obs, train_rows)
    validation_before = angular_metrics(
        poses, points, obs, validation_rows)
    print(
        f"initial angular: train p50/p90 "
        f"{train_before['median']:.3f}/{train_before['p90']:.3f} deg; "
        f"validation {validation_before['median']:.3f}/"
        f"{validation_before['p90']:.3f} deg")

    problem, values, variable_points = make_global_problem(
        poses, points, alive, obs, imu, args)
    cg = jaxls.ConjugateGradientConfig(
        tolerance_min=args.cg_tolerance_min,
        tolerance_max=args.cg_tolerance_max,
        preconditioner="block_jacobi",
    )
    termination = jaxls.TerminationConfig(
        max_iterations=args.lm_iters,
        early_termination=False,
    )
    initial_cost = problem.compute_residual_vector(values)
    initial_cost = float(jnp.sum(initial_cost ** 2))
    solve_jit = jax.jit(lambda problem, values: problem.solve(
        values,
        linear_solver=cg,
        sparse_mode="blockrow",
        trust_region=jaxls.TrustRegionConfig(
            lambda_initial=args.lambda_initial),
        termination=termination,
        verbose=False,
        return_summary=True,
    ))
    print("compiling and solving global Schur-CG problem")
    solve_start = time.time()
    solution, summary = solve_jit(problem, values)
    pose_var_type = CamCenterVar if args.freeze_rotations else jaxls.SE3Var
    jax.block_until_ready(solution[pose_var_type])
    final_cost = problem.compute_residual_vector(solution)
    final_cost = float(jnp.sum(final_cost ** 2))
    accepted = np.isfinite(final_cost) and final_cost <= initial_cost
    print(
        f"global solve: accepted={accepted}, "
        f"cost ratio={final_cost / max(initial_cost, 1e-12):.6f}, "
        f"LM iterations={int(summary.iterations) + 1}, "
        f"solve wall={time.time() - solve_start:.1f}s")
    if accepted:
        if args.freeze_rotations:
            poses = centers_to_poses(
                poses[:, :4],
                np.asarray(solution[CamCenterVar]),
            )
        else:
            poses = np.asarray(solution[jaxls.SE3Var].wxyz_xyz)
        points[variable_points] = np.asarray(solution[Point3Var])

    points, alive, median_angle, positive_fraction = retriangulate(
        poses,
        points,
        obs,
        args.robust_scale,
        args.obs_per_landmark,
        args.max_point_med_ang,
        args.min_positive_depth_frac,
    )
    train_after = angular_metrics(poses, points, obs, train_rows)
    validation_after = angular_metrics(
        poses, points, obs, validation_rows)
    print(
        f"final angular: train p50/p90 "
        f"{train_after['median']:.3f}/{train_after['p90']:.3f} deg; "
        f"validation {validation_after['median']:.3f}/"
        f"{validation_after['p90']:.3f} deg")
    write_probe_output(
        args.out,
        frame_idx,
        poses,
        points,
        alive,
        median_angle,
        positive_fraction,
        obs,
        {
            "global_schur_cg": True,
            "global_accepted": accepted,
            "global_initial_cost": initial_cost,
            "global_final_cost": final_cost,
            "global_lm_iterations": int(summary.iterations) + 1,
            "metric_train_median_deg": [
                train_before["median"], train_after["median"]],
            "metric_train_p90_deg": [
                train_before["p90"], train_after["p90"]],
            "metric_validation_median_deg": [
                validation_before["median"],
                validation_after["median"],
            ],
            "metric_validation_p90_deg": [
                validation_before["p90"],
                validation_after["p90"],
            ],
            "config_obs_per_landmark": args.obs_per_landmark,
            "config_lm_iters": args.lm_iters,
            "config_lambda_initial": args.lambda_initial,
            "config_freeze_rotations": args.freeze_rotations,
            "config_cg_tolerance_min": args.cg_tolerance_min,
            "config_cg_tolerance_max": args.cg_tolerance_max,
            "config_imu_rotation_weight": args.imu_rot_weight,
            "config_gravity_weight": args.gravity_weight,
            "config_constant_velocity_weight":
                args.constant_velocity_weight,
            "wall_seconds": time.time() - wall_start,
        },
    )
    print(f"wall total: {time.time() - wall_start:.1f}s")


if __name__ == "__main__":
    main()
