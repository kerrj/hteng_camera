#!/usr/bin/env python3
"""Plot window-local motion and optimizer convergence from windowed VIO output."""

import argparse

import matplotlib.pyplot as plt
import numpy as np


def camera_centers(poses):
    q = poses[:, :4].astype(np.float64)
    q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    t = poses[:, 4:].astype(np.float64)
    w, x, y, z = q.T
    rotation = np.stack([
        np.stack([1 - 2 * (y*y + z*z), 2 * (x*y - z*w),
                  2 * (x*z + y*w)], axis=-1),
        np.stack([2 * (x*y + z*w), 1 - 2 * (x*x + z*z),
                  2 * (y*z - x*w)], axis=-1),
        np.stack([2 * (x*z - y*w), 2 * (y*z + x*w),
                  1 - 2 * (x*x + y*y)], axis=-1),
    ], axis=-2)
    return -np.einsum("nji,nj->ni", rotation, t)


def normalized_history(history):
    scale = np.maximum(history[:, :1], 1e-12)
    return history / scale


def plot_history(ax, history, title):
    normalized = normalized_history(history)
    iterations = np.arange(normalized.shape[1])
    ax.plot(iterations, normalized.T, color="#9ca3af", alpha=0.18, lw=0.7)
    ax.fill_between(
        iterations,
        np.percentile(normalized, 10, axis=0),
        np.percentile(normalized, 90, axis=0),
        color="#2563eb",
        alpha=0.18,
        label="p10-p90",
    )
    ax.plot(iterations, np.median(normalized, axis=0),
            color="#1d4ed8", lw=2, label="median")
    ax.set(title=title, xlabel="LM iteration", ylabel="cost / initial cost")
    ax.set_yscale("log")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trajectory")
    parser.add_argument("--out", default="/tmp/vio_window_diagnostics.png")
    parser.add_argument("--overlap-frames", type=int, default=30)
    args = parser.parse_args()

    data = np.load(args.trajectory)
    starts = data["window_starts"].astype(int)
    path, step_p95, max_step, accel_p95, max_accel = [], [], [], [], []
    if "window_centers" in data:
        local_windows = data["window_centers"]
        motion_title = "Exact motion within each optimized window"
    else:
        centers = camera_centers(data["pose_wxyz_xyz"])
        if len(starts) < 2:
            raise ValueError("At least two windows are required")
        window_size = int(starts[1] - starts[0] + args.overlap_frames)
        coverage_end = starts[0] + window_size
        owned_ranges = [(starts[0], coverage_end)]
        for start in starts[1:]:
            owned_ranges.append((coverage_end, start + window_size))
            coverage_end = start + window_size
        local_windows = [centers[begin:end] for begin, end in owned_ranges]
        motion_title = "Motion within owned frames (handoffs excluded)"

    for local in local_windows:
        step = np.linalg.norm(np.diff(local, axis=0), axis=1)
        accel = np.linalg.norm(np.diff(local, n=2, axis=0), axis=1)
        path.append(np.sum(step))
        step_p95.append(np.percentile(step, 95) if len(step) else 0.0)
        max_step.append(np.max(step) if len(step) else 0.0)
        accel_p95.append(np.percentile(accel, 95) if len(accel) else 0.0)
        max_accel.append(np.max(accel) if len(accel) else 0.0)

    path = np.asarray(path)
    step_p95 = np.asarray(step_p95)
    max_step = np.asarray(max_step)
    accel_p95 = np.asarray(accel_p95)
    max_accel = np.asarray(max_accel)
    windows = np.arange(len(starts))

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    axes[0, 0].plot(windows, path, color="#111827", lw=1.2)
    axes[0, 0].scatter(windows, path, c=max_step, cmap="magma", s=20)
    axes[0, 0].set(
        title=motion_title,
        xlabel="window",
        ylabel="path length (m)",
    )
    axes[0, 0].grid(alpha=0.2)

    axes[0, 1].plot(windows, step_p95, label="step p95", color="#2563eb")
    axes[0, 1].plot(windows, max_step, label="max step", color="#dc2626")
    axes[0, 1].plot(windows, accel_p95, label="acceleration p95",
                    color="#059669")
    axes[0, 1].set(
        title="Within-window motion consistency",
        xlabel="window",
        ylabel="meters per frame",
    )
    axes[0, 1].grid(alpha=0.2)
    axes[0, 1].legend(frameon=False)

    plot_history(
        axes[1, 0], data["positioning_cost_history"],
        "Positioning total objective",
    )
    plot_history(
        axes[1, 1], data["refine_cost_history"],
        "SE(3) refinement total objective",
    )
    fig.savefig(args.out, dpi=160)

    for name, values in (
        ("owned path (m)", path),
        ("step p95 (m)", step_p95),
        ("max step (m)", max_step),
        ("acceleration p95 (m)", accel_p95),
        ("max acceleration (m)", max_accel),
    ):
        worst = np.argsort(values)[-8:][::-1]
        print(f"{name}: p50={np.median(values):.4g} "
              f"p95={np.percentile(values, 95):.4g} max={values.max():.4g}")
        print("  worst:", ", ".join(
            f"w{index}={values[index]:.4g}" for index in worst))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
