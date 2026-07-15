"""Diagnostics on stage 5 output: per-landmark angular residuals + depth
stats (for outlier filtering design), and trajectory kink detection
(frame-to-frame velocity/turn-angle spikes, compared against the IMU prior).

Run (from data_processing/vio/):
    python vio_diagnose.py ../../testimu
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def quat_to_matrix(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("recording")
    ap.add_argument("--trajectory", default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    traj_path = args.trajectory or os.path.join(args.recording, "trajectory.npz")
    d = np.load(traj_path)
    frame_idx = d["frame_idx"]
    pose_wxyz_xyz = d["pose_wxyz_xyz"]
    points = d["points"]
    n_frames = len(frame_idx)
    n_points = len(points)

    # --- camera centers / rotations in world frame ---
    R_wl = np.stack([quat_to_matrix(q) for q in pose_wxyz_xyz[:, :4]])
    t_wl = pose_wxyz_xyz[:, 4:]
    cam_pos = -np.einsum("nij,nj->ni", R_wl.transpose(0, 2, 1), t_wl)

    # ---------- trajectory kink analysis ----------
    v = np.diff(cam_pos, axis=0)
    speed = np.linalg.norm(v, axis=1)
    # turn angle between consecutive velocity vectors
    v0, v1 = v[:-1], v[1:]
    cosang = np.einsum("ni,ni->n", v0, v1) / (
        np.linalg.norm(v0, axis=1) * np.linalg.norm(v1, axis=1) + 1e-12)
    turn_deg = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
    # relative rotation magnitude between consecutive poses (optimized)
    rel_rot_deg = np.zeros(n_frames - 1)
    for i in range(n_frames - 1):
        R_rel = R_wl[i] @ R_wl[i + 1].T
        rel_rot_deg[i] = np.degrees(np.arccos(np.clip((np.trace(R_rel) - 1) / 2, -1, 1)))

    # IMU-predicted relative rotation magnitude for comparison
    imu = np.load(os.path.join(args.recording, "imu_relative.npz"))
    imu_frame_idx = imu["frame_idx"]
    imu_valid = imu["frame_valid"]
    rel_quat = imu["rel_quat"]
    # map surviving frames back
    pos_of = {int(f): i for i, f in enumerate(imu_frame_idx)}
    imu_rot_deg = np.full(n_frames - 1, np.nan)
    for i in range(n_frames - 1):
        a, b = pos_of.get(int(frame_idx[i])), pos_of.get(int(frame_idx[i + 1]))
        if a is not None and b is not None and b == a + 1 and imu["rel_valid"][a]:
            w = np.clip(abs(rel_quat[a, 0]), -1, 1)
            imu_rot_deg[i] = np.degrees(2 * np.arccos(w))

    rot_mismatch = np.abs(rel_rot_deg - imu_rot_deg)

    print("=== trajectory kink analysis ===")
    print(f"speed (m/frame): median {np.median(speed):.4f}, p99 {np.percentile(speed, 99):.4f}, max {speed.max():.4f}")
    order = np.argsort(speed)[::-1][:10]
    print("top-10 speed spikes (pose slot i -> i+1, video frame):")
    for i in order:
        print(f"  slot {i:4d} (frame {frame_idx[i]:4d}->{frame_idx[i+1]:4d}): "
              f"{speed[i]:.4f} m, opt rot {rel_rot_deg[i]:.2f} deg, imu rot {imu_rot_deg[i]:.2f} deg")
    ok = ~np.isnan(rot_mismatch)
    print(f"rot mismatch opt-vs-imu (deg): median {np.median(rot_mismatch[ok]):.4f}, "
          f"p99 {np.percentile(rot_mismatch[ok], 99):.4f}, max {np.nanmax(rot_mismatch):.4f}")
    order = np.argsort(np.where(ok, rot_mismatch, -1))[::-1][:10]
    print("top-10 rotation mismatches:")
    for i in order:
        print(f"  slot {i:4d} (frame {frame_idx[i]:4d}->{frame_idx[i+1]:4d}): "
              f"opt {rel_rot_deg[i]:.3f} vs imu {imu_rot_deg[i]:.3f} deg, speed {speed[i]:.4f} m")
    # sharp cusp: high turn angle AND meaningful speed on both sides
    sig = (np.linalg.norm(v0, axis=1) > np.percentile(speed, 50)) & \
          (np.linalg.norm(v1, axis=1) > np.percentile(speed, 50))
    cusp_score = np.where(sig, turn_deg, 0)
    order = np.argsort(cusp_score)[::-1][:10]
    print("top-10 direction cusps (turn angle at meaningful speed):")
    for i in order:
        print(f"  slot {i+1:4d} (frame {frame_idx[i+1]:4d}): turn {turn_deg[i]:.1f} deg, "
              f"speeds {np.linalg.norm(v0[i]):.4f}/{np.linalg.norm(v1[i]):.4f} m")

    # ---------- per-landmark residual/depth analysis ----------
    print("\n=== landmark analysis ===")
    tracks_path = os.path.join(args.recording, "tracks.jsonl")
    max_frame = int(frame_idx[-1])
    frame_to_pose = {int(f): i for i, f in enumerate(frame_idx)}

    import h5py
    with h5py.File(os.path.join(args.recording, "features.h5"), "r") as f:
        ls, rs = f.attrs["left_serial"], f.attrs["right_serial"]
    calib = {}
    for s in (ls, rs):
        c = json.load(open(os.path.join(args.recording, f"calib_{s}.json")))["intrinsics"]
        calib[s] = (np.array(c["K"]), np.array(c["dist"]))
    st = json.load(open(os.path.join(args.recording, f"stereo_{ls}_{rs}.json")))
    R_st = np.array(st["R"])
    t_st = np.array(st["t"]).reshape(3)

    pose_ids, point_ids, obs_px, obs_is_right = [], [], [], []
    track_len = np.zeros(n_points, dtype=np.int64)
    with open(tracks_path) as f:
        pt_idx = 0
        for line in f:
            r = json.loads(line)
            obs = [o for o in r["observations"] if o["frame"] <= max_frame]
            if len(obs) < 2:
                continue
            for o in obs:
                if o["frame"] not in frame_to_pose:
                    continue
                pose_ids.append(frame_to_pose[o["frame"]])
                point_ids.append(pt_idx)
                obs_px.append(o["px"])
                obs_is_right.append(o["eye"] == "right")
            track_len[pt_idx] = len(obs)
            pt_idx += 1
    pose_ids = np.array(pose_ids)
    point_ids = np.array(point_ids)
    obs_px = np.array(obs_px, dtype=np.float64)
    obs_is_right = np.array(obs_is_right)
    n_obs = len(pose_ids)
    assert pt_idx == n_points, (pt_idx, n_points)

    import torch
    import fisheye_pinhole as FP
    ray_cam = np.zeros((n_obs, 3))
    for eye, serial in (("left", ls), ("right", rs)):
        m = obs_is_right == (eye == "right")
        K, D = calib[serial]
        px = torch.from_numpy(obs_px[m].astype(np.float32)).to(args.device)
        K_t = torch.tensor(K, dtype=torch.float32, device=args.device)
        D_t = torch.tensor(D, dtype=torch.float32, device=args.device)
        ray_cam[m] = FP.fisheye_unproject(px[:, 0], px[:, 1], K_t, D_t).cpu().numpy()

    # per-observation: angle between observed ray and point direction, and depth
    R_obs = R_wl[pose_ids]
    t_obs = t_wl[pose_ids]
    right = obs_is_right
    R_obs = np.where(right[:, None, None], np.einsum("ij,njk->nik", R_st, R_obs), R_obs)
    t_obs = np.where(right[:, None], (np.einsum("ij,nj->ni", R_st, t_obs) + t_st), t_obs)
    p_cam = np.einsum("nij,nj->ni", R_obs, points[point_ids]) + t_obs
    depth = p_cam[:, 2]
    dir_cam = p_cam / (np.linalg.norm(p_cam, axis=1, keepdims=True) + 1e-12)
    ang = np.degrees(np.arccos(np.clip(np.einsum("ni,ni->n", dir_cam, ray_cam), -1, 1)))

    # per-landmark aggregates
    def agg(vals, fn, init):
        out = np.full(n_points, init)
        np.minimum.at(out, point_ids, vals) if fn == "min" else None
        return out
    lm_max_ang = np.zeros(n_points)
    np.maximum.at(lm_max_ang, point_ids, ang)
    lm_med_ang = np.full(n_points, np.nan)
    order = np.argsort(point_ids, kind="stable")
    splits = np.searchsorted(point_ids[order], np.arange(n_points))
    for k in range(n_points):
        lo = splits[k]
        hi = splits[k + 1] if k + 1 < n_points else n_obs
        lm_med_ang[k] = np.median(ang[order[lo:hi]])
    lm_min_depth = np.full(n_points, np.inf)
    np.minimum.at(lm_min_depth, point_ids, depth)
    lm_dist = np.linalg.norm(points - cam_pos.mean(axis=0), axis=1)

    print(f"{n_obs} observations, {n_points} landmarks")
    print(f"per-obs angular residual (deg): median {np.median(ang):.3f}, "
          f"p90 {np.percentile(ang, 90):.3f}, p99 {np.percentile(ang, 99):.3f}, max {ang.max():.1f}")
    neg = lm_min_depth < 0
    print(f"landmarks with any negative-depth obs: {neg.sum()} ({100*neg.mean():.1f}%)")
    for thr in (0.5, 1.0, 2.0, 5.0):
        bad = lm_med_ang > thr
        print(f"landmarks with median angular residual > {thr} deg: {bad.sum()} ({100*bad.mean():.1f}%)")
    far = lm_dist > 10 * np.median(lm_dist)
    print(f"landmarks farther than 10x median distance from trajectory centroid: {far.sum()}")
    print(f"distance from centroid: median {np.median(lm_dist):.2f}, p99 {np.percentile(lm_dist, 99):.2f}, max {lm_dist.max():.1f}")
    # cross-tabs: are negative-depth landmarks also high-residual?
    print(f"neg-depth AND median residual > 1 deg: {(neg & (lm_med_ang > 1)).sum()}")
    print(f"neg-depth AND median residual <= 1 deg: {(neg & (lm_med_ang <= 1)).sum()}")
    print(f"track length of neg-depth landmarks: median {np.median(track_len[neg]) if neg.any() else 0:.0f} "
          f"vs positive: {np.median(track_len[~neg]):.0f}")
    # residual vs track length
    for lo, hi in ((2, 3), (3, 5), (5, 10), (10, 10**9)):
        m = (track_len >= lo) & (track_len < hi)
        if m.sum():
            print(f"track len [{lo},{hi}): {m.sum()} landmarks, median residual "
                  f"{np.nanmedian(lm_med_ang[m]):.3f} deg, neg-depth {100*neg[m].mean():.1f}%")

    np.savez(os.path.join(args.recording, "diagnostics.npz"),
             lm_med_ang=lm_med_ang, lm_max_ang=lm_max_ang, lm_min_depth=lm_min_depth,
             lm_dist=lm_dist, track_len=track_len,
             speed=speed, turn_deg=turn_deg, rel_rot_deg=rel_rot_deg, imu_rot_deg=imu_rot_deg)
    print(f"wrote {os.path.join(args.recording, 'diagnostics.npz')}")


if __name__ == "__main__":
    main()
