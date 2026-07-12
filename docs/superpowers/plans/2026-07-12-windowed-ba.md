# Windowed Bundle Adjustment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Converged full-length VIO trajectories for long recordings by solving overlapping ~30 s BA windows, stitching them with a closed-form 4-DOF alignment, and re-running the full solve from the stitched init.

**Architecture:** `vio/vio_bundle_adjust.py` gains two flags (`--start-frame` window restriction, `--init-trajectory` init override) and stays the single source of truth for solver math. New `vio/vio_stitch.py` holds pure-numpy 4-DOF gauge math (unit-tested). New `vio/vio_windowed_ba.py` orchestrates: window ranges → multi-GPU subprocess fanout → stitch/blend → global refine.

**Tech Stack:** Python 3.10 (conda env `eyeball` on sphynx), numpy, jax/jaxls/jaxlie (existing solver), subprocess multi-GPU (8× RTX A6000), pytest 9.1.1.

**Spec:** `docs/superpowers/specs/2026-07-12-windowed-ba-design.md`

## Global Constraints

- All commits go to branch `depth_understanding`. NEVER push to, merge into, or modify `main`.
- Recording data stays out of git: `/long-test1/`, `/long-test2/` already gitignored; this plan adds `/testimu/`.
- Run everything in conda env `eyeball`: `conda run --no-capture-output -n eyeball python …`. GPU jobs pin `CUDA_VISIBLE_DEVICES`; parallel jax subprocesses set `XLA_PYTHON_CLIENT_PREALLOCATE=false`.
- Pipeline conventions: positional `recording` arg; inputs default from `<recording>/derived/`; final output `<recording>/derived/trajectory.npz` with the existing schema (keys `frame_idx, pose_wxyz_xyz, points, point_first_frame, point_first_is_right, point_first_px, point_alive, point_med_ang, cost_history`). Poses are WORLD→CAMERA wxyz_xyz; camera center `c = -Rᵀ t`.
- Do not change solver math in `vio_bundle_adjust.py` — the change surface is exactly the two new CLI flags and a `load_tracks` `min_frame` bound.
- Tests live in `data_processing/vio/`, run as: `cd data_processing/vio && conda run --no-capture-output -n eyeball python -m pytest <file> -v`.
- Commit messages end with:
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`
  `Claude-Session: https://claude.ai/code/session_01BdNbSVuPYhr1KeDeTySjF3`
- Key facts: long-test2 `imu_relative.npz` spans video frames 0..11042 (11043 entries, 1 invalid-timestamp frame dropped in-solve). jkerr's testimu reference lives at `/home/jkerr/hteng_camera/testimu` (FLAT layout, no `derived/`), reference full-solve residual ~0.23°. Windowed solves of ≤ ~900 frames converge to 0.2–0.6°; the full-length long-test2 solve plateaus at 8–9° (IMU gyro-chain init drift — the problem this plan fixes).

---

### Task 1: 4-DOF stitch math (`vio_stitch.py`)

Pure numpy gauge math used by the orchestrator. Windows share gravity alignment (+z up) and metric scale, so two windows' worlds differ by yaw-about-z + translation: `X_a = Rz(θ) X_b + t`.

**Files:**
- Create: `data_processing/vio/vio_stitch.py`
- Test: `data_processing/vio/test_vio_stitch.py`

**Interfaces:**
- Produces (used by Task 4):
  - `quat_to_R(q: (...,4) wxyz) -> (...,3,3)`
  - `quat_mul(a, b) -> (...,4)` Hamilton product, wxyz
  - `yaw_R(theta: float) -> (3,3)`
  - `centers_of(poses: (N,7)) -> (N,3)` camera centers `-Rᵀt`
  - `fit_yaw_translation(poses_a, poses_b) -> (theta: float, t: (3,), diag: dict)` with diag keys `yaw_deg, yaw_spread_deg, center_rms_m, n_shared`
  - `apply_yaw_translation(poses: (N,7), theta, t) -> (N,7)` re-expresses gauge-b poses in gauge a
  - `compose_yaw_translation(theta1, t1, theta2, t2) -> (theta, t)` (apply 2 first, then 1)
  - `blend_poses(poses_a, poses_b, w: (N,)) -> (N,7)` per-frame blend, w=0→a, w=1→b

- [ ] **Step 1: Write the failing tests**

Create `data_processing/vio/test_vio_stitch.py`:

```python
import numpy as np

import vio_stitch as ST


def rand_poses(n, seed):
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    q[q[:, 0] < 0] *= -1
    return np.concatenate([q, rng.normal(size=(n, 3))], axis=1)


def gauge_b_of(poses_a, theta, t):
    """Re-express gauge-a poses in gauge b, where X_a = Rz(theta) X_b + t.
    The inverse map is X_b = Rz(-theta) X_a - Rz(-theta) t."""
    return ST.apply_yaw_translation(poses_a, -theta, -(ST.yaw_R(-theta) @ t))


def test_quat_mul_matches_matrix_product():
    a, b = rand_poses(8, 0), rand_poses(8, 1)
    R_ab = ST.quat_to_R(ST.quat_mul(a[:, :4], b[:, :4]))
    assert np.allclose(R_ab, ST.quat_to_R(a[:, :4]) @ ST.quat_to_R(b[:, :4]),
                       atol=1e-12)


def test_fit_recovers_known_gauge():
    poses_a = rand_poses(50, 2)
    theta_true, t_true = 1.234, np.array([0.5, -2.0, 0.3])
    poses_b = gauge_b_of(poses_a, theta_true, t_true)
    theta, t, diag = ST.fit_yaw_translation(poses_a, poses_b)
    assert abs(theta - theta_true) < 1e-9
    assert np.allclose(t, t_true, atol=1e-9)
    assert diag["center_rms_m"] < 1e-9
    assert diag["yaw_spread_deg"] < 1e-6
    assert diag["n_shared"] == 50


def test_fit_near_pi_wraparound_with_noise():
    rng = np.random.default_rng(4)
    poses_a = rand_poses(200, 5)
    theta_true, t_true = np.pi - 0.01, np.array([3.0, 0.0, 1.0])
    poses_b = gauge_b_of(poses_a, theta_true, t_true)
    poses_b[:, 4:] += rng.normal(scale=1e-3, size=(200, 3))  # mm noise on t
    theta, t, _ = ST.fit_yaw_translation(poses_a, poses_b)
    assert abs((theta - theta_true + np.pi) % (2 * np.pi) - np.pi) < 1e-3
    assert np.allclose(t, t_true, atol=5e-3)


def test_apply_roundtrip():
    poses_a = rand_poses(20, 3)
    theta, t = -0.7, np.array([1.0, 2.0, -0.5])
    poses_b = gauge_b_of(poses_a, theta, t)
    back = ST.apply_yaw_translation(poses_b, theta, t)
    sign = np.sign((poses_a[:, :4] * back[:, :4]).sum(1, keepdims=True))
    assert np.allclose(poses_a[:, :4], back[:, :4] * sign, atol=1e-9)
    assert np.allclose(poses_a[:, 4:], back[:, 4:], atol=1e-9)


def test_compose_equals_sequential_apply():
    poses = rand_poses(15, 8)
    th1, t1 = 0.4, np.array([1.0, -1.0, 0.2])
    th2, t2 = -1.1, np.array([0.3, 2.0, -0.7])
    seq = ST.apply_yaw_translation(ST.apply_yaw_translation(poses, th2, t2), th1, t1)
    thc, tc = ST.compose_yaw_translation(th1, t1, th2, t2)
    comp = ST.apply_yaw_translation(poses, thc, tc)
    assert np.allclose(ST.centers_of(seq), ST.centers_of(comp), atol=1e-9)
    assert np.allclose(ST.quat_to_R(seq[:, :4]), ST.quat_to_R(comp[:, :4]), atol=1e-9)


def test_blend_endpoints_and_midpoint():
    pa, pb = rand_poses(10, 6), rand_poses(10, 7)
    b0 = ST.blend_poses(pa, pb, np.zeros(10))
    b1 = ST.blend_poses(pa, pb, np.ones(10))
    assert np.allclose(b0, pa, atol=1e-12)
    assert np.allclose(ST.quat_to_R(b1[:, :4]), ST.quat_to_R(pb[:, :4]), atol=1e-12)
    assert np.allclose(ST.centers_of(b1), ST.centers_of(pb), atol=1e-12)
    mid = ST.blend_poses(pa, pb, np.full(10, 0.5))
    assert np.allclose(ST.centers_of(mid),
                       0.5 * (ST.centers_of(pa) + ST.centers_of(pb)), atol=1e-12)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/smahapatra/hteng_camera/data_processing/vio && conda run --no-capture-output -n eyeball python -m pytest test_vio_stitch.py -v`
Expected: collection error — `ModuleNotFoundError: No module named 'vio_stitch'`

- [ ] **Step 3: Implement `vio_stitch.py`**

Create `data_processing/vio/vio_stitch.py`:

```python
"""4-DOF stitching math for windowed VIO trajectories.

Windowed BA solves (vio_bundle_adjust.py over a --start-frame range) share
gravity alignment (+z up: roll/pitch fixed by the gravity prior) and metric
scale (stereo baseline), so two windows' world gauges differ by exactly a yaw
about +z plus a translation:

    X_a = Rz(theta) @ X_b + t

Poses are (N,7) wxyz_xyz WORLD->CAMERA (jaxls SE3Var convention):
X_cam = R @ X_world + t_pose, camera center c = -R^T @ t_pose.
Pure numpy, no GPU.
"""
import numpy as np


def quat_to_R(q):
    """(...,4) wxyz -> (...,3,3)."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], -2)


def quat_mul(a, b):
    """Hamilton product (wxyz), broadcasting: R(quat_mul(a,b)) = R(a) @ R(b)."""
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw], -1)


def yaw_R(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def centers_of(poses):
    """(N,7) -> (N,3) camera centers c = -R^T t."""
    R = quat_to_R(poses[:, :4])
    return -np.einsum("nji,nj->ni", R, poses[:, 4:])


def fit_yaw_translation(poses_a, poses_b):
    """4-DOF gauge alignment from the SAME frames solved in two windows.

    Pose relation: X_cam = R_a X_a + t_a = R_b X_b + t_b with
    X_a = Rz X_b + t  =>  R_b = R_a Rz, so each shared frame votes
    Rz ~ R_a^T R_b (nearly pure z-rotation since both gauges are
    gravity-aligned). theta = circular mean of the votes; then
    t = mean(c_a - Rz c_b) over camera centers.

    Returns (theta, t, diag): diag has yaw_deg, yaw_spread_deg (vote std),
    center_rms_m (post-alignment center disagreement), n_shared.
    """
    Ra = quat_to_R(poses_a[:, :4])
    Rb = quat_to_R(poses_b[:, :4])
    M = np.einsum("nji,njk->nik", Ra, Rb)  # R_a^T R_b
    votes = np.arctan2(M[:, 1, 0] - M[:, 0, 1], M[:, 0, 0] + M[:, 1, 1])
    theta = float(np.arctan2(np.sin(votes).mean(), np.cos(votes).mean()))
    Rz = yaw_R(theta)
    ca, cb = centers_of(poses_a), centers_of(poses_b)
    t = (ca - cb @ Rz.T).mean(axis=0)
    resid = ca - (cb @ Rz.T + t)
    wrap = (votes - theta + np.pi) % (2 * np.pi) - np.pi
    diag = {"yaw_deg": float(np.degrees(theta)),
            "yaw_spread_deg": float(np.degrees(wrap.std())),
            "center_rms_m": float(np.sqrt((resid ** 2).sum(1).mean())),
            "n_shared": len(poses_a)}
    return theta, t, diag


def apply_yaw_translation(poses, theta, t):
    """Re-express WORLD->CAM poses solved in gauge b in gauge a
    (X_a = Rz X_b + t): R' = R Rz^T, c' = Rz c + t, t' = -R' c'."""
    qz_inv = np.array([np.cos(theta / 2), 0.0, 0.0, -np.sin(theta / 2)])
    q_new = quat_mul(poses[:, :4], qz_inv[None, :])
    c_new = centers_of(poses) @ yaw_R(theta).T + t
    t_new = -np.einsum("nij,nj->ni", quat_to_R(q_new), c_new)
    return np.concatenate([q_new, t_new], axis=1)


def compose_yaw_translation(theta1, t1, theta2, t2):
    """Composition (theta1,t1) o (theta2,t2): apply 2 first, then 1.
    X0 = Rz1 (Rz2 X + t2) + t1 = Rz(th1+th2) X + (Rz1 t2 + t1)."""
    return theta1 + theta2, yaw_R(theta1) @ t2 + t1


def blend_poses(poses_a, poses_b, w):
    """Per-frame blend of two ALIGNED pose sets; w in [0,1], 0 -> a, 1 -> b.
    Camera-center lerp + quaternion nlerp (sign-aligned) -- adequate for the
    mm/sub-degree disagreements left after 4-DOF alignment."""
    qa, qb = poses_a[:, :4], poses_b[:, :4].copy()
    qb[(qa * qb).sum(1) < 0] *= -1
    q = (1 - w[:, None]) * qa + w[:, None] * qb
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    c = (1 - w[:, None]) * centers_of(poses_a) + w[:, None] * centers_of(poses_b)
    t = -np.einsum("nij,nj->ni", quat_to_R(q), c)
    return np.concatenate([q, t], axis=1)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/smahapatra/hteng_camera/data_processing/vio && conda run --no-capture-output -n eyeball python -m pytest test_vio_stitch.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
cd /home/smahapatra/hteng_camera
git add data_processing/vio/vio_stitch.py data_processing/vio/test_vio_stitch.py
git commit -m "vio: 4-DOF (yaw+translation) stitch math for windowed BA

Pure-numpy gauge alignment between gravity-aligned metric windows:
per-frame yaw votes (R_a^T R_b) -> circular mean, center-mean translation,
compose/apply/blend helpers. Unit-tested (recovery, wraparound, roundtrip,
compose, blend endpoints)."
```

(Append the two Global Constraints trailer lines to this and every commit message.)

---

### Task 2: `--start-frame` window restriction in the solver

**Files:**
- Modify: `data_processing/vio/vio_bundle_adjust.py` (argparse ~line 58; frame-cap block ~lines 252–260; `load_tracks` ~line 118; call site ~line 302)
- Test: `data_processing/vio/test_vio_bundle_adjust_window.py`
- Modify: `.gitignore` (add `/testimu/`)

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: `vio_bundle_adjust.py --start-frame S --n-frames N` solves only video frames `[S, S+N]` (inclusive, matched against `frame_idx`); `load_tracks(tracks_path, min_frame, max_frame)` (both bounds inclusive, `None` = unbounded). Behavior without `--start-frame` is unchanged. Task 4 shells out to this.

- [ ] **Step 1: Write the failing test**

Create `data_processing/vio/test_vio_bundle_adjust_window.py`:

```python
import json

import vio_bundle_adjust as BA


def _write_tracks(path, recs):
    with open(path, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")


def test_load_tracks_min_max_window(tmp_path):
    p = str(tmp_path / "tracks.jsonl")
    _write_tracks(p, [
        # 10-frame track: frames 0..9 survive windowing to [3,7] with 5 obs
        {"observations": [{"eye": "left", "frame": f, "px": [1.0, 2.0]}
                          for f in range(10)]},
        # straddler: only 1 obs inside [3,7] -> dropped (<2 obs rule)
        {"observations": [{"eye": "left", "frame": 0, "px": [0.0, 0.0]},
                          {"eye": "right", "frame": 5, "px": [0.0, 0.0]},
                          {"eye": "left", "frame": 9, "px": [0.0, 0.0]}]},
    ])
    tracks = BA.load_tracks(p, 3, 7)
    assert len(tracks) == 1
    frames = [f for _, f, _ in tracks[0]]
    assert min(frames) == 3 and max(frames) == 7 and len(frames) == 5


def test_load_tracks_none_bounds_keep_everything(tmp_path):
    p = str(tmp_path / "tracks.jsonl")
    _write_tracks(p, [{"observations": [{"eye": "left", "frame": f, "px": [0.0, 0.0]}
                                        for f in range(4)]}])
    assert len(BA.load_tracks(p, None, None)[0]) == 4
```

Note: the straddler has 2 obs total outside plus 1 inside — after windowing it has 1 obs, below the existing `len(obs) < 2` cutoff, so it must vanish.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/smahapatra/hteng_camera/data_processing/vio && conda run --no-capture-output -n eyeball python -m pytest test_vio_bundle_adjust_window.py -v`
Expected: FAIL — `TypeError: load_tracks() takes 2 positional arguments but 3 were given` (the import itself is heavy — jax/torch — and takes ~10 s; that's normal).

- [ ] **Step 3: Implement**

In `data_processing/vio/vio_bundle_adjust.py`, three edits:

(a) argparse — directly under the existing `--n-frames` argument:

```python
    p.add_argument("--start-frame", type=int, default=None,
                    help="first video frame of the solve window; with --n-frames "
                         "this solves frames [start, start+n] (windowed BA); "
                         "default: the recording's first frame")
```

(b) Replace the `if args.n_frames is not None:` block (currently prefix-only slicing) with contiguous-range slicing:

```python
    if args.n_frames is not None or args.start_frame is not None:
        lo = args.start_frame if args.start_frame is not None else int(frame_idx[0])
        hi = lo + args.n_frames if args.n_frames is not None else int(frame_idx[-1])
        keep = (frame_idx >= lo) & (frame_idx <= hi)
        a = int(np.argmax(keep))
        b = a + int(keep.sum())
        assert keep.any() and keep[a:b].all(), \
            f"frame window [{lo},{hi}] empty or non-contiguous in frame_idx"
        frame_idx = frame_idx[a:b]
        frame_valid = frame_valid[a:b]
        rel_quat = rel_quat[a:b - 1]
        rel_valid = rel_valid[a:b - 1]
        gravity_cam = gravity_cam[a:b]
        gravity_weight = gravity_weight[a:b]
```

(`rel_quat[i]` ties pose `i` to `i+1`, so a kept pose range `a..b-1` keeps edges `a..b-2` = `rel_quat[a:b-1]`. With `--n-frames` alone, `a == 0` reproduces the old prefix behavior exactly. The IMU chain seed already uses `gravity_cam[0]` of the SLICED arrays, so each window automatically re-seeds gravity-aligned at its own first frame — no further change needed; that is the entire windowing fix.)

(c) `load_tracks` gains a `min_frame` bound:

```python
def load_tracks(tracks_path, min_frame, max_frame):
    """Returns list of tracks; each track is a list of (eye, frame, px (2,))
    observations, restricted to min_frame <= frame <= max_frame (either bound
    None = unbounded)."""
    tracks = []
    with open(tracks_path) as f:
        for line in f:
            r = json.loads(line)
            obs = [o for o in r["observations"]
                   if (min_frame is None or o["frame"] >= min_frame)
                   and (max_frame is None or o["frame"] <= max_frame)]
            if len(obs) < 2:
                continue
            tracks.append([(o["eye"], o["frame"], np.array(o["px"], dtype=np.float64))
                            for o in obs])
    return tracks
```

and its call site becomes:

```python
    tracks = load_tracks(tracks_path, int(frame_idx[0]), max_frame)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/smahapatra/hteng_camera/data_processing/vio && conda run --no-capture-output -n eyeball python -m pytest test_vio_bundle_adjust_window.py test_vio_stitch.py -v`
Expected: 8 passed

- [ ] **Step 5: Set up the testimu harness (symlinks into jkerr's flat-layout copy)**

jkerr's testimu predates the `derived/` reorg; build the expected layout with symlinks (read-only use, no 4 GB copy):

```bash
cd /home/smahapatra/hteng_camera
mkdir -p testimu/derived
ln -sf /home/jkerr/hteng_camera/testimu/calib_046060323008.json testimu/
ln -sf /home/jkerr/hteng_camera/testimu/calib_046060323001.json testimu/
ln -sf /home/jkerr/hteng_camera/testimu/stereo_046060323008_046060323001.json testimu/
ln -sf /home/jkerr/hteng_camera/testimu/features.h5 testimu/derived/
ln -sf /home/jkerr/hteng_camera/testimu/tracks.jsonl testimu/derived/
ln -sf /home/jkerr/hteng_camera/testimu/imu_relative.npz testimu/derived/
```

Append to `.gitignore` under the existing recording block:

```
/testimu/
```

- [ ] **Step 6: GPU smoke test — window solve on testimu**

```bash
cd /home/smahapatra/hteng_camera/data_processing/vio
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n eyeball python vio_bundle_adjust.py \
    ../../testimu --start-frame 300 --n-frames 300 \
    --out ../../testimu/derived/windows/smoke_300_600.npz \
    --loss-plot ../../testimu/derived/windows/smoke_loss.png
```

(Create `testimu/derived/windows/` first: `mkdir -p ../../testimu/derived/windows`.)
Expected: prints `~301 pose frames (idx 300..600)`, both stages converge, `median angular residual` **< 1.0 deg** (testimu solves at ~0.23° full-length, a 300-frame window should be similar). Verify the output range:

```bash
conda run --no-capture-output -n eyeball python -c "
import numpy as np
d = np.load('../../testimu/derived/windows/smoke_300_600.npz')
print(d['frame_idx'][0], d['frame_idx'][-1], len(d['frame_idx']))
assert d['frame_idx'][0] >= 300 and d['frame_idx'][-1] <= 600"
```

Expected: `300 600 301` (or ±1 if a frame is invalid), assert passes.

- [ ] **Step 7: Commit**

```bash
cd /home/smahapatra/hteng_camera
git add data_processing/vio/vio_bundle_adjust.py \
        data_processing/vio/test_vio_bundle_adjust_window.py .gitignore
git commit -m "vio_bundle_adjust: --start-frame window restriction

Solve an arbitrary [start, start+n] frame range: contiguous slicing of the
IMU arrays (rel edges a:b-1) and a min_frame bound in load_tracks. The IMU
rotation chain seeds gravity-aligned at the window's first frame (existing
behavior on the sliced arrays), so gyro drift resets per window. --n-frames
alone is unchanged (prefix case)."
```

---

### Task 3: `--init-trajectory` init override in the solver

**Files:**
- Modify: `data_processing/vio/vio_bundle_adjust.py` (argparse; init block right after `point_init = point_init.at[:, 2].add(args.init_depth)` ~line 363)

**Interfaces:**
- Consumes: a trajectory npz (`frame_idx`, `pose_wxyz_xyz`) whose `frame_idx` is a superset of the solve's surviving frames — Task 4 passes the stitched trajectory here.
- Produces: `vio_bundle_adjust.py --init-trajectory <npz>` replaces stage-1's frozen rotations (IMU gyro chain) and center inits (random) with values from the npz. Landmarks/scales stay randomly initialized.

- [ ] **Step 1: Implement**

(a) argparse, after `--pose-init-seed`:

```python
    p.add_argument("--init-trajectory", default=None,
                    help="trajectory npz (frame_idx, pose_wxyz_xyz) to take "
                         "stage-1 frozen rotations + camera-center inits from, "
                         "instead of the IMU gyro chain + random centers. Used "
                         "by vio_windowed_ba.py's global refine: the stitched "
                         "windowed solution is a near-correct init everywhere, "
                         "where the raw gyro chain drifts over minutes.")
```

(b) Insert directly after the `point_init = point_init.at[:, 2].add(args.init_depth)` line (i.e. after both `rot_init_wxyz` and the random `center_init`/`point_init` exist, before any cost construction):

```python
    if args.init_trajectory:
        init = np.load(args.init_trajectory)
        init_fi = init["frame_idx"]
        pos = np.searchsorted(init_fi, frame_idx)
        if pos.max() >= len(init_fi) or not np.array_equal(init_fi[pos], frame_idx):
            raise ValueError(f"--init-trajectory {args.init_trajectory} does not "
                             f"cover the solve's frames "
                             f"({frame_idx[0]}..{frame_idx[-1]})")
        init_poses = np.asarray(init["pose_wxyz_xyz"])[pos]
        rot_init_wxyz = jnp.asarray(init_poses[:, :4], dtype=jnp.float32)
        R_init = np.asarray(jax.vmap(lambda q: jaxlie.SO3(q).as_matrix())(
            jnp.asarray(init_poses[:, :4])))
        center_init = jnp.asarray(
            -np.einsum("nji,nj->ni", R_init, init_poses[:, 4:]), dtype=jnp.float32)
        print(f"stage-1 rotations + centers initialized from "
              f"{args.init_trajectory} ({len(frame_idx)} frames)")
```

(The gyro-chain/gravity-seed code above it still runs and is simply overridden — harmless, keeps the diff minimal. `searchsorted` + `array_equal` allows the init npz to be a superset — the stitched trajectory covers all surviving frames, and a subrange re-solve for debugging also works.)

- [ ] **Step 2: Regression check — unit tests still pass**

Run: `cd /home/smahapatra/hteng_camera/data_processing/vio && conda run --no-capture-output -n eyeball python -m pytest test_vio_bundle_adjust_window.py test_vio_stitch.py -v`
Expected: 8 passed

- [ ] **Step 3: GPU smoke test — re-solve a window from its own output**

```bash
cd /home/smahapatra/hteng_camera/data_processing/vio
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n eyeball python vio_bundle_adjust.py \
    ../../testimu --start-frame 300 --n-frames 300 \
    --init-trajectory ../../testimu/derived/windows/smoke_300_600.npz \
    --out /tmp/claude-1000845/-home-smahapatra-hteng-camera/87c1bb25-0989-44d9-8102-068a2b853e99/scratchpad/smoke_reinit.npz \
    --loss-plot /tmp/claude-1000845/-home-smahapatra-hteng-camera/87c1bb25-0989-44d9-8102-068a2b853e99/scratchpad/smoke_reinit_loss.png
```

Expected: prints `stage-1 rotations + centers initialized from …`; stage-1 **initial** cost is far below the Task-2 smoke run's initial cost (compare the first `iter 0: cost` lines — good init ⇒ much smaller starting residuals on centers, though landmark terms still start random); final `median angular residual` ≤ the Task-2 smoke value (~same, this is a fixed-point check).

- [ ] **Step 4: Verify error path — wrong trajectory rejected**

```bash
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n eyeball python vio_bundle_adjust.py \
    ../../testimu --start-frame 0 --n-frames 500 \
    --init-trajectory ../../testimu/derived/windows/smoke_300_600.npz \
    --out /tmp/claude-1000845/-home-smahapatra-hteng-camera/87c1bb25-0989-44d9-8102-068a2b853e99/scratchpad/should_fail.npz
```

Expected: `ValueError: --init-trajectory … does not cover the solve's frames (0..500)` (the npz only covers 300..600).

- [ ] **Step 5: Commit**

```bash
cd /home/smahapatra/hteng_camera
git add data_processing/vio/vio_bundle_adjust.py
git commit -m "vio_bundle_adjust: --init-trajectory init override

Take stage-1 frozen rotations + center inits from a prior trajectory npz
(superset frame coverage asserted) instead of the gyro chain + random.
Hook for windowed BA's global refine: full-length solve from the stitched
near-correct init. Landmarks/scales stay randomly initialized (proven to
converge under good rotations)."
```

---

### Task 4: Orchestrator (`vio_windowed_ba.py`) + testimu end-to-end

**Files:**
- Create: `data_processing/vio/vio_windowed_ba.py`
- Test: `data_processing/vio/test_vio_windowed_ba.py`

**Interfaces:**
- Consumes: `vio_stitch` (Task 1, all functions listed there); `vio_bundle_adjust.py --start-frame/--n-frames/--init-trajectory/--tracks/--imu-relative/--out/--loss-plot` (Tasks 2–3).
- Produces: the stage CLI `python vio_windowed_ba.py <recording> --gpus …`; pure helpers `window_ranges(first, last, window, overlap) -> [(s, e)]` and `merge_blend(placed) -> (frame_idx, poses)` where `placed` is a list of `(frame_idx, poses, points, npz)` tuples (only elements 0–1 used by `merge_blend`).
- Outputs on disk: `<recording>/derived/windows/window_<s>_<e>.npz` (+ `.log`, `_loss.png`), `<recording>/derived/trajectory_stitched.npz`, final `<recording>/derived/trajectory.npz`.

- [ ] **Step 1: Write the failing unit tests**

Create `data_processing/vio/test_vio_windowed_ba.py`:

```python
import numpy as np

import vio_stitch as ST
import vio_windowed_ba as WBA


def test_window_ranges_long_test2_shape():
    r = WBA.window_ranges(0, 11042, 900, 300)
    assert r[0][0] == 0 and r[-1][1] == 11042
    for (s0, e0), (s1, e1) in zip(r, r[1:]):
        assert s1 == s0 + 600          # stride = window - overlap
        assert e0 - s1 >= 300          # shared span >= overlap
    lens = [e - s for s, e in r]
    assert lens[:-1] == [900] * (len(r) - 1)
    assert 900 <= lens[-1] <= 900 + 600  # tail folded into final window
    assert len(r) == 17


def test_window_ranges_short_recording_single_window():
    assert WBA.window_ranges(0, 500, 900, 300) == [(0, 500)]


def test_window_ranges_rejects_bad_overlap():
    import pytest
    with pytest.raises(AssertionError):
        WBA.window_ranges(0, 5000, 900, 500)   # overlap > window/2


def _rand_poses(n, seed):
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return np.concatenate([q, rng.normal(size=(n, 3))], axis=1)


def test_merge_blend_agreeing_windows_pass_through():
    fi_a = np.arange(0, 10)
    pa = _rand_poses(10, 0)
    fi_b = np.arange(5, 15)
    pb = np.concatenate([pa[5:], _rand_poses(5, 1)])  # agrees on shared 5..9
    fi, poses = WBA.merge_blend([(fi_a, pa, None, None), (fi_b, pb, None, None)])
    assert np.array_equal(fi, np.arange(15))
    assert np.allclose(ST.quat_to_R(poses[5:10, :4]), ST.quat_to_R(pa[5:10, :4]),
                       atol=1e-9)
    assert np.allclose(ST.centers_of(poses[5:10]), ST.centers_of(pa[5:10]),
                       atol=1e-9)
    assert np.allclose(poses[:5], pa[:5]) and np.allclose(poses[10:], pb[5:])


def test_merge_blend_ramps_between_disagreeing_windows():
    fi_a, fi_b = np.arange(0, 10), np.arange(5, 15)
    pa, pb = _rand_poses(10, 2), _rand_poses(10, 3)
    fi, poses = WBA.merge_blend([(fi_a, pa, None, None), (fi_b, pb, None, None)])
    ca = ST.centers_of(pa[5:10])
    cb = ST.centers_of(pb[:5])
    cm = ST.centers_of(poses[5:10])
    w = np.linspace(0.0, 1.0, 7)[1:-1]
    assert np.allclose(cm, (1 - w[:, None]) * ca + w[:, None] * cb, atol=1e-9)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/smahapatra/hteng_camera/data_processing/vio && conda run --no-capture-output -n eyeball python -m pytest test_vio_windowed_ba.py -v`
Expected: collection error — `ModuleNotFoundError: No module named 'vio_windowed_ba'`

- [ ] **Step 3: Implement `vio_windowed_ba.py`**

Create `data_processing/vio/vio_windowed_ba.py`:

```python
"""VIO stage 5w: windowed bundle adjustment for LONG recordings.

The full-length two-stage solve (vio_bundle_adjust.py) converges on short
spans but plateaus on multi-minute recordings: stage 1 freezes rotations from
the IMU gyro chain seeded at frame 0, and gyro drift over minutes puts that
scaffold too far from truth (long-test2 11k frames: 8-9 deg median residual
vs 0.2-0.6 deg on <=30 s spans). This stage resets the drift by windowing:

  1. solve overlapping ~30 s windows independently (each window's chain
     re-seeds gravity-aligned at its own first frame) -- parallel across GPUs
  2. stitch: consecutive windows share ~overlap solved frames; both gauges
     are gravity-aligned + metric, so they differ by exactly yaw+translation
     (closed form, vio_stitch.py), chained into window-0's world
  3. blend poses across overlaps (center lerp + quat nlerp)
  4. global refine: full-length vio_bundle_adjust.py --init-trajectory
     <stitched>, so stage 1's frozen rotations are near-correct everywhere;
     stage 2 irons out seams. --refine-tracks tracks_loop.jsonl folds in
     loop closure.

Design doc: docs/superpowers/specs/2026-07-12-windowed-ba-design.md

Run (from data_processing/vio/):
    python vio_windowed_ba.py ../../long-test2 --gpus 0 1 2 3 4 5 6 7 \
        --refine-tracks ../../long-test2/derived/tracks_loop.jsonl
"""
import argparse
import os
import subprocess
import sys
import time

import numpy as np

import vio_stitch as ST

BA_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "vio_bundle_adjust.py")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("recording")
    p.add_argument("--tracks", default=None,
                    help="default: <recording>/derived/tracks.jsonl")
    p.add_argument("--imu-relative", default=None,
                    help="default: <recording>/derived/imu_relative.npz")
    p.add_argument("--out", default=None,
                    help="final refined trajectory; default: "
                         "<recording>/derived/trajectory.npz")
    p.add_argument("--gpus", type=int, nargs="+", required=True)
    p.add_argument("--window-frames", type=int, default=900,
                    help="window length in video frames (~30 s @ 30 fps: short "
                         "enough that the per-window gyro-chain init is accurate)")
    p.add_argument("--overlap-frames", type=int, default=300,
                    help="shared frames between consecutive windows (the stitch "
                         "estimates 4 DOF from these; must be <= window/2)")
    p.add_argument("--min-overlap", type=int, default=30,
                    help="min shared SOLVED frames per seam; fail loudly below")
    p.add_argument("--no-refine", action="store_true",
                    help="stop after stitching (trajectory_stitched.npz only; "
                         "does NOT write --out)")
    p.add_argument("--refine-tracks", default=None,
                    help="tracks file for the global refine, e.g. "
                         "tracks_loop.jsonl to fold in loop closure; "
                         "default: same as --tracks")
    p.add_argument("--resume", action="store_true",
                    help="skip window solves whose output npz already exists")
    return p.parse_args()


def window_ranges(first, last, window, overlap):
    """Inclusive [s, e] video-frame windows covering [first, last]: stride =
    window - overlap; the tail (< stride frames) folds into the final window
    rather than becoming a runt. overlap <= window/2 guarantees only
    CONSECUTIVE windows share frames (blend + stitch assume pairwise seams)."""
    assert window > overlap >= 0, "need window > overlap >= 0"
    assert overlap * 2 <= window, "overlap > window/2 would triple-overlap frames"
    if last - first <= window:
        return [(first, last)]
    stride = window - overlap
    starts = list(range(first, last - window + 1, stride))
    ranges = [[s, s + window] for s in starts]
    ranges[-1][1] = last
    return [tuple(r) for r in ranges]


def solve_windows(args, ranges, win_dir, tracks, imu_rel):
    """Fan window solves out across GPUs, one subprocess per window, one
    window per GPU at a time. Blocks until all succeed; raises on any rc!=0."""
    os.makedirs(win_dir, exist_ok=True)
    jobs = []
    for s, e in ranges:
        out = os.path.join(win_dir, f"window_{s}_{e}.npz")
        if args.resume and os.path.exists(out):
            print(f"[resume] window {s}-{e} exists, skipping", flush=True)
            continue
        jobs.append((s, e, out))
    free = list(args.gpus)
    running = []  # (proc, gpu, s, e, log_path)
    while jobs or running:
        while jobs and free:
            s, e, out = jobs.pop(0)
            g = free.pop(0)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g),
                       XLA_PYTHON_CLIENT_PREALLOCATE="false")
            cmd = [sys.executable, BA_SCRIPT, args.recording,
                   "--tracks", tracks, "--imu-relative", imu_rel,
                   "--start-frame", str(s), "--n-frames", str(e - s),
                   "--out", out, "--loss-plot", out[:-4] + "_loss.png"]
            log_path = out[:-4] + ".log"
            log_f = open(log_path, "w")
            print(f"[gpu {g}] window {s}-{e} -> {out}", flush=True)
            proc = subprocess.Popen(cmd, env=env, stdout=log_f,
                                    stderr=subprocess.STDOUT)
            log_f.close()  # child holds the fd
            running.append((proc, g, s, e, log_path))
        still = []
        for proc, g, s, e, log_path in running:
            if proc.poll() is None:
                still.append((proc, g, s, e, log_path))
                continue
            free.append(g)
            if proc.returncode != 0:
                raise RuntimeError(f"window {s}-{e} failed "
                                   f"(rc {proc.returncode}), see {log_path}")
            print(f"[gpu {g}] window {s}-{e} done", flush=True)
        running = still
        if running:
            time.sleep(3)


def load_and_stitch(ranges, win_dir, min_overlap):
    """Load window npzs, chain 4-DOF seam alignments into window-0's world.
    Returns placed = [(frame_idx, poses_world0 (N,7) f64, points_world0, npz)]."""
    wins = []
    for s, e in ranges:
        d = np.load(os.path.join(win_dir, f"window_{s}_{e}.npz"))
        med = float(np.nanmedian(d["point_med_ang"][d["point_alive"]]))
        flag = "  <-- WARNING: > 1 deg, check this window" if med > 1.0 else ""
        print(f"window {s}-{e}: {len(d['frame_idx'])} frames, "
              f"median residual {med:.3f} deg{flag}", flush=True)
        wins.append(d)
    theta_cum, t_cum = 0.0, np.zeros(3)
    placed = []
    for k, win in enumerate(wins):
        if k > 0:
            prev = wins[k - 1]
            shared, ia, ib = np.intersect1d(prev["frame_idx"], win["frame_idx"],
                                             return_indices=True)
            if len(shared) < min_overlap:
                raise RuntimeError(f"seam {k}: only {len(shared)} shared frames "
                                   f"(< {min_overlap}) -- windows too disjoint")
            th, tt, diag = ST.fit_yaw_translation(
                np.asarray(prev["pose_wxyz_xyz"], np.float64)[ia],
                np.asarray(win["pose_wxyz_xyz"], np.float64)[ib])
            print(f"seam {k}: yaw {diag['yaw_deg']:+8.3f} deg "
                  f"(vote spread {diag['yaw_spread_deg']:.3f} deg), "
                  f"center rms {diag['center_rms_m'] * 100:.2f} cm, "
                  f"{diag['n_shared']} shared frames", flush=True)
            theta_cum, t_cum = ST.compose_yaw_translation(theta_cum, t_cum, th, tt)
        poses = ST.apply_yaw_translation(
            np.asarray(win["pose_wxyz_xyz"], np.float64), theta_cum, t_cum)
        points = np.asarray(win["points"], np.float64) @ ST.yaw_R(theta_cum).T + t_cum
        placed.append((np.asarray(win["frame_idx"]), poses, points, win))
    return placed


def merge_blend(placed):
    """Merge placed windows into one pose-per-frame trajectory; overlap frames
    blend with a linear ramp (0 -> earlier window, 1 -> later window)."""
    fi = placed[0][0].copy()
    poses = placed[0][1].copy()
    for k in range(1, len(placed)):
        fi_b, pb = placed[k][0], placed[k][1]
        shared, ia, ib = np.intersect1d(fi, fi_b, return_indices=True)
        w = np.linspace(0.0, 1.0, len(shared) + 2)[1:-1]
        poses[ia] = ST.blend_poses(poses[ia], pb[ib], w)
        new = ~np.isin(fi_b, shared)
        fi = np.concatenate([fi, fi_b[new]])
        poses = np.concatenate([poses, pb[new]])
        order = np.argsort(fi)
        fi, poses = fi[order], poses[order]
    return fi, poses


def save_stitched(path, fi, poses, placed):
    """trajectory.npz-compatible stitched output. Landmarks are the per-window
    clouds transformed to world-0, concatenated (overlap duplicates are fine
    for visualization; the refine re-solves landmarks from scratch anyway)."""
    np.savez(
        path, frame_idx=fi, pose_wxyz_xyz=poses,
        points=np.concatenate([p[2] for p in placed]),
        point_first_frame=np.concatenate([p[3]["point_first_frame"] for p in placed]),
        point_first_is_right=np.concatenate([p[3]["point_first_is_right"] for p in placed]),
        point_first_px=np.concatenate([p[3]["point_first_px"] for p in placed]),
        point_alive=np.concatenate([p[3]["point_alive"] for p in placed]),
        point_med_ang=np.concatenate([p[3]["point_med_ang"] for p in placed]),
        cost_history=np.zeros(1))


def main():
    args = parse_args()
    rec = args.recording
    tracks = args.tracks or os.path.join(rec, "derived", "tracks.jsonl")
    imu_rel = args.imu_relative or os.path.join(rec, "derived", "imu_relative.npz")
    out = args.out or os.path.join(rec, "derived", "trajectory.npz")
    win_dir = os.path.join(rec, "derived", "windows")

    imu = np.load(imu_rel)
    first, last = int(imu["frame_idx"][0]), int(imu["frame_idx"][-1])
    ranges = window_ranges(first, last, args.window_frames, args.overlap_frames)
    print(f"{len(ranges)} windows over frames {first}..{last} "
          f"(window {args.window_frames}, overlap {args.overlap_frames}, "
          f"gpus {args.gpus})", flush=True)

    solve_windows(args, ranges, win_dir, tracks, imu_rel)
    placed = load_and_stitch(ranges, win_dir, args.min_overlap)
    fi, poses = merge_blend(placed)
    stitched = os.path.join(rec, "derived", "trajectory_stitched.npz")
    save_stitched(stitched, fi, poses, placed)
    print(f"wrote {stitched} ({len(fi)} poses)", flush=True)

    if args.no_refine:
        print("--no-refine: stopping after stitch (final refine skipped)")
        return
    print("=== global refine from stitched init ===", flush=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpus[0]),
               XLA_PYTHON_CLIENT_PREALLOCATE="false")
    subprocess.run([sys.executable, BA_SCRIPT, rec,
                    "--tracks", args.refine_tracks or tracks,
                    "--imu-relative", imu_rel,
                    "--init-trajectory", stitched,
                    "--out", out], env=env, check=True)
    print(f"final trajectory: {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run unit tests to verify they pass**

Run: `cd /home/smahapatra/hteng_camera/data_processing/vio && conda run --no-capture-output -n eyeball python -m pytest test_vio_windowed_ba.py test_vio_stitch.py test_vio_bundle_adjust_window.py -v`
Expected: 13 passed

- [ ] **Step 5: End-to-end integration on testimu (2 windows, 2 GPUs)**

```bash
cd /home/smahapatra/hteng_camera/data_processing/vio
conda run --no-capture-output -n eyeball python vio_windowed_ba.py ../../testimu \
    --gpus 0 1 --window-frames 400 --overlap-frames 150
```

Expected console flow:
- `2 windows over frames 0..896` → windows `(0,400)` and `(250,896)` (tail folded)
- both window solves finish, each `median residual` ≤ ~0.5 deg, no WARNING flag
- `seam 1: yaw … (vote spread < 0.5 deg), center rms < 1.00 cm, ~151 shared frames`
- `wrote ../../testimu/derived/trajectory_stitched.npz (~897 poses)`
- global refine runs full-length and reports `median angular residual` ≈ **0.23 deg** (jkerr's full-solve reference — testimu converges even unwindowed, so this validates the machinery end-to-end, not the drift fix itself)

- [ ] **Step 6: Compare final trajectory against jkerr's reference**

```bash
cd /home/smahapatra/hteng_camera/data_processing/vio
conda run --no-capture-output -n eyeball python - <<'EOF'
import numpy as np
import vio_stitch as ST

ours = np.load("../../testimu/derived/trajectory.npz")
ref = np.load("/home/jkerr/hteng_camera/testimu/trajectory.npz")
shared, ia, ib = np.intersect1d(ours["frame_idx"], ref["frame_idx"],
                                 return_indices=True)
th, t, diag = ST.fit_yaw_translation(
    np.asarray(ref["pose_wxyz_xyz"], np.float64)[ib],
    np.asarray(ours["pose_wxyz_xyz"], np.float64)[ia])
aligned = ST.apply_yaw_translation(
    np.asarray(ours["pose_wxyz_xyz"], np.float64)[ia], th, t)
dc = np.linalg.norm(ST.centers_of(aligned)
                    - ST.centers_of(np.asarray(ref["pose_wxyz_xyz"], np.float64)[ib]),
                    axis=1)
med = np.nanmedian(ours["point_med_ang"][ours["point_alive"]])
print(f"median center diff vs reference: {np.median(dc)*100:.2f} cm "
      f"(p95 {np.percentile(dc,95)*100:.2f} cm)")
print(f"our median angular residual: {med:.3f} deg (reference ~0.23)")
EOF
```

Expected: median center diff **< ~5 cm** (context: jkerr measured loop-vs-noloop full solves differing by median 2.3 cm — same-ballpark gauge/solver noise is a pass), residual ≤ ~0.3 deg. If wildly off (> 20 cm / > 1 deg), STOP and debug the stitch/refine before Task 5.

- [ ] **Step 7: Commit**

```bash
cd /home/smahapatra/hteng_camera
git add data_processing/vio/vio_windowed_ba.py data_processing/vio/test_vio_windowed_ba.py
git commit -m "vio: windowed BA stage (solve windows -> 4-DOF stitch -> refine)

vio_windowed_ba.py: overlapping ~30s windows solved in parallel across
GPUs (per-window gyro-chain init stays drift-free), closed-form
yaw+translation seam alignment chained into window-0's world, overlap
blending, then a full-length re-solve from the stitched init
(--init-trajectory). Validated end-to-end on testimu against jkerr's
converged reference."
```

---

### Task 5: long-test2 production run, dense fusion, docs

The actual payoff: converged full-length long-test2 poses → dense world.

**Files:**
- Modify: `data_processing/vio/CLAUDE.md` (stage table + status), `data_processing/CLAUDE.md` (branch notes)
- Outputs (gitignored): `long-test2/derived/{windows/, trajectory_stitched.npz, trajectory.npz}`, `data_processing/out/lt2_world_windowed.ply`

**Interfaces:**
- Consumes: the full stage from Task 4; existing range maps `data_processing/out/lt2_video` (meta.json + range stacks); `data_processing/ffs_fuse_world.py`; `data_processing/ffs_scene_viser.py`.

- [ ] **Step 1: Back up the old (blob) trajectory, then run the stage on all 8 GPUs**

```bash
cd /home/smahapatra/hteng_camera
[ -f long-test2/derived/trajectory.npz ] && \
    cp long-test2/derived/trajectory.npz long-test2/derived/trajectory_fullba_8deg_backup.npz
cd data_processing/vio
conda run --no-capture-output -n eyeball python vio_windowed_ba.py ../../long-test2 \
    --gpus 0 1 2 3 4 5 6 7 \
    --refine-tracks ../../long-test2/derived/tracks_loop.jsonl \
    2>&1 | tee ../../long-test2/derived/windowed_ba_run.log
```

Expected: `17 windows over frames 0..11042`; 3 waves across 8 GPUs; **every window's median residual in ~0.2–0.8 deg** (this is the drift-fix moment — the same solver full-length gave 8–9 deg); 16 seams each with sub-degree vote spread and ≲ few-cm center rms; then the full-length refine (stage 1 ~15 iters is minutes; stage 2 is the slow part — the full solve ran before, so it fits). Final `median angular residual` ≪ 8 deg — **target < 1.5 deg** (seams + long-range drift make it looser than a single window; anything ≲ 1 deg is excellent).

Decision point: if the refine plateaus > 2 deg, the stitched trajectory itself (`trajectory_stitched.npz`) is still likely good — fuse from it instead (Step 2 with `--trajectory long-test2/derived/trajectory_stitched.npz`) and report both numbers before tuning further.

- [ ] **Step 2: Dense world fusion with the new poses**

```bash
cd /home/smahapatra/hteng_camera
conda run --no-capture-output -n eyeball python data_processing/ffs_fuse_world.py \
    --range-dir data_processing/out/lt2_video \
    --trajectory long-test2/derived/trajectory.npz \
    --out data_processing/out/lt2_world_windowed.ply \
    --frame-stride 10 --max-range 3.0 --voxel 0.01 --clean
```

Expected: accumulates ~1000 frames' clouds; after voxel + clean, several million points; no "smeared blob" (the long-test2 failure mode with the 8–9 deg poses).

- [ ] **Step 3: Visual check in viser**

```bash
conda run --no-capture-output -n eyeball python data_processing/ffs_scene_viser.py \
    --ply data_processing/out/lt2_world_windowed.ply --up +z --port 8090 --share
```

Acceptance (the spec's criterion 4): recognizable room/scene geometry — walls, furniture, coherent surfaces — comparable in quality to the testimu dense world. Share the URL with the user; leave the server running for them.

- [ ] **Step 4: Update docs**

`data_processing/vio/CLAUDE.md`: add a stage-table row and a short section:

```markdown
| 5w. Windowed BA (long recordings) | `vio_windowed_ba.py` | `tracks.jsonl`, `imu_relative.npz` | `derived/windows/*.npz`, `trajectory_stitched.npz`, `trajectory.npz` | For recordings where the full-length solve plateaus (gyro-chain init drift over minutes; long-test2: 8-9 deg). Solves overlapping ~30 s windows in parallel across GPUs (`--gpus`), aligns them with a closed-form 4-DOF stitch (`vio_stitch.py`: gravity fixes roll/pitch, baseline fixes scale -> yaw+translation per seam, circular-mean yaw votes + center-mean t), blends overlaps, then re-runs the full solve from the stitched init (`vio_bundle_adjust.py --init-trajectory`). `--refine-tracks tracks_loop.jsonl` folds loop closure into the refine. `--no-refine` stops after stitching. Seam diagnostics (yaw vote spread, center rms) printed per seam; per-window residuals flagged > 1 deg. |
```

and in the Status section append: windowed BA validated on testimu (matches the reference full solve) and long-test2 (fill in the measured final residual vs the 8–9 deg baseline).

`data_processing/CLAUDE.md`: append one paragraph to the `depth_understanding` branch notes: long-test2 VIO convergence fixed via `vio/vio_windowed_ba.py` (windowed BA + 4-DOF stitch + `--init-trajectory` refine); dense fusion now works on long-test2 (fill in the measured numbers).

- [ ] **Step 5: Final commit**

```bash
cd /home/smahapatra/hteng_camera
git add data_processing/vio/CLAUDE.md data_processing/CLAUDE.md
git commit -m "vio/CLAUDE.md: document windowed BA stage + long-test2 results"
```

Then report to the user: per-window residuals, seam diagnostics, final full-length residual vs the 8–9° baseline, and the viser link to the dense long-test2 world. Offer (do not start) the follow-ups: hands overlay on the dense world, meshing, pushing the branch.

---

## Self-Review Notes

- **Spec coverage:** windowing/per-window solve → Tasks 2+4; 4-DOF stitch + blending + seam diagnostics → Tasks 1+4; `--init-trajectory` global refine + loop-closure composition → Tasks 3+5; outputs/schema compatibility → Task 4 (`save_stitched`, refine writes standard npz); acceptance criteria 1–3 → Task 4 steps 5–6 and Task 5 step 1; criterion 4 (dense fusion) → Task 5 steps 2–3; edge cases (invalid frames via `frame_idx` intersection, `--min-overlap` assert, per-window residual flag, `XLA_PYTHON_CLIENT_PREALLOCATE`) → Tasks 2–4. One deliberate deviation from the spec: `--init-trajectory` accepts a frame *superset* (searchsorted + array_equal) rather than demanding exact equality — strictly more general, needed for subrange debugging, noted in Task 3.
- **Type consistency:** poses are (N,7) wxyz_xyz float64 throughout stitch code (windows load as float32 → cast at use); `placed` tuple shape `(fi, poses, points, npz)` consistent between `load_and_stitch`, `merge_blend` (uses [0],[1]), `save_stitched` (uses [2],[3]); `window_ranges` returns list of int 2-tuples everywhere.
- **Numbers checked:** long-test2 first/last = 0/11042 → `range(0, 10143, 600)` = 17 starts, last window (9600, 11042), length 1442 ≤ 1500 ✓; testimu 0..896 with 400/150 → starts [0, 250] → 2 windows, seam ≈ 151 shared frames ✓.
