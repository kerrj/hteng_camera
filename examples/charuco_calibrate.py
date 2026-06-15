"""ChArUco calibration in viser — mono intrinsics *and* stereo cam2cam.

Open one or two HTENG cameras. A ChArUco board (rendered to a window here, or a
physical board you move around) is detected live; pose-diverse views are captured
automatically. Then:

  * per-camera **intrinsics** (fisheye Kannala-Brandt or pinhole Brown-Conrady),
  * the **cam2cam pose** (R, t with ``X_right = R @ X_left + t``).

Stereo extrinsics are solved *without* cv2.stereoCalibrate (its fisheye path is
broken in 4.x). Instead: PnP gives the board pose in each camera per synchronized
frame; the board corners are expressed in both camera frames and a single rigid
alignment over *all* common corners across *all* frames (Kabsch/Umeyama, closed
form) gives R, t. A scipy bundle-adjustment pass then refines (R, t) jointly with
the per-frame board poses by minimizing reprojection error in both cameras.

Sync without a hardware trigger: a stereo pair is captured only while the board is
held **steady** (the left-camera pose is stable frame-to-frame) and the pose is
novel. Holding still means both cameras see the same board position regardless of
small frame-time offsets — the natural "move, pause, capture" workflow, and it
works the same whether the board moves or the cameras do.

Metric note: ``t``/baseline are in the board's units, so ``--square-mm`` must be
the *true* square size (measure it) for real millimetres. Intrinsics don't depend
on scale; the cam2cam translation does.

Run::

    pip install hteng-camera[viser] scipy
    python examples/charuco_calibrate.py --square-mm 30.0

Open the URL viser prints. Drag the board window next to the browser. Use each
camera's "Close & release" before quitting.
"""

import argparse
import glob
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import viser
from concurrent.futures import ThreadPoolExecutor

from hteng_camera import (
    CameraCalibration, HTCamera, Intrinsics, StereoCalibration,
    convert, enums, list_cameras,
)

try:
    from scipy.optimize import least_squares
    from scipy.sparse import lil_matrix
    _HAVE_SCIPY = True
except ImportError:  # BA refinement is skipped; Kabsch closed form still runs
    _HAVE_SCIPY = False

JPEG_QUALITY = 85
SIDE_W = 760            # per-camera tile width in the side-by-side preview
SIDES = ("L", "R")

# Preview / detector tone curve: BT.709 standard SDR gamma, matching record.py's
# default master (bt709 takes no param). OpenCV's marker detection thresholds
# locally, so the milder shadow-lift vs the old log/120 doesn't hurt detection on
# a well-lit board; switch back to dict(curve="log", param=120.0) for very dark
# or high-dynamic-range capture conditions.
TONE = dict(curve="bt709")

# Board "held steady" thresholds — below these frame-to-frame pose deltas the
# board counts as stationary, so a stereo pair is safe to capture (both cameras
# see the same board position regardless of small capture-time offsets).
STILL_TRANS_M = 0.004
STILL_ROT_DEG = 0.7

# Consecutive empty grabs (each ~300ms) before a camera is declared lost and its
# handle released — long enough to ride out transient USB hiccups (~3s).
LOST_AFTER = 10

DICTS = {
    "4X4_50": cv2.aruco.DICT_4X4_50,
    "4X4_100": cv2.aruco.DICT_4X4_100,
    "5X5_100": cv2.aruco.DICT_5X5_100,
    "5X5_250": cv2.aruco.DICT_5X5_250,
}


# --------------------------------------------------------------------------- #
# Pure geometry helpers (independently testable; no camera/GUI state).
# Undistort-map building lives on hteng_camera.Intrinsics.undistort_maps.
# --------------------------------------------------------------------------- #
def corner_sharpness(gray, corners, patch=4, max_corners=40):
    """Blur metric: median normalized gradient steepness at charuco corners.

    At each detected corner, take a (2*patch+1)^2 window and compute
    max|Sobel| / (4 * intensity range) — for an ideal step edge this is ~1.0
    and it falls off roughly with 1/blur-radius, so motion blur (or defocus)
    drags it down fast. Contrast normalization makes it exposure-invariant.
    Median over corners resists a few outliers (e.g. corners on the board edge).

    Rough scale: >0.5 crisp, ~0.35 slightly soft, <0.25 visibly blurred.
    """
    h, w = gray.shape[:2]
    pts = corners.reshape(-1, 2)
    if len(pts) > max_corners:
        pts = pts[:: len(pts) // max_corners + 1]
    vals = []
    for x, y in pts:
        x, y = int(round(x)), int(round(y))
        x0, x1 = max(0, x - patch), min(w, x + patch + 1)
        y0, y1 = max(0, y - patch), min(h, y + patch + 1)
        win = gray[y0:y1, x0:x1].astype(np.float32)
        rng = float(win.max() - win.min())
        if rng < 8.0:        # featureless patch — no edge to measure
            continue
        gx = cv2.Sobel(win, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(win, cv2.CV_32F, 0, 1, ksize=3)
        grad = float(np.sqrt(gx * gx + gy * gy).max())
        vals.append(grad / (4.0 * rng))
    return float(np.median(vals)) if vals else 0.0


def rotation_angle_deg(rvec_a, rvec_b):
    """Geodesic angle (degrees) between two rotations given as Rodrigues vectors."""
    Ra, _ = cv2.Rodrigues(rvec_a)
    Rb, _ = cv2.Rodrigues(rvec_b)
    R = Ra @ Rb.T
    cos = (np.trace(R) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def is_novel(rvec, tvec, kept, min_trans_m, min_rot_deg):
    """True if (rvec, tvec) differs from *every* kept pose by >trans OR >rot.

    Returns (novel, nearest_trans_cm, nearest_rot_deg). Redundant only when some
    kept view is within BOTH thresholds.
    """
    nearest_t = nearest_r = float("inf")
    novel = True
    for r_k, t_k in kept:
        dt = float(np.linalg.norm(tvec - t_k))
        dr = rotation_angle_deg(rvec, r_k)
        nearest_t = min(nearest_t, dt)
        nearest_r = min(nearest_r, dr)
        if dt < min_trans_m and dr < min_rot_deg:
            novel = False
    return novel, nearest_t * 100.0, nearest_r


def guessed_K(w, h):
    """Rough pinhole guess for pose-gating only (focal ~ image width)."""
    f = float(w)
    return np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1.0]], np.float64)


def kabsch(A, B):
    """Closed-form rigid transform with B ≈ R @ A + t. A, B are (N, 3)."""
    muA, muB = A.mean(0), B.mean(0)
    H = (A - muA).T @ (B - muB)
    U, _S, Vt = np.linalg.svd(H)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, muB - R @ muA


def calibrate_fisheye(obj_pts, img_pts, w, h):
    """Kannala-Brandt intrinsic fit. Returns (rms, K, D, used_indices).

    A single ill-conditioned view aborts fisheye.calibrate; on failure we drop
    the offending view (index parsed from the error, else the last) and retry.
    """
    flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_FIX_SKEW
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
    obj = [o.reshape(-1, 1, 3).astype(np.float64) for o in obj_pts]
    img = [i.reshape(-1, 1, 2).astype(np.float64) for i in img_pts]
    idx = list(range(len(obj)))
    while len(obj) >= 6:
        K = np.zeros((3, 3))
        D = np.zeros((4, 1))
        try:
            rms, K, D, _r, _t = cv2.fisheye.calibrate(
                obj, img, (w, h), K, D, flags=flags, criteria=crit)
            return rms, K, D, idx
        except cv2.error as exc:
            drop = len(obj) - 1
            for tok in str(exc).replace(".", " ").split():
                if tok.isdigit() and int(tok) < len(obj):
                    drop = int(tok)
            obj.pop(drop); img.pop(drop); idx.pop(drop)
    raise RuntimeError("fisheye solve failed: too few well-conditioned views")


def pnp_pose(objp, imgp, intr):
    """Board pose (rvec, tvec) in a camera, honoring its distortion model.

    ``intr`` is an :class:`hteng_camera.Intrinsics`.
    """
    objp = objp.reshape(-1, 1, 3).astype(np.float64)
    imgp = imgp.reshape(-1, 1, 2).astype(np.float64)
    if intr.model == "fisheye":
        # No cv2.fisheye.solvePnP — undistort to pinhole-K pixels, then solvePnP.
        und = cv2.fisheye.undistortPoints(imgp, intr.K, intr.dist.reshape(4, 1),
                                          P=intr.K)
        return cv2.solvePnP(objp, und, intr.K, None, flags=cv2.SOLVEPNP_ITERATIVE)
    return cv2.solvePnP(objp, imgp, intr.K, intr.dist,
                        flags=cv2.SOLVEPNP_ITERATIVE)


def project_pts(objp, rvec, tvec, intr):
    """Project board points into a camera (model-aware). Returns (N, 2)."""
    objp = objp.reshape(-1, 1, 3).astype(np.float64)
    rvec = np.asarray(rvec, np.float64).reshape(3, 1)
    tvec = np.asarray(tvec, np.float64).reshape(3, 1)
    if intr.model == "fisheye":
        p, _ = cv2.fisheye.projectPoints(objp, rvec, tvec, intr.K,
                                         intr.dist.reshape(4, 1))
    else:
        p, _ = cv2.projectPoints(objp, rvec, tvec, intr.K, intr.dist)
    return p.reshape(-1, 2)


def solve_cam2cam(pairs, intrL, intrR):
    """Cam2cam (R, t) from synchronized board views. X_right = R @ X_left + t.

    ``pairs`` is a list of (objp, imgL, imgR) for common corners. Per pair, PnP
    gives the board pose in each camera; the corners are expressed in both frames
    and one Kabsch alignment over all of them gives a closed-form (R, t). If scipy
    is present, a bundle-adjustment pass refines (R, t) + per-frame board poses by
    minimizing reprojection error in both cameras, and is kept only if it lowers
    the reprojection RMS.
    """
    A, B, frames = [], [], []
    for objp, imgL, imgR in pairs:
        okL, rL, tL = pnp_pose(objp, imgL, intrL)
        okR, rR, tR = pnp_pose(objp, imgR, intrR)
        if not (okL and okR):
            continue
        op = objp.reshape(-1, 3).astype(np.float64)
        A.append((cv2.Rodrigues(rL)[0] @ op.T).T + tL.ravel())
        B.append((cv2.Rodrigues(rR)[0] @ op.T).T + tR.ravel())
        frames.append(dict(objp=op, imgL=imgL.reshape(-1, 2).astype(np.float64),
                           imgR=imgR.reshape(-1, 2).astype(np.float64),
                           rL=rL.ravel(), tL=tL.ravel()))
    if len(frames) < 3:
        raise RuntimeError(f"need >=3 usable stereo pairs, have {len(frames)}")

    R0, t0 = kabsch(np.vstack(A), np.vstack(B))

    # Residual layout: per frame, left reproj then right reproj (variable counts).
    M = len(frames)
    layout, off = [], 0
    for f in frames:
        n = len(f["objp"])
        layout.append((off, off + 2 * n, off + 2 * n, off + 4 * n))
        off += 4 * n
    total = off

    def residuals(x):
        Rc = cv2.Rodrigues(x[0:3])[0]
        tc = x[3:6]
        out = np.empty(total)
        for i, f in enumerate(frames):
            rb = x[6 + 6 * i:6 + 6 * i + 3]
            tb = x[6 + 6 * i + 3:6 + 6 * i + 6]
            a, b, c, dd = layout[i]
            out[a:b] = (project_pts(f["objp"], rb, tb, intrL) - f["imgL"]).ravel()
            Rr = Rc @ cv2.Rodrigues(rb)[0]
            tr = Rc @ tb + tc
            out[c:dd] = (project_pts(f["objp"], cv2.Rodrigues(Rr)[0], tr, intrR)
                         - f["imgR"]).ravel()
        return out

    x0 = np.concatenate([cv2.Rodrigues(R0)[0].ravel(), t0]
                        + [np.concatenate([f["rL"], f["tL"]]) for f in frames])
    kabsch_rms = float(np.sqrt((residuals(x0) ** 2).mean()))
    R, t, rms, method = R0, t0, kabsch_rms, "kabsch"

    if _HAVE_SCIPY:
        S = lil_matrix((total, 6 + 6 * M), dtype=int)
        for i in range(M):
            a, b, c, dd = layout[i]
            S[a:b, 6 + 6 * i:6 + 6 * i + 6] = 1   # left block: board pose only
            S[c:dd, 0:6] = 1                       # right block: cam2cam ...
            S[c:dd, 6 + 6 * i:6 + 6 * i + 6] = 1   # ... and board pose
        sol = least_squares(residuals, x0, jac_sparsity=S, method="trf",
                            x_scale="jac", xtol=1e-10, ftol=1e-10, max_nfev=100)
        ba_rms = float(np.sqrt((sol.fun ** 2).mean()))
        if ba_rms <= kabsch_rms:
            R = cv2.Rodrigues(sol.x[0:3])[0]
            t = sol.x[3:6]
            rms, method = ba_rms, "kabsch+BA"

    return dict(R=R, t=t, reproj_rms=rms, kabsch_rms=kabsch_rms,
                baseline=float(np.linalg.norm(t)),
                rot_deg=rotation_angle_deg(cv2.Rodrigues(R)[0], np.zeros(3)),
                num_pairs=M, method=method)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--squares-x", type=int, default=7, help="board columns")
    ap.add_argument("--squares-y", type=int, default=5, help="board rows")
    ap.add_argument("--square-mm", type=float, default=30.0,
                    help="TRUE square edge (displayed or printed) — measure it!")
    ap.add_argument("--marker-ratio", type=float, default=0.75,
                    help="aruco marker size as a fraction of the square")
    ap.add_argument("--dict", default="5X5_100", choices=list(DICTS))
    ap.add_argument("--square-px", type=int, default=160,
                    help="pixels per square in the rendered board")
    ap.add_argument("--no-board-window", action="store_true",
                    help="don't render a board (using a physical board instead)")
    ap.add_argument("--min-sharpness", type=float, default=0.35,
                    help="reject views blurrier than this (normalized corner "
                         "gradient: >0.5 crisp, <0.25 visibly blurred; 0 = off)")
    ap.add_argument("--no-save-images", action="store_true",
                    help="don't write captured views as 16-bit PNGs to the "
                         "session folder")
    ap.add_argument("--calib-dir", default="calibrations",
                    help="where to publish calib_<serial>.json (git-tracked, "
                         "image-free; the library loads from here)")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()

    square_len = args.square_mm / 1000.0
    marker_len = square_len * args.marker_ratio
    dictionary = cv2.aruco.getPredefinedDictionary(DICTS[args.dict])
    board = cv2.aruco.CharucoBoard(
        (args.squares_x, args.squares_y), square_len, marker_len, dictionary)
    board.setLegacyPattern(False)
    detector = cv2.aruco.CharucoDetector(board)
    board_obj = board.getChessboardCorners().astype(np.float64)  # (Ncorners, 3)
    n_board_corners = (args.squares_x - 1) * (args.squares_y - 1)
    min_corners = max(6, n_board_corners // 2)

    if not args.no_board_window:
        bw = args.squares_x * args.square_px
        bh = args.squares_y * args.square_px
        board_img = board.generateImage((bw, bh), marginSize=args.square_px // 4)
        WIN = "ChArUco board (aim cameras here)"
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN, bw, bh)
        cv2.moveWindow(WIN, 40, 40)
        cv2.imshow(WIN, board_img)
        cv2.waitKey(1)

    server = viser.ViserServer(port=args.port)

    # Per-side mutable state. intr_cap = accumulated intrinsic views; intr = the
    # last calibration result for that side (an Intrinsics).
    state = {
        "cams": {"L": None, "R": None},
        "serials": {"L": None, "R": None},      # kept after close, for saving
        "intr_cap": {s: {"obj": [], "img": [], "poses": [], "lins": []}
                     for s in SIDES},
        "image_size": {"L": None, "R": None},
        "intr": {"L": None, "R": None},
        "prev_pose": {"L": None, "R": None},   # last frame's pose, for stillness
        "stereo_pairs": [],                     # (objp, imgL, imgR)
        "stereo_poses": [],                     # left rvec/tvec, for novelty
        "stereo_lins": [],                      # (linL, linR) raw frames
        "png_written": {"L": 0, "R": 0, "stereo": 0},  # already-on-disk counts
        "stereo": None,                         # cam2cam result
        "do_intr": False, "do_stereo": False, "force": False,
        "file_calib": {}, "umaps": {"L": {}, "R": {}},
        "fail": {"L": 0, "R": 0},   # consecutive empty grabs, for drop detection
        "session_dir": None,        # capture session folder, created lazily
        "quit": False,
    }

    # -- Capture session folder, under the calibration dir. Captured views are
    # held in memory (a 5 MP uint16 frame is ~30 MB; a typical 20-40 view
    # session fits fine) and written as lossless 16-bit linear PNGs only when a
    # Calibrate button runs — PNG encoding is a few hundred ms each and would
    # stall the live loop. The PNGs are gitignored (provenance, not deliverable);
    # the published JSONs in the parent calibrations/ dir are what's tracked.
    def session_dir():
        if state["session_dir"] is None:
            d = (Path(args.calib_dir)
                 / f"calib_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            d.mkdir(parents=True, exist_ok=True)
            state["session_dir"] = d
            print(f"[session] writing capture images into {d}/")
        return state["session_dir"]

    def save_capture_png(name, lin):
        """Write a linear uint16 frame as a 16-bit PNG (<<4 to fill the range)."""
        cv2.imwrite(str(session_dir() / f"{name}_lin16.png"),
                    cv2.cvtColor(lin << 4, cv2.COLOR_RGB2BGR))

    def flush_capture_pngs():
        """Write captured views not yet on disk (incremental, idempotent)."""
        if args.no_save_images:
            return
        tasks = []
        for s in SIDES:
            lins = state["intr_cap"][s]["lins"]
            serial = state["serials"][s]
            for i in range(state["png_written"][s], len(lins)):
                tasks.append((f"{s}_{serial}_view{i + 1:03d}", lins[i]))
            state["png_written"][s] = len(lins)
        for i in range(state["png_written"]["stereo"], len(state["stereo_lins"])):
            for s, lin in zip(SIDES, state["stereo_lins"][i]):
                tasks.append((f"stereo{i + 1:03d}_{s}_{state['serials'][s]}", lin))
        state["png_written"]["stereo"] = len(state["stereo_lins"])
        if not tasks:
            return
        with ThreadPoolExecutor() as pool:
            futs = [pool.submit(save_capture_png, name, lin) for name, lin in tasks]
            for f in futs:
                f.result()
        print(f"[session] wrote {len(tasks)} capture PNG(s) to {session_dir()}/")

    cams = list_cameras()
    cam_opts = [f'{c["serial"]}  ({c["name"]})' for c in cams] or ["<none found>"]
    serial_by_label = {opt: c["serial"] for opt, c in zip(cam_opts, cams)}

    # -- Per-camera connection --------------------------------------------------
    conn_text = {}
    cam_dd = {}
    for side, default_idx in (("L", 0), ("R", min(1, len(cam_opts) - 1))):
        with server.gui.add_folder(f"Camera {side}"):
            cam_dd[side] = server.gui.add_dropdown(
                "Device", options=cam_opts, initial_value=cam_opts[default_idx])
            open_b = server.gui.add_button("Open", icon=viser.Icon.PLUG)
            close_b = server.gui.add_button("Close & release", icon=viser.Icon.PLUG_X)
            conn_text[side] = server.gui.add_text("State", initial_value="closed",
                                                  disabled=True)

        def make_open(side):
            def _(_e):
                if state["cams"][side] is not None or not cams:
                    if not cams:
                        conn_text[side].value = "no camera found"
                    return
                serial = serial_by_label[cam_dd[side].value]
                other = state["cams"]["R" if side == "L" else "L"]
                if other is not None and other.serial == serial:
                    conn_text[side].value = "that serial is the other camera"
                    return
                conn_text[side].value = f"opening {serial}…"
                try:
                    cam = HTCamera(serial=serial)
                    lo, hi, step = cam.gain_range()
                    gain_slider.min, gain_slider.max, gain_slider.step = lo, hi, step
                    pushed[side].update(ae=None, exp=None, gain=None, speed=None)
                    state["cams"][side] = cam
                    state["serials"][side] = cam.serial
                    state["fail"][side] = 0
                    conn_text[side].value = f"open: {cam.serial} ({cam.name})"
                except Exception as exc:
                    conn_text[side].value = f"open failed: {exc}"
            return _

        def make_close(side):
            def _(_e):
                if state["cams"][side] is not None:
                    release_side(side, "closed (released)")
            return _

        open_b.on_click(make_open(side))
        close_b.on_click(make_close(side))

    # -- Exposure / gain (shared, applied to whichever cameras are open) --------
    with server.gui.add_folder("Exposure / gain (both)"):
        ae_box = server.gui.add_checkbox("Auto exposure", initial_value=False)
        exp_slider = server.gui.add_slider(
            "Exposure (ms)", min=0.1, max=200.0, step=0.1, initial_value=5.0)
        gain_slider = server.gui.add_slider(
            "Analog gain (x)", min=1.0, max=22.0, step=0.1, initial_value=1.0)
        speed_dropdown = server.gui.add_dropdown(
            "Frame speed", options=("Low", "Mid", "High"), initial_value="Mid",
            hint="Lower = less USB bandwidth. Two 5MP cameras on one bus can drop "
                 "frames at High; Low/Mid is plenty for calibration.")

    # -- Calibration ------------------------------------------------------------
    with server.gui.add_folder("Calibration"):
        server.gui.add_text("Board", disabled=True,
                            initial_value=f"{args.squares_x}x{args.squares_y}  "
                                          f"{args.square_mm:g}mm/sq  {args.dict}")
        model_dropdown = server.gui.add_dropdown(
            "Distortion model", options=("fisheye", "pinhole"),
            initial_value="fisheye")
        auto_box = server.gui.add_checkbox(
            "Auto-capture diverse poses", initial_value=True)
        steady_box = server.gui.add_checkbox(
            "Stereo only when board steady", initial_value=True,
            hint="Capture a stereo pair only while the board is held still, so "
                 "both cameras see the same board position (frame-sync proxy).")
        min_trans_slider = server.gui.add_slider(
            "Min translation (cm)", min=0.0, max=30.0, step=0.5, initial_value=5.0)
        min_rot_slider = server.gui.add_slider(
            "Min rotation (deg)", min=0.0, max=45.0, step=1.0, initial_value=10.0)
        count_text = server.gui.add_text("Captured", initial_value="L 0  R 0  stereo 0",
                                         disabled=True)
        manual_btn = server.gui.add_button("Capture this view now")
        intr_btn = server.gui.add_button("Calibrate intrinsics (both)",
                                         icon=viser.Icon.CALCULATOR)
        stereo_btn = server.gui.add_button("Calibrate stereo (cam2cam)",
                                           icon=viser.Icon.CALCULATOR)
        save_btn = server.gui.add_button("Save calibration")
        reset_btn = server.gui.add_button("Reset captures")
        intr_text = server.gui.add_text("Intrinsics", initial_value="—", disabled=True)
        stereo_text = server.gui.add_text("Stereo", initial_value="—", disabled=True)

    # -- Undistort preview ------------------------------------------------------
    def undistort_options():
        # Per-sensor files in the CWD and in calib_session_*/ folders.
        files = sorted(set(glob.glob("calib_*.json") +
                           glob.glob("calib_session_*/calib_*.json")))
        return ["off", "current (live calib)"] + files

    with server.gui.add_folder("Undistort (preview)"):
        undist_dropdown = server.gui.add_dropdown(
            "Source", options=undistort_options(), initial_value="off",
            hint="Undistort both live feeds with each camera's intrinsics.")
        undist_refresh_btn = server.gui.add_button("Rescan files")
        balance_slider = server.gui.add_slider(
            "FoV balance", min=0.0, max=1.0, step=0.05, initial_value=0.0)

    @undist_refresh_btn.on_click
    def _(_e):
        undist_dropdown.options = undistort_options()

    speed_map = {"Low": enums.FRAME_SPEED_LOW, "Mid": enums.FRAME_SPEED_NORMAL,
                 "High": enums.FRAME_SPEED_HIGH}
    pushed = {s: {"ae": None, "exp": None, "gain": None, "speed": None} for s in SIDES}

    def apply_settings(side, cam):
        p = pushed[side]
        if ae_box.value != p["ae"]:
            cam.set_ae(ae_box.value); p["ae"] = ae_box.value
        if not ae_box.value:
            if exp_slider.value != p["exp"]:
                cam.set_exposure_ms(exp_slider.value); p["exp"] = exp_slider.value
            if gain_slider.value != p["gain"]:
                cam.set_analog_gain(gain_slider.value); p["gain"] = gain_slider.value
        if speed_dropdown.value != p["speed"]:
            cam.set_frame_speed(speed_map[speed_dropdown.value])
            p["speed"] = speed_dropdown.value

    def update_counts():
        count_text.value = (f"L {len(state['intr_cap']['L']['poses'])}  "
                            f"R {len(state['intr_cap']['R']['poses'])}  "
                            f"stereo {len(state['stereo_pairs'])}")

    def release_side(side, msg):
        """Drop a camera and free its handle without hanging if the bus is dead.

        Skips CameraPause (its retry/ReConnect loop can block ~12s on a wedged
        channel) and UnInits directly — captures are kept, only the handle goes.
        """
        cam = state["cams"][side]
        state["cams"][side] = None
        state["prev_pose"][side] = None
        state["fail"][side] = 0
        if cam is not None:
            try:
                cam._playing = False     # make close()'s pause() a no-op
                cam.close()
            except Exception:
                pass
        conn_text[side].value = msg

    @manual_btn.on_click
    def _(_e):
        state["force"] = True

    @reset_btn.on_click
    def _(_e):
        for s in SIDES:
            state["intr_cap"][s] = {"obj": [], "img": [], "poses": [], "lins": []}
        state["stereo_pairs"].clear()
        state["stereo_poses"].clear()
        state["stereo_lins"].clear()
        state["png_written"] = {"L": 0, "R": 0, "stereo": 0}
        update_counts()
        intr_text.value = stereo_text.value = "captures cleared"

    @intr_btn.on_click
    def _(_e):
        state["do_intr"] = True

    @stereo_btn.on_click
    def _(_e):
        state["do_stereo"] = True

    @save_btn.on_click
    def _(_e):
        """Publish per-sensor calib_<serial>.json (+ stereo) to calibrations/.

        calibrations/ is the git-tracked, image-free home the library loads from
        (HTCamera auto-loads calib_<serial>.json on open). Each per-sensor file
        merges into any existing calibration for that serial, so the color
        section from color_calibrate.py is preserved. The raw capture PNGs live
        in a gitignored session subfolder as re-runnable provenance.
        """
        if not any(state["intr"].values()):
            stereo_text.value = "calibrate intrinsics first"
            return
        flush_capture_pngs()
        board_notes = {"board": {
            "squares_x": args.squares_x, "squares_y": args.squares_y,
            "square_mm": args.square_mm, "marker_ratio": args.marker_ratio,
            "dict": args.dict}}
        pubdir = Path(args.calib_dir)
        pubdir.mkdir(parents=True, exist_ok=True)
        written = []
        for s in SIDES:
            intr = state["intr"][s]
            serial = state["serials"][s]
            if intr is None or serial is None:
                continue
            cal = CameraCalibration.load_or_new(serial, dir=pubdir)
            cal.intrinsics = intr
            cal.notes.update(board_notes)
            cal.save(dir=pubdir)
            written.append(f"calib_{serial}.json")
        st = state["stereo"]
        if st is not None and state["serials"]["L"] and state["serials"]["R"]:
            stereo = StereoCalibration(
                serial_left=state["serials"]["L"],
                serial_right=state["serials"]["R"],
                R=st["R"], t=st["t"], reproj_rms_px=st["reproj_rms"],
                num_pairs=st["num_pairs"], method=st["method"],
                notes=board_notes)
            stereo.save(dir=pubdir)
            written.append(stereo.default_path(
                state["serials"]["L"], state["serials"]["R"]).name)
        stereo_text.value = f"published {', '.join(written)} -> {pubdir}/"
        print(f"[save] published {', '.join(written)} to {pubdir}/")

    def run_intrinsics():
        flush_capture_pngs()
        msgs = []
        for s in SIDES:
            cap = state["intr_cap"][s]
            n = len(cap["obj"])
            if n < 6:
                msgs.append(f"{s}:{n}<6")
                continue
            w, h = state["image_size"][s]
            model = model_dropdown.value
            try:
                if model == "fisheye":
                    rms, K, dist, idx = calibrate_fisheye(cap["obj"], cap["img"], w, h)
                    used = len(idx)
                else:
                    rms, K, dist, _rv, _tv = cv2.calibrateCamera(
                        cap["obj"], cap["img"], (w, h), None, None, flags=0)
                    used = n
            except (cv2.error, RuntimeError) as exc:
                msgs.append(f"{s}:fail({exc})")
                continue
            state["intr"][s] = Intrinsics(
                model=model, image_size=(w, h), K=K, dist=dist,
                rms_reproj_px=float(rms), num_views=int(used))
            state["umaps"][s].clear()
            msgs.append(f"{s}: rms {rms:.3f}px fx {K[0,0]:.1f} cx {K[0,2]:.1f} "
                        f"({used}v,{model})")
        intr_text.value = "  |  ".join(msgs) if msgs else "no views"
        print("[intr] " + intr_text.value)

    def run_stereo():
        if not (state["intr"]["L"] and state["intr"]["R"]):
            stereo_text.value = "need both cameras' intrinsics first"
            return
        if len(state["stereo_pairs"]) < 3:
            stereo_text.value = f"need >=3 stereo pairs, have {len(state['stereo_pairs'])}"
            return
        try:
            res = solve_cam2cam(state["stereo_pairs"], state["intr"]["L"],
                                state["intr"]["R"])
        except (RuntimeError, cv2.error) as exc:
            stereo_text.value = f"stereo solve failed: {exc}"
            return
        state["stereo"] = res
        t = np.asarray(res["t"]).ravel()
        stereo_text.value = (f"baseline {res['baseline']*1000:.2f}mm  "
                             f"rot {res['rot_deg']:.3f}°  reproj {res['reproj_rms']:.3f}px  "
                             f"t[{t[0]*1000:.1f},{t[1]*1000:.1f},{t[2]*1000:.1f}]mm  "
                             f"({res['num_pairs']} pairs, {res['method']})")
        print("[stereo] " + stereo_text.value)

    # -- Per-frame analysis -----------------------------------------------------
    def analyze(lin, w, h):
        disp = convert.tonemap_linear(lin, **TONE)
        gray = cv2.cvtColor(disp, cv2.COLOR_RGB2GRAY)
        cc, ci, _mc, _mi = detector.detectBoard(gray)
        n = 0 if ci is None else len(ci)
        objp = imgp = pose = None
        sharp = None
        if n >= min_corners:
            # Motion-blur gate: skip pose/capture entirely for blurred views —
            # blurred corners are localized poorly and poison the solve.
            sharp = corner_sharpness(gray, cc)
            if sharp >= args.min_sharpness:
                objp, imgp = board.matchImagePoints(cc, ci)
                ok, rv, tv = cv2.solvePnP(objp, imgp, guessed_K(w, h), None,
                                          flags=cv2.SOLVEPNP_ITERATIVE)
                if ok:
                    pose = (rv, tv)
        return dict(lin=lin, disp=disp, cc=cc, ci=ci, n=n, objp=objp, imgp=imgp,
                    pose=pose, sharp=sharp)

    def common_corrs(ciL, ccL, ciR, ccR):
        idL = {int(i): ccL[k, 0] for k, i in enumerate(ciL.ravel())}
        idR = {int(i): ccR[k, 0] for k, i in enumerate(ciR.ravel())}
        common = sorted(set(idL) & set(idR))
        if len(common) < min_corners:
            return None
        objp = board_obj[common].reshape(-1, 1, 3).astype(np.float32)
        imgL = np.array([idL[i] for i in common], np.float32).reshape(-1, 1, 2)
        imgR = np.array([idR[i] for i in common], np.float32).reshape(-1, 1, 2)
        return objp, imgL, imgR

    def side_intr_for_undistort(side):
        """Intrinsics for this side's undistort preview, or None.

        A calib_<serial>.json applies to whichever open side has that serial.
        """
        sel = undist_dropdown.value
        if sel == "off":
            return None
        if sel.startswith("current"):
            return state["intr"][side]
        cal = state["file_calib"].get(sel)
        if cal is None:
            try:
                cal = CameraCalibration.load(sel)
                state["file_calib"][sel] = cal
            except Exception:
                return None
        if cal.intrinsics is not None and cal.serial == state["serials"][side]:
            return cal.intrinsics
        return None

    def tile(side, disp):
        """Resize a side's display to a tile, undistorting if a source is set."""
        h, w = disp.shape[:2]
        tw = SIDE_W
        th = int(round(h * tw / w))
        small = cv2.resize(disp, (tw, th), interpolation=cv2.INTER_AREA)
        intr = side_intr_for_undistort(side)
        if intr is not None:
            cache = state["umaps"][side]
            key = (id(intr), undist_dropdown.value, round(balance_slider.value, 3), tw, th)
            if cache.get("key") != key:
                try:
                    m1, m2 = intr.undistort_maps(tw, th, balance_slider.value)
                    cache.update(key=key, m1=m1, m2=m2)
                except Exception:
                    cache.clear()
            if cache.get("m1") is not None:
                small = cv2.remap(small, cache["m1"], cache["m2"], cv2.INTER_LINEAR)
        cv2.putText(small, side, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 255, 0), 2, cv2.LINE_AA)
        return small

    print(f"Viser up on :{args.port} — open the URL. Open Camera L (and R for "
          f"stereo). {'Physical board mode.' if args.no_board_window else 'Drag the board window next to the browser.'}")
    fps_t0 = time.time()
    frames_n = 0

    try:
        while not state["quit"]:
            cv2.waitKey(1)
            if state["do_intr"]:
                state["do_intr"] = False; run_intrinsics()
            if state["do_stereo"]:
                state["do_stereo"] = False; run_stereo()

            force = state["force"]
            state["force"] = False
            results = {}     # side -> analyze() dict
            any_open = False
            for s in SIDES:
                cam = state["cams"][s]
                if cam is None:
                    continue
                any_open = True
                try:
                    apply_settings(s, cam)
                    lin, _info = cam.grab(timeout_ms=300)   # short, so one flaky
                except Exception:                            # camera can't stall
                    lin = None                               # the healthy one
                if lin is None:
                    state["fail"][s] += 1
                    # A few empty grabs are normal hiccups; a sustained run means
                    # the device fell off the bus — release it so the GUI surfaces
                    # it (one-click reopen after replug) and the good camera keeps
                    # streaming at full rate instead of waiting on the dead one.
                    if state["fail"][s] >= LOST_AFTER:
                        release_side(s, "LOST — replug USB & reopen")
                        print(f"[{s}] no frames for ~{LOST_AFTER*0.3:.0f}s; released. "
                              f"Replug the USB and click Open.")
                    continue
                state["fail"][s] = 0
                h, w = lin.shape[:2]
                if state["image_size"][s] is None:
                    state["image_size"][s] = (w, h)
                results[s] = analyze(lin, w, h)

            if not any_open:
                time.sleep(0.05)
                continue

            # Per-side intrinsic capture (novelty on that camera's own pose).
            steady = {}
            for s, r in results.items():
                if r["pose"] is None:
                    state["prev_pose"][s] = None
                    continue
                rv, tv = r["pose"]
                prev = state["prev_pose"][s]
                steady[s] = (prev is not None
                             and float(np.linalg.norm(tv - prev[1])) < STILL_TRANS_M
                             and rotation_angle_deg(rv, prev[0]) < STILL_ROT_DEG)
                state["prev_pose"][s] = (rv, tv)
                cap = state["intr_cap"][s]
                novel, _c, _d = is_novel(rv, tv, cap["poses"],
                                         min_trans_slider.value / 100.0,
                                         min_rot_slider.value)
                if force or (auto_box.value and novel):
                    cap["obj"].append(r["objp"]); cap["img"].append(r["imgp"])
                    cap["poses"].append((rv, tv))
                    cap["lins"].append(r["lin"])

            # Stereo pair capture: both see the board, steady (unless disabled),
            # and the left pose is novel vs kept stereo poses.
            stereo_status = ""
            if "L" in results and "R" in results and results["L"]["pose"] and results["R"]["pose"]:
                corr = common_corrs(results["L"]["ci"], results["L"]["cc"],
                                    results["R"]["ci"], results["R"]["cc"])
                if corr is not None:
                    rvL, tvL = results["L"]["pose"]
                    novel, ncm, ndeg = is_novel(rvL, tvL, state["stereo_poses"],
                                                min_trans_slider.value / 100.0,
                                                min_rot_slider.value)
                    is_steady = steady.get("L", False) and steady.get("R", False)
                    ok_steady = is_steady or not steady_box.value
                    take = force or (auto_box.value and novel and ok_steady)
                    if take:
                        state["stereo_pairs"].append(corr)
                        state["stereo_poses"].append((rvL, tvL))
                        state["stereo_lins"].append(
                            tuple(results[s]["lin"] for s in SIDES))
                    common_n = len(corr[0])
                    tag = ("CAPTURED" if take else
                           "moving" if (novel and not ok_steady) else
                           "novel" if novel else "redundant")
                    stereo_status = f"stereo {common_n}c {tag}"
                else:
                    stereo_status = "stereo: too few common corners"

            update_counts()

            # Composite preview: overlay corners, tile each side, hstack.
            tiles = []
            for s in SIDES:
                r = results.get(s)
                if r is None:
                    continue
                disp = r["disp"]
                if r["ci"] is not None:
                    disp = cv2.aruco.drawDetectedCornersCharuco(
                        np.ascontiguousarray(disp), r["cc"], r["ci"], (0, 255, 0))
                tiles.append(tile(s, disp))
            if tiles:
                hmax = max(t.shape[0] for t in tiles)
                tiles = [cv2.copyMakeBorder(t, 0, hmax - t.shape[0], 0, 0,
                                            cv2.BORDER_CONSTANT, value=0)
                         for t in tiles]
                server.scene.set_background_image(
                    np.hstack(tiles), format="jpeg", jpeg_quality=JPEG_QUALITY)

            frames_n += 1
            now = time.time()
            if now - fps_t0 >= 0.5:
                fps = frames_n / (now - fps_t0)
                det = " ".join(
                    f"{s}:{results[s]['n']}c" + (
                        f" BLUR({results[s]['sharp']:.2f})"
                        if results[s]["sharp"] is not None
                        and results[s]["sharp"] < args.min_sharpness else "")
                    for s in SIDES if s in results)
                for s in SIDES:
                    if state["cams"][s] is not None:
                        conn_text[s].value = (
                            f"open: {state['cams'][s].serial}  {fps:.1f}fps  "
                            f"{det}  {stereo_status}")
                fps_t0 = now
                frames_n = 0
    except KeyboardInterrupt:
        print("\nstopping (Ctrl-C)")
    finally:
        for s in SIDES:
            if state["cams"][s] is not None:
                state["cams"][s].close()
        cv2.destroyAllWindows()
        print("cameras released. safe to relaunch.")


if __name__ == "__main__":
    main()
