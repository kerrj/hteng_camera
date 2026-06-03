"""Intrinsic camera calibration from a ChArUco board shown on a monitor.

Workflow:
  1. A ChArUco board is rendered fullscreen in an OpenCV window on this monitor.
  2. You point the HTENG camera at the screen and move it around (vary angle and
     distance). The live camera feed — with detected corners overlaid — streams to
     a viser page in your browser, alongside the calibration controls.
  3. With "Auto-capture" on, only *pose-diverse* views are kept: a frame is saved
     only if its board pose differs from every already-saved view by more than the
     translation OR rotation threshold (default 5 cm / 10°). This avoids dumping a
     hundred near-identical frames and keeps the calibration well-conditioned.
  4. Hit "Calibrate" once you have a dozen-plus diverse views.

The capture/display path mirrors ``viser_control.py``: ``grab()`` yields linear
uint16 RGB; the preview (and the image fed to the detector) is log-tone-mapped to
uint8 via ``convert.to_display(curve="log", param=120)``.

Pose gating needs intrinsics to run ``solvePnP``, but intrinsics are what we're
solving for — so gating uses a *rough* guessed K (fx=fy≈width, centered principal
point, no distortion). That's accurate enough to judge *relative* novelty; the
real intrinsics come out of the final ``cv2.calibrateCamera``.

Metric note: the 5 cm translation gate is only true centimeters if ``--square-mm``
matches the square size as *displayed on your monitor* — measure one square with a
ruler and pass it. Intrinsics (fx, fy, cx, cy, distortion) don't depend on scale;
only the translation gate and reported tvecs do.

Run::

    pip install hteng-camera[viser]
    python examples/charuco_calibrate.py            # default 7x5 board
    python examples/charuco_calibrate.py --square-mm 28.5

Open the URL viser prints. Use "Close & release" before quitting.
"""

import argparse
import glob
import json
import time
from datetime import datetime

import cv2
import numpy as np
import viser

from hteng_camera import HTCamera, list_cameras, convert, enums

JPEG_QUALITY = 85
DISPLAY_WIDTH = 1280

# Preview / detector tone curve (per request): log space, strength 120.
TONE = dict(curve="log", param=120.0)

# ChArUco dictionaries keyed by a friendly name (must hold enough markers for the
# board: a SxS board uses ceil(squares_x*squares_y / 2) markers).
DICTS = {
    "4X4_50": cv2.aruco.DICT_4X4_50,
    "4X4_100": cv2.aruco.DICT_4X4_100,
    "5X5_100": cv2.aruco.DICT_5X5_100,
    "5X5_250": cv2.aruco.DICT_5X5_250,
}


def downscale(img, target_w):
    h, w = img.shape[:2]
    if w <= target_w:
        return img
    step = max(1, w // target_w)
    return img[::step, ::step]


def build_undistort_maps(calib, out_w, out_h, balance):
    """Remap tables that undistort a frame of size (out_w, out_h).

    ``calib`` is the saved JSON dict (K, dist, model, image_size). The calibration
    K is for the full-res ``image_size``; if the preview is a different size we
    scale K accordingly so the maps line up with whatever we actually display.
    Fisheye uses the Kannala-Brandt undistort; pinhole uses Brown-Conrady. The
    ``balance`` (0..1) trades retained field-of-view against black border: 0 crops
    to all-valid pixels, 1 keeps every source pixel (curved black corners).
    """
    cw, ch = calib["image_size"]
    s = out_w / float(cw)
    K = np.array(calib["K"], np.float64)
    K = K * np.array([[s, s, s], [s, s, s], [0, 0, 1.0]])  # scale fx,fy,cx,cy; keep 1
    K[2, 2] = 1.0
    dist = np.array(calib["dist"], np.float64)
    size = (out_w, out_h)
    if calib.get("model") == "fisheye":
        D = dist.reshape(4, 1)
        # NOTE: cv2.fisheye.estimateNewCameraMatrixForUndistortRectify returns a
        # wildly off-center newK for wide fisheyes (cx lands thousands of px
        # off-screen), so the remap zooms into a sliver -> stretched/garbled. Build
        # newK by hand: principal point at the (preview) image centre, focal scaled
        # by `balance`. balance 0 keeps the original focal (fills the frame, crops
        # the fisheye periphery); higher zooms out to keep more field of view, with
        # black borders at the limit.
        newK = K.copy()
        f_scale = 1.0 - 0.7 * balance
        newK[0, 0] *= f_scale
        newK[1, 1] *= f_scale
        newK[0, 2] = out_w / 2.0
        newK[1, 2] = out_h / 2.0
        return cv2.fisheye.initUndistortRectifyMap(
            K, D, np.eye(3), newK, size, cv2.CV_16SC2)
    newK, _ = cv2.getOptimalNewCameraMatrix(K, dist, size, balance, size)
    return cv2.initUndistortRectifyMap(
        K, dist, np.eye(3), newK, size, cv2.CV_16SC2)


def rotation_angle_deg(rvec_a, rvec_b):
    """Geodesic angle (degrees) between two rotations given as Rodrigues vectors."""
    Ra, _ = cv2.Rodrigues(rvec_a)
    Rb, _ = cv2.Rodrigues(rvec_b)
    R = Ra @ Rb.T
    cos = (np.trace(R) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def is_novel(rvec, tvec, kept, min_trans_m, min_rot_deg):
    """True if (rvec, tvec) is far from *every* kept pose.

    A view is redundant only when some saved view is within BOTH the translation
    and the rotation threshold — i.e. it's kept if it moved >min_trans OR rotated
    >min_rot relative to its nearest neighbour. Returns (novel, nearest_trans_cm,
    nearest_rot_deg) so the GUI can show how close the current view is.
    """
    nearest_t = float("inf")
    nearest_r = float("inf")
    novel = True
    for r_k, t_k in kept:
        dt = float(np.linalg.norm(tvec - t_k))
        dr = rotation_angle_deg(rvec, r_k)
        # Track the single most-similar neighbour for the readout.
        if dt < nearest_t:
            nearest_t = dt
        if dr < nearest_r:
            nearest_r = dr
        if dt < min_trans_m and dr < min_rot_deg:
            novel = False  # close in both -> redundant, but keep scanning for readout
    return novel, nearest_t * 100.0, nearest_r


def guessed_K(w, h):
    """Rough pinhole guess for pose-gating only (focal ~ image width)."""
    f = float(w)
    return np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1.0]], np.float64)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--squares-x", type=int, default=7, help="board columns")
    ap.add_argument("--squares-y", type=int, default=5, help="board rows")
    ap.add_argument("--square-mm", type=float, default=30.0,
                    help="square edge as DISPLAYED on the monitor (measure it!)")
    ap.add_argument("--marker-ratio", type=float, default=0.75,
                    help="aruco marker size as a fraction of the square")
    ap.add_argument("--dict", default="5X5_100", choices=list(DICTS),
                    help="aruco dictionary")
    ap.add_argument("--square-px", type=int, default=160,
                    help="pixels per square in the rendered board")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()

    square_len = args.square_mm / 1000.0           # metres
    marker_len = square_len * args.marker_ratio
    dictionary = cv2.aruco.getPredefinedDictionary(DICTS[args.dict])
    board = cv2.aruco.CharucoBoard(
        (args.squares_x, args.squares_y), square_len, marker_len, dictionary)
    board.setLegacyPattern(False)
    detector = cv2.aruco.CharucoDetector(board)
    n_board_corners = (args.squares_x - 1) * (args.squares_y - 1)
    min_corners = max(6, n_board_corners // 2)     # accept a frame only if this many seen

    # --- Render the board and show it fullscreen on the monitor ----------------
    board_w = args.squares_x * args.square_px
    board_h = args.squares_y * args.square_px
    board_img = board.generateImage((board_w, board_h), marginSize=args.square_px // 4)
    WIN = "ChArUco board (aim camera here)"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)        # resizable, not fullscreen
    cv2.resizeWindow(WIN, board_w, board_h)
    cv2.moveWindow(WIN, 40, 40)                    # park it top-left; drag it next to the browser
    cv2.imshow(WIN, board_img)
    cv2.waitKey(1)

    server = viser.ViserServer(port=args.port)

    state = {
        "cam": None,
        "latest_linear": None,
        "obj_pts": [],          # per-view object points (matchImagePoints output)
        "img_pts": [],          # per-view image points
        "poses": [],            # per-view (rvec, tvec) for gating
        "image_size": None,     # (w, h)
        "calib": None,          # last calibration result dict
        "do_calibrate": False,  # set by button, serviced in main loop
        "file_calib": {},       # filename -> loaded JSON calib (cache)
        "umaps": {"key": None, "m1": None, "m2": None},  # cached remap tables
        "quit": False,
    }

    cams = list_cameras()
    options = [f'{c["serial"]}  ({c["name"]})' for c in cams] or ["<none found>"]
    serial_by_label = {opt: c["serial"] for opt, c in zip(options, cams)}

    # -- Connection -------------------------------------------------------------
    with server.gui.add_folder("Camera"):
        cam_dropdown = server.gui.add_dropdown(
            "Device", options=options, initial_value=options[0])
        open_btn = server.gui.add_button("Open", icon=viser.Icon.PLUG)
        close_btn = server.gui.add_button("Close & release", icon=viser.Icon.PLUG_X)
        conn_text = server.gui.add_text("State", initial_value="closed", disabled=True)

    # -- Exposure / gain (defaults: AE off, 5 ms, 1x) ---------------------------
    with server.gui.add_folder("Exposure / gain"):
        ae_box = server.gui.add_checkbox("Auto exposure", initial_value=False)
        exp_slider = server.gui.add_slider(
            "Exposure (ms)", min=0.1, max=200.0, step=0.1, initial_value=5.0)
        gain_slider = server.gui.add_slider(
            "Analog gain (x)", min=1.0, max=22.0, step=0.1, initial_value=1.0)
        speed_dropdown = server.gui.add_dropdown(
            "Frame speed", options=("Low", "Mid", "High"), initial_value="High")

    # -- Calibration ------------------------------------------------------------
    with server.gui.add_folder("Calibration"):
        board_text = server.gui.add_text(
            "Board", disabled=True,
            initial_value=f"{args.squares_x}x{args.squares_y}  "
                          f"{args.square_mm:g}mm/sq  {args.dict}")
        auto_box = server.gui.add_checkbox(
            "Auto-capture diverse poses", initial_value=True,
            hint="Save a frame only when its board pose is novel vs. all saved views.")
        min_trans_slider = server.gui.add_slider(
            "Min translation (cm)", min=0.0, max=30.0, step=0.5, initial_value=5.0)
        min_rot_slider = server.gui.add_slider(
            "Min rotation (deg)", min=0.0, max=45.0, step=1.0, initial_value=10.0)
        captured_text = server.gui.add_text(
            "Captured views", initial_value="0", disabled=True)
        model_dropdown = server.gui.add_dropdown(
            "Distortion model", options=("fisheye", "pinhole"),
            initial_value="fisheye",
            hint="fisheye: Kannala-Brandt (k1-k4), for wide/fisheye lenses.  "
                 "pinhole: Brown-Conrady (k1,k2,p1,p2,k3), for normal lenses.")
        manual_btn = server.gui.add_button("Capture this view now")
        calib_btn = server.gui.add_button("Calibrate", icon=viser.Icon.CALCULATOR)
        save_btn = server.gui.add_button("Save calibration")
        reset_btn = server.gui.add_button("Reset captures")
        result_text = server.gui.add_text("Result", initial_value="—", disabled=True)

    # -- Undistort the live preview ---------------------------------------------
    def undistort_options():
        # "off", the just-computed in-memory calib, then any saved JSON files.
        return ["off", "current (live calib)"] + sorted(glob.glob("calib_*.json"))

    with server.gui.add_folder("Undistort (preview)"):
        undist_dropdown = server.gui.add_dropdown(
            "Source", options=undistort_options(), initial_value="off",
            hint="Undistort the live feed using a calibration. 'current' uses the "
                 "last Calibrate result; or pick a saved calib_*.json.")
        undist_refresh_btn = server.gui.add_button("Rescan files")
        balance_slider = server.gui.add_slider(
            "FoV balance", min=0.0, max=1.0, step=0.05, initial_value=0.0,
            hint="0 = crop to all-valid pixels (no black border).  "
                 "1 = keep every source pixel (curved black corners).")
        undist_text = server.gui.add_text("Undistort", initial_value="off",
                                          disabled=True)

    @undist_refresh_btn.on_click
    def _(_e):
        undist_dropdown.options = undistort_options()

    speed_map = {"Low": enums.FRAME_SPEED_LOW, "Mid": enums.FRAME_SPEED_NORMAL,
                 "High": enums.FRAME_SPEED_HIGH}
    pushed = {"ae": None, "exp": None, "gain": None, "speed": None}

    def apply_settings(cam):
        if ae_box.value != pushed["ae"]:
            cam.set_ae(ae_box.value)
            pushed["ae"] = ae_box.value
        if not ae_box.value:
            if exp_slider.value != pushed["exp"]:
                cam.set_exposure_ms(exp_slider.value)
                pushed["exp"] = exp_slider.value
            if gain_slider.value != pushed["gain"]:
                cam.set_analog_gain(gain_slider.value)
                pushed["gain"] = gain_slider.value
        if speed_dropdown.value != pushed["speed"]:
            cam.set_frame_speed(speed_map[speed_dropdown.value])
            pushed["speed"] = speed_dropdown.value

    @open_btn.on_click
    def _(_e):
        if state["cam"] is not None or not cams:
            if not cams:
                conn_text.value = "no camera found"
            return
        serial = serial_by_label[cam_dropdown.value]
        conn_text.value = f"opening {serial}…"
        try:
            cam = HTCamera(serial=serial)
            lo, hi, step = cam.gain_range()
            gain_slider.min, gain_slider.max, gain_slider.step = lo, hi, step
            pushed.update(ae=None, exp=None, gain=None, speed=None)
            state["cam"] = cam
            conn_text.value = f"open: {cam.serial} ({cam.name})"
        except Exception as exc:
            conn_text.value = f"open failed: {exc}"

    @close_btn.on_click
    def _(_e):
        cam = state["cam"]
        if cam is None:
            return
        state["cam"] = None
        state["latest_linear"] = None
        cam.close()
        server.scene.set_background_image(
            np.zeros((360, 640, 3), np.uint8), format="jpeg")
        conn_text.value = "closed (released)"

    @manual_btn.on_click
    def _(_e):
        # Force-capture the latest detected view, bypassing the novelty gate.
        state["_force_capture"] = True

    @reset_btn.on_click
    def _(_e):
        state["obj_pts"].clear()
        state["img_pts"].clear()
        state["poses"].clear()
        state["calib"] = None
        captured_text.value = "0"
        result_text.value = "captures cleared"

    @calib_btn.on_click
    def _(_e):
        state["do_calibrate"] = True

    @save_btn.on_click
    def _(_e):
        calib = state["calib"]
        if calib is None:
            result_text.value = "calibrate first"
            return
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = f"calib_{ts}"
        with open(f"{stem}.json", "w") as f:
            json.dump(calib["json"], f, indent=2)
        np.savez(f"{stem}.npz", K=calib["K"], dist=calib["dist"],
                 image_size=np.array(calib["json"]["image_size"]))
        result_text.value = f"saved {stem}.json + .npz  (rms {calib['rms']:.3f}px)"

    def calibrate_fisheye(obj_pts, img_pts, w, h):
        """Kannala-Brandt fisheye fit. Returns (rms, K, D, used_indices).

        fisheye.calibrate is picky: object/image points must be float64 shaped
        (N,1,3)/(N,1,2), and a single ill-conditioned view aborts the whole solve
        with CALIB_CHECK_COND. We don't pass CHECK_COND, but degenerate views can
        still raise — so on failure we drop the offending view (parsed from the
        error, else the last one) and retry until it converges.
        """
        flags = (cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
                 | cv2.fisheye.CALIB_FIX_SKEW)
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
                # Error text often ends with "... input array N" — drop view N.
                drop = len(obj) - 1
                for tok in str(exc).replace(".", " ").split():
                    if tok.isdigit() and int(tok) < len(obj):
                        drop = int(tok)
                obj.pop(drop); img.pop(drop); idx.pop(drop)
        raise RuntimeError("fisheye solve failed: too few well-conditioned views")

    def run_calibration():
        n = len(state["obj_pts"])
        if n < 6:
            result_text.value = f"need >=6 views, have {n}"
            return
        w, h = state["image_size"]
        model = model_dropdown.value
        used = n
        try:
            if model == "fisheye":
                rms, K, dist, idx = calibrate_fisheye(
                    state["obj_pts"], state["img_pts"], w, h)
                used = len(idx)
            else:
                rms, K, dist, _rv, _tv = cv2.calibrateCamera(
                    state["obj_pts"], state["img_pts"], (w, h), None, None, flags=0)
        except (cv2.error, RuntimeError) as exc:
            result_text.value = f"{model} calibrate failed: {exc}"
            print("[calib] " + result_text.value)
            return
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        dropped = n - used
        info = {
            "model": model,
            "image_size": [int(w), int(h)],
            "num_views": int(used),
            "dropped_views": int(dropped),
            "rms_reproj_px": float(rms),
            "K": K.tolist(),
            "dist": dist.ravel().tolist(),
            "fx": float(fx), "fy": float(fy), "cx": float(cx), "cy": float(cy),
            "board": {"squares_x": args.squares_x, "squares_y": args.squares_y,
                      "square_mm": args.square_mm, "marker_ratio": args.marker_ratio,
                      "dict": args.dict},
        }
        state["calib"] = {"K": K, "dist": dist, "rms": float(rms), "json": info}
        state["umaps"]["key"] = None  # force remap rebuild if previewing "current"
        drop_tag = f"  (-{dropped} ill-cond)" if dropped else ""
        dist_tag = " ".join(f"{c:+.4f}" for c in dist.ravel())
        result_text.value = (f"[{model}] rms {rms:.3f}px  fx {fx:.1f} fy {fy:.1f}  "
                             f"cx {cx:.1f} cy {cy:.1f}  dist [{dist_tag}]  "
                             f"({used} views{drop_tag})")
        print("[calib] " + result_text.value)

    def resolve_undistort_calib():
        """Return (calib_dict, label) for the dropdown selection, or (None, ...)."""
        sel = undist_dropdown.value
        if sel == "off":
            return None, "off"
        if sel.startswith("current"):
            c = state["calib"]
            return (c["json"], "current") if c else (None, "no live calib yet")
        cached = state["file_calib"].get(sel)
        if cached is None:
            try:
                with open(sel) as f:
                    cached = json.load(f)
                state["file_calib"][sel] = cached
            except Exception as exc:
                return None, f"load failed: {exc}"
        return cached, sel

    def apply_undistort(img):
        """Undistort `img` per the dropdown; returns (out, status). No-op if off."""
        calib, label = resolve_undistort_calib()
        if calib is None:
            undist_text.value = label
            return img, label
        h, w = img.shape[:2]
        key = (label, round(balance_slider.value, 3), w, h, calib.get("model"))
        cache = state["umaps"]
        if cache["key"] != key:
            try:
                m1, m2 = build_undistort_maps(calib, w, h, balance_slider.value)
            except Exception as exc:
                undist_text.value = f"map build failed: {exc}"
                return img, "undistort error"
            cache.update(key=key, m1=m1, m2=m2)
        out = cv2.remap(img, cache["m1"], cache["m2"], cv2.INTER_LINEAR)
        undist_text.value = (f"{label}  [{calib.get('model')}]  "
                             f"balance {balance_slider.value:.2f}")
        return out, "undistorted"

    print(f"Viser up on :{args.port} — open the URL. Drag the ChArUco board window "
          f"next to the browser. Press 'Open' to connect a camera.")
    fps_t0 = time.time()
    frames = 0

    try:
        while not state["quit"]:
            cv2.waitKey(1)  # keep the fullscreen board window responsive (macOS)

            if state["do_calibrate"]:
                state["do_calibrate"] = False
                run_calibration()

            cam = state["cam"]
            if cam is None:
                time.sleep(0.05)
                continue

            apply_settings(cam)
            lin, _info = cam.grab(timeout_ms=500)
            if lin is None:
                continue
            state["latest_linear"] = lin
            h, w = lin.shape[:2]
            if state["image_size"] is None:
                state["image_size"] = (w, h)

            disp = convert.to_display(lin, **TONE)            # uint8 RGB, log-mapped
            gray = cv2.cvtColor(disp, cv2.COLOR_RGB2GRAY)

            ch_corners, ch_ids, _mk_corners, _mk_ids = detector.detectBoard(gray)
            status = "no board"
            n_seen = 0 if ch_ids is None else len(ch_ids)

            if n_seen >= min_corners:
                obj_pts, img_pts = board.matchImagePoints(ch_corners, ch_ids)
                K0 = guessed_K(w, h)
                ok, rvec, tvec = cv2.solvePnP(
                    obj_pts, img_pts, K0, None, flags=cv2.SOLVEPNP_ITERATIVE)
                if ok:
                    novel, near_cm, near_deg = is_novel(
                        rvec, tvec, state["poses"],
                        min_trans_slider.value / 100.0, min_rot_slider.value)
                    forced = state.pop("_force_capture", False)
                    take = forced or (auto_box.value and novel)
                    if take:
                        state["obj_pts"].append(obj_pts)
                        state["img_pts"].append(img_pts)
                        state["poses"].append((rvec, tvec))
                        captured_text.value = str(len(state["poses"]))
                    if not state["poses"]:
                        status = f"{n_seen} corners  (first view ready)"
                    else:
                        tag = "CAPTURED" if take else ("novel" if novel else "redundant")
                        status = (f"{n_seen} corners  {tag}  "
                                  f"nearest Δ {near_cm:.1f}cm / {near_deg:.1f}°")
                else:
                    status = f"{n_seen} corners (pnp failed)"
                # Overlay the detected charuco corners on the preview.
                disp = cv2.aruco.drawDetectedCornersCharuco(
                    np.ascontiguousarray(disp), ch_corners, ch_ids, (0, 255, 0))
            else:
                state.pop("_force_capture", False)
                if n_seen:
                    status = f"{n_seen} corners (need {min_corners})"

            preview = downscale(disp, DISPLAY_WIDTH)
            preview, _u = apply_undistort(preview)
            server.scene.set_background_image(
                preview, format="jpeg", jpeg_quality=JPEG_QUALITY)

            frames += 1
            now = time.time()
            if now - fps_t0 >= 0.5:
                fps = frames / (now - fps_t0)
                conn_text.value = (f"open: {cam.serial}  {fps:.1f} fps  |  {status}"
                                   if cam else status)
                fps_t0 = now
                frames = 0
    except KeyboardInterrupt:
        print("\nstopping (Ctrl-C)")
    finally:
        if state["cam"] is not None:
            state["cam"].close()
        cv2.destroyAllWindows()
        print("camera released. safe to relaunch.")


if __name__ == "__main__":
    main()
