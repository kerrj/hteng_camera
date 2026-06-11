"""HTENG camera control GUI in viser.

A single self-contained example: pick a camera, open/close it, drive exposure /
gain / frame-speed, and adjust the display tone curve (gamma etc.) live. The
camera image is shown as the viser scene background (full window), so the GUI
panel stays uncluttered.

The capture path is the package's fast path: raw 12-bit Bayer -> cv2 demosaic ->
*linear* uint16 RGB. Gamma is applied here, for display only, via
``convert.tonemap_linear`` — the linear signal is what a snapshot saves.

Run::

    pip install hteng-camera[viser]
    python examples/viser_control.py

Open the URL viser prints. Use the GUI's "Close & release" before quitting so
the camera is always released cleanly (a leaked handle wedges the next run).
"""

import time
from datetime import datetime

import cv2
import numpy as np
import viser

from hteng_camera import HTCamera, list_cameras, convert, enums

JPEG_QUALITY = 80
DISPLAY_WIDTH = 1280   # normal preview width; capture is always full-res
LOWRES_WIDTH = 480     # "Low-res preview" width — cheaper encode + far fewer bytes


def downscale(img, target_w):
    """Resize to target_w wide (INTER_AREA: proper box filter, no aliasing).

    The old stride-skip (img[::step, ::step]) silently passed full-res frames
    through whenever width < 2*target_w — e.g. a 2448-wide sensor at a 1280
    target shipped every full frame to the JPEG encoder.
    """
    h, w = img.shape[:2]
    if w <= target_w:
        return img
    target_h = round(h * target_w / w)
    return cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)


def main():
    server = viser.ViserServer(port=8080)

    # Mutable session state shared across GUI callbacks and the capture loop.
    state = {
        "cam": None,          # HTCamera or None
        "latest_linear": None,  # most recent full-res linear uint16 RGB
        "quit": False,
        "full_w": 0,          # native sensor resolution, learned on open
        "full_h": 0,
    }

    cams = list_cameras()
    options = [f'{c["serial"]}  ({c["name"]})' for c in cams] or ["<none found>"]
    serial_by_label = {opt: c["serial"] for opt, c in zip(options, cams)}

    # -- Connection --------------------------------------------------------
    with server.gui.add_folder("Camera"):
        cam_dropdown = server.gui.add_dropdown(
            "Device", options=options, initial_value=options[0])
        open_btn = server.gui.add_button("Open", icon=viser.Icon.PLUG)
        close_btn = server.gui.add_button("Close & release", icon=viser.Icon.PLUG_X)
        conn_text = server.gui.add_text("State", initial_value="closed", disabled=True)

    # -- Exposure / gain (thin SDK pass-throughs) --------------------------
    with server.gui.add_folder("Exposure / gain"):
        ae_box = server.gui.add_checkbox("Auto exposure", initial_value=False)
        exp_slider = server.gui.add_slider(
            "Exposure (ms)", min=0.1, max=200.0, step=0.1, initial_value=15.0)
        gain_slider = server.gui.add_slider(
            "Analog gain (x)", min=1.0, max=22.0, step=0.1, initial_value=1.0)
        speed_dropdown = server.gui.add_dropdown(
            "Frame speed", options=("Low", "Mid", "High"), initial_value="High")

    # -- Sensor ROI (firmware crop -> less USB bandwidth) ------------------
    # set_roi() sets a true sensor-readout window, so only the cropped region
    # is transferred over USB. Defaults to full res; slider bounds + the full
    # baseline are filled in from current_resolution() when a camera opens.
    ROI_STEP = 16  # most MindVision sensors want width/offset on a 16-px grid
    with server.gui.add_folder("Sensor ROI (USB bandwidth)"):
        roi_full_box = server.gui.add_checkbox(
            "Full resolution", initial_value=True,
            hint="Read out the whole sensor. Uncheck to crop the readout (e.g. "
                 "to a fisheye's image circle) and cut USB transfer time.")
        roi_w = server.gui.add_slider(
            "Width", min=ROI_STEP, max=4096, step=ROI_STEP, initial_value=4096,
            disabled=True)
        roi_h = server.gui.add_slider(
            "Height", min=ROI_STEP, max=4096, step=ROI_STEP, initial_value=4096,
            disabled=True)
        roi_x = server.gui.add_slider(
            "X offset", min=0, max=4096, step=ROI_STEP, initial_value=0,
            disabled=True)
        roi_y = server.gui.add_slider(
            "Y offset", min=0, max=4096, step=ROI_STEP, initial_value=0,
            disabled=True)
        roi_center_btn = server.gui.add_button("Center ROI")
        roi_apply_btn = server.gui.add_button("Apply ROI")
        roi_text = server.gui.add_text("Sensor", initial_value="—", disabled=True)

    # -- Display tone curve (preview only) ---------------------------------
    # Per-curve "strength" slider range/default — the slider's meaning adapts to
    # the selected curve (gamma exponent / log shadow-lift a / reinhard k).
    curve_param_cfg = {
        "bt709":    dict(min=0.0, max=1.0, step=0.1, default=0.0,
                         hint="BT.709 standard SDR gamma — no parameter (slider "
                              "unused). Matches what record.py bakes into the "
                              "default master, so the preview is WYSIWYG."),
        "gamma":    dict(min=0.4, max=4.0, step=0.05, default=2.0,
                         hint="Gamma exponent. 2.0=sqrt, 1.0=linear. High gamma "
                              "lifts darks but washes out mid/highlights."),
        "log":      dict(min=2.0, max=300.0, step=1.0, default=80.0,
                         hint="Shadow-lift strength a. Higher = more dark detail. "
                              "Best all-round for HDR scenes."),
        "reinhard": dict(min=0.02, max=1.0, step=0.01, default=0.18,
                         hint="Shoulder k. Lower = brighter midtones."),
    }
    with server.gui.add_folder("Display tone curve (preview only)"):
        curve_dropdown = server.gui.add_dropdown(
            "Curve", options=("bt709", "gamma", "log", "reinhard"),
            initial_value="bt709",
            hint="bt709: standard SDR gamma, matches the recorded master.  "
                 "gamma: classic power curve.  log: equal detail per stop (HDR).  "
                 "reinhard: smooth global tone map.")
        param_slider = server.gui.add_slider(
            "Curve strength", min=0.0, max=1.0, step=0.1, initial_value=0.0,
            hint=curve_param_cfg["bt709"]["hint"])
        ev_slider = server.gui.add_slider(
            "Exposure mult", min=0.1, max=16.0, step=0.1, initial_value=1.0,
            hint="Brightens the linear signal before clipping (display only).")
        black_slider = server.gui.add_slider(
            "Black level", min=0.0, max=0.5, step=0.005, initial_value=0.0)
        white_slider = server.gui.add_slider(
            "White level", min=0.5, max=1.0, step=0.005, initial_value=1.0)
        lowres_box = server.gui.add_checkbox(
            "Low-res preview", initial_value=False,
            hint=f"Downsample the preview to {LOWRES_WIDTH}px wide before encoding "
                 f"(vs {DISPLAY_WIDTH}px). Cheaper encode + much less bandwidth; "
                 f"capture and snapshots stay full-res.")
        reset_btn = server.gui.add_button("Reset tone curve")

    @curve_dropdown.on_update
    def _(_e):
        # Retarget the strength slider to the selected curve's range/default.
        cfg = curve_param_cfg[curve_dropdown.value]
        param_slider.min = cfg["min"]
        param_slider.max = cfg["max"]
        param_slider.step = cfg["step"]
        param_slider.value = cfg["default"]
        param_slider.hint = cfg["hint"]

    # -- Capture -----------------------------------------------------------
    with server.gui.add_folder("Capture"):
        snap_btn = server.gui.add_button("Save snapshot (linear16 + preview)")
        status_text = server.gui.add_text("Status", initial_value="—", disabled=True)

    speed_map = {"Low": enums.FRAME_SPEED_LOW, "Mid": enums.FRAME_SPEED_NORMAL,
                 "High": enums.FRAME_SPEED_HIGH}
    # last-pushed settings, so we only issue control writes on change
    pushed = {"ae": None, "exp": None, "gain": None, "speed": None}

    def apply_settings(cam):
        """Push only changed settings to the camera (avoids spamming writes)."""
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
        if state["cam"] is not None:
            return
        if not cams:
            conn_text.value = "no camera found"
            return
        serial = serial_by_label[cam_dropdown.value]
        conn_text.value = f"opening {serial}…"
        try:
            cam = HTCamera(serial=serial)
            lo, hi, step = cam.gain_range()
            gain_slider.min, gain_slider.max, gain_slider.step = lo, hi, step
            pushed.update(ae=None, exp=None, gain=None, speed=None)  # force re-push

            # Learn the native sensor size and set up the ROI sliders at full res.
            res = cam.current_resolution()
            fw = max(int(res.iWidthFOV), int(res.iWidth))
            fh = max(int(res.iHeightFOV), int(res.iHeight))
            state["full_w"], state["full_h"] = fw, fh
            roi_w.max, roi_w.value = fw, fw
            roi_h.max, roi_h.value = fh, fh
            roi_x.max, roi_x.value = fw, 0
            roi_y.max, roi_y.value = fh, 0
            roi_full_box.value = True
            roi_text.value = f"full {fw}x{fh}"

            state["cam"] = cam
            wb = "WB calibrated" if cam.wb_gains is not None else "no WB calib"
            conn_text.value = f"open: {cam.serial} ({cam.name}) [{wb}]"
        except Exception as exc:  # surface SDK errors in the GUI, don't crash
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

    def tone_kwargs():
        """Current tone-curve settings as tonemap_linear() keyword args.

        Includes the open camera's calibrated WB gains (None if uncalibrated) —
        folded into the tone LUT, so applying them is free per frame.
        """
        cam = state["cam"]
        return dict(curve=curve_dropdown.value, param=param_slider.value,
                    exposure=ev_slider.value, black=black_slider.value,
                    white=white_slider.value,
                    wb_gains=cam.wb_gains if cam else None)

    @reset_btn.on_click
    def _(_e):
        curve_dropdown.value = "bt709"          # also retargets param slider via on_update
        param_slider.value = 0.0
        ev_slider.value = 1.0
        black_slider.value = 0.0
        white_slider.value = 1.0

    @snap_btn.on_click
    def _(_e):
        lin = state["latest_linear"]
        if lin is None:
            status_text.value = "no frame yet"
            return
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        # The real data: full-depth linear RGB (12-bit values aligned to 16-bit).
        lin16 = lin << 4
        cv2.imwrite(f"snap_{ts}_linear16.png", cv2.cvtColor(lin16, cv2.COLOR_RGB2BGR))
        prev = convert.tonemap_linear(lin, **tone_kwargs())
        cv2.imwrite(f"snap_{ts}_preview.png", cv2.cvtColor(prev, cv2.COLOR_RGB2BGR))
        status_text.value = (f"saved snap_{ts}_linear16.png + _preview.png "
                             f"({lin.shape[1]}x{lin.shape[0]})")

    def _snap(v, step):
        """Round to the ROI grid (avoids the SDK rejecting off-grid windows)."""
        return int(round(v / step)) * step

    @roi_full_box.on_update
    def _(_e):
        # Greyed-out sliders when full-res; enabled when cropping.
        full = roi_full_box.value
        for s in (roi_w, roi_h, roi_x, roi_y):
            s.disabled = full
        if full:
            roi_w.value, roi_h.value = state["full_w"], state["full_h"]
            roi_x.value, roi_y.value = 0, 0
            _apply_roi()

    @roi_center_btn.on_click
    def _(_e):
        fw, fh = state["full_w"], state["full_h"]
        roi_x.value = max(0, _snap((fw - roi_w.value) / 2, ROI_STEP))
        roi_y.value = max(0, _snap((fh - roi_h.value) / 2, ROI_STEP))

    def _apply_roi():
        cam = state["cam"]
        if cam is None:
            roi_text.value = "open a camera first"
            return
        fw, fh = state["full_w"], state["full_h"]
        if roi_full_box.value:
            w, h, x, y = fw, fh, 0, 0
        else:
            # Clamp to a valid on-grid window inside the sensor.
            w = min(_snap(roi_w.value, ROI_STEP), fw)
            h = min(_snap(roi_h.value, ROI_STEP), fh)
            x = min(_snap(roi_x.value, ROI_STEP), fw - w)
            y = min(_snap(roi_y.value, ROI_STEP), fh - h)
        try:
            cam.set_roi(w, h, x, y)
            state["latest_linear"] = None  # old-size frame is stale
            roi_text.value = f"ROI {w}x{h} @ ({x},{y})  of {fw}x{fh}"
        except Exception as exc:  # surface SDK rejection, keep streaming
            roi_text.value = f"ROI failed: {exc}"

    @roi_apply_btn.on_click
    def _(_e):
        _apply_roi()

    print("Viser up — open the printed URL. Use 'Open' to connect a camera.")
    fps_t0 = time.time()
    frames = 0

    try:
        while not state["quit"]:
            cam = state["cam"]
            if cam is None:
                time.sleep(0.05)
                continue

            apply_settings(cam)
            lin, _info = cam.grab(timeout_ms=500)   # linear uint16 RGB, GET_NEWEST
            if lin is None:
                continue
            state["latest_linear"] = lin

            target_w = LOWRES_WIDTH if lowres_box.value else DISPLAY_WIDTH
            disp_lin = downscale(lin, target_w)
            disp = convert.tonemap_linear(disp_lin, **tone_kwargs())
            server.scene.set_background_image(disp, format="jpeg",
                                              jpeg_quality=JPEG_QUALITY)

            frames += 1
            now = time.time()
            if now - fps_t0 >= 1.0:
                fps = frames / (now - fps_t0)
                tag = "AE on" if ae_box.value else \
                    f"exp {exp_slider.value:.1f}ms gain {gain_slider.value:.2f}x"
                preview_tag = f"preview {disp.shape[1]}x{disp.shape[0]}" \
                    + ("  [low-res]" if lowres_box.value else "")
                status_text.value = (
                    f"{lin.shape[1]}x{lin.shape[0]}  {tag}  "
                    f"lin mean={lin.mean():.0f} max={int(lin.max())}  "
                    f"{curve_dropdown.value} {param_slider.value:.2f} ev {ev_slider.value:.1f}x  "
                    f"{preview_tag}  {fps:.1f} fps")
                fps_t0 = now
                frames = 0
    except KeyboardInterrupt:
        print("\nstopping (Ctrl-C)")
    finally:
        if state["cam"] is not None:
            state["cam"].close()
        print("camera released. safe to relaunch.")


if __name__ == "__main__":
    main()
