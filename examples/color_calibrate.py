"""Color calibration GUI in viser — white balance now, ColorChecker/CCM later.

Point the camera at a neutral reference (gray card, or good white paper) under
the *working* illuminant, put the on-screen patch box over it, and Measure. WB
gains are computed in linear light from the patch means:

    gain_c = max(mean_R, mean_G, mean_B) / mean_c        (min gain = 1.0)

normalized so no channel is attenuated — a sensor-clipped highlight then stays
clipped-white after WB instead of picking up a tint. After measuring, the
preview shows the gains applied live; Save merges them into that sensor's
``calib_<serial>.json`` (intrinsics from the ChArUco tool are preserved) and
writes the raw measurement frame as a 16-bit PNG next to it.

Exposure discipline: the patch should sit at ~40-70% of full scale. The GUI
shows the patch level and warns outside that band — too dark is noisy, too
bright risks per-channel clipping which silently skews the ratios.

Gains are only valid under the measured illuminant (they're a von Kries
correction, not magic) — note it in the Illuminant field.

Run::

    pip install hteng-camera[viser]
    python examples/color_calibrate.py [--serial 046060323003]
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import viser

from hteng_camera import (
    CameraCalibration, ColorCalibration, HTCamera, convert, list_cameras,
)

JPEG_QUALITY = 85
DISPLAY_WIDTH = 1280
PATCH_LO, PATCH_HI = 0.40, 0.70   # healthy patch level, fraction of full scale


def measure_wb(lin, frac):
    """WB gains + diagnostics from the centre patch of a linear RGB frame.

    ``frac`` is the patch edge as a fraction of the frame's short side.
    Returns (gains(3,), means(3,), level, clipped_frac).
    """
    h, w = lin.shape[:2]
    half = int(min(h, w) * frac / 2)
    cy, cx = h // 2, w // 2
    patch = lin[cy - half:cy + half, cx - half:cx + half].astype(np.float64)
    means = patch.reshape(-1, 3).mean(axis=0)
    gains = means.max() / np.maximum(means, 1e-6)
    level = float(means.max() / 4095.0)
    clipped = float((patch >= 4080).any(axis=2).mean())
    return gains, means, level, clipped


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--serial", default=None, help="camera serial (default: first)")
    ap.add_argument("--calib-dir", default="calibrations",
                    help="where to publish calib_<serial>.json (git-tracked; the "
                         "library loads from here). The raw measurement PNG lands "
                         "here too but is gitignored.")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()

    cams = list_cameras()
    if not cams:
        raise SystemExit("No camera found.")
    serial = args.serial or cams[0]["serial"]

    print(f"[info] opening {serial}…")
    cam = HTCamera(serial=serial)
    cam.set_ae(False)

    server = viser.ViserServer(port=args.port)

    with server.gui.add_folder("Exposure"):
        exp_slider = server.gui.add_slider(
            "Exposure (ms)", min=0.1, max=200.0, step=0.1, initial_value=15.0)
        gain_slider = server.gui.add_slider(
            "Analog gain (x)", min=1.0, max=22.0, step=0.1, initial_value=1.0)
        level_text = server.gui.add_text("Patch level", initial_value="—",
                                         disabled=True)

    with server.gui.add_folder("White balance"):
        patch_slider = server.gui.add_slider(
            "Patch size", min=0.05, max=0.6, step=0.05, initial_value=0.2,
            hint="Patch edge as a fraction of the frame's short side. Keep the "
                 "neutral reference fully inside the green box.")
        illum_text = server.gui.add_text(
            "Illuminant", initial_value="",
            hint="What light the reference is under (e.g. 'sunlight', "
                 "'bench LED'). Gains are only valid under this light.")
        measure_btn = server.gui.add_button("Measure WB from patch")
        apply_box = server.gui.add_checkbox(
            "Preview with gains applied", initial_value=True)
        gains_text = server.gui.add_text("Gains (R,G,B)", initial_value="—",
                                         disabled=True)
        save_btn = server.gui.add_button("Save to calib_<serial>.json")
        status_text = server.gui.add_text("Status", initial_value="—", disabled=True)

    state = {"gains": None, "frame_at_measure": None, "do_measure": False}
    pushed = {"exp": None, "gain": None}

    @measure_btn.on_click
    def _(_e):
        state["do_measure"] = True

    @save_btn.on_click
    def _(_e):
        """Publish WB gains into calibrations/calib_<serial>.json.

        Merges into the existing file for this serial, so intrinsics from the
        ChArUco tool are preserved. The library auto-loads from calibrations/,
        so the next HTCamera open white-balances automatically.
        """
        if state["gains"] is None:
            status_text.value = "measure first"
            return
        pubdir = Path(args.calib_dir)
        pubdir.mkdir(parents=True, exist_ok=True)
        cal = CameraCalibration.load_or_new(serial, dir=pubdir)
        cal.color = ColorCalibration(wb_gains=state["gains"],
                                     wb_illuminant=illum_text.value)
        path = cal.save(dir=pubdir)
        # Raw measurement frame: the evidence behind the gains, re-measurable.
        # Untracked (the published JSON is the deliverable; the PNG is provenance).
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        png = pubdir / f"wb_measurement_{serial}_{ts}_lin16.png"
        cv2.imwrite(str(png), cv2.cvtColor(
            state["frame_at_measure"] << 4, cv2.COLOR_RGB2BGR))
        status_text.value = f"published {path.name} -> {pubdir}/  (+ {png.name})"
        print(f"[save] published {path}  (gains {np.round(state['gains'], 4).tolist()}, "
              f"illuminant {illum_text.value!r})  + {png.name}")

    print("Viser up — open the printed URL. Aim the green box at the neutral "
          "reference, set exposure so the patch sits in the green band, Measure, "
          "then Save.")

    try:
        while True:
            if exp_slider.value != pushed["exp"]:
                cam.set_exposure_ms(exp_slider.value); pushed["exp"] = exp_slider.value
            if gain_slider.value != pushed["gain"]:
                cam.set_analog_gain(gain_slider.value); pushed["gain"] = gain_slider.value

            lin, _info = cam.grab(timeout_ms=500)
            if lin is None:
                continue

            gains, means, level, clipped = measure_wb(lin, patch_slider.value)
            band = ("CLIPPING" if clipped > 0.001 else
                    "too bright" if level > PATCH_HI else
                    "too dark" if level < PATCH_LO else "ok")
            level_text.value = (f"{level*100:.0f}% of full scale [{band}]  "
                                f"RGB {means.round(0).astype(int).tolist()}")

            if state["do_measure"]:
                state["do_measure"] = False
                if clipped > 0.001:
                    status_text.value = (f"refusing: {clipped*100:.1f}% of patch "
                                         "clipped — lower exposure")
                elif level < 0.15:
                    status_text.value = "refusing: patch too dark (<15%) — raise exposure"
                else:
                    state["gains"] = gains
                    state["frame_at_measure"] = lin
                    gains_text.value = f"{gains.round(4).tolist()}"
                    status_text.value = (f"measured at {level*100:.0f}% level"
                                         + ("" if band == "ok" else f" [{band}]"))

            # Preview: optional live WB, tone curve, patch box.
            disp_lin = lin
            if apply_box.value and state["gains"] is not None:
                disp_lin = np.clip(lin * state["gains"], 0, 4095).astype(np.uint16)
            disp = convert.tonemap_linear(disp_lin, curve="bt709")
            h, w = disp.shape[:2]
            if w > DISPLAY_WIDTH:
                disp = cv2.resize(disp, (DISPLAY_WIDTH, round(h * DISPLAY_WIDTH / w)),
                                  interpolation=cv2.INTER_AREA)
            dh, dw = disp.shape[:2]
            half = int(min(dh, dw) * patch_slider.value / 2)
            cv2.rectangle(disp, (dw // 2 - half, dh // 2 - half),
                          (dw // 2 + half, dh // 2 + half), (0, 255, 0), 2)
            server.scene.set_background_image(disp, format="jpeg",
                                              jpeg_quality=JPEG_QUALITY)
    except KeyboardInterrupt:
        print("\nstopping (Ctrl-C)")
    finally:
        cam.close()
        print("camera released.")


if __name__ == "__main__":
    main()
