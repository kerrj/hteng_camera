#!/usr/bin/env python3
"""Validate the software AutoExposure controller against a live camera.

Runs the gain-first controller in a tight grab loop and prints, once a second,
the metered brightness, the smoothed EMA, and the exposure/gain the controller
settled on — so you can watch it converge and check it stays calm (no pumping)
while you wave the camera around.

  python examples/test_auto_exposure.py                 # first camera
  python examples/test_auto_exposure.py --serial 046060323001 --target 0.35
"""

import argparse
import time

from hteng_camera import HTCamera, enums, list_cameras, AutoExposure


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--serial", default=None, help="Camera serial (default: first)")
    ap.add_argument("--target", type=float, default=0.35, help="Target brightness 0..1")
    ap.add_argument("--exp-min", type=float, default=5.0, help="Exposure floor (ms)")
    ap.add_argument("--exp-max", type=float, default=17.0, help="Exposure cap (ms)")
    ap.add_argument("--gain-max", type=float, default=4.0, help="Max analog gain (x)")
    ap.add_argument("--flicker-hz", type=float, default=60.0,
                    help="Mains freq for anti-flicker (60 US / 50 EU), 0 to disable")
    ap.add_argument("--responsiveness", type=float, default=0.25,
                    help="Loop gain (0,1]: fraction of error corrected per step")
    ap.add_argument("--duration", type=float, default=None, help="Seconds (default: Ctrl+C)")
    args = ap.parse_args()

    serial = args.serial
    if serial is None:
        cams = list_cameras()
        if not cams:
            raise SystemExit("no camera found")
        serial = cams[0]["serial"]

    cam = HTCamera(serial=serial, demosaic_quality="ea")
    cam.set_frame_speed(enums.FRAME_SPEED_HIGH)
    cam.set_ae(False)                       # we drive exposure/gain ourselves

    gain_lo, gain_hi, _ = cam.gain_range()
    exp, gain = args.exp_min, 1.0
    cam.set_exposure_ms(exp)
    cam.set_analog_gain(gain)

    ae = AutoExposure(gain_lo, min(gain_hi, args.gain_max),
                      exp_min=args.exp_min, exp_max=args.exp_max,
                      target=args.target, responsiveness=args.responsiveness,
                      flicker_hz=args.flicker_hz)

    print(f"[info] {serial}: gain {gain_lo:.1f}..{min(gain_hi, args.gain_max):.1f}x, "
          f"target {args.target:.2f}, exp {args.exp_min}..{args.exp_max} ms, "
          f"anti-flicker {args.flicker_hz:g} Hz → steps {ae._exposure_steps()} ms")
    print(f"[info] {'t':>6}  {'meas':>6} {'err':>6}  {'exp_ms':>7} {'gain_x':>7}  note")

    t0 = time.monotonic()
    last_print = 0.0
    last_info = None
    grabs = 0
    try:
        while True:
            bayer, _ = cam.grab_bayer12(timeout_ms=500)
            if bayer is None:
                continue
            grabs += 1
            new_exp, new_gain, info = ae.update(bayer, exp, gain)
            # apply() every frame: it counts down any deferred (light-increasing)
            # write so a step-switch dims briefly rather than flashing bright.
            exp, gain = ae.apply(cam, new_exp, new_gain, exp, gain)
            last_info = info

            now = time.monotonic()
            if now - last_print >= 1.0:
                note = "→adjusting" if info["changed"] else "settled"
                print(f"      {now - t0:6.1f}  {info['measured']:6.3f} "
                      f"{info['error']:+6.3f}  {exp:7.2f} {gain:7.2f}  {note} "
                      f"({grabs} grabs/s)")
                last_print = now
                grabs = 0
            if args.duration is not None and now - t0 >= args.duration:
                break
    except KeyboardInterrupt:
        print("\n[info] stopping")
    finally:
        cam.close()
        if last_info is not None:
            print(f"[info] final: exp {exp:.2f} ms, gain {gain:.2f}x, "
                  f"meas {last_info['measured']:.3f} (target {args.target:.2f})")


if __name__ == "__main__":
    main()
