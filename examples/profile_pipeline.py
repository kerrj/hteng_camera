"""Profile the fast-path capture pipeline, stage by stage (no viser).

Breaks one frame into its constituent steps and times each over N frames:

    capture   CameraGetImageBufferPriority — waits for the next frame; bounded
              below by exposure time + USB transfer (NOT cpu-bound)
    unpack    12-bit packed Bayer bytes -> uint16 Bayer plane (numpy)
    release   hand the SDK buffer back to the kernel
    demosaic  Bayer -> linear uint16 RGB (cv2 SIMD)
    display   tonemap_linear: linear -> uint8 with gamma (sqrt) encode

The capture stage is dominated by exposure, so lower --exposure to see the true
CPU cost of the rest. Use --no-display to skip the gamma encode (it's a preview-
only step). Times are per-frame; the summary shows mean/median/min/max + share
of the CPU-side total (everything except capture).

    python examples/profile_pipeline.py
    python examples/profile_pipeline.py --serial 044162023020 --exposure 2 --frames 300
"""

import argparse
import ctypes
import time
from ctypes import POINTER, byref, c_ubyte

import numpy as np

from hteng_camera import HTCamera, convert, enums
from hteng_camera import _sdk
from hteng_camera._sdk import tSdkFrameHead


def parse():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--serial", default=None, help="camera serial (default: first found)")
    p.add_argument("--index", type=int, default=None, help="camera index fallback")
    p.add_argument("--exposure", type=float, default=5.0, help="exposure ms (default 5)")
    p.add_argument("--gain", type=float, default=1.0, help="analog gain x (default 1)")
    p.add_argument("--frames", type=int, default=200, help="frames to time (default 200)")
    p.add_argument("--warmup", type=int, default=15, help="warmup frames (default 15)")
    p.add_argument("--gamma", type=float, default=2.0, help="display gamma (default 2.0=sqrt)")
    p.add_argument("--no-display", action="store_true", help="skip the tonemap_linear stage")
    p.add_argument("--timeout", type=int, default=2000, help="grab timeout ms")
    return p.parse_args()


def summarize(name, samples_ms, total_cpu_ms):
    a = np.asarray(samples_ms)
    share = (a.mean() / total_cpu_ms * 100.0) if total_cpu_ms else 0.0
    return (f"  {name:<9} mean {a.mean():7.3f}  med {np.median(a):7.3f}  "
            f"min {a.min():7.3f}  max {a.max():7.3f}  std {a.std():6.3f} ms"
            f"  ({share:4.1f}% cpu)")


def main():
    a = parse()

    with HTCamera(serial=a.serial, index=a.index) as cam:
        cam.set_ae(False)
        cam.set_exposure_ms(a.exposure)
        cam.set_analog_gain(a.gain)
        cam.set_frame_speed(enums.FRAME_SPEED_HIGH)
        cam.play()

        mt = next((m for m in cam.media_types() if m["code"] == cam._media_code), None)
        print(f"camera : {cam.serial} ({cam.name})")
        print(f"format : {mt['desc'] if mt else '?'}  (0x{cam._media_code:08X})")
        print(f"config : exposure {a.exposure} ms  gain {a.gain}x  speed High  "
              f"frames {a.frames} (+{a.warmup} warmup)")
        print(f"display: gamma {a.gamma}{'  [skipped]' if a.no_display else ''}\n")

        # per-stage timing buffers
        t_capture, t_unpack, t_release = [], [], []
        t_demosaic, t_display, t_frame = [], [], []

        def grab_raw(timeout_ms):
            """Mirror grab_bayer12's buffer fetch, but timed in pieces.

            Returns (bayer_plane, w, h) or None on timeout.
            """
            head = tSdkFrameHead()
            raw_ptr = POINTER(c_ubyte)()
            t0 = time.perf_counter()
            rc = _sdk.CameraGetImageBufferPriority(
                cam.h, byref(head), byref(raw_ptr), timeout_ms, enums.GET_NEWEST)
            t1 = time.perf_counter()
            if rc != 0:
                return None
            try:
                w, h, nb = head.iWidth, head.iHeight, head.uBytes
                raw = np.frombuffer(
                    (c_ubyte * nb).from_address(ctypes.addressof(raw_ptr.contents)),
                    dtype=np.uint8, count=nb)
                t2 = time.perf_counter()
                bayer = convert.unpack_bayer12_packed(raw, w, h)
                t3 = time.perf_counter()
            finally:
                _sdk.CameraReleaseImageBuffer(cam.h, raw_ptr)
            t4 = time.perf_counter()
            t_capture.append((t1 - t0) * 1e3)
            t_unpack.append((t3 - t2) * 1e3)
            t_release.append((t4 - t3) * 1e3)
            return bayer, w, h

        # warmup (let exposure/gain settle, prime caches, JIT cv2 kernels)
        for _ in range(a.warmup):
            r = grab_raw(a.timeout)
            if r is not None:
                convert.demosaic(r[0], cam._cv_code)
        t_capture.clear(); t_unpack.clear(); t_release.clear()

        misses = 0
        for _ in range(a.frames):
            f0 = time.perf_counter()
            r = grab_raw(a.timeout)
            if r is None:
                misses += 1
                continue
            bayer, w, h = r

            d0 = time.perf_counter()
            rgb = convert.demosaic(bayer, cam._cv_code)
            d1 = time.perf_counter()
            t_demosaic.append((d1 - d0) * 1e3)

            if not a.no_display:
                p0 = time.perf_counter()
                convert.tonemap_linear(rgb, gamma=a.gamma)
                p1 = time.perf_counter()
                t_display.append((p1 - p0) * 1e3)

            t_frame.append((time.perf_counter() - f0) * 1e3)

    # ---- report ----
    n = len(t_frame)
    if n == 0:
        print("No frames captured — check the camera / timeout.")
        return

    cpu_stage_means = [np.mean(t_unpack), np.mean(t_release), np.mean(t_demosaic)]
    if t_display:
        cpu_stage_means.append(np.mean(t_display))
    total_cpu = float(sum(cpu_stage_means))

    print(f"timed {n} frames ({misses} misses)\n")
    print(summarize("capture", t_capture, 0))
    print("  " + "-" * 72)
    print(summarize("unpack", t_unpack, total_cpu))
    print(summarize("release", t_release, total_cpu))
    print(summarize("demosaic", t_demosaic, total_cpu))
    if t_display:
        print(summarize("display", t_display, total_cpu))
    print("  " + "-" * 72)

    frame = np.asarray(t_frame)
    cap_mean = np.mean(t_capture)
    print(f"  {'cpu sum':<9} mean {total_cpu:7.3f} ms  "
          f"(unpack+release+demosaic{'+display' if t_display else ''})")
    print(f"  {'frame':<9} mean {frame.mean():7.3f}  med {np.median(frame):7.3f}  "
          f"min {frame.min():7.3f}  max {frame.max():7.3f} ms  "
          f"(capture {cap_mean:.2f} + cpu {total_cpu:.2f})")
    print(f"\n  throughput: {1000.0 / frame.mean():6.1f} fps end-to-end   "
          f"| cpu-only ceiling: {1000.0 / total_cpu:6.1f} fps")
    print(f"  (capture is exposure/USB-bound; lower --exposure to expose CPU cost)")


if __name__ == "__main__":
    main()
