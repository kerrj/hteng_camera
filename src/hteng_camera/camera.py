"""HTCamera — a clean, fast interface to one HTENG/MindVision camera.

Design goals:
  * One ``HTCamera`` instance per physical camera. Two cameras = two instances,
    each with its own handle, control channel, and ISP state — fully independent.
    Select by serial number so identity is stable across replug/reboot.
  * Fast path only: the sensor streams raw 12-bit packed Bayer; we unpack +
    cv2-demosaic in :mod:`convert` (~10x faster than the SDK ISP). ``grab()``
    returns *linear* uint16 RGB — gamma is never baked in; encode for display
    with :func:`convert.to_display`.
  * Robust against this hardware's quirks: a USB control channel that comes up
    wedged (-52) is recovered in-session via CameraReConnect; handles are always
    released (atexit + SIGTERM/SIGINT + __del__) so a leaked handle never wedges
    the next run (-18 DEVICE_IS_OPENED).
"""

import atexit
import ctypes
import os
import signal
import threading
import time
from ctypes import POINTER, byref, c_double, c_float, c_int, c_ubyte, cast, c_void_p

import numpy as np

from . import _sdk
from . import enums
from . import convert
from ._sdk import (
    tSdkCameraDevInfo, tSdkFrameHead, tSdkImageResolution, _CapHead, check,
)

# 12-bit packed media types, richest-first preference for the fast path.
_PACKED12 = (
    enums.MEDIA_BAYGR12_PACKED, enums.MEDIA_BAYRG12_PACKED,
    enums.MEDIA_BAYGB12_PACKED, enums.MEDIA_BAYBG12_PACKED,
)


# ---------------------------------------------------------------------------
# Process-wide SDK init + handle bookkeeping (shared by all instances)
# ---------------------------------------------------------------------------

_sdk_init_lock = threading.Lock()
_sdk_initialised = False
_OPEN_HANDLES = set()


def _ensure_sdk_init():
    global _sdk_initialised
    with _sdk_init_lock:
        if not _sdk_initialised:
            check(_sdk.CameraSdkInit(1), "CameraSdkInit")
            _sdk_initialised = True


def _uninit_all():
    for h in list(_OPEN_HANDLES):
        try:
            _sdk.CameraUnInit(h)
        except Exception:
            pass
        _OPEN_HANDLES.discard(h)


atexit.register(_uninit_all)


def _install_signal_cleanup():
    """UnInit on SIGTERM/SIGINT, then chain to the previous handler/default."""
    def _make(prev):
        def _handler(signum, frame):
            _uninit_all()
            if callable(prev):
                prev(signum, frame)
            else:
                signal.signal(signum, signal.SIG_DFL)
                os.kill(os.getpid(), signum)
        return _handler

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, _make(signal.getsignal(sig)))
        except (ValueError, OSError):
            # not on the main thread; atexit + __del__ still cover us
            pass


_install_signal_cleanup()


# ---------------------------------------------------------------------------
# Enumeration
# ---------------------------------------------------------------------------

def list_cameras():
    """Enumerate attached cameras. Returns a list of dicts with keys:
    ``index``, ``serial``, ``name``, ``port``, ``sensor`` and the raw ``info``
    struct (needed to open). Order is the SDK's enumeration order — use
    ``serial`` for stable identity across two cameras.
    """
    _ensure_sdk_init()
    devs = (tSdkCameraDevInfo * 16)()
    n = c_int(16)
    check(_sdk.CameraEnumerateDevice(devs, byref(n)), "CameraEnumerateDevice")
    out = []
    for i in range(n.value):
        d = devs[i]
        out.append({
            "index": i,
            "serial": d.acSn.decode(errors="replace"),
            "name": d.acFriendlyName.decode(errors="replace"),
            "product": d.acProductName.decode(errors="replace"),
            "port": d.acPortType.decode(errors="replace"),
            "sensor": d.acSensorType.decode(errors="replace"),
            "info": d,
        })
    return out


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------

class HTCamera:
    """One physical HTENG camera. Open by serial (preferred) or index.

    Example::

        from hteng_camera import HTCamera, convert
        with HTCamera(serial="044162023020") as cam:
            cam.set_exposure_ms(15)
            cam.set_analog_gain(1.0)
            rgb16, info = cam.grab()             # linear uint16 (H, W, 3); info["time"] µs
            preview = convert.to_display(rgb16)  # sqrt-encoded uint8 for display
    """

    def __init__(self, serial=None, index=None, *, fov=None, frame_speed=None,
                 auto_exposure=False, open_now=True):
        self.h = 0
        self.info = None
        self.serial = None
        self.name = None
        self._playing = False
        self._media_index = None
        self._media_code = None
        self._cv_code = None
        # stash construction-time config applied during open
        self._init_fov = fov
        self._init_speed = frame_speed
        self._init_ae = auto_exposure
        self._want_serial = serial
        self._want_index = index
        if open_now:
            self.open()

    # -- lifecycle ---------------------------------------------------------

    def open(self):
        if self.h:
            return self
        cams = list_cameras()
        if not cams:
            raise RuntimeError("No HTENG/MindVision camera detected.")

        chosen = self._select(cams)
        self.info = chosen["info"]
        self.serial = chosen["serial"]
        self.name = chosen["name"]

        handle = c_int(0)
        check(_sdk.CameraInit(byref(self.info), -1, -1, byref(handle)),
              "CameraInit")
        self.h = handle.value
        _OPEN_HANDLES.add(self.h)

        # Establish a healthy control channel BEFORE any other control write.
        # The channel comes up all-good or all-bad (-52) per init; a write on a
        # bad channel knocks the device off the bus. _ctrl recovers in-session
        # via CameraReConnect. Setting continuous trigger doubles as the probe.
        self._ctrl(lambda: _sdk.CameraSetTriggerMode(self.h, 0),
                   "CameraSetTriggerMode")

        # Lock the sensor to a 12-bit packed Bayer transmission format — the
        # raw, linear, full-bit-depth feed the fast path is built on.
        self._select_packed12()

        # Apply construction-time config.
        self.set_ae(self._init_ae)
        if self._init_speed is not None:
            self.set_frame_speed(self._init_speed)
        if self._init_fov is not None:
            self.set_roi(*self._init_fov)
        return self

    def _select(self, cams):
        if self._want_serial is not None:
            for c in cams:
                if c["serial"] == self._want_serial:
                    return c
            avail = ", ".join(c["serial"] for c in cams)
            raise RuntimeError(
                f"No camera with serial {self._want_serial!r}. Present: {avail}")
        idx = 0 if self._want_index is None else self._want_index
        if idx >= len(cams):
            raise IndexError(
                f"Asked for index {idx} but only {len(cams)} camera(s) present.")
        return cams[idx]

    def close(self):
        if self.h:
            try:
                self.pause()
            except Exception:
                pass
            _sdk.CameraUnInit(self.h)
            _OPEN_HANDLES.discard(self.h)
            self.h = 0

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def play(self):
        if not self._playing:
            # Route through _ctrl so a wedged control channel (-52) recovers via
            # CameraReConnect instead of raising — same as every other write.
            self._ctrl(lambda: _sdk.CameraPlay(self.h), "CameraPlay")
            self._playing = True

    def pause(self):
        if self._playing:
            self._ctrl(lambda: _sdk.CameraPause(self.h), "CameraPause")
            self._playing = False

    # -- robust control writes --------------------------------------------

    def _ctrl(self, call, name, tries=20, settle=0.6):
        """Run a control write, recovering a wedged USB control channel in
        session via CameraReConnect. Most writes succeed on the first try (no
        delay); only a bad channel triggers the reconnect/retry loop.
        """
        last = None
        for _ in range(tries):
            rc = call()
            if rc == 0:
                return
            last = rc
            _sdk.CameraReConnect(self.h)
            time.sleep(settle)
        raise RuntimeError(
            f"{name} failed (status {last}) after {tries} retries — unplug/"
            f"replug the camera (control channel wedged).")

    # -- transmission format ----------------------------------------------

    def media_types(self):
        """List the camera's RAW transmission formats (index/code/bits/desc)."""
        buf = (c_ubyte * 8192)()
        check(_sdk.CameraGetCapability(self.h, cast(buf, c_void_p)),
              "CameraGetCapability")
        cap = cast(buf, POINTER(_CapHead)).contents
        out = []
        for i in range(cap.iMediaTypdeDesc):
            mt = cap.pMediaTypeDesc[i]
            code = int(mt.iMediaType)
            out.append({
                "index": int(mt.iIndex),
                "code": code,
                "bits": (code >> 16) & 0xFF,
                "desc": mt.acDescription.decode(errors="replace"),
            })
        return out

    def _select_packed12(self):
        """Pick a 12-bit packed Bayer media type and cache its cv2 demosaic code."""
        types = self.media_types()
        packed = [m for m in types if m["code"] in _PACKED12]
        if not packed:
            avail = ", ".join(f"{m['desc']}(0x{m['code']:08X})" for m in types)
            raise RuntimeError(
                "Camera does not expose a 12-bit packed Bayer format (required "
                f"for the fast path). Available: {avail}")
        chosen = packed[0]
        self._set_media_type(chosen["index"], chosen["code"])

    def _set_media_type(self, index, code):
        was_playing = self._playing
        if was_playing:
            self.pause()
        self._ctrl(lambda: _sdk.CameraSetMediaType(self.h, int(index)),
                   "CameraSetMediaType")
        self._media_index = int(index)
        self._media_code = int(code)
        self._cv_code = enums.cv_bayer_code(code)
        if self._cv_code is None:
            raise RuntimeError(f"No demosaic mapping for media 0x{code:08X}")
        if was_playing:
            self.play()

    # -- settings ----------------------------------------------------------

    def set_ae(self, enabled):
        self._ctrl(lambda: _sdk.CameraSetAeState(self.h, 1 if enabled else 0),
                   "CameraSetAeState")

    def set_exposure_ms(self, ms):
        self._ctrl(lambda: _sdk.CameraSetExposureTime(self.h, float(ms) * 1000.0),
                   "CameraSetExposureTime")

    def get_exposure_ms(self):
        us = c_double(0.0)
        check(_sdk.CameraGetExposureTime(self.h, byref(us)), "CameraGetExposureTime")
        return us.value / 1000.0

    def set_analog_gain(self, x):
        """Analog (sensor) gain as an x-multiplier (1.0 .. ~22.0)."""
        self._ctrl(lambda: _sdk.CameraSetAnalogGainX(self.h, float(x)),
                   "CameraSetAnalogGainX")

    def get_analog_gain(self):
        x = c_float(0.0)
        check(_sdk.CameraGetAnalogGainX(self.h, byref(x)), "CameraGetAnalogGainX")
        return x.value

    def gain_range(self):
        """(min_x, max_x, step) analog-gain multipliers."""
        lo, hi, step = c_float(0.0), c_float(0.0), c_float(0.0)
        check(_sdk.CameraGetAnalogGainXRange(self.h, byref(lo), byref(hi), byref(step)),
              "CameraGetAnalogGainXRange")
        return lo.value, hi.value, step.value

    def set_frame_speed(self, mode):
        self._ctrl(lambda: _sdk.CameraSetFrameSpeed(self.h, int(mode)),
                   "CameraSetFrameSpeed")

    def set_roi(self, fov_w, fov_h, h_offset=0, v_offset=0):
        """Set sensor ROI (crop). FOV-shaped output, no hardware binning."""
        was_playing = self._playing
        if was_playing:
            self.pause()
        res = tSdkImageResolution()
        res.iIndex = 0xFF
        res.iHOffsetFOV = h_offset
        res.iVOffsetFOV = v_offset
        res.iWidthFOV = fov_w
        res.iHeightFOV = fov_h
        res.iWidth = fov_w
        res.iHeight = fov_h
        res.uBinSumMode = 0
        self._ctrl(lambda: _sdk.CameraSetImageResolution(self.h, byref(res)),
                   "CameraSetImageResolution")
        if was_playing:
            self.play()

    def current_resolution(self):
        res = tSdkImageResolution()
        check(_sdk.CameraGetImageResolution(self.h, byref(res)),
              "CameraGetImageResolution")
        return res

    # -- grabbing ----------------------------------------------------------

    def grab_bayer12(self, timeout_ms=500, priority=enums.GET_NEWEST):
        """Grab one raw 12-bit Bayer frame, unpacked to a uint16 (H, W) plane
        (0..4095). No demosaic.

        Returns ``(bayer, info)``. ``info`` is a dict carrying per-frame metadata
        — currently ``{"time": <microseconds>}``, the camera's hardware frame
        timestamp (frame-head counter, 100 µs resolution, promoted to µs). Reset
        the counter with :meth:`reset_timestamp` to establish a common time base
        across cameras. On timeout returns ``(None, {})``.
        """
        self.play()
        head = tSdkFrameHead()
        raw_ptr = POINTER(c_ubyte)()
        if _sdk.CameraGetImageBufferPriority(
                self.h, byref(head), byref(raw_ptr), timeout_ms, priority) != 0:
            return None, {}
        try:
            w, h, nb = head.iWidth, head.iHeight, head.uBytes
            if nb * 2 != w * h * 3:
                raise RuntimeError(
                    f"expected 12-bit packed (1.5 B/px); got {nb/(w*h):.2f} B/px "
                    f"(media 0x{head.uiMediaType:08X}).")
            raw = np.frombuffer(
                (c_ubyte * nb).from_address(ctypes.addressof(raw_ptr.contents)),
                dtype=np.uint8, count=nb)
            bayer = convert.unpack_bayer12_packed(raw, w, h)
            # uiTimeStamp is in 0.1 ms units; promote to integer microseconds.
            info = {"time": int(head.uiTimeStamp) * 100}
        finally:
            _sdk.CameraReleaseImageBuffer(self.h, raw_ptr)
        return bayer, info

    def grab(self, timeout_ms=2000, drop_first=0, priority=enums.GET_NEWEST,
             align_to_16bit=False):
        """Grab one **linear** RGB frame as (H, W, 3) uint16 — the primary path.

        Raw 12-bit Bayer is unpacked and cv2-demosaiced (fast path). The result
        is linear (no gamma); encode for display with :func:`convert.to_display`.

        align_to_16bit: left-shift to the full 0..65520 range (for 16-bit PNG
        export). Default keeps native 0..4095.

        Returns ``(rgb, info)`` — see :meth:`grab_bayer12` for ``info`` (carries
        the per-frame ``"time"`` in microseconds). On timeout returns
        ``(None, {})``.
        """
        for _ in range(drop_first):
            self.grab_bayer12(timeout_ms, priority)
        bayer, info = self.grab_bayer12(timeout_ms, priority)
        if bayer is None:
            return None, {}
        rgb = convert.demosaic(bayer, self._cv_code, align_to_16bit=align_to_16bit)
        return rgb, info

    def reset_timestamp(self):
        """Reset this camera's hardware frame-timestamp counter to 0.

        To pair frames across two cameras with no wiring: call this on both
        cameras back-to-back from the same thread to establish a shared time
        base, then match frames by the ``"time"`` value in each grab's info dict.
        Sync accuracy is bounded by the gap between the two reset calls
        (~50-200 µs) plus per-frame USB jitter — typically 1-2 ms on a quiet
        Linux host. For true simultaneous exposure, use a hardware trigger.
        """
        self._ctrl(lambda: _sdk.CameraRstTimeStamp(self.h), "CameraRstTimeStamp")
