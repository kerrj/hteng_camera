"""Bind the optional native acceleration kernel (libhteng_fast).

This is a *speed* layer, not a correctness one: every native kernel has a numpy
fallback that produces bit-identical output, so the package works whether or not
the binary is present. We load it from the same per-platform ``_libs`` dir as
the vendor SDK lib, so a built wheel ships it and the target needs no compiler.

Build it (for development) with ``native/build.sh``. ``HTENG_NO_NATIVE=1``
forces the numpy fallback (handy for benchmarking the two against each other).
"""

import ctypes
import os
import platform
from ctypes import POINTER, c_int, c_size_t, c_ubyte, c_uint16
from pathlib import Path

_LIBS_DIR = Path(__file__).resolve().parent / "_libs"

# Same platform map as _loader, but for *our* kernel's filename.
_PLATFORM_MAP = {
    ("Linux", ("x86_64", "amd64")): ("linux-x86_64", "libhteng_fast.so"),
    ("Linux", ("aarch64", "arm64")): ("linux-aarch64", "libhteng_fast.so"),
    ("Darwin", ("arm64",)): ("darwin-arm64", "libhteng_fast.dylib"),
}

_lib = None
available = False

# Thread cap passed to the native gamma kernel (0 = let it auto-pick all cores).
# convert.set_num_threads keeps this in sync with cv2's cap.
_num_threads = 0


def set_num_threads(n):
    """Set the thread count the native kernels use (0 = auto / all cores)."""
    global _num_threads
    _num_threads = max(0, int(n))


def num_threads():
    return _num_threads


def _candidate_path():
    system = platform.system()
    machine = platform.machine().lower()
    for (sys_name, machines), (subdir, fname) in _PLATFORM_MAP.items():
        if system == sys_name and machine in machines:
            return _LIBS_DIR / subdir / fname
    return None


def _try_load():
    global _lib, available
    if os.environ.get("HTENG_NO_NATIVE"):
        return
    p = _candidate_path()
    if p is None or not p.exists():
        return
    try:
        lib = ctypes.CDLL(str(p))
        lib.hteng_fast_version.restype = c_int
        lib.hteng_fast_version.argtypes = []
        lib.hteng_apply_lut_u16.restype = None
        lib.hteng_apply_lut_u16.argtypes = [
            POINTER(c_uint16),  # in
            POINTER(c_ubyte),   # out
            c_size_t,           # n
            POINTER(c_ubyte),   # lut
            c_int,              # lut_size
            c_int,              # n_threads
        ]
        lib.hteng_apply_lut_u16_u16.restype = None
        lib.hteng_apply_lut_u16_u16.argtypes = [
            POINTER(c_uint16),  # in
            POINTER(c_uint16),  # out
            c_size_t,           # n
            POINTER(c_uint16),  # lut (65536 entries)
            c_int,              # n_threads
        ]
        lib.hteng_unpack_bayer12.restype = None
        lib.hteng_unpack_bayer12.argtypes = [
            POINTER(c_ubyte),   # in  (packed bytes)
            POINTER(c_uint16),  # out (bayer plane)
            c_size_t,           # n_pixels
        ]
        # Per-channel (RGB) LUT kernels: lut = 3 concatenated 65536-entry tables.
        lib.hteng_apply_lut3_u16.restype = None
        lib.hteng_apply_lut3_u16.argtypes = [
            POINTER(c_uint16),  # in (interleaved RGB)
            POINTER(c_ubyte),   # out
            c_size_t,           # n_pixels
            POINTER(c_ubyte),   # lut (3 * 65536)
            c_int,              # n_threads
        ]
        lib.hteng_apply_lut3_u16_u16.restype = None
        lib.hteng_apply_lut3_u16_u16.argtypes = [
            POINTER(c_uint16),  # in (interleaved RGB)
            POINTER(c_uint16),  # out
            c_size_t,           # n_pixels
            POINTER(c_uint16),  # lut (3 * 65536)
            c_int,              # n_threads
        ]
        _lib = lib
        available = True
    except (OSError, AttributeError):
        # Missing/incompatible binary, or an older build lacking a newer symbol —
        # stay on the numpy fallback silently.
        _lib = None
        available = False


_try_load()


def version():
    return _lib.hteng_fast_version() if available else None
