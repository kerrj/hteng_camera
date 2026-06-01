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
        _lib = lib
        available = True
    except OSError:
        # Missing/incompatible binary — stay on the numpy fallback silently.
        _lib = None
        available = False


_try_load()


def version():
    return _lib.hteng_fast_version() if available else None
