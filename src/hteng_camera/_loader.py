"""Locate and load the vendored MindVision/HTENG shared library.

The SDK's ABI is identical across platforms (same headers, same struct layout),
so the *only* platform-specific decision is which binary file to dlopen. We ship
the vendor binaries inside the package (``_libs/<platform>-<arch>/``) and load
them straight from there — no system install, no copy to ``/lib``, nothing that
needs root. That is what makes the package ``pip install``-and-go.

Linux is the primary target; macOS (Apple Silicon) is supported for free because
the binding layer above this file is platform-agnostic.

Override the auto-pick with the ``HTENG_LIB`` env var (absolute path to a
libMVSDK.so / libmvsdk.dylib), e.g. to test a newer vendor drop in place.
"""

import ctypes
import os
import platform
from pathlib import Path

_LIBS_DIR = Path(__file__).resolve().parent / "_libs"

# (system, machine-aliases) -> (subdir, filename)
_PLATFORM_MAP = {
    ("Linux", ("x86_64", "amd64")): ("linux-x86_64", "libMVSDK.so"),
    ("Linux", ("aarch64", "arm64")): ("linux-aarch64", "libMVSDK.so"),
    ("Darwin", ("arm64",)): ("darwin-arm64", "libmvsdk.dylib"),
}


def _candidate_path():
    system = platform.system()
    machine = platform.machine().lower()
    for (sys_name, machines), (subdir, fname) in _PLATFORM_MAP.items():
        if system == sys_name and machine in machines:
            return _LIBS_DIR / subdir / fname
    return None


def library_path():
    """Resolve the shared-library path that :func:`load` would use.

    Honours ``HTENG_LIB``; otherwise the bundled binary for this platform/arch.
    Raises a clear error if the platform is unsupported or the file is missing.
    """
    override = os.environ.get("HTENG_LIB")
    if override:
        p = Path(override)
        if not p.exists():
            raise FileNotFoundError(f"HTENG_LIB points at a missing file: {p}")
        return p

    p = _candidate_path()
    if p is None:
        raise RuntimeError(
            f"No bundled HTENG SDK binary for {platform.system()}/"
            f"{platform.machine()}. Supported: Linux x86_64, Linux aarch64, "
            f"macOS arm64. Set HTENG_LIB to a libMVSDK.so to override."
        )
    if not p.exists():
        raise FileNotFoundError(
            f"Bundled SDK binary missing at {p} — the package may be built "
            f"without binaries for this platform."
        )
    return p


def load():
    """dlopen the vendor library and return the ``ctypes.CDLL`` handle."""
    path = library_path()
    try:
        return ctypes.CDLL(str(path))
    except OSError as e:  # pragma: no cover - environment dependent
        raise OSError(
            f"Failed to load HTENG SDK from {path}: {e}\n"
            f"On Linux, ensure libusb-1.0 is present (apt install libusb-1.0-0) "
            f"and the udev rules are installed (see packaging/)."
        ) from e
