"""Per-sensor calibration data: intrinsics + color, one JSON file per camera.

Calibration is keyed by camera *serial* (stable across replug/reboot, like
everything else in this package), one file per physical sensor:

    calib_<serial>.json     CameraCalibration  (intrinsics + color)
    stereo_<L>_<R>.json     StereoCalibration  (cam2cam pose — a pair property)

Different tools fill in different pieces at different times — the ChArUco GUI
writes ``intrinsics``, the color GUI writes ``color`` — so the intended flow is
load-or-new, set your piece, save: an existing file's other sections are kept.

The JSON is versioned (``format`` tag) and all arrays are plain nested lists,
so files are stable, diffable, and readable from any language.
"""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

FORMAT_TAG = "hteng-camera-calibration/1"
STEREO_FORMAT_TAG = "hteng-camera-stereo/1"
STEREO_CONVENTION = "X_right = R @ X_left + t  (t in board units)"


def search_dirs():
    """Directories searched (in order) for calib_<serial>.json.

    1. ``HTENG_CALIB_DIR`` env var, if set
    2. ``./calibrations`` — the git-tracked, image-free home for published
       calibrations (session folders keep the raw images, untracked)
    3. ``.`` — the CWD itself
    """
    dirs = []
    env = os.environ.get("HTENG_CALIB_DIR")
    if env:
        dirs.append(Path(env))
    dirs += [Path("calibrations"), Path(".")]
    return dirs


def find(serial):
    """Locate + load the calibration for ``serial`` from :func:`search_dirs`.

    Returns a :class:`CameraCalibration` or None. Unreadable files are skipped
    (with a warning) rather than raised — a corrupt published file shouldn't
    make every camera open fail.
    """
    for d in search_dirs():
        p = CameraCalibration.default_path(serial, d)
        if p.exists():
            try:
                return CameraCalibration.load(p)
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as e:
                import warnings
                warnings.warn(f"skipping unreadable calibration {p}: {e}")
    return None


def _mat3(x):
    return np.asarray(x, np.float64).reshape(3, 3)


# --------------------------------------------------------------------------- #
@dataclass
class Intrinsics:
    """One camera's projection model.

    model        "fisheye" (Kannala-Brandt, 4 dist coeffs) or
                 "pinhole" (Brown-Conrady, OpenCV's 5+ coeffs)
    image_size   (w, h) the calibration was solved at; K is scaled to other
                 sizes by :meth:`undistort_maps`
    K            3x3 camera matrix
    dist         distortion coefficients, flat
    """
    model: str
    image_size: tuple
    K: np.ndarray
    dist: np.ndarray
    rms_reproj_px: float | None = None
    num_views: int | None = None

    def __post_init__(self):
        if self.model not in ("fisheye", "pinhole"):
            raise ValueError(f"model must be 'fisheye' or 'pinhole', got {self.model!r}")
        self.K = _mat3(self.K)
        self.dist = np.asarray(self.dist, np.float64).ravel()
        w, h = self.image_size
        self.image_size = (int(w), int(h))

    def to_dict(self):
        return {"model": self.model, "image_size": list(self.image_size),
                "K": self.K.tolist(), "dist": self.dist.tolist(),
                "rms_reproj_px": self.rms_reproj_px, "num_views": self.num_views}

    @classmethod
    def from_dict(cls, d):
        return cls(model=d["model"], image_size=tuple(d["image_size"]),
                   K=d["K"], dist=d["dist"],
                   rms_reproj_px=d.get("rms_reproj_px"),
                   num_views=d.get("num_views"))

    def undistort_maps(self, out_w, out_h, balance=0.0):
        """cv2.remap tables that undistort a frame of size (out_w, out_h).

        K is calibrated for ``image_size``; if the target size differs, K is
        scaled so the maps still line up. Fisheye builds newK by hand (OpenCV's
        estimateNewCameraMatrixForUndistortRectify returns an off-centre newK
        for wide fisheyes -> garbled remap): centred principal point, focal
        scaled by ``balance`` (0 = original focal / cropped periphery, higher =
        zoom out to keep more field of view, black borders at the limit).
        """
        import cv2

        cw, _ch = self.image_size
        s = out_w / float(cw)
        K = self.K * np.array([[s, s, s], [s, s, s], [0, 0, 1.0]])
        K[2, 2] = 1.0
        size = (out_w, out_h)
        if self.model == "fisheye":
            D = self.dist.reshape(4, 1)
            newK = K.copy()
            f_scale = 1.0 - 0.7 * balance
            newK[0, 0] *= f_scale
            newK[1, 1] *= f_scale
            newK[0, 2] = out_w / 2.0
            newK[1, 2] = out_h / 2.0
            return cv2.fisheye.initUndistortRectifyMap(
                K, D, np.eye(3), newK, size, cv2.CV_16SC2)
        newK, _ = cv2.getOptimalNewCameraMatrix(K, self.dist, size, balance, size)
        return cv2.initUndistortRectifyMap(
            K, self.dist, np.eye(3), newK, size, cv2.CV_16SC2)


# --------------------------------------------------------------------------- #
@dataclass
class ColorCalibration:
    """Sensor -> display color, applied in *linear* light.

    wb_gains       (3,) per-channel white-balance multipliers, normalized so
                   min(gains) == 1 (no channel attenuated; a sensor-clipped
                   highlight stays clipped-white instead of tinting). For
                   typical CMOS (green strongest) this is the G=1 convention.
    wb_illuminant  free-text note on the light the gains were measured under
                   ("sunlight", "bench LED 5600K", ...) — gains are only valid
                   under that illuminant.
    ccm            future: 3x3 color-correction matrix, linear WB'd sensor RGB
                   -> linear BT.709, rows summing to 1 (preserves neutrals).
    """
    wb_gains: np.ndarray | None = None
    wb_illuminant: str = ""
    ccm: np.ndarray | None = None

    def __post_init__(self):
        if self.wb_gains is not None:
            self.wb_gains = np.asarray(self.wb_gains, np.float64).ravel()
            if self.wb_gains.shape != (3,):
                raise ValueError("wb_gains must be 3 values (R, G, B)")
        if self.ccm is not None:
            self.ccm = _mat3(self.ccm)

    def to_dict(self):
        return {"wb_gains": None if self.wb_gains is None else self.wb_gains.tolist(),
                "wb_illuminant": self.wb_illuminant,
                "ccm": None if self.ccm is None else self.ccm.tolist()}

    @classmethod
    def from_dict(cls, d):
        return cls(wb_gains=d.get("wb_gains"),
                   wb_illuminant=d.get("wb_illuminant", ""),
                   ccm=d.get("ccm"))


# --------------------------------------------------------------------------- #
@dataclass
class CameraCalibration:
    """Everything calibrated about one physical sensor, keyed by serial.

    ``intrinsics`` and ``color`` are independently optional — each calibration
    tool fills in its piece and :meth:`save` keeps the rest::

        cal = CameraCalibration.load_or_new("046060323003")
        cal.color = ColorCalibration(wb_gains=[1.21, 1.0, 1.34],
                                     wb_illuminant="sunlight")
        cal.save()                       # -> calib_046060323003.json

    ``notes`` is a free-form dict for provenance (board geometry, etc.).
    """
    serial: str
    intrinsics: Intrinsics | None = None
    color: ColorCalibration | None = None
    notes: dict = field(default_factory=dict)
    saved_at: str | None = None

    @staticmethod
    def default_path(serial, dir="."):
        return Path(dir) / f"calib_{serial}.json"

    def to_dict(self):
        return {"format": FORMAT_TAG, "serial": self.serial,
                "saved_at": self.saved_at,
                "intrinsics": self.intrinsics.to_dict() if self.intrinsics else None,
                "color": self.color.to_dict() if self.color else None,
                "notes": self.notes}

    @classmethod
    def from_dict(cls, d):
        fmt = d.get("format", "")
        if not fmt.startswith("hteng-camera-calibration/"):
            raise ValueError(f"not a camera-calibration file (format={fmt!r})")
        intr = d.get("intrinsics")
        color = d.get("color")
        return cls(serial=d["serial"],
                   intrinsics=Intrinsics.from_dict(intr) if intr else None,
                   color=ColorCalibration.from_dict(color) if color else None,
                   notes=d.get("notes") or {},
                   saved_at=d.get("saved_at"))

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def load_or_new(cls, serial, dir="."):
        """Existing calibration for ``serial`` in ``dir``, or a fresh empty one."""
        p = cls.default_path(serial, dir)
        if p.exists():
            cal = cls.load(p)
            if cal.serial != serial:
                raise ValueError(f"{p} holds serial {cal.serial!r}, expected {serial!r}")
            return cal
        return cls(serial=serial)

    def save(self, dir=".", path=None):
        """Write to ``path`` (default ``<dir>/calib_<serial>.json``). Returns the path."""
        p = Path(path) if path else self.default_path(self.serial, dir)
        self.saved_at = datetime.now().astimezone().isoformat(timespec="seconds")
        with open(p, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return p


# --------------------------------------------------------------------------- #
@dataclass
class StereoCalibration:
    """Cam2cam pose for a pair of sensors: X_right = R @ X_left + t."""
    serial_left: str
    serial_right: str
    R: np.ndarray
    t: np.ndarray
    reproj_rms_px: float | None = None
    num_pairs: int | None = None
    method: str = ""
    notes: dict = field(default_factory=dict)
    saved_at: str | None = None

    def __post_init__(self):
        self.R = _mat3(self.R)
        self.t = np.asarray(self.t, np.float64).ravel()

    @staticmethod
    def default_path(serial_left, serial_right, dir="."):
        return Path(dir) / f"stereo_{serial_left}_{serial_right}.json"

    def to_dict(self):
        return {"format": STEREO_FORMAT_TAG, "convention": STEREO_CONVENTION,
                "serial_left": self.serial_left, "serial_right": self.serial_right,
                "saved_at": self.saved_at,
                "R": self.R.tolist(), "t": self.t.tolist(),
                "baseline": float(np.linalg.norm(self.t)),
                "reproj_rms_px": self.reproj_rms_px, "num_pairs": self.num_pairs,
                "method": self.method, "notes": self.notes}

    @classmethod
    def from_dict(cls, d):
        fmt = d.get("format", "")
        if not fmt.startswith("hteng-camera-stereo/"):
            raise ValueError(f"not a stereo-calibration file (format={fmt!r})")
        return cls(serial_left=d["serial_left"], serial_right=d["serial_right"],
                   R=d["R"], t=d["t"], reproj_rms_px=d.get("reproj_rms_px"),
                   num_pairs=d.get("num_pairs"), method=d.get("method", ""),
                   notes=d.get("notes") or {}, saved_at=d.get("saved_at"))

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def save(self, dir=".", path=None):
        p = Path(path) if path else self.default_path(
            self.serial_left, self.serial_right, dir)
        self.saved_at = datetime.now().astimezone().isoformat(timespec="seconds")
        with open(p, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return p
