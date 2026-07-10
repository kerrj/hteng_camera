"""hteng_camera — a clean, fast Python interface to HTENG/MindVision cameras.

Fast path: the sensor streams raw 12-bit packed Bayer; this package unpacks and
cv2-demosaics it (~10x faster than the SDK's on-host ISP). ``grab()`` returns a
*linear* uint16 RGB frame — gamma is never baked in; encode for display/export
with :func:`convert.tonemap_linear`.

Quick start::

    from hteng_camera import HTCamera, list_cameras, convert

    for cam in list_cameras():
        print(cam["serial"], cam["name"])

    with HTCamera(serial="044162023020") as cam:
        cam.set_exposure_ms(15)
        cam.set_analog_gain(1.0)
        rgb16, info = cam.grab()            # linear uint16 (H, W, 3), GET_NEWEST
        preview = convert.tonemap_linear(rgb16) # sqrt-encoded uint8 for screen/PNG
        # info["time"] = hardware frame timestamp in microseconds

Two cameras are two independent ``HTCamera`` instances — open each by serial.
Pair their frames in software via ``cam.reset_timestamp()`` + ``info["time"]``.
"""

from . import calibration, convert, enums
from .auto_exposure import AutoExposure
from .calibration import (
    CameraCalibration, ColorCalibration, Intrinsics, StereoCalibration,
)
from .camera import HTCamera, list_cameras
from .convert import set_num_threads
from .enums import (
    FRAME_SPEED_HIGH, FRAME_SPEED_LOW, FRAME_SPEED_NORMAL, FRAME_SPEED_SUPER,
    GET_NEWEST, GET_NEXT, GET_OLDEST,
)
from .imu import ImuSample, YbImu

__version__ = "0.1.0"

__all__ = [
    "HTCamera", "list_cameras", "AutoExposure", "calibration", "convert", "enums",
    "set_num_threads",
    "CameraCalibration", "Intrinsics", "ColorCalibration", "StereoCalibration",
    "FRAME_SPEED_LOW", "FRAME_SPEED_NORMAL", "FRAME_SPEED_HIGH", "FRAME_SPEED_SUPER",
    "GET_OLDEST", "GET_NEWEST", "GET_NEXT",
    "YbImu", "ImuSample",
]
