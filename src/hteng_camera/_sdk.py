"""ctypes binding to the MindVision/HTENG SDK.

This is the raw C boundary: struct definitions, function signatures, and the
status-code check. It is platform-agnostic — the only platform decision happens
in :mod:`_loader`, which hands us a ``CDLL``. The struct layouts are verified
byte-identical across the Linux and macOS SDK headers.

Most code should use :class:`hteng_camera.camera.HTCamera`, not these directly.
"""

import ctypes
from ctypes import (
    POINTER, Structure, byref, c_char, c_double, c_float, c_int, c_ubyte,
    c_uint, c_void_p,
)

from ._loader import load

_lib = load()


# ---------------------------------------------------------------------------
# Structs (subset we cross the boundary with)
# ---------------------------------------------------------------------------

class tSdkCameraDevInfo(Structure):
    _fields_ = [
        ("acProductSeries", c_char * 32),
        ("acProductName", c_char * 32),
        ("acFriendlyName", c_char * 32),
        ("acLinkName", c_char * 32),
        ("acDriverVersion", c_char * 32),
        ("acSensorType", c_char * 32),
        ("acPortType", c_char * 32),
        ("acSn", c_char * 32),
        ("uInstance", c_uint),
    ]


class tSdkImageResolution(Structure):
    _fields_ = [
        ("iIndex", c_int),
        ("acDescription", c_char * 32),
        ("uBinSumMode", c_uint),
        ("uBinAverageMode", c_uint),
        ("uSkipMode", c_uint),
        ("uResampleMask", c_uint),
        ("iHOffsetFOV", c_int),
        ("iVOffsetFOV", c_int),
        ("iWidthFOV", c_int),
        ("iHeightFOV", c_int),
        ("iWidth", c_int),
        ("iHeight", c_int),
        ("iWidthZoomHd", c_int),
        ("iHeightZoomHd", c_int),
        ("iWidthZoomSw", c_int),
        ("iHeightZoomSw", c_int),
    ]


class tSdkMediaType(Structure):
    _fields_ = [
        ("iIndex", c_int),
        ("acDescription", c_char * 32),
        ("iMediaType", c_uint),
    ]


class _CapHead(Structure):
    """Prefix view of tSdkCameraCapbility up to the media-type list.

    CameraGetCapability fills the whole (much larger) struct; we only read the
    media-type prefix, so callers MUST back this with an over-sized buffer and
    cast, never allocate it raw.
    """
    _fields_ = [
        ("pTriggerDesc", c_void_p), ("iTriggerDesc", c_int),
        ("pImageSizeDesc", c_void_p), ("iImageSizeDesc", c_int),
        ("pClrTempDesc", c_void_p), ("iClrTempDesc", c_int),
        ("pMediaTypeDesc", POINTER(tSdkMediaType)), ("iMediaTypdeDesc", c_int),
    ]


class tSdkFrameHead(Structure):
    _fields_ = [
        ("uiMediaType", c_uint),
        ("uBytes", c_uint),
        ("iWidth", c_int),
        ("iHeight", c_int),
        ("iWidthZoomSw", c_int),
        ("iHeightZoomSw", c_int),
        ("bIsTrigger", c_int),
        ("uiTimeStamp", c_uint),
        ("uiExpTime", c_uint),
        ("fAnalogGain", c_float),
        ("iGamma", c_int),
        ("iContrast", c_int),
        ("iSaturation", c_int),
        ("fRgain", c_float),
        ("fGgain", c_float),
        ("fBgain", c_float),
    ]


# ---------------------------------------------------------------------------
# Function signatures
# ---------------------------------------------------------------------------

def _sig(name, restype, argtypes):
    fn = getattr(_lib, name)
    fn.restype = restype
    fn.argtypes = argtypes
    return fn


CameraSdkInit = _sig("CameraSdkInit", c_int, [c_int])
CameraEnumerateDevice = _sig("CameraEnumerateDevice", c_int,
                             [POINTER(tSdkCameraDevInfo), POINTER(c_int)])
CameraInit = _sig("CameraInit", c_int,
                  [POINTER(tSdkCameraDevInfo), c_int, c_int, POINTER(c_int)])
CameraUnInit = _sig("CameraUnInit", c_int, [c_int])
CameraPlay = _sig("CameraPlay", c_int, [c_int])
CameraPause = _sig("CameraPause", c_int, [c_int])
CameraReConnect = _sig("CameraReConnect", c_int, [c_int])

CameraGetCapability = _sig("CameraGetCapability", c_int, [c_int, c_void_p])
CameraSetMediaType = _sig("CameraSetMediaType", c_int, [c_int, c_int])
CameraGetMediaType = _sig("CameraGetMediaType", c_int, [c_int, POINTER(c_int)])
CameraSetIspOutFormat = _sig("CameraSetIspOutFormat", c_int, [c_int, c_uint])
CameraSetTriggerMode = _sig("CameraSetTriggerMode", c_int, [c_int, c_int])

CameraSetAeState = _sig("CameraSetAeState", c_int, [c_int, c_int])
CameraSetExposureTime = _sig("CameraSetExposureTime", c_int, [c_int, c_double])
CameraGetExposureTime = _sig("CameraGetExposureTime", c_int,
                             [c_int, POINTER(c_double)])
CameraSetAnalogGainX = _sig("CameraSetAnalogGainX", c_int, [c_int, c_float])
CameraGetAnalogGainX = _sig("CameraGetAnalogGainX", c_int,
                            [c_int, POINTER(c_float)])
CameraGetAnalogGainXRange = _sig("CameraGetAnalogGainXRange", c_int,
                                 [c_int, POINTER(c_float), POINTER(c_float),
                                  POINTER(c_float)])
CameraSetFrameSpeed = _sig("CameraSetFrameSpeed", c_int, [c_int, c_int])
CameraSetImageResolution = _sig("CameraSetImageResolution", c_int,
                                [c_int, POINTER(tSdkImageResolution)])
CameraGetImageResolution = _sig("CameraGetImageResolution", c_int,
                                [c_int, POINTER(tSdkImageResolution)])

CameraGetImageBufferPriority = _sig("CameraGetImageBufferPriority", c_int,
                                    [c_int, POINTER(tSdkFrameHead),
                                     POINTER(POINTER(c_ubyte)), c_uint, c_uint])
CameraReleaseImageBuffer = _sig("CameraReleaseImageBuffer", c_int,
                                [c_int, POINTER(c_ubyte)])

# Reset the per-camera frame-timestamp counter to 0 (for software cross-camera
# frame pairing — see HTCamera.reset_timestamp).
CameraRstTimeStamp = _sig("CameraRstTimeStamp", c_int, [c_int])


class SdkError(RuntimeError):
    """A vendor SDK call returned a non-zero status code."""

    def __init__(self, fn_name, status):
        self.fn_name = fn_name
        self.status = status
        super().__init__(f"{fn_name} failed with status {status}")


def check(rc, fn_name):
    if rc != 0:
        raise SdkError(fn_name, rc)
