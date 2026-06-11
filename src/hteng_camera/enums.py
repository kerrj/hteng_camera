"""Constants from the vendor SDK headers (CameraDefine.h) that this package uses.

Only the values on the fast path are mirrored, so the C headers are never needed
at runtime. These encode the GenICam-style pixel format / SDK enums and are
stable across SDK versions.
"""

# --- 12-bit *packed* Bayer transmission formats ---------------------------
# OCCUPY12BIT -> 1.5 bytes/px, hi-byte-first. This is what the cameras stream by
# default and the only format the fast path consumes (raw, linear, full depth).
# Code = MONO(0x01000000) | OCCUPY12BIT(0x000C0000) | ordinal(0x2A..0x2D).
MEDIA_BAYGR12_PACKED = 0x010C002A
MEDIA_BAYRG12_PACKED = 0x010C002B
MEDIA_BAYGB12_PACKED = 0x010C002C
MEDIA_BAYBG12_PACKED = 0x010C002D

# --- frame speed (CameraSetFrameSpeed) ------------------------------------
FRAME_SPEED_LOW = 0
FRAME_SPEED_NORMAL = 1
FRAME_SPEED_HIGH = 2
FRAME_SPEED_SUPER = 3

# --- get-buffer priority (CameraGetImageBufferPriority) -------------------
GET_OLDEST = 0  # FIFO from the SDK queue (default CameraGetImageBuffer behaviour)
GET_NEWEST = 1  # drop queued stale frames, return the freshest — best for live view
GET_NEXT = 2    # interrupt the current exposure if the camera supports it


# Demosaic quality -> cv2 enum suffix. Bilinear is fastest but shows zipper
# patterns and false-colour fringing on edges/fine detail; EA (edge-aware)
# removes most of it for a small cost. (OpenCV's VNG variant is omitted on
# purpose: it asserts depth==CV_8U, so it can't run on our 12-bit uint16 Bayer
# plane without an 8-bit downconvert that would discard HDR range.)
DEMOSAIC_QUALITY = ("bilinear", "ea")
_DEMOSAIC_SUFFIX = {"bilinear": "", "ea": "_EA"}

# Packed-12 media ordinal -> the sensor's CFA tile label (first row's first two
# pixels, GenICam naming: "GR" = rows G R G R / B G B G ...).
_TILE_BY_ORDINAL = {0x2A: "GR", 0x2B: "RG", 0x2C: "GB", 0x2D: "BG"}


def bayer_tile(media_type):
    """CFA tile label ("GR"/"RG"/"GB"/"BG") for a packed-12 media type, or None."""
    return _TILE_BY_ORDINAL.get(media_type & 0xFF)


def cv_bayer_code(media_type, quality="ea"):
    """Return the cv2.COLOR_Bayer*2RGB code for a packed-12 Bayer media type.

    OpenCV's Bayer enum naming is offset one row/col from the sensor's own tile
    label, so the mapping below is the empirically-correct one (a sensor that
    reports its tile as "BG" demosaics correctly with COLOR_BayerRG2RGB, etc).

    ``quality`` selects the demosaic algorithm (see :data:`DEMOSAIC_QUALITY`):
    "bilinear" or "ea" (edge-aware, default). Both share the same tile naming,
    so this just appends OpenCV's suffix to the base constant.

    Resolved lazily so importing this module never requires cv2. Returns None
    for an unrecognised (non packed-12) media type.
    """
    import cv2

    tile = bayer_tile(media_type)
    if tile is None:
        return None
    cv_pattern = {"GR": "GB", "RG": "BG", "GB": "GR", "BG": "RG"}[tile]
    suffix = _DEMOSAIC_SUFFIX[quality]
    return getattr(cv2, f"COLOR_Bayer{cv_pattern}2RGB{suffix}")
