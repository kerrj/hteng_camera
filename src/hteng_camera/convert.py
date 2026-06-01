"""Pure pixel math: raw 12-bit packed Bayer -> linear RGB16 -> display RGB8.

Nothing in here touches the SDK or a camera handle — it's plain numpy/cv2 on
buffers, which makes it independently unit-testable and the natural place to
later drop in a native (C++/SIMD) unpack+demosaic without changing callers.

The pipeline is deliberately split so the *linear* signal and the *display*
encode never get conflated:

    raw bytes  --unpack_bayer12_packed-->  uint16 Bayer plane (0..4095)
    Bayer      --demosaic-->               uint16 linear RGB  (0..4095, hi-aligned optional)
    linear RGB --to_display-->             uint8 RGB for screen/export (gamma encoded)

``grab()`` on the camera returns the *linear* RGB; gamma is applied only by
``to_display`` and only for preview/export. The encode is a cached lookup table
(see :func:`to_display`): building the curve over 65536 entries is ~0.08 ms,
while applying it is a single uint16->uint8 gather — ~2.6x faster than the
equivalent float32 round-trip over a 5 MP frame, and bit-identical to it.
"""

import numpy as np

from . import _fast

try:  # cv2 is required for demosaic; imported lazily-friendly for clear errors
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


def unpack_bayer12_packed(raw, width, height):
    """Unpack a 12-bit *packed* Bayer buffer to a uint16 plane (values 0..4095).

    Packing is hi-byte-first, 2 pixels per 3 bytes (GigE/USB3 "12 packed"):

        p0 = (b0 << 4) | (b1 & 0x0F)
        p1 = (b2 << 4) | (b1 >> 4)

    ``raw`` may be a numpy uint8 array or any buffer of length width*height*3/2.
    Returns an (H, W) uint16 array. ~7 ms for 5 MP, the bulk of the fast path.
    """
    buf = np.frombuffer(raw, dtype=np.uint8)
    expected = width * height * 3 // 2
    if buf.size < expected:
        raise ValueError(
            f"packed-12 buffer too small: got {buf.size} bytes, "
            f"need {expected} for {width}x{height}"
        )
    b = buf[:expected].reshape(-1, 3)
    lo = b[:, 1]
    out = np.empty(width * height, dtype=np.uint16)
    # Only the high bytes need promoting to uint16; the nibble ORs stay in range.
    out[0::2] = (b[:, 0].astype(np.uint16) << 4) | (lo & 0x0F)
    out[1::2] = (b[:, 2].astype(np.uint16) << 4) | (lo >> 4)
    return out.reshape(height, width)


def demosaic(bayer12, cv_code, align_to_16bit=False):
    """Demosaic a uint16 Bayer plane (0..4095) to linear RGB (H, W, 3) uint16.

    ``cv_code`` is the cv2.COLOR_Bayer*2RGB constant for this sensor's tile
    (see :func:`hteng_camera.enums.cv_bayer_code`). cv2's SIMD bilinear
    demosaic is ~0.5 ms for 5 MP.

    align_to_16bit: if True, left-shift the result by 4 so values span the full
    0..65520 uint16 range (useful for 16-bit PNG export). Default keeps the
    native 0..4095 scale, which is what :func:`to_display` expects.
    """
    if cv2 is None:
        raise RuntimeError("opencv-python is required for demosaic()")
    rgb = cv2.cvtColor(bayer12, cv_code)  # uint16, 0..4095
    if align_to_16bit:
        return rgb << 4
    return rgb


# Cache the most recent display LUT so repeated calls with unchanged tone-curve
# params (the common case in a live loop) reuse it. The table is keyed on the
# full parameter tuple; one entry is plenty since callers vary params slowly.
_LUT_CACHE = {"key": None, "lut": None}


def _display_lut(gamma, exposure, black, white, max_in):
    """Build (or fetch from cache) the uint16->uint8 display-encode table.

    The table has 65536 entries — one per possible uint16 input — so indexing it
    with any uint16 frame is always in-bounds, no clamping of indices needed.
    Building it touches 65536 values (~0.08 ms), vs millions for a per-pixel
    float path, so the sqrt/pow runs a few thousand times instead of millions.
    """
    key = (gamma, exposure, black, white, max_in)
    if _LUT_CACHE["key"] == key:
        return _LUT_CACHE["lut"]

    f = np.arange(65536, dtype=np.float32) * (exposure / max_in)
    if black != 0.0 or white != 1.0:
        f = (f - black) / max(1e-6, (white - black))
    np.clip(f, 0.0, 1.0, out=f)
    if gamma == 2.0:
        np.sqrt(f, out=f)            # gamma 2.0 == sqrt
    elif gamma != 1.0:
        f **= (1.0 / gamma)
    lut = (f * 255.0 + 0.5).astype(np.uint8)

    _LUT_CACHE["key"] = key
    _LUT_CACHE["lut"] = lut
    return lut


def to_display(linear, gamma=2.0, exposure=1.0, black=0.0, white=1.0,
               max_in=4095.0):
    """Encode a linear RGB (or mono) frame to uint8 for display/export.

        norm = clip((linear/max_in * exposure - black) / (white - black), 0, 1)
        out  = norm ** (1/gamma) * 255

    Implemented as a cached lookup table (built once per distinct tone curve),
    then a single uint16->uint8 gather — bit-identical to the float computation
    but ~2.6x faster on a 5 MP frame, since it never materialises a float buffer
    the size of the image. gamma=2.0 (default) is sqrt; 1.0 is linear; 2.2 is
    sRGB-ish.

    ``linear`` must be an integer array (uint8/uint16) — it indexes the LUT.
    ``max_in`` defaults to 4095 (native 12-bit). Pass 65535 if you fed in a
    16-bit-aligned frame (demosaic(..., align_to_16bit=True)).

    When the native kernel (libhteng_fast) is present it does the LUT apply
    multithreaded (~10x the numpy gather on a 5 MP frame); otherwise numpy's
    fancy-index gather is used. Output is identical either way. Set
    ``HTENG_NO_NATIVE=1`` to force the numpy path.
    """
    lut = _display_lut(gamma, exposure, black, white, max_in)

    # Native fast path: contiguous uint16 in, threaded LUT apply into uint8 out.
    if _fast.available and linear.dtype == np.uint16:
        src = np.ascontiguousarray(linear)
        out = np.empty(src.shape, dtype=np.uint8)
        _fast._lib.hteng_apply_lut_u16(
            src.ctypes.data_as(_fast.POINTER(_fast.c_uint16)),
            out.ctypes.data_as(_fast.POINTER(_fast.c_ubyte)),
            src.size,
            lut.ctypes.data_as(_fast.POINTER(_fast.c_ubyte)),
            lut.size,          # 65536 -> no index clamping needed
            0,                 # auto-select thread count
        )
        return out

    return lut[linear]
