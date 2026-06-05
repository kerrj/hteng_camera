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

import os

import numpy as np

from . import _fast

try:  # cv2 is required for demosaic; imported lazily-friendly for clear errors
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


def set_num_threads(n):
    """Cap CPU threads used by the pixel pipeline (cv2's demosaic + our gamma).

    Demosaic runs in cv2, which by default grabs *every* core (e.g. 10 threads
    behind your back). This caps it process-wide so the camera pipeline doesn't
    saturate the machine. The native unpack is single-threaded regardless.
    Call once at startup; ``HTENG_NUM_THREADS`` sets the initial value.

    Platform note: cv2's cap works under TBB/pthreads (the Linux default). Under
    macOS's GCD backend ``cv2.setNumThreads`` is a documented no-op for n>0 (only
    n=0 has an effect there: fully serial); our native gamma kernel still honors
    the cap on every platform.
    """
    if cv2 is not None:
        cv2.setNumThreads(int(n))
    _fast.set_num_threads(int(n))


# Apply an initial thread cap so importing the package doesn't silently hand cv2
# all cores. Default 4 is a sane balance; override with HTENG_NUM_THREADS.
_default_threads = int(os.environ.get("HTENG_NUM_THREADS", "4"))
set_num_threads(_default_threads)


def unpack_bayer12_packed(raw, width, height):
    """Unpack a 12-bit *packed* Bayer buffer to a uint16 plane (values 0..4095).

    Packing is hi-byte-first, 2 pixels per 3 bytes (GigE/USB3 "12 packed"):

        p0 = (b0 << 4) | (b1 & 0x0F)
        p1 = (b2 << 4) | (b1 >> 4)

    ``raw`` may be a numpy uint8 array or any buffer of length width*height*3/2.
    Returns an (H, W) uint16 array.

    Uses the native single-threaded kernel when available (~0.5 ms for 5 MP, 15x
    the numpy path below) and falls back to numpy otherwise — bit-identical
    either way. This is the step that dominated the old pure-Python pipeline.
    """
    buf = np.frombuffer(raw, dtype=np.uint8)
    expected = width * height * 3 // 2
    if buf.size < expected:
        raise ValueError(
            f"packed-12 buffer too small: got {buf.size} bytes, "
            f"need {expected} for {width}x{height}"
        )

    if _fast.available:
        src = np.ascontiguousarray(buf[:expected])
        out = np.empty(width * height, dtype=np.uint16)
        _fast._lib.hteng_unpack_bayer12(
            src.ctypes.data_as(_fast.POINTER(_fast.c_ubyte)),
            out.ctypes.data_as(_fast.POINTER(_fast.c_uint16)),
            width * height,
        )
        return out.reshape(height, width)

    b = buf[:expected].reshape(-1, 3)
    lo = b[:, 1]
    out = np.empty(width * height, dtype=np.uint16)
    # Only the high bytes need promoting to uint16; the nibble ORs stay in range.
    out[0::2] = (b[:, 0].astype(np.uint16) << 4) | (lo & 0x0F)
    out[1::2] = (b[:, 2].astype(np.uint16) << 4) | (lo >> 4)
    return out.reshape(height, width)


def demosaic(bayer12, cv_code, align_to_16bit=False):
    """Demosaic a uint16 Bayer plane (0..4095) to linear RGB (H, W, 3) uint16.

    ``cv_code`` is the cv2.COLOR_Bayer*2RGB constant for this sensor's tile and
    chosen quality (see :func:`hteng_camera.enums.cv_bayer_code`). cv2's SIMD
    bilinear demosaic is ~0.5 ms for 5 MP; edge-aware (EA) and VNG variants are
    higher quality at a modest / larger cost respectively.

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


# Tone curves: pure 0..1 -> 0..1 pointwise mappings applied to the normalised
# linear signal. Each is monotonic so it round-trips a 1-D LUT exactly. Add new
# ones here and they immediately work through the cache + native kernel.
#
#   gamma     out = x**(1/gamma)            — classic power curve; one knob, but
#                                             lifting shadows (high gamma) washes
#                                             out mid/highlights (see README).
#   log       out = log(1+a*x)/log(1+a)     — allocates output per stop of light:
#                                             strong shadow/mid detail, gentle
#                                             highlight rolloff, black stays black.
#                                             Best all-round for HDR machine-vision
#                                             scenes. `a` = shadow-lift strength.
#   reinhard  out = (x/(x+k)) * (1+k)/1     — smooth global tone map; great
#                                             midtones, soft highlights. `k` sets
#                                             the shoulder.
def _curve_gamma(x, gamma):
    if gamma == 2.0:
        return np.sqrt(x)               # exact, and the common fast default
    if gamma == 1.0:
        return x
    return x ** (1.0 / gamma)


def _curve_log(x, a):
    a = max(1e-3, float(a))
    return np.log1p(a * x) / np.log1p(a)


def _curve_reinhard(x, k):
    k = max(1e-3, float(k))
    return (x / (x + k)) * (1.0 + k)     # scaled so x=1 -> 1


_CURVES = {"gamma": _curve_gamma, "log": _curve_log, "reinhard": _curve_reinhard}

# Default shape param per curve, used when the caller doesn't pass `param`.
_CURVE_DEFAULT_PARAM = {"gamma": 2.0, "log": 80.0, "reinhard": 0.18}


# Cache the most recent display LUT so repeated calls with an unchanged tone
# curve (the common case in a live loop) reuse it. Keyed on the full spec —
# curve name, shape param, and the exposure/black/white normalisation — so any
# change rebuilds (~0.08 ms) and anything steady is a free lookup.
_LUT_CACHE = {"key": None, "lut": None}


def _display_lut(curve, param, exposure, black, white, max_in):
    """Build (or fetch from cache) the uint16->uint8 display-encode table.

    The table has 65536 entries — one per possible uint16 input — so indexing it
    with any uint16 frame is always in-bounds, no clamping of indices needed.
    The curve runs over 65536 values (~0.08 ms) instead of per-pixel, so even an
    expensive curve costs the same at apply time as a plain sqrt.

    ``curve`` is a name in ``_CURVES`` or a callable ``f(x01_array) -> 0..1``.
    ``param`` is the curve's shape knob (gamma exponent / log a / reinhard k);
    None uses that curve's default. Callables ignore ``param``.
    """
    key = (curve if isinstance(curve, str) else id(curve),
           param, exposure, black, white, max_in)
    if _LUT_CACHE["key"] == key:
        return _LUT_CACHE["lut"]

    # Normalise linear input -> 0..1 with exposure gain and black/white levels.
    f = np.arange(65536, dtype=np.float32) * (exposure / max_in)
    if black != 0.0 or white != 1.0:
        f = (f - black) / max(1e-6, (white - black))
    np.clip(f, 0.0, 1.0, out=f)

    # Apply the tone curve.
    if callable(curve):
        f = np.clip(curve(f), 0.0, 1.0)
    else:
        fn = _CURVES.get(curve)
        if fn is None:
            raise ValueError(f"unknown curve {curve!r}; options: {list(_CURVES)}")
        p = _CURVE_DEFAULT_PARAM[curve] if param is None else param
        f = fn(f, p)

    lut = (f * 255.0 + 0.5).astype(np.uint8)
    _LUT_CACHE["key"] = key
    _LUT_CACHE["lut"] = lut
    return lut


def to_display(linear, gamma=None, exposure=1.0, black=0.0, white=1.0,
               max_in=4095.0, curve="gamma", param=None):
    """Encode a linear RGB (or mono) frame to uint8 for display/export.

        norm = clip((linear/max_in * exposure - black) / (white - black), 0, 1)
        out  = curve(norm) * 255

    ``curve`` selects the tone curve (see :data:`_CURVES`):
      * ``"gamma"`` — power curve; ``param`` is the gamma exponent (default 2.0,
        a plain sqrt). 1.0 = linear, 2.2 = sRGB-ish. Lifting shadows via high
        gamma washes out mid/highlights.
      * ``"log"`` — ``out = log(1+a*x)/log(1+a)``; ``param`` = a, the shadow-lift
        strength (default 80). Best all-round for HDR scenes: strong shadow and
        midtone detail, gentle highlight rolloff, black stays black.
      * ``"reinhard"`` — smooth global tone map; ``param`` = k (default 0.18).
      * a callable ``f(x01) -> 0..1`` for an arbitrary pointwise curve.

    ``gamma=`` is kept as a back-compat shortcut: passing it is equivalent to
    ``curve="gamma", param=<gamma>``.

    Implemented as a cached lookup table (built once per distinct curve), then a
    single uint16->uint8 gather — the curve runs over 65536 values, not per
    pixel, so even an expensive curve is free per-frame. ``linear`` must be an
    integer array (uint8/uint16) — it indexes the LUT.

    When the native kernel (libhteng_fast) is present it does the LUT apply
    multithreaded (~10x the numpy gather on a 5 MP frame); otherwise numpy's
    fancy-index gather is used. Output is identical either way. Set
    ``HTENG_NO_NATIVE=1`` to force the numpy path.
    """
    # Back-compat: gamma= is shorthand for curve="gamma", param=gamma.
    if gamma is not None:
        curve, param = "gamma", gamma
    lut = _display_lut(curve, param, exposure, black, white, max_in)

    # Native fast path: contiguous uint16 in, threaded LUT apply into uint8 out.
    if _fast.available and linear.dtype == np.uint16:
        src = np.ascontiguousarray(linear)
        out = np.empty(src.shape, dtype=np.uint8)
        _fast._lib.hteng_apply_lut_u16(
            src.ctypes.data_as(_fast.POINTER(_fast.c_uint16)),
            out.ctypes.data_as(_fast.POINTER(_fast.c_ubyte)),
            src.size,
            lut.ctypes.data_as(_fast.POINTER(_fast.c_ubyte)),
            lut.size,                 # 65536 -> no index clamping needed
            _fast.num_threads(),      # honor the central thread cap
        )
        return out

    return lut[linear]
