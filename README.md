# hteng_camera

A clean, fast Python interface to HTENG / MindVision machine-vision cameras.

The sensor streams raw **12-bit packed Bayer** over USB3. This package unpacks
it and demosaics with OpenCV (~10× faster than the SDK's on-host ISP), handing
you a **linear `uint16` RGB** frame. Gamma is never baked in — you encode for
display/export yourself with `convert.tonemap_linear` (default is a fast `sqrt`).

The vendor shared library is **bundled in the wheel** (Linux x86_64, Linux
aarch64, macOS arm64), so there's nothing to install system-wide and the right
binary is picked automatically at runtime.

## Install

```bash
pip install hteng-camera            # core
pip install hteng-camera[viser]     # + the live control-GUI example
```

### Linux: udev rules (one-time, for non-root USB access)

The Python package needs no system install, but Linux USB device nodes are
root-only by default. Install the rules once so a normal user can open the
camera:

```bash
sudo ./packaging/install_udev.sh    # then replug the camera
```

## Quick start

```python
from hteng_camera import HTCamera, list_cameras, convert

for c in list_cameras():
    print(c["serial"], c["name"], c["port"])

with HTCamera(serial="044162023020") as cam:
    cam.set_exposure_ms(15)
    cam.set_analog_gain(1.0)

    rgb16, info = cam.grab()                    # linear uint16 (H, W, 3), freshest frame
    preview = convert.tonemap_linear(rgb16)     # sqrt-encoded uint8 for screen/PNG
    # info["time"] = hardware frame timestamp in microseconds
```

### Two cameras

Two cameras are simply two independent instances — open each **by serial** so
identity is stable across replug/reboot:

```python
left  = HTCamera(serial="044162023020")
right = HTCamera(serial="046060323003")
l, l_info = left.grab()
r, r_info = right.grab()
```

To pair frames across the two in software, call `reset_timestamp()` on both
back-to-back to establish a shared time base, then match frames by
`info["time"]` (typically 1–2 ms accuracy; use a hardware trigger for true
simultaneous exposure).

## API

### `list_cameras() -> list[dict]`
Enumerate attached cameras. Each dict has `index`, `serial`, `name`, `product`,
`port`, `sensor`.

### `HTCamera(serial=None, index=None, *, fov=None, frame_speed=None, auto_exposure=False, demosaic_quality="ea", open_now=True)`
One physical camera. Prefer `serial=`; `index=` is a convenience fallback.

| Method | Description |
|---|---|
| `grab(timeout_ms=2000, drop_first=0, priority=GET_NEWEST, align_to_16bit=False)` | One **linear** `uint16` RGB frame (fast path). Returns `(rgb, info)`; `(None, {})` on timeout. `info["time"]` is the hardware frame timestamp in µs. |
| `grab_bayer12(timeout_ms=500, priority=GET_NEWEST)` | Raw 12-bit Bayer plane, `uint16` `(H, W)` 0..4095, no demosaic. Same `(frame, info)` return shape. |
| `set_exposure_ms(ms)` / `get_exposure_ms()` | Exposure in milliseconds. |
| `set_analog_gain(x)` / `get_analog_gain()` / `gain_range()` | Analog gain (x-multiplier). |
| `set_ae(enabled)` | Auto-exposure on/off. |
| `set_frame_speed(mode)` | `FRAME_SPEED_LOW` / `_NORMAL` / `_HIGH` / `_SUPER`. |
| `set_demosaic_quality(q)` | `"ea"` (edge-aware, default) or `"bilinear"` (faster, zipper artifacts). |
| `set_roi(w, h, h_offset=0, v_offset=0)` / `current_resolution()` | Sensor crop (full SDK ROI). |
| `reset_timestamp()` | Zero the hardware frame-timestamp counter (cross-camera pairing). |
| `media_types()` | Available transmission formats. |
| `play()` / `pause()` / `close()` | Streaming + lifecycle. Use as a context manager for auto-close. |

`priority` options: `GET_NEWEST` (freshest, default — best for live view),
`GET_OLDEST` (FIFO), `GET_NEXT` (interrupt current exposure).

### `convert` — pure pixel math (no SDK dependency)

| Function | Description |
|---|---|
| `unpack_bayer12_packed(raw, w, h)` | Packed-12 bytes → `uint16` Bayer plane. |
| `demosaic(bayer12, cv_code, align_to_16bit=False)` | Bayer → linear `uint16` RGB. |
| `tonemap_linear(linear, gamma=None, exposure=1.0, black=0.0, white=1.0, max_in=4095.0, curve="gamma", param=None, out_dtype=np.uint8)` | Linear → display/encode through a tone curve (cached LUT). |
| `set_num_threads(n)` | Cap CPU threads used by the pixel pipeline (cv2 + native kernels). |

`tonemap_linear` curves: `"gamma"` (power curve; `param`=exponent, default 2.0 =
`sqrt`), `"log"` (`param`=shadow-lift strength, best all-round for HDR scenes),
`"reinhard"` (smooth global tone map), `"bt709"` (the standard, *reversible*
BT.709 OETF — what `record.py` bakes into its default master), or any callable
`f(x01) -> 0..1`. `out_dtype=np.uint16` produces a full-range encode master for
a 10-bit+ encoder instead of a uint8 preview.

## Examples

| Script | What it does |
|---|---|
| `examples/viser_control.py` | Live control GUI: exposure/gain/ROI, tone-curve preview, snapshots. |
| `examples/record.py` | Record to 10-bit HEVC MP4 (NVENC / VideoToolbox / x265). |
| `examples/make_viewable.py` | Transcode/tone-map a record.py master to a plays-anywhere 8-bit H.264. |
| `examples/charuco_calibrate.py` | ChArUco intrinsics + stereo cam2cam calibration GUI. |
| `examples/profile_pipeline.py` | Per-stage pipeline timing (capture/unpack/demosaic/display). |

```bash
python examples/viser_control.py
```

A self-contained [viser](https://viser.studio) web GUI: open/close the camera,
drive exposure / gain / frame-speed, and tune the display tone curve (gamma,
exposure mult, black/white) live. The camera feed is the scene background;
snapshots save both the linear-16 source and the tone-mapped preview.

## Native acceleration (optional)

The hot pixel ops (`unpack_bayer12_packed` and `tonemap_linear`'s LUT apply)
have an optional native C++ kernel, `libhteng_fast` — multithreaded LUT apply
(~9x the numpy gather on a 5 MP frame, ≈12 ms → ≈1.3 ms) and a ~15x unpack,
bit-identical output. It is **optional**: if the binary isn't present the
package falls back to numpy automatically, so everything works either way.

A built wheel ships the kernel. To build it from source (e.g. on a fresh Linux
checkout):

```bash
./native/build.sh          # compiles into src/hteng_camera/_libs/<platform>/
```

Requires a C++17 compiler (`g++` on Linux, `clang++` on macOS) — no other
dependencies; flags are portable `-O3` (the kernel is memory-bandwidth bound, so
`-march=native` is neither used nor needed). Set `HTENG_NO_NATIVE=1` to force the
numpy fallback (useful for benchmarking the two).

## Notes

- **Always release the camera** (`close()`, the context manager, or the GUI's
  "Close & release"). A leaked handle wedges the next run (`-18 DEVICE_IS_OPENED`).
  The package installs `atexit` + SIGTERM/SIGINT handlers as a backstop — never
  `pkill` a camera-holding process.
- The USB control channel can come up wedged (`-52`); `HTCamera` recovers in
  session via `CameraReConnect` automatically.
- **Color is raw sensor-native**: the fast path bypasses the SDK ISP, so no
  white balance or color-correction matrix is applied — expect a green-ish cast
  vs a consumer camera. Recorded files are tagged BT.709 for player
  compatibility; proper WB/CCM handling is future work.
- Override the bundled binary with `HTENG_LIB=/path/to/libMVSDK.so` (e.g. to
  test a newer vendor drop).
- `HTENG_NUM_THREADS` (default 4) caps the pixel pipeline's CPU threads at
  import time; `set_num_threads()` adjusts it at runtime.
