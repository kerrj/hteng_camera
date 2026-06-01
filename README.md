# hteng_camera

A clean, fast Python interface to HTENG / MindVision machine-vision cameras.

The sensor streams raw **12-bit packed Bayer** over USB3. This package unpacks
it and demosaics with OpenCV (~10× faster than the SDK's on-host ISP), handing
you a **linear `uint16` RGB** frame. Gamma is never baked in — you encode for
display/export yourself with `convert.to_display` (default is a fast `sqrt`).

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

    rgb16 = cam.grab()                       # linear uint16 (H, W, 3), freshest frame
    preview = convert.to_display(rgb16)      # sqrt-encoded uint8 for screen/PNG
```

### Two cameras

Two cameras are simply two independent instances — open each **by serial** so
identity is stable across replug/reboot:

```python
left  = HTCamera(serial="044162023020")
right = HTCamera(serial="046060323003")
l = left.grab()
r = right.grab()
```

## API

### `list_cameras() -> list[dict]`
Enumerate attached cameras. Each dict has `index`, `serial`, `name`, `product`,
`port`, `sensor`.

### `HTCamera(serial=None, index=None, *, fov=None, frame_speed=None, auto_exposure=False, open_now=True)`
One physical camera. Prefer `serial=`; `index=` is a convenience fallback.

| Method | Description |
|---|---|
| `grab(timeout_ms=2000, drop_first=0, priority=GET_NEWEST, align_to_16bit=False)` | One **linear** `uint16` RGB frame (fast path). `None` on timeout. |
| `grab_bayer12(timeout_ms=500, priority=GET_NEWEST)` | Raw 12-bit Bayer plane, `uint16` `(H, W)` 0..4095, no demosaic. |
| `set_exposure_ms(ms)` / `get_exposure_ms()` | Exposure in milliseconds. |
| `set_analog_gain(x)` / `get_analog_gain()` / `gain_range()` | Analog gain (x-multiplier). |
| `set_ae(enabled)` | Auto-exposure on/off. |
| `set_frame_speed(mode)` | `FRAME_SPEED_LOW` / `_NORMAL` / `_HIGH` / `_SUPER`. |
| `set_roi(w, h, h_offset=0, v_offset=0)` / `current_resolution()` | Sensor crop (full SDK ROI). |
| `media_types()` | Available transmission formats. |
| `play()` / `pause()` / `close()` | Streaming + lifecycle. Use as a context manager for auto-close. |

`priority` options: `GET_NEWEST` (freshest, default — best for live view),
`GET_OLDEST` (FIFO), `GET_NEXT` (interrupt current exposure).

### `convert` — pure pixel math (no SDK dependency)

| Function | Description |
|---|---|
| `unpack_bayer12_packed(raw, w, h)` | Packed-12 bytes → `uint16` Bayer plane. |
| `demosaic(bayer12, cv_code, align_to_16bit=False)` | Bayer → linear `uint16` RGB. |
| `to_display(linear, gamma=2.0, exposure=1.0, black=0.0, white=1.0, max_in=4095.0)` | Linear → display `uint8`. `gamma=2.0` is a fast `sqrt`. |

## Example: live control GUI

```bash
python examples/viser_control.py
```

A self-contained [viser](https://viser.studio) web GUI: open/close the camera,
drive exposure / gain / frame-speed, and tune the display tone curve (gamma,
exposure mult, black/white) live. The camera feed is the scene background;
snapshots save both the linear-16 source and the tone-mapped preview.

## Notes

- **Always release the camera** (`close()`, the context manager, or the GUI's
  "Close & release"). A leaked handle wedges the next run (`-18 DEVICE_IS_OPENED`).
  The package installs `atexit` + SIGTERM/SIGINT handlers as a backstop — never
  `pkill` a camera-holding process.
- The USB control channel can come up wedged (`-52`); `HTCamera` recovers in
  session via `CameraReConnect` automatically.
- Override the bundled binary with `HTENG_LIB=/path/to/libMVSDK.so` (e.g. to
  test a newer vendor drop).
```
