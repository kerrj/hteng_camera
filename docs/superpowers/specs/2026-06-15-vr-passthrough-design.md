# VR Passthrough — Design Spec

**Date:** 2026-06-15
**Status:** Approved (design); pending implementation plan
**Component:** new `examples/vr_passthrough.py` + `examples/vr_web/` (no package-core changes)

## Goal

Stream the calibrated wide-FOV **stereo** cameras into a **Quest 3S** as a
"look-around-the-world" VR passthrough view. The captured fisheye image is
mapped into 3D so the user can freely look around *within* the wide field of
view, rendered at headset refresh and decoupled from the camera frame rate.

## Core constraint (the design driver)

Camera frames arrive at a fixed (and possibly modest) rate, but the headset must
render **as fast as it can** (90 Hz on Quest 3S, 120 Hz opt-in) so head-look is
always smooth. Therefore the latest frame is treated as a **texture mapped into
the 3D scene**, and the WebXR render loop redraws the user's *current* head pose
against that texture every refresh. Head-look latency = headset render loop;
imagery freshness = camera + transport. The two never block each other.

## Key decisions

| Area | Decision | Notes |
|---|---|---|
| Runtime | **WebXR** in the Quest browser (Three.js) | Reuses the Python→web pattern; no app install |
| Depth model | **True stereo** (left cam → left eye, right cam → right eye) | The payoff of the calibrated stereo rig |
| Projection | **B: fisheye-sampled sphere** (client shader); **C: equirect** as fallback | B = most accurate, least host work |
| Capture host | **Mac now, Linux later** — host-agnostic encode/transport boundary | VideoToolbox now, NVENC later, behind one interface |
| Transport | **MJPEG over WebSocket** (TurboJPEG); boundary swappable to **WebRTC** later | Simple, low-latency on LAN/USB |
| Deployment | **Tethered USB-C via `adb reverse`** (primary); wifi/LAN (fallback) | Tether → `localhost` → no TLS cert needed |
| v1 scope | **Minimal live viewer** | No extra UI, no recording |

## Non-goals (v1)

- In-headset/web controls for exposure/gain/FOV/IPD (future).
- Recording or VR180 playback (future).
- Driving any physical pan/robot from head pose (future).
- Motion parallax from head translation, or CCM color correction (out of scope;
  mild green cast is accepted, per the package README).
- WebRTC transport (designed-for, not built in v1).

## Architecture

```
HOST  (capture + encode + serve; Mac now → Linux later, behind one interface)
  Left+Right cameras (USB3, timestamp-synced)
    → grab() fast path → tonemap_linear (WB + BT.709/sRGB) → uint8 RGB
    → downscale (--width) → JPEG encode (TurboJPEG), per eye
    → newest-wins mailbox (per eye)
    → aiohttp server: serves static client page + per-eye frame WebSocket
        · calibration JSON (K, dist, image_size, stereo R) sent once on connect
                              │  WebSocket (newest JPEG per eye, at camera rate)
                              ▼
HEADSET BROWSER  (WebXR, Three.js)
  WS recv + createImageBitmap → GL texture per eye
    → WebXR render loop @ 90–120 Hz:
        left sphere  (layer 1) → left eye
        right sphere (layer 2) → right eye
        fisheye Kannala-Brandt shader; right sphere oriented by stereo R
        head pose: rotation only (spheres centered on head; translation ignored)
    → your eyes (stereo wide-FOV look-around)
```

### Components

- **`examples/vr_passthrough.py`** — the host app. Opens both cameras by serial,
  runs the per-eye pixel pipeline in per-camera threads, runs an **aiohttp**
  server (already in the env) that serves the static client *and* the per-eye
  frame WebSocket, and (when tethered) runs `adb reverse` and prints the URL.
- **`examples/vr_web/`** — the static WebXR client: `index.html`, `app.js`
  (Three.js + WebXR session, scene/spheres, render loop), `fisheye.glsl` (the
  projection shader). No build step.
- **No package-core changes.** Uses existing `HTCamera`, `convert.tonemap_linear`,
  and `calibration` (`find`, `CameraCalibration`, `StereoCalibration`). Anything
  reusable can graduate into the package later.

## Data flow & decoupling

- **Per eye, host:** `grab()` (newest, fast path) → `tonemap_linear` with that
  camera's `wb_gains` + BT.709/sRGB curve → uint8 → downscale → JPEG (TurboJPEG)
  → single-slot newest-wins mailbox. Two cameras in independent threads.
- **Transport:** the WebSocket sends only the freshest JPEG per eye; if the
  client is slower than the cameras, stale frames are **dropped, never queued**
  (latency does not grow).
- **Client:** each WS message → `createImageBitmap` → upload to that eye's GL
  texture. The WebXR render loop runs separately at headset refresh, drawing the
  latest textures against the current head pose.
- **Calibration** (`K`, `dist`, `image_size`, stereo `R`) is sent once on connect
  as shader uniforms.

## Projection & stereo math

- **Fisheye-sphere shader, per eye.** Each eye renders a large inward-facing
  sphere centered on the head, textured with that camera's latest frame. The
  fragment shader takes the viewing-ray direction and applies the camera's
  **Kannala-Brandt forward model** to find the source pixel:
  - `θ = atan2(‖xy‖, z)`
  - `r = θ·(1 + k₁θ² + k₂θ⁴ + k₃θ⁶ + k₄θ⁸)`
  - `u = fx·(x/‖xy‖)·r + cx`, `v = fy·(y/‖xy‖)·r + cy`
  - Sample texture at `(u/W, v/H)`. Rays past the calibrated FOV or off-image
    render black — the natural "edge of the captured world."
- **Per-eye routing (Three.js layers).** Left sphere on layer 1, right on layer
  2; the XR camera's two sub-cameras are bound to layers 1 and 2 respectively, so
  each eye sees only its image (standard stereo-360/VR180 technique).
- **Co-aligning eyes with stereo `R`.** Orient the right sphere by the calibrated
  `R` relative to the left (optionally split symmetrically à la
  `cv2.stereoRectify` R1/R2). Removes vertical disparity and relative tilt,
  leaving horizontal disparity for depth → comfortable fusion.
- **Head pose = rotation only.** Spheres stay centered on the head each frame, so
  rotation reveals different parts of the FOV (look-around) while translation is
  ignored (one fixed viewpoint has no motion parallax to give).
- **Limitations (inherent to fixed-camera VR180):** spheres at infinity → disparity
  accurate for distant content; very near objects have slight vergence mismatch,
  and if camera baseline ≠ user IPD, depth scale is somewhat off. Not correctable
  without per-pixel depth.

## Color, resolution, latency

- **Color:** `convert.tonemap_linear` with each camera's `wb_gains` + BT.709/sRGB
  curve → uint8; texture flagged sRGB so it displays correctly. No CCM (mild green
  cast accepted).
- **Resolution:** Quest 3S ≈ 1832×1920/eye, 90 Hz (120 Hz opt-in). Wide FOV
  spreads pixels over ~96°, so a per-eye texture of **~1280 px** (configurable
  `--width`) looks good. Render loop targets 90 Hz; imagery at 30–60 fps suffices.
- **Bandwidth:** stereo 1280×1024 JPEG @ 30 fps ≈ **80–140 Mbit/s**. Comfortable
  on USB tether and on 5 GHz/Wi-Fi 6; not on 2.4 GHz. Tunable via `--width`,
  `--fps`, `--quality`. WebRTC (future) would cut this substantially.

## Networking & deployment

- **Primary — tethered USB-C (`adb reverse`):** `adb reverse tcp:<port>
  tcp:<port>` forwards a Quest port to the Mac; the Quest browser opens
  **`http://localhost:<port>`**, routed over USB.
  - `localhost` is a **secure context**, so WebXR `immersive-vr` works over plain
    **HTTP/WS — no HTTPS, no self-signed cert.**
  - USB (≥480 Mbit/s) dwarfs the bandwidth need and is lower-latency/more stable
    than wifi.
  - **Prereqs (one-time):** Quest developer mode + USB debugging on; `adb`
    (Android platform-tools) on the Mac. The host app runs `adb reverse` and
    prints the URL.
- **Fallback — wifi/LAN:** the host serves **HTTPS with an auto-generated
  self-signed cert** (so the frame socket is **WSS**); the user accepts the cert
  warning once in the Quest browser.

## Failure & edge handling

- **Camera lifecycle:** reuse package safeguards — `-52` auto-recovers via
  `CameraReConnect`; atexit/SIGTERM release already installed; surface `-18
  DEVICE_IS_OPENED` clearly. Host uses context managers so both cameras release.
- **Missing calibration:** no intrinsics for a serial → **fail fast** ("run
  `charuco_calibrate.py` first"); the projection needs `K`/`dist`. Missing stereo
  file → warn, fall back to `R = I` (assume parallel). Missing `wb_gains` → skip
  WB with a note.
- **Degraded capture:** only one camera opens or one drops → run **mono** (same
  image to both eyes) with a warning. Frame timeout → client keeps the last frame
  (newest-wins); host attempts reconnect.
- **Client/secure-context errors:** WS disconnect is clean and reconnectable
  without restarting capture. If the session can't go immersive (not secure / no
  XR), the page shows the fix (tethered localhost URL, or accept the HTTPS cert).
- **Bandwidth overrun (wifi):** newest-wins drops stale frames; `--width`,
  `--fps`, `--quality` knobs to tune down.

## Testing strategy

- **Shader-math parity (unit):** a Python reference of the KB forward map checked
  against `cv2.fisheye.projectPoints` for sample rays; the GLSL/JS port checked
  against the same vectors. Guarantees the shader samples the right pixels.
- **Synthetic source + desktop preview (integration, no hardware):** a
  `--test-pattern` mode feeds a known image (lat-long grid / ChArUco) instead of
  cameras, and the client has a **non-XR fallback** (mouse-drag look-around on a
  normal canvas). Together these validate the full host→WS→client path, stereo
  routing, and projection **on the Mac browser before donning the headset**.
- **Manual headset checklist:** enters VR; correct image per eye; smooth 90 Hz
  look-around; comfortable stereo fusion; FOV edges where expected; acceptable
  imagery latency.

## v1 success criteria

Tether the Quest 3S to the Mac over USB-C, run `python examples/vr_passthrough.py`,
open the printed `http://localhost:<port>` in the Quest browser, enter VR, and
see a **live true-stereo wide-FOV feed** mapped to the sphere that you can **look
around within, smoothly at 90 Hz**, white-balanced via the existing pipeline,
with FOV edges where the lens stops. No extra UI, no recording.

## Deliverables / file layout

```
examples/vr_passthrough.py      # host app: capture pipeline + aiohttp server + adb reverse
examples/vr_web/index.html      # WebXR client page
examples/vr_web/app.js          # Three.js scene, WebXR session, render loop, WS client
examples/vr_web/fisheye.glsl    # Kannala-Brandt projection shader (B); equirect path (C) fallback
tests/test_fisheye_projection.py  # KB forward-map parity vs cv2.fisheye.projectPoints
```

## Future work

WebRTC transport (hardware H.264/HEVC via VideoToolbox/NVENC); in-headset/web
controls (exposure/gain/FOV, IPD/convergence comfort knobs); stereo recording +
VR180 playback; Linux/robot-host deployment; optional head-pose-driven physical
pan; CCM color correction.
