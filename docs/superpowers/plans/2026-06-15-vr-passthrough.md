# VR Passthrough Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stream the calibrated wide-FOV stereo cameras into a Quest 3S as a live, look-around-the-world VR passthrough view, rendered in the headset browser via WebXR.

**Architecture:** A Python host app (`examples/vr_passthrough.py`) opens both cameras, runs a per-eye pixel pipeline (grab → white-balance/tone-map → downscale → JPEG) into newest-wins mailboxes, and serves a static WebXR client plus a per-eye frame WebSocket via aiohttp. The client (`examples/vr_web/`) maps each eye's latest JPEG onto an inward-facing sphere using the camera's Kannala-Brandt fisheye model in a fragment shader, routing left/right images to the two eyes via Three.js layers. Head rotation reveals different parts of the captured FOV; the render loop runs at headset refresh, decoupled from the camera frame rate. Primary deployment is USB-C tether via `adb reverse` (localhost = secure context, no TLS), with a wifi/HTTPS fallback.

**Tech Stack:** Python (numpy, opencv, PyTurboJPEG, aiohttp), Three.js + WebXR (vendored, no build step), GLSL ES 1.00. Existing package APIs: `HTCamera`, `convert.tonemap_linear`, `calibration`.

**Spec:** `docs/superpowers/specs/2026-06-15-vr-passthrough-design.md`

---

## File Structure

| File | Responsibility |
|---|---|
| `examples/vr_passthrough.py` | Host app: KB reference, Mailbox, JPEG encode, frame sources (camera + test pattern), per-eye pipeline threads, calib payload, aiohttp server (static + WS), adb-reverse + URL/cert logic, argparse, main |
| `examples/vr_web/index.html` | Client page: import map, Enter-VR button, canvas, loads `app.js` |
| `examples/vr_web/shaders.js` | ES module exporting the fisheye vertex + fragment shader strings |
| `examples/vr_web/app.js` | Three.js scene, two spheres + layers, WebXR session, WS client, render loop, desktop (non-XR) preview |
| `examples/vr_web/vendor/three.module.js` | Vendored Three.js (offline-robust) |
| `examples/vr_web/vendor/VRButton.js` | Vendored Three.js VRButton helper |
| `tests/conftest.py` | Puts `examples/` on `sys.path` so tests can import `vr_passthrough` |
| `tests/test_fisheye_projection.py` | KB forward-map parity vs `cv2.fisheye.projectPoints` |
| `tests/test_host_pipeline.py` | Mailbox newest-wins, JPEG encode round-trip, calib payload, eye selection, server smoke test |

**Design note (deviation from spec file list):** the spec listed `fisheye.glsl`; we use `shaders.js` instead so shaders load as an ES module (no extra `fetch`/MIME setup). The equirect fallback (projection "C") is **not** built in v1 — it remains documented future work; v1 ships only path "B".

**Convention reference (used by the shader and the stereo-R task):**
- Lens frame = OpenCV: **X right, Y down, Z forward**; pixel origin top-left, `u` right, `v` down.
- Three.js reference frame: **X right, Y up, Z toward viewer** (forward = −Z).
- Basis change OpenCV↔Three: `B = diag(1, -1, -1)` (its own inverse).
- A Three local direction `p` maps to a lens-frame direction `d = (p.x, -p.y, -p.z)`.
- Kannala-Brandt forward: `θ=atan2(‖d.xy‖, d.z)`, `θd=θ(1+k₁θ²+k₂θ⁴+k₃θ⁶+k₄θ⁸)`, `u=fx·(d.x/‖d.xy‖)·θd+cx`, `v=fy·(d.y/‖d.xy‖)·θd+cy`.

---

## Task 1: Test infra + Kannala-Brandt projection parity

**Files:**
- Create: `tests/conftest.py`
- Create: `examples/vr_passthrough.py` (only `kb_project` for now)
- Test: `tests/test_fisheye_projection.py`

- [ ] **Step 1: Create the test path shim**

`tests/conftest.py`:

```python
import sys
from pathlib import Path

# Make the importable host helpers in examples/vr_passthrough.py available as
# `import vr_passthrough` without turning examples/ into a package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))
```

- [ ] **Step 2: Write the failing parity test**

`tests/test_fisheye_projection.py`:

```python
import numpy as np
import cv2
import pytest

from vr_passthrough import kb_project


def _sample_dirs():
    # Rays spread across a wide fisheye field, all with positive z (in front).
    rng = np.random.default_rng(0)
    az = rng.uniform(-np.pi, np.pi, 200)
    theta = rng.uniform(0.0, np.deg2rad(85), 200)  # up to 85 deg off-axis
    x = np.sin(theta) * np.cos(az)
    y = np.sin(theta) * np.sin(az)
    z = np.cos(theta)
    return np.stack([x, y, z], axis=1).astype(np.float64)


def test_kb_project_matches_opencv():
    K = np.array([[800.0, 0, 960.0], [0, 800.0, 540.0], [0, 0, 1.0]])
    dist = np.array([-0.02, 0.004, -0.0008, 0.0001])  # k1..k4
    dirs = _sample_dirs()

    # OpenCV reference: project unit-depth 3D points through the fisheye model.
    obj = dirs.reshape(-1, 1, 3)
    ref, _ = cv2.fisheye.projectPoints(
        obj, np.zeros(3), np.zeros(3), K, dist.reshape(4, 1))
    ref = ref.reshape(-1, 2)

    ours = kb_project(dirs, K, dist)

    assert ours.shape == ref.shape
    assert np.allclose(ours, ref, atol=1e-3), np.abs(ours - ref).max()
```

- [ ] **Step 3: Run it, verify it fails**

Run: `pytest tests/test_fisheye_projection.py -v`
Expected: FAIL — `ImportError: cannot import name 'kb_project'` (module/function not defined).

- [ ] **Step 4: Implement `kb_project`**

Create `examples/vr_passthrough.py` with exactly this (more is added in later tasks):

```python
"""HTENG stereo → Quest 3S WebXR passthrough host.

Run:  python examples/vr_passthrough.py
See docs/superpowers/specs/2026-06-15-vr-passthrough-design.md
"""
import numpy as np


def kb_project(dirs, K, dist):
    """Kannala-Brandt forward projection (the exact math the client shader mirrors).

    dirs : (N,3) ray directions in the OpenCV lens frame (x right, y down, z fwd).
    K    : (3,3) camera matrix.  dist : (4,) [k1,k2,k3,k4].
    Returns (N,2) pixel coords. Mirrors cv2.fisheye.projectPoints for z>0 rays.
    """
    dirs = np.asarray(dirs, np.float64).reshape(-1, 3)
    K = np.asarray(K, np.float64).reshape(3, 3)
    k1, k2, k3, k4 = np.asarray(dist, np.float64).ravel()[:4]
    x, y, z = dirs[:, 0], dirs[:, 1], dirs[:, 2]
    rxy = np.hypot(x, y)
    theta = np.arctan2(rxy, z)
    t2 = theta * theta
    thetad = theta * (1.0 + k1 * t2 + k2 * t2**2 + k3 * t2**3 + k4 * t2**4)
    scale = np.where(rxy > 1e-9, thetad / np.maximum(rxy, 1e-12), 1.0)
    u = K[0, 0] * (x * scale) + K[0, 2]
    v = K[1, 1] * (y * scale) + K[1, 2]
    return np.stack([u, v], axis=1)
```

- [ ] **Step 5: Run it, verify it passes**

Run: `pytest tests/test_fisheye_projection.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add tests/conftest.py tests/test_fisheye_projection.py examples/vr_passthrough.py
git commit -m "feat(vr): KB fisheye projection reference + parity test"
```

---

## Task 2: Newest-wins Mailbox + JPEG encode

**Files:**
- Modify: `examples/vr_passthrough.py`
- Test: `tests/test_host_pipeline.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_host_pipeline.py`:

```python
import numpy as np
import cv2

from vr_passthrough import Mailbox, encode_jpeg


def test_mailbox_newest_wins():
    mb = Mailbox()
    assert mb.get_latest() is None
    mb.put(b"a")
    mb.put(b"b")
    # Only the newest survives; reading does not consume.
    assert mb.get_latest() == b"b"
    assert mb.get_latest() == b"b"


def test_encode_jpeg_roundtrips():
    img = np.zeros((64, 96, 3), np.uint8)
    img[:, :48] = (200, 30, 30)  # left half red-ish (RGB)
    jpg = encode_jpeg(img, quality=90)
    assert isinstance(jpg, (bytes, bytearray)) and len(jpg) > 0
    # Decode (OpenCV gives BGR); the left half should be red-dominant.
    bgr = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
    assert bgr.shape == (64, 96, 3)
    b, g, r = bgr[10, 10]
    assert r > b and r > g
```

- [ ] **Step 2: Run, verify failure**

Run: `pytest tests/test_host_pipeline.py -v`
Expected: FAIL — `ImportError` for `Mailbox` / `encode_jpeg`.

- [ ] **Step 3: Implement Mailbox + encode_jpeg**

Add to `examples/vr_passthrough.py` (after `kb_project`):

```python
import threading

try:
    from turbojpeg import TurboJPEG, TJPF_RGB
    _TJ = TurboJPEG()
except Exception:  # libturbojpeg missing or init failed → cv2 fallback
    _TJ = None
    import cv2


class Mailbox:
    """Single-slot, newest-wins handoff between a producer thread and the
    async sender. put() overwrites; get_latest() peeks (never consumes)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._item = None

    def put(self, item):
        with self._lock:
            self._item = item

    def get_latest(self):
        with self._lock:
            return self._item


def encode_jpeg(rgb, quality=85):
    """RGB uint8 (H,W,3) → JPEG bytes. TurboJPEG if available, else cv2."""
    if _TJ is not None:
        return _TJ.encode(np.ascontiguousarray(rgb), quality=quality,
                          pixel_format=TJPF_RGB)
    import cv2
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return buf.tobytes()
```

- [ ] **Step 4: Run, verify pass**

Run: `pytest tests/test_host_pipeline.py -v`
Expected: PASS (both tests).

- [ ] **Step 5: Commit**

```bash
git add examples/vr_passthrough.py tests/test_host_pipeline.py
git commit -m "feat(vr): newest-wins Mailbox and JPEG encode helper"
```

---

## Task 3: Frame sources + per-eye pipeline

**Files:**
- Modify: `examples/vr_passthrough.py`
- Test: `tests/test_host_pipeline.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_host_pipeline.py`:

```python
from vr_passthrough import TestPatternSource, EyePipeline


def test_test_pattern_source_shape():
    src = TestPatternSource(width=320, height=240, eye="left")
    rgb = src.read()
    assert rgb.shape == (240, 320, 3) and rgb.dtype == np.uint8
    # Left/right patterns differ so we can tell the eyes apart on the headset.
    right = TestPatternSource(width=320, height=240, eye="right").read()
    assert not np.array_equal(rgb, right)


def test_eye_pipeline_produces_jpeg():
    src = TestPatternSource(width=640, height=480, eye="left")
    pipe = EyePipeline(src, out_width=320, quality=80)
    jpg = pipe.process_once()
    assert isinstance(jpg, (bytes, bytearray)) and len(jpg) > 0
    bgr = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
    # Downscaled to out_width preserving aspect (640x480 -> 320x240).
    assert bgr.shape == (240, 320, 3)
```

- [ ] **Step 2: Run, verify failure**

Run: `pytest tests/test_host_pipeline.py -v`
Expected: FAIL — `ImportError` for `TestPatternSource` / `EyePipeline`.

- [ ] **Step 3: Implement sources + pipeline**

Add to `examples/vr_passthrough.py`:

```python
import cv2  # safe: opencv is a hard dependency


def _downscale(rgb, out_width):
    h, w = rgb.shape[:2]
    if w == out_width:
        return rgb
    out_h = max(1, round(h * out_width / w))
    return cv2.resize(rgb, (out_width, out_h), interpolation=cv2.INTER_AREA)


class TestPatternSource:
    """Hardware-free source: a lat/long grid with an eye label, so the full
    host→WS→client path and the projection can be validated without cameras."""

    def __init__(self, width=1280, height=960, eye="left"):
        self.width, self.height, self.eye = width, height, eye
        self._img = self._render()

    def _render(self):
        w, h = self.width, self.height
        img = np.full((h, w, 3), 20, np.uint8)
        step = max(16, w // 24)
        img[::step, :] = (90, 90, 90)
        img[:, ::step] = (90, 90, 90)
        cv2.circle(img, (w // 2, h // 2), min(w, h) // 3, (60, 120, 220), 3)
        color = (60, 220, 120) if self.eye == "left" else (220, 120, 60)
        cv2.putText(img, self.eye.upper(), (w // 2 - 60, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 4)
        return img

    def read(self):
        return self._img.copy()

    def close(self):
        pass


class EyePipeline:
    """One eye: read source → (camera path white-balances/tone-maps) →
    downscale → JPEG. Sources already return display-ready uint8 RGB."""

    def __init__(self, source, out_width=1280, quality=85):
        self.source = source
        self.out_width = out_width
        self.quality = quality

    def process_once(self):
        rgb = self.source.read()
        if rgb is None:
            return None
        rgb = _downscale(rgb, self.out_width)
        return encode_jpeg(rgb, quality=self.quality)
```

- [ ] **Step 4: Run, verify pass**

Run: `pytest tests/test_host_pipeline.py -v`
Expected: PASS (all tests).

- [ ] **Step 5: Add the camera source (no separate test — exercised live in Task 9)**

Add to `examples/vr_passthrough.py`:

```python
from hteng_camera import HTCamera, convert


class CameraSource:
    """Live HTENG camera → display-ready uint8 RGB (WB + BT.709 tone map)."""

    def __init__(self, serial):
        self.cam = HTCamera(serial=serial)
        self.serial = serial

    def read(self):
        rgb16, info = self.cam.grab()
        if rgb16 is None:
            return None
        return convert.tonemap_linear(
            rgb16, wb_gains=self.cam.wb_gains, curve="bt709",
            out_dtype=np.uint8)

    def close(self):
        self.cam.close()
```

- [ ] **Step 6: Commit**

```bash
git add examples/vr_passthrough.py tests/test_host_pipeline.py
git commit -m "feat(vr): test-pattern + camera sources and per-eye pipeline"
```

---

## Task 4: Calibration payload

**Files:**
- Modify: `examples/vr_passthrough.py`
- Test: `tests/test_host_pipeline.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_host_pipeline.py`:

```python
from vr_passthrough import build_calib_payload


def test_build_calib_payload_shape():
    intr = {
        "model": "fisheye",
        "image_size": [1920, 1080],
        "K": [[800, 0, 960], [0, 800, 540], [0, 0, 1]],
        "dist": [-0.02, 0.004, -0.0008, 0.0001],
    }
    payload = build_calib_payload(
        left_intr=intr, right_intr=intr,
        stereo_R=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], max_fov_deg=150.0)
    assert payload["type"] == "calib"
    for eye in ("left", "right"):
        e = payload[eye]
        assert e["fx"] == 800 and e["cx"] == 960
        assert e["dist"] == [-0.02, 0.004, -0.0008, 0.0001]
        assert e["width"] == 1920 and e["height"] == 1080
    assert abs(payload["maxTheta"] - np.deg2rad(75.0)) < 1e-9  # half of FOV
    assert payload["R"] == [1, 0, 0, 0, 1, 0, 0, 0, 1]  # row-major flat
```

- [ ] **Step 2: Run, verify failure**

Run: `pytest tests/test_host_pipeline.py::test_build_calib_payload_shape -v`
Expected: FAIL — `ImportError` for `build_calib_payload`.

- [ ] **Step 3: Implement the payload builder**

Add to `examples/vr_passthrough.py`:

```python
def _eye_calib(intr):
    K = np.asarray(intr["K"], float).reshape(3, 3)
    w, h = intr["image_size"]
    return {
        "fx": float(K[0, 0]), "fy": float(K[1, 1]),
        "cx": float(K[0, 2]), "cy": float(K[1, 2]),
        "dist": [float(x) for x in np.asarray(intr["dist"]).ravel()[:4]],
        "width": int(w), "height": int(h),
    }


def build_calib_payload(left_intr, right_intr, stereo_R, max_fov_deg=150.0):
    """JSON-able dict sent once on WS connect → client shader uniforms."""
    R = np.asarray(stereo_R, float).reshape(3, 3)
    return {
        "type": "calib",
        "left": _eye_calib(left_intr),
        "right": _eye_calib(right_intr),
        "R": [float(x) for x in R.ravel()],          # row-major, OpenCV frame
        "maxTheta": float(np.deg2rad(max_fov_deg / 2.0)),
    }
```

- [ ] **Step 4: Run, verify pass**

Run: `pytest tests/test_host_pipeline.py::test_build_calib_payload_shape -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add examples/vr_passthrough.py tests/test_host_pipeline.py
git commit -m "feat(vr): build client calibration payload"
```

---

## Task 5: Deployment logic — eye selection, URL/cert decision, self-signed cert

**Files:**
- Modify: `examples/vr_passthrough.py`
- Test: `tests/test_host_pipeline.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_host_pipeline.py`:

```python
from vr_passthrough import choose_url, make_self_signed_cert


def test_choose_url_tethered_uses_localhost_http():
    # Tethered (adb reverse active) → localhost is a secure context → plain http.
    assert choose_url(tethered=True, lan_ip="192.168.1.5", port=8000) == \
        "http://localhost:8000"


def test_choose_url_wifi_uses_https_lan():
    assert choose_url(tethered=False, lan_ip="192.168.1.5", port=8000) == \
        "https://192.168.1.5:8000"


def test_make_self_signed_cert_writes_files(tmp_path):
    cert, key = make_self_signed_cert(tmp_path)
    assert cert.exists() and key.exists()
    assert cert.read_bytes().startswith(b"-----BEGIN CERTIFICATE-----")
```

- [ ] **Step 2: Run, verify failure**

Run: `pytest tests/test_host_pipeline.py -k "choose_url or self_signed" -v`
Expected: FAIL — `ImportError`.

- [ ] **Step 3: Implement deployment helpers**

Add to `examples/vr_passthrough.py`:

```python
import subprocess
from pathlib import Path


def choose_url(tethered, lan_ip, port):
    """Tethered → http://localhost (secure context, no cert). Wifi → https LAN."""
    if tethered:
        return f"http://localhost:{port}"
    return f"https://{lan_ip}:{port}"


def adb_reverse(port):
    """Forward Quest localhost:port → host localhost:port over USB. Returns
    True if a device was set up. Never raises (adb absent / no device)."""
    try:
        out = subprocess.run(["adb", "devices"], capture_output=True,
                             text=True, timeout=5).stdout
        lines = [l for l in out.splitlines()[1:] if l.strip().endswith("device")]
        if not lines:
            return False
        subprocess.run(["adb", "reverse", f"tcp:{port}", f"tcp:{port}"],
                       check=True, timeout=5)
        return True
    except (OSError, subprocess.SubprocessError):
        return False


def make_self_signed_cert(out_dir):
    """Generate a self-signed cert/key for the wifi/HTTPS fallback.
    Returns (cert_path, key_path)."""
    from datetime import datetime, timedelta, timezone
    from cryptography import x509
    from cryptography.x509.oid import NameOID
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    out_dir = Path(out_dir)
    cert_path, key_path = out_dir / "vr_cert.pem", out_dir / "vr_key.pem"
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "hteng-vr")])
    now = datetime.now(timezone.utc)
    cert = (x509.CertificateBuilder()
            .subject_name(name).issuer_name(name).public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - timedelta(days=1))
            .not_valid_after(now + timedelta(days=825))
            .sign(key, hashes.SHA256()))
    key_path.write_bytes(key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.TraditionalOpenSSL,
        serialization.NoEncryption()))
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    return cert_path, key_path
```

- [ ] **Step 4: Confirm `cryptography` is importable in the env**

Run: `python -c "import cryptography; print(cryptography.__version__)"`
Expected: prints a version. If `ModuleNotFoundError`, run `pip install cryptography` (it is a common transitive dep; install if absent), then re-run.

- [ ] **Step 5: Run, verify pass**

Run: `pytest tests/test_host_pipeline.py -k "choose_url or self_signed" -v`
Expected: PASS (all three).

- [ ] **Step 6: Commit**

```bash
git add examples/vr_passthrough.py tests/test_host_pipeline.py
git commit -m "feat(vr): tether/cert deployment helpers"
```

---

## Task 6: aiohttp server — static files + per-eye WebSocket

**Files:**
- Modify: `examples/vr_passthrough.py`
- Create: `examples/vr_web/index.html` (placeholder so static serving has something real to serve; replaced in Task 7)
- Test: `tests/test_host_pipeline.py`

- [ ] **Step 1: Create a minimal client page so the static route is real**

Create `examples/vr_web/index.html`:

```html
<!doctype html>
<meta charset="utf-8">
<title>HTENG VR passthrough</title>
<body><p>placeholder — replaced in Task 7</p></body>
```

- [ ] **Step 2: Write the failing server smoke test**

Append to `tests/test_host_pipeline.py`:

```python
import asyncio
import json
import aiohttp

from vr_passthrough import build_app, Mailbox, build_calib_payload


def test_server_sends_calib_then_frames():
    intr = {"model": "fisheye", "image_size": [320, 240],
            "K": [[200, 0, 160], [0, 200, 120], [0, 0, 1]],
            "dist": [0, 0, 0, 0]}
    calib = build_calib_payload(intr, intr, np.eye(3).tolist())
    left, right = Mailbox(), Mailbox()
    left.put(b"LEFTJPEG"); right.put(b"RIGHTJPEG")

    async def run():
        app = build_app(calib=calib, mailboxes={"left": left, "right": right},
                        web_dir=Path("examples/vr_web"), send_fps=60)
        runner = aiohttp.web.AppRunner(app)
        await runner.setup()
        site = aiohttp.web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = list(runner.addresses)[0][1] if False else \
            runner.server.sockets[0].getsockname()[1]
        try:
            async with aiohttp.ClientSession() as s:
                # static index served
                async with s.get(f"http://127.0.0.1:{port}/") as r:
                    assert r.status == 200
                # ws: first text msg = calib, then binary frames tagged by eye
                async with s.ws_connect(f"http://127.0.0.1:{port}/ws") as ws:
                    first = await asyncio.wait_for(ws.receive(), 2)
                    assert json.loads(first.data)["type"] == "calib"
                    seen = set()
                    for _ in range(8):
                        m = await asyncio.wait_for(ws.receive(), 2)
                        if m.type == aiohttp.WSMsgType.BINARY:
                            seen.add(m.data[0])  # 0=left, 1=right
                        if {0, 1} <= seen:
                            break
                    assert {0, 1} <= seen
        finally:
            await runner.cleanup()

    asyncio.run(run())
```

- [ ] **Step 3: Run, verify failure**

Run: `pytest tests/test_host_pipeline.py::test_server_sends_calib_then_frames -v`
Expected: FAIL — `ImportError` for `build_app`.

- [ ] **Step 4: Implement the server**

Add to `examples/vr_passthrough.py`:

```python
import asyncio
import json
from aiohttp import web, WSMsgType

EYE_ID = {"left": 0, "right": 1}


def build_app(calib, mailboxes, web_dir, send_fps=30):
    """aiohttp app: static client at /, frame stream at /ws.
    Protocol: text calib JSON on connect, then binary [eye_id_byte]+jpeg."""
    app = web.Application()
    web_dir = Path(web_dir)

    async def ws_handler(request):
        ws = web.WebSocketResponse(max_msg_size=0)
        await ws.prepare(request)
        await ws.send_str(json.dumps(calib))
        period = 1.0 / send_fps
        last = {"left": None, "right": None}
        try:
            while not ws.closed:
                for eye, mb in mailboxes.items():
                    jpg = mb.get_latest()
                    if jpg is not None and jpg is not last[eye]:
                        last[eye] = jpg
                        await ws.send_bytes(bytes([EYE_ID[eye]]) + bytes(jpg))
                await asyncio.sleep(period)
        except (ConnectionResetError, asyncio.CancelledError):
            pass
        return ws

    async def index(request):
        return web.FileResponse(web_dir / "index.html")

    app.router.add_get("/", index)
    app.router.add_get("/ws", ws_handler)
    app.router.add_static("/", web_dir)  # app.js, shaders.js, vendor/*
    return app
```

Note the test references `aiohttp.web`; it already imports `aiohttp`. Add `from aiohttp import web` at the test's top if running it standalone — it is imported by the module under test, but make the test self-sufficient:

```python
from aiohttp import web  # add near the other imports in test_host_pipeline.py
```

- [ ] **Step 5: Run, verify pass**

Run: `pytest tests/test_host_pipeline.py::test_server_sends_calib_then_frames -v`
Expected: PASS.

- [ ] **Step 6: Run the full suite**

Run: `pytest -v`
Expected: all tests PASS.

- [ ] **Step 7: Commit**

```bash
git add examples/vr_passthrough.py examples/vr_web/index.html tests/test_host_pipeline.py
git commit -m "feat(vr): aiohttp static + per-eye WebSocket server"
```

---

## Task 7: Client shell — vendor Three.js, enter VR, per-eye solid color

**Files:**
- Create: `examples/vr_web/vendor/three.module.js`, `examples/vr_web/vendor/VRButton.js`
- Replace: `examples/vr_web/index.html`
- Create: `examples/vr_web/app.js`

This task validates the WebXR session and **layer routing** (each eye sees a different image) before any shader work.

- [ ] **Step 1: Vendor Three.js**

```bash
mkdir -p examples/vr_web/vendor
curl -L -o examples/vr_web/vendor/three.module.js \
  https://unpkg.com/three@0.160.0/build/three.module.js
curl -L -o examples/vr_web/vendor/VRButton.js \
  https://unpkg.com/three@0.160.0/examples/jsm/webxr/VRButton.js
```

Then make `VRButton.js` resolve `three` locally — verify the top of the file imports from `'three'` (it does in 0.160.0). The import map in Step 2 maps `'three'` to the vendored module.

- [ ] **Step 2: Write index.html**

Replace `examples/vr_web/index.html`:

```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>HTENG VR passthrough</title>
  <style>html,body{margin:0;height:100%;background:#000;overflow:hidden}</style>
  <script type="importmap">
    { "imports": { "three": "./vendor/three.module.js" } }
  </script>
</head>
<body>
  <script type="module" src="./app.js"></script>
</body>
</html>
```

- [ ] **Step 3: Write app.js (solid-color eyes)**

Create `examples/vr_web/app.js`:

```js
import * as THREE from 'three';
import { VRButton } from './vendor/VRButton.js';

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.xr.enabled = true;
renderer.outputColorSpace = THREE.SRGBColorSpace;
document.body.appendChild(renderer.domElement);
document.body.appendChild(VRButton.createButton(renderer));

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(
  70, window.innerWidth / window.innerHeight, 0.1, 1000);

// Two big inward-facing spheres, one per eye, on layers 1 and 2.
function makeSphere(color, layer) {
  const geo = new THREE.SphereGeometry(500, 60, 40);
  const mat = new THREE.MeshBasicMaterial({ color, side: THREE.BackSide });
  const mesh = new THREE.Mesh(geo, mat);
  mesh.layers.set(layer);
  scene.add(mesh);
  return mesh;
}
const leftSphere = makeSphere(0x208020, 1);   // green to LEFT eye
const rightSphere = makeSphere(0x802020, 2);  // red to RIGHT eye

renderer.xr.addEventListener('sessionstart', () => {
  const xrCam = renderer.xr.getCamera();
  xrCam.cameras[0].layers.enable(1);  // left eye renders layer 1
  xrCam.cameras[1].layers.enable(2);  // right eye renders layer 2
});

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

renderer.setAnimationLoop(() => renderer.render(scene, camera));
```

- [ ] **Step 4: Serve and verify on the headset**

Run the host in test-pattern mode (full main lands in Task 9; for now run a one-line static server from the web dir):

```bash
cd examples/vr_web && python -m http.server 8000
```

Tether the Quest, then `adb reverse tcp:8000 tcp:8000` (or use the Quest's wifi to `http://<mac-lan-ip>:8000` — but note: WebXR needs a secure context, so for this pre-host step use the tethered `http://localhost:8000`). Open `http://localhost:8000` in the Quest browser, tap **Enter VR**.

Expected: you are inside; **left eye sees green, right eye sees red**. Close one eye at a time to confirm. If both eyes show the same color, layer routing is wrong — confirm `cameras[0]`/`cameras[1]` get layers 1/2 in `sessionstart`.

- [ ] **Step 5: Commit**

```bash
git add examples/vr_web/
git commit -m "feat(vr): WebXR client shell with per-eye layer routing"
```

---

## Task 8: Client fisheye shader + sphere sampling (desktop-verifiable)

**Files:**
- Create: `examples/vr_web/shaders.js`
- Modify: `examples/vr_web/app.js`

Validated on the **Mac desktop browser** (no headset) via a non-XR mouse-look preview against the test pattern.

- [ ] **Step 1: Write the shaders**

Create `examples/vr_web/shaders.js`:

```js
// Kannala-Brandt fisheye sampling. vDir is the sphere-local outward direction
// (sphere orientation encodes the lens frame). See plan "Convention reference".
export const VERT = `
varying vec3 vDir;
void main() {
  vDir = position;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}`;

export const FRAG = `
precision highp float;
varying vec3 vDir;
uniform sampler2D map;
uniform float fx, fy, cx, cy;
uniform float k1, k2, k3, k4;
uniform float imgW, imgH, maxTheta;
void main() {
  // Three local frame -> OpenCV lens frame (x right, y down, z forward).
  vec3 d = normalize(vec3(vDir.x, -vDir.y, -vDir.z));
  float rxy = length(d.xy);
  float theta = atan(rxy, d.z);
  if (theta > maxTheta) { gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0); return; }
  float t2 = theta * theta;
  float thetad = theta * (1.0 + k1*t2 + k2*t2*t2 + k3*t2*t2*t2 + k4*t2*t2*t2*t2);
  float scale = rxy > 1e-6 ? thetad / rxy : 1.0;
  float u = (fx * d.x * scale + cx) / imgW;
  float v = (fy * d.y * scale + cy) / imgH;
  if (u < 0.0 || u > 1.0 || v < 0.0 || v > 1.0) {
    gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0); return;     // outside captured image
  }
  gl_FragColor = texture2D(map, vec2(u, 1.0 - v));        // v down -> GL up
}`;
```

- [ ] **Step 2: Rewrite app.js to use the shader + a WS texture + desktop preview**

Replace `examples/vr_web/app.js`:

```js
import * as THREE from 'three';
import { VRButton } from './vendor/VRButton.js';
import { VERT, FRAG } from './shaders.js';

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.xr.enabled = true;
renderer.outputColorSpace = THREE.SRGBColorSpace;
document.body.appendChild(renderer.domElement);
document.body.appendChild(VRButton.createButton(renderer));

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(
  80, window.innerWidth / window.innerHeight, 0.1, 1000);

function makeTexture() {
  const t = new THREE.Texture();
  t.flipY = false;                       // we flip v in the shader
  t.colorSpace = THREE.SRGBColorSpace;
  t.minFilter = THREE.LinearFilter;
  t.magFilter = THREE.LinearFilter;
  t.generateMipmaps = false;
  t.wrapS = t.wrapT = THREE.ClampToEdgeWrapping;
  return t;
}

function makeEye(layer) {
  const tex = makeTexture();
  const mat = new THREE.ShaderMaterial({
    vertexShader: VERT, fragmentShader: FRAG, side: THREE.BackSide,
    uniforms: {
      map: { value: tex },
      fx: { value: 1 }, fy: { value: 1 }, cx: { value: 0 }, cy: { value: 0 },
      k1: { value: 0 }, k2: { value: 0 }, k3: { value: 0 }, k4: { value: 0 },
      imgW: { value: 1 }, imgH: { value: 1 }, maxTheta: { value: 1.4 },
    },
  });
  const mesh = new THREE.Mesh(new THREE.SphereGeometry(500, 64, 48), mat);
  mesh.layers.set(layer);
  scene.add(mesh);
  return { tex, mat, mesh };
}
const eyes = { left: makeEye(1), right: makeEye(2) };

function applyCalib(calib) {
  for (const name of ['left', 'right']) {
    const u = eyes[name].mat.uniforms, c = calib[name];
    u.fx.value = c.fx; u.fy.value = c.fy; u.cx.value = c.cx; u.cy.value = c.cy;
    u.k1.value = c.dist[0]; u.k2.value = c.dist[1];
    u.k3.value = c.dist[2]; u.k4.value = c.dist[3];
    u.imgW.value = c.width; u.imgH.value = c.height;
    u.maxTheta.value = calib.maxTheta;
  }
}

function setEyeImage(name, bitmap) {
  const tex = eyes[name].tex;
  tex.image = bitmap;
  tex.needsUpdate = true;
}

// ---- WebSocket: text calib, then [eyeByte]+jpeg binary frames ----
function connect() {
  const ws = new WebSocket(`ws://${location.host}/ws`);
  ws.binaryType = 'arraybuffer';
  ws.onmessage = async (ev) => {
    if (typeof ev.data === 'string') { applyCalib(JSON.parse(ev.data)); return; }
    const bytes = new Uint8Array(ev.data);
    const name = bytes[0] === 0 ? 'left' : 'right';
    const blob = new Blob([bytes.subarray(1)], { type: 'image/jpeg' });
    setEyeImage(name, await createImageBitmap(blob));
  };
  ws.onclose = () => setTimeout(connect, 1000);   // reconnect without restart
}
connect();

// ---- per-eye layer routing in XR ----
renderer.xr.addEventListener('sessionstart', () => {
  const xrCam = renderer.xr.getCamera();
  xrCam.cameras[0].layers.enable(1);
  xrCam.cameras[1].layers.enable(2);
});

// ---- keep spheres centered on the head (ignore translation, keep rotation) ----
const headPos = new THREE.Vector3();
function recenter(cam) {
  cam.getWorldPosition(headPos);
  eyes.left.mesh.position.copy(headPos);
  eyes.right.mesh.position.copy(headPos);
}

// ---- desktop (non-XR) preview: drag to look, render LEFT eye on both layers ----
let yaw = 0, pitch = 0, dragging = false, px = 0, py = 0;
camera.layers.enable(1);
addEventListener('pointerdown', (e) => { dragging = true; px = e.clientX; py = e.clientY; });
addEventListener('pointerup', () => { dragging = false; });
addEventListener('pointermove', (e) => {
  if (!dragging) return;
  yaw -= (e.clientX - px) * 0.005; pitch -= (e.clientY - py) * 0.005;
  pitch = Math.max(-1.5, Math.min(1.5, pitch));
  px = e.clientX; py = e.clientY;
  camera.quaternion.setFromEuler(new THREE.Euler(pitch, yaw, 0, 'YXZ'));
});

addEventListener('resize', () => {
  camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});

renderer.setAnimationLoop(() => {
  const cam = renderer.xr.isPresenting ? renderer.xr.getCamera() : camera;
  recenter(cam);
  renderer.render(scene, camera);
});
```

- [ ] **Step 3: Add a `--test-pattern` flag to the host (so the desktop browser gets frames)**

Add to `examples/vr_passthrough.py` a guarded `main()` that, for now, serves the test pattern (camera path added in Task 9):

```python
import argparse


def _run_test_pattern(args):
    web_dir = Path(__file__).resolve().parent / "vr_web"
    intr = {"model": "fisheye", "image_size": [args.width, round(args.width * 0.75)],
            "K": [[args.width * 0.42, 0, args.width / 2],
                  [0, args.width * 0.42, args.width * 0.375], [0, 0, 1]],
            "dist": [0.0, 0.0, 0.0, 0.0]}
    calib = build_calib_payload(intr, intr, np.eye(3).tolist(), args.max_fov_deg)
    mailboxes = {"left": Mailbox(), "right": Mailbox()}
    pipes = {n: EyePipeline(TestPatternSource(args.width, round(args.width*0.75), n),
                            out_width=args.width, quality=args.quality)
             for n in ("left", "right")}
    for n, mb in mailboxes.items():
        mb.put(pipes[n].process_once())
    app = build_app(calib, mailboxes, web_dir, send_fps=args.fps)
    web.run_app(app, host="0.0.0.0", port=args.port)


def main():
    ap = argparse.ArgumentParser(description="HTENG stereo → Quest 3S WebXR passthrough")
    ap.add_argument("--test-pattern", action="store_true",
                    help="serve a synthetic grid instead of cameras (no hardware)")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--quality", type=int, default=85)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--max-fov-deg", type=float, default=150.0)
    args = ap.parse_args()
    if args.test_pattern:
        _run_test_pattern(args)
        return
    raise SystemExit("camera mode lands in Task 9 — use --test-pattern for now")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Desktop verification**

```bash
python examples/vr_passthrough.py --test-pattern --port 8000
```

Open `http://localhost:8000` in the **Mac** browser (not VR). Drag to look around.
Expected: the grid + center circle render **upright and centered** when looking forward; dragging pans within the image; beyond the image edge is **black**. The text reads "LEFT" (desktop preview shows the left eye).
If the image is mirrored or upside-down, the v-flip or the `(p.x,-p.y,-p.z)` mapping is off — re-check against the "Convention reference". If the image is tiny/huge, `maxTheta`/`K` scale in the test intrinsics is off.

- [ ] **Step 5: Commit**

```bash
git add examples/vr_web/shaders.js examples/vr_web/app.js examples/vr_passthrough.py
git commit -m "feat(vr): fisheye sphere shader + desktop preview (test pattern)"
```

---

## Task 9: Live cameras end-to-end

**Files:**
- Modify: `examples/vr_passthrough.py`

- [ ] **Step 1: Implement camera mode in `main()`**

Add to `examples/vr_passthrough.py`:

```python
import threading as _threading
from hteng_camera import list_cameras, calibration


def _resolve_pair(args):
    """Return (left_serial, right_serial). Explicit flags win; else use the
    stereo calibration file in ./calibrations; else the two enumerated cams."""
    if args.left and args.right:
        return args.left, args.right
    for d in (Path("calibrations"), Path(".")):
        for p in d.glob("stereo_*_*.json") if d.exists() else []:
            st = calibration.StereoCalibration.load(p)
            return st.serial_left, st.serial_right
    cams = list_cameras()
    if len(cams) == 2:
        return cams[0]["serial"], cams[1]["serial"]
    raise SystemExit("specify --left and --right serials (could not auto-resolve)")


def _intr_dict(serial):
    cal = calibration.find(serial)
    if cal is None or cal.intrinsics is None:
        raise SystemExit(f"no intrinsics for {serial}: run charuco_calibrate.py first")
    i = cal.intrinsics
    return {"model": i.model, "image_size": list(i.image_size),
            "K": i.K.tolist(), "dist": i.dist.tolist()}


def _capture_loop(pipe, mailbox, stop):
    while not stop.is_set():
        jpg = pipe.process_once()
        if jpg is not None:
            mailbox.put(jpg)


def _run_cameras(args):
    web_dir = Path(__file__).resolve().parent / "vr_web"
    left_serial, right_serial = _resolve_pair(args)
    left_intr, right_intr = _intr_dict(left_serial), _intr_dict(right_serial)

    # stereo R: prefer the calibration file; else assume parallel (identity).
    R = np.eye(3)
    sp = calibration.StereoCalibration.default_path(left_serial, right_serial, "calibrations")
    if sp.exists():
        R = calibration.StereoCalibration.load(sp).R
    else:
        print(f"[warn] no stereo calibration for {left_serial}/{right_serial}; assuming parallel")
    calib = build_calib_payload(left_intr, right_intr, R.tolist(), args.max_fov_deg)

    sources, mailboxes, threads, stop = {}, {}, [], _threading.Event()
    try:
        for name, serial in (("left", left_serial), ("right", right_serial)):
            try:
                sources[name] = CameraSource(serial)
            except Exception as e:
                print(f"[warn] could not open {name} camera {serial}: {e}")
        if not sources:
            raise SystemExit("no cameras opened")
        # Degraded: one camera → mono (feed it to both eyes).
        if len(sources) == 1:
            only = next(iter(sources.values()))
            sources = {"left": only, "right": only}
            print("[warn] one camera only → mono to both eyes")
        for name in ("left", "right"):
            mb = Mailbox(); mailboxes[name] = mb
            pipe = EyePipeline(sources[name], out_width=args.width, quality=args.quality)
            t = _threading.Thread(target=_capture_loop, args=(pipe, mb, stop), daemon=True)
            t.start(); threads.append(t)

        tethered = adb_reverse(args.port)
        lan_ip = _lan_ip()
        url = choose_url(tethered, lan_ip, args.port)
        print(f"\n  Open in the Quest browser:  {url}\n"
              f"  ({'USB-tethered (adb reverse)' if tethered else 'wifi — accept the cert warning'})\n")

        app = build_app(calib, mailboxes, web_dir, send_fps=args.fps)
        if tethered:
            web.run_app(app, host="127.0.0.1", port=args.port, print=None)
        else:
            cert, key = make_self_signed_cert(web_dir.parent)
            import ssl
            ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ctx.load_cert_chain(cert, key)
            web.run_app(app, host="0.0.0.0", port=args.port, ssl_context=ctx, print=None)
    finally:
        stop.set()
        for s in set(sources.values()):
            s.close()


def _lan_ip():
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80)); return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()
```

- [ ] **Step 2: Wire flags + dispatch in `main()`**

In `main()`, add these arguments before `args = ap.parse_args()`:

```python
    ap.add_argument("--left", help="left camera serial")
    ap.add_argument("--right", help="right camera serial")
```

and replace the final `raise SystemExit(...)` line with:

```python
    _run_cameras(args)
```

- [ ] **Step 3: Verify the suite still passes (no regressions)**

Run: `pytest -v`
Expected: all PASS.

- [ ] **Step 4: Live headset verification (hardware + Quest required)**

```bash
python examples/vr_passthrough.py --left <L_SERIAL> --right <R_SERIAL>
```

Tether the Quest by USB-C, open the printed `http://localhost:8000`, Enter VR.
Expected: live stereo wide-FOV feed; turning your head looks around within the captured FOV; smooth at headset refresh even though imagery updates at ~`--fps`; FOV edge is black where the lens stops; white-balanced (mild green cast OK).

- [ ] **Step 5: Commit**

```bash
git add examples/vr_passthrough.py
git commit -m "feat(vr): live stereo camera capture end-to-end"
```

---

## Task 10: Stereo R co-alignment (comfort)

**Files:**
- Modify: `examples/vr_web/app.js`
- Test: `tests/test_host_pipeline.py`

The host already sends `R` (OpenCV frame, row-major). The client orients the right sphere so the two images fuse without vertical offset/roll. Right-frame→left-frame rotation is `Rᵀ`; converted to Three's basis it is `B·Rᵀ·B` with `B=diag(1,-1,-1)`.

- [ ] **Step 1: Add a parity test for the basis conversion (pure JS mirrored in Python)**

Append to `tests/test_host_pipeline.py`:

```python
from vr_passthrough import opencv_R_to_three_right

def test_basis_conversion_identity():
    out = opencv_R_to_three_right(np.eye(3).ravel().tolist())
    assert np.allclose(np.array(out).reshape(3, 3), np.eye(3))

def test_basis_conversion_known_yaw():
    # 10 deg yaw about OpenCV y(down) -> Three y(up) flips sign of the angle.
    a = np.deg2rad(10.0)
    R = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    out = np.array(opencv_R_to_three_right(R.ravel().tolist())).reshape(3, 3)
    B = np.diag([1.0, -1.0, -1.0])
    assert np.allclose(out, B @ R.T @ B)
```

- [ ] **Step 2: Run, verify failure**

Run: `pytest tests/test_host_pipeline.py -k basis_conversion -v`
Expected: FAIL — `ImportError` for `opencv_R_to_three_right`.

- [ ] **Step 3: Implement the reference (host) and verify the client mirrors it**

Add to `examples/vr_passthrough.py` (this documents the exact matrix the client computes; it is also unit-tested):

```python
def opencv_R_to_three_right(R_flat):
    """OpenCV stereo R (row-major flat 9) → Three.js right-sphere rotation
    matrix (row-major flat 9): B · Rᵀ · B,  B = diag(1,-1,-1)."""
    R = np.asarray(R_flat, float).reshape(3, 3)
    B = np.diag([1.0, -1.0, -1.0])
    return [float(x) for x in (B @ R.T @ B).ravel()]
```

- [ ] **Step 4: Run, verify pass**

Run: `pytest tests/test_host_pipeline.py -k basis_conversion -v`
Expected: PASS (both).

- [ ] **Step 5: Apply R to the right sphere in app.js**

In `examples/vr_web/app.js`, inside `applyCalib(calib)`, after the per-eye uniform loop, add:

```js
  // Co-align the right eye: B * R^T * B (B = diag(1,-1,-1)), R row-major OpenCV.
  const R = calib.R;
  const Rt = [R[0], R[3], R[6], R[1], R[4], R[7], R[2], R[5], R[8]]; // transpose
  const b = [1, -1, -1];
  const m = new THREE.Matrix3();
  // (B*Rt*B)[i][j] = b[i]*Rt[i][j]*b[j]
  m.set(
    b[0]*Rt[0]*b[0], b[0]*Rt[1]*b[1], b[0]*Rt[2]*b[2],
    b[1]*Rt[3]*b[0], b[1]*Rt[4]*b[1], b[1]*Rt[5]*b[2],
    b[2]*Rt[6]*b[0], b[2]*Rt[7]*b[1], b[2]*Rt[8]*b[2]);
  const e = m.elements; // column-major
  const m4 = new THREE.Matrix4().set(
    e[0], e[3], e[6], 0,
    e[1], e[4], e[7], 0,
    e[2], e[5], e[8], 0,
    0,    0,    0,    1);
  eyes.right.mesh.quaternion.setFromRotationMatrix(m4);
```

- [ ] **Step 6: Live verification**

Re-run the camera command from Task 9, Step 4. Look at an object visible to both eyes near the center.
Expected: it fuses comfortably with **no vertical offset / roll** between the eyes (compare to before — vertical misalignment should be reduced). Horizontal disparity (depth) remains.
If vertical offset got worse, the rotation sense is inverted: swap `Rt` back to `R` (use `R` instead of the transpose) — but the parity test pins `Rᵀ` as correct for the documented `X_right = R·X_left + t` convention.

- [ ] **Step 7: Commit**

```bash
git add examples/vr_web/app.js examples/vr_passthrough.py tests/test_host_pipeline.py
git commit -m "feat(vr): stereo R co-alignment of the right eye"
```

---

## Task 11: Failure handling polish + client error messaging

**Files:**
- Modify: `examples/vr_passthrough.py`, `examples/vr_web/app.js`
- Test: `tests/test_host_pipeline.py`

- [ ] **Step 1: Write tests for the pure failure-path helpers**

Append to `tests/test_host_pipeline.py`:

```python
import pytest
from vr_passthrough import _intr_dict_from_cal

def test_missing_intrinsics_fails_fast():
    class _Cal: intrinsics = None
    with pytest.raises(SystemExit, match="charuco_calibrate"):
        _intr_dict_from_cal(_Cal(), "SERIAL123")
```

- [ ] **Step 2: Run, verify failure**

Run: `pytest tests/test_host_pipeline.py -k missing_intrinsics -v`
Expected: FAIL — `ImportError` for `_intr_dict_from_cal`.

- [ ] **Step 3: Refactor `_intr_dict` to a testable pure helper**

In `examples/vr_passthrough.py`, replace `_intr_dict` with:

```python
def _intr_dict_from_cal(cal, serial):
    if cal is None or getattr(cal, "intrinsics", None) is None:
        raise SystemExit(f"no intrinsics for {serial}: run charuco_calibrate.py first")
    i = cal.intrinsics
    return {"model": i.model, "image_size": list(i.image_size),
            "K": i.K.tolist(), "dist": i.dist.tolist()}


def _intr_dict(serial):
    return _intr_dict_from_cal(calibration.find(serial), serial)
```

- [ ] **Step 4: Run, verify pass**

Run: `pytest tests/test_host_pipeline.py -k missing_intrinsics -v`
Expected: PASS.

- [ ] **Step 5: Add the client secure-context / XR-unavailable message**

In `examples/vr_web/index.html`, add inside `<body>` before the script tag:

```html
  <div id="msg" style="position:fixed;top:12px;left:12px;color:#ddd;
       font:14px sans-serif;background:#000a;padding:8px;border-radius:6px"></div>
```

In `examples/vr_web/app.js`, after the renderer is created, add:

```js
const msg = document.getElementById('msg');
if (!window.isSecureContext) {
  msg.textContent = 'Not a secure context — open the tethered http://localhost URL, '
    + 'or accept the HTTPS cert on the wifi URL. WebXR will not start otherwise.';
} else if (!navigator.xr) {
  msg.textContent = 'WebXR not available in this browser.';
} else {
  navigator.xr.isSessionSupported('immersive-vr').then((ok) => {
    if (!ok) msg.textContent = 'immersive-vr not supported on this device.';
  });
}
```

- [ ] **Step 6: Run the full suite**

Run: `pytest -v`
Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add examples/vr_passthrough.py examples/vr_web/index.html examples/vr_web/app.js tests/test_host_pipeline.py
git commit -m "feat(vr): fail-fast on missing calib + client secure-context messaging"
```

---

## Task 12: Docs

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add a VR passthrough row to the Examples table**

In `README.md`, in the Examples table (after the `charuco_calibrate.py` row), add:

```markdown
| `examples/vr_passthrough.py` | Live stereo VR passthrough to a Quest 3S browser (WebXR). |
```

- [ ] **Step 2: Add a usage subsection after the examples table**

In `README.md`, after the examples table block, add:

````markdown
### VR passthrough (Quest 3S)

Live stereo, look-around-the-world passthrough in the headset browser.

```bash
python examples/vr_passthrough.py            # auto-resolves the calibrated stereo pair
python examples/vr_passthrough.py --test-pattern   # no hardware: synthetic grid
```

**Tethered (recommended):** enable Quest developer mode + USB debugging, install
`adb` (Android platform-tools), connect USB-C. The script runs `adb reverse` and
prints `http://localhost:<port>` — open it in the Quest browser and tap *Enter VR*.
`localhost` is a secure context, so no TLS is needed.

**Wifi fallback:** without a tether the script serves HTTPS with a self-signed cert
on the LAN IP; accept the browser warning once. Use 5 GHz / Wi-Fi 6.

Requires per-camera fisheye intrinsics (`charuco_calibrate.py`) and, for comfortable
stereo, the stereo `R` (also from `charuco_calibrate.py`). Knobs: `--width`,
`--fps`, `--quality`, `--max-fov-deg`, `--port`.
````

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs(vr): document the VR passthrough example"
```

---

## Self-Review

**Spec coverage:**
- WebXR client → Tasks 7–11. True stereo (L/R eyes via layers) → Tasks 7, 9. Fisheye-sphere projection (B) → Tasks 1, 8. Equirect (C) → explicitly deferred (documented in File Structure note + spec future work). Host-agnostic capture → Tasks 3, 9 (sources behind a `read()` interface). MJPEG/WebSocket transport → Tasks 2, 6. Tethered adb-reverse + localhost secure context → Tasks 5, 9. Wifi HTTPS/WSS fallback → Tasks 5, 9. Newest-wins decoupling → Tasks 2, 6, 8. Color (WB + BT.709) → Task 3. Resolution/fps/quality knobs → Tasks 3, 8, 9. Stereo R co-alignment → Task 10. Head rotation only / sphere recenters → Task 8. Failure handling (missing calib fail-fast, missing stereo→R=I, one camera→mono, ws reconnect, secure-context message) → Tasks 9, 11. Testing (KB parity, synthetic source, desktop preview, manual checklist) → Tasks 1, 3, 8, 9. Docs → Task 12. **No uncovered spec requirements.**

**Placeholder scan:** No "TBD/TODO/handle edge cases" steps; every code step shows complete code; every verification step states an explicit expected result.

**Type/name consistency:** `Mailbox.put/get_latest`, `encode_jpeg(rgb, quality)`, `EyePipeline.process_once`, `TestPatternSource(width,height,eye).read()`, `CameraSource(serial).read()`, `build_calib_payload(left_intr, right_intr, stereo_R, max_fov_deg)`, `build_app(calib, mailboxes, web_dir, send_fps)`, `choose_url(tethered, lan_ip, port)`, `make_self_signed_cert(out_dir)`, `opencv_R_to_three_right(R_flat)`, `_intr_dict_from_cal(cal, serial)` are used consistently across tasks and tests. WS protocol (text calib, then `[eye_byte]+jpeg`; 0=left, 1=right) is identical in Task 6 (server) and Task 8 (client). Shader uniform names match between `shaders.js` and `applyCalib` in `app.js`.

**Note for the executor:** Tasks 1–6, 10, 11 run fully on the Mac with no hardware (`pytest`). Task 8 verifies on the Mac desktop browser (no headset). Tasks 7, 9, and the comfort check in 10 require the Quest 3S (and Task 9 the cameras). Sign/orientation issues in the shader or stereo-R surface only at those manual steps — each lists the exact symptom and the knob to adjust.
