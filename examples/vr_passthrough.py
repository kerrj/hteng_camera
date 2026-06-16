"""HTENG stereo → Quest 3S WebXR passthrough host.

Run:  python examples/vr_passthrough.py
See docs/superpowers/specs/2026-06-15-vr-passthrough-design.md
"""
import argparse
import asyncio
import json
import subprocess
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from aiohttp import web, WSMsgType

from hteng_camera import HTCamera, convert, list_cameras, calibration

try:
    from turbojpeg import TurboJPEG, TJPF_RGB
    _TJ = TurboJPEG()
except Exception:  # libturbojpeg missing or init failed → cv2 fallback
    _TJ = None


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


def _downscale(rgb, out_width):
    h, w = rgb.shape[:2]
    if w == out_width:
        return rgb
    out_h = max(1, round(h * out_width / w))
    return cv2.resize(rgb, (out_width, out_h), interpolation=cv2.INTER_AREA)


class TestPatternSource:
    """Hardware-free source: a lat/long grid with an eye label, so the full
    host→WS→client path and the projection can be validated without cameras."""

    __test__ = False  # not a pytest test class despite the "Test" prefix

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
        self.last_encode_ms = 0.0

    def process_once(self):
        rgb = self.source.read()
        if rgb is None:
            return None
        rgb = _downscale(rgb, self.out_width)
        t0 = time.monotonic()
        jpg = encode_jpeg(rgb, quality=self.quality)
        self.last_encode_ms = (time.monotonic() - t0) * 1000.0
        return jpg


_FRAME_SPEED = {"low": 0, "normal": 1, "high": 2, "super": 3}


class CameraSource:
    """Live HTENG camera → display-ready uint8 RGB (WB + BT.709 tone map).

    Exposure is capped short by default: a long exposure both over-exposes the
    image and throttles the sensor frame rate (long integration = fewer fps)."""

    def __init__(self, serial, exposure_ms=10.0, gain=1.0, auto_exposure=False,
                 frame_speed="high", demosaic="ea", out_width=None):
        self.cam = HTCamera(serial=serial, demosaic_quality=demosaic)
        self.serial = serial
        self.out_width = out_width
        self.cam.set_frame_speed(_FRAME_SPEED[frame_speed])
        if auto_exposure:
            self.cam.set_ae(True)
        else:
            self.cam.set_ae(False)
            self.cam.set_exposure_ms(exposure_ms)
            self.cam.set_analog_gain(gain)

    def read(self):
        rgb16, info = self.cam.grab()
        if rgb16 is None:
            return None
        # Downscale in linear light *before* the tone curve: tonemap then touches
        # far fewer pixels (cheaper), and linear-space resampling is more correct.
        if self.out_width:
            rgb16 = _downscale(rgb16, self.out_width)
        return convert.tonemap_linear(
            rgb16, wb_gains=self.cam.wb_gains, curve="bt709",
            out_dtype=np.uint8)

    def close(self):
        self.cam.close()


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


def opencv_R_to_three_right(R_flat):
    """OpenCV stereo R (row-major flat 9) → Three.js right-sphere rotation
    matrix (row-major flat 9): B · Rᵀ · B,  B = diag(1,-1,-1)."""
    R = np.asarray(R_flat, float).reshape(3, 3)
    B = np.diag([1.0, -1.0, -1.0])
    return [float(x) for x in (B @ R.T @ B).ravel()]


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


EYE_ID = {"left": 0, "right": 1}


def build_app(calib, mailboxes, web_dir, send_fps=30):
    """aiohttp app: static client at /, frame stream at /ws.
    Protocol: text calib JSON on connect, then binary [eye_id_byte]+jpeg."""

    @web.middleware
    async def _no_cache(request, handler):
        resp = await handler(request)
        if not isinstance(resp, web.WebSocketResponse):
            resp.headers["Cache-Control"] = "no-store"  # headset must not cache JS/shaders
        return resp

    app = web.Application(middlewares=[_no_cache])
    web_dir = Path(web_dir)

    async def ws_handler(request):
        ws = web.WebSocketResponse(max_msg_size=0, heartbeat=5.0)
        await ws.prepare(request)
        await ws.send_str(json.dumps(calib))
        period = (1.0 / send_fps) if send_fps and send_fps > 0 else 0.0
        last = {"left": None, "right": None}

        async def sender():
            while not ws.closed:
                sent_any = False
                for eye, mb in mailboxes.items():
                    jpg = mb.get_latest()
                    if jpg is not None and jpg is not last[eye]:
                        last[eye] = jpg
                        await ws.send_bytes(bytes([EYE_ID[eye]]) + bytes(jpg))
                        sent_any = True
                # capped: tick at 1/fps. uncapped (fps<=0): ship newest as soon as
                # it's ready — TCP backpressure self-throttles to the link rate and
                # newest-wins keeps it fresh, so no buffer bloat.
                await asyncio.sleep(period if period else (0.0 if sent_any else 0.001))

        # Run the frame sender alongside draining incoming messages; the
        # `async for` ends as soon as the client closes, so we never leak the
        # handler on disconnect (and the heartbeat catches dead connections).
        task = asyncio.ensure_future(sender())
        try:
            async for msg in ws:
                if msg.type != WSMsgType.TEXT:
                    continue
                try:
                    d = json.loads(msg.data)
                except (ValueError, TypeError):
                    continue
                if d.get("type") == "ping":            # transport RTT probe
                    await ws.send_str(json.dumps({"type": "pong", "t": d.get("t")}))
                elif d.get("type") == "stats":          # client-measured latency
                    print(f"[client] {d.get('fps')} f/s, decode {d.get('decode_ms')}ms, "
                          f"transport rtt {d.get('rtt_ms')}ms", flush=True)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        return ws

    async def index(request):
        return web.FileResponse(web_dir / "index.html")

    app.router.add_get("/", index)
    app.router.add_get("/ws", ws_handler)
    app.router.add_static("/", web_dir)  # app.js, shaders.js, vendor/*
    return app


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
    tethered = adb_reverse(args.port)
    where = ("Quest is USB-tethered — open it in the Quest browser and tap Enter VR"
             if tethered else "on this Mac (desktop preview); drag to look around")
    print(f"\n  Open:  http://localhost:{args.port}\n  ({where})\n")
    web.run_app(app, host="0.0.0.0", port=args.port, print=None)


def _resolve_pair(args):
    """Return (left_serial, right_serial). Explicit flags win; else use the
    stereo calibration file in ./calibrations; else the two enumerated cams."""
    if args.left and args.right:
        return args.left, args.right
    for d in (Path("calibrations"), Path(".")):
        for p in (d.glob("stereo_*_*.json") if d.exists() else []):
            st = calibration.StereoCalibration.load(p)
            return st.serial_left, st.serial_right
    cams = list_cameras()
    if len(cams) == 2:
        return cams[0]["serial"], cams[1]["serial"]
    raise SystemExit("specify --left and --right serials (could not auto-resolve)")


def _intr_dict_from_cal(cal, serial):
    if cal is None or getattr(cal, "intrinsics", None) is None:
        raise SystemExit(f"no intrinsics for {serial}: run charuco_calibrate.py first")
    i = cal.intrinsics
    return {"model": i.model, "image_size": list(i.image_size),
            "K": i.K.tolist(), "dist": i.dist.tolist()}


def _intr_dict(serial):
    return _intr_dict_from_cal(calibration.find(serial), serial)


def _capture_loop(pipe, mailbox, stop, label=""):
    n, t0, enc = 0, time.monotonic(), 0.0
    while not stop.is_set():
        jpg = pipe.process_once()
        if jpg is not None:
            mailbox.put(jpg)
            n += 1
            enc += pipe.last_encode_ms
        now = time.monotonic()
        if now - t0 >= 2.0:
            print(f"[capture {label}] {n / (now - t0):.1f} fps, "
                  f"encode {enc / max(n, 1):.1f}ms", flush=True)
            n, t0, enc = 0, now, 0.0


def _lan_ip():
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80)); return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


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

    sources, mailboxes, threads, stop = {}, {}, [], threading.Event()
    try:
        for name, serial in (("left", left_serial), ("right", right_serial)):
            try:
                sources[name] = CameraSource(
                    serial, exposure_ms=args.exposure_ms, gain=args.gain,
                    auto_exposure=args.ae, frame_speed=args.frame_speed,
                    demosaic=args.demosaic, out_width=args.width)
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
            t = threading.Thread(target=_capture_loop, args=(pipe, mb, stop, name), daemon=True)
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


def main():
    ap = argparse.ArgumentParser(description="HTENG stereo → Quest 3S WebXR passthrough")
    ap.add_argument("--test-pattern", action="store_true",
                    help="serve a synthetic grid instead of cameras (no hardware)")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--fps", type=int, default=0,
                    help="WS send-rate cap (0 = uncapped: deliver every captured frame)")
    ap.add_argument("--quality", type=int, default=85)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--max-fov-deg", type=float, default=150.0)
    ap.add_argument("--left", help="left camera serial")
    ap.add_argument("--right", help="right camera serial")
    ap.add_argument("--exposure-ms", type=float, default=10.0,
                    help="manual exposure in ms (lower = darker + higher fps)")
    ap.add_argument("--gain", type=float, default=1.0, help="analog gain multiplier")
    ap.add_argument("--ae", action="store_true",
                    help="auto-exposure (adapts brightness; overrides --exposure-ms)")
    ap.add_argument("--frame-speed", choices=["low", "normal", "high", "super"],
                    default="high", help="sensor readout speed")
    ap.add_argument("--demosaic", choices=["ea", "bilinear"], default="ea",
                    help="'bilinear' is faster (slight zipper artifacts)")
    args = ap.parse_args()
    if args.test_pattern:
        _run_test_pattern(args)
        return
    _run_cameras(args)


if __name__ == "__main__":
    main()
