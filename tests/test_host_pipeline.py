import asyncio
import json
from pathlib import Path

import numpy as np
import cv2
import aiohttp
from aiohttp import web

from vr_passthrough import (
    Mailbox, encode_jpeg, TestPatternSource, EyePipeline, build_calib_payload,
    choose_url, make_self_signed_cert, build_app)


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
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = runner.addresses[0][1]
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
