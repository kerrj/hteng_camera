import numpy as np
import cv2

from vr_passthrough import Mailbox, encode_jpeg, TestPatternSource, EyePipeline


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
