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
