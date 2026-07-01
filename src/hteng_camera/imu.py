"""YbImu — a clean interface to the Yahboom YbImu 9-axis IMU (USB-serial).

Transport: pyserial, 115200 baud, a small checksummed frame protocol
(``0x7E 0x23 <len> <func> <payload...> <checksum>``). Once :meth:`YbImu.start`
is running, the board continuously auto-broadcasts three frame types
regardless of fusion mode: raw accelerometer+gyroscope+magnetometer, an
onboard fused quaternion, and onboard Euler angles. The report rate is
configurable via :meth:`YbImu.set_report_rate` in [10, 100] Hz, but 100 Hz is
also a verified FIRMWARE ceiling for this board — requesting a higher rate
(even bypassing this class's clamp) has no effect, confirmed by direct
testing.

Why this module only exposes raw accel/gyro/mag, not the onboard fusion
------------------------------------------------------------------------
:meth:`set_algo_type` selects the board's ONBOARD fusion mode (6-axis:
accel+gyro; 9-axis: adds the magnetometer). Raw accelerometer and gyroscope
values are unaffected by this and always available — BUT the raw
magnetometer channel is NOT: in 6-axis mode the firmware appears to stop
sampling the magnetometer entirely, and ``get_magnetometer()``/the raw frame
report a constant ``(0.0, 0.0, 0.0)`` regardless of motion, confirmed by
direct testing (switching to 9-axis makes real magnetometer values reappear
immediately). If you need magnetometer data for anything — including just
logging it for later offline 9-axis fusion comparisons — you must run in
9-axis mode; 6-axis mode silently zeros that column, it doesn't just exclude
it from the onboard fusion math. For an egomotion capture rig feeding a
downstream VIO/SLAM pipeline, log the RAW stream via :meth:`start`'s
``on_raw_sample`` callback, not the onboard fusion: cheap AHRS filters
(complementary/Mahony/Madgwick-style, which is almost certainly what this
board runs internally) assume the accelerometer measures gravity, i.e. they
assume near-static motion. During real translational acceleration — exactly
what egomotion capture involves — they can't distinguish gravity from linear
acceleration and misattribute real motion as tilt error. A VIO backend doing
its own IMU preintegration from raw data, with known noise characteristics,
does not have this failure mode. (Yahboom's own ROS2 reference package
corroborates this: it republishes raw ``imu/data_raw`` and re-fuses it
host-side via a Madgwick filter, rather than trusting the onboard quaternion
downstream.)

Calibration
-----------
:meth:`calibrate_imu` (accelerometer + gyroscope bias) requires the board to
be stationary and flat for several seconds, and persists to the board's
flash — a one-time step per physical mount, not a per-session one.
:meth:`calibrate_mag` (magnetometer hard/soft-iron correction) requires
slowly rotating the board through all axes, and is environment-dependent —
redo it if the mount or nearby ferrous/magnetic material changes.

Capturing samples: callback, not polling
-----------------------------------------
:meth:`start` takes an optional ``on_raw_sample`` callback, invoked
synchronously from the receive thread for every validated raw frame. This is
the only way to capture a complete, non-duplicated sample stream — the board
pushes frames asynchronously at its own rate, so polling
:meth:`get_accelerometer`/:meth:`get_gyroscope` on an independent timer can
skip or duplicate samples relative to the wire rate.

Scope
-----
Only the USB-serial transport is ported here. The vendor library also ships
an I2C transport (via ``smbus2``), left out deliberately: no current hardware
uses it, and it needs a Linux I2C bus unavailable on this project's macOS
capture host. If a future embedded-Linux capture host needs it, add
``imu_i2c.py`` mirroring this module's public surface. Likewise, the vendor
library's barometer support, temperature-compensation calibration, and flash
reset command aren't ported — they're unused by this project's capture
pipeline and this board's barometer channel reported nothing during testing.

Example (exact port known)::

    from hteng_camera.imu import YbImu

    def on_sample(sample):
        print(sample.host_time_s, sample.ax, sample.gx, sample.mx)

    imu = YbImu("/dev/cu.usbserial-1440")
    imu.set_report_rate(100)
    imu.set_algo_type(6)
    imu.start(on_raw_sample=on_sample)
    ...
    imu.close()

Example (autodetect)::

    imu = YbImu.autodetect()   # None if no YbImu found; already streaming
    if imu is not None:
        imu.on_raw_sample = on_sample
"""

import math
import struct
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

import serial
from serial.tools import list_ports as _list_ports

__all__ = ["YbImu", "ImuSample", "list_candidate_ports"]

# Substrings/VIDs of common USB-serial chips (CH340, CP210x, FTDI) used to
# recognize candidate ports for autodetection without guessing by OS-specific
# device-path naming. CH340 (0x1A86) is what this board's own adapter reports.
_USB_SERIAL_MARKERS = ("usb serial", "usbserial", "usb-serial", "ch340", "cp210", "ftdi")
_USB_SERIAL_VIDS = {0x1A86, 0x10C4, 0x0403}


def list_candidate_ports() -> list[str]:
    """List serial device paths that look like real USB-serial adapters, as
    opposed to e.g. Bluetooth or debug-console ports.

    Used by :meth:`YbImu.autodetect` to avoid probing obviously-unrelated
    serial devices — writing this board's protocol bytes to some other
    serial device (e.g. a camera trigger relay) could have side effects, so
    autodetection is deliberately conservative about what it tries.
    """
    candidates = []
    for p in _list_ports.comports():
        haystack = f"{p.description or ''} {p.manufacturer or ''}".lower()
        if p.vid in _USB_SERIAL_VIDS or any(m in haystack for m in _USB_SERIAL_MARKERS):
            candidates.append(p.device)
    return candidates


@dataclass(frozen=True)
class ImuSample:
    """One raw accelerometer+gyroscope+magnetometer frame, plus the host
    time it was parsed.

    ``host_time_s`` is ``time.perf_counter()`` at the moment this frame was
    parsed off the wire — a host-side receive timestamp, not a hardware
    sample timestamp. It shares no clock epoch with anything else unless the
    caller establishes one (see ``examples/record_ego.py`` for how the
    capture pipeline relates this to camera frame timestamps).
    """

    host_time_s: float
    ax: float
    ay: float
    az: float
    gx: float
    gy: float
    gz: float
    mx: float
    my: float
    mz: float


class YbImu:
    """One Yahboom YbImu 9-axis IMU on a USB-serial transport.

    See the module docstring for the full protocol/design rationale.
    """

    _BAUD_RATE = 115200

    # Frame protocol constants (internal — callers use the methods below,
    # not these codes directly).
    _HEAD1 = 0x7E
    _HEAD2 = 0x23
    _MAX_FRAME_LEN = 40

    _FUNC_VERSION = 0x01
    _FUNC_REPORT_IMU_RAW = 0x04
    _FUNC_REPORT_IMU_QUAT = 0x16
    _FUNC_REPORT_IMU_EULER = 0x26
    _FUNC_REPORT_RATE = 0x60
    _FUNC_ALGO_TYPE = 0x61
    _FUNC_CALIB_IMU = 0x70
    _FUNC_CALIB_MAG = 0x71
    _FUNC_REQUEST_DATA = 0x80
    _FUNC_RETURN_STATE = 0x81

    # Raw-value -> physical-unit scale factors (16-bit signed ADC counts).
    _ACCEL_G_PER_COUNT = 16 / 32767.0
    _GYRO_RAD_PER_COUNT = (2000 / 32767.0) * (math.pi / 180.0)
    _MAG_UT_PER_COUNT = 800.0 / 32767.0

    def __init__(self, port: str, *, debug: bool = False):
        self._debug = debug
        self._dev = serial.Serial(str(port), self._BAUD_RATE)

        # Cached latest sensor state, updated by the receive thread and read
        # by the get_*() methods.
        self._ax = self._ay = self._az = 0.0
        self._gx = self._gy = self._gz = 0.0
        self._mx = self._my = self._mz = 0.0
        self._roll = self._pitch = self._yaw = 0.0
        self._q0, self._q1, self._q2, self._q3 = 1.0, 0.0, 0.0, 0.0
        self._firmware_version: Optional[tuple[int, int, int]] = None

        # Calibration/status-request bookkeeping (see _wait_for_status).
        self._last_status_func = 0
        self._last_status_value = 0

        # Byte-level frame parser state.
        self._parse_state = 0
        self._frame_len = 0
        self._frame_func = 0
        self._frame_payload: list[int] = []

        self.on_raw_sample: Optional[Callable[[ImuSample], None]] = None
        self._stop = threading.Event()
        self._receive_thread: Optional[threading.Thread] = None
        self._started = False

        if not self._dev.is_open:
            raise RuntimeError(f"failed to open YbImu serial port {port!r}")

    def __enter__(self) -> "YbImu":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    @property
    def port(self) -> str:
        """The serial device path this instance is connected to."""
        return self._dev.port

    @classmethod
    def autodetect(cls, timeout_s: float = 0.5, debug: bool = False) -> Optional["YbImu"]:
        """Probe candidate USB-serial ports (see :func:`list_candidate_ports`)
        and return an already-started ``YbImu`` for the first one that
        responds to a version query with this board's protocol.

        Returns ``None`` if no matching device is found. Only ports that
        look like USB-serial adapters are tried — never Bluetooth or
        debug-console ports — since writing this board's protocol bytes to
        unrelated serial hardware (e.g. a camera trigger relay on the same
        USB-serial family) could have side effects.

        The returned instance is already streaming (``start()`` has been
        called with no callback); assign ``imu.on_raw_sample`` afterward to
        start capturing samples.
        """
        for candidate in list_candidate_ports():
            try:
                imu = cls(candidate, debug=debug)
            except Exception:
                continue
            try:
                imu.start()
                if imu.get_version(timeout_s=timeout_s) is not None:
                    return imu
            except Exception:
                pass
            imu.close()
        return None

    # -- lifecycle -----------------------------------------------------

    def start(self, on_raw_sample: Optional[Callable[[ImuSample], None]] = None) -> None:
        """Start the background receive thread.

        If given, ``on_raw_sample`` is called synchronously, from the
        receive thread, for every validated raw accel/gyro/mag frame — the
        supported way to capture a complete sample stream (see the module
        docstring on why polling the getters instead can skip/duplicate
        samples). A callback that raises is logged and skipped rather than
        killing the receive thread.
        """
        self.on_raw_sample = on_raw_sample
        self._dev.reset_input_buffer()
        self._receive_thread = threading.Thread(
            target=self._receive_loop, name="ybimu-receive", daemon=True,
        )
        self._receive_thread.start()
        self._started = True

    def close(self) -> None:
        """Stop the receive thread and close the serial port."""
        self._stop.set()
        if self._receive_thread is not None:
            self._receive_thread.join(timeout=1.0)
        self._dev.close()

    # -- wire protocol ---------------------------------------------------

    def _log(self, label: str, frame: bytes) -> None:
        if self._debug:
            print(f"{label}: [{frame.hex().upper()}]")

    def _require_started(self, what: str) -> None:
        """Guard against a footgun found by direct hardware testing: sending
        a command before the receive thread is draining the serial port can
        silently wedge the link — the device stops responding to ANYTHING,
        including future commands, with no exception and no error. The
        vendor's own reference code avoids this by always starting the
        receive thread immediately after opening the port, before issuing
        any other command; enforce the same order here instead of leaving it
        as a comment someone can miss.
        """
        if not self._started:
            raise RuntimeError(
                f"YbImu.start() must be called before {what}() — sending a "
                "command before the receive thread is running can silently "
                "wedge the serial link."
            )

    def _build_command(self, func: int, payload: bytes = b"") -> bytes:
        header = bytes([self._HEAD1, self._HEAD2])
        # Length counts everything from HEAD1 through the checksum byte.
        length = 2 + 1 + 1 + len(payload) + 1
        body = header + bytes([length, func]) + payload
        checksum = sum(body) & 0xFF
        return body + bytes([checksum])

    def _send(self, func: int, payload: bytes = b"", label: str = "send") -> None:
        frame = self._build_command(func, payload)
        self._dev.write(frame)
        self._log(label, frame)

    def _receive_loop(self) -> None:
        while not self._stop.is_set():
            if self._dev.in_waiting <= 0:
                time.sleep(0.001)
                continue
            for byte in self._dev.read_all():
                self._feed_byte(byte)

    def _feed_byte(self, byte: int) -> None:
        """Frame state machine: HEAD1 -> HEAD2 -> length -> func -> payload
        -> checksum. On a validated checksum, dispatches to
        :meth:`_handle_frame`; on a mismatch, drops the frame silently
        (debug mode logs it)."""
        if self._parse_state == 0:
            if byte == self._HEAD1:
                self._parse_state = 1
        elif self._parse_state == 1:
            self._parse_state = 2 if byte == self._HEAD2 else 0
        elif self._parse_state == 2:
            self._frame_len = byte
            self._parse_state = 3 if byte <= self._MAX_FRAME_LEN else 0
        elif self._parse_state == 3:
            self._frame_func = byte
            self._frame_payload = []
            self._parse_state = 4
        elif self._parse_state == 4:
            self._frame_payload.append(byte)
            # frame_len counts HEAD1, HEAD2, len, func, payload, and checksum,
            # so the payload is (frame_len - 5) bytes.
            if len(self._frame_payload) >= self._frame_len - 5:
                self._parse_state = 5
        elif self._parse_state == 5:
            self._parse_state = 0
            checksum = (
                self._HEAD1 + self._HEAD2 + self._frame_len + self._frame_func
                + sum(self._frame_payload)
            ) & 0xFF
            if byte == checksum:
                self._handle_frame(self._frame_func, self._frame_payload)
            elif self._debug:
                print(f"checksum mismatch: got {byte}, expected {checksum}")

    def _handle_frame(self, func: int, payload: list) -> None:
        """Update cached sensor state for a validated frame, and — for a raw
        accel/gyro/mag frame — invoke the registered sample callback."""
        host_time_s = time.perf_counter()
        data = bytearray(payload)

        if func == self._FUNC_REPORT_IMU_RAW:
            self._ax = struct.unpack_from("h", data, 0)[0] * self._ACCEL_G_PER_COUNT
            self._ay = struct.unpack_from("h", data, 2)[0] * self._ACCEL_G_PER_COUNT
            self._az = struct.unpack_from("h", data, 4)[0] * self._ACCEL_G_PER_COUNT
            self._gx = struct.unpack_from("h", data, 6)[0] * self._GYRO_RAD_PER_COUNT
            self._gy = struct.unpack_from("h", data, 8)[0] * self._GYRO_RAD_PER_COUNT
            self._gz = struct.unpack_from("h", data, 10)[0] * self._GYRO_RAD_PER_COUNT
            self._mx = struct.unpack_from("h", data, 12)[0] * self._MAG_UT_PER_COUNT
            self._my = struct.unpack_from("h", data, 14)[0] * self._MAG_UT_PER_COUNT
            self._mz = struct.unpack_from("h", data, 16)[0] * self._MAG_UT_PER_COUNT
            if self.on_raw_sample is not None:
                sample = ImuSample(
                    host_time_s=host_time_s,
                    ax=self._ax, ay=self._ay, az=self._az,
                    gx=self._gx, gy=self._gy, gz=self._gz,
                    mx=self._mx, my=self._my, mz=self._mz,
                )
                try:
                    self.on_raw_sample(sample)
                except Exception as e:
                    print(f"[warn] YbImu on_raw_sample callback raised: {e}")

        elif func == self._FUNC_REPORT_IMU_EULER:
            self._roll = struct.unpack_from("f", data, 0)[0]
            self._pitch = struct.unpack_from("f", data, 4)[0]
            self._yaw = struct.unpack_from("f", data, 8)[0]

        elif func == self._FUNC_REPORT_IMU_QUAT:
            self._q0 = struct.unpack_from("f", data, 0)[0]
            self._q1 = struct.unpack_from("f", data, 4)[0]
            self._q2 = struct.unpack_from("f", data, 8)[0]
            self._q3 = struct.unpack_from("f", data, 12)[0]

        elif func == self._FUNC_VERSION:
            self._firmware_version = (data[0], data[1], data[2])

        elif func == self._FUNC_RETURN_STATE:
            self._last_status_func = data[0]
            self._last_status_value = data[1]

    def _wait_for_status(self, func: int, timeout_ms: Optional[int]) -> Optional[int]:
        """Poll for a FUNC_RETURN_STATE reply matching ``func`` (used to
        collect the result of a calibration command). Returns the status
        byte (1 = success), or None on timeout."""
        deadline = None if timeout_ms is None else time.monotonic() + timeout_ms / 1000.0
        while True:
            if self._last_status_func == func:
                return self._last_status_value
            if deadline is not None and time.monotonic() >= deadline:
                return None
            time.sleep(0.001)

    # -- configuration -----------------------------------------------------

    def set_report_rate(self, rate: int) -> None:
        """Set the auto-report rate in Hz, clamped to [10, 100].

        100 Hz is also a verified FIRMWARE ceiling for this board — this
        isn't just a defensive clamp; requesting more (even bypassing it)
        has no effect.
        """
        self._require_started("set_report_rate")
        rate = max(10, min(100, int(rate)))
        self._send(self._FUNC_REPORT_RATE, bytes([rate, 0x5F]), "report rate")
        time.sleep(1)  # firmware settle time

    def set_algo_type(self, algo: int) -> None:
        """Select the onboard fusion mode: 6 (accel+gyro) or 9 (+ magnetometer).

        Affects the onboard fused quaternion/Euler output as expected, but
        ALSO the raw magnetometer channel: in 6-axis mode the firmware stops
        sampling it and :meth:`get_magnetometer` reads a constant
        ``(0.0, 0.0, 0.0)`` regardless of motion — confirmed by direct
        testing. Raw accelerometer/gyroscope are unaffected either way. See
        the module docstring for the full explanation.
        """
        self._require_started("set_algo_type")
        if algo not in (6, 9):
            raise ValueError(f"algo must be 6 or 9, got {algo!r}")
        self._send(self._FUNC_ALGO_TYPE, bytes([algo, 0x5F]), "algo type")
        time.sleep(1)  # firmware settle time

    # -- calibration ---------------------------------------------------

    def calibrate_imu(self, timeout_ms: int = 7000) -> Optional[int]:
        """Calibrate accelerometer + gyroscope bias. The board must be
        stationary and flat for the duration of this call. Persists to
        flash — a one-time step per physical mount.

        Returns 1 on success, another status code on failure, or None if no
        result arrived within ``timeout_ms``.
        """
        self._require_started("calibrate_imu")
        self._last_status_func = 0
        self._last_status_value = 0
        self._send(self._FUNC_CALIB_IMU, bytes([0x01, 0x5F]), "calibrate imu")
        return self._wait_for_status(self._FUNC_CALIB_IMU, timeout_ms)

    def calibrate_mag(self, timeout_ms: int = 20000) -> Optional[int]:
        """Calibrate magnetometer hard/soft-iron distortion. Rotate the
        board slowly through all axes for the duration of this call.
        Environment-dependent — redo if the mount or nearby ferrous/magnetic
        material changes.

        Returns 1 on success, another status code on failure, or None if no
        result arrived within ``timeout_ms``.
        """
        self._require_started("calibrate_mag")
        self._last_status_func = 0
        self._last_status_value = 0
        self._send(self._FUNC_CALIB_MAG, bytes([0x01, 0x5F]), "calibrate mag")
        return self._wait_for_status(self._FUNC_CALIB_MAG, timeout_ms)

    # -- reading ---------------------------------------------------------

    def get_accelerometer(self) -> tuple[float, float, float]:
        """Latest cached accelerometer reading, in g."""
        return (self._ax, self._ay, self._az)

    def get_gyroscope(self) -> tuple[float, float, float]:
        """Latest cached gyroscope reading, in rad/s."""
        return (self._gx, self._gy, self._gz)

    def get_magnetometer(self) -> tuple[float, float, float]:
        """Latest cached magnetometer reading, in microtesla.

        Reads a constant ``(0.0, 0.0, 0.0)`` in 6-axis mode — the firmware
        stops sampling the magnetometer, not just excluding it from onboard
        fusion. Call ``set_algo_type(9)`` first if you need real values.
        """
        return (self._mx, self._my, self._mz)

    def get_euler(self, degrees: bool = True) -> tuple[float, float, float]:
        """Latest cached ONBOARD fused attitude, as (roll, pitch, yaw).

        See the module docstring for why this project logs raw data rather
        than this onboard fusion output for downstream VIO use.
        """
        if degrees:
            return (math.degrees(self._roll), math.degrees(self._pitch), math.degrees(self._yaw))
        return (self._roll, self._pitch, self._yaw)

    def get_quaternion(self) -> tuple[float, float, float, float]:
        """Latest cached ONBOARD fused quaternion, as (w, x, y, z).

        See the module docstring for why this project logs raw data rather
        than this onboard fusion output for downstream VIO use.
        """
        return (self._q0, self._q1, self._q2, self._q3)

    def get_version(self, timeout_s: float = 0.5) -> Optional[str]:
        """Query the firmware version, e.g. "V1.0.0". Returns None on
        timeout."""
        self._require_started("get_version")
        self._firmware_version = None
        self._send(self._FUNC_REQUEST_DATA, bytes([self._FUNC_VERSION, 0x00]), "get version")
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self._firmware_version is not None:
                major, minor, patch = self._firmware_version
                return f"V{major}.{minor}.{patch}"
            time.sleep(0.005)
        return None
