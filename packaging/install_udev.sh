#!/usr/bin/env bash
# Install udev rules so non-root users can open HTENG/MindVision USB cameras.
#
# The Python package itself needs no system install — it loads its bundled
# libMVSDK.so directly. But on Linux the USB device nodes are root-only by
# default; these rules set MODE=0666 for the camera's vendor IDs so a normal
# user (and thus a pip-installed venv) can claim the device.
#
# Run once per machine:   sudo ./install_udev.sh
set -euo pipefail

if [[ $EUID -ne 0 ]]; then
    echo "Must run as root: sudo $0" >&2
    exit 1
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

install -m 0644 "$HERE/99-mvusb.rules" /etc/udev/rules.d/99-mvusb.rules
install -m 0644 "$HERE/88-mvusb.rules" /etc/udev/rules.d/88-mvusb.rules
echo "Installed udev rules to /etc/udev/rules.d/"

udevadm control --reload-rules
udevadm trigger
echo "Reloaded udev rules. Replug the camera to apply."
