#!/usr/bin/env bash
# Build the hteng_fast native kernel into the package's _libs/<platform>/ dir,
# right next to the vendor SDK binary. ctypes loads it from there at runtime, so
# the wheel ships a prebuilt binary and the target needs no compiler.
#
# Flags are portable on purpose: the kernels are memory-bandwidth bound, so
# -march=native buys nothing measurable while breaking on older CPUs. Plain -O3
# is safe to ship on any x86-64 / arm64.
#
#   ./build.sh            # build for the current platform/arch
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$HERE/src/hteng_fast.cpp"
PKG_LIBS="$HERE/../src/hteng_camera/_libs"

uname_s="$(uname -s)"
uname_m="$(uname -m)"

case "$uname_s" in
    Linux)
        case "$uname_m" in
            x86_64|amd64) subdir="linux-x86_64" ;;
            aarch64|arm64) subdir="linux-aarch64" ;;
            *) echo "unsupported Linux arch: $uname_m" >&2; exit 1 ;;
        esac
        out="$PKG_LIBS/$subdir/libhteng_fast.so"
        CXX="${CXX:-g++}"
        flags="-O3 -std=c++17 -fPIC -shared -pthread"
        ;;
    Darwin)
        subdir="darwin-arm64"
        out="$PKG_LIBS/$subdir/libhteng_fast.dylib"
        CXX="${CXX:-clang++}"
        flags="-O3 -std=c++17 -fPIC -dynamiclib"
        ;;
    *)
        echo "unsupported OS: $uname_s" >&2; exit 1 ;;
esac

mkdir -p "$(dirname "$out")"
echo "building $out  ($CXX $flags)"
$CXX $flags "$SRC" -o "$out"
echo "done: $out"
