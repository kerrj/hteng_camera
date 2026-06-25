"""One-time: rewrite a MANO ``.pkl`` so its chumpy arrays become plain numpy.

WiLoR's MANO model ships as a pickle containing ``chumpy.Ch`` objects. chumpy
0.70 is unmaintained and does not import on numpy >= 2 / py >= 3.11 (it uses
removed aliases like ``np.bool`` and ``inspect.getargspec``). Rather than keep
chumpy importable at runtime, we convert the pickle *once* to pure numpy; after
that nothing in WiLoR/smplx ever needs chumpy again.

Usage (in an env where chumpy is installed, e.g. ``pip install --no-deps
--no-build-isolation chumpy``)::

    python mano_dechumpy.py /path/to/MANO_RIGHT.pkl

Writes a ``.chumpy.bak`` next to the original, then overwrites it in place.
"""
import inspect
import pickle
import shutil
import sys

import numpy as np

# --- shim numpy aliases removed in numpy 2.x so ancient chumpy imports ---
for _name, _t in {"bool": bool, "int": int, "float": float, "complex": complex,
                   "object": object, "str": str, "unicode": str}.items():
    if not hasattr(np, _name):
        setattr(np, _name, _t)

# --- shim inspect.getargspec, removed in py3.11, used by chumpy ---
if not hasattr(inspect, "getargspec"):
    from collections import namedtuple
    _ArgSpec = namedtuple("ArgSpec", "args varargs keywords defaults")

    def _getargspec(func):
        fs = inspect.getfullargspec(func)
        return _ArgSpec(fs.args, fs.varargs, fs.varkw, fs.defaults)

    inspect.getargspec = _getargspec

import chumpy as ch  # noqa: E402  (must follow the shims above)


def to_numpy(obj):
    """Recursively replace chumpy arrays with their evaluated numpy values."""
    if isinstance(obj, ch.Ch):
        return np.asarray(obj.r)
    if isinstance(obj, np.ndarray):
        return obj
    if isinstance(obj, dict):
        return {k: to_numpy(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_numpy(v) for v in obj)
    return obj


def has_chumpy(obj):
    if isinstance(obj, ch.Ch):
        return True
    if isinstance(obj, dict):
        return any(has_chumpy(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return any(has_chumpy(v) for v in obj)
    return False


def main(path):
    shutil.copy2(path, path + ".chumpy.bak")
    with open(path, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    clean = to_numpy(data)
    assert not has_chumpy(clean), "chumpy objects remain after conversion!"
    with open(path, "wb") as f:
        pickle.dump(clean, f)
    keys = list(clean.keys()) if isinstance(clean, dict) else type(clean)
    print(f"CONVERTED {path}\nkeys: {keys}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python mano_dechumpy.py /path/to/MANO_RIGHT.pkl")
    main(sys.argv[1])
