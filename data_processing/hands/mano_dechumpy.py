"""Rewrite a MANO ``.pkl`` so its chumpy arrays become plain NumPy arrays.

WiLoR's MANO model ships as a pickle containing ``chumpy.Ch`` objects. chumpy
0.70 is unmaintained and does not import on numpy >= 2 / py >= 3.11 (it uses
removed aliases like ``np.bool`` and ``inspect.getargspec``). Rather than keep
chumpy importable at runtime, we convert the pickle *once* to pure numpy; after
that nothing in WiLoR/smplx ever needs chumpy again.

Usage (in an env where chumpy is installed, e.g. ``pip install --no-deps
--no-build-isolation chumpy``)::

    python mano_dechumpy.py /path/to/MANO_RIGHT.pkl

The first conversion preserves the upstream pickle as ``.chumpy.bak``. Repeated
runs detect an already-clean model and leave both files unchanged.
"""
import argparse
import inspect
import os
import pickle
import shutil
import tempfile
import warnings

import numpy as np

# --- shim numpy aliases removed in numpy 2.x so ancient chumpy imports ---
for _name, _t in {"bool": bool, "int": int, "float": float, "complex": complex,
                   "object": object, "str": str, "unicode": str}.items():
    if _name not in np.__dict__:
        setattr(np, _name, _t)

# --- shim inspect.getargspec, removed in py3.11, used by chumpy ---
if not hasattr(inspect, "getargspec"):
    from collections import namedtuple
    _ArgSpec = namedtuple("ArgSpec", "args varargs keywords defaults")

    def _getargspec(func):
        fs = inspect.getfullargspec(func)
        return _ArgSpec(fs.args, fs.varargs, fs.varkw, fs.defaults)

    inspect.getargspec = _getargspec

def is_chumpy(obj):
    """Identify a loaded chumpy value without retaining a runtime dependency."""
    return type(obj).__module__.split(".", 1)[0] == "chumpy"


def to_numpy(obj):
    """Recursively replace chumpy arrays with their evaluated numpy values."""
    if is_chumpy(obj):
        return np.asarray(obj.r)
    if isinstance(obj, np.ndarray):
        return obj
    if isinstance(obj, dict):
        return {k: to_numpy(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_numpy(v) for v in obj)
    return obj


def has_chumpy(obj):
    if is_chumpy(obj):
        return True
    if isinstance(obj, dict):
        return any(has_chumpy(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return any(has_chumpy(v) for v in obj)
    return False


def convert_model(path):
    """Convert ``path`` in place, returning whether a rewrite was required."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with open(path, "rb") as f:
                data = pickle.load(f, encoding="latin1")
    except ModuleNotFoundError as exc:
        if exc.name == "chumpy":
            raise RuntimeError(
                f"{path} is still chumpy-backed, but chumpy is not installed.\n"
                "Install it with:\n"
                "  python -m pip install --no-deps --no-build-isolation chumpy\n"
                "Then rerun prepare_wilor_models.py. Removing chumpy afterward "
                "is optional.") from exc
        raise
    except Exception as exc:
        raise RuntimeError(
            f"could not read MANO pickle {path}; restore it from "
            f"{path}.chumpy.bak or rerun prepare_wilor_models.py") from exc

    if not has_chumpy(data):
        print(f"already converted: {path}")
        return False

    clean = to_numpy(data)
    if has_chumpy(clean):
        raise RuntimeError(f"chumpy objects remain after converting {path}")

    backup = path + ".chumpy.bak"
    if not os.path.exists(backup):
        shutil.copy2(path, backup)
        print(f"preserved original: {backup}")

    directory = os.path.dirname(os.path.abspath(path))
    fd, temporary = tempfile.mkstemp(
        prefix=".mano_dechumpy_", suffix=".pkl", dir=directory)
    try:
        with os.fdopen(fd, "wb") as f:
            pickle.dump(clean, f)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)

    keys = list(clean.keys()) if isinstance(clean, dict) else type(clean)
    print(f"converted: {path}\nkeys: {keys}")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="path to WiLoR's MANO_RIGHT.pkl")
    args = parser.parse_args()
    path = os.path.expanduser(args.model)
    if not os.path.isfile(path):
        parser.error(
            f"MANO model not found: {path}\n"
            "Run data_processing/hands/prepare_wilor_models.py to download it.")
    try:
        convert_model(path)
    except RuntimeError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
