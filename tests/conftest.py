import sys
from pathlib import Path

# Make the importable host helpers in examples/vr_passthrough.py available as
# `import vr_passthrough` without turning examples/ into a package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))
