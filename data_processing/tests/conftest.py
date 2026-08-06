"""Puts each source subpackage on sys.path so tests can `import module_under_test`
the same way they did when they lived next to their modules.
"""
import os
import sys

_data_processing = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _sub in ("", "hands", "vio", "voice"):
    sys.path.insert(0, os.path.join(_data_processing, _sub))
