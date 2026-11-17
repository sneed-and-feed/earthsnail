"""Unittest discovery entrypoint for EarthSnail Python bindings (`python -m unittest discover -s python`)."""

import pathlib
import sys

_PY_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_PY_ROOT) not in sys.path:
    sys.path.insert(0, str(_PY_ROOT))

from earthsnail.selftest import EarthSnailBindingsTest  # noqa: E402, F401
