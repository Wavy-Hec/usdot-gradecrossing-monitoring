"""AI grade-crossing monitor: zone stabilisation, vehicle tracking, stuck-vehicle alerting, debris detection.

Importing the package stays light (config + types only). ``Monitor``, ``open_source`` and
``draw_overlay`` are exposed lazily so that ``import gradecrossing`` (and
``python -m gradecrossing.evaluate``) does not pull in OpenCV until something needs it.
"""
from __future__ import annotations

import importlib
from typing import Any

__version__ = "0.3.0"  # the one place the version is written; pyproject.toml reads it from here

from .config import MonitorConfig, load_config  # noqa: F401
from .types import Assessment, Detection, Event, FrameResult, Status  # noqa: F401

# name -> (module, attribute), resolved on first access (PEP 562).
_LAZY = {
    "Monitor": ("gradecrossing.monitor", "Monitor"),
    "open_source": ("gradecrossing.monitor", "open_source"),
    "draw_overlay": ("gradecrossing.visualize", "draw_overlay"),
}

__all__ = [
    "Assessment",
    "Detection",
    "Event",
    "FrameResult",
    "Monitor",
    "MonitorConfig",
    "Status",
    "__version__",
    "draw_overlay",
    "load_config",
    "open_source",
]


def __getattr__(name: str) -> Any:
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module 'gradecrossing' has no attribute {name!r}")
    value = getattr(importlib.import_module(target[0]), target[1])
    globals()[name] = value  # cache so the next access is a plain attribute lookup
    return value
