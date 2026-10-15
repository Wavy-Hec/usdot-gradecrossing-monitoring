"""Console-script entry points (``[project.scripts]`` in pyproject.toml).

``gradecrossing-monitor`` reads frames, so it needs OpenCV, which only the extras install
(the base install is numpy + PyYAML). A script pointed straight at
``gradecrossing.monitor:main`` crashed on a base install with a ModuleNotFoundError
traceback before argparse ran, even for ``--help``. :func:`monitor` imports the monitor
inside the call instead and turns a missing (or unloadable) OpenCV into one line on stderr
that names the extra to install, with exit status 2: like a configuration error, it is a
setup mistake that running again cannot fix (1 stays "the crossing is not being watched").
``gradecrossing-evaluate`` needs no OpenCV and points at ``gradecrossing.evaluate:main``.
"""
from __future__ import annotations

import sys
from typing import Optional, Sequence

# The monitor's exit status for a configuration / usage error (gradecrossing.monitor.EXIT_CONFIG,
# which cannot be imported when the monitor itself cannot be).
EXIT_CONFIG = 2

OPENCV_INSTALL_HINT = (
    "install an OpenCV extra: from a checkout, pip install \".[detector]\" (headless: Jetson, Colab, "
    "servers; same as requirements.txt) or \".[display]\" (GUI OpenCV for --display; "
    "requirements-display.txt)")


def monitor(argv: Optional[Sequence[str]] = None) -> int:
    """``gradecrossing-monitor``: :func:`gradecrossing.monitor.main`, or exit status 2 with one
    line when the monitor cannot be imported because a third-party module (OpenCV) is missing
    or broken. An import error inside the package itself is a bug and keeps its traceback."""
    try:
        from .monitor import main
    except ImportError as exc:
        if str(getattr(exc, "name", None) or "").split(".")[0] == __package__:
            raise
        print(f"error: gradecrossing-monitor cannot start ({' '.join(str(exc).split())}): {OPENCV_INSTALL_HINT}",
              file=sys.stderr)
        return EXIT_CONFIG
    return main(argv)
