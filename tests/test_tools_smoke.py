"""Smoke runs of the tools the rest of the suite never executes (ours-cleanup OC-13), without torch.

Every tool's ``--help`` is covered by tests/test_detection.py::test_tool_compiles_and_help_works;
this runs tools/benchmark.py end to end on a synthetic clip with the detector stubbed out.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ENV = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONPATH=str(ROOT))


def _clip(path: Path, n: int = 40, w: int = 640, h: int = 360) -> Path:
    rng = np.random.default_rng(7)
    base = cv2.GaussianBlur(rng.integers(0, 256, (h, w, 3), dtype=np.uint8), (7, 7), 0)  # textured, static
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (w, h))
    assert writer.isOpened()
    for _ in range(n):
        writer.write(base)
    writer.release()
    return path


def test_benchmark_runs_without_a_detector(tmp_path):
    clip = _clip(tmp_path / "clip.avi")
    cfg = tmp_path / "site.yaml"
    cfg.write_text("zone:\n  polygon: [[100, 150], [540, 150], [540, 300], [100, 300]]\n", encoding="utf-8")
    report = tmp_path / "bench.json"
    proc = subprocess.run([sys.executable, str(ROOT / "tools" / "benchmark.py"), "--config", str(cfg),
                           "--source", str(clip), "--no-detector", "--frames", "30", "--warmup", "5", "--include-debris-warmup",
                           "--json", str(report)], capture_output=True, text=True, env=ENV, cwd=str(tmp_path),
                          timeout=300)
    assert proc.returncode == 0, proc.stderr
    stats = json.loads(report.read_text(encoding="utf-8"))
    assert stats["frames_timed"] > 0 and stats["fps"] > 0
    assert "end-to-end" in proc.stdout and proc.stdout.isascii()
