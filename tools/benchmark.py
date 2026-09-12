#!/usr/bin/env python
"""Measure per-stage milliseconds and end-to-end FPS of the Monitor on a source.

Why: the Jetson budget is tight and "it feels slow" is not a measurement. This runs the
real ``Monitor.process_frame`` loop (same code path as ``python -m gradecrossing.monitor``)
for N frames, collects the ``timings_ms`` every FrameResult carries, and prints mean /
median / p95 / max per stage plus the share of the total, so the bottleneck is obvious
before anyone tunes ``detect_every_n_frames``, ``resize_width`` or ``imgsz``.

Examples::

    python tools/benchmark.py --config configs/mysite.yaml --frames 600
    python tools/benchmark.py --source videos/clip.mov --weights yolo11n.pt --resize-width 960 --detect-every 2
    python tools/benchmark.py --source videos/clip.mov --no-detector      # zone + debris cost only, no torch

``--no-detector`` swaps in a detector that returns nothing, which is how you measure the
non-neural stages on a machine without ultralytics. Output is ASCII; ``--json`` writes the
numbers for a report.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))  # allow `python tools/benchmark.py` from any cwd

from gradecrossing.config import MonitorConfig, load_config  # noqa: E402
from gradecrossing.monitor import ConfigClassifier, Monitor, open_source  # noqa: E402
from gradecrossing.types import Detection  # noqa: E402

STAGES = ("resize", "zone", "detect", "track", "infra", "debris", "alert", "total")


class NullDetector:
    """Detector that never detects anything (measures every stage except the network)."""

    def __init__(self, cfg: MonitorConfig) -> None:
        self._classify = ConfigClassifier(cfg.detector)

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        return []

    def classify(self, cls_name: str) -> str:
        return self._classify(cls_name)

    def reset(self) -> None:
        pass


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Per-stage timing and FPS of the grade-crossing Monitor.")
    p.add_argument("--config", metavar="YAML", help="config file (default: built-in defaults)")
    p.add_argument("--source", help="video file, camera index or stream URL (overrides video.source)")
    p.add_argument("--weights", help="vehicle detector weights (overrides detector.weights)")
    p.add_argument("--frames", type=int, default=300, help="frames to time (default 300; 0 = whole file)")
    p.add_argument("--warmup", type=int, default=10, help="leading frames excluded from the statistics (default 10)")
    p.add_argument("--detect-every", type=int, help="detector.detect_every_n_frames")
    p.add_argument("--resize-width", type=int, help="video.resize_width")
    p.add_argument("--imgsz", type=int, help="detector.imgsz")
    p.add_argument("--device", help="detector/infrastructure device ('' auto, '0' CUDA, 'cpu')")
    p.add_argument("--no-stabilize", action="store_true", help="zone.stabilize = false")
    p.add_argument("--no-debris", action="store_true", help="debris.enabled = false")
    p.add_argument("--no-detector", action="store_true", help="use a detector that returns nothing (no ultralytics)")
    p.add_argument("--infra-weights", help="enable the infrastructure detector with these weights")
    p.add_argument("--json", metavar="PATH", help="write the statistics as JSON")
    return p


def configure(args: argparse.Namespace) -> MonitorConfig:
    """Load the config, apply the benchmark flags, and silence every output side effect."""
    cfg = load_config(args.config)
    if args.source is not None:
        cfg.video.source = args.source
    if args.weights is not None:
        cfg.detector.weights = args.weights
    if args.detect_every is not None:
        cfg.detector.detect_every_n_frames = max(1, int(args.detect_every))
    if args.resize_width is not None:
        cfg.video.resize_width = int(args.resize_width)
    if args.imgsz is not None:
        cfg.detector.imgsz = int(args.imgsz)
    if args.device is not None:
        cfg.detector.device = args.device
        cfg.infrastructure.device = args.device
    if args.no_stabilize:
        cfg.zone.stabilize = False
    if args.no_debris:
        cfg.debris.enabled = False
    if args.infra_weights:
        cfg.infrastructure.enabled = True
        cfg.infrastructure.weights = args.infra_weights
    # A benchmark must not print events, open windows or write files.
    cfg.alerts.stdout = False
    cfg.alerts.events_jsonl = ""
    cfg.alerts.webhook_url = ""
    cfg.output.display = False
    cfg.output.save_video = ""
    cfg.output.snapshot_dir = ""
    cfg.output.print_every_n_frames = 0
    cfg.video.loop = False
    return cfg


def summarise(samples: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
    """mean / median / p95 / max per stage in ms."""
    out: Dict[str, Dict[str, float]] = {}
    for stage, values in samples.items():
        if not values:
            continue
        arr = np.asarray(values, dtype=np.float64)
        out[stage] = {
            "mean_ms": float(arr.mean()),
            "median_ms": float(np.median(arr)),
            "p95_ms": float(np.percentile(arr, 95)),
            "max_ms": float(arr.max()),
            "n": int(arr.size),
        }
    return out


def format_table(stats: Dict[str, Dict[str, float]]) -> str:
    total_mean = stats.get("total", {}).get("mean_ms", 0.0)
    lines = [f"{'stage':<8} {'mean ms':>9} {'median':>9} {'p95':>9} {'max':>9} {'share':>7}"]
    for stage in STAGES:
        s = stats.get(stage)
        if s is None:
            continue
        share = (100.0 * s["mean_ms"] / total_mean) if (total_mean > 0 and stage != "total") else 100.0
        lines.append(f"{stage:<8} {s['mean_ms']:>9.2f} {s['median_ms']:>9.2f} {s['p95_ms']:>9.2f} {s['max_ms']:>9.2f} {share:>6.1f}%")
    return "\n".join(lines)


def run_benchmark(cfg: MonitorConfig, frames: int, warmup: int, no_detector: bool) -> Dict[str, Any]:
    """Time ``frames`` frames of ``cfg.video.source`` through Monitor.process_frame."""
    detector = NullDetector(cfg) if no_detector else None
    mon = Monitor(cfg, detector=detector)
    cap, fps, is_live = open_source(cfg.video)
    samples: Dict[str, List[float]] = {s: [] for s in STAGES}
    statuses: Dict[str, int] = {}
    processed = 0
    frame_shape = None
    t_wall = None
    t0 = time.monotonic()
    try:
        while frames <= 0 or processed < frames:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            timestamp = (time.monotonic() - t0) if is_live else processed / fps
            result = mon.process_frame(frame, timestamp, processed)
            if processed == warmup:
                t_wall = time.perf_counter()  # wall clock starts after warm-up (model load, first-frame costs)
            if processed >= warmup:
                for stage, ms in result.timings_ms.items():
                    samples.setdefault(stage, []).append(float(ms))
                statuses[result.assessment.status.value] = statuses.get(result.assessment.status.value, 0) + 1
            frame_shape = mon.last_frame.shape
            processed += 1
    finally:
        cap.release()
        mon.close()
    timed = max(0, processed - warmup)
    elapsed = (time.perf_counter() - t_wall) if (t_wall is not None and timed > 0) else 0.0
    stats = summarise(samples)
    return {
        "source": str(cfg.video.source),
        "frames_processed": processed,
        "frames_timed": timed,
        "warmup_frames": min(warmup, processed),
        "source_fps": fps,
        "processing_shape": list(frame_shape) if frame_shape is not None else None,
        "detect_every_n_frames": cfg.detector.detect_every_n_frames,
        "detector": "none" if no_detector else cfg.detector.weights,
        "elapsed_s": elapsed,
        "fps": (timed / elapsed) if elapsed > 0 else 0.0,
        "stages": stats,
        "status_counts": statuses,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cfg = configure(args)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        print(f"error: cannot load config: {exc}", file=sys.stderr)
        return 2
    try:
        report = run_benchmark(cfg, args.frames, max(0, args.warmup), args.no_detector)
    except (RuntimeError, FileNotFoundError, ImportError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if report["frames_timed"] == 0:
        print(f"error: only {report['frames_processed']} frame(s) read; nothing to time (need more than --warmup)",
              file=sys.stderr)
        return 1
    shape = report["processing_shape"]
    print(f"source: {report['source']} ({report['source_fps']:.1f} fps), processing at "
          f"{shape[1]}x{shape[0]}, detector={report['detector']} every {report['detect_every_n_frames']} frame(s)")
    print(f"timed {report['frames_timed']} frames after {report['warmup_frames']} warm-up frames")
    print(format_table(report["stages"]))
    print(f"end-to-end: {report['frames_timed']} frames in {report['elapsed_s']:.2f} s = {report['fps']:.1f} fps "
          f"(real time needs {report['source_fps']:.1f} fps); statuses {report['status_counts']}")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
