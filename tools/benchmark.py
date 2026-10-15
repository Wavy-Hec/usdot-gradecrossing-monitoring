#!/usr/bin/env python
"""Measure per-stage milliseconds and end-to-end FPS of the Monitor on a source.

Why: the Jetson budget is tight and "it feels slow" is not a measurement. This runs the
real ``Monitor.process_frame`` loop (same code path as ``python -m gradecrossing.monitor``)
and collects the ``timings_ms`` every FrameResult carries, and prints mean / median / p95 /
max per stage plus the share of the total, so the bottleneck is obvious before anyone tunes
``detect_every_n_frames``, ``resize_width`` or ``imgsz``.

What is timed: ``--frames`` steady-state frames, i.e. after the first ``--warmup`` frames
(model load, first-frame costs) AND after the debris detector's warm-up, during which it
only learns the scene and costs about half as much as when it monitors (on clip01 7 ms
against 16 ms per frame): timing the warm-up would under-report the debris stage and
over-report FPS (ours-correctness M4). The warm-up frames are reported separately, with
the frame at which monitoring starts; ``--include-debris-warmup`` times them as well.

Examples::

    python tools/benchmark.py --config configs/mysite.yaml --frames 600
    python tools/benchmark.py --source videos/clip.mov --weights yolo11n.pt --resize-width 960 --detect-every 2
    python tools/benchmark.py --source videos/clip.mov --no-detector      # zone + debris cost only, no torch

``--no-detector`` swaps in a detector that returns nothing, which is how you measure the
non-neural stages on a machine without ultralytics. The flags shared with the monitor CLI
come from ``gradecrossing.monitor.CLI_OVERRIDES``. Output is ASCII; ``--json`` writes the
numbers for a report. Exit status: 0 measured, 1 the run failed (source, model, nothing
to time), 2 configuration or usage error (as for the monitor).
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
from gradecrossing.monitor import (  # noqa: E402
    CLI_OVERRIDES,
    CliOverride,
    ConfigClassifier,
    Monitor,
    add_cli_arguments,
    apply_cli_overrides,
    one_line,
    open_source,
)
from gradecrossing.types import Detection  # noqa: E402

STAGES = ("resize", "zone", "detect", "track", "infra", "debris", "alert", "total")

# The monitor flags that make sense for a benchmark (same rows: same help, same effect),
# plus two of its own in the same format.
SHARED_FLAGS = ("--source", "--weights", "--detect-every", "--resize-width", "--device", "--no-stabilize",
                "--infra-weights")
BENCHMARK_OVERRIDES = tuple(row for row in CLI_OVERRIDES if row.flag in SHARED_FLAGS) + (
    CliOverride("--imgsz", ("detector.imgsz",), int, "detector inference size (overrides detector.imgsz)"),
    CliOverride("--no-debris", ("debris.enabled",), None, "disable the debris detector (debris.enabled=false)",
                const=False),
)


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
    add_cli_arguments(p, BENCHMARK_OVERRIDES)
    p.add_argument("--frames", type=int, default=300,
                   help="steady-state frames to time (default 300; 0 = to the end of the file)")
    p.add_argument("--warmup", type=int, default=10,
                   help="leading frames never timed: model load, first-frame costs (default 10)")
    p.add_argument("--include-debris-warmup", action="store_true",
                   help="also time the frames of the debris warm-up (reported separately by default)")
    p.add_argument("--no-detector", action="store_true", help="use a detector that returns nothing (no ultralytics)")
    p.add_argument("--json", metavar="PATH", help="write the statistics as JSON")
    return p


def configure(args: argparse.Namespace) -> MonitorConfig:
    """Load the config, apply the benchmark flags, and silence every output side effect."""
    cfg = load_config(args.config)
    apply_cli_overrides(cfg, args, BENCHMARK_OVERRIDES)
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


def run_benchmark(cfg: MonitorConfig, frames: int, warmup: int, no_detector: bool,
                  include_debris_warmup: bool = False, monitor: Optional[Monitor] = None) -> Dict[str, Any]:
    """Time ``frames`` steady-state frames of ``cfg.video.source`` through Monitor.process_frame.

    The first ``warmup`` frames are never timed; frames of the debris warm-up are collected
    in ``debris_warmup`` unless ``include_debris_warmup``. ``fps`` is timed frames over the
    wall-clock time of exactly those frames (read + process).
    """
    mon = monitor if monitor is not None else Monitor(cfg, detector=NullDetector(cfg) if no_detector else None)
    cap, fps, is_live = open_source(cfg.video)
    samples: Dict[str, List[float]] = {s: [] for s in STAGES}
    warm_samples: Dict[str, List[float]] = {s: [] for s in STAGES}
    statuses: Dict[str, int] = {}
    processed = timed = warm_frames = 0
    monitoring_from: Optional[int] = None  # first frame with the debris model warmed up (or off)
    frame_shape = None
    elapsed = 0.0
    t0 = time.monotonic()
    try:
        while frames <= 0 or timed < frames:
            t_frame = time.perf_counter()
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            timestamp = (time.monotonic() - t0) if is_live else processed / fps
            result = mon.process_frame(frame, timestamp, processed)
            dt = time.perf_counter() - t_frame
            frame_shape = mon.last_frame.shape
            in_debris_warmup = result.debris is not None and not result.debris.warmed_up
            if not in_debris_warmup and monitoring_from is None:
                monitoring_from = processed
            processed += 1
            if processed <= warmup:
                continue
            if in_debris_warmup:
                warm_frames += 1
                for stage, ms in result.timings_ms.items():
                    warm_samples.setdefault(stage, []).append(float(ms))
                if not include_debris_warmup:
                    continue
            timed += 1
            elapsed += dt
            for stage, ms in result.timings_ms.items():
                samples.setdefault(stage, []).append(float(ms))
            statuses[result.assessment.status.value] = statuses.get(result.assessment.status.value, 0) + 1
    finally:
        cap.release()
        mon.close()
    return {
        "source": str(cfg.video.source),
        "frames_processed": processed,
        "frames_timed": timed,
        "warmup_frames": min(warmup, processed),
        "monitoring_starts_at_frame": monitoring_from,
        "debris_warmup": {"frames": warm_frames, "timed": bool(include_debris_warmup),
                          "stages": summarise(warm_samples)},
        "source_fps": fps,
        "processing_shape": list(frame_shape) if frame_shape is not None else None,
        "detect_every_n_frames": cfg.detector.detect_every_n_frames,
        "detector": "none" if no_detector else cfg.detector.weights,
        "elapsed_s": elapsed,
        "fps": (timed / elapsed) if elapsed > 0 else 0.0,
        "stages": summarise(samples),
        "status_counts": statuses,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cfg = configure(args)
    except Exception as exc:  # noqa: BLE001 - YAML syntax, unknown key, bad override: one line, exit 2
        print(f"error: cannot load config {args.config!r}: {one_line(exc)}", file=sys.stderr)
        return 2
    try:
        mon = Monitor(cfg, detector=NullDetector(cfg) if args.no_detector else None)
    except (ValueError, TypeError, KeyError, OSError) as exc:
        print(f"error: invalid configuration: {one_line(exc)}", file=sys.stderr)
        return 2
    try:
        report = run_benchmark(cfg, args.frames, max(0, args.warmup), args.no_detector,
                               args.include_debris_warmup, monitor=mon)
    except (RuntimeError, OSError, ImportError, ValueError) as exc:
        print(f"error: {one_line(exc)}", file=sys.stderr)
        return 1
    if report["frames_timed"] == 0:
        print(f"error: only {report['frames_processed']} frame(s) read and none of them after the warm-up "
              f"(--warmup {args.warmup}, debris warm-up {report['debris_warmup']['frames']} frame(s)); "
              f"nothing to time", file=sys.stderr)
        return 1
    shape = report["processing_shape"]
    print(f"source: {report['source']} ({report['source_fps']:.1f} fps), processing at "
          f"{shape[1]}x{shape[0]}, detector={report['detector']} every {report['detect_every_n_frames']} frame(s)")
    warm = report["debris_warmup"]
    if warm["frames"]:
        warm_total = warm["stages"].get("total", {}).get("mean_ms", 0.0)
        warm_debris = warm["stages"].get("debris", {}).get("mean_ms", 0.0)
        print(f"debris warm-up: {warm['frames']} frame(s), mean total {warm_total:.2f} ms (debris {warm_debris:.2f} ms)"
              f" - {'timed below' if warm['timed'] else 'not timed'}; monitoring starts at frame "
              f"{report['monitoring_starts_at_frame']}")
    what = "frames (including the debris warm-up)" if warm["timed"] else "steady-state frames"
    print(f"timed {report['frames_timed']} {what} after {report['warmup_frames']} start-up frames")
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
