#!/usr/bin/env python
"""Hyperparameter tuning for the vehicle / infrastructure YOLO models, done so the result is valid.

What went wrong in the original Colab notebook (``notebooks/.../yolo11n_infrastructure_tuning.py``)
------------------------------------------------------------------------------------------------
The notebook ran ``model.tune(..., val=False)`` for 100 iterations. With validation disabled the
tuner never obtains a mAP for an iteration, so every row of ``tune_results.csv`` has
``fitness=0.0``; the "best" hyperparameters it wrote (``best_hyperparameters.yaml``, header:
"Best fitness=0.0 observed at iteration 1") are just the untouched defaults of iteration 1.
Hours of GPU time produced no information, and the follow-up training run silently used those
defaults. (The tune call also pointed at ``data.yml`` while training used ``data.yaml``; if the
tune path did not exist every iteration failed and, again, scored 0.)

What this tool does differently
-------------------------------
1. Refuses to start unless the dataset YAML exists and its ``train`` / ``val`` entries resolve
   to real files or directories (relative to the YAML's ``path`` key or its own directory).
2. Calls ``model.tune(..., val=True, plots=False, save=False)`` with argparse-controlled
   ``epochs`` (per iteration) and ``iterations``. Defaults 20/20 are a compromise: the fitness of
   an iteration is only meaningful if its short training run got past warm-up.
3. Reads the tuner's results file (``tune_results.csv`` on ultralytics 8.3, ``tune_results.ndjson``
   on 8.4) and **exits non-zero if the best fitness is 0**, renaming the useless
   ``best_hyperparameters.yaml`` to ``best_hyperparameters.INVALID.yaml`` so it cannot be picked
   up by ``train_vehicle.py --cfg`` by accident. With a valid result it optionally copies the file
   to ``--out``.

Usage::

    python tools/tune_hyperparameters.py --data datasets/vehicles/data.yaml --model yolo11n.pt \
        --epochs 20 --iterations 20 --out configs/vehicle_hyp.yaml

    python tools/tune_hyperparameters.py --data ... --dry-run   # validate the dataset only

Cannot run on a CPU-only laptop in reasonable time; use a CUDA GPU (Colab is fine).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class DatasetError(RuntimeError):
    """Raised when the dataset YAML is missing or its split entries do not resolve."""


def _load_yaml(path: Path) -> Dict[str, Any]:
    import yaml  # PyYAML is a runtime requirement of the package; imported here for --help speed

    with open(str(path), "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise DatasetError("{} does not contain a mapping".format(path))
    return data


def _dataset_root(yaml_path: Path, data: Dict[str, Any]) -> Path:
    """Directory that relative split entries are resolved against (ultralytics semantics).

    ultralytics uses the ``path`` key when present, else the YAML's own directory. A relative
    ``path`` is tried against the YAML directory first and then the ultralytics DATASETS_DIR
    setting (what ultralytics itself does) when that is available.
    """
    raw = data.get("path")
    if not raw:
        return yaml_path.parent
    root = Path(str(raw))
    if root.is_absolute():
        return root
    candidate = (yaml_path.parent / root).resolve()
    if candidate.exists():
        return candidate
    try:
        from ultralytics.utils import DATASETS_DIR  # optional; only for parity with ultralytics

        alt = (Path(DATASETS_DIR) / root).resolve()
        if alt.exists():
            return alt
    except Exception:
        pass
    return candidate


def resolve_splits(yaml_path: str, required: Tuple[str, ...] = ("train", "val")) -> Dict[str, List[Path]]:
    """Return {split: [resolved paths]} for train/val/test; raise DatasetError on any problem."""
    p = Path(yaml_path)
    if not p.is_file():
        raise DatasetError("dataset yaml not found: {} (cwd={})".format(p, Path.cwd()))
    data = _load_yaml(p)
    root = _dataset_root(p, data)
    problems: List[str] = []
    resolved: Dict[str, List[Path]] = {}
    for split in ("train", "val", "test"):
        entry = data.get(split)
        if entry is None:
            if split in required:
                problems.append("'{}:' key missing".format(split))
            continue
        entries = entry if isinstance(entry, (list, tuple)) else [entry]
        paths: List[Path] = []
        for e in entries:
            ep = Path(str(e))
            if not ep.is_absolute():
                ep = root / ep
            if not ep.exists():
                problems.append("'{}' entry {} does not exist (resolved against {})".format(split, ep, root))
            paths.append(ep)
        resolved[split] = paths
    if "names" not in data and "nc" not in data:
        problems.append("neither 'names' nor 'nc' present")
    if problems:
        raise DatasetError("dataset yaml {} is not usable:\n  - {}".format(p, "\n  - ".join(problems)))
    return resolved


def find_tune_dir(project: str, name: str) -> Optional[Path]:
    """Locate the directory the tuner wrote (ultralytics may append 2, 3... to ``name``)."""
    base = Path(project)
    if not base.is_dir():
        return None
    candidates = [d for d in base.iterdir() if d.is_dir() and d.name.startswith(name)]
    if not candidates:
        return None
    # newest first; the run that just finished is the most recently modified one
    candidates.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    return candidates[0]


def read_fitness(tune_dir: Path) -> List[float]:
    """Read the fitness column from tune_results.csv (8.3) or tune_results.ndjson (8.4)."""
    csv_path = tune_dir / "tune_results.csv"
    ndjson_path = tune_dir / "tune_results.ndjson"
    values: List[float] = []
    if csv_path.is_file():
        with open(str(csv_path), "r", encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                values.append(_to_float(row.get("fitness")))
    elif ndjson_path.is_file():
        with open(str(ndjson_path), "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    values.append(_to_float(json.loads(line).get("fitness")))
                except ValueError:
                    values.append(0.0)
    else:
        raise FileNotFoundError("no tune_results.csv / tune_results.ndjson in {}".format(tune_dir))
    return values


def _to_float(value: Any) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if math.isnan(f) else f


def max_fitness(values: List[float]) -> float:
    return max(values) if values else 0.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Tune YOLO hyperparameters with validation enabled and verify the result is meaningful.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", required=True, help="dataset YAML (train/val/names)")
    parser.add_argument("--model", default="yolo11n.pt", help="starting weights")
    parser.add_argument("--epochs", type=int, default=20, help="training epochs per tuning iteration")
    parser.add_argument("--iterations", type=int, default=20, help="number of hyperparameter sets to try")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--optimizer", default="AdamW", help="optimizer used for every iteration")
    parser.add_argument("--device", default="", help="'' = auto, '0' = first GPU, 'cpu'")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--project", default="runs/tune", help="parent directory for tuning runs")
    parser.add_argument("--name", default="tune", help="run name (ultralytics appends a number if it exists)")
    parser.add_argument("--resume", action="store_true", help="continue the run in --project/--name")
    parser.add_argument("--out", default="", help="copy the validated best_hyperparameters.yaml here")
    parser.add_argument("--dry-run", action="store_true", help="validate the dataset yaml and print the planned call; do not train")
    return parser


def planned_call(args: argparse.Namespace) -> Dict[str, Any]:
    """The exact keyword arguments handed to ``model.tune`` (also printed by --dry-run)."""
    call: Dict[str, Any] = {
        "data": args.data,
        "epochs": int(args.epochs),
        "iterations": int(args.iterations),
        "imgsz": int(args.imgsz),
        "batch": int(args.batch),
        "optimizer": args.optimizer,
        "device": args.device or None,
        "workers": int(args.workers),
        "seed": int(args.seed),
        "project": args.project,
        "name": args.name,
        "val": True,  # the fix: without validation every iteration scores fitness 0
        "plots": False,
        "save": False,
    }
    if args.resume:
        call["resume"] = True
    return call


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.epochs < 1 or args.iterations < 1:
        print("ERROR: --epochs and --iterations must be >= 1", file=sys.stderr)
        return 2
    try:
        splits = resolve_splits(args.data)
    except DatasetError as exc:
        print("ERROR: {}".format(exc), file=sys.stderr)
        return 2
    for split, paths in splits.items():
        print("{:<5} -> {}".format(split, ", ".join(str(p) for p in paths)))
    call = planned_call(args)
    print("model.tune call: " + json.dumps(call, indent=2, default=str))
    if args.dry_run:
        print("dry run: dataset OK, nothing trained")
        return 0

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics is not installed (pip install ultralytics)", file=sys.stderr)
        return 2
    model = YOLO(args.model)
    model.tune(**call)

    tune_dir = find_tune_dir(args.project, args.name)
    if tune_dir is None:
        print("ERROR: tuning finished but no run directory found under {}".format(args.project), file=sys.stderr)
        return 1
    try:
        fitness = read_fitness(tune_dir)
    except FileNotFoundError as exc:
        print("ERROR: {}".format(exc), file=sys.stderr)
        return 1
    best = max_fitness(fitness)
    best_yaml = tune_dir / "best_hyperparameters.yaml"
    print("tune dir: {}   iterations recorded: {}   best fitness: {:.5f}".format(tune_dir, len(fitness), best))
    if best <= 0.0:
        if best_yaml.is_file():
            invalid = tune_dir / "best_hyperparameters.INVALID.yaml"
            best_yaml.replace(invalid)
            print("renamed {} -> {}".format(best_yaml.name, invalid.name))
        print(
            "ERROR: every tuning iteration scored fitness 0. The search learned nothing, so no\n"
            "best_hyperparameters.yaml is provided. Usual causes: the dataset yaml/split paths were\n"
            "wrong inside the training subprocess, labels are empty, epochs too small for any mAP,\n"
            "or a GPU/out-of-memory failure in every iteration. Inspect the per-iteration logs in\n"
            "{} before re-running.".format(tune_dir),
            file=sys.stderr,
        )
        return 1
    if not best_yaml.is_file():
        print("ERROR: fitness > 0 but {} was not written by ultralytics".format(best_yaml), file=sys.stderr)
        return 1
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(best_yaml), str(out))
        print("copied validated hyperparameters to {}".format(out))
    print("OK: use --cfg {} with tools/train_vehicle.py".format(args.out or best_yaml))
    return 0


if __name__ == "__main__":
    sys.exit(main())
