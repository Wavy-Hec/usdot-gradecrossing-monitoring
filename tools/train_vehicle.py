#!/usr/bin/env python
"""Reproducible fine-tuning of the vehicle (or infrastructure) YOLO detector.

Why a script instead of a notebook cell
---------------------------------------
The earlier Colab runs left no record of *which* data YAML, hyperparameter file and arguments
produced a given ``best.pt`` (see ``tools/inspect_checkpoint.py`` for what the checkpoint itself
remembers: Google-Drive paths that no longer resolve). This tool:

1. validates the dataset YAML before spending GPU time (train/val must resolve; a ``test`` split
   is strongly recommended, see below),
2. runs ``model.train`` with every argument on the command line (seeded, deterministic),
3. evaluates the resulting ``best.pt`` on the **test** split, which the training loop never saw
   (the ``val`` split drives early stopping and checkpoint selection, so val mAP is optimistic),
4. writes ``metrics.json`` next to ``best.pt`` and copies the data YAML (and the hyperparameter
   file when ``--cfg`` is used) into the same ``weights/`` directory, so the folder is
   self-describing when it is copied to the Jetson.

Usage::

    python tools/train_vehicle.py --data datasets/vehicles/data.yaml --model yolo11n.pt \
        --epochs 100 --imgsz 640 --batch 16 --patience 30 --project runs/detect --name vehicle
    python tools/train_vehicle.py --data ... --cfg configs/vehicle_hyp.yaml     # from tune_hyperparameters.py
    python tools/train_vehicle.py --data ... --dry-run                          # validate only

Needs a CUDA GPU for any realistic dataset; ``--device cpu`` works for smoke tests only.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class DatasetError(RuntimeError):
    """Raised when the dataset YAML is missing or its split entries do not resolve."""


def _load_yaml(path: Path) -> Dict[str, Any]:
    import yaml

    with open(str(path), "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise DatasetError("{} does not contain a mapping".format(path))
    return data


def _dataset_root(yaml_path: Path, data: Dict[str, Any]) -> Path:
    """Directory relative split entries resolve against (``path`` key, else the YAML's folder)."""
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
        from ultralytics.utils import DATASETS_DIR

        alt = (Path(DATASETS_DIR) / root).resolve()
        if alt.exists():
            return alt
    except Exception:
        pass
    return candidate


def resolve_splits(yaml_path: str) -> Dict[str, List[Path]]:
    """Return {split: [resolved paths]}; train and val are required, test optional but checked."""
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
            if split in ("train", "val"):
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a YOLO detector reproducibly and evaluate it on the test split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", required=True, help="dataset YAML (train/val[/test]/names)")
    parser.add_argument("--model", default="yolo11n.pt", help="starting weights (COCO-pretrained or a previous best.pt)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--patience", type=int, default=30, help="early-stop after this many epochs without val improvement")
    parser.add_argument("--cfg", default="", help="optional hyperparameter YAML (e.g. from tune_hyperparameters.py)")
    parser.add_argument("--project", default="runs/detect")
    parser.add_argument("--name", default="vehicle")
    parser.add_argument("--device", default="", help="'' = auto, '0' = first GPU, 'cpu'")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--exist-ok", action="store_true", help="overwrite --project/--name instead of incrementing")
    parser.add_argument("--dry-run", action="store_true", help="validate inputs and print the planned call; do not train")
    return parser


def planned_call(args: argparse.Namespace) -> Dict[str, Any]:
    """Keyword arguments for ``model.train`` (explicit values override anything in --cfg)."""
    call: Dict[str, Any] = {
        "data": args.data,
        "epochs": int(args.epochs),
        "imgsz": int(args.imgsz),
        "batch": int(args.batch),
        "patience": int(args.patience),
        "project": args.project,
        "name": args.name,
        "device": args.device or None,
        "workers": int(args.workers),
        "seed": int(args.seed),
        "deterministic": True,
        "exist_ok": bool(args.exist_ok),
        "val": True,
        "plots": True,
    }
    if args.cfg:
        call["cfg"] = args.cfg
    return call


def _float_list(values: Any) -> List[float]:
    try:
        return [float(v) for v in values]
    except Exception:
        return []


def collect_metrics(metrics: Any, names: Dict[int, str]) -> Dict[str, Any]:
    """Turn an ultralytics ``DetMetrics`` object into plain JSON (defensive: never raise)."""
    out: Dict[str, Any] = {}
    try:
        out["summary"] = {k: float(v) for k, v in dict(metrics.results_dict).items()}
    except Exception:
        out["summary"] = {}
    try:
        per_class: Dict[str, Dict[str, float]] = {}
        class_index = list(getattr(metrics, "ap_class_index", []))
        for i, cls_id in enumerate(class_index):
            p, r, map50, map5095 = [float(v) for v in metrics.class_result(i)]
            per_class[names.get(int(cls_id), str(cls_id))] = {
                "precision": p, "recall": r, "mAP50": map50, "mAP50-95": map5095,
            }
        out["per_class"] = per_class
        out["maps_per_class"] = _float_list(getattr(metrics, "maps", []))
    except Exception:
        out["per_class"] = {}
    try:
        out["speed_ms"] = {k: float(v) for k, v in dict(metrics.speed).items()}
    except Exception:
        out["speed_ms"] = {}
    return out


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        splits = resolve_splits(args.data)
    except DatasetError as exc:
        print("ERROR: {}".format(exc), file=sys.stderr)
        return 2
    if args.cfg and not Path(args.cfg).is_file():
        print("ERROR: --cfg {} not found".format(args.cfg), file=sys.stderr)
        return 2
    for split, paths in splits.items():
        print("{:<5} -> {}".format(split, ", ".join(str(p) for p in paths)))
    eval_split = "test" if "test" in splits else "val"
    if eval_split == "val":
        print(
            "WARNING: no 'test:' split in the data yaml; the final evaluation will use 'val', which "
            "also selected best.pt, so the reported numbers are optimistic."
        )
    call = planned_call(args)
    print("model.train call: " + json.dumps(call, indent=2, default=str))
    if args.dry_run:
        print("dry run: inputs OK, nothing trained (final evaluation split would be '{}')".format(eval_split))
        return 0

    try:
        import ultralytics
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics is not installed (pip install ultralytics)", file=sys.stderr)
        return 2

    model = YOLO(args.model)
    train_metrics = model.train(**call)
    trainer = getattr(model, "trainer", None)
    save_dir = Path(getattr(trainer, "save_dir", Path(args.project) / args.name))
    weights_dir = save_dir / "weights"
    best = weights_dir / "best.pt"
    if not best.is_file():
        print("ERROR: training finished but {} does not exist".format(best), file=sys.stderr)
        return 1
    print("best weights: {}".format(best))

    # Evaluate the *saved* best.pt (not the in-memory last epoch) on the held-out split.
    best_model = YOLO(str(best))
    names = {int(k): str(v) for k, v in dict(best_model.names).items()}
    test_metrics = best_model.val(
        data=args.data,
        split=eval_split,
        imgsz=int(args.imgsz),
        batch=int(args.batch),
        device=args.device or None,
        plots=False,
        project=str(save_dir),
        name="eval_{}".format(eval_split),
        exist_ok=True,
        verbose=True,
    )

    record: Dict[str, Any] = {
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "ultralytics_version": getattr(ultralytics, "__version__", None),
        "weights": str(best),
        "data_yaml": str(Path(args.data).resolve()),
        "train_call": call,
        "names": {str(k): v for k, v in names.items()},
        "epochs_completed": (int(getattr(trainer, "epoch", -1)) + 1) if trainer is not None else None,
        "best_fitness": float(trainer.best_fitness) if trainer is not None and getattr(trainer, "best_fitness", None) is not None else None,
        "val_metrics_at_best": collect_metrics(train_metrics, names) if train_metrics is not None else {},
        "final_eval_split": eval_split,
        "final_eval": collect_metrics(test_metrics, names),
    }
    metrics_path = weights_dir / "metrics.json"
    with open(str(metrics_path), "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2, default=str)
    shutil.copy2(args.data, str(weights_dir / "data.yaml"))
    if args.cfg:
        shutil.copy2(args.cfg, str(weights_dir / "hyperparameters.yaml"))
    print("wrote {} and copied data.yaml next to best.pt".format(metrics_path))
    summary = record["final_eval"].get("summary", {})
    print("{} split: mAP50={:.4f} mAP50-95={:.4f} precision={:.4f} recall={:.4f}".format(
        eval_split,
        summary.get("metrics/mAP50(B)", float("nan")),
        summary.get("metrics/mAP50-95(B)", float("nan")),
        summary.get("metrics/precision(B)", float("nan")),
        summary.get("metrics/recall(B)", float("nan")),
    ))
    print("next: python tools/export_tensorrt.py --weights {} --half   (on the Jetson)".format(best))
    return 0


if __name__ == "__main__":
    sys.exit(main())
