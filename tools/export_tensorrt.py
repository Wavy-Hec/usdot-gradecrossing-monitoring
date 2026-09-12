#!/usr/bin/env python
"""Export a trained YOLO ``.pt`` to a TensorRT ``.engine`` for the Jetson.

``gradecrossing`` loads ``.pt`` and ``.engine`` weights interchangeably (ultralytics keeps the
class names and input size in a metadata block inside the engine), so after export you only
change ``detector.weights`` in the YAML config. A TensorRT engine is roughly 2-3x faster than
PyTorch on the same Jetson GPU at FP16, which is what turns a 5-8 FPS monitor into a real-time one.

Usage::

    python tools/export_tensorrt.py --weights runs/detect/vehicle/weights/best.pt --half
    python tools/export_tensorrt.py --weights best.pt --int8 --data datasets/vehicles/data.yaml
    python tools/export_tensorrt.py --weights best.pt --dry-run       # print the plan and notes only

Read the Jetson notes printed at the end (also shown by ``--dry-run``); the most important one is
that the export must run **on the target Jetson**, because an engine only loads on the exact
TensorRT / CUDA / GPU combination that built it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

JETSON_NOTES = """
Jetson notes
------------
1. Build on the target. A .engine is bound to the TensorRT + CUDA version and the GPU
   architecture that built it (Orin != Xavier != Nano, JetPack 5 != JetPack 6). An engine exported
   on a desktop GPU will not load on the Jetson. Copy best.pt to the Jetson and run this tool there.
2. Software on the Jetson: use the torch/torchvision wheels NVIDIA publishes for your JetPack,
   then `pip install ultralytics`. TensorRT itself ships with JetPack - do NOT `pip install tensorrt`.
3. Precision: --half (FP16) is the sweet spot on Jetson tensor cores (Orin/Xavier): ~2x faster than
   FP32 with negligible accuracy change. --int8 is faster again but needs --data for calibration and
   can lose recall on small/far vehicles; SAFETY: re-validate an INT8 engine on the test split with
   tools/train_vehicle.py-style val before deploying, since a missed vehicle is a missed alarm.
4. Input size: the engine has a fixed input unless --dynamic. Use the same --imgsz as
   `detector.imgsz` in your config (default 640); a mismatch forces a rebuild or a resize per frame.
5. Batch: keep --batch 1 for a single live camera; the monitor feeds one frame at a time.
6. Memory: on 8 GB modules set --workspace 2 (GiB); the first build takes several minutes and
   prints many TensorRT log lines - that is normal. Close other GPU users during the build.
7. Clocks: `sudo nvpmodel -m 0 && sudo jetson_clocks` before benchmarking, otherwise the DVFS
   governor makes FPS numbers meaningless. tools/benchmark.py reports per-stage ms.
8. Config: set `detector.weights: path/to/best.engine`, keep `detector.tracker: bytetrack.yaml`
   (the tracker runs on the CPU and is unaffected), and `detector.half: true` for an FP16 engine.
   Class names are read from the engine metadata, so VehicleDetector.classify works unchanged.
9. The .engine file is not portable and not versioned: keep best.pt (and its metrics.json) as the
   source of truth and rebuild the engine after every JetPack upgrade.
""".strip("\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export YOLO .pt weights to a TensorRT .engine (run on the Jetson itself).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--weights", required=True, help="trained .pt checkpoint")
    parser.add_argument("--imgsz", type=int, default=640, help="engine input size; match detector.imgsz")
    parser.add_argument("--half", action="store_true", help="FP16 engine (recommended on Jetson)")
    parser.add_argument("--int8", action="store_true", help="INT8 engine; requires --data for calibration")
    parser.add_argument("--data", default="", help="dataset YAML used for INT8 calibration images")
    parser.add_argument("--device", default="0", help="CUDA device index; TensorRT cannot target 'cpu'")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--workspace", type=float, default=None, help="TensorRT workspace in GiB (e.g. 2 on 8 GB Jetsons)")
    parser.add_argument("--dynamic", action="store_true", help="dynamic input shapes (slower; normally off)")
    parser.add_argument("--dry-run", action="store_true", help="print the planned export call and the Jetson notes; do not export")
    return parser


def planned_call(args: argparse.Namespace) -> Dict[str, Any]:
    """Keyword arguments for ``model.export`` (``format='engine'`` is TensorRT in ultralytics)."""
    call: Dict[str, Any] = {
        "format": "engine",
        "imgsz": int(args.imgsz),
        "device": args.device,
        "batch": int(args.batch),
        "dynamic": bool(args.dynamic),
    }
    if args.half:
        call["half"] = True
    if args.int8:
        call["int8"] = True
        call["data"] = args.data
    if args.workspace is not None:
        call["workspace"] = float(args.workspace)
    return call


def validate(args: argparse.Namespace) -> Optional[str]:
    """Return an error message for impossible requests, else None."""
    weights = Path(args.weights)
    if not weights.is_file():
        return "weights not found: {}".format(weights)
    if weights.suffix.lower() != ".pt":
        return "expected a .pt checkpoint, got {}".format(weights.name)
    if str(args.device).strip().lower() == "cpu":
        return "TensorRT engines are built for and run on the CUDA GPU; --device cannot be 'cpu'"
    if args.int8 and not args.data:
        return "--int8 needs --data <dataset.yaml> for calibration images"
    if args.int8 and args.data and not Path(args.data).is_file():
        return "calibration dataset yaml not found: {}".format(args.data)
    if args.int8 and args.half:
        return "choose one of --half / --int8 (int8 engines already use fp16 for unquantised layers)"
    if args.imgsz % 32 != 0:
        return "--imgsz must be a multiple of 32 (got {})".format(args.imgsz)
    return None


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    error = validate(args)
    if error:
        print("ERROR: {}".format(error), file=sys.stderr)
        return 2
    call = planned_call(args)
    print("model.export call: " + json.dumps(call, indent=2))
    if args.dry_run:
        print(JETSON_NOTES)
        print("dry run: nothing exported")
        return 0
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics is not installed (pip install ultralytics)", file=sys.stderr)
        return 2
    model = YOLO(args.weights)
    try:
        out_path = model.export(**call)
    except Exception as exc:  # TensorRT missing, CUDA unavailable, out of memory...
        print("ERROR: export failed: {}: {}".format(type(exc).__name__, exc), file=sys.stderr)
        print(JETSON_NOTES)
        return 1
    print("exported: {}".format(out_path))
    print("set detector.weights: {} in your config".format(out_path))
    print(JETSON_NOTES)
    return 0


if __name__ == "__main__":
    sys.exit(main())
