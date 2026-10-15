#!/usr/bin/env python
"""Print the metadata stored in an ultralytics ``.pt`` checkpoint WITHOUT importing torch.

Why this exists
---------------
A YOLO checkpoint is a zip archive (``best/data.pkl`` + one file per tensor storage). Loading it
with ``torch.load`` needs torch, ultralytics and often the *same* versions that wrote the file,
and it takes seconds. All the interesting bookkeeping (epoch, training arguments, validation
metrics, class names) lives in the small ``data.pkl`` pickle, so we unpickle only that file with
a custom :class:`pickle.Unpickler`:

* ``find_class`` hands out real objects only from a fixed allowlist of plain containers and value
  types (``set``, ``OrderedDict``, ``datetime``, pure paths, numpy arrays/dtypes/scalars, the
  ``_codecs.encode`` that pickles bytes) - none of them can run code when the pickle calls them;
* every ``torch.*`` / ``ultralytics.*`` (and other ``numpy``) symbol becomes a named stub class,
  so ``nn.Module`` objects become plain attribute bags and tensors become stubs;
* anything else is refused with an error naming the global: a checkpoint that asks for
  ``builtins.eval``, ``getattr``, ``os.system``... is not a plain ultralytics checkpoint, and
  loading it with ``YOLO()`` / ``torch.load(weights_only=False)`` would run that code;
* ``persistent_load`` returns a placeholder for every tensor storage, so no tensor bytes are read.

Typical questions this answers for students: "did this run finish or stop early?", "which
dataset / image size / hyperparameters produced it?", "what were the final mAP numbers?",
"can I resume training from it?" (only if the optimizer is still present - a stripped ``best.pt``
has ``epoch=-1`` and no optimizer).

Usage::

    python tools/inspect_checkpoint.py 50_epochs_best.pt
    python tools/inspect_checkpoint.py path/to/best.pt --json
    python tools/inspect_checkpoint.py a.pt b.pt --all-train-args

Runs on Python 3.8+ with only the standard library (numpy optional: numeric scalars stored as
numpy types are decoded when numpy is importable, otherwise shown as stubs).
"""
from __future__ import annotations

import _codecs
import argparse
import collections
import datetime
import importlib
import io
import json
import pathlib
import pickle
import sys
import zipfile
from pathlib import Path, PurePath
from typing import Any, Dict, List, Optional, Tuple

# The only real objects the unpickler hands out: containers and value types whose constructors
# cannot run code, plus ``_codecs.encode`` (how protocol-2 pickles store bytes). Concrete
# PosixPath/WindowsPath become the matching pure path, which can be built on any platform.
_SAFE_GLOBALS: Dict[Tuple[str, str], Any] = {("_codecs", "encode"): _codecs.encode}
for _module in ("builtins", "__builtin__"):  # __builtin__: protocol-2 pickles written by Python 3
    for _obj in (set, frozenset, slice, complex, bytearray, bytes, range, int, float, str, bool, list,
                 tuple, dict):
        _SAFE_GLOBALS[(_module, _obj.__name__)] = _obj
for _name in ("OrderedDict", "defaultdict", "Counter", "deque"):
    _SAFE_GLOBALS[("collections", _name)] = getattr(collections, _name)
for _name in ("datetime", "date", "time", "timedelta", "timezone"):
    _SAFE_GLOBALS[("datetime", _name)] = getattr(datetime, _name)
for _module in ("pathlib", "pathlib._local"):  # Python 3.13 pickles paths from pathlib._local
    for _name, _obj in (("PurePath", pathlib.PurePath), ("Path", pathlib.PurePath),
                        ("PurePosixPath", pathlib.PurePosixPath), ("PosixPath", pathlib.PurePosixPath),
                        ("PureWindowsPath", pathlib.PureWindowsPath), ("WindowsPath", pathlib.PureWindowsPath)):
        _SAFE_GLOBALS[(_module, _name)] = _obj

# numpy objects reconstructed for real (imported lazily: numpy is optional). numpy 2 pickles
# them from numpy._core, numpy 1 from numpy.core; either spelling is tried on either version.
_SAFE_NUMPY = {
    ("numpy", "ndarray"), ("numpy", "dtype"),
    ("numpy.core.multiarray", "_reconstruct"), ("numpy.core.multiarray", "scalar"),
    ("numpy._core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "scalar"),
}

# Module roots whose other symbols become named stubs (never imported, never called for real).
_STUB_ROOTS = ("torch", "ultralytics", "numpy")

# train_args keys most people want to see; --all-train-args prints everything.
TRAIN_ARGS_SUBSET = (
    "model", "data", "epochs", "patience", "batch", "imgsz", "device", "optimizer", "lr0", "lrf",
    "momentum", "weight_decay", "warmup_epochs", "cos_lr", "close_mosaic", "seed", "deterministic",
    "pretrained", "freeze", "workers", "amp", "cfg", "project", "name", "mosaic", "mixup", "fliplr",
    "flipud", "degrees", "scale", "translate", "hsv_h", "hsv_s", "hsv_v", "box", "cls", "dfl",
)


class _Stub:
    """Stand-in for any class we refuse to import. Behaves like an attribute bag.

    ``pickle`` builds objects either via ``cls(*args)`` (REDUCE) or ``cls.__new__(cls)`` + BUILD
    (NEWOBJ), and then hands the state to ``__setstate__``. ``nn.Module`` state is its ``__dict__``,
    so after unpickling a ``DetectionModel`` stub simply has ``.names`` and ``.yaml`` attributes.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._stub_args = args
        self._stub_kwargs = kwargs

    def __setstate__(self, state: Any) -> None:
        if isinstance(state, dict):
            self.__dict__.update(state)
        elif isinstance(state, tuple) and len(state) == 2 and isinstance(state[0], (dict, type(None))):
            # (dict_state, slot_state) form used by objects with __slots__
            if state[0]:
                self.__dict__.update(state[0])
            if isinstance(state[1], dict):
                self.__dict__.update(state[1])
        else:
            self._stub_state = state

    def __repr__(self) -> str:
        return "<stub {}.{}>".format(type(self).__module__, type(self).__name__)


class _StoragePlaceholder:
    """Placeholder returned for every persistent id (tensor storage); the bytes are never read."""

    def __init__(self, pid: Any) -> None:
        self.pid = pid

    def __repr__(self) -> str:
        return "<storage {}>".format(self.pid[1].__name__ if isinstance(self.pid, tuple) and len(self.pid) > 1 and hasattr(self.pid[1], "__name__") else "?")


class UnsafeCheckpointError(pickle.UnpicklingError):
    """The pickle asks for a global outside the allowlist (one that could run code)."""


def _numpy_global(module: str, name: str) -> Any:
    """The real numpy object for an allowlisted (module, name), or None (numpy missing/renamed)."""
    spellings = [module]
    if module.startswith("numpy._core."):
        spellings.append("numpy.core." + module[len("numpy._core."):])
    elif module.startswith("numpy.core."):
        spellings.append("numpy._core." + module[len("numpy.core."):])
    for spelling in spellings:
        try:
            return getattr(importlib.import_module(spelling), name)
        except Exception:  # numpy absent, or this spelling does not exist in the installed numpy
            continue
    return None


class NoTorchUnpickler(pickle.Unpickler):
    """Unpickler that never imports torch/ultralytics and never hands out a callable that can run code.

    Allowlisted globals are returned for real, other torch/ultralytics/numpy symbols become
    named :class:`_Stub` subclasses, and anything else raises :class:`UnsafeCheckpointError`.
    """

    def __init__(self, file: Any) -> None:
        super().__init__(file)
        self._stub_cache: Dict[Tuple[str, str], type] = {}

    def find_class(self, module: str, name: str) -> Any:
        key = (module, name)
        if key in _SAFE_GLOBALS:
            return _SAFE_GLOBALS[key]
        if key in _SAFE_NUMPY:
            obj = _numpy_global(module, name)
            if obj is not None:
                return obj
        if module.split(".")[0] not in _STUB_ROOTS:
            raise UnsafeCheckpointError(
                "refusing to load global '{}.{}': not a plain container/value type and not a "
                "torch/ultralytics class, so this is not a plain ultralytics checkpoint (loading it with "
                "YOLO() or torch.load could run code); nothing was executed".format(module, name))
        if key not in self._stub_cache:
            self._stub_cache[key] = type(name, (_Stub,), {"__module__": module, "__qualname__": name})
        return self._stub_cache[key]

    def persistent_load(self, pid: Any) -> Any:
        return _StoragePlaceholder(pid)


def load_checkpoint_dict(path: str) -> Dict[str, Any]:
    """Return the top-level checkpoint dict of an ultralytics/torch zip checkpoint (tensors stubbed)."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError("checkpoint not found: {}".format(p))
    if not zipfile.is_zipfile(str(p)):
        raise ValueError(
            "{} is not a zip-based torch checkpoint (legacy tar/pickle format is not supported)".format(p)
        )
    with zipfile.ZipFile(str(p)) as zf:
        pkl_names = [n for n in zf.namelist() if n.endswith("data.pkl")]
        if not pkl_names:
            raise ValueError("{} contains no data.pkl entry".format(p))
        payload = zf.read(pkl_names[0])
    obj = NoTorchUnpickler(io.BytesIO(payload)).load()
    if not isinstance(obj, dict):
        raise ValueError("unexpected checkpoint root object: {}".format(type(obj).__name__))
    return obj


def _plain(value: Any, depth: int = 0) -> Any:
    """Recursively convert unpickled values to JSON-friendly Python (stubs -> dict or repr)."""
    if depth > 12:
        return repr(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if hasattr(value, "item") and not isinstance(value, _Stub):  # numpy scalar
        try:
            return value.item()
        except Exception:
            return repr(value)
    if isinstance(value, PurePath):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _plain(v, depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_plain(v, depth + 1) for v in value]
    if isinstance(value, _Stub):
        fields = {k: v for k, v in vars(value).items() if not k.startswith("_stub_")}
        if fields:
            return {str(k): _plain(v, depth + 1) for k, v in fields.items()}
        return repr(value)
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            return repr(value)
    return repr(value)


def _as_mapping(value: Any) -> Dict[str, Any]:
    """train_args is a dict in recent checkpoints and an IterableSimpleNamespace stub in older ones."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, _Stub):
        return {k: v for k, v in vars(value).items() if not k.startswith("_stub_")}
    return {}


def inspect_checkpoint(path: str, all_train_args: bool = False) -> Dict[str, Any]:
    """Collect the human-relevant metadata of ``path`` into a JSON-serialisable dict."""
    ck = load_checkpoint_dict(path)
    model_obj = ck.get("ema") if ck.get("ema") is not None else ck.get("model")
    names_raw = getattr(model_obj, "names", None)
    names: Dict[int, str] = {}
    if isinstance(names_raw, dict):
        names = {int(k): str(v) for k, v in names_raw.items()}
    elif isinstance(names_raw, (list, tuple)):
        names = {i: str(v) for i, v in enumerate(names_raw)}
    model_yaml = getattr(model_obj, "yaml", None)
    model_yaml = model_yaml if isinstance(model_yaml, dict) else {}
    nc = model_yaml.get("nc")
    if nc is None and names:
        nc = len(names)

    train_args = _as_mapping(ck.get("train_args"))
    if not all_train_args:
        train_args = {k: train_args[k] for k in TRAIN_ARGS_SUBSET if k in train_args}

    train_results = ck.get("train_results")
    results_last: Dict[str, Any] = {}
    results_rows = 0
    if isinstance(train_results, dict):
        for key, col in train_results.items():
            if isinstance(col, (list, tuple)) and len(col) > 0:
                results_rows = max(results_rows, len(col))
                results_last[str(key)] = _plain(col[-1])
            else:
                results_last[str(key)] = _plain(col)

    epoch = _plain(ck.get("epoch"))
    optimizer_present = ck.get("optimizer") is not None
    ema_present = ck.get("ema") is not None
    p = Path(path)
    return {
        "file": str(p),
        "size_mb": round(p.stat().st_size / (1024 * 1024), 2),
        "keys": [str(k) for k in ck.keys()],
        "date": _plain(ck.get("date")),
        "version": _plain(ck.get("version")),
        "license": _plain(ck.get("license")),
        "epoch": epoch,
        "best_fitness": _plain(ck.get("best_fitness")),
        "updates": _plain(ck.get("updates")),
        "optimizer_present": optimizer_present,
        "ema_present": ema_present,
        "model_present": ck.get("model") is not None,
        # ultralytics strip_optimizer() sets epoch=-1 and drops optimizer/ema when a run finishes:
        # such a file holds the final (EMA) weights and cannot be used to resume training.
        "stripped": (epoch == -1) and not optimizer_present,
        "git": _plain(ck.get("git")),
        "model_yaml_file": _plain(model_yaml.get("yaml_file")),
        "train_args": _plain(train_args),
        "train_metrics": _plain(ck.get("train_metrics")),
        "train_results_rows": results_rows,
        "train_results_last": results_last,
        "names": {str(k): v for k, v in sorted(names.items())},
        "nc": nc,
    }


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return "{:.5g}".format(value)
    return str(value)


def format_report(info: Dict[str, Any]) -> str:
    """Render :func:`inspect_checkpoint` output as an ASCII text report."""
    lines: List[str] = []
    lines.append("checkpoint : {} ({} MB)".format(info["file"], info["size_mb"]))
    lines.append("saved by   : ultralytics {} on {}".format(info.get("version"), info.get("date")))
    lines.append("epoch      : {}{}".format(
        info.get("epoch"),
        "  (-1 = stripped final checkpoint: EMA weights, no optimizer, not resumable)" if info.get("stripped") else "",
    ))
    lines.append("best_fitness: {}".format(_fmt(info.get("best_fitness"))))
    lines.append("optimizer  : {}   ema: {}   model: {}   updates: {}".format(
        "present" if info["optimizer_present"] else "absent",
        "present" if info["ema_present"] else "absent",
        "present" if info["model_present"] else "absent",
        info.get("updates"),
    ))
    if info.get("git"):
        lines.append("git        : {}".format(info["git"]))
    lines.append("model yaml : {}   nc: {}".format(info.get("model_yaml_file"), info.get("nc")))
    lines.append("names      :")
    for k, v in info["names"].items():
        lines.append("    {:>3}: {}".format(k, v))
    lines.append("train_args :")
    for k, v in info["train_args"].items():
        lines.append("    {:<16} {}".format(k, _fmt(v)))
    lines.append("train_metrics (final validation of the epoch that produced these weights):")
    tm = info.get("train_metrics") or {}
    if isinstance(tm, dict) and tm:
        for k, v in tm.items():
            lines.append("    {:<28} {}".format(k, _fmt(v)))
    else:
        lines.append("    (none stored)")
    lines.append("train_results: {} epoch rows; last row:".format(info.get("train_results_rows", 0)))
    tr = info.get("train_results_last") or {}
    if tr:
        for k, v in tr.items():
            lines.append("    {:<28} {}".format(k, _fmt(v)))
    else:
        lines.append("    (none stored)")
    lines.append("top-level keys: {}".format(", ".join(info["keys"])))
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoints", nargs="+", help="one or more ultralytics .pt files")
    parser.add_argument("--json", action="store_true", help="print a JSON object (or list) instead of text")
    parser.add_argument("--all-train-args", action="store_true", help="print every train_args key, not just the common subset")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    reports: List[Dict[str, Any]] = []
    failures = 0
    for ck in args.checkpoints:
        try:
            info = inspect_checkpoint(ck, all_train_args=args.all_train_args)
        except Exception as exc:  # keep going for the remaining files
            failures += 1
            print("ERROR {}: {}: {}".format(ck, type(exc).__name__, exc), file=sys.stderr)
            continue
        reports.append(info)
        if not args.json:
            print(format_report(info))
            print("")
    if args.json:
        print(json.dumps(reports[0] if len(reports) == 1 else reports, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
