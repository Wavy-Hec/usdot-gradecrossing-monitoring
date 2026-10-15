"""Dataset-YAML validation shared by ``tools/train_vehicle.py`` and ``tools/tune_hyperparameters.py``.

Both tools refuse to spend GPU time on a dataset YAML whose ``train`` / ``val`` entries do
not resolve to real files or directories, the most common reason a Colab run "trained" on
nothing. Not a tool itself: each tool puts ``tools/`` on ``sys.path`` and imports
:class:`DatasetError` and :func:`resolve_splits` from here.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple


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
    """Return {split: [resolved paths]} for train/val/test; raise DatasetError on any problem.

    The ``required`` splits must be present; ``test`` is optional but, when present, checked
    like the others.
    """
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
