"""tools/tune_hyperparameters.py and tools/train_vehicle.py without torch or a GPU.

``model.tune`` is replaced by a fake ultralytics that writes the results tree the way each
ultralytics generation lays it out (ours-correctness F15: 8.4 nests a relative ``project``
under ``SETTINGS['runs_dir']/<task>/``, so the tool never found a finished run), and the
dataset-YAML validation both tools share lives in tools/_dataset_yaml.py (ours-cleanup OC-06).
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
import types
from pathlib import Path
from typing import List

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"


def _load_tool(name: str):
    spec = importlib.util.spec_from_file_location("tool_" + name, str(TOOLS / (name + ".py")))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _dataset(tmp_path: Path) -> Path:
    for split in ("train", "val"):
        (tmp_path / "images" / split).mkdir(parents=True)
    yaml_path = tmp_path / "data.yaml"
    yaml_path.write_text("path: {}\ntrain: images/train\nval: images/val\nnames:\n  0: car\n".format(
        tmp_path.as_posix()), encoding="utf-8")
    return yaml_path


class _FakeYOLO:
    """Stand-in for ``ultralytics.YOLO`` whose ``tune`` writes a results tree and nothing else.

    ``layout`` picks the generation: "8.4" (relative project nested under runs_dir/<task>,
    tune_results.ndjson), "8.3" (relative project against the cwd, tune_results.csv) or
    "8.2" (like 8.3 but the directory is always named ``tune``).
    """

    layout = "8.4"
    runs_dir = Path(".")
    fitness: List[float] = [0.0, 0.31]
    write = True
    calls: List[dict] = []

    def __init__(self, model: str) -> None:
        self.model = model
        self.task = "detect"

    def tune(self, **kw) -> None:
        _FakeYOLO.calls.append(kw)
        if not _FakeYOLO.write:
            return
        project = Path(kw["project"])
        if not project.is_absolute():
            project = (_FakeYOLO.runs_dir / self.task / project) if self.layout == "8.4" else (Path.cwd() / project)
        name = "tune" if self.layout == "8.2" else (kw.get("name") or "tune")
        out, n = project / name, 2
        while out.exists() and not kw.get("resume"):  # ultralytics increment_path
            out, n = project / "{}{}".format(name, n), n + 1
        out.mkdir(parents=True, exist_ok=True)
        if self.layout == "8.4":
            (out / "tune_results.ndjson").write_text(
                "".join(json.dumps({"iteration": i + 1, "fitness": f}) + "\n" for i, f in enumerate(self.fitness)),
                encoding="utf-8")
        else:
            (out / "tune_results.csv").write_text(
                "fitness,lr0\n" + "".join("{},0.01\n".format(f) for f in self.fitness), encoding="utf-8")
        if max(self.fitness) > 0:
            (out / "best_hyperparameters.yaml").write_text("lr0: 0.01\n", encoding="utf-8")


@pytest.fixture
def fake_ultralytics(monkeypatch, tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=_FakeYOLO))
    monkeypatch.setattr(_FakeYOLO, "layout", "8.4")
    monkeypatch.setattr(_FakeYOLO, "runs_dir", tmp_path / "ultralytics_settings_runs")
    monkeypatch.setattr(_FakeYOLO, "fitness", [0.0, 0.31])
    monkeypatch.setattr(_FakeYOLO, "write", True)
    monkeypatch.setattr(_FakeYOLO, "calls", [])
    return work


@pytest.mark.parametrize("layout", ["8.4", "8.3", "8.2"])
def test_a_finished_tuning_run_is_found_and_exits_0(fake_ultralytics, tmp_path, capsys, layout):
    _FakeYOLO.layout = layout
    tune = _load_tool("tune_hyperparameters")
    data = _dataset(tmp_path / "ds")
    out = tmp_path / "hyp" / "vehicle_hyp.yaml"
    rc = tune.main(["--data", str(data), "--project", "runs/tune", "--name", "vehicle", "--out", str(out)])
    captured = capsys.readouterr()
    assert rc == 0, captured.err
    # the tuner was handed an absolute project, so 8.4 did not nest it under its runs_dir
    project = Path(_FakeYOLO.calls[0]["project"])
    assert project.is_absolute() and project == (fake_ultralytics / "runs" / "tune").resolve()
    assert not _FakeYOLO.runs_dir.exists()
    assert "best fitness: 0.31000" in captured.out
    assert out.read_text(encoding="utf-8") == "lr0: 0.01\n"


def test_a_stale_run_is_never_reported_as_this_runs_result(fake_ultralytics, tmp_path, capsys):
    tune = _load_tool("tune_hyperparameters")
    data = _dataset(tmp_path / "ds")
    stale = fake_ultralytics / "runs" / "tune" / "tune"
    stale.mkdir(parents=True)
    (stale / "tune_results.ndjson").write_text('{"iteration": 1, "fitness": 0.9}\n', encoding="utf-8")
    (stale / "best_hyperparameters.yaml").write_text("lr0: 0.5\n", encoding="utf-8")
    old = time.time() - 3600.0
    os.utime(str(stale / "tune_results.ndjson"), (old, old))

    # this run writes tune2: its fitness is reported, not the stale 0.9
    assert tune.main(["--data", str(data), "--out", str(tmp_path / "hyp.yaml")]) == 0
    out = capsys.readouterr().out
    assert "tune2" in out and "best fitness: 0.31000" in out
    assert (tmp_path / "hyp.yaml").read_text(encoding="utf-8") == "lr0: 0.01\n"

    # a run that wrote nothing (crashed, or a layout nobody knows yet) is an error, not tune/ or tune2/
    for d in (stale, stale.parent / "tune2"):
        os.utime(str(d / "tune_results.ndjson"), (old, old))
    _FakeYOLO.write = False
    assert tune.main(["--data", str(data)]) == 1
    assert "written by this run" in capsys.readouterr().err


def test_a_search_that_learned_nothing_still_exits_1(fake_ultralytics, tmp_path, capsys):
    _FakeYOLO.fitness = [0.0, 0.0]
    tune = _load_tool("tune_hyperparameters")
    assert tune.main(["--data", str(_dataset(tmp_path / "ds"))]) == 1
    assert "every tuning iteration scored fitness 0" in capsys.readouterr().err


def test_find_tune_dir_matches_ultralytics_names_only(tmp_path):
    tune = _load_tool("tune_hyperparameters")
    for name in ("vehicle", "vehicle3", "tune2", "vehicle_backup", "other"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "tune_results.csv").write_text("fitness\n0.1\n", encoding="utf-8")
    t0 = time.time() - 100.0
    for k, name in enumerate(("vehicle", "vehicle3", "tune2", "vehicle_backup", "other")):
        os.utime(str(tmp_path / name / "tune_results.csv"), (t0 + k, t0 + k))
    # 'vehicle_backup' and 'other' are newer but not names ultralytics gives a tuning run
    assert tune.find_tune_dir(str(tmp_path), "vehicle") == tmp_path / "tune2"
    assert tune.find_tune_dir(str(tmp_path), "vehicle", since=time.time()) is None
    (tmp_path / "vehicle4").mkdir()  # no results file: not a finished run
    assert tune.find_tune_dir(str(tmp_path), "vehicle") == tmp_path / "tune2"


def test_dry_run_prints_the_absolute_project(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tune = _load_tool("tune_hyperparameters")
    assert tune.main(["--data", str(_dataset(tmp_path / "ds")), "--dry-run", "--project", "runs/tune"]) == 0
    call = json.loads(capsys.readouterr().out.split("model.tune call: ", 1)[1].split("\ndry run", 1)[0])
    assert Path(call["project"]) == (tmp_path / "runs" / "tune").resolve()


def test_both_training_tools_share_one_dataset_validator(tmp_path):
    tune = _load_tool("tune_hyperparameters")
    train = _load_tool("train_vehicle")
    helper = sys.modules["_dataset_yaml"]
    assert tune.resolve_splits is train.resolve_splits is helper.resolve_splits
    assert tune.DatasetError is train.DatasetError is helper.DatasetError
    good = _dataset(tmp_path / "ds")
    splits = helper.resolve_splits(str(good))
    assert sorted(splits) == ["train", "val"] and splits["val"] == [tmp_path / "ds" / "images" / "val"]
    only_train = tmp_path / "train_only.yaml"
    only_train.write_text("path: {}\ntrain: images/train\nnc: 1\n".format((tmp_path / "ds").as_posix()),
                          encoding="utf-8")
    with pytest.raises(helper.DatasetError, match="'val:' key missing"):
        helper.resolve_splits(str(only_train))
    assert list(helper.resolve_splits(str(only_train), required=("train",))) == ["train"]
    not_a_mapping = tmp_path / "list.yaml"
    not_a_mapping.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(helper.DatasetError, match="does not contain a mapping"):
        helper.resolve_splits(str(not_a_mapping))
