"""Tests for gradecrossing/detection.py and the training/inspection tools in tools/.

Fast unit tests use a stub model (no torch). The two smoke tests load the real local weights and
are skipped when ultralytics is not importable or the files are absent.
"""
from __future__ import annotations

import importlib.util
import json
import logging
import os
import subprocess
import sys
import types
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pytest

from gradecrossing.config import DetectorConfig, InfrastructureConfig
from gradecrossing.detection import (
    CATEGORY_OTHER,
    CATEGORY_PERSON,
    CATEGORY_TRAIN,
    CATEGORY_VEHICLE,
    InfrastructureDetector,
    VehicleDetector,
    load_yolo,
    results_to_detections,
)
from gradecrossing.types import Detection, GateState

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
VEHICLE_WEIGHTS = ROOT / "50_epochs_best.pt"
INFRA_WEIGHTS = ROOT / "notebooks" / "infrastructure tuning best weights" / "best.pt"
SUBPROCESS_ENV = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONPATH=str(ROOT))


# --------------------------------------------------------------------------------------------
# Stub ultralytics objects (numpy only)
# --------------------------------------------------------------------------------------------
class _FakeBoxes:
    def __init__(self, xyxy: List[List[float]], conf: List[float], cls: List[int], ids: Optional[List[int]]):
        self.xyxy = np.asarray(xyxy, dtype=np.float64).reshape(-1, 4)
        self.conf = np.asarray(conf, dtype=np.float32)
        self.cls = np.asarray(cls, dtype=np.float32)  # ultralytics stores class ids as floats
        self.id = None if ids is None else np.asarray(ids, dtype=np.float32)

    def __len__(self) -> int:
        return int(self.xyxy.shape[0])


class _FakeResult:
    def __init__(self, boxes: _FakeBoxes):
        self.boxes = boxes


class _FakeModel:
    """Records every track/predict call; returns scripted boxes per call (last script repeats)."""

    def __init__(self, names: Dict[int, str], scripts: Optional[List[Optional[_FakeBoxes]]] = None):
        self.names = names
        self.scripts = scripts or []
        self.track_calls: List[Dict[str, Any]] = []
        self.predict_calls: List[Dict[str, Any]] = []

    def _next(self) -> list:
        n = len(self.track_calls) + len(self.predict_calls) - 1
        if not self.scripts:
            return [_FakeResult(_FakeBoxes([], [], [], None))]
        boxes = self.scripts[min(n, len(self.scripts) - 1)]
        if boxes is None:
            boxes = _FakeBoxes([], [], [], None)
        return [_FakeResult(boxes)]

    def track(self, frame, **kwargs):
        self.track_calls.append(dict(kwargs, frame_shape=frame.shape))
        return self._next()

    def predict(self, frame, **kwargs):
        self.predict_calls.append(dict(kwargs, frame_shape=frame.shape))
        return self._next()


VEHICLE_NAMES = {0: "Car", 1: "Jeep", 2: "Motorcycle", 3: "Tricycle", 4: "Truck", 5: "Van", 6: "person", 7: "train"}
INFRA_NAMES = {0: "railroad-crossing-down", 1: "railroad-crossings", 2: "railroad-sign",
               3: "railroad-sign-light-off", 4: "railroad-sign-light-on"}


def _frame(h: int = 360, w: int = 640) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)


def _load_tool(name: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("tool_" + name, TOOLS / (name + ".py"))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------------------------
# VehicleDetector with a stub model
# --------------------------------------------------------------------------------------------
def test_classify_is_case_insensitive_and_covers_all_categories():
    det = VehicleDetector(DetectorConfig(), model=_FakeModel(VEHICLE_NAMES))
    assert det.classify("Car") == CATEGORY_VEHICLE
    assert det.classify("TRUCK") == CATEGORY_VEHICLE
    assert det.classify(" jeep ") == CATEGORY_VEHICLE
    assert det.classify("bus") == CATEGORY_VEHICLE
    assert det.classify("person") == CATEGORY_PERSON
    assert det.classify("Pedestrian") == CATEGORY_PERSON
    assert det.classify("Train") == CATEGORY_TRAIN
    assert det.classify("railroad-sign") == CATEGORY_OTHER
    assert det.classify("") == CATEGORY_OTHER
    # custom config lists are honoured
    det2 = VehicleDetector(DetectorConfig(vehicle_classes=["forklift"], train_classes=["Locomotive"]),
                           model=_FakeModel(VEHICLE_NAMES))
    assert det2.classify("FORKLIFT") == CATEGORY_VEHICLE
    assert det2.classify("car") == CATEGORY_OTHER
    assert det2.classify("locomotive") == CATEGORY_TRAIN


def test_unmapped_model_classes_are_warned_about_at_startup(caplog):
    """A class in none of the three lists is a blind spot: it raises no WARNING/ALARM and is
    still excluded from the debris mask, so the operator has to be told, loudly and once."""
    names = {0: "Car", 1: "boat", 2: "bench"}
    with caplog.at_level(logging.WARNING, logger="gradecrossing.detection"):
        VehicleDetector(DetectorConfig(), model=_FakeModel(names))
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING
                and "not mapped" in r.getMessage()]
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert "boat" in message and "bench" in message and "Car" not in message
    assert "vehicle_classes" in message

    # A model whose classes are all mapped says nothing.
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="gradecrossing.detection"):
        VehicleDetector(DetectorConfig(), model=_FakeModel({0: "car", 1: "person", 2: "train"}))
    assert not [r for r in caplog.records if "not mapped" in r.getMessage()]


def test_names_property_returns_int_keyed_copy():
    model = _FakeModel({"0": "Car", "1": "Truck"})
    det = VehicleDetector(DetectorConfig(), model=model)
    names = det.names
    assert names == {0: "Car", 1: "Truck"}
    names[99] = "hack"
    assert 99 not in det.names  # a copy, not the internal dict


def test_detect_uses_track_with_persist_and_converts_boxes():
    boxes = _FakeBoxes([[10, 20, 50, 80], [100, 100, 200, 220]], [0.9, 0.42], [0, 4], [5, 9])
    model = _FakeModel(VEHICLE_NAMES, scripts=[boxes])
    cfg = DetectorConfig(conf=0.3, iou=0.6, imgsz=512, max_det=50, device="", half=False, tracker="bytetrack.yaml")
    det = VehicleDetector(cfg, model=model)
    out = det.detect(_frame())

    assert len(model.track_calls) == 1 and not model.predict_calls
    call = model.track_calls[0]
    assert call["persist"] is True
    assert call["tracker"] == "bytetrack.yaml"
    assert call["conf"] == 0.3 and call["iou"] == 0.6 and call["imgsz"] == 512 and call["max_det"] == 50
    assert call["device"] is None  # "" -> None (ultralytics auto)
    assert call["verbose"] is False
    assert "half" not in call  # half=False is the ultralytics default; not sent (avoids deprecation spam)

    assert len(out) == 2 and all(isinstance(d, Detection) for d in out)
    assert out[0].xyxy.dtype == np.float32 and out[0].xyxy.shape == (4,)
    np.testing.assert_allclose(out[0].xyxy, [10, 20, 50, 80])
    assert out[0].conf == pytest.approx(0.9, abs=1e-6)
    assert out[0].cls_id == 0 and out[0].cls_name == "Car" and out[0].track_id == 5
    assert out[1].cls_id == 4 and out[1].cls_name == "Truck" and out[1].track_id == 9
    assert isinstance(out[1].track_id, int)
    assert out[0].bottom_center == (30.0, 80.0)


def test_tracker_can_get_low_score_boxes_but_idless_boxes_keep_the_decision_threshold():
    """team-david D21 (corrected): model.track at conf 0.25 leaves ByteTrack's second (0.1-0.25)
    association stage empty. With detector.track_conf the tracker is fed lower; a box it returns
    with an id is a tracked box (ByteTrack's own thresholds decided it), while a box without an
    id (ultralytics hands back raw boxes when no track is active) is a plain detection and still
    needs conf. Opt-in: by default (None) the tracker gets conf, exactly the pre-port call."""
    boxes = _FakeBoxes([[10, 20, 50, 80], [100, 100, 200, 220], [300, 50, 340, 90], [400, 50, 450, 90]],
                       [0.15, 0.6, 0.12, 0.4], [0, 4, 0, 0], None)
    tracked = _FakeBoxes([[10, 20, 50, 80], [100, 100, 200, 220]], [0.15, 0.6], [0, 4], [3, 4])
    model = _FakeModel(VEHICLE_NAMES, scripts=[tracked, boxes])
    det = VehicleDetector(DetectorConfig(track_conf=0.1), model=model)
    out = det.detect(_frame())
    assert [(d.track_id, round(d.conf, 2)) for d in out] == [(3, 0.15), (4, 0.6)]  # a 0.15 box keeps id 3
    assert model.track_calls[0]["conf"] == 0.1
    out = det.detect(_frame())  # no ids: only the boxes at or above conf (0.25) survive
    assert [round(d.conf, 2) for d in out] == [0.6, 0.4]
    # the default sends the pre-port call (conf) and filters nothing
    assert DetectorConfig().track_conf is None
    model2 = _FakeModel(VEHICLE_NAMES, scripts=[boxes])
    out = VehicleDetector(DetectorConfig(), model=model2).detect(_frame())
    assert model2.track_calls[0]["conf"] == 0.25 and len(out) == 4
    # plain detection ignores track_conf
    model3 = _FakeModel(VEHICLE_NAMES, scripts=[boxes])
    VehicleDetector(DetectorConfig(tracker="", track_conf=0.1), model=model3).detect(_frame())
    assert model3.predict_calls[0]["conf"] == 0.25
    for bad in ({"track_conf": 0.0}, {"conf": 1.5}):
        with pytest.raises(ValueError):
            VehicleDetector(DetectorConfig(**bad), model=_FakeModel(VEHICLE_NAMES))


def test_bad_detector_scores_fail_when_the_monitor_is_built():
    """The Monitor builds the detector on the first frame, so a bad detector.track_conf used to
    raise on every frame instead: five tracebacks, the run loop gave up, and the CLI exited 0.
    The check now runs in Monitor.__init__, like the tracker's and the engine's (no torch here)."""
    from gradecrossing.config import MonitorConfig
    from gradecrossing.monitor import Monitor

    for key, value in (("track_conf", 0.0), ("conf", 1.5), ("conf", 0.0)):
        cfg = MonitorConfig()
        setattr(cfg.detector, key, value)
        with pytest.raises(ValueError, match=f"detector.{key}"):
            Monitor(cfg)
    assert Monitor(MonitorConfig())._detector is None  # valid: still built lazily on the first frame
    VehicleDetector.check_config(DetectorConfig(track_conf=1.0))  # (0, 1] is valid


def test_ignore_classes_are_dropped_and_not_warned_about(caplog):
    """ours-correctness F21: a static false positive inside the zone (COCO 'traffic light' on the
    crossing signals) blocked the background and keyframe refresh with no configuration fix."""
    names = {0: "car", 1: "traffic light", 2: "person"}
    boxes = _FakeBoxes([[0, 0, 10, 10], [20, 20, 30, 40], [50, 50, 60, 80]], [0.9, 0.8, 0.7], [0, 1, 2], [1, 2, 3])
    with caplog.at_level(logging.WARNING, logger="gradecrossing.detection"):
        det = VehicleDetector(DetectorConfig(ignore_classes=["Traffic Light"]), model=_FakeModel(names, [boxes]))
    assert not [r for r in caplog.records if "not mapped" in r.getMessage()]
    assert [d.cls_name for d in det.detect(_frame())] == ["car", "person"]
    plain = VehicleDetector(DetectorConfig(tracker=""), model=_FakeModel(names, [boxes]))
    assert [d.cls_name for d in plain.detect(_frame())] == ["car", "traffic light", "person"]


def test_half_true_is_sent_on_every_call():
    # ultralytics rebuilds the predictor (and loses tracker state) if the precision flag changes
    # between calls, so when half is on it must be present on *every* call.
    model = _FakeModel(VEHICLE_NAMES)
    det = VehicleDetector(DetectorConfig(half=True, device="0"), model=model)
    for _ in range(3):
        det.detect(_frame())
    assert all(call.get("half") is True for call in model.track_calls)
    assert all(call["device"] == "0" for call in model.track_calls)


def test_detect_uses_predict_when_tracker_disabled():
    boxes = _FakeBoxes([[0, 0, 10, 10]], [0.5], [1], None)
    model = _FakeModel(VEHICLE_NAMES, scripts=[boxes])
    det = VehicleDetector(DetectorConfig(tracker=""), model=model)
    out = det.detect(_frame())
    assert len(model.predict_calls) == 1 and not model.track_calls
    assert "persist" not in model.predict_calls[0] and "tracker" not in model.predict_calls[0]
    assert len(out) == 1 and out[0].track_id is None and out[0].cls_name == "Jeep"


def test_detect_handles_missing_ids_unknown_class_and_empty_results():
    boxes = _FakeBoxes([[1, 2, 3, 4]], [0.7], [42], None)  # class id not in names
    model = _FakeModel(VEHICLE_NAMES, scripts=[boxes, None])
    det = VehicleDetector(DetectorConfig(), model=model)
    first = det.detect(_frame())
    assert first[0].track_id is None and first[0].cls_name == "42"
    assert det.detect(_frame()) == []
    assert results_to_detections([], VEHICLE_NAMES) == []
    assert results_to_detections([types.SimpleNamespace(boxes=None)], VEHICLE_NAMES) == []


def test_reset_is_noop_without_predictor_and_resets_stub_trackers():
    model = _FakeModel(VEHICLE_NAMES)
    det = VehicleDetector(DetectorConfig(), model=model)
    det.reset()  # no predictor attribute yet -> nothing to do, must not raise

    class _Tracker:
        def __init__(self):
            self.reset_calls = 0

        def reset(self):
            self.reset_calls += 1

    tracker = _Tracker()
    model.predictor = types.SimpleNamespace(trackers=[tracker])
    det.reset()
    assert tracker.reset_calls == 1


def test_missing_local_weights_fail_fast(tmp_path):
    with pytest.raises(FileNotFoundError):
        VehicleDetector(DetectorConfig(weights=str(tmp_path / "nope.pt")))
    with pytest.raises(FileNotFoundError):
        VehicleDetector(DetectorConfig(weights="not_an_official_name.pt"))


def test_without_ultralytics_the_advice_keeps_one_opencv_and_nothing_promises_a_download(
        monkeypatch, tmp_path, caplog, capsys):
    """Final review Q5: the error said 'pip install ultralytics', which in a [detector]/[headless]
    environment adds the GUI opencv-python on top of opencv-python-headless (one cv2/ directory,
    two owners), and the 'will try to download' notice for yolo11n.pt came before the ImportError."""
    monkeypatch.setitem(sys.modules, "ultralytics", None)  # import ultralytics -> ImportError
    monkeypatch.chdir(tmp_path)  # no yolo11n.pt here
    with caplog.at_level(logging.INFO, logger="gradecrossing.detection"):
        with pytest.raises(ImportError) as info:
            load_yolo("yolo11n.pt")
    message = str(info.value)
    assert "ultralytics is required" in message  # the CLI test (exit 1) matches on this
    assert '".[detector]"' in message and '".[display]"' in message and "NVIDIA's torch" in message
    assert "not a bare 'pip install ultralytics'" in message
    assert [r.getMessage() for r in caplog.records if "download" in r.getMessage()] == []
    with pytest.raises(FileNotFoundError):  # a missing custom path is still the first error
        load_yolo(str(tmp_path / "best_typo.pt"))
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"\x00")
    for tool, argv in (("export_tensorrt", ["--weights", str(weights)]),):
        assert _load_tool(tool).main(argv) == 2
        err = capsys.readouterr().err
        assert "ultralytics is not installed" in err and '".[detector]"' in err and "not a bare" in err


# --------------------------------------------------------------------------------------------
# InfrastructureDetector with a stub model
# --------------------------------------------------------------------------------------------
def _infra_boxes(down_confs: List[float], light_on: bool = False) -> _FakeBoxes:
    xyxy, conf, cls = [], [], []
    for c in down_confs:
        xyxy.append([10, 10, 60, 60]); conf.append(c); cls.append(0)
    if light_on:
        xyxy.append([100, 10, 120, 40]); conf.append(0.9); cls.append(4)
    return _FakeBoxes(xyxy, conf, cls, None)


def test_infrastructure_cadence_hold_and_confidence():
    scripts = [
        _infra_boxes([0.55, 0.8], light_on=True),  # run 1 (frame 0)
        None,                                       # run 2 (frame 3)
        None,                                       # run 3 (frame 6)
        _infra_boxes([0.6]),                        # run 4 (frame 9)
    ]
    model = _FakeModel(INFRA_NAMES, scripts=scripts)
    cfg = InfrastructureConfig(every_n_frames=3, hold_seconds=2.0, conf=0.35, imgsz=416)
    infra = InfrastructureDetector(cfg, model=model)
    frame = _frame()

    s0 = infra.update(frame, 0.0, 0)
    assert isinstance(s0, GateState)
    assert s0.available is True and s0.gates_down is True and s0.lights_on is True
    assert s0.confidence == pytest.approx(0.8, abs=1e-6)  # max over gate-down boxes
    assert s0.last_update == 0.0 and s0.last_seen_down == 0.0
    assert len(model.predict_calls) == 1 and not model.track_calls
    assert model.predict_calls[0]["conf"] == 0.35 and model.predict_calls[0]["imgsz"] == 416
    assert "iou" not in model.predict_calls[0] and "max_det" not in model.predict_calls[0]
    assert len(infra.last_detections) == 3

    # frames 1 and 2: cached, no model call, same object
    assert infra.update(frame, 0.5, 1) is s0
    assert infra.update(frame, 1.0, 2) is s0
    assert len(model.predict_calls) == 1

    # frame 3 (t=1.5): nothing detected but still inside hold window -> gates_down held True
    s3 = infra.update(frame, 1.5, 3)
    assert len(model.predict_calls) == 2
    assert s3.gates_down is True and s3.lights_on is False and s3.confidence == 0.0
    assert s3.last_seen_down == 0.0 and s3.last_update == 1.5

    # frame 6 (t=3.0): hold expired -> False
    s6 = infra.update(frame, 3.0, 6)
    assert s6.gates_down is False and s6.last_seen_down == 0.0

    # frame 9 (t=4.5): gate down again
    s9 = infra.update(frame, 4.5, 9)
    assert s9.gates_down is True and s9.confidence == pytest.approx(0.6, abs=1e-6) and s9.last_seen_down == 4.5

    infra.reset()
    assert infra.state.last_seen_down is None and infra.state.gates_down is False and infra.state.available is True


def test_infrastructure_runs_on_first_call_even_off_cadence():
    model = _FakeModel(INFRA_NAMES, scripts=[_infra_boxes([0.7])])
    infra = InfrastructureDetector(InfrastructureConfig(every_n_frames=15), model=model)
    state = infra.update(_frame(), 12.3, 7)  # e.g. after a start_seconds seek
    assert state.last_update == 12.3 and state.gates_down is True
    assert len(model.predict_calls) == 1
    assert infra.update(_frame(), 12.4, 8).last_update == 12.3  # cached until frame 15
    assert infra.update(_frame(), 12.9, 15).last_update == 12.9


def test_infrastructure_class_matching_is_case_insensitive():
    boxes = _FakeBoxes([[0, 0, 5, 5]], [0.5], [0], None)
    model = _FakeModel({0: "Railroad-Crossing-DOWN"}, scripts=[boxes])
    infra = InfrastructureDetector(InfrastructureConfig(), model=model)
    assert infra.update(_frame(), 0.0, 0).gates_down is True


# --------------------------------------------------------------------------------------------
# Smoke tests with the real local weights (skipped without ultralytics)
# --------------------------------------------------------------------------------------------
@pytest.mark.skipif(not VEHICLE_WEIGHTS.is_file(), reason="50_epochs_best.pt not present")
def test_vehicle_detector_smoke_with_real_weights():
    pytest.importorskip("ultralytics")
    cfg = DetectorConfig(weights=str(VEHICLE_WEIGHTS), tracker="bytetrack.yaml", device="cpu", imgsz=320)
    det = VehicleDetector(cfg, model=None)
    assert det.names == {0: "Car", 1: "Jeep", 2: "Motorcycle", 3: "Tricycle", 4: "Truck", 5: "Van"}
    frame = _frame()
    tracker_ids = []
    for _ in range(3):
        out = det.detect(frame)
        assert isinstance(out, list) and all(isinstance(d, Detection) for d in out)  # noise -> usually []
        trackers = det._model.predictor.trackers
        assert len(trackers) == 1
        tracker_ids.append(id(trackers[0]))
    # persist=True: the very same tracker object survives all three calls
    assert len(set(tracker_ids)) == 1
    det.reset()
    assert det._model.predictor.trackers[0].frame_id == 0


@pytest.mark.skipif(not INFRA_WEIGHTS.is_file(), reason="infrastructure best.pt not present")
def test_infrastructure_detector_smoke_with_real_weights():
    pytest.importorskip("ultralytics")
    cfg = InfrastructureConfig(weights=str(INFRA_WEIGHTS), device="cpu", imgsz=320, every_n_frames=5)
    infra = InfrastructureDetector(cfg)
    assert infra.names[0] == "railroad-crossing-down" and infra.names[4] == "railroad-sign-light-on"
    frame = _frame()
    s0 = infra.update(frame, 0.0, 0)
    assert s0.available is True and s0.last_update == 0.0
    assert infra.update(frame, 0.04, 1) is s0
    boxes = infra.detect(frame)
    assert isinstance(boxes, list)


# --------------------------------------------------------------------------------------------
# tools/inspect_checkpoint.py
# --------------------------------------------------------------------------------------------
INSPECT_SNIPPET = r"""
import importlib.util, json, sys
spec = importlib.util.spec_from_file_location("ic", sys.argv[1])
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
info = mod.inspect_checkpoint(sys.argv[2])
assert "torch" not in sys.modules, "torch was imported"
assert "ultralytics" not in sys.modules, "ultralytics was imported"
print(json.dumps(info))
"""


@pytest.mark.parametrize(
    "weights, expected_names, expected_nc, expected_last_epoch, expected_map50",
    [
        pytest.param(VEHICLE_WEIGHTS, ["Car", "Jeep", "Motorcycle", "Tricycle", "Truck", "Van"], 6, 50, 0.92295, id="vehicle"),
        pytest.param(INFRA_WEIGHTS, ["railroad-crossing-down", "railroad-crossings", "railroad-sign",
                                     "railroad-sign-light-off", "railroad-sign-light-on"], 5, 213, 0.9303, id="infra"),
    ],
)
def test_inspect_checkpoint_without_torch(weights, expected_names, expected_nc, expected_last_epoch, expected_map50):
    if not weights.is_file():
        pytest.skip("{} not present".format(weights))
    proc = subprocess.run(
        [sys.executable, "-c", INSPECT_SNIPPET, str(TOOLS / "inspect_checkpoint.py"), str(weights)],
        capture_output=True, text=True, env=SUBPROCESS_ENV, cwd=str(ROOT), timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    info = json.loads(proc.stdout.strip().splitlines()[-1])
    assert [info["names"][str(i)] for i in range(expected_nc)] == expected_names
    assert info["nc"] == expected_nc
    assert info["epoch"] == -1 and info["stripped"] is True  # final best.pt: strip_optimizer() ran
    assert info["best_fitness"] is None
    assert info["optimizer_present"] is False and info["ema_present"] is False and info["model_present"] is True
    assert str(info["version"]).startswith("8.")
    assert str(info["date"]).startswith("2026-")
    assert info["train_metrics"]["metrics/mAP50(B)"] == pytest.approx(expected_map50, abs=1e-5)
    assert info["train_results_rows"] == expected_last_epoch
    assert info["train_results_last"]["epoch"] == expected_last_epoch
    assert info["train_args"]["imgsz"] == 640 and info["train_args"]["batch"] == 16
    assert info["model_yaml_file"] == "yolo11n.yaml"


def test_inspect_checkpoint_text_report_and_errors(capsys, tmp_path):
    ic = _load_tool("inspect_checkpoint")
    if VEHICLE_WEIGHTS.is_file():
        assert ic.main([str(VEHICLE_WEIGHTS)]) == 0
        out = capsys.readouterr().out
        assert "names" in out and "Car" in out and "train_metrics" in out and "epoch" in out
        assert out.isascii()
        assert ic.main([str(VEHICLE_WEIGHTS), "--json"]) == 0
        assert json.loads(capsys.readouterr().out)["nc"] == 6
    bogus = tmp_path / "bogus.pt"
    bogus.write_bytes(b"not a zip")
    assert ic.main([str(bogus)]) == 1
    assert ic.main([str(tmp_path / "missing.pt")]) == 1
    with pytest.raises(FileNotFoundError):
        ic.load_checkpoint_dict(str(tmp_path / "missing.pt"))


# --------------------------------------------------------------------------------------------
# tools/*.py compile, --help and --dry-run paths (no GPU / dataset needed)
# --------------------------------------------------------------------------------------------
TOOL_NAMES = sorted(p.stem for p in TOOLS.glob("*.py") if not p.stem.startswith("_"))  # _*.py: shared helpers


@pytest.mark.parametrize("tool", TOOL_NAMES)
def test_tool_compiles_and_help_works(tool):
    path = TOOLS / (tool + ".py")
    compile(path.read_text(encoding="utf-8"), str(path), "exec")
    proc = subprocess.run(
        [sys.executable, str(path), "--help"],
        capture_output=True, text=True, env=SUBPROCESS_ENV, cwd=str(ROOT), timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert "usage" in proc.stdout.lower()
    assert proc.stdout.isascii()


def _make_dataset(tmp_path: Path, with_test: bool = True, break_val: bool = False) -> Path:
    for split in ("train", "val") + (("test",) if with_test else ()):
        if split == "val" and break_val:
            continue
        (tmp_path / "images" / split).mkdir(parents=True)
    yaml_path = tmp_path / "data.yaml"
    lines = ["path: {}".format(tmp_path.as_posix()), "train: images/train", "val: images/val"]
    if with_test:
        lines.append("test: images/test")
    lines += ["names:", "  0: car", "  1: truck"]
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return yaml_path


def test_tune_tool_validates_dataset_and_dry_runs(tmp_path, capsys):
    tune = _load_tool("tune_hyperparameters")
    good = _make_dataset(tmp_path / "good")
    assert tune.main(["--data", str(good), "--dry-run", "--epochs", "5", "--iterations", "3"]) == 0
    out = capsys.readouterr().out
    assert '"val": true' in out and '"plots": false' in out and '"save": false' in out
    assert '"epochs": 5' in out and '"iterations": 3' in out

    broken = _make_dataset(tmp_path / "broken", break_val=True)
    assert tune.main(["--data", str(broken), "--dry-run"]) == 2
    assert "does not exist" in capsys.readouterr().err
    assert tune.main(["--data", str(tmp_path / "missing.yaml"), "--dry-run"]) == 2
    assert tune.main(["--data", str(good), "--dry-run", "--iterations", "0"]) == 2
    with pytest.raises(tune.DatasetError):
        tune.resolve_splits(str(broken))


def test_tune_tool_reads_fitness_from_csv_and_ndjson(tmp_path):
    tune = _load_tool("tune_hyperparameters")
    csv_dir = tmp_path / "tune"
    csv_dir.mkdir()
    (csv_dir / "tune_results.csv").write_text(
        "fitness,lr0,lrf\n0.0,0.01,0.01\n0.0,0.008,0.01\n", encoding="utf-8")
    assert tune.max_fitness(tune.read_fitness(csv_dir)) == 0.0  # the legacy failure mode
    (csv_dir / "tune_results.csv").write_text(
        "fitness,lr0,lrf\n0.0,0.01,0.01\n0.4123,0.008,0.01\nnan,0.1,0.1\n", encoding="utf-8")
    assert tune.max_fitness(tune.read_fitness(csv_dir)) == pytest.approx(0.4123)

    nd_dir = tmp_path / "tune2"
    nd_dir.mkdir()
    (nd_dir / "tune_results.ndjson").write_text(
        '{"iteration": 1, "fitness": 0.0}\n{"iteration": 2, "fitness": 0.31}\n\n', encoding="utf-8")
    assert tune.max_fitness(tune.read_fitness(nd_dir)) == pytest.approx(0.31)
    with pytest.raises(FileNotFoundError):
        tune.read_fitness(tmp_path)
    os.utime(str(csv_dir / "tune_results.csv"), (1.0e9, 1.0e9))
    assert tune.find_tune_dir(str(tmp_path), "tune") == nd_dir
    assert tune.find_tune_dir(str(tmp_path / "nothing"), "tune") is None


def test_train_tool_validates_dataset_and_dry_runs(tmp_path, capsys):
    train = _load_tool("train_vehicle")
    good = _make_dataset(tmp_path / "good")
    assert train.main(["--data", str(good), "--dry-run", "--epochs", "3", "--patience", "7"]) == 0
    out = capsys.readouterr().out
    assert "split would be 'test'" in out and '"patience": 7' in out and '"deterministic": true' in out

    no_test = _make_dataset(tmp_path / "notest", with_test=False)
    assert train.main(["--data", str(no_test), "--dry-run"]) == 0
    assert "split would be 'val'" in capsys.readouterr().out

    assert train.main(["--data", str(good), "--dry-run", "--cfg", str(tmp_path / "missing_hyp.yaml")]) == 2
    broken = _make_dataset(tmp_path / "broken", break_val=True)
    assert train.main(["--data", str(broken), "--dry-run"]) == 2

    hyp = tmp_path / "hyp.yaml"
    hyp.write_text("lr0: 0.005\n", encoding="utf-8")
    assert train.main(["--data", str(good), "--dry-run", "--cfg", str(hyp)]) == 0
    assert '"cfg"' in capsys.readouterr().out


def test_train_tool_collect_metrics_is_defensive():
    train = _load_tool("train_vehicle")

    class _Metrics:
        results_dict = {"metrics/mAP50(B)": 0.9, "fitness": 0.8}
        ap_class_index = [0, 1]
        maps = np.array([0.7, 0.6])
        speed = {"inference": 3.2}

        def class_result(self, i):
            return (0.5 + i, 0.6, 0.7, 0.8)

    rec = train.collect_metrics(_Metrics(), {0: "car", 1: "truck"})
    assert rec["summary"]["fitness"] == 0.8
    assert rec["per_class"]["truck"]["precision"] == 1.5
    assert rec["maps_per_class"] == [0.7, 0.6] and rec["speed_ms"] == {"inference": 3.2}
    bogus = train.collect_metrics(object(), {})  # not a DetMetrics: must not raise after hours of training
    assert bogus["summary"] == {} and bogus["per_class"] == {} and bogus["speed_ms"] == {}
    assert bogus.get("maps_per_class", []) == []


def test_export_tool_dry_run_and_validation(tmp_path, capsys):
    export = _load_tool("export_tensorrt")
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"\x00")
    assert export.main(["--weights", str(weights), "--half", "--dry-run", "--workspace", "2"]) == 0
    out = capsys.readouterr().out
    assert '"format": "engine"' in out and '"half": true' in out and '"workspace": 2.0' in out
    assert "Jetson" in out and out.isascii()

    assert export.main(["--weights", str(tmp_path / "missing.pt"), "--dry-run"]) == 2
    assert export.main(["--weights", str(weights), "--int8", "--dry-run"]) == 2  # needs --data
    assert export.main(["--weights", str(weights), "--device", "cpu", "--dry-run"]) == 2
    assert export.main(["--weights", str(weights), "--imgsz", "600", "--dry-run"]) == 2
    assert export.main(["--weights", str(weights), "--half", "--int8", "--data", str(weights), "--dry-run"]) == 2
    capsys.readouterr()
    assert export.main(["--weights", str(weights), "--int8", "--data", str(weights), "--dry-run"]) == 0
    assert '"int8": true' in capsys.readouterr().out
