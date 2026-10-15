"""Monitor wiring: each link between the components is checked end to end, so a broken
link fails here even when every component passes its own tests.

Links: ZoneState.to_reference -> DebrisDetector.update (to_reference, registration_ok,
train hold); detector cadence -> tracker.update(detector_ran=...); ZoneState.to_reference
-> the tracker's camera-motion compensation; ALARM_UPDATE -> the sinks; AlertEngine built
as (stuck_cfg, alert_cfg); one class mapping (ConfigClassifier, CATEGORY_*) for detector,
tracker and monitor; Monitor.reset closes an open episode (MISS-01). Synthetic frames and
scripted detections from tests/test_monitor.py; nothing imports ultralytics.
"""
from __future__ import annotations

import dataclasses
import logging
import subprocess
import sys
from typing import Any, Dict, List

import numpy as np
import pytest

import gradecrossing.alerting as alerting
import gradecrossing.detection as detection
import gradecrossing.tracking as tracking
import gradecrossing.visualize as visualize
from gradecrossing.alerting import AlertEngine, CallbackSink
from gradecrossing.config import AlertConfig, DetectorConfig, StuckConfig
from gradecrossing.monitor import ConfigClassifier, Monitor
from gradecrossing.tracking import PredictedDetection
from gradecrossing.types import Detection, Event, GateState, ObjectState, Status

from test_monitor import FPS, H, W, ROOT, Session, car_det, make_cfg, stop_script  # noqa: E402


def parked(x: float, y: float = 230.0, start: int = 0, tid: int = 1):
    """A car that stands still at bottom-center (x, y) from frame ``start`` on."""
    return lambda i: [car_det(x, y, tid=tid)] if i >= start else []


def spy(obj: Any, name: str, calls: List[Dict[str, Any]]) -> None:
    """Record every call of ``obj.name`` (args and kwargs) and still run it."""
    original = getattr(obj, name)

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        calls.append({"args": args, "kwargs": kwargs})
        return original(*args, **kwargs)

    setattr(obj, name, wrapper)


# ----------------------------------------------------------------------------- zone -> debris
def test_debris_is_registered_with_zone_to_reference_and_held_by_trains():
    train = np.array([195.0, 145.0, 445.0, 275.0], np.float32)  # over the whole zone

    def script(i: int) -> List[Detection]:
        if 50 <= i < 60:
            return [Detection(xyxy=train.copy(), conf=0.9, cls_id=6, cls_name="train", track_id=9)]
        return []

    s = Session(make_cfg(), script)
    calls: List[Dict[str, Any]] = []
    spy(s.mon.debris, "update", calls)
    res = s.run(60)
    assert len(calls) == 60
    for r, call in zip(res, calls):
        kw = call["kwargs"]
        assert r.zone.healthy and r.zone.to_reference is not None
        assert kw["registration_ok"] is True
        np.testing.assert_allclose(kw["to_reference"], r.zone.to_reference)  # the stabiliser's own matrix
        assert kw["hold"] is (50 <= r.frame_index < 60)  # a train box over the zone holds the timers
        assert call["args"][3] is None or call["args"][3] >= 0.0  # zone_empty_seconds
    # a held zone: no registration, and the debris result says why it is held
    uniform = np.full((H, W, 3), 128, dtype=np.uint8)
    r = s.mon.process_frame(uniform, 60 / FPS, 60)
    assert not r.zone.healthy
    assert calls[-1]["kwargs"]["registration_ok"] is False and calls[-1]["kwargs"]["to_reference"] is None
    assert r.debris.held and r.debris.state == "held (registration)"


# ----------------------------------------------------------------------------- detector -> tracker
def test_tracker_is_told_which_frames_the_detector_ran_on():
    s = Session(make_cfg(detector={"detect_every_n_frames": 3}), parked(320.0))
    calls: List[Dict[str, Any]] = []
    spy(s.mon.tracker, "update", calls)
    res = s.run(10)
    assert [c["kwargs"]["detector_ran"] for c in calls] == [i % 3 == 0 for i in range(10)]
    assert s.fake.calls == [0, 3, 6, 9]
    for r, c in zip(res, calls):
        assert c["args"][1] is r.zone  # the tracker gets this frame's zone state
        predicted = [isinstance(d, PredictedDetection) for d in c["args"][0]]
        assert predicted == [r.frame_index % 3 != 0] * len(predicted) and predicted


def test_tracker_measures_motion_through_zone_to_reference():
    """The monitor hands the stabiliser's ``to_reference`` to the tracker, which keeps its
    speed history in reference pixels: with a (forced) +50 px reference shift, the history
    of a parked car sits 50 px to the right of its image position; with
    stuck.camera_motion_compensation off it is the image position."""
    shift = np.array([[1.0, 0.0, 50.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    xs = {}
    for compensate in (True, False):
        s = Session(make_cfg(debris={"enabled": False}, stuck={"camera_motion_compensation": compensate}),
                    parked(320.0))
        s.run(1)  # builds the stabiliser
        stab = s.mon.stabilizer
        real_update = stab.update

        def shifted_update(*args: Any, _real=real_update, **kwargs: Any):
            z = _real(*args, **kwargs)
            return dataclasses.replace(z, to_reference=shift @ z.to_reference)

        stab.update = shifted_update
        r = s.run(10)[-1]
        assert r.zone.healthy
        xs[compensate] = r.objects[0].history[-1][1]
    assert xs[True] == pytest.approx(320.0 + 50.0, abs=1.0)
    assert xs[False] == pytest.approx(320.0, abs=1.0)


# ----------------------------------------------------------------------------- engine -> sinks
def test_alarm_update_reaches_the_sinks_and_the_frame_result():
    def script(i: int) -> List[Detection]:
        out = [car_det(250.0, 230.0, tid=1)]
        if i >= 60:
            out.append(car_det(390.0, 230.0, tid=2))
        return out

    s = Session(make_cfg(), script)
    res = s.run(120)
    updates = s.events_of("ALARM_UPDATE")
    assert len(updates) == 1 and updates[0].extra["new_track_ids"] == [2]
    in_results = [e for r in res for e in r.events if e.type == "ALARM_UPDATE"]
    assert in_results == updates  # the sink got exactly the events the FrameResults carry
    assert [e.type for e in s.events_of("ALARM_START")] == ["ALARM_START"]


def test_engine_is_built_from_the_stuck_and_alert_configs_only(caplog):
    cfg = make_cfg()
    with caplog.at_level(logging.WARNING, logger="gradecrossing.alerting"):
        mon = Monitor(cfg, detector=object())
    assert mon.engine.stuck_cfg is cfg.stuck and mon.engine.alert_cfg is cfg.alerts
    assert not hasattr(mon.engine, "debris_cfg")
    assert not [r for r in caplog.records if "deprecated" in r.getMessage()]


# ----------------------------------------------------------------------------- one class mapping
def test_one_class_mapping_and_one_set_of_category_constants():
    for name in ("CATEGORY_VEHICLE", "CATEGORY_PERSON", "CATEGORY_TRAIN", "CATEGORY_OTHER"):
        assert getattr(tracking, name) is getattr(detection, name)
    assert ConfigClassifier is detection.ConfigClassifier  # re-exported by the monitor
    source = (ROOT / "gradecrossing" / "tracking.py").read_text(encoding="utf-8")
    assert "CATEGORY_VEHICLE =" not in source  # imported, not redefined

    class StubModel:
        names = {0: "car", 1: "person", 2: "train", 3: "boat"}

    cfg = DetectorConfig()
    det = detection.VehicleDetector(cfg, model=StubModel())
    mapping = ConfigClassifier(cfg)
    for name in ("car", " Car ", "TRUCK", "person", "Pedestrian", "train", "boat", "", "jeep "):
        assert det.classify(name) == mapping(name)


def test_detection_imports_without_torch():
    code = ("import sys; import gradecrossing.detection as d; from gradecrossing.monitor import ConfigClassifier; "
            "assert ConfigClassifier is d.ConfigClassifier; "
            "bad = [m for m in ('torch', 'ultralytics') if m in sys.modules]; assert not bad, bad")
    proc = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


def test_console_text_goes_through_ascii_safe():
    assert visualize.ascii_safe is alerting.ascii_safe
    s = Session(make_cfg())
    r = s.run(1)[-1]
    r = dataclasses.replace(r, degraded=True, degraded_reasons=("zone held → café",))
    line = Monitor._status_line(r, None)
    assert line.isascii() and "DEGRADED(zone held ? caf?)" in line
    visualize.draw_overlay(s.mon.last_frame, r)  # Hershey text: no exception on non-ASCII input


# ----------------------------------------------------------------------------- MISS-01: reset closes episodes
def test_reset_closes_an_open_alarm_and_the_hazard_alarms_again():
    s = Session(make_cfg(), stop_script(start=0, x0=320.0, speed=120.0, y=230.0, stop_x=320.0, leave=10 ** 6))
    s.run(int(3.0 * FPS))
    assert s.mon.engine.status is Status.ALARM
    start = s.events_of("ALARM_START")[0]
    last = s.results[-1]
    n = len(s.events)
    s.mon.reset()
    ends = s.events[n:]
    assert [e.type for e in ends] == ["ALARM_END"]
    end = ends[0]
    assert end.extra["reason"] == "pipeline re-initialised" and end.status == "CLEAR"
    assert end.timestamp == last.timestamp and end.frame_index == last.frame_index
    assert end.track_ids == [1] and end.extra["episode_start"] == start.timestamp
    assert s.mon.engine.status is Status.CLEAR
    # the car is still there: the re-initialised monitor raises the alarm again
    s.run(int(3.0 * FPS))
    again = [e for e in s.events[n + 1:] if e.type == "ALARM_START"]
    assert again and s.mon.engine.status is Status.ALARM
    # an explicit timestamp is used as is; nothing open -> nothing emitted
    n = len(s.events)
    s.mon.reset(timestamp=123.0)
    assert [(e.type, e.timestamp) for e in s.events[n:]] == [("ALARM_END", 123.0)]
    n = len(s.events)
    s.mon.reset()
    assert s.events[n:] == []


def test_reset_closes_an_open_warning():
    s = Session(make_cfg(stuck={"alarm_seconds": 30.0}),
                stop_script(start=0, x0=320.0, speed=120.0, y=230.0, stop_x=320.0, leave=10 ** 6))
    s.run(int(1.5 * FPS))
    assert s.mon.engine.status is Status.WARNING
    n = len(s.events)
    s.mon.reset()
    assert [(e.type, e.extra.get("reason")) for e in s.events[n:]] == [("WARNING_END", "pipeline re-initialised")]


def _obj(tid: int, stuck: float) -> ObjectState:
    return ObjectState(track_id=tid, cls_name="car", is_vehicle=True, is_person=False, is_train=False,
                       last_box=np.array([300, 190, 360, 230], np.float32), last_seen=0.0, first_seen=0.0,
                       in_zone=True, in_zone_since=0.0, dwell_seconds=stuck, speed=0.0, speed_units="px/s",
                       stationary_seconds=stuck, stuck_seconds=stuck, zone_overlap=1.0, zone_distance_bh=0.0)


def test_engine_close_open_episodes():
    got: List[Event] = []
    eng = AlertEngine(StuckConfig(), AlertConfig(stdout=False))
    eng.add_sink(CallbackSink(got.append))
    assert eng.close_open_episodes(1.0, 30) == [] and got == []  # CLEAR: nothing to close
    eng.evaluate([_obj(4, 5.0)], None, GateState(), 2.0, 60)
    assert eng.status is Status.ALARM and eng.latched_until is not None
    events = eng.close_open_episodes(2.5, 75, reason="pipeline re-initialised")
    assert [e.type for e in events] == ["ALARM_END"] and got[-1] is events[0]
    assert events[0].track_ids == [4] and events[0].reasons and events[0].extra["episode_start"] == 2.0
    assert eng.status is Status.CLEAR and eng.latched_until is None and eng.alarm_track_ids == []
    # still stuck on the next frame: a new episode opens with a fresh ALARM_START
    _, again = eng.evaluate([_obj(4, 5.1)], None, GateState(), 2.6, 78)
    assert [e.type for e in again] == ["ALARM_START"]
