"""Scenarios behind the rollback check (tests/test_legacy_profile.py).

The goldens in tests/data/legacy_events.json were recorded by tools/record_legacy_goldens.py
from commit ae66b8f (the package before the YOLO-geometry port) with no overrides; the test
replays the same scenarios on the current package with configs/legacy_rule.yaml applied.

* Monitor scenarios: test_monitor.py's A (crossing car), B (stop, latch, clear), C (gates
  down) and E (id swap), with the debris detector off so that changes to the debris path
  (another module, not part of the vehicle rule) cannot move these goldens. Scenario D is
  debris-only and is left out for the same reason.
* Tracker scenarios: the five test_tracking.py tests that pin the pre-port semantics.

Everything here uses only APIs that exist in ae66b8f, because the recorder imports this file
against that commit's package.
"""
from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional

import numpy as np

import test_monitor as tm
import test_tracking as tt

FPS = tm.FPS
# Timing is set by each scenario itself (make_cfg's fast thresholds): the profile's 3 s / 8 s
# must not override it.
SCENARIO_TIMING = ("warning_seconds", "alarm_seconds")


def _merge(base: Dict[str, Dict[str, Any]], top: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out = copy.deepcopy(base)
    for section, values in top.items():
        out.setdefault(section, {}).update(values)
    return out


def _swap_script():
    base = tm.stop_script(start=30, x0=80.0, speed=120.0, y=230.0, stop_x=320.0, leave=10 ** 6)

    def script(i: int):
        dets = base(i)
        for d in dets:
            d.track_id = 2 if i >= int(6.0 * FPS) else 1
        return dets

    return script


def monitor_scenarios() -> Dict[str, Dict[str, Any]]:
    """name -> {sections, script, frames, infra} with the parameters of test_monitor.py."""
    no_debris = {"debris": {"enabled": False}}
    return {
        "A_crossing_car": dict(sections=no_debris, frames=160,
                               script=tm.crossing_script(start=30, x0=80.0, speed=120.0, y=230.0)),
        "B_stop_latch_clear": dict(sections=no_debris, frames=int(11.0 * FPS),
                                   script=tm.stop_script(start=30, x0=80.0, speed=120.0, y=230.0, stop_x=320.0,
                                                         leave=int(8.0 * FPS))),
        "C_gates_down": dict(sections=_merge(no_debris, {"stuck": {"alarm_seconds": 3.0, "warning_seconds": 2.0,
                                                                   "gates_down_alarm_seconds": 0.5,
                                                                   "gates_down_warning_seconds": 0.2}}),
                             frames=int(6.0 * FPS), infra=True,
                             script=tm.stop_script(start=30, x0=80.0, speed=120.0, y=230.0, stop_x=320.0,
                                                   leave=int(9.0 * FPS))),
        "E_id_swap": dict(sections=_merge(no_debris, {"stuck": {"alarm_seconds": 4.0, "warning_seconds": 1.0}}),
                          frames=int(10.0 * FPS), script=_swap_script()),
    }


def run_monitor_scenario(name: str, profile: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
    spec = monitor_scenarios()[name]
    base = copy.deepcopy(profile or {})
    for key in SCENARIO_TIMING:
        base.get("stuck", {}).pop(key, None)
    sections = _merge(base, spec["sections"])
    session = tm.Session(tm.make_cfg(**sections), spec["script"], infra=tm.FakeInfra(down=True) if spec.get("infra") else None)
    results = session.run(spec["frames"])
    events = [dict(type=e.type, timestamp=float(e.timestamp), frame_index=int(e.frame_index), status=e.status,
                   track_ids=[int(i) for i in e.track_ids], reasons=list(e.reasons), extra=_plain(e.extra))
              for e in session.events]
    return {"events": events, "statuses": [r.assessment.status.value for r in results]}


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, float):
        return value if value == value and abs(value) != float("inf") else None
    return value


def _snap(tracker, objs) -> Dict[str, Any]:
    return {"objects": [dict(id=int(o.track_id), in_zone=bool(o.in_zone), dwell=float(o.dwell_seconds),
                             stuck=float(o.stuck_seconds), stationary=float(o.stationary_seconds),
                             speed=float(o.speed), units=o.speed_units) for o in objs],
            "occupied": bool(tracker.zone_occupied),
            "empty_since": None if tracker.zone_empty_since is None else float(tracker.zone_empty_since)}


def _frames_enters_then_stops():
    out = []
    for i in range(181):
        t = i * tt.DT
        out.append((t, [tt.det(40.0 + 200.0 * min(t, 1.5), 300.0)]))
    return out


def run_tracker_scenario(name: str, stuck: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Per-frame object states of one of the five pre-port tracker scenarios."""
    stuck = dict(stuck or {})
    zone = tt.make_zone()
    out: List[Dict[str, Any]] = []

    def feed(tracker, frames):
        for t, dets in frames:
            out.append(_snap(tracker, tracker.update(dets, zone, t)))

    if name == "moving_car":
        feed(tt.make_tracker(**stuck), tt.moving_car_frames(40.0, 200.0, 300.0, 91))
    elif name == "enters_then_stops":
        feed(tt.make_tracker(**stuck), _frames_enters_then_stops())
    elif name == "predicted_box_membership":
        tracker = tt.make_tracker(**stuck)
        frames = list(tt.moving_car_frames(-10.0, 200.0, 300.0, 31, tid=1))
        feed(tracker, frames)
        t = frames[-1][0] + 0.2
        feed(tracker, [(t, tracker.predict_boxes(t))])
        feed(tracker, [(t + tt.DT, [tt.det(230.0, 300.0, tid=1)])])
    elif name == "ground_plane_homography":
        H = [[0.01, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 1.0]]
        feed(tt.make_tracker(ground_plane_homography=H, speed_threshold=1.0, **stuck),
             tt.moving_car_frames(40.0, 200.0, 300.0, 61))
        feed(tt.make_tracker(ground_plane_homography=H, speed_threshold=1.0, **stuck),
             tt.stationary_car_frames(320.0, 280.0, 91, np.random.default_rng(5)))
    elif name == "zone_empty_since":
        tracker = tt.make_tracker(**stuck)
        feed(tracker, [(0.5, []), (1.0, [])])
        feed(tracker, tt.moving_car_frames(40.0, 200.0, 300.0, 91, t0=1.0))
    else:
        raise KeyError(name)
    return out


TRACKER_SCENARIOS = ("moving_car", "enters_then_stops", "predicted_box_membership", "ground_plane_homography",
                     "zone_empty_since")


def record_all(profile: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Every scenario's output, with ``profile`` ({section: {key: value}}) applied."""
    profile = profile or {}
    return {"monitor": {name: run_monitor_scenario(name, profile) for name in monitor_scenarios()},
            "tracker": {name: run_tracker_scenario(name, profile.get("stuck")) for name in TRACKER_SCENARIOS}}
