"""Vehicles stopped *too close* to the tracks (stuck.close_policy) and the on-or-near streak.

The engine tests use hand-made ObjectStates, like tests/test_alerting.py, with the pre-port
3 s / 8 s timing pinned so the thresholds read the same as in the plan. The team's "merged"
policy (ALARM on the tracks OR too close, F1 0.94 vs 0.80 on-track only, EXPERIMENTS.md E6)
is close_policy="alarm"; the default "warning" is the advisor's three-level policy
(evaluate_policies.py: "ALARM = on the tracks; WARNING = stopped too close"): it keeps ALARM
meaning "on the tracks" and reports too-close stops one tier lower, because the merged
policy raised 26-55 alarms/h at busy intersections (E6/E8) and false-alarmed on cars waiting
at the gate of the low-angle team clips.
"""
from __future__ import annotations

import logging

import numpy as np
import pytest

from gradecrossing.alerting import AlertEngine, CallbackSink
from gradecrossing.config import AlertConfig, StuckConfig
from gradecrossing.geometry import footprint_relation, polygon_mask
from gradecrossing.tracking import StuckVehicleTracker
from gradecrossing.types import Detection, GateState, ObjectState, Status, ZoneState

FPS = 30.0
DT = 1.0 / FPS


def obj(tid=1, kind="vehicle", in_zone=False, near=False, stuck=0.0, near_stuck=0.0, dwell=0.0) -> ObjectState:
    return ObjectState(track_id=tid, cls_name={"vehicle": "car", "person": "person", "train": "train"}[kind],
                       is_vehicle=kind == "vehicle", is_person=kind == "person", is_train=kind == "train",
                       last_box=np.array([0, 0, 10, 10], np.float32), last_seen=0.0, first_seen=0.0,
                       in_zone=in_zone, in_zone_since=0.0 if in_zone else None, dwell_seconds=dwell,
                       speed=0.0, speed_units="px/s", stationary_seconds=stuck, stuck_seconds=stuck,
                       near_zone=near, near_stuck_seconds=near_stuck, zone_distance_bh=0.5 if near else 0.0)


def engine(policy="warning", **stuck) -> AlertEngine:
    cfg = StuckConfig(**{"warning_seconds": 3.0, "alarm_seconds": 8.0, "close_policy": policy, **stuck})
    return AlertEngine(cfg, AlertConfig(latch_seconds=0.0, repeat_notify_seconds=0.0, warning_hold_seconds=0.0,
                                        stdout=False))


GATE_UP = GateState(available=True)
GATE_DOWN = GateState(gates_down=True, available=True)


def classify(name: str) -> str:
    return {"car": "vehicle", "truck": "vehicle", "person": "person", "train": "train"}.get(name, "other")


def zone_state(poly) -> ZoneState:
    p = np.asarray(poly, dtype=np.int32)
    return ZoneState(polygon=p, mask=polygon_mask((360, 640), p), homography=None, inliers=0,
                     method="identity", healthy=True)


def det(b, tid=1, cls="car") -> Detection:
    return Detection(xyxy=np.asarray(b, np.float32), conf=0.9, cls_id=2, cls_name=cls, track_id=tid)


# ----------------------------------------------------------------------------- engine tiers
def test_near_vehicle_warns_by_default_and_never_alarms():
    eng = engine()
    a, _ = eng.evaluate([obj(near=True, near_stuck=2.9)], None, GATE_UP, 2.9, 87)
    assert a.status is Status.CLEAR
    a, ev = eng.evaluate([obj(near=True, near_stuck=3.0)], None, GATE_UP, 3.0, 90)
    assert a.status is Status.WARNING and a.warning_track_ids == [1]
    assert a.reasons == ["vehicle 1 (car) stopped near the tracks 3.0s"]
    a, _ = eng.evaluate([obj(near=True, near_stuck=600.0)], None, GATE_UP, 600.0, 18000)
    assert a.status is Status.WARNING  # however long: ALARM still means "on the tracks"


def test_merged_policy_alarms_on_near_stops():
    eng = engine("alarm")
    a, _ = eng.evaluate([obj(near=True, near_stuck=3.5)], None, GATE_UP, 3.5, 105)
    assert a.status is Status.WARNING
    a, _ = eng.evaluate([obj(near=True, near_stuck=8.0)], None, GATE_UP, 8.0, 240)
    assert a.status is Status.ALARM and a.alarm_track_ids == [1]
    assert a.reasons == ["vehicle 1 (car) stopped on/near the tracks 8.0s"]
    # gates down: the shorter gates-down thresholds apply to near stops too
    eng2 = engine("alarm")
    a, _ = eng2.evaluate([obj(near=True, near_stuck=3.1)], None, GATE_DOWN, 3.1, 93)
    assert a.status is Status.ALARM and a.reasons[0].endswith("(gates down)")


def test_policy_off_restores_the_old_behaviour():
    eng = engine("off")
    a, _ = eng.evaluate([obj(near=True, near_stuck=60.0)], None, GATE_UP, 60.0, 1800)
    assert a.status is Status.CLEAR and a.reasons == []


def test_on_track_tiers_are_unchanged_and_take_precedence():
    eng = engine("alarm")
    a, _ = eng.evaluate([obj(in_zone=True, stuck=8.2, near_stuck=8.2)], None, GATE_UP, 8.2, 246)
    assert a.reasons == ["vehicle 1 (car) stationary in zone 8.2s"]
    # stopped near for 4 s, then crept onto the zone 1 s ago: the merged clock keeps the WARNING
    eng2 = engine("warning")
    a, _ = eng2.evaluate([obj(in_zone=True, stuck=1.0, near_stuck=4.0)], None, GATE_UP, 4.0, 120)
    assert a.status is Status.WARNING and a.reasons == ["vehicle 1 (car) stopped near the tracks 4.0s"]


def test_near_never_applies_to_persons_or_trains():
    eng = engine("alarm")
    objs = [obj(1, kind="person", near=True, near_stuck=60.0), obj(2, kind="train", near=True, near_stuck=60.0)]
    a, _ = eng.evaluate(objs, None, GATE_UP, 60.0, 1800)
    assert a.status is Status.CLEAR


def test_near_warning_events_and_bad_policy():
    eng = engine()
    got = []
    eng.add_sink(CallbackSink(got.append))
    eng.evaluate([obj(near=True, near_stuck=3.2)], None, GATE_UP, 3.2, 96)
    eng.evaluate([], None, GATE_UP, 3.3, 99)
    assert [e.type for e in got] == ["WARNING_START", "WARNING_END"]
    assert got[0].extra["objects"][0]["near_zone"] is True
    with pytest.raises(ValueError):
        engine("loud")


def test_close_policy_with_bottom_center_logs_a_noop_warning(caplog):
    with caplog.at_level(logging.WARNING, logger="gradecrossing.alerting"):
        engine("warning", zone_test="bottom_center")
    assert any("no effect" in r.getMessage() and "bottom_center" in r.getMessage() for r in caplog.records)
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="gradecrossing.alerting"):
        engine("off", zone_test="bottom_center")
        engine("warning")
    assert not any("no effect" in r.getMessage() for r in caplog.records)


# ----------------------------------------------------------------------------- tracker streak
BAND = [[0, 300], [640, 300], [640, 330], [0, 330]]


def near_box(y2: float) -> np.ndarray:
    return np.array([260.0, y2 - 80.0, 380.0, y2], np.float32)


def test_near_streak_survives_flicker_at_the_close_boundary():
    """The footprint-to-zone distance alternates 0.74 / 0.76 box heights (one pixel row): the
    on-or-near streak is held by the exit debounce, so near_stuck keeps growing; without the
    debounce every 'not near' frame restarted it."""
    z = zone_state(BAND)
    assert footprint_relation(near_box(242.0), z.polygon, z.mask)[1] < 0.75 < \
        footprint_relation(near_box(240.0), z.polygon, z.mask)[1]
    for grace, grows in ((0.5, True), (0.0, False)):
        tr = StuckVehicleTracker(StuckConfig(zone_exit_grace_seconds=grace), classify)
        for i in range(int(6 * FPS)):
            o = tr.update([det(near_box((242.0, 240.0)[i % 2]))], z, i * DT)[0]
        assert not o.in_zone
        assert (o.near_stuck_seconds >= 5.5) is grows, grace


def test_near_streak_survives_an_id_switch():
    """A duplicate-box id switch the re-keying cannot match: the near streak, like the in-zone
    streak, belongs to the still cluster the new id takes over."""
    z = zone_state(BAND)
    tr = StuckVehicleTracker(StuckConfig(), classify)
    for i in range(int(6 * FPS) + 1):
        t = i * DT
        dets = [det(near_box(242.0), tid=1 if t < 3.0 else 2)]
        if 2.9 <= t < 3.0:
            dets.append(det(near_box(243.0), tid=2))
        objs = {o.track_id: o for o in tr.update(dets, z, t)}
    o = objs[2]
    assert o.near_zone and not o.in_zone
    assert o.near_stuck_seconds == pytest.approx(6.0 - DT, abs=0.05)
