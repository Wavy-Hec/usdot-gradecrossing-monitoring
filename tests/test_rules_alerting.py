"""AlertEngine changes that came with the YOLO-geometry port.

* shipped timing: WARNING 1.5 s, ALARM 3 s (the task definition "more than 3 s"), and a log
  line for gates-down thresholds that no longer escalate anything (P4, ours-realfootage M1);
* ALARM_UPDATE when a new cause joins an active ALARM (ours-correctness F06);
* the WARNING hold for a vehicle that has not moved (ours-realfootage F16);
* event payloads that can be tied back to objects (ours-realfootage F12);
* the constructor without the unused DebrisConfig (ours-cleanup OC-10).
"""
from __future__ import annotations

import json
import logging
from typing import List

import numpy as np
import pytest

from gradecrossing.alerting import AlertEngine, CallbackSink, event_to_json
from gradecrossing.config import AlertConfig, DebrisConfig, StuckConfig
from gradecrossing.types import DebrisResult, Event, GateState, ObjectState, Status

GATE = GateState(available=True)


def car(tid: int, stuck: float = 0.0, dwell: float = 0.0, in_zone: bool = True, near: bool = False,
        near_stuck: float = 0.0, cls: str = "car") -> ObjectState:
    return ObjectState(track_id=tid, cls_name=cls, is_vehicle=True, is_person=False, is_train=False,
                       last_box=np.array([100, 100, 160, 140], np.float32), last_seen=0.0, first_seen=0.0,
                       in_zone=in_zone, in_zone_since=0.0 if in_zone else None, dwell_seconds=dwell if in_zone else 0.0,
                       speed=1.5, speed_units="px/s", stationary_seconds=stuck, stuck_seconds=stuck if in_zone else 0.0,
                       zone_overlap=0.4 if in_zone else 0.0, zone_distance_bh=0.0 if in_zone else 0.5,
                       near_zone=near, near_stuck_seconds=near_stuck, still_seconds=stuck)


def debris(triggered: bool = True) -> DebrisResult:
    return DebrisResult(mask=np.zeros((4, 4), np.uint8), area_px=10, area_fraction=0.12, blobs=[],
                        persist_seconds=3.0, triggered=triggered, learning_rate_used=0.0, warmed_up=True)


def make(latch: float = 10.0, repeat: float = 0.0, hold: float = 0.0, **stuck) -> AlertEngine:
    return AlertEngine(StuckConfig(**stuck), AlertConfig(latch_seconds=latch, repeat_notify_seconds=repeat,
                                                         warning_hold_seconds=hold, stdout=False))


def collect(eng: AlertEngine) -> List[Event]:
    got: List[Event] = []
    eng.add_sink(CallbackSink(got.append))
    return got


# ----------------------------------------------------------------------------- P4 timing
def test_default_timing_is_3s_alarm_1_5s_warning():
    cfg = StuckConfig()
    assert (cfg.warning_seconds, cfg.alarm_seconds) == (1.5, 3.0)
    assert AlertConfig().warning_hold_seconds == 2.0
    eng = AlertEngine(StuckConfig(), AlertConfig(stdout=False))
    a, _ = eng.evaluate([car(7, stuck=1.49, dwell=1.49)], None, GATE, 1.49, 45)
    assert a.status is Status.CLEAR
    a, _ = eng.evaluate([car(7, stuck=1.5, dwell=1.5)], None, GATE, 1.5, 45)
    assert a.status is Status.WARNING
    a, _ = eng.evaluate([car(7, stuck=2.99, dwell=2.99)], None, GATE, 2.99, 90)
    assert a.status is Status.WARNING
    a, _ = eng.evaluate([car(7, stuck=3.0, dwell=3.0)], None, GATE, 3.0, 90)
    assert a.status is Status.ALARM and a.reasons == ["vehicle 7 (car) stationary in zone 3.0s"]


def test_gates_down_thresholds_not_shorter_logs_a_warning(caplog):
    with caplog.at_level(logging.WARNING, logger="gradecrossing.alerting"):
        AlertEngine(StuckConfig(), AlertConfig(stdout=False))  # shipped: gates-down alarm 3 s = alarm 3 s
    msgs = [r.getMessage() for r in caplog.records if "gates-down" in r.getMessage()]
    assert len(msgs) == 1 and "gates_down_alarm_seconds 3 s >= alarm_seconds 3 s" in msgs[0]
    assert "gates_down_warning_seconds" not in msgs[0]  # 1 s < 1.5 s still escalates
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="gradecrossing.alerting"):
        AlertEngine(StuckConfig(gates_down_alarm_seconds=2.0), AlertConfig(stdout=False))
        AlertEngine(StuckConfig(warning_seconds=3.0, alarm_seconds=8.0), AlertConfig(stdout=False))
    assert not [r for r in caplog.records if "gates-down" in r.getMessage()]
    # gates down still applies the (equal) thresholds: no ALARM before 3 s
    eng = make()
    a, _ = eng.evaluate([car(1, stuck=2.5, dwell=2.5)], None, GateState(gates_down=True, available=True), 2.5, 75)
    assert a.status is Status.WARNING and a.severity_scale == pytest.approx(1.0)


# ----------------------------------------------------------------------------- F06 ALARM_UPDATE
def test_a_new_vehicle_during_an_active_debris_alarm_emits_alarm_update():
    """ours-correctness probe: a ghost debris ALARM is active; a car then stops in the zone. The
    first event naming it used to be a repeat up to 30 s later (never with repeats off)."""
    eng = make(repeat=0.0)
    got = collect(eng)
    eng.evaluate([], debris(), GATE, 10.0, 300)
    eng.evaluate([car(9, stuck=2.0, dwell=2.0)], debris(), GATE, 12.0, 360)  # WARNING-level only: no update
    a, ev = eng.evaluate([car(9, stuck=3.1, dwell=3.1)], debris(), GATE, 13.1, 393)
    assert [e.type for e in ev] == ["ALARM_UPDATE"]
    u = ev[0]
    assert u.status == "ALARM" and u.track_ids == [9]
    assert u.extra["new_track_ids"] == [9]
    assert u.extra["new_reasons"] == ["vehicle 9 (car) stationary in zone 3.1s"]
    assert [o["id"] for o in u.extra["objects"]] == [9]
    # the same cause again is not news; a second new car is
    _, ev = eng.evaluate([car(9, stuck=4.0, dwell=4.0)], debris(), GATE, 14.0, 420)
    assert ev == []
    _, ev = eng.evaluate([car(9, stuck=5.0, dwell=5.0), car(11, stuck=3.0, dwell=3.0)], debris(), GATE, 15.0, 450)
    assert [e.type for e in ev] == ["ALARM_UPDATE"] and ev[0].extra["new_track_ids"] == [11]
    assert [e.type for e in got] == ["ALARM_START", "ALARM_UPDATE", "ALARM_UPDATE"]


def test_a_re_identified_vehicle_is_not_a_new_cause():
    """On clip05 the car waiting behind the passing tank cars came back under 15 ByteTrack ids;
    the tracker bridges those switches (same ObjectState.uid), so they are no ALARM_UPDATE. The
    END event still lists every id the episode used."""
    eng = make(latch=10.0)
    got = collect(eng)
    first = car(5, stuck=3.0, dwell=3.0)
    first.uid = 41
    eng.evaluate([first], None, GATE, 3.0, 90)
    again = car(77, stuck=9.0, dwell=9.0)  # same physical track, new detector id
    again.uid = 41
    _, ev = eng.evaluate([again], None, GATE, 9.0, 270)
    assert ev == []
    other = car(78, stuck=3.0, dwell=3.0)
    other.uid = 42
    _, ev = eng.evaluate([again, other], None, GATE, 10.0, 300)
    assert [e.type for e in ev] == ["ALARM_UPDATE"] and ev[0].extra["new_track_ids"] == [78]
    _, ev = eng.evaluate([], None, GATE, 30.0, 900)
    assert ev[0].type == "ALARM_END" and ev[0].track_ids == [5, 77, 78]
    assert [e.type for e in got] == ["ALARM_START", "ALARM_UPDATE", "ALARM_END"]


def test_alarm_update_while_latched_and_repeats_unchanged():
    eng = make(latch=10.0, repeat=5.0)
    got = collect(eng)
    eng.evaluate([car(1, stuck=3.5, dwell=3.5)], None, GATE, 3.5, 105)
    eng.evaluate([], None, GATE, 4.0, 120)  # car 1 gone: latched
    a, ev = eng.evaluate([car(2, stuck=3.0, dwell=3.0)], None, GATE, 6.0, 180)  # a new incident in the latch
    assert [e.type for e in ev] == ["ALARM_UPDATE"] and ev[0].extra["new_track_ids"] == [2]
    _, ev = eng.evaluate([car(2, stuck=5.0, dwell=5.0)], None, GATE, 8.5, 255)
    assert [(e.type, e.extra) for e in ev] == [("ALARM_START", {"repeat": True})]  # repeats stay as they were
    eng.evaluate([], debris(), GATE, 9.0, 270)  # debris joins
    assert got[-1].type == "ALARM_UPDATE" and got[-1].extra["new_reasons"][0].startswith("debris")
    _, ev = eng.evaluate([], None, GATE, 20.0, 600)
    assert [e.type for e in ev] == ["ALARM_END"]
    assert ev[0].track_ids == [1, 2]  # every cause of the episode
    assert ev[0].reasons == ["vehicle 1 (car) stationary in zone 3.5s"]


# ----------------------------------------------------------------------------- F16 WARNING hold
def test_warning_is_held_while_the_vehicle_has_not_moved():
    """ours-realfootage F16: clip01 stresstest, WARNING_END 13.32 s and WARNING_START 13.64 s for an
    SUV that never left (its stuck clock ended on a speed blip, the dwell reason took over 0.3 s
    later). Held, it is one WARNING episode."""
    eng = make(hold=2.0)
    got = collect(eng)
    eng.evaluate([car(2, stuck=1.6, dwell=5.0)], None, GATE, 13.2, 396)
    a, _ = eng.evaluate([car(2, stuck=0.0, dwell=5.1)], None, GATE, 13.32, 400)  # clock blip, still in zone
    assert a.status is Status.WARNING and a.reasons == ["held"] and a.warning_track_ids == []
    a, _ = eng.evaluate([car(2, stuck=0.2, dwell=5.4)], None, GATE, 13.64, 409)
    assert a.status is Status.WARNING
    eng.evaluate([car(2, stuck=1.6, dwell=5.5)], None, GATE, 13.8, 414)  # the condition is back
    assert [e.type for e in got] == ["WARNING_START"]
    # the hold runs out 2 s after the last WARNING condition
    a, _ = eng.evaluate([car(2, stuck=0.0, dwell=6.0)], None, GATE, 15.79, 474)
    assert a.status is Status.WARNING
    a, _ = eng.evaluate([car(2, stuck=0.0, dwell=6.1)], None, GATE, 15.8, 474)
    assert a.status is Status.CLEAR and [e.type for e in got] == ["WARNING_START", "WARNING_END"]


def test_warning_hold_ends_at_once_when_the_vehicle_leaves():
    for gone in ([], [car(2, in_zone=False)]):  # dropped, or out of (and not near) the zone
        eng = make(hold=2.0)
        got = collect(eng)
        eng.evaluate([car(2, stuck=1.6, dwell=1.6)], None, GATE, 1.6, 48)
        a, _ = eng.evaluate(gone, None, GATE, 1.7, 51)
        assert a.status is Status.CLEAR and [e.type for e in got] == ["WARNING_START", "WARNING_END"]
    # with the hold off a blip flaps, as before (alerts.warning_hold_seconds 0)
    eng = make(hold=0.0)
    got = collect(eng)
    for t, stuck in ((1.6, 1.6), (1.7, 0.0), (1.8, 1.7)):
        eng.evaluate([car(2, stuck=stuck, dwell=t)], None, GATE, t, int(t * 30))
    assert [e.type for e in got] == ["WARNING_START", "WARNING_END", "WARNING_START"]
    with pytest.raises(ValueError):
        make(hold=-1.0)


# ----------------------------------------------------------------------------- F12 payloads
def test_event_payloads_name_their_objects_and_separate_warning_reasons():
    eng = make(latch=1.0)
    got = collect(eng)
    eng.evaluate([car(1, stuck=3.2, dwell=3.2), car(159, stuck=1.8, dwell=1.8)], None, GATE, 8.0, 240)
    start = got[0]
    assert start.type == "ALARM_START" and start.track_ids == [1]
    assert start.reasons == ["vehicle 1 (car) stationary in zone 3.2s"]  # alarm reasons only
    assert start.extra["warning_reasons"] == ["vehicle 159 (car) stationary in zone 1.8s"]
    obj = start.extra["objects"][0]
    assert set(obj) >= {"id", "class", "box", "zone_overlap", "zone_distance_bh", "dwell", "stuck", "still", "speed"}
    assert obj["id"] == 1 and obj["class"] == "car" and obj["box"] == [100.0, 100.0, 160.0, 140.0]
    assert obj["stuck"] == 3.2 and obj["zone_overlap"] == 0.4 and obj["zone_distance_bh"] == 0.0
    # latched with no alarm left: "latched", and the END names the episode
    eng.evaluate([], None, GATE, 8.5, 255)
    eng.evaluate([], None, GATE, 9.5, 285)
    end = got[-1]
    assert end.type == "ALARM_END" and end.status == "CLEAR"
    assert end.track_ids == [1] and end.reasons == ["vehicle 1 (car) stationary in zone 3.2s"]
    assert end.extra["episode_start"] == 8.0 and end.extra["objects"] == []
    # every payload is plain JSON (an infinite zone distance is written as null)
    far = car(3, stuck=1.6, dwell=1.6)
    far.zone_distance_bh = float("inf")
    _, ev = eng.evaluate([far], None, GATE, 12.0, 360)
    line = json.loads(event_to_json(ev[0]))
    assert line["type"] == "WARNING_START" and line["extra"]["objects"][0]["zone_distance_bh"] is None
    assert "Infinity" not in event_to_json(ev[0])


def test_alarm_track_ids_property_follows_the_last_assessment():
    eng = make(latch=10.0)
    assert eng.alarm_track_ids == []
    eng.evaluate([car(4, stuck=3.0, dwell=3.0)], None, GATE, 3.0, 90)
    assert eng.alarm_track_ids == [4]
    eng.evaluate([], None, GATE, 3.1, 93)
    assert eng.alarm_track_ids == [] and eng.status is Status.ALARM  # latched; raw ids are honest
    eng.reset()
    assert eng.alarm_track_ids == []


# ----------------------------------------------------------------------------- OC-10 constructor
def test_constructor_takes_stuck_and_alert_configs_and_still_accepts_the_old_signature(caplog):
    alerts = AlertConfig(stdout=False, latch_seconds=0.0)
    eng = AlertEngine(StuckConfig(), alerts)
    assert eng.alert_cfg is alerts and not hasattr(eng, "debris_cfg")
    with caplog.at_level(logging.WARNING, logger="gradecrossing.alerting"):
        old = AlertEngine(StuckConfig(), DebrisConfig(), alerts)  # pre-port call sites
    assert old.alert_cfg is alerts
    assert any("deprecated" in r.getMessage() for r in caplog.records)
    with pytest.raises(TypeError):
        AlertEngine(StuckConfig(), alerts, alerts)
    with pytest.raises(ValueError):
        AlertEngine(StuckConfig(close_policy="loud"), alerts)
