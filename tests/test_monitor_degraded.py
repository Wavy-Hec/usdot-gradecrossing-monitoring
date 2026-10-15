"""DEGRADED: the monitor says when it cannot fully see (team-occupancy PORT-04, OCC-08).

``FrameResult.degraded`` / ``degraded_reasons`` are per frame; a change that lasts
``alerts.health_debounce_seconds`` becomes one HEALTH event with extra
{'component': 'monitor', 'degraded': bool, 'reasons': [...]}. Zone health is reported by
the zone's own (debounced) HEALTH events and is not reported a second time, unless
zone.health_events is off; debris warm-ups are reported by the debris component only, and
its trusted/untrusted changes are debounced the same way. Synthetic frames and scripted
detections from tests/test_monitor.py.
"""
from __future__ import annotations

import dataclasses
import logging
from typing import List

import numpy as np
import pytest

from gradecrossing.monitor import Monitor
from gradecrossing.types import Detection, Event, GateState
from gradecrossing.visualize import draw_overlay

from test_monitor import FPS, H, W, FakeInfra, Session, car_det, make_cfg, no_cars, zone_health  # noqa: E402

UNIFORM = np.full((H, W, 3), 128, dtype=np.uint8)


def monitor_health(events: List[Event]) -> List[Event]:
    return [e for e in events if e.type == "HEALTH" and e.extra.get("component") == "monitor"]


def first(results, pred) -> int:
    return next(r.frame_index for r in results if pred(r))


def debris_health(events: List[Event]) -> List[Event]:
    return [e for e in events if e.type == "HEALTH" and e.extra.get("component") == "debris"]


def test_debris_warm_up_is_degraded_but_reported_once_by_the_debris_component():
    """FO9: every run used to open with four HEALTH events for one fact (debris inactive,
    monitor degraded: debris warming up, debris active, monitor no longer degraded). The
    warm-up is still DEGRADED on every frame (FrameResult, status line, overlay), but only
    the debris component reports it: its start and its end."""
    s = Session(make_cfg(debris={"warmup_frames": 60}))
    res = s.run(150)
    ok = first(res, lambda r: not r.degraded)
    assert 55 <= ok <= 65
    assert all(r.degraded_reasons == ("debris warming up",) for r in res[:ok])
    assert not any(r.degraded for r in res[ok:])
    assert monitor_health(s.events) == []
    texts = [e.reasons[0] for e in debris_health(s.events)]
    assert len(texts) == 2 and texts[0].startswith("debris detection inactive until the background warm-up")
    assert texts[1].startswith("debris detection active (warm-up complete")
    # status line and overlay show it
    assert "CLEAR   DEGRADED(debris warming up) objs=" in Monitor._status_line(res[5], None)
    assert "DEGRADED" not in Monitor._status_line(res[-1], None)
    frame = s.mon.last_frame
    shown = draw_overlay(frame, res[5])
    hidden = draw_overlay(frame, dataclasses.replace(res[5], degraded=False, degraded_reasons=()))
    assert np.any(shown != hidden)
    # another reason during the warm-up is reported, without the warm-up
    s2 = Session(make_cfg(debris={"warmup_frames": 60}), infra=FlakyInfra(range(0, 45)))
    res2 = s2.run(100)
    assert res2[10].degraded_reasons == ("debris warming up", "gate detector failing")
    events = monitor_health(s2.events)
    assert [(e.frame_index, e.extra["degraded"], e.extra["reasons"]) for e in events] == [
        (30, True, ["gate detector failing"]),  # failing since frame 0, reported once it lasted 1.0 s
        (75, False, []),
    ]
    assert events[0].reasons == ["monitor degraded for 1.00 s: gate detector failing"]
    assert events[1].reasons == ["monitor no longer degraded for 1.00 s"]
    assert events[0] in res2[30].events  # also in the FrameResult, like every HEALTH event
    # debounce 0: every change is reported at once
    s0 = Session(make_cfg(debris={"enabled": False}, alerts={"health_debounce_seconds": 0.0}),
                 infra=FlakyInfra(range(10, 20)))
    s0.run(40)
    assert [(e.frame_index, e.extra["degraded"]) for e in monitor_health(s0.events)] == [(10, True), (20, False)]
    with pytest.raises(ValueError, match="health_debounce_seconds"):
        Monitor(make_cfg(alerts={"health_debounce_seconds": -1.0}), detector=object())


def test_debris_trust_flicker_is_debounced():
    """FO7 (flip_probe.py on clip03): a registration that fails every other frame made 30
    debris HEALTH events per second ('untrusted: held (registration)' / 'trusted again'),
    enough to push queued WARNING / RUN events out of a full webhook queue. A trust change
    is now reported once it has lasted alerts.health_debounce_seconds, like DEGRADED."""

    def run(debounce: float, failing) -> List[Event]:
        s = Session(make_cfg(debris={"warmup_frames": 15}, alerts={"health_debounce_seconds": debounce}))
        orig = s.mon._debris_registration
        s.mon._debris_registration = (  # type: ignore[assignment]
            lambda zone: (None, False) if failing(s.i) else orig(zone))
        s.run(240)
        return [e for e in debris_health(s.events) if "trusted" in e.extra and "registration_ok" in e.extra]

    flicker = lambda i: 30 <= i < 120 and i % 2 == 0  # noqa: E731 - 3 s of every other frame
    assert len(run(0.0, flicker)) == 90  # undebounced: one event per change
    assert run(1.0, flicker) == []
    sustained = run(1.0, lambda i: 60 <= i < 120)  # 2 s without registration
    assert [(e.frame_index, e.extra["trusted"]) for e in sustained] == [(90, False), (150, True)]
    assert sustained[0].reasons[0] == "debris detection untrusted for 1.00 s: held (registration)"
    assert sustained[1].reasons[0].startswith("debris detection trusted again for 1.00 s")


def test_zone_held_is_degraded_and_left_to_the_zone_events():
    for zone_events in (True, False):
        s = Session(make_cfg(debris={"enabled": False}, zone={"health_events": zone_events}))
        s.run(5)
        res = []
        for i in range(5, 5 + 45):  # 1.5 s of frames the stabiliser cannot use
            s.fake.index = i
            res.append(s.mon.process_frame(UNIFORM, i / FPS, i))
        assert all(r.degraded_reasons == ("zone held",) for r in res)
        line = Monitor._status_line(res[-1], None)
        assert "DEGRADED(zone held)" in line and "zone=held/" in line and " HELD " in line
        mon_events = monitor_health(s.events)
        if zone_events:
            assert len(zone_health(s.events)) == 1  # the zone reports itself...
            assert mon_events == []  # ...and is not reported twice
        else:
            assert zone_health(s.events) == []
            assert [(e.frame_index, e.extra["reasons"]) for e in mon_events] == [(5 + 30, ["zone held"])]


def test_uninitialised_zone_is_shown_as_such():
    s = Session(make_cfg(debris={"enabled": False}))
    r = s.mon.process_frame(UNIFORM, 0.0, 0)
    assert r.zone.method == "uninitialized" and r.degraded_reasons == ("zone uninitialised",)
    assert "NOT ANCHORED" in Monitor._status_line(r, None) and "HELD" not in Monitor._status_line(r, None)


def test_train_hold_of_the_debris_backstop_is_degraded():
    train = np.array([200.0, 150.0, 440.0, 270.0], np.float32)

    def script(i: int) -> List[Detection]:
        if 60 <= i < 120:
            return [Detection(xyxy=train.copy(), conf=0.9, cls_id=6, cls_name="train", track_id=50)]
        return []

    s = Session(make_cfg(debris={"warmup_frames": 15}), script)  # warm-up shorter than the debounce
    res = s.run(260)
    held = [r.frame_index for r in res if "debris held (train)" in r.degraded_reasons]
    assert held[0] == 60 and held[-1] >= 120  # the hold lingers after the train (debris.hold_linger_seconds)
    events = monitor_health(s.events)
    assert [(e.frame_index, e.extra["degraded"]) for e in events] == [(90, True), (held[-1] + 31, False)]
    assert events[0].extra["reasons"] == ["debris held (train)"]


class FlakyInfra(FakeInfra):
    """Gate detector that raises on the frames in ``bad``."""

    def __init__(self, bad) -> None:
        super().__init__(down=False)
        self.bad = set(bad)
        self.i = 0

    def update(self, frame_bgr, timestamp, frame_index) -> GateState:
        if frame_index in self.bad:
            raise RuntimeError("gate model crashed")
        return super().update(frame_bgr, timestamp, frame_index)


def test_failing_gate_detector_costs_no_frame_and_is_degraded(caplog):
    infra = FlakyInfra(list(range(10, 20)) + list(range(40, 100)))
    s = Session(make_cfg(debris={"enabled": False}), infra=infra)
    with caplog.at_level(logging.ERROR, logger="gradecrossing.monitor"):
        res = s.run(150)  # never raises
    failing = [r.frame_index for r in res if r.degraded_reasons == ("gate detector failing",)]
    assert failing == list(range(10, 20)) + list(range(40, 100))
    assert all(not r.gate.available for r in res if r.frame_index in infra.bad)
    assert all(r.gate.available for r in res if r.frame_index not in infra.bad)
    # the 10-frame failure is shorter than the debounce; the 2 s one is reported, both ways
    assert [(e.frame_index, e.extra["degraded"]) for e in monitor_health(s.events)] == [(70, True), (130, False)]
    assert len([r for r in caplog.records if "gate detector failed" in r.getMessage()]) == 2  # once per streak


def test_detector_slower_than_the_still_gap_is_degraded():
    script = lambda i: [car_det(320.0, 230.0)]  # noqa: E731
    s = Session(make_cfg(debris={"enabled": False}, detector={"detect_every_n_frames": 40}), script)
    res = s.run(120)
    slow = [r.frame_index for r in res if r.degraded_reasons == ("detector interval > still gap",)]
    assert slow == list(range(40, 120))  # known from the second detector run on
    assert [(e.frame_index, e.extra["degraded"]) for e in monitor_health(s.events)] == [(70, True)]
    assert len([e for e in s.events if e.type == "HEALTH" and "detection_gap_s" in e.extra]) == 1
    # the speed rule alone does not need the still clock
    s2 = Session(make_cfg(debris={"enabled": False}, detector={"detect_every_n_frames": 40},
                          stuck={"stationarity": "speed"}), script)
    assert not any(r.degraded for r in s2.run(120))


def test_reported_state_survives_reset():
    s = Session(make_cfg(debris={"enabled": False}), infra=FlakyInfra(range(0, 200)))
    s.run(45)
    assert [e.extra["degraded"] for e in monitor_health(s.events)] == [True]
    s.mon.reset()  # still failing afterwards: the receivers were already told
    res = s.run(60)
    assert all(r.degraded for r in res[45:]) and len(monitor_health(s.events)) == 1
