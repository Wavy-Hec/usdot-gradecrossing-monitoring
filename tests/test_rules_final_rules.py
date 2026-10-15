"""Decision-rule fixes from the final review (digest/final: MA1, MA3, MA4).

* MA1: a train covers a stuck car gradually. Its box is cut short first, which restarts its still
  and speed clocks, so only the latch holds the ALARM by the time the car is lost under the train.
  The tracker now gets the ALARM episode's ids (latched included) as "alarmed", so the car is kept
  as a hidden hazard and the ALARM holds until occlusion_memory_seconds after the train
  (work/rev-missed-alarms/hidden_probe.py 'partial': ALARM_END 18.6 s before, CLEAR from 20 s with
  the car under the train until 30 s). An episode vehicle that has driven off is not hidden.
* MA3: an alarmed car whose box merges with the arriving train is labelled 'train' for longer than
  label_hold_seconds (clip07's car 29 at 4.4 s). It keeps its vehicle category while it is in the
  ALARM episode and in or near the zone (hidden_probe_long.py 'relabel': ALARM_END 19.0 s before).
* MA4: a box cut by the image border restarts its still cluster's clock only with a train box beside
  it, so a vehicle stopped at the edge of a full-width zone alarms at alarm_seconds (the clip01 SUV
  moved to the right edge: ALARM 12.35 s before, 5.42 s now, as with still_border_margin 0).

Synthetic detections at 30 fps; the tracker, the alert engine and (for the wiring) the Monitor are
the real ones.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pytest

from gradecrossing.alerting import AlertEngine, CallbackSink
from gradecrossing.config import AlertConfig, MonitorConfig, StuckConfig
from gradecrossing.geometry import polygon_mask
from gradecrossing.monitor import ConfigClassifier, Monitor
from gradecrossing.tracking import StuckVehicleTracker
from gradecrossing.types import Detection, Event, GateState, ObjectState, Status, ZoneState

FPS = 30.0
DT = 1.0 / FPS
CLASSIFY = ConfigClassifier(MonitorConfig().detector)
BAND = [[0, 190], [640, 190], [640, 260], [0, 260]]  # a zone across the whole picture, like clip01-06
CAR = np.array([300, 200, 380, 250], np.float32)
T_TRAIN = 8.0  # the train front reaches the car's left edge
T_TRAIN_END = 30.0
# After the last train box: the train track lives lost_track_grace_seconds (1 s), the hidden hazard
# occlusion_memory_seconds (3 s), then the latch (10 s) runs out.
HELD_UNTIL = T_TRAIN_END + 1.0 + 3.0 + 10.0


def zone_state(poly=BAND) -> ZoneState:
    p = np.asarray(poly, np.int32)
    return ZoneState(polygon=p, mask=polygon_mask((360, 640), p), homography=None, inliers=100,
                     method="identity", healthy=True)


def det(b: Sequence[float], tid: Optional[int] = 1, cls: str = "car") -> Detection:
    return Detection(xyxy=np.asarray(b, np.float32), conf=0.9, cls_id=2, cls_name=cls, track_id=tid)


def cover_dets(t: float, mode: str, hide_at: float) -> List[Detection]:
    """A car stopped in the zone from t = 0; a train box arrives from the left at T_TRAIN and covers
    it until T_TRAIN_END; the car is not detected from ``hide_at``. ``mode`` 'partial': the visible
    car box shrinks as the train front advances over it (what YOLO returns for a partly hidden car);
    'relabel': the car box keeps its size but is labelled 'train' from T_TRAIN."""
    out = []
    if t < hide_at:
        b = CAR.copy()
        if mode == "partial" and t >= T_TRAIN:
            b[0] = CAR[0] + (t - T_TRAIN) / (hide_at - T_TRAIN) * 70.0
        out.append(det(b, 1, "train" if mode == "relabel" and t >= T_TRAIN else "car"))
    if T_TRAIN <= t < T_TRAIN_END:
        front = CAR[0] + (t - T_TRAIN) / (hide_at - T_TRAIN) * (CAR[2] - CAR[0]) if t < hide_at else 639.0
        out.append(det([0, 120, min(front, 639.0), 260], 50, "train"))
    return out


def cover_run(mode: str, hide_at: float, raw_ids: bool = False, seconds: float = 50.0, **stuck):
    """Tracker + engine wired like Monitor.process_frame. ``raw_ids``: feed the tracker the raw ALARM
    ids of the last frame (the behaviour before the final review)."""
    cfg = StuckConfig(**stuck)
    tr = StuckVehicleTracker(cfg, CLASSIFY)
    eng = AlertEngine(cfg, AlertConfig(stdout=False, repeat_notify_seconds=0.0))
    events: List[Event] = []
    eng.add_sink(CallbackSink(events.append))
    z = zone_state()
    log = []
    for i in range(int(seconds * FPS)):
        t = i * DT
        alarmed = eng.alarm_track_ids if raw_ids else eng.alarm_episode_track_ids
        objs = tr.update(cover_dets(t, mode, hide_at), z, t, alarmed_ids=alarmed)
        a, _ = eng.evaluate(objs, None, GateState(), t, i)
        log.append((t, a, {o.track_id: (o.is_vehicle, o.is_train, o.occluded, o.cls_name) for o in objs}))
    return events, log


def alarm_spans(events: Sequence[Event]) -> List[Tuple[str, float]]:
    return [(e.type, round(e.timestamp, 2)) for e in events if e.type in ("ALARM_START", "ALARM_END")]


# ----------------------------------------------------------------------------- MA1
def test_alarm_episode_track_ids_cover_the_latched_alarm():
    eng = AlertEngine(StuckConfig(), AlertConfig(stdout=False, latch_seconds=10.0, repeat_notify_seconds=0.0))
    assert eng.alarm_episode_track_ids == []

    def car(tid: int, stuck: float) -> ObjectState:
        return ObjectState(track_id=tid, cls_name="car", is_vehicle=True, is_person=False, is_train=False,
                           last_box=np.array([0, 0, 10, 10], np.float32), last_seen=0.0, first_seen=0.0,
                           in_zone=True, in_zone_since=0.0, dwell_seconds=stuck, speed=0.0, speed_units="px/s",
                           stationary_seconds=stuck, stuck_seconds=stuck, history=[])

    eng.evaluate([car(4, 1.6)], None, GateState(), 1.6, 48)  # WARNING: not alarmed
    assert eng.status is Status.WARNING and eng.alarm_episode_track_ids == []
    eng.evaluate([car(4, 3.0), car(7, 3.1)], None, GateState(), 3.0, 90)
    assert eng.alarm_episode_track_ids == [4, 7]
    eng.evaluate([car(7, 3.2)], None, GateState(), 3.1, 93)
    assert eng.alarm_track_ids == [7] and eng.alarm_episode_track_ids == [7, 4]
    eng.evaluate([], None, GateState(), 5.0, 150)  # latched: no raw ids, the episode's ids stay
    assert eng.status is Status.ALARM and eng.alarm_track_ids == [] and eng.alarm_episode_track_ids == [4, 7]
    eng.evaluate([], None, GateState(), 13.2, 396)  # the latch ran out
    assert eng.status is Status.CLEAR and eng.alarm_episode_track_ids == []
    eng.evaluate([car(9, 3.0)], None, GateState(), 20.0, 600)
    eng.reset()
    assert eng.alarm_episode_track_ids == []


def test_a_car_the_train_covers_gradually_stays_an_alarm_until_the_train_has_gone():
    events, log = cover_run("partial", hide_at=9.0)
    assert alarm_spans(events) == [("ALARM_START", round(3.0 + DT, 2)), ("ALARM_END", pytest.approx(HELD_UNTIL, abs=0.1))]
    # the raw ALARM ids went empty while the box was cut short; the car was hidden nonetheless
    assert any(not a.alarm_track_ids for t, a, _ in log if T_TRAIN <= t < 9.0)
    covered = [(t, a, objs) for t, a, objs in log if 10.0 <= t < T_TRAIN_END + 4.0]
    assert all(a.status is Status.ALARM and any("occluded by train" in r for r in a.reasons) for _, a, _ in covered)
    assert all(objs[1][2] for _, _, objs in covered)
    # before the fix: the car was dropped, the ALARM ended with the latch, CLEAR under the train
    events_raw, log_raw = cover_run("partial", hide_at=9.0, raw_ids=True)
    ends = [ts for ty, ts in alarm_spans(events_raw) if ty == "ALARM_END"]
    assert ends and ends[0] < 20.0
    assert all(a.status is Status.CLEAR for t, a, _ in log_raw if 20.0 <= t < T_TRAIN_END)


class _CoverDetector:
    """The 'partial' scenario as a detector for the Monitor."""

    def __init__(self) -> None:
        self.t = 0.0

    def detect(self, frame) -> List[Detection]:
        return cover_dets(self.t, "partial", 9.0)

    def classify(self, name: str) -> str:
        return CLASSIFY(name)


def test_the_monitor_gives_the_tracker_the_alarm_episode():
    """The wiring: Monitor.process_frame passes AlertEngine.alarm_episode_track_ids to the tracker."""
    cfg = MonitorConfig()
    cfg.zone.polygon = [list(p) for p in BAND]
    cfg.zone.stabilize = False
    cfg.debris.enabled = False
    cfg.alerts.stdout = False
    cfg.alerts.repeat_notify_seconds = 0.0
    events: List[Event] = []
    fake = _CoverDetector()
    mon = Monitor(cfg, detector=fake, sinks=[CallbackSink(events.append)])
    frame = np.random.default_rng(0).integers(0, 255, (360, 640, 3), dtype=np.uint8)
    status: Dict[int, Status] = {}
    for i in range(int(46 * FPS)):
        t = i / FPS
        fake.t = t
        r = mon.process_frame(frame, t, i)
        if i % int(FPS) == 0:
            status[int(round(t))] = r.assessment.status
    assert all(status[s] is Status.ALARM for s in range(4, 44))
    assert alarm_spans(events)[-1] == ("ALARM_END", pytest.approx(HELD_UNTIL, abs=0.1))


def test_an_episode_vehicle_that_drove_off_is_not_kept_behind_a_train():
    """The episode's ids include a vehicle that has left the zone; lost under a train box far from
    the zone it is forgotten as before (only vehicles last seen in or near the zone are hidden)."""
    cfg = StuckConfig()
    tr = StuckVehicleTracker(cfg, CLASSIFY)
    eng = AlertEngine(cfg, AlertConfig(stdout=False, repeat_notify_seconds=0.0))
    events: List[Event] = []
    eng.add_sink(CallbackSink(events.append))
    z = zone_state()
    occluded = False
    for i in range(int(30 * FPS)):
        t = i * DT
        dets = []
        if t < 9.0:
            dy = 60.0 * max(0.0, t - 6.0)  # stuck until 6 s, then drives up and away from the band
            dets.append(det(CAR - np.array([0, dy, 0, dy], np.float32), 1))
        if 8.5 <= t < 25.0:
            dets.append(det([0, 0, 640, 120], 50, "train"))
        objs = tr.update(dets, z, t, alarmed_ids=eng.alarm_episode_track_ids)
        if 8.9 <= t < 9.0:
            assert 1 in eng.alarm_episode_track_ids and eng.alarm_track_ids == []  # latched, still "alarmed"
            assert not objs[0].in_zone and not objs[0].near_zone
        a, _ = eng.evaluate(objs, None, GateState(), t, i)
        occluded = occluded or any(o.occluded for o in objs)
    assert not occluded
    ends = [ts for ty, ts in alarm_spans(events) if ty == "ALARM_END"]
    assert ends and ends[0] < 18.0  # the latch after the car left, not the train passage


# ----------------------------------------------------------------------------- MA3
def test_an_alarmed_car_labelled_train_as_the_train_arrives_stays_a_vehicle():
    events, log = cover_run("relabel", hide_at=10.5)
    assert alarm_spans(events) == [("ALARM_START", round(3.0 + DT, 2)), ("ALARM_END", pytest.approx(HELD_UNTIL, abs=0.1))]
    merged = [objs[1] for t, _, objs in log if T_TRAIN <= t < 10.5]
    assert all(v and not trn and name == "car" for v, trn, _, name in merged)  # category and label kept
    # its stuck clock never stopped: the raw ALARM went on until it was hidden
    assert all(a.alarm_track_ids == [1] for t, a, _ in log if 3.1 <= t < 10.5)
    assert all(a.status is Status.ALARM for t, a, _ in log if 3.1 <= t < HELD_UNTIL - 0.1)
    # with label_hold_seconds 0 every frame follows the label: a 'train', the ALARM ends with the latch
    events_off, log_off = cover_run("relabel", hide_at=10.5, label_hold_seconds=0.0)
    assert log_off[int(T_TRAIN * FPS) + 1][2][1][1]  # is_train
    ends = [ts for ty, ts in alarm_spans(events_off) if ty == "ALARM_END"]
    assert ends and ends[0] < 20.0


def test_a_warned_car_labelled_train_is_not_held_beyond_the_label_hold():
    """Only the ALARM episode's vehicles are held: a car at WARNING whose label runs 'train' for
    2 s becomes a train after label_hold_seconds, as before (and, a train, it raises nothing)."""
    tr = StuckVehicleTracker(StuckConfig(), CLASSIFY)
    eng = AlertEngine(StuckConfig(), AlertConfig(stdout=False, repeat_notify_seconds=0.0))
    z = zone_state()
    first_train = None
    for i in range(int(3.7 * FPS)):
        t = i * DT
        name = "train" if 1.8 <= t < 3.8 else "car"
        objs = tr.update([det(CAR, 1, name)], z, t, alarmed_ids=eng.alarm_episode_track_ids)
        a, _ = eng.evaluate(objs, None, GateState(), t, i)
        assert a.status is not Status.ALARM
        if objs[0].is_train and first_train is None:
            first_train = t
    assert first_train == pytest.approx(2.8, abs=2 * DT)


def test_an_alarmed_car_labelled_train_that_leaves_the_zone_follows_its_label():
    """The hold ends when the vehicle is neither in nor near the zone: a track that drives off
    labelled 'train' is a train again."""
    tr = StuckVehicleTracker(StuckConfig(), CLASSIFY)
    eng = AlertEngine(StuckConfig(), AlertConfig(stdout=False, repeat_notify_seconds=0.0))
    z = zone_state()
    trace = []
    for i in range(int(12 * FPS)):
        t = i * DT
        dy = 60.0 * max(0.0, t - 6.0)
        objs = tr.update([det(CAR - np.array([0, dy, 0, dy], np.float32), 1, "train" if t >= 5.0 else "car")], z, t,
                         alarmed_ids=eng.alarm_episode_track_ids)
        eng.evaluate(objs, None, GateState(), t, i)
        o = objs[0]
        trace.append((t, o.is_train, o.in_zone or o.near_zone))
    assert not any(trn for t, trn, _ in trace if t < 7.0)  # in the episode and in the zone: held
    first_train = next(t for t, trn, _ in trace if trn)
    last_on = max(t for t, _, on in trace if on)
    assert first_train == pytest.approx(last_on + DT, abs=DT)


# ----------------------------------------------------------------------------- MA4
def edge_stop(train_until: Optional[float] = None, seconds: float = 14.0, **stuck):
    """A car stopped at the right image edge inside a full-width zone; its box breathes like the
    clip06 'bus' (left edge +-8 px, bottom +-6 px), so the speed rule restarts and only the still
    clock can alarm. A train box beside it (gap 5 px, same band) until ``train_until``. Returns the
    first ALARM time and the (t, still) trace."""
    cfg = StuckConfig(**stuck)
    tr = StuckVehicleTracker(cfg, CLASSIFY)
    eng = AlertEngine(cfg, AlertConfig(stdout=False, repeat_notify_seconds=0.0))
    z = zone_state()
    first, trace = None, []
    for i in range(int(seconds * FPS)):
        t = i * DT
        x1 = 540.0 + 8.0 * math.sin(2 * math.pi * t / 2.0)
        y2 = 250.0 + 6.0 * math.sin(2 * math.pi * t / 1.7 + 1.0)
        dets = [det([x1, 200.0, 640.0, y2], 7, "car")]
        if train_until is not None and t < train_until:
            dets.append(det([0.0, 130.0, 530.0, 255.0], 50, "train"))
        objs = {o.track_id: o for o in tr.update(dets, z, t)}
        trace.append((t, objs[7].still_seconds))
        a, _ = eng.evaluate(list(objs.values()), None, GateState(), t, i)
        if a.status is Status.ALARM and first is None:
            first = t
    return first, trace


def test_a_car_stopped_at_the_image_edge_of_a_full_width_zone_alarms_on_time():
    first, trace = edge_stop()
    first_m0, trace_m0 = edge_stop(still_border_margin=0.0)
    assert first == pytest.approx(3.0 + DT, abs=DT) and first == first_m0
    assert trace == trace_m0  # no train beside it: the border rule does nothing


def test_a_border_box_beside_a_train_restarts_its_still_clock_until_the_train_is_gone():
    """With a train box beside it the edge box is the clip06 case: no still time while a train box
    was seen within 2 s, then the clock runs from the end of that window."""
    first, trace = edge_stop(train_until=99.0)
    assert first is None and max(s for _, s in trace) == 0.0
    first, trace = edge_stop(train_until=6.0)
    last_window = 6.0 - DT + 2.0  # the last frame that saw a train box within 2 s
    assert max(s for t, s in trace if t <= last_window + 1e-6) == 0.0
    assert first == pytest.approx(last_window + 3.0 + DT, abs=2 * DT)
