"""Regression tests for the tracker robustness fixes that came with the YOLO-geometry port.

* sparse detection never ages a track out, and one missed detector run does not drop it
  (ours-correctness F03; impl review issue 1);
* id re-keying: a vehicle box adopts only a stopped vehicle track, a person box only a person
  track, and the clocks go back to the old id (F10, corrected; impl review issue 4);
* a clean id switch the re-keying refuses keeps the in-zone streak (cluster streak + min age);
* camera motion is removed from speed and still clusters via ZoneState.to_reference (M2);
* an alarmed vehicle hidden by a train stays a hazard (M3; only alarmed vehicles, F8 corrected);
* a short label flicker does not demote a stopped in-zone vehicle (F17 / F9, corrected).

Synthetic detections at 30 fps unless stated; the alert engine is the real one.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pytest

from gradecrossing.alerting import AlertEngine, CallbackSink
from gradecrossing.config import AlertConfig, MonitorConfig, StuckConfig
from gradecrossing.geometry import polygon_mask
from gradecrossing.monitor import ConfigClassifier
from gradecrossing.tracking import StuckVehicleTracker
from gradecrossing.types import Detection, Event, GateState, ObjectState, Status, ZoneState

from test_monitor import Session, car_det, make_cfg  # noqa: E402

FPS = 30.0
DT = 1.0 / FPS
CLASSIFY = ConfigClassifier(MonitorConfig().detector)
POLY = [[200, 150], [440, 150], [440, 270], [200, 270]]


def zone_state(poly=POLY, offset: Tuple[float, float] = (0.0, 0.0), to_reference: bool = False,
               healthy: bool = True) -> ZoneState:
    p = np.asarray(poly, np.float32) + np.asarray(offset, np.float32)
    zs = ZoneState(polygon=np.round(p).astype(np.int32), mask=polygon_mask((360, 640), p), homography=None,
                   inliers=100, method="lk", healthy=healthy)
    if to_reference:
        # current-frame pixels -> reference-frame pixels (the contract of ZoneState.to_reference)
        zs.to_reference = np.array([[1, 0, -offset[0]], [0, 1, -offset[1]], [0, 0, 1]], np.float64)  # type: ignore[attr-defined]
    return zs


def det(b: Sequence[float], tid: Optional[int] = 1, cls: str = "car") -> Detection:
    return Detection(xyxy=np.asarray(b, np.float32), conf=0.9, cls_id=2, cls_name=cls, track_id=tid)


CAR = np.array([300, 210, 360, 250], np.float32)  # bottom-center (330, 250), inside POLY


def tracker(**stuck) -> StuckVehicleTracker:
    return StuckVehicleTracker(StuckConfig(**stuck), CLASSIFY)


def engine(**stuck) -> AlertEngine:
    return AlertEngine(StuckConfig(**stuck), AlertConfig(stdout=False, repeat_notify_seconds=0.0))


def by_id(objs: Sequence[ObjectState]) -> Dict[int, ObjectState]:
    return {o.track_id: o for o in objs}


# ----------------------------------------------------------------------------- F03 sparse detection
def test_prediction_frames_never_age_a_track_out():
    """Detector every 7th frame at 5 fps (one run every 1.4 s > the 1 s grace): the parked car
    must survive the prediction-only frames between runs and keep its clocks."""
    tr, z = tracker(), zone_state()
    alive, stuck = [], []
    for i in range(int(20 * 5)):
        t = i / 5.0
        ran = i % 7 == 0
        dets = [det(CAR)] if ran else tr.predict_boxes(t)
        objs = tr.update(dets, z, t, detector_ran=ran)
        alive.append(bool(objs))
        stuck.append(objs[0].stuck_seconds if objs else 0.0)
    assert all(alive)
    assert max(stuck) > 10.0
    # a car that really left (last detected at frame 98) survives the first detector run that
    # misses it and is dropped at the second (lost_track_missed_runs = 2; both runs > grace)
    runs = 0
    for i in range(100, 120):
        t = i / 5.0
        ran = i % 7 == 0
        objs = tr.update([] if ran else tr.predict_boxes(t), z, t, detector_ran=ran)
        runs += int(ran)
        if runs < 2:
            assert objs, i
        else:
            assert objs == [] and i == 112
            break
    else:
        pytest.fail("the lost car was never dropped")


def sparse_run(missed_runs: Sequence[int], **stuck) -> Tuple[Optional[float], float, int]:
    """review issue 1: a parked car at 5 fps, detector every 7th frame (runs 1.4 s apart, longer
    than the 1 s grace); the runs listed in ``missed_runs`` do not detect it. Returns the first
    ALARM time, the largest stuck time and how many times the track was dropped."""
    tr, eng = tracker(**stuck), engine(**stuck)
    first, worst, drops, had = None, 0.0, 0, False
    for i in range(int(40 * 5)):
        t = i / 5.0
        ran = i % 7 == 0
        if ran:
            dets = [] if i // 7 in missed_runs else [det(CAR)]
        else:
            dets = tr.predict_boxes(t)
        objs = tr.update(dets, zone_state(), t, detector_ran=ran, alarmed_ids=eng.alarm_track_ids)
        drops += int(had and not objs)
        had = bool(objs)
        a, _ = eng.evaluate(objs, None, GateState(), t, i)
        worst = max([worst] + [o.stuck_seconds for o in objs])
        if a.status is Status.ALARM and first is None:
            first = t
    return first, worst, drops


def test_one_missed_detector_run_does_not_restart_a_parked_car():
    """With runs 1.4 s apart, f1aee43 dropped the parked car at the first run that missed it
    (1.4 s > the 1 s grace): one miss delayed the ALARM 5.8 -> 12.8 s, one miss in every four
    runs delayed it to 39.4 s. Two missed runs in a row are needed now."""
    first, worst, drops = sparse_run(())
    assert first == pytest.approx(5.8) and drops == 0
    assert sparse_run((4,)) == (first, worst, 0)  # the miss changes nothing
    first_q, worst_q, drops_q = sparse_run((3, 7, 11, 15, 19, 23))
    assert first_q == pytest.approx(first) and drops_q == 0 and worst_q == pytest.approx(worst)
    # two misses in a row (before the ALARM): dropped, and a new track restarts the clocks
    first_2, _, drops_2 = sparse_run((2, 3))
    assert drops_2 == 1 and first_2 is not None and first_2 > first + 4.0
    # lost_track_missed_runs = 1 is the previous behaviour
    first_1, _, drops_1 = sparse_run((4,), lost_track_missed_runs=1)
    assert drops_1 == 1 and first_1 == pytest.approx(12.8)


def test_with_every_frame_detected_the_grace_still_decides():
    """At 30 fps with the detector on every frame the run count never matters: kept for 1 s of
    misses (30 runs), dropped after."""
    tr, z = tracker(), zone_state()
    for i in range(30):
        tr.update([det(CAR)], z, i * DT)
    t_last = 29 * DT
    k = 1
    while tr.update([], z, t_last + k * DT):
        k += 1
    assert k * DT > 1.0 and (k - 1) * DT <= 1.0 + 1e-6  # dropped on the first frame past the grace


def test_a_take_over_is_remembered_across_sparse_detector_runs():
    """At 5 fps / every 7th frame a truck is first detected with a new id on top of the stopped,
    hidden car; the car's own id is back one run (1.4 s, more than the grace) later. The clocks
    must still go back to the car."""
    tr, z = tracker(), zone_state()
    i = 0
    while i < 40:  # the car alone, parked since t = 0
        t = i / 5.0
        ran = i % 7 == 0
        tr.update([det(CAR, tid=1)] if ran else tr.predict_boxes(t), z, t, detector_ran=ran)
        i += 1
    stuck_before = by_id(tr.update([det(CAR, tid=1)], z, 42 / 5.0, detector_ran=True))[1].stuck_seconds
    assert stuck_before > 5.0
    for i in range(43, 49):
        tr.update(tr.predict_boxes(i / 5.0), z, i / 5.0, detector_ran=False)
    objs = by_id(tr.update([det(CAR + np.array([-5, -8, 5, 4], np.float32), tid=7, cls="truck")], z, 49 / 5.0,
                           detector_ran=True))
    assert set(objs) == {7}  # the truck took the car's (unclaimed) track over
    for i in range(50, 56):
        tr.update(tr.predict_boxes(i / 5.0), z, i / 5.0, detector_ran=False)
    objs = by_id(tr.update([det(CAR, tid=1)], z, 56 / 5.0, detector_ran=True))
    assert 1 in objs and objs[1].stuck_seconds == pytest.approx(stuck_before + 14 / 5.0, abs=0.05)


def test_grace_comparison_has_an_epsilon():
    """6/5 + 1.0 - 6/5 = 1.0000000000000002 > 1.0: float noise must not decide a drop."""
    tr, z = tracker(), zone_state()
    t1, t2 = 6 / 5.0, (6 + 5) / 5.0
    assert t2 - t1 > 1.0
    tr.update([det(CAR)], z, t1)
    assert tr.update([], z, t2)  # missed exactly lost_track_grace_seconds later: still live
    assert tr.update([], z, t2 + 0.2) == []


def test_monitor_with_sparse_detection_at_5fps_still_alarms():
    """ours-correctness F03: 5 fps, detect_every_n_frames=7 (a plausible Jetson setting) used to
    drop the parked car on every prediction frame, so nothing ever alarmed."""
    cfg = make_cfg(detector={"detect_every_n_frames": 7}, debris={"enabled": False},
                   stuck={"warning_seconds": 1.5, "alarm_seconds": 3.0})
    s = Session(cfg, lambda i: [car_det(320.0, 230.0)] if i >= 5 else [])
    for i in range(int(20 * 5)):
        s.fake.index = i
        s.mon.process_frame(s.frame(i), i / 5.0, i)
    starts = s.events_of("ALARM_START")
    assert starts and starts[0].timestamp <= 8.0
    assert any(e.type == "HEALTH" and "detection_gap_s" in e.extra for e in s.events)  # still clock is off


# ----------------------------------------------------------------------------- F10 re-keying
def truck_box(t: float) -> np.ndarray:
    # 100x70 truck driving left->right at 300 px/s; its first box is centred on the hidden car
    x = 330 + (t - 6.0) * 300.0
    return np.array([x - 50, 192, x + 50, 262], np.float32)


def steal_run(policy: str, stationarity: str = "speed_or_still", alarm_seconds: float = 8.0):
    """ours-correctness probe_rekey_steal: car 1 stops at 1 s; at 6.0 s a truck passing in front
    is first detected with a new id on top of it (IoU 0.34) while the car is occluded until 6.6 s,
    when ByteTrack re-finds the car under its old id."""
    tr = tracker(rekey_policy=policy, stationarity=stationarity)
    eng = engine(rekey_policy=policy, warning_seconds=3.0, alarm_seconds=alarm_seconds)
    first_alarm, stuck_at_7 = None, None
    for i in range(int(20 * FPS)):
        t = i / FPS
        dets = []
        if t >= 1.0 and not 6.0 <= t < 6.6:
            dets.append(det(CAR, tid=1))
        if 6.0 <= t < 7.5:
            dets.append(det(truck_box(t), tid=2, cls="truck"))
        objs = tr.update(dets, zone_state(), t)
        a, _ = eng.evaluate(objs, None, GateState(), t, i)
        if a.status is Status.ALARM and first_alarm is None:
            first_alarm = t
        if abs(t - 7.0) < 1e-9:
            stuck_at_7 = by_id(objs)[1].stuck_seconds
    return first_alarm, stuck_at_7


def test_a_truck_passing_in_front_does_not_keep_a_stopped_cars_clocks():
    for stationarity in ("speed_or_still", "speed"):  # the hand-back works without still clusters too
        first, stuck = steal_run("strict", stationarity)
        assert stuck == pytest.approx(7.0 - 1.0 - DT, abs=0.1), stationarity  # the clocks went back to the car
        assert first == pytest.approx(9.0 + DT, abs=0.1), stationarity  # as if the truck had never been there
    first_any, stuck_any = steal_run("any", "speed")  # the pre-port rule: ALARM 5.6 s late
    assert stuck_any < 1.0 and first_any > 14.0


def test_a_vehicle_box_adopts_only_a_stopped_vehicle_track():
    """Strict policy: a person box never takes a vehicle's track, a vehicle box never a person's,
    and a new id on top of a moving vehicle is a new object."""
    z = zone_state()
    tr = tracker()
    for i in range(int(3 * FPS)):
        tr.update([det(CAR, tid=1)], z, i * DT)
    # a person first detected on top of the stopped car (the car hidden behind them)
    objs = by_id(tr.update([det(CAR, tid=5, cls="person")], z, 3.0))
    assert 1 in objs and 5 in objs and objs[5].stuck_seconds == 0.0
    # nor can a vehicle box take a person's track over
    tr3 = tracker()
    for i in range(int(3 * FPS)):
        tr3.update([det(PERSON, tid=1, cls="person")], z, i * DT)
    assert set(by_id(tr3.update([det(PERSON, tid=6)], z, 3.0))) == {1, 6}
    # a new id on top of a *moving* car is a new object
    tr2 = tracker()
    for i in range(int(2 * FPS)):
        tr2.update([det(CAR + np.array([150 * i * DT, 0, 150 * i * DT, 0], np.float32), tid=1)], z, i * DT)
    t = 2 * FPS * DT
    moved = CAR + np.array([150 * t, 0, 150 * t, 0], np.float32)
    assert set(by_id(tr2.update([det(moved, tid=9)], z, t))) == {1, 9}


PERSON = np.array([320, 180, 340, 240], np.float32)


def person_swap_run(policy: str = "strict", swap_every: Optional[float] = None) -> Optional[float]:
    """review issue 4: a person stands in the zone from t = 0; ByteTrack gives them a new id at
    6 s (or every ``swap_every`` seconds). Person ALARM is 10 s of dwell."""
    tr, eng = tracker(rekey_policy=policy), engine(rekey_policy=policy)
    first = None
    for i in range(int(20 * FPS)):
        t = i * DT
        tid = (1 + int(t // swap_every)) if swap_every else (1 if t < 6.0 else 2)
        objs = tr.update([det(PERSON, tid=tid, cls="person")], zone_state(), t)
        a, _ = eng.evaluate(objs, None, GateState(), t, i)
        if a.status is Status.ALARM and first is None:
            first = t
    return first


def test_an_id_swap_does_not_restart_a_standing_persons_dwell():
    """The strict policy re-keys person -> person (stopped or not: dwell counts walking along
    the tracks too); f1aee43 refused it and the ALARM came at 16 s instead of 10 s, never with
    a swap every 8 s."""
    assert person_swap_run() == pytest.approx(10.0 + DT, abs=DT)
    assert person_swap_run(swap_every=8.0) == pytest.approx(10.0 + DT, abs=DT)
    assert person_swap_run("any") == pytest.approx(10.0 + DT, abs=DT)


def test_a_person_passing_in_front_gives_the_clocks_back():
    """Person 2 walks past person 1 (standing since t = 0) and is first detected on top of them
    while 1 is hidden: 2 takes 1's clocks for a moment, and they go back when 1 is seen again."""
    tr, z = tracker(), zone_state()
    for i in range(int(5 * FPS)):
        tr.update([det(PERSON, tid=1, cls="person")], z, i * DT)
    objs = by_id(tr.update([det(PERSON + np.array([4, 0, 4, 0], np.float32), tid=2, cls="person")], z, 5.0))
    assert set(objs) == {2} and objs[2].dwell_seconds > 4.5  # adopted
    objs = by_id(tr.update([det(PERSON, tid=1, cls="person"),
                            det(PERSON + np.array([30, 0, 30, 0], np.float32), tid=2, cls="person")], z, 5.0 + 5 * DT))
    assert objs[1].dwell_seconds == pytest.approx(5.0 + 5 * DT, abs=0.05)
    assert objs[2].dwell_seconds < 0.5  # the passer-by starts over


def test_the_physical_track_uid_survives_bridged_id_switches():
    """ObjectState.uid follows the physical track through a re-key, a hand-back and a hidden
    hazard re-detection; a genuinely new object gets a new one."""
    z = zone_state()
    tr = tracker()
    for i in range(int(3 * FPS)):
        objs = tr.update([det(CAR, tid=1)], z, i * DT)
    uid = objs[0].uid
    assert uid > 0
    objs = by_id(tr.update([det(CAR, tid=2)], z, 3.0))  # id swap in place: re-keyed
    assert set(objs) == {2} and objs[2].uid == uid
    objs = by_id(tr.update([det(CAR, tid=1)], z, 3.0 + DT))  # the old id is back: clocks handed back
    assert set(objs) == {1} and objs[1].uid == uid
    far = by_id(tr.update([det(CAR, tid=1), det(CAR + 200, tid=9)], z, 3.0 + 2 * DT))
    assert far[9].uid not in (0, uid)
    _, log = train_run()
    t, a, objs = next((t, a, o) for t, a, o in log if t >= 50.5)
    before = next(o for t, a, o in log if t >= 10.0)
    assert by_id(objs)[3].uid == by_id(before)[1].uid  # the hazard re-detected after the train


def test_a_same_car_id_flip_with_a_different_box_size_is_rekeyed():
    """clip06 10.18 s: far-side car 159 -> 295, boxes 21x12 vs 29x18 (area ratio 2.0, IoU 0.46).
    No box-size gate: the stuck clock must survive."""
    small = np.array([400, 200, 421, 212], np.float32)
    big = np.array([397, 197, 426, 215], np.float32)
    z = zone_state([[300, 150], [540, 150], [540, 270], [300, 270]])
    tr = tracker()
    for i in range(int(5 * FPS)):
        tr.update([det(small, tid=159)], z, i * DT)
    objs = by_id(tr.update([det(big, tid=295)], z, 5.0))
    assert set(objs) == {295}
    assert objs[295].stuck_seconds == pytest.approx(5.0 - DT, abs=0.05)
    # ... and the flip back a moment later returns the clocks to the old id
    objs = by_id(tr.update([det(small, tid=159)], z, 5.0 + 16 * DT))
    assert set(objs) == {159} and objs[159].stuck_seconds == pytest.approx(5.0 + 15 * DT, abs=0.05)


def test_a_clean_id_switch_the_rekeying_refuses_keeps_the_in_zone_streak():
    """The re-keying is made to miss (fallback_iou_threshold 0.99 with box jitter): the new id
    takes the still cluster over, and its first frame - not zone evidence by min_track_age_frames
    - must not end the cluster's in-zone streak."""
    rng = np.random.default_rng(3)
    tr, z = tracker(fallback_iou_threshold=0.99), zone_state()
    for i in range(int(6 * FPS) + 1):
        t = i * DT
        b = CAR + rng.uniform(-1.0, 1.0, 4).astype(np.float32)
        objs = by_id(tr.update([det(b, tid=1 if t < 3.0 else 2)], z, t))
    assert 2 in objs
    assert objs[2].first_seen == pytest.approx(3.0, abs=DT)  # really a new track, not re-keyed
    assert objs[2].stuck_seconds == pytest.approx(6.0 - DT, abs=0.05)


# ----------------------------------------------------------------------------- M2 camera motion
def shift_at(t: float) -> Tuple[float, float]:
    """The image (zone + car) moves by (-10, +9) px over 0.27 s at t = 6 s (clip02's bump)."""
    a = min(1.0, max(0.0, (t - 6.0) / 0.27))
    return -10.0 * a, 9.0 * a


def camera_shift_run(to_reference: bool, **stuck) -> List[Tuple[float, float]]:
    rng = np.random.default_rng(5)
    tr = tracker(**stuck)
    out = []
    for i in range(int(14 * FPS) + 1):
        t = i * DT
        off = shift_at(t)
        z = zone_state(offset=off, to_reference=to_reference)
        b = CAR + np.array([off[0], off[1], off[0], off[1]], np.float32) + rng.uniform(-0.3, 0.3, 4).astype(np.float32)
        o = tr.update([det(b)], z, t)[0]
        out.append((t, o.stuck_seconds))
    return out


def test_camera_motion_is_not_vehicle_motion():
    """ours-realfootage-verify M2: a camera bump read as vehicle motion reset a parked car's
    stuck clock (6.24 s -> 0 at 6.41 s). Mapped into reference coordinates, it does not move."""
    with_ref = camera_shift_run(True)
    assert with_ref[-1][1] == pytest.approx(14.0 - DT, abs=0.05)
    raw = camera_shift_run(False)
    assert min(s for t, s in raw if 6.0 < t < 7.0) < 0.5  # the documented failure without the transform
    off = camera_shift_run(True, camera_motion_compensation=False)
    assert off[-1][1] < 8.0  # the switch restores the raw-pixel behaviour


def test_held_zone_keeps_the_last_healthy_transform():
    """An unhealthy (held) frame carries no usable transform: the last healthy one is kept, so the
    whole camera offset is not read as one jump of every vehicle."""
    tr = tracker()
    off = (-10.0, 9.0)
    b = CAR + np.array([off[0], off[1], off[0], off[1]], np.float32)
    for i in range(int(3 * FPS)):
        tr.update([det(b)], zone_state(offset=off, to_reference=True), i * DT)
    for i in range(int(3 * FPS), int(5 * FPS)):
        z = zone_state(offset=off, to_reference=False, healthy=False)  # held: no transform
        o = tr.update([det(b)], z, i * DT)[0]
    assert o.speed < 1.0 and o.stuck_seconds == pytest.approx(5.0 - 2 * DT, abs=0.05)


# ----------------------------------------------------------------------------- M3 hidden hazard
TRAIN = np.array([0, 120, 640, 265], np.float32)


def train_run(car_back_at: Optional[float] = 50.0, stop_at: float = 1.0, **stuck):
    """A car stops in the zone at ``stop_at``; a train box covers it from 20 s to 50 s (the car
    is not detected then); the car is detected again from ``car_back_at`` with a new id 3."""
    tr = tracker(**stuck)
    eng = engine(**stuck)
    events: List[Event] = []
    eng.add_sink(CallbackSink(events.append))
    log = []
    for i in range(int(70 * FPS)):
        t = i / FPS
        dets = []
        if 20.0 <= t < 50.0:
            dets.append(det(TRAIN, tid=2, cls="train"))
        elif t >= stop_at and (t < 20.0 or (car_back_at is not None and t >= car_back_at)):
            dets.append(det(CAR, tid=1 if t < 20.0 else 3))
        objs = tr.update(dets, zone_state(), t, alarmed_ids=eng.alarm_track_ids)
        a, _ = eng.evaluate(objs, None, GateState(), t, i)
        log.append((t, a, [dataclass_copy(o) for o in objs]))
    return events, log


def dataclass_copy(o: ObjectState) -> ObjectState:
    import dataclasses
    return dataclasses.replace(o, last_box=o.last_box.copy(), history=list(o.history))


def test_a_train_passing_in_front_of_an_alarmed_car_does_not_end_the_alarm():
    events, log = train_run()
    types = [(e.type, round(e.timestamp, 2)) for e in events]
    assert [ty for ty, _ in types if ty in ("ALARM_START", "ALARM_END")] == ["ALARM_START"], types
    occluded = [(t, a) for t, a, _ in log if 25.0 <= t < 50.0]
    assert all(a.status is Status.ALARM for _, a in occluded)
    assert all(any("occluded by train" in r for r in a.reasons) for _, a in occluded)
    t, a, objs = next((t, a, o) for t, a, o in log if t >= 30.0)
    assert any(o.occluded and o.track_id == 1 for o in objs)
    # re-detected as id 3 after the train: the car's clocks are back
    t, a, objs = next((t, a, o) for t, a, o in log if t >= 50.5)
    car = by_id(objs)[3]
    assert not car.occluded and car.stuck_seconds > 49.0
    assert a.status is Status.ALARM and a.alarm_track_ids == [3]


def test_a_hidden_hazard_that_is_not_seen_again_expires():
    events, log = train_run(car_back_at=None)
    ends = [e for e in events if e.type == "ALARM_END"]
    # the hazard is held while a train box covers it (the train track itself lives 1 s past its
    # last detection), then occlusion_memory_seconds (3 s), then the latch (10 s) runs out
    assert len(ends) == 1 and ends[0].timestamp == pytest.approx(50.0 + 1.0 + 3.0 + 10.0, abs=0.1)
    assert ends[0].track_ids == [1] and "stationary in zone" in ends[0].reasons[0]


def test_only_alarmed_vehicles_are_remembered_behind_a_train():
    """ours-realfootage F8 (corrected): remembering every vehicle kept the clocks of cars waiting
    at the gates running and added false alarms. A car stopped 1.5 s before the train (WARNING,
    not ALARM) is forgotten as before; with the memory off the alarm ends after the latch."""
    events, log = train_run(stop_at=18.5)
    assert not [e for e in events if e.type == "ALARM_START" and e.timestamp < 50.0]
    assert not any(o.occluded for _, _, objs in log for o in objs)
    t, a, objs = next((t, a, o) for t, a, o in log if t >= 50.5)
    assert by_id(objs)[3].stuck_seconds < 0.6  # seen again after the train: a new object, fresh clocks
    events_off, _ = train_run(occlusion_memory_seconds=0.0)
    ends = [e for e in events_off if e.type == "ALARM_END"]
    assert ends and ends[0].timestamp < 32.0  # the pre-fix behaviour: CLEAR while the car is behind the train


# ----------------------------------------------------------------------------- F17 label hold
def label_run(labels, moving: bool = False, **stuck):
    """A car stopped in the zone (or crossing it at 120 px/s) since t = 0; ``labels(t)`` gives the
    detector's class name at time t."""
    tr, z = tracker(**stuck), zone_state()
    out = []
    for i in range(int(6 * FPS)):
        t = i * DT
        dx = 120.0 * t if moving else 0.0
        b = np.array([150 + dx, 210, 210 + dx, 250], np.float32) if moving else CAR
        o = tr.update([det(b, cls=labels(t))], z, t)[0]
        out.append((t, o.is_vehicle, o.is_train, o.cls_name, o.stuck_seconds))
    return out


def test_a_short_label_flicker_does_not_demote_a_stopped_vehicle():
    """The clip01 SUV was labelled 'train' on 17 detections while its box merged with the driver;
    a stopped, confidently labelled vehicle keeps its category through a run shorter than 1 s."""
    out = label_run(lambda t: "train" if 3.0 <= t < 3.5 else "car")
    assert all(v and not tr for t, v, tr, _, _ in out if t >= 0.0)
    assert out[-1][4] == pytest.approx(6.0 - 2 * DT, abs=0.05)  # the stuck clock was never interrupted
    # a run of 1.5 s is a real label change: it becomes a train after the 1 s hold
    long = label_run(lambda t: "train" if 3.0 <= t < 4.5 else "car")
    flipped = [t for t, v, tr, _, _ in long if tr]
    assert flipped and flipped[0] == pytest.approx(4.0, abs=2 * DT)
    # with the hold off, every frame follows the label (the pre-port behaviour)
    off = label_run(lambda t: "train" if 3.0 <= t < 3.5 else "car", label_hold_seconds=0.0)
    assert any(tr for t, v, tr, _, _ in off if 3.0 <= t < 3.5)


def test_moving_tracks_always_follow_their_label():
    """No vote: a passing train labelled 'car' for a while is a train again on its next 'train'
    frame (clip06/clip07 trains labelled bus/car must not become in-zone vehicles)."""
    out = label_run(lambda t: "car" if t < 1.0 else "train", moving=True)
    first_train = next(t for t, v, tr, _, _ in out if tr)
    assert first_train == pytest.approx(1.0, abs=DT)


def test_robustness_keys_are_validated():
    for bad in ({"rekey_policy": "all"}, {"rekey_policy": "stopped_vehicles"}, {"label_hold_seconds": -1.0},
                {"occlusion_memory_seconds": -1.0}, {"lost_track_missed_runs": 0},
                {"lost_track_missed_runs": 1.5}, {"lost_track_missed_runs": True}):
        with pytest.raises(ValueError):
            tracker(**bad)
    tracker(lost_track_missed_runs=1)
    tracker(lost_track_missed_runs=3.0)  # a YAML float that is a whole number is fine
