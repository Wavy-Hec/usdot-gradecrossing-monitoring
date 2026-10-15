"""MA5 (digest/final): a vehicle that brakes into the zone and stops raises ALARM on time.

clip07's trapped van (ground truth 2.3-5.8 s) only reached WARNING, for two reasons
(work/rev-missed-alarms, out_clip07_def.trace.jsonl):

(a) its still cluster was anchored at 2.6 s, while the van still rolled at 93 px/s, 45 px before
    its stop. The stopped box overlapped that anchor at IoU 0.58 only, so a ~20 px creep at 4.6 s
    left the cluster and restarted the still clock. The anchor now moves once, to the owner's box
    where its speed first drops below speed_threshold; the cluster keeps its start.
(b) from 5.0 s the train hid the van: a live but undetected track kept the stuck time of its last
    detection until it was dropped. On detector runs the dwell and stuck clocks of a live,
    undetected, in-zone vehicle now keep running from their start until it is dropped
    (stuck.undetected_clock_while_live, false in configs/legacy_rule.yaml).

Replayed on the recorded detections of the 8 team clips: ALARM recall 1/3 -> 3/3 (clip07 ALARM
5.61 s, clip08 5.84 s, before the train hits it), one false-alarm episode fewer, +13.5 s of false
ALARM (+12.9 s on clip06, whose two zone-geometry false ALARMs merge into one; +0.6 s on clip05).

Synthetic detections at 30 fps; the tracker and the alert engine are the real ones.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import pytest

from gradecrossing.alerting import AlertEngine, CallbackSink
from gradecrossing.config import AlertConfig, MonitorConfig, StuckConfig, load_config
from gradecrossing.geometry import box_iou, polygon_mask
from gradecrossing.monitor import ConfigClassifier
from gradecrossing.tracking import StuckVehicleTracker
from gradecrossing.types import Detection, Event, GateState, ObjectState, Status, ZoneState

FPS = 30.0
DT = 1.0 / FPS
ROOT = Path(__file__).resolve().parents[1]
CLASSIFY = ConfigClassifier(MonitorConfig().detector)
BAND = [[0, 190], [640, 190], [640, 260], [0, 260]]  # a zone across the whole picture, like clip01-06
CAR = np.array([300, 200, 380, 250], np.float32)  # stands in BAND (footprint 237.5-250)


def zone_state(poly=BAND) -> ZoneState:
    p = np.asarray(poly, np.int32)
    return ZoneState(polygon=p, mask=polygon_mask((360, 640), p), homography=None, inliers=100,
                     method="identity", healthy=True)


def det(b: Sequence[float], tid: Optional[int] = 1, cls: str = "car") -> Detection:
    return Detection(xyxy=np.asarray(b, np.float32), conf=0.9, cls_id=2, cls_name=cls, track_id=tid)


def run(dets_at: Callable[[float], List[Detection]], seconds: float, zone_poly=BAND, **stuck):
    """Tracker + engine wired like Monitor.process_frame, the detector on every frame. The log
    keeps a copy of every frame's objects (the tracker updates its ObjectStates in place)."""
    cfg = StuckConfig(**stuck)
    tr = StuckVehicleTracker(cfg, CLASSIFY)
    eng = AlertEngine(cfg, AlertConfig(stdout=False, repeat_notify_seconds=0.0))
    events: List[Event] = []
    eng.add_sink(CallbackSink(events.append))
    z = zone_state(zone_poly)
    log: List[Tuple[float, Status, List[ObjectState]]] = []
    for i in range(int(round(seconds * FPS))):
        t = i * DT
        objs = tr.update(dets_at(t), z, t, alarmed_ids=eng.alarm_episode_track_ids)
        a, _ = eng.evaluate(objs, None, GateState(), t, i)
        log.append((t, a.status, [copy.copy(o) for o in objs]))
    return tr, events, log


def first(events: Sequence[Event], kind: str) -> Optional[float]:
    return next((e.timestamp for e in events if e.type == kind), None)


# ----------------------------------------------------------------------------- (a) the anchor
# A 140 px van (clip07's is 140 px wide) at 150 px/s brakes evenly from T_BRAKE to a stop at
# T_STOP inside a zone on the right half of the picture, stands, then creeps 20 px forward over
# 0.5 s from T_CREEP. Its last still cluster opens while it brakes, ~40 px before the stop.
VAN_W = 140.0
V0 = 150.0
T_BRAKE, T_STOP = 1.1, 2.6
T_CREEP = T_STOP + 1.0
CREEP_PX = 20.0
X_STOP = 400.0
RIGHT_PART = [[300, 190], [640, 190], [640, 260], [300, 260]]  # the van enters it at ~0.3 s


def van_x(t: float, creep: bool) -> float:
    d_brake = V0 * (T_STOP - T_BRAKE) / 2.0
    x0 = X_STOP - d_brake - V0 * T_BRAKE
    if t < T_BRAKE:
        x = x0 + V0 * t
    elif t < T_STOP:
        u = t - T_BRAKE
        x = x0 + V0 * T_BRAKE + V0 * u - V0 / (T_STOP - T_BRAKE) * u * u / 2.0
    else:
        x = X_STOP
    if creep and t >= T_CREEP:
        x += CREEP_PX * min(1.0, (t - T_CREEP) / 0.5)
    return x


def van_box(t: float, creep: bool = False) -> np.ndarray:
    x = van_x(t, creep)
    return np.array([x, 180.0, x + VAN_W, 260.0], np.float32)


def van_run(creep: bool, **stuck):
    return run(lambda t: [det(van_box(t, creep), 1, "truck")], 10.0, RIGHT_PART, **stuck)


def test_a_van_that_brakes_into_the_zone_alarms_on_time_despite_a_creep_after_the_stop():
    alarm_s = StuckConfig().alarm_seconds
    tr, events, log = van_run(creep=True)
    crept = van_box(T_CREEP + 1.0, creep=True)
    stopped = van_box(T_STOP)
    cluster = tr._tracks[1].cluster
    # the setting: the cluster opened while the van rolled; the creep leaves that first box ...
    assert cluster.start < T_STOP - 0.2 and cluster.settled
    assert box_iou(van_box(cluster.start), crept) < 0.5 <= box_iou(van_box(cluster.start), stopped)
    # ... but not the box where it stopped, which is now the anchor
    assert box_iou(cluster.anchor, stopped) > 0.99 and box_iou(cluster.anchor, crept) >= 0.5
    # ALARM no later than alarm_seconds after the stop, held through the creep, as without it
    alarm = first(events, "ALARM_START")
    assert alarm is not None and alarm <= T_STOP + alarm_s + DT + 1e-6
    assert first(events, "ALARM_END") is None and log[-1][1] is Status.ALARM
    _, events_still, _ = van_run(creep=False)
    assert first(events_still, "ALARM_START") == pytest.approx(alarm, abs=1e-9)
    # it is the still clock that carries it: the creep restarts the speed clock
    assert any(not o.stationary_seconds for t, _, objs in log if T_CREEP < t < T_CREEP + 1.0 for o in objs)


def test_the_anchor_moves_once_so_a_creeping_vehicle_still_leaves_its_cluster():
    """A vehicle creeping at 6 px/s (below speed_threshold) is re-anchored once, then leaves the
    cluster after a third of its width like before: its still clock restarts."""
    w, v = 120.0, 6.0

    def dets(t: float) -> List[Detection]:
        x = 300.0 + v * t
        return [det([x, 200, x + w, 250])]

    tr, _, log = run(dets, 12.0)
    still = [(t, objs[0].still_seconds) for t, _, objs in log]
    restarts = [t for (t, s), (_, prev) in zip(still[1:], still[:-1]) if s < prev]
    # IoU (w - d) / (w + d) < 0.5 once it has moved d > w / 3 from the (re-)anchor
    assert restarts and restarts[0] == pytest.approx((w / 3.0) / v, abs=0.1)
    assert tr._tracks[1].cluster.settled


def test_the_speed_only_rule_has_no_clusters_to_move():
    """The rollback profile's stationarity 'speed' never builds still clusters, so (a) cannot act
    there; the legacy goldens (tests/test_legacy_profile.py) check the rest."""
    tr, _, _ = van_run(creep=True, stationarity="speed")
    assert tr._tracks[1].cluster is None and tr._clusters.active == []
    legacy = load_config(str(ROOT / "configs" / "legacy_rule.yaml")).stuck
    assert legacy.stationarity == "speed" and legacy.undetected_clock_while_live is False


# ----------------------------------------------------------------------------- (b) undetected clocks
T_MISS, T_BACK = 2.5, 3.3  # the detector misses the stopped car for 0.8 s < lost_track_grace_seconds


def missed_car(t: float) -> List[Detection]:
    return [] if T_MISS - DT / 2 <= t < T_BACK - DT / 2 else [det(CAR)]


def test_a_stopped_car_the_detector_misses_briefly_keeps_its_stuck_clock():
    cfg = StuckConfig()
    assert T_BACK - T_MISS < cfg.lost_track_grace_seconds
    _, events, log = run(missed_car, 6.0)
    _, events_seen, _ = run(lambda t: [det(CAR)], 6.0)
    # ALARM when a car seen on every frame alarms (3.0 s + the first frame of the track), unseen
    assert first(events, "ALARM_START") == pytest.approx(cfg.alarm_seconds + DT, abs=1e-6)
    assert first(events, "ALARM_START") == pytest.approx(first(events_seen, "ALARM_START"), abs=1e-9)
    missed = [(t, objs) for t, _, objs in log if T_MISS - DT / 2 <= t < T_BACK - DT / 2]
    assert missed and all(len(objs) == 1 and objs[0].in_zone for _, objs in missed)  # live, not dropped
    for t, objs in missed:
        assert objs[0].stuck_seconds == pytest.approx(t - DT, abs=1e-6)
        assert objs[0].dwell_seconds == pytest.approx(t - DT, abs=1e-6)
    assert first(events, "ALARM_END") is None


def test_with_the_key_off_the_clocks_of_an_undetected_car_freeze():
    _, events, log = run(missed_car, 6.0, undetected_clock_while_live=False)
    frozen = T_MISS - DT - DT  # the stuck time of its last detection
    for t, _, objs in log:
        if T_MISS - DT / 2 <= t < T_BACK - DT / 2:
            assert objs[0].stuck_seconds == pytest.approx(frozen, abs=1e-6)
    # the ALARM waits for the re-detection
    assert first(events, "ALARM_START") == pytest.approx(T_BACK, abs=1e-6)


def test_a_car_that_drives_out_of_the_zone_and_vanishes_does_not_alarm():
    """A car stopped 2.5 s in the zone (WARNING) drives off across the zone edge and is lost 0.3 s
    later, still inside its zone-exit grace: its clocks run from its last detection until it is
    dropped, which is far from an ALARM."""
    cfg = StuckConfig()
    t_go, t_gone = 2.5, 2.8

    def dets(t: float) -> List[Detection]:
        if t >= t_gone - DT / 2:
            return []
        dy = 150.0 * max(0.0, t - t_go)  # down, away from the band
        return [det(CAR + np.array([0, dy, 0, dy], np.float32))]

    _, events, log = run(dets, 8.0)
    assert first(events, "WARNING_START") is not None
    assert first(events, "ALARM_START") is None
    after = [(t, objs) for t, _, objs in log if t >= t_gone - DT / 2]
    assert after[0][1] and after[0][1][0].in_zone  # lost while still held in the zone: (b) runs
    assert max(o.stuck_seconds for _, objs in after for o in objs) < cfg.warning_seconds
    gone = [t for t, objs in after if not objs]
    assert gone and gone[0] <= t_gone + cfg.lost_track_grace_seconds + 2 * DT  # dropped on time
    assert log[-1][1] is Status.CLEAR
