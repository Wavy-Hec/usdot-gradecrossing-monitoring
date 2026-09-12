"""Synthetic-detection tests for gradecrossing.tracking.StuckVehicleTracker.

All scenarios are scripted at 30 fps (unless stated) with hand-built ``Detection``
objects, so no detector weights or torch are needed. numpy is seeded for the jitter
used in the stationary-car scenarios.
"""
from __future__ import annotations

from typing import Callable, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import pytest

from gradecrossing.config import DetectorConfig, StuckConfig
from gradecrossing.geometry import polygon_mask
from gradecrossing.tracking import PredictedDetection, StuckVehicleTracker
from gradecrossing.types import Detection, ObjectState, ZoneState

FPS = 30.0
DT = 1.0 / FPS

# Zone: a rectangle x in [200, 440], y in [200, 320] on a 640x480 frame.
ZONE_POLY = [[200, 200], [440, 200], [440, 320], [200, 320]]


# ----------------------------------------------------------------------------- helpers

def make_classify(cfg: Optional[DetectorConfig] = None) -> Callable[[str], str]:
    """Build the class-name -> category callable from DetectorConfig lists.

    detection.py provides the real one (VehicleDetector.classify); this mirrors its
    contract (case-insensitive membership) so the tracker can be tested in isolation.
    """
    cfg = cfg or DetectorConfig()
    vehicles = {c.lower() for c in cfg.vehicle_classes}
    persons = {c.lower() for c in cfg.person_classes}
    trains = {c.lower() for c in cfg.train_classes}

    def classify(name: str) -> str:
        n = name.lower()
        if n in vehicles:
            return "vehicle"
        if n in persons:
            return "person"
        if n in trains:
            return "train"
        return "other"

    return classify


def make_zone(poly=ZONE_POLY, shape=(480, 640)) -> ZoneState:
    p = np.asarray(poly, dtype=np.int32)
    mask = polygon_mask(shape, p)
    return ZoneState(polygon=p, mask=mask, homography=None, inliers=0, method="identity",
                     healthy=True, area_px=int(cv2.countNonZero(mask)))


def det(bx: float, by: float, w: float = 60.0, h: float = 40.0, cls: str = "car",
        tid: Optional[int] = 1, conf: float = 0.9, cls_id: int = 0) -> Detection:
    """Detection whose bottom-center is (bx, by)."""
    xyxy = np.array([bx - w / 2.0, by - h, bx + w / 2.0, by], dtype=np.float32)
    return Detection(xyxy=xyxy, conf=conf, cls_id=cls_id, cls_name=cls, track_id=tid)


def make_tracker(**overrides) -> StuckVehicleTracker:
    cfg = StuckConfig(**overrides)
    return StuckVehicleTracker(cfg, make_classify())


def run(tracker: StuckVehicleTracker, zone: ZoneState,
        frames: Iterable[Tuple[float, List[Detection]]]) -> List[Tuple[float, List[ObjectState]]]:
    """Feed scripted (t, detections) frames; return a snapshot of every object per frame."""
    out = []
    for t, dets in frames:
        objs = tracker.update(dets, zone, t)
        snap = []
        for o in objs:
            snap.append(ObjectState(**{**o.__dict__, "last_box": o.last_box.copy(),
                                       "history": list(o.history)}))
        out.append((t, snap))
    return out


def by_id(objs: List[ObjectState], tid: int) -> Optional[ObjectState]:
    for o in objs:
        if o.track_id == tid:
            return o
    return None


def moving_car_frames(x0: float, speed: float, y: float, n_frames: int, tid: Optional[int] = 1,
                      t0: float = 0.0, fps: float = FPS):
    for i in range(n_frames):
        t = t0 + i / fps
        yield t, [det(x0 + speed * (t - t0), y, tid=tid)]


def stationary_car_frames(x: float, y: float, n_frames: int, rng: np.random.Generator,
                          tid: Optional[int] = 1, t0: float = 0.0, fps: float = FPS,
                          jitter: float = 0.5, cls: str = "car"):
    for i in range(n_frames):
        t = t0 + i / fps
        jx, jy = rng.uniform(-jitter, jitter, size=2)
        yield t, [det(x + jx, y + jy, tid=tid, cls=cls)]


# ----------------------------------------------------------------------------- tests

def test_moving_car_never_stuck_and_dwell_resets():
    """Spec test 1: a car crossing at 200 px/s produces no stuck time; dwell grows then resets."""
    tracker = make_tracker()
    zone = make_zone()
    # x goes 40 -> 640 over 3 s; inside the zone (x in [200,440]) for t in [0.8, 2.0].
    snaps = run(tracker, zone, moving_car_frames(40.0, 200.0, 300.0, 91))
    dwell = []
    for i, (t, objs) in enumerate(snaps):
        o = by_id(objs, 1)
        assert o is not None
        assert o.is_vehicle and not o.is_person and not o.is_train
        assert o.stuck_seconds == 0.0
        assert o.stationary_seconds == 0.0
        if i < 2:
            assert o.speed == 0.0  # no estimate before min_history_for_speed samples
        else:
            assert 195.0 <= o.speed <= 205.0
            assert o.speed_units == "px/s"
        dwell.append(o.dwell_seconds)
    assert 1.1 <= max(dwell) <= 1.3  # 240 px wide zone at 200 px/s = 1.2 s
    assert dwell[-1] == 0.0 and not snaps[-1][1][0].in_zone
    assert dwell[0] == 0.0
    # dwell is monotone while inside, and a single contiguous streak
    inside = [d > 0 for d in dwell]
    first = inside.index(True)
    last = len(inside) - 1 - inside[::-1].index(True)
    assert all(inside[first:last + 1])
    assert all(dwell[i] <= dwell[i + 1] + 1e-9 for i in range(first, last))


def test_stationary_car_stuck_after_3s():
    """Spec test 2: a car standing in the zone reports stuck_seconds ~= 3 after 3 s."""
    rng = np.random.default_rng(0)
    tracker = make_tracker()
    zone = make_zone()
    snaps = run(tracker, zone, stationary_car_frames(320.0, 280.0, 91, rng))
    # Frames 0/1: no speed estimate yet, so nothing may accumulate.
    for t, objs in snaps[:2]:
        o = by_id(objs, 1)
        assert o.speed == 0.0 and o.stuck_seconds == 0.0 and o.stationary_seconds == 0.0
    t, objs = snaps[-1]
    assert abs(t - 3.0) < 1e-9
    o = by_id(objs, 1)
    assert o.in_zone
    assert o.speed < tracker.cfg.speed_threshold
    assert 2.9 <= o.stuck_seconds <= 3.1
    assert 2.9 <= o.stationary_seconds <= 3.1
    assert 2.9 <= o.dwell_seconds <= 3.1
    assert o.in_zone_since is not None and abs(o.in_zone_since - DT) < 1e-9  # forced out on frame 0
    assert tracker.zone_occupied
    assert tracker.zone_empty_since is None


def test_car_enters_then_stops_counter_starts_after_stop():
    """Stuck time must not accumulate while moving, and must start within one speed window of the stop."""
    tracker = make_tracker()
    zone = make_zone()
    t_stop = 1.5  # x = 40 + 200*1.5 = 340, inside the zone
    frames = []
    for i in range(181):  # 0..6 s
        t = i * DT
        x = 40.0 + 200.0 * min(t, t_stop)
        frames.append((t, [det(x, 300.0)]))
    snaps = run(tracker, zone, frames)
    first_stuck_t = None
    for t, objs in snaps:
        o = by_id(objs, 1)
        if t <= t_stop:
            assert o.stuck_seconds == 0.0
        elif first_stuck_t is None and o.stuck_seconds > 0.0:
            first_stuck_t = t
    assert first_stuck_t is not None
    # The clock starts on the first frame the condition holds (stuck_seconds == 0 there),
    # so the first frame with stuck_seconds > 0 is one frame after the onset.
    onset_t = first_stuck_t - DT
    window = tracker.cfg.speed_window_seconds
    assert t_stop < onset_t <= t_stop + window + 0.1
    t, objs = snaps[-1]
    o = by_id(objs, 1)
    assert abs(o.stuck_seconds - (t - onset_t)) < 1e-6
    assert abs(o.dwell_seconds - (6.0 - 0.8)) < 0.1  # entered the zone at t = 0.8


def test_fallback_iou_assigns_stable_ids_without_track_ids():
    """Spec test 3: detections with track_id=None still get one stable id each."""
    rng = np.random.default_rng(1)
    tracker = make_tracker()
    zone = make_zone()
    a = stationary_car_frames(320.0, 280.0, 91, rng, tid=None)
    b = moving_car_frames(20.0, 150.0, 420.0, 91, tid=None)  # far below the zone
    frames = [(ta, da + db) for (ta, da), (_, db) in zip(a, b)]
    snaps = run(tracker, zone, frames)
    ids_seen = set()
    for t, objs in snaps:
        assert len(objs) == 2
        for o in objs:
            ids_seen.add(o.track_id)
            assert isinstance(o.track_id, int)
    assert len(ids_seen) == 2
    t, objs = snaps[-1]
    stuck = [o for o in objs if o.in_zone]
    assert len(stuck) == 1
    assert 2.9 <= stuck[0].stuck_seconds <= 3.1
    other = [o for o in objs if not o.in_zone][0]
    assert 145.0 <= other.speed <= 155.0 and other.stuck_seconds == 0.0


def test_fallback_new_id_when_no_overlap():
    tracker = make_tracker()
    zone = make_zone()
    tracker.update([det(100.0, 100.0, tid=None)], zone, 0.0)
    objs = tracker.update([det(100.0, 100.0, tid=None), det(500.0, 400.0, tid=None)], zone, DT)
    assert sorted(o.track_id for o in objs) == [1, 2]
    # And the two persist with the same ids on the next frame.
    objs = tracker.update([det(500.0, 400.0, tid=None), det(100.0, 100.0, tid=None)], zone, 2 * DT)
    assert {o.track_id for o in objs} == {1, 2}
    assert by_id(objs, 1).last_box[0] == pytest.approx(70.0)
    assert by_id(objs, 2).last_box[0] == pytest.approx(470.0)


def test_person_train_other_categories():
    """Spec test 4: person -> is_person; train -> is_train, not a vehicle; other -> none."""
    tracker = make_tracker()
    zone = make_zone()
    frames = []
    for i in range(3):
        t = i * DT
        frames.append((t, [det(300.0, 300.0, cls="person", tid=1, w=20, h=60),
                           det(350.0, 300.0, cls="TRAIN", tid=2, w=200, h=80),
                           det(400.0, 300.0, cls="dog", tid=3, w=30, h=20)]))
    snaps = run(tracker, zone, frames)
    t, objs = snaps[-1]
    p, tr, dog = by_id(objs, 1), by_id(objs, 2), by_id(objs, 3)
    assert p.is_person and not p.is_vehicle and not p.is_train
    assert tr.is_train and not tr.is_vehicle and not tr.is_person
    assert not dog.is_vehicle and not dog.is_person and not dog.is_train
    assert p.in_zone and tr.in_zone and dog.in_zone
    assert tracker.zone_occupied


def test_zone_occupied_counts_unmapped_classes():
    """An object of a class the config does not map ("dog", COCO "bench", any class of a
    fine-tuned model missing from the lists) never feeds the stuck logic and is still
    subtracted from the debris mask, so blocking the background refresh is the only
    protection left: it must count as occupying the zone."""
    tracker = make_tracker()
    zone = make_zone()
    for i in range(3):
        tracker.update([det(300.0, 300.0, cls="dog", tid=7)], zone, i * DT)
    o = by_id(tracker.objects(), 7)
    assert o.in_zone
    assert not (o.is_vehicle or o.is_person or o.is_train)  # invisible to the alert engine
    assert tracker.zone_occupied
    assert tracker.zone_empty_since is None
    # ... and the zone reads empty again once it leaves (bottom-center outside the polygon).
    for i in range(3, 6):
        tracker.update([det(300.0, 400.0, cls="dog", tid=7)], zone, i * DT)
    assert not tracker.zone_occupied
    assert tracker.zone_empty_since == pytest.approx(3 * DT)


def test_track_dropped_after_grace_and_new_id_restarts():
    """Spec test 5: a vanished track is dropped after the grace period; a new id starts fresh."""
    rng = np.random.default_rng(2)
    tracker = make_tracker(lost_track_grace_seconds=1.0)
    zone = make_zone()
    run(tracker, zone, stationary_car_frames(320.0, 280.0, 61, rng, tid=1))  # 0..2 s
    o = by_id(tracker.objects(), 1)
    assert 1.85 <= o.stuck_seconds <= 2.05
    # Vanish: empty frames. Within grace it is still live with frozen counters.
    t = 2.0
    for i in range(1, 16):  # up to t = 2.5
        t = 2.0 + i * DT
        objs = tracker.update([], zone, t)
        assert by_id(objs, 1) is not None
        assert tracker.zone_occupied
    assert by_id(tracker.objects(), 1).stuck_seconds == o.stuck_seconds
    # Beyond grace: gone, zone reported empty from the frame it was dropped.
    dropped_at = None
    while t < 4.0:
        t += DT
        objs = tracker.update([], zone, t)
        if by_id(objs, 1) is None and dropped_at is None:
            dropped_at = t
    assert dropped_at is not None
    assert 3.0 < dropped_at <= 3.0 + 2 * DT + 1e-9
    assert tracker.objects() == []
    assert tracker.zone_empty_since == pytest.approx(dropped_at)
    assert not tracker.zone_occupied
    # Reappears with a new id: counters start over.
    snaps = run(tracker, zone, stationary_car_frames(320.0, 280.0, 31, rng, tid=2, t0=4.0))  # 4..5 s
    t, objs = snaps[-1]
    assert by_id(objs, 1) is None
    o2 = by_id(objs, 2)
    assert o2.first_seen == pytest.approx(4.0)
    assert 0.85 <= o2.stuck_seconds <= 1.0
    assert tracker.zone_empty_since is None


def test_id_swap_in_place_keeps_the_stuck_clock_running():
    """A stationary vehicle whose detector id changes (ByteTrack re-identifying it after an
    occlusion - typically the train it is trapped in front of) must not restart its
    counters: the box has not moved, so it is the same object. Before the re-keying, the
    new id started at zero with no speed estimate and the ALARM was pushed out by the
    grace period plus the speed window plus the whole threshold again."""
    rng = np.random.default_rng(11)
    tracker = make_tracker(lost_track_grace_seconds=1.0)
    zone = make_zone()
    run(tracker, zone, stationary_car_frames(320.0, 280.0, 181, rng, tid=1))  # 0..6 s
    before = by_id(tracker.objects(), 1)
    assert 5.8 <= before.stuck_seconds <= 6.05

    # From t = 6 s the detector calls the same box id 2.
    snaps = run(tracker, zone, stationary_car_frames(320.0, 280.0, 61, rng, tid=2, t0=6.0))
    t, objs = snaps[-1]
    assert by_id(objs, 1) is None  # the old id is gone: the track was re-keyed, not duplicated
    o = by_id(objs, 2)
    assert o is not None and len(objs) == 1
    assert o.first_seen == 0.0  # same track, same history
    assert 7.8 <= o.stuck_seconds <= 8.05, "the stuck clock restarted on the id swap"
    assert o.speed < 12.0 and o.in_zone

    # The swap happens on the first frame of the new id, with no gap at all.
    first_t, first_objs = snaps[0]
    assert by_id(first_objs, 2).stuck_seconds == pytest.approx(first_t, abs=0.1)


def test_id_swap_does_not_steal_a_track_another_detection_owns():
    """Re-keying must never cannibalise a track that is still being detected under its own
    id: a second vehicle stopping next to the first gets its own track and its own clock."""
    rng = np.random.default_rng(12)
    tracker = make_tracker()
    zone = make_zone()
    run(tracker, zone, stationary_car_frames(320.0, 280.0, 61, rng, tid=1))  # 0..2 s
    # id 1 keeps being reported *and* a new id 2 appears on an overlapping box.
    for i in range(30):
        t = 2.0 + i * DT
        tracker.update([det(320.0, 280.0, tid=1), det(330.0, 285.0, tid=2)], zone, t)
    objs = tracker.objects()
    assert {o.track_id for o in objs} == {1, 2}
    assert by_id(objs, 1).stuck_seconds > 2.5  # the original clock is untouched
    assert by_id(objs, 2).stuck_seconds < 1.1  # the newcomer starts from zero


def test_id_swap_needs_overlap():
    """A new id on a box that does not overlap any live track is a new object."""
    rng = np.random.default_rng(13)
    tracker = make_tracker()
    zone = make_zone()
    run(tracker, zone, stationary_car_frames(220.0, 280.0, 61, rng, tid=1))
    objs = tracker.update([det(420.0, 300.0, tid=9)], zone, 2.0)  # far away
    assert {o.track_id for o in objs} == {1, 9}
    assert by_id(objs, 9).first_seen == 2.0 and by_id(objs, 9).stuck_seconds == 0.0


def test_short_dropout_within_grace_keeps_counters():
    """Timestamp-based counters bridge a detector dropout shorter than the grace period."""
    rng = np.random.default_rng(3)
    tracker = make_tracker(lost_track_grace_seconds=1.0)
    zone = make_zone()
    run(tracker, zone, stationary_car_frames(320.0, 280.0, 61, rng, tid=1))  # 0..2 s
    for i in range(1, 10):  # 0.3 s of nothing
        tracker.update([], zone, 2.0 + i * DT)
    snaps = run(tracker, zone, stationary_car_frames(320.0, 280.0, 31, rng, tid=1, t0=2.4))  # 2.4..3.4
    t, objs = snaps[-1]
    o = by_id(objs, 1)
    assert 3.25 <= o.stuck_seconds <= 3.4  # continuous since ~0.07 s, not restarted at 2.4
    assert o.first_seen == 0.0


def test_corner_overlap_but_bottom_center_outside_is_not_in_zone():
    """Spec test 6: bbox pixel overlap is never the criterion; only the bottom-center is."""
    tracker = make_tracker()
    zone = make_zone()
    # Box x in [270,330], y in [280,360]: its top corners are inside the zone (y<320),
    # but the bottom-center (300, 360) is below the zone's bottom edge (y=320).
    box = Detection(xyxy=np.array([270, 280, 330, 360], dtype=np.float32), conf=0.9, cls_id=0,
                    cls_name="car", track_id=5)
    for i in range(5):
        objs = tracker.update([box], zone, i * DT)
    o = by_id(objs, 5)
    assert not o.in_zone and o.dwell_seconds == 0.0
    assert not tracker.zone_occupied
    # Sanity: a box whose bottom-center IS on the zone edge counts as inside.
    edge = Detection(xyxy=np.array([270, 280, 330, 320], dtype=np.float32), conf=0.9, cls_id=0,
                     cls_name="car", track_id=6)
    for i in range(5, 8):
        objs = tracker.update([edge], zone, i * DT)
    assert by_id(objs, 6).in_zone


def test_predict_boxes_propagates_velocity_and_preserves_ids():
    tracker = make_tracker()
    zone = make_zone()
    snaps = run(tracker, zone, moving_car_frames(40.0, 200.0, 300.0, 31, tid=3))  # 0..1 s
    t_last = snaps[-1][0]
    last_box = by_id(snaps[-1][1], 3).last_box
    preds = tracker.predict_boxes(t_last + 0.5)
    assert len(preds) == 1
    p = preds[0]
    assert isinstance(p, Detection) and isinstance(p, PredictedDetection)
    assert p.track_id == 3 and p.cls_name == "car" and p.conf == pytest.approx(0.9)
    assert p.xyxy.dtype == np.float32
    shift = p.xyxy - last_box
    assert np.allclose(shift[[0, 2]], 100.0, atol=3.0)  # 200 px/s * 0.5 s
    assert np.allclose(shift[[1, 3]], 0.0, atol=1.0)
    # No velocity estimate yet -> zero shift.
    fresh = make_tracker()
    fresh.update([det(100.0, 100.0, tid=9)], zone, 0.0)
    p0 = fresh.predict_boxes(1.0)[0]
    assert np.array_equal(p0.xyxy, det(100.0, 100.0, tid=9).xyxy)


def test_predicted_boxes_fed_back_advance_clocks_but_are_not_evidence():
    """Feeding predict_boxes output into update keeps counters ticking, yet a track the
    detector has lost is still dropped after the grace period and its speed history
    does not grow from its own predictions."""
    rng = np.random.default_rng(4)
    tracker = make_tracker(lost_track_grace_seconds=1.0)
    zone = make_zone()
    run(tracker, zone, stationary_car_frames(320.0, 280.0, 61, rng, tid=1))  # 0..2 s
    o = by_id(tracker.objects(), 1)
    hist_len = len(o.history)
    stuck_before = o.stuck_seconds
    # Skipped detector frames: predicted boxes only.
    t = 2.0 + DT
    objs = tracker.update(tracker.predict_boxes(t), zone, t)
    o = by_id(objs, 1)
    assert o.stuck_seconds == pytest.approx(stuck_before + DT)  # clock advanced
    assert len(o.history) == hist_len  # no fake position samples
    assert o.last_seen == pytest.approx(2.0)  # last *real* detection
    assert o.in_zone
    # Keep predicting with no real detections: must die after grace, not live forever.
    alive_at = []
    while t < 4.0:
        t += DT
        objs = tracker.update(tracker.predict_boxes(t), zone, t)
        if by_id(objs, 1) is not None:
            alive_at.append(t)
    assert alive_at and alive_at[-1] <= 3.0 + 1e-9
    assert tracker.objects() == []
    assert tracker.predict_boxes(t) == []


def test_predicted_box_does_not_change_zone_membership():
    """Only real detections may move an object in or out of the zone."""
    tracker = make_tracker()
    zone = make_zone()
    # Car at 200 px/s heading for the zone, last real detection at x = 190 (just outside).
    frames = list(moving_car_frames(-10.0, 200.0, 300.0, 31, tid=1))  # x: -10 .. 190
    run(tracker, zone, frames)
    assert not by_id(tracker.objects(), 1).in_zone
    t = frames[-1][0] + 0.2  # prediction puts it at x ~ 230, inside the zone
    preds = tracker.predict_boxes(t)
    assert 200.0 < (preds[0].xyxy[0] + preds[0].xyxy[2]) / 2.0 < 440.0
    objs = tracker.update(preds, zone, t)
    assert not by_id(objs, 1).in_zone and by_id(objs, 1).dwell_seconds == 0.0
    # A real detection at the same place does count.
    objs = tracker.update([det(230.0, 300.0, tid=1)], zone, t + DT)
    assert by_id(objs, 1).in_zone


def test_ground_plane_homography_reports_metres_per_second():
    # Pure scale: 100 px = 1 m. 200 px/s -> 2.0 m/s.
    H = [[0.01, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 1.0]]
    tracker = make_tracker(ground_plane_homography=H, speed_threshold=1.0)
    zone = make_zone()
    snaps = run(tracker, zone, moving_car_frames(40.0, 200.0, 300.0, 61))
    o = by_id(snaps[-1][1], 1)
    assert o.speed_units == "m/s"
    assert abs(o.speed - 2.0) < 0.05
    assert o.stuck_seconds == 0.0
    # Stationary car under the same homography accumulates stuck time (0 m/s < 1 m/s).
    rng = np.random.default_rng(5)
    tracker2 = make_tracker(ground_plane_homography=H, speed_threshold=1.0)
    snaps = run(tracker2, zone, stationary_car_frames(320.0, 280.0, 91, rng))
    o = by_id(snaps[-1][1], 1)
    assert o.speed_units == "m/s" and o.speed < 1.0
    assert 2.9 <= o.stuck_seconds <= 3.1
    with pytest.raises(ValueError):
        make_tracker(ground_plane_homography=[[1.0, 0.0], [0.0, 1.0]])


def test_min_track_age_forces_in_zone_false_for_new_ids():
    zone = make_zone()
    tracker = make_tracker(min_track_age_frames=2)
    objs = tracker.update([det(320.0, 280.0, tid=1)], zone, 0.0)
    assert not by_id(objs, 1).in_zone and not tracker.zone_occupied
    objs = tracker.update([det(320.0, 280.0, tid=1)], zone, DT)
    assert by_id(objs, 1).in_zone and tracker.zone_occupied
    tracker3 = make_tracker(min_track_age_frames=3)
    for i in range(2):
        objs = tracker3.update([det(320.0, 280.0, tid=1)], zone, i * DT)
        assert not by_id(objs, 1).in_zone
    objs = tracker3.update([det(320.0, 280.0, tid=1)], zone, 2 * DT)
    assert by_id(objs, 1).in_zone


def test_counters_are_frame_rate_independent():
    """The same 6 s of a stationary car at 10 fps and 30 fps gives the same stuck time."""
    results = {}
    for fps in (10.0, 30.0):
        rng = np.random.default_rng(6)
        tracker = make_tracker()
        zone = make_zone()
        n = int(round(6.0 * fps)) + 1
        snaps = run(tracker, zone, stationary_car_frames(320.0, 280.0, n, rng, fps=fps))
        t, objs = snaps[-1]
        assert abs(t - 6.0) < 1e-6
        results[fps] = by_id(objs, 1).stuck_seconds
    assert abs(results[10.0] - results[30.0]) <= 0.2
    assert results[10.0] > 5.7 and results[30.0] > 5.7


def test_zone_empty_since_lifecycle():
    tracker = make_tracker()
    zone = make_zone()
    assert tracker.zone_empty_since is None  # nothing known before the first update
    tracker.update([], zone, 0.5)
    assert tracker.zone_empty_since == 0.5
    tracker.update([], zone, 1.0)
    assert tracker.zone_empty_since == 0.5  # unchanged while it stays empty
    frames = list(moving_car_frames(40.0, 200.0, 300.0, 91, t0=1.0))  # inside for t in [1.8, 3.0]
    became_occupied = became_empty = None
    for t, dets in frames:
        tracker.update(dets, zone, t)
        if tracker.zone_occupied and became_occupied is None:
            became_occupied = t
        if became_occupied is not None and not tracker.zone_occupied and became_empty is None:
            became_empty = t
            assert tracker.zone_empty_since == pytest.approx(t)
    assert 1.8 <= became_occupied <= 1.8 + 2 * DT
    assert 3.0 <= became_empty <= 3.0 + 2 * DT + 1e-9
    assert tracker.zone_empty_since == pytest.approx(became_empty)


def test_reset_clears_everything():
    rng = np.random.default_rng(7)
    tracker = make_tracker()
    zone = make_zone()
    run(tracker, zone, stationary_car_frames(320.0, 280.0, 31, rng, tid=None))
    assert tracker.objects() and tracker.zone_occupied
    tracker.reset()
    assert tracker.objects() == []
    assert not tracker.zone_occupied
    assert tracker.zone_empty_since is None
    assert tracker.predict_boxes(5.0) == []
    objs = tracker.update([det(100.0, 100.0, tid=None)], zone, 5.0)
    assert [o.track_id for o in objs] == [1]  # fallback ids restart


def test_duplicate_ids_and_class_flicker_are_tolerated():
    tracker = make_tracker()
    zone = make_zone()
    objs = tracker.update([det(320.0, 280.0, tid=1, cls="car"), det(100.0, 100.0, tid=1, cls="car")],
                          zone, 0.0)
    assert len(objs) == 1 and objs[0].last_box[0] == pytest.approx(290.0)  # first one wins
    objs = tracker.update([det(320.0, 280.0, tid=1, cls="truck")], zone, DT)
    assert objs[0].cls_name == "truck" and objs[0].is_vehicle
    objs = tracker.update([det(320.0, 280.0, tid=1, cls="person")], zone, 2 * DT)
    assert objs[0].is_person and not objs[0].is_vehicle
    # history is exposed as (t, x, y) bottom-centers
    assert len(objs[0].history) == 3
    assert objs[0].history[-1][1:] == pytest.approx((320.0, 280.0))
