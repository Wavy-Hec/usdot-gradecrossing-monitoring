"""ID-free stationarity (StationaryClusters) and how it feeds the stuck clock.

Synthetic detections at 30 fps (unless stated), seeded numpy, no detector. The jitter
scenario mirrors what YOLO11n + ByteTrack did on the FOX13 clip (clip01): the SUV stood still
from ~14 s to the end, yet its bottom-center least-squares speed crossed 12 px/s nine times
(box changes as a gate arm and the train passed), restarting the speed-based clock, while one
ID-free cluster covered the whole stop (work/ours-port-design/analyze_stationarity.py).
The image-border scenario mirrors clip06 at 62.5 s: the end of a passing train cut off by the
right edge of the picture, labelled 'bus', whose box stood still while the train moved.
"""
from __future__ import annotations

import math
from typing import List, Optional

import numpy as np
import pytest

from gradecrossing.alerting import AlertEngine
from gradecrossing.config import AlertConfig, StuckConfig
from gradecrossing.geometry import box_iou, polygon_mask
from gradecrossing.tracking import StationaryClusters, StuckVehicleTracker
from gradecrossing.types import Detection, GateState, Status, ZoneState

FPS = 30.0
DT = 1.0 / FPS
ZONE_POLY = [[100, 100], [540, 100], [540, 330], [100, 330]]


def classify(name: str) -> str:
    return {"car": "vehicle", "truck": "vehicle", "bus": "vehicle", "person": "person",
            "train": "train"}.get(name, "other")


def zone(poly=ZONE_POLY) -> ZoneState:
    p = np.asarray(poly, dtype=np.int32)
    return ZoneState(polygon=p, mask=polygon_mask((360, 640), p), homography=None, inliers=0,
                     method="identity", healthy=True)


def box(cx: float, by: float, w: float = 120.0, h: float = 80.0) -> np.ndarray:
    return np.array([cx - w / 2, by - h, cx + w / 2, by], dtype=np.float32)


def det(b: np.ndarray, tid: Optional[int] = 1, cls: str = "car") -> Detection:
    return Detection(xyxy=np.asarray(b, np.float32), conf=0.9, cls_id=0, cls_name=cls, track_id=tid)


def run(tr: StuckVehicleTracker, frames) -> List:
    out = []
    z = zone()
    for t, dets in frames:
        objs = tr.update(dets, z, t)
        out.append((t, {o.track_id: (o.stuck_seconds, o.still_seconds, o.stationary_seconds) for o in objs}))
    return out


def id_switch_frames():
    """Old id 1 still reported on a duplicate box for 3 frames while the new id 2 appears, then
    only id 2: the re-keying cannot match it (id 1 is claimed on those frames)."""
    frames = []
    for i in range(int(6 * FPS) + 1):
        t = i * DT
        dets = [det(box(320, 260), tid=1 if t < 3.0 else 2)]
        if 2.9 <= t < 3.0:
            dets.append(det(box(322, 261), tid=2))
        frames.append((t, dets))
    return frames


# ----------------------------------------------------------------------------- clusters
def test_clusters_follow_the_anchor_not_the_latest_box():
    sc = StationaryClusters(min_iou=0.5, max_gap_seconds=1.0)
    a = sc.update(0.0, [box(300, 200)])[0]
    # 30 px drift on a 120 px wide box: IoU 0.6 with the anchor -> same cluster
    assert sc.update(1.0, [box(330, 200)])[0] is a and a.last == 1.0 and a.count == 2
    # 45 px from the anchor (IoU 0.45) starts a new cluster, even though it is only 15 px
    # from the previous box: creeping cannot keep one cluster alive forever.
    b = sc.update(2.0, [box(345, 200)])[0]
    assert b is not a and b.start == 2.0


def test_cluster_expires_after_the_gap():
    sc = StationaryClusters(min_iou=0.5, max_gap_seconds=1.0)
    a = sc.update(0.0, [box(300, 200)])[0]
    assert sc.update(1.0, [box(300, 200)])[0] is a  # gap of exactly 1 s is bridged
    b = sc.update(2.05, [box(300, 200)])[0]  # 1.05 s without a match: a new stop
    assert b is not a and not sc.alive(a, 2.05) and sc.alive(b, 2.05)
    sc.revive(a, 2.1)  # a hidden hazard re-detected: its cluster is back in play
    assert sc.alive(a, 2.1) and sum(c is a for c in sc.active) == 1
    sc.revive(a, 2.2)
    assert sum(c is a for c in sc.active) == 1
    sc.reset()
    assert sc.active == [] and not sc.alive(a, 2.2) and not sc.alive(b, 2.2)


class _LoopClusters:
    """The f1aee43 matcher (one box_iou call per box and live cluster): the reference."""

    def __init__(self, min_iou: float, max_gap: float) -> None:
        self.min_iou, self.max_gap, self.active, self.made = float(min_iou), float(max_gap), [], 0

    def update(self, t: float, boxes) -> List[dict]:
        self.active = [c for c in self.active if t - c["last"] <= self.max_gap + 1e-6]
        out = []
        for b in boxes:
            b = np.asarray(b, np.float32).reshape(4)
            best, best_iou = None, self.min_iou
            for c in self.active:
                v = box_iou(c["anchor"], b)
                if v >= best_iou:
                    best, best_iou = c, v
            if best is None:
                best = {"anchor": b.copy(), "start": t, "last": t, "count": 1, "n": self.made}
                self.made += 1
                self.active.append(best)
            else:
                best["last"], best["count"] = t, best["count"] + 1
            out.append(best)
        return out


def test_vectorised_matcher_assigns_exactly_like_the_loop():
    """review issue 5: the per-frame IoU matrix replaced 1.8 M box_iou calls (30 moving vehicles,
    600 frames). Same clusters, same order, same tie-break (the newest of equal best IoUs), same
    expiry, including clusters opened earlier in the same frame and exact duplicate boxes."""
    rng = np.random.default_rng(11)
    spots = rng.uniform(0, 500, size=(8, 2)).round()
    new, ref = StationaryClusters(0.5, 1.0), _LoopClusters(0.5, 1.0)
    index, seen = {}, []  # creation order of our clusters (kept alive so ids are not reused)
    t = 0.0
    for _ in range(600):
        t += float(rng.choice([1 / 30, 1 / 30, 1 / 30, 0.4, 1.2]))
        boxes = []
        for _ in range(int(rng.integers(0, 9))):
            x, y = spots[int(rng.integers(0, len(spots)))]
            w, h = (60.0, 40.0) if rng.random() < 0.7 else (90.0, 60.0)
            dx, dy = rng.integers(-12, 13, size=2) if rng.random() < 0.8 else (0, 0)
            boxes.append(np.array([x + dx, y + dy, x + dx + w, y + dy + h], np.float32))
        got = new.update(t, boxes)
        want = ref.update(t, boxes)
        for g, w_ in zip(got, want):
            if id(g) not in index:
                index[id(g)] = len(index)
                seen.append(g)
            n = index[id(g)]
            assert (n, g.start, g.last, g.count) == (w_["n"], w_["start"], w_["last"], w_["count"])
        assert len(new.active) == len(ref.active)
    assert len(index) > 100  # the stream really opened, matched and expired many clusters


def test_a_tie_goes_to_the_newest_cluster_and_same_frame_clusters_count():
    sc = StationaryClusters(0.5, 1.0)
    a_box = np.array([0, 0, 100, 100], np.float32)
    b_box = np.array([40, 0, 140, 100], np.float32)  # IoU 0.43 with a_box: its own cluster
    a, b = sc.update(0.0, [a_box, b_box])
    assert a is not b
    c_box = np.array([20, 0, 120, 100], np.float32)  # IoU 2/3 with both anchors
    assert box_iou(a_box, c_box) == box_iou(b_box, c_box)
    assert sc.update(0.1, [c_box])[0] is b
    d, e = sc.update(0.2, [np.array([300, 300, 360, 340], np.float32), np.array([302, 301, 362, 341], np.float32)])
    assert d is e and d.count == 2  # the second box joins the cluster the first opened this frame


# ----------------------------------------------------------------------------- tracker
def test_still_clock_survives_box_jitter_that_resets_the_speed_clock():
    """Every 2 s the box bottom jumps up 20 px for 0.3 s (partial occlusion). The speed path
    restarts each time; ID-free stationarity (IoU 0.75 with the anchor) does not."""
    rng = np.random.default_rng(0)
    frames = []
    for i in range(int(8 * FPS) + 1):
        t = i * DT
        occluded = (t % 2.0) > 1.7
        h = 60.0 if occluded else 80.0
        by = 260.0 - (20.0 if occluded else 0.0)
        b = box(320 + rng.uniform(-0.5, 0.5), by + rng.uniform(-0.5, 0.5), h=h)
        b[1] = 260.0 - 80.0  # the top edge does not move; only the visible bottom does
        frames.append((t, [det(b)]))
    speed = run(StuckVehicleTracker(StuckConfig(stationarity="speed"), classify), frames)
    both = run(StuckVehicleTracker(StuckConfig(), classify), frames)
    assert max(s[1][0] for _, s in speed) < 2.5, "the speed clock should have been restarted"
    t_end, last = both[-1]
    stuck, still, _ = last[1]
    assert still == pytest.approx(t_end, abs=0.05)
    assert stuck == pytest.approx(t_end - DT, abs=0.05)  # in zone from the 2nd frame (min_track_age_frames)


def test_stuck_clock_survives_an_id_switch_the_rekeying_cannot_match():
    """Not only still_seconds: the ALARM clock (stuck_seconds) keeps the stop and the in-zone
    streak that began at t = 0, because both belong to the cluster the new id now owns."""
    out = run(StuckVehicleTracker(StuckConfig(), classify), id_switch_frames())
    t_end, last = out[-1]
    assert set(last) == {2}
    assert last[2][1] == pytest.approx(t_end, abs=0.05)  # still since the car stopped at t = 0
    assert last[2][0] == pytest.approx(t_end - DT, abs=0.05)  # stuck since the 2nd frame, not since the switch
    speed_only = run(StuckVehicleTracker(StuckConfig(stationarity="speed"), classify), id_switch_frames())
    assert speed_only[-1][1][2][0] < 3.1


def test_moving_vehicle_still_time_is_bounded():
    """A box moving at v px/s keeps IoU >= 0.5 with its own anchor for at most (w/3)/v s; as a
    guest passing a parked vehicle's anchor it gets at most (2w/3)/v s, never the parked
    vehicle's age."""
    for v, w in ((200.0, 60.0), (60.0, 120.0), (30.0, 120.0)):
        tr = StuckVehicleTracker(StuckConfig(), classify)
        worst = 0.0
        for i in range(int(4 * FPS)):
            t = i * DT
            objs = tr.update([det(box(110 + v * t, 260, w=w))], zone(), t)
            worst = max(worst, objs[0].still_seconds)
        assert worst <= (w / 3.0) / v + DT + 1e-6, (v, w, worst)
    # ... which is also why a very slow, very long vehicle can look stopped: documented, not hidden
    assert (120.0 / 3.0) / 10.0 >= StuckConfig().warning_seconds
    # A guest: car 2 drives through (in image space) the box of car 1, parked since t = 0.
    v, w = 40.0, 120.0
    tr = StuckVehicleTracker(StuckConfig(), classify)
    worst = 0.0
    for i in range(int(10 * FPS)):
        t = i * DT
        dets = [det(box(320, 260), tid=1)]
        x = 120 + v * max(0.0, t - 2.0)
        if t >= 2.0 and x < 520:
            dets.append(det(box(x, 262), tid=2))
        objs = {o.track_id: o for o in tr.update(dets, zone(), t)}
        if 2 in objs:
            worst = max(worst, objs[2].still_seconds, objs[2].stuck_seconds)
    assert worst <= (2.0 * w / 3.0) / v + DT + 1e-6, worst
    assert objs[1].still_seconds == pytest.approx(t, abs=1e-6)  # the parked car keeps its stop


@pytest.mark.parametrize("v,gap", [(100.0, 10.0), (100.0, 30.0), (150.0, 30.0), (70.0, 10.0)])
def test_a_dense_moving_queue_never_inherits_a_stop(v, gap):
    """Bumper-to-bumper traffic that never stops: each car stays within the guest bound; it
    must not inherit the age of the cluster the car ahead started (the designer's hand-over
    gave 3.6-5.5 s, an ALARM at 3 s)."""
    w = 120.0
    z = zone([[100, 150], [540, 150], [540, 330], [100, 330]])
    tr = StuckVehicleTracker(StuckConfig(), classify)
    worst = 0.0
    for i in range(int(20 * FPS)):
        t = i * DT
        dets = []
        for k in range(100):
            x1 = -w - k * (w + gap) + v * t
            if x1 > 640:
                continue
            if x1 + w < 0:
                break
            dets.append(det([x1, 220.0, x1 + w, 300.0], tid=k + 1))
        worst = max([worst] + [o.stuck_seconds for o in tr.update(dets, z, t)])
    assert worst <= (2.0 * w / 3.0) / v + 2 * DT, worst


def test_predicted_boxes_and_other_categories_never_feed_clusters():
    tr = StuckVehicleTracker(StuckConfig(lost_track_grace_seconds=2.0), classify)
    for i in range(int(2 * FPS)):
        tr.update([det(box(320, 260))], zone(), i * DT)
    t = 2.0
    while t < 3.5:  # detector skipped: only the tracker's own predictions
        t += DT
        objs = tr.update(tr.predict_boxes(t), zone(), t, detector_ran=False)
    assert objs and objs[0].still_seconds == 0.0  # the cluster died 1 s after the last real box
    tr2 = StuckVehicleTracker(StuckConfig(), classify)
    for i in range(int(4 * FPS)):
        objs = tr2.update([det(box(320, 260), tid=1, cls="person"), det(box(200, 200), tid=2, cls="train")],
                          zone(), i * DT)
    assert all(o.still_seconds == 0.0 for o in objs)
    assert tr2._clusters.active == []  # noqa: SLF001 - nothing but vehicles is clustered


def test_still_is_frame_rate_independent():
    res = {}
    for fps in (5.0, 30.0):
        tr = StuckVehicleTracker(StuckConfig(stationarity="still"), classify)
        n = int(round(5 * fps)) + 1
        for i in range(n):
            objs = tr.update([det(box(320, 260))], zone(), i / fps)
        res[fps] = (objs[0].still_seconds, objs[0].stuck_seconds)
    assert res[5.0][0] == pytest.approx(res[30.0][0]) == pytest.approx(5.0)
    assert abs(res[5.0][1] - res[30.0][1]) <= 0.2 + 1e-9  # min_track_age_frames costs one frame


@pytest.mark.parametrize("bad", [{"stationarity": "ids"}, {"still_min_iou": 0.0}, {"still_min_iou": 1.5},
                                 {"still_max_gap_seconds": 0.0}, {"still_border_margin": -0.01},
                                 {"still_border_margin": 0.5}])
def test_stationarity_and_still_keys_are_validated(bad):
    with pytest.raises(ValueError):
        StuckVehicleTracker(StuckConfig(**bad), classify)


PICKUP = np.array([-1, 219, 107, 305], np.float32)
CAR_NEXT_TO_IT = np.array([-9, 221, 151, 302], np.float32)
# Both real clip05 boxes touch the left image edge, so by default (still_border_margin) their
# clusters never count as still; the two ownership tests below switch that off.
NO_BORDER = {"still_border_margin": 0.0}


def test_a_second_vehicle_does_not_inherit_a_visible_vehicles_stop():
    """clip05, t = 9.21 s: a car pulled up next to a pickup that had stood there since t = 0;
    its box overlapped the pickup's first box with IoU 0.65 and, before cluster ownership,
    was reported stationary for 9.2 s on its first frame there. Here the car is first seen
    at that spot (born in the cluster): it takes the spot over only once the pickup is gone."""
    tr = StuckVehicleTracker(StuckConfig(**NO_BORDER), classify)
    for i in range(int(9 * FPS)):
        tr.update([det(PICKUP, tid=2, cls="truck")], zone(), i * DT)
    t = 9 * FPS * DT
    for k in range(5):
        objs = tr.update([det(PICKUP, tid=2, cls="truck"), det(CAR_NEXT_TO_IT, tid=1)], zone(), t + k * DT)
    by = {o.track_id: o for o in objs}
    assert by[2].still_seconds == pytest.approx(t + 4 * DT, abs=1e-6)  # the owner keeps its stop
    assert by[1].still_seconds <= 4 * DT + 1e-6  # the newcomer starts from its own arrival
    # once the pickup is gone, the car (born in the cluster) takes the spot over and inherits it
    for k in range(5, 10):
        objs = tr.update([det(CAR_NEXT_TO_IT, tid=1)], zone(), t + k * DT)
    assert {o.track_id: o for o in objs}[1].still_seconds == pytest.approx(t + 9 * DT, abs=1e-6)


def test_a_vehicle_that_drove_into_the_spot_never_inherits_the_stop():
    """The same pickup; the car drives in from the right (born outside the cluster), stops on
    the pickup's spot, and the pickup leaves: the car stays a guest and keeps only its own time."""
    tr = StuckVehicleTracker(StuckConfig(**NO_BORDER), classify)
    for i in range(int(9 * FPS)):
        tr.update([det(PICKUP, tid=2, cls="truck")], zone(), i * DT)
    t0 = 9 * FPS * DT
    k = 0
    arrived = None
    while True:  # 300 px/s approach from x = +300 px to the stop
        t = t0 + k * DT
        dx = max(0.0, 300.0 - 300.0 * k * DT)
        car = CAR_NEXT_TO_IT + np.array([dx, 0, dx, 0], np.float32)
        dets = [det(car, tid=1)] + ([det(PICKUP, tid=2, cls="truck")] if t < t0 + 3.0 else [])
        objs = {o.track_id: o for o in tr.update(dets, zone(), t)}
        if dx == 0.0 and arrived is None:
            arrived = t
        k += 1
        if t >= t0 + 6.0:
            break
    car_state = objs[1]
    assert 2 not in objs or objs[2].last_seen < t0 + 3.0
    assert car_state.still_seconds <= (t - arrived) + 0.4 + 1e-6  # its own time there (+ the approach in the window)
    assert car_state.still_seconds < 7.0  # the pickup's stop would be 15 s by now


# ----------------------------------------------------------------------------- image border
EDGE_BAND = [[0, 207], [640, 203], [640, 224], [0, 228]]  # clip06's zone: a band across the whole picture


def edge_run(inset: float = 0.0, **stuck):
    """review issue 2, clip06 62.5 s: the visible end of a passenger train passing the right edge
    of the picture, labelled 'bus' (box about [522, 130, 640, 225]). Its box breathes like the real
    one (left edge +-8 px, bottom +-6 px), keeping IoU ~0.8 with its first box, while the train's
    own box comes and goes beside it. ``inset`` moves everything left, away from the border.
    Returns the first ALARM time and the largest still / speed-stationary times of the 'bus'."""
    cfg = StuckConfig(**stuck)
    tr = StuckVehicleTracker(cfg, classify)
    eng = AlertEngine(cfg, AlertConfig(stdout=False, repeat_notify_seconds=0.0))
    z = zone(EDGE_BAND)
    first, still, stationary = None, 0.0, 0.0
    for i in range(int(12 * FPS)):
        t = i * DT
        x1 = 523.0 + 8.0 * math.sin(2 * math.pi * t / 2.0)
        y2 = 219.0 + 6.0 * math.sin(2 * math.pi * t / 1.7 + 1.0)
        dets = [det(np.array([x1 - inset, 130.0, 640.0 - inset, y2], np.float32), tid=1483, cls="bus")]
        if int(t / 0.7) % 2 == 0:
            dets.append(det(np.array([30.0, 135.0, 600.0 - inset, 226.0], np.float32), tid=1513, cls="train"))
        objs = {o.track_id: o for o in tr.update(dets, z, t)}
        still = max(still, objs[1483].still_seconds)
        stationary = max(stationary, objs[1483].stationary_seconds)
        a, _ = eng.evaluate(list(objs.values()), None, GateState(), t, i)
        if a.status is Status.ALARM and first is None:
            first = t
    return first, still, stationary


def test_a_box_cut_by_the_image_border_is_never_still():
    """The train end at the frame edge raised a 3 s ALARM through the still clock (f1aee43; 11 s
    of false ALARM on clip06). A cluster that takes a box within still_border_margin of the border
    never counts as still, and the speed rule - restarted by the breathing box, as on base - decides
    alone."""
    first, still, stationary = edge_run()
    assert first is None and still == 0.0 and stationary < 1.0
    first_old, still_old, _ = edge_run(still_border_margin=0.0)
    assert first_old == pytest.approx(3.0 + DT, abs=DT) and still_old > 11.0  # the f1aee43 behaviour
    # the same breathing box 30 px inside the picture is a stopped vehicle (clip01's SUV): it alarms
    first_in, still_in, _ = edge_run(inset=30.0)
    assert first_in == pytest.approx(3.0 + DT, abs=DT) and still_in > 11.0


CLIP07_ZONE = [[212, 130], [252, 130], [445, 300], [330, 300]]


def entry_run(v0: float, t_stop: float, x_start: float, margin: float):
    """A 150 px van drives in from the right image edge (its box cut by the border at first),
    brakes evenly from ``v0`` px/s and stands still from ``t_stop`` on, inside clip07's zone.
    Returns the first WARNING and ALARM times and the (t, still, stuck) trace of the van."""
    cfg = StuckConfig(still_border_margin=margin)
    tr = StuckVehicleTracker(cfg, classify)
    eng = AlertEngine(cfg, AlertConfig(stdout=False, repeat_notify_seconds=0.0))
    z = zone(CLIP07_ZONE)
    first_w = first_a = None
    trace = []
    for i in range(int(8 * FPS)):
        t = i * DT
        tt = min(t, t_stop)
        x1 = x_start - (v0 * tt - 0.5 * (v0 / t_stop) * tt * tt)
        b = np.array([x1, 145.0, min(640.0, x1 + 150.0), 223.0], np.float32)
        objs = tr.update([det(b, tid=12, cls="truck")], z, t)
        a, _ = eng.evaluate(objs, None, GateState(), t, i)
        if a.status is Status.WARNING and first_w is None:
            first_w = t
        if a.status is Status.ALARM and first_a is None:
            first_a = t
        trace.append((t, objs[0].still_seconds, objs[0].stuck_seconds))
    return first_w, first_a, trace


@pytest.mark.parametrize("v0,t_stop,x_start", [(115.0, 4.0, 540.0), (120.0, 4.5, 520.0), (115.0, 3.5, 540.0)])
def test_a_vehicle_that_drives_in_from_the_border_stops_like_one_that_did_not(v0, t_stop, x_start):
    """clip07: the trapped van came in through the right image edge. e9e6f86 kept its cut boxes out
    of the clusters, which moved every later anchor and shifted the still clock of its stop (WARNING
    1.8 -> 2.24 s after the event start; up to 0.67 s either way in this scenario). The cut boxes
    always feed the clusters, and with no train beside them they are ordinary boxes: the whole run
    is the one without the border rule. (Until the final review the clusters that took a cut box
    lost their still time for good, train or not - this test asserted that; see the full-width
    zone test below for what it cost.)"""
    first_w, first_a, trace = entry_run(v0, t_stop, x_start, margin=0.01)
    ref_w, ref_a, ref = entry_run(v0, t_stop, x_start, margin=0.0)
    assert first_w is not None and first_a is not None
    assert (first_w, first_a) == (ref_w, ref_a)
    assert trace[0][1] == 0.0 and ref[0][1] == 0.0  # (no still time on the first frame either way)
    assert trace == ref
