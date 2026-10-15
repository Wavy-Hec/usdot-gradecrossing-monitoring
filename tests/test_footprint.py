"""Footprint zone relation, the "too close" region and the in-zone exit debounce
(team EXPERIMENTS.md E4/E6; stuck.zone_test="footprint").

Geometry is taken from the team clips so the tests pin the cases that motivated the port:
clip08's thin oblique zone with the trapped van's YOLO11n box at t = 3.30 s (bottom-center
outside the zone, bottom-quarter footprint 41 % inside), and clip06's foreground car waiting
at the gate 0.5 box heights in front of a thin horizontal zone.
"""
from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from gradecrossing.config import DetectorConfig, StuckConfig
from gradecrossing.geometry import footprint_box, footprint_relation, polygon_mask, rect_polygon_distance
from gradecrossing.tracking import StuckVehicleTracker
from gradecrossing.types import Detection, ZoneState

FPS = 30.0
DT = 1.0 / FPS
SHAPE = (360, 640)
CLIP08_ZONE = [[268, 110], [303, 110], [520, 300], [415, 300]]
CLIP08_VAN = [245.0, 117.0, 402.0, 199.0]  # YOLO11n "truck", clip08 t = 3.30 s
CLIP06_ZONE = [[0, 207], [640, 203], [640, 224], [0, 228]]
CLIP06_CAR = [47.3, 197.3, 318.5, 322.3]  # median box of the car waiting at the gate


def classify(name: str) -> str:
    cfg = DetectorConfig()
    n = name.lower()
    if n in {c.lower() for c in cfg.vehicle_classes}:
        return "vehicle"
    if n in {c.lower() for c in cfg.person_classes}:
        return "person"
    return "train" if n in {c.lower() for c in cfg.train_classes} else "other"


def zone_state(poly, shape=SHAPE) -> ZoneState:
    p = np.asarray(poly, dtype=np.int32)
    mask = polygon_mask(shape, p)
    return ZoneState(polygon=p, mask=mask, homography=None, inliers=0, method="identity", healthy=True,
                     area_px=int(cv2.countNonZero(mask)))


def det(box, cls="car", tid=1) -> Detection:
    return Detection(xyxy=np.asarray(box, dtype=np.float32), conf=0.9, cls_id=0, cls_name=cls, track_id=tid)


def box_at(cx: float, by: float, w: float = 120.0, h: float = 80.0) -> np.ndarray:
    return np.array([cx - w / 2, by - h, cx + w / 2, by], np.float32)


def dt_reference(box, poly, fraction=0.25):
    """The team's definition (pipelines/common/geometry.py Zone.relation): footprint pixels
    on the mask, and min of a cv2.distanceTransform over the footprint / box height."""
    h, w = SHAPE
    mask = polygon_mask(SHAPE, np.asarray(poly, np.float32))
    dist = cv2.distanceTransform(255 - mask, cv2.DIST_L2, 5)
    x1, y1, x2, y2 = box
    bh = max(1.0, y2 - y1)
    fy1 = int(np.clip(y2 - fraction * bh, 0, h - 1))
    fy2 = int(np.clip(y2, fy1 + 1, h))
    fx1 = int(np.clip(x1, 0, w - 1))
    fx2 = int(np.clip(x2, fx1 + 1, w))
    return float((mask[fy1:fy2, fx1:fx2] > 0).mean()), float(dist[fy1:fy2, fx1:fx2].min()) / bh


# ----------------------------------------------------------------------------- geometry
def test_footprint_box_is_the_bottom_quarter():
    fp = footprint_box([10, 20, 110, 100])
    assert fp.dtype == np.float32
    assert fp.tolist() == [10.0, 80.0, 110.0, 100.0]
    assert footprint_box([10, 20, 110, 100], 0.5).tolist() == [10.0, 60.0, 110.0, 100.0]


def test_rect_polygon_distance_exact_cases():
    square = np.array([[100, 100], [200, 100], [200, 200], [100, 200]], np.float32)
    assert rect_polygon_distance([0, 0, 50, 50], square) == pytest.approx(math.hypot(50, 50))
    assert rect_polygon_distance([0, 120, 70, 130], square) == pytest.approx(30.0)
    assert rect_polygon_distance([150, 150, 160, 160], square) == 0.0  # rect inside polygon
    assert rect_polygon_distance([0, 0, 300, 300], square) == 0.0  # polygon inside rect
    # A thin band crossing a footprint strip with no vertex of either inside the other.
    band = np.array([[0, 140], [640, 140], [640, 150], [0, 150]], np.float32)
    assert rect_polygon_distance([300, 130, 340, 160], band) == 0.0
    # Non-convex (L-shaped) zone: inside the notch the nearest boundary is an inner edge.
    ell = np.array([[0, 0], [100, 0], [100, 40], [40, 40], [40, 100], [0, 100]], np.float32)
    assert rect_polygon_distance([60, 60, 90, 90], ell) == pytest.approx(20.0)
    assert rect_polygon_distance([60, 60, 90, 90], ell) == pytest.approx(
        min(-cv2.pointPolygonTest(ell.reshape(-1, 1, 2), c, True) for c in [(60, 60), (90, 60), (60, 90), (90, 90)]))


def test_footprint_relation_matches_the_team_definition_on_random_boxes():
    rng = np.random.default_rng(0)
    zone = zone_state(CLIP08_ZONE)
    worst = 0.0
    for _ in range(400):
        x1, y1 = rng.uniform(0, 600), rng.uniform(0, 330)
        box = [x1, y1, min(639.0, x1 + rng.uniform(8, 160)), min(359.0, y1 + rng.uniform(8, 90))]
        ov, d = footprint_relation(box, zone.polygon, zone.mask)
        ov_ref, d_ref = dt_reference(box, CLIP08_ZONE)
        assert ov == ov_ref  # same pixel block as the team's Zone.relation, so the 5 % means the same
        bh = box[3] - box[1]
        # The reference is approximate: cv2.distanceTransform(DIST_L2, 5) is within ~3 % of
        # Euclid, and it measures from integer pixel indices (int() truncation of the box).
        worst = max(worst, abs(d - d_ref) * bh - 0.03 * d_ref * bh)
    assert worst <= 2.0


# ----------------------------------------------------------------------------- tracker
def test_clip08_van_is_in_the_zone_only_by_its_footprint():
    zone = zone_state(CLIP08_ZONE)
    bx, by = (CLIP08_VAN[0] + CLIP08_VAN[2]) / 2, CLIP08_VAN[3]
    assert cv2.pointPolygonTest(zone.polygon.reshape(-1, 1, 2).astype(np.float32), (bx, by), False) < 0
    ov, d = footprint_relation(CLIP08_VAN, zone.polygon, zone.mask)
    assert 0.35 <= ov <= 0.47 and d == 0.0
    for test, expected in (("footprint", True), ("bottom_center", False)):
        tr = StuckVehicleTracker(StuckConfig(zone_test=test), classify)
        for i in range(3):
            objs = tr.update([det(CLIP08_VAN, cls="truck")], zone, i * DT)
        assert objs[0].in_zone is expected, test
        assert tr.zone_occupied is expected


def test_clip06_waiting_car_is_near_not_in_the_zone():
    zone = zone_state(CLIP06_ZONE)
    ov, d = footprint_relation(CLIP06_CAR, zone.polygon, zone.mask)
    assert ov == 0.0 and 0.45 <= d <= 0.55
    tr = StuckVehicleTracker(StuckConfig(), classify)
    for i in range(3):
        objs = tr.update([det(CLIP06_CAR)], zone, i * DT)
    o = objs[0]
    assert not o.in_zone and o.near_zone
    assert o.zone_distance_bh == pytest.approx(d)
    assert not tr.zone_occupied  # "near" never blocks the background / keyframe refresh
    # close_box_heights below the measured distance, or 0, turns "near" off
    for close in (0.4, 0.0):
        tr2 = StuckVehicleTracker(StuckConfig(close_box_heights=close), classify)
        for i in range(3):
            objs = tr2.update([det(CLIP06_CAR)], zone, i * DT)
        assert not objs[0].near_zone


def test_persons_trains_and_unmapped_classes_keep_the_point_test():
    zone = zone_state(CLIP08_ZONE)
    for cls in ("person", "train", "bench"):
        tr = StuckVehicleTracker(StuckConfig(), classify)
        for i in range(3):
            objs = tr.update([det(CLIP08_VAN, cls=cls)], zone, i * DT)
        assert not objs[0].in_zone and not objs[0].near_zone, cls


def test_bad_zone_test_value_is_rejected():
    with pytest.raises(ValueError):
        StuckVehicleTracker(StuckConfig(zone_test="center"), classify)


@pytest.mark.parametrize("bad", [{"footprint_fraction": 0.0}, {"footprint_fraction": 2.0},
                                 {"footprint_min_overlap": 1.5}, {"footprint_min_overlap": -0.1},
                                 {"close_box_heights": -1.0}, {"zone_exit_grace_seconds": -0.5}])
def test_numeric_zone_keys_are_validated(bad):
    with pytest.raises(ValueError):
        StuckVehicleTracker(StuckConfig(**bad), classify)


# ----------------------------------------------------------------------------- exit debounce
def test_one_pixel_boundary_flicker_does_not_restart_the_stuck_clock():
    """A stopped car whose footprint overlap alternates 5 % / 0 % (its bottom edge jitters by
    1 px on the band edge) must still accumulate stuck time; without the exit debounce every
    out-of-zone frame restarted in_zone_since and the stuck clock (max stuck 0.00 s in 6 s)."""
    z = zone_state([[0, 250], [640, 250], [640, 300], [0, 300]])
    assert footprint_relation([260.0, 171.0, 380.0, 251.0], z.polygon, z.mask)[0] == pytest.approx(0.05)
    assert footprint_relation([260.0, 170.0, 380.0, 250.0], z.polygon, z.mask)[0] == 0.0
    for grace, expect_long in ((0.5, True), (0.0, False)):
        tr = StuckVehicleTracker(StuckConfig(zone_exit_grace_seconds=grace), classify)
        best = 0.0
        for i in range(int(6 * FPS)):
            y2 = (251.0, 250.0)[i % 2]
            o = tr.update([det([260.0, y2 - 80.0, 380.0, y2])], z, i * DT)[0]
            best = max(best, o.stuck_seconds)
        assert (o.stuck_seconds >= 5.5) is expect_long, grace
        if not expect_long:
            assert best < 0.1  # the pre-debounce behaviour: the clock restarts every other frame


def test_a_car_that_really_leaves_stops_being_in_the_zone_within_the_grace():
    z = zone_state([[100, 100], [540, 100], [540, 330], [100, 330]])
    tr = StuckVehicleTracker(StuckConfig(), classify)
    last_in = None
    for i in range(int(6 * FPS)):
        t = i * DT
        cx = 320 + (0 if t < 2.0 else min(400.0, 200.0 * (t - 2.0)))
        if tr.update([det(box_at(cx, 260))], z, t)[0].in_zone:
            last_in = t
    # bare footprint test: the box leaves x = 540 at t = 3.37 s; the grace may add 0.5 s + 1 frame
    assert last_in <= 3.37 + 0.5 + 2 * DT


def test_wide_vehicle_on_a_narrow_zone_is_near_not_in():
    """The one way the footprint test is NOT a superset of the bottom-center test: a 420 px bus
    on a 16 px zone strip (rails along the camera axis) overlaps 4 % < 5 %. It is 'near'
    (WARNING under close_policy='warning'), never in the zone, so it can never ALARM; with the
    bottom-center test it was in the zone. On the team clips the bottom-center-only boxes were
    far-side cars touching the band edge, and OR-ing the point test back in added false alarms."""
    z = zone_state([[300, 0], [316, 0], [316, 360], [300, 360]])
    bus = [100.0, 150.0, 520.0, 300.0]
    tr = StuckVehicleTracker(StuckConfig(), classify)
    for i in range(int(4 * FPS)):
        o = tr.update([det(bus, cls="bus")], z, i * DT)[0]
    assert not o.in_zone and o.near_zone and o.near_stuck_seconds > 3.5
    assert 0.0 < o.zone_overlap < 0.05
    legacy = StuckVehicleTracker(StuckConfig(zone_test="bottom_center"), classify)
    for i in range(3):
        o = legacy.update([det(bus, cls="bus")], z, i * DT)[0]
    assert o.in_zone
