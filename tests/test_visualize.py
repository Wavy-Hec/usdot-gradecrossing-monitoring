"""Tests for gradecrossing.visualize.draw_overlay on hand-built FrameResults.

The overlay is cosmetic, so the tests only pin the contract that matters: it runs on
every combination of inputs the monitor can produce (no detections, predicted boxes,
debris, held zone, unavailable gate, grayscale frame), returns a *copy* of the same
shape, and uses the engine's per-id severity (not its own thresholds) for colours.
"""
from __future__ import annotations

import cv2
import numpy as np

from gradecrossing.geometry import polygon_mask
from gradecrossing.tracking import PredictedDetection
from gradecrossing.types import (
    Assessment,
    DebrisResult,
    Detection,
    FrameResult,
    GateState,
    ObjectState,
    Status,
    ZoneState,
)
from gradecrossing.visualize import COLOR_ALARM, COLOR_CLEAR, COLOR_OTHER, COLOR_WARNING, draw_overlay, object_color, object_label

H, W = 360, 640
POLY = np.array([[200, 150], [440, 150], [440, 270], [200, 270]], dtype=np.int32)


def make_zone(healthy: bool = True, method: str = "lk") -> ZoneState:
    mask = polygon_mask((H, W), POLY)
    return ZoneState(polygon=POLY.copy(), mask=mask, homography=None, inliers=120 if healthy else 3,
                     method=method, healthy=healthy, area_px=int(cv2.countNonZero(mask)))


def make_obj(tid: int, cls: str = "car", vehicle: bool = True, person: bool = False, train: bool = False,
             stuck: float = 0.0, speed: float = 40.0, in_zone: bool = True) -> ObjectState:
    return ObjectState(track_id=tid, cls_name=cls, is_vehicle=vehicle, is_person=person, is_train=train,
                       last_box=np.array([300, 200, 360, 240], np.float32), last_seen=1.0, first_seen=0.0,
                       in_zone=in_zone, in_zone_since=0.0 if in_zone else None, dwell_seconds=1.0 if in_zone else 0.0,
                       speed=speed, speed_units="px/s", stationary_seconds=stuck, stuck_seconds=stuck, history=[])


def make_det(tid, cls="car", predicted=False) -> Detection:
    xyxy = np.array([300, 200, 360, 240], np.float32)
    if predicted:
        return PredictedDetection(xyxy=xyxy, conf=0.9, cls_id=0, cls_name=cls, track_id=tid)
    return Detection(xyxy=xyxy, conf=0.9, cls_id=0, cls_name=cls, track_id=tid)


def make_assessment(status: Status = Status.CLEAR, warn=(), alarm=(), reasons=(), debris=False,
                    latched=None) -> Assessment:
    return Assessment(status=status, reasons=list(reasons), warning_track_ids=list(warn),
                     alarm_track_ids=list(alarm), debris_alarm=debris, person_alert=False, gates_down=False,
                     latched_until=latched, severity_scale=1.0)


def make_result(**kw) -> FrameResult:
    base = dict(frame_index=12, timestamp=0.4, assessment=make_assessment(), zone=make_zone(),
                detections=[make_det(1)], objects=[make_obj(1)], debris=None, gate=GateState(available=False),
                timings_ms={"total": 4.2, "zone": 2.0}, events=[])
    base.update(kw)
    return FrameResult(**base)


def frame(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.clip(rng.normal(120, 20, (H, W, 3)), 0, 255).astype(np.uint8)


def test_overlay_returns_copy_same_shape():
    f = frame()
    before = f.copy()
    out = draw_overlay(f, make_result(), fps_estimate=25.0)
    assert out.shape == f.shape and out.dtype == np.uint8
    assert out is not f
    assert np.array_equal(f, before)  # the input frame is never modified
    assert not np.array_equal(out, f)  # something was drawn


def test_overlay_handles_every_input_combination():
    f = frame(1)
    debris = DebrisResult(mask=np.zeros((H, W), np.uint8), area_px=900, area_fraction=0.031,
                          blobs=[(250, 180, 30, 30, 900)], persist_seconds=1.2, triggered=False,
                          learning_rate_used=0.0, warmed_up=True)
    cases = [
        make_result(detections=[], objects=[]),
        make_result(detections=[make_det(1, predicted=True)]),
        make_result(debris=debris, assessment=make_assessment(Status.ALARM, alarm=[1], reasons=["debris 3.1% of zone for 3.2s"], debris=True)),
        make_result(zone=make_zone(healthy=False, method="held")),
        make_result(gate=GateState(gates_down=True, lights_on=True, confidence=0.8, available=True)),
        make_result(assessment=make_assessment(Status.WARNING, warn=[1], reasons=["vehicle 1 (car) stationary in zone 3.2s"])),
        make_result(assessment=make_assessment(Status.ALARM, reasons=["latched"], latched=9.0)),
        make_result(detections=[make_det(None)], objects=[]),  # untracked detection
        make_result(detections=[make_det(7, cls="train")], objects=[make_obj(7, "train", vehicle=False, train=True)]),
        make_result(timings_ms={}),
    ]
    for res in cases:
        out = draw_overlay(f, res)
        assert out.shape == f.shape
    # very long reason list is truncated, not crashing
    long = make_result(assessment=make_assessment(Status.ALARM, alarm=[1], reasons=["x" * 500]))
    assert draw_overlay(f, long).shape == f.shape


def test_overlay_accepts_grayscale_frame():
    g = cv2.cvtColor(frame(2), cv2.COLOR_BGR2GRAY)
    out = draw_overlay(g, make_result())
    assert out.shape == (H, W, 3)


def test_object_colors_follow_engine_ids_not_thresholds():
    obj = make_obj(1, stuck=100.0)  # a huge stuck time by itself does not colour red
    assert object_color(obj, warning_ids=[], alarm_ids=[]) == COLOR_CLEAR
    assert object_color(obj, warning_ids=[1], alarm_ids=[]) == COLOR_WARNING
    assert object_color(obj, warning_ids=[1], alarm_ids=[1]) == COLOR_ALARM
    train = make_obj(2, "train", vehicle=False, train=True)
    assert object_color(train, [], []) == COLOR_OTHER
    assert object_color(None, [], []) == COLOR_OTHER


def test_object_label_format():
    obj = make_obj(3, stuck=2.34, speed=5.6)
    label = object_label(obj, "car", 3)
    assert label.startswith("3 car")
    assert "px/s" in label and "2.3s" in label
    assert object_label(None, "bus", None) == "? bus"
    assert object_label(None, "bus", 9) == "9 bus"
    label.encode("ascii")  # labels are ASCII


def test_zone_polygon_colour_reflects_health():
    f = np.zeros((H, W, 3), np.uint8)
    ok = draw_overlay(f, make_result(detections=[], objects=[]))
    held = draw_overlay(f, make_result(detections=[], objects=[], zone=make_zone(healthy=False, method="held")))
    # sample a point on the left edge of the polygon, away from the text
    x, y = 200, 250
    assert tuple(int(v) for v in ok[y, x]) == COLOR_CLEAR
    assert tuple(int(v) for v in held[y, x]) == COLOR_ALARM
