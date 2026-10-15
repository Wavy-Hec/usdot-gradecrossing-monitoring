"""Monitor wiring of the debris backstop (process_frame step 6), with a scripted detector.

What is pinned here and not in the detector's own tests: which boxes the monitor passes as
exclusions (vehicles, persons and trains - not unmapped classes, but a track keeps its
exclusion for a few seconds after its last mapped label), the train hold, registration of
the frame through the zone stabiliser (across keyframe re-anchors), the ignore polygons,
and the HEALTH events / status text for the debris state.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pytest

from gradecrossing.alerting import CallbackSink
from gradecrossing.config import DetectorConfig, MonitorConfig
from gradecrossing.monitor import ConfigClassifier, Monitor
from gradecrossing.types import Detection, Event, FrameResult, Status

H, W = 360, 640
FPS = 30.0
ZONE = [[200, 150], [440, 150], [440, 270], [200, 270]]
BLOB = (300, 190, 360, 230)  # 60x40 obstruction inside the zone


class Scene:
    """Textured static background (corners for the stabiliser) + painted rectangles."""

    def __init__(self, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        base = cv2.GaussianBlur(rng.normal(120.0, 40.0, size=(H, W)).astype(np.float32), (0, 0), 1.5)
        base = cv2.merge([base, base * 0.95, base * 1.05])
        for _ in range(40):
            x, y = int(rng.integers(0, W - 40)), int(rng.integers(0, H - 30))
            if 150 < x < 500 and 100 < y < 300:
                continue
            base[y:y + 25, x:x + 35] = rng.uniform(30, 220, size=3)
        for x in (215, 275, 335, 395):  # crossing markings inside the zone (static background)
            base[158:180, x:x + 30] = 215.0
        self.base = np.clip(base, 0, 255).astype(np.float32)
        self.noise = rng.normal(0.0, 2.5, size=(8, H, W, 3)).astype(np.float32)
        self.calls = 0

    def frame(self, rects: Sequence[Tuple[Tuple[int, int, int, int], Tuple[float, float, float]]] = (),
              shift: Tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
        img = self.base.copy()
        for (x1, y1, x2, y2), color in rects:
            img[y1:y2, x1:x2] = color
        if shift != (0.0, 0.0):
            m = np.float32([[1, 0, shift[0]], [0, 1, shift[1]]])
            img = cv2.warpAffine(img, m, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        img = img + self.noise[self.calls % 8]
        self.calls += 1
        return np.clip(img, 0, 255).astype(np.uint8)


class Scripted:
    """detect() returns script(index); the class mapping is the default config's."""

    def __init__(self, script: Callable[[int], List[Detection]]) -> None:
        self.script = script
        self.index = 0
        self._classify = ConfigClassifier(DetectorConfig())

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        return self.script(self.index)

    def classify(self, cls_name: str) -> str:
        return self._classify(cls_name)


def det(box: Sequence[float], cls: str = "car", tid: Optional[int] = 1) -> Detection:
    return Detection(xyxy=np.asarray(box, dtype=np.float32), conf=0.8, cls_id=0, cls_name=cls, track_id=tid)


def make_cfg(**sections) -> MonitorConfig:
    cfg = MonitorConfig()
    cfg.zone.polygon = [list(p) for p in ZONE]
    cfg.alerts.stdout = False
    cfg.alerts.latch_seconds = 1.0
    cfg.alerts.repeat_notify_seconds = 0.0
    cfg.debris.warmup_seconds = 1.0
    cfg.debris.persist_seconds = 1.0
    for section, fields in sections.items():
        for k, v in fields.items():
            setattr(getattr(cfg, section), k, v)
    return cfg


class Session:
    def __init__(self, cfg: MonitorConfig, script: Callable[[int], List[Detection]] = lambda i: [],
                 paint: Optional[Callable[[int], list]] = None,
                 shift: Optional[Callable[[int], Tuple[float, float]]] = None) -> None:
        self.scene = Scene(0)
        self.fake = Scripted(script)
        self.paint = paint
        self.shift = shift
        self.events: List[Event] = []
        self.results: List[FrameResult] = []
        self.mon = Monitor(cfg, detector=self.fake, sinks=[CallbackSink(self.events.append)])
        self.i = 0

    def run(self, seconds: float) -> List[FrameResult]:
        for _ in range(int(round(seconds * FPS))):
            i = self.i
            self.fake.index = i
            frame = self.scene.frame(self.paint(i) if self.paint else (),
                                     self.shift(i) if self.shift else (0.0, 0.0))
            self.results.append(self.mon.process_frame(frame, i / FPS, i))
            self.i += 1
        return self.results

    def debris_alarm_starts(self) -> List[float]:
        return [e.timestamp for e in self.events
                if e.type == "ALARM_START" and any(r.startswith("debris") for r in e.reasons)]

    def debris_health(self) -> List[Event]:
        return [e for e in self.events if e.type == "HEALTH" and e.extra.get("component") == "debris"]


DARK = (45.0, 40.0, 50.0)


def blob_from(t0: float, rect=BLOB, color=DARK) -> Callable[[int], list]:
    return lambda i: [(rect, color)] if i >= int(t0 * FPS) else []


def test_unmapped_class_obstruction_raises_a_debris_alarm():
    """M2: a cow (or suitcase, bench ...) that YOLO names with an unmapped class used to be
    excluded from the debris mask on every frame while the alert engine skips it: no alarm
    at all. Only vehicle / person / train boxes are exclusions now."""
    script = (lambda i: [det(BLOB, "cow", 9001)] if i >= 90 else [])
    s = Session(make_cfg(), script, paint=blob_from(3.0))
    s.run(6.0)
    starts = s.debris_alarm_starts()
    assert starts and 3.0 + 1.0 <= starts[0] <= 3.0 + 1.6
    # the same object boxed as a car every frame is the tracker's business, not debris
    s2 = Session(make_cfg(), lambda i: [det(BLOB, "car", 9001)] if i >= 90 else [], paint=blob_from(3.0))
    s2.run(6.0)
    assert not s2.debris_alarm_starts()


def test_sporadically_detected_obstruction_still_raises_a_debris_alarm():
    """M2: boxed as 'car' once every 45 frames - every box used to restart the debris clock."""
    script = (lambda i: [det(BLOB, "car", None)] if i >= 90 and i % 45 == 0 else [])
    s = Session(make_cfg(), script, paint=blob_from(3.0))
    s.run(6.0)
    starts = s.debris_alarm_starts()
    assert starts and starts[0] <= 3.0 + 1.6


def test_vehicle_relabelled_with_an_unmapped_class_stays_excluded():
    """A car waiting with its box over the zone but its bottom-centre outside (not in the
    zone for the tracker) is labelled 'boat' on every third frame, as YOLO does with vans
    and cars. Its track id had a vehicle label moments ago, so it stays an exclusion."""
    car = (300, 230, 380, 300)  # bottom edge below the zone (y = 270)
    script = (lambda i: [det(car, "boat" if i % 3 == 0 else "car", 7)] if i >= 60 else [])
    paint = (lambda i: [(car, DARK)] if i >= 60 else [])
    s = Session(make_cfg(), script, paint=paint)
    res = s.run(10.0)
    assert not s.debris_alarm_starts()
    assert max(r.debris.persist_seconds for r in res if r.debris is not None) < 0.5


def test_train_over_the_zone_holds_the_debris_timers():
    """A train detection overlapping the zone sets the hold: timers freeze, nothing new becomes
    a candidate, nothing is learned (an object first seen under a passing train starts its
    clock only after the train)."""
    train_box = (150, 120, 500, 280)
    script = (lambda i: [det(train_box, "train", 50)] if 60 <= i < 150 else [])
    paint = (lambda i: ([(BLOB, DARK)] if i >= 90 else []))
    cfg = make_cfg(debris={"hold_linger_seconds": 0.5})
    s = Session(cfg, script, paint=paint)
    res = s.run(8.0)
    held = [r for r in res if r.debris is not None and r.debris.held]
    assert held and all(r.debris.state == "held (train)" for r in held)
    assert held[0].timestamp == pytest.approx(2.0, abs=0.05)
    assert held[-1].timestamp == pytest.approx(5.5, abs=0.1)  # 1 s train ends at 5.0 s + linger
    starts = s.debris_alarm_starts()
    assert starts and starts[0] >= 5.5 + 1.0 - 0.05


def test_camera_shift_tracked_by_the_stabiliser_raises_no_debris():
    """F05: a permanent (8, 4) px bump at 3 s. The stabiliser tracks it; the debris frame is
    registered with the inverse homography, so no debris. With stabilisation off the same
    bump becomes debris (the old behaviour, now only without registration)."""
    shift = (lambda i: (8.0, 4.0) if i >= 90 else (0.0, 0.0))
    s = Session(make_cfg(), shift=shift)
    res = s.run(8.0)
    assert not s.debris_alarm_starts()
    assert all(r.debris.registered for r in res if r.debris is not None)
    assert max(r.debris.area_fraction for r in res[100:]) < 0.02
    # (ghost healing off: it would re-learn the half of the displaced markings that is
    # visible only in the model, which hides part of the contrast)
    s2 = Session(make_cfg(zone={"stabilize": False}, debris={"ghost_heal_seconds": 0.0}), shift=shift)
    s2.run(8.0)
    assert s2.debris_alarm_starts()


def test_registration_survives_keyframe_reanchors():
    """The stabiliser re-anchors its keyframe after a static period; the debris model must
    keep being registered to the frame it was learned in (the cumulative transform), or a
    bump before the re-anchor would turn into debris after it."""
    shift = (lambda i: (6.0, 3.0) if i >= 60 else (0.0, 0.0))
    cfg = make_cfg(zone={"keyframe_refresh_static_seconds": 1.0})
    s = Session(cfg, shift=shift)
    s.run(9.0)
    assert s.mon.stabilizer.health()["keyframe_age_s"] < 3.0  # it did re-anchor after the bump
    assert not s.debris_alarm_starts()
    blob = Session(cfg, shift=shift, paint=blob_from(6.0))
    blob.run(9.0)
    assert blob.debris_alarm_starts()  # and a real object after the re-anchor still alarms


def test_ignore_polygons_are_scaled_with_the_frame_and_applied():
    """debris.ignore_polygons are calibration pixels, like zone.polygon: with resize_width
    they are scaled by the same factor. An object resting inside one never alarms."""
    ignore = [[[290, 180], [370, 180], [370, 240], [290, 240]]]  # around BLOB, native pixels
    cfg = make_cfg(video={"resize_width": 320}, debris={"ignore_polygons": ignore})
    s = Session(cfg, paint=blob_from(2.0))
    s.run(5.0)
    assert not s.debris_alarm_starts()
    s2 = Session(make_cfg(video={"resize_width": 320}), paint=blob_from(2.0))
    s2.run(5.0)
    assert s2.debris_alarm_starts()


def test_debris_health_events_and_status_text():
    """HEALTH events for the debris state: warm-up start, warm-up complete (with the unknown
    share), and trusted flips; the periodic status line shows the warm-up instead of
    'debris=0.0%'."""
    bus = (190, 140, 450, 280)  # covers the whole zone during the warm-up
    script = (lambda i: [det(bus, "bus", 3)] if i < 45 else [])
    paint = (lambda i: [(bus, (200.0, 180.0, 40.0))] if i < 45 else [])
    cfg = make_cfg(debris={"update_time_constant_seconds": 4.0, "learn_seconds": 0.5})
    s = Session(cfg, script, paint=paint)
    res = s.run(10.0)
    assert "debris=warmup" in Monitor._status_line(res[5], None)
    health = s.debris_health()
    texts = [e.reasons[0] for e in health]
    assert texts[0] == "debris detection inactive until the background warm-up completes (1.0 s)"
    assert health[0].timestamp == 0.0 and health[0].extra["warming_up"] is True
    assert texts[1].startswith("debris detection active (warm-up complete; 100%")
    assert "untrusted" in texts[1] and health[1].extra["trusted"] is False
    assert health[1].timestamp == pytest.approx(1.0, abs=0.05)
    assert texts[-1].startswith("debris detection trusted again")
    assert res[-1].debris.trusted and res[-1].debris.unknown_fraction == 0.0
    assert "debris=0.0%" in Monitor._status_line(res[-1], None)
    assert not s.debris_alarm_starts()
    assert Status.ALARM not in {r.assessment.status for r in res}


def test_debris_warmup_restart_is_reported_once_per_restart():
    """A silent re-warm (the detector rebuilt its model because the frame geometry or the
    zone's region changed) is reported like the first warm-up: debris is blind until the new
    warm-up completes, and the operator must know. One event per restart, not per frame."""
    s = Session(make_cfg())
    s.run(2.0)
    starts = [e for e in s.debris_health() if e.extra.get("warming_up")]
    assert [e.timestamp for e in starts] == [0.0]
    s.mon.debris.reset()  # what update() does on a geometry or region change
    s.run(2.0)
    health = s.debris_health()
    starts = [e for e in health if e.extra.get("warming_up")]
    assert [e.timestamp for e in starts] == [0.0, pytest.approx(2.0)]
    texts = [e.reasons[0] for e in health]
    assert sum(t.startswith("debris detection active (warm-up complete") for t in texts) == 2
    assert s.results[-1].debris.warmed_up
