"""Calibration geometry of the zone: reference size, reference image, start-up checks.

Covers ``gradecrossing.zone.build_stabilizer`` and its helpers, the monitor's use of them
(``Monitor._build_stabilizer`` / step 2 of ``process_frame``) and ``tools/calibrate_zone.py``:

* F07 / MISS-01: a live stream that renegotiates its resolution was re-initialised with
  the polygon still in calibration pixels - the zone landed in the wrong place and a car
  stuck on the crossing never alarmed;
* MISS-04: nothing checked that the polygon fits the frame (the built-in default is the
  FOX13 clip's polygon and covers a sliver of a 640x360 frame);
* GAEL-21: the stabilizer was anchored on the first live frame, so a camera that moved
  between calibration and start-up kept an offset zone that reported healthy.
"""
from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
from typing import Callable, List

import cv2
import numpy as np
import pytest

from gradecrossing.alerting import CallbackSink
from gradecrossing.config import DetectorConfig, MonitorConfig, ZoneConfig, load_config
from gradecrossing.monitor import ConfigClassifier, Monitor
from gradecrossing.types import Detection, Event, Status
from gradecrossing.zone import (
    ZoneGeometryError,
    build_stabilizer,
    calibration_scale,
    polygon_inside_fraction,
    zone_thickness,
)

ROOT = Path(__file__).resolve().parents[1]
W, H = 640, 360
ZONE = [[200, 150], [440, 150], [440, 270], [200, 270]]


def scene(seed: int = 0, w: int = W, h: int = H) -> np.ndarray:
    """Textured static background (rectangles everywhere but the zone) of size w x h."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(120.0, 40.0, size=(H, W)).astype(np.float32)
    base = cv2.merge([cv2.GaussianBlur(noise, (0, 0), 1.5)] * 3)
    for _ in range(40):
        x, y = int(rng.integers(0, W - 40)), int(rng.integers(0, H - 30))
        if 150 < x < 500 and 100 < y < 300:
            continue
        base[y:y + 25, x:x + 35] = rng.uniform(30, 220, size=3)
    img = np.clip(base, 0, 255).astype(np.uint8)
    if (w, h) != (W, H):
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    return img


def shifted(img: np.ndarray, dx: float, dy: float) -> np.ndarray:
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REFLECT)


class FakeDetector:
    def __init__(self, script: Callable[[np.ndarray], List[Detection]]) -> None:
        self.script = script
        self._classify = ConfigClassifier(DetectorConfig())

    def detect(self, frame: np.ndarray) -> List[Detection]:
        return self.script(frame)

    def classify(self, name: str) -> str:
        return self._classify(name)

    def reset(self) -> None:
        pass


def parked_car(frame: np.ndarray) -> List[Detection]:
    """A car parked in the middle of the zone, reported in the pixels of the frame received."""
    s = frame.shape[1] / float(W)
    box = np.array([280 * s, 210 * s, 360 * s, 250 * s], np.float32)
    return [Detection(xyxy=box, conf=0.9, cls_id=2, cls_name="car", track_id=1)]


def monitor_cfg(**zone) -> MonitorConfig:
    cfg = MonitorConfig()
    cfg.zone.polygon = [list(p) for p in ZONE]
    for k, v in zone.items():
        setattr(cfg.zone, k, v)
    cfg.alerts.stdout = False
    cfg.alerts.latch_seconds = 1.0
    cfg.alerts.repeat_notify_seconds = 0.0
    cfg.stuck.warning_seconds = 0.3
    cfg.stuck.alarm_seconds = 0.6
    cfg.debris.enabled = False
    return cfg


# ----------------------------------------------------------------------------- helpers
def test_polygon_inside_fraction():
    assert polygon_inside_fraction(np.float32(ZONE), (W, H)) == pytest.approx(1.0)
    # a band drawn edge to edge (the advisor's zones) is fully inside
    assert polygon_inside_fraction(np.float32([[0, 262], [640, 265], [640, 284], [0, 284]]), (W, H)) == 1.0
    half = [[-100, 0], [100, 0], [100, 100], [-100, 100]]
    assert polygon_inside_fraction(np.float32(half), (W, H)) == pytest.approx(0.5)
    assert polygon_inside_fraction(np.float32([[700, 400], [800, 400], [800, 500]]), (W, H)) == 0.0
    fox13 = MonitorConfig().zone.polygon  # built-in default, drawn on a 1772x872 frame
    assert polygon_inside_fraction(np.float32(fox13), (1772, 872)) == pytest.approx(1.0)
    assert polygon_inside_fraction(np.float32(fox13), (W, H)) < 0.05


def test_zone_thickness_is_the_height_of_a_band():
    assert zone_thickness(np.float32([[0, 262], [640, 265], [640, 284], [0, 284]])) == pytest.approx(20.0, abs=1.0)
    assert zone_thickness(np.float32(ZONE)) == pytest.approx(2 * 240 * 120 / 720.0)


def test_calibration_scale_and_aspect_refusal():
    assert calibration_scale((1280, 720), (640, 360)) == (0.5, 0.5)
    sx, sy = calibration_scale((1772, 872), (960, 472))  # resize_width=960 of the FOX13 clip
    assert sx == pytest.approx(960 / 1772) and sy == pytest.approx(472 / 872)
    with pytest.raises(ZoneGeometryError, match="aspect"):
        calibration_scale((640, 480), (640, 360))
    with pytest.raises(ZoneGeometryError):
        calibration_scale((0, 480), (640, 360))


# ----------------------------------------------------------------------------- build_stabilizer
def test_reference_size_scales_the_polygon_on_both_axes():
    cfg = ZoneConfig(polygon=[[400, 300], [880, 300], [880, 540], [400, 540]], reference_size=[1280, 720])
    stab, scale, problems = build_stabilizer(cfg, (W, H), default_reference_size=(W, H))
    assert scale == (0.5, 0.5) and problems == []
    assert np.allclose(stab.reference_polygon, np.float32(ZONE))
    # without reference_size the default (first native frame) size is used
    stab, scale, _ = build_stabilizer(ZoneConfig(polygon=ZONE), (W, H), default_reference_size=(W, H))
    assert scale == (1.0, 1.0) and np.allclose(stab.reference_polygon, np.float32(ZONE))
    stab, scale, _ = build_stabilizer(ZoneConfig(polygon=ZONE), (2 * W, 2 * H), default_reference_size=(W, H))
    assert scale == (2.0, 2.0) and np.allclose(stab.reference_polygon, 2 * np.float32(ZONE))


def test_aspect_mismatch_and_bad_reference_size_are_refused():
    with pytest.raises(ZoneGeometryError, match="aspect"):
        build_stabilizer(ZoneConfig(polygon=ZONE, reference_size=[640, 480]), (W, H), (W, H))
    with pytest.raises(ZoneGeometryError, match="reference_size"):
        build_stabilizer(ZoneConfig(polygon=ZONE, reference_size=[640]), (W, H), (W, H))


def test_polygon_mostly_outside_the_frame_is_reported():
    _stab, _scale, problems = build_stabilizer(ZoneConfig(), (W, H), default_reference_size=(W, H))
    assert len(problems) == 1 and "inside" in problems[0] and "FOX13" in problems[0]


def test_reference_image_anchors_the_stabilizer_and_registers_the_first_live_frame(tmp_path):
    """GAEL-21: the camera moved by (6, -5) px between calibration and start-up. Anchored on
    the calibration frame, the first live frame already has the corrected zone; anchored on
    the first live frame (old behaviour) the offset is never noticed."""
    calib = scene(0)
    path = tmp_path / "site_reference.png"
    assert cv2.imwrite(str(path), calib)
    live = shifted(calib, 6, -5)
    stab, _scale, problems = build_stabilizer(ZoneConfig(polygon=ZONE, reference_image=str(path)), (W, H), (999, 999))
    assert problems == [] and stab.initialized
    state = stab.update(live, 0.0)
    assert state.healthy and state.method == "lk"
    assert np.abs(state.polygon - (np.int32(ZONE) + [6, -5])).max() <= 1
    assert np.allclose(state.to_reference[:2, 2], [-6, 5], atol=0.3)
    assert stab.registered and stab.health()["registered"] is True
    # the registered live frame is the keyframe from then on: plain LK, still registered
    state = stab.update(shifted(calib, 7, -5), 1 / 30.0)
    assert state.method == "lk" and np.abs(state.polygon - (np.int32(ZONE) + [7, -5])).max() <= 1
    assert np.allclose(state.to_reference[:2, 2], [-7, 5], atol=0.3)
    # the old behaviour: anchored on the live frame, the zone stays where it was drawn
    old, _scale, _ = build_stabilizer(ZoneConfig(polygon=ZONE), (W, H), (W, H))
    assert np.array_equal(old.update(live, 0.0).polygon, np.int32(ZONE))


def test_reference_image_size_defines_the_calibration_size(tmp_path):
    path = tmp_path / "big.png"
    assert cv2.imwrite(str(path), scene(0, 2 * W, 2 * H))
    cfg = ZoneConfig(polygon=[[2 * x, 2 * y] for x, y in ZONE], reference_image=str(path))
    stab, scale, problems = build_stabilizer(cfg, (W, H), default_reference_size=(W, H))
    assert scale == (0.5, 0.5) and problems == [] and stab.initialized
    assert np.allclose(stab.reference_polygon, np.float32(ZONE))
    state = stab.update(scene(0), 0.0)
    assert state.healthy and np.abs(state.polygon - np.int32(ZONE)).max() <= 1
    # an image whose shape contradicts reference_size is refused
    with pytest.raises(ZoneGeometryError, match="aspect"):
        build_stabilizer(ZoneConfig(polygon=ZONE, reference_image=str(path), reference_size=[640, 480]), (W, H), (W, H))


def test_unreadable_and_featureless_reference_images(tmp_path):
    """An optional calibration image that is missing, unreadable or blank is reported but
    never stops the crossing being watched: the stabilizer anchors on the live frame."""
    for bad in (tmp_path / "missing.png", tmp_path / "corrupt.png"):
        if bad.name == "corrupt.png":
            bad.write_bytes(b"not a png")
        stab, scale, problems = build_stabilizer(ZoneConfig(polygon=ZONE, reference_image=str(bad)), (W, H), (W, H))
        assert scale == (1.0, 1.0) and not stab.initialized
        assert len(problems) == 1 and "cannot read zone.reference_image" in problems[0]
        assert "zone.reference_size is not set" in problems[0]  # the polygon's frame is a guess too
        state = stab.update(scene(0), 0.0)
        assert state.method == "identity" and state.healthy
    _stab, _scale, problems = build_stabilizer(
        ZoneConfig(polygon=ZONE, reference_image=str(tmp_path / "missing.png"), reference_size=[W, H]), (W, H), (W, H))
    assert len(problems) == 1 and "reference_size is not set" not in problems[0]
    black = tmp_path / "black.png"
    assert cv2.imwrite(str(black), np.zeros((H, W, 3), np.uint8))
    stab, _scale, problems = build_stabilizer(ZoneConfig(polygon=ZONE, reference_image=str(black)), (W, H), (W, H))
    assert not stab.initialized and len(problems) == 1 and "features" in problems[0]
    assert stab.update(scene(0), 0.0).method == "identity"  # falls back to the first live frame


# ----------------------------------------------------------------------------- monitor
def test_monitor_reports_the_builtin_default_polygon_on_a_small_frame(caplog):
    """MISS-04: running without a site config watched 0.4% of a 640x360 frame silently."""
    cfg = MonitorConfig()
    cfg.alerts.stdout = False
    cfg.debris.enabled = False
    events: List[Event] = []
    mon = Monitor(cfg, detector=FakeDetector(lambda f: []), sinks=[CallbackSink(events.append)])
    with caplog.at_level(logging.ERROR, logger="gradecrossing.monitor"):
        result = mon.process_frame(scene(0), 0.0, 0)
    health = [e for e in events if e.type == "HEALTH"]
    assert len(health) == 1 and "inside" in health[0].reasons[0] and health[0].extra.get("zone_setup")
    assert result.events and result.events[0].type == "HEALTH"
    assert any(r.levelno == logging.ERROR and "inside" in r.getMessage() for r in caplog.records)
    mon.process_frame(scene(0), 1 / 30.0, 1)
    assert len([e for e in events if e.type == "HEALTH"]) == 1  # reported once, at start-up


def test_monitor_scales_by_reference_size():
    cfg = monitor_cfg(polygon=[[400, 300], [880, 300], [880, 540], [400, 540]], reference_size=[1280, 720])
    mon = Monitor(cfg, detector=FakeDetector(lambda f: []))
    r = mon.process_frame(scene(0), 0.0, 0)
    assert np.array_equal(r.zone.polygon, np.int32(ZONE)) and not r.events


def test_monitor_refuses_a_stream_with_another_aspect_ratio(caplog):
    events: List[Event] = []
    mon = Monitor(monitor_cfg(reference_size=[640, 480]), detector=FakeDetector(lambda f: []),
                  sinks=[CallbackSink(events.append)])
    with caplog.at_level(logging.ERROR, logger="gradecrossing.monitor"):
        with pytest.raises(ZoneGeometryError, match="aspect"):
            mon.process_frame(scene(0), 0.0, 0)
    assert [e.type for e in events] == ["HEALTH"] and "zone refused" in events[0].reasons[0]
    assert any("zone refused" in r.getMessage() for r in caplog.records)
    # every later frame is refused too (the run loop counts them and stops), reported once
    with pytest.raises(ZoneGeometryError):
        mon.process_frame(scene(0), 1 / 30.0, 1)
    assert len(events) == 1 and mon.stabilizer is None


def test_live_resolution_change_reinitialises_with_a_rescaled_polygon():
    """MISS-01 / F07: the stream switches 640x360 -> 1280x720 with no zone.reference_size.
    The monitor resizes it to the first frame's size (MA2: nothing is reset); after a reset()
    the rebuilt zone must be scaled by new/first native size, so the parked car still alarms."""
    events: List[Event] = []
    mon = Monitor(monitor_cfg(), detector=FakeDetector(parked_car), sinks=[CallbackSink(events.append)])
    small, big = scene(0), scene(0, 2 * W, 2 * H)
    for i in range(30):
        mon.process_frame(small, i / 30.0, i)
    assert mon.engine.status is Status.ALARM
    assert mon.process_frame(big, 30 / 30.0, 30).zone.mask.shape == (H, W)  # MA2: resized, not refused
    mon.reset()  # a new video of another size
    n_before = len(events)
    for i in range(31, 75):
        r = mon.process_frame(big, i / 30.0, i)
    assert np.allclose(mon.stabilizer.reference_polygon, 2 * np.float32(ZONE))
    assert r.zone.mask.shape == (2 * H, 2 * W) and r.objects[0].in_zone
    assert "ALARM_START" in [e.type for e in events[n_before:]]
    # the same change with a pinned processing size keeps the original polygon
    pinned = monitor_cfg()
    pinned.video.resize_width = W
    mon2 = Monitor(pinned, detector=FakeDetector(parked_car))
    mon2.process_frame(small, 0.0, 0)
    r2 = mon2.process_frame(big, 1 / 30.0, 1)
    assert np.array_equal(r2.zone.polygon, np.int32(ZONE))


def test_monitor_anchors_on_the_reference_image(tmp_path):
    calib = scene(0)
    path = tmp_path / "ref.png"
    assert cv2.imwrite(str(path), calib)
    mon = Monitor(monitor_cfg(reference_image=str(path), reference_size=[W, H]), detector=FakeDetector(lambda f: []))
    r = mon.process_frame(shifted(calib, 6, -5), 0.0, 0)
    assert r.zone.healthy and np.abs(r.zone.polygon - (np.int32(ZONE) + [6, -5])).max() <= 1


def test_monitor_survives_black_first_frames():
    """D13 / OCC-03: a stream that starts black must not crash or stall the zone. A black
    start that lasts zone.health_report_seconds is reported, and so is the recovery."""
    events: List[Event] = []
    mon = Monitor(monitor_cfg(), detector=FakeDetector(lambda f: []), sinks=[CallbackSink(events.append)])
    black = np.zeros((H, W, 3), np.uint8)
    for i in range(20):
        r = mon.process_frame(black, i / 30.0, i)
        assert r.zone.method == "uninitialized" and not r.zone.healthy and r.zone.to_reference is None
    for i in range(20, 40):
        r = mon.process_frame(scene(0), i / 30.0, i)
    assert r.zone.healthy and r.zone.method == "lk"
    health = [e for e in events if e.type == "HEALTH"]
    assert len(health) == 2
    assert "uninitialised" in health[0].reasons[0] and health[0].extra["initialized"] is False
    assert health[0].frame_index == 15  # 0.5 s after the first black frame
    assert health[1].extra["healthy"] is True and health[1].extra["initialized"] is True
    assert health[1].frame_index == 35  # healthy for 0.5 s
    # a short black start is not worth an event
    events.clear()
    mon2 = Monitor(monitor_cfg(), detector=FakeDetector(lambda f: []), sinks=[CallbackSink(events.append)])
    for i in range(30):
        mon2.process_frame(black if i < 5 else scene(0), i / 30.0, i)
    assert not [e for e in events if e.type == "HEALTH"]


# ----------------------------------------------------------------------------- calibrate_zone
def _calibrate_zone():
    spec = importlib.util.spec_from_file_location("calibrate_zone_under_test", str(ROOT / "tools" / "calibrate_zone.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_calibrate_zone_writes_reference_size_and_a_clean_reference_image(tmp_path, capsys):
    tool = _calibrate_zone()
    frame = scene(0)
    src = tmp_path / "shot.png"
    assert cv2.imwrite(str(src), frame)
    out = tmp_path / "configs" / "site.yaml"
    rc = tool.main(["--source", str(src), "--polygon", "200,150 440,150 440,270 200,270", "--out", str(out),
                    "--no-seed", "--save-frame", str(tmp_path / "overlay.jpg")])
    assert rc == 0
    cfg = load_config(str(out))
    assert cfg.zone.reference_size == [W, H]
    ref = Path(cfg.zone.reference_image)
    assert ref.name == "site_reference.png" and ref.parent == out.parent
    saved = cv2.imread(str(ref))
    assert saved is not None and np.array_equal(saved, frame)  # clean: no seed, no polygon drawn
    assert "reference_image" in capsys.readouterr().out
    # the monitor accepts the result and anchors on the saved frame
    stab, scale, problems = build_stabilizer(cfg.zone, (W, H), (W, H))
    assert scale == (1.0, 1.0) and problems == [] and stab.initialized
    # re-calibrating without an image clears the stale one
    rc = tool.main(["--source", str(src), "--polygon", "210,150 440,150 440,270 210,270", "--out", str(out),
                    "--no-seed", "--no-reference-image"])
    assert rc == 0
    cfg = load_config(str(out))
    assert cfg.zone.reference_image == "" and cfg.zone.reference_size == [W, H]


def test_calibrate_zone_warns_about_an_untrackable_frame(tmp_path, capsys):
    tool = _calibrate_zone()
    src = tmp_path / "black.png"
    assert cv2.imwrite(str(src), np.zeros((H, W, 3), np.uint8))
    rc = tool.main(["--source", str(src), "--polygon", "200,150 440,150 440,270 200,270",
                    "--out", str(tmp_path / "site.yaml"), "--no-seed", "--no-reference-image"])
    assert rc == 0
    assert "too little texture" in capsys.readouterr().err
