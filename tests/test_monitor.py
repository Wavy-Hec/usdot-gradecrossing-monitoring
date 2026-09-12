"""End-to-end tests for gradecrossing.monitor.Monitor with a scripted FakeDetector.

The scene is a seeded, textured 640x360 background (so the LK stabilizer has corners to
track) with a rectangular zone in the middle. Every "car" the FakeDetector reports is
also *painted* into the frame at the detection's box, so the MOG2 debris detector sees
it too: this is what proves that the exclusion boxes (real or velocity-predicted) keep
the debris area at zero while a detected vehicle is in the zone.

Time thresholds are shortened through config overrides so each scenario runs in a few
hundred frames, but frames are always timestamped at 30 fps so the timestamp-based
counters behave exactly as they would on real footage. No test imports ultralytics.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pytest

from gradecrossing.alerting import CallbackSink
from gradecrossing.config import DetectorConfig, MonitorConfig, config_from_dict, load_config, save_config
from gradecrossing.monitor import (
    MAX_CONSECUTIVE_FRAME_ERRORS,
    ConfigClassifier,
    Monitor,
    apply_cli_overrides,
    build_parser,
    format_summary,
    is_live_source,
    main,
    open_source,
)
from gradecrossing.tracking import PredictedDetection
from gradecrossing.types import Detection, Event, FrameResult, GateState, Status

ROOT = Path(__file__).resolve().parents[1]
H, W = 360, 640
FPS = 30.0
ZONE = [[200, 150], [440, 150], [440, 270], [200, 270]]  # 240x120 px rectangle, area 28800
CAR_W, CAR_H = 60, 40
CAR_COLOR = (40, 40, 40)
BLOB_COLOR = (250, 250, 250)


# ----------------------------------------------------------------------------- scene


class Scene:
    """Textured static background + painted cars/blobs + mild per-frame sensor noise."""

    NOISE_BANK = 8

    def __init__(self, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        noise = rng.normal(120.0, 40.0, size=(H, W)).astype(np.float32)
        base = cv2.GaussianBlur(noise, (0, 0), 1.5)
        base = cv2.merge([base, base * 0.95, base * 1.05])
        # Rectangles outside the zone give the stabilizer strong, well-spread corners.
        for _ in range(40):
            x = int(rng.integers(0, W - 40))
            y = int(rng.integers(0, H - 30))
            if 150 < x < 500 and 100 < y < 300:
                continue
            base[y:y + 25, x:x + 35] = rng.uniform(30, 220, size=3)
        self.base = np.clip(base, 0, 255).astype(np.float32)
        self.noise = rng.normal(0.0, 2.5, size=(self.NOISE_BANK, H, W, 3)).astype(np.float32)
        self.calls = 0

    def frame(self, boxes: Sequence[np.ndarray] = (), blob: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
        img = self.base.copy()
        for box in boxes:
            x1, y1, x2, y2 = [int(round(float(v))) for v in box]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            if x2 > x1 and y2 > y1:
                img[y1:y2, x1:x2] = CAR_COLOR
        if blob is not None:
            x1, y1, x2, y2 = blob
            img[y1:y2, x1:x2] = BLOB_COLOR
        img += self.noise[self.calls % self.NOISE_BANK]
        self.calls += 1
        return np.clip(img, 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------------- fakes


def car_det(bx: float, by: float, tid: Optional[int] = 1, cls: str = "car") -> Detection:
    """Detection whose bottom-center is (bx, by)."""
    xyxy = np.array([bx - CAR_W / 2, by - CAR_H, bx + CAR_W / 2, by], dtype=np.float32)
    return Detection(xyxy=xyxy, conf=0.9, cls_id=0, cls_name=cls, track_id=tid)


class FakeDetector:
    """Returns ``script(frame_index)``; the test sets ``index`` before every process_frame."""

    def __init__(self, script: Callable[[int], List[Detection]]) -> None:
        self.script = script
        self.index = 0
        self.calls: List[int] = []
        self.resets = 0
        self._classify = ConfigClassifier(DetectorConfig())

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        assert frame_bgr.ndim == 3 and frame_bgr.dtype == np.uint8
        self.calls.append(self.index)
        return [Detection(xyxy=d.xyxy.copy(), conf=d.conf, cls_id=d.cls_id, cls_name=d.cls_name, track_id=d.track_id)
                for d in self.script(self.index)]

    def classify(self, cls_name: str) -> str:
        return self._classify(cls_name)

    def reset(self) -> None:
        self.resets += 1


class FakeInfra:
    """Always reports the gates down."""

    def __init__(self, down: bool = True) -> None:
        self.down = down
        self.calls = 0

    def update(self, frame_bgr: np.ndarray, timestamp: float, frame_index: int) -> GateState:
        self.calls += 1
        return GateState(gates_down=self.down, lights_on=self.down, confidence=0.9 if self.down else 0.0,
                         last_update=timestamp, last_seen_down=timestamp if self.down else None, available=True)

    def reset(self) -> None:
        pass


def no_cars(_i: int) -> List[Detection]:
    return []


def crossing_script(start: int, x0: float, speed: float, y: float, end: Optional[int] = None,
                    tid: int = 1) -> Callable[[int], List[Detection]]:
    """A car that appears at frame ``start`` and drives at ``speed`` px/s (until ``end``)."""

    def script(i: int) -> List[Detection]:
        if i < start or (end is not None and i >= end):
            return []
        return [car_det(x0 + speed * (i - start) / FPS, y, tid=tid)]

    return script


def stop_script(start: int, x0: float, speed: float, y: float, stop_x: float, leave: int,
                seed: int = 1, jitter: float = 0.3, tid: int = 1) -> Callable[[int], List[Detection]]:
    """A car drives in, stops at ``stop_x`` (jittered box), and vanishes at frame ``leave``."""
    rng = np.random.default_rng(seed)
    n_move = int(round((stop_x - x0) / speed * FPS))
    jit = rng.uniform(-jitter, jitter, size=(leave + 1, 2))

    def script(i: int) -> List[Detection]:
        if i < start or i >= leave:
            return []
        k = i - start
        if k < n_move:
            return [car_det(x0 + speed * k / FPS, y, tid=tid)]
        return [car_det(stop_x + jit[i, 0], y + jit[i, 1], tid=tid)]

    return script


# ----------------------------------------------------------------------------- session


def make_cfg(**sections) -> MonitorConfig:
    """Fast thresholds; ``sections`` are {section: {field: value}} overrides."""
    cfg = MonitorConfig()
    cfg.zone.polygon = [list(p) for p in ZONE]
    cfg.alerts.stdout = False
    cfg.alerts.latch_seconds = 1.0
    cfg.alerts.repeat_notify_seconds = 0.0
    cfg.stuck.warning_seconds = 0.5
    cfg.stuck.alarm_seconds = 1.0
    cfg.debris.warmup_frames = 30
    cfg.debris.persist_seconds = 0.5
    for section, fields in sections.items():
        for k, v in fields.items():
            setattr(getattr(cfg, section), k, v)
    return cfg


class Session:
    """Drives a Monitor through synthetic frames at 30 fps and records results/events."""

    def __init__(self, cfg: MonitorConfig, script: Callable[[int], List[Detection]] = no_cars,
                 blob: Optional[Callable[[int], Optional[Tuple[int, int, int, int]]]] = None,
                 infra: Optional[FakeInfra] = None, seed: int = 0) -> None:
        self.cfg = cfg
        self.script = script
        self.blob = blob
        self.scene = Scene(seed)
        self.fake = FakeDetector(script)
        self.events: List[Event] = []
        self.results: List[FrameResult] = []
        self.mon = Monitor(cfg, detector=self.fake, infra_detector=infra, sinks=[CallbackSink(self.events.append)])
        self.i = 0

    def frame(self, i: int) -> np.ndarray:
        boxes = [d.xyxy for d in self.script(i)]
        return self.scene.frame(boxes, self.blob(i) if self.blob else None)

    def run(self, n_frames: int) -> List[FrameResult]:
        for _ in range(n_frames):
            i = self.i
            self.fake.index = i
            res = self.mon.process_frame(self.frame(i), i / FPS, i)
            assert res.frame_index == i and abs(res.timestamp - i / FPS) < 1e-9
            self.results.append(res)
            self.i += 1
        return self.results

    def events_of(self, etype: str, repeats: bool = False) -> List[Event]:
        return [e for e in self.events if e.type == etype and (repeats or not e.extra.get("repeat"))]

    def status_at(self, t: float) -> Status:
        return self.results[int(round(t * FPS))].assessment.status


# ----------------------------------------------------------------------------- scenarios


def test_scenario_a_crossing_car_never_alarms():
    np.random.seed(0)
    # Warm-up (30 frames), then a car crosses the 240 px zone in 2 s at 120 px/s.
    s = Session(make_cfg(), crossing_script(start=30, x0=80.0, speed=120.0, y=230.0))
    res = s.run(160)
    statuses = {r.assessment.status for r in res}
    assert Status.ALARM not in statuses
    assert not s.events_of("ALARM_START", repeats=True)
    assert all(o.stuck_seconds == 0.0 for r in res for o in r.objects)
    assert max(o.dwell_seconds for r in res for o in r.objects) <= 2.1
    # the car was seen inside the zone at some point (bottom-center test)
    assert any(o.in_zone for r in res for o in r.objects)
    # debris: the painted car is excluded by its own detection box, so no debris is ever reported
    after = [r.debris for r in res[30:] if r.debris is not None]
    assert after and all(d.warmed_up for d in after)
    assert max(d.area_px for d in after) == 0
    # the zone stayed healthy (LK on the textured background) throughout
    assert all(r.zone.healthy for r in res)
    assert res[-1].zone.method == "lk"
    # every stage reported a timing
    assert {"resize", "zone", "detect", "track", "infra", "debris", "alert", "total"} <= set(res[-1].timings_ms)


def test_scenario_b_stopped_car_alarms_latches_and_clears():
    np.random.seed(0)
    cfg = make_cfg()
    stop_t, leave_t = 3.0, 8.0  # car stops at x=320 (zone center) at t=3.0 s, vanishes at t=8.0 s
    script = stop_script(start=30, x0=80.0, speed=120.0, y=230.0, stop_x=320.0, leave=int(leave_t * FPS))
    s = Session(cfg, script)
    res = s.run(int(11.0 * FPS))

    starts = s.events_of("ALARM_START")
    assert len(starts) == 1, [e.type for e in s.events]
    t_alarm = starts[0].timestamp
    # stuck time starts once the 1 s speed window is free of the approach samples, then alarm_seconds
    assert stop_t + cfg.stuck.alarm_seconds <= t_alarm <= stop_t + cfg.stuck.alarm_seconds + cfg.stuck.speed_window_seconds + 0.3
    assert starts[0].track_ids == [1]
    assert any("stationary in zone" in r for r in starts[0].reasons)
    warn = s.events_of("WARNING_START")
    assert warn and warn[0].timestamp < t_alarm
    assert s.status_at(t_alarm - 0.1) is not Status.ALARM
    assert s.status_at(t_alarm + 0.5) is Status.ALARM

    # The car vanished at t=8.0: the track survives lost_track_grace_seconds (1 s) with frozen
    # counters, then the latch (1 s) holds the ALARM, then it clears exactly once.
    grace = cfg.stuck.lost_track_grace_seconds
    assert s.status_at(leave_t + 0.5) is Status.ALARM
    latched = s.results[int(round((leave_t + grace + 0.5) * FPS))].assessment
    assert latched.status is Status.ALARM and "latched" in latched.reasons
    ends = s.events_of("ALARM_END")
    assert len(ends) == 1
    assert leave_t + grace + cfg.alerts.latch_seconds - 0.15 <= ends[0].timestamp <= leave_t + grace + cfg.alerts.latch_seconds + 0.15
    assert res[-1].assessment.status is Status.CLEAR
    assert res[-1].objects == []
    # no debris while the car sat in the zone (exclusion box) or after it left (background frozen)
    assert max(r.debris.area_px for r in res[30:]) == 0


def test_scenario_c_gates_down_shortens_alarm_threshold():
    np.random.seed(0)
    cfg = make_cfg(stuck={"alarm_seconds": 3.0, "warning_seconds": 2.0,
                          "gates_down_alarm_seconds": 0.5, "gates_down_warning_seconds": 0.2})
    stop_t = 3.0
    script = stop_script(start=30, x0=80.0, speed=120.0, y=230.0, stop_x=320.0, leave=int(9.0 * FPS))
    infra = FakeInfra(down=True)
    s = Session(cfg, script, infra=infra)
    res = s.run(int(6.0 * FPS))
    starts = s.events_of("ALARM_START")
    assert len(starts) == 1
    t_alarm = starts[0].timestamp
    assert t_alarm < stop_t + cfg.stuck.alarm_seconds  # earlier than the normal 3 s threshold would allow
    assert t_alarm <= stop_t + cfg.stuck.gates_down_alarm_seconds + cfg.stuck.speed_window_seconds + 0.3
    assert any("(gates down)" in r for r in starts[0].reasons)
    last = res[-1]
    assert last.gate.available and last.gate.gates_down
    assert last.assessment.gates_down
    assert abs(last.assessment.severity_scale - 0.5 / 3.0) < 1e-9
    assert infra.calls == len(res)  # an injected infra detector is used even with infrastructure.enabled=False


def test_scenario_d_static_debris_alarms_and_never_self_clears():
    np.random.seed(0)
    # persist 1.5 s = 45 frames, longer than the ~21 frames MOG2 needs to absorb a static change at
    # the default empty_learning_rate. The blob appears at t=6 s, after the zone has been "empty"
    # for more than empty_seconds_before_refresh (5 s), i.e. exactly when the refresh path would be
    # active. The monitor must withhold zone_empty_seconds while candidate debris persists.
    cfg = make_cfg(debris={"persist_seconds": 1.5})
    appear_t = 6.0
    blob = (290, 190, 350, 230)  # 60x40 = 2400 px = 8.3 % of the zone, inside it

    def blob_at(i: int):
        return blob if i >= int(appear_t * FPS) else None

    s = Session(cfg, no_cars, blob=blob_at)
    res = s.run(int(26.0 * FPS))
    starts = s.events_of("ALARM_START")
    assert len(starts) == 1
    t_alarm = starts[0].timestamp
    assert appear_t + cfg.debris.persist_seconds <= t_alarm <= appear_t + cfg.debris.persist_seconds + 0.2
    assert any(r.startswith("debris") for r in starts[0].reasons)
    assert starts[0].track_ids == []
    # never clears: no ALARM_END, ALARM on every frame after the trigger, still triggered at the end
    assert not s.events_of("ALARM_END")
    first_alarm = int(round(t_alarm * FPS))
    assert all(r.assessment.status is Status.ALARM for r in res[first_alarm:])
    assert all(r.assessment.debris_alarm for r in res[first_alarm:])
    last = res[-1].debris
    assert last is not None and last.triggered
    initial = res[int(appear_t * FPS) + 5].debris.area_px
    assert initial > 0
    assert last.area_px >= 0.9 * initial
    assert last.learning_rate_used == 0.0  # frozen while the alarm holds
    # before the blob, with the zone empty for > 5 s, the slow refresh was allowed (and harmless)
    pre = res[int(5.5 * FPS)].debris
    assert pre.learning_rate_used == cfg.debris.empty_learning_rate and pre.area_px == 0
    # the candidate blob suppressed the refresh before it triggered
    during = res[int((appear_t + 0.5) * FPS)].debris
    assert during.persist_seconds > 0 and during.learning_rate_used == 0.0


def test_scenario_e_track_id_swap_does_not_delay_the_alarm():
    """A stationary car whose detector id changes (ByteTrack re-identifying it after the
    train that trapped it passed in front) keeps its clocks: the ALARM still arrives at
    alarm_seconds after the stop, not alarm_seconds after the swap."""
    np.random.seed(0)
    cfg = make_cfg(stuck={"alarm_seconds": 4.0, "warning_seconds": 1.0})
    stop_t, swap_t = 3.0, 6.0
    base = stop_script(start=30, x0=80.0, speed=120.0, y=230.0, stop_x=320.0, leave=10 ** 6)

    def script(i: int) -> List[Detection]:
        dets = base(i)
        for d in dets:  # same box, new id from t = 6 s
            d.track_id = 2 if i >= int(swap_t * FPS) else 1
        return dets

    s = Session(cfg, script)
    res = s.run(int(10.0 * FPS))
    starts = s.events_of("ALARM_START")
    assert len(starts) == 1, [(e.type, e.timestamp) for e in s.events]
    t_alarm = starts[0].timestamp
    assert stop_t + cfg.stuck.alarm_seconds <= t_alarm <= (
        stop_t + cfg.stuck.alarm_seconds + cfg.stuck.speed_window_seconds + 0.3)
    assert t_alarm > swap_t  # the alarm really is raised after the id changed
    assert starts[0].track_ids == [2]
    # one physical vehicle stays one track: the old id was re-keyed, not duplicated
    assert len(res[-1].objects) == 1 and res[-1].objects[0].track_id == 2


def test_stationary_unmapped_class_blocks_the_background_refresh():
    """An object of a class the config does not map is invisible to the alert engine and is
    still excluded from the debris mask, so the one protection left is that it stops the
    empty-zone refresh: the background must not learn a stationary obstruction away. This
    also documents the residual hole - no WARNING or ALARM is raised for such an object, so
    map every class that can obstruct the crossing in detector.vehicle_classes /
    person_classes (the detector logs the unmapped ones at start-up)."""
    np.random.seed(0)
    cfg = make_cfg()
    appear = int(6.0 * FPS)  # after empty_seconds_before_refresh (5 s) has elapsed

    def script(i: int) -> List[Detection]:
        return [car_det(320.0, 230.0, tid=1, cls="bench")] if i >= appear else []

    s = Session(cfg, script)
    res = s.run(int(10.0 * FPS))
    obj = res[-1].objects[0]
    assert obj.in_zone and not (obj.is_vehicle or obj.is_person or obj.is_train)
    # before it arrives the zone is empty and the slow refresh is allowed
    assert res[appear - 1].debris.learning_rate_used == pytest.approx(cfg.debris.empty_learning_rate)
    # From its second frame on the refresh is off, so it can never be learned as background.
    # (min_track_age_frames ignores a brand-new id for one frame, so exactly one frame is
    # still learned at the refresh rate - 0.5 % of the weight MOG2 would need to absorb it.)
    assert all(r.debris.learning_rate_used == 0.0 for r in res[appear + 1:])
    assert s.mon.tracker.zone_occupied and s.mon.tracker.zone_empty_since is None
    assert Status.ALARM not in {r.assessment.status for r in res}  # the documented hole


def test_detect_every_3_frames_keeps_exclusion_boxes_on_the_moving_car():
    np.random.seed(0)
    cfg = make_cfg(detector={"detect_every_n_frames": 3})
    script = crossing_script(start=30, x0=80.0, speed=120.0, y=230.0)
    s = Session(cfg, script)
    res = s.run(160)
    assert s.fake.calls == [i for i in range(160) if i % 3 == 0]
    assert Status.ALARM not in {r.assessment.status for r in res}
    # debris stays 0 on every frame although the painted car crosses the zone
    assert max(r.debris.area_px for r in res[30:]) == 0
    # skipped frames carry predicted boxes for the same id, close to the true car position
    worst = 0.0
    for i in range(31, 150):
        r = res[i]
        if i % 3 == 0:
            assert all(not isinstance(d, PredictedDetection) for d in r.detections)
            continue
        assert r.detections and all(isinstance(d, PredictedDetection) for d in r.detections)
        assert r.detections[0].track_id == 1
        true_x = script(i)[0].bottom_center[0]
        worst = max(worst, abs(r.detections[0].bottom_center[0] - true_x))
        assert r.objects and r.objects[0].track_id == 1  # the object stays live on skipped frames
    assert worst < cfg.debris.exclusion_dilate_px


# ----------------------------------------------------------------------------- plumbing


def test_resize_width_scales_zone_polygon():
    cfg = make_cfg(video={"resize_width": 320})
    s = Session(cfg, crossing_script(start=0, x0=160.0, speed=0.0, y=115.0))
    res = s.run(3)
    r = res[-1]
    assert s.mon.last_frame.shape[:2] == (180, 320)
    assert r.zone.mask.shape == (180, 320)
    expected = np.asarray(ZONE, dtype=np.float32) * 0.5
    assert np.abs(r.zone.polygon.astype(np.float32) - expected).max() <= 1.0
    assert r.zone.area_px == pytest.approx(120 * 60, rel=0.05)
    # detection coordinates are in processing pixels: (160, 115) is inside the scaled zone
    assert r.objects and r.objects[0].in_zone
    # config polygon itself is untouched (still native coordinates)
    assert cfg.zone.polygon == ZONE


def test_health_events_on_zone_health_transitions():
    np.random.seed(0)
    s = Session(make_cfg())
    s.run(5)
    assert not s.events_of("HEALTH")
    uniform = np.full((H, W, 3), 128, dtype=np.uint8)
    s.fake.index = 5
    r = s.mon.process_frame(uniform, 5 / FPS, 5)
    assert not r.zone.healthy and r.zone.method == "held"
    health = [e for e in s.events if e.type == "HEALTH"]
    assert len(health) == 1 and health[0].extra["healthy"] is False and health[0].extra["method"] == "held"
    assert health[0].status == "CLEAR" and health[0].frame_index == 5
    assert r.events and r.events[0].type == "HEALTH"  # also visible in the FrameResult
    # textured frame again: healthy -> second HEALTH event
    s.run(1)
    health = [e for e in s.events if e.type == "HEALTH"]
    assert len(health) == 2 and health[1].extra["healthy"] is True
    # disabled: no events
    s2 = Session(make_cfg(zone={"health_events": False}))
    s2.run(2)
    s2.mon.process_frame(uniform, 2 / FPS, 2)
    assert not [e for e in s2.events if e.type == "HEALTH"]


def test_reset_forgets_everything_and_reinitialises_zone():
    np.random.seed(0)
    cfg = make_cfg()
    s = Session(cfg, stop_script(start=0, x0=320.0, speed=120.0, y=230.0, stop_x=320.0, leave=10 ** 6))
    s.run(int(3.0 * FPS))
    assert s.mon.engine.status is Status.ALARM
    s.mon.reset()
    assert s.mon.stabilizer is None
    assert s.mon.tracker.objects() == []
    assert s.mon.engine.status is Status.CLEAR
    assert s.fake.resets == 1
    assert not s.mon.debris.warmed_up
    n_events = len(s.events)
    r = s.run(1)[-1]
    assert r.zone.method == "identity" and r.zone.healthy
    assert r.assessment.status is Status.CLEAR
    assert len(s.events) == n_events  # reset emits nothing


def test_frame_result_objects_are_snapshots():
    np.random.seed(0)
    s = Session(make_cfg(), stop_script(start=0, x0=320.0, speed=120.0, y=230.0, stop_x=320.0, leave=10 ** 6))
    res = s.run(int(2.0 * FPS))
    early = res[45].objects[0].stuck_seconds
    assert res[-1].objects[0].stuck_seconds > early  # earlier results were not mutated by later frames


def test_debris_disabled_gives_none():
    s = Session(make_cfg(debris={"enabled": False}))
    r = s.run(2)[-1]
    assert r.debris is None and s.mon.debris is None
    assert r.timings_ms["debris"] >= 0.0


def test_package_parses_on_the_oldest_supported_python():
    """pyproject declares requires-python >= 3.8 (JetPack 5 Jetsons ship Python 3.8), so
    every module must parse under 3.8 syntax: no walrus-in-comprehension edge cases, no
    ``X | Y`` annotations evaluated at runtime, no ``match``. This is a syntax check, not a
    runtime one - keep 3.9+/3.10+ library calls out too (removeprefix, functools.cache,
    is_relative_to, math.lcm), which the grep below pins."""
    import ast

    sources = sorted(ROOT.glob("gradecrossing/*.py")) + sorted(ROOT.glob("tools/*.py"))
    assert len(sources) > 15
    for path in sources:
        text = path.read_text(encoding="utf-8")
        ast.parse(text, filename=str(path), feature_version=(3, 8))
        for newer in ("removeprefix(", "removesuffix(", "is_relative_to(", "functools.cache",
                      "math.lcm", "graphlib"):
            assert newer not in text, f"{path.name} uses a post-3.8 API: {newer}"


def test_monitor_and_visualize_do_not_import_ultralytics():
    code = ("import sys, gradecrossing.monitor, gradecrossing.visualize; "
            "print('ultralytics' in sys.modules, 'torch' in sys.modules)")
    env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONIOENCODING="utf-8")
    out = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), env=env, capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False False"


# ----------------------------------------------------------------------------- sources / run()


def write_clip(path: Path, n_frames: int = 45, seed: int = 0) -> Path:
    scene = Scene(seed)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    if not writer.isOpened():
        pytest.skip("cv2.VideoWriter cannot write mp4v here")
    for _ in range(n_frames):
        writer.write(scene.frame())
    writer.release()
    return path


def test_is_live_source_guess():
    assert is_live_source("0") and is_live_source(" 2 ")
    assert is_live_source("rtsp://cam/stream") and is_live_source("HTTP://host/x.m3u8") and is_live_source("udp://@:5000")
    assert not is_live_source("videos/clip.mov") and not is_live_source("C:/x/y.mp4")


def test_open_source_errors_fps_and_seek(tmp_path):
    cfg = make_cfg(video={"source": str(tmp_path / "missing.mp4")})
    with pytest.raises(RuntimeError):
        open_source(cfg.video)
    clip = write_clip(tmp_path / "clip.mp4")
    cap, fps, live = open_source(MonitorConfig().video.__class__(source=str(clip)))
    assert not live and fps == pytest.approx(30.0)
    cap.release()
    cap, fps, live = open_source(MonitorConfig().video.__class__(source=str(clip), fps_override=12.5, start_seconds=0.5))
    assert fps == 12.5
    assert abs(cap.get(cv2.CAP_PROP_POS_FRAMES) - 15) <= 1  # seek uses the container clock (30 fps)
    cap.release()


def test_run_on_file_writes_video_events_and_snapshot(tmp_path, capsys):
    clip = write_clip(tmp_path / "clip.mp4", n_frames=40)
    out_video = tmp_path / "out" / "annotated.mp4"
    events_path = tmp_path / "events.jsonl"
    snaps = tmp_path / "snaps"
    cfg = make_cfg(
        video={"source": str(clip), "max_frames": 20},
        stuck={"alarm_seconds": 0.2, "warning_seconds": 0.1},
        debris={"warmup_frames": 5},
        alerts={"events_jsonl": str(events_path)},
        output={"save_video": str(out_video), "snapshot_dir": str(snaps), "print_every_n_frames": 10},
    )  # display stays False: the GUI fallback has its own test with cv2.imshow monkeypatched,
       # so the suite never opens a window on an OpenCV build that has GUI support.
    fake = FakeDetector(lambda i: [car_det(320.0, 230.0)])  # parked in the zone from frame 0
    mon = Monitor(cfg, detector=fake)
    summary = mon.run()
    assert summary["frames"] == 20 and summary["alarms"] == 1 and summary["fps"] > 0
    assert summary["final_status"] == "ALARM" and summary["reconnects"] == 0 and not summary["interrupted"]
    assert set(summary["timings_ms"]) >= {"zone", "detect", "track", "debris", "alert", "total"}
    assert len(fake.calls) == 20  # detector ran on every frame (index is not set by run(); count only)
    # events log: exactly the Event keys, ALARM_START once, timestamps = frame/fps
    lines = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    alarm_starts = [e for e in lines if e["type"] == "ALARM_START" and not e["extra"].get("repeat")]
    assert len(alarm_starts) == 1
    assert set(alarm_starts[0]) == {"type", "timestamp", "frame_index", "status", "reasons", "track_ids", "extra"}
    assert alarm_starts[0]["timestamp"] == pytest.approx(alarm_starts[0]["frame_index"] / 30.0)
    # annotated video at processing resolution with one frame per processed frame
    assert out_video.exists() and out_video.stat().st_size > 0
    cap = cv2.VideoCapture(str(out_video))
    n = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        assert fr.shape == (H, W, 3)
        n += 1
    cap.release()
    assert n == 20
    # one snapshot for the ALARM_START
    jpgs = sorted(snaps.glob("alarm_*.jpg"))
    assert len(jpgs) == 1 and cv2.imread(str(jpgs[0])) is not None
    # periodic status lines are ASCII and mention the status
    captured = capsys.readouterr().out
    status_lines = [ln for ln in captured.splitlines() if ln.startswith("[f=")]
    assert len(status_lines) == 2 and all(ln.isascii() for ln in status_lines)
    assert "ALARM" in status_lines[-1]
    # the JSONL sink was closed by run(); a second run must reopen cleanly (append mode)
    summary2 = Monitor(cfg, detector=FakeDetector(lambda i: [])).run()
    assert summary2["frames"] == 20 and summary2["alarms"] == 0


def test_display_failure_disables_display_without_stopping_the_run(tmp_path, monkeypatch):
    """opencv-python-headless (CI, a Jetson without X) raises cv2.error from imshow. The run
    must continue with the display switched off. imshow is monkeypatched so this branch is
    exercised identically on GUI builds - and so the suite never pops up a window."""
    clip = write_clip(tmp_path / "clip.mp4", n_frames=12)
    shown: List[int] = []

    def no_gui(*_a, **_k):
        shown.append(1)
        raise cv2.error("The function is not implemented (no GUI support)")

    monkeypatch.setattr(cv2, "imshow", no_gui)
    monkeypatch.setattr(cv2, "waitKey", lambda *_a, **_k: -1)
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)
    cfg = make_cfg(video={"source": str(clip), "max_frames": 6}, debris={"warmup_frames": 2},
                   output={"display": True})
    summary = Monitor(cfg, detector=FakeDetector(no_cars)).run()
    assert summary["frames"] == 6 and summary["errors"] == 0
    assert len(shown) == 1  # tried once, then display was disabled for the rest of the run


def test_run_survives_a_transient_frame_error(tmp_path):
    """A crossing must keep being watched through one bad frame: the failure is counted and
    logged, the loop continues, and the summary says how many frames were lost."""
    clip = write_clip(tmp_path / "clip.mp4", n_frames=12)

    class FlakyDetector(FakeDetector):
        def detect(self, frame_bgr):
            if len(self.calls) == 3:
                self.calls.append(-1)
                raise RuntimeError("transient inference failure")
            return super().detect(frame_bgr)

    cfg = make_cfg(video={"source": str(clip), "max_frames": 10}, debris={"warmup_frames": 2})
    summary = Monitor(cfg, detector=FlakyDetector(no_cars)).run()
    assert summary["errors"] == 1
    # max_frames counts *processed* frames, so the lost frame costs one extra read, not a
    # missing result: 11 frames were read from the 12-frame clip and 10 were processed.
    assert summary["frames"] == 10
    assert not summary["interrupted"]


def test_run_aborts_after_repeated_frame_errors(tmp_path):
    """A systematic failure (not a transient one) still stops the loop instead of spinning."""
    clip = write_clip(tmp_path / "clip.mp4", n_frames=30)

    class BrokenDetector(FakeDetector):
        def detect(self, frame_bgr):
            raise RuntimeError("detector is broken")

    cfg = make_cfg(video={"source": str(clip)}, debris={"warmup_frames": 2})
    summary = Monitor(cfg, detector=BrokenDetector(no_cars)).run()
    assert summary["frames"] == 0
    assert summary["errors"] == MAX_CONSECUTIVE_FRAME_ERRORS


def test_live_geometry_change_reinitialises_instead_of_exiting(tmp_path):
    """An RTSP stream that renegotiates its resolution makes ZoneStabilizer raise ValueError.
    On a live source the pipeline re-initialises on the new geometry (and says so with a
    HEALTH event) rather than ending the run; pin the geometry with video.resize_width to
    avoid the re-initialisation entirely."""
    clip = write_clip(tmp_path / "clip.mp4", n_frames=12)
    events: List[Event] = []

    class ResizingDetector(FakeDetector):
        """Stands in for the stream: the frame handed to the zone changes size once."""

        def detect(self, frame_bgr):
            if len(self.calls) == 4:
                self.calls.append(-1)
                raise ValueError("frame size 320x180 does not match the reference keyframe 640x360")
            return super().detect(frame_bgr)

    cfg = make_cfg(video={"source": str(clip), "max_frames": 8, "live": True},
                   debris={"warmup_frames": 2})
    fake = ResizingDetector(no_cars)
    mon = Monitor(cfg, detector=fake, sinks=[CallbackSink(events.append)])
    summary = mon.run()
    assert summary["errors"] == 1 and summary["frames"] == 8
    resets = [e for e in events if e.type == "HEALTH" and e.extra.get("reset")]
    assert len(resets) == 1 and "re-initialised" in resets[0].reasons[0]
    assert summary["health_events"] >= 1
    assert fake.resets == 1  # Monitor.reset() propagated to the detector


def test_run_loop_and_start_seconds(tmp_path):
    clip = write_clip(tmp_path / "clip.mp4", n_frames=30)
    cfg = make_cfg(video={"source": str(clip), "loop": True, "max_frames": 50, "start_seconds": 0.5},
                   debris={"warmup_frames": 5})
    fake = FakeDetector(no_cars)
    mon = Monitor(cfg, detector=fake)
    summary = mon.run()
    assert summary["frames"] == 50
    assert summary["loops"] >= 2  # 15 frames per pass after the 0.5 s seek
    assert fake.resets == summary["loops"]  # every loop restart resets the detector too


# ----------------------------------------------------------------------------- configs / CLI


def test_default_yaml_matches_dataclass_defaults():
    cfg = load_config(str(ROOT / "configs" / "default.yaml"))
    assert cfg == MonitorConfig()


def test_stresstest_yaml_loads():
    cfg = load_config(str(ROOT / "configs" / "stresstest.yaml"))
    assert cfg.zone.polygon == [[100, 400], [500, 350], [1650, 480], [1500, 600]]
    assert cfg.video.source == "videos/Rail-Grade-Crossing-StressTest.mov"
    assert cfg.detector.weights == "yolo11n.pt"
    assert not is_live_source(cfg.video.source)


def test_cli_overrides_every_flag():
    args = build_parser().parse_args([
        "--source", "rtsp://cam/1", "--weights", "w.pt", "--display", "--save", "o.mp4", "--events", "e.jsonl",
        "--max-frames", "7", "--infra-weights", "g.pt", "--no-stabilize", "--print-every", "5", "--fps", "25",
        "--resize-width", "960", "--start-seconds", "1.5", "--loop", "--live", "--detect-every", "2",
        "--device", "cpu", "--snapshot-dir", "snaps", "--webhook", "http://h/x", "--no-stdout", "--log-level", "DEBUG",
    ])
    cfg = apply_cli_overrides(MonitorConfig(), args)
    assert cfg.video.source == "rtsp://cam/1" and cfg.detector.weights == "w.pt" and cfg.output.display
    assert cfg.output.save_video == "o.mp4" and cfg.alerts.events_jsonl == "e.jsonl" and cfg.video.max_frames == 7
    assert cfg.infrastructure.enabled and cfg.infrastructure.weights == "g.pt"
    assert not cfg.zone.stabilize and cfg.output.print_every_n_frames == 5 and cfg.video.fps_override == 25.0
    assert cfg.video.resize_width == 960 and cfg.video.start_seconds == 1.5 and cfg.video.loop and cfg.video.live is True
    assert cfg.detector.detect_every_n_frames == 2 and cfg.detector.device == "cpu" and cfg.infrastructure.device == "cpu"
    assert cfg.output.snapshot_dir == "snaps" and cfg.alerts.webhook_url == "http://h/x" and not cfg.alerts.stdout
    assert cfg.output.log_level == "DEBUG"
    # no flags: nothing changes
    assert apply_cli_overrides(MonitorConfig(), build_parser().parse_args([])) == MonitorConfig()
    assert format_summary({"frames": 3, "alarms": 1, "fps": 12.5, "timings_ms": {"total": 4.0}}).isascii()


def test_cli_error_paths(tmp_path, capsys):
    assert main(["--config", str(tmp_path / "nope.yaml")]) == 2
    bad = tmp_path / "bad.yaml"
    bad.write_text("video:\n  sourc: x\n", encoding="utf-8")  # typo -> KeyError from config_from_dict
    assert main(["--config", str(bad)]) == 2
    # Every other way a config file goes wrong is the same exit status 2, never a traceback.
    tabbed = tmp_path / "tabbed.yaml"
    tabbed.write_text("video:\n\tsource: x\n", encoding="utf-8")  # tab indent -> yaml.YAMLError
    assert main(["--config", str(tabbed)]) == 2
    listed = tmp_path / "listed.yaml"
    listed.write_text("- video\n- zone\n", encoding="utf-8")  # top level is a list, not a mapping
    assert main(["--config", str(listed)]) == 2
    scalar = tmp_path / "scalar.yaml"
    scalar.write_text("just a string\n", encoding="utf-8")
    assert main(["--config", str(scalar)]) == 2
    assert capsys.readouterr().err.count("cannot load config") == 5
    with pytest.raises(TypeError):
        config_from_dict(["video"])  # type: ignore[arg-type]
    cfg = make_cfg(video={"source": str(tmp_path / "missing.mp4")})
    good = tmp_path / "good.yaml"
    save_config(cfg, str(good))
    assert main(["--config", str(good), "--no-stdout"]) == 1  # cannot open the source (before any model load)
    err = capsys.readouterr().err
    assert "cannot open video source" in err


def test_cli_smoke_with_real_detector(tmp_path):
    pytest.importorskip("ultralytics")
    weights = ROOT / "50_epochs_best.pt"
    if not weights.exists():
        pytest.skip("50_epochs_best.pt not available")
    clip = write_clip(tmp_path / "clip.mp4", n_frames=6)
    cfg = make_cfg(video={"source": str(clip)}, detector={"imgsz": 320, "device": "cpu"})
    cfg_path = tmp_path / "cfg.yaml"
    save_config(cfg, str(cfg_path))
    events = tmp_path / "events.jsonl"
    rc = main(["--config", str(cfg_path), "--weights", str(weights), "--max-frames", "3", "--no-stdout",
               "--events", str(events), "--print-every", "1", "--log-level", "WARNING"])
    assert rc == 0
    assert events.exists()  # created by the JSONL sink even when nothing was emitted
