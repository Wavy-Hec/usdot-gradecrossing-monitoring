"""Registration of live frames against ``zone.reference_image`` and refusal of a zone
that cannot be mapped onto the stream (exit status 2).

The calibration image is taken once, usually in daylight; the monitor may start at night,
in rain, after a re-aim. These tests pin what happens then:

* a global exposure difference (gain/gamma) is registered with histogram-matched LK, and
  the registered live frame becomes the keyframe (no per-frame ORB against the image);
* an image that cannot be registered at all no longer holds the zone forever (the debris
  path treats a held zone as "registration lost" and never warms up): after
  ``registration_timeout_seconds`` the stabilizer tracks from a live frame at the
  calibration pose, says so once, and registers later at a keyframe refresh;
* a missing reference image is reported and the crossing is still watched;
* a zone whose aspect ratio does not fit the stream stops ``Monitor.run`` and the CLI
  exits 2 instead of ending "successfully" after five frame errors.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, List

import cv2
import numpy as np
import pytest

import gradecrossing.monitor as monitor_mod
from gradecrossing.alerting import CallbackSink
from gradecrossing.config import DetectorConfig, MonitorConfig, ZoneConfig, save_config
from gradecrossing.monitor import ConfigClassifier, Monitor, main
from gradecrossing.types import Detection, Event
from gradecrossing.zone import ZoneGeometryError, build_stabilizer, resolve_reference_path

ROOT = Path(__file__).resolve().parents[1]
W, H = 640, 360
ZONE = [[200, 150], [440, 150], [440, 270], [200, 270]]
FPS = 30.0


def scene(seed: int = 0) -> np.ndarray:
    """Textured static background (rectangles everywhere but the zone), 640x360 BGR."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(120.0, 40.0, size=(H, W)).astype(np.float32)
    base = cv2.merge([cv2.GaussianBlur(noise, (0, 0), 1.5)] * 3)
    for _ in range(40):
        x, y = int(rng.integers(0, W - 40)), int(rng.integers(0, H - 30))
        if 150 < x < 500 and 100 < y < 300:
            continue
        base[y:y + 25, x:x + 35] = rng.uniform(30, 220, size=3)
    return np.clip(base, 0, 255).astype(np.uint8)


def shifted(img: np.ndarray, dx: float, dy: float) -> np.ndarray:
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REFLECT)


def exposure(img: np.ndarray, alpha: float, gamma: float, noise: float = 0.0, seed: int = 0) -> np.ndarray:
    """out = alpha * 255 * (in / 255) ** gamma (+ Gaussian noise): dusk, night, another camera mode."""
    lut = alpha * 255.0 * (np.arange(256) / 255.0) ** gamma
    out = lut[img]
    if noise > 0:
        out = out + np.random.default_rng(seed).normal(0.0, noise, img.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


class FakeDetector:
    def __init__(self, script: Callable[[np.ndarray], List[Detection]] = lambda f: []) -> None:
        self.script = script
        self._classify = ConfigClassifier(DetectorConfig())

    def detect(self, frame: np.ndarray) -> List[Detection]:
        return self.script(frame)

    def classify(self, name: str) -> str:
        return self._classify(name)

    def reset(self) -> None:
        pass


def monitor_cfg(**zone) -> MonitorConfig:
    cfg = MonitorConfig()
    cfg.zone.polygon = [list(p) for p in ZONE]
    for k, v in zone.items():
        setattr(cfg.zone, k, v)
    cfg.alerts.stdout = False
    cfg.debris.enabled = False
    return cfg


@pytest.fixture(autouse=True)
def _seed_opencv():
    cv2.setRNGSeed(0)


@pytest.fixture()
def calibration(tmp_path) -> Path:
    path = tmp_path / "site_reference.png"
    assert cv2.imwrite(str(path), scene(0))
    return path


def build(path: Path, **zone):
    return build_stabilizer(ZoneConfig(polygon=ZONE, reference_image=str(path), **zone), (W, H), (W, H))


# ----------------------------------------------------------------------------- registration
def test_reference_image_in_other_light_is_registered_by_histogram_matched_lk(calibration):
    """GAEL-21 at dusk: the camera moved (6, -5) px since the daylight calibration and the
    picture is darker with another gamma. Plain LK cannot register it (the review measured
    ORB on 900/900 frames at 3-4x the cost); the histogram-matched retry does, on the first
    frame, and the registered live frame becomes the keyframe."""
    stab, _scale, problems = build(calibration)
    assert problems == [] and stab.initialized and not stab.registered
    live = exposure(shifted(scene(0), 6, -5), 0.6, 1.3)
    gray = cv2.cvtColor(live, cv2.COLOR_BGR2GRAY)
    assert stab._lk_estimate(stab._reference, gray)[0] is None  # plain LK really fails
    state = stab.update(live, 0.0)
    assert state.healthy and state.method == "lk"
    assert np.abs(state.polygon - (np.int32(ZONE) + [6, -5])).max() <= 1
    assert np.allclose(state.to_reference[:2, 2], [-6, 5], atol=0.3)
    assert stab.registered and stab._keyframe is not stab._reference
    assert stab.health_reports == []
    for i in range(1, 10):  # later frames track the live keyframe: LK every frame, no ORB
        state = stab.update(exposure(shifted(scene(0), 6, -5), 0.6, 1.3, noise=1.5, seed=i), i / FPS)
        assert state.method == "lk" and state.healthy
        assert np.abs(state.polygon - (np.int32(ZONE) + [6, -5])).max() <= 1


def test_unregistrable_reference_image_falls_back_to_the_live_frame_after_the_timeout(calibration):
    """The live view has nothing in common with the calibration image (a re-aimed camera,
    a different site, a night picture no estimator can match): the zone is held while the
    registration is tried, then tracked from a live frame at the calibration pose."""
    stab, _scale, _problems = build(calibration, registration_timeout_seconds=0.5)
    live = scene(7)
    reports = []
    for i in range(30):
        state = stab.update(shifted(live, 0.2 * (i % 2), 0.0), i / FPS)
        reports += stab.health_reports
        if i / FPS < 0.5:
            assert state.method == "held" and not state.healthy, i
            assert "registering against zone.reference_image" in stab.health()["reason"]
            assert np.array_equal(state.polygon, np.int32(ZONE)) and np.array_equal(state.to_reference, np.eye(3))
        elif i / FPS < 0.5 + 1.0 / FPS:
            assert state.method == "identity" and state.healthy, i
        else:
            assert state.method == "lk" and state.healthy, i
    assert not stab.registered and stab.health()["registered"] is False
    assert len(reports) == 1 and "not registered against zone.reference_image" in reports[0]
    assert "NOT corrected" in reports[0]
    assert np.array_equal(state.polygon, np.int32(ZONE))


def test_registration_is_retried_and_applied_at_a_keyframe_refresh(calibration):
    """Night: no estimate against the daylight image (simulated), so the stabilizer falls
    back to the calibration pose, although the camera has moved (8, 3) px. In the morning
    the direct estimate at a keyframe refresh succeeds: the zone moves onto the tracks and
    the correction is reported. Unverified refreshes before that keep the zone healthy."""
    stab, _scale, _problems = build(calibration, registration_timeout_seconds=0.2,
                                    keyframe_refresh_static_seconds=0.5)
    night = {"on": True}
    original = stab._estimate

    def estimate(kf, gray):
        if night["on"] and kf is stab._reference:
            return None, 0, "simulated night"
        return original(kf, gray)

    stab._estimate = estimate  # type: ignore[assignment]
    live = shifted(scene(0), 8, 3)
    reports: List[str] = []
    t = 0.0
    for i in range(45):  # 1.5 s of night: fallback, then refreshes that cannot be verified
        t = i / FPS
        state = stab.update(live, t, scene_static=True)
        reports += stab.health_reports
    assert state.healthy and not stab.registered
    assert np.array_equal(state.polygon, np.int32(ZONE))  # uncorrected: calibration pose
    assert stab.health()["reanchors"] >= 1 and stab.health()["unverified_reanchors"] >= 1
    assert len(reports) == 1 and "not registered" in reports[0]
    night["on"] = False
    for i in range(45, 90):
        t = i / FPS
        state = stab.update(live, t, scene_static=True)
        reports += stab.health_reports
        if stab.registered:
            break
    assert stab.registered and state.healthy
    assert np.abs(state.polygon - (np.int32(ZONE) + [8, 3])).max() <= 1
    assert np.allclose(state.to_reference[:2, 2], [-8, -3], atol=0.3)
    assert len(reports) == 2 and "registered against zone.reference_image" in reports[1]
    moved = re.search(r"moved ([0-9.]+) px", reports[1])
    assert moved is not None and float(moved.group(1)) == pytest.approx(np.hypot(8, 3), abs=0.3)
    assert stab.health()["unverified_reanchors"] == 0
    state = stab.update(live, t + 1 / FPS, scene_static=True)
    assert state.healthy and np.abs(state.polygon - (np.int32(ZONE) + [8, 3])).max() <= 1


def test_monitor_with_a_reference_image_never_stays_held_at_night(calibration):
    """Review probe in miniature: calibrated in daylight, started in deep night (gain 0.15,
    gamma 1.6, sensor noise). The zone must not stay held for the whole night - the debris
    path treats that as 'registration lost' and never warms up. Here the night picture is
    still registered (histogram-matched LK); a view that cannot be registered at all is
    tracked from the live frame after registration_timeout_seconds, and reported."""
    events: List[Event] = []
    mon = Monitor(monitor_cfg(reference_image=str(calibration), reference_size=[W, H]),
                  detector=FakeDetector(), sinks=[CallbackSink(events.append)])
    night = [exposure(scene(0), 0.15, 1.6, noise=5.0, seed=s) for s in range(4)]
    for i in range(int(2 * FPS)):
        r = mon.process_frame(night[i % 4], i / FPS, i)
        assert r.zone.healthy and np.array_equal(r.zone.polygon, np.int32(ZONE)), i
    assert mon.stabilizer.registered and not [e for e in events if e.type == "HEALTH"]

    elsewhere = [exposure(scene(7), 0.15, 1.6, noise=5.0, seed=s) for s in range(4)]
    mon = Monitor(monitor_cfg(reference_image=str(calibration), reference_size=[W, H]),
                  detector=FakeDetector(), sinks=[CallbackSink(events.append)])
    healthy = []
    for i in range(int(4 * FPS)):
        healthy.append(mon.process_frame(elsewhere[i % 4], i / FPS, i).zone.healthy)
    first = healthy.index(True)
    assert first / FPS == pytest.approx(mon.cfg.zone.registration_timeout_seconds, abs=1.5 / FPS)
    assert np.mean(healthy[first:]) > 0.5 and not mon.stabilizer.registered
    texts = [e.reasons[0] for e in events if e.type == "HEALTH"]
    assert any("not registered against zone.reference_image" in t for t in texts)


def test_monitor_reports_a_missing_reference_image_and_keeps_watching(tmp_path):
    for stabilize in (True, False):
        events: List[Event] = []
        cfg = monitor_cfg(reference_image=str(tmp_path / "moved_away.png"), reference_size=[W, H],
                          stabilize=stabilize)
        mon = Monitor(cfg, detector=FakeDetector(), sinks=[CallbackSink(events.append)])
        r = mon.process_frame(scene(0), 0.0, 0)
        r = mon.process_frame(shifted(scene(0), 0.4, 0.0), 1 / FPS, 1)
        assert r.zone.healthy and np.array_equal(r.zone.polygon, np.int32(ZONE))
        health = [e for e in events if e.type == "HEALTH"]
        assert len(health) == 1 and health[0].extra.get("zone_setup")
        assert "cannot read zone.reference_image" in health[0].reasons[0]
        assert ("anchors on the first textured live frame" in health[0].reasons[0]) == stabilize


def test_relative_reference_image_is_also_looked_up_from_the_repository_root(tmp_path, monkeypatch):
    """calibrate_zone writes configs/<site>_reference.png relative to the repository; a
    monitor started from another working directory (systemd) still finds it."""
    monkeypatch.chdir(tmp_path)
    assert resolve_reference_path("configs/default.yaml") == ROOT / "configs" / "default.yaml"
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "default.yaml").write_text("x", encoding="utf-8")
    assert resolve_reference_path("configs/default.yaml") == Path("configs/default.yaml")  # cwd first
    assert resolve_reference_path("nowhere/x.png") == Path("nowhere/x.png")


# ----------------------------------------------------------------------------- refusal / exit 2
def write_clip(path: Path, n_frames: int = 12) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    if not writer.isOpened():
        pytest.skip("cv2.VideoWriter cannot write mp4v here")
    for _ in range(n_frames):
        writer.write(scene(0))
    writer.release()
    return path


def test_cli_exits_2_when_the_zone_does_not_fit_the_stream(tmp_path, capsys):
    """A 640x480 calibration on a 640x360 stream used to end the run after five frame
    errors with exit status 0 - a supervisor saw success while nothing was watched."""
    cfg = monitor_cfg(reference_size=[640, 480])
    cfg.video.source = str(write_clip(tmp_path / "clip.mp4"))
    cfg.alerts.events_jsonl = str(tmp_path / "events.jsonl")
    path = tmp_path / "site.yaml"
    save_config(cfg, str(path))
    assert main(["--config", str(path)]) == 2
    err = capsys.readouterr().err
    assert "zone refused" in err and "aspect" in err
    log = (tmp_path / "events.jsonl").read_text(encoding="utf-8")
    assert log.count("zone refused") == 1


class LiveCapture:
    """A live stream that renegotiates from 640x360 to 640x480 after ``switch`` frames."""

    def __init__(self, switch: int = 5, total: int = 40) -> None:
        self.i, self.switch, self.total = 0, switch, total

    def read(self):
        self.i += 1
        if self.i > self.total:
            return False, None
        if self.i <= self.switch:
            return True, scene(0)
        return True, cv2.resize(scene(0), (640, 480))

    def isOpened(self) -> bool:  # noqa: N802 - cv2.VideoCapture API
        return True

    def release(self) -> None:
        pass


def test_live_stream_that_changes_aspect_ratio_stops_the_run(monkeypatch):
    """The polygon cannot be mapped onto a 4:3 picture of a 16:9 calibration: the run stops
    with ZoneGeometryError (one 're-initialised' and one 'zone refused' HEALTH event)
    instead of five misleading re-initialisations, and the CLI exits 2."""
    monkeypatch.setattr(monitor_mod, "open_source", lambda video: (LiveCapture(), FPS, True))
    events: List[Event] = []
    cfg = monitor_cfg()
    cfg.video.source = "rtsp://camera/stream"
    mon = Monitor(cfg, detector=FakeDetector(), sinks=[CallbackSink(events.append)])
    with pytest.raises(ZoneGeometryError, match="aspect"):
        mon.run()
    texts = [e.reasons[0] for e in events if e.type == "HEALTH"]
    assert len(texts) == 2 and "re-initialised" in texts[0] and "zone refused" in texts[1]
    monkeypatch.setattr(monitor_mod, "Monitor", lambda c: Monitor(c, detector=FakeDetector()))
    assert main(["--source", "rtsp://camera/stream", "--no-stdout"]) == 2
