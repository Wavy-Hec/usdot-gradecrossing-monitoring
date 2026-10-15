"""Monitor.run driven with fake captures: how a run ends, what it tells the sinks, and why.

``monitor._make_capture`` is monkeypatched with ``FakeCapture`` objects (scripted frames,
container clock, open/closed) and ``time.sleep`` with a recorder, so the file-end, live
reconnect, give-up, repeated-error and Ctrl-C paths run in milliseconds and without a
camera. The frames are the textured synthetic scene of ``test_monitor`` so the zone
stabiliser initialises exactly as on a real clip; live tests that need deterministic
timestamps replace the monitor's clock (``fake_clock``). No test imports ultralytics.
"""
from __future__ import annotations

import json
import logging
import time
import types
from typing import Any, Callable, Dict, Iterable, List, Optional

import cv2
import numpy as np
import pytest

import gradecrossing
import gradecrossing.monitor as monitor_mod
from gradecrossing.alerting import CallbackSink, WebhookSink
from gradecrossing.config import MonitorConfig
from gradecrossing.evaluate import TruthSet, evaluate_clip
from gradecrossing.monitor import (
    DEFAULT_FPS,
    MAX_CONSECUTIVE_FRAME_ERRORS,
    RECONNECT_ATTEMPTS,
    RECONNECT_BACKOFF_S,
    Monitor,
    config_sha256,
    open_source,
)
from gradecrossing.types import Event
from gradecrossing.zone import ZoneGeometryError

from test_monitor import FPS, FakeDetector, Scene, car_det, make_cfg, no_cars  # noqa: E402

RUN_START_KEYS = {"run_id", "source", "config_sha256", "package_version"}  # gradecrossing.evaluate contract
RUN_END_KEYS = {"run_id", "frames", "duration_s", "last_timestamp", "final_status", "stop_reason"}

_SCENE = Scene(0)
FRAME = _SCENE.frame()


class FakeCapture:
    """cv2.VideoCapture stand-in: ``n`` frames, then ``read()`` fails.

    ``pos_msec`` is the container clock per frame (CAP_PROP_POS_MSEC after reading it);
    the default is frame_index * 1000 / fps, as a constant-frame-rate file reports it.
    """

    def __init__(self, n: int = 10, fps: float = FPS, opened: bool = True,
                 pos_msec: Optional[List[float]] = None, frame: Optional[np.ndarray] = None) -> None:
        self.n, self.fps, self.opened = n, fps, opened
        self.pos_msec = pos_msec if pos_msec is not None else [1000.0 * i / fps for i in range(n)]
        self.frame = FRAME if frame is None else frame
        self.i = 0
        self.released = False

    def isOpened(self) -> bool:  # noqa: N802 - cv2 API
        return self.opened

    def read(self):
        if not self.opened or self.i >= self.n:
            return False, None
        self.i += 1
        return True, self.frame.copy()

    def get(self, prop: int) -> float:
        if prop == cv2.CAP_PROP_FPS:
            return self.fps
        if prop == cv2.CAP_PROP_POS_MSEC:
            return float(self.pos_msec[self.i - 1]) if self.i > 0 else 0.0
        return 0.0

    def set(self, prop: int, value: float) -> bool:
        return False

    def release(self) -> None:
        self.released = True


def patch_captures(monkeypatch, captures: Iterable[FakeCapture]) -> List[FakeCapture]:
    """Every _make_capture() call (open_source, reconnects) returns the next capture."""
    made: List[FakeCapture] = []
    pending = list(captures)

    def make(_source: str, _timeout_s: float = 0.0) -> FakeCapture:
        cap = pending.pop(0) if pending else FakeCapture(opened=False)
        made.append(cap)
        return cap

    monkeypatch.setattr(monitor_mod, "_make_capture", make)
    return made


@pytest.fixture
def sleeps(monkeypatch) -> List[float]:
    """time.sleep as seen by the monitor, recorded instead of slept."""
    delays: List[float] = []
    monkeypatch.setattr(monitor_mod.time, "sleep", lambda s: delays.append(float(s)))
    return delays


def run_monitor(cfg: MonitorConfig, detector: Any = None) -> "tuple":
    events: List[Event] = []
    mon = Monitor(cfg, detector=detector if detector is not None else FakeDetector(no_cars),
                  sinks=[CallbackSink(events.append)])
    return mon, mon.run(), events


def of_type(events: List[Event], etype: str) -> List[Event]:
    return [e for e in events if e.type == etype]


def file_cfg(**sections) -> MonitorConfig:
    video = dict(sections.pop("video", {}))
    video.setdefault("source", "clip.mp4")
    sections.setdefault("debris", {"warmup_frames": 2})
    return make_cfg(video=video, **sections)


def live_cfg(**sections) -> MonitorConfig:
    video = dict(sections.pop("video", {}))
    video.setdefault("source", "rtsp://user:secret@camera/stream")
    sections.setdefault("debris", {"warmup_frames": 2})
    return make_cfg(video=video, **sections)


# ----------------------------------------------------------------------------- how a run ends
def test_file_end_processes_every_frame(monkeypatch):
    caps = patch_captures(monkeypatch, [FakeCapture(n=12)])
    mon, summary, events = run_monitor(file_cfg())
    assert summary["frames"] == 12 and summary["errors"] == 0 and summary["reconnects"] == 0
    assert not summary["interrupted"] and summary["loops"] == 0
    assert caps[0].released


def test_live_read_failure_reconnects_and_continues(monkeypatch, sleeps):
    patch_captures(monkeypatch, [FakeCapture(n=5), FakeCapture(opened=False), FakeCapture(n=50)])
    mon, summary, events = run_monitor(live_cfg(video={"max_frames": 10}))
    # attempt 1 opened a dead capture, attempt 2 a live one (its first frame is the check read)
    assert summary["reconnects"] == 1 and summary["frames"] == 10
    assert sleeps == [RECONNECT_BACKOFF_S, 2 * RECONNECT_BACKOFF_S]


def test_live_reconnect_gives_up_after_the_last_attempt(monkeypatch, sleeps):
    patch_captures(monkeypatch, [FakeCapture(n=5)])  # every reconnect gets a closed capture
    mon, summary, events = run_monitor(live_cfg())
    assert summary["frames"] == 5 and summary["reconnects"] == 0
    assert sleeps == [RECONNECT_BACKOFF_S * 2 ** k for k in range(RECONNECT_ATTEMPTS)]


class BrokenDetector(FakeDetector):
    def detect(self, frame_bgr):
        raise RuntimeError("detector is broken")


def test_consecutive_frame_errors_abort(monkeypatch):
    patch_captures(monkeypatch, [FakeCapture(n=30)])
    mon, summary, events = run_monitor(file_cfg(), BrokenDetector(no_cars))
    assert summary["frames"] == 0 and summary["errors"] == MAX_CONSECUTIVE_FRAME_ERRORS


class InterruptingDetector(FakeDetector):
    def detect(self, frame_bgr):
        if len(self.calls) == 4:
            raise KeyboardInterrupt
        return super().detect(frame_bgr)


def test_ctrl_c_stops_cleanly_with_a_summary(monkeypatch):
    caps = patch_captures(monkeypatch, [FakeCapture(n=30)])
    mon, summary, events = run_monitor(file_cfg(), InterruptingDetector(no_cars))
    assert summary["interrupted"] and summary["frames"] == 4
    assert caps[0].released


# ----------------------------------------------------------------------------- RUN_START / RUN_END
def health(events: List[Event], component: str) -> List[Event]:
    return [e for e in events if e.type == "HEALTH" and e.extra.get("component") == component]


def test_run_start_and_run_end_frame_the_run_with_the_evaluate_contract(monkeypatch):
    patch_captures(monkeypatch, [FakeCapture(n=12)])
    cfg = file_cfg()
    mon, summary, events = run_monitor(cfg)
    assert events[0].type == "RUN_START" and events[-1].type == "RUN_END"
    assert len(of_type(events, "RUN_START")) == len(of_type(events, "RUN_END")) == 1
    start, end = events[0].extra, events[-1].extra
    assert set(start) == RUN_START_KEYS and set(end) == RUN_END_KEYS
    assert start["config_sha256"] == config_sha256(cfg) and len(start["config_sha256"]) == 64
    assert start["package_version"] == gradecrossing.__version__ and start["source"] == "clip.mp4"
    assert start["run_id"] == end["run_id"] == summary["run_id"]
    assert end["frames"] == 12 and end["stop_reason"] == "end_of_input" and end["final_status"] == "CLEAR"
    assert end["duration_s"] == pytest.approx(12 / FPS)  # video seconds covered: frames / fps
    assert end["last_timestamp"] == pytest.approx(11 / FPS)
    assert summary["stop_reason"] == "end_of_input" and summary["aborted"] is False
    patch_captures(monkeypatch, [FakeCapture(n=3)])
    assert run_monitor(file_cfg())[1]["run_id"] != summary["run_id"]  # unique per run() call


def test_config_sha256_is_deterministic_and_tracks_the_config():
    a, b = MonitorConfig(), MonitorConfig()
    assert config_sha256(a) == config_sha256(b)
    b.stuck.alarm_seconds = 4.0
    assert config_sha256(a) != config_sha256(b)


def test_max_frames_stop_reason(monkeypatch):
    patch_captures(monkeypatch, [FakeCapture(n=30)])
    mon, summary, events = run_monitor(file_cfg(video={"max_frames": 7}))
    assert (summary["frames"], summary["stop_reason"], summary["aborted"]) == (7, "max_frames", False)
    assert events[-1].extra["stop_reason"] == "max_frames" and events[-1].extra["frames"] == 7


def test_input_ending_during_an_alarm_closes_with_run_end_and_evaluate_clips_it(monkeypatch, tmp_path):
    """ours-realfootage F18: the last event is RUN_END (final_status ALARM); no ALARM_END is
    invented, and evaluate ends the open episode at the run's video duration."""
    patch_captures(monkeypatch, [FakeCapture(n=60)])
    log = tmp_path / "clip.events.jsonl"
    cfg = file_cfg(alerts={"events_jsonl": str(log)})
    mon, summary, events = run_monitor(cfg, FakeDetector(lambda i: [car_det(320.0, 230.0)]))
    assert summary["final_status"] == "ALARM" and of_type(events, "ALARM_START")
    assert not of_type(events, "ALARM_END")
    assert events[-1].type == "RUN_END" and events[-1].extra["final_status"] == "ALARM"
    lines = [json.loads(x) for x in log.read_text(encoding="utf-8").splitlines() if x.strip()]
    assert lines[0]["type"] == "RUN_START" and lines[-1]["type"] == "RUN_END"
    r = evaluate_clip(str(log), TruthSet("clip", None, []), 1.0, "any_overlap")
    alarm_start = of_type(events, "ALARM_START")[0].timestamp
    assert r["notes"] == [] and r["duration_source"] == "RUN_END" and r["duration_s"] == pytest.approx(60 / FPS)
    assert r["fp"] == 1 and r["fa_seconds"] == pytest.approx(60 / FPS - alarm_start)


def test_looped_file_sends_nothing_of_a_pass_after_its_run_end(monkeypatch, tmp_path):
    """FO10: a pass that ends during an ALARM is closed by its RUN_END (the contract above);
    the restart used to follow it with an ALARM_END 'pipeline re-initialised' of that pass.
    Now the next event is the new pass's RUN_START, and the car alarms afresh in pass 2."""
    patch_captures(monkeypatch, [FakeCapture(n=60), FakeCapture(n=60)])
    log = tmp_path / "loop.events.jsonl"
    cfg = file_cfg(video={"loop": True, "max_frames": 100}, alerts={"events_jsonl": str(log)})
    mon, summary, events = run_monitor(cfg, FakeDetector(lambda i: [car_det(320.0, 230.0)]))
    flow = [e for e in events if e.type != "HEALTH"]
    ends = [k for k, e in enumerate(flow) if e.type == "RUN_END"]
    assert len(ends) == 2 and flow[ends[0]].extra["final_status"] == "ALARM"
    assert flow[ends[0] + 1].type == "RUN_START"
    assert not of_type(events, "ALARM_END") and len(of_type(events, "ALARM_START")) == 2
    r = evaluate_clip(str(log), TruthSet("loop", None, []), 1.0, "any_overlap", run="all")
    assert r["notes"] == []  # no 'ALARM_END after RUN_END' any more


def test_looped_file_gets_one_run_per_pass(monkeypatch, tmp_path):
    patch_captures(monkeypatch, [FakeCapture(n=10), FakeCapture(n=10), FakeCapture(n=10)])
    log = tmp_path / "loop.events.jsonl"
    mon, summary, events = run_monitor(file_cfg(video={"loop": True, "max_frames": 25},
                                                alerts={"events_jsonl": str(log)}))
    starts, ends = of_type(events, "RUN_START"), of_type(events, "RUN_END")
    assert summary["loops"] == 2 and len(starts) == len(ends) == 3
    assert [e.extra["stop_reason"] for e in ends] == ["loop", "loop", "max_frames"]
    assert [e.extra["frames"] for e in ends] == [10, 10, 5]
    ids = [e.extra["run_id"] for e in starts]
    assert ids == [e.extra["run_id"] for e in ends] and len(set(ids)) == 3 and ids[1] == summary["run_id"] + "-p2"
    r = evaluate_clip(str(log), TruthSet("loop", None, []), 1.0, "any_overlap", run="all")
    assert r["notes"] == [] and r["runs"]["in_log"] == 3


# ----------------------------------------------------------------------------- aborts and HEALTH
def test_reconnect_outcomes_are_health_events(monkeypatch, sleeps):
    patch_captures(monkeypatch, [FakeCapture(n=5), FakeCapture(opened=False), FakeCapture(n=50)])
    mon, summary, events = run_monitor(live_cfg(video={"max_frames": 10}))
    states = [e.extra.get("state") for e in health(events, "source")]
    assert states == ["lost", "attempt_failed", "reconnected"]
    assert summary["aborted"] is False and summary["stop_reason"] == "max_frames"
    assert summary["health_events"] >= 3


def test_reconnect_give_up_aborts_and_tells_every_sink(monkeypatch, sleeps):
    patch_captures(monkeypatch, [FakeCapture(n=5)])
    mon, summary, events = run_monitor(live_cfg())
    states = [e.extra.get("state") for e in health(events, "source")]
    assert states == ["lost"] + ["attempt_failed"] * RECONNECT_ATTEMPTS + ["gave_up"]
    assert summary["aborted"] is True and summary["stop_reason"] == "reconnect_failed"
    assert events[-1].type == "RUN_END" and events[-1].extra["stop_reason"] == "reconnect_failed"
    assert events[-1].extra["duration_s"] >= 0.0
    # the camera password never leaves the machine
    assert not any("secret" in json.dumps(e.extra) + " ".join(e.reasons) for e in events)
    assert events[0].extra["source"] == "rtsp://***@camera/stream"


def test_frame_errors_abort_with_a_health_event(monkeypatch):
    patch_captures(monkeypatch, [FakeCapture(n=30)])
    mon, summary, events = run_monitor(file_cfg(), BrokenDetector(no_cars))
    assert summary["aborted"] is True and summary["stop_reason"] == "frame_errors"
    (ev,) = health(events, "pipeline")
    assert ev.extra["consecutive_errors"] == MAX_CONSECUTIVE_FRAME_ERRORS and "detector is broken" in ev.extra["error"]
    assert events[-1].type == "RUN_END" and events[-1].extra["stop_reason"] == "frame_errors"
    assert events[-1].extra["frames"] == 0


def test_ctrl_c_still_ends_the_run(monkeypatch):
    patch_captures(monkeypatch, [FakeCapture(n=30)])
    mon, summary, events = run_monitor(file_cfg(), InterruptingDetector(no_cars))
    assert summary["stop_reason"] == "interrupted" and summary["aborted"] is False
    assert events[-1].type == "RUN_END" and events[-1].extra["stop_reason"] == "interrupted"


def test_missing_weights_end_the_run_with_one_clean_error(monkeypatch, tmp_path, caplog):
    """The detector is built on the first frame (after the zone set-up); weights that do not
    exist used to give five tracebacks and a run that looked finished (OC-01)."""
    patch_captures(monkeypatch, [FakeCapture(n=30)])
    events: List[Event] = []
    cfg = file_cfg(detector={"weights": str(tmp_path / "nope.pt")})
    mon = Monitor(cfg, sinks=[CallbackSink(events.append)])
    with caplog.at_level(logging.ERROR), pytest.raises(FileNotFoundError, match="nope.pt"):
        mon.run()
    assert not [r for r in caplog.records if "consecutive error" in r.getMessage()]
    assert [e.type for e in events] == ["RUN_START", "HEALTH", "RUN_END"]
    assert events[-1].extra["stop_reason"] == "detector_unavailable" and events[-1].extra["frames"] == 0
    assert mon.last_summary["aborted"] is True


def test_refused_zone_is_not_a_live_geometry_change_and_ends_the_run(monkeypatch):
    patch_captures(monkeypatch, [FakeCapture(n=30)])
    events: List[Event] = []
    cfg = live_cfg(zone={"polygon": [[200, 150], [440, 150], [440, 270], [200, 270]], "reference_size": [640, 480]})
    mon = Monitor(cfg, detector=FakeDetector(no_cars), sinks=[CallbackSink(events.append)])
    with pytest.raises(ZoneGeometryError):
        mon.run()
    assert not [e for e in events if e.extra.get("reset")]  # no re-initialisation attempt
    assert events[-1].type == "RUN_END" and events[-1].extra["stop_reason"] == "zone_refused"
    assert mon.last_summary["aborted"] is True and mon.last_summary["stop_reason"] == "zone_refused"


# ----------------------------------------------------------------------------- MA2: live geometry
class SceneCapture(FakeCapture):
    """A live camera on the synthetic scene: ``frame_at(i)`` is frame i, and each read sets
    the monitor's clock (``fake_clock``) to i / fps, so live timestamps are deterministic."""

    def __init__(self, frame_at: Callable[[int], np.ndarray], n: int, clock: Dict[str, float]) -> None:
        super().__init__(n=n)
        self.frame_at, self.clock = frame_at, clock

    def read(self):
        if self.i >= self.n:
            return False, None
        self.clock["t"] = self.i / self.fps
        self.i += 1
        return True, self.frame_at(self.i - 1)


@pytest.fixture
def fake_clock(monkeypatch) -> Dict[str, float]:
    """The monitor module's ``time``, with a monotonic clock the capture advances."""
    clock = {"t": 0.0}
    fake = types.SimpleNamespace(monotonic=lambda: clock["t"], perf_counter=time.perf_counter,
                                 sleep=lambda s: None, strftime=time.strftime, time=time.time)
    monkeypatch.setattr(monitor_mod, "time", fake)
    return clock


def test_live_resolution_change_during_a_debris_alarm_keeps_the_alarm(monkeypatch, fake_clock):
    """MA2 (rev-missed-alarms live_reinit.py on clip04): a live stream that renegotiates
    640x360 -> 1280x720 during a debris ALARM used to re-initialise the pipeline: a CLEAR
    ALARM_END, then a debris warm-up that learned the obstruction - silent to the end. The
    frames are now resized to the first frame's size: nothing is lost or reset."""
    scene, blob = Scene(0), (300, 190, 360, 230)
    switch = 120

    def frame_at(i: int) -> np.ndarray:
        img = scene.frame(blob=blob if i >= 60 else None)
        return cv2.resize(img, (1280, 720), interpolation=cv2.INTER_LINEAR) if i >= switch else img

    patch_captures(monkeypatch, [SceneCapture(frame_at, 400, fake_clock)])
    cfg = live_cfg(video={"max_frames": 240}, debris={"warmup_frames": 30, "persist_seconds": 0.5})
    mon, summary, events = run_monitor(cfg)
    (start,) = of_type(events, "ALARM_START")
    assert 2.4 <= start.timestamp < switch / FPS and start.reasons[0].startswith("debris")
    assert not of_type(events, "ALARM_END") and summary["final_status"] == "ALARM"
    assert summary["errors"] == 0 and summary["frames"] == 240 and summary["stop_reason"] == "max_frames"
    assert not [e for e in events if e.extra.get("reset")]
    assert len([e for e in events if e.extra.get("warming_up")]) == 1  # the start-up one only
    (changed,) = [e for e in health(events, "source") if e.extra.get("state") == "resolution_changed"]
    assert changed.frame_index == switch and changed.extra["processing_size"] == [640, 360]
    assert mon.last_result.debris.triggered and mon.last_frame.shape == (360, 640, 3)


def test_live_aspect_change_reinitialises_with_the_alarm_kept(monkeypatch, fake_clock):
    """MA2: a new aspect ratio cannot be resized: the perception is re-initialised (one lost
    frame, one HEALTH event with extra.reset) but the alert state is kept - no CLEAR
    ALARM_END while the car is still there - and the re-initialised pipeline raises the car
    again inside the latch. (zone.reference_size admits both shapes, 640x363 and 640x358,
    which are 1.4% apart: usually the new shape is refused and the run stops with
    zone_refused, see test_zone_registration.)"""
    scene = Scene(0)
    car = car_det(320.0, 230.0)

    def frame_at(i: int) -> np.ndarray:
        return cv2.resize(scene.frame([car.xyxy]), (640, 363) if i < 90 else (640, 358),
                          interpolation=cv2.INTER_LINEAR)

    patch_captures(monkeypatch, [SceneCapture(frame_at, 400, fake_clock)])
    cfg = live_cfg(video={"max_frames": 180}, zone={"reference_size": [640, 360]},
                   alerts={"latch_seconds": 3.0}, debris={"enabled": False})
    fake = FakeDetector(lambda i: [car_det(320.0, 230.0)])
    mon, summary, events = run_monitor(cfg, fake)
    (reset,) = [e for e in events if e.extra.get("reset")]
    assert reset.frame_index == 90 and reset.extra["previous_status"] == "ALARM" and reset.status == "ALARM"
    assert "re-initialised" in reset.reasons[0] and "status ALARM kept" in reset.reasons[0]
    assert not of_type(events, "ALARM_END")
    assert [e.frame_index < 40 for e in of_type(events, "WARNING_END")] == [True]  # WARNING -> ALARM only
    assert len(of_type(events, "ALARM_START")) == 1 and summary["final_status"] == "ALARM"
    assert summary["errors"] == 1 and fake.resets == 1 and mon.last_frame.shape == (358, 640, 3)
    assert mon.last_result.assessment.alarm_track_ids  # raised again by the new pipeline, not only latched


# ----------------------------------------------------------------------------- FO6: stream timeouts
class RecordingVideoCapture:
    """cv2.VideoCapture stand-in that records how it was called."""

    calls: List[tuple] = []
    reject_params = False

    def __init__(self, *args) -> None:
        if len(args) > 1 and RecordingVideoCapture.reject_params:
            raise TypeError("VideoCapture() takes at most 2 arguments")
        RecordingVideoCapture.calls.append(args)


def test_stream_urls_are_opened_with_timeouts(monkeypatch):
    RecordingVideoCapture.calls, RecordingVideoCapture.reject_params = [], False
    monkeypatch.setattr(cv2, "VideoCapture", RecordingVideoCapture)
    monitor_mod._make_capture("rtsp://user:pw@cam/stream", 10.0)
    monitor_mod._make_capture("clip.mp4", 10.0)
    monitor_mod._make_capture("0", 10.0)
    monitor_mod._make_capture("http://cam/video.mjpg", 0.0)  # 0 = the backend default
    open_ms, read_ms = cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, cv2.CAP_PROP_READ_TIMEOUT_MSEC
    assert RecordingVideoCapture.calls == [
        ("rtsp://user:pw@cam/stream", cv2.CAP_FFMPEG, [open_ms, 10000, read_ms, 10000]),
        ("clip.mp4",), (0,), ("http://cam/video.mjpg",)]
    RecordingVideoCapture.calls, RecordingVideoCapture.reject_params = [], True
    monitor_mod._make_capture("rtsp://cam/stream", 5.0)  # a binding without the params overload
    assert RecordingVideoCapture.calls == [("rtsp://cam/stream",)]
    # open_source and the reconnects pass video.stream_timeout_seconds
    got: List[tuple] = []
    monkeypatch.setattr(monitor_mod, "_make_capture", lambda s, t=0.0: got.append((s, t)) or FakeCapture(n=1))
    open_source(live_cfg(video={"stream_timeout_seconds": 4.0}).video)
    assert got == [("rtsp://user:secret@camera/stream", 4.0)]


@pytest.mark.skipif(monitor_mod._stream_timeout_params(1.0) is None,
                    reason="this OpenCV build cannot set stream timeouts")
def test_a_dead_stream_url_gives_up_after_the_timeout():
    """A refused RTSP connection blocked cv2.VideoCapture for the backend default (30 s
    measured, OpenCV 4.14 on Windows); with the timeout it gives up after it."""
    t0 = time.monotonic()
    cap = monitor_mod._make_capture("rtsp://127.0.0.1:9/stream", 0.5)
    try:
        assert not cap.isOpened()
    finally:
        cap.release()
    assert time.monotonic() - t0 < 10.0


# ----------------------------------------------------------------------------- outputs (MISS-03)
def test_unusable_snapshot_dir_fails_before_the_first_frame(monkeypatch, tmp_path):
    patch_captures(monkeypatch, [FakeCapture(n=10)])
    blocker = tmp_path / "a_file"
    blocker.write_text("x", encoding="utf-8")
    events: List[Event] = []
    mon = Monitor(file_cfg(output={"snapshot_dir": str(blocker / "snaps")}), detector=FakeDetector(no_cars),
                  sinks=[CallbackSink(events.append)])
    with pytest.raises(RuntimeError, match="output.snapshot_dir"):
        mon.run()
    assert events == []  # the run never started


def test_snapshot_failure_disables_snapshots_and_monitoring_continues(monkeypatch, tmp_path):
    patch_captures(monkeypatch, [FakeCapture(n=60)])
    calls: List[str] = []
    monkeypatch.setattr(cv2, "imwrite", lambda path, img: calls.append(path) or False)
    cfg = file_cfg(output={"snapshot_dir": str(tmp_path / "snaps")})
    mon, summary, events = run_monitor(cfg, FakeDetector(lambda i: [car_det(320.0, 230.0)]))
    assert summary["frames"] == 60 and summary["alarms"] >= 1 and len(calls) == 1
    (ev,) = health(events, "pipeline")
    assert ev.extra["output"] == "snapshot" and cfg.output.snapshot_dir == str(tmp_path / "snaps")


def test_overlay_failure_keeps_the_video_and_the_run(monkeypatch, tmp_path):
    patch_captures(monkeypatch, [FakeCapture(n=8)])

    def broken(*_a, **_k):
        raise RuntimeError("overlay bug")

    monkeypatch.setattr(monitor_mod, "draw_overlay", broken)
    out = tmp_path / "out" / "annotated.mp4"
    mon, summary, events = run_monitor(file_cfg(output={"save_video": str(out)}))
    assert summary["frames"] == 8 and summary["errors"] == 0
    assert [e.extra["output"] for e in health(events, "pipeline")] == ["overlay"]
    assert out.exists() and out.stat().st_size > 0  # the plain frames were written


def test_video_writer_failure_disables_the_video_without_touching_the_config(monkeypatch, tmp_path):
    patch_captures(monkeypatch, [FakeCapture(n=8)])

    class DeadWriter:
        def __init__(self, *a, **k):
            pass

        def isOpened(self):  # noqa: N802
            return False

        def release(self):
            pass

    monkeypatch.setattr(cv2, "VideoWriter", DeadWriter)
    cfg = file_cfg(output={"save_video": str(tmp_path / "v.mp4")})
    mon, summary, events = run_monitor(cfg)
    assert summary["frames"] == 8
    assert [e.extra["output"] for e in health(events, "pipeline")] == ["save_video"]
    assert cfg.output.save_video == str(tmp_path / "v.mp4")  # OC-08: run() never writes to cfg


# ----------------------------------------------------------------------------- timestamps
def record_timestamps(mon: Monitor) -> List[float]:
    seen: List[float] = []
    orig = mon.process_frame

    def wrapped(frame, timestamp, frame_index):
        seen.append(timestamp)
        return orig(frame, timestamp, frame_index)

    mon.process_frame = wrapped  # type: ignore[assignment]
    return seen


def run_clock(monkeypatch, pos_msec: List[float], **video) -> "tuple":
    patch_captures(monkeypatch, [FakeCapture(n=len(pos_msec), pos_msec=pos_msec)])
    events: List[Event] = []
    mon = Monitor(file_cfg(video=video), detector=FakeDetector(no_cars), sinks=[CallbackSink(events.append)])
    seen = record_timestamps(mon)
    mon.run()
    return seen, events


def test_file_timestamps_follow_the_container_clock(monkeypatch, caplog):
    """ours-realfootage M3: clip01 is variable frame rate (8-58 ms gaps); frame_index / fps ran
    up to 0.84 s ahead of the container clock."""
    pos = [0.0, 8.3, 50.0, 58.3, 100.0, 300.0, 316.7, 333.3]
    with caplog.at_level(logging.WARNING, logger="gradecrossing.monitor"):
        seen, events = run_clock(monkeypatch, pos)
    assert seen == pytest.approx([p / 1000.0 for p in pos])
    warned = [r for r in caplog.records if "container clock" in r.getMessage()]
    assert len(warned) == 1  # once, when the two clocks first differ by more than 0.1 s
    assert events[-1].extra["duration_s"] == pytest.approx(0.3333 + 1 / FPS)


def test_file_timestamps_without_a_usable_clock(monkeypatch):
    zeros, _ = run_clock(monkeypatch, [0.0] * 6)
    assert zeros == pytest.approx([i / FPS for i in range(6)])  # frame_index / fps
    back, _ = run_clock(monkeypatch, [0.0, 33.3, 66.7, 50.0, 133.3])  # one value goes backwards
    assert back == pytest.approx([0.0, 0.0333, 0.0667, 0.0667 + 1 / FPS, 0.1333], abs=1e-4)
    assert all(b > a for a, b in zip(back, back[1:]))
    forced, _ = run_clock(monkeypatch, [0.0, 8.3, 50.0, 58.3], fps_override=10.0)
    assert forced == pytest.approx([0.0, 0.1, 0.2, 0.3])  # fps_override: frame_index / fps


@pytest.mark.parametrize("reported, expected", [(1.0, 1.0), (0.5, 0.5), (500.0, 500.0), (0.0, DEFAULT_FPS),
                                                (float("nan"), DEFAULT_FPS), (-3.0, DEFAULT_FPS)])
def test_fps_fallback_accepts_any_finite_positive_rate(monkeypatch, reported, expected):
    """ours-correctness F18: a genuine 1 fps recording used to be re-timed at 30 fps."""
    patch_captures(monkeypatch, [FakeCapture(n=1, fps=reported, pos_msec=[0.0])])
    cap, fps, live = open_source(file_cfg().video)
    assert fps == expected and not live


# ----------------------------------------------------------------------------- sinks
def test_close_calls_close_on_every_config_sink(monkeypatch):
    closed: List[Any] = []

    class Sink:
        def __init__(self, fail: bool = False) -> None:
            self.fail = fail

        def emit(self, event):
            pass

        def close(self, *args, **kwargs):
            closed.append((self, args, kwargs))
            if self.fail:
                raise OSError("disk gone")

    mon = Monitor(file_cfg(), detector=FakeDetector(no_cars))
    sinks = [Sink(fail=True), Sink()]
    mon._config_sinks = sinks
    mon.close()  # a failing close is logged, the next sink is still closed
    assert [c[0] for c in closed] == sinks and all(c[1:] == ((), {}) for c in closed)
    calls: List[Any] = []
    monkeypatch.setattr(WebhookSink, "close", lambda self, timeout=None: calls.append(timeout) or True)
    Monitor(file_cfg(alerts={"webhook_url": "http://127.0.0.1:9/hook"}), detector=FakeDetector(no_cars)).close()
    assert calls == [None]  # WebhookSink.close() default: one per-event delivery deadline
