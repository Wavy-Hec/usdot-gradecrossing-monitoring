"""End-to-end Monitor scenarios for the footprint / too-close / ID-free rule and its timing
(synthetic frames, scripted detections; helpers from tests/test_monitor.py)."""
from __future__ import annotations

import logging
from typing import List

import numpy as np

from gradecrossing.alerting import CallbackSink
from gradecrossing.config import StuckConfig
from gradecrossing.monitor import Monitor
from gradecrossing.types import Detection, Status

from test_monitor import FPS, FakeDetector, Session, make_cfg, no_cars, write_clip  # noqa: E402

# clip08's thin oblique zone and the trapped van's YOLO11n box (t = 3.30 s): bottom-center
# outside the zone, bottom-quarter footprint 41 % inside it.
OBLIQUE_ZONE = [[268, 110], [303, 110], [520, 300], [415, 300]]
VAN = np.array([245.0, 117.0, 402.0, 199.0], np.float32)


def van_script(start: int, end: int = 10 ** 6, tid_every: int = 0, jitter: float = 0.0, seed: int = 0):
    rng = np.random.default_rng(seed)
    noise = rng.uniform(-jitter, jitter, (4000, 4)).astype(np.float32) if jitter else None

    def script(i: int) -> List[Detection]:
        if i < start or i >= end:
            return []
        tid = 1 + ((i - start) // tid_every if tid_every else 0)
        box = VAN + (noise[i] if noise is not None else 0.0)
        return [Detection(xyxy=box, conf=0.9, cls_id=7, cls_name="truck", track_id=tid)]

    return script


def test_scenario_f_oblique_zone_alarms_by_footprint_only():
    for test, alarms in (("footprint", True), ("bottom_center", False)):
        cfg = make_cfg(zone={"polygon": OBLIQUE_ZONE}, stuck={"zone_test": test})
        s = Session(cfg, van_script(start=30))
        res = s.run(int(4.0 * FPS))
        starts = s.events_of("ALARM_START")
        assert bool(starts) is alarms, test
        if alarms:
            # stationary from frame 30 (t = 1.0 s): ALARM at alarm_seconds (1.0 s) after the stop,
            # + one frame of min_track_age_frames
            assert 2.0 <= starts[0].timestamp <= 2.0 + 2.0 / FPS
            assert any("stationary in zone" in r for r in starts[0].reasons)
            assert res[-1].objects[0].in_zone and s.mon.tracker.zone_occupied


def test_scenario_g_box_jitter_does_not_delay_the_alarm():
    """The van's box jitters by +-6 px and the detector re-ids it every 0.5 s: the ID-free clock
    keeps one stop and alarms on time; speed-only stationarity is pushed out by the jitter."""
    stuck = {"alarm_seconds": 2.0, "warning_seconds": 1.0}
    s = Session(make_cfg(zone={"polygon": OBLIQUE_ZONE}, stuck=stuck), van_script(start=30, tid_every=15, jitter=6.0))
    s.run(int(5.0 * FPS))
    starts = s.events_of("ALARM_START")
    assert starts and starts[0].timestamp <= 1.0 + 2.0 + 2.0 / FPS
    s2 = Session(make_cfg(zone={"polygon": OBLIQUE_ZONE}, stuck=dict(stuck, stationarity="speed")),
                 van_script(start=30, tid_every=15, jitter=6.0))
    s2.run(int(5.0 * FPS))
    late = s2.events_of("ALARM_START")
    assert not late or late[0].timestamp >= starts[0].timestamp + 0.3


def test_scenario_h_car_waiting_next_to_the_zone_warns_but_never_alarms():
    zone = [[0, 150], [640, 150], [640, 175], [0, 175]]  # thin band
    car = np.array([200.0, 160.0, 320.0, 240.0], np.float32)  # footprint 40 px below: 0.5 box heights

    def script(i: int) -> List[Detection]:
        return [Detection(xyxy=car.copy(), conf=0.9, cls_id=2, cls_name="car", track_id=1)] if i >= 30 else []

    s = Session(make_cfg(zone={"polygon": zone}), script)
    res = s.run(int(6.0 * FPS))
    assert Status.ALARM not in {r.assessment.status for r in res}
    warn = s.events_of("WARNING_START")
    assert warn and any("stopped near the tracks" in r for r in warn[0].reasons)
    assert not s.mon.tracker.zone_occupied  # a near car does not block the background refresh
    s2 = Session(make_cfg(zone={"polygon": zone}, stuck={"close_policy": "alarm"}), script)
    s2.run(int(6.0 * FPS))
    assert any("stopped on/near the tracks" in r for e in s2.events_of("ALARM_START") for r in e.reasons)


def test_scenario_clip07_like_stop_alarms_while_the_train_hides_the_van():
    """clip07/clip08: the van stands on the crossing for ~2.3 s, then the train hides it. Its track
    lives lost_track_grace_seconds (1 s) longer, and its clocks keep running while it does
    (undetected_clock_while_live, MA5): with the shipped thresholds (WARNING 1.5 s, ALARM 3 s) the
    ALARM comes at 3 s of stuck time, while the van is hidden. The cost: a vehicle the detector
    loses after 2 s in the zone alarms whether or not it is still there. With the clocks frozen at
    the last detection (the rule before MA5) it is a WARNING and no ALARM; nothing at all with the
    pre-port rule's 3 s / 8 s and frozen clocks."""
    shipped = {"warning_seconds": StuckConfig().warning_seconds, "alarm_seconds": StuckConfig().alarm_seconds}
    assert shipped == {"warning_seconds": 1.5, "alarm_seconds": 3.0}
    last_seen = (30 + int(2.3 * FPS) - 1) / FPS
    script = van_script(start=30, end=30 + int(2.3 * FPS))
    s = Session(make_cfg(zone={"polygon": OBLIQUE_ZONE}, stuck=shipped), script)
    s.run(int(6.0 * FPS))
    warn = s.events_of("WARNING_START")
    assert warn and 1.0 + 1.5 <= warn[0].timestamp <= 1.0 + 1.5 + 2.0 / FPS
    assert any("stationary in zone" in r for r in warn[0].reasons)
    alarm = s.events_of("ALARM_START", repeats=True)
    assert len(alarm) == 1 and 1.0 + 3.0 <= alarm[0].timestamp <= 1.0 + 3.0 + 2.0 / FPS
    assert last_seen < alarm[0].timestamp <= last_seen + StuckConfig().lost_track_grace_seconds
    assert any("stationary in zone" in r for r in alarm[0].reasons)
    frozen = dict(shipped, undetected_clock_while_live=False)
    s = Session(make_cfg(zone={"polygon": OBLIQUE_ZONE}, stuck=frozen), script)
    s.run(int(6.0 * FPS))
    assert s.events_of("WARNING_START") and not s.events_of("ALARM_START", repeats=True)
    legacy = Session(make_cfg(zone={"polygon": OBLIQUE_ZONE}, stuck={"warning_seconds": 3.0, "alarm_seconds": 8.0,
                                                                    "undetected_clock_while_live": False}),
                     script)
    legacy.run(int(6.0 * FPS))
    assert not legacy.events_of("WARNING_START") and not legacy.events_of("ALARM_START")


def test_run_warns_when_detection_is_slower_than_the_still_gap(tmp_path, caplog):
    clip = write_clip(tmp_path / "clip.mp4", n_frames=45)
    events = []
    cfg = make_cfg(video={"source": str(clip)}, detector={"detect_every_n_frames": 40})
    with caplog.at_level(logging.WARNING):
        Monitor(cfg, detector=FakeDetector(no_cars), sinks=[CallbackSink(events.append)]).run()
    assert any("still_max_gap_seconds" in r.getMessage() for r in caplog.records)
    health = [e for e in events if e.type == "HEALTH" and "detection_gap_s" in e.extra]
    assert len(health) == 1 and abs(health[0].extra["detection_gap_s"] - 40 / FPS) < 1e-6


def test_slow_live_detection_emits_one_health_event():
    """A live source delivering frames 1.5 s apart (slower than its nominal rate): one HEALTH
    event per run says the ID-free clock is off; nothing with the speed-only rule."""
    for stationarity, expected in (("speed_or_still", 1), ("speed", 0)):
        s = Session(make_cfg(stuck={"stationarity": stationarity}), no_cars)
        for i in range(6):
            s.fake.index = i
            s.mon.process_frame(s.frame(i), 1.5 * i, i)
        health = [e for e in s.events if e.type == "HEALTH" and "detection_gap_s" in e.extra]
        assert len(health) == expected, stationarity
        if expected:
            assert "still_max_gap_seconds" in health[0].reasons[0]
        s.mon.reset()  # a new run reports again
        for i in range(3):
            s.mon.process_frame(s.frame(i), 1.5 * i, i)
        again = [e for e in s.events if e.type == "HEALTH" and "detection_gap_s" in e.extra]
        assert len(again) == 2 * expected
