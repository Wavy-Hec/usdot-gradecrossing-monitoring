"""Tests for gradecrossing.alerting (AlertEngine + sinks).

All inputs are hand-built ObjectState / DebrisResult / GateState instances so the
decision logic is exercised without a detector, tracker or video.
"""
from __future__ import annotations

import http.server
import json
import threading
import time
from typing import List, Optional

import numpy as np
import pytest

from gradecrossing.alerting import (
    AlertEngine,
    CallbackSink,
    JsonlSink,
    StdoutSink,
    WebhookSink,
    build_sinks,
    event_to_json,
)
from gradecrossing.config import AlertConfig, StuckConfig
from gradecrossing.types import DebrisResult, Event, GateState, ObjectState, Status


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def make_obj(track_id: int, kind: str = "vehicle", cls_name: Optional[str] = None,
             stuck: float = 0.0, dwell: float = 0.0, in_zone: bool = True,
             speed: float = 0.0, t: float = 0.0) -> ObjectState:
    """Build an ObjectState directly (kind: vehicle | person | train | other)."""
    if cls_name is None:
        cls_name = {"vehicle": "car", "person": "person", "train": "train"}.get(kind, "dog")
    return ObjectState(
        track_id=track_id,
        cls_name=cls_name,
        is_vehicle=(kind == "vehicle"),
        is_person=(kind == "person"),
        is_train=(kind == "train"),
        last_box=np.array([100, 100, 160, 140], dtype=np.float32),
        last_seen=t,
        first_seen=max(0.0, t - dwell),
        in_zone=in_zone,
        in_zone_since=(t - dwell) if in_zone else None,
        dwell_seconds=dwell if in_zone else 0.0,
        speed=speed,
        speed_units="px/s",
        stationary_seconds=stuck,
        stuck_seconds=stuck if in_zone else 0.0,
        history=[],
    )


def make_debris(triggered: bool, area_fraction: float = 0.041, persist: float = 3.2) -> DebrisResult:
    return DebrisResult(
        mask=np.zeros((10, 10), dtype=np.uint8),
        area_px=int(area_fraction * 100),
        area_fraction=area_fraction,
        blobs=[],
        persist_seconds=persist,
        triggered=triggered,
        learning_rate_used=0.0,
        warmed_up=True,
    )


def make_gate(down: bool = False) -> GateState:
    return GateState(gates_down=down, lights_on=down, confidence=0.9 if down else 0.0,
                     last_update=0.0, last_seen_down=0.0 if down else None, available=True)


def make_engine(latch: float = 10.0, repeat: float = 0.0, hold: float = 0.0, **stuck_overrides) -> AlertEngine:
    """Engine with the pre-port 3 s / 8 s stuck timing pinned (these tests were written for it;
    the shipped 1.5 s / 3 s defaults are pinned in tests/test_rules_alerting.py) and no WARNING hold."""
    stuck = StuckConfig(**{"warning_seconds": 3.0, "alarm_seconds": 8.0, **stuck_overrides})
    alerts = AlertConfig(latch_seconds=latch, repeat_notify_seconds=repeat, warning_hold_seconds=hold, stdout=False)
    return AlertEngine(stuck, alerts)


def collect(engine: AlertEngine) -> List[Event]:
    got: List[Event] = []
    engine.add_sink(CallbackSink(got.append))
    return got


# --------------------------------------------------------------------------------------
# decision logic
# --------------------------------------------------------------------------------------
def test_clear_with_moving_vehicles():
    eng = make_engine()
    events = collect(eng)
    gate = make_gate(False)
    # Traffic crossing the tracks: in zone briefly, moving fast, never stuck.
    for i in range(30):
        t = i / 30.0
        objs = [make_obj(1, dwell=t, speed=200.0, t=t), make_obj(2, dwell=0.0, in_zone=False, speed=150.0, t=t)]
        a, ev = eng.evaluate(objs, None, gate, t, i)
        assert a.status is Status.CLEAR
        assert a.reasons == []
        assert a.alarm_track_ids == [] and a.warning_track_ids == []
        assert a.latched_until is None
        assert a.severity_scale == pytest.approx(1.0)
        assert ev == []
    assert events == []
    assert eng.status is Status.CLEAR


def test_warning_then_alarm_as_stuck_crosses_thresholds():
    eng = make_engine(latch=0.0)
    gate = make_gate(False)
    # defaults: warning 3 s, alarm 8 s
    a, _ = eng.evaluate([make_obj(7, stuck=2.9, dwell=2.9)], None, gate, 2.9, 87)
    assert a.status is Status.CLEAR
    a, ev = eng.evaluate([make_obj(7, stuck=3.0, dwell=3.0)], None, gate, 3.0, 90)
    assert a.status is Status.WARNING
    assert a.warning_track_ids == [7] and a.alarm_track_ids == []
    assert [e.type for e in ev] == ["WARNING_START"]
    assert "vehicle 7" in a.reasons[0] and "stationary" in a.reasons[0]
    a, ev = eng.evaluate([make_obj(7, stuck=7.9, dwell=7.9)], None, gate, 7.9, 237)
    assert a.status is Status.WARNING and ev == []
    a, ev = eng.evaluate([make_obj(7, stuck=8.3, dwell=8.3)], None, gate, 8.3, 249)
    assert a.status is Status.ALARM
    assert a.alarm_track_ids == [7] and a.warning_track_ids == []
    assert [e.type for e in ev] == ["WARNING_END", "ALARM_START"]
    assert ev[1].track_ids == [7]
    assert ev[1].status == "ALARM"
    assert a.reasons == ["vehicle 7 (car) stationary in zone 8.3s"]
    assert not a.debris_alarm and not a.person_alert and not a.gates_down


def test_dwell_fallback_alarms_slow_creeping_vehicle():
    eng = make_engine(latch=0.0)
    gate = make_gate(False)
    # Creeping (speed above threshold so stuck never accumulates) but in the zone 26 s.
    a, _ = eng.evaluate([make_obj(3, stuck=0.0, dwell=13.0, speed=20.0)], None, gate, 13.0, 390)
    assert a.status is Status.WARNING and a.warning_track_ids == [3]
    assert "in zone 13.0s" in a.reasons[0]
    a, _ = eng.evaluate([make_obj(3, stuck=0.0, dwell=26.0, speed=20.0)], None, gate, 26.0, 780)
    assert a.status is Status.ALARM and a.alarm_track_ids == [3]


def test_gate_down_shortens_thresholds():
    eng = make_engine(latch=0.0)
    # 3.5 s stationary: only a WARNING with gates up ...
    a, _ = eng.evaluate([make_obj(7, stuck=3.5, dwell=3.5)], None, make_gate(False), 3.5, 105)
    assert a.status is Status.WARNING
    assert a.severity_scale == pytest.approx(1.0)
    # ... but an ALARM with gates down (gates_down_alarm_seconds = 3.0)
    a, _ = eng.evaluate([make_obj(7, stuck=3.5, dwell=3.5)], None, make_gate(True), 3.6, 108)
    assert a.status is Status.ALARM
    assert a.gates_down is True
    assert a.severity_scale == pytest.approx(3.0 / 8.0)
    assert a.reasons == ["vehicle 7 (car) stationary in zone 3.5s (gates down)"]
    # gates-down warning at 1 s
    eng.reset()
    a, _ = eng.evaluate([make_obj(8, stuck=1.2, dwell=1.2)], None, make_gate(True), 1.2, 36)
    assert a.status is Status.WARNING and a.warning_track_ids == [8]
    # gates-down dwell alarm at 6 s regardless of speed
    eng.reset()
    a, _ = eng.evaluate([make_obj(9, stuck=0.0, dwell=6.5, speed=30.0)], None, make_gate(True), 6.5, 195)
    assert a.status is Status.ALARM and a.alarm_track_ids == [9]


def test_latch_holds_alarm_then_releases():
    eng = make_engine(latch=10.0)
    events = collect(eng)
    gate = make_gate(False)
    a, _ = eng.evaluate([make_obj(7, stuck=9.0, dwell=9.0)], None, gate, 10.0, 300)
    assert a.status is Status.ALARM
    assert a.latched_until == pytest.approx(20.0)
    # Object vanishes (tracker dropped it): ALARM must be held.
    a, ev = eng.evaluate([], None, gate, 11.0, 330)
    assert a.status is Status.ALARM
    assert a.reasons == ["latched"]
    assert a.alarm_track_ids == []  # raw ids are reported honestly
    assert a.latched_until == pytest.approx(20.0)
    assert ev == []
    a, _ = eng.evaluate([], None, gate, 19.99, 599)
    assert a.status is Status.ALARM
    # Latch expires -> CLEAR and ALARM_END exactly once.
    a, ev = eng.evaluate([], None, gate, 20.0, 600)
    assert a.status is Status.CLEAR
    assert a.latched_until is None
    assert a.reasons == []
    assert [e.type for e in ev] == ["ALARM_END"]
    # The END event names what ended: the episode's ids and the reason that opened it (it
    # used to carry the previous frame's raw ids, [] after a latch).
    assert ev[0].track_ids == [7]
    assert ev[0].reasons == ["vehicle 7 (car) stationary in zone 9.0s"]
    assert [e.type for e in events] == ["ALARM_START", "ALARM_END"]
    assert events[0].track_ids == [7]


def test_latch_is_extended_while_condition_persists():
    eng = make_engine(latch=5.0)
    gate = make_gate(False)
    for t in (10.0, 11.0, 12.0):
        a, _ = eng.evaluate([make_obj(1, stuck=9.0, dwell=9.0)], None, gate, t, int(t * 30))
        assert a.latched_until == pytest.approx(t + 5.0)
    # Condition gone at t=13: latched until 17
    a, _ = eng.evaluate([], None, gate, 16.9, 507)
    assert a.status is Status.ALARM
    a, _ = eng.evaluate([], None, gate, 17.0, 510)
    assert a.status is Status.CLEAR


def test_latched_alarm_keeps_underlying_warning_reasons():
    eng = make_engine(latch=10.0)
    gate = make_gate(False)
    eng.evaluate([make_obj(1, stuck=9.0, dwell=9.0)], None, gate, 10.0, 300)
    # Vehicle 1 left; vehicle 2 is only at WARNING while the latch holds.
    a, _ = eng.evaluate([make_obj(2, stuck=4.0, dwell=4.0)], None, gate, 12.0, 360)
    assert a.status is Status.ALARM
    assert a.reasons[0] == "latched"
    assert any("vehicle 2" in r for r in a.reasons[1:])
    assert a.warning_track_ids == [2]
    # Latch expires while vehicle 2 is still WARNING -> ALARM_END then WARNING_START.
    a, ev = eng.evaluate([make_obj(2, stuck=5.0, dwell=5.0)], None, gate, 20.5, 615)
    assert a.status is Status.WARNING
    assert [e.type for e in ev] == ["ALARM_END", "WARNING_START"]
    assert ev[1].track_ids == [2]


def test_debris_alarm():
    eng = make_engine(latch=0.0)
    gate = make_gate(False)
    a, ev = eng.evaluate([], make_debris(False, persist=1.0), gate, 1.0, 30)
    assert a.status is Status.CLEAR and not a.debris_alarm
    a, ev = eng.evaluate([], make_debris(True, area_fraction=0.041, persist=3.2), gate, 3.2, 96)
    assert a.status is Status.ALARM
    assert a.debris_alarm is True
    assert a.alarm_track_ids == []
    assert a.reasons == ["debris 4.1% of zone for 3.2s"]
    assert [e.type for e in ev] == ["ALARM_START"]
    # debris=None (detector disabled) never alarms
    eng.reset()
    a, _ = eng.evaluate([], None, gate, 4.0, 120)
    assert a.status is Status.CLEAR and not a.debris_alarm


def test_person_warning_and_alarm():
    eng = make_engine(latch=0.0)
    gate = make_gate(False)
    a, _ = eng.evaluate([make_obj(3, kind="person", dwell=1.5)], None, gate, 1.5, 45)
    assert a.status is Status.CLEAR and not a.person_alert
    a, ev = eng.evaluate([make_obj(3, kind="person", dwell=2.5)], None, gate, 2.5, 75)
    assert a.status is Status.WARNING
    assert a.person_alert is True
    assert a.warning_track_ids == [3]
    assert a.reasons == ["person 3 in zone 2.5s"]
    assert [e.type for e in ev] == ["WARNING_START"]
    a, _ = eng.evaluate([make_obj(3, kind="person", dwell=10.0)], None, gate, 10.0, 300)
    assert a.status is Status.ALARM and a.alarm_track_ids == [3] and a.person_alert
    # A person outside the zone (dwell 0) is ignored.
    eng.reset()
    a, _ = eng.evaluate([make_obj(4, kind="person", dwell=0.0, in_zone=False)], None, gate, 11.0, 330)
    assert a.status is Status.CLEAR and not a.person_alert


def test_train_and_other_classes_are_ignored():
    eng = make_engine(latch=0.0)
    gate = make_gate(True)
    objs = [make_obj(1, kind="train", stuck=60.0, dwell=120.0),
            make_obj(2, kind="other", stuck=60.0, dwell=120.0)]
    a, ev = eng.evaluate(objs, None, gate, 120.0, 3600)
    assert a.status is Status.CLEAR
    assert a.reasons == [] and ev == []


def test_vehicle_outside_zone_never_alerts_even_with_zero_thresholds():
    # Defensive: a threshold of 0 s must not alarm on traffic that never entered the zone.
    eng = make_engine(latch=0.0, warning_seconds=0.0, alarm_seconds=0.0)
    a, _ = eng.evaluate([make_obj(1, in_zone=False, speed=0.0)], None, make_gate(False), 1.0, 30)
    assert a.status is Status.CLEAR
    a, _ = eng.evaluate([make_obj(1, in_zone=True, stuck=0.0, dwell=0.0)], None, make_gate(False), 1.1, 33)
    assert a.status is Status.ALARM  # in zone with a 0 s threshold does alarm


def test_multiple_objects_sorted_and_mixed_severity():
    eng = make_engine(latch=0.0)
    gate = make_gate(False)
    objs = [make_obj(12, stuck=9.0, dwell=9.0), make_obj(3, stuck=4.0, dwell=4.0),
            make_obj(5, kind="person", dwell=3.0), make_obj(1, dwell=0.5, speed=100.0)]
    a, ev = eng.evaluate(objs, make_debris(True), gate, 9.0, 270)
    assert a.status is Status.ALARM
    assert a.alarm_track_ids == [12]
    assert a.warning_track_ids == [3, 5]
    assert a.person_alert and a.debris_alarm
    # alarm reasons first (vehicle then debris), then warnings in id order
    assert a.reasons[0].startswith("vehicle 12")
    assert a.reasons[1].startswith("debris")
    assert a.reasons[2].startswith("vehicle 3")
    assert a.reasons[3].startswith("person 5")
    assert [e.type for e in ev] == ["ALARM_START"]


def test_event_sequence_warning_then_alarm_then_clear():
    eng = make_engine(latch=0.0)
    events = collect(eng)
    gate = make_gate(False)
    eng.evaluate([make_obj(7, stuck=4.0, dwell=4.0)], None, gate, 4.0, 120)
    eng.evaluate([make_obj(7, stuck=9.0, dwell=9.0)], None, gate, 9.0, 270)
    eng.evaluate([], None, gate, 10.0, 300)
    assert [e.type for e in events] == ["WARNING_START", "WARNING_END", "ALARM_START", "ALARM_END"]
    assert [e.status for e in events] == ["WARNING", "ALARM", "ALARM", "CLEAR"]
    assert [e.frame_index for e in events] == [120, 270, 270, 300]
    assert events[1].track_ids == [7]  # WARNING_END carries the ids that were warning
    assert events[3].track_ids == [7]  # ALARM_END carries the ids that were alarming
    # Transition events describe the objects involved (they used to carry an empty extra).
    assert [[o["id"] for o in e.extra["objects"]] for e in events] == [[7], [7], [7], []]
    assert all("repeat" not in e.extra for e in events)


def test_warning_start_end_without_alarm():
    eng = make_engine(latch=0.0)
    events = collect(eng)
    gate = make_gate(False)
    eng.evaluate([make_obj(7, stuck=4.0, dwell=4.0)], None, gate, 4.0, 120)
    eng.evaluate([make_obj(7, stuck=0.0, dwell=0.0, speed=50.0)], None, gate, 5.0, 150)
    assert [e.type for e in events] == ["WARNING_START", "WARNING_END"]


def test_repeat_notify():
    eng = make_engine(latch=0.0, repeat=5.0)
    events = collect(eng)
    gate = make_gate(False)
    for i in range(13):
        t = float(i)
        eng.evaluate([make_obj(7, stuck=9.0 + t, dwell=9.0 + t)], None, gate, t, i)
    starts = [e for e in events if e.type == "ALARM_START"]
    assert len(starts) == 3
    assert "repeat" not in starts[0].extra
    assert [e.timestamp for e in starts[1:]] == [5.0, 10.0]
    assert all(e.extra == {"repeat": True} for e in starts[1:])
    assert all(e.track_ids == [7] for e in starts)
    # repeat disabled
    eng2 = make_engine(latch=0.0, repeat=0.0)
    ev2 = collect(eng2)
    for i in range(100):
        eng2.evaluate([make_obj(7, stuck=9.0)], None, gate, float(i), i)
    assert [e.type for e in ev2] == ["ALARM_START"]


def test_repeat_notify_continues_through_latch_and_resets_after_end():
    eng = make_engine(latch=6.0, repeat=4.0)
    events = collect(eng)
    gate = make_gate(False)
    eng.evaluate([make_obj(7, stuck=9.0)], None, gate, 0.0, 0)  # ALARM_START
    eng.evaluate([], None, gate, 4.0, 120)  # latched, repeat at 4
    eng.evaluate([], None, gate, 5.0, 150)  # latched
    eng.evaluate([], None, gate, 6.0, 180)  # latch expired -> ALARM_END
    eng.evaluate([make_obj(8, stuck=9.0)], None, gate, 7.0, 210)  # fresh ALARM_START (not repeat)
    types = [(e.type, bool(e.extra.get("repeat"))) for e in events]
    assert types == [("ALARM_START", False), ("ALARM_START", True), ("ALARM_END", False), ("ALARM_START", False)]


def test_reset_clears_latch_and_status_silently():
    eng = make_engine(latch=100.0)
    events = collect(eng)
    eng.evaluate([make_obj(7, stuck=9.0)], None, make_gate(False), 1.0, 30)
    assert eng.status is Status.ALARM and eng.latched_until is not None
    eng.reset()
    assert eng.status is Status.CLEAR and eng.latched_until is None
    a, ev = eng.evaluate([], None, make_gate(False), 2.0, 60)
    assert a.status is Status.CLEAR and ev == []
    assert [e.type for e in events] == ["ALARM_START"]  # no ALARM_END from reset


def test_numpy_track_ids_and_counters_are_coerced():
    eng = make_engine(latch=0.0)
    obj = make_obj(np.int64(42), stuck=np.float32(9.0), dwell=np.float64(9.0))
    a, ev = eng.evaluate([obj], None, make_gate(False), np.float64(9.0), np.int64(270))
    assert a.alarm_track_ids == [42] and type(a.alarm_track_ids[0]) is int
    assert type(ev[0].frame_index) is int and type(ev[0].timestamp) is float
    json.loads(event_to_json(ev[0]))


# --------------------------------------------------------------------------------------
# sinks
# --------------------------------------------------------------------------------------
def test_add_sink_rejects_objects_without_emit():
    eng = make_engine()
    with pytest.raises(TypeError):
        eng.add_sink(object())
    with pytest.raises(TypeError):
        CallbackSink("not callable")  # type: ignore[arg-type]


def test_faulty_sink_does_not_break_evaluate(caplog):
    eng = make_engine(latch=0.0)
    calls = []

    def boom(event):
        raise RuntimeError("sink exploded")

    eng.add_sink(CallbackSink(boom))
    eng.add_sink(CallbackSink(calls.append))
    a, ev = eng.evaluate([make_obj(7, stuck=9.0)], None, make_gate(False), 9.0, 270)
    assert a.status is Status.ALARM
    assert len(ev) == 1 and len(calls) == 1  # the healthy sink still received the event
    assert any("sink exploded" in r.getMessage() or "failed" in r.getMessage() for r in caplog.records)


def test_jsonl_sink_writes_valid_json_lines(tmp_path):
    path = tmp_path / "sub" / "events.jsonl"  # parent dir created by the sink
    eng = make_engine(latch=0.0, repeat=2.0)
    sink = JsonlSink(str(path))
    eng.add_sink(sink)
    gate = make_gate(True)
    eng.evaluate([make_obj(np.int64(7), stuck=1.5, dwell=1.5)], None, gate, 1.5, 45)  # WARNING
    eng.evaluate([make_obj(np.int64(7), stuck=3.5, dwell=3.5)], None, gate, 3.5, 105)  # ALARM
    eng.evaluate([make_obj(np.int64(7), stuck=5.5, dwell=5.5)], None, gate, 5.5, 165)  # repeat
    eng.evaluate([], make_debris(True), gate, 6.0, 180)  # still ALARM; debris is a new cause: ALARM_UPDATE
    eng.evaluate([], None, gate, 7.0, 210)  # ALARM_END
    sink.close()

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 6 == sink.count
    records = [json.loads(line) for line in lines]
    assert [r["type"] for r in records] == ["WARNING_START", "WARNING_END", "ALARM_START", "ALARM_START",
                                            "ALARM_UPDATE", "ALARM_END"]
    assert set(records[0]) == {"type", "timestamp", "frame_index", "status", "reasons", "track_ids", "extra"}
    assert records[2]["track_ids"] == [7]
    assert records[2]["status"] == "ALARM"
    assert records[2]["reasons"] == ["vehicle 7 (car) stationary in zone 3.5s (gates down)"]
    assert records[3]["extra"] == {"repeat": True}
    assert records[4]["extra"]["new_reasons"] == ["debris 4.1% of zone for 3.2s"]
    assert records[5]["status"] == "CLEAR"
    # append mode: a second sink on the same path keeps the earlier lines
    with JsonlSink(str(path)) as sink2:
        sink2.emit(Event("HEALTH", 8.0, 240, "CLEAR", ["stabilizer unhealthy"], [], {"inliers": 3}))
    assert len(path.read_text(encoding="utf-8").splitlines()) == 7
    # emitting after close is logged, not raised
    sink.emit(Event("HEALTH", 9.0, 270, "CLEAR", [], [], {}))
    assert sink.count == 6


def test_stdout_sink_prints_ascii_line(capsys):
    sink = StdoutSink()
    sink.emit(Event("ALARM_START", 12.345, 370, "ALARM", ["vehicle 7 (car) stationary in zone 8.3s", "café"], [7], {}))
    sink.emit(Event("ALARM_START", 42.0, 1260, "ALARM", ["x"], [7], {"repeat": True}))
    out = capsys.readouterr().out
    out.encode("ascii")  # must not raise
    lines = out.splitlines()
    assert len(lines) == 2
    assert "ALARM_START" in lines[0] and "12.345" in lines[0] and "f=    370" in lines[0]
    assert "ids=7" in lines[0] and "vehicle 7" in lines[0] and "caf?" in lines[0]
    assert lines[1].endswith("repeat")
    # explicit stream
    import io
    buf = io.StringIO()
    StdoutSink(stream=buf).emit(Event("WARNING_END", 1.0, 30, "CLEAR", [], [], {}))
    assert "WARNING_END" in buf.getvalue() and "ids=-" in buf.getvalue()


def test_callback_sink_and_engine_emit():
    got: List[Event] = []
    eng = make_engine()
    eng.add_sink(CallbackSink(got.append))
    assert len(eng.sinks) == 1
    health = Event("HEALTH", 3.0, 90, "CLEAR", ["zone held"], [], {"method": "held"})
    eng.emit(health)
    assert got == [health]


def test_build_sinks_from_config(tmp_path):
    cfg = AlertConfig(stdout=True, events_jsonl=str(tmp_path / "e.jsonl"), webhook_url="http://127.0.0.1:9/x")
    sinks = build_sinks(cfg)
    assert [type(s).__name__ for s in sinks] == ["StdoutSink", "JsonlSink", "WebhookSink"]
    sinks[1].close()
    assert build_sinks(AlertConfig(stdout=False)) == []


# --------------------------------------------------------------------------------------
# webhook sink (local HTTP server; no external network)
# --------------------------------------------------------------------------------------
class _Recorder(http.server.BaseHTTPRequestHandler):
    bodies: List[dict] = []
    delay_s = 0.0
    received = threading.Event()

    def do_POST(self):  # noqa: N802 (http.server naming)
        n = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(n)
        if _Recorder.delay_s:
            time.sleep(_Recorder.delay_s)
        _Recorder.bodies.append(json.loads(body.decode("utf-8")))
        self.send_response(200)
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"ok")
        _Recorder.received.set()

    def log_message(self, *args):  # keep pytest output quiet
        return


@pytest.fixture
def local_server():
    _Recorder.bodies = []
    _Recorder.delay_s = 0.0
    _Recorder.received = threading.Event()
    srv = http.server.HTTPServer(("127.0.0.1", 0), _Recorder)
    th = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    th.start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}/hook"
    finally:
        srv.shutdown()
        srv.server_close()


def test_webhook_sink_posts_json(local_server):
    sink = WebhookSink(local_server, timeout=2.0)
    ev = Event("ALARM_START", 8.3, 249, "ALARM", ["vehicle 7 (car) stationary in zone 8.3s"], [7], {})
    sink.emit(ev)
    assert sink.flush(timeout=5.0)
    assert _Recorder.received.wait(5.0)
    assert sink.delivered == 1 and sink.failures == 0
    assert _Recorder.bodies == [json.loads(event_to_json(ev))]
    assert _Recorder.bodies[0]["type"] == "ALARM_START"


def test_webhook_sink_does_not_block_frame_loop(local_server):
    _Recorder.delay_s = 0.5
    sink = WebhookSink(local_server, timeout=3.0)
    t0 = time.perf_counter()
    for i in range(3):
        sink.emit(Event("ALARM_START", float(i), i, "ALARM", [], [], {"repeat": i > 0}))
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.2, f"emit blocked for {elapsed:.3f}s"
    assert sink.flush(timeout=10.0)
    assert sink.delivered == 3
    assert [b["timestamp"] for b in _Recorder.bodies] == [0.0, 1.0, 2.0]  # order preserved


def test_webhook_sink_failure_is_logged_not_raised(caplog):
    # Grab a free port and close it so the connection is refused immediately.
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    sink = WebhookSink(f"http://127.0.0.1:{port}/hook", timeout=1.0)
    sink.emit(Event("ALARM_START", 1.0, 30, "ALARM", [], [], {}))
    assert sink.flush(timeout=5.0)
    assert sink.failures == 1 and sink.delivered == 0
    assert any("WebhookSink POST" in r.getMessage() for r in caplog.records)
    # A malformed URL also never raises.
    bad = WebhookSink("not a url", timeout=1.0)
    bad.emit(Event("ALARM_START", 1.0, 30, "ALARM", [], [], {}))
    assert bad.flush(timeout=5.0)
    assert bad.failures == 1


def test_webhook_sink_drops_when_queue_full():
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    sink = WebhookSink(f"http://127.0.0.1:{port}/hook", timeout=1.0, max_queue=1)
    for i in range(50):
        sink.emit(Event("ALARM_START", float(i), i, "ALARM", [], [], {}))
    assert sink.flush(timeout=10.0)
    assert sink.dropped + sink.failures + sink.delivered == 50
    assert sink.dropped > 0
