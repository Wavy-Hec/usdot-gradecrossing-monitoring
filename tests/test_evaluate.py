"""Tests for gradecrossing.evaluate (truth loading, alarm-interval extraction, scoring, CLI)."""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from gradecrossing.alerting import AlertEngine, JsonlSink
from gradecrossing.config import AlertConfig, DebrisConfig, StuckConfig
from gradecrossing.evaluate import GroundTruthEvent, load_alarm_intervals, load_truth_csv, main, score
from gradecrossing.types import GateState, ObjectState

ROOT = Path(__file__).resolve().parents[1]


def T(start: float, end: float, kind: str = "stuck_vehicle", video: str = "clip") -> GroundTruthEvent:
    return GroundTruthEvent(video=video, kind=kind, start_s=start, end_s=end)


# --------------------------------------------------------------------------------------
# score
# --------------------------------------------------------------------------------------
def test_score_known_tp_fp_fn_and_time_to_alert():
    truth = [T(10, 20), T(40, 50, kind="debris"), T(70, 80, kind="person")]
    alarms = [
        (12.0, 22.0),   # overlaps truth 1 -> TP, time-to-alert 2.0
        (38.5, 45.0),   # starts 1.5 s early but overlaps truth 2 -> TP, tta clipped to 0
        (100.0, 105.0),  # nothing there -> FP
    ]
    r = score(truth, alarms, duration_s=120.0, tolerance_s=2.0)
    assert (r["tp"], r["fp"], r["fn"]) == (2, 1, 1)
    assert r["precision"] == pytest.approx(2 / 3)
    assert r["recall"] == pytest.approx(2 / 3)
    assert r["f1"] == pytest.approx(2 / 3)
    assert r["time_to_alert_s"] == [2.0, 0.0]
    assert r["median_time_to_alert_s"] == pytest.approx(1.0)
    assert r["false_alarms_per_hour"] == pytest.approx(1 / (120.0 / 3600.0))  # 30/h
    assert r["duration_s"] == 120.0
    assert r["n_truth"] == 3 and r["n_predicted"] == 3
    d = r["details"]
    assert [m["kind"] for m in d["matches"]] == ["stuck_vehicle", "debris"]
    assert [m["kind"] for m in d["missed"]] == ["person"]
    assert d["false_alarms"] == [{"pred_start_s": 100.0, "pred_end_s": 105.0}]
    json.dumps(r)  # serialisable


def test_score_tolerance_early_start_without_overlap():
    truth = [T(10, 20)]
    early = [(8.5, 9.5)]  # ends before truth starts: no overlap, start 1.5 s early
    assert score(truth, early, 60.0, tolerance_s=2.0)["tp"] == 1
    r = score(truth, early, 60.0, tolerance_s=1.0)
    assert (r["tp"], r["fp"], r["fn"]) == (0, 1, 1)
    assert r["precision"] == 0.0 and r["recall"] == 0.0 and r["f1"] == 0.0
    # late start within tolerance but no overlap (alarm after the incident ended)
    late = [(21.5, 30.0)]
    assert score(truth, late, 60.0, tolerance_s=2.0)["tp"] == 0
    assert score([T(10, 20)], [(21.5, 30.0)], 60.0, tolerance_s=12.0)["tp"] == 1


def test_score_each_truth_matched_at_most_once():
    # Each incident still yields one TP. The later alarms inside the same incident (the
    # alarm latched off and fired again) used to count as false alarms, inflating FA/h for
    # long incidents (review finding F12); they are now reported as duplicates.
    truth = [T(10, 20)]
    alarms = [(11.0, 12.0), (15.0, 16.0), (18.0, 25.0)]
    r = score(truth, alarms, 60.0)
    assert (r["tp"], r["fp"], r["fn"]) == (1, 0, 0)
    assert r["duplicates"] == 2 and r["details"]["matches"][0]["n_alarms"] == 3
    assert r["time_to_alert_s"] == [1.0]
    assert r["false_alarms_per_hour"] == 0.0
    # an alarm that starts after the incident (beyond the tolerance) is still a false alarm
    r = score(truth, alarms + [(30.0, 31.0)], 60.0)
    assert (r["tp"], r["fp"], r["duplicates"]) == (1, 1, 2)
    assert r["false_alarms_per_hour"] == pytest.approx(1 / (60.0 / 3600.0))


def test_score_open_ended_intervals():
    truth = [T(10, math.inf)]  # incident runs to the end of the clip
    r = score(truth, [(30.0, math.inf)], duration_s=100.0)  # alarm never ended either
    assert r["tp"] == 1 and r["time_to_alert_s"] == [20.0]


def test_score_empty_cases():
    r = score([], [], duration_s=10.0)
    assert (r["tp"], r["fp"], r["fn"]) == (0, 0, 0)
    assert r["precision"] == 1.0 and r["recall"] == 1.0 and r["f1"] == 1.0
    assert r["median_time_to_alert_s"] is None
    assert r["false_alarms_per_hour"] == 0.0
    r = score([T(1, 2)], [], duration_s=10.0)
    assert r["recall"] == 0.0 and r["precision"] == 1.0 and r["fn"] == 1
    r = score([], [(1.0, 2.0)], duration_s=0.0)
    assert r["fp"] == 1 and r["false_alarms_per_hour"] is None


# --------------------------------------------------------------------------------------
# loaders
# --------------------------------------------------------------------------------------
def _write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(rec if isinstance(rec, str) else json.dumps(rec))
            fh.write("\n")


def _ev(etype, t, extra=None):
    return {"type": etype, "timestamp": t, "frame_index": int(t * 30), "status": "ALARM",
            "reasons": [], "track_ids": [], "extra": extra or {}}


def test_load_alarm_intervals(tmp_path):
    p = tmp_path / "events.jsonl"
    _write_jsonl(p, [
        _ev("WARNING_START", 1.0),
        _ev("WARNING_END", 3.0),
        _ev("ALARM_START", 3.0),
        _ev("ALARM_START", 8.0, {"repeat": True}),   # repeat: ignored
        _ev("ALARM_START", 9.0),                      # duplicate start: ignored
        "",                                            # blank line
        "{this is not json",                          # truncated line: skipped
        _ev("ALARM_END", 12.5),
        _ev("ALARM_END", 13.0),                       # end without start: ignored
        _ev("HEALTH", 14.0),
        _ev("ALARM_START", 20.0),                     # never ends -> +inf
    ])
    assert load_alarm_intervals(str(p)) == [(3.0, 12.5), (20.0, math.inf)]
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    assert load_alarm_intervals(str(empty)) == []


def test_load_truth_csv(tmp_path):
    p = tmp_path / "truth.csv"
    p.write_text(
        "video,kind,start_s,end_s\n"
        "b.mp4,debris,5,9.5\n"
        "\n"
        "a.mp4, stuck_vehicle , 10.0 , 20\n"
        "#comment,x,1,2\n"
        "a.mp4,person,30,\n",
        encoding="utf-8")
    rows = load_truth_csv(str(p))
    assert [(r.video, r.kind, r.start_s, r.end_s) for r in rows] == [
        ("a.mp4", "stuck_vehicle", 10.0, 20.0),
        ("a.mp4", "person", 30.0, math.inf),
        ("b.mp4", "debris", 5.0, 9.5),
    ]
    bad = tmp_path / "bad.csv"
    bad.write_text("video,start_s\na,1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_truth_csv(str(bad))
    bad.write_text("video,kind,start_s,end_s\na,k,abc,2\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_truth_csv(str(bad))
    bad.write_text("video,kind,start_s,end_s\na,k,5,2\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_truth_csv(str(bad))


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def test_main_prints_table_and_writes_json(tmp_path, capsys):
    events = tmp_path / "events.jsonl"
    truth = tmp_path / "truth.csv"
    out = tmp_path / "out" / "metrics.json"
    _write_jsonl(events, [_ev("ALARM_START", 12.0), _ev("ALARM_END", 22.0),
                          _ev("ALARM_START", 50.0), _ev("ALARM_END", 51.0)])
    truth.write_text("video,kind,start_s,end_s\nclip,stuck_vehicle,10,20\nother,debris,1,2\nclip,person,70,\n",
                     encoding="utf-8")
    rc = main(["--events", str(events), "--truth", str(truth), "--video", "clip",
               "--duration-s", "120", "--out", str(out)])
    assert rc == 0
    text = capsys.readouterr().out
    text.encode("ascii")
    assert "precision" in text and "TP 1  FP 1  FN 1" in text and "MISSED" in text
    assert "false alarms: 50.00-51.00" in text
    data = json.loads(out.read_text(encoding="utf-8"))
    assert (data["tp"], data["fp"], data["fn"]) == (1, 1, 1)
    assert data["video"] == "clip" and data["duration_s"] == 120.0
    assert data["details"]["missed"][0]["truth_end_s"] is None  # inf sanitised for strict JSON


def test_main_duration_fallback_and_kind_filter(tmp_path, capsys):
    events = tmp_path / "events.jsonl"
    truth = tmp_path / "truth.csv"
    _write_jsonl(events, [_ev("ALARM_START", 12.0), _ev("ALARM_END", 22.0), _ev("HEALTH", 33.0)])
    truth.write_text("video,kind,start_s,end_s\nclip,stuck_vehicle,10,20\nclip,debris,25,30\n", encoding="utf-8")
    rc = main(["--events", str(events), "--truth", str(truth), "--kind", "stuck_vehicle", "--quiet"])
    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "duration" in captured.err and "33.0" in captured.err


def test_main_missing_file_returns_1(tmp_path, capsys):
    truth = tmp_path / "truth.csv"
    truth.write_text("video,kind,start_s,end_s\n", encoding="utf-8")
    rc = main(["--events", str(tmp_path / "nope.jsonl"), "--truth", str(truth)])
    assert rc == 1
    assert "error" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main([])  # argparse usage error


def test_module_entry_point_exists():
    proc = subprocess.run([sys.executable, "-m", "gradecrossing.evaluate", "--help"], capture_output=True, text=True,
                          env=dict(os.environ, PYTHONPATH=str(ROOT)), cwd=str(ROOT), timeout=60)
    assert proc.returncode == 0, proc.stderr
    assert "--events" in proc.stdout and "--truth" in proc.stdout


# --------------------------------------------------------------------------------------
# end-to-end: AlertEngine -> JsonlSink -> load_alarm_intervals -> score
# --------------------------------------------------------------------------------------
def _obj(tid: int, stuck: float, t: float) -> ObjectState:
    return ObjectState(track_id=tid, cls_name="car", is_vehicle=True, is_person=False, is_train=False,
                       last_box=np.array([0, 0, 10, 10], dtype=np.float32), last_seen=t, first_seen=t - stuck,
                       in_zone=True, in_zone_since=t - stuck, dwell_seconds=stuck, speed=0.0, speed_units="px/s",
                       stationary_seconds=stuck, stuck_seconds=stuck, history=[])


def test_end_to_end_engine_log_scoring(tmp_path):
    log = tmp_path / "events.jsonl"
    eng = AlertEngine(StuckConfig(warning_seconds=3.0, alarm_seconds=8.0),  # the 8 s alarm this test expects
                      AlertConfig(latch_seconds=2.0, repeat_notify_seconds=3.0, stdout=False))
    sink = JsonlSink(str(log))
    eng.add_sink(sink)
    gate = GateState()
    fps = 30
    # Car stops in the zone at t=10 and is stuck until t=30; alarm expected at t=18 (8 s).
    for i in range(int(40 * fps)):
        t = i / fps
        objs = [_obj(1, stuck=max(0.0, t - 10.0), t=t)] if 10.0 <= t < 30.0 else []
        eng.evaluate(objs, None, gate, t, i)
    sink.close()
    intervals = load_alarm_intervals(str(log))
    assert len(intervals) == 1
    start, end = intervals[0]
    assert start == pytest.approx(18.0, abs=0.05)
    assert end == pytest.approx(32.0, abs=0.05)  # latch 2 s after the car vanished
    r = score([T(10.0, 30.0)], intervals, duration_s=40.0)
    assert (r["tp"], r["fp"], r["fn"]) == (1, 0, 0)
    assert r["time_to_alert_s"][0] == pytest.approx(8.0, abs=0.05)
