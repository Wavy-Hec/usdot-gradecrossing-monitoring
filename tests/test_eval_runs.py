"""The event-log contract of gradecrossing.evaluate: which events start or end an episode,
how RUN_START / RUN_END and timestamps going backwards split a log into runs, which run is
scored, and where the duration comes from."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from gradecrossing.alerting import AlertEngine, JsonlSink
from gradecrossing.config import AlertConfig, DebrisConfig, StuckConfig
from gradecrossing.evaluate import (
    GroundTruthEvent,
    TruthSet,
    evaluate_clip,
    load_alarm_intervals,
    main,
    read_event_log,
    select_runs,
)
from gradecrossing.types import Event, GateState, ObjectState


def rec(etype, t=None, **extra):
    r = {"type": etype, "frame_index": 0, "status": "ALARM", "reasons": [], "track_ids": [], "extra": extra}
    if t is not None:
        r["timestamp"] = t
    return r


def write(path, records):
    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write((r if isinstance(r, str) else json.dumps(r)) + "\n")
    return path


def run_start(run_id, t=0.0):
    return rec("RUN_START", t, run_id=run_id, source="clip.mp4", config_sha256="ab" * 32, package_version="0.2.0")


def run_end(run_id, duration_s, last_timestamp=None, t=None):
    last = duration_s if last_timestamp is None else last_timestamp
    return rec("RUN_END", last if t is None else t, run_id=run_id, frames=int(duration_s * 30),
               duration_s=duration_s, last_timestamp=last, final_status="CLEAR")


def test_alarm_update_and_unknown_types_never_start_an_episode(tmp_path):
    p = write(tmp_path / "e.jsonl", [
        rec("ALARM_UPDATE", 1.0, new_ids=[4]),  # an update with no open alarm: not a start
        rec("SOMETHING_NEW", 2.0),
        rec("ALARM_START", 3.0),
        rec("ALARM_UPDATE", 4.0, new_ids=[9]),  # a new cause joined the active alarm
        rec("ALARM_START", 5.0, repeat=True),
        rec("ALARM_END", 6.0),
        rec("ALARM_UPDATE", 7.0),
        rec("HEALTH", 8.0),
        rec("ALARM_END", 9.0),  # end without start: ignored
        rec("WARNING_START", 10.0),
        rec("WARNING_START", 10.5, repeat=True),
        rec("WARNING_END", 11.0),
    ])
    assert load_alarm_intervals(str(p)) == [(3.0, 6.0)]
    log = read_event_log(str(p))
    (run,) = log.runs
    assert [(w.start, w.end) for w in run.warnings] == [(10.0, 11.0)]
    assert run.counts["ALARM_UPDATE"] == 3 and run.counts["SOMETHING_NEW"] == 1 and run.counts["HEALTH"] == 1
    assert run.run_id is None and run.started_by == "start of file" and log.notes == []


def test_bad_lines_are_reported_not_fatal(tmp_path):
    p = write(tmp_path / "e.jsonl", [rec("ALARM_START", 1.0), "{truncated", "[1, 2]", rec("HEALTH"),
                                     rec("HEALTH", "x"), rec("ALARM_END", 2.0)])
    log = read_event_log(str(p))
    assert [(a.start, a.end) for a in log.runs[0].alarms] == [(1.0, 2.0)]
    assert len(log.notes) == 3  # bad JSON, missing timestamp, bad timestamp


def test_run_markers_split_the_log_and_the_last_run_is_scored(tmp_path):
    p = write(tmp_path / "e.jsonl", [
        run_start("aaa"), rec("ALARM_START", 9.0), rec("ALARM_END", 24.0), run_end("aaa", 30.0),
        run_start("bbb"), rec("ALARM_START", 9.5), rec("ALARM_END", 20.0), run_end("bbb", 30.0),
    ])
    log = read_event_log(str(p))
    assert [(r.run_id, r.started_by, r.end_info["duration_s"]) for r in log.runs] == [
        ("aaa", "RUN_START", 30.0), ("bbb", "RUN_START", 30.0)]
    assert log.runs[0].start_info["config_sha256"] == "ab" * 32
    assert load_alarm_intervals(str(p)) == [(9.5, 20.0)]  # default: the last run
    assert load_alarm_intervals(str(p), run="aaa") == [(9.0, 24.0)]
    assert load_alarm_intervals(str(p), run="all") == [(9.0, 24.0), (9.5, 20.0)]
    with pytest.raises(ValueError):
        select_runs(log.runs, "zzz")
    truth = TruthSet("clip", None, [GroundTruthEvent("clip", "vehicle", 10.0, 30.0)])
    r = evaluate_clip(str(p), truth, 2.0, "one_to_one")
    assert (r["tp"], r["fp"], r["duration_s"], r["duration_source"]) == (1, 0, 30.0, "RUN_END")
    assert r["runs"]["in_log"] == 2 and r["runs"]["scored"][0]["run_id"] == "bbb"
    assert any("2 runs" in n for n in r["notes"])
    # --run all: each run is its own pass over the clip, so both detections count
    both = evaluate_clip(str(p), truth, 2.0, "one_to_one", run="all")
    assert (both["tp"], both["fn"], both["n_truth"], both["duplicates"], both["duration_s"]) == (2, 0, 2, 0, 60.0)
    assert [(s["run_id"], s["tp"], s["duration_s"]) for s in both["runs"]["scored"]] == [
        ("aaa", 1, 30.0), ("bbb", 1, 30.0)]
    assert [m["run"] for m in both["details"]["matches"]] == [0, 1]
    assert both["time_to_alert_s"] == [0.0, 0.0] and both["recall"] == 1.0


def test_rerun_appended_without_run_markers_is_not_counted_twice(tmp_path, capsys):
    # F13: the same 30 s run appended twice to one log scored tp 1, fp 1, 120 false alarms/h.
    once = [rec("WARNING_START", 4.0), rec("WARNING_END", 9.07), rec("ALARM_START", 9.07), rec("ALARM_END", 24.3)]
    p = write(tmp_path / "e.jsonl", once + once)
    log = read_event_log(str(p))
    assert [r.started_by for r in log.runs] == ["start of file", "timestamps went backwards"]
    assert any("went back from 24.30 s to 4.00 s" in n for n in log.notes)
    assert load_alarm_intervals(str(p)) == [(9.07, 24.3)]
    truth = tmp_path / "truth.csv"
    truth.write_text("video,kind,start_s,end_s\nclip,stuck_vehicle,8,30\n", encoding="utf-8")
    out = tmp_path / "m.json"
    assert main(["--events", str(p), "--truth", str(truth), "--video", "clip", "--duration-s", "30",
                 "--out", str(out)]) == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert (data["tp"], data["fp"], data["false_alarms_per_hour"]) == (1, 0, 0.0)
    err = capsys.readouterr().err
    assert "went back" in err and "2 runs" in err
    assert main(["--events", str(p), "--truth", str(truth), "--run", "nope"]) == 1


def test_events_after_run_end_stay_in_that_run(tmp_path):
    # A late event (say a HEALTH written while shutting down) must not become a new, empty
    # "last run" that hides the real one.
    p = write(tmp_path / "e.jsonl", [
        run_start("r1"), rec("ALARM_START", 5.0), rec("ALARM_END", 7.0), run_end("r1", 8.0),
        rec("HEALTH", 8.0),
    ])
    log = read_event_log(str(p))
    assert len(log.runs) == 1 and log.runs[0].counts["HEALTH"] == 1
    assert any("HEALTH after RUN_END" in n for n in log.notes)
    assert load_alarm_intervals(str(p)) == [(5.0, 7.0)]


def test_alarm_open_at_run_end_and_merged_runs(tmp_path):
    p = write(tmp_path / "e.jsonl", [
        run_start("r1"), rec("ALARM_START", 5.0), run_end("r1", 8.0),  # still alarming at the end (F18)
        run_start("r2"), rec("ALARM_START", 2.0), rec("ALARM_END", 3.0), run_end("r2", 8.0),
    ])
    assert load_alarm_intervals(str(p), run="r1") == [(5.0, math.inf)]
    # merged with a later run, an alarm left open by an earlier run ends at that run's end
    assert load_alarm_intervals(str(p), run="all") == [(2.0, 3.0), (5.0, 8.0)]
    truth = TruthSet("clip", None, [])
    r = evaluate_clip(str(p), truth, 1.0, "any_overlap", run="r1")
    assert r["fp"] == 1 and r["fa_seconds"] == pytest.approx(3.0)  # open alarm clipped at RUN_END's 8 s
    both = evaluate_clip(str(p), truth, 1.0, "any_overlap", run="all")
    assert (both["fp"], both["duration_s"]) == (2, 16.0) and both["fa_seconds"] == pytest.approx(3.0 + 1.0)


def test_run_all_scores_each_run_on_its_own_and_adds_them_up(tmp_path, capsys):
    # The same false alarm (10-40 s of a 60 s clip without incidents) in two runs: two false
    # alarms in 120 s and 60 FA-seconds. Laid over one timeline the two alarms would count
    # twice but their FA-seconds only once (30 s).
    lines = []
    for rid in ("a", "b"):
        lines += [run_start(rid), rec("ALARM_START", 10.0), rec("ALARM_END", 40.0), run_end(rid, 60.0)]
    p = write(tmp_path / "neg.events.jsonl", lines)
    neg = TruthSet("neg", None, [])
    one = evaluate_clip(str(p), neg, 1.0, "any_overlap", run="a")
    assert (one["fp"], one["duration_s"], one["fa_seconds"]) == (1, 60.0, 30.0)
    both = evaluate_clip(str(p), neg, 1.0, "any_overlap", run="all")
    assert (both["fp"], both["duration_s"], both["fa_seconds"]) == (2, 120.0, 60.0)
    assert both["false_alarms_per_hour"] == pytest.approx(60.0) and both["n_predicted"] == 2
    assert [f["run"] for f in both["details"]["false_alarms"]] == [0, 1]
    assert both["duration_source"] == "RUN_END" and both["notes"] == []
    # a log with one run scores exactly as before, without run labels
    pos = TruthSet("neg", 60.0, [GroundTruthEvent("neg", "vehicle", 12.0, 30.0)])
    log2 = write(tmp_path / "pos.events.jsonl", [rec("ALARM_START", 13.0), rec("ALARM_END", 31.0),
                                                 rec("ALARM_START", 50.0), rec("ALARM_END", 55.0)])
    r = evaluate_clip(str(log2), pos, 1.0, "any_overlap", run="all")
    assert (r["tp"], r["fp"], r["n_truth"]) == (1, 1, 1) and "run" not in r["details"]["matches"][0]
    # an incident missed in both runs: recall counts it once per pass, the report says which run
    truth_csv = tmp_path / "t.csv"
    truth_csv.write_text("video,kind,start_s,end_s\nneg,vehicle,50,55\n", encoding="utf-8")
    assert main(["--events", str(p), "--truth", str(truth_csv), "--video", "neg", "--run", "all"]) == 0
    out = capsys.readouterr().out
    assert "TP 0  FP 2  FN 2" in out and "10.00-40.00 (run 0), 10.00-40.00 (run 1)" in out
    assert " run  kind" in out


def test_run_marker_timestamps_do_not_split_runs(tmp_path):
    # RUN_START/RUN_END timestamps are not part of the monotonic-time check
    p = write(tmp_path / "e.jsonl", [run_start("r1", t=1e9), rec("ALARM_START", 2.0), rec("ALARM_END", 3.0),
                                     run_end("r1", 4.0, t=0.0)])
    log = read_event_log(str(p))
    assert len(log.runs) == 1 and log.notes == []


def test_duration_resolution_order(tmp_path):
    truth_json = TruthSet("clip", 50.0, [])
    no_duration = TruthSet("clip", None, [GroundTruthEvent("clip", "vehicle", 10.0, 70.0)])
    marked = write(tmp_path / "marked.jsonl", [run_start("r"), rec("ALARM_START", 1.0), rec("ALARM_END", 2.0),
                                               run_end("r", 40.0)])
    plain = write(tmp_path / "plain.jsonl", [rec("ALARM_START", 1.0), rec("ALARM_END", 2.0), rec("HEALTH", 33.0)])
    r = evaluate_clip(str(marked), truth_json, 1.0, "any_overlap", duration_s=12.0)
    assert (r["duration_s"], r["duration_source"]) == (12.0, "--duration-s")
    r = evaluate_clip(str(marked), truth_json, 1.0, "any_overlap")
    assert (r["duration_s"], r["duration_source"], r["notes"]) == (40.0, "RUN_END", [])
    r = evaluate_clip(str(plain), truth_json, 1.0, "any_overlap")
    assert (r["duration_s"], r["duration_source"]) == (50.0, "truth") and "ground-truth duration" in r["notes"][0]
    r = evaluate_clip(str(plain), no_duration, 2.0, "one_to_one")
    assert (r["duration_s"], r["duration_source"]) == (70.0, "last event")  # the truth ends after the log
    r = evaluate_clip(str(plain), TruthSet("clip", None, []), 2.0, "one_to_one")
    assert r["duration_s"] == 33.0 and "--duration-s not given" in r["notes"][0]


def _obj(tid: int, stuck: float, t: float) -> ObjectState:
    return ObjectState(track_id=tid, cls_name="car", is_vehicle=True, is_person=False, is_train=False,
                       last_box=np.array([0, 0, 10, 10], dtype=np.float32), last_seen=t, first_seen=t - stuck,
                       in_zone=True, in_zone_since=t - stuck, dwell_seconds=stuck, speed=0.0, speed_units="px/s",
                       stationary_seconds=stuck, stuck_seconds=stuck, history=[])


def test_engine_log_with_run_markers_scores_the_latest_run_end_to_end(tmp_path):
    """AlertEngine -> JsonlSink (append) twice, each run framed by RUN_START/RUN_END the way the
    monitor writes them; the evaluator must score only the second run and use its duration."""
    log = tmp_path / "events.jsonl"
    fps = 10
    for run_id, stop_at in (("first", 30.0), ("second", 25.0)):
        eng = AlertEngine(StuckConfig(warning_seconds=3.0, alarm_seconds=8.0), DebrisConfig(),
                          AlertConfig(latch_seconds=2.0, repeat_notify_seconds=3.0, stdout=False))
        sink = JsonlSink(str(log))  # appends, like a re-run with the same --events path
        eng.add_sink(sink)
        sink.emit(Event("RUN_START", 0.0, 0, "CLEAR", [], [], {"run_id": run_id, "source": "clip.mp4",
                                                                "config_sha256": "0" * 64, "package_version": "x"}))
        for i in range(int(40 * fps)):
            t = i / fps
            objs = [_obj(1, stuck=max(0.0, t - 10.0), t=t)] if 10.0 <= t < stop_at else []
            eng.evaluate(objs, None, GateState(), t, i)
        sink.emit(Event("RUN_END", 39.9, 399, "CLEAR", [], [], {"run_id": run_id, "frames": 400, "duration_s": 40.0,
                                                                 "last_timestamp": 39.9, "final_status": "CLEAR"}))
        sink.close()
    truth = TruthSet("clip", None, [GroundTruthEvent("clip", "stuck_vehicle", 10.0, 25.0)])
    r = evaluate_clip(str(log), truth, 2.0, "one_to_one")
    assert r["runs"]["scored"][0]["run_id"] == "second" and r["duration_s"] == 40.0
    assert (r["tp"], r["fp"], r["fn"], r["duplicates"]) == (1, 0, 0, 0)
    assert r["details"]["matches"][0]["pred_start_s"] == pytest.approx(18.0, abs=0.11)
    assert r["details"]["matches"][0]["pred_end_s"] == pytest.approx(27.0, abs=0.11)  # 2 s latch after 25 s
    assert r["event_counts"]["ALARM_START"] >= 2  # the first ALARM_START and >= 1 repeat reminder
