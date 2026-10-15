"""gradecrossing.evaluate with the team's ground truth: the JSON format (labelled negatives,
ignore spans), the two matching modes, FA-seconds, WARNING-or-ALARM metrics and the batch
CLI over directories of clips.

The randomized test checks our ``match="any_overlap"`` against a line-by-line transcription
of the team's scorer (occupancy_monitor/crossing_monitor/evaluate.py::score_video at
upstream/sergei-crossing-experiments 2f90d19). The same comparison against the team's
unmodified file on 224 real alarm-span sets is in the impl-eval report.
"""
from __future__ import annotations

import json
import math
import random

import pytest

from gradecrossing.evaluate import (
    GroundTruthEvent,
    TruthSet,
    evaluate_batch,
    evaluate_clip,
    format_report,
    load_truth_dir,
    load_truth_json,
    main,
    score,
)

CLIP07 = {"video": "clip07", "duration": 7.04, "container_duration": 7.9,
          "events": [{"start": 2.3, "end": 5.8, "label": "vehicle", "should_alarm": True,
                      "note": "van trapped on the track ~3 s before the train hits it"}],
          "ignore": [[5.8, 7.9]]}
CLIP06 = {"video": "clip06", "duration": 110.01,
          "events": [{"start": 10, "end": 110, "should_alarm": False, "note": "gates down"},
                     {"start": 38.0, "end": 113.8, "should_alarm": False, "note": "car waiting at the gate"}],
          "note": "normal operation with an occluding car", "ignore": []}


def write(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def jsonl(path, records):
    with open(path, "w", encoding="utf-8") as fh:
        for etype, t, extra in records:
            fh.write(json.dumps({"type": etype, "timestamp": t, "frame_index": int(t * 30), "status": "ALARM",
                                 "reasons": [f"{etype} at {t}"], "track_ids": [], "extra": extra}) + "\n")
    return path


def alarms_log(path, spans, warnings=()):
    recs = []
    for a, b in spans:
        recs.append((a, 1, "ALARM_START"))
        if b is not None:
            recs.append((b, 0, "ALARM_END"))
    for a, b in warnings:
        recs.append((a, 1, "WARNING_START"))
        recs.append((b, 0, "WARNING_END"))
    return jsonl(path, [(etype, t, {}) for t, _, etype in sorted(recs)])


# --------------------------------------------------------------------------------------
# loader
# --------------------------------------------------------------------------------------
def test_load_truth_json_team_format(tmp_path):
    ts = load_truth_json(str(write(tmp_path / "clip07.json", CLIP07)))
    assert (ts.video, ts.duration_s, ts.container_duration_s, ts.ignore) == ("clip07", 7.04, 7.9, [(5.8, 7.9)])
    assert [(e.kind, e.start_s, e.end_s) for e in ts.events] == [("vehicle", 2.3, 5.8)]
    assert ts.events[0].note.startswith("van trapped") and ts.negatives == []
    ts6 = load_truth_json(str(write(tmp_path / "clip06.json", CLIP06)))
    assert ts6.events == [] and [(n.note, n.start_s, n.end_s) for n in ts6.negatives] == [
        ("gates down", 10.0, 110.0), ("car waiting at the gate", 38.0, 113.8)]
    assert ts6.negatives[0].kind == ""  # negatives keep their (missing) label, as the team tags them
    no_video = {k: v for k, v in CLIP07.items() if k != "video"}
    no_label = {"events": [{"start": 1, "end": 2}]}
    assert load_truth_json(str(write(tmp_path / "clipX.json", no_video))).video == "clipX"
    assert load_truth_json(str(write(tmp_path / "y.json", no_label))).events[0].kind == "vehicle"
    bad_files = [
        [1, 2],
        {"events": {"start": 1}},
        {"events": [{"start": 3}]},
        {"events": [{"start": 3, "end": 1}]},
        {"events": [{"start": "a", "end": 1}]},
        {"events": [{"start": True, "end": 1}]},
        {"events": [{"start": 1, "end": 2, "should_alarm": "false"}]},
        {"events": [], "ignore": [[3, 1]]},
        {"events": [], "ignore": [5]},
        {"events": [], "duration": -1},
        {},  # no events list: not a ground-truth file
        {"video": "clip07", "duration": 7.04},
        {"vertices": [[212, 130], [252, 130], [445, 300]], "source": "hand-drawn"},  # a zone file
    ]
    for i, bad in enumerate(bad_files):
        with pytest.raises(ValueError):
            load_truth_json(str(write(tmp_path / f"bad{i}.json", bad)))
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError):
        load_truth_json(str(broken))


def test_load_truth_dir(tmp_path):
    write(tmp_path / "clip07.json", CLIP07)
    write(tmp_path / "clip06.json", CLIP06)
    assert sorted(load_truth_dir(str(tmp_path))) == ["clip06", "clip07"]
    write(tmp_path / "copy.json", CLIP07)  # a second file for clip07
    with pytest.raises(ValueError):
        load_truth_dir(str(tmp_path))
    with pytest.raises(ValueError):
        load_truth_dir(str(tmp_path / "empty_dir_that_does_not_exist"))


def test_a_directory_of_zone_files_is_not_ground_truth(tmp_path, capsys):
    # --truth pointed at the zones directory used to score every clip as a clip without
    # incidents (exit 0, the missed positives gone from the report)
    zones = tmp_path / "zones"
    zones.mkdir()
    write(zones / "clip07.json", {"vertices": [[212, 130], [252, 130], [445, 300], [330, 300]],
                                  "source": "hand-drawn", "note": "danger zone in source pixels"})
    with pytest.raises(ValueError, match="zone file"):
        load_truth_dir(str(zones))
    events = alarms_log(tmp_path / "clip07.events.jsonl", [(5.2, 5.6)])
    assert main(["--events", str(events), "--truth", str(zones)]) == 1
    assert "'events' list" in capsys.readouterr().err


# --------------------------------------------------------------------------------------
# matching modes
# --------------------------------------------------------------------------------------
def test_any_overlap_reproduces_the_team_rules(tmp_path):
    ts = load_truth_json(str(write(tmp_path / "clip07.json", CLIP07)))
    # two episodes inside one incident (a flapping alarm): both are hits, neither is a false alarm
    r = score(ts.events, [(5.0, 5.5), (6.5, 7.0)], ts.duration_s, tolerance_s=1.0, ignore=ts.ignore,
              match="any_overlap")
    assert (r["tp"], r["fp"], r["fn"]) == (1, 0, 0) and r["time_to_alert_s"] == [pytest.approx(2.7)]
    assert r["details"]["matches"][0]["n_alarms"] == 2 and r["match"] == "any_overlap"
    # an unmatched alarm inside the ignore span (collision aftermath) is not a false alarm
    # (6.9 s is past the window [1.3, 6.8] and starts after the span start minus the tolerance)...
    for match in ("any_overlap", "one_to_one"):
        r1 = score(ts.events, [(5.0, 5.5), (6.9, 7.2)], ts.duration_s, tolerance_s=1.0, ignore=ts.ignore,
                   match=match)
        assert (r1["tp"], r1["fp"], r1["ignored_alarms"]) == (1, 0, 1), match
        assert r1["details"]["ignored"][0]["ignore_span"] == [5.8, 7.9]
    # ...but one that starts more than the tolerance before the span still is
    r3 = score([], [(3.0, 7.0)], 7.9, tolerance_s=1.0, ignore=[(5.8, 7.9)], match="any_overlap")
    assert r3["fp"] == 1 and r3["ignored_alarms"] == 0
    # false alarms are tagged with the labelled negative they overlap (first one wins)
    ts6 = load_truth_json(str(write(tmp_path / "clip06.json", CLIP06)))
    r6 = score(ts6.events, [(3.0, 9.0), (40.0, 70.0)], ts6.duration_s, tolerance_s=1.0,
               negatives=ts6.negatives, match="any_overlap")
    assert [f.get("negative") for f in r6["details"]["false_alarms"]] == [None, "gates down"]
    assert r6["false_alarms_per_hour"] == pytest.approx(2 / (110.01 / 3600.0))
    with pytest.raises(ValueError):
        score([], [], 1.0, match="best")


def test_one_to_one_reports_re_alarms_inside_an_incident_as_duplicates():
    # F12 case 1: both alarms lie inside the real incident -> 1 TP, 1 duplicate, no false alarm
    truth = [GroundTruthEvent("v", "vehicle", 10.0, 70.0)]
    r = score(truth, [(18.0, 35.0), (45.0, 70.0)], 3600.0)
    assert (r["tp"], r["fp"], r["fn"], r["duplicates"]) == (1, 0, 0, 1)
    assert r["precision"] == 1.0 and r["false_alarms_per_hour"] == 0.0
    assert r["details"]["duplicates"] == [{"pred_start_s": 45.0, "pred_end_s": 70.0,
                                           "truth_start_s": 10.0, "truth_end_s": 70.0}]
    assert r["details"]["matches"][0]["n_alarms"] == 2
    # a re-alarm after the incident (outside the tolerance) is still a false alarm
    r2 = score(truth, [(18.0, 35.0), (75.0, 80.0)], 3600.0)
    assert (r2["tp"], r2["fp"], r2["duplicates"]) == (1, 1, 0)
    # tolerance-only matches: start within the tolerance of the incident start, and not
    # later than the tolerance after its end (F12 case 3 stays a match: 12 <= 10.5 + 2)
    short = [GroundTruthEvent("v", "vehicle", 10.0, 10.5)]
    assert score(short, [(12.0, 15.0)], 60.0)["tp"] == 1
    assert score(short, [(12.6, 15.0)], 60.0)["tp"] == 0
    assert score(short, [(12.6, 15.0)], 60.0, tolerance_s=3.0)["tp"] == 1


def test_matches_by_an_alarm_that_was_already_on_are_flagged():
    # PORT-05 / OCC-10: an alarm from 20 to 55 s "detects" an incident at 50-60 s with TTA 0
    truth = [GroundTruthEvent("v", "vehicle", 50.0, 60.0)]
    for match in ("one_to_one", "any_overlap"):
        r = score(truth, [(20.0, 55.0)], 100.0, tolerance_s=1.0, match=match)
        assert r["tp"] == 1 and r["time_to_alert_s"] == [0.0]
        assert r["pre_existing_matches"] == 1 and r["details"]["matches"][0]["pre_existing"] is True
    early = score(truth, [(49.5, 55.0)], 100.0, tolerance_s=1.0)  # within the tolerance: a real alarm
    assert early["pre_existing_matches"] == 0 and early["details"]["matches"][0]["pre_existing"] is False


def _team_score_video(alarms, gt, tolerance=1.0):
    """Transcription of the team's score_video on alarm spans (instead of an events.jsonl)."""
    duration = gt.get("duration")
    ignore = [tuple(x) for x in gt.get("ignore", [])]
    positives = [e for e in gt.get("events", []) if e.get("should_alarm", True)]
    negatives = [e for e in gt.get("events", []) if not e.get("should_alarm", True)]

    def ov(a0, a1, b0, b1):
        return a0 <= b1 and b0 <= a1

    matched = set()
    per_event = []
    for e in positives:
        hits = [i for i, (s, en) in enumerate(alarms) if ov(s, en, e["start"] - tolerance, e["end"] + tolerance)]
        matched.update(hits)
        lat = max(0.0, min(alarms[i][0] for i in hits) - e["start"]) if hits else None
        per_event.append({"detected": bool(hits), "time_to_alarm": None if lat is None else round(lat, 2)})
    fas = []
    for i, (s, en) in enumerate(alarms):
        if i in matched:
            continue
        if any(ov(s, en, i0, i1) and s >= i0 - tolerance for i0, i1 in ignore):
            continue
        tag = next((n.get("note") or n.get("label") for n in negatives if ov(s, en, n["start"], n["end"])), None)
        fas.append({"start": s, "end": en, "negative": tag})
    hours = max(duration, 1e-6) / 3600.0
    return {"detected": sum(p["detected"] for p in per_event), "events": per_event, "false_alarms": fas,
            "false_alarm_count": len(fas), "false_alarms_per_hour": round(len(fas) / hours, 2)}


def test_any_overlap_equals_the_team_scorer_on_random_cases(tmp_path):
    rng = random.Random(7)
    for case in range(400):
        duration = rng.uniform(5.0, 120.0)
        events = []
        for _ in range(rng.randint(0, 4)):
            a = rng.uniform(0, duration)
            e = {"start": round(a, 2), "end": round(min(duration, a + rng.uniform(0, 30)), 2),
                 "should_alarm": rng.random() < 0.5}
            if rng.random() < 0.7:
                e["note"] = rng.choice(["gates down", "car waiting", "van trapped"])
            if rng.random() < 0.3:
                e["label"] = "vehicle"
            events.append(e)
        ignore = []
        for _ in range(rng.randint(0, 2)):
            a = rng.uniform(0, duration)
            ignore.append([round(a, 2), round(min(duration, a + rng.uniform(0, 10)), 2)])
        gt = {"video": f"c{case}", "duration": round(duration, 2), "events": events, "ignore": ignore}
        spans, t = [], 0.0
        for _ in range(rng.randint(0, 6)):  # non-overlapping episodes, like a status-level log
            t += rng.uniform(0.0, 20.0)
            end = t + rng.uniform(0.0, 15.0)
            spans.append((round(t, 2), round(end, 2)))
            t = end
        theirs = _team_score_video(spans, gt)
        ts = load_truth_json(str(write(tmp_path / "gt.json", gt)))
        ours = score(ts.events, spans, ts.duration_s, tolerance_s=1.0, ignore=ts.ignore,
                     negatives=ts.negatives, match="any_overlap")
        assert ours["tp"] == theirs["detected"], case
        assert ours["fp"] == theirs["false_alarm_count"], case
        assert sorted(round(x, 2) for x in ours["time_to_alert_s"]) == sorted(
            e["time_to_alarm"] for e in theirs["events"] if e["time_to_alarm"] is not None), case
        assert sorted(str(f.get("negative")) for f in ours["details"]["false_alarms"]) == sorted(
            str(f["negative"]) for f in theirs["false_alarms"]), case
        assert round(ours["false_alarms_per_hour"], 2) == theirs["false_alarms_per_hour"], case


# --------------------------------------------------------------------------------------
# FA-seconds and WARNING-or-ALARM
# --------------------------------------------------------------------------------------
def test_fa_seconds_and_warning_metrics():
    truth = [GroundTruthEvent("v", "vehicle", 10.0, 20.0)]
    alarms = [(12.0, 25.0),  # matched; 21-25 is outside the window [9, 21]
              (40.0, 50.0),  # 10 s of false alarm
              (55.0, 58.0),  # inside the ignore span 54-60
              (90.0, math.inf)]  # open at the end of a 100 s clip: 10 s
    warnings = [(5.0, 12.0),  # starts 4 s before the incident window: 5-9 counts
                (30.0, 40.0),  # outside: a warning outside the incidents
                (85.0, 90.0)]  # outside
    r = score(truth, alarms, 100.0, tolerance_s=1.0, ignore=[(54.0, 60.0)], match="any_overlap",
              warnings=warnings)
    assert (r["tp"], r["fp"], r["ignored_alarms"]) == (1, 2, 1)
    assert r["fa_seconds"] == pytest.approx(4.0 + 10.0 + 10.0)
    assert r["warn_or_alarm_fa_seconds"] == pytest.approx(4.0 + 4.0 + 10.0 + 10.0 + 15.0)
    assert r["warnings_outside_positives"] == 2 and r["n_warnings"] == 3
    assert r["detected_warning_or_alarm"] == 1
    assert r["warning_or_alarm_time_to_alert_s"] == [0.0]  # the warning started before the incident
    # without warnings the W|A keys are absent and FA-seconds still works
    r0 = score(truth, alarms, 100.0, tolerance_s=1.0, ignore=[(54.0, 60.0)], match="any_overlap")
    assert "warn_or_alarm_fa_seconds" not in r0 and r0["fa_seconds"] == r["fa_seconds"]
    # a warning-only detection
    rw = score(truth, [], 100.0, warnings=[(13.0, 19.0)])
    assert (rw["tp"], rw["detected_warning_or_alarm"], rw["warning_or_alarm_time_to_alert_s"]) == (0, 1, [3.0])


# --------------------------------------------------------------------------------------
# clips, batches and the CLI
# --------------------------------------------------------------------------------------
def _bench(tmp_path):
    gt = tmp_path / "gt"
    runs = tmp_path / "run"
    gt.mkdir()
    runs.mkdir()
    write(gt / "clip07.json", CLIP07)
    write(gt / "clip06.json", CLIP06)
    write(gt / "clip01.json", {"video": "clip01", "duration": 32.277, "events": [
        {"start": 2.0, "end": 32.7, "label": "vehicle", "should_alarm": True, "note": "SUV"}], "ignore": []})
    alarms_log(runs / "clip07.events.jsonl", [(5.2, 5.6), (7.0, None)], warnings=[(3.0, 5.2)])
    alarms_log(runs / "clip06.events.jsonl", [(8.0, 37.9), (92.5, 104.1)], warnings=[(3.0, 8.0), (58.0, 60.0)])
    alarms_log(runs / "clip01.events.jsonl", [])
    return gt, runs


def test_evaluate_clip_attaches_reasons_and_notes(tmp_path):
    gt, runs = _bench(tmp_path)
    ts = load_truth_json(str(gt / "clip06.json"))
    r = evaluate_clip(str(runs / "clip06.events.jsonl"), ts, 1.0, "any_overlap")
    assert r["video"] == "clip06" and r["duration_s"] == 110.01 and r["duration_source"] == "truth"
    assert any("no RUN_END" in n for n in r["notes"])
    fa = r["details"]["false_alarms"]
    assert [f["negative"] for f in fa] == ["gates down", "gates down"]
    assert fa[0]["reasons"] == ["ALARM_START at 8.0"]
    assert r["fa_seconds"] == pytest.approx((37.9 - 8.0) + (104.1 - 92.5))
    assert r["warnings_outside_positives"] == 2 and r["event_counts"]["ALARM_START"] == 2


def test_batch_totals(tmp_path):
    gt, runs = _bench(tmp_path)
    truths = load_truth_dir(str(gt))
    logs = {v: str(runs / f"{v}.events.jsonl") for v in ("clip01", "clip06", "clip07")}
    b = evaluate_batch(truths, logs, ["clip01", "clip06", "clip07", "clip09"], 1.0, "any_overlap")
    s = b["summary"]
    assert b["missing"] == ["clip09"] and s["clips"] == 3
    assert (s["positives"], s["detected"], s["detected_warning_or_alarm"]) == (2, 1, 1)
    assert s["recall"] == pytest.approx(0.5) and s["time_to_alert_s"] == {"clip01": [], "clip07": [pytest.approx(2.9)]}
    assert s["false_alarms"] == 2 and s["ignored_alarms"] == 1
    assert s["duration_s"] == pytest.approx(32.277 + 110.01 + 7.04)
    assert s["false_alarms_per_hour"] == pytest.approx(2 / (s["duration_s"] / 3600.0))
    assert s["false_alarms_by_negative"] == {"gates down": 2}
    assert s["per_clip_fa"] == {"clip01": 0, "clip06": 2, "clip07": 0}
    # clip07: the open alarm at 7.0 s lies in the ignore span; its FA-seconds are excluded
    assert [c["fa_seconds"] for c in b["clips"]][2] == pytest.approx(0.0)


def test_cli_batch_over_directories(tmp_path, capsys):
    gt, runs = _bench(tmp_path)
    out = tmp_path / "report.json"
    assert main(["--events", str(runs), "--truth", str(gt), "--out", str(out)]) == 0
    captured = capsys.readouterr()
    captured.out.encode("ascii")
    assert "clip06" in captured.out and "[gates down]" in captured.out and "recall 1/2" in captured.out
    assert "missed: clip01 2.00-32.70" in captured.out
    assert "no RUN_END" in captured.err
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["mode"] == "batch" and data["match"] == "any_overlap" and data["tolerance_s"] == 1.0
    assert [c["video"] for c in data["clips"]] == ["clip01", "clip06", "clip07"]
    assert data["clips"][2]["details"]["false_alarms"] == []  # strict JSON: the open alarm's inf is null
    assert data["summary"]["false_alarms"] == 2
    # a subset, and a clip whose log is missing (exit 1, still reported)
    assert main(["--events", str(runs), "--truth", str(gt), "--video", "clip07", "--quiet"]) == 0
    (runs / "clip01.events.jsonl").unlink()
    assert main(["--events", str(runs), "--truth", str(gt), "--quiet"]) == 1
    assert "no event log" in capsys.readouterr().err
    assert main(["--events", str(runs), "--truth", str(gt), "--video", "clip99"]) == 1
    with pytest.raises(SystemExit):
        main(["--events", str(runs), "--truth", str(gt), "--duration-s", "10"])  # one clip only


def test_cli_single_team_json_defaults_and_ignored_alarms(tmp_path, capsys):
    gt = write(tmp_path / "clip07.json", CLIP07)
    events = alarms_log(tmp_path / "clip07.events.jsonl", [(5.2, 5.6), (6.0, 6.2), (7.0, 7.5)])
    out = tmp_path / "m.json"
    assert main(["--events", str(events), "--truth", str(gt), "--out", str(out)]) == 0
    text = capsys.readouterr().out
    text.encode("ascii")
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["video"] == "clip07" and data["duration_s"] == 7.04 and data["tolerance_s"] == 1.0
    assert data["match"] == "any_overlap" and (data["tp"], data["fp"], data["fn"]) == (1, 0, 0)
    # 5.2-5.6 and 6.0-6.2 overlap [2.3 - 1, 5.8 + 1]; 7.0-7.5 does not, but it lies in the
    # ignore span (collision aftermath) -> not a false alarm, reported separately
    assert data["n_predicted"] == 3 and data["ignored_alarms"] == 1
    assert "alarms inside ignore spans (not counted): 1" in text
    # the team JSON in a directory, video taken from the log's file name
    (tmp_path / "gt").mkdir()
    write(tmp_path / "gt" / "clip07.json", CLIP07)
    assert main(["--events", str(events), "--truth", str(tmp_path / "gt"), "--out", str(out), "--quiet"]) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["video"] == "clip07"
    assert main(["--events", str(events), "--truth", str(tmp_path / "gt"), "--video", "clip08"]) == 1
    # one_to_one can be forced on team JSON: 6.0-6.2 starts after the incident ended (no
    # overlap, 3.7 s after its start), so it is not a match or a duplicate; like 7.0-7.5 it
    # falls in the ignore span
    assert main(["--events", str(events), "--truth", str(gt), "--match", "one_to_one", "--out", str(out),
                 "--quiet"]) == 0
    forced = json.loads(out.read_text(encoding="utf-8"))
    assert forced["match"] == "one_to_one"
    assert (forced["tp"], forced["fp"], forced["duplicates"], forced["ignored_alarms"]) == (1, 0, 0, 2)


def test_cli_csv_truth_with_an_events_directory(tmp_path, capsys):
    runs = tmp_path / "run"
    runs.mkdir()
    alarms_log(runs / "a.events.jsonl", [(12.0, 22.0)])
    alarms_log(runs / "b.events.jsonl", [(5.0, 6.0)])  # no CSV rows: a clip with no incident
    truth = tmp_path / "truth.csv"
    truth.write_text("video,kind,start_s,end_s\na,stuck_vehicle,10,20\n", encoding="utf-8")
    out = tmp_path / "r.json"
    assert main(["--events", str(runs), "--truth", str(truth), "--out", str(out), "--quiet"]) == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["match"] == "one_to_one" and data["tolerance_s"] == 2.0
    assert [(c["video"], c["tp"], c["fp"]) for c in data["clips"]] == [("a", 1, 0), ("b", 0, 1)]
    assert data["summary"]["recall"] == 1.0


def test_truthset_defaults_and_report_for_a_clip_without_incidents():
    ts = TruthSet("v", None, [])
    assert ts.negatives == [] and ts.ignore == [] and ts.container_duration_s is None
    r = score([], [(1.0, 2.0)], 10.0, warnings=[])
    assert r["recall"] == 1.0  # the documented convention of the dict...
    text = format_report(r)
    assert "recall n/a (no incidents)" in text  # ...but the report does not claim a perfect recall
    assert "FA-seconds 1.0" in text and "warnings outside incidents 0" in text
