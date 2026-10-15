"""Offline evaluation of monitor runs against hand-labelled ground truth.

Workflow
--------
1. Run the monitor on a clip with ``--events events.jsonl`` (``JsonlSink``).
2. Label the clip once, either

   * in a CSV with columns ``video,kind,start_s,end_s``, one row per incident that *should*
     have produced an ALARM (a vehicle stuck on the tracks, debris, a pedestrian), or
   * in the team's ground-truth JSON (``benchmarks/team_clips/ground_truth/<clip>.json``)::

         {"video": "clip07", "duration": 7.04,
          "events": [{"start": 2.3, "end": 5.8, "label": "vehicle", "should_alarm": true,
                      "note": "van trapped on the track"}],
          "ignore": [[5.8, 7.9]]}

     ``events`` is required (a clip without incidents has an empty list). Events with
     ``should_alarm: false`` are labelled negative situations ("gates down"): they are
     never matched, they only tag the false alarms that overlap them. ``ignore`` spans
     (collision aftermath, a camera cut) are excluded from false-alarm counting.
3. ``python -m gradecrossing.evaluate --events events.jsonl --truth truth.csv --video NAME
   [--out metrics.json]`` for one clip, or ``--events RUN_DIR --truth GT_DIR`` for a whole
   benchmark (``RUN_DIR/<video>.events.jsonl`` scored against ``GT_DIR/<video>.json``).

Matching
--------
Every ``ALARM_START .. ALARM_END`` pair of the scored run is one *predicted alarm*.

* ``one_to_one`` (default for CSV truth, tolerance 2 s). In order of start, each prediction
  takes the earliest unmatched incident that it overlaps, or whose start is within
  ``tolerance_s`` of the prediction start (a tolerance-only match also needs the prediction
  to start no later than ``tolerance_s`` after the incident ends). A prediction that only
  matches an incident that is already taken is a *duplicate* (the alarm latched off and
  fired again during the same incident): reported, but not a false alarm.
* ``any_overlap`` (default for team JSON, tolerance 1 s; the rules of the team's
  ``occupancy_monitor/crossing_monitor/evaluate.py``). Every prediction that overlaps
  ``[start - tolerance_s, end + tolerance_s]`` of an incident is a hit; the earliest gives
  the time-to-alert.

In both modes a leftover prediction that overlaps an ignore span and starts no earlier than
``tolerance_s`` before the span is ignored, and every false alarm is tagged with the note
of the first labelled negative it overlaps. Time-to-alert is ``pred_start - truth_start``
clipped at 0.

The metrics that matter for a safety monitor are recall (did we catch every incident),
time-to-alert (how fast), false alarms per hour (will operators start ignoring it) and
FA-seconds: the time spent in ALARM outside the incident windows (widened by the
tolerance) and the ignore spans. The same is reported for WARNING-or-ALARM, together with
the WARNING episodes outside the incident windows. Precision/F1 are reported for
completeness.

Event-log contract
------------------
One JSON object per line with ``type, timestamp, frame_index, status, reasons, track_ids,
extra`` (written by ``JsonlSink``):

* ``ALARM_START`` opens an ALARM episode. ``extra.repeat`` marks a reminder of the open
  episode, and a second start while one is open is ignored. ``ALARM_END`` closes it. An
  episode still open when its run ends lasts until the end of the clip.
* ``WARNING_START`` / ``WARNING_END``: the same for WARNING episodes.
* ``ALARM_UPDATE`` (a new cause joined the active alarm), ``HEALTH`` and any unknown type
  are counted but never start an episode.
* ``RUN_START`` (``extra``: ``run_id, source, config_sha256, package_version``) starts a
  run. ``RUN_END`` (``extra``: ``run_id, frames, duration_s, last_timestamp,
  final_status``) records its summary; ``duration_s`` is the seconds of video the run
  covered. Events after a RUN_END stay in that run (with a warning). The sink appends, so
  one file can hold several runs: only the last run is scored unless ``--run all`` or
  ``--run <run_id>`` is given. Timestamps that go backwards without a RUN_START (an older
  log, a re-run appended to the same file, a looped file) also start a new run, with a
  warning. RUN_START/RUN_END timestamps are not used.

When several runs are scored, each is a separate pass over the clip: it is scored on its
own against the same ground truth, and the counts, durations and FA-seconds are added up
(as for the clips of a batch; listed items carry the run's index as ``run``).

The duration of a run (for FA/h and for alarms still open at the end) is ``--duration-s``,
else its ``RUN_END.duration_s``, else the truth JSON ``duration`` (with a warning), else
the last event or truth end (with a warning).
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .alerting import ascii_safe

logger = logging.getLogger(__name__)

TRUTH_COLUMNS = ("video", "kind", "start_s", "end_s")
MATCH_MODES = ("one_to_one", "any_overlap")
EVENTS_SUFFIXES = (".events.jsonl", ".jsonl")  # <video><suffix> inside an --events directory

# Event types with a meaning for scoring; everything else is only counted.
_EPISODE_TYPES = {"ALARM_START": ("alarm", True), "ALARM_END": ("alarm", False),
                  "WARNING_START": ("warning", True), "WARNING_END": ("warning", False)}


@dataclass
class GroundTruthEvent:
    """One labelled incident. ``end_s`` may be ``inf`` when the incident runs to the clip end."""

    video: str
    kind: str
    start_s: float
    end_s: float
    note: str = ""


@dataclass
class TruthSet:
    """The ground truth of one clip (one team JSON file, or the CSV rows of one video).

    ``events`` are the incidents that should raise an ALARM. ``negatives`` are labelled
    situations that should not (``should_alarm: false``); they are never matched and only
    tag the false alarms that overlap them. ``ignore`` spans are excluded from false-alarm
    counting. ``duration_s`` is the clip length (None when unknown); the benchmark files
    keep the container's longer claim in ``container_duration_s``.
    """

    video: str
    duration_s: Optional[float]
    events: List[GroundTruthEvent]
    negatives: List[GroundTruthEvent] = field(default_factory=list)
    ignore: List[Tuple[float, float]] = field(default_factory=list)
    container_duration_s: Optional[float] = None


@dataclass
class Episode:
    """One ALARM or WARNING episode of an event log; ``end`` is inf while it is still open."""

    start: float
    end: float
    reasons: List[str] = field(default_factory=list)
    track_ids: List[int] = field(default_factory=list)


@dataclass
class RunLog:
    """The events of one monitor run inside an event log (see the event-log contract)."""

    index: int  # position in the file, 0 = first
    run_id: Optional[str]  # RUN_START extra.run_id; None when the run had no RUN_START
    started_by: str  # "RUN_START", "start of file" or "timestamps went backwards"
    first_line: int
    alarms: List[Episode] = field(default_factory=list)
    warnings: List[Episode] = field(default_factory=list)
    counts: Dict[str, int] = field(default_factory=dict)  # events per type
    last_timestamp: Optional[float] = None
    start_info: Dict[str, Any] = field(default_factory=dict)  # RUN_START extra
    end_info: Optional[Dict[str, Any]] = None  # RUN_END extra


@dataclass
class EventLog:
    path: str
    runs: List[RunLog]
    notes: List[str]  # problems found while reading (bad lines, timestamps going backwards)


# --------------------------------------------------------------------------------------
# Loading: ground truth
# --------------------------------------------------------------------------------------
def _parse_seconds(text: str, field: str, line_no: int) -> float:
    text = (text or "").strip()
    if text == "" or text.lower() in ("inf", "+inf", "end"):
        return math.inf
    try:
        return float(text)
    except ValueError as exc:
        raise ValueError(f"truth CSV line {line_no}: bad {field} value {text!r}") from exc


def load_truth_csv(path: str) -> List[GroundTruthEvent]:
    """Read ``video,kind,start_s,end_s`` rows. Blank lines and ``#`` comments are skipped.

    Raises ``ValueError`` when a required column is missing or a time does not parse, so
    a typo in the label file is caught before it silently skews the metrics.
    """
    events: List[GroundTruthEvent] = []
    with open(path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            return events
        fieldnames = [f.strip() for f in reader.fieldnames]
        missing = [c for c in TRUTH_COLUMNS if c not in fieldnames]
        if missing:
            raise ValueError(f"truth CSV {path} is missing columns {missing}; has {fieldnames}")
        for line_no, row in enumerate(reader, start=2):
            row = {(k or "").strip(): (v or "") for k, v in row.items()}
            video = row.get("video", "").strip()
            if not video or video.startswith("#"):
                continue
            start_s = _parse_seconds(row.get("start_s", ""), "start_s", line_no)
            end_s = _parse_seconds(row.get("end_s", ""), "end_s", line_no)
            if math.isinf(start_s):
                raise ValueError(f"truth CSV line {line_no}: start_s must be finite")
            if end_s < start_s:
                raise ValueError(f"truth CSV line {line_no}: end_s {end_s} < start_s {start_s}")
            events.append(GroundTruthEvent(video=video, kind=row.get("kind", "").strip(),
                                           start_s=start_s, end_s=end_s))
    events.sort(key=lambda e: (e.video, e.start_s))
    return events


def _json_number(value: Any, what: str, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{path}: {what} must be a finite number, got {value!r}")
    return float(value)


def load_truth_json(path: str) -> TruthSet:
    """Read one team ground-truth JSON file (see :class:`TruthSet` and the module docstring).

    ``should_alarm`` defaults to true, ``label`` to ``"vehicle"`` for incidents and
    ``video`` to the file name stem. Raises ``ValueError`` on a malformed file (a root that
    is not an object, no top-level ``events`` list (a clip without incidents has an empty
    one), a missing or non-numeric start/end, an end before its start, a ``should_alarm``
    that is not a boolean, a bad ignore span or duration) so a typo, or ``--truth`` pointed
    at the wrong directory (the zone files, say), cannot silently skew the metrics.
    """
    with open(path, "r", encoding="utf-8") as fh:
        try:
            data = json.load(fh)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}: not valid JSON ({exc})") from exc
    if not isinstance(data, dict) or not isinstance(data.get("events"), list):
        hint = " (this looks like a zone file)" if isinstance(data, dict) and "vertices" in data else ""
        raise ValueError(f"{path}: expected a ground-truth JSON object with an 'events' list{hint}")
    video = str(data.get("video") or Path(path).stem)
    events: List[GroundTruthEvent] = []
    negatives: List[GroundTruthEvent] = []
    for i, e in enumerate(data["events"]):
        if not isinstance(e, dict) or "start" not in e or "end" not in e:
            raise ValueError(f"{path}: event {i} needs 'start' and 'end'")
        start = _json_number(e["start"], f"event {i} start", path)
        end = _json_number(e["end"], f"event {i} end", path)
        if end < start:
            raise ValueError(f"{path}: event {i} ends ({end}) before it starts ({start})")
        should_alarm = e.get("should_alarm", True)
        if not isinstance(should_alarm, bool):
            raise ValueError(f"{path}: event {i} should_alarm must be true or false, got {should_alarm!r}")
        label = str(e.get("label") or "")
        note = str(e.get("note") or "")
        if should_alarm:
            events.append(GroundTruthEvent(video, label or "vehicle", start, end, note))
        else:  # the team tags a false alarm with note or label; keep both as written
            negatives.append(GroundTruthEvent(video, label, start, end, note))
    ignore: List[Tuple[float, float]] = []
    for span in data.get("ignore") or []:
        if not isinstance(span, (list, tuple)) or len(span) != 2:
            raise ValueError(f"{path}: ignore span {span!r} must be [start, end]")
        a = _json_number(span[0], "ignore start", path)
        b = _json_number(span[1], "ignore end", path)
        if b < a:
            raise ValueError(f"{path}: ignore span {span} ends before it starts")
        ignore.append((a, b))
    durations = []
    for key in ("duration", "container_duration"):
        value = data.get(key)
        if value is not None:
            value = _json_number(value, key, path)
            if value <= 0:
                raise ValueError(f"{path}: {key} must be positive, got {value}")
        durations.append(value)
    return TruthSet(video=video, duration_s=durations[0], events=sorted(events, key=lambda ev: ev.start_s),
                    negatives=negatives, ignore=ignore, container_duration_s=durations[1])


def load_truth_dir(path: str) -> Dict[str, TruthSet]:
    """Load every ``*.json`` of a directory, keyed by video. Duplicate videos raise ``ValueError``."""
    truths: Dict[str, TruthSet] = {}
    files = sorted(Path(path).glob("*.json"))
    if not files:
        raise ValueError(f"no ground-truth *.json files in {path}")
    for p in files:
        ts = load_truth_json(str(p))
        if ts.video in truths:
            raise ValueError(f"{p}: video {ts.video!r} is already defined by another file in {path}")
        truths[ts.video] = ts
    return truths


def truth_sets_from_csv(rows: Iterable[GroundTruthEvent]) -> Dict[str, TruthSet]:
    """Group CSV rows into one :class:`TruthSet` per video (no duration, negatives or ignore)."""
    truths: Dict[str, TruthSet] = {}
    for row in rows:
        truths.setdefault(row.video, TruthSet(row.video, None, [])).events.append(row)
    return truths


# --------------------------------------------------------------------------------------
# Loading: event logs
# --------------------------------------------------------------------------------------
def _finite(value: Any) -> Optional[float]:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def read_event_log(path: str) -> EventLog:
    """Split a JSONL event log into runs and pair its ALARM / WARNING episodes.

    See the event-log contract in the module docstring. Malformed lines (for example a
    truncated last line after a crash) are skipped and reported in ``notes``.
    """
    runs: List[RunLog] = []
    notes: List[str] = []
    cur: Optional[RunLog] = None
    ended = False  # the current run has seen its RUN_END (later events stay in it, with a note)
    open_ep: Dict[str, Optional[Episode]] = {"alarm": None, "warning": None}

    def finish() -> None:
        if cur is None:
            return
        for kind in ("alarm", "warning"):
            ep = open_ep[kind]
            if ep is not None:
                (cur.alarms if kind == "alarm" else cur.warnings).append(ep)
            open_ep[kind] = None
        cur.alarms.sort(key=lambda e: e.start)
        cur.warnings.sort(key=lambda e: e.start)

    def start_run(run_id: Optional[str], started_by: str, line_no: int) -> RunLog:
        finish()
        run = RunLog(index=len(runs), run_id=run_id, started_by=started_by, first_line=line_no)
        runs.append(run)
        return run

    with open(path, "r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                notes.append(f"{path} line {line_no}: not valid JSON, skipped")
                continue
            if not isinstance(rec, dict):
                continue
            etype = str(rec.get("type", ""))
            extra = rec.get("extra")
            extra = extra if isinstance(extra, dict) else {}
            if etype == "RUN_START":
                run_id = extra.get("run_id")
                cur = start_run(None if run_id is None else str(run_id), "RUN_START", line_no)
                cur.start_info = dict(extra)
                cur.counts[etype] = 1
                ended = False
                continue
            if etype == "RUN_END":
                if cur is None:
                    cur = start_run(None, "start of file", line_no)
                cur.end_info = dict(extra)
                cur.counts[etype] = cur.counts.get(etype, 0) + 1
                ended = True
                continue
            t = _finite(rec.get("timestamp"))
            if t is None:
                notes.append(f"{path} line {line_no}: missing or bad timestamp, skipped")
                continue
            if cur is None:
                cur = start_run(None, "start of file", line_no)
            elif cur.last_timestamp is not None and t < cur.last_timestamp - 1e-6:
                notes.append(f"{path} line {line_no}: timestamp went back from {cur.last_timestamp:.2f} s "
                             f"to {t:.2f} s without RUN_START; treated as a new run")
                cur = start_run(cur.run_id, "timestamps went backwards", line_no)
                ended = False
            elif ended:
                notes.append(f"{path} line {line_no}: {etype or 'event'} after RUN_END; kept in that run")
                ended = False  # one note per run
            cur.counts[etype] = cur.counts.get(etype, 0) + 1
            cur.last_timestamp = t if cur.last_timestamp is None else max(cur.last_timestamp, t)
            kind_open = _EPISODE_TYPES.get(etype)
            if kind_open is None:
                continue  # ALARM_UPDATE, HEALTH and unknown types never start an episode
            kind, is_start = kind_open
            if is_start:
                if extra.get("repeat") or open_ep[kind] is not None:
                    continue  # reminder, or a duplicate start of the open episode
                reasons = rec.get("reasons") if isinstance(rec.get("reasons"), list) else []
                ids = rec.get("track_ids") if isinstance(rec.get("track_ids"), list) else []
                open_ep[kind] = Episode(t, math.inf, [str(r) for r in reasons], list(ids))
            elif open_ep[kind] is not None:
                ep = open_ep[kind]
                ep.end = max(t, ep.start)
                (cur.alarms if kind == "alarm" else cur.warnings).append(ep)
                open_ep[kind] = None
    finish()
    return EventLog(path=str(path), runs=runs, notes=notes)


def select_runs(runs: Sequence[RunLog], run: str = "last") -> List[RunLog]:
    """Pick the runs to score: ``"last"`` (default), ``"all"`` or a RUN_START ``run_id``.

    A run id that the log does not contain raises ``ValueError``.
    """
    if run == "last":
        return list(runs[-1:])
    if run == "all":
        return list(runs)
    chosen = [r for r in runs if r.run_id == run]
    if not chosen:
        known = [r.run_id for r in runs if r.run_id is not None]
        raise ValueError(f"run id {run!r} is not in the log (run ids: {known or 'none'})")
    return chosen


def _merge_runs(runs: Sequence[RunLog]) -> Tuple[List[Episode], List[Episode]]:
    """Episodes of the selected runs on one timeline. An episode left open by a run that is
    not the last one ends at that run's last timestamp instead of running on."""
    alarms: List[Episode] = []
    warnings: List[Episode] = []
    for i, r in enumerate(runs):
        stop = math.inf
        if i < len(runs) - 1:
            stop = _finite((r.end_info or {}).get("last_timestamp")) or r.last_timestamp or 0.0
        for src, dst in ((r.alarms, alarms), (r.warnings, warnings)):
            for ep in src:
                end = ep.end if math.isfinite(ep.end) else max(stop, ep.start)
                dst.append(Episode(ep.start, end, ep.reasons, ep.track_ids))
    alarms.sort(key=lambda e: e.start)
    warnings.sort(key=lambda e: e.start)
    return alarms, warnings


def load_alarm_intervals(events_jsonl_path: str, run: str = "last") -> List[Tuple[float, float]]:
    """Return ``(start, end)`` pairs for every ALARM episode of the scored run(s).

    An episode opens on an ``ALARM_START`` (repeat notifications, marked with
    ``extra.repeat``, and duplicate starts are ignored) and closes on the next
    ``ALARM_END``. An episode still open at the end of the run gets ``end = +inf``.
    ``run`` selects the run as in :func:`select_runs` (default: the last one, so a clip
    re-run into the same appended log is not counted twice). With several runs the
    episodes share one timeline (each run's own video time); to score several runs use
    :func:`evaluate_clip`, which scores each run on its own. Malformed lines are skipped
    with a warning instead of aborting the evaluation.
    """
    log = read_event_log(events_jsonl_path)
    for note in log.notes:
        logger.warning("%s", note)
    alarms, _ = _merge_runs(select_runs(log.runs, run))
    return [(ep.start, ep.end) for ep in alarms]


# --------------------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------------------
def _overlaps(a: Tuple[float, float], b: Tuple[float, float]) -> bool:
    return a[0] <= b[1] and b[0] <= a[1]


def _one_to_one_hit(ps: float, pe: float, tr: GroundTruthEvent, tolerance_s: float) -> bool:
    """Overlap, or a start within the tolerance of the incident start (and not later than
    the tolerance after its end)."""
    if _overlaps((ps, pe), (tr.start_s, tr.end_s)):
        return True
    return abs(ps - tr.start_s) <= tolerance_s and ps <= tr.end_s + tolerance_s


def _union_length(spans: Iterable[Tuple[float, float]], lo: float, hi: float) -> float:
    clipped = sorted((max(lo, a), min(hi, b)) for a, b in spans if b > lo and a < hi)
    total = 0.0
    cur_a: Optional[float] = None
    cur_b = 0.0
    for a, b in clipped:
        if b <= a:
            continue
        if cur_a is None or a > cur_b:
            if cur_a is not None:
                total += cur_b - cur_a
            cur_a, cur_b = a, b
        else:
            cur_b = max(cur_b, b)
    if cur_a is not None:
        total += cur_b - cur_a
    return total


def _seconds_outside(spans: Sequence[Tuple[float, float]], excluded: Sequence[Tuple[float, float]],
                     lo: float, hi: float) -> float:
    """Seconds of ``[lo, hi]`` covered by ``spans`` but by none of the ``excluded`` windows."""
    inside = [(max(a, x0), min(b, x1)) for a, b in spans for x0, x1 in excluded if min(b, x1) > max(a, x0)]
    return max(0.0, _union_length(spans, lo, hi) - _union_length(inside, lo, hi))


def _rates(tp: int, fp: int, fn: int, duration_s: float) -> Dict[str, Optional[float]]:
    """Precision, recall, F1 and false alarms per hour (conventions in :func:`score`)."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    if fp == 0:
        fa_per_hour: Optional[float] = 0.0
    elif duration_s > 0:
        fa_per_hour = fp / (duration_s / 3600.0)
    else:
        fa_per_hour = None  # unknown duration: cannot normalise
    return {"precision": precision, "recall": recall, "f1": f1, "false_alarms_per_hour": fa_per_hour}


def score(truth: List[GroundTruthEvent], alarms: List[Tuple[float, float]], duration_s: float,
          tolerance_s: float = 2.0, ignore: Sequence[Tuple[float, float]] = (),
          negatives: Sequence[GroundTruthEvent] = (), match: str = "one_to_one",
          warnings: Optional[Sequence[Tuple[float, float]]] = None) -> Dict[str, Any]:
    """Match predicted alarm intervals to truth incidents and compute the metrics.

    ``match`` is ``"one_to_one"`` or ``"any_overlap"`` (rules in the module docstring).
    Precision/recall are defined as 1.0 when their denominator is zero (nothing predicted /
    nothing to find is not an error); F1 is 0.0 when both are 0. Duplicates and ignored
    alarms are neither true nor false positives.

    ``ignore`` spans and ``negatives`` come from a team ground-truth JSON. ``warnings``
    (WARNING episode intervals) adds the WARNING-or-ALARM metrics: an incident counts as
    alerted when any WARNING or ALARM overlaps ``[start - tolerance_s, end + tolerance_s]``.

    A match whose alarm was already on more than ``tolerance_s`` before the incident began
    is flagged ``pre_existing`` (counted in ``pre_existing_matches``): it is credited, with
    a time-to-alert of 0, but says nothing about how fast this incident was detected.

    Returns a JSON-serialisable dict with ``tp, fp, fn, precision, recall, f1,
    time_to_alert_s, median_time_to_alert_s, false_alarms_per_hour, duration_s,
    duplicates, ignored_alarms, pre_existing_matches, fa_seconds`` (and with ``warnings``:
    ``detected_warning_or_alarm, warning_or_alarm_time_to_alert_s,
    warn_or_alarm_fa_seconds, warnings_outside_positives``), plus a ``details`` block
    listing every match, miss, duplicate, ignored and false alarm for the report.
    """
    if match not in MATCH_MODES:
        raise ValueError(f"match must be one of {MATCH_MODES}, got {match!r}")
    tol = float(tolerance_s)
    truth_sorted = sorted(truth, key=lambda e: e.start_s)
    preds = sorted(((float(s), float(e)) for s, e in alarms), key=lambda iv: iv[0])
    match_of: Dict[int, Dict[str, Any]] = {}  # truth index -> match record
    duplicates: List[Dict[str, Any]] = []
    unmatched: List[Tuple[float, float]] = []

    def record(i: int, ps: float, pe: float, n_alarms: int) -> None:
        tr = truth_sorted[i]
        match_of[i] = {"kind": tr.kind, "truth_start_s": tr.start_s, "truth_end_s": tr.end_s,
                       "pred_start_s": ps, "pred_end_s": pe,
                       "time_to_alert_s": max(0.0, ps - tr.start_s), "n_alarms": n_alarms,
                       "pre_existing": ps < tr.start_s - tol}

    if match == "one_to_one":
        for ps, pe in preds:
            hits = [i for i, tr in enumerate(truth_sorted) if _one_to_one_hit(ps, pe, tr, tol)]
            free = [i for i in hits if i not in match_of]
            if free:
                record(free[0], ps, pe, 1)
            elif hits:
                taken = match_of[hits[0]]
                taken["n_alarms"] += 1
                duplicates.append({"pred_start_s": ps, "pred_end_s": pe,
                                   "truth_start_s": taken["truth_start_s"], "truth_end_s": taken["truth_end_s"]})
            else:
                unmatched.append((ps, pe))
    else:
        hit = set()
        for i, tr in enumerate(truth_sorted):
            window = (tr.start_s - tol, tr.end_s + tol)
            hits = [k for k, iv in enumerate(preds) if _overlaps(iv, window)]
            if hits:
                hit.update(hits)
                first = min(hits, key=lambda k: preds[k][0])
                record(i, preds[first][0], preds[first][1], len(hits))
        unmatched = [iv for k, iv in enumerate(preds) if k not in hit]

    false_alarms: List[Dict[str, Any]] = []
    ignored: List[Dict[str, Any]] = []
    for ps, pe in unmatched:
        span = next(((float(a), float(b)) for a, b in ignore
                     if _overlaps((ps, pe), (float(a), float(b))) and ps >= float(a) - tol), None)
        if span is not None:
            ignored.append({"pred_start_s": ps, "pred_end_s": pe, "ignore_span": list(span)})
            continue
        tag = next((n.note or n.kind or None for n in negatives if _overlaps((ps, pe), (n.start_s, n.end_s))), None)
        fa: Dict[str, Any] = {"pred_start_s": ps, "pred_end_s": pe}
        if tag is not None:
            fa["negative"] = tag
        false_alarms.append(fa)

    matches = [match_of[i] for i in sorted(match_of)]
    missed = [{"kind": tr.kind, "truth_start_s": tr.start_s, "truth_end_s": tr.end_s}
              for i, tr in enumerate(truth_sorted) if i not in match_of]
    matches.sort(key=lambda m: m["truth_start_s"])

    tp = len(matches)
    fp = len(false_alarms)
    fn = len(missed)
    tta = [m["time_to_alert_s"] for m in matches]
    median_tta = float(statistics.median(tta)) if tta else None
    duration_s = float(duration_s)
    rates = _rates(tp, fp, fn, duration_s)

    # Time in ALARM outside every incident window (widened by the tolerance) and ignore span.
    excluded = [(tr.start_s - tol, tr.end_s + tol) for tr in truth_sorted] + [(float(a), float(b)) for a, b in ignore]
    finite = [x for iv in list(preds) + list(warnings or []) for x in iv if math.isfinite(x)]
    hi = duration_s if duration_s > 0 else max(finite, default=0.0)

    result: Dict[str, Any] = {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": rates["precision"],
        "recall": rates["recall"],
        "f1": rates["f1"],
        "time_to_alert_s": tta,
        "median_time_to_alert_s": median_tta,
        "false_alarms_per_hour": rates["false_alarms_per_hour"],
        "duration_s": duration_s,
        "tolerance_s": tol,
        "match": match,
        "n_truth": len(truth_sorted),
        "n_predicted": len(preds),
        "duplicates": len(duplicates),
        "ignored_alarms": len(ignored),
        "pre_existing_matches": sum(m["pre_existing"] for m in matches),
        "fa_seconds": _seconds_outside(preds, excluded, 0.0, hi),
    }
    details: Dict[str, Any] = {"matches": matches, "missed": missed, "false_alarms": false_alarms,
                               "duplicates": duplicates, "ignored": ignored}
    if warnings is not None:
        warns = sorted(((float(s), float(e)) for s, e in warnings), key=lambda iv: iv[0])
        alerted = []
        for tr in truth_sorted:
            window = (tr.start_s - tol, tr.end_s + tol)
            starts = [s for s, e in list(preds) + warns if _overlaps((s, e), window)]
            alerted.append({"kind": tr.kind, "truth_start_s": tr.start_s, "truth_end_s": tr.end_s,
                            "time_to_alert_s": max(0.0, min(starts) - tr.start_s) if starts else None})
        outside = [{"start_s": s, "end_s": e} for s, e in warns
                   if not any(_overlaps((s, e), x) for x in excluded)]
        result.update({
            "detected_warning_or_alarm": sum(a["time_to_alert_s"] is not None for a in alerted),
            "warning_or_alarm_time_to_alert_s": [a["time_to_alert_s"] for a in alerted
                                                 if a["time_to_alert_s"] is not None],
            "warn_or_alarm_fa_seconds": _seconds_outside(preds + warns, excluded, 0.0, hi),
            "n_warnings": len(warns),
            "warnings_outside_positives": len(outside),
        })
        details["warning_or_alarm"] = alerted
        details["warnings_outside_positives"] = outside
    result["details"] = details
    return result


# --------------------------------------------------------------------------------------
# Clips and batches
# --------------------------------------------------------------------------------------
def default_match_and_tolerance(team_json: bool) -> Tuple[str, float]:
    """``("any_overlap", 1.0)`` for team JSON truth (its own rules), else ``("one_to_one", 2.0)``."""
    return ("any_overlap", 1.0) if team_json else ("one_to_one", 2.0)


def _resolve_duration(explicit: Optional[float], runs: Sequence[RunLog], truth: TruthSet,
                      log_name: str) -> Tuple[float, str, Optional[str]]:
    """(duration, where it came from, warning or None); order in the module docstring."""
    if explicit is not None:
        return float(explicit), "--duration-s", None
    ends = [_finite((r.end_info or {}).get("duration_s")) for r in runs]
    if runs and all(d is not None and d > 0 for d in ends):
        return float(sum(ends)), "RUN_END", None  # type: ignore[arg-type]
    if truth.duration_s:
        return float(truth.duration_s), "truth", (
            f"{log_name}: no RUN_END duration in the log; using the ground-truth duration "
            f"{truth.duration_s:.2f} s")
    last = max([r.last_timestamp or 0.0 for r in runs]
               + [_finite((r.end_info or {}).get("last_timestamp")) or 0.0 for r in runs], default=0.0)
    labelled = [x for e in list(truth.events) + list(truth.negatives) for x in (e.start_s, e.end_s)
                if math.isfinite(x)]
    duration = max([last] + labelled)
    return duration, "last event", (
        f"{log_name}: --duration-s not given and the log has no RUN_END; using {duration:.1f} s "
        f"(last event or truth end)")


_SUMMED_KEYS = ("tp", "fp", "fn", "n_truth", "n_predicted", "duplicates", "ignored_alarms",
                "pre_existing_matches", "duration_s", "fa_seconds", "detected_warning_or_alarm",
                "warn_or_alarm_fa_seconds", "n_warnings", "warnings_outside_positives")
_LIST_KEYS = ("time_to_alert_s", "warning_or_alarm_time_to_alert_s")


def _combine_runs(results: Sequence[Dict[str, Any]], labels: Sequence[Any]) -> Dict[str, Any]:
    """Add up the :func:`score` results of several runs of one clip (each a separate pass
    over the same ground truth, like the clips of a batch). Listed items get ``run``."""
    out = dict(results[0])
    for key in _SUMMED_KEYS:
        if key in out:
            out[key] = sum(r[key] for r in results)
    for key in _LIST_KEYS:
        if key in out:
            out[key] = [x for r in results for x in r[key]]
    out.update(_rates(out["tp"], out["fp"], out["fn"], out["duration_s"]))
    tta = out["time_to_alert_s"]
    out["median_time_to_alert_s"] = float(statistics.median(tta)) if tta else None
    details: Dict[str, List[Any]] = {}
    for r, label in zip(results, labels):
        for key, items in r["details"].items():
            details.setdefault(key, []).extend(dict(item, run=label) for item in items)
    out["details"] = details
    return out


def evaluate_clip(events_path: str, truth: TruthSet, tolerance_s: float, match: str, run: str = "last",
                  duration_s: Optional[float] = None, kind: Optional[str] = None) -> Dict[str, Any]:
    """Score one clip's event log against its :class:`TruthSet` (see the module docstring).

    Returns :func:`score`'s dict plus ``video, events, duration_source, runs`` (what was
    scored, with each run's own counts), ``event_counts`` and ``notes`` (warnings worth
    printing). False alarms also carry the ``reasons`` and ``track_ids`` of their
    ALARM_START. Several selected runs (``run="all"``, or a run id that a timestamp jump
    split) are scored one by one and added up; ``duration_s`` then applies to each run.
    """
    log = read_event_log(events_path)
    name = Path(events_path).name
    notes = list(log.notes)
    selected = select_runs(log.runs, run)
    if run == "last" and len(log.runs) > 1:
        notes.append(f"{name}: {len(log.runs)} runs in the log; scoring only the last "
                     f"(--run all to score each of them)")
    positives = [e for e in truth.events if kind is None or e.kind == kind]
    results: List[Dict[str, Any]] = []
    sources: List[str] = []
    scored: List[Dict[str, Any]] = []
    for r in list(selected) or [None]:  # an empty log is one run without events
        if r is None:
            alarms: List[Episode] = []
            warnings: List[Episode] = []
            label = name
        else:
            alarms, warnings = r.alarms, r.warnings
            label = name if len(selected) == 1 else f"{name} run {r.index}"
        duration, source, note = _resolve_duration(duration_s, [r] if r is not None else [], truth, label)
        if note:
            notes.append(note)
        res = score(positives, [(a.start, a.end) for a in alarms], duration, tolerance_s=tolerance_s,
                    ignore=truth.ignore, negatives=truth.negatives, match=match,
                    warnings=[(w.start, w.end) for w in warnings])
        by_span = {(a.start, a.end): a for a in alarms}
        for fa in res["details"]["false_alarms"]:
            ep = by_span.get((fa["pred_start_s"], fa["pred_end_s"]))
            if ep is not None:
                fa["reasons"] = ep.reasons[:3]
                fa["track_ids"] = ep.track_ids
        results.append(res)
        if source not in sources:
            sources.append(source)
        if r is not None:
            scored.append({"index": r.index, "run_id": r.run_id, "started_by": r.started_by,
                           "first_line": r.first_line, "run_end": r.end_info, "duration_s": res["duration_s"],
                           "duration_source": source, "tp": res["tp"], "fp": res["fp"], "fn": res["fn"],
                           "fa_seconds": res["fa_seconds"]})
    if len(results) == 1:
        result = results[0]
    else:
        result = _combine_runs(results, [r.index for r in selected])
    counts: Dict[str, int] = {}
    for r in selected:
        for etype, n in r.counts.items():
            counts[etype] = counts.get(etype, 0) + n
    result.update({
        "video": truth.video,
        "events": str(events_path),
        "duration_source": ",".join(sources),
        "runs": {"selected": run, "in_log": len(log.runs), "scored": scored},
        "event_counts": counts,
        "notes": notes,
    })
    return result


def summarize(clips: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Totals over per-clip :func:`evaluate_clip` results (recall is None without positives)."""
    duration = sum(c["duration_s"] for c in clips)
    positives = sum(c["n_truth"] for c in clips)
    detected = sum(c["tp"] for c in clips)
    tta = [t for c in clips for t in c["time_to_alert_s"]]
    fas = sum(c["fp"] for c in clips)
    by_negative: Dict[str, int] = {}
    for c in clips:
        for fa in c["details"]["false_alarms"]:
            key = str(fa.get("negative") or "(untagged)")
            by_negative[key] = by_negative.get(key, 0) + 1
    return {
        "clips": len(clips),
        "duration_s": duration,
        "positives": positives,
        "detected": detected,
        "recall": (detected / positives) if positives else None,
        "detected_warning_or_alarm": sum(c.get("detected_warning_or_alarm", 0) for c in clips),
        "time_to_alert_s": {c["video"]: c["time_to_alert_s"] for c in clips if c["n_truth"]},
        "mean_time_to_alert_s": (sum(tta) / len(tta)) if tta else None,
        "max_time_to_alert_s": max(tta) if tta else None,
        "false_alarms": fas,
        "false_alarms_per_hour": (fas / (duration / 3600.0)) if duration > 0 else None,
        "fa_seconds": sum(c["fa_seconds"] for c in clips),
        "warn_or_alarm_fa_seconds": sum(c.get("warn_or_alarm_fa_seconds", 0.0) for c in clips),
        "warnings_outside_positives": sum(c.get("warnings_outside_positives", 0) for c in clips),
        "duplicates": sum(c["duplicates"] for c in clips),
        "ignored_alarms": sum(c["ignored_alarms"] for c in clips),
        "pre_existing_matches": sum(c["pre_existing_matches"] for c in clips),
        "per_clip_fa": {c["video"]: c["fp"] for c in clips},
        "false_alarms_by_negative": by_negative,
    }


def find_event_logs(directory: str) -> Dict[str, str]:
    """Map video -> log path for ``<video>.events.jsonl`` (preferred) or ``<video>.jsonl``."""
    logs: Dict[str, str] = {}
    for suffix in reversed(EVENTS_SUFFIXES):  # the preferred suffix is written last and wins
        for p in sorted(Path(directory).glob("*" + suffix)):
            name = p.name[: -len(suffix)]
            if suffix == ".jsonl" and name.endswith(".events"):
                continue
            logs[name] = str(p)
    return logs


def evaluate_batch(truths: Mapping[str, TruthSet], event_logs: Mapping[str, str], videos: Sequence[str],
                   tolerance_s: float, match: str, run: str = "last",
                   kind: Optional[str] = None) -> Dict[str, Any]:
    """Score ``videos`` (each against ``truths[v]``, or no incidents when absent) and total them.

    A video without an event log is listed in ``missing`` and left out of the totals.
    """
    clips = []
    missing = []
    for video in videos:
        path = event_logs.get(video)
        if path is None:
            missing.append(video)
            continue
        truth = truths.get(video) or TruthSet(video, None, [])
        clips.append(evaluate_clip(path, truth, tolerance_s, match, run=run, kind=kind))
    return {"mode": "batch", "match": match, "tolerance_s": float(tolerance_s), "run": run,
            "clips": clips, "missing": missing, "summary": summarize(clips)}


# --------------------------------------------------------------------------------------
# Reporting / CLI
# --------------------------------------------------------------------------------------
def _fmt_t(value: float) -> str:
    return "end" if math.isinf(value) else f"{value:.2f}"


def _fmt_opt(value: Optional[float], spec: str = ".1f") -> str:
    return "-" if value is None else format(value, spec)


def _run_note(item: Mapping[str, Any]) -> str:
    """`` (run N)`` for an item of a several-run result, else ``""``."""
    return f" (run {item['run']})" if item.get("run") is not None else ""


def format_report(result: Dict[str, Any], title: str = "") -> str:
    """Render ``score``/``evaluate_clip`` output as an ASCII table (safe for cp1252 consoles)."""
    lines: List[str] = []
    if title:
        lines.append(ascii_safe(title))
    fa = result.get("false_alarms_per_hour")
    med = result.get("median_time_to_alert_s")
    lines.append(f"duration {result['duration_s']:.1f} s, tolerance {result['tolerance_s']:.1f} s"
                 + (f", {result['match']} matching" if result.get("match") else ""))
    lines.append(f"truth incidents: {result['n_truth']}   predicted alarms: {result['n_predicted']}")
    recall = f"{result['recall']:.3f}" if result["n_truth"] else "n/a (no incidents)"
    lines.append(f"TP {result['tp']}  FP {result['fp']}  FN {result['fn']}   "
                 f"precision {result['precision']:.3f}  recall {recall}  f1 {result['f1']:.3f}")
    lines.append("median time-to-alert " + ("n/a" if med is None else f"{med:.2f} s")
                 + "   false alarms/hour " + ("n/a" if fa is None else f"{fa:.1f}"))
    if "fa_seconds" in result:
        line = f"FA-seconds {result['fa_seconds']:.1f} (time in ALARM outside incidents and ignore spans)"
        if "warn_or_alarm_fa_seconds" in result:
            line += (f"   WARNING-or-ALARM FA-seconds {result['warn_or_alarm_fa_seconds']:.1f}"
                     f"   warnings outside incidents {result['warnings_outside_positives']}")
        lines.append(line)
    details = result.get("details", {})
    rows = []
    for m in details.get("matches", []):
        pred = f"{_fmt_t(m['pred_start_s'])}-{_fmt_t(m['pred_end_s'])}"
        if m.get("n_alarms", 1) > 1:
            pred += f" (+{m['n_alarms'] - 1})"
        rows.append((m.get("run"), m["kind"], m["truth_start_s"], m["truth_end_s"], pred,
                     f"{m['time_to_alert_s']:.2f}"))
    for m in details.get("missed", []):
        rows.append((m.get("run"), m["kind"], m["truth_start_s"], m["truth_end_s"], "MISSED", "-"))
    rows.sort(key=lambda r: (-1 if r[0] is None else r[0], r[2]))
    if rows:
        by_run = any(r[0] is not None for r in rows)  # several runs scored: say which one
        lines.append("")
        lines.append(f"{'#':>3}  " + (f"{'run':>4}  " if by_run else "")
                     + f"{'kind':<16}{'truth start':>12}{'truth end':>12}  {'matched alarm':<20}{'time-to-alert':>14}")
        for i, (run, kind, ts, te, pred, tta) in enumerate(rows, start=1):
            run_txt = f"{'' if run is None else run:>4}  " if by_run else ""
            lines.append(ascii_safe(f"{i:>3}  {run_txt}{kind:<16}{_fmt_t(ts):>12}{_fmt_t(te):>12}  {pred:<20}{tta:>14}"))
    fas = details.get("false_alarms", [])
    if fas:
        lines.append(ascii_safe("false alarms: " + ", ".join(
            f"{_fmt_t(f['pred_start_s'])}-{_fmt_t(f['pred_end_s'])}"
            + (f" [{f['negative']}]" if f.get("negative") else "") + _run_note(f) for f in fas)))
    dups = details.get("duplicates", [])
    if dups:
        lines.append("re-alarms inside an already matched incident (not false alarms): " + ", ".join(
            f"{_fmt_t(d['pred_start_s'])}-{_fmt_t(d['pred_end_s'])}" + _run_note(d) for d in dups))
    if result.get("ignored_alarms"):
        lines.append(f"alarms inside ignore spans (not counted): {result['ignored_alarms']}")
    if result.get("pre_existing_matches"):
        lines.append(f"incidents credited to an alarm already active before they began (time-to-alert 0): "
                     f"{result['pre_existing_matches']}")
    return "\n".join(lines)


def format_batch_report(batch: Dict[str, Any], title: str = "") -> str:
    """Render ``evaluate_batch`` output: one row per clip, totals, then the false alarms."""
    s = batch["summary"]
    lines = [ascii_safe(title or f"evaluation: {s['clips']} clip(s), {batch['match']} matching, "
                                 f"tolerance {batch['tolerance_s']:.1f} s, run={batch['run']}")]
    header = (f"{'video':<12}{'dur(s)':>8}{'pos':>5}{'det':>5}{'TTA(s)':>8}{'W|A':>5}{'FA':>5}{'FA/h':>8}"
              f"{'FA-s':>8}{'W|A-FA-s':>10}{'warn-out':>10}{'dup':>5}{'ign':>5}")
    lines.append(header)
    for c in batch["clips"]:
        tta = min(c["time_to_alert_s"]) if c["time_to_alert_s"] else None
        lines.append(ascii_safe(
            f"{c['video']:<12}{c['duration_s']:>8.1f}{c['n_truth']:>5}{c['tp']:>5}{_fmt_opt(tta):>8}"
            f"{(c.get('detected_warning_or_alarm', 0) if c['n_truth'] else '-'):>5}{c['fp']:>5}"
            f"{_fmt_opt(c['false_alarms_per_hour']):>8}{c['fa_seconds']:>8.1f}"
            f"{c.get('warn_or_alarm_fa_seconds', 0.0):>10.1f}{c.get('warnings_outside_positives', 0):>10}"
            f"{c['duplicates']:>5}{c['ignored_alarms']:>5}"))
    lines.append("-" * len(header))
    tta = s["mean_time_to_alert_s"]
    lines.append(
        f"{'total':<12}{s['duration_s']:>8.1f}{s['positives']:>5}{s['detected']:>5}{_fmt_opt(tta):>8}"
        f"{s['detected_warning_or_alarm']:>5}{s['false_alarms']:>5}{_fmt_opt(s['false_alarms_per_hour']):>8}"
        f"{s['fa_seconds']:>8.1f}{s['warn_or_alarm_fa_seconds']:>10.1f}{s['warnings_outside_positives']:>10}"
        f"{s['duplicates']:>5}{s['ignored_alarms']:>5}")
    lines.append(f"recall {s['detected']}/{s['positives']} (warning-or-alarm {s['detected_warning_or_alarm']}/"
                 f"{s['positives']})  mean TTA {_fmt_opt(s['mean_time_to_alert_s'], '.2f')} s  max TTA "
                 f"{_fmt_opt(s['max_time_to_alert_s'], '.2f')} s  false alarms {s['false_alarms']} "
                 f"({_fmt_opt(s['false_alarms_per_hour'])}/h over {s['duration_s']:.1f} s)  FA-seconds "
                 f"{s['fa_seconds']:.2f}  W|A FA-seconds {s['warn_or_alarm_fa_seconds']:.2f}")
    fa_lines = []
    missed = []
    for c in batch["clips"]:
        for f in c["details"]["false_alarms"]:
            why = "; ".join(f.get("reasons") or [])
            fa_lines.append(ascii_safe(f"  {c['video']:<10} {_fmt_t(f['pred_start_s']):>8}-{_fmt_t(f['pred_end_s']):<8}"
                                       f" [{f.get('negative') or 'untagged'}]{_run_note(f)} {why}".rstrip()))
        for m in c["details"]["missed"]:
            missed.append(f"{c['video']} {_fmt_t(m['truth_start_s'])}-{_fmt_t(m['truth_end_s'])}{_run_note(m)}")
    if fa_lines:
        lines.append("false alarms:")
        lines.extend(fa_lines)
    if missed:
        lines.append(ascii_safe("missed: " + ", ".join(missed)))
    if s.get("pre_existing_matches"):
        lines.append(f"incidents credited to an alarm already active before they began (time-to-alert 0): "
                     f"{s['pre_existing_matches']}")
    if batch.get("missing"):
        lines.append("no event log for: " + ", ".join(batch["missing"]))
    return "\n".join(lines)


def _json_safe(value: Any) -> Any:
    """Replace +/-inf and NaN with ``None`` so the written JSON is strictly valid.

    ``json.dump`` would otherwise emit the non-standard token ``Infinity`` which many
    downstream tools (jq, browsers) refuse to parse.
    """
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and (math.isinf(value) or math.isnan(value)):
        return None
    return value


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m gradecrossing.evaluate",
        description="Score monitor event logs against ground truth (one clip, or a directory of clips).")
    p.add_argument("--events", required=True,
                   help="JSONL event log written by JsonlSink, or a directory of <video>.events.jsonl")
    p.add_argument("--truth", required=True,
                   help="CSV with columns video,kind,start_s,end_s, a team ground-truth .json (events with "
                        "should_alarm, ignore spans) or a directory of them")
    p.add_argument("--video", default=None,
                   help="video to score (CSV 'video' column); with a directory, a comma-separated list "
                        "(default: every video)")
    p.add_argument("--kind", default=None, help="only score truth incidents of this kind / label")
    p.add_argument("--duration-s", type=float, default=None,
                   help="clip length in seconds, one clip only, used for every scored run (default: "
                        "RUN_END in the log, else the truth JSON duration, else the last event)")
    p.add_argument("--tolerance-s", type=float, default=None,
                   help="matching tolerance (default 2.0 for CSV, 1.0 for team JSON as in its evaluate.py)")
    p.add_argument("--match", choices=list(MATCH_MODES), default=None,
                   help="one_to_one (default for CSV) or any_overlap (the team's rules, default for JSON)")
    p.add_argument("--run", default="last",
                   help="which run of an appended log to score: last (default), a RUN_START run_id, or all "
                        "(each run scored on its own, then added up)")
    p.add_argument("--out", default=None, help="write the metrics as JSON here")
    p.add_argument("--quiet", action="store_true", help="do not print the table")
    return p


def _load_truth_arg(path: str) -> Tuple[str, Any]:
    """('dir', {video: TruthSet}) | ('json', TruthSet) | ('csv', [GroundTruthEvent])."""
    if Path(path).is_dir():
        return "dir", load_truth_dir(path)
    if str(path).lower().endswith(".json"):
        return "json", load_truth_json(path)
    return "csv", load_truth_csv(path)


def _video_of_log(path: str) -> str:
    name = Path(path).name
    for suffix in EVENTS_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return Path(path).stem


def write_json(path: str, data: Any) -> None:
    """Write ``data`` as strictly valid JSON (inf/NaN become null), creating parent dirs."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(_json_safe(data), fh, indent=2)


def _print_notes(notes: Iterable[str]) -> None:
    for note in notes:
        print(ascii_safe(f"warning: {note}"), file=sys.stderr)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point. Returns 0 on success, 1 on a bad input file, an unknown run id or a
    batch with a missing event log."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        truth_kind, truth = _load_truth_arg(args.truth)
        if not Path(args.events).exists():
            raise OSError(f"event log {args.events} does not exist")
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    match, tolerance = default_match_and_tolerance(truth_kind != "csv")
    match = args.match or match
    tolerance = args.tolerance_s if args.tolerance_s is not None else tolerance

    if Path(args.events).is_dir():
        if args.duration_s is not None:
            parser.error("--duration-s applies to one clip; in a batch durations come from RUN_END "
                         "or the truth files")
        logs = find_event_logs(args.events)
        if truth_kind == "dir":
            truths = dict(truth)
            names = sorted(truths)
            for extra_log in sorted(set(logs) - set(truths)):
                print(f"warning: {extra_log}: event log has no ground-truth file; skipped", file=sys.stderr)
        elif truth_kind == "json":
            truths = {truth.video: truth}
            names = [truth.video]
        else:  # CSV lists incidents only: a logged clip without rows is a clip with no incident
            truths = truth_sets_from_csv(truth)
            names = sorted(set(truths) | set(logs))
        if args.video:
            wanted = [v.strip() for v in str(args.video).split(",") if v.strip()]
            unknown = [v for v in wanted if v not in names]
            if unknown:
                print(f"error: unknown video(s) {unknown}; known: {names}", file=sys.stderr)
                return 1
            names = wanted
        try:
            batch = evaluate_batch(truths, logs, names, tolerance, match, run=args.run, kind=args.kind)
        except (OSError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        batch.update({"events": str(args.events), "truth": str(args.truth)})
        _print_notes(n for c in batch["clips"] for n in c["notes"])
        if not args.quiet:
            print(format_batch_report(batch))
        if args.out:
            write_json(args.out, batch)
            if not args.quiet:
                print(f"wrote {args.out}")
        if batch["missing"]:
            print(f"error: no event log in {args.events} for {batch['missing']}", file=sys.stderr)
            return 1
        return 0

    # One clip.
    notes: List[str] = []
    if truth_kind == "csv":
        rows = truth
        videos = sorted({e.video for e in rows})
        if args.video is not None:
            rows = [e for e in rows if e.video == args.video]
            if not rows and videos:
                notes.append(f"no truth rows for video {args.video!r} (scored as a clip with no incidents); "
                             f"videos in the CSV: {videos}")
        elif len(videos) > 1:
            notes.append(f"truth CSV has {len(videos)} videos and --video was not given; scoring all rows together")
        clip_truth = TruthSet(args.video or (videos[0] if len(videos) == 1 else ""), None, rows)
    elif truth_kind == "json":
        clip_truth = truth
        if args.video is not None and args.video != truth.video:
            notes.append(f"--video {args.video!r} ignored: {args.truth} is the ground truth of {truth.video!r}")
    else:
        video = args.video or _video_of_log(args.events)
        if video not in truth:
            print(f"error: no ground truth for video {video!r} in {args.truth} (videos: {sorted(truth)})",
                  file=sys.stderr)
            return 1
        clip_truth = truth[video]
    try:
        result = evaluate_clip(args.events, clip_truth, tolerance, match, run=args.run,
                               duration_s=args.duration_s, kind=args.kind)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    result["notes"] = notes + result["notes"]
    result["truth"] = str(args.truth)
    result["video"] = args.video if truth_kind == "csv" else clip_truth.video
    _print_notes(result["notes"])
    if not args.quiet:
        title = f"evaluation: {Path(args.events).name} vs {Path(args.truth).name}"
        if result["video"]:
            title += f" (video={result['video']})"
        print(format_report(result, title))
    if args.out:
        write_json(args.out, result)
        if not args.quiet:
            print(f"wrote {args.out}")
    return 0


__all__ = [
    "GroundTruthEvent",
    "TruthSet",
    "Episode",
    "RunLog",
    "EventLog",
    "load_truth_csv",
    "load_truth_json",
    "load_truth_dir",
    "truth_sets_from_csv",
    "read_event_log",
    "select_runs",
    "load_alarm_intervals",
    "score",
    "default_match_and_tolerance",
    "evaluate_clip",
    "evaluate_batch",
    "find_event_logs",
    "summarize",
    "format_report",
    "format_batch_report",
    "write_json",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
