"""Offline evaluation of a monitor run against hand-labelled ground truth.

Workflow
--------
1. Run the monitor on a clip with ``--events events.jsonl`` (``JsonlSink``).
2. Label the clip once in a CSV with columns ``video,kind,start_s,end_s`` where each row
   is one incident that *should* have produced an ALARM (a vehicle stuck on the tracks,
   debris, a pedestrian).
3. ``python -m gradecrossing.evaluate --events events.jsonl --truth truth.csv --video NAME
   --duration-s 32.7 [--out metrics.json]``.

Scoring is interval based: every ``ALARM_START .. ALARM_END`` pair in the event log is
one *predicted alarm*; each is matched to at most one unmatched truth incident when the
two intervals overlap, or when the prediction starts within ``tolerance_s`` of the truth
start (an alarm that fires slightly early because the labeller was generous is not a
false alarm). Time-to-alert is ``pred_start - truth_start`` clipped at 0.

The metrics that matter for a safety monitor are recall (did we catch every incident),
time-to-alert (how fast) and false alarms per hour (will operators start ignoring it).
Precision/F1 are reported for completeness.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

TRUTH_COLUMNS = ("video", "kind", "start_s", "end_s")


@dataclass
class GroundTruthEvent:
    """One labelled incident. ``end_s`` may be ``inf`` when the incident runs to the clip end."""

    video: str
    kind: str
    start_s: float
    end_s: float


# --------------------------------------------------------------------------------------
# Loading
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


def load_alarm_intervals(events_jsonl_path: str) -> List[Tuple[float, float]]:
    """Return ``(start, end)`` pairs for every ALARM episode in a JSONL event log.

    An episode opens on an ``ALARM_START`` (repeat notifications, marked with
    ``extra.repeat``, and duplicate starts are ignored) and closes on the next
    ``ALARM_END``. An episode still open at the end of the file gets ``end = +inf``.
    Malformed lines (for example a truncated last line after a crash) are skipped with a
    warning instead of aborting the evaluation.
    """
    intervals: List[Tuple[float, float]] = []
    open_start: Optional[float] = None
    with open(events_jsonl_path, "r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("%s line %d: not valid JSON, skipped", events_jsonl_path, line_no)
                continue
            if not isinstance(rec, dict):
                continue
            etype = rec.get("type")
            try:
                t = float(rec.get("timestamp", 0.0))
            except (TypeError, ValueError):
                logger.warning("%s line %d: bad timestamp, skipped", events_jsonl_path, line_no)
                continue
            extra = rec.get("extra") or {}
            if etype == "ALARM_START":
                if isinstance(extra, dict) and extra.get("repeat"):
                    continue
                if open_start is None:
                    open_start = t
            elif etype == "ALARM_END":
                if open_start is not None:
                    intervals.append((open_start, max(t, open_start)))
                    open_start = None
    if open_start is not None:
        intervals.append((open_start, math.inf))
    intervals.sort(key=lambda iv: iv[0])
    return intervals


# --------------------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------------------
def _overlaps(a: Tuple[float, float], b: Tuple[float, float]) -> bool:
    return a[0] <= b[1] and b[0] <= a[1]


def score(truth: List[GroundTruthEvent], alarms: List[Tuple[float, float]], duration_s: float,
          tolerance_s: float = 2.0) -> Dict[str, Any]:
    """Match predicted alarm intervals to truth incidents and compute the metrics.

    Matching is greedy in order of prediction start: each prediction takes the earliest
    unmatched truth incident that it overlaps or whose start is within ``tolerance_s``
    of the prediction start. Predictions left over are false alarms; truth left over are
    misses. Precision/recall are defined as 1.0 when their denominator is zero (nothing
    predicted / nothing to find is not an error); F1 is 0.0 when both are 0.

    Returns a JSON-serialisable dict with ``tp, fp, fn, precision, recall, f1,
    time_to_alert_s, median_time_to_alert_s, false_alarms_per_hour, duration_s`` plus a
    ``details`` block listing every match, miss and false alarm for the report table.
    """
    truth_sorted = sorted(truth, key=lambda e: e.start_s)
    preds = sorted(((float(s), float(e)) for s, e in alarms), key=lambda iv: iv[0])
    matched_truth = [False] * len(truth_sorted)
    matches: List[Dict[str, Any]] = []
    false_alarms: List[Dict[str, float]] = []

    for ps, pe in preds:
        chosen = None
        for i, tr in enumerate(truth_sorted):
            if matched_truth[i]:
                continue
            if _overlaps((ps, pe), (tr.start_s, tr.end_s)) or abs(ps - tr.start_s) <= tolerance_s:
                chosen = i
                break
        if chosen is None:
            false_alarms.append({"pred_start_s": ps, "pred_end_s": pe})
            continue
        matched_truth[chosen] = True
        tr = truth_sorted[chosen]
        matches.append({
            "kind": tr.kind,
            "truth_start_s": tr.start_s,
            "truth_end_s": tr.end_s,
            "pred_start_s": ps,
            "pred_end_s": pe,
            "time_to_alert_s": max(0.0, ps - tr.start_s),
        })

    missed = [{"kind": tr.kind, "truth_start_s": tr.start_s, "truth_end_s": tr.end_s}
              for i, tr in enumerate(truth_sorted) if not matched_truth[i]]
    matches.sort(key=lambda m: m["truth_start_s"])

    tp = len(matches)
    fp = len(false_alarms)
    fn = len(missed)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    tta = [m["time_to_alert_s"] for m in matches]
    median_tta = float(statistics.median(tta)) if tta else None
    duration_s = float(duration_s)
    if fp == 0:
        fa_per_hour: Optional[float] = 0.0
    elif duration_s > 0:
        fa_per_hour = fp / (duration_s / 3600.0)
    else:
        fa_per_hour = None  # unknown duration: cannot normalise

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "time_to_alert_s": tta,
        "median_time_to_alert_s": median_tta,
        "false_alarms_per_hour": fa_per_hour,
        "duration_s": duration_s,
        "tolerance_s": float(tolerance_s),
        "n_truth": len(truth_sorted),
        "n_predicted": len(preds),
        "details": {"matches": matches, "missed": missed, "false_alarms": false_alarms},
    }


# --------------------------------------------------------------------------------------
# Reporting / CLI
# --------------------------------------------------------------------------------------
def _fmt_t(value: float) -> str:
    return "end" if math.isinf(value) else f"{value:.2f}"


def format_report(result: Dict[str, Any], title: str = "") -> str:
    """Render ``score`` output as an ASCII table (safe for cp1252 consoles)."""
    lines: List[str] = []
    if title:
        lines.append(title)
    fa = result.get("false_alarms_per_hour")
    med = result.get("median_time_to_alert_s")
    lines.append(f"duration {result['duration_s']:.1f} s, tolerance {result['tolerance_s']:.1f} s")
    lines.append(f"truth incidents: {result['n_truth']}   predicted alarms: {result['n_predicted']}")
    lines.append(f"TP {result['tp']}  FP {result['fp']}  FN {result['fn']}   "
                 f"precision {result['precision']:.3f}  recall {result['recall']:.3f}  f1 {result['f1']:.3f}")
    lines.append("median time-to-alert " + ("n/a" if med is None else f"{med:.2f} s")
                 + "   false alarms/hour " + ("n/a" if fa is None else f"{fa:.1f}"))
    details = result.get("details", {})
    rows = []
    for m in details.get("matches", []):
        rows.append((m["kind"], m["truth_start_s"], m["truth_end_s"],
                     f"{_fmt_t(m['pred_start_s'])}-{_fmt_t(m['pred_end_s'])}", f"{m['time_to_alert_s']:.2f}"))
    for m in details.get("missed", []):
        rows.append((m["kind"], m["truth_start_s"], m["truth_end_s"], "MISSED", "-"))
    rows.sort(key=lambda r: r[1])
    if rows:
        lines.append("")
        lines.append(f"{'#':>3}  {'kind':<16}{'truth start':>12}{'truth end':>12}  {'matched alarm':<20}{'time-to-alert':>14}")
        for i, (kind, ts, te, pred, tta) in enumerate(rows, start=1):
            lines.append(f"{i:>3}  {kind:<16}{_fmt_t(ts):>12}{_fmt_t(te):>12}  {pred:<20}{tta:>14}")
    fas = details.get("false_alarms", [])
    if fas:
        lines.append("false alarms: " + ", ".join(
            f"{_fmt_t(f['pred_start_s'])}-{_fmt_t(f['pred_end_s'])}" for f in fas))
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


def _max_event_timestamp(events_jsonl_path: str) -> Optional[float]:
    """Last timestamp seen in the event log; used when ``--duration-s`` is not given."""
    best: Optional[float] = None
    with open(events_jsonl_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                t = float(json.loads(line).get("timestamp", 0.0))
            except (ValueError, AttributeError, TypeError):
                continue
            best = t if best is None else max(best, t)
    return best


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m gradecrossing.evaluate",
        description="Score a monitor event log against a ground-truth CSV.")
    p.add_argument("--events", required=True, help="JSONL event log written by JsonlSink")
    p.add_argument("--truth", required=True, help="CSV with columns video,kind,start_s,end_s")
    p.add_argument("--video", default=None,
                   help="value of the 'video' column to evaluate (default: all rows)")
    p.add_argument("--kind", default=None, help="only score truth rows of this kind")
    p.add_argument("--duration-s", type=float, default=None,
                   help="clip length in seconds (default: last event timestamp)")
    p.add_argument("--tolerance-s", type=float, default=2.0,
                   help="early-start tolerance for matching (default 2.0)")
    p.add_argument("--out", default=None, help="write the metrics dict as JSON here")
    p.add_argument("--quiet", action="store_true", help="do not print the table")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point. Returns 0 on success, 1 on a bad input file."""
    args = build_parser().parse_args(argv)
    try:
        truth = load_truth_csv(args.truth)
        alarms = load_alarm_intervals(args.events)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    videos = sorted({e.video for e in truth})
    if args.video is not None:
        truth = [e for e in truth if e.video == args.video]
        if not truth and videos:
            print(f"warning: no truth rows for video {args.video!r}; known videos: {videos}",
                  file=sys.stderr)
    elif len(videos) > 1:
        print(f"warning: truth CSV has {len(videos)} videos and --video was not given; "
              f"scoring all rows together", file=sys.stderr)
    if args.kind is not None:
        truth = [e for e in truth if e.kind == args.kind]

    duration = args.duration_s
    if duration is None:
        last = _max_event_timestamp(args.events)
        last_truth = max((e.end_s for e in truth if not math.isinf(e.end_s)), default=0.0)
        duration = max(last or 0.0, last_truth)
        print(f"note: --duration-s not given; using {duration:.1f} s from the logs", file=sys.stderr)

    result = score(truth, alarms, duration_s=duration, tolerance_s=args.tolerance_s)
    result["events"] = str(args.events)
    result["truth"] = str(args.truth)
    result["video"] = args.video
    if not args.quiet:
        title = f"evaluation: {Path(args.events).name} vs {Path(args.truth).name}"
        if args.video:
            title += f" (video={args.video})"
        print(format_report(result, title))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(_json_safe(result), fh, indent=2)
        if not args.quiet:
            print(f"wrote {args.out}")
    return 0


__all__ = [
    "GroundTruthEvent",
    "load_truth_csv",
    "load_alarm_intervals",
    "score",
    "format_report",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
