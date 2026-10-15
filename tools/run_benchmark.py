#!/usr/bin/env python
"""Run the monitor over the team benchmark clips and score the run the way the team does.

For every clip of ``benchmarks/team_clips/clips.yaml`` this writes a MINIMAL config: only
``video.source``, ``zone.polygon`` (the clip's hand-drawn zone, source pixels),
``detector.weights`` / ``detector.device``, ``alerts.stdout: false`` and
``output.print_every_n_frames: 0``, plus every ``--extra-yaml`` / ``--set`` override. So a
run measures the checkout's DEFAULTS plus the overrides you name. Each clip is processed by

    python -m gradecrossing.monitor --config OUT/configs/CLIP.yaml --events OUT/CLIP.events.jsonl

(a fresh log per clip, cwd = OUT) and the logs are then scored with ``gradecrossing.evaluate``
in batch mode against ``benchmarks/team_clips/ground_truth``: the team's rules (any overlap,
1 s tolerance, ignore spans, false alarms tagged by labelled negative), decoded durations.

Examples::

    python tools/fetch_clips.py --clips clip07,clip08
    python tools/run_benchmark.py --out runs/bench_defaults --clips clip07,clip08 --device 0
    python tools/run_benchmark.py --out runs/bench_alarm3 --set stuck.alarm_seconds=3 --set stuck.warning_seconds=1.5
    python tools/run_benchmark.py --out runs/bench_defaults --score-only      # re-score existing logs

Outputs in OUT: configs/, logs/, CLIP.events.jsonl, run_meta.json, report.txt and report.json.
Exit status: 0 when every selected clip ran (exit 0) and was scored, 1 when a clip file is
missing or does not verify, its zone file cannot be loaded, a monitor run failed or a log is
missing, 2 for bad arguments.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)  # allow `python tools/run_benchmark.py` from any cwd

import fetch_clips  # noqa: E402
from gradecrossing import evaluate  # noqa: E402
from gradecrossing.config import load_zone_file  # noqa: E402

GT_SUBDIR = "ground_truth"


def set_dotted(cfg: Dict[str, Any], dotted: str, value: Any) -> None:
    node = cfg
    keys = dotted.split(".")
    for key in keys[:-1]:
        nxt = node.setdefault(key, {})
        if not isinstance(nxt, dict):
            raise ValueError(f"--set {dotted}: {key} is not a section")
        node = nxt
    node[keys[-1]] = value


def deep_merge(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            deep_merge(dst[key], value)
        else:
            dst[key] = value
    return dst


def parse_sets(items: Sequence[str]) -> List[Any]:
    """``dotted.key=yaml_value`` strings -> [(key, value)]. Raises ValueError on a bad item."""
    import yaml

    out = []
    for item in items:
        key, sep, raw = item.partition("=")
        if not sep or not key.strip():
            raise ValueError(f"--set expects dotted.key=value, got {item!r}")
        out.append((key.strip(), yaml.safe_load(raw)))
    return out


def make_config(entry: Dict[str, Any], base_dir: Path, video: Path, args: argparse.Namespace,
                extra: Dict[str, Any], sets: Sequence[Any]) -> Dict[str, Any]:
    """The clip's minimal config. The zone file is read with the monitor's own loader
    (``gradecrossing.config.load_zone_file``: "vertices"/"polygon"/"points", a BOM, finite
    numbers), so the benchmark accepts exactly the files ``--zone`` and ``zone.polygon_file``
    accept; it raises ``OSError`` / ``ValueError`` like that loader."""
    cfg: Dict[str, Any] = {
        "video": {"source": video.resolve().as_posix()},
        "zone": {"polygon": load_zone_file(str(base_dir / str(entry["zone"])))},
        "detector": {"weights": args.weights, "device": args.device},
        "alerts": {"stdout": False},
        "output": {"print_every_n_frames": 0},
    }
    deep_merge(cfg, json.loads(json.dumps(extra)))  # copy: one dict per clip
    for key, value in sets:
        set_dotted(cfg, key, value)
    return cfg


def run_clips(entries: Sequence[Dict[str, Any]], configs: Dict[str, Path], out: Path,
              args: argparse.Namespace) -> Dict[str, Any]:
    """Run the monitor once per clip (``--jobs`` at a time); returns per-clip exit codes/times."""
    (out / "logs").mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(args.repo).resolve()) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.setdefault("PYTHONIOENCODING", "utf-8")
    if args.jobs > 1:
        env.setdefault("OMP_NUM_THREADS", "2")  # several torch processes share the CPU
    pending = list(entries)
    procs: Dict[str, Any] = {}
    meta: Dict[str, Any] = {}
    while pending or procs:
        while pending and len(procs) < args.jobs:
            entry = pending.pop(0)
            cid = entry["id"]
            events = out / f"{cid}.events.jsonl"
            if events.exists():
                events.unlink()  # the sink appends; each benchmark run starts a fresh log
            log = open(out / "logs" / f"{cid}.log", "w", encoding="utf-8")
            cmd = [args.python, "-m", "gradecrossing.monitor", "--config", str(configs[cid]), "--events", str(events)]
            procs[cid] = (subprocess.Popen(cmd, cwd=str(out), env=env, stdout=log, stderr=subprocess.STDOUT),
                          log, time.monotonic())
        time.sleep(0.2)
        for cid in list(procs):
            proc, log, t0 = procs[cid]
            rc = proc.poll()
            if rc is None:
                continue
            log.close()
            text = (out / "logs" / f"{cid}.log").read_text(encoding="utf-8", errors="replace")
            summary = [ln for ln in text.splitlines() if ln.startswith("summary:")]
            meta[cid] = {"exit_code": rc, "wall_s": round(time.monotonic() - t0, 1),
                         "summary_line": summary[-1] if summary else ""}
            print(f"{cid}: exit {rc} in {meta[cid]['wall_s']:.1f}s {meta[cid]['summary_line']}", flush=True)
            del procs[cid]
    return meta


def score_run(registry: Dict[str, Any], entries: Sequence[Dict[str, Any]], out: Path,
              run: str = "last") -> Dict[str, Any]:
    """Batch-score OUT/<clip>.events.jsonl against the registry's ground truth (team rules)."""
    base = Path(registry["base_dir"])
    truths = {}
    for entry in entries:
        ts = evaluate.load_truth_json(str(base / str(entry.get("ground_truth") or f"{GT_SUBDIR}/{entry['id']}.json")))
        truths[entry["id"]] = ts
    logs = {e["id"]: str(out / f"{e['id']}.events.jsonl") for e in entries
            if (out / f"{e['id']}.events.jsonl").is_file()}
    match, tolerance = evaluate.default_match_and_tolerance(True)
    return evaluate.evaluate_batch(truths, logs, [e["id"] for e in entries], tolerance, match, run=run)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run gradecrossing.monitor on the team benchmark clips and score it.")
    p.add_argument("--out", required=True, help="run directory (configs, logs, event logs, report)")
    p.add_argument("--registry", default=str(fetch_clips.DEFAULT_REGISTRY), help="clip registry (default: %(default)s)")
    p.add_argument("--clips-dir", default=None,
                   help="directory with the videos (default: the tools/fetch_clips.py cache)")
    p.add_argument("--clips", default="", help="comma-separated clip ids (default: all)")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="config override, dotted.key=yaml_value (repeatable), e.g. stuck.alarm_seconds=3")
    p.add_argument("--extra-yaml", default="", help="YAML file deep-merged into every clip config")
    p.add_argument("--weights", default="yolo11n.pt",
                   help="detector weights; an existing path is made absolute (default: %(default)s)")
    p.add_argument("--device", default="", help="detector.device: '' auto, '0' first CUDA GPU, 'cpu'")
    p.add_argument("--python", default=sys.executable, help="interpreter that runs the monitor (default: this one)")
    p.add_argument("--repo", default=str(ROOT), help="checkout to benchmark (put on PYTHONPATH; default: this one)")
    p.add_argument("--jobs", type=int, default=1, help="clips processed in parallel")
    p.add_argument("--run", default="last", help="which run of each log to score (see gradecrossing.evaluate)")
    p.add_argument("--no-verify", action="store_true", help="skip the size/sha256 check of the videos")
    p.add_argument("--score-only", action="store_true", help="do not run the monitor; re-score the logs in --out")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.jobs < 1:
        parser.error("--jobs must be >= 1")
    try:
        registry = fetch_clips.load_registry(args.registry)
        entries = fetch_clips.select_clips(registry, args.clips)
        sets = parse_sets(args.set)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    meta_path = out / "run_meta.json"
    ok = True
    if not args.score_only:
        import yaml

        extra: Dict[str, Any] = {}
        if args.extra_yaml:
            with open(args.extra_yaml, "r", encoding="utf-8") as fh:
                extra = yaml.safe_load(fh) or {}
        if Path(args.weights).is_file():
            args.weights = str(Path(args.weights).resolve())
        clips_dir = Path(args.clips_dir).expanduser() if args.clips_dir else fetch_clips.default_cache_dir()
        base = Path(registry["base_dir"])
        configs: Dict[str, Path] = {}
        runnable = []
        (out / "configs").mkdir(parents=True, exist_ok=True)
        for entry in entries:
            video = fetch_clips.clip_path(clips_dir, entry)
            problem = None
            if not video.is_file():
                problem = "not found (run tools/fetch_clips.py or pass --clips-dir)"
            elif not args.no_verify:
                problem = fetch_clips.verify_file(video, entry)
            if problem:
                print(f"{entry['id']}: {video}: {problem}", file=sys.stderr)
                stale = out / f"{entry['id']}.events.jsonl"
                if stale.exists():
                    stale.unlink()  # never score an older run's log for a clip that did not run
                ok = False
                continue
            try:
                cfg = make_config(entry, base, video, args, extra, sets)
            except (OSError, ValueError) as exc:  # the clip's zone file (or a --set through a value)
                print(f"{entry['id']}: {' '.join(str(exc).split())}", file=sys.stderr)
                stale = out / f"{entry['id']}.events.jsonl"
                if stale.exists():
                    stale.unlink()
                ok = False
                continue
            path = out / "configs" / f"{entry['id']}.yaml"
            with open(path, "w", encoding="utf-8", newline="\n") as fh:
                yaml.safe_dump(cfg, fh, sort_keys=False)
            configs[entry["id"]] = path
            runnable.append(entry)
        meta = run_clips(runnable, configs, out, args)
        ok = ok and all(m["exit_code"] == 0 for m in meta.values())
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump({"repo": str(Path(args.repo).resolve()), "python": args.python, "registry": registry["path"],
                       "clips_dir": str(clips_dir), "weights": args.weights, "device": args.device,
                       "sets": args.set, "extra_yaml": args.extra_yaml, "runs": meta}, fh, indent=1)
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    try:
        batch = score_run(registry, entries, out, run=args.run)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    batch["run_meta"] = meta
    header = [f"benchmark: {out.name}  repo: {meta.get('repo', '?')}  sets: {meta.get('sets', [])}",
              "exit codes: " + json.dumps({c: m.get("exit_code") for c, m in (meta.get("runs") or {}).items()})]
    for clip in batch["clips"]:
        for note in clip["notes"]:
            print(f"warning: {note}", file=sys.stderr)
    text = "\n".join(header + [evaluate.format_batch_report(batch)])
    print(text)
    (out / "report.txt").write_text(text + "\n", encoding="utf-8")
    evaluate.write_json(str(out / "report.json"), batch)
    if batch["missing"]:
        ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
