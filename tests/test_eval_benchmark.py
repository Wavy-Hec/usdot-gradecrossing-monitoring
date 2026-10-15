"""The team benchmark inside the repo (benchmarks/team_clips), tools/fetch_clips.py and
tools/run_benchmark.py. No video, network or torch: downloads come from a local HTTP
server and the monitor is replaced by a stub package on PYTHONPATH."""
from __future__ import annotations

import hashlib
import http.server
import importlib.util
import json
import os
import subprocess
import sys
import threading
import types
from pathlib import Path

import pytest
import yaml

from gradecrossing.evaluate import load_truth_json

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
BENCH = ROOT / "benchmarks" / "team_clips"
ENV = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONPATH=str(ROOT))


def load_tool(name: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("tool_" + name, TOOLS / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fetch = load_tool("fetch_clips")
bench = load_tool("run_benchmark")


# --------------------------------------------------------------------------------------
# the data in benchmarks/team_clips
# --------------------------------------------------------------------------------------
def test_registry_describes_the_eight_team_clips():
    reg = fetch.load_registry()
    assert list(reg["clips"]) == [f"clip{i:02d}" for i in range(1, 9)]
    for cid, c in reg["clips"].items():
        assert len(c["drive_id"]) >= 25 and c["file"] == f"{cid}.mp4" and c["size_bytes"] > 100_000
        # decoded counts: fewer frames than the container claims, duration = frames / fps
        assert c["frames"] <= c["container_frames"]
        assert c["duration_s"] == pytest.approx(c["frames"] / c["fps"], abs=1e-3)
        zone = json.loads((BENCH / c["zone"]).read_text(encoding="utf-8"))
        assert len(zone["vertices"]) >= 3 and zone["source"] == "hand-drawn"
        assert all(0 <= x <= c["width"] and 0 <= y <= c["height"] for x, y in zone["vertices"])
        ts = load_truth_json(str(BENCH / c["ground_truth"]))
        assert ts.video == cid and len(ts.events) == c["positives"]
        assert ts.duration_s == pytest.approx(c["duration_s"], abs=1e-3)
        assert ts.container_duration_s > ts.duration_s  # the team's value, kept for reference
    assert sum(c["positives"] for c in reg["clips"].values()) == 3
    assert reg["download_url"].startswith("https://drive.usercontent.google.com/download?id={drive_id}")


def test_ground_truth_keeps_the_team_labels():
    gt = {cid: json.loads((BENCH / "ground_truth" / f"{cid}.json").read_text(encoding="utf-8"))
          for cid in ("clip01", "clip02", "clip07")}
    assert gt["clip01"]["events"][0]["start"] == 2.0 and gt["clip01"]["events"][0]["end"] == 32.7
    assert gt["clip01"]["container_duration"] == 32.7 and gt["clip01"]["duration"] == pytest.approx(32.277)
    assert gt["clip02"]["events"] == [{"start": 8, "end": 66, "should_alarm": False, "note": "gates down"}]
    assert gt["clip07"]["ignore"] == [[5.8, 7.9]] and gt["clip07"]["duration"] == pytest.approx(7.04)
    assert list(gt["clip02"]) == ["video", "duration", "container_duration", "events", "note", "ignore"]


def test_tools_compile_and_print_ascii_help():
    for name in ("fetch_clips", "run_benchmark"):
        proc = subprocess.run([sys.executable, str(TOOLS / f"{name}.py"), "--help"], capture_output=True,
                              text=True, env=ENV, cwd=str(ROOT), timeout=60)
        assert proc.returncode == 0, proc.stderr
        assert "usage" in proc.stdout.lower() and proc.stdout.isascii()


# --------------------------------------------------------------------------------------
# fetch_clips
# --------------------------------------------------------------------------------------
class _Files(http.server.BaseHTTPRequestHandler):
    files = {}
    hits = []

    def do_GET(self):  # noqa: N802
        name = self.path.strip("/")
        _Files.hits.append(name)
        body, ctype = _Files.files.get(name, (b"missing", "text/plain"))
        self.send_response(200 if name in _Files.files else 404)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        return


@pytest.fixture
def file_server():
    _Files.files = {}
    _Files.hits = []
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Files)
    threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}/{{drive_id}}"
    finally:
        srv.shutdown()
        srv.server_close()


def make_registry(tmp_path, clips):
    """clips: {id: bytes}. Writes clips.yaml + zone/ground-truth files; returns its path."""
    (tmp_path / "zones").mkdir(exist_ok=True)
    (tmp_path / "ground_truth").mkdir(exist_ok=True)
    entries = []
    for cid, data in clips.items():
        (tmp_path / "zones" / f"{cid}.json").write_text(json.dumps(
            {"vertices": [[10, 10], [100, 10], [100, 50], [10, 50]], "source": "hand-drawn"}), encoding="utf-8")
        (tmp_path / "ground_truth" / f"{cid}.json").write_text(json.dumps(
            {"video": cid, "duration": 8.0, "container_duration": 8.7,
             "events": [{"start": 2.0, "end": 5.0, "label": "vehicle", "should_alarm": True, "note": "van"}],
             "ignore": [[5.0, 8.0]]}), encoding="utf-8")
        entries.append({"id": cid, "drive_id": f"id_{cid}", "file": f"{cid}.mp4", "size_bytes": len(data),
                        "sha256": hashlib.sha256(data).hexdigest(), "fps": 30.0, "frames": 240, "duration_s": 8.0,
                        "container_frames": 261, "width": 640, "height": 360, "zone": f"zones/{cid}.json",
                        "ground_truth": f"ground_truth/{cid}.json", "positives": 1})
    path = tmp_path / "clips.yaml"
    path.write_text(yaml.safe_dump({"download_url": "http://unused/{drive_id}", "clips": entries}), encoding="utf-8")
    return path


def test_fetch_downloads_verifies_and_reuses_the_cache(tmp_path, file_server, capsys):
    a, b = b"\x00\x00\x00\x18ftypmp42" + b"a" * 5000, b"\x00\x00\x00\x18ftypmp42" + b"b" * 3000
    (tmp_path / "reg").mkdir()
    registry = make_registry(tmp_path / "reg", {"clipA": a, "clipB": b})
    _Files.files = {"id_clipA": (a, "video/mp4"), "id_clipB": (b, "application/octet-stream")}
    cache = tmp_path / "cache"
    args = ["--registry", str(registry), "--cache-dir", str(cache), "--url-template", file_server]
    assert fetch.main(args) == 0
    assert (cache / "clipA.mp4").read_bytes() == a and (cache / "clipB.mp4").read_bytes() == b
    assert sorted(_Files.hits) == ["id_clipA", "id_clipB"]
    assert fetch.main(args) == 0 and len(_Files.hits) == 2  # verified cache: no second download
    assert "cached" in capsys.readouterr().out
    (cache / "clipB.mp4").write_bytes(b"truncated")  # a damaged cache entry is fetched again
    assert fetch.main(args + ["--clips", "clipB"]) == 0 and _Files.hits[-1] == "id_clipB"
    assert fetch.main(args + ["--verify-only", "--clips", "clipA,clipB"]) == 0
    assert fetch.main(args + ["--force", "--clips", "clipA"]) == 0 and _Files.hits[-1] == "id_clipA"


def test_fetch_rejects_wrong_bytes_html_pages_and_bad_registries(tmp_path, file_server, capsys, monkeypatch):
    good = b"\x00\x00\x00\x18ftypmp42" + b"x" * 2000
    (tmp_path / "reg").mkdir()
    registry = make_registry(tmp_path / "reg", {"clipA": good, "clipB": good})
    _Files.files = {"id_clipA": (good[:-1] + b"y", "video/mp4"),  # same size, different content
                    "id_clipB": (b"<html>quota exceeded</html>", "text/html; charset=utf-8")}
    cache = tmp_path / "cache"
    args = ["--registry", str(registry), "--cache-dir", str(cache), "--url-template", file_server]
    assert fetch.main(args) == 1
    err = capsys.readouterr().err
    assert "sha256" in err and "HTML page" in err
    assert not list(cache.glob("*"))  # nothing unverified is left behind, not even .part files
    assert fetch.main(args + ["--verify-only"]) == 1
    assert fetch.main(args + ["--clips", "clipZ"]) == 2
    assert fetch.main(args[:-1] + ["http://127.0.0.1:1/no-placeholder"]) == 2
    bad = tmp_path / "bad.yaml"
    bad.write_text("clips:\n  - {id: c1, drive_id: x, file: c1.mp4, size_bytes: 3}\n", encoding="utf-8")
    assert fetch.main(["--registry", str(bad), "--cache-dir", str(cache)]) == 2
    assert "missing ['sha256']" in capsys.readouterr().err
    monkeypatch.setenv("GRADECROSSING_CLIPS", str(tmp_path / "envcache"))
    assert fetch.default_cache_dir() == tmp_path / "envcache"
    monkeypatch.delenv("GRADECROSSING_CLIPS")
    assert fetch.default_cache_dir().parts[-2:] == ("gradecrossing", "team_clips")


# --------------------------------------------------------------------------------------
# run_benchmark
# --------------------------------------------------------------------------------------
def test_minimal_config_and_overrides(tmp_path):
    clip = b"video"
    (tmp_path / "reg").mkdir()
    registry = fetch.load_registry(str(make_registry(tmp_path / "reg", {"clipA": clip})))
    video = tmp_path / "clipA.mp4"
    video.write_bytes(clip)
    args = bench.build_parser().parse_args(["--out", str(tmp_path / "o"), "--weights", "w.pt", "--device", "0"])
    sets = bench.parse_sets(["stuck.alarm_seconds=3", "debris.enabled=false", "zone.stabilize=false"])
    cfg = bench.make_config(registry["clips"]["clipA"], Path(registry["base_dir"]), video, args,
                            {"stuck": {"warning_seconds": 1.5}, "alerts": {"latch_seconds": 5.0}}, sets)
    assert cfg["video"]["source"] == video.resolve().as_posix()
    assert cfg["zone"] == {"polygon": [[10.0, 10.0], [100.0, 10.0], [100.0, 50.0], [10.0, 50.0]],
                           "stabilize": False}
    assert cfg["detector"] == {"weights": "w.pt", "device": "0"}
    assert cfg["alerts"] == {"stdout": False, "latch_seconds": 5.0}
    assert cfg["stuck"] == {"warning_seconds": 1.5, "alarm_seconds": 3} and cfg["debris"] == {"enabled": False}
    with pytest.raises(ValueError):
        bench.parse_sets(["no_equals_sign"])


@pytest.mark.parametrize("content", [
    '﻿{"vertices": [[10, 10], [100, 10], [100, 50], [10, 50]]}',  # a BOM from a Windows editor
    '{"points": [[10, 10], [100, 10], [100, 50], [10, 50]]}',  # the key Gael's extract_roi.py writes
    '[[10, 10], [100, 10], [100, 50], [10, 50]]',  # a bare list
])
def test_the_benchmark_reads_zone_files_with_the_monitors_loader(tmp_path, content):
    """Final review Q9: run_benchmark had its own zone reader (no BOM, no 'points', no finiteness
    check), so a zone file that --zone and zone.polygon_file accept could crash the harness."""
    clip = b"video"
    (tmp_path / "reg").mkdir()
    registry = fetch.load_registry(str(make_registry(tmp_path / "reg", {"clipA": clip})))
    (tmp_path / "reg" / "zones" / "clipA.json").write_text(content, encoding="utf-8")
    video = tmp_path / "clipA.mp4"
    video.write_bytes(clip)
    args = bench.build_parser().parse_args(["--out", str(tmp_path / "o")])
    cfg = bench.make_config(registry["clips"]["clipA"], Path(registry["base_dir"]), video, args, {}, [])
    assert cfg["zone"] == {"polygon": [[10.0, 10.0], [100.0, 10.0], [100.0, 50.0], [10.0, 50.0]]}
    assert not hasattr(bench, "zone_vertices")  # one loader


def test_a_bad_zone_file_fails_its_clip_with_one_line(tmp_path, capsys):
    clip = b"clip a bytes"
    (tmp_path / "reg").mkdir()
    registry = make_registry(tmp_path / "reg", {"clipA": clip})
    (tmp_path / "reg" / "zones" / "clipA.json").write_text('{"vertices": [[0, 0], [10, NaN], [20, 20]]}',
                                                          encoding="utf-8")
    videos = tmp_path / "videos"
    videos.mkdir()
    (videos / "clipA.mp4").write_bytes(clip)
    out = tmp_path / "run"
    assert bench.main(["--registry", str(registry), "--clips-dir", str(videos), "--out", str(out),
                       "--python", sys.executable]) == 1
    err = capsys.readouterr().err
    assert "clipA: zone file" in err and "not finite" in err and "Traceback" not in err
    assert not (out / "configs" / "clipA.yaml").exists() and not (out / "clipA.events.jsonl").exists()


STUB_MONITOR = '''
"""Stand-in for gradecrossing.monitor: reads --config/--events, writes a fixed log."""
import json, sys
import yaml
args = dict(zip(sys.argv[1::2], sys.argv[2::2]))
cfg = yaml.safe_load(open(args["--config"], encoding="utf-8"))
source = cfg["video"]["source"]
if source.endswith("clipB.mp4"):
    print("simulated crash", file=sys.stderr)
    sys.exit(3)
alarm = float(cfg.get("stuck", {}).get("alarm_seconds", 8.0))
def ev(t, typ, **extra):
    return json.dumps({"type": typ, "timestamp": t, "frame_index": int(t * 30), "status": "ALARM",
                       "reasons": [typ.lower()], "track_ids": [1], "extra": extra})
with open(args["--events"], "a", encoding="utf-8") as fh:
    fh.write(ev(0.0, "RUN_START", run_id="stub", source=source) + "\\n")
    fh.write(ev(alarm, "ALARM_START") + "\\n")
    fh.write(ev(alarm + 1.0, "ALARM_END") + "\\n")
    fh.write(ev(8.0, "RUN_END", run_id="stub", duration_s=8.0, frames=240) + "\\n")
print("summary: frames=240 alarms=1 (stub)")
'''


def test_run_benchmark_runs_each_clip_and_scores_the_logs(tmp_path, capsys):
    clip_a, clip_b = b"clip a bytes", b"clip b bytes"
    (tmp_path / "reg").mkdir()
    registry = make_registry(tmp_path / "reg", {"clipA": clip_a, "clipB": clip_b})
    videos = tmp_path / "videos"
    videos.mkdir()
    (videos / "clipA.mp4").write_bytes(clip_a)
    (videos / "clipB.mp4").write_bytes(clip_b)
    fake_repo = tmp_path / "fake_repo" / "gradecrossing"
    fake_repo.mkdir(parents=True)
    (fake_repo / "__init__.py").write_text("", encoding="utf-8")
    (fake_repo / "monitor.py").write_text(STUB_MONITOR, encoding="utf-8")
    out = tmp_path / "run"
    base = ["--registry", str(registry), "--clips-dir", str(videos), "--out", str(out),
            "--repo", str(fake_repo.parent), "--python", sys.executable]
    rc = bench.main(base + ["--clips", "clipA", "--set", "stuck.alarm_seconds=3", "--jobs", "2"])
    assert rc == 0
    captured = capsys.readouterr()
    assert "clipA: exit 0" in captured.out and "summary: frames=240 alarms=1 (stub)" in captured.out
    cfg = yaml.safe_load((out / "configs" / "clipA.yaml").read_text(encoding="utf-8"))
    assert cfg["stuck"] == {"alarm_seconds": 3} and cfg["alerts"] == {"stdout": False}
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    (clip,) = report["clips"]
    assert (clip["video"], clip["tp"], clip["fp"], clip["duration_source"]) == ("clipA", 1, 0, "RUN_END")
    assert clip["time_to_alert_s"] == [1.0]  # alarm at 3 s, incident from 2 s
    assert report["run_meta"]["sets"] == ["stuck.alarm_seconds=3"]
    assert "recall 1/1" in (out / "report.txt").read_text(encoding="utf-8")
    # re-running appends nothing: the log is fresh, one run
    assert bench.main(base + ["--clips", "clipA"]) == 0
    assert (out / "clipA.events.jsonl").read_text(encoding="utf-8").count("RUN_START") == 1
    # a crashing monitor run fails the benchmark (exit 1) but the other clips are still scored
    assert bench.main(base) == 1
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["run_meta"]["runs"]["clipB"]["exit_code"] == 3
    assert [c["video"] for c in report["clips"]] == ["clipA"] and report["missing"] == ["clipB"]
    # score-only re-scores what is there; a corrupted video is refused before running
    assert bench.main(base + ["--score-only", "--clips", "clipA"]) == 0
    (videos / "clipA.mp4").write_bytes(b"changed")
    assert bench.main(base + ["--clips", "clipA"]) == 1
    assert "size" in capsys.readouterr().err and not (out / "clipA.events.jsonl").exists()
    assert bench.main(base + ["--clips", "nope"]) == 2
