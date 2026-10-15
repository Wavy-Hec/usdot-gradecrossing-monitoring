"""The team's zone files load directly (P6, critic-amended spec).

The team writes ``{"vertices": [[x, y], ...], ...}`` in source-video pixels
(occupancy_monitor/zones/clipNN.json, copied to benchmarks/team_clips/zones/); their loaders
also read a "polygon" / "points" key or a bare list. ``zone.polygon_file`` / ``--zone``
loads such a file in place of ``zone.polygon``. The Monitor reads it ONCE, when it is built:
a missing or malformed file is an error from the constructor (before any frame, sink or
model) and from the CLI (exit status 2), never a run that fails frame after frame, and
editing the file during a run does not move the zone on the next reset() (loop, live
re-initialisation).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import gradecrossing.monitor as monitor_mod
from gradecrossing.config import MonitorConfig, ZoneConfig, load_config, load_zone_file, zone_polygon
from gradecrossing.monitor import Monitor, apply_cli_overrides, build_parser, main

ROOT = Path(__file__).resolve().parents[1]
TEAM_ZONES = sorted((ROOT / "benchmarks" / "team_clips" / "zones").glob("clip*.json"))
W, H = 640, 360
FILE_ZONE = [[212, 130], [252, 130], [445, 300], [330, 300]]  # the team's clip07 zone
CONFIG_ZONE = [[10, 10], [60, 10], [60, 60], [10, 60]]  # somewhere else entirely


class NoDetections:
    def detect(self, frame):
        return []


def write(path: Path, content) -> Path:
    path.write_text(content if isinstance(content, str) else json.dumps(content), encoding="utf-8")
    return path


def frame(w: int = W, h: int = H) -> np.ndarray:
    return np.random.default_rng(0).integers(0, 255, size=(h, w, 3), dtype=np.uint8)


def monitor_cfg(zone_file: Path, **video) -> MonitorConfig:
    cfg = MonitorConfig()
    cfg.zone.polygon = [list(p) for p in CONFIG_ZONE]
    cfg.zone.polygon_file = str(zone_file)
    cfg.zone.stabilize = False  # the polygon as configured (scaled), no tracking
    cfg.alerts.stdout = False
    cfg.debris.enabled = False
    for k, v in video.items():
        setattr(cfg.video, k, v)
    return cfg


# ----------------------------------------------------------------------------- the loader
def test_load_zone_file_team_formats(tmp_path):
    assert len(TEAM_ZONES) == 8
    for path in TEAM_ZONES:  # the team's real files, as committed
        vertices = json.loads(path.read_text(encoding="utf-8"))["vertices"]
        points = load_zone_file(str(path))
        assert points == [[float(x), float(y)] for x, y in vertices] and len(points) >= 3, path.name
        assert all(isinstance(v, float) for p in points for v in p)
    for content in ({"vertices": FILE_ZONE, "source": "hand-drawn"}, {"polygon": FILE_ZONE}, {"points": FILE_ZONE},
                    FILE_ZONE):
        assert load_zone_file(str(write(tmp_path / "z.json", content))) == [[float(x), float(y)] for x, y in FILE_ZONE]
    bom = tmp_path / "bom.json"
    bom.write_bytes(b"\xef\xbb\xbf" + json.dumps({"vertices": FILE_ZONE}).encode("utf-8"))  # a Windows editor
    assert len(load_zone_file(str(bom))) == 4


@pytest.mark.parametrize("content, fragment", [
    ({"vertices": [[0, 0], [10, 0]]}, "at least 3 [x, y] points, got 2"),
    ({"verts": FILE_ZONE}, "no vertex list"),
    ({"vertices": [[0, 0], [10, 0], ["10", 10]]}, "point 2 must be [x, y]"),
    ({"vertices": [[0, 0], [10, 0], [True, 10]]}, "point 2 must be [x, y]"),
    ({"vertices": [[0, 0], [10, 0], [10]]}, "point 2 must be [x, y]"),
    ('{"vertices": [[0, 0], [10, 0], [NaN, 10]]}', "point 2 is not finite"),
    ('{"vertices": [[0, 0], [10, 0], [Infinity, 10]]}', "point 2 is not finite"),
    ('{"vertices": [[0, 0], [10, 0], [1e999, 10]]}', "point 2 is not finite"),
    ({"vertices": "0,0 10,0 10,10"}, "expected a list of [x, y] points"),
    (42, "expected a list of [x, y] points"),
    ('{"vertices": [[0, 0], ', "not valid JSON"),
])
def test_load_zone_file_rejects_bad_files(tmp_path, content, fragment):
    path = write(tmp_path / "bad.json", content)
    with pytest.raises(ValueError) as info:
        load_zone_file(str(path))
    assert fragment in str(info.value) and "bad.json" in str(info.value)


def test_a_missing_zone_file_is_an_os_error(tmp_path):
    with pytest.raises(OSError):
        load_zone_file(str(tmp_path / "nope.json"))


def test_zone_polygon_is_the_file_when_set(tmp_path):
    path = write(tmp_path / "z.json", {"vertices": FILE_ZONE})
    assert zone_polygon(ZoneConfig(polygon=CONFIG_ZONE, polygon_file=str(path))) == [[float(x), float(y)]
                                                                                       for x, y in FILE_ZONE]
    assert zone_polygon(ZoneConfig(polygon=CONFIG_ZONE)) == [[float(x), float(y)] for x, y in CONFIG_ZONE]


def test_yaml_polygon_file_loads_and_a_bad_one_is_a_load_error(tmp_path):
    good = write(tmp_path / "z.json", {"vertices": FILE_ZONE})
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(f"zone:\n  polygon_file: '{good.as_posix()}'\n", encoding="utf-8")
    assert load_config(str(cfg_path)).zone.polygon_file == good.as_posix()
    write(good, {"vertices": FILE_ZONE[:2]})
    with pytest.raises(ValueError, match="cannot load zone"):
        load_config(str(cfg_path))
    good.unlink()
    with pytest.raises(ValueError, match="cannot load zone"):
        load_config(str(cfg_path))


# ----------------------------------------------------------------------------- the Monitor
def test_monitor_uses_the_zone_file_scaled_by_resize_width(tmp_path):
    path = write(tmp_path / "z.json", {"vertices": FILE_ZONE})
    r = Monitor(monitor_cfg(path), detector=NoDetections()).process_frame(frame(), 0.0, 0)
    assert np.allclose(r.zone.polygon, FILE_ZONE, atol=1)  # the file, not zone.polygon
    r = Monitor(monitor_cfg(path, resize_width=W // 2), detector=NoDetections()).process_frame(frame(), 0.0, 0)
    assert np.allclose(r.zone.polygon, np.float32(FILE_ZONE) / 2, atol=1)  # source pixels -> processing pixels


def test_monitor_constructor_rejects_a_bad_zone_file(tmp_path):
    events = tmp_path / "events.jsonl"
    two = write(tmp_path / "two.json", {"vertices": FILE_ZONE[:2]})
    for path, error in ((two, ValueError), (tmp_path / "missing.json", OSError)):
        cfg = monitor_cfg(path)
        cfg.alerts.events_jsonl = str(events)
        with pytest.raises(error):
            Monitor(cfg)  # detector=None: it would be built on the first frame, which never comes
        assert not events.exists()  # before any sink was opened


def test_zone_file_is_read_once(tmp_path):
    path = write(tmp_path / "z.json", {"vertices": FILE_ZONE})
    mon = Monitor(monitor_cfg(path), detector=NoDetections())
    write(path, {"vertices": CONFIG_ZONE})  # edited during the run...
    assert np.allclose(mon.process_frame(frame(), 0.0, 0).zone.polygon, FILE_ZONE, atol=1)
    path.unlink()  # ...or deleted
    mon.reset()  # a loop restart / live re-initialisation rebuilds the zone from the stored points
    assert mon.stabilizer is None
    assert np.allclose(mon.process_frame(frame(), 1.0, 30).zone.polygon, FILE_ZONE, atol=1)


# ----------------------------------------------------------------------------- the CLI
def test_cli_zone_flag_and_bad_zone_exit_status(tmp_path, monkeypatch, capsys):
    good = write(tmp_path / "z.json", {"vertices": FILE_ZONE})
    cfg = apply_cli_overrides(MonitorConfig(), build_parser().parse_args(["--zone", str(good)]))
    assert cfg.zone.polygon_file == str(good)
    assert apply_cli_overrides(MonitorConfig(), build_parser().parse_args([])).zone.polygon_file == ""

    def no_monitor(*args, **kwargs):
        raise AssertionError("the Monitor (and so any model) must not be built")

    monkeypatch.setattr(monitor_mod, "Monitor", no_monitor)
    two = write(tmp_path / "two.json", {"vertices": FILE_ZONE[:2]})
    for bad in (two, tmp_path / "missing.json"):
        with pytest.raises(SystemExit) as info:  # argparse: usage error, exit status 2
            main(["--zone", str(bad), "--source", str(tmp_path / "missing.mp4"), "--no-stdout"])
        assert info.value.code == 2
        err = capsys.readouterr().err
        assert "--zone" in err and "cannot load zone" in err and bad.name in err
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(f"zone:\n  polygon_file: '{two.as_posix()}'\n", encoding="utf-8")
    assert main(["--config", str(cfg_path), "--no-stdout"]) == 2  # same mistake in the YAML: a config error
    assert "cannot load zone" in capsys.readouterr().err


def test_python_m_monitor_with_a_bad_zone_exits_2_before_any_model(tmp_path):
    bad = write(tmp_path / "bad.json", {"vertices": [[0, 0], [10, 0]]})
    env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE="1")
    proc = subprocess.run(
        [sys.executable, "-m", "gradecrossing.monitor", "--zone", str(bad), "--source", str(tmp_path / "x.mp4"),
         "--weights", str(tmp_path / "no_such_weights.pt"), "--no-stdout"],
        cwd=str(tmp_path), env=env, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 2, proc.stderr
    assert "cannot load zone" in proc.stderr and "at least 3" in proc.stderr
    assert "loading vehicle detector" not in proc.stderr and "Traceback" not in proc.stderr
