"""tools/calibrate_zone.py headless: what it writes and how it fails (ours-cleanup OC-23, OC-13).

The site config holds only what the site needs (video.source and the zone calibration) on top
of the keys its base file sets; it never freezes the package defaults. A base config that
cannot be loaded exits 2 with a one-line error before anything is written.
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from gradecrossing.config import MonitorConfig, load_config, zone_polygon

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "calibrate_zone.py"
W, H = 640, 360
POLY = "200,150 440,150 440,270 200,270"


def _tool():
    spec = importlib.util.spec_from_file_location("calibrate_zone_tool_under_test", str(TOOL))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def shot(tmp_path):
    rng = np.random.default_rng(3)
    frame = cv2.GaussianBlur(rng.integers(0, 256, (H, W, 3), dtype=np.uint8), (5, 5), 0)  # textured
    path = tmp_path / "shot.png"
    assert cv2.imwrite(str(path), frame)
    return path


def test_a_new_site_file_holds_only_the_site_keys(tmp_path, shot, capsys):
    out = tmp_path / "configs" / "site.yaml"
    assert _tool().main(["--source", str(shot), "--polygon", POLY, "--out", str(out), "--no-seed"]) == 0
    text = out.read_text(encoding="utf-8")
    assert text.startswith("# Site config written by tools/calibrate_zone.py (gradecrossing ")
    data = yaml.safe_load(text)
    assert data == {
        "video": {"source": str(shot)},
        "zone": {"polygon": [[200.0, 150.0], [440.0, 150.0], [440.0, 270.0], [200.0, 270.0]],
                 "reference_size": [W, H],
                 "reference_image": (out.parent / "site_reference.png").as_posix()},
    }
    assert len(text.splitlines()) < 15  # was ~120 lines of frozen defaults
    cfg = load_config(str(out))
    defaults = MonitorConfig()
    assert cfg.stuck == defaults.stuck and cfg.debris == defaults.debris and cfg.detector == defaults.detector
    assert "only this site's keys" in capsys.readouterr().out


def test_recalibrating_keeps_the_sites_own_keys_and_adds_no_defaults(tmp_path, shot):
    out = tmp_path / "site.yaml"
    out.write_text("video:\n  source: rtsp://cam/1\n  resize_width: 960\nstuck:\n  alarm_seconds: 3.0\n"
                   "zone:\n  polygon: [[0, 0], [10, 0], [10, 10]]\n  reference_image: old_reference.png\n"
                   "  stabilize: false\n", encoding="utf-8")
    assert _tool().main(["--source", str(shot), "--polygon", POLY, "--out", str(out), "--no-seed",
                         "--keep-source", "--no-reference-image"]) == 0
    data = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert data["video"] == {"source": "rtsp://cam/1", "resize_width": 960}  # --keep-source
    assert data["stuck"] == {"alarm_seconds": 3.0}  # pinned by the site: kept
    assert data["zone"] == {"polygon": [[200.0, 150.0], [440.0, 150.0], [440.0, 270.0], [200.0, 270.0]],
                            "stabilize": False, "reference_size": [W, H]}  # stale image removed
    assert sorted(data) == ["stuck", "video", "zone"]


def test_recalibrating_a_site_with_a_zone_file_replaces_the_file(tmp_path, shot, capsys):
    """Final review Q1: zone.polygon_file wins over zone.polygon in the monitor, so a kept file
    left the site on the old zone while the tool reported the new polygon."""
    old_zone = tmp_path / "old_zone.json"
    old_zone.write_text('{"vertices": [[10, 10], [20, 10], [20, 20], [10, 20]]}', encoding="utf-8")
    out = tmp_path / "site.yaml"
    out.write_text(f"video:\n  source: rtsp://cam/1\nzone:\n  polygon_file: {old_zone.as_posix()}\n", encoding="utf-8")
    assert zone_polygon(load_config(str(out)).zone) == [[10.0, 10.0], [20.0, 10.0], [20.0, 20.0], [10.0, 20.0]]
    assert _tool().main(["--source", str(shot), "--polygon", POLY, "--out", str(out), "--no-seed",
                         "--keep-source", "--no-reference-image"]) == 0
    new = [[200.0, 150.0], [440.0, 150.0], [440.0, 270.0], [200.0, 270.0]]
    assert "polygon_file" not in yaml.safe_load(out.read_text(encoding="utf-8"))["zone"]
    assert zone_polygon(load_config(str(out)).zone) == new  # what the monitor will watch
    assert "zone.polygon_file" in capsys.readouterr().out and old_zone.exists()  # said so; the file is not deleted


def test_a_full_default_site_file_from_0_2_0_is_refused_with_the_reason(tmp_path, shot):
    """0.2.0's calibrate_zone froze every default into the site file (final review Q2): re-running
    the tool on one stops before anything is written and says what to delete."""
    out = tmp_path / "site.yaml"
    out.write_text("video:\n  source: rtsp://cam/1\nstuck:\n  warning_seconds: 3.0\n  alarm_seconds: 8.0\n"
                   "debris:\n  warmup_frames: 300\n  empty_seconds_before_refresh: 5.0\n  empty_learning_rate: 0.005\n",
                   encoding="utf-8")
    before = out.read_text(encoding="utf-8")
    proc = subprocess.run([sys.executable, str(TOOL), "--source", str(shot), "--polygon", POLY, "--out", str(out),
                           "--no-seed"], capture_output=True, text=True, timeout=120,
                          env=dict(os.environ, PYTHONIOENCODING="utf-8"), cwd=str(tmp_path))
    assert proc.returncode == 2, proc.stderr
    (err,) = proc.stderr.strip().splitlines()
    for fragment in ("debris.empty_seconds_before_refresh", "debris.empty_learning_rate", "stuck.alarm_seconds 8.0",
                     "gradecrossing 0.2.0", "configs/legacy_rule.yaml"):
        assert fragment in err, fragment
    assert out.read_text(encoding="utf-8") == before


def test_a_base_config_contributes_its_keys_only(tmp_path, shot):
    base = tmp_path / "profile.yaml"
    base.write_text("detector:\n  imgsz: 960\ndebris:\n  enabled: false\n", encoding="utf-8")
    out = tmp_path / "site.yaml"
    assert _tool().main(["--source", str(shot), "--polygon", POLY, "--out", str(out), "--config", str(base),
                         "--no-seed", "--no-reference-image"]) == 0
    data = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert data["detector"] == {"imgsz": 960} and data["debris"] == {"enabled": False}
    assert data["video"] == {"source": str(shot)} and sorted(data["zone"]) == ["polygon", "reference_size"]


@pytest.mark.parametrize("content, needle", [
    ("zone:\n  polygon: [[0, 0], [1, 0]\n", "while parsing"),          # YAML syntax error
    ("zone:\n\tpolygon: []\n", "cannot start any token"),              # a tab in the indentation
    ("zone:\n  polygonn: [[0, 0], [1, 0], [1, 1]]\n", "unknown config key 'zone.polygonn'"),
    ("- zone\n", "mapping"),                                          # top level is not a mapping
])
def test_a_broken_base_config_exits_2_with_one_line_and_writes_nothing(tmp_path, shot, content, needle):
    base = tmp_path / "base.yaml"
    base.write_text(content, encoding="utf-8")
    out = tmp_path / "site.yaml"
    proc = subprocess.run([sys.executable, str(TOOL), "--source", str(shot), "--polygon", POLY, "--out", str(out),
                           "--config", str(base), "--no-seed"], capture_output=True, text=True, timeout=120,
                          env=dict(os.environ, PYTHONIOENCODING="utf-8"), cwd=str(tmp_path))
    assert proc.returncode == 2, proc.stderr
    err = proc.stderr.strip().splitlines()
    assert len(err) == 1 and err[0].startswith("error: cannot load config") and needle in err[0], err
    assert proc.stdout == ""  # failed before the frame was even read
    assert not out.exists() and not (tmp_path / "site_reference.png").exists()


def test_a_broken_existing_out_and_a_missing_config_exit_2(tmp_path, shot, capsys):
    tool = _tool()
    out = tmp_path / "site.yaml"
    out.write_text("stuck: [1, 2]\n", encoding="utf-8")
    assert tool.main(["--source", str(shot), "--polygon", POLY, "--out", str(out), "--no-seed"]) == 2
    assert "must be a mapping" in capsys.readouterr().err
    assert out.read_text(encoding="utf-8") == "stuck: [1, 2]\n"  # left alone
    assert tool.main(["--source", str(shot), "--polygon", POLY, "--out", str(tmp_path / "new.yaml"),
                      "--config", str(tmp_path / "missing.yaml"), "--no-seed"]) == 2
    assert "missing.yaml" in capsys.readouterr().err


def test_headless_paths_and_usage_errors(tmp_path, shot, capsys):
    tool = _tool()
    out = tmp_path / "site.yaml"
    assert tool.main(["--source", str(shot), "--headless", "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "{}x{}".format(W, H) in printed and "headless: nothing written" in printed and printed.isascii()
    assert not out.exists()
    assert tool.main(["--source", str(shot), "--polygon", "1,2 3", "--out", str(out), "--no-seed"]) == 2
    assert tool.main(["--source", str(shot), "--polygon", "1,1 2,2", "--out", str(out), "--no-seed"]) == 2
    assert "bad --polygon" in capsys.readouterr().err
    assert tool.main(["--source", str(tmp_path / "missing.png"), "--polygon", POLY, "--out", str(out)]) == 1
    assert not out.exists()
    frame_out = tmp_path / "frame.jpg"
    assert tool.main(["--source", str(shot), "--headless", "--save-frame", str(frame_out)]) == 0
    assert cv2.imread(str(frame_out)).shape == (H, W, 3)
