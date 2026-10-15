"""configs/default.yaml is the complete reference, and a 0.2.0 site file cannot keep the old rule silently.

Two things that loading a config cannot show by itself:

* default.yaml's header promises every key with its dataclass default. ``load_config`` merges a
  file onto the defaults, so comparing loaded values (tests/test_monitor.py) passes even when a key
  is missing from the file; here the key sets are compared (final review Q7).
* tools/calibrate_zone.py of gradecrossing 0.2.0 (ae66b8f) wrote every default into each site file.
  Such a file names two keys that no longer exist and pins three defaults that have changed since:
  stuck.warning_seconds 3.0, stuck.alarm_seconds 8.0 and debris.warmup_frames 300 (10 s at 30 fps),
  so the 3 s ALARM would never reach a site calibrated with the old tool. Loaded today it fails
  once, naming every retired key and the pinned rule; with the retired keys deleted it loads, with
  one warning per load naming each pinned key, its old and today's default and
  configs/legacy_rule.yaml (final review Q2).
"""
from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Any, Dict

import pytest
import yaml

from gradecrossing.config import (
    CHANGED_DEFAULTS,
    RETIRED_KEYS,
    ConfigError,
    MonitorConfig,
    config_from_dict,
    load_config,
    stale_defaults,
)

ROOT = Path(__file__).resolve().parents[1]

# MonitorConfig().to_dict() of gradecrossing 0.2.0 (git show ae66b8f:gradecrossing/config.py): what
# that release's tools/calibrate_zone.py (save_config) wrote into every site file, and what its
# configs/default.yaml listed.
DEFAULTS_0_2_0: Dict[str, Dict[str, Any]] = {
    "video": {
        "source": "0", "fps_override": None, "resize_width": None, "start_seconds": 0.0, "max_frames": None,
        "loop": False, "live": None,
    },
    "zone": {
        "polygon": [[100, 400], [500, 350], [1650, 480], [1500, 600]], "stabilize": True, "max_corners": 400,
        "quality_level": 0.01, "min_distance": 20, "block_size": 7, "lk_win_size": 21, "lk_max_level": 3,
        "ransac_reproj_threshold": 3.0, "min_inliers": 30, "min_inlier_fraction": 0.2, "orb_fallback": True,
        "orb_features": 1500, "orb_ratio_test": 0.75, "orb_min_matches": 25, "max_scale_change": 0.35,
        "max_area_change": 0.5, "keyframe_refresh_static_seconds": 60.0, "exclude_features_in_zone": True,
        "health_events": True,
    },
    "detector": {
        "weights": "yolo11n.pt", "conf": 0.25, "iou": 0.5, "imgsz": 640, "device": "", "half": False,
        "tracker": "bytetrack.yaml", "detect_every_n_frames": 1,
        "vehicle_classes": ["car", "truck", "bus", "motorcycle", "motorbike", "jeep", "van", "tricycle", "bicycle"],
        "person_classes": ["person", "pedestrian"], "train_classes": ["train"], "max_det": 100,
    },
    "infrastructure": {
        "enabled": False, "weights": "notebooks/infrastructure tuning best weights/best.pt", "conf": 0.35,
        "imgsz": 640, "device": "", "half": False, "every_n_frames": 15,
        "gate_down_classes": ["railroad-crossing-down"], "light_on_classes": ["railroad-sign-light-on"],
        "hold_seconds": 4.0,
    },
    "stuck": {
        "speed_threshold": 12.0, "speed_window_seconds": 1.0, "min_history_for_speed": 3,
        "ground_plane_homography": None, "warning_seconds": 3.0, "alarm_seconds": 8.0,
        "dwell_warning_seconds": 12.0, "dwell_alarm_seconds": 25.0, "gates_down_warning_seconds": 1.0,
        "gates_down_alarm_seconds": 3.0, "gates_down_dwell_alarm_seconds": 6.0, "lost_track_grace_seconds": 1.0,
        "min_track_age_frames": 2, "fallback_iou_threshold": 0.3, "person_warning_seconds": 2.0,
        "person_alarm_seconds": 10.0,
    },
    "debris": {
        "enabled": True, "history": 300, "var_threshold": 50.0, "detect_shadows": True, "warmup_frames": 300,
        "monitoring_learning_rate": 0.0, "empty_seconds_before_refresh": 5.0, "empty_learning_rate": 0.005,
        "exclusion_dilate_px": 12, "morph_kernel": 5, "min_blob_area_fraction": 0.002, "min_area_fraction": 0.03,
        "persist_seconds": 3.0, "max_area_fraction": 0.75, "static_motion_diff": 12.0,
        "static_motion_fraction": 0.15,
    },
    "alerts": {
        "latch_seconds": 10.0, "repeat_notify_seconds": 30.0, "stdout": True, "events_jsonl": "", "webhook_url": "",
        "webhook_timeout_seconds": 3.0,
    },
    "output": {"display": False, "save_video": "", "snapshot_dir": "", "log_level": "INFO", "print_every_n_frames": 0},
}


def flat(data: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Nested config mapping -> {dotted key: value} (no config value is itself a mapping)."""
    out: Dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, dict):
            out.update(flat(value, f"{prefix}{key}."))
        else:
            out[f"{prefix}{key}"] = value
    return out


def same(a: Any, b: Any) -> bool:
    """Equal as YAML values: 8 == 8.0, but true is not 1."""
    return isinstance(a, bool) == isinstance(b, bool) and a == b


def write(tmp_path: Path, data: Dict[str, Any], name: str = "site.yaml") -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def site_file_0_2_0() -> Dict[str, Any]:
    """A site file as 0.2.0's calibrate_zone wrote it: every default, plus the site's source and polygon."""
    data = copy.deepcopy(DEFAULTS_0_2_0)
    data["video"]["source"] = "rtsp://camera/stream"
    data["zone"]["polygon"] = [[10.0, 200.0], [600.0, 210.0], [620.0, 300.0], [5.0, 290.0]]
    return data


def config_warnings(caplog) -> list:
    return [r.getMessage() for r in caplog.records if r.name == "gradecrossing.config" and r.levelno >= logging.WARNING]


def stale_fragment(key: str) -> str:
    """How the messages describe a key pinned at its 0.2.0 default: '<key> <old> (the 0.2.0 default; now <new>)'."""
    old, release = CHANGED_DEFAULTS[key]
    now = flat(MonitorConfig().to_dict())[key]
    return f"{key} {json.dumps(old)} (the {release} default; now {json.dumps(now)})"


# ----------------------------------------------------------------------------- default.yaml (Q7)
def test_default_yaml_lists_every_key_with_its_default():
    """A key added to a dataclass but not to configs/default.yaml (or left there after removal) fails
    here; loading the file cannot show it, because load_config fills the gaps with the defaults."""
    listed = flat(yaml.safe_load((ROOT / "configs" / "default.yaml").read_text(encoding="utf-8")))
    defaults = flat(MonitorConfig().to_dict())
    assert sorted(set(defaults) - set(listed)) == [], "keys missing from configs/default.yaml"
    assert sorted(set(listed) - set(defaults)) == [], "keys in configs/default.yaml that the config does not have"
    assert {k: v for k, v in listed.items() if not same(v, defaults[k])} == {}, "values that are not the default"


# ----------------------------------------------------------------------------- 0.2.0 site files (Q2)
def test_changed_defaults_is_the_diff_against_0_2_0():
    """Every 0.2.0 key still exists, is retired (RETIRED_KEYS) or, when its default changed, has a
    CHANGED_DEFAULTS row with its 0.2.0 value; every row is still a change. Changing a 0.2.0 default
    without adding a row fails here."""
    old, now = flat(DEFAULTS_0_2_0), flat(MonitorConfig().to_dict())
    changed = {}
    for key, value in old.items():
        if key in RETIRED_KEYS:
            assert key not in now, key
            continue
        assert key in now, f"{key} was removed without a RETIRED_KEYS row"
        if not same(now[key], value):
            changed[key] = value
    assert {k: v for k, (v, release) in CHANGED_DEFAULTS.items() if release == "0.2.0"} == changed
    assert set(changed) == {"stuck.warning_seconds", "stuck.alarm_seconds", "debris.warmup_frames"}
    for key, (value, _release) in CHANGED_DEFAULTS.items():
        assert key in now and not same(now[key], value), f"CHANGED_DEFAULTS row {key} is not a change any more"


def test_a_full_default_site_file_from_0_2_0_fails_once_and_says_why(tmp_path, capsys):
    path = write(tmp_path, site_file_0_2_0())
    with pytest.raises(ConfigError) as info:
        load_config(str(path))
    message = str(info.value)
    for key, hint in RETIRED_KEYS.items():  # every retired key in the one error, not one per attempt
        assert f"'{key}' ({hint})" in message, key
    for key in CHANGED_DEFAULTS:
        assert stale_fragment(key) in message, key
    for fragment in ("full-default site file of gradecrossing 0.2.0", "tools/calibrate_zone.py",
                     "configs/legacy_rule.yaml"):
        assert fragment in message, fragment
    from gradecrossing.monitor import main  # the CLI: exit 2, one line, no traceback

    assert main(["--config", str(path), "--no-stdout"]) == 2
    err = capsys.readouterr().err.strip().splitlines()
    assert len(err) == 1 and err[0].startswith("error: cannot load config"), err
    assert all(key in err[0] for key in RETIRED_KEYS) and "stuck.alarm_seconds 8.0" in err[0]


def test_without_the_retired_keys_it_loads_its_values_with_one_warning(tmp_path, caplog):
    data = site_file_0_2_0()
    for key in RETIRED_KEYS:
        section, name = key.split(".")
        del data[section][name]
    path = write(tmp_path, data)
    with caplog.at_level(logging.WARNING, logger="gradecrossing.config"):
        cfg = load_config(str(path))
    # the file's values are used: a warning, not a silent change and not a failed start
    assert (cfg.stuck.warning_seconds, cfg.stuck.alarm_seconds, cfg.debris.warmup_frames) == (3.0, 8.0, 300)
    (warning,) = config_warnings(caplog)  # one warning per load, every pinned key in it
    for key in CHANGED_DEFAULTS:
        assert stale_fragment(key) in warning, key
    assert path.name in warning and "configs/legacy_rule.yaml" in warning
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="gradecrossing.config"):
        config_from_dict(data)  # a mapping built in code is not a file: no warning
    assert config_warnings(caplog) == []


def test_only_a_value_equal_to_the_old_default_is_stale(tmp_path, caplog):
    assert stale_defaults({"stuck": {"alarm_seconds": 8}}) == [
        ("stuck.alarm_seconds", 8, MonitorConfig().stuck.alarm_seconds, "0.2.0")]  # 8 is 8.0 in YAML
    assert stale_defaults({"stuck": {"alarm_seconds": 5.0, "warning_seconds": MonitorConfig().stuck.warning_seconds},
                           "debris": {"warmup_frames": None}, "video": None}) == []
    assert stale_defaults(None) == [] and stale_defaults(["stuck"]) == []
    path = write(tmp_path, {"stuck": {"alarm_seconds": 5.0}, "debris": {"warmup_frames": 60}})
    with caplog.at_level(logging.WARNING, logger="gradecrossing.config"):
        load_config(str(path))
    assert config_warnings(caplog) == []


def test_a_retired_key_alone_is_named_without_the_0_2_0_story():
    with pytest.raises(ConfigError) as info:
        config_from_dict({"debris": {"empty_learning_rate": 0.01}, "stuck": {"alarm_seconds": 3.0}})
    message = str(info.value)
    assert message.startswith("retired config key (delete it): 'debris.empty_learning_rate' (removed: ")
    assert "full-default" not in message


def test_the_legacy_profile_loads_and_its_warning_says_it_is_on_purpose(caplog):
    """configs/legacy_rule.yaml sets the 0.2.0 WARNING/ALARM times on purpose: it loads with them,
    and the warning names the profile as the deliberate way to keep that rule."""
    with caplog.at_level(logging.WARNING, logger="gradecrossing.config"):
        cfg = load_config(str(ROOT / "configs" / "legacy_rule.yaml"))
    assert (cfg.stuck.warning_seconds, cfg.stuck.alarm_seconds) == (3.0, 8.0)
    (warning,) = config_warnings(caplog)
    assert stale_fragment("stuck.alarm_seconds") in warning
    assert "configs/legacy_rule.yaml is the profile that restores the 0.2.0 vehicle rule on purpose" in warning
