"""Config values are checked against their dataclass field types (F09).

Before, ``_apply`` stored whatever the YAML held: ``detector: {vehicle_classes: car}``
loaded as the string 'car', the class sets were built from its characters, every vehicle
became 'other' and a car parked on the crossing for a minute raised nothing. Every value
is now checked when a config is loaded and again when a ``Monitor`` is built (so a config
assembled in code gets the same checks); a mistake is a ``ValueError`` naming the dotted
key, the expected type and the value, which the CLI reports with exit status 2.
"""
from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest

from gradecrossing.config import (
    RETIRED_KEYS,
    ConfigError,
    MonitorConfig,
    config_from_dict,
    load_config,
    validate_config,
)
from gradecrossing.monitor import Monitor, main

ROOT = Path(__file__).resolve().parents[1]
SHIPPED = sorted((ROOT / "configs").glob("*.yaml"))


class NoDetections:
    def detect(self, frame):
        return []


def rejects(overrides, *fragments):
    """config_from_dict(overrides) must raise a ValueError whose message has every fragment."""
    with pytest.raises(ValueError) as info:
        config_from_dict(overrides)
    for fragment in fragments:
        assert fragment in str(info.value), (fragment, str(info.value))
    return info.value


# ----------------------------------------------------------------------------- what loads
def test_defaults_validate():
    cfg = MonitorConfig()
    assert validate_config(cfg) is cfg


@pytest.mark.parametrize("path", SHIPPED, ids=[p.name for p in SHIPPED])
def test_every_shipped_config_loads(path):
    assert isinstance(load_config(str(path)), MonitorConfig)


def test_ints_are_floats_and_null_is_none_where_optional():
    cfg = config_from_dict({"stuck": {"alarm_seconds": 3}, "video": {"fps_override": None, "max_frames": None},
                            "zone": {"reference_size": [1280, 720]}})
    assert cfg.stuck.alarm_seconds == 3 and cfg.video.fps_override is None and cfg.zone.reference_size == [1280, 720]
    homography = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    assert config_from_dict({"stuck": {"ground_plane_homography": homography}}).stuck.ground_plane_homography == homography


def test_code_may_use_tuples_and_numpy_arrays_for_lists():
    cfg = MonitorConfig()
    cfg.zone.polygon = np.array([[0, 0], [10, 0], [10, 10]], dtype=np.float32)
    cfg.zone.reference_size = (640, 360)
    cfg.debris.ignore_polygons = [[(0, 0), (5, 0), (5, 5)]]
    cfg.stuck.speed_threshold = np.float32(12.0)
    assert validate_config(cfg) is cfg


# ----------------------------------------------------------------------------- what does not
def test_a_scalar_class_list_is_rejected(tmp_path, capsys):
    """F09: the one-line slip that silenced every stuck-vehicle alarm."""
    rejects({"detector": {"vehicle_classes": "car"}},
            "config key detector.vehicle_classes: expected list[str], got 'car' (str)", "YAML list")
    bad = tmp_path / "bad.yaml"
    bad.write_text("detector:\n  vehicle_classes: car, truck, bus\n", encoding="utf-8")
    with pytest.raises(ValueError, match="detector.vehicle_classes"):
        load_config(str(bad))
    assert main(["--config", str(bad), "--no-stdout"]) == 2
    assert "detector.vehicle_classes" in capsys.readouterr().err
    rejects({"detector": {"vehicle_classes": ["car", 3]}}, "detector.vehicle_classes[1]", "expected str", "3 (int)")


@pytest.mark.parametrize("section, key, value, expected, shown", [
    ("stuck", "alarm_seconds", "3", "stuck.alarm_seconds: expected float", "'3' (str)"),
    ("stuck", "alarm_seconds", True, "stuck.alarm_seconds: expected float", "True (bool)"),
    ("stuck", "min_track_age_frames", 2.5, "stuck.min_track_age_frames: expected int", "2.5 (float)"),
    ("stuck", "min_track_age_frames", False, "stuck.min_track_age_frames: expected int", "False (bool)"),
    ("zone", "stabilize", 1, "zone.stabilize: expected bool", "1 (int)"),
    ("zone", "stabilize", "yes", "zone.stabilize: expected bool", "'yes' (str)"),
    ("video", "source", 0, "video.source: expected str", "0 (int)"),
    ("video", "fps_override", "25", "video.fps_override: expected float or null", "'25' (str)"),
    ("zone", "reference_size", "1280x720", "zone.reference_size: expected list[int] or null", "'1280x720' (str)"),
    ("zone", "polygon", None, "zone.polygon: expected list[list[float]]", "None (NoneType)"),
    ("zone", "polygon_file", 3, "zone.polygon_file: expected str", "3 (int)"),
    ("debris", "ignore_polygons", [[0, 0], [1, 0], [1, 1]], "debris.ignore_polygons[0][0]: expected list[float]",
     "0 (int)"),
])
def test_wrong_types_name_the_key_the_type_and_the_value(section, key, value, expected, shown):
    exc = rejects({section: {key: value}}, "config key " + expected, "got " + shown)
    assert isinstance(exc, ConfigError) and isinstance(exc, TypeError)  # a TypeError too, as sections raised before


@pytest.mark.parametrize("key, value, fragment", [
    ("zone.polygon", [[0, 0], [1, 1]], "at least 3 [x, y] points, got 2"),
    ("zone.polygon", [[0, 0], [1, 1], [2]], "point 2 must be [x, y]"),
    ("zone.polygon", [[0, 0], [1, 1], ["a", 2]], "zone.polygon[2][0]: expected float"),
    ("zone.polygon", [[0, 0], [1, 1], [float("nan"), 2]], "point 2 is not finite"),
    ("debris.ignore_polygons", [[[0, 0], [1, 1]]], "debris.ignore_polygons[0]: a polygon needs at least 3"),
    ("zone.reference_size", [1280, 720, 3], "zone.reference_size: expected 2 values, got 3"),
    ("stuck.ground_plane_homography", [[1, 0], [0, 1]], "stuck.ground_plane_homography: expected 3 values"),
    ("stuck.ground_plane_homography", [[1, 0, 0], [0, 1], [0, 0, 1]],
     "stuck.ground_plane_homography[1]: expected 3 values"),
])
def test_polygons_and_fixed_length_lists_are_checked(key, value, fragment):
    section, name = key.split(".")
    rejects({section: {name: value}}, fragment)


def test_sections_must_be_mappings():
    rejects({"zone": 5}, "zone", "mapping")
    cfg = MonitorConfig()
    cfg.zone = {"polygon": [[0, 0], [1, 0], [1, 1]]}  # type: ignore[assignment]
    with pytest.raises(ValueError, match="config key zone: expected a mapping of ZoneConfig"):
        validate_config(cfg)
    with pytest.raises(TypeError):  # unchanged: a top level that is not a mapping
        config_from_dict(["video"])  # type: ignore[arg-type]


def test_unknown_keys_still_raise_and_retired_keys_say_what_replaced_them():
    """A typo is a KeyError listing the valid keys. A retired key is a ConfigError (exit 2 like any
    config error; it was a KeyError, one key per attempt, before final review Q2) that names every
    retired key the file sets at once, each with what replaced it."""
    with pytest.raises(KeyError, match="valid:"):
        config_from_dict({"video": {"sourc": "x"}})
    for dotted, hint in RETIRED_KEYS.items():
        section, key = dotted.split(".")
        assert key not in {f.name for f in fields(getattr(MonitorConfig(), section))}, dotted
        with pytest.raises(ConfigError) as info:
            config_from_dict({section: {key: 1.0}})
        assert dotted in str(info.value) and hint in str(info.value)
    assert {"debris.empty_seconds_before_refresh", "debris.empty_learning_rate"} <= set(RETIRED_KEYS)
    both = {"debris": {"empty_seconds_before_refresh": 5.0, "empty_learning_rate": 0.005}, "video": {"sourc": "x"}}
    with pytest.raises(ConfigError) as info:  # all retired keys first, in one message
        config_from_dict(both)
    assert str(info.value).startswith("retired config keys (delete them): ")
    assert all(dotted in str(info.value) for dotted in RETIRED_KEYS)


def test_the_monitor_checks_a_config_built_in_code(tmp_path):
    """Programmatic configs get the same checks, before any sink, frame or model."""
    cfg = MonitorConfig()
    cfg.alerts.stdout = False
    cfg.alerts.events_jsonl = str(tmp_path / "events.jsonl")
    cfg.detector.vehicle_classes = "car"  # type: ignore[assignment]
    with pytest.raises(ValueError, match="detector.vehicle_classes"):
        Monitor(cfg, detector=NoDetections())
    assert not (tmp_path / "events.jsonl").exists()  # failed before the JSONL sink was opened
    cfg.detector.vehicle_classes = ["car"]
    cfg.stuck.alarm_seconds = "3"  # type: ignore[assignment]
    with pytest.raises(ValueError, match="stuck.alarm_seconds"):
        Monitor(cfg, detector=NoDetections())
