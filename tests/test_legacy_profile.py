"""The rollback profile configs/legacy_rule.yaml restores the pre-port vehicle rule exactly.

The YOLO-geometry port changed several defaults at once (footprint test, ID-free
stationarity, the too-close WARNING, the exit debounce, 1.5 s / 3 s timing, and the
robustness switches); "zone_test: bottom_center restores the old rule" is only true when all
of them are set back. Two checks:

(a) the profile sets every behavioural key that is new or changed since ae66b8f, so it stays
    complete when a key is added (add the key to the profile, or to INERT with the reason);
(b) the test_monitor.py scenarios A, B, C, E and the five pre-port tracker scenarios, replayed
    with the profile, match goldens recorded once from ae66b8f
    (tools/record_legacy_goldens.py -> tests/data/legacy_events.json): every event's type, time
    (1e-6), frame and status; START events' track ids and reasons; every frame's status; every
    tracker frame's in_zone / dwell / stuck / stationary / speed. END events are compared by
    type and time only: their payload deliberately changed (they now name the episode's ids
    and opening reason instead of the previous frame's raw ids), as did ALARM events' reasons,
    which keep warning-level reasons apart in extra['warning_reasons'] (re-joined here).
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest
import yaml

from gradecrossing.config import MonitorConfig, load_config

import legacy_scenarios  # noqa: E402  (tests/ is on sys.path, like test_monitor for test_port_scenarios)

ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = ROOT / "configs" / "legacy_rule.yaml"
PROFILE = yaml.safe_load(PROFILE_PATH.read_text(encoding="utf-8"))
GOLDENS = json.loads((ROOT / "tests" / "data" / "legacy_events.json").read_text(encoding="utf-8"))

# New stuck keys that cannot change anything once the profile's switches are set.
INERT = {
    "footprint_fraction": "only used by zone_test='footprint'",
    "footprint_min_overlap": "only used by zone_test='footprint'",
    "close_box_heights": "'near' needs zone_test='footprint'; close_policy is off",
    "still_min_iou": "only used when stationarity is not 'speed'",
    "still_max_gap_seconds": "only used when stationarity is not 'speed'",
    "still_border_margin": "only used when stationarity is not 'speed'",
}
# Keys outside the stuck section that the port added (other modules own the rest of those sections).
PORT_KEYS = {"alerts": ["warning_hold_seconds"], "detector": ["track_conf"]}


def test_profile_sets_every_behavioural_key_the_port_changed():
    old = GOLDENS["defaults"]
    new = dataclasses.asdict(MonitorConfig())
    changed = sorted(k for k, v in new["stuck"].items() if k not in old["stuck"] or old["stuck"][k] != v)
    assert {"zone_test", "stationarity", "close_policy", "alarm_seconds"} <= set(changed)  # sanity
    missing = [k for k in changed if k not in PROFILE["stuck"] and k not in INERT]
    assert not missing, f"configs/legacy_rule.yaml does not roll back {missing} (or list them in INERT)"
    for key, value in PROFILE["stuck"].items():
        if key in old["stuck"]:
            assert value == old["stuck"][key], key  # a pre-existing key goes back to its ae66b8f default
    for section, keys in PORT_KEYS.items():
        assert set(keys) <= set(PROFILE[section]), section
    assert PROFILE["detector"]["track_conf"] is None  # = conf: ByteTrack gets what it got
    assert (old["stuck"]["warning_seconds"], old["stuck"]["alarm_seconds"]) == (3.0, 8.0)
    cfg = load_config(str(PROFILE_PATH))
    assert (cfg.stuck.warning_seconds, cfg.stuck.alarm_seconds) == (3.0, 8.0)
    assert (cfg.stuck.zone_test, cfg.stuck.stationarity, cfg.stuck.close_policy) == ("bottom_center", "speed", "off")
    assert cfg.alerts.warning_hold_seconds == 0.0 and cfg.stuck.occlusion_memory_seconds == 0.0


@pytest.mark.parametrize("name", sorted(legacy_scenarios.monitor_scenarios()))
def test_monitor_scenarios_match_ae66b8f(name):
    got = legacy_scenarios.run_monitor_scenario(name, PROFILE)
    want = GOLDENS["scenarios"]["monitor"][name]
    assert got["statuses"] == want["statuses"]
    assert [e["type"] for e in got["events"]] == [e["type"] for e in want["events"]]
    for g, w in zip(got["events"], want["events"]):
        assert g["timestamp"] == pytest.approx(w["timestamp"], abs=1e-6)
        assert (g["frame_index"], g["status"]) == (w["frame_index"], w["status"])
        assert bool(g["extra"].get("repeat")) == bool(w["extra"].get("repeat"))
        if not g["type"].endswith("_END"):
            assert g["track_ids"] == w["track_ids"]
            assert g["reasons"] + g["extra"].get("warning_reasons", []) == w["reasons"]


@pytest.mark.parametrize("name", legacy_scenarios.TRACKER_SCENARIOS)
def test_tracker_scenarios_match_ae66b8f(name):
    got = legacy_scenarios.run_tracker_scenario(name, PROFILE["stuck"])
    want = GOLDENS["scenarios"]["tracker"][name]
    assert len(got) == len(want)
    for i, (g, w) in enumerate(zip(got, want)):
        assert (g["occupied"], g["empty_since"]) == (w["occupied"], w["empty_since"]), i
        assert [o["id"] for o in g["objects"]] == [o["id"] for o in w["objects"]], i
        for go, wo in zip(g["objects"], w["objects"]):
            assert (go["in_zone"], go["units"]) == (wo["in_zone"], wo["units"]), i
            for key in ("dwell", "stuck", "stationary", "speed"):
                assert go[key] == pytest.approx(wo[key], abs=1e-6), (i, key)


def test_team_rule_profile_loads():
    """configs/team_rule.yaml (approximation of the advisor's merged rule, for comparisons) only
    sets existing keys."""
    cfg = load_config(str(ROOT / "configs" / "team_rule.yaml"))
    assert (cfg.stuck.stationarity, cfg.stuck.close_policy, cfg.stuck.alarm_seconds) == ("still", "alarm", 3.0)
    assert (cfg.detector.imgsz, cfg.detector.iou, cfg.detector.detect_every_n_frames) == (960, 0.7, 6)
    assert cfg.debris.enabled is False


def test_the_defaults_are_not_the_legacy_rule():
    """The same scenarios at the shipped defaults differ (otherwise the check above proves nothing)."""
    got = legacy_scenarios.run_tracker_scenario("enters_then_stops", None)
    want = GOLDENS["scenarios"]["tracker"]["enters_then_stops"]
    assert any(abs(g["objects"][0]["stuck"] - w["objects"][0]["stuck"]) > 0.1 for g, w in zip(got, want))
