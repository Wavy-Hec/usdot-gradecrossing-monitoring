"""``python -m gradecrossing.monitor`` and ``tools/benchmark.py`` as an operator meets them.

Exit status: 0 = the input ended normally or the run was stopped (Ctrl-C, SIGTERM), 1 = the
crossing stopped being watched or watching never started, 2 = configuration or usage error (the
DECISIONS of the run-loop work; ours-cleanup OC-01, ours-correctness F08). Captures are
faked as in ``test_monitor_run``; where a detector is needed ``monitor.Monitor`` is
replaced by one with an injected fake, so nothing imports ultralytics. Also: the one CLI
override table (OC-19) and the benchmark's separate debris warm-up (ours-correctness M4).
"""
from __future__ import annotations

import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, List

import pytest

import gradecrossing.monitor as monitor_mod
from gradecrossing.config import MonitorConfig, save_config
from gradecrossing.monitor import (
    CLI_OVERRIDES,
    EXIT_CONFIG,
    EXIT_NOT_WATCHING,
    EXIT_OK,
    Monitor,
    apply_cli_overrides,
    build_parser,
    format_summary,
    main,
)

from test_monitor import FakeDetector, car_det, no_cars  # noqa: E402
from test_monitor_run import (  # noqa: E402,F401 - sleeps is a fixture
    BrokenDetector,
    FakeCapture,
    InterruptingDetector,
    file_cfg,
    live_cfg,
    patch_captures,
    sleeps,
)

ROOT = Path(__file__).resolve().parents[1]


def config_file(tmp_path: Path, cfg: MonitorConfig) -> str:
    cfg.alerts.stdout = False
    path = tmp_path / "site.yaml"
    save_config(cfg, str(path))
    return str(path)


def inject(monkeypatch, make_detector: Callable[[], Any]) -> None:
    """main() builds Monitor(cfg): give it a fake detector instead of YOLO."""
    monkeypatch.setattr(monitor_mod, "Monitor", lambda c: Monitor(c, detector=make_detector()))


def error_lines(err: str):
    return [ln for ln in err.splitlines() if ln.startswith("error:")]


# ----------------------------------------------------------------------------- exit 0
def test_exit_0_when_the_input_ends(monkeypatch, tmp_path, capsys):
    patch_captures(monkeypatch, [FakeCapture(n=8)])
    inject(monkeypatch, lambda: FakeDetector(no_cars))
    assert main(["--config", config_file(tmp_path, file_cfg())]) == EXIT_OK
    out = capsys.readouterr().out
    assert "summary: frames=8" in out and "ABORTED" not in out


def test_exit_0_when_the_input_ends_during_an_alarm(monkeypatch, tmp_path):
    patch_captures(monkeypatch, [FakeCapture(n=60)])
    inject(monkeypatch, lambda: FakeDetector(lambda i: [car_det(320.0, 230.0)]))
    assert main(["--config", config_file(tmp_path, file_cfg())]) == EXIT_OK


def test_exit_0_on_ctrl_c(monkeypatch, tmp_path):
    patch_captures(monkeypatch, [FakeCapture(n=30)])
    inject(monkeypatch, lambda: InterruptingDetector(no_cars))
    assert main(["--config", config_file(tmp_path, file_cfg())]) == EXIT_OK


# ----------------------------------------------------------------------------- FO4: SIGTERM / SIGHUP
def read_events(path: Path) -> List[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def complete_lines(path: Path) -> int:
    """Lines another process has finished writing to ``path`` (0 while it does not exist)."""
    return path.read_text(encoding="utf-8").count("\n") if path.exists() else 0


def test_stop_signals_raise_keyboard_interrupt_while_main_runs():
    """The handler logic (os.kill(SIGTERM) is TerminateProcess on Windows: nothing to catch
    there, so the real-signal test below is POSIX only; the Jetson is Linux)."""
    before = signal.getsignal(signal.SIGTERM)
    installed = monitor_mod._install_stop_handlers()
    try:
        names = {signal.Signals(s).name for s, _ in installed}
        assert "SIGTERM" in names and ("SIGHUP" in names) == hasattr(signal, "SIGHUP")
        assert signal.getsignal(signal.SIGTERM) is monitor_mod._stop_on_signal
        with pytest.raises(KeyboardInterrupt, match="SIGTERM"):
            monitor_mod._stop_on_signal(signal.SIGTERM, None)
    finally:
        monitor_mod._restore_handlers(installed)
    assert signal.getsignal(signal.SIGTERM) == before


def test_exit_0_and_run_end_on_sigterm(monkeypatch, tmp_path):
    """FO4 (sigterm_probe.py): systemctl stop / docker stop send SIGTERM, which killed the
    process without run()'s finally - no RUN_END, an open WARNING never closed, queued webhook
    events lost. main() now stops the run like Ctrl-C: RUN_END 'interrupted', exit 0."""

    class SigtermDetector(FakeDetector):
        def detect(self, frame_bgr):
            if len(self.calls) == 45:  # 1.5 s in: the car has a WARNING open
                signal.raise_signal(signal.SIGTERM)
            return super().detect(frame_bgr)

    before = signal.getsignal(signal.SIGTERM)
    patch_captures(monkeypatch, [FakeCapture(n=300)])
    inject(monkeypatch, lambda: SigtermDetector(lambda i: [car_det(320.0, 230.0)]))
    log = tmp_path / "events.jsonl"
    cfg = file_cfg(stuck={"alarm_seconds": 30.0})
    assert main(["--config", config_file(tmp_path, cfg), "--events", str(log)]) == EXIT_OK
    rows = read_events(log)
    assert rows[-1]["type"] == "RUN_END" and rows[-1]["extra"]["stop_reason"] == "interrupted"
    assert rows[-1]["extra"]["final_status"] == "WARNING" and rows[-1]["extra"]["frames"] < 50
    assert signal.getsignal(signal.SIGTERM) == before  # main() put the handler back


SIGTERM_CHILD = """
import sys, time
sys.path[:0] = [{tests!r}, {root!r}]
import gradecrossing.monitor as m
from test_monitor import FakeDetector, no_cars
from test_monitor_run import FakeCapture

class SlowCapture(FakeCapture):
    def read(self):
        time.sleep(0.01)
        return super().read()

m._make_capture = lambda source, timeout_s=0.0: SlowCapture(n=100000)
Monitor = m.Monitor
m.Monitor = lambda cfg: Monitor(cfg, detector=FakeDetector(no_cars))
sys.exit(m.main(sys.argv[1:]))
"""


@pytest.mark.skipif(sys.platform == "win32", reason="os.kill(SIGTERM) is TerminateProcess on Windows")
def test_real_sigterm_from_a_service_manager(tmp_path):
    script = tmp_path / "child.py"
    script.write_text(SIGTERM_CHILD.format(tests=str(ROOT / "tests"), root=str(ROOT)), encoding="utf-8")
    log = tmp_path / "events.jsonl"
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    proc = subprocess.Popen([sys.executable, str(script), "--config", config_file(tmp_path, file_cfg()),
                             "--events", str(log)], cwd=str(tmp_path), env=env)
    try:
        deadline = time.monotonic() + 60.0
        while time.monotonic() < deadline and complete_lines(log) < 3:  # RUN_START and the debris warm-up
            time.sleep(0.1)
        proc.send_signal(signal.SIGTERM)
        assert proc.wait(timeout=60) == EXIT_OK
    finally:
        if proc.poll() is None:
            proc.kill()
    rows = read_events(log)
    assert rows[0]["type"] == "RUN_START"
    assert rows[-1]["type"] == "RUN_END" and rows[-1]["extra"]["stop_reason"] == "interrupted"


# ----------------------------------------------------------------------------- exit 1
def test_exit_1_when_the_weights_are_missing(monkeypatch, tmp_path, capsys):
    patch_captures(monkeypatch, [FakeCapture(n=30)])
    cfg = file_cfg(detector={"weights": str(tmp_path / "models" / "best_vehicle_typo.pt")})
    assert main(["--config", config_file(tmp_path, cfg)]) == EXIT_NOT_WATCHING
    err = capsys.readouterr().err
    assert len(error_lines(err)) == 1 and "weights file not found" in err and "Traceback" not in err


def test_exit_1_without_ultralytics(monkeypatch, tmp_path, capsys):
    patch_captures(monkeypatch, [FakeCapture(n=30)])
    weights = tmp_path / "w.pt"
    weights.write_bytes(b"not a model")
    monkeypatch.setitem(sys.modules, "ultralytics", None)  # import ultralytics -> ImportError
    assert main(["--config", config_file(tmp_path, file_cfg(detector={"weights": str(weights)}))]) == EXIT_NOT_WATCHING
    assert "ultralytics is required" in capsys.readouterr().err


def test_exit_1_when_the_source_cannot_be_opened(monkeypatch, tmp_path, capsys):
    patch_captures(monkeypatch, [FakeCapture(opened=False)])
    assert main(["--config", config_file(tmp_path, file_cfg())]) == EXIT_NOT_WATCHING
    assert "cannot open video source" in capsys.readouterr().err


def test_exit_1_after_repeated_frame_errors(monkeypatch, tmp_path, capsys):
    patch_captures(monkeypatch, [FakeCapture(n=30)])
    inject(monkeypatch, lambda: BrokenDetector(no_cars))
    assert main(["--config", config_file(tmp_path, file_cfg())]) == EXIT_NOT_WATCHING
    assert "ABORTED stop_reason=frame_errors" in capsys.readouterr().out


def test_exit_1_when_a_live_source_cannot_be_reconnected(monkeypatch, tmp_path, capsys, sleeps):
    patch_captures(monkeypatch, [FakeCapture(n=5)])
    inject(monkeypatch, lambda: FakeDetector(no_cars))
    assert main(["--config", config_file(tmp_path, live_cfg())]) == EXIT_NOT_WATCHING
    assert "ABORTED stop_reason=reconnect_failed" in capsys.readouterr().out


@pytest.mark.parametrize("section, key, leaf", [("output", "snapshot_dir", "snaps"),
                                                ("output", "save_video", "out/v.mp4"),
                                                ("alerts", "events_jsonl", "logs/events.jsonl")])
def test_exit_1_when_an_output_path_cannot_be_created(monkeypatch, tmp_path, capsys, section, key, leaf):
    """MISS-03: an unusable path used to crash the monitor at the first ALARM_START (snapshots)
    or give a traceback (events log); now it stops the start-up with one line."""
    patch_captures(monkeypatch, [FakeCapture(n=30)])
    inject(monkeypatch, lambda: FakeDetector(no_cars))
    blocker = tmp_path / "a_file"
    blocker.write_text("x", encoding="utf-8")
    cfg = file_cfg()
    setattr(getattr(cfg, section), key, str(blocker / leaf))
    assert main(["--config", config_file(tmp_path, cfg)]) == EXIT_NOT_WATCHING
    err = capsys.readouterr().err
    assert len(error_lines(err)) == 1 and f"cannot create {section}.{key}" in err


# ----------------------------------------------------------------------------- exit 2
def test_exit_2_for_configuration_errors(monkeypatch, tmp_path, capsys):
    patch_captures(monkeypatch, [FakeCapture(n=30)])
    tabbed = tmp_path / "tabbed.yaml"
    tabbed.write_text("video:\n\tsource: x\n", encoding="utf-8")
    assert main(["--config", str(tabbed)]) == EXIT_CONFIG
    wrong_type = file_cfg()
    wrong_type.detector.conf = "high"  # type: ignore[assignment] - a value of the wrong type
    assert main(["--config", config_file(tmp_path, wrong_type)]) == EXIT_CONFIG
    not_a_list = file_cfg()
    not_a_list.detector.vehicle_classes = 5  # type: ignore[assignment] - a scalar where a list is expected
    assert main(["--config", config_file(tmp_path, not_a_list)]) == EXIT_CONFIG
    not_a_path = file_cfg()
    not_a_path.alerts.events_jsonl = 5  # type: ignore[assignment]
    assert main(["--config", config_file(tmp_path, not_a_path)]) == EXIT_CONFIG
    err = capsys.readouterr().err
    # each mistake is one line: the YAML error and the three wrong types are all reported by the
    # loader's validation ('cannot load config ...'); errors only Monitor(cfg) can see say
    # 'invalid configuration'
    lines = error_lines(err)
    assert len(lines) == 4 and "Traceback" not in err
    assert all("cannot load config" in line or "invalid configuration" in line for line in lines), lines


def test_exit_2_when_the_zone_does_not_fit_the_stream(monkeypatch, tmp_path, capsys):
    patch_captures(monkeypatch, [FakeCapture(n=30)])  # 640x360 frames
    inject(monkeypatch, lambda: FakeDetector(no_cars))
    cfg = file_cfg(zone={"polygon": [[200, 150], [440, 150], [440, 270], [200, 270]], "reference_size": [640, 480]})
    assert main(["--config", config_file(tmp_path, cfg)]) == EXIT_CONFIG
    captured = capsys.readouterr()
    assert "zone refused" in captured.err and "ABORTED stop_reason=zone_refused" in captured.out


def test_help_documents_the_exit_status():
    text = build_parser().format_help()
    assert "exit status:" in text
    assert all(f"\n  {code}  " in text for code in (0, 1, 2))
    assert "exit status" in (main.__doc__ or "").lower()


def test_format_summary_says_why_a_run_aborted():
    base = {"frames": 3, "alarms": 0, "fps": 12.5, "timings_ms": {"total": 4.0}}
    assert "ABORTED" not in format_summary(dict(base, aborted=False, stop_reason="end_of_input"))
    line = format_summary(dict(base, aborted=True, stop_reason="reconnect_failed"))
    assert "ABORTED stop_reason=reconnect_failed" in line and line.isascii()


# ----------------------------------------------------------------------------- one override table
def test_parser_and_overrides_come_from_one_table():
    parser = build_parser()
    flags = {opt for action in parser._actions for opt in action.option_strings}
    assert {row.flag for row in CLI_OVERRIDES} <= flags
    zone_file = str(Path(__file__).resolve().parents[1] / "benchmarks" / "team_clips" / "zones" / "clip07.json")
    special = {"--detect-every": "3", "--zone": zone_file}  # --zone must name a loadable zone file
    argv = []
    for row in CLI_OVERRIDES:
        argv += [row.flag] if row.type is None else [row.flag, special.get(row.flag, "7" if row.type in (int, float) else "x")]
    cfg = apply_cli_overrides(MonitorConfig(), parser.parse_args(argv))
    for row in CLI_OVERRIDES:
        for path in row.paths:
            section, _, key = path.partition(".")
            value = getattr(getattr(cfg, section), key)
            if row.type is None:
                assert value == row.const, row.flag
            else:
                assert value == row.type(special.get(row.flag, "7" if row.type in (int, float) else "x"))
        for path, const in row.also:
            section, _, key = path.partition(".")
            assert getattr(getattr(cfg, section), key) == const
    assert apply_cli_overrides(MonitorConfig(), parser.parse_args([])) == MonitorConfig()
    assert apply_cli_overrides(MonitorConfig(), parser.parse_args(["--detect-every", "0"])).detector.detect_every_n_frames == 1
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--detect-every", "x"])
    assert exc.value.code == 2


# ----------------------------------------------------------------------------- tools/benchmark.py
@pytest.fixture(scope="module")
def bench():
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True  # never leave __pycache__ in tools/
    try:
        spec = importlib.util.spec_from_file_location("benchmark_tool", ROOT / "tools" / "benchmark.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    finally:
        sys.dont_write_bytecode = old
    return module


def test_benchmark_reuses_the_monitor_override_rows(bench):
    shared = [row for row in bench.BENCHMARK_OVERRIDES if row.flag in bench.SHARED_FLAGS]
    assert len(shared) == len(bench.SHARED_FLAGS) and all(row in CLI_OVERRIDES for row in shared)
    args = bench.build_parser().parse_args(["--source", "clip.mp4", "--detect-every", "0", "--imgsz", "320",
                                            "--no-debris", "--device", "cpu", "--infra-weights", "g.pt"])
    cfg = bench.configure(args)
    assert (cfg.video.source, cfg.detector.detect_every_n_frames, cfg.detector.imgsz) == ("clip.mp4", 1, 320)
    assert not cfg.debris.enabled and cfg.detector.device == cfg.infrastructure.device == "cpu"
    assert cfg.infrastructure.enabled and cfg.infrastructure.weights == "g.pt" and not cfg.alerts.stdout


def test_benchmark_config_errors_exit_2_with_one_line(bench, tmp_path, capsys):
    tabbed = tmp_path / "tabbed.yaml"
    tabbed.write_text("video:\n\tsource: x\n", encoding="utf-8")
    assert bench.main(["--config", str(tabbed), "--no-detector"]) == 2
    unknown = tmp_path / "unknown.yaml"
    unknown.write_text("video:\n  sourc: x\n", encoding="utf-8")
    assert bench.main(["--config", str(unknown), "--no-detector"]) == 2
    err = capsys.readouterr().err
    assert len(error_lines(err)) == 2 and len(err.strip().splitlines()) == 2 and "Traceback" not in err


def test_benchmark_times_the_steady_state_and_reports_the_debris_warmup(bench, monkeypatch, capsys):
    patch_captures(monkeypatch, [FakeCapture(n=60)])
    cfg = bench.configure(bench.build_parser().parse_args(["--source", "clip.mp4"]))
    cfg.debris.warmup_frames, cfg.debris.warmup_seconds = None, 0.5  # 15 frames at 30 fps
    report = bench.run_benchmark(cfg, frames=20, warmup=2, no_detector=True)
    start = report["monitoring_starts_at_frame"]
    assert start is not None and 10 <= start <= 20
    assert report["frames_timed"] == 20 and report["debris_warmup"]["frames"] == start - 2
    assert report["frames_processed"] == start + 20
    assert report["stages"]["total"]["n"] == 20 and report["debris_warmup"]["stages"]["total"]["n"] == start - 2
    patch_captures(monkeypatch, [FakeCapture(n=60)])
    both = bench.run_benchmark(cfg, frames=20, warmup=2, no_detector=True, include_debris_warmup=True)
    assert both["frames_processed"] == 22 and both["frames_timed"] == 20
    patch_captures(monkeypatch, [FakeCapture(n=120)])  # default 2 s debris warm-up = 60 frames
    assert bench.main(["--source", "clip.mp4", "--no-detector", "--frames", "5", "--warmup", "1"]) == 0
    out = capsys.readouterr().out
    assert "debris warm-up:" in out and "monitoring starts at frame" in out and "steady-state frames" in out
