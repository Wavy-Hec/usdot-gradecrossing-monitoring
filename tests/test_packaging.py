"""Packaging metadata: one version, parseable dependency lists, one OpenCV build per install.

Nothing is installed and nothing touches the network: the tests read pyproject.toml, setup.py
and the requirements files, and run two small subprocesses that hide OpenCV and torch to show
what the base install (``pip install .``, no extra) can do.
"""
from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import pytest

import gradecrossing

if sys.version_info >= (3, 11):
    import tomllib
else:  # pytest itself depends on tomli below 3.11
    tomllib = pytest.importorskip("tomli")
Requirement = pytest.importorskip("packaging.requirements").Requirement
canonicalize_name = pytest.importorskip("packaging.utils").canonicalize_name
Version = pytest.importorskip("packaging.version").Version

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
PROJECT = PYPROJECT["project"]
EXTRAS: Dict[str, List[str]] = PROJECT["optional-dependencies"]
TEAM_REPO = "https://github.com/ruby-gonzalez/usdot-gradecrossing-monotoring"

# Distributions that put a cv2/ directory into the environment, by build; the ultralytics
# distributions pull theirs in. Two kinds in one environment overwrite each other's files.
HEADLESS = {"opencv-python-headless", "opencv-contrib-python-headless", "ultralytics-opencv-headless"}
GUI = {"opencv-python", "opencv-contrib-python", "ultralytics"}
TORCH_STACK = {"ultralytics", "ultralytics-opencv-headless", "torch", "torchvision", "lap"}


def names(requirements: List[str]) -> set:
    return {canonicalize_name(Requirement(r).name) for r in requirements}


def normalised(requirements: List[str]) -> List[str]:
    return sorted(str(Requirement(r)) for r in requirements)


def read_requirements(filename: str) -> List[str]:
    lines = (ROOT / filename).read_text(encoding="utf-8").splitlines()
    return [line.split(" #")[0].strip() for line in lines if line.strip() and not line.lstrip().startswith("#")]


def test_version_is_single_sourced():
    """pyproject.toml reads the version from gradecrossing/__init__.py instead of repeating it.
    setuptools resolves the attr without importing the package (the build environment has no
    numpy), so the assignment has to stay a single string literal."""
    assert "version" not in PROJECT
    assert "version" in PROJECT["dynamic"]
    assert PYPROJECT["tool"]["setuptools"]["dynamic"]["version"] == {"attr": "gradecrossing.__version__"}
    tree = ast.parse((ROOT / "gradecrossing" / "__init__.py").read_text(encoding="utf-8"))
    literals = [ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "__version__" for t in node.targets)]
    assert literals == [gradecrossing.__version__]
    assert str(Version(gradecrossing.__version__)) == gradecrossing.__version__  # PEP 440, normalised


def test_setup_py_is_a_bare_shim():
    """pip < 21.3 (JetPack 5 ships 20.0.2) needs a setup.py for ``pip install -e .``. It must not
    become a second place for metadata, and the isolated build needs a setuptools that reads
    pyproject's [project] table (61+)."""
    tree = ast.parse((ROOT / "setup.py").read_text(encoding="utf-8"))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "setup"]
    assert len(calls) == 1 and not calls[0].args and not calls[0].keywords
    build = PYPROJECT["build-system"]
    assert build["build-backend"] == "setuptools.build_meta"
    setuptools_req = [Requirement(r) for r in build["requires"] if canonicalize_name(Requirement(r).name) == "setuptools"]
    assert len(setuptools_req) == 1
    assert not setuptools_req[0].specifier.contains("60.10.0")


def test_dependency_lists_parse():
    groups = {"dependencies": PROJECT["dependencies"], **EXTRAS}
    assert set(EXTRAS) == {"headless", "detector", "display", "dev"}
    for group, requirements in groups.items():
        parsed = [Requirement(r) for r in requirements]  # raises InvalidRequirement on a typo
        assert parsed, group
        assert len(names(requirements)) == len(parsed), f"{group} names a distribution twice"


def test_base_install_is_torch_free_and_opencv_free():
    base = names(PROJECT["dependencies"])
    assert base == {"numpy", "pyyaml"}
    assert base.isdisjoint(TORCH_STACK | HEADLESS | GUI)
    assert names(EXTRAS["headless"]).isdisjoint(TORCH_STACK)


def test_every_install_gets_exactly_one_opencv_build():
    for extra in ("headless", "detector", "display"):
        installed = names(PROJECT["dependencies"] + EXTRAS[extra])
        assert installed & (HEADLESS | GUI), f"[{extra}] installs no OpenCV"
        assert not (installed & HEADLESS and installed & GUI), f"[{extra}] mixes headless and GUI OpenCV"
    assert names(EXTRAS["detector"]) >= {"opencv-python-headless", "ultralytics-opencv-headless"}
    assert names(EXTRAS["display"]) >= {"opencv-python", "ultralytics"}
    for filename in ("requirements.txt", "requirements-display.txt", "requirements-dev.txt"):
        installed = names(read_requirements(filename))
        assert not (installed & HEADLESS and installed & GUI), filename
    assert names(read_requirements("requirements-dev.txt")).isdisjoint(HEADLESS | GUI)


def test_requirements_files_match_pyproject():
    """The requirements files serve a checkout that is not pip-installed; they repeat the
    pyproject lists, so pin them together."""
    base = PROJECT["dependencies"]
    assert normalised(read_requirements("requirements.txt")) == normalised(base + EXTRAS["detector"])
    assert normalised(read_requirements("requirements-display.txt")) == normalised(base + EXTRAS["display"])
    assert normalised(read_requirements("requirements-dev.txt")) == normalised(EXTRAS["dev"])


def test_classifiers_and_urls():
    floor = re.fullmatch(r">=\s*3\.(\d+)", PROJECT["requires-python"])
    assert floor is not None
    minors = sorted(int(m.group(1)) for c in PROJECT["classifiers"]
                    for m in [re.fullmatch(r"Programming Language :: Python :: 3\.(\d+)", c)] if m)
    assert minors[0] == int(floor.group(1)) == 8
    assert minors == list(range(minors[0], minors[-1] + 1)) and minors[-1] >= 12
    assert PROJECT["urls"] and all(url.startswith(TEAM_REPO) for url in PROJECT["urls"].values())


BLOCK_HEAVY = "import sys; sys.modules.update({'cv2': None, 'torch': None, 'ultralytics': None}); "


@pytest.mark.parametrize("code, expected", [
    ("import gradecrossing; print(gradecrossing.__version__, type(gradecrossing.load_config('configs/default.yaml')).__name__)",
     f"{gradecrossing.__version__} MonitorConfig"),
    ("import runpy; sys.argv = ['gradecrossing-evaluate', '--help']; "
     "runpy.run_module('gradecrossing.evaluate', run_name='__main__')", "usage"),
])
def test_base_install_runs_without_opencv_or_torch(code: str, expected: str):
    """The base install (numpy + pyyaml) promises config/types and gradecrossing-evaluate.
    ``sys.modules[name] = None`` makes ``import name`` fail exactly as on a machine without it."""
    env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1")
    proc = subprocess.run([sys.executable, "-c", BLOCK_HEAVY + code], cwd=str(ROOT), env=env,
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    assert expected in proc.stdout


def test_console_scripts_resolve():
    """[project.scripts] targets exist; the monitor's goes through gradecrossing._entry (below)."""
    import importlib

    scripts = PROJECT["scripts"]
    assert scripts == {"gradecrossing-monitor": "gradecrossing._entry:monitor",
                       "gradecrossing-evaluate": "gradecrossing.evaluate:main"}
    for target in scripts.values():
        module, _, attr = target.partition(":")
        assert callable(getattr(importlib.import_module(module), attr)), target


@pytest.mark.parametrize("argv", [["--help"], ["--source", "clip.mp4", "--no-stdout"]])
def test_monitor_script_on_a_base_install_says_which_extra_to_install(argv: List[str]):
    """Final review Q4: on the base install (no OpenCV) gradecrossing-monitor crashed with a
    ModuleNotFoundError traceback and exit 1 ("not watching"), even for --help. It now prints one
    line naming the extras and exits 2 (a setup error; running again cannot help)."""
    code = f"from gradecrossing._entry import monitor; raise SystemExit(monitor({argv!r}))"
    env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1")
    proc = subprocess.run([sys.executable, "-c", BLOCK_HEAVY + code], cwd=str(ROOT), env=env,
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 2, proc.stderr
    assert proc.stdout == "" and "Traceback" not in proc.stderr
    (line,) = proc.stderr.strip().splitlines()
    assert line.startswith("error: gradecrossing-monitor cannot start (") and "cv2" in line
    assert '".[detector]"' in line and '".[display]"' in line


def test_monitor_script_runs_the_monitor_when_opencv_is_there(capsys):
    from gradecrossing._entry import monitor

    with pytest.raises(SystemExit) as info:  # argparse's --help: the arguments reach main()
        monitor(["--help"])
    assert info.value.code == 0 and "exit status:" in capsys.readouterr().out
