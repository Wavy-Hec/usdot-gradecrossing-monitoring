"""tools/inspect_checkpoint.py never runs code from the file it inspects (ours-correctness F16).

Its unpickler hands out real objects only from an allowlist of plain containers and value
types; torch/ultralytics symbols become named stubs and any other global (``eval``,
``getattr``, ``os.system``...) is refused with an error naming it. Crafted checkpoints are
built here with the standard pickler or opcode by opcode; the real ones are the repo's.
"""
from __future__ import annotations

import collections
import importlib.util
import io
import json
import os
import pathlib
import pickle
import struct
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "inspect_checkpoint.py"
ENV = dict(os.environ, PYTHONIOENCODING="utf-8")


def _tool():
    spec = importlib.util.spec_from_file_location("inspect_checkpoint_under_test", str(TOOL))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _checkpoint(path: Path, payload: bytes) -> Path:
    """A torch-style zip checkpoint whose data.pkl is ``payload``."""
    with zipfile.ZipFile(str(path), "w") as zf:
        zf.writestr("crafted/data.pkl", payload)
        zf.writestr("crafted/version", "3\n")
    return path


class _Call:
    """Pickles as ``func(*args)``: what a malicious checkpoint puts in its data.pkl."""

    def __init__(self, func, *args):
        self.func, self.args = func, args

    def __reduce__(self):
        return self.func, self.args


def _run(path: Path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(TOOL), str(path)], capture_output=True, text=True, env=ENV,
                          cwd=str(path.parent), timeout=60)


def _marker_code(marker: Path) -> str:
    return "open({!r}, 'w').write('code ran inside inspect_checkpoint') and -1".format(str(marker))


@pytest.mark.parametrize("protocol", [2, 4])
def test_a_checkpoint_that_calls_eval_is_refused_and_nothing_runs(tmp_path, protocol):
    marker = tmp_path / "marker.txt"
    payload = pickle.dumps({"epoch": _Call(eval, _marker_code(marker))}, protocol=protocol)
    proc = _run(_checkpoint(tmp_path / "evil.pt", payload))
    assert proc.returncode == 1
    assert not marker.exists()
    err = proc.stderr.strip()
    assert "refusing to load global" in err and ".eval'" in err and len(err.splitlines()) == 1
    assert proc.stdout == ""


def test_os_system_and_getattr_chains_are_refused(tmp_path):
    marker = tmp_path / "marker.txt"
    cmd = '"{}" -c "open(r\'{}\', \'w\').close()"'.format(sys.executable, marker)
    for k, obj in enumerate([
        _Call(os.system, cmd),
        _Call(getattr, _Call(__import__, "os"), "system"),
        _Call(exec, "open({!r}, 'w').close()".format(str(marker))),
        _Call(subprocess.call, cmd),
    ]):
        proc = _run(_checkpoint(tmp_path / "evil{}.pt".format(k), pickle.dumps({"model": obj}, protocol=2)))
        assert proc.returncode == 1, proc.stdout
        assert "refusing to load global" in proc.stderr
    assert not marker.exists()


def test_the_unpickler_allowlist():
    ic = _tool()
    up = ic.NoTorchUnpickler(io.BytesIO(b""))
    assert up.find_class("collections", "OrderedDict") is collections.OrderedDict
    assert up.find_class("__builtin__", "set") is set and up.find_class("builtins", "frozenset") is frozenset
    assert up.find_class("pathlib", "PosixPath") is pathlib.PurePosixPath  # buildable on Windows too
    assert up.find_class("pathlib", "WindowsPath") is pathlib.PureWindowsPath
    np = pytest.importorskip("numpy")
    # numpy 2 writes numpy._core, numpy 1 numpy.core: both spellings work on either version
    assert up.find_class("numpy._core.multiarray", "scalar") is up.find_class("numpy.core.multiarray", "scalar")
    assert up.find_class("numpy", "dtype") is np.dtype
    for module, name in (("torch", "FloatStorage"), ("torch._utils", "_rebuild_tensor_v2"),
                         ("ultralytics.nn.tasks", "DetectionModel"), ("numpy.random", "seed")):
        stub = up.find_class(module, name)
        assert issubclass(stub, ic._Stub) and stub.__name__ == name and stub.__module__ == module
        assert isinstance(stub(1, 2), ic._Stub)  # calling a stub only records its arguments
    for module, name in (("builtins", "eval"), ("builtins", "exec"), ("builtins", "getattr"),
                         ("__builtin__", "__import__"), ("builtins", "compile"), ("builtins", "open"),
                         ("os", "system"), ("nt", "system"), ("posix", "system"), ("subprocess", "Popen"),
                         ("_codecs", "decode"), ("collections", "namedtuple"), ("copyreg", "_reconstructor")):
        with pytest.raises(ic.UnsafeCheckpointError, match="refusing to load global"):
            up.find_class(module, name)


def _u(text: str) -> bytes:  # BINUNICODE
    raw = text.encode("utf-8")
    return b"X" + struct.pack("<I", len(raw)) + raw


def _g(module: str, name: str) -> bytes:  # GLOBAL
    return b"c" + module.encode() + b"\n" + name.encode() + b"\n"


def test_a_plain_ultralytics_style_checkpoint_still_inspects(tmp_path):
    """Opcode by opcode, the shape of a real best.pt: a DetectionModel (stub) with names/yaml,
    a train_args dict holding a PosixPath (written on Linux/Colab), torch storages and a set."""
    model = (_g("ultralytics.nn.tasks", "DetectionModel") + b")\x81"  # EMPTY_TUPLE NEWOBJ
             + b"}"  # the model's state dict, applied by BUILD below
             + _u("names") + b"}" + b"K\x00" + _u("car") + b"s" + b"s"
             + _u("yaml") + b"}" + _u("nc") + b"K\x01" + b"s" + _u("yaml_file") + _u("yolo11n.yaml") + b"s" + b"s"
             + _u("stride") + _g("torch._utils", "_rebuild_tensor_v2")
             + b"(" + _u("storage") + _g("torch", "FloatStorage") + _u("0") + _u("cpu") + b"K\x03" + b"t"
             + b"Q" + b"K\x00" + b"\x86" + b"R" + b"s"  # BINPERSID: a tensor storage, never read
             + b"b")  # BUILD
    train_args = (b"}" + _u("data") + _g("pathlib", "PosixPath") + _u("/content/data.yaml") + b"\x85R" + b"s"
                  + _u("imgsz") + b"M\x80\x02" + b"s"
                  + _u("classes") + _g("__builtin__", "set") + b"]" + b"K\x02" + b"a" + b"\x85R" + b"s")
    payload = (b"\x80\x02}" + _u("model") + model + b"s" + _u("train_args") + train_args + b"s"
               + _u("epoch") + b"J\xff\xff\xff\xff" + b"s" + _u("version") + _u("8.4.0") + b"s" + b".")
    path = _checkpoint(tmp_path / "plain.pt", payload)
    ic = _tool()
    ck = ic.load_checkpoint_dict(str(path))
    assert ck["train_args"]["classes"] == {2} and isinstance(ck["model"].stride, ic._Stub)
    info = ic.inspect_checkpoint(str(path))
    assert info["names"] == {"0": "car"} and info["nc"] == 1 and info["model_yaml_file"] == "yolo11n.yaml"
    assert info["train_args"] == {"data": "/content/data.yaml", "imgsz": 640}
    assert info["epoch"] == -1 and info["stripped"] is True and info["version"] == "8.4.0"
    proc = _run(path)
    assert proc.returncode == 0, proc.stderr
    assert "car" in proc.stdout and "/content/data.yaml" in proc.stdout


def test_the_repo_checkpoint_with_numpy_scalars_inspects(capsys):
    """best_vehicle.pt (not stripped) holds numpy scalars, _codecs.encode bytes, torch.device and
    an IterableSimpleNamespace - every allowlist and stub path of a real file."""
    weights = ROOT / "best_vehicle.pt"
    if not weights.is_file():
        pytest.skip("best_vehicle.pt not present")
    pytest.importorskip("numpy")
    ic = _tool()
    info = ic.inspect_checkpoint(str(weights))
    assert info["epoch"] == 21 and info["stripped"] is False and info["optimizer_present"] is True
    assert info["best_fitness"] == pytest.approx(0.79708, abs=1e-5)  # a numpy float64 scalar
    assert info["nc"] == 6 and info["names"]["0"] == "Car"
    assert info["train_args"]["imgsz"] == 640
    assert ic.main([str(weights), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["updates"] == 2235
