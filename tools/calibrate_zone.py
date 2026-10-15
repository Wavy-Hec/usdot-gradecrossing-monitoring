#!/usr/bin/env python
"""Calibrate the crossing zone polygon for a camera and write it into a YAML config.

Why a separate tool
-------------------
Every safety decision the monitor makes (is the vehicle *in* the zone, is that blob
*inside* the crossing) depends on the polygon being right, and there is no way for the
software to verify that by itself. So the zone is drawn once by a person, on a frame of
the actual camera, at native resolution, and stored in the config. The monitor then
keeps it locked to the scene with the stabilizer (``gradecrossing/zone.py``) and scales
it automatically when ``video.resize_width`` is set.

Workflow
--------
::

    python tools/calibrate_zone.py --source rtsp://camera/stream --out configs/mysite.yaml
    python tools/calibrate_zone.py --source videos/clip.mov --frame 120 --config configs/default.yaml --out configs/clip.yaml

A window shows the chosen frame. Left-click the corners of the crossing (in order around
the boundary), right-click or ``u`` undoes the last point, ``r`` clears, ``s`` accepts the
automatic seed polygon (yellow, from ``gradecrossing.autozone.propose_zone``; it is only a
starting guess and is often wrong - always check it), ``Enter``/``y`` saves, ``q``/``Esc``
quits without saving. ``--out`` receives only what the site needs: ``video.source``, the
polygon and the calibration geometry below, on top of the keys ``--config`` (or the existing
``--out``) sets itself, except a ``zone.polygon_file``, which the new polygon replaces; every
other key keeps the package default, so a later default change reaches the site. A base
config that cannot be loaded exits 2 with a one-line error (a full-default site file of
gradecrossing 0.2.0, which still names retired keys, is one: the error says what to delete).

The calibration geometry is saved with the polygon: ``zone.reference_size`` is the frame
size the polygon was drawn in (the monitor scales the polygon to its processing size and
refuses a stream with another aspect ratio), and the calibration frame itself is written,
clean (no overlay), next to the config as ``<config stem>_reference.png`` and recorded as
``zone.reference_image``. The monitor registers live frames against that image (a global
exposure difference is compensated), so a camera that moved between calibration and
start-up is corrected on the first live frame; when no live frame can be registered within
``zone.registration_timeout_seconds`` (night against a daylight frame, a changed view) it
says so in a HEALTH event, tracks from a live frame at the calibration pose and registers
later. Calibrate on a frame where the crossing is clear (no train or vehicle in the zone)
and re-run the tool whenever the camera is re-aimed. ``--no-reference-image`` skips the image.

Headless use (SSH to a Jetson, CI): the tool always prints the frame size and the seed
polygon, ``--save-frame shot.jpg`` writes the frame (with the seed drawn) so you can pick
coordinates in any image viewer, and ``--polygon "x1,y1 x2,y2 x3,y3 x4,y4"`` writes a
polygon without opening a window. Every printed line is plain ASCII.
"""
from __future__ import annotations

import argparse
import copy
import datetime
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))  # allow `python tools/calibrate_zone.py` from any cwd

from gradecrossing.autozone import propose_zone  # noqa: E402
from gradecrossing.config import (  # noqa: E402
    InfrastructureConfig, MonitorConfig, VideoConfig, config_from_dict, load_config)
from gradecrossing.geometry import polygon_area  # noqa: E402
from gradecrossing.monitor import open_source  # noqa: E402
from gradecrossing.types import Detection  # noqa: E402
from gradecrossing.zone import ZoneStabilizer  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
WINDOW = "calibrate_zone - left click: add point | right click/u: undo | s: use seed | Enter: save | q: quit"
COLOR_SEED = (0, 255, 255)
COLOR_POLY = (0, 220, 0)
COLOR_TEXT = (255, 255, 255)


# --------------------------------------------------------------------------------------
# frame acquisition
# --------------------------------------------------------------------------------------
def grab_frame(source: str, frame_index: int = 0) -> np.ndarray:
    """Return frame ``frame_index`` of a video/camera/stream, or the image at ``source``.

    Uses ``gradecrossing.monitor.open_source`` so camera indices, RTSP URLs and files are
    treated exactly as the monitor treats them. For a live source the "frame index" is
    simply the Nth frame received.
    """
    if Path(source).suffix.lower() in IMAGE_EXTS:
        img = cv2.imread(str(source))
        if img is None:
            raise RuntimeError(f"cannot read image {source!r}")
        return img
    cap, _fps, _live = open_source(VideoConfig(source=source))
    try:
        frame = None
        for k in range(int(frame_index) + 1):
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError(f"could not read frame {k} from {source!r}")
        return frame
    finally:
        cap.release()


def infra_boxes(frame: np.ndarray, weights: Optional[str]) -> Optional[List[Detection]]:
    """Gate/sign boxes from the infrastructure model, used to bound the seed polygon.

    Optional and lazy: loading the model needs ultralytics/torch, so this is only done
    when ``--infra-weights`` is given.
    """
    if not weights:
        return None
    from gradecrossing.detection import InfrastructureDetector  # local import (torch)

    det = InfrastructureDetector(InfrastructureConfig(enabled=True, weights=weights))
    boxes = det.detect(frame)
    print(f"infrastructure model found {len(boxes)} box(es): "
          + ", ".join(f"{b.cls_name}({b.conf:.2f})" for b in boxes))
    return boxes


# --------------------------------------------------------------------------------------
# polygon helpers
# --------------------------------------------------------------------------------------
def parse_polygon(text: str) -> List[List[float]]:
    """Parse ``"x1,y1 x2,y2 ..."`` or a JSON list ``[[x, y], ...]`` into a point list."""
    text = text.strip()
    if text.startswith("["):
        pts = json.loads(text)
    else:
        pts = []
        for token in text.replace(";", " ").split():
            x, y = token.split(",")
            pts.append([float(x), float(y)])
    pts = [[float(p[0]), float(p[1])] for p in pts]
    if len(pts) < 3:
        raise ValueError("a polygon needs at least 3 points")
    return pts


def format_polygon(points: Sequence[Sequence[float]]) -> str:
    """YAML-ready ``[[x, y], ...]`` with integer-looking floats printed as integers."""

    def fmt(v: float) -> str:
        return str(int(round(v))) if abs(v - round(v)) < 1e-6 else f"{v:.1f}"

    return "[" + ", ".join(f"[{fmt(x)}, {fmt(y)}]" for x, y in points) + "]"


def reference_image_path(out: str) -> Path:
    """Where the clean calibration frame of config ``out`` goes: ``<dir>/<stem>_reference.png``.

    The path keeps the form ``--out`` was given in (relative paths stay relative to the
    working directory, which is how the monitor resolves every path in a config; for this
    one it also tries the repository root, so a monitor started elsewhere still finds it).
    """
    p = Path(out)
    return p.with_name(p.stem + "_reference.png")


class ConfigFileError(ValueError):
    """A base config (``--config`` or the existing ``--out``) that cannot be loaded."""


def base_config_path(out: str, base: Optional[str]) -> Optional[str]:
    """The file whose settings the written config keeps: ``--config``, else an existing ``--out``."""
    if base:
        return base
    return out if Path(out).exists() else None


def read_site_yaml(path: Optional[str]) -> Dict[str, Any]:
    """The mapping a site YAML sets (only its own keys, no defaults merged in), validated.

    Raises :class:`ConfigFileError` with a one-line message for a missing file, a YAML
    syntax error, a top level that is not a mapping or an unknown key.
    """
    if path is None:
        return {}
    import yaml

    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        config_from_dict(data)  # unknown sections/keys and non-mapping sections raise here
    except Exception as exc:  # noqa: BLE001 - OSError, yaml.YAMLError, KeyError, TypeError...
        raise ConfigFileError(f"cannot load config {path!r}: {' '.join(str(exc).split())}") from exc
    return copy.deepcopy(data)


def write_config(points: Sequence[Sequence[float]], out: str, base: Optional[str], source: Optional[str],
                 keep_source: bool, frame_size: Optional[Sequence[int]] = None,
                 reference_image: Optional[str] = None) -> MonitorConfig:
    """Write the site config: the base file's own keys plus the source and the zone calibration.

    Only what the site needs is written - ``video.source`` and ``zone.polygon`` /
    ``reference_size`` / ``reference_image`` - on top of the keys the base file itself sets
    (``--config`` if given, else the existing ``--out``, so re-calibrating a site keeps its
    other settings). Nothing is copied from the built-in defaults, so a later change of a
    default reaches the site; a key that must stay fixed is pinned by writing it in the file.
    ``frame_size = (w, h)`` becomes ``zone.reference_size``; ``reference_image`` becomes
    ``zone.reference_image`` ("" removes an image left by an earlier calibration, which no
    longer matches the new polygon). A ``zone.polygon_file`` in the base is removed (and
    said so): the monitor prefers the file over ``zone.polygon``, so keeping it would leave
    the site on the old zone. Returns the written file loaded as a full config.
    """
    data = read_site_yaml(base_config_path(out, base))
    if source and not keep_source:
        video = data.get("video") or {}
        video["source"] = str(source)
        data["video"] = video
    zone = data.get("zone") or {}
    zone["polygon"] = [[float(round(x, 1)), float(round(y, 1))] for x, y in points]
    old_file = zone.pop("polygon_file", None)
    if old_file:
        print(f"zone.polygon_file {old_file!r} removed from the config: the new calibration (zone.polygon) replaces it")
    if frame_size is not None:
        zone["reference_size"] = [int(frame_size[0]), int(frame_size[1])]
    if reference_image:
        zone["reference_image"] = str(reference_image)
    elif reference_image is not None:
        zone.pop("reference_image", None)
    data["zone"] = zone
    import yaml

    from gradecrossing import __version__

    header = (f"# Site config written by tools/calibrate_zone.py (gradecrossing {__version__}) on "
              f"{datetime.date.today().isoformat()}.\n"
              "# Only this site's keys are listed; every other key takes the package default\n"
              "# (configs/default.yaml). Add a key here to pin it for this site.\n")
    class _SiteDumper(yaml.SafeDumper):
        """Block-style sections; a list of scalars (a point, a size) on one line."""

    _SiteDumper.add_representer(list, lambda dumper, seq: dumper.represent_sequence(
        "tag:yaml.org,2002:seq", seq, flow_style=not any(isinstance(v, (list, dict)) for v in seq)))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(header)
        yaml.dump(data, fh, Dumper=_SiteDumper, sort_keys=False, default_flow_style=False)
    return load_config(out)


def trackable(frame: np.ndarray, cfg: MonitorConfig) -> bool:
    """True when the zone stabilizer can anchor on ``frame`` (enough texture outside the zone)."""
    stab = ZoneStabilizer(cfg.zone)
    stab.initialize(frame, 0.0)
    return stab.initialized


def draw_polygons(frame: np.ndarray, points: Sequence[Sequence[float]], seed: Optional[np.ndarray],
                  message: str = "") -> np.ndarray:
    """Frame copy with the seed (yellow, thin) and the operator polygon (green, with corner dots)."""
    img = frame.copy()
    h, w = img.shape[:2]
    scale = max(0.5, min(1.0, w / 1600.0))
    if seed is not None and len(seed) >= 3:
        cv2.polylines(img, [np.round(seed).astype(np.int32).reshape(-1, 1, 2)], True, COLOR_SEED, 1, cv2.LINE_AA)
        x, y = seed[0]
        cv2.putText(img, "seed (press s to use)", (int(x), int(y) - 6), cv2.FONT_HERSHEY_SIMPLEX, scale * 0.6,
                    COLOR_SEED, 1, cv2.LINE_AA)
    if points:
        pts = np.round(np.asarray(points, dtype=np.float32)).astype(np.int32).reshape(-1, 1, 2)
        if len(points) >= 2:
            cv2.polylines(img, [pts], len(points) >= 3, COLOR_POLY, 2, cv2.LINE_AA)
        for k, (x, y) in enumerate(points):
            cv2.circle(img, (int(round(x)), int(round(y))), 5, COLOR_POLY, -1)
            cv2.putText(img, str(k + 1), (int(x) + 6, int(y) - 6), cv2.FONT_HERSHEY_SIMPLEX, scale * 0.6,
                        COLOR_POLY, 1, cv2.LINE_AA)
    if message:
        cv2.rectangle(img, (0, 0), (w, int(28 * scale) + 10), (0, 0, 0), -1)
        cv2.putText(img, message, (8, int(22 * scale) + 4), cv2.FONT_HERSHEY_SIMPLEX, scale * 0.7, COLOR_TEXT, 1, cv2.LINE_AA)
    return img


# --------------------------------------------------------------------------------------
# GUI
# --------------------------------------------------------------------------------------
class PolygonEditor:
    """Minimal OpenCV click-to-draw editor. Raises ``cv2.error`` on a headless build."""

    def __init__(self, frame: np.ndarray, seed: Optional[np.ndarray]) -> None:
        self.frame = frame
        self.seed = seed
        self.points: List[List[float]] = []

    def _on_mouse(self, event: int, x: int, y: int, _flags: int, _param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append([float(x), float(y)])
        elif event == cv2.EVENT_RBUTTONDOWN and self.points:
            self.points.pop()

    def run(self) -> Optional[List[List[float]]]:
        """Show the window until the operator saves (returns points) or quits (None)."""
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW, self._on_mouse)
        try:
            while True:
                n = len(self.points)
                msg = (f"{n} point(s). left click: add | right click/u: undo | r: clear | s: seed | "
                       f"Enter/y: save ({'ok' if n >= 3 else 'need >= 3'}) | q/Esc: quit")
                cv2.imshow(WINDOW, draw_polygons(self.frame, self.points, self.seed, msg))
                key = cv2.waitKey(20) & 0xFF
                if key in (27, ord("q")):
                    return None
                if key in (13, 10, ord("y")) and n >= 3:
                    return [list(p) for p in self.points]
                if key in (ord("u"), 8) and self.points:
                    self.points.pop()
                elif key == ord("r"):
                    self.points = []
                elif key == ord("s") and self.seed is not None:
                    self.points = [[float(x), float(y)] for x, y in self.seed]
        finally:
            try:
                cv2.destroyWindow(WINDOW)
            except cv2.error:
                pass


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Draw the crossing zone polygon on a frame of the camera and write it into a YAML config.",
        epilog="Headless: add --polygon 'x1,y1 x2,y2 x3,y3 x4,y4' (no window) or --save-frame shot.jpg to pick "
               "coordinates in an image viewer first.",
    )
    p.add_argument("--source", required=True, help="video file, image, camera index or stream URL")
    p.add_argument("--out", default="configs/mysite.yaml", help="YAML config to write/merge (default: configs/mysite.yaml)")
    p.add_argument("--config", help="base YAML to merge into (default: existing --out, else built-in defaults)")
    p.add_argument("--frame", type=int, default=0, help="frame index to calibrate on (default 0)")
    p.add_argument("--no-seed", action="store_true", help="do not compute the automatic Hough-line seed polygon")
    p.add_argument("--infra-weights", help="infrastructure model weights: gate/sign boxes bound the seed (needs ultralytics)")
    p.add_argument("--polygon", help="write this polygon without a window: 'x1,y1 x2,y2 ...' or JSON [[x,y],...]")
    p.add_argument("--headless", action="store_true", help="never open a window; print the seed and exit")
    p.add_argument("--save-frame", metavar="PATH", help="write the calibration frame (seed drawn) to this image file")
    p.add_argument("--keep-source", action="store_true", help="do not overwrite video.source in the config")
    p.add_argument("--no-reference-image", action="store_true",
                   help="do not save the clean calibration frame next to the config (zone.reference_image)")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Exit status: 0 written (or headless look only), 1 no frame / nothing written / write
    failed, 2 usage or config error (bad ``--polygon``, a base config that cannot be loaded),
    3 no GUI available."""
    args = build_parser().parse_args(argv)
    try:  # a broken base config is reported before a camera is opened or anything is written
        read_site_yaml(base_config_path(args.out, args.config))
    except ConfigFileError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    try:
        frame = grab_frame(args.source, args.frame)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    h, w = frame.shape[:2]
    print(f"frame {args.frame} of {args.source}: {w}x{h} (polygon coordinates are in these native pixels)")

    seed = None
    if not args.no_seed:
        try:
            seed = propose_zone(frame, infra_boxes(frame, args.infra_weights))
        except Exception as exc:  # the seed is a convenience; never let it stop calibration
            print(f"seed proposal failed: {exc}", file=sys.stderr)
        if seed is None:
            print("seed: none (autozone found no pair of rail-like lines; draw the polygon by hand)")
        else:
            print(f"seed: {format_polygon(seed)}  (area {polygon_area(seed):.0f} px^2; a guess - verify it)")

    if args.save_frame:
        shot = draw_polygons(frame, [], seed, f"{w}x{h}; seed in yellow")
        if cv2.imwrite(args.save_frame, shot):
            print(f"frame written to {args.save_frame}")
        else:
            print(f"error: could not write {args.save_frame}", file=sys.stderr)
            return 1

    points: Optional[List[List[float]]] = None
    if args.polygon:
        try:
            points = parse_polygon(args.polygon)
        except (ValueError, json.JSONDecodeError) as exc:
            print(f"error: bad --polygon: {exc}", file=sys.stderr)
            return 2
        for x, y in points:
            if not (0 <= x <= w and 0 <= y <= h):
                print(f"warning: point ({x:.0f}, {y:.0f}) lies outside the {w}x{h} frame", file=sys.stderr)
    elif args.headless:
        print("headless: nothing written. Re-run with --polygon 'x1,y1 x2,y2 x3,y3 x4,y4' to save a polygon.")
        return 0
    else:
        try:
            points = PolygonEditor(frame, seed).run()
        except cv2.error:
            print("OpenCV has no GUI here (headless build or no display). Use --save-frame to look at the "
                  "frame and --polygon 'x1,y1 x2,y2 ...' to write the zone.", file=sys.stderr)
            return 3
        if points is None:
            print("cancelled; nothing written")
            return 1

    reference = ""
    if not args.no_reference_image:
        ref_path = reference_image_path(args.out)
        ref_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(ref_path), frame):  # the frame as read: no seed, no polygon drawn
            print(f"error: could not write the reference frame {ref_path}", file=sys.stderr)
            return 1
        reference = ref_path.as_posix()
        print(f"clean calibration frame written to {reference} (zone.reference_image)")
    try:
        cfg = write_config(points, args.out, args.config, args.source, args.keep_source,
                           frame_size=(w, h), reference_image=reference)
    except ConfigFileError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"error: could not write {args.out}: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {args.out} (only this site's keys; everything else is the package default)")
    print("zone:")
    print(f"  polygon: {format_polygon(cfg.zone.polygon)}")
    print(f"  reference_size: [{w}, {h}]")
    if reference:
        print(f"  reference_image: {reference}")
    if not trackable(frame, cfg):
        print("warning: this frame has too little texture outside the zone for the zone stabilizer to "
              "anchor on (black, blurred or uniform?); calibrate on a clear, textured frame", file=sys.stderr)
    print(f"area {polygon_area(np.asarray(cfg.zone.polygon, dtype=np.float32)):.0f} px^2 "
          f"({100.0 * polygon_area(np.asarray(cfg.zone.polygon, dtype=np.float32)) / float(w * h):.1f}% of the frame)")
    print(f"next: python -m gradecrossing.monitor --config {args.out} --events events.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
