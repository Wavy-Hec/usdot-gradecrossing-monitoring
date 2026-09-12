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
quits without saving. The polygon is merged into ``--config`` (or the existing ``--out``,
or the built-in defaults) and written to ``--out`` together with ``video.source``.

Headless use (SSH to a Jetson, CI): the tool always prints the frame size and the seed
polygon, ``--save-frame shot.jpg`` writes the frame (with the seed drawn) so you can pick
coordinates in any image viewer, and ``--polygon "x1,y1 x2,y2 x3,y3 x4,y4"`` writes a
polygon without opening a window. Every printed line is plain ASCII.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Sequence

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))  # allow `python tools/calibrate_zone.py` from any cwd

from gradecrossing.autozone import propose_zone  # noqa: E402
from gradecrossing.config import InfrastructureConfig, MonitorConfig, VideoConfig, load_config, save_config  # noqa: E402
from gradecrossing.geometry import polygon_area  # noqa: E402
from gradecrossing.monitor import open_source  # noqa: E402
from gradecrossing.types import Detection  # noqa: E402

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


def write_config(points: Sequence[Sequence[float]], out: str, base: Optional[str], source: Optional[str],
                 keep_source: bool) -> MonitorConfig:
    """Merge the polygon (and source) into a config and save it to ``out``.

    Precedence for the base config: ``--config`` if given, else the existing ``--out``
    file (so re-calibrating a site keeps its other settings), else the built-in defaults.
    """
    if base:
        cfg = load_config(base)
    elif Path(out).exists():
        cfg = load_config(out)
    else:
        cfg = MonitorConfig()
    cfg.zone.polygon = [[float(round(x, 1)), float(round(y, 1))] for x, y in points]
    if source and not keep_source:
        cfg.video.source = str(source)
    save_config(cfg, out)
    return cfg


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
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
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

    cfg = write_config(points, args.out, args.config, args.source, args.keep_source)
    print(f"wrote {args.out}")
    print("zone:")
    print(f"  polygon: {format_polygon(cfg.zone.polygon)}")
    print(f"area {polygon_area(np.asarray(cfg.zone.polygon, dtype=np.float32)):.0f} px^2 "
          f"({100.0 * polygon_area(np.asarray(cfg.zone.polygon, dtype=np.float32)) / float(w * h):.1f}% of the frame)")
    print(f"next: python -m gradecrossing.monitor --config {args.out} --events events.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
