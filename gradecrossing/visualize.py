"""Annotated overlay for the monitor (display window, saved video, alarm snapshots).

``draw_overlay`` paints one :class:`~gradecrossing.types.FrameResult` onto a copy of
the frame. It is the only place in the package that draws anything, so a change of
colour scheme or label format never touches the safety logic.

What an operator needs to see at a glance, in priority order:

1. The **status banner** (CLEAR / WARNING / ALARM) with the reasons the engine gave.
2. The **zone polygon** and whether it can be trusted (green = freshly estimated,
   red = held from the last healthy estimate because the stabilizer lost track).
3. Each **tracked object** with its id, class, speed and stuck time, coloured by the
   severity the alert engine assigned to *that id* (red alarm, orange warning); trains
   and unmapped classes are grey so it is obvious they are ignored by the stuck logic.
4. **Debris blobs** in magenta, drawn from the connected-component rectangles the
   debris detector already computed (no per-frame mask blending: the overlay must stay
   cheap on a Jetson, where every millisecond is detector time).
5. Gate state and a one-line zone-health / timing readout.

All text is ASCII. Nothing here calls ``cv2.imshow``; the monitor owns the window.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import cv2
import numpy as np

from .tracking import PredictedDetection
from .types import FrameResult, ObjectState, Status

# BGR colours (OpenCV order).
COLOR_CLEAR = (60, 200, 60)
COLOR_WARNING = (0, 165, 255)
COLOR_ALARM = (0, 0, 255)
COLOR_HELD = (0, 0, 255)
COLOR_OTHER = (160, 160, 160)
COLOR_DEBRIS = (255, 0, 255)
COLOR_TEXT = (255, 255, 255)
COLOR_SHADOW = (0, 0, 0)

STATUS_COLORS = {
    Status.CLEAR: COLOR_CLEAR,
    Status.WARNING: COLOR_WARNING,
    Status.ALARM: COLOR_ALARM,
}

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _ascii(text: str) -> str:
    """Hershey fonts only cover ASCII; replace anything else so labels never show garbage."""
    return text.encode("ascii", "replace").decode("ascii")


def _put_text(img: np.ndarray, text: str, org: Tuple[int, int], scale: float,
              color: Tuple[int, int, int], thickness: int = 1) -> None:
    """Text with a thin dark outline so it stays readable on any background."""
    text = _ascii(text)
    cv2.putText(img, text, org, _FONT, scale, COLOR_SHADOW, thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, _FONT, scale, color, thickness, cv2.LINE_AA)


def object_color(obj: Optional[ObjectState], warning_ids: Sequence[int],
                 alarm_ids: Sequence[int]) -> Tuple[int, int, int]:
    """Colour for one tracked object: red alarm, orange warning, grey ignored class, green otherwise.

    The severity comes from the *alert engine's per-id lists*, not from re-deriving
    thresholds here, so the overlay can never disagree with the events that were sent.
    """
    if obj is None:
        return COLOR_OTHER
    if obj.track_id in alarm_ids:
        return COLOR_ALARM
    if obj.track_id in warning_ids:
        return COLOR_WARNING
    if not (obj.is_vehicle or obj.is_person):
        return COLOR_OTHER  # trains and unmapped classes never feed the stuck logic
    return COLOR_CLEAR


def object_label(obj: Optional[ObjectState], cls_name: str, track_id: Optional[int]) -> str:
    """``id cls speed stuck_s`` for a box; falls back to the raw detection when untracked."""
    if obj is None:
        tid = "?" if track_id is None else str(track_id)
        return f"{tid} {cls_name}"
    units = "m/s" if obj.speed_units == "m/s" else "px/s"
    speed = f"{obj.speed:.1f}{units}" if units == "m/s" else f"{obj.speed:.0f}{units}"
    zone_flag = "*" if obj.in_zone else ""
    return f"{obj.track_id} {obj.cls_name}{zone_flag} {speed} {obj.stuck_seconds:.1f}s"


def draw_overlay(frame_bgr: np.ndarray, result: FrameResult,
                 fps_estimate: Optional[float] = None) -> np.ndarray:
    """Return a copy of ``frame_bgr`` annotated with everything in ``result``.

    ``frame_bgr`` must be at the *processing* resolution (the one ``result`` was computed
    at), otherwise the polygon and boxes land in the wrong place. Cost is a handful of
    ``cv2.polylines``/``rectangle``/``putText`` calls; no full-frame mask is blended.
    """
    img = frame_bgr.copy()
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = img.shape[:2]
    scale = max(0.4, min(1.2, w / 1280.0))  # font scale that stays legible from 320 to 1920 px wide
    thick = 1 if w < 800 else 2
    line_h = int(22 * scale + 8)

    assessment = result.assessment
    status_color = STATUS_COLORS.get(assessment.status, COLOR_OTHER)

    # ---- zone polygon: green when the estimate is fresh, red when held (unhealthy)
    zone = result.zone
    if zone is not None and zone.polygon is not None and len(zone.polygon) >= 3:
        poly = np.asarray(zone.polygon, dtype=np.int32).reshape(-1, 1, 2)
        zone_color = COLOR_CLEAR if zone.healthy else COLOR_HELD
        cv2.polylines(img, [poly], isClosed=True, color=zone_color, thickness=thick + 1, lineType=cv2.LINE_AA)

    # ---- debris blobs (rectangles from connectedComponentsWithStats; no mask blending)
    debris = result.debris
    if debris is not None and debris.blobs:
        for (bx, by, bw, bh, _area) in debris.blobs:
            cv2.rectangle(img, (int(bx), int(by)), (int(bx + bw), int(by + bh)), COLOR_DEBRIS, thick)
        _put_text(img, f"debris {100.0 * debris.area_fraction:.1f}% {debris.persist_seconds:.1f}s",
                  (int(debris.blobs[0][0]), max(line_h, int(debris.blobs[0][1]) - 4)), scale * 0.6, COLOR_DEBRIS, thick)

    # ---- detections, coloured by the severity the engine assigned to their id
    objects: Dict[int, ObjectState] = {int(o.track_id): o for o in result.objects}
    warning_ids = list(assessment.warning_track_ids)
    alarm_ids = list(assessment.alarm_track_ids)
    for det in result.detections:
        x1, y1, x2, y2 = [int(round(float(v))) for v in det.xyxy]
        obj = objects.get(int(det.track_id)) if det.track_id is not None else None
        color = object_color(obj, warning_ids, alarm_ids)
        # Predicted (velocity-propagated) boxes on skipped detector frames are drawn
        # thinner so an operator can tell "seen" from "assumed".
        predicted = isinstance(det, PredictedDetection)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1 if predicted else thick + 1)
        label = object_label(obj, det.cls_name, det.track_id)
        if predicted:
            label = "~" + label
        _put_text(img, label, (x1, max(line_h, y1 - 4)), scale * 0.6, color, thick)
        # Bottom-center dot: the point that is actually tested against the zone.
        cx, cy = det.bottom_center
        cv2.circle(img, (int(round(cx)), int(round(cy))), max(2, thick + 1), color, -1)

    # ---- status banner
    banner_h = line_h + 6
    cv2.rectangle(img, (0, 0), (w, banner_h), status_color, -1)
    reasons = "; ".join(assessment.reasons) if assessment.reasons else "-"
    banner = f"{assessment.status.value}  t={result.timestamp:.1f}s f={result.frame_index}  {reasons}"
    max_chars = max(20, int(w / (11 * scale)))
    if len(banner) > max_chars:
        banner = banner[: max_chars - 3] + "..."
    _put_text(img, banner, (8, banner_h - 8), scale * 0.7, COLOR_TEXT, thick)

    # ---- gate state + zone health + timing, below the banner
    gate = result.gate
    if gate is not None and gate.available:
        gate_txt = f"gates {'DOWN' if gate.gates_down else 'up'} ({gate.confidence:.2f})"
        if gate.lights_on:
            gate_txt += " lights ON"
        gate_color = COLOR_ALARM if gate.gates_down else COLOR_TEXT
    else:
        gate_txt = "gates n/a"
        gate_color = COLOR_OTHER
    _put_text(img, gate_txt, (8, banner_h + line_h), scale * 0.6, gate_color, thick)

    if zone is not None:
        health = f"zone {zone.method} {zone.inliers} inl {'ok' if zone.healthy else 'HELD'}"
        health_color = COLOR_TEXT if zone.healthy else COLOR_ALARM
    else:
        health, health_color = "zone n/a", COLOR_OTHER
    total_ms = float(result.timings_ms.get("total", sum(result.timings_ms.values())))
    perf = f"{total_ms:.1f} ms"
    if fps_estimate is not None and fps_estimate > 0:
        perf += f" {fps_estimate:.1f} fps"
    _put_text(img, f"{health}  {perf}", (8, banner_h + 2 * line_h), scale * 0.6, health_color, thick)

    if assessment.latched_until is not None and "latched" in assessment.reasons:
        _put_text(img, "LATCHED", (w - int(120 * scale) - 8, banner_h + line_h), scale * 0.7, COLOR_ALARM, thick)
    return img


__all__ = ["draw_overlay", "object_color", "object_label", "STATUS_COLORS"]
