"""Shared value types for the grade-crossing monitor.

Everything here is a plain dataclass so that modules can be unit-tested without
ultralytics/torch installed. Coordinates are image pixels (x right, y down).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np


class Status(str, Enum):
    CLEAR = "CLEAR"
    WARNING = "WARNING"
    ALARM = "ALARM"


@dataclass
class Detection:
    """One detector output box. ``track_id`` is None when no tracker assigned an id."""

    xyxy: np.ndarray  # shape (4,), float32, pixel coords x1,y1,x2,y2
    conf: float
    cls_id: int
    cls_name: str
    track_id: Optional[int] = None

    @property
    def bottom_center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = [float(v) for v in self.xyxy]
        return ((x1 + x2) / 2.0, y2)

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = [float(v) for v in self.xyxy]
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def height(self) -> float:
        return float(self.xyxy[3] - self.xyxy[1])

    @property
    def width(self) -> float:
        return float(self.xyxy[2] - self.xyxy[0])


@dataclass
class ZoneState:
    """Where the zone of interest is in the *current* frame."""

    polygon: np.ndarray  # (N,2) int32 pixel polygon in the current frame
    mask: np.ndarray  # HxW uint8, 255 inside the polygon
    homography: Optional[np.ndarray]  # 3x3 reference->current, None if identity/held
    inliers: int  # RANSAC inlier count (0 if none computed)
    method: str  # "identity" | "lk" | "orb" | "held"
    healthy: bool  # True when the estimate met the inlier/geometry sanity checks
    area_px: int = 0


@dataclass
class ObjectState:
    """Per-tracked-object measurements maintained by StuckVehicleTracker."""

    track_id: int
    cls_name: str
    is_vehicle: bool
    is_person: bool
    is_train: bool
    last_box: np.ndarray  # (4,) xyxy
    last_seen: float  # timestamp (s)
    first_seen: float
    in_zone: bool
    in_zone_since: Optional[float]  # timestamp when current in-zone streak began
    dwell_seconds: float  # continuous seconds inside zone (0 if outside)
    speed: float  # current speed estimate (px/s, or m/s when ground homography is set)
    speed_units: str  # "px/s" | "m/s"
    stationary_seconds: float  # continuous seconds with speed < threshold (any location)
    stuck_seconds: float  # continuous seconds simultaneously in_zone AND below speed threshold
    history: List[Tuple[float, float, float]] = field(default_factory=list)  # (t, x, y) bottom-center


@dataclass
class DebrisResult:
    mask: np.ndarray  # HxW uint8 debris mask (inside zone, after exclusions)
    area_px: int
    area_fraction: float  # area_px / zone area
    blobs: List[Tuple[int, int, int, int, int]]  # (x, y, w, h, area) of kept components
    persist_seconds: float  # continuous seconds area_fraction >= min_area_fraction
    triggered: bool  # persist_seconds >= persist_seconds threshold
    learning_rate_used: float
    warmed_up: bool


@dataclass
class GateState:
    gates_down: bool = False
    lights_on: bool = False
    confidence: float = 0.0
    last_update: Optional[float] = None  # timestamp of last detector run
    last_seen_down: Optional[float] = None  # timestamp gate-down was last detected
    available: bool = False  # False when no infrastructure detector is configured


@dataclass
class Assessment:
    status: Status
    reasons: List[str]
    warning_track_ids: List[int]
    alarm_track_ids: List[int]
    debris_alarm: bool
    person_alert: bool
    gates_down: bool
    latched_until: Optional[float]  # when an ALARM latch expires (None if not latched)
    severity_scale: float  # 1.0 normally, <1 when gates are down (thresholds shortened)


@dataclass
class Event:
    """Emitted on status transitions; serialised to JSONL by sinks."""

    type: str  # "ALARM_START" | "ALARM_END" | "WARNING_START" | "WARNING_END" | "HEALTH"
    timestamp: float
    frame_index: int
    status: str
    reasons: List[str]
    track_ids: List[int]
    extra: Dict[str, object] = field(default_factory=dict)


@dataclass
class FrameResult:
    frame_index: int
    timestamp: float
    assessment: Assessment
    zone: ZoneState
    detections: List[Detection]
    objects: List[ObjectState]
    debris: Optional[DebrisResult]
    gate: GateState
    timings_ms: Dict[str, float] = field(default_factory=dict)
    events: List[Event] = field(default_factory=list)
