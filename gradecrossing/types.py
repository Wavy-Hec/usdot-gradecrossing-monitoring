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
    method: str  # "identity" | "lk" | "orb" | "held" | "uninitialized"
    healthy: bool  # True when the estimate met the inlier/geometry sanity checks
    area_px: int = 0
    # 3x3 float64 mapping pixels of the CURRENT processed frame to pixels of the original
    # reference (calibration) frame, composed across keyframe re-anchors and never
    # dead-zoned; the last accepted value while holding; None until the zone is initialised
    # (method "uninitialized", healthy False, while the first frames are featureless).
    # ``homography`` above is relative to the current keyframe, not to this reference.
    to_reference: Optional[np.ndarray] = None


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
    stuck_seconds: float  # continuous seconds simultaneously in_zone AND stopped (see StuckConfig.stationarity)
    # (t, x, y) bottom-center samples of the speed window; reference-frame pixels when the
    # zone provides ZoneState.to_reference, image pixels otherwise
    history: List[Tuple[float, float, float]] = field(default_factory=list)
    # Footprint relation of the last real detection (vehicles with zone_test="footprint").
    zone_overlap: float = 0.0  # share of the footprint inside the zone (0..1)
    zone_distance_bh: float = float("inf")  # footprint-to-zone distance in box heights (0 inside)
    near_zone: bool = False  # not in the zone but within close_box_heights of it ("too close")
    still_seconds: float = 0.0  # ID-free stationary time: the box kept re-appearing where it stood
    near_stuck_seconds: float = 0.0  # continuous seconds stopped while in the zone OR too close
    occluded: bool = False  # alarmed vehicle lost under a train box, kept as a hidden hazard
    # Serial of the physical track (> 0; 0 = unknown): kept when the tracker bridges an id
    # switch (re-keying, a hidden hazard re-detected), so a re-identified object is not "new".
    uid: int = 0


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
    unknown_fraction: float = 0.0  # zone share the model cannot judge (never seen box-free, off-frame, clipped)
    covered_fraction: float = 0.0  # zone share under exclusion boxes on this frame
    registered: bool = False  # the frame was warped into the background's reference geometry
    trusted: bool = True  # False while registration fails or unknown_fraction is too high
    gain: float = 1.0  # mean global gain (current / background) divided out before classification
    held: bool = False  # timers frozen this frame (train over the zone, registration lost)
    state: str = ""  # short text for status lines, e.g. "warmup 1.2/2.0s", "ok", "held (train)"


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

    # "ALARM_START" | "ALARM_UPDATE" | "ALARM_END" | "WARNING_START" | "WARNING_END" | "HEALTH"
    # | "RUN_START" | "RUN_END" (Monitor.run; the event-log contract in gradecrossing.evaluate)
    type: str
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
    # True while some part of the pipeline cannot fully see on this frame (zone not anchored
    # or held, debris backstop warming up / held / untrusted, detector slower than the still
    # clock needs, gate detector failing); the status may then read CLEAR only because the
    # monitor is partly blind. Per frame, not debounced: the monitor reports lasting changes
    # as HEALTH events (extra component='monitor'; zone health and debris warm-ups are left to
    # the zone's and the debris component's own events). ``degraded_reasons`` are short,
    # stable texts such as "zone held" or "debris held (train)".
    degraded: bool = False
    degraded_reasons: Tuple[str, ...] = ()
