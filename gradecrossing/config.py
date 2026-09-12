"""Configuration for the grade-crossing monitor.

All tunables live here as dataclasses with documented defaults. A YAML file overrides
any subset of fields (nested by section name). Nothing else in the package hardcodes a
pixel threshold: area thresholds are fractions of the zone area and time thresholds are
seconds, so the same config transfers between cameras and resolutions.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class VideoConfig:
    source: str = "0"  # file path, RTSP/HTTP URL, or camera index as a string
    fps_override: Optional[float] = None  # force a frame rate when the container reports 0/garbage
    resize_width: Optional[int] = None  # downscale frames to this width before processing (None = native)
    start_seconds: float = 0.0  # seek before processing (files only)
    max_frames: Optional[int] = None  # stop after N frames (None = run to end / forever)
    loop: bool = False  # loop a file (useful for soak tests)
    live: Optional[bool] = None  # None = auto: True for URLs/camera indices, False for files


@dataclass
class ZoneConfig:
    # Polygon in pixel coordinates of the *reference* frame (the frame the monitor
    # initialises on). Order the points around the boundary (clockwise or anticlockwise).
    polygon: List[List[float]] = field(default_factory=lambda: [[100, 400], [500, 350], [1650, 480], [1500, 600]])
    stabilize: bool = True
    # Shi-Tomasi feature detection on the reference keyframe
    max_corners: int = 400
    quality_level: float = 0.01
    min_distance: int = 20
    block_size: int = 7
    # Lucas-Kanade
    lk_win_size: int = 21
    lk_max_level: int = 3
    # RANSAC
    ransac_reproj_threshold: float = 3.0
    min_inliers: int = 30  # below this the LK estimate is considered unhealthy
    # An LK estimate must also keep at least this fraction of the reference features as
    # inliers: accepted only when ``inliers >= max(min_inliers, min_inlier_fraction *
    # number_of_reference_features)``. An absolute count alone is not enough, because LK
    # can lock a minority of points onto self-similar texture (rail sleepers, ballast,
    # road markings) with a *consistent* wrong flow and still clear min_inliers; when
    # most of the reference features disagree, the estimate must not be trusted.
    # 0 disables the relative test (absolute min_inliers only).
    min_inlier_fraction: float = 0.2
    # ORB fallback when LK is unhealthy (handles large appearance change)
    orb_fallback: bool = True
    orb_features: int = 1500
    orb_ratio_test: float = 0.75
    orb_min_matches: int = 25
    # Sanity limits for an accepted homography
    max_scale_change: float = 0.35
    max_area_change: float = 0.5
    # Reference keyframe refresh: after the scene has been static (no tracked objects,
    # healthy homography) for this many seconds, re-anchor the reference to the current
    # frame so slow lighting change does not erode the feature set. 0 disables.
    keyframe_refresh_static_seconds: float = 60.0
    # Exclude features that sit inside the zone (they belong to vehicles/trains, not background)
    exclude_features_in_zone: bool = True
    # Log a HEALTH event when the stabilizer transitions healthy <-> unhealthy
    health_events: bool = True


@dataclass
class DetectorConfig:
    weights: str = "yolo11n.pt"  # COCO-pretrained by default; or a fine-tuned .pt/.engine
    conf: float = 0.25
    iou: float = 0.5
    imgsz: int = 640
    device: str = ""  # "" = auto, "0" = first CUDA GPU, "cpu"
    half: bool = False  # FP16 on CUDA/TensorRT
    tracker: str = "bytetrack.yaml"  # ultralytics tracker config; "" disables tracking
    detect_every_n_frames: int = 1  # >1 skips detector frames and propagates boxes from track velocity
    # Class names (case-insensitive) treated as road vehicles for the stuck logic.
    # Covers COCO (car, truck, bus, motorcycle) and the fine-tuned CCTV_Vehicles names.
    vehicle_classes: List[str] = field(default_factory=lambda: [
        "car", "truck", "bus", "motorcycle", "motorbike", "jeep", "van", "tricycle", "bicycle"])
    person_classes: List[str] = field(default_factory=lambda: ["person", "pedestrian"])
    train_classes: List[str] = field(default_factory=lambda: ["train"])
    # Everything detected (any class) is excluded from the debris mask.
    max_det: int = 100


@dataclass
class InfrastructureConfig:
    enabled: bool = False
    weights: str = "notebooks/infrastructure tuning best weights/best.pt"
    conf: float = 0.35
    imgsz: int = 640
    device: str = ""
    half: bool = False
    every_n_frames: int = 15  # gate state changes slowly; no need to run every frame
    gate_down_classes: List[str] = field(default_factory=lambda: ["railroad-crossing-down"])
    light_on_classes: List[str] = field(default_factory=lambda: ["railroad-sign-light-on"])
    hold_seconds: float = 4.0  # gates_down stays True this long after the last gate-down detection


@dataclass
class StuckConfig:
    # Speed below which an object is considered stationary.
    speed_threshold: float = 12.0  # px/s at processing resolution (or m/s if ground homography set)
    # Speed is estimated by least-squares fit over the last ``speed_window_seconds`` of bottom-center positions.
    speed_window_seconds: float = 1.0
    min_history_for_speed: int = 3
    # Optional 3x3 homography mapping pixel -> ground-plane metres; when set speeds are in m/s.
    ground_plane_homography: Optional[List[List[float]]] = None
    # Stuck = in zone AND stationary continuously for this long.
    warning_seconds: float = 3.0
    alarm_seconds: float = 8.0
    # Dwell fallback: in zone this long regardless of speed (creeping/queued traffic on the tracks).
    dwell_warning_seconds: float = 12.0
    dwell_alarm_seconds: float = 25.0
    # Overrides applied when the infrastructure detector reports gates down.
    gates_down_warning_seconds: float = 1.0
    gates_down_alarm_seconds: float = 3.0
    gates_down_dwell_alarm_seconds: float = 6.0
    # Track bookkeeping
    lost_track_grace_seconds: float = 1.0  # keep an unseen track's state this long before dropping it
    min_track_age_frames: int = 2  # ignore brand-new ids (tracker flicker) for zone logic
    # Fallback IoU tracker used when detections carry no track ids
    fallback_iou_threshold: float = 0.3
    # Pedestrians
    person_warning_seconds: float = 2.0
    person_alarm_seconds: float = 10.0


@dataclass
class DebrisConfig:
    enabled: bool = True
    history: int = 300
    var_threshold: float = 50.0
    detect_shadows: bool = True
    warmup_frames: int = 300  # frames learned with the default rate before monitoring starts
    monitoring_learning_rate: float = 0.0  # 0 = background frozen while monitoring
    # When the tracker reports the zone empty for this long, let MOG2 adapt slowly so
    # lighting drift does not accumulate as false debris. 0 disables refresh.
    empty_seconds_before_refresh: float = 5.0
    empty_learning_rate: float = 0.005
    # Exclusion of detected objects from the foreground mask
    exclusion_dilate_px: int = 12
    # Morphology
    morph_kernel: int = 5
    # Thresholds as fractions of the zone area (resolution independent)
    min_blob_area_fraction: float = 0.002  # drop connected components smaller than this
    min_area_fraction: float = 0.03  # total debris area needed to start the persistence timer
    persist_seconds: float = 3.0  # continuous seconds over threshold before ALARM
    # Upper bound: foreground covering more than this fraction of the zone while the
    # tracker reports the zone empty cannot be an obstruction (nothing that big can sit
    # on a crossing unseen). It is an illumination step - a floodlight switching, a
    # day/night camera's IR-cut filter at dusk, an auto-exposure jump - which would
    # otherwise start the persistence timer, block the refresh and latch an ALARM that
    # can never clear. Instead the background model is re-warmed and a HEALTH event is
    # reported. 0 disables the check (the step then alarms as debris). See background.py.
    max_area_fraction: float = 0.75
    # Immobility criterion for candidate debris. Debris is by definition *static*: the
    # persistence timer only advances while the candidate blob pixels barely change from
    # one frame to the next. Without this, any object the detector misses (a train with
    # weights that have no "train" class, a crowd) raises a debris ALARM merely by taking
    # longer than persist_seconds to cross the zone.
    static_motion_diff: float = 12.0  # per-pixel grey-level change that counts as "moving"
    static_motion_fraction: float = 0.15  # blob fraction allowed to change before the timer restarts


@dataclass
class AlertConfig:
    latch_seconds: float = 10.0  # ALARM stays raised this long after the last alarm condition
    repeat_notify_seconds: float = 30.0  # re-send an ALARM notification while it persists (0 = never)
    stdout: bool = True
    events_jsonl: str = ""  # path for the JSONL event log ("" = disabled)
    webhook_url: str = ""  # POST JSON events here ("" = disabled)
    webhook_timeout_seconds: float = 3.0


@dataclass
class OutputConfig:
    display: bool = False  # cv2.imshow window
    save_video: str = ""  # annotated output video path ("" = disabled)
    snapshot_dir: str = ""  # save an annotated JPEG on every ALARM_START ("" = disabled)
    log_level: str = "INFO"
    print_every_n_frames: int = 0  # periodic status line to stdout (0 = off)


@dataclass
class MonitorConfig:
    video: VideoConfig = field(default_factory=VideoConfig)
    zone: ZoneConfig = field(default_factory=ZoneConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    infrastructure: InfrastructureConfig = field(default_factory=InfrastructureConfig)
    stuck: StuckConfig = field(default_factory=StuckConfig)
    debris: DebrisConfig = field(default_factory=DebrisConfig)
    alerts: AlertConfig = field(default_factory=AlertConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def _apply(section: Any, overrides: Dict[str, Any], path: str) -> None:
    valid = {f.name for f in fields(section)}
    for key, value in overrides.items():
        if key not in valid:
            raise KeyError(f"unknown config key '{path}.{key}' (valid: {sorted(valid)})")
        current = getattr(section, key)
        if is_dataclass(current) and isinstance(value, dict):
            _apply(current, value, f"{path}.{key}")
        else:
            setattr(section, key, value)


def config_from_dict(data: Optional[Dict[str, Any]]) -> MonitorConfig:
    """Build a MonitorConfig from a nested dict (e.g. parsed YAML). Unknown keys raise.

    ``None`` / an empty mapping returns the defaults. Anything else that is not a mapping
    (a YAML file whose top level is a list or a bare scalar) raises ``TypeError`` rather
    than an ``AttributeError`` deep inside this function, so the CLI can report it as a
    config error (exit status 2) instead of printing a traceback.
    """
    cfg = MonitorConfig()
    if not data:
        return cfg
    if not isinstance(data, dict):
        raise TypeError(
            f"config root must be a mapping of section name -> settings, got {type(data).__name__}")
    valid = {f.name for f in fields(cfg)}
    for section, overrides in data.items():
        if section not in valid:
            raise KeyError(f"unknown config section '{section}' (valid: {sorted(valid)})")
        if overrides is None:
            continue
        if not isinstance(overrides, dict):
            raise TypeError(f"config section '{section}' must be a mapping")
        _apply(getattr(cfg, section), overrides, section)
    return cfg


def load_config(path: Optional[str]) -> MonitorConfig:
    """Load a YAML config file; ``None`` returns defaults."""
    if path is None:
        return MonitorConfig()
    import yaml  # local import so the package imports without PyYAML for tests

    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return config_from_dict(data)


def save_config(cfg: MonitorConfig, path: str) -> None:
    import yaml

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg.to_dict(), fh, sort_keys=False)
