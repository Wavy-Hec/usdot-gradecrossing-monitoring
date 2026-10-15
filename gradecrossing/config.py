"""Configuration for the grade-crossing monitor.

All tunables live here as dataclasses with documented defaults. A YAML file overrides
any subset of fields (nested by section name). Most thresholds do not depend on the
camera: areas are fractions of the zone (or frame) area and times are seconds. A few are
pixels and are worth re-checking for a camera whose processed resolution is very
different: ``stuck.speed_threshold`` (px/s unless a ground homography is set), the zone
stabiliser's feature, Lucas-Kanade and RANSAC sizes and its ``max_step_px``,
``deadzone_px`` and ``reanchor_max_drift_px`` (pixels of the processed frame), and
``debris.exclusion_dilate_px`` / ``morph_kernel`` (pixels at ``debris.process_width``).

A file that sets a key removed since an earlier release (``RETIRED_KEYS``) fails to load;
one that sets a key to a default that has since changed (``CHANGED_DEFAULTS``) loads with
a warning, because it silently keeps the old behaviour.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
import numbers
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, get_args, get_origin, get_type_hints

LOGGER = logging.getLogger(__name__)


@dataclass
class VideoConfig:
    source: str = "0"  # file path, RTSP/HTTP URL, or camera index as a string
    fps_override: Optional[float] = None  # force a frame rate when the container reports 0/garbage
    resize_width: Optional[int] = None  # downscale frames to this width before processing (None = native)
    start_seconds: float = 0.0  # seek before processing (files only)
    max_frames: Optional[int] = None  # stop after N frames (None = run to end / forever)
    loop: bool = False  # loop a file (useful for soak tests)
    live: Optional[bool] = None  # None = auto: True for URLs/camera indices, False for files
    # Open and read timeout of a stream URL (rtsp/http/...; OpenCV 4.5.2+ with FFmpeg): a dead
    # camera fails each (re)connect attempt after this long, a frozen stream fails read().
    # Raise it for a camera whose keyframes are further apart. 0 = the backend default (30 s).
    stream_timeout_seconds: float = 10.0


@dataclass
class ZoneConfig:
    # Polygon in pixel coordinates of the *calibration* frame. Order the points around the
    # boundary (clockwise or anticlockwise). The default is the FOX13 stress-test clip's
    # polygon (1772x872); it is not meaningful for any other camera.
    polygon: List[List[float]] = field(default_factory=lambda: [[100, 400], [500, 350], [1650, 480], [1500, 600]])
    # Zone file in the team's JSON format, {"vertices": [[x, y], ...]} ("polygon" / "points"
    # keys and a bare [[x, y], ...] list load too), in the same calibration pixels. When set
    # it replaces polygon; the Monitor reads it once, when it is built. Relative paths resolve
    # against the working directory. "" = use polygon. CLI: --zone ZONE.json
    polygon_file: str = ""
    # [width, height] of the frame the polygon was drawn on (tools/calibrate_zone.py writes
    # it). The monitor scales the polygon by processed/reference size on both axes and
    # refuses to run when the aspect ratio differs. None = the native size of the first
    # frame (or of reference_image when that is set).
    reference_size: Optional[List[int]] = None
    # Clean calibration frame (no overlay) the polygon was drawn on. When set, live frames
    # are registered against it, so camera motion between calibration and start-up is
    # corrected on the first live frame. "" = anchor on the first live frame. A missing or
    # unreadable image is reported (HEALTH) and the first live frame is used instead.
    reference_image: str = ""
    # How long live frames are registered against reference_image (held meanwhile) before
    # the stabilizer gives up for now and tracks from a live frame at the calibration pose
    # (reported; registration is retried at every keyframe refresh). Night against a
    # daylight image is the usual reason.
    registration_timeout_seconds: float = 2.0
    stabilize: bool = True
    # Camera motion model: "similarity" (translation + rotation + uniform scale, right for a
    # fixed pole camera), "affine" or "homography" (8 DOF; fits perspective distortions to
    # whatever minority of points survives a passing train).
    motion_model: str = "similarity"
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
    # An ORB estimate must also be supported by this share of its ratio-test matches.
    orb_min_inlier_fraction: float = 0.5
    # Sanity limits for an accepted transform (relative to the calibration frame)
    max_scale_change: float = 0.35
    max_area_change: float = 0.5
    max_rotation_deg: float = 10.0
    max_translation_fraction: float = 0.25  # zone centroid shift, as a fraction of the frame's larger side
    # Temporal consistency: an estimate that moves a vertex more than
    # max(max_step_px, max_step_fraction * zone thickness) from the last accepted polygon
    # in one frame is held (healthy=False) unless the frame itself refutes the last pose
    # (the camera really moved) or confirm_frames consecutive estimates agree on the new
    # pose. confirm_frames counts estimates, not seconds: it is how many independent
    # measurements must agree. confirm_frames <= 1 or both limits 0 disables the gate.
    max_step_px: float = 3.0
    max_step_fraction: float = 0.1
    confirm_frames: int = 3
    # Reported polygon only: motion smaller than this (largest vertex displacement)
    # snaps to the previous polygon, so a static camera gives a perfectly still zone.
    # ZoneState.to_reference (debris registration) is never dead-zoned. 0 disables.
    deadzone_px: float = 1.0
    # Reference keyframe refresh: after the scene has been static (no tracked objects,
    # healthy estimate) for this many seconds, re-anchor the keyframe to the current
    # frame so slow lighting change does not erode the feature set. 0 disables.
    keyframe_refresh_static_seconds: float = 60.0
    # A re-anchor is checked against a direct estimate from the calibration frame and
    # refused when the two disagree by more than this (largest vertex distance, px).
    reanchor_max_drift_px: float = 1.0
    # Exclude features that sit inside the zone (they belong to vehicles/trains, not background)
    exclude_features_in_zone: bool = True
    # Emit zone HEALTH events: health changes (healthy <-> held/uninitialised) and the
    # reference_image registration notices
    health_events: bool = True
    # A health change is reported only once it has lasted this long, so single held frames
    # and an estimate flickering in deep low light do not flood the sinks (ZoneState.healthy
    # itself stays per frame). 0 = report every change.
    health_report_seconds: float = 0.5


@dataclass
class DetectorConfig:
    weights: str = "yolo11n.pt"  # COCO-pretrained by default; or a fine-tuned .pt/.engine
    # Decision threshold for plain detection (tracker "") and for boxes the tracker returns
    # without an id; also what the tracker is fed unless track_conf is set.
    conf: float = 0.25
    # Score threshold handed to the tracker (model.track); None = conf. ByteTrack associates
    # boxes scoring between track_low_thresh (0.1) and track_high_thresh (0.25) in a second
    # stage that keeps a partly occluded vehicle on its id; fed at 0.25 that stage is empty.
    # New tracks still need new_track_thresh (0.25 in bytetrack.yaml). Measured on the 8 team
    # clips, 0.1 cut ByteTrack id restarts 71 -> 63 but added 9-26 s of false ALARM (over 461 s
    # of negatives, depending on the other branches merged) and delayed the clip08 WARNING by
    # 0.56 s, so it is opt-in. Unused when tracker is "".
    track_conf: Optional[float] = None
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
    # Class names (case-insensitive) dropped right after detection: never tracked, never
    # alerted on and never blocking the zone keyframe refresh. For static false positives
    # inside the zone, e.g. COCO "traffic light" on the crossing signals. The debris mask
    # excludes only vehicle, person and train boxes (and a track that had one of those labels
    # in the last few seconds); every other class, ignored or not, stays visible to it.
    ignore_classes: List[str] = field(default_factory=list)
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
    # Stuck = in zone AND stopped (see ``stationarity``) continuously for this long. ALARM at
    # 3 s is the advisor's task definition ("stops on the railway tracks ... for more than
    # 3 s") and the operating point the team measured (EXPERIMENTS.md E8). The 1.5 s WARNING is
    # provisional: it rests on one trapped-van event seen by two cameras (clip07/clip08).
    warning_seconds: float = 1.5
    alarm_seconds: float = 3.0
    # Dwell fallback: in zone this long regardless of speed (creeping/queued traffic on the tracks).
    dwell_warning_seconds: float = 12.0
    dwell_alarm_seconds: float = 25.0
    # Overrides applied when the infrastructure detector reports gates down. A gates-down
    # threshold only escalates when it is shorter than the normal one (the AlertEngine logs a
    # warning otherwise): with the 3 s ALARM, gates down no longer shortens the stuck alarm.
    # The gate model is not validated on the team clips (it is blind on clip01), which is
    # why infrastructure.enabled stays false.
    gates_down_warning_seconds: float = 1.0
    gates_down_alarm_seconds: float = 3.0
    gates_down_dwell_alarm_seconds: float = 6.0
    # Track bookkeeping
    # Drop a track once lost_track_missed_runs detector runs in a row have missed it AND its
    # last detection is older than lost_track_grace_seconds. Frames on which the detector is
    # skipped never age a track out. With the detector on every frame the grace decides; with
    # sparse detection (5 fps, detect_every_n_frames 7: runs 1.4 s apart) the run count does,
    # so one missed run does not restart a parked car's clocks.
    lost_track_grace_seconds: float = 1.0
    lost_track_missed_runs: int = 2
    # Until then, a live in-zone vehicle that a detector run missed keeps its dwell and stuck
    # clocks running from their start instead of freezing them at its last detection until it
    # is seen again. The trapped van needs it (clip07: hidden by the train from 5.0 s): with
    # the still-cluster re-anchoring (code, not a key) clip07 and clip08 ALARM at 5.61 s and
    # 5.84 s instead of WARNING only. Measured cost on the 8 team clips: +12.9 s of false ALARM
    # (clip06's two zone-geometry false ALARMs merge into one). false = freeze
    # (configs/legacy_rule.yaml).
    undetected_clock_while_live: bool = True
    min_track_age_frames: int = 2  # ignore brand-new ids (tracker flicker) for zone logic
    # Fallback IoU tracker used when detections carry no track ids
    fallback_iou_threshold: float = 0.3
    # Pedestrians
    person_warning_seconds: float = 2.0
    person_alarm_seconds: float = 10.0
    # --- Where a vehicle is relative to the zone (team EXPERIMENTS.md E4/E6) -------------
    # "footprint": the bottom footprint_fraction of the box (full width) must overlap the
    # zone by >= footprint_min_overlap. "bottom_center": the pre-port single-point test.
    # Persons, trains and unmapped classes always use the bottom-center point.
    zone_test: str = "footprint"
    footprint_fraction: float = 0.25
    footprint_min_overlap: float = 0.05
    # "Too close": a vehicle not on the zone whose footprint is within this many box heights
    # of it (0 disables). 0.75 was chosen independently on each of two sites (E6).
    close_box_heights: float = 0.75
    # What a vehicle stopped too close raises: "off", "warning" (WARNING after
    # warning_seconds; the advisor's three-level policy) or "alarm" (the team's "merged"
    # policy: ALARM after alarm_seconds).
    close_policy: str = "warning"
    # A vehicle in the zone stays in it until its real detections have been out for longer
    # than this (one jittering frame on the zone edge does not restart its clocks). 0 = exit
    # on the first out-of-zone detection. Also holds the on-or-near streak of close_policy.
    zone_exit_grace_seconds: float = 0.5
    # --- How "stopped" is decided ----------------------------------------------------------
    # "speed": least-squares speed < speed_threshold; "still": ID-free - the box keeps
    # re-appearing with IoU >= still_min_iou to where it was first seen, gaps <=
    # still_max_gap_seconds (+9 recall points over tracker ids, E6); "speed_or_still": either.
    stationarity: str = "speed_or_still"
    still_min_iou: float = 0.5
    still_max_gap_seconds: float = 1.0  # must exceed the detector interval (a HEALTH event says when not)
    # A vehicle box reaching within this fraction of the frame width (left/right edges) or
    # height (top/bottom) of the image border restarts its still cluster's clock while a
    # train box was seen beside it within the last 2 s: the visible end of a train passing
    # the edge of the picture, labelled bus/truck, keeps one box while the train moves
    # (clip06: 10 s of false "stopped", an 11 s false ALARM). The speed rule alone decides
    # while that lasts. A vehicle stopped at the edge with no train beside it keeps its still
    # clock (most team zones span the full frame width). 0 = off.
    still_border_margin: float = 0.01
    # --- Identity and labels -----------------------------------------------------------------
    # Which live track a brand-new detector id may take over (an id swap in place):
    # "strict" = a vehicle box only a stopped vehicle track, a person box only a person track,
    # with the clocks handed back if the old id reappears while the take-over is recent;
    # "any" = any unclaimed track of any class (the pre-port rule).
    rekey_policy: str = "strict"
    # A stopped in-zone track labelled vehicle for at least this long keeps its vehicle
    # category through a shorter run of other labels (car -> train/boat flicker). 0 = always
    # follow the latest label. There is no vote: passing trains labelled bus/car stay trains
    # only as long as their label says so. A vehicle of the ALARM episode in progress (latched
    # included) keeps its category while it is in or near the zone, however long the run (a
    # stuck car merging with the arriving train is labelled 'train' for seconds).
    label_hold_seconds: float = 1.0
    # An ALARMED vehicle (one of the ALARM episode in progress, latched included) whose track
    # is lost in or near the zone under a train box is kept as a hidden hazard:
    # ALARM is held ("occluded by train") while a train box overlaps its last box and for this
    # long after; a re-detection at IoU >= 0.5 within that time restores its clocks. 0 = off.
    # Measured cost on the 8 team clips: no true alarm changes; false ALARMs already raised by
    # far-side cars waiting at the gate inside the thin hand-drawn zones are held through the
    # train passage: +58 s of false ALARM over 461 s of negatives (189 -> 247 s; 7 episodes
    # merge into 5). On by default for the vehicle the train really hides; 0 trades that hold
    # for fewer false-alarm seconds until the zones cover only the rails (a team decision).
    occlusion_memory_seconds: float = 3.0
    # When the zone provides ZoneState.to_reference (current-frame -> reference-frame pixels),
    # fit speeds and match still clusters in reference coordinates, so camera motion is not
    # read as vehicle motion. false = raw image coordinates.
    camera_motion_compensation: bool = True


@dataclass
class DebrisConfig:
    enabled: bool = True
    history: int = 300  # MOG2 history (sets the automatic learning rate of the warm-up)
    var_threshold: float = 50.0
    detect_shadows: bool = True
    # Warm-up: the scene is learned for this long before monitoring starts. Pixels under a
    # (dilated) detection box are never learned; a pixel that was box-free for less than
    # half of the warm-up stays UNKNOWN until it has been seen box-free (see learn_seconds).
    # 2 s: long enough for a stable MOG2 model, short enough to finish before the gate arms
    # move (3-5 s on the team's clips) and well inside the 7-8 s positive clips.
    warmup_seconds: float = 2.0
    warmup_frames: Optional[int] = None  # legacy override: a frame count instead of warmup_seconds
    monitoring_learning_rate: float = 0.0  # rate of the per-frame classification apply (0 = frozen)
    # Selective background update: every update_interval_seconds the model learns, with this
    # time constant, ONLY the pixels classified background that lie outside the dilated
    # detection boxes and outside every debris candidate; everything else is pasted from the
    # model's own background. Lighting drift is followed even on a crossing that is never
    # empty, and no foreground pixel in the zone is ever absorbed. 0 disables the update.
    update_time_constant_seconds: float = 30.0
    update_interval_seconds: float = 0.5
    # An UNKNOWN pixel (never seen box-free, or a healed ghost) joins the model once it has been
    # box-free and classified background for this long; until then it is excluded from debris.
    learn_seconds: float = 1.0
    # Exclusion of detected objects from the foreground mask
    exclusion_dilate_px: int = 12
    # Morphology
    morph_kernel: int = 5
    # Size thresholds. Debris must cover min(min_area_fraction of the zone,
    # min_object_frame_fraction of the whole processing frame): the zone share alone missed a
    # 30x24 px object on a large zone, the frame share keeps that bar resolution independent.
    min_blob_area_fraction: float = 0.002  # drop connected components smaller than this share of the zone
    min_area_fraction: float = 0.03  # debris area (share of the zone) that starts the persistence timer...
    min_object_frame_fraction: float = 0.002  # ...or this share of the processing frame, whichever is smaller
    persist_seconds: float = 3.0  # seconds a candidate must be seen static before a debris ALARM
    # A candidate covered by a detection box (or briefly unobservable) keeps its persistence
    # instead of restarting, for at most this long; the covered time does not count.
    occlusion_bridge_seconds: float = 5.0
    # Global illumination: per-channel gain (current / background, estimated on background
    # pixels outside the zone and outside boxes) is divided out before classification, so
    # exposure ramps and steps do not become debris. Pixels the gain would clip are unknown.
    gain_compensation: bool = True
    # Re-warm the model when a channel gain stays beyond [1/rewarm_gain, rewarm_gain] for 1 s
    # (the compensation is no longer trustworthy). 0 disables.
    rewarm_gain: float = 2.0
    # Change compensation cannot explain: foreground over more than this fraction of the
    # zone AND of the modelled region outside the zone (an IR-cut filter switch, floodlights)
    # re-warms the model with a HEALTH event instead of latching an ALARM. An obstruction,
    # however big, leaves the rest of the scene alone. Never while debris is triggered. 0 disables.
    max_area_fraction: float = 0.75
    # Immobility criterion for candidate debris. Debris is by definition *static*: a candidate
    # restarts while more than static_motion_fraction of its pixels changed by more than
    # static_motion_diff grey levels against a frame static_motion_lag_seconds old (beyond a
    # +-1 px tolerance, so sub-pixel camera jitter is not motion), and a pixel only ages while
    # it stays foreground, so a sliding outline never persists. Without this, any object the
    # detector misses (a train under weights with no "train" class, a crowd) alarms by being slow.
    static_motion_diff: float = 12.0  # per-pixel grey-level change that counts as "moving"
    static_motion_fraction: float = 0.15  # blob fraction allowed to change before the timer restarts
    static_motion_lag_seconds: float = 0.5  # age of the comparison frame (motion must accumulate)
    # The monitor sets hold while a train is detected over the zone: timers freeze, no new
    # candidates, no learning. The hold lasts this long after the last such frame: 3 s bridges
    # most gaps between the detections of a passing train (clip03: up to 3.5 s), at the cost of
    # debris being blind that long after it. A hold that starts again less than persist_seconds
    # after the last one ended also restarts the candidates of the gap (the same train).
    hold_linger_seconds: float = 3.0
    # Ghost healing: a candidate whose outline exists only in the background model (something
    # learned during the warm-up that has left) is re-learned after this long - before
    # persist_seconds, so the ghost raises no ALARM; never while debris is triggered. 0 disables.
    ghost_heal_seconds: float = 2.0
    # ...but only if its model content had been seen as background for less than this before it
    # changed (a caption that vanishes, something that moves off right after start-up). An
    # obstruction arrives on road the model has watched for longer, so it is never healed;
    # the image evidence alone confuses a dark object across a bright rail with a rail-shaped
    # ghost. 0 = no age limit (heal on image evidence alone).
    ghost_heal_max_seen_seconds: float = 5.0
    # DebrisResult.trusted is False above this unknown share of the zone (or while the frame
    # cannot be registered); the monitor reports the flips as HEALTH events.
    max_unknown_fraction: float = 0.5
    # Frames wider than this are processed at this width (the zone, boxes and results are
    # scaled); keeps the cost of HD sources bounded. 0 = native resolution.
    process_width: int = 640
    # Polygons (calibration pixels, like zone.polygon) excluded from debris decisions, e.g.
    # where a lowered gate arm rests inside the zone. Default: none.
    ignore_polygons: List[List[List[float]]] = field(default_factory=list)


@dataclass
class AlertConfig:
    latch_seconds: float = 10.0  # ALARM stays raised this long after the last alarm condition
    repeat_notify_seconds: float = 30.0  # re-send an ALARM notification while it persists (0 = never)
    # WARNING stays raised this long after its last condition while one of its objects is
    # still in (or near) the zone, so a vehicle that has not moved does not flap the
    # WARNING when its stuck clock blips (0 = no hold).
    warning_hold_seconds: float = 2.0
    # The monitor's DEGRADED state (FrameResult.degraded) is reported as a HEALTH event only
    # once a change has lasted this long (0 = every change).
    health_debounce_seconds: float = 1.0
    stdout: bool = True
    events_jsonl: str = ""  # path for the JSONL event log ("" = disabled)
    webhook_url: str = ""  # POST JSON events here ("" = disabled)
    webhook_timeout_seconds: float = 3.0  # per socket operation (connect, send, each read)
    # Failed POSTs (refused, timeout, HTTP 5xx/408/429) are retried: webhook_attempts in
    # all, the first retry after webhook_backoff_seconds and the wait doubling each time,
    # all within webhook_deadline_seconds of wall-clock time per event (None = attempts x
    # timeout + the back-off waits); the attempt still running at the deadline is aborted.
    webhook_attempts: int = 3
    webhook_backoff_seconds: float = 0.5
    webhook_deadline_seconds: Optional[float] = None
    webhook_max_queue: int = 256  # events awaiting delivery; when full the oldest non-ALARM one is dropped


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


# --------------------------------------------------------------------------------------
# Loading and validation
# --------------------------------------------------------------------------------------
class ConfigError(ValueError, TypeError):
    """A config value of the wrong type or shape; the message names the dotted key.

    A ``ValueError`` (the CLI reports every config error as exit status 2), and also a
    ``TypeError``, which is what a section that is not a mapping raised before.
    """


# Keys removed or renamed since the package was first published (ae66b8f), with what to do
# instead. A config that still sets any of them fails to load with one ConfigError that
# lists them all, with these hints. Add a row whenever a key is removed or renamed.
RETIRED_KEYS: Dict[str, str] = {
    "debris.empty_seconds_before_refresh": (
        "removed: the background no longer waits for an empty zone to adapt; the selective update "
        "(debris.update_time_constant_seconds, debris.update_interval_seconds) follows the scene all the time"),
    "debris.empty_learning_rate": (
        "removed: the selective update's time constant, debris.update_time_constant_seconds, sets how fast "
        "the background adapts"),
}

# Defaults that have changed since an earlier release: dotted key -> (the old default, the
# last release whose default it was). A file that still sets the old value keeps the old
# behaviour without saying so - tools/calibrate_zone.py of 0.2.0 copied every default into
# each site file, and people copy configs/default.yaml - so load_config warns about it (see
# stale_defaults). Add a row whenever a default changes; tests/test_config_defaults.py
# checks the table against the full 0.2.0 defaults.
CHANGED_DEFAULTS: Dict[str, Tuple[Any, str]] = {
    "stuck.warning_seconds": (3.0, "0.2.0"),
    "stuck.alarm_seconds": (8.0, "0.2.0"),
    "debris.warmup_frames": (300, "0.2.0"),  # 300 frames = 10 s at 30 fps; now warmup_seconds 2.0 decides
}

# Shapes the type annotations cannot express, checked after the types: "polygon" = at least
# 3 finite [x, y] points, "polygons" = a list of those, a tuple = the length at each level.
_SHAPES: Dict[str, Any] = {
    "zone.polygon": "polygon",
    "zone.reference_size": (2,),
    "stuck.ground_plane_homography": (3, 3),
    "debris.ignore_polygons": "polygons",
}

# Where a zone file keeps its vertices, in lookup order: the keys the team's loaders read
# (occupancy_monitor zone.py, Gael's extract_roi.py).
ZONE_FILE_KEYS = ("vertices", "polygon", "points")


def _show(value: Any) -> str:
    """A config value for an error message: its (shortened) repr and its type."""
    text = repr(value)
    return f"{text if len(text) <= 80 else text[:77] + '...'} ({type(value).__name__})"


def _is_number(value: Any) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def _describe(tp: Any) -> str:
    """An annotation as error messages show it: 'float', 'list[str]', 'int or null'."""
    origin, args = get_origin(tp), get_args(tp)
    if origin is Union:
        return " or ".join([_describe(a) for a in args if a is not type(None)]
                           + (["null"] if type(None) in args else []))
    if origin in (list, tuple):
        return f"list[{_describe(args[0])}]" if args else "list"
    if origin is dict:
        return "mapping"
    return getattr(tp, "__name__", str(tp))


def _mismatch(value: Any, tp: Any, key: str) -> Optional[tuple]:
    """First place where ``value`` does not match annotation ``tp``: (dotted key, expected,
    value), or None. Annotations this checker does not know are not checked."""
    origin, args = get_origin(tp), get_args(tp)
    if origin is Union:
        if value is None and type(None) in args:
            return None
        found = None
        for option in [a for a in args if a is not type(None)]:
            found = _mismatch(value, option, key)
            if found is None:
                return None
        # Name the wrong element of a list (zone.polygon[2]), else the whole expected type.
        return found if found is not None and found[0] != key and len(args) <= 2 else (key, _describe(tp), value)
    if tp is bool:
        ok = isinstance(value, bool)
    elif tp is int:
        ok = isinstance(value, numbers.Integral) and not isinstance(value, bool)
    elif tp is float:
        ok = _is_number(value)  # an int is a float here (YAML writes 3 for 3.0)
    elif tp is str:
        ok = isinstance(value, str)
    elif origin in (list, tuple) or tp in (list, tuple):
        if hasattr(value, "tolist") and not isinstance(value, (list, tuple)):
            value = value.tolist()  # a numpy array set in code
        if not isinstance(value, (list, tuple)):
            return key, _describe(tp), value
        item_tp = args[0] if args else Any
        for i, item in enumerate(value):
            found = _mismatch(item, item_tp, f"{key}[{i}]")
            if found is not None:
                return found
        return None
    elif is_dataclass(tp) and isinstance(tp, type):
        ok = isinstance(value, tp)
    elif origin is dict or tp is dict:
        ok = isinstance(value, dict)
    else:
        return None
    return None if ok else (key, _describe(tp), value)


def _polygon_points(value: Any, what: str) -> List[List[float]]:
    """``value`` as [[x, y], ...] floats; ConfigError unless >= 3 finite numeric [x, y] points."""
    if hasattr(value, "tolist") and not isinstance(value, (list, tuple)):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        raise ConfigError(f"{what}: expected a list of [x, y] points, got {_show(value)}")
    points: List[List[float]] = []
    for i, point in enumerate(value):
        if not isinstance(point, (list, tuple)) or len(point) != 2 or not all(_is_number(v) for v in point):
            raise ConfigError(f"{what}: point {i} must be [x, y] (two numbers), got {_show(point)}")
        x, y = float(point[0]), float(point[1])
        if not (math.isfinite(x) and math.isfinite(y)):
            raise ConfigError(f"{what}: point {i} is not finite, got {_show(point)}")
        points.append([x, y])
    if len(points) < 3:
        raise ConfigError(f"{what}: a polygon needs at least 3 [x, y] points, got {len(points)}")
    return points


def _check_lengths(value: Any, lengths: tuple, key: str) -> None:
    if len(value) != lengths[0]:
        raise ConfigError(f"config key {key}: expected {lengths[0]} values, got {len(value)}: {_show(value)}")
    for i, row in enumerate(value if len(lengths) > 1 else ()):
        _check_lengths(row, lengths[1:], f"{key}[{i}]")


def _validate_section(section: Any, prefix: str) -> None:
    hints = get_type_hints(type(section))  # this module's annotations are strings (postponed)
    for f in fields(section):
        key, value, tp = prefix + f.name, getattr(section, f.name), hints.get(f.name, Any)
        if is_dataclass(tp) and isinstance(tp, type):  # a nested section
            if not isinstance(value, tp):
                raise ConfigError(f"config key {key}: expected a mapping of {tp.__name__} settings, got {_show(value)}")
            _validate_section(value, key + ".")
            continue
        found = _mismatch(value, tp, key)
        if found is not None:
            bad_key, expected, bad = found
            hint = ""
            if expected == "str" and _is_number(bad):
                hint = "; quote it in YAML"
            elif expected.startswith("list") and isinstance(bad, str):
                hint = "; write a YAML list, e.g. [a, b]"
            raise ConfigError(f"config key {bad_key}: expected {expected}, got {_show(bad)}{hint}")
        shape = _SHAPES.get(key)
        if shape == "polygon":
            _polygon_points(value, f"config key {key}")
        elif shape == "polygons":
            for i, polygon in enumerate(value):
                _polygon_points(polygon, f"config key {key}[{i}]")
        elif shape is not None and value is not None:
            _check_lengths(value, shape, key)


def validate_config(cfg: MonitorConfig) -> MonitorConfig:
    """Check every value of ``cfg`` against its dataclass field's type; return ``cfg``.

    int / float fields take numbers, never a bool or a string (an int is fine for a float);
    bool fields take true/false, never 0/1 or a string; str fields take strings;
    ``Optional[...]`` also takes None (null); a list field takes a list (a scalar there is
    the classic slip: ``vehicle_classes: car`` would make every vehicle 'other' and silence
    every stuck-vehicle alarm) whose items are checked the same way; polygons need at least
    3 finite numeric [x, y] points; fixed-length lists (zone.reference_size,
    stuck.ground_plane_homography) need their length; sections must be mappings.

    Raises :class:`ConfigError` (a ValueError) naming the dotted key, the expected type and
    the value. :func:`config_from_dict` / :func:`load_config` run it on every file and
    ``Monitor`` runs it again when it is built, so a config assembled in code is checked
    the same way. No file is read here (see :func:`zone_polygon`).
    """
    if not isinstance(cfg, MonitorConfig):
        raise ConfigError(f"expected a MonitorConfig, got {_show(cfg)}")
    _validate_section(cfg, "")
    return cfg


def load_zone_file(path: str) -> List[List[float]]:
    """Read a zone polygon from a JSON file (``zone.polygon_file``, ``--zone``).

    The team's format ``{"vertices": [[x, y], ...], ...}`` (the vertex list may also be under
    ``"polygon"`` or ``"points"``) or a bare ``[[x, y], ...]`` list, in calibration pixels
    like ``zone.polygon``. Returns the vertices as floats. Raises ``OSError`` when the file
    cannot be read and ``ValueError`` (:class:`ConfigError`) unless it holds at least 3
    finite numeric [x, y] vertices. Standard library only.
    """
    with open(path, "r", encoding="utf-8-sig") as fh:  # -sig: a BOM from a Windows editor is fine
        text = fh.read()
    what = f"zone file {str(path)!r}"
    try:
        data = json.loads(text)
    except ValueError as exc:
        raise ConfigError(f"{what} is not valid JSON: {exc}") from None
    if isinstance(data, dict):
        key = next((k for k in ZONE_FILE_KEYS if k in data), None)
        if key is None:
            raise ConfigError(f"{what} has no vertex list: expected one of the keys {list(ZONE_FILE_KEYS)}, "
                              f"found {sorted(data)}")
        data = data[key]
    return _polygon_points(data, what)


def zone_polygon(zone_cfg: ZoneConfig) -> List[List[float]]:
    """The zone polygon in calibration pixels, as floats: the contents of
    ``zone_cfg.polygon_file`` when that is set, else (a copy of) ``zone_cfg.polygon``.
    Raises like :func:`load_zone_file`. ``Monitor`` calls it once, when it is built."""
    if zone_cfg.polygon_file:
        return load_zone_file(zone_cfg.polygon_file)
    return _polygon_points(zone_cfg.polygon, "config key zone.polygon")


def zone_file_argument(path: str) -> str:
    """argparse ``type=`` of ``--zone``: the path, once the file loads; otherwise argparse
    exits with status 2 and the reason, before any config, frame or model is loaded."""
    try:
        load_zone_file(path)
    except (OSError, ValueError) as exc:
        raise argparse.ArgumentTypeError(f"cannot load zone: {exc}") from None
    return path


def _unknown_key(dotted: str, valid: Any) -> KeyError:
    hint = RETIRED_KEYS.get(dotted)
    return KeyError(f"unknown config key '{dotted}' ({hint if hint else f'valid: {sorted(valid)}'})")


def _settings(data: Dict[str, Any], section: Any, prefix: str = "") -> Dict[str, Any]:
    """Dotted key -> value of every setting a config mapping makes, sections flattened.

    A mapping is descended only where the config has a section of that name, so an unknown
    or retired key is a leaf like any value; nothing here raises."""
    names = {f.name for f in fields(section)} if is_dataclass(section) else set()
    out: Dict[str, Any] = {}
    for key, value in data.items():
        dotted = f"{prefix}{key}"
        current = getattr(section, key) if key in names else None
        if isinstance(value, dict) and is_dataclass(current):
            out.update(_settings(value, current, dotted + "."))
        else:
            out[dotted] = value
    return out


def _same_value(a: Any, b: Any) -> bool:
    """``a == b`` as YAML means it: 8 equals 8.0, but true is not 1."""
    if isinstance(a, bool) != isinstance(b, bool):
        return False
    try:
        return bool(a == b)
    except Exception:  # noqa: BLE001 - e.g. an array compared element-wise
        return False


def _yaml_value(value: Any) -> str:
    try:
        return json.dumps(value)  # null, true, 8.0, "text"
    except (TypeError, ValueError):
        return repr(value)


def stale_defaults(data: Any) -> List[Tuple[str, Any, Any, str]]:
    """The settings of config mapping ``data`` (nested by section, as in a YAML file) that
    still hold an old default listed in ``CHANGED_DEFAULTS``: ``(dotted key, value, today's
    default, release whose default the value was)``, in table order. Nothing is checked or
    raised here; ``load_config`` warns about what this finds."""
    if not isinstance(data, dict):
        return []
    defaults = MonitorConfig()
    settings = _settings(data, defaults)
    found = []
    for key, (old, release) in CHANGED_DEFAULTS.items():
        if key not in settings or not _same_value(settings[key], old):
            continue
        now: Any = defaults
        try:
            for part in key.split("."):
                now = getattr(now, part)
        except AttributeError:  # a row for a key that was later removed: RETIRED_KEYS reports it
            continue
        if not _same_value(now, old):
            found.append((key, settings[key], now, release))
    return found


def _describe_stale(stale: List[Tuple[str, Any, Any, str]]) -> str:
    return ", ".join(f"{key} {_yaml_value(value)} (the {release} default; now {_yaml_value(now)})"
                     for key, value, now, release in stale)


def _retired_error(retired: List[str], data: Dict[str, Any]) -> ConfigError:
    """One error for every retired key a config sets; when it also pins old defaults, it is
    almost certainly a full-default site file from an earlier release, and says so."""
    many = len(retired) > 1
    text = (f"retired config key{'s' if many else ''} (delete {'them' if many else 'it'}): "
            + "; ".join(f"'{key}' ({RETIRED_KEYS[key]})" for key in retired))
    stale = stale_defaults(data)
    if stale:
        releases = "/".join(sorted({release for _key, _value, _now, release in stale}))
        text += (f". The file also sets {_describe_stale(stale)}: it looks like a full-default site file of "
                 f"gradecrossing {releases}, whose tools/calibrate_zone.py copied every default into it. Keep only "
                 "the keys this site sets on purpose (video.source, zone, ...) or write a new file with "
                 "tools/calibrate_zone.py, which now writes only those; configs/legacy_rule.yaml restores the 0.2.0 "
                 "vehicle rule (WARNING 3 s, ALARM 8 s) if the site needs it")
    return ConfigError(text)


def _apply(section: Any, overrides: Dict[str, Any], path: str) -> None:
    valid = {f.name for f in fields(section)}
    for key, value in overrides.items():
        if key not in valid:
            raise _unknown_key(f"{path}.{key}", valid)
        current = getattr(section, key)
        if is_dataclass(current):
            if not isinstance(value, dict):
                raise ConfigError(f"config key {path}.{key}: expected a mapping of settings, got {_show(value)}")
            _apply(current, value, f"{path}.{key}")
        else:
            setattr(section, key, value)  # checked by validate_config once every key is set


def config_from_dict(data: Optional[Dict[str, Any]]) -> MonitorConfig:
    """Build a validated MonitorConfig from a nested dict (e.g. parsed YAML).

    ``None`` / an empty mapping returns the defaults. Unknown keys raise ``KeyError``; keys
    that were removed or renamed (``RETIRED_KEYS``) raise one :class:`ConfigError` that
    lists every one the mapping sets, with what replaced each (and, when the mapping also
    holds old ``CHANGED_DEFAULTS`` values, that it looks like a full-default site file of an
    earlier release). A value of the wrong type or shape raises :class:`ConfigError` (see
    :func:`validate_config`). Anything else that is not a mapping (a YAML file whose top
    level is a list or a bare scalar) raises :class:`ConfigError`, which is also a
    ``TypeError``, rather than an ``AttributeError`` deep inside this function, so the CLI
    can report it as a config error (exit status 2) instead of printing a traceback.
    """
    cfg = MonitorConfig()
    if not data:
        return cfg
    if not isinstance(data, dict):
        raise ConfigError(
            f"config root must be a mapping of section name -> settings, got {type(data).__name__}")
    retired = [key for key in _settings(data, cfg) if key in RETIRED_KEYS]
    if retired:
        raise _retired_error(retired, data)
    valid = {f.name for f in fields(cfg)}
    for section, overrides in data.items():
        if section not in valid:
            raise KeyError(f"unknown config section '{section}' (valid: {sorted(valid)})")
        if overrides is None:
            continue
        if not isinstance(overrides, dict):
            raise ConfigError(f"config section '{section}' must be a mapping, got {_show(overrides)}")
        _apply(getattr(cfg, section), overrides, section)
    return validate_config(cfg)


def load_config(path: Optional[str]) -> MonitorConfig:
    """Load a YAML config file and validate it; ``None`` returns defaults.

    A ``zone.polygon_file`` is read here too, so a missing or malformed zone file is a load
    error like any other bad value (CLI exit status 2), found before any frame or model.

    A file that sets a key to an old default that has since changed (``CHANGED_DEFAULTS``,
    e.g. ``stuck.alarm_seconds: 8.0`` from a 0.2.0 site file) loads with its value, and one
    warning naming each such key, its old and today's default and configs/legacy_rule.yaml.
    """
    if path is None:
        return MonitorConfig()
    import yaml  # local import so the package imports without PyYAML for tests

    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    cfg = config_from_dict(data)
    if cfg.zone.polygon_file:
        try:
            load_zone_file(cfg.zone.polygon_file)
        except (OSError, ValueError) as exc:
            raise ConfigError(f"cannot load zone (zone.polygon_file): {exc}") from exc
    stale = stale_defaults(data)
    if stale:
        LOGGER.warning(
            "config file %r sets %s. The file's values are used. A site file written by tools/calibrate_zone.py "
            "of an earlier release (0.2.0 copied every default into it) or a copy of that release's "
            "configs/default.yaml keeps them without meaning to: delete these keys to get today's defaults. "
            "configs/legacy_rule.yaml is the profile that restores the 0.2.0 vehicle rule on purpose.",
            str(path), _describe_stale(stale))
    return cfg


def save_config(cfg: MonitorConfig, path: str) -> None:
    import yaml

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg.to_dict(), fh, sort_keys=False)
