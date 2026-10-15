# Architecture and module contracts

This document is the contract every module in `gradecrossing/` is written against. If you
change a signature or a behaviour described here, change the module, its tests and this
file in the same commit. Field comments in `gradecrossing/types.py` and `gradecrossing/config.py`
are part of the spec.

```
video / RTSP --> Monitor.run (I/O, timestamps, RUN_START / RUN_END, reconnect, outputs, stop signals)
                   |
                   v
                 Monitor.process_frame(frame, t, i)
                   1 resize to the processing size the first frame fixed (video.resize_width)
                   2 ZoneStabilizer.update(frame, t, scene_static)         -> ZoneState (polygon, mask, to_reference, healthy)
                   3 VehicleDetector.detect(frame) every N processed frames -> [Detection] with ByteTrack ids
                     (else StuckVehicleTracker.predict_boxes(t))
                   4 StuckVehicleTracker.update(dets, zone, t, detector_ran,
                         alarmed_ids=AlertEngine.alarm_episode_track_ids)  -> [ObjectState]
                   5 InfrastructureDetector.update(frame, t, i) (optional)  -> GateState
                   6 DebrisDetector.update(frame, zone.mask, boxes, empty_s, t,
                         to_reference=, registration_ok=, hold=, ignore_mask=) -> DebrisResult
                   7 DEGRADED reasons (+ HEALTH event when a change lasts)
                   8 AlertEngine.evaluate(objects, debris, gate, t, i)      -> Assessment + [Event]
                   9 timings per stage                                      -> FrameResult
                                                 events -> sinks: stdout / JSONL / webhook / callbacks
```

## Design rules

* **Per object, not per frame.** Every vehicle and pedestrian decision is made on a tracked
  object. "Any box overlapping the zone" is never a trigger.
* **A vehicle is where its ground footprint is.** With `stuck.zone_test: footprint` (default)
  a vehicle is in the zone when the bottom `footprint_fraction` (0.25) of its box, full width,
  overlaps the zone by at least `footprint_min_overlap` (5 %), and *near* (too close) when it
  is not in it but within `close_box_heights` (0.75) box heights of it. The upper part of a
  box (a car on the road beyond the crossing) never counts. Persons, trains and unmapped
  classes use the bottom-center point (`zone_test: bottom_center` restores the point test for
  vehicles too).
* **Stuck = in zone AND stopped, continuously, for T seconds.** Stopped is the speed rule
  OR the ID-free still-cluster rule (`stuck.stationarity`). Moving traffic is not stuck, but
  it can carry a short still time (a vehicle stays in its own still cluster for about (w/3)/v
  seconds, up to twice that when it slows below `speed_threshold` on the way and the anchor
  moves once; a long slow bus longer), so slow or queued traffic can reach a WARNING;
  `tests/test_monitor.py` pins that a car crossing the zone at 120 px/s raises neither.
* **Class filtered.** `vehicle_classes` feed the stuck, near and dwell rules; `person_classes`
  the pedestrian rule; `train_classes` never alert. Classes in none of the lists raise no
  alert and are visible to the debris backstop; `ignore_classes` are dropped right after
  detection.
* **The background model never learns what it has to report.** Debris classification runs
  at learning rate 0; the model learns only during a box-free warm-up and through a
  selective update of pixels that are background, outside boxes and outside candidates.
* **The zone is never silently wrong.** An estimate that cannot be trusted holds the last
  good polygon with `healthy=False`; the debris timers are held meanwhile; lasting changes are
  HEALTH events; `FrameResult.degraded` says when the monitor is partly blind.
* **Alarms latch; warnings hold.** ALARM persists `latch_seconds` after its last condition;
  WARNING persists `warning_hold_seconds` while its objects are still in or near the zone.
* **Units.** Areas are fractions of the zone (or of the frame for
  `debris.min_object_frame_fraction`), times are seconds, speeds are px/s at processing
  resolution (m/s with a ground-plane homography). Some knobs are processing pixels and do
  not scale with resolution: `stuck.speed_threshold`, `zone.min_distance`, `zone.block_size`,
  `zone.lk_win_size`, `zone.ransac_reproj_threshold`, `zone.max_step_px`, `zone.deadzone_px`,
  `zone.reanchor_max_drift_px`, `debris.exclusion_dilate_px`, `debris.morph_kernel`, plus
  constants such as `zone.FB_MAX_ERROR_PX` (1 px) and `autozone.HOUGH_MAX_GAP` (10 px).
* **Testable without torch.** Only `detection.py` imports ultralytics, and only inside
  `load_yolo`. `Monitor` accepts an injected detector, so the whole loop runs with scripted
  detections; `VehicleDetector` and `InfrastructureDetector` accept an injected `model`.
* **Sinks cannot stop the frame loop.** Every sink call is wrapped; the webhook delivers
  from a daemon thread.

Timestamps are float seconds. For files, `Monitor.run` uses the container clock
(`CAP_PROP_POS_MSEC`) while it is positive and increasing, otherwise the previous timestamp
plus 1/fps (which is `frame_index / fps` when the container has no usable clock);
`video.fps_override` forces `frame_index / fps`. For live sources, `t` is `time.monotonic()`
relative to the start of the run. Frame indices start at 0 (at `start_seconds * fps` after a
seek).

---

## `gradecrossing/types.py`

`Status`, `Detection`, `ZoneState`, `ObjectState`, `DebrisResult`, `GateState`, `Assessment`,
`Event`, `FrameResult`. Read the file; the field comments are the spec. Fields added since
the first version (all defaulted, so old constructors keep working):

* `ZoneState.to_reference`: 3x3 float64, current processed-frame pixels -> calibration-frame
  pixels (see the zone section).
* `ObjectState`: `zone_overlap`, `zone_distance_bh`, `near_zone`, `still_seconds`,
  `near_stuck_seconds`, `occluded`, `uid` (serial of the physical track, kept across id
  switches).
* `DebrisResult`: `unknown_fraction`, `covered_fraction`, `registered`, `trusted`, `gain`,
  `held`, `state`.
* `FrameResult`: `degraded: bool`, `degraded_reasons: Tuple[str, ...]`.

`Detection.center` was removed (nothing used it).

## `gradecrossing/geometry.py`

`as_polygon`, `polygon_mask`, `polygon_area`, `point_in_polygon`, `bottom_center`, `box_iou`,
`box_iou_matrix`, `box_mask_overlap_fraction`, `footprint_box(xyxy, fraction=0.25)`,
`rect_polygon_distance(rect, polygon)`, `footprint_relation(xyxy, polygon, mask,
fraction=0.25) -> (overlap, distance_in_box_heights)`, `dilate_box`, `transform_points`,
`transform_box(xyxy, H)`, `homography_is_sane`. (`box_center` was removed.)

## `gradecrossing/config.py`

Dataclasses `VideoConfig, ZoneConfig, DetectorConfig, InfrastructureConfig, StuckConfig,
DebrisConfig, AlertConfig, OutputConfig, MonitorConfig`; `configs/default.yaml` lists the same
keys with the same defaults (a test keeps the key set and the values equal).

```python
def load_config(path: Optional[str]) -> MonitorConfig      # YAML; also checks zone.polygon_file; warns on stale defaults
def config_from_dict(data: Optional[dict]) -> MonitorConfig # retired keys first, ends with validate_config
def validate_config(cfg: MonitorConfig) -> MonitorConfig    # raises ConfigError naming the dotted key
def save_config(cfg: MonitorConfig, path: str) -> None
def load_zone_file(path: str) -> List[List[float]]          # {"vertices"|"polygon"|"points": [[x, y], ...]} or a bare list
def zone_polygon(zone_cfg: ZoneConfig) -> List[List[float]] # polygon_file contents if set, else polygon
def zone_file_argument(path: str) -> str                    # argparse type= of --zone
def stale_defaults(data) -> List[Tuple[str, Any, Any, str]] # (key, value, today's default, release) per old default set
class ConfigError(ValueError, TypeError)
RETIRED_KEYS: Dict[str, str]                                # removed keys -> hint in the ConfigError
CHANGED_DEFAULTS: Dict[str, Tuple[Any, str]]                # key -> (old default, last release with it)
```

* An unknown section or key raises `KeyError`. Retired keys are checked first, over the
  whole mapping: one `ConfigError` names every `RETIRED_KEYS` entry the mapping sets with
  its hint (`debris.empty_seconds_before_refresh` and `debris.empty_learning_rate`, the only
  keys removed since `ae66b8f`; add a row whenever a key is removed or renamed). When the
  mapping also holds old `CHANGED_DEFAULTS` values the error adds that it looks like a
  full-default site file of 0.2.0 and what to do (keep only the site's keys or re-run
  `tools/calibrate_zone.py`; `configs/legacy_rule.yaml` for the old vehicle rule).
* `CHANGED_DEFAULTS` lists the defaults changed since a release: `stuck.warning_seconds`
  3.0, `stuck.alarm_seconds` 8.0 and `debris.warmup_frames` 300, all 0.2.0 (add a row
  whenever a default changes; `tests/test_config_defaults.py` checks the table against a
  snapshot of the 0.2.0 defaults). `load_config` logs one WARNING (logger
  `gradecrossing.config`) when a *file* sets any of them to its old value, naming each key
  with its old and today's default and `configs/legacy_rule.yaml`; the file's values are
  used. `config_from_dict` on a mapping built in code does not warn.
* `validate_config` checks every value against its field annotation: int/float take numbers,
  never bool or str; bool takes only true/false; str only strings; `Optional` also null; lists
  must be lists and every item is checked. Shapes: `zone.polygon` and every
  `debris.ignore_polygons` entry need >= 3 finite [x, y] points, `zone.reference_size` 2
  values, `stuck.ground_plane_homography` 3x3. A root or section that is not a mapping is a
  `ConfigError`.
* `Monitor.__init__` runs `validate_config` again and resolves the zone once
  (`zone_polygon`), so a config built in code gets the same checks.
* Enum-like strings and ranges (`stuck.zone_test`, `stationarity`, `rekey_policy`,
  `close_policy`, `zone.motion_model`, `detector.conf`, ...) are checked by the component that
  uses them (`ValueError` at construction, CLI exit 2).

## `gradecrossing/zone.py`

```python
class ZoneStabilizer:
    def __init__(self, cfg: ZoneConfig): ...           # ValueError for an unknown cfg.motion_model
    def initialize(self, frame_bgr: np.ndarray, timestamp: float) -> ZoneState
    def initialize_reference(self, image_bgr: np.ndarray, timestamp: float = 0.0) -> bool
    def update(self, frame_bgr: np.ndarray, timestamp: float, scene_static: bool = False) -> ZoneState
    initialized: bool                           # property: a textured calibration keyframe is stored
    registered: bool                            # property: the pose is tied to the calibration frame
    motion_model: str                           # property
    reference_polygon: np.ndarray               # property: (N,2) float32 in the current keyframe
    reference_features: np.ndarray              # property
    health_changed: bool                        # healthy flipped on this update (per frame, not debounced)
    health_reports: List[str]                   # HEALTH texts for this update (debounced, registration)
    def max_step_px(self) -> float              # temporal-gate step limit for this zone
    def min_lk_inliers(self) -> int
    def health(self) -> dict                    # method, inliers, healthy, keyframe_age_s, consecutive_unhealthy,
                                                # initialized, motion_model, reason, offset_px, reanchors,
                                                # unverified_reanchors, refused_reanchors, drift_px,
                                                # registered, overlay_features
def build_stabilizer(cfg: ZoneConfig, processed_size, default_reference_size, timestamp=0.0)
        -> (ZoneStabilizer, (sx, sy), problems)  # ZoneGeometryError on aspect mismatch / bad reference_size
class ZoneGeometryError(ValueError)              # the monitor CLI exits 2 on it
def calibration_scale(reference_size, processed_size) -> (sx, sy)
def polygon_inside_fraction(polygon, frame_size) -> float
def zone_thickness(polygon) -> float             # 2 * area / perimeter
def resolve_reference_path(path) -> Path         # working directory, then the repository root
def load_reference_image(path) -> np.ndarray     # ZoneGeometryError if unreadable
```

`ZoneState.to_reference` (3x3 float64) maps pixels of the current processed frame to pixels
of the calibration keyframe, composed across keyframe re-anchors, never dead-zoned, the last
accepted value while holding, `np.eye(3)` right after initialisation and when
`stabilize=False`, `None` while uninitialised. `ZoneState.homography` stays the raw
keyframe -> current transform of a fresh estimate (`None` when identity/held). The tracker
and the debris detector both consume `to_reference`.

Behaviour

* `cfg.polygon` is in processed pixels; the monitor gets the stabilizer from `build_stabilizer`,
  which scales the configured polygon (`zone.polygon`, or `zone.polygon_file`) by
  processed / calibration size on both axes. The calibration size is `zone.reference_size`,
  else the size of `zone.reference_image`, else the native size of the first frame the
  monitor ever processed (kept across `reset()`, so a live re-initialisation on a new
  frame geometry still scales from the calibration pixels). An aspect ratio that differs by
  more than 1% (`ASPECT_TOLERANCE`) raises `ZoneGeometryError` (the monitor emits one HEALTH
  event, `Monitor.run` stops, the CLI exits 2). A polygon less than 90% inside the frame, and a missing, unreadable or
  featureless `zone.reference_image`, are start-up problems (ERROR log + HEALTH event with
  `extra.zone_setup`; monitoring continues, anchored on the first textured live frame). A
  relative `reference_image` is looked up from the working directory, then from the
  repository root.
* `initialize` stores the grayscale calibration keyframe and its Shi-Tomasi features (features
  inside the polygon are discarded when `cfg.exclude_features_in_zone`). A featureless frame
  (grey-level std < 3, or fewer than `min_inliers` corners) is not stored: the state is
  `method="uninitialized"`, `healthy=False`, `to_reference=None`, and the next `update` retries.
  It is called once at start-up; a keyframe refresh re-anchors through a private `_anchor`.
* `initialize_reference` (called by `build_stabilizer` with `zone.reference_image` resized to the
  processed size) makes the image the calibration keyframe; `update` then *registers* live
  frames against it (steps 2-4 below, from the calibration pose) while holding the zone at the
  calibration pose. The first registered live frame becomes the keyframe LK tracks from. After
  `registration_timeout_seconds` without a registration the stabilizer tracks from a textured
  live frame at the calibration pose (`registered=False`, one HEALTH report "not registered");
  a later keyframe refresh registers it when its direct estimate is a small step or refutes the
  uncorrected pose (HEALTH report "registered ... moved N px").
* `update`:
  1. If `not cfg.stabilize` return the identity state.
  2. LK from the keyframe with a forward-backward check (round trip <= 1 px), then a RANSAC fit
     of `cfg.motion_model`: `similarity` (default, `estimateAffinePartial2D`), `affine`
     (`estimateAffine2D`) or `homography` (`findHomography`). Accept when
     `inliers >= max(min_inliers, ceil(min_inlier_fraction * n_keyframe_features))` and the
     calibration -> current pose is sane (scale, rotation `max_rotation_deg`, area, centroid
     shift `max_translation_fraction` of the frame, perspective factor for a homography).
  3. Otherwise, when the frame's grey-level distribution (outside the zone) differs from the
     keyframe's by more than 4 levels, LK again on the frame remapped to the keyframe's
     histogram (gain/gamma/offset change: dusk, auto-exposure, a calibration image in other
     light). Otherwise, if `orb_fallback`: ORB + ratio test on the frame as captured, same model
     and sanity checks, needing `orb_min_matches` inliers and `orb_min_inlier_fraction` of the
     matches.
  4. Temporal gate (LK and ORB alike): a vertex step above `max(max_step_px, max_step_fraction *
     2*area/perimeter)` from the last accepted polygon is accepted at once only when the frame
     refutes the last pose (the old pose fails the acceptance bar, fits at most 25% of the
     correspondences, and the new pose fits twice as many); otherwise only after `confirm_frames`
     consecutive estimates agree on the new pose. Until then: hold.
  5. Hold: **keep the last healthy polygon** (never the raw reference polygon), `method="held"`,
     `healthy=False`, `homography=None`, `to_reference` unchanged, `health()["reason"]` says why.
  6. Dead-zone: the reported polygon only moves when the accepted polygon is `deadzone_px` or
     more away from it (static camera -> bit-identical polygon every frame).
  7. Static overlays: a keyframe corner that stayed within 0.5 px on 3 accepted LK estimates
     that moved it by >= 2 px is a burned-in overlay; a disc around it is masked out of LK, ORB,
     the gate's vote and every later keyframe (at most half of a keyframe's corners).
  8. Keyframe refresh: after `keyframe_refresh_static_seconds` of `scene_static` frames with a
     fresh healthy estimate, re-anchor the keyframe on the current frame. The new keyframe's
     pose is checked with a direct estimate against the calibration keyframe: within
     `reanchor_max_drift_px` -> anchored on the direct pose (no random walk); further apart ->
     refused; no direct estimate possible -> chained pose, counted in `unverified_reanchors`.
  9. `ZoneState.polygon` is `np.int32 (N,2)` rounded; `mask` via `polygon_mask` (cached while
     the polygon does not change); `area_px` is `countNonZero(mask)`.
  10. `health_reports`: a change of `healthy` once it has lasted `health_report_seconds`
      (0 = every change), plus the registration notices; the monitor turns each into a HEALTH
      event (`extra` = `health()`) when `zone.health_events`.
* A frame of another size than the first one raises `ValueError`. The monitor never passes
  one: `process_frame` step 1 brings every frame to the processing size the first frame
  fixed, or raises `FrameGeometryError` for another aspect ratio (see `monitor.py`).

Tests: `tests/test_zone.py` (LK/ORB/hold/refresh basics on a warped 640x480 texture,
histogram-matched LK), `tests/test_zone_temporal.py` (motion models, a train-occlusion scene
with shake, dead-zone, temporal gate incl. a glitch frame, a moving occluder, a bump behind a
static banner, ORB inlier fraction, `to_reference` across re-anchors, re-anchor drift,
featureless start-up), `tests/test_zone_calibration.py` (reference size/image, aspect refusal,
polygon outside the frame, live resolution change, black first frames,
`tools/calibrate_zone.py`), `tests/test_zone_registration.py` (reference image in other light,
registration timeout and late registration, missing image, exit status 2) and
`tests/test_zone_health.py` (debounced health reports, static-overlay learning).

## `gradecrossing/autozone.py` (experimental, stage-1 seed)

```python
def propose_zone(frame_bgr: np.ndarray, infra_detections: Optional[List[Detection]] = None,
                 gate_classes: Sequence[str] = ("railroad-crossing-down", "railroad-crossings")) -> Optional[np.ndarray]
```

Heuristic: Canny + `HoughLinesP`; cluster line segments by angle; pick the two strongest
near-parallel clusters as rail edges; if gate/sign boxes are provided, use their horizontal
extent to bound the crossing; return a 4-point float32 polygon or None. It is a *seed* for
`tools/calibrate_zone.py`, not a production zone. Tests: `tests/test_autozone.py`.

## `gradecrossing/detection.py`

```python
CATEGORY_VEHICLE, CATEGORY_PERSON, CATEGORY_TRAIN, CATEGORY_OTHER   # "vehicle" | "person" | "train" | "other"
class ConfigClassifier:
    def __init__(self, cfg: DetectorConfig): ...
    def __call__(self, cls_name: str) -> str              # case-insensitive membership in the three lists

class VehicleDetector:
    @staticmethod
    def check_config(cfg: DetectorConfig) -> None        # ValueError: conf / track_conf outside (0, 1]
    def __init__(self, cfg: DetectorConfig, model=None): ...   # model=None: load_yolo(cfg.weights) lazily
    names: Dict[int, str]                                 # property
    def detect(self, frame_bgr: np.ndarray) -> List[Detection]
    def reset(self) -> None                               # reset the ByteTrack state (new video)
    def classify(self, cls_name: str) -> str              # delegates to ConfigClassifier

class InfrastructureDetector:
    def __init__(self, cfg: InfrastructureConfig, model=None): ...
    def update(self, frame_bgr: np.ndarray, timestamp: float, frame_index: int) -> GateState
    def reset(self) -> None

def load_yolo(weights: str, task: str = "detect")         # the only ultralytics import
```

* `detect` calls `model.track(...)` when `cfg.tracker` is set (else `model.predict(...)`)
  with `conf = cfg.track_conf` when set, else `cfg.conf`, plus `iou`, `imgsz`, `max_det`,
  `device` and `half=True` only when `cfg.half` (passing `half=False` explicitly triggers an
  ultralytics 8.4 deprecation warning on every frame). Boxes are converted to `Detection`
  (`track_id` from `boxes.id` if present). With `track_conf < conf`, a box the tracker returns
  without an id must still score `conf`. Boxes of `ignore_classes` are dropped.
* `load_yolo` refuses a missing local weights path (`FileNotFoundError`), except official
  ultralytics asset names such as `yolo11n.pt`, which ultralytics downloads. Without
  ultralytics it raises `ImportError` with `ULTRALYTICS_INSTALL_HINT` (install `.[detector]`
  / `requirements.txt` or `.[display]` / `requirements-display.txt`, not a bare
  `pip install ultralytics`; NVIDIA's torch wheel first on a Jetson); the "not found
  locally, ultralytics will try to download" warning is logged only once ultralytics has
  imported.
* Class names of the loaded model that are in none of the three lists are logged once, at
  WARNING: such a class raises no vehicle/person alert; the debris backstop still sees it.
* `InfrastructureDetector.update` runs the model only every `every_n_frames` (and on its first
  call); otherwise returns the cached `GateState`. `gates_down = (last_seen_down is not None and
  timestamp - last_seen_down <= hold_seconds)`. `available=True`.
* Tests: `tests/test_detection.py` (stub models, class mapping, thresholds, tool smoke tests;
  two smoke tests load real weights and skip without ultralytics).

## `gradecrossing/tracking.py`

```python
class StuckVehicleTracker:
    def __init__(self, cfg: StuckConfig, classify: Callable[[str], str]): ...   # ValueError on bad enums/ranges
    def update(self, detections: List[Detection], zone: ZoneState, timestamp: float,
               detector_ran: bool = True, alarmed_ids: Iterable[int] = ()) -> List[ObjectState]
    def predict_boxes(self, timestamp: float) -> List[Detection]   # PredictedDetection per live track
    zone_occupied: bool                       # property: a live object (any class) or hidden hazard is in the zone
    zone_empty_since: Optional[float]         # property
    def objects(self) -> List[ObjectState]    # live tracks, then hidden hazards
    def pop_detection_gap(self) -> Optional[float]   # once per run: detector interval > still_max_gap_seconds
    def reset(self) -> None

class StationaryClusters:                     # ID-free stationarity (port of the team's geometry.py)
    def __init__(self, min_iou: float = 0.5, max_gap_seconds: float = 1.0): ...
    def update(self, timestamp: float, boxes) -> List[StillCluster]
class PredictedDetection(Detection)           # marks boxes from predict_boxes: never evidence
```

`update` returns every live object plus every hidden hazard (`occluded=True`), not only the
ones seen this frame. `detector_ran` says whether the detections are a real detector run or
the tracker's own predictions; `alarmed_ids` are the ids of the engine's ALARM episode in
progress on the previous frame, latched included (`AlertEngine.alarm_episode_track_ids`:
the raw `alarm_track_ids` plus every id of the episode while the status is ALARM).

* **Identity.** Detections without `track_id` are matched by greedy IoU
  (`fallback_iou_threshold`). A detection with a `track_id` the tracker has never seen is
  first matched, by IoU at the velocity-predicted box, against the live tracks no detection
  claims this frame; a match re-keys that track to the new id with every counter intact
  (ByteTrack's id swap after an occlusion). `rekey_policy: strict` (default): a vehicle box
  may only adopt a *stopped* vehicle track, a person box only a person track, and if the old
  id reappears while the take-over is recent the clocks go back to it; `any` is the pre-port
  rule. A stopped in-zone vehicle keeps its category through label runs shorter than
  `label_hold_seconds` (car -> train/boat flicker). A vehicle in `alarmed_ids` that is in or
  near the zone keeps its vehicle category and label through any run of other labels,
  moving or not (a stopped car merging with the arriving train is labelled 'train' for
  seconds), so it stays evaluated, feeds its still cluster and can become a hidden hazard;
  once the episode ends or the vehicle leaves the zone and its near band, a run already
  longer than the hold changes the category at the next detection. `label_hold_seconds: 0`
  follows every label, alarmed or not.
* **Zone relation** (real detections only; predicted boxes never change it). Vehicles with
  `zone_test: footprint`: `footprint_relation` against the zone of this frame gives
  `zone_overlap` and `zone_distance_bh`; in zone when the overlap is > 0 and >=
  `footprint_min_overlap`; near when not in it and within `close_box_heights` (0 = off).
  Everything else: `point_in_polygon(bottom_center, zone.polygon)`, never near. A vehicle
  leaves the zone (and its on-or-near streak ends) only after its real detections have been
  out for `zone_exit_grace_seconds`. Ids seen fewer than `min_track_age_frames` times are
  never in or near.
* **Speed**: a least-squares velocity of the bottom-center over `speed_window_seconds`
  (px/s, or m/s with `ground_plane_homography`), in calibration-frame pixels when the zone
  provides `to_reference` and `camera_motion_compensation` is on (on a held frame the last
  healthy matrix is reused). No estimate until `min_history_for_speed` samples exist, and
  without one nothing is stationary (no instant alarms).
* **Still clusters**: every real vehicle box joins the live cluster whose *anchor* it
  overlaps with IoU >= `still_min_iou`, or opens one; a cluster expires after
  `still_max_gap_seconds` without a match. The anchor is the cluster's first box, moved once
  (`StillCluster.settled`) to the owner's box on the first observation where the owner's
  speed estimate is below `speed_threshold`; the cluster keeps its start. A vehicle braking
  into the zone opens its cluster while it still rolls, and anchored there a short creep
  after the stop would leave the cluster and restart the clock; a vehicle that keeps
  creeping still leaves it after about a third of its width. `still_seconds` is the owner's
  age of its cluster (a guest only its own time there). A cluster changes owner only when
  the owner's id is absent and only to a track born in it; it carries the owner's in-zone
  and on-or-near streaks across such a change. A box within `still_border_margin` of the
  image border restarts its cluster's still clock (`StillCluster.cut_t`; the clock counts
  from max(start, cut_t)) only when a real train box seen in the last 2 s
  (`_BORDER_TRAIN_SECONDS`) is beside it, that is overlaps the box grown by its own height on
  every side: the visible end of a train passing the edge of the picture. A vehicle stopped
  at the edge with no train beside it keeps its still clock.
* **Counters** (timestamp differences, not frame counts): `dwell_seconds` (continuous in
  zone), `stationary_seconds` (speed below `speed_threshold`), `still_seconds`, `stuck_seconds`
  (in zone AND stopped, stopped = per `stationarity`: `speed`, `still` or `speed_or_still`),
  `near_stuck_seconds` (in zone or near AND stopped).
* **Track loss**: on detector runs only, a track is dropped once `lost_track_missed_runs`
  runs in a row missed it and its last detection is older than `lost_track_grace_seconds`.
  Skipped detector frames never age a track. Until it is dropped, a live vehicle track in
  the zone that a detector run missed keeps its `dwell_seconds` and `stuck_seconds` running
  from their start (`undetected_clock_while_live`, default true; false, as in
  `configs/legacy_rule.yaml`, leaves them at their last value until the next detection).
* **Hidden hazards**: a vehicle in `alarmed_ids` that is dropped while a train box overlaps
  its last box, and was last seen in or near the zone, becomes a hidden hazard
  (`occluded=True`, the engine holds ALARM "occluded by train") while a train box overlaps
  it and for `occlusion_memory_seconds` after; a re-detection at IoU >= 0.5 restores its
  track and clocks. Only alarmed vehicles get this; because `alarmed_ids` is the whole
  episode, a car the train covers gradually (its box cut short, its clocks restarted, the
  ALARM held only by the latch when it is lost) is still kept, and an episode vehicle that
  drove off is not.
* `predict_boxes` returns the last detected box of every live track shifted by its velocity
  (hidden hazards get none). Predicted boxes advance the clocks but never refresh
  `last_seen`, add speed samples, feed clusters or change zone membership.
* `zone_occupied` counts any category, `other` included, and hidden hazards; the monitor uses
  it for the zone keyframe refresh (`scene_static`) and, through `zone_empty_since`, for the
  debris illumination guard when the zone fills the modelled region.

Tests: `tests/test_tracking.py`, `tests/test_footprint.py`, `tests/test_still.py`,
`tests/test_rules_tracking.py` (sparse detection, re-keying policy, label hold, hidden
hazards, camera-motion compensation, border clusters), `tests/test_rules_final_rules.py`
(the ALARM episode as `alarmed_ids`, a car the train covers gradually, the alarmed label
hold, border boxes with and without a train beside them).

## `gradecrossing/background.py`

```python
class DebrisDetector:
    def __init__(self, cfg: DebrisConfig): ...
    def update(self, frame_bgr: np.ndarray, zone_mask: np.ndarray, exclusion_boxes: Sequence[np.ndarray],
               zone_empty_seconds: Optional[float], timestamp: float, *,
               to_reference: Optional[np.ndarray] = None, registration_ok: bool = True,
               hold: bool = False, ignore_mask: Optional[np.ndarray] = None) -> DebrisResult
    warmed_up: bool                    # property
    frames_seen: int                   # property
    triggered: bool                    # property
    last_illumination_event: Optional[dict]   # set on the update that re-warmed the model
    def background_image(self) -> Optional[np.ndarray]
    @staticmethod
    def absorption_frames(learning_rate, background_ratio=0.9) -> float
    def reset(self) -> None
```

* `zone_mask` is `ZoneState.mask` (current geometry). `to_reference` maps current pixels to
  the background's reference pixels; a wrong shape raises `ValueError`, non-finite entries
  mean `registration_ok=False`. `ignore_mask` is in reference geometry. `zone_empty_seconds`
  is only read by the illumination guard when the zone fills the whole modelled region.
* **Region and scale.** Only the zone's bounding box plus a margin (at least 32 px) is
  modelled; frames wider than `process_width` (640) are downscaled first; results are mapped
  back to the caller's pixels. A change of frame geometry or zone region re-warms the model.
* **Warm-up** (`warmup_seconds`, 2 s, at least 5 frames; `warmup_frames` overrides): MOG2 at
  its automatic rate, but pixels under dilated exclusion boxes are pasted from the previous
  warm-up frame, so they are never learned. A pixel box-free for less than half the warm-up is
  *unknown* and excluded from every decision until it has been box-free and classified
  background for `learn_seconds`. While `registration_ok` is False the warm-up waits.
* **Classification** at `monitoring_learning_rate` (0: nothing is absorbed), after dividing
  out the per-channel gain measured on background pixels outside the zone and the boxes
  (`gain_compensation`); pixels the gain would clip are unknown for the frame. Foreground =
  MOG2 255 (shadows 127 dropped), opened and closed with `morph_kernel`, restricted to the
  observable zone (learned, in frame, not clipped, not under a box, not ignored); components
  below `min_blob_area_fraction` of the zone are dropped, and a component whose texture
  correlates with the background (NCC >= 0.7 on a textured background) is lighting.
* **Selective update**: every `update_interval_seconds` one learning `apply` with rate
  `interval / update_time_constant_seconds` on a composite frame: zone pixels classified
  background outside boxes and candidates (plus a margin), and box-free pixels outside the
  zone, show the frame; everything else is the model's own background, dithered by its
  residual standard deviation. No in-zone foreground pixel is ever learned.
* **Persistence** is per pixel: an observable foreground pixel that stays static ages; a pixel
  seen as background resets at once; a candidate pixel under a box (or unobservable) keeps
  its age, without counting the covered time, for `occlusion_bridge_seconds`. Immobility:
  a component restarts when more than `static_motion_fraction` of its pixels changed by more
  than `static_motion_diff` grey levels against a frame `static_motion_lag_seconds` old (with
  a +-1 px tolerance, counted only where the pixel was foreground then). `triggered` once
  `k = min(min_area_fraction * zone area, min_object_frame_fraction * processing-frame area)`
  pixels are at least `persist_seconds` old; `DebrisResult.persist_seconds` is the age of the
  k-th oldest.
* **Hold**: while `hold` (a train box over the zone, set by the monitor) and
  `hold_linger_seconds` (3 s) after, and while `registration_ok` is False, timers are
  frozen, no candidate starts and nothing is learned (`held=True`, `state` "held (train)" /
  "held (registration)"). The linger bridges most gaps between the 'train' labels of one
  passage (clip03: up to 3.5 s), at the cost of 3 s of blindness after the last one. On the
  first held frame of a train hold that starts less than `persist_seconds` after the
  previous hold ended (the same passage), and while debris is not triggered, the candidates
  born since that hold ended restart (age, covered and ghost time 0) and the persistence
  reported while held is recomputed without them; candidates from before the previous hold
  keep their age. Without this a uniform freight car seen in the gaps added up its age over
  the gaps of one passage.
* **Re-warm** (never while triggered or held): a channel gain outside
  `[1/rewarm_gain, rewarm_gain]` for 1 s, or foreground over `max_area_fraction` of the
  observable zone and of the modelled region outside it, resets the model, starts a new
  warm-up and sets `last_illumination_event` (`reason` "gain" or "illumination step").
* **Ghost healing**: a candidate whose outline step exists only in the model, for
  `ghost_heal_seconds`, on young model content (at most 25 % of its pixels seen as
  background for `ghost_heal_max_seen_seconds` or longer), is made unknown and re-learned.
  Never while triggered.
* `trusted` is False while registration fails or `unknown_fraction` exceeds
  `max_unknown_fraction` (recovers at 0.8 of it).

Tests: `tests/test_background.py`, `tests/test_debris_robustness.py` (one reviewed real-footage
failure per test: warm-up ghosts, occlusion bridging, shake, gain, hazard lights, trains,
ghost healing limits) and `tests/test_debris_monitor.py` (the monitor's wiring of step 6).

## `gradecrossing/alerting.py`

```python
class AlertEngine:
    def __init__(self, stuck_cfg: StuckConfig, alert_cfg: AlertConfig): ...
        # ValueError: unknown close_policy, negative warning_hold_seconds. The old
        # AlertEngine(stuck_cfg, debris_cfg, alert_cfg) call still works and logs a deprecation line.
    def evaluate(self, objects: List[ObjectState], debris: Optional[DebrisResult], gate: GateState,
                 timestamp: float, frame_index: int) -> Tuple[Assessment, List[Event]]
    def add_sink(self, sink) -> None             # sink has .emit(event); .close() is used if present
    def emit(self, event: Event) -> None         # deliver an external event (HEALTH, RUN_*) to every sink
    def close_open_episodes(self, timestamp: float, frame_index: int = -1,
                            reason: str = "pipeline re-initialised") -> List[Event]   # END event, then reset()
    def reset(self) -> None                      # forget all state, emit nothing
    status: Status                               # property: last evaluated status
    latched_until: Optional[float]               # property
    alarm_track_ids: List[int]                   # property: alarm ids of the last evaluate (raw: none while latched)
    alarm_episode_track_ids: List[int]           # property: alarm_track_ids plus every id of the ALARM episode
                                                 # in progress while the status is ALARM (latched included)
    sinks: Tuple                                 # property

class StdoutSink(stream=None), JsonlSink(path, append=True), CallbackSink(fn)
class WebhookSink(url, timeout=3.0, max_queue=256, headers=None, attempts=3,
                  backoff_seconds=0.5, deadline_seconds=None)   # .flush(timeout), .close(timeout), counters
def build_sinks(alert_cfg: AlertConfig) -> List       # stdout / JSONL / webhook from the config
def event_id(event) -> str                            # "<session>-<n>", or extra["event_id"]
def event_to_json(event, include_id=True) -> str
def ascii_safe(text) -> str
```

Decision (`evaluate`):

* Thresholds: with `gate.gates_down` the `gates_down_warning_seconds`, `gates_down_alarm_seconds`
  and `gates_down_dwell_alarm_seconds` replace `warning_seconds`, `alarm_seconds` and
  `dwell_alarm_seconds`; `dwell_warning_seconds` has no gates-down variant. At construction the
  engine logs every gates-down threshold that is not shorter than the normal one (with the
  defaults: the 3 s stuck ALARM, so gates down only shortens the WARNING and the dwell ALARM).
  `severity_scale = alarm / alarm_seconds`.
* Per vehicle, first match wins: occluded (hidden hazard) -> ALARM; in zone and (stuck >=
  alarm or dwell >= dwell alarm) -> ALARM; `close_policy == "alarm"` and near_stuck >= alarm
  -> ALARM; in zone and (stuck >= warn or dwell >= dwell warning) -> WARNING;
  `close_policy != "off"` and near_stuck >= warn -> WARNING ("stopped near the tracks").
  Objects neither in the zone nor (vehicles) near or occluded are skipped.
* Per person in the zone: dwell >= `person_alarm_seconds` -> ALARM, >= `person_warning_seconds`
  -> WARNING (`person_alert`). Trains and unmapped classes: nothing.
* Debris: `debris.triggered` -> ALARM.
* Raw status: ALARM if any alarm, WARNING if any warning, else CLEAR. Latch: a raw ALARM sets
  `latched_until = t + latch_seconds`; while `t < latched_until` the status stays ALARM with
  reason `"latched"`. WARNING hold: a WARNING whose condition just ended stays WARNING (reason
  `"held"`) for `warning_hold_seconds` while one of its episode's objects is still in, near or
  hidden in the zone.
* Reasons: `"vehicle 7 (car) stationary in zone 3.2s"`, `"vehicle 7 (car) in zone 25.0s"`
  (dwell), `"vehicle 7 (car) stopped near the tracks 1.6s"`,
  `"vehicle 7 (car) occluded by train, stationary in zone 9.0s"`, `"person 3 in zone 2.5s"`,
  `"debris 4.1% of zone for 3.2s"`; vehicle reasons end in `" (gates down)"` when the gate
  thresholds applied.

Events (`Event.type`, `status` = status after the transition):

| Type | When | `track_ids`, `reasons`, `extra` |
|---|---|---|
| `WARNING_START` | CLEAR -> WARNING, ALARM -> WARNING | warning ids; frame reasons; `objects` |
| `ALARM_START` | CLEAR/WARNING -> ALARM | alarm ids; alarm-level reasons; `objects`, `warning_reasons` if any |
| `ALARM_START` repeat | every `repeat_notify_seconds` while ALARM (0 = never) | as above, `repeat: true`, no `objects` |
| `ALARM_UPDATE` | a new cause (new physical track by `ObjectState.uid`, or debris) joins an ALARM | current alarm ids; alarm reasons; `new_track_ids`, `new_reasons`, `objects` of the new ids, `warning_reasons` |
| `WARNING_END`, `ALARM_END` | the episode ends | every id of the episode; `[opening reason]`; `objects`, `episode_start` |
| END from `close_open_episodes` | pipeline re-initialised | status CLEAR; `objects: []`, `episode_start`, `reason` |
| `HEALTH` | emitted by the monitor | see `monitor.py` |
| `RUN_START`, `RUN_END` | emitted by `Monitor.run` | see `monitor.py` |

WARNING -> ALARM emits `WARNING_END` (status ALARM) then `ALARM_START`; ALARM -> WARNING emits
`ALARM_END` then `WARNING_START`. While only the latch holds an ALARM its reasons are
`["latched"]`. `objects` entries: `id, class, box, in_zone, near_zone, zone_overlap,
zone_distance_bh` (null when infinite)`, dwell, stuck, still, speed[, occluded]`.

Sinks:

* `StdoutSink`: one ASCII line per event. `JsonlSink`: one `dataclasses.asdict(event)` JSON
  object per line, appended, **without** `event_id` (the evaluate contract).
* `WebhookSink`: JSON POST from one daemon worker, payload = event fields + top-level
  `event_id`. Retries refused/reset connections, timeouts and HTTP 5xx/408/429, `attempts` in
  all, waiting `backoff_seconds` doubling; `timeout` is urllib's per socket operation; all
  attempts of one event share `deadline_seconds` (default attempts x timeout + waits = 10.5 s),
  and an attempt still running at the deadline is aborted and sends nothing afterwards. ALARM
  events are queued ahead of other events (order kept within each group); the WARNING_END of a
  WARNING -> ALARM transition is sent right after its ALARM_START. A full queue (`max_queue`)
  drops the oldest non-ALARM event first. Delivery is at-least-once: receivers de-duplicate on
  `event_id` and order by its sequence number (or by `timestamp` within a run).
* Every sink has `close()`; `Monitor.close` calls it on the sinks it built from the config.
  Sinks passed to `Monitor(sinks=...)` belong to the caller.
* `WebhookSink.close()` first moves queued HEALTH notices behind the other queued events
  (order kept within each group, ALARM priority kept, so an END still never overtakes its
  START): the RUN_END and the closing WARNING_END / ALARM_END are not stuck behind start-up
  notices on a slow endpoint. It then waits up to `timeout` per queued event, at least
  `deadline_seconds` and at most `_CLOSE_MAX_SECONDS` (30 s; the deadline when that is
  longer), cut back to one deadline as soon as a delivery fails meanwhile (a dead endpoint
  costs one deadline). HEALTH notices still queued are delivered after RUN_END. An explicit
  `close(timeout)` flushes for exactly that long; events left are still attempted by the
  daemon worker until the process exits.

Tests: `tests/test_alerting.py`, `tests/test_rules_alerting.py`, `tests/test_close_policy.py`,
`tests/test_sinks.py` (event ids, ASCII, close order and budget, retries, deadline,
priority, overflow and eviction order, late POSTs).

## `gradecrossing/evaluate.py`

```python
@dataclass class GroundTruthEvent: video, kind, start_s, end_s, note=""
@dataclass class TruthSet: video, duration_s, events, negatives=[], ignore=[], container_duration_s=None
def load_truth_csv(path) -> List[GroundTruthEvent]           # columns video,kind,start_s,end_s
def load_truth_json(path) -> TruthSet                        # team format; needs a top-level "events" list
def load_truth_dir(path) -> Dict[str, TruthSet]
def read_event_log(path) -> EventLog                         # runs split by RUN_START / backwards timestamps
def select_runs(runs, run="last") -> List[RunLog]            # "last" | "all" | <run_id>
def load_alarm_intervals(events_jsonl_path, run="last") -> List[Tuple[float, float]]
def score(truth, alarms, duration_s, tolerance_s=2.0, ignore=(), negatives=(),
          match="one_to_one", warnings=None) -> dict
def evaluate_clip(events_path, truth, tolerance_s, match, run="last", duration_s=None, kind=None) -> dict
def evaluate_batch(...) -> dict; def summarize(clips) -> dict
def main(argv=None) -> int                                   # 0 ok, 1 bad input / missing log, 2 usage
```

* Episodes: `ALARM_START .. ALARM_END` (a repeat or a second start while open is ignored;
  `ALARM_UPDATE`, `HEALTH` and unknown types never start one); the same for WARNING. An episode
  still open when its run ends lasts to the end of the run.
* Runs: `RUN_START` starts a run, `RUN_END` records `duration_s` (seconds of video covered);
  events after a RUN_END stay in that run (with a warning); timestamps going backwards without
  a RUN_START also start a new run. Only the last run is scored unless `run` says otherwise;
  several runs are scored one by one against the same truth and added up.
* Matching: `one_to_one` (default for CSV, tolerance 2 s; duplicates of an already matched
  incident are reported, not false alarms) or `any_overlap` (default for team JSON, tolerance
  1 s; the team's `occupancy_monitor/crossing_monitor/evaluate.py` rules). Unmatched alarms that
  overlap an ignore span (starting no earlier than `tolerance_s` before it) are ignored; false
  alarms are tagged with the first labelled negative they overlap.
* Metrics: `tp, fp, fn, precision, recall, f1, time_to_alert_s, median_time_to_alert_s,
  false_alarms_per_hour, duration_s, duplicates, ignored_alarms, pre_existing_matches,
  fa_seconds` and, with warnings, the WARNING-or-ALARM detection, time-to-alert, FA-seconds
  and `warnings_outside_positives`; a `details` block lists every match and false alarm.
* Duration of a run: `--duration-s`, else `RUN_END.duration_s`, else the truth JSON
  `duration`, else the last event (with a warning).
* CLI: `python -m gradecrossing.evaluate --events FILE|DIR --truth CSV|JSON|DIR [--video]
  [--kind] [--duration-s] [--tolerance-s] [--match] [--run last|all|RUN_ID] [--out] [--quiet]`.
  With a directory of event logs (`<video>.events.jsonl`) it scores a batch and prints one
  table.

Tests: `tests/test_evaluate.py`, `tests/test_eval_team.py`, `tests/test_eval_runs.py`,
`tests/test_eval_benchmark.py`.

## `gradecrossing/visualize.py`

```python
def draw_overlay(frame_bgr, result: FrameResult, fps_estimate: Optional[float] = None,
                 footprint_fraction: Optional[float] = None) -> np.ndarray
```

Draws the zone polygon (green healthy; red when held, "HELD", or not anchored, "NOT
ANCHORED"), each object box with id, class, speed and stuck time (red alarm, orange warning,
grey when not a vehicle; `*` marks an object in the zone, `+` one near it; the footprint strip when
`footprint_fraction` is given), debris blobs (magenta), a status banner with reasons, gate
state, a zone-health / timing line and `DEGRADED: <reasons>` while `result.degraded`. All text
is ASCII. Returns a copy. Test: `tests/test_visualize.py`.

## `gradecrossing/monitor.py`

```python
class Monitor:
    def __init__(self, cfg: MonitorConfig, detector=None, infra_detector=None, sinks=()): ...
        # validate_config + zone_polygon here: ValueError / OSError before any frame, sink or model
    def process_frame(self, frame_bgr: np.ndarray, timestamp: float, frame_index: int) -> FrameResult
    def run(self) -> dict          # summary; RuntimeError when the source/output cannot be opened;
                                   # re-raises ZoneGeometryError and a detector that cannot be loaded
    def reset(self, timestamp: Optional[float] = None, keep_alerts: bool = False) -> None
                                   # closes an open episode first unless keep_alerts
    def close(self) -> None        # sink.close() on the config-built sinks
    last_result: Optional[FrameResult]; last_summary: Optional[dict]
class FrameGeometryError(ValueError)   # a frame whose aspect ratio differs from the processing size
def open_source(video: VideoConfig) -> Tuple[cv2.VideoCapture, float, bool]   # (cap, fps, is_live)
def is_live_source(source: str) -> bool
STOP_SIGNALS = ("SIGTERM", "SIGHUP")   # main() stops the run on them like Ctrl-C
def config_sha256(cfg: MonitorConfig) -> str
class CliOverride(NamedTuple); CLI_OVERRIDES: Tuple[CliOverride, ...]
def build_parser(); def add_cli_arguments(parser, overrides=CLI_OVERRIDES); def apply_cli_overrides(cfg, args, overrides=CLI_OVERRIDES)
def format_summary(summary) -> str; def main(argv=None) -> int     # 0 / 1 / 2
```

`process_frame` order:

1. the first frame fixes the processing size: its own, or `video.resize_width` wide with
   the aspect kept. A later frame of another resolution with the same aspect ratio (within
   `zone.ASPECT_TOLERANCE`, 1 %; a live stream that renegotiated it) is resized to that size,
   so no stage sees a new geometry and nothing is reset; each change is one HEALTH event
   (`component` "source", `state` "resolution_changed", `previous_size`, `native_size`,
   `processing_size`). Another aspect ratio raises `FrameGeometryError` before any state
   changes;
2. zone: on the first frame `build_stabilizer` (start-up problems -> HEALTH `zone_setup`; a
   `ZoneGeometryError` -> HEALTH `zone_error`, re-raised), then `update` with
   `scene_static = not tracker.zone_occupied and not debris triggered`; each
   `health_reports` text -> HEALTH (`extra` = `health()`) when `zone.health_events`;
3. detector on every `detect_every_n_frames`-th frame of an internal processed-frame counter
   (zeroed by `reset()`, so the detector runs on the first frame after construction, reset or
   a seek), else `tracker.predict_boxes(t)`; the detector is built here on the first frame;
4. `tracker.update(dets, zone, t, detector_ran=..., alarmed_ids=engine.alarm_episode_track_ids)`;
   a `pop_detection_gap()` -> HEALTH (`extra.detection_gap_s`);
5. gate state when an infrastructure detector is enabled or injected; one that raises leaves
   the frame without gate information (normal thresholds) and the monitor DEGRADED;
6. debris: exclusion boxes = vehicle, person and train detections (real or predicted), plus a
   track that had one of those labels within `DEBRIS_MAPPED_MEMORY_S` (5 s); `hold` when a
   train box overlaps the zone; `to_reference = zone.to_reference` and
   `registration_ok = zone.healthy`; `ignore_mask` from `debris.ignore_polygons` (scaled like
   the zone); `zone_empty_seconds` from the tracker, withheld while a candidate accumulates
   persistence. HEALTH (`component` "debris"): warm-up start (first frame, after `reset()`, after
   a rebuilt model), warm-up complete (with `unknown_fraction`), re-warms, and a change of
   `trusted` once it has lasted `alerts.health_debounce_seconds` (nothing is tracked during a
   warm-up; the value compared against is the one the warm-up-complete event announced);
7. DEGRADED: `zone uninitialised`, `zone held`, `debris warming up`, `debris held (train)`,
   `debris held (registration)` (left out when the zone hold causes it), `debris untrusted`,
   `detector interval > still gap` (unless `stationarity: speed`), `gate detector failing` ->
   `FrameResult.degraded` / `degraded_reasons` per frame; a change of the degraded flag that
   lasts `alerts.health_debounce_seconds` -> HEALTH (`component` "monitor", `degraded`,
   `reasons`). The reasons the monitor reports leave out `debris warming up` (the debris
   component reports each warm-up's start and end, step 6) and the zone reasons, which the
   zone's own events report, unless `zone.health_events` is off;
8. `engine.evaluate(...)`;
9. timings per stage in ms (`resize, zone, detect, track, infra, debris, alert, total`).

`reset(timestamp=None)` calls `engine.close_open_episodes` (END event at the last processed
frame's timestamp, `reason` "pipeline re-initialised"), then forgets the zone, tracks,
background model, detector tracker state, cadence and processing size (the next frame fixes
a new one); the DEGRADED state last reported is kept. `reset(keep_alerts=True)` skips the
END and leaves the alert engine alone: status, latch and open episode carry on and end the
normal way or with RUN_END. `run` uses it for a live stream re-initialised on a new aspect
ratio (the hazard is probably still there). Between passes of a looped file `run` calls
`engine.reset()` first (the pass's RUN_END closed the episode) and then `reset()`, so nothing
of a pass follows its RUN_END.

`run`:

* Opens the source (`open_source`; `RuntimeError` when it cannot) and creates the output
  directories (`RuntimeError`). Emits `RUN_START` (`run_id`, `source` with `user:pass@`
  redacted, `config_sha256`, `package_version`) before the first frame and `RUN_END`
  (`run_id`, `frames`, `duration_s` of video covered, `last_timestamp`, `final_status`,
  `stop_reason`) after the last one, also on abort, Ctrl-C, SIGTERM/SIGHUP (under `main`) or
  an exception. When the input ends during WARNING or ALARM, RUN_END is the closing event (no
  END is invented). A looped file gets one RUN_START / RUN_END pair per pass
  (`<run_id>-p<N>`); the event after a pass's RUN_END is the next pass's RUN_START (an episode
  open at the end of a pass is closed by that RUN_END, no END follows it).
* Stop reasons: `end_of_input`, `max_frames`, `interrupted` (Ctrl-C, or SIGTERM/SIGHUP under
  `main`), `user_quit`, `loop` (one pass of a looped file), and the aborting ones
  `frame_errors`, `reconnect_failed`, `zone_refused`, `detector_unavailable`, `error`. The
  summary has `frames, alarms, fps, warnings, health_events, elapsed_s, reconnects, errors,
  loops, final_status, interrupted, aborted, stop_reason, run_id, timings_ms`.
* A frame that makes a stage raise is logged, counted in `errors` and skipped;
  `MAX_CONSECUTIVE_FRAME_ERRORS` (5) in a row abort (`frame_errors`, HEALTH `component`
  "pipeline"). If the detector does not exist yet after a failed frame it is built at once;
  a model that cannot be loaded ends the run (`detector_unavailable`). A `FrameGeometryError`
  on a *live* source (another aspect ratio) costs that frame and re-initialises the
  perception with the alert state kept: `reset(keep_alerts=True)` (no END event) and a HEALTH
  event (`extra`: `error, reset, previous_status, debris_untrusted`; the last is true when a
  debris ALARM was on, because the background is re-learned with the current view). The zone
  set-up on the next frame then decides whether the calibration fits the new geometry;
  usually it does not and the run stops with `zone_refused`. Any other `ValueError`, and a
  `FrameGeometryError` on a file, is a lost frame like any other error. A
  `ZoneGeometryError` stops the run (`zone_refused`) and is re-raised after RUN_END.
* A live read failure reconnects `RECONNECT_ATTEMPTS` (5) times with back-off
  1, 2, 4, 8, 16 s; HEALTH `component` "source" for the loss (`state` lost), each attempt
  (`attempt_failed` / `reconnected`) and the give-up (`gave_up`, stop reason
  `reconnect_failed`). A stream URL is opened (first open and every reconnect) by the
  FFmpeg backend with open and read timeouts of `video.stream_timeout_seconds` (10 s; 0 =
  the backend default, 30 s measured with OpenCV 4.14) where the build supports them
  (OpenCV 4.5.2+ with FFmpeg as a stream backend), else plainly; camera indices and files
  are opened plainly. A dead camera is given up after about 31 s of back-off plus 5 x 10 s.
* Outputs (overlay, alarm snapshot, video writer, display, status line) are each guarded: a
  failure is logged, reported once (HEALTH `component` "pipeline", `output`) and that output is
  switched off while monitoring continues. `run` never writes to `cfg`.
* `main`: exit 0 (input ended, user stop, SIGTERM/SIGHUP), 1 (not watching: source, model,
  output path, frame errors, reconnect give-up, unexpected error), 2 (config or usage error,
  the monitor rejects a setting, zone refused); one line on stderr, never a traceback. Right
  before `run` it installs a handler for each of `STOP_SIGNALS` this platform has (SIGHUP
  does not exist on Windows) that raises `KeyboardInterrupt`, so `run` ends with stop reason
  `interrupted`, emits RUN_END and closes the sinks (webhook drain included); the previous
  handlers are restored afterwards. The console script `gradecrossing-monitor` is
  `gradecrossing._entry:monitor`, which calls `main` and exits 2 with one line naming the
  extra to install when OpenCV is missing (`python -m gradecrossing.monitor` without OpenCV
  still fails on the `cv2` import).

CLI: `python -m gradecrossing.monitor [--config YAML]` plus the `CLI_OVERRIDES` flags, each
overriding the config key in brackets: `--source` (video.source), `--weights`
(detector.weights), `--display` (output.display), `--save` (output.save_video), `--events`
(alerts.events_jsonl), `--max-frames` (video.max_frames), `--infra-weights`
(infrastructure.weights, and enables it), `--infra` (infrastructure.enabled), `--zone`
(zone.polygon_file; the file is checked at parse time), `--no-stabilize` (zone.stabilize
false), `--print-every` (output.print_every_n_frames), `--fps` (video.fps_override),
`--resize-width` (video.resize_width), `--start-seconds` (video.start_seconds), `--loop`
(video.loop), `--live` (video.live), `--detect-every` (detector.detect_every_n_frames),
`--device` (detector.device and infrastructure.device), `--snapshot-dir`
(output.snapshot_dir), `--webhook` (alerts.webhook_url), `--no-stdout` (alerts.stdout false),
`--log-level` (output.log_level). `--help` prints the exit statuses.

Tests: `tests/test_monitor.py` (scripted `FakeDetector`, synthetic 640x360 frames: crossing car,
stopped car, gate-down, static debris blob, id swap, detector cadence, unmapped class, frame
errors, same-aspect resize, live aspect change, display failure, malformed configs),
`tests/test_port_scenarios.py`, `tests/test_monitor_run.py` (fake captures: file end,
reconnect, give-up, stream timeouts, repeated errors, Ctrl-C, RUN events, timestamps, loops
and their event order, a resolution change during a debris ALARM),
`tests/test_monitor_degraded.py` (debounce, the warm-up reported once, debris trust
flicker), `tests/test_wiring.py` (each link between components, end to end),
`tests/test_cli.py` (exit statuses, stop signals; the real-SIGTERM subprocess test is
POSIX-only), `tests/test_config_validation.py`, `tests/test_config_defaults.py` (retired
keys, stale defaults, default.yaml), `tests/test_team_interop.py` (zone files).

## Tools (`tools/`)

* `calibrate_zone.py`: open a video/image/stream, show a frame (optionally with the
  `autozone.propose_zone` seed), click the polygon corners, write the site config: only
  `video.source`, `zone.polygon`, `zone.reference_size`, `zone.reference_image`
  (`<stem>_reference.png`, unless `--no-reference-image`) on top of the keys the base file sets.
  Headless: `--polygon`, `--headless`, `--save-frame`. Exit 0 written, 1 frame unreadable /
  cancelled / write error, 2 bad polygon or base config, 3 no GUI. A `zone.polygon_file` in
  the base config is removed from the written config, with a notice (the JSON file is
  kept): it would override the new polygon.
* `fetch_clips.py`: download the team benchmark clips from Drive into a cache
  (`$GRADECROSSING_CLIPS`, else `~/.cache/gradecrossing/team_clips`), check size and sha256.
* `run_benchmark.py`: run the monitor on the benchmark clips with minimal configs (source,
  zone, weights, device, `--set` / `--extra-yaml`) and score them in batch mode. Zone files
  are read with the monitor's `load_zone_file`; one that cannot be loaded fails its clip
  with one line on stderr (exit 1).
* `benchmark.py`: per-stage ms and FPS of the steady state (after the debris warm-up;
  `--include-debris-warmup`, `--no-detector`); shares the monitor's CLI rows; exit 0 / 1 / 2.
* `inspect_checkpoint.py`: ultralytics checkpoint metadata without torch; an allowlisting
  unpickler refuses checkpoints that reference code outside torch/ultralytics/numpy stubs.
* `tune_hyperparameters.py`: `model.tune` with a validated data yaml, `val=True`, an absolute
  `--project`, finds the run directory ultralytics 8.4 writes, refuses a zero-fitness result.
* `train_vehicle.py`: reproducible `model.train`, writes `metrics.json`, copies `data.yaml`,
  validates on the test split. `_dataset_yaml.py` is their shared dataset check.
* `export_tensorrt.py`: `model.export(format="engine", ...)` for Jetson; needs a local weights
  file.
* `record_legacy_goldens.py`: records the `ae66b8f` event goldens that
  `tests/test_legacy_profile.py` replays with `configs/legacy_rule.yaml`.

## Configs (`configs/`)

* `default.yaml`: every key with its default and a comment (equal to the dataclasses).
* `stresstest.yaml`: the FOX13 clip (clip01): source, polygon, `reference_size [1772, 872]`.
* `legacy_rule.yaml`: rolls back the vehicle decision rule only, to that of `ae66b8f`
  (bottom-center point test, speed only, no near tier, WARNING 3 s / ALARM 8 s,
  `rekey_policy: any`, per-frame labels, no hidden hazards, no WARNING hold,
  `undetected_clock_while_live: false`, ...), to merge under a site config.
  `tests/test_legacy_profile.py` replays the `ae66b8f` goldens with it. The zone stabiliser,
  the debris backstop and the container-clock timestamps are code, not keys, so whole-run
  events still differ from `ae66b8f`. Its 3 s / 8 s are old defaults, so loading it logs the
  stale-default warning, which names it as the deliberate profile.
* `team_rule.yaml`: an approximation of the advisor's "merged" rule (still clusters only,
  near -> ALARM, detector at 5 fps, debris off), for comparisons.

## Benchmark (`benchmarks/team_clips/`)

`clips.yaml` (Drive ids, sizes, sha256, decoded frame counts), `zones/clipNN.json` (the
advisor's zones) and `ground_truth/clipNN.json` (positives, labelled negatives, ignore spans,
decoded durations). Provenance and caveats are in its README.

## Packaging

`pyproject.toml` (version read from `gradecrossing.__version__`): base dependencies numpy +
PyYAML; extras `detector` (headless OpenCV, `ultralytics-opencv-headless`, lap), `display`
(GUI OpenCV, `ultralytics`, lap), `headless` (headless OpenCV only), `dev` (pytest).
`requirements.txt` = base + detector, `requirements-display.txt` = base + display,
`requirements-dev.txt` = pytest; `tests/test_packaging.py` keeps them in step. `setup.py` is a
bare shim for pip < 21.3 editable installs. Console scripts: `gradecrossing-monitor` =
`gradecrossing._entry:monitor`, which imports `gradecrossing.monitor:main` inside the call;
without a usable OpenCV it prints one line naming `.[detector]` / `.[display]` and the
requirements files and exits 2 (an import error inside the package keeps its traceback);
`gradecrossing-evaluate` = `gradecrossing.evaluate:main`, which needs no OpenCV.
