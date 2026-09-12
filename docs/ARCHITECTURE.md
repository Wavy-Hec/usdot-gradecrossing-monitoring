# Architecture and module contracts

This document is the contract every module in `gradecrossing/` is written against. If you
change a signature here, change the module and its tests in the same commit.

```
video/RTSP ──► Monitor.process_frame(frame, t, i)
                 │
                 ├─ ZoneStabilizer.update(frame, t, scene_static)        → ZoneState (polygon, mask, health)
                 ├─ VehicleDetector.detect(frame)  (every N frames)       → [Detection] with track ids (ByteTrack)
                 ├─ StuckVehicleTracker.update(dets, zone, t)             → [ObjectState] dwell / speed / stuck_seconds
                 ├─ InfrastructureDetector.update(frame, t) (every M)     → GateState (gates_down, lights_on)
                 ├─ DebrisDetector.update(frame, zone.mask, boxes, empty) → DebrisResult (area fraction, persistence)
                 └─ AlertEngine.evaluate(objects, debris, gate, t)        → Assessment (CLEAR/WARNING/ALARM) + Events
                                                                            └─ sinks: stdout / JSONL / webhook
```

Design rules

* **Per-object, not per-frame.** Every safety decision is made on a tracked object with an id.
  "Any box overlapping the zone" is never a trigger.
* **Point-in-polygon on the bottom-center of the box**, never bbox-pixel overlap. A vehicle on the
  adjacent road whose box corner crosses the zone in image space is not in the zone.
* **Stuck = in zone AND speed below threshold, continuously, for T seconds.** Moving traffic that
  crosses the tracks produces at most a transient WARNING, never an ALARM.
* **Class filtered.** Only `vehicle_classes` feed the stuck logic; `person_classes` feed the
  pedestrian logic; `train_classes` are excluded from both (a train in the zone is normal) but still
  excluded from the debris mask.
* **Background frozen while monitoring** (`learningRate=0`), refreshed slowly only when the tracker
  says the zone has been empty for a while *and* nothing is accumulating debris persistence. A
  stationary obstruction must never be absorbed into the background. Debris must also be immobile
  before it can trigger, and a foreground too large to be an obstruction on an empty zone is an
  illumination step that re-warms the model.
* **Alarms latch.** Once raised, ALARM persists `latch_seconds` after the last alarm condition.
* **No hardcoded pixels.** Areas are fractions of the zone area; times are seconds; speeds are px/s
  at processing resolution (or m/s when a ground-plane homography is configured).
* **Testable without torch.** Only `detection.py` imports ultralytics, and only inside functions.
  `Monitor` accepts an injected detector so the whole loop runs with synthetic detections.

Timestamps are float seconds. For files, `t = frame_index / fps`; for live sources, `t` is
`time.monotonic()` relative to start. Frame indices start at 0.

---

## `gradecrossing/types.py` (done)

`Status`, `Detection`, `ZoneState`, `ObjectState`, `DebrisResult`, `GateState`, `Assessment`,
`Event`, `FrameResult`. Read the file; field comments are the spec.

## `gradecrossing/geometry.py` (done)

`as_polygon`, `polygon_mask`, `polygon_area`, `point_in_polygon`, `bottom_center`, `box_center`,
`box_iou`, `box_mask_overlap_fraction`, `dilate_box`, `transform_points`, `homography_is_sane`.

## `gradecrossing/config.py` (done)

Dataclasses `VideoConfig, ZoneConfig, DetectorConfig, InfrastructureConfig, StuckConfig,
DebrisConfig, AlertConfig, OutputConfig, MonitorConfig`; `load_config(path)`, `config_from_dict`,
`save_config`. Unknown keys raise `KeyError`; a top level that is not a mapping raises `TypeError`.

---

## `gradecrossing/zone.py`

```python
class ZoneStabilizer:
    def __init__(self, cfg: ZoneConfig): ...
    def initialize(self, frame_bgr: np.ndarray, timestamp: float) -> ZoneState
    def update(self, frame_bgr: np.ndarray, timestamp: float, scene_static: bool = False) -> ZoneState
    @property
    def reference_polygon(self) -> np.ndarray   # (N,2) float32 in the current reference keyframe
    def health(self) -> dict                    # {"method","inliers","healthy","keyframe_age_s","consecutive_unhealthy"}
```

Behaviour

* `initialize` stores the grayscale reference keyframe, the reference polygon (`cfg.polygon`),
  and Shi-Tomasi features. If `cfg.exclude_features_in_zone`, features inside the polygon are
  discarded (use a `mask` argument to `goodFeaturesToTrack`). Returns a `ZoneState` with
  `method="identity"`, `healthy=True`, `homography=None`.
* `update`:
  1. If `not cfg.stabilize` return identity state.
  2. Track reference features into the current frame with `calcOpticalFlowPyrLK`
     (`winSize=(lk_win_size, lk_win_size)`, `maxLevel=lk_max_level`). Use forward-backward
     check: track back to the reference and drop points whose round-trip error > 1 px.
  3. `findHomography(ref_pts, cur_pts, RANSAC, ransac_reproj_threshold)`. Accept when
     `inliers >= max(min_inliers, ceil(min_inlier_fraction * len(reference_features)))` and
     `homography_is_sane(H, reference_polygon, max_scale_change, max_area_change)`.
     → `method="lk"`, `healthy=True`. The relative bar (`min_inlier_fraction`, default 0.2,
     exposed as `ZoneStabilizer.min_lk_inliers()`) exists because a minority of corners locked
     onto self-similar sleeper/ballast texture can agree on a consistent *wrong* flow and still
     clear an absolute count; 0 disables it. ORB keeps its own `orb_min_matches` bar.
  4. Otherwise, if `orb_fallback`: ORB (`orb_features`) on reference and current, `BFMatcher(NORM_HAMMING)`
     knnMatch k=2 with ratio test `orb_ratio_test`, need `>= orb_min_matches`, then RANSAC homography
     + sanity check. → `method="orb"`, `healthy=True`.
  5. Otherwise **hold the last healthy polygon** (never fall back to the raw reference polygon
     silently): `method="held"`, `healthy=False`, `homography=None`, `inliers=` whatever LK produced.
  6. Keyframe refresh: if `cfg.keyframe_refresh_static_seconds > 0`, `scene_static` is True, the
     current estimate is healthy, and the scene has been static for that long, re-anchor: reference
     keyframe = current gray frame, reference polygon = current warped polygon, re-detect features.
     Reset the static timer on any non-static frame.
  7. `ZoneState.polygon` is `np.int32 (N,2)` rounded; `mask` via `polygon_mask`; `area_px` is
     `countNonZero(mask)`.
* Must cope with frames of a different size than the reference (raise `ValueError`).

Tests (`tests/test_zone.py`, synthetic): build a 640x480 textured image (random noise blurred +
some rectangles), define a polygon; warp the image by a known small affine/homography (e.g. shift
(7,-4) and 2° rotation) with `cv2.warpPerspective`; assert `update` returns a polygon within 1.5 px
RMS of the analytically warped polygon and `method=="lk"`. Second test: feed a blank/uniform frame
→ `healthy=False`, `method=="held"`, polygon equals the last healthy polygon. Third: after N static
frames beyond `keyframe_refresh_static_seconds`, the reference polygon updates (health()["keyframe_age_s"]
resets). Fourth: ORB fallback path is exercised by blurring/large brightness change that makes LK fail
but ORB succeed (if it is hard to construct, test ORB directly through a private helper).

## `gradecrossing/autozone.py` (experimental, stage-1 seed)

```python
def propose_zone(frame_bgr: np.ndarray, infra_detections: Optional[List[Detection]] = None,
                 gate_classes: Sequence[str] = ("railroad-crossing-down", "railroad-crossings")) -> Optional[np.ndarray]
```

Heuristic: Canny + `HoughLinesP`; cluster line segments by angle; pick the two strongest near-parallel
clusters as rail edges; if gate/sign boxes are provided, use their horizontal extent to bound the
crossing; return a 4-point float32 polygon or None. Document clearly that it is a *seed* for
`tools/calibrate_zone.py`, not a production zone. One test: synthetic image with two parallel bright
lines → returns a polygon whose long edges are within 5° of the lines.

## `gradecrossing/detection.py`

```python
class VehicleDetector:
    def __init__(self, cfg: DetectorConfig): ...          # loads ultralytics lazily (import inside)
    @property
    def names(self) -> Dict[int, str]
    def detect(self, frame_bgr: np.ndarray) -> List[Detection]
    def reset(self) -> None                               # reset tracker state (new video)
    def classify(self, cls_name: str) -> str              # "vehicle" | "person" | "train" | "other"

class InfrastructureDetector:
    def __init__(self, cfg: InfrastructureConfig): ...
    def update(self, frame_bgr: np.ndarray, timestamp: float, frame_index: int) -> GateState
```

* When `cfg.tracker` is non-empty, `detect` calls `model.track(frame, persist=True,
  tracker=cfg.tracker, conf=cfg.conf, iou=cfg.iou, imgsz=cfg.imgsz, device=cfg.device or None,
  max_det=cfg.max_det, verbose=False)` and adds `half=True` only when `cfg.half` is True (passing
  `half=False` explicitly triggers an ultralytics 8.4 deprecation warning on every frame; the
  keyword set is identical on every call so the predictor and its trackers are not rebuilt);
  otherwise `model.predict(...)` with the same keywords. Convert `results[0].boxes` to
  `Detection` (xyxy float32 numpy, conf float, cls_id int, cls_name from `names`, track_id from
  `boxes.id` if present else None).
* Class names of the loaded model that are in none of the three lists are logged once, at
  WARNING: such a class raises no alert and is still excluded from the debris mask, so it is a
  blind spot the operator has to close in the config.
* `classify` is case-insensitive against `vehicle_classes / person_classes / train_classes`.
* `InfrastructureDetector.update` runs the model only every `every_n_frames` (and on frame 0);
  otherwise returns the cached `GateState`. `gates_down = (last_seen_down is not None and
  timestamp - last_seen_down <= hold_seconds)`. `confidence` is the max conf of gate-down boxes on
  the last run. `available=True`.
* Tests: `tests/test_detection.py` — `classify` with a stub (no model load); a smoke test that
  loads `50_epochs_best.pt` from the repo root and runs `detect` on a synthetic frame, skipped when
  `ultralytics` is not importable (`pytest.importorskip`).

## `gradecrossing/tracking.py`

```python
class StuckVehicleTracker:
    def __init__(self, cfg: StuckConfig, classify: Callable[[str], str]): ...
    def update(self, detections: List[Detection], zone: ZoneState, timestamp: float) -> List[ObjectState]
    def predict_boxes(self, timestamp: float) -> List[Detection]   # velocity-propagated boxes for skipped frames
    @property
    def zone_occupied(self) -> bool           # any live tracked object (any class) with in_zone True
    @property
    def zone_empty_since(self) -> Optional[float]
    def objects(self) -> List[ObjectState]
    def reset(self) -> None
```

* Detections without `track_id` are assigned ids by an internal greedy IoU matcher
  (`fallback_iou_threshold`), so the module works with any detector.
* A detection carrying a `track_id` the tracker has never seen is first matched, by IoU at the
  velocity-predicted box and above `fallback_iou_threshold`, against the live tracks that no
  detection claims this frame; a match re-keys that track to the new id, keeping `history`,
  `stuck_since`, `in_zone_since` and `seen_count`. This is the in-place id swap ByteTrack
  produces when it re-identifies a vehicle after an occlusion (the train it is stuck in front
  of): a fresh track would zero the counters and drop the status back to CLEAR for a vehicle
  that never moved. Only *live* tracks are candidates - a track already dropped after
  `lost_track_grace_seconds` is never resurrected (see test 5).
* Per id keep a `deque` of `(t, x, y)` bottom-center samples covering at least
  `speed_window_seconds`. Speed = magnitude of the least-squares velocity over the window (px/s).
  If `ground_plane_homography` is set, map points to ground metres first and report m/s
  (`speed_units="m/s"`). Speed is 0 until `min_history_for_speed` samples exist... but
  `stuck_seconds` must NOT accumulate until a speed estimate exists (avoid instant alarms).
* `in_zone` = `point_in_polygon(bottom_center, zone.polygon)`. `dwell_seconds` counts continuous
  in-zone time; `stationary_seconds` counts continuous `speed < speed_threshold`; `stuck_seconds`
  counts continuous (in_zone and stationary). Leaving the zone or moving resets the relevant counter.
* Objects unseen for more than `lost_track_grace_seconds` are dropped. Objects seen fewer than
  `min_track_age_frames` times have `in_zone=False` forced (tracker flicker).
* `zone_occupied` is True when any live object of **any** class is in the zone, `other`
  included: an unmapped class raises no alert and is still excluded from the debris mask, so
  blocking the background refresh is the only protection left against a real obstruction being
  learned into the background.
* `predict_boxes` returns the last box of each live object shifted by velocity × Δt (ids preserved).

Tests (`tests/test_tracking.py`, synthetic detections at 30 fps):
1. Car drives across the zone at 200 px/s → `stuck_seconds` stays 0, `dwell_seconds` grows then resets.
2. Car stops inside the zone → after 3 s `stuck_seconds ≈ 3` (±0.1), `in_zone=True`.
3. Same car with `track_id=None` on every detection still gets a stable id via IoU fallback.
4. `person` detection uses `is_person`; `train` gives `is_train` and does not count as vehicle.
5. Track vanishes → dropped after grace; reappears with a new id → counters start over.
   In contrast, an id that changes *in place* while the old track is still live re-keys that
   track: a stationary car whose id changes at t = 6 s still alarms `alarm_seconds` after it
   stopped (`tests/test_monitor.py::test_scenario_e_track_id_swap_does_not_delay_the_alarm`).
6. Box whose corner overlaps the zone but whose bottom-center is outside → `in_zone=False`.

## `gradecrossing/background.py`

```python
class DebrisDetector:
    def __init__(self, cfg: DebrisConfig): ...
    def update(self, frame_bgr: np.ndarray, zone_mask: np.ndarray, exclusion_boxes: List[np.ndarray],
               zone_empty_seconds: Optional[float], timestamp: float) -> DebrisResult
    @property
    def warmed_up(self) -> bool
    def reset(self) -> None
```

* First `warmup_frames` calls: `mog2.apply(frame)` with the default rate, return a `DebrisResult`
  with `warmed_up=False`, `triggered=False`, `area_px=0`.
* Afterwards choose the learning rate: `empty_learning_rate` if `empty_seconds_before_refresh > 0`
  and `zone_empty_seconds is not None and zone_empty_seconds >= empty_seconds_before_refresh`
  and the previous result was not `triggered`, otherwise `monitoring_learning_rate`. (The Monitor
  additionally passes `zone_empty_seconds=None` while the previous `DebrisResult.persist_seconds > 0`,
  so candidate debris is never absorbed before it can trigger.) Call `mog2.apply(frame, learningRate=lr)`.
* **Refresh is also suppressed while an over-threshold streak is merely in progress**
  (`_over_since is not None`), not only while `triggered`. The tracker cannot see debris, so
  "zone empty" is exactly the state in which undetected debris is present, and at the default
  `empty_learning_rate` MOG2 absorbs a static change in ~21 frames — well inside the default 3 s
  `persist_seconds`. What is left for the refresh is change still below `min_area_fraction`, i.e.
  gradual lighting drift on its way up, which is nibbled away before it can become a candidate.
  (One frame is still learned at the refresh rate: the rate for a frame is chosen before that
  frame is classified. That is ~0.5 % of the weight MOG2 needs, and the test pins it.)
* Foreground: pixels == 255 only (shadows 127 dropped); morphology open+close with
  `morph_kernel`; AND with `zone_mask`; AND NOT with the union of `exclusion_boxes` each dilated by
  `exclusion_dilate_px` (`dilate_box`).
* Connected components; keep those with `area >= min_blob_area_fraction * zone_area`;
  `area_px` = sum of kept areas; `area_fraction = area_px / zone_area`.
* Persistence timer: continuous seconds with `area_fraction >= min_area_fraction` **and the
  candidate immobile**; `triggered` when `>= persist_seconds`. Track with timestamps (not frame
  counts). Immobile means both: at most `static_motion_fraction` of the candidate pixels changed
  by more than `static_motion_diff` grey levels since the previous frame (content), and the
  running intersection of every frame's candidate mask since the streak began still covers
  `min_area_fraction` of the zone (footprint). Either test failing restarts the streak at the
  current timestamp — which is a no-op on the frame an object appears, so the timing of dropped
  debris is unchanged. Rationale: without it, any object the detector does not fire on (a train
  under weights with no `train` class, a crowd) alarms simply by taking longer than
  `persist_seconds` to cross, and the latch then holds the alarm after it has gone. Residual
  limitation: something very large and uniform that is slower than `persist_seconds` across its
  own width can still look static.
* **Illumination guard.** If `max_area_fraction > 0`, the foreground covers at least that
  fraction of the zone *and* at least that fraction of the frame outside the zone (a lighting
  change affects the whole image; even a lorry parked across a tight zone does not), and
  `zone_empty_seconds is not None` (the tracker agrees nothing is
  there), the frame is a lighting step, not an obstruction: `reset()` the model, start a new
  warm-up on this frame, set `last_illumination_event` (a dict the Monitor turns into a HEALTH
  event) and return a warm-up result. Without it such a step starts the timer, the timer blocks
  the refresh that would have absorbed it, and the latch holds an ALARM until the process is
  restarted.
* `learning_rate_used` reported in the result.

Tests (`tests/test_background.py`, synthetic 320x240 frames with static noisy background):
1. After warm-up, a static rectangle placed in the zone remains in the debris mask for 20 s of
   frames with `monitoring_learning_rate=0` (area stays ≥ 90% of its initial value) and
   `triggered` becomes True after `persist_seconds`.
2. With `monitoring_learning_rate=-1` (OpenCV auto) and `history=30`, the same rectangle is absorbed
   (area falls below 50% within a few hundred frames) — documents the failure mode the default fixes.
3. Exclusion: a rectangle covered by an exclusion box produces `area_px == 0`.
4. A rectangle outside the zone produces `area_px == 0`.
5. Refresh: with `zone_empty_seconds` large, a change *below* `min_area_fraction` is absorbed
   within `absorption_frames` frames, while an object over the threshold is **not** absorbed and
   triggers after `persist_seconds`. A global brightness step (over `max_area_fraction`) is
   reported as an illumination event and re-warms the model instead of alarming; with
   `max_area_fraction=0` the same step alarms as debris.
6. A block sliding across the zone with no detection to exclude it never triggers, however long
   it takes; the moment it stops, it does.

## `gradecrossing/alerting.py`

```python
class AlertEngine:
    def __init__(self, stuck_cfg: StuckConfig, debris_cfg: DebrisConfig, alert_cfg: AlertConfig): ...
    def evaluate(self, objects: List[ObjectState], debris: Optional[DebrisResult], gate: GateState,
                 timestamp: float, frame_index: int) -> Tuple[Assessment, List[Event]]
    def add_sink(self, sink) -> None      # sink has .emit(event: Event) -> None
    def reset(self) -> None

class StdoutSink, JsonlSink(path), WebhookSink(url, timeout), CallbackSink(fn)
```

Decision (evaluate):

* thresholds: if `gate.gates_down`: `warn = gates_down_warning_seconds`, `alarm = gates_down_alarm_seconds`,
  `dwell_alarm = gates_down_dwell_alarm_seconds`; else the normal values. `severity_scale = alarm/alarm_seconds`.
* per vehicle object: `stuck_seconds >= alarm or dwell_seconds >= dwell_alarm` → alarm id;
  elif `stuck_seconds >= warn or dwell_seconds >= dwell_warning_seconds` → warning id.
* per person object in zone: `dwell_seconds >= person_alarm_seconds` → alarm; `>= person_warning_seconds` → warning.
  `person_alert` True if any person warning/alarm.
* debris: `debris is not None and debris.triggered` → `debris_alarm=True`.
* raw status: ALARM if any alarm; WARNING if any warning; else CLEAR.
* latch: when raw is ALARM set `latched_until = t + latch_seconds`. If raw < ALARM but
  `t < latched_until`, status = ALARM with reason `"latched"`. Latch clears when it expires.
* reasons: human-readable strings, e.g. `"vehicle 7 stationary in zone 8.3s (gates down)"`,
  `"debris 4.1% of zone for 3.2s"`, `"person 3 in zone 2.5s"`.
* events: on transitions emit `WARNING_START/END`, `ALARM_START/END` (an ALARM_START from WARNING
  implies WARNING_END first). Also while ALARM persists, re-emit `ALARM_START` with
  `extra={"repeat": True}` every `repeat_notify_seconds` (0 disables). Events go to all sinks.
* `WebhookSink` posts JSON with `urllib.request` on a daemon thread; failures are logged, never raised.

Tests (`tests/test_alerting.py`): construct `ObjectState`s directly. Cover: CLEAR with moving vehicles;
WARNING then ALARM as `stuck_seconds` crosses thresholds; gate-down shortening; latch holds ALARM after
object leaves and releases after `latch_seconds`; debris alarm; person warning; event sequence
(`WARNING_START, WARNING_END, ALARM_START, ALARM_END`); JsonlSink writes valid JSON lines; repeat notify.

## `gradecrossing/evaluate.py`

```python
@dataclass
class GroundTruthEvent: video: str; kind: str; start_s: float; end_s: float
def load_truth_csv(path) -> List[GroundTruthEvent]          # columns: video,kind,start_s,end_s
def load_alarm_intervals(events_jsonl_path) -> List[Tuple[float, float]]   # ALARM_START..ALARM_END pairs (open end = +inf)
def score(truth: List[GroundTruthEvent], alarms: List[Tuple[float,float]], duration_s: float,
          tolerance_s: float = 2.0) -> dict
```

`score` returns `{"tp","fp","fn","precision","recall","f1","time_to_alert_s":[...],
"median_time_to_alert_s","false_alarms_per_hour","duration_s"}`. Matching: a predicted alarm interval
matches an unmatched truth interval if they overlap or the prediction starts within `tolerance_s`
of the truth start; time-to-alert = pred_start − truth_start (clipped at 0). Unmatched predictions
are false alarms. CLI: `python -m gradecrossing.evaluate --events events.jsonl --truth truth.csv
--video NAME --duration-s 32.7` prints a table and writes JSON with `--out`.
Tests: hand-built truth/alarm lists with known TP/FP/FN and time-to-alert.

## `gradecrossing/visualize.py`

```python
def draw_overlay(frame_bgr, result: FrameResult, fps_estimate: Optional[float] = None) -> np.ndarray
```

Draws the zone polygon (green healthy / red held), each detection box with `id cls speed stuck_s`
(orange when warning, red when alarm, grey when not a vehicle), debris blobs (magenta), a status banner,
gate state, and a health line (`method inliers`). Returns a copy. Keep it cheap: no full-frame masks
per box. Test: runs without error on a synthetic FrameResult.

## `gradecrossing/monitor.py`

```python
class Monitor:
    def __init__(self, cfg: MonitorConfig, detector=None, infra_detector=None, sinks=()): ...
    def process_frame(self, frame_bgr: np.ndarray, timestamp: float, frame_index: int) -> FrameResult
    def run(self) -> dict     # opens cfg.video.source, loops, returns summary {"frames","alarms","fps"}
    def reset(self) -> None
def open_source(video: VideoConfig) -> Tuple[cv2.VideoCapture, float, bool]   # (cap, fps, is_live)
def main(argv=None) -> int   # CLI
```

`process_frame` order: (1) optional resize, (2) zone update with `scene_static = not tracker.zone_occupied`
and no debris trigger, (3) detector on every `detect_every_n_frames`-th frame counted on an internal
processed-frame counter that `reset()` zeroes (so the detector always runs on the first frame after
construction, reset or a `start_seconds` seek, whatever the source `frame_index`), else
`tracker.predict_boxes(t)`, (4) tracker update, (5) infrastructure update (if enabled),
(6) debris update with exclusion boxes = all detection boxes (any class, real or predicted) and
`zone_empty_seconds = t - tracker.zone_empty_since`, withheld (None) while the previous
`DebrisResult.persist_seconds > 0`; a `DebrisDetector.last_illumination_event` is emitted as a
HEALTH event, (7) alert evaluate, (8) timings per stage in ms. `run` handles files vs live
(`open_source`), fps override, `max_frames`, `loop`, display, save_video, snapshots on `ALARM_START`,
periodic status prints, and a final summary. On a live source a failed `read()` is retried with
reconnect (up to 5 attempts, backoff) rather than exiting.

`run` must also survive a stage raising on one frame: the exception is logged with its traceback,
counted in the summary's `errors`, and the frame is skipped; only `MAX_CONSECUTIVE_FRAME_ERRORS`
(5) failures in a row abort the loop. A `ValueError` on a *live* source is treated as a geometry
change (the stabilizer refusing a frame whose size no longer matches the keyframe, e.g. an RTSP
stream renegotiating its resolution): `self.reset()` re-initialises the pipeline on the new
geometry and a HEALTH event says so. `video.resize_width` pins the processing geometry and avoids
this entirely. `main` returns 0 on success, 1 when the run cannot start, and 2 for *any* failure to
load the config — missing file, YAML syntax error (a tab in the indentation), a top level that is
not a mapping, an unknown key or a value of the wrong type — never a traceback.

CLI: `python -m gradecrossing.monitor --config configs/stresstest.yaml [--source PATH] [--weights W]
[--display] [--save OUT.mp4] [--events events.jsonl] [--max-frames N] [--infra-weights W]
[--no-stabilize] [--print-every N]`. Any flag overrides the YAML. The implementation also exposes
`--infra`, `--fps`, `--resize-width`, `--start-seconds`, `--loop`, `--live`, `--detect-every`,
`--device`, `--snapshot-dir`, `--webhook`, `--no-stdout` and `--log-level`, each mapping to the
config key of the same name (`--help` lists them).

Tests (`tests/test_monitor.py`): a `FakeDetector` that returns scripted detections per frame index;
synthetic 640x360 frames. Scenario A: a car crosses the zone in 2 s → no ALARM ever, at most WARNING.
Scenario B: a car stops in the zone → ALARM after `alarm_seconds` (+ tolerance), `ALARM_START` event
emitted once, latched after the car disappears, cleared after `latch_seconds`. Scenario C: gate-down
state injected via a fake infra detector → ALARM at the shorter threshold. Scenario D: a static blob
drawn into the frames inside the zone with no detection → debris ALARM after `persist_seconds`, and it
does not self-clear over 20 s. Scenario E: a stopped car whose detector id changes mid-dwell still
alarms `alarm_seconds` after it stopped, and stays one track. Also test `detect_every_n_frames=3`
keeps exclusion boxes moving with the object (debris area stays 0 while a moving detected car
passes); a stationary object of an unmapped class holds the background refresh at rate 0; a frame
that raises is counted in `errors` and skipped while five in a row abort the run; a `ValueError` on
a live source re-initialises the pipeline and emits a HEALTH event; `cv2.imshow` raising `cv2.error`
(monkeypatched, so GUI builds never open a window in the suite) disables the display without
stopping the run; a malformed config - unknown key, tab indentation, non-mapping root - exits 2.

## Tools (`tools/`)

* `calibrate_zone.py` — open a video/image, show first frame (optionally with `autozone.propose_zone`
  seed), click polygon corners, write/merge into a YAML config (`--out configs/mysite.yaml`). Print the
  polygon so headless users can paste it.
* `inspect_checkpoint.py` — print ultralytics checkpoint metadata (epoch, train_args, metrics, names,
  optimizer presence) without torch (zip + custom unpickler).
* `tune_hyperparameters.py` — corrected `model.tune` call: validates the data yaml exists before
  starting, `val=True`, sensible `iterations`/`epochs` args, asserts the resulting fitness > 0 and
  refuses to write `best_hyperparameters.yaml` otherwise.
* `train_vehicle.py` — reproducible `model.train` with argparse, writes `metrics.json` and copies
  `data.yaml` next to the weights; validates on the **test** split at the end.
* `export_tensorrt.py` — `model.export(format="engine", half=True, imgsz=...)` with notes for Jetson.
* `benchmark.py` — runs the Monitor on a source for N frames and prints per-stage ms and FPS.

## Configs (`configs/`)

* `default.yaml` — every section with commented defaults.
* `stresstest.yaml` — the FOX13 clip: polygon `[[100,400],[500,350],[1650,480],[1500,600]]`,
  source `videos/Rail-Grade-Crossing-StressTest.mov`, `detector.weights: yolo11n.pt`.

## Packaging

`requirements.txt` (runtime), `requirements-dev.txt` (pytest), `pyproject.toml` with
`[project.scripts] gradecrossing-monitor = "gradecrossing.monitor:main"`.
