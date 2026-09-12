# Audit of the MOG2 + YOLO prototype (September 2026)

This document records what was verified about the prototype that existed in this repository as
of commit `30de329` (notebook `notebooks/MOG2_YOLO_Detection.ipynb`, helper module
`scripts/pipeline_helpers.py`, the tuning script and the three checkpoints), and what the
`gradecrossing/` package does about each point. Every finding below was checked directly against
the committed files, the notebook's stored outputs, the checkpoint headers or a reproduction
script; the evidence is quoted so anyone can re-check it.

Cell numbers refer to the notebook **before** the banner cell was added on 2026-09-11 (so "cell
12" is now the 14th cell). The team's own write-up is `docs/MOG2_YOLO_StressTest_Findings.pdf`.

## Summary

| ID | Severity | Finding | Fixed in package by |
|---|---|---|---|
| S1 | critical | Cell 12 alarms the instant any box touches the zone; the vehicle timer and WARNING state are unreachable | `AlertEngine` decides only from per-object `stuck_seconds` / dwell |
| S2 | critical | No class filter: a train, a pedestrian, a bird, any COCO class counts as a "vehicle" | `DetectorConfig.vehicle_classes`, separate person/train handling |
| S3 | high | "In zone" means any pixel of the axis-aligned box overlaps the polygon; a pickup on the shoulder is flagged in the team's own snapshot | bottom-center `pointPolygonTest` |
| S4 | high | No tracking or identity: one global boolean; the timer resets on a single missed detection (13.1 s to 2.8 s on the trapped SUV) | ByteTrack ids, per-object history, grace period |
| S5 | high | No latch, no alert output: status is drawn on a frame that is discarded | latched ALARM, events to stdout / JSONL / webhook |
| C1 | critical | MOG2 keeps learning while monitoring; a stationary object disappears from the mask in about 1 s, below the 3 s debris threshold, so the debris alarm is unreachable for stationary debris | `learningRate=0` while monitoring, slow refresh only when the zone is empty |
| C2 | medium | The exclusion mask is stale on 4 of every 5 frames | detector every frame by default, velocity-propagated boxes otherwise |
| C3 | medium | Stabiliser anchored to frame 1 forever, no inlier gate, silent fallback to the raw polygon | forward-backward LK, inlier and geometry checks, ORB fallback, hold-last-good, keyframe refresh, HEALTH events |
| C4 | medium | Shadow pixels are discarded, so dark objects on grey asphalt can be dropped; `varThreshold=50` is about 3x the OpenCV default | configurable; documented limitation |
| C5 | medium | MOG2 sees the raw frame while only the polygon is warped | documented; zone health gates decisions |
| T1 | high | Hyperparameter "tuning" never ran: fitness 0.0 in all 100 iterations, defaults committed as `best_hyperparameters.yaml` | `tools/tune_hyperparameters.py` validates data, uses `val=True`, refuses a zero-fitness result |
| T2 | high | The infrastructure model's headline 0.93 mAP50 is the validation split used for early stopping; held-out test is 0.60 / 0.36 on 45 images | reported honestly in README |
| T3 | high | Neither vehicle checkpoint is reproducible: both started from an uncommitted `vehicle_tuning/best.pt`, with an uncommitted cfg, on two different datasets; `best_vehicle.pt` is an interrupted epoch-21 snapshot | `tools/train_vehicle.py` writes metrics and copies `data.yaml`; `tools/inspect_checkpoint.py` |
| T4 | medium | The pipeline never loads any fine-tuned checkpoint; all reported results are COCO yolo11n | `--weights` flag, class lists cover both name sets |
| E1 | high | No entry point, no requirements, timers are frame counts, capture rewind and exit-on-read-failure are impossible on a live stream | `python -m gradecrossing.monitor`, `requirements.txt`, timestamp-based timers, reconnect |
| E2 | medium | Every threshold is an absolute pixel count for one 1772x872 clip | fractions of zone area, seconds, px/s |
| D1 | medium | The write-up's numbers do not match the notebook (961 vs 696 vs 872 debris pixels for the same frame) and its "correct" verdicts are structurally guaranteed by S1 | evaluation harness with ground-truth intervals |

## Safety logic

### S1. Any box touching the zone is an immediate ALARM

Cell 12, status block:

```python
# Status — vehicle in ZOI triggers ALARM immediately
if vehicle_in_zoi:
    status_text  = "ALARM"
    status_color = (0, 0, 255)
elif debris_time >= TIME_THRESHOLD:
    ...
else:
    status_text, status_color = evaluate_status(
        debris_time, vehicle_time, vehicle_in_zoi, TIME_THRESHOLD, VEHICLE_THRESHOLD)
```

`evaluate_status()` is only reached when `vehicle_in_zoi` is False, at which point the vehicle
timer block above it has already reset `vehicle_time` to 0.0. So `VEHICLE_THRESHOLD` (3.0 s in
cell 12, 5.0 s in cell 11, commented "seconds a known vehicle can stay in ZOI before alarm") and
the WARNING branch can never fire. The runtime state machine is CLEAR / ALARM. The stress-test
write-up section 6.1 documents this as the intended fix ("immediately triggers ALARM regardless of
the debris pixel count or timer"), and cell 11's stored output shows the same 14 s frame evaluating
to WARNING when `evaluate_status()` is allowed to decide.

Consequence: at 31.45 fps a car that takes 1 to 2 s to cross produces 30 to 60 consecutive ALARM
frames. Wired to dispatch, every routine crossing is an alert.

### S2. No class or confidence filter

`scripts/pipeline_helpers.py`, `check_vehicle_in_zoi()` (original):

```python
for result in yolo_results:
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = result.names[int(box.cls[0])]
        cv2.rectangle(exclusion_mask, (x1, y1), (x2, y2), 255, -1)
        ...
        if cv2.countNonZero(cv2.bitwise_and(bbox_mask, zoi_mask)) > 0:
            vehicle_in_zoi = True
```

`box.cls` is read only to look up a label; `box.conf` is never read; cell 12 calls
`model(frame, verbose=False)` with no `classes=` or `conf=`. The model is the stock COCO
`yolo11n.pt` (cell 9), so `train` (class 6), `person` (0), `bird`, `dog` all set
`vehicle_in_zoi`. The write-up's own table says YOLO "excluded train" at 28 s, i.e. the train's
box was detected and, by this rule, is itself an alarm source. A train occupying the crossing is
the one event that is supposed to happen there.

### S3. Zone test is "any box pixel overlaps the polygon"

Same function: the test is the pixel overlap of a filled axis-aligned rectangle with the polygon
mask. With an oblique camera the box of a vehicle on the road beside the crossing overlaps the
polygon tips in image space. The notebook's own 7 s snapshot (below) labels the pickup on the
right shoulder "car IN ZOI". The standard test is whether the box's bottom-center (ground contact
point) is inside the polygon.

### S4. No tracking; the timer resets on one missed detection

Cell 12 runs `model(frame, ...)` (predict, not track) every fifth frame and collapses all boxes
into one boolean. The vehicle timer is:

```python
if vehicle_in_zoi:
    if vehicle_timer_start is None:
        vehicle_timer_start = current_time
    vehicle_time = current_time - vehicle_timer_start
else:
    vehicle_timer_start = None
    vehicle_time = 0.0
```

A single YOLO frame with no overlapping box resets it. The notebook's stored figure shows the
consequence on the trapped SUV, which never left the zone: "Vehicle time: 13.1s" at 13.9 s and
"Vehicle time: 2.8s" at 27.9 s. The class also flickers `car` / `truck` between panels.

![Cell 12 snapshots from the committed notebook run](img/notebook_cell12_snapshots.jpg)

### S5. No latch and no alert output

Status is recomputed from instantaneous conditions each frame, with the timer start as the only
memory. Nothing logs, writes or sends an alert: `draw_frame()` paints the status onto a frame that
is then discarded except for four hard-coded snapshot timestamps. There is no VideoWriter either.

## Computer-vision pipeline

### C1. The background model absorbs a stationary object in about one second

`apply_mog2_mask()` calls `mog2.apply(frame)` with no `learningRate`, so OpenCV uses its
automatic rate `1/history` on every monitoring frame. The subtractor is built with
`history=300, varThreshold=50, detectShadows=True`. A new static object becomes background once
the old background mode's weight decays below `backgroundRatio` (0.9), i.e. after
`ln(0.9) / ln(1 - 1/300) = 31.6` frames. Reproduction with a synthetic 320x240 scene and the
notebook's settings (script in the package tests, `tests/test_background.py`):

```
history=300 varThreshold=50: object area frame1=4800px, <50% at frame 33   (1.00 s @31.45fps)
history=500 varThreshold=50: object area frame1=4800px, <50% at frame 56   (1.67 s)
learningRate=0:              area at frame 1 = 4800 | at frame 600 = 4800
```

The debris ALARM requires the area to stay above 5000 px for 3.0 s (94 frames) continuously. It
cannot be reached by stationary debris; only by something that keeps moving. The write-up's claim
that MOG2 "detected the gate arm damage" is the transient of the arm moving, and it did not trigger
anything.

### C2. Stale exclusion mask

YOLO runs when `frame_count % 5 == 0`; the boxes from frame 5k are subtracted from the MOG2 masks
of frames 5k+1 .. 5k+4, and frames 1 to 4 have no exclusion at all. A moving vehicle's fresh blob
leaks out from under its old rectangle and counts as debris; slow queued traffic can push that over
the area threshold.

### C3. Stabilisation anchored to frame 1, fail-open

`stabilize_zoi()` tracks the first frame's Shi-Tomasi points (detected without a mask, so some
sit on vehicles) into every later frame, fits a homography with as few as 4 points, applies it
without any inlier count or geometry check, and returns the raw `zoi_points` when
`findHomography` returns None. Nothing records the inlier count in the full pipeline, and the
">100 inliers is reliable" rule quoted in the write-up exists nowhere in code. The stress-test
camera was fixed, so camera motion was never exercised.

### C4. Shadows and sensitivity

`fg_mask[fg_mask == 127] = 0` (and cell 10's `threshold(fg_mask, 200, ...)`) delete every pixel
MOG2 labels as shadow. MOG2 labels as shadow any pixel that is a darker version of the background
colour, so a dark neutral object on grey asphalt or ballast (tyre, black bag, dark car at night) is
removed before it is counted. `varThreshold=50` is about three times the OpenCV default of 16,
which also raises the minimum contrast an object needs.

### C5. Two coordinate frames under camera motion

The polygon is warped by the homography but MOG2 is fed the raw frame, so under real camera
motion the background model compares against a background learned in the original pose.

## Training and models

### T1. The hyperparameter tuning never ran

`notebooks/infrastructure tuning best weights/best_hyperparameters.yaml` header:

```
# 100/100 iterations complete ✅ (383.41s)
# Best fitness=0.0 observed at iteration 1
# Best fitness metrics are {}
```

`tune_results.csv` has 100 rows and the `fitness` column is 0.0 in every one. 100 iterations of
30 epochs cannot complete in 383 s; every iteration failed immediately. Two causes are visible in
`yolo11n_infrastructure_tuning.py`: `model.tune(data=".../data.yml", ...)` while
`model.train(data=".../data.yaml", ...)` (the tune path does not exist), and `val=False`, which by
itself prevents fitness from being computed. The yaml body is Ultralytics' defaults verbatim, and
the training call then used it as `cfg`. The tune also used `optimizer="AdamW"` while training used
the default `auto`, so even a successful search would not have transferred. Both vehicle checkpoints
were trained with a `vehicle_tuning/best_hyperparameters.yaml` produced the same way.

### T2. Validation vs test metrics of the infrastructure model

Checkpoint header (`tools/inspect_checkpoint.py`): 213 epochs logged with `patience=50`, best
validation mAP50 0.930, mAP50-95 0.649. The screenshot committed next to it shows the held-out test
split: 45 images, 106 instances, mAP50 0.60, mAP50-95 0.36, with `railroad-crossing-down` (the class
that matters for gate state) at mAP50 0.478. The validation number selected the checkpoint, so it is
optimistic; the test set is too small to bound the real number tightly. The YOLO-World result in
`fine-tuning/` reports 0.95 mAP50 on the same 45 validation images and has no test metrics, weights
or code.

### T3. Vehicle checkpoints are not reproducible

From the checkpoint headers:

| File | Started from | Dataset | Epochs | Optimizer state | Best val mAP50-95 |
|---|---|---|---|---|---|
| `best_vehicle.pt` (15.9 MB) | `vehicle_tuning/best.pt` | `CCTV_Vehicles.v7i.yolov11/data.yaml` | 21 of 300 planned (interrupted) | present | 0.797 |
| `50_epochs_best.pt` (5.3 MB) | `vehicle_tuning/best.pt` | `vehicle_tuning/vehicle_subfolder/data.yaml` | 50 of 50 | stripped | 0.807 |

Neither started from `yolo11n.pt`; the starting checkpoint and the cfg are not in the repository,
the two runs used different data (per-epoch time 17.6 s vs 149 s suggests the subfolder is a much
smaller subset), both have 6 classes (`Car, Jeep, Motorcycle, Tricycle, Truck, Van`) while the
released dataset has 7 (Bus is missing), and the `notebooks/vehicletuning` file that was deleted in
`f474b5d` was a one-line placeholder. There is no measured detection accuracy on any crossing
footage; the CCTV_Vehicles taxonomy (Jeep, Tricycle) points to a different traffic domain.

### T4. None of the fine-tuned checkpoints is used

Cell 9: `model = YOLO("yolo11n.pt")`. Every reported pipeline result is COCO yolo11n behaviour.
Because S2 treats every class as a vehicle, swapping in the infrastructure checkpoint would make
static signage inside the polygon a permanent alarm.

## Engineering

### E1. Not runnable outside the notebook, not runnable on a stream

There is no script, CLI or service; cell 12 depends on globals from six earlier cells. Cell 1
imports `pipeline_helpers` from `notebooks/` while the module lives in `scripts/` (fixed by adding
`../scripts` to `sys.path`). Cell 0 pip-installs unpinned packages; there is no requirements file;
the stored outputs are from macOS arm64 / Python 3.9. Cell 12 seeks back to frame 0 after warm-up
and breaks out of the loop on the first failed `read()`, both impossible on RTSP. Timers use
`frame_count / fps`, so on a device that cannot keep up, or a stream reporting FPS 0, the
3-second thresholds are wrong or the code divides by zero.

### E2. Everything is in absolute pixels for one clip

`AREA_THRESHOLD = 5000` pixels (3.2 % of the 155 276-pixel zone), the polygon, feature
parameters, MOG2 parameters and font positions are all tuned to 1772x872. A different camera or a
resized stream silently changes the meaning of every threshold.

Repository hygiene: 27 MB of model binaries at the root (one an unstripped mid-run checkpoint),
13.7 MB of embedded notebook outputs, no ignore rules for generated artefacts.

## Documentation vs code

### D1. The write-up's numbers and verdicts

- The debris pixel count for the 14 s frame is 961 in the write-up (three times), 696 in cells
  10 and 11's stored output, and 872 in the cell 12 overlay.
- The summary table marks "Gate damage (14 s) - ALARM - Correct", but section 5.2 of the same
  document says the damage "did not trigger an alarm"; the ALARM at 14 s comes from the SUV.
- "Empty crossing (0 s) - CLEAR - Correct" is vacuous: YOLO first runs on frame 5, and the
  document's own timeline says the SUV is already entering at 0 s.
- "Validated" means four hand-picked overlay frames from one positive-case video. No per-frame
  log, no ground truth, no metric, and the second released test video is never used.
- Section 2.1 describes a right-side-only zone with placeholder coordinates; the code's polygon
  spans from x=100 to x=1650 of a 1772-wide frame.
- The write-up says the pipeline "works correctly across videos of any length and frame rate";
  only the snapshot timestamps were converted to seconds. MOG2 warm-up (300 frames), history and
  the YOLO cadence are frame counts.

## What was verified how

| Method | Used for |
|---|---|
| Reading the committed notebook JSON (source and stored outputs) | S1-S5, C2, C3, E1, D1 |
| Reading `scripts/pipeline_helpers.py` | S2, S3, C1, C3, C4 |
| Viewing the stored cell 12 figure | S3, S4, D1 |
| Parsing `tune_results.csv` and the yaml header; reading the tuning script | T1 |
| Unpickling the three `.pt` files without torch (`tools/inspect_checkpoint.py`) | T2, T3 |
| Viewing the committed validation screenshot | T2 |
| Synthetic reproduction with OpenCV 4.14 | C1 |
| GitHub release metadata of the upstream repository | T3, datasets |

## Review of the package itself

The new package was reviewed the same way (one independent review pass over every module, each
candidate defect confirmed by re-reading the code path or running a probe script). Eight defects
were confirmed and fixed before this document was finalised; they are worth knowing about because
they show where the remaining risk is:

- A ByteTrack id change on a stationary vehicle (re-identification after an occlusion) reset its
  stuck clock. Fixed: a new id whose box overlaps an unclaimed live track inherits that track.
- A stationary object with a class in none of the configured lists raised nothing and was learned
  into the background. Partly fixed: it now blocks the background refresh and the detector warns
  about unmapped class names at start-up; it still raises no alert. Map every class that can
  obstruct a crossing.
- The debris timer had no immobility test, so an undetected moving object (a train under weights
  with no `train` class) tripped a debris alarm after `persist_seconds`. Fixed: the candidate's
  pixels and footprint must both be static for the timer to advance.
- A sudden global illumination step latched a permanent "debris 100 % of zone" alarm. Fixed:
  foreground covering more than `debris.max_area_fraction` of an empty zone *and* of the frame
  outside it is treated as an illumination step; the model re-warms and a HEALTH event is logged.
- One transient exception inside a frame ended the whole run. Fixed: the frame is skipped and
  counted; only five consecutive failures stop the loop.
- Malformed YAML produced a traceback and exit 1 instead of the documented exit 2. Fixed.
- `requires-python` was 3.9 although the code is 3.8-clean (JetPack 5 ships 3.8). Lowered, but
  only syntax-checked: run the suite once on the Jetson's own interpreter.
- One test opened a real window on GUI builds of OpenCV. Fixed.

Two hardenings were added on top: the background refresh is also refused while any candidate
debris streak is in progress (not only after it has triggered), and an LK estimate must keep at
least `zone.min_inlier_fraction` of the reference features as inliers, not just an absolute count.

## Open items the package does not close

- No evaluation on real footage yet: every package test uses synthetic scenes. The evaluation
  harness exists; the labelled event set does not. All new thresholds (`max_area_fraction`,
  `static_motion_*`, `min_inlier_fraction`) are reasoned defaults, not measured ones.
- The illumination-step guard cannot fire once a debris streak has begun, so a lighting step that
  lands while small debris is accumulating persistence is still treated as debris.
- The infrastructure detector's gate-down class is the weakest one it has (mAP50 0.48 on test);
  gate state is a threshold modifier, not a hard gate, for that reason.
- Speeds are in pixels per second unless a per-site ground-plane homography is configured.
- The fine-tuned vehicle weights remain unreproduced until `tools/train_vehicle.py` is rerun with
  a committed `data.yaml`.
- The live-stream path (RTSP reconnect, wall-clock timestamps) has no test coverage because no
  live source was available; only the file path is exercised.
