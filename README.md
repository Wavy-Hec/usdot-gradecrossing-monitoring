# USDOT Grade Crossing Monitoring

AI monitoring for rail grade crossings from ordinary fixed cameras. The monitor watches a
zone of interest (ZOI) on the tracks, tracks every vehicle and pedestrian passing through it,
raises a **WARNING** when something stops there and an **ALARM** when it stays stopped (sooner
when the gates are down), and flags debris that YOLO cannot name. It is designed to run on an
NVIDIA Jetson at the crossing and push events to a dispatch system.

> **Status: research prototype.** The pipeline has been exercised on one 32-second news clip
> of an SUV trapped on the tracks and on synthetic test scenes. There is not yet an
> evaluation set of normal traffic, so false-alarm rates are unmeasured. See
> [docs/AUDIT.md](docs/AUDIT.md) for the verified findings that drove the current design and
> the [Roadmap](#roadmap) for what is next.

## The four-stage plan and where it stands

| Stage | Goal | State in this repo |
|---|---|---|
| 1. Dynamic zone detection | Find the ZOI automatically and keep it locked under camera motion | Zone is **calibrated once** with `tools/calibrate_zone.py` (an experimental Hough-line seed is offered), then **stabilised** every frame with Lucas-Kanade + RANSAC, ORB fallback, keyframe refresh and a health signal |
| 2. Vehicle tracking | Detect and track individual vehicles | YOLO11 + **ByteTrack** ids, per-object bottom-center (ground contact point) history, speed and dwell time (StuckVehicleTracker samples `bottom_center(box)`, never the box centroid) |
| 3. Stuck-vehicle alerting | Alert when a vehicle is stationary in the zone | Per-object **stuck = in zone AND below speed threshold for T seconds**, WARNING then ALARM, thresholds shortened when the infrastructure model sees gates down, alarms latch |
| 4. Debris detection | Flag unknown objects dropped on the crossing | MOG2 foreground minus every detected box of any class (tracked or not, real or velocity-predicted on skipped detector frames), **background frozen while monitoring**, persistence timer, area thresholds as a fraction of the zone |

## Quick start

```bash
git clone https://github.com/Wavy-Hec/usdot-gradecrossing-monitoring.git
cd usdot-gradecrossing-monitoring
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Get the test clip from the upstream Releases tab (v1.1) and put it in `videos/`:
<https://github.com/ruby-gonzalez/usdot-gradecrossing-monotoring/releases>

Run the stress-test clip with the COCO-pretrained YOLO11n (downloads automatically on first use):

```bash
python -m gradecrossing.monitor --config configs/stresstest.yaml --display --events events.jsonl
```

Run on a live camera or RTSP stream after calibrating a zone for it:

```bash
python tools/calibrate_zone.py --source rtsp://camera/stream --out configs/mysite.yaml
python -m gradecrossing.monitor --config configs/mysite.yaml --events events.jsonl --save annotated.mp4
```

On a headless box (no OpenCV window) pass the corners directly, or save the first frame with the
Hough seed drawn on it and read the coordinates off that:

```bash
python tools/calibrate_zone.py --source clip.mp4 --polygon "100,400 500,350 1650,480 1500,600" --out configs/mysite.yaml
```

Score a run against labelled events:

```bash
python -m gradecrossing.evaluate --events events.jsonl --truth data/truth.csv --video Rail-Grade-Crossing-StressTest.mov --duration-s 32.7
```

Run the tests (no GPU or model download needed):

```bash
pip install -r requirements-dev.txt && python -m pytest tests -q
```

## How it works

```
frame ─► ZoneStabilizer ─► zone polygon + mask + health
      ├► VehicleDetector (YOLO11 + ByteTrack) ─► boxes with ids
      │      └► StuckVehicleTracker ─► per id: in_zone, dwell_s, speed, stuck_s
      ├► InfrastructureDetector (optional, every N frames) ─► gates_down / lights_on
      ├► DebrisDetector (MOG2 minus all detected boxes, inside zone) ─► area %, persistence
      └► AlertEngine ─► CLEAR / WARNING / ALARM + events ─► stdout / JSONL / webhook
```

The decision rules, in plain language:

- **A vehicle is "in the zone" when the bottom-center of its box is inside the polygon.** Box
  corners overlapping the zone do not count, so traffic on the adjacent road is ignored.
- **Stuck** means in the zone *and* slower than `stuck.speed_threshold` continuously. After
  `warning_seconds` the status is WARNING, after `alarm_seconds` it is ALARM. A car that crosses
  normally never trips either. A vehicle that creeps inside the zone for a long time trips the
  separate dwell thresholds.
- **Gates down shorten everything.** With the infrastructure model enabled, a gate-down
  detection switches to `gates_down_*` thresholds (default 1 s / 3 s).
- **Only vehicle classes feed the stuck logic.** Trains in the zone are normal and ignored;
  pedestrians have their own thresholds. Every detected box of any class is subtracted from the
  debris mask. A class that is in *none* of the lists is therefore a blind spot: it raises no
  alert and is not debris either (it does block the background refresh). The detector logs the
  unmapped names of its model at start-up — map anything that can obstruct a crossing into
  `detector.vehicle_classes` / `person_classes`.
- **Debris has to be immobile.** The persistence timer only advances while the candidate's
  pixels stop changing *and* its footprint stops moving, so an object the detector missed does
  not raise a debris alarm merely by taking longer than `persist_seconds` to cross the zone.
  A long train under weights that have no `train` class can still trip it: use weights with a
  train class on clips with rail traffic, or set `debris.enabled: false` for them.
- **The background model is frozen while monitoring.** A stationary obstruction is never
  absorbed into the background, so a debris alarm does not self-clear. The model adapts slowly
  only when the tracker has seen the zone empty for a while and no candidate debris is
  accumulating persistence, so gradual lighting drift does not pile up. A sudden global change
  (lights switching, a day/night camera's IR-cut switch) that covers more than
  `debris.min_area_fraction` of the zone is treated as debris, alarms after `persist_seconds`
  and is never absorbed until the monitor is restarted. The one exception is a change big
  enough to be impossible as an obstruction — more than `debris.max_area_fraction` (0.75) of a
  zone the tracker reports empty *and* of the frame outside it: that is recognised as an
  illumination step, so the background model re-warms itself and a HEALTH event is logged
  instead of a permanent alarm.
- **Alarms latch** for `alerts.latch_seconds` after the last alarm condition, and an ALARM is
  re-notified every `repeat_notify_seconds` while it persists.
- **Zone health is a first-class signal.** When the homography cannot be trusted the monitor
  holds the last good polygon, marks the zone unhealthy and emits a HEALTH event instead of
  silently using the uncalibrated polygon.

All thresholds live in one YAML file and none of them is a pixel count: areas are fractions of the
zone, times are seconds, speeds are pixels per second at processing resolution (or metres per
second if you supply a ground-plane homography). See [configs/default.yaml](configs/default.yaml)
for every knob with comments and [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for module contracts.

## Repository layout

```
gradecrossing/          the monitor package
  monitor.py            Monitor class + CLI (python -m gradecrossing.monitor)
  zone.py               ZoneStabilizer (LK + RANSAC, ORB fallback, keyframe refresh, health)
  autozone.py           experimental Hough-line zone seed for calibration
  detection.py          YOLO11/ByteTrack wrapper + infrastructure (gate) detector
  tracking.py           per-object dwell / speed / stuck bookkeeping
  background.py         MOG2 debris detector with frozen background
  alerting.py           CLEAR/WARNING/ALARM engine, latching, event sinks
  evaluate.py           event-level precision / recall / time-to-alert / false alarms per hour
  visualize.py          annotated overlay
  config.py, types.py, geometry.py
configs/                default.yaml (documented defaults), stresstest.yaml (FOX13 clip)
tools/                  calibrate_zone, benchmark, inspect_checkpoint, tune_hyperparameters,
                        train_vehicle, export_tensorrt
tests/                  pytest suite with synthetic scenes (runs on CPU, no downloads)
notebooks/              the original MOG2+YOLO prototype notebook (kept for reference)
scripts/                legacy helpers used by the notebook
docs/                   ARCHITECTURE.md, AUDIT.md, stress-test findings, code reference PDFs
fine-tuning/            early YOLO-World fine-tuning results
```

## Models

| File | What it is | Trained on | Reported metric | Notes |
|---|---|---|---|---|
| `yolo11n.pt` (auto-download) | COCO-pretrained detector | COCO | n/a | What the monitor uses by default. Classes car / truck / bus / motorcycle / person / train |
| `notebooks/infrastructure tuning best weights/best.pt` | 5-class infrastructure detector (`railroad-crossing-down`, `railroad-crossings`, `railroad-sign`, `railroad-sign-light-off`, `railroad-sign-light-on`) | Roboflow *Railroad v3* (200 images), yolo11n, early-stopped at epoch 213 | test mAP50 0.60, mAP50-95 0.36 (45 images) | Enable with `infrastructure.enabled: true` to get gate state |
| `50_epochs_best.pt` | 6-class vehicle detector (Car, Jeep, Motorcycle, Tricycle, Truck, Van) | Roboflow *CCTV_Vehicles* subset, 50 epochs, started from an earlier fine-tuned checkpoint | val mAP50 0.92, mAP50-95 0.81 | Not used by default; try it with `--weights 50_epochs_best.pt` |
| `best_vehicle.pt` | Same classes, interrupted run (epoch 21 of 300, optimizer state still attached, 16 MB) | CCTV_Vehicles | val mAP50-95 0.80 at epoch 21 | Superseded by `50_epochs_best.pt`; kept for reference |

`python tools/inspect_checkpoint.py <file.pt>` prints exactly what a checkpoint was trained with,
without needing torch. The reported metrics above come from those headers; the hyperparameter
"tuning" that produced the training config never ran successfully (see the audit), so treat the
fine-tuned weights as unreproduced until `tools/train_vehicle.py` is rerun with a committed
`data.yaml` and metrics.

Weights belong on the Releases tab rather than in git; they are in the tree today only because
that is where the team put them.

## Datasets

- **Railroad v3** (infrastructure, 200 labelled images with augmentation):
  <https://universe.roboflow.com/obstacle-detection/railroad-crossings-object-detection> and upstream release v1.0
- **CCTV_Vehicles** (AtledTech, ~6.6k images, 7 classes): upstream release v1.2. The Jeep and
  Tricycle classes suggest South-East-Asian traffic; a US crossing set or UA-DETRAC is a better
  domain match for the next vehicle model.
- **Test videos**: upstream release v1.1 (`Rail-Grade-Crossing-StressTest.mov`, the FOX13 clip,
  and a longer `Rail-Grade-Crossing-Video.mov`).

## Evaluation

There is no quantitative evaluation of the *pipeline* yet, only of the detectors on still
images. `gradecrossing/evaluate.py` scores a run's event log against a CSV of ground-truth
intervals (`video,kind,start_s,end_s`) and reports event-level precision and recall,
time-to-alert, and false alarms per hour. The intended evaluation set:

1. Hours of normal traffic from public live crossing streams (day, night, rain, train passages).
2. Every trapped-vehicle news clip that can be found, with event start and end labelled.
3. Pedestrian and cyclist crossings.

Until that exists, any "validated" claim refers to the single stress-test clip.

## Jetson deployment

- **Python 3.8 or newer** (`pyproject.toml` pins `requires-python = ">=3.8"`, so `pip install -e .`
  works on a JetPack 5 box). The package is 3.8-syntax clean and uses no 3.9+ runtime API — a test
  parses every module with `ast.parse(..., feature_version=(3, 8))` — but CI runs 3.12 and the team
  runs 3.9/3.10, so run the suite once on the Jetson's own interpreter after installing.
- Export the detector once: `python tools/export_tensorrt.py --weights yolo11n.pt --half` and
  point `detector.weights` at the resulting `.engine`.
- **Pin the processing geometry on live streams** with `video.resize_width`. An RTSP source that
  renegotiates its resolution mid-run changes the frame size under the zone polygon; the monitor
  then logs a HEALTH event and re-initialises the pipeline on the new geometry (losing the
  background model and every track), which `resize_width` avoids entirely. A frame that makes any
  stage raise is counted in the run summary's `errors` and skipped — only
  `MAX_CONSECUTIVE_FRAME_ERRORS` failures in a row stop the loop.
- Measure before tuning: `python tools/benchmark.py --config configs/mysite.yaml --frames 600`
  prints per-stage milliseconds and end-to-end FPS.
- `detector.detect_every_n_frames` trades detector cost for tracker-propagated boxes on skipped
  frames; `video.resize_width` trades resolution for speed (the zone polygon is scaled with it).
- Use `alerts.webhook_url` to POST events to dispatch, `alerts.events_jsonl` for the local log,
  and `output.snapshot_dir` to keep a JPEG of every alarm.

## Roadmap

1. Build the evaluation set and report event-level numbers before any further modelling.
2. Rerun vehicle training to completion with the fixed tuning script, a committed `data.yaml`
   and metrics; reconsider the dataset domain.
3. Auto-ZOI: seed the polygon from the infrastructure detector (gate and signal masts on both
   sides) plus track lines, confirm once per install.
4. Ground-plane calibration per site so speeds and dwell are in metres and seconds.
5. Long-run soak test on a Jetson with a live stream; log zone-health and FPS.

## The original notebook

`notebooks/MOG2_YOLO_Detection.ipynb` and `scripts/pipeline_helpers.py` are the prototype this
package grew from. They are kept so the stress-test PDF in `docs/` still matches something
runnable, but the notebook's cell 12 status logic alarms on every vehicle that touches the zone;
use the package for anything beyond a demo.

## Contributors

Ruby Gonzalez, Gael Marquez, David Sanchez

**Advisor:** Dr. Sergei Chuprov, UTRGV
