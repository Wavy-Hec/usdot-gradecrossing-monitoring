# USDOT Grade Crossing Monitoring

AI monitoring for rail grade crossings from ordinary fixed cameras. The monitor watches a
zone of interest (ZOI) drawn over the tracks, tracks every vehicle and pedestrian near it,
raises a **WARNING** when a vehicle stops on the tracks (or stops too close to them) and an
**ALARM** when a vehicle stays stopped on the tracks for 3 seconds, and runs a background
model as a backstop for obstructions the detector cannot name (debris). It is designed to
run on an NVIDIA Jetson at the crossing and push events to a dispatch system.

> **Status: research prototype.** The package has been run on the advisor's 8 labelled team
> clips (7.9 minutes, 3 positives, two of them the same event seen by two cameras) and on
> synthetic test scenes. That shows how it behaves; it is far too little data to measure
> detection or false-alarm *rates*. It has not been deployed or soak-tested on a live
> camera. See [Results on the team clips](#results-on-the-team-clips),
> [docs/AUDIT.md](docs/AUDIT.md) for the verified findings that drove the design and
> [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the module contracts.

## The four-stage plan and where it stands

| Stage | Goal | State in this repo |
|---|---|---|
| 1. Dynamic zone detection | Find the ZOI and keep it locked under camera motion | The zone is **drawn once** per camera with `tools/calibrate_zone.py` (an experimental Hough-line seed is offered), which also saves the frame size and a clean calibration image. Every frame it is **stabilised**: Lucas-Kanade + RANSAC (similarity transform by default), ORB fallback, a temporal-consistency gate, registration against the calibration image, verified keyframe refresh, and a health signal. Automatic zone *finding* is not done. |
| 2. Vehicle tracking | Detect and track individual vehicles | YOLO11 (COCO `yolo11n.pt` by default) + **ByteTrack** ids, plus an ID-free "still cluster" so an id switch does not restart a stopped vehicle's clock. Speed is fitted on the bottom-center of the box, in calibration-frame pixels so camera shake is not read as motion. |
| 3. Stuck-vehicle alerting | Alert when a vehicle is stopped on the tracks | Per object: **in the zone** (bottom quarter of the box overlaps the zone by >= 5 %) **and stopped** (speed rule OR still cluster): WARNING at 1.5 s, ALARM at 3 s. A vehicle stopped *near* the tracks gets a WARNING. Alarms latch. |
| 4. Debris detection | Flag unknown objects left on the crossing | MOG2 background, never taught what it must report: box-free warm-up, frozen classification, a selective update of background pixels only, registration to the calibration frame, per-pixel persistence of static candidates, lighting compensation and a hold while a train passes. |

## Install

Python 3.8 to 3.12 (`requires-python = ">=3.8"`; JetPack 5 ships 3.8, the team works on
3.9, the suite is also run on 3.12). Pick **one** OpenCV build per environment:
`opencv-python` (GUI) and `opencv-python-headless` unpack into the same `cv2/` directory,
pip keeps whichever came last, and uninstalling either one breaks `cv2`.

```bash
git clone https://github.com/ruby-gonzalez/usdot-gradecrossing-monotoring.git
cd usdot-gradecrossing-monotoring
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
```

(Until the package is merged there, it lives on the fork
`https://github.com/Wavy-Hec/usdot-gradecrossing-monitoring`, branch `main`.)

| Install | What you get | Use it for |
|---|---|---|
| `pip install -r requirements.txt` or `pip install ".[detector]"` | the monitor (YOLO + ByteTrack) with **headless** OpenCV: `opencv-python-headless`, `ultralytics-opencv-headless`, `lap` | the default: Jetson, Colab, servers |
| `pip install -r requirements-display.txt` or `pip install ".[display]"` | the monitor with a **GUI** OpenCV (`opencv-python`, `ultralytics`, `lap`) | `--display` and the click-to-draw zone editor on a laptop; use it *instead of* the headless set, never on top of it |
| `pip install ".[headless]"` | headless OpenCV, no torch | `calibrate_zone --polygon`, `benchmark --no-detector`, the test suite |
| `pip install .` | numpy + PyYAML only | config, types and `gradecrossing-evaluate`; the monitor cannot import `cv2` (`gradecrossing-monitor` then prints which extra to install and exits 2) |
| `pip install -r requirements-dev.txt` (on top of either runtime file) or `pip install ".[headless,dev]"` | pytest | running the tests |

To switch an environment from the headless to the GUI build:
`pip uninstall -y opencv-python-headless ultralytics-opencv-headless && pip install -r requirements-display.txt`.
Team laptops that already have `ultralytics` + `opencv-python` from the notebooks should use
the display set; adding `[detector]` there installs the headless pair next to it. When
ultralytics is missing, the monitor and the training/export tools give the same advice:
install the detector set or the display set above, not a bare `pip install ultralytics`,
which adds the GUI `opencv-python` next to the headless one (on a Jetson, NVIDIA's torch
wheel first).

On a **Jetson** (JetPack 5, Python 3.8): `python3 -m pip install -U pip` first, install
NVIDIA's torch/torchvision wheels, then `pip install -e ".[detector]"`. The editable install
itself works on the stock pip 20.0.2 (a `setup.py` shim; checked in a py3.8 venv with pip
20.0.2, not on a Jetson), but one dependency of `[detector]` (polars, pulled in by
ultralytics) has only a `manylinux_2_24` aarch64 wheel for cp38, which pip 20.0.2 cannot
use. Ultralytics' Jetson guide pins `numpy==1.23.5` for onnxruntime-gpu, which conflicts
with this package's `numpy>=1.24` floor; neither point has been tried on hardware.

## Quick start

Get the FOX13 stress-test clip (an SUV trapped on the tracks) from the upstream Releases tab
(v1.1) into `videos/`, or fetch it as `clip01` of the team benchmark (see
[Evaluation](#evaluation)): <https://github.com/ruby-gonzalez/usdot-gradecrossing-monotoring/releases>

```bash
python -m gradecrossing.monitor --config configs/stresstest.yaml --events events.jsonl
python -m gradecrossing.evaluate --events events.jsonl --truth benchmarks/team_clips/ground_truth/clip01.json
```

The COCO `yolo11n.pt` is downloaded by ultralytics into the working directory on first use
(root-level `*.pt` files are git-ignored). Add `--display` for a window; it needs the display
install. On a headless build the window fails, is switched off with a HEALTH event, and
monitoring continues.

Run on a camera or RTSP stream after calibrating a zone for it:

```bash
python tools/calibrate_zone.py --source rtsp://camera/stream --out configs/mysite.yaml
python -m gradecrossing.monitor --config configs/mysite.yaml --events events.jsonl --save annotated.mp4
```

`calibrate_zone.py` writes only the site keys into `--out` (on top of the keys the `--config`
base file or an existing `--out` already sets): `video.source`, `zone.polygon`,
`zone.reference_size` (the frame size the polygon was drawn in) and `zone.reference_image`
(a clean copy of the calibration frame, saved as `<config stem>_reference.png`; skip it with
`--no-reference-image`). Everything else keeps the package default. Calibrate on a frame
where the crossing is clear and re-run the tool whenever the camera is re-aimed. The click
editor needs a GUI OpenCV; on a headless box print or save the frame and pass the corners:

```bash
python tools/calibrate_zone.py --source clip.mp4 --save-frame shot.jpg --headless
python tools/calibrate_zone.py --source clip.mp4 --polygon "100,400 500,350 1650,480 1500,600" --out configs/mysite.yaml
```

A zone file in the team's JSON format (`{"vertices": [[x, y], ...]}`) can be used instead of
`zone.polygon`: `zone.polygon_file` in the YAML or `--zone ZONE.json` on the command line.
A `zone.polygon_file` takes precedence over `zone.polygon`, so when `calibrate_zone.py`
re-calibrates a site whose config has one, it removes the key from the config it writes and
says so (the JSON file itself is kept): the new `zone.polygon` is the zone.

Run the tests (CPU only, synthetic frames, no model download; 3 tests need ultralytics and
are skipped without it, and the one that sends a real SIGTERM runs on Linux/macOS only):

```bash
pip install -r requirements-dev.txt && python -m pytest tests -q
```

### Exit status

`python -m gradecrossing.monitor` (also installed as `gradecrossing-monitor`) exits with

| Code | Meaning |
|---|---|
| 0 | the input ended (end of file, `--max-frames`) or the run was stopped (Ctrl-C, q/Esc in the window, SIGTERM/SIGHUP: `systemctl stop`, `docker stop`) |
| 1 | the crossing stopped being watched, or watching never started: the source cannot be opened, the detector cannot be loaded (missing weights, no ultralytics), an output path cannot be created, 5 consecutive frame errors, a live source that could not be reconnected (the summary line says `ABORTED stop_reason=...`) |
| 2 | configuration or usage error: the config cannot be loaded (missing file, YAML syntax error, unknown or retired key, a value of the wrong type or shape), a command-line override cannot be applied, the monitor rejects a setting, the zone calibration does not fit the stream (`zone refused`: another aspect ratio than `zone.reference_size`), or `gradecrossing-monitor` was installed without OpenCV (the message names the extra to install) |

A one-line message goes to stderr, never a traceback. SIGTERM and SIGHUP stop a run the way
Ctrl-C does: RUN_END with `stop_reason` `interrupted`, the sinks are closed (the webhook
queue is drained, see [Events and delivery](#events-and-delivery)) and the exit status is 0.

## How it works

```
frame -> resize to the processing size the first frame fixed (video.resize_width)
      -> ZoneStabilizer        zone polygon + mask + to_reference (current -> calibration pixels) + health
      -> VehicleDetector       YOLO11 + ByteTrack every detect_every_n_frames, predicted boxes in between
      -> StuckVehicleTracker   per object: in/near zone, dwell, speed, still time, stuck time, hidden hazards
      -> InfrastructureDetector (off by default) gates_down / lights_on
      -> DebrisDetector        registered background model, vehicle/person/train boxes excluded
      -> DEGRADED check        what the monitor cannot see right now (HEALTH when it lasts)
      -> AlertEngine           CLEAR / WARNING / ALARM + events -> stdout / JSONL / webhook
```

### Decision rules

- **On the tracks** means: the bottom quarter of a vehicle's box (its full width, the part
  nearest the ground) overlaps the zone by at least 5 % (`stuck.zone_test: footprint`,
  `footprint_fraction`, `footprint_min_overlap`). A box whose *upper* part crosses the zone
  (a car on the road beyond it) is not on the tracks. Persons, trains and unmapped classes
  use the bottom-center point. A vehicle leaves the zone only after its detections have
  been outside it for 0.5 s (`zone_exit_grace_seconds`).
- **Near the tracks** means: not on them, but with the footprint within 0.75 box heights of
  the zone (`close_box_heights`).
- **Stopped** means either rule (`stuck.stationarity: speed_or_still`): the least-squares
  speed of the bottom-center over the last 1 s is below `speed_threshold` (12 px/s at
  processing resolution), or the box keeps re-appearing with IoU >= 0.5 where it stands
  (a still cluster, ID-free, a port of the team's rule). A cluster is anchored on its first
  box and re-anchored once, where its vehicle's speed first drops below `speed_threshold`
  (its clock keeps running), so a vehicle that brakes into its stop and then creeps a little
  stays in its cluster. A new track has no speed until it has 3 samples, so it cannot alarm
  instantly. A box cut by the image border (`still_border_margin`) restarts its cluster's
  still clock only while a train box is beside it (the visible end of a train passing the
  edge of the picture); a vehicle stopped at the edge with no train beside it is still like
  any other.
- **WARNING** when a vehicle has been on the tracks *and* stopped for `warning_seconds`
  (1.5 s), or stopped near the tracks for 1.5 s (`close_policy: warning`; "stopped near the
  tracks"), or on the tracks for 12 s whatever its speed (`dwell_warning_seconds`).
  **ALARM** when it has been on the tracks and stopped for `alarm_seconds` (3 s), or on
  them for 25 s whatever its speed. A vehicle near the tracks never raises an ALARM under the
  default policy (`close_policy: alarm` is the team's "merged" rule).
- **The 3 s / 1.5 s policy is provisional.** The 3 s ALARM is the task definition ("stops on
  the railway tracks ... for more than 3 s") and the team's measured operating point; the
  1.5 s WARNING rests on one trapped-van event seen by two cameras. The team has to confirm
  both. **Rollback:** `configs/legacy_rule.yaml` restores the vehicle rule this package used
  before (bottom-center point test, speed only, no near tier, WARNING 3 s / ALARM 8 s); copy
  its keys into a site config, or run it directly with
  `--config configs/legacy_rule.yaml --zone ZONE.json --source ...`. It rolls back the rule
  only: the zone, debris and timestamp changes stay. On the 8 team clips it gives 0 / 3
  recall (WARNING-or-ALARM 0 / 3), 7 false alarms and 136.2 s of false ALARM (the "before"
  column below has 154.1 s; the difference is the new debris backstop, which no longer
  joins those alarms). Its WARNING 3 s / ALARM 8 s are the old defaults, so loading it logs
  the old-defaults warning (see [Configuration](#configuration)), which names it as the
  profile that restores the old rule on purpose. `configs/team_rule.yaml` approximates the
  advisor's own rule.
- **Gates down no longer shortens the stuck ALARM.** With the infrastructure (gate) model
  enabled, gates down switches to `gates_down_*` thresholds, but only those shorter than the
  normal ones escalate: the WARNING (1.0 s) and the dwell ALARM (6 s), not the 3 s stuck
  ALARM. The engine logs which ones do nothing. The gate model is not validated on the team
  clips (it does not see the gates on clip01), so `infrastructure.enabled` is false.
- **Pedestrians**: a person in the zone (bottom-center point) gives a WARNING after 2 s and
  an ALARM after 10 s. **Trains** never raise an alert. A class in none of
  `detector.vehicle_classes / person_classes / train_classes` raises no alert of its own; the
  detector lists such class names at start-up. Map anything that can obstruct a crossing.
  `detector.ignore_classes` drops a class right after detection (for static false positives).
- **Identity.** A new ByteTrack id that appears on top of a stopped vehicle's live track
  takes over its clocks (an id swap after an occlusion), vehicle to stopped vehicle and
  person to person only; a stopped vehicle keeps its category through label flicker shorter
  than 1 s, and a vehicle of the ALARM episode in progress keeps its vehicle category
  through any run of other labels (a stopped car merging with an arriving train is labelled
  `train` for seconds) as long as the episode lasts and it is in or near the zone. A track
  is dropped only when 2 detector runs in a row missed it and its last detection is more
  than 1 s old; until then a vehicle in the zone that the detector missed keeps its dwell
  and stuck clocks running (`stuck.undetected_clock_while_live`), so missed detections do
  not hold its clocks back.
- **Hidden hazards.** A vehicle of the ALARM episode in progress (the latched part
  included) whose track is lost in or near the zone while a train box covers it stays an
  ALARM ("occluded by train") while covered and for `occlusion_memory_seconds` (3 s) after.
  The episode counts, not only the vehicles raising the ALARM on the last frame: a train
  usually covers a stopped car gradually, which cuts its box short and restarts its clocks,
  so by the time the car is lost often only the latch holds the ALARM. A vehicle of the
  episode last seen neither in nor near the zone (it drove off) is not kept. Measured cost
  on the team clips: no true alarm changes, but false ALARMs that were already raised by
  cars waiting inside the thin zones are held through the train passage: 198.2 s -> 256.4 s
  of false ALARM, 7 false-alarm episodes merged into 5. Setting it to 0 trades the hold for
  fewer false-alarm seconds; that is a team decision.
- **Latch and hold.** An ALARM stays raised for `alerts.latch_seconds` (10 s) after its last
  condition and is re-sent every `repeat_notify_seconds` (30 s). A new cause joining an
  active ALARM is announced at once with ALARM_UPDATE. A WARNING is held for
  `warning_hold_seconds` (2 s) while its vehicles are still in or near the zone, so a clock
  blip does not flap it.

### Debris backstop

The detector only knows its classes. Anything else on the crossing (a shed load, a
stalled car the detector misses at night) is left to the debris detector:

- **Warm-up** (`debris.warmup_seconds`, 2 s): the scene is learned, never under a detection
  box. Pixels not seen box-free are *unknown* and excluded until they have been.
- **Nothing in the zone is ever absorbed.** Every frame is classified against a frozen
  model; every 0.5 s a learning step with a 30 s time constant learns only pixels that are
  background, outside boxes and outside every candidate. Slow lighting drift is followed on
  a crossing that is never empty.
- **Registration.** The frame is warped into the calibration geometry with the zone's
  `to_reference`; while the zone is held the debris timers are held too.
- **Exclusions.** Only vehicle, person and train boxes are excluded (with a 5 s memory for a
  track that recently had one of those labels), so an unmapped class stays visible to it.
- **Must be static.** Persistence is per pixel and only ages while the pixel stays
  foreground; a candidate restarts when more than 15 % of it changed against a frame 0.5 s
  old. It triggers once `min(3 % of the zone, 0.2 % of the frame)` pixels have been static
  for `persist_seconds` (3 s). A candidate covered by a passing vehicle keeps its age for up
  to 5 s.
- **Lighting.** A global gain measured outside the zone is divided out; a component that
  keeps the background's texture is lighting, not an object; a gain beyond 2x for 1 s or
  foreground over 75 % of the zone and of the region around it re-warms the model with a
  HEALTH event (never while debris is triggered).
- **Trains.** While a train box overlaps the zone, and for `hold_linger_seconds` (3 s)
  after, the timers are held: the detector does not label a passing train on every frame
  (gaps of up to 3.5 s on clip03), and debris is blind for those 3 s after the last label.
  A hold that starts again less than `persist_seconds` after the previous one ended (the
  same train) restarts the candidates that appeared in between; candidates from before the
  previous hold keep their age, and nothing restarts while debris is triggered.
- **Ghosts.** Something learned during the warm-up that leaves within 5 s of being learned
  is re-learned instead of alarming; older content that leaves raises the ALARM (the
  fail-safe side).

On the team clips the debris detector raises no false alarm (3 false-alarm episodes
involved debris before). Known gaps: an undetected obstruction already on the crossing while
the model warms up is learned as background; partly visible dark objects of a few hundred
pixels fall under the size bar; the train hold relies on the detector calling the train a
`train`.

### Zone health, DEGRADED and HEALTH events

The zone is never silently wrong. When no transform can be trusted the stabiliser holds the
last good polygon (`healthy=False`); a hold that lasts `zone.health_report_seconds` (0.5 s)
and the recovery are HEALTH events. Until a textured frame arrives the zone is the
calibration polygon, reported as not anchored. With `zone.reference_image` the first live
frames are registered against the calibration image (corrects a camera that moved since
calibration); if none registers within 2 s the stabiliser tracks from the calibration pose
and says so. A polygon less than 90 % inside the frame, or a missing reference image, is a
start-up HEALTH event; an aspect ratio that differs from `zone.reference_size` stops the run
(exit 2).

Every `FrameResult` says whether the monitor is **DEGRADED**, that is partly blind, and why:
`zone uninitialised`, `zone held`, `debris warming up`, `debris held (train)`,
`debris held (registration)`, `debris untrusted`, `detector interval > still gap`,
`gate detector failing`. A CLEAR status while degraded means "nothing seen", not "nothing
there". The status line and the overlay show it; a change that lasts
`alerts.health_debounce_seconds` (1 s) is a HEALTH event with `extra.component == "monitor"`
(a debris warm-up is left out of that report: the debris component reports it).
Other HEALTH events carry `component` `"debris"` (warm-up start and end, trust changes once
they last `alerts.health_debounce_seconds`, re-warms), `"source"` (stream lost, each
reconnect attempt, give-up, a resolution change) or `"pipeline"` (frame-error abort,
detector cannot be loaded, an output switched off). Zone events have no `component`: health
changes carry the stabiliser's `health()`, start-up problems `extra.zone_setup`, a refused
zone `extra.zone_error`. On the team clips there are 3-5 HEALTH events per clip (31 in 7.9
minutes, none before this change). Each run reports its debris warm-up once (start and
end, by the debris component); a train passage adds a DEGRADED report
(`debris held (train)`) and its recovery.

### Events and delivery

Events go to every sink: stdout (one ASCII line), a JSONL log (`alerts.events_jsonl`,
appended, read by `gradecrossing.evaluate`) and a webhook (`alerts.webhook_url`, JSON POST).

- `RUN_START` (`run_id`, `source` with credentials redacted, `config_sha256`,
  `package_version`) and `RUN_END` (`run_id`, `frames`, `duration_s` of video covered,
  `last_timestamp`, `final_status`, `stop_reason`) frame every run, also an aborted one. When
  the input ends during an ALARM, RUN_END is the closing event; no ALARM_END is invented.
- `WARNING_START` / `WARNING_END`, `ALARM_START` / `ALARM_END` on transitions;
  `ALARM_START` with `extra.repeat` every 30 s while an ALARM lasts; `ALARM_UPDATE` when a new
  cause joins it. START and UPDATE events list the objects involved (`extra.objects`: id,
  class, box, in/near zone, overlap, distance, dwell, stuck, still, speed). END events carry
  every track id of the episode, the reason that opened it and `extra.episode_start`. An
  integrator that calls `Monitor.reset()` mid-episode gets an END with status CLEAR and
  `extra.reason = "pipeline re-initialised"`. A live stream that changes resolution keeps
  its alert state (no END), and each pass of a looped file ends with its RUN_END; nothing of
  a pass follows it.
- `HEALTH` (above).

Webhook delivery contract, for whoever writes the receiver: delivery is **at-least-once**
(refused connections, timeouts and HTTP 5xx/408/429 are retried: 3 attempts in all, 0.5 s
and then 1 s apart, within `webhook_deadline_seconds` per event, 10.5 s by default), so
**de-duplicate on the payload's `event_id`** (`"<session>-<n>"`, repeated by every retry).
Under a backlog ALARM events overtake older queued WARNING, HEALTH and RUN events (and a full
queue of 256 drops the oldest non-ALARM event first), so a receiver that tracks the current
status must **order events by the sequence number `n` of `event_id`** (or by `timestamp`
within a run) and skip an event older than the last one it applied; an event's `status` is
the status at its own timestamp. Sinks never block the frame loop. At shutdown queued
HEALTH notices are moved behind the other queued events, so RUN_END and the closing
WARNING_END / ALARM_END are not stuck behind them, and the queue is drained for up to one
`webhook_timeout_seconds` (3 s) per queued event: at least one deadline, at most 30 s (or
one deadline, if that is longer), and only one deadline once a delivery fails (a dead
endpoint does not hold the shutdown up). HEALTH notices still queued then arrive after
RUN_END, which the `event_id` order shows. The JSONL log has no `event_id`.

## Configuration

Everything is in one YAML file; [configs/default.yaml](configs/default.yaml) lists every key
with its default and a comment. A site config only needs the keys that differ. Loading is
strict: an unknown key, a value of the wrong type (`vehicle_classes: car` instead of a list,
an unquoted number for a string), a polygon with fewer than 3 finite points or a malformed
`reference_size` is a config error that names the key (exit 2). Keys removed since the first
version of the package (`debris.empty_seconds_before_refresh`, `debris.empty_learning_rate`)
fail with one error that lists every such key the file sets and what replaced each.

A file that still sets a default that has changed since 0.2.0 (`stuck.warning_seconds: 3.0`,
`stuck.alarm_seconds: 8.0`, `debris.warmup_frames: 300`; the 0.2.0 `calibrate_zone.py`
copied every default into each site file) silently kept the old behaviour. It now loads
with its values and one warning that names each such key with its old and today's default,
and points at `configs/legacy_rule.yaml` as the profile that restores the 0.2.0 vehicle rule
on purpose. When such a file also has a retired key, the load error says it looks like a
full-default 0.2.0 site file. Keep only the keys a site sets on purpose (a fresh
`calibrate_zone.py` file does).

Units: areas are fractions of the zone (or of the frame for `debris.min_object_frame_fraction`),
times are seconds, speeds are pixels per second at processing resolution (metres per second
with `stuck.ground_plane_homography`). Not everything is resolution-independent: these are
in processing pixels and may need retuning when `video.resize_width` or the camera changes a
lot: `stuck.speed_threshold` (px/s), `zone.min_distance`, `zone.block_size`,
`zone.lk_win_size`, `zone.ransac_reproj_threshold`, `zone.max_step_px`, `zone.deadzone_px`,
`zone.reanchor_max_drift_px`, `debris.exclusion_dilate_px` and `debris.morph_kernel` (the
debris model works at most 640 px wide, `debris.process_width`, which bounds the last two).

## Evaluation

`python -m gradecrossing.evaluate` scores the event log of a run against labelled incidents:
recall, time-to-alert, false alarms per hour and FA-seconds (time in ALARM outside the
incidents), the same for WARNING-or-ALARM, plus precision/F1.

```bash
# one clip, team ground-truth JSON (team rules: any overlap, 1 s tolerance, ignore spans)
python -m gradecrossing.evaluate --events events.jsonl --truth benchmarks/team_clips/ground_truth/clip07.json
# one clip, your own CSV with columns video,kind,start_s,end_s (no CSV is committed)
python -m gradecrossing.evaluate --events events.jsonl --truth mytruth.csv --video NAME
# batch: RUN_DIR/<clip>.events.jsonl against GT_DIR/<clip>.json, one table
python -m gradecrossing.evaluate --events RUN_DIR --truth benchmarks/team_clips/ground_truth --out metrics.json
```

The JSONL sink appends, so a log can hold several runs: only the last run is scored unless
`--run all` (each run scored on its own, then added up) or `--run <run_id>` is given.
`--match one_to_one|any_overlap` and `--tolerance-s` override the defaults (one_to_one / 2 s
for CSV, any_overlap / 1 s for team JSON). The duration of a run comes from `--duration-s`,
else its RUN_END, else the truth file. Exit status: 0 scored, 1 a bad input file or missing
log, 2 bad arguments.

### Team-clip benchmark

[benchmarks/team_clips](benchmarks/team_clips/README.md) packages the advisor's 8 labelled
clips: his hand-drawn zones and ground truth (copied from the team's branches, with
provenance), and a registry of the videos with sizes and sha256. The videos stay on the
team's Google Drive:

```bash
python tools/fetch_clips.py                     # download + verify into ~/.cache/gradecrossing/team_clips ($GRADECROSSING_CLIPS)
python tools/run_benchmark.py --out runs/bench --device 0
python tools/run_benchmark.py --out runs/bench_legacy --extra-yaml configs/legacy_rule.yaml
python -m gradecrossing.evaluate --events runs/bench --truth benchmarks/team_clips/ground_truth
```

`run_benchmark.py` writes a minimal config per clip (source, the clip's zone, weights,
device, stdout off, plus your `--set key=value` / `--extra-yaml` overrides), so a run
measures the checkout's defaults; it then scores the logs in batch mode.

### Results on the team clips

COCO `yolo11n.pt` on GPU, the advisor's zones, scored with the team's rules. "Before" is
commit `ae66b8f` (the package as first published), "after" is this branch with its
defaults (a GPU run of the final code). Times-to-alert are measured from the labelled
start of the incident.

| | before | after |
|---|---|---|
| ALARM recall | 0 / 3 | 3 / 3 (clip01 3.48 s, clip07 3.31 s, clip08 2.54 s) |
| WARNING-or-ALARM recall | 0 / 3 | 3 / 3 (WARNING clip01 2.01 s, clip07 1.80 s, clip08 1.04 s) |
| false ALARM episodes (per hour over 475 s) | 7 (53.0) | 5 (37.9) |
| false ALARM seconds | 154.1 | 256.4 |
| WARNING-or-ALARM false seconds | 190.1 | 359.8 |
| WARNING episodes outside the positives | 13 | 6 |
| false alarms involving debris | 3 | 0 |
| HEALTH events per clip | 0 | 3-5 |

What the numbers say, and what they do not:

- **8 clips with 3 positives show behaviour, not rates.** Two of the positives are the same
  van seen by two cameras; one clip more or less changes every ratio.
- clip07 and clip08 (the van) reach ALARM at 5.61 s and 5.84 s, 0.2 s and 0.4 s before the
  train hits it (the van is trapped for 3.5 s and 2.9 s). Before the last rule change they
  got a WARNING only. On clip07 two details decided it: the van's still cluster opened while
  it was still braking, so a 20 px creep after the stop left it (the cluster is now
  re-anchored where a vehicle stops), and from 5.0 s the train hides the van, whose clock
  froze (it now runs until the track is dropped, `stuck.undetected_clock_while_live`). An
  event that short leaves a 3 s stop rule almost no margin, which is what the earlier WARNING
  is for.
- All five remaining false alarms are "vehicle ... stationary in zone" while the gates are
  down: cars (and a truck) waiting at the gates inside the thin hand-drawn zones, which at
  these low camera angles cover road next to the rails. The false-alarm *seconds* grew
  because the 3 s rule now catches those waiting cars sooner, the hidden-hazard memory holds
  them through the train (see above), and a waiting car the detector stops seeing keeps its
  clock until its track is dropped (clip06: two false ALARMs merge into one, +12.9 s).
  **A per-site review of the zones is the next step**; no rule change can tell a car waiting
  inside the zone from one stopped on the rails.
- clip01 is a variable-frame-rate file. Its timestamps follow the container clock; with
  `--fps 31.447` (frame_index / fps, as the advisor's tools count) its times differ by up to
  0.8 s.

## Repository layout

```
gradecrossing/          the monitor package
  monitor.py            Monitor class, run loop, CLI (python -m gradecrossing.monitor)
  zone.py               ZoneStabilizer, build_stabilizer (calibration scaling, reference image)
  autozone.py           experimental Hough-line zone seed for calibration
  detection.py          YOLO11/ByteTrack wrapper, class mapping, infrastructure (gate) detector
  tracking.py           per-object zone relation, speed, still clusters, stuck clocks, hidden hazards
  background.py         debris backstop (selectively updated MOG2)
  alerting.py           CLEAR/WARNING/ALARM engine, latch, events, sinks (stdout, JSONL, webhook)
  evaluate.py           scoring of event logs against ground truth (single clip and batch)
  visualize.py          annotated overlay
  config.py, types.py, geometry.py
  _entry.py             console scripts (gradecrossing-monitor says which extra to install without OpenCV)
configs/                default.yaml (every key, documented), stresstest.yaml (FOX13 clip),
                        legacy_rule.yaml (rollback of the decision rule), team_rule.yaml
benchmarks/team_clips/  the advisor's 8 clips: zones, ground truth, video registry
tools/                  calibrate_zone, fetch_clips, run_benchmark, benchmark, inspect_checkpoint,
                        tune_hyperparameters, train_vehicle, export_tensorrt, record_legacy_goldens
tests/                  pytest suite with synthetic scenes (CPU, no downloads)
notebooks/              the original MOG2+YOLO prototype notebook (kept for reference)
scripts/                legacy helpers used by the notebook
docs/                   ARCHITECTURE.md, AUDIT.md, stress-test findings, code reference PDFs
fine-tuning/            early YOLO-World fine-tuning results
```

## Models

| File | What it is | Trained on | Reported metric | Notes |
|---|---|---|---|---|
| `yolo11n.pt` (auto-download) | COCO-pretrained detector | COCO | n/a | What the monitor uses by default; it has a `train` class, which the train rules need |
| `notebooks/infrastructure tuning best weights/best.pt` | 5-class infrastructure detector (`railroad-crossing-down`, `railroad-crossings`, `railroad-sign`, `railroad-sign-light-off`, `railroad-sign-light-on`) | Roboflow *Railroad v3* (200 images), yolo11n, early-stopped at epoch 213 | test mAP50 0.60, mAP50-95 0.36 (45 images) | Enable with `infrastructure.enabled: true` to get gate state; not validated on crossing video |
| `50_epochs_best.pt` | 6-class vehicle detector (Car, Jeep, Motorcycle, Tricycle, Truck, Van) | Roboflow *CCTV_Vehicles* subset, 50 epochs, started from an earlier fine-tuned checkpoint | val mAP50 0.92, mAP50-95 0.81 | Not used by default; try it with `--weights 50_epochs_best.pt` (no `train` class: trains are left to the debris backstop) |
| `best_vehicle.pt` | Same classes, interrupted run (epoch 21 of 300, optimizer state still attached, 16 MB) | CCTV_Vehicles | val mAP50-95 0.80 at epoch 21 | Superseded by `50_epochs_best.pt`; kept for reference |

`python tools/inspect_checkpoint.py <file.pt>` prints what a checkpoint was trained with,
without torch (torch, ultralytics and numpy objects are stubbed; a checkpoint that references
any other code is refused, nothing in it runs).
The reported metrics above come from those headers; the hyperparameter "tuning" that
produced the training config never ran successfully (see the audit), so treat the
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
  and a longer `Rail-Grade-Crossing-Video.mov`); the team benchmark clips (above).

## Jetson deployment

- Install as in [Install](#install) and run the suite once on the Jetson's own interpreter:
  aarch64, NVIDIA's torch wheels and JetPack's numpy/OpenCV are not what a desktop venv tests.
- Export the detector once. The export tool needs a local weights file (only the monitor
  downloads `yolo11n.pt`), so run the monitor once or download the file first, then
  `python tools/export_tensorrt.py --weights yolo11n.pt --half` and point `detector.weights`
  at the resulting `.engine`.
- **A live stream may change resolution.** The first frame fixes the processing size
  (`video.resize_width` applied to it). An RTSP source that renegotiates its resolution
  mid-run with the same aspect ratio (a camera or NVR profile change, often after a
  reconnect) is resized to that size: the zone, the tracks, the debris background and the
  alert state are kept, and one HEALTH event (`component` "source", `state`
  "resolution_changed") reports it. A new aspect ratio re-initialises the zone, the tracks
  and the debris background with the alert state kept (no END event; a HEALTH event with
  `extra.reset`); the zone set-up then usually refuses the new geometry: HEALTH event, exit
  status 2. A frame that makes any stage raise is counted in the run summary's `errors` and
  skipped; 5 failures in a row stop the run (exit 1).
- A lost live stream is reconnected 5 times with 1, 2, 4, 8, 16 s back-off (HEALTH events for
  each step), then the run stops with exit 1. Each open and read of a stream URL gives up
  after `video.stream_timeout_seconds` (10 s; 0 = the backend default, 30 s) where the
  OpenCV build supports it (4.5.2+ with FFmpeg), so a dead camera is given up after about
  81 s; raise it for a camera whose keyframes are further apart. Use a supervisor
  (`Restart=on-failure`). Live frames are timestamped with the wall clock; file frames with
  the container clock.
- `systemctl stop` / `docker stop` (SIGTERM) and a closed terminal (SIGHUP) end the run like
  Ctrl-C: RUN_END, sinks closed, exit 0. Draining the webhook queue can take up to 30 s on a
  slow endpoint; Docker's default 10 s stop timeout kills the process before that, so give
  the container a longer one (`docker stop -t 40`, `stop_grace_period` in compose).
- Measure before tuning: `python tools/benchmark.py --config configs/mysite.yaml --frames 600`
  prints per-stage milliseconds and FPS of the steady state (after the debris warm-up;
  `--include-debris-warmup` times it too; `--no-detector` measures the non-neural stages).
- `detector.detect_every_n_frames` trades detector cost for tracker-propagated boxes on
  skipped frames; keep the detector interval below `stuck.still_max_gap_seconds` (1 s) or the
  still clusters cannot work (the monitor reports `detector interval > still gap`).
- `alerts.webhook_url` POSTs events to dispatch, `alerts.events_jsonl` keeps the local log,
  `output.snapshot_dir` keeps a JPEG of every ALARM_START.

## Roadmap

1. **Per-site zone review.** Redraw the benchmark and site zones so they cover the rails
   (and the agreed fouling distance), not the road where cars wait at the gates, with
   `tools/calibrate_zone.py` and a reference image; then re-run the benchmark.
2. Confirm the WARNING / ALARM policy (1.5 s / 3 s, near-the-tracks WARNING, hidden-hazard
   memory) with the advisor, and grow the evaluation set (hours of normal traffic, night,
   rain, more positives) before any further tuning.
3. Validate the gate model on crossing video before enabling `infrastructure.enabled`.
4. Long-run soak test on a Jetson with a live stream: reconnects, FPS, HEALTH volume, a
   real `systemctl stop` (the SIGTERM test only runs on Linux/macOS and has not run on the
   Jetson).
5. Rerun vehicle training to completion with the fixed tuning script, a committed `data.yaml`
   and metrics; reconsider the dataset domain.
6. Ground-plane calibration per site so speeds are in metres per second.

## The original notebook

`notebooks/MOG2_YOLO_Detection.ipynb` and `scripts/pipeline_helpers.py` are the prototype this
package grew from. They are kept so the stress-test PDF in `docs/` still matches something
runnable. Commit `d75b495` corrected the notebook's status logic (it used to alarm on any box
touching the zone; it now lets `evaluate_status()` decide: WARNING while a vehicle is in the
zone, ALARM after `VEHICLE_THRESHOLD`), but it still has no tracking, one global timer and a
background model that learns every frame. Use the package for anything beyond a demo.

## Contributors

Ruby Gonzalez, Gael Marquez, David Sanchez

**Advisor:** Dr. Sergei Chuprov, UTRGV
