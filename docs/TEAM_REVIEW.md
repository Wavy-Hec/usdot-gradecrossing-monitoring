# Review of the team branches (October 2026)

This is a review of the work pushed to the team repository
(`ruby-gonzalez/usdot-gradecrossing-monotoring`) since the April prototype. Upstream `main` is still
the April commit `30de329`. All later work lives on branches:

| Branch (tip) | Author | What it is |
|---|---|---|
| `dataset-pipeline` = `ruby-branch` (`23bf06d`, 2026-06-16); PR #1 `4e93ead` closed unmerged | David Sanchez (last two commits by gaelm2005) | `pipeline/` package replacing the April notebook: video manifest + cached download, MOG2 + YOLO/ByteTrack, per-ID dwell timers, run folders |
| `gaels_branch` (`c30836a`, 2026-07-27); PR #2 (Venom-Ocelot) merged into it | Gael Marquez | `bgs_playground/`: pybgs SuBSENSE/ViBe background subtraction, blob analysis, DeepSORT, dwell state machine, ROI stabiliser, rail-segmentation ROI |
| `sergei-crossing-experiments` (`2f90d19`, 2026-10-04), built on `gaels_branch` | Dr. Sergei Chuprov | `occupancy_monitor/` package (33 tests), `pipelines/` (YOLO-only, V-JEPA, occupancy), experiments E1-E12 in `EXPERIMENTS.md` |

**How this was checked.** Every finding below comes from running the code, not from reading it:
the branch tips were exported and run on the 8 labelled team clips (`clip01`-`clip08`, 7.9 min of
decoded video), with the advisor's hand-drawn zones and ground truth
(`occupancy_monitor/zones/`, `occupancy_monitor/ground_truth/`). Each reviewer's findings were then
re-checked by an independent verifier who tried to refute them. Nothing was refuted outright; where
a verifier corrected a number or a claim, the corrected version is the one given here. Severity is
by safety impact: a missed alarm ranks above a false alarm, which ranks above maintainability.

The same benchmark is now part of our package (`benchmarks/team_clips/`, `tools/run_benchmark.py`),
so every number for a pipeline that writes our event format can be re-run with one command.

---

## Summary

| | Blocking errors | Main accuracy problem | Worth keeping |
|---|---|---|---|
| **dataset-pipeline** | Manifest at the tip has unresolved merge-conflict markers, so the notebook fails at cell 1. `gdown>=4.4` resolves to gdown 6, which has no `fuzzy=` argument (TypeError on Python >= 3.10). | Any pixel of a box touching the zone counts as "in the zone": in ALARM 52 % (team model) to 78 % (COCO) of normal-operation time. The team checkpoint misses both trapped-van clips. | Manifest + cached download, per-run provenance folder, the never-built "blind-spot" diagnostic idea |
| **gaels_branch** | Cannot be imported on Python 3.9 (PEP 604 annotations); `reqs.txt` cannot be installed on macOS or a Jetson; 645 MB of videos committed; the branch deletes `main`'s weights, PDFs and notebook. | The dwell state machine alarms on anything that stays 30 frames in the zone, moving or not: every negative clip false-alarms (trains, gate arms). The stopped van alarms for 1.0 s of 3.5 s and clears 0.8 s before impact. | The ROI stabiliser's similarity model + dead-zone (ported into our zone stabiliser), the rail-segmentation corridor as a calibration seed |
| **occupancy_monitor** | None: tests pass on 3.9 and 3.12, E2 reproduces exactly. | A ghost baked into the bootstrap reference becomes a permanent ALARM; a train detection cancels active alarms zone-wide; cells behind a waiting vehicle are blind with no DEGRADED status. | Frame registration, per-cell reference comparison, grace/hold timers, seconds-based parameters, DEGRADED status, the ground-truth format (ported or adopted) |
| **experiments E1-E12** | None: every result file that can be rebuilt from committed inputs is reproduced byte for byte. | The recommended YOLO-geometry rule does not transfer to the team's low-angle gated clips (1/3 recall, 7 false alarms in 7.9 min, in ALARM 62 % of normal time). Labels are by the rule's author with system evidence on screen; recall is pooled over the scanners' own finds. | Footprint test, ID-free stationarity idea, labelled negatives + ignore spans (all ported) |

Our own package did not do better on the same clips before this review: **0 of 3 positives, 7 false
alarms, 154 s of false ALARM**. What changed in response is in section 5.

---

## 1. David Sanchez: `dataset-pipeline` and PR #1

**What it does.** `pipeline/runner.py` trains MOG2 on the first 300 frames of the clip in a separate
pass, then reads the clip again. Every frame it warps a hand-typed zone with LK + RANSAC; every 5th
frame it runs YOLO `model.track` (ByteTrack, conf 0.25). `VehicleTracker` keeps a dwell timer per
track id (since `0b2ae8c`). Debris = frozen-MOG2 foreground minus all YOLO boxes. ALARM when a box
touches the zone and the longest dwell passes a per-clip threshold, or when debris >= 5000 px lasts
long enough. `io.py` adds a YAML video manifest with cached Google Drive downloads and one output
folder per run.

**Measured end to end** (tip runner, advisor zones, David's last valid thresholds from `ca5ca4e`):
team 6-class checkpoint `50_epochs_best.pt`: recall 1/3 (both trapped-van clips missed), 5 false
alarms, 222 s in false ALARM over 428 s of normal operation. COCO `yolo11n`: recall 3/3, but every
negative clip alarms (335 s, 78 %).

### Errors

- **High: the tip cannot run.** `data/manifests/videos.yaml` has unresolved conflict markers from
  merge `23bf06d` (lines 10/18/25 and 66/69/86); `yaml.safe_load` raises ScannerError, so
  `find_video()` and notebook cell 1 fail. The two sides also disagree on clip07's thresholds.
- **Medium: `gdown>=4.4.0` does not fix the download** (`pipeline/io.py:78`). pip resolves gdown 6.x
  on Python >= 3.10, and gdown 6 removed `fuzzy=`: `TypeError: download() got an unexpected keyword
  argument 'fuzzy'` before any network access. This is exactly the committed cell 1 output. Pin
  `gdown>=4.4,<6` or drop `fuzzy=True`.
- **High: any pixel of the box counts as in the zone** (`yolo_detector.py:50`, same test in
  `visualization.py:41`). Cars beside or in front of the thin zones keep the clip in ALARM; replaying
  the same detections with a bottom-quarter footprint test cuts the team-model false-ALARM time from
  222 s to 66 s (clip06: 108 s to 0). With COCO the gain is small (335 s to 319 s), because far-side
  cars really sit inside the thin bands at these low camera angles.
- **High: the "safety latch" is a hair trigger** (`runner.py:163-189`). After the first ALARM of a
  run, both timers are clamped to their thresholds for the rest of the run, so every later box that
  touches the zone is an instant ALARM, even an id-less box, which `tracker.py:15-17` says cannot
  alarm. It still drops to CLEAR as soon as the box leaves, so it does not hold an alarm either.
- **High: no class handling** (`yolo_detector.py:44-53`). With the team checkpoint (no train class)
  the train becomes debris (clip03 ALARM for the whole 32 s freight-train passage); with COCO,
  persons, trains, boats and airplanes in the zone all count.
- **High: a dwell timer is deleted the first stride its id is missing** (`tracker.py:31-35`): a car
  stopped 6 s and missed on 1 stride in 4 never alarms. On clip07 the miss is detector recall (no
  box at all from 4.5 to 5.84 s), not the timer.
- **Medium: MOG2 is trained on the clip being monitored** (`mog2_anomaly.py:13-32`). clip07/08 have
  only 211/241 decodable frames, so the trapped van is part of the "background"; the zone
  foreground at 3-4 s drops from 2,090-4,261 px to 40-592 px. The two-pass read cannot work on a
  live stream.
- **Medium: ByteTrack state leaks between videos** when the notebook reuses the model: clip08 after
  clip07 starts at tracker frame 43 with clip07's lost tracks revived (77 vs 62 ALARM frames).
- **Medium: crash on a featureless first frame** (`zoi.py:15`): `goodFeaturesToTrack` returns None
  and `calcOpticalFlowPyrLK` asserts. The homography is applied with as few as 4 points and no
  inlier or geometry check (one clip05 frame moves the zone 211 px).
- **Medium: the committed notebook watches the wrong area.** Cell 1 selects clip08 (640x360) while
  cell 3's zone was drawn for clip01 (1772x872): 833 px of it fall inside the frame. The zone cannot
  be stored per video (`VideoEntry(**entry)` rejects unknown keys) and is not saved with the run.
- **Low:** `evaluate_status()` defaults reference `config.TIME_THRESHOLD`/`VEHICLE_THRESHOLD`,
  deleted in `0b2ae8c` (AttributeError). Detection logs use the box centre while the alarm uses any
  pixel, and `alerts.csv` has no track id, so an alarm cannot be traced to a vehicle. PR #1 could not
  be installed (`setuptools.backends.legacy`, fixed in `479080b`) and its README command ends in a
  KeyError.
- **Hygiene** (`25e14d5`): a root file named `=4.4.0` (pip output from an unquoted
  `pip install gdown>=4.4.0`), committed `__pycache__/*.pyc` (with a local path) and `egg-info`; the
  notebook is 2.7 MB of outputs; `.vscode/tasks.json` runs `pip install` on folder open.

### Methodology

- **Results are not tied to a checkpoint.** `weights/best.pt` is untracked. The two team checkpoints
  with the same 6 classes give opposite outcomes on the positives: `50_epochs_best.pt` 1/3,
  `best_vehicle.pt` 3/3. The latter "detects" clip07/08 through frame-sized hallucinated boxes before
  the van arrives (387.8 s of false ALARM).
- **Per-clip thresholds are tuned on the test clips** (`videos.yaml`): only the positives deviate
  from 1.5/2.0 s, and replays show they change nothing. A live camera has no "clip length".
- **The per-ID dwell commit (`0b2ae8c`) changed nothing measurable**: replaying PR #1's global timer
  over all 8 clips gives identical totals. No tree on the branch has tests, ground truth or a scorer.

### Worth keeping

The clip manifest with cached downloads (our `benchmarks/team_clips/` + `tools/fetch_clips.py`
reuse its Drive ids), the per-run provenance folder (config used, metadata, snapshots), and the
"blind-spot" diagnostic from the April proposal (foreground in the zone that no detection
explains), which was never implemented.

---

## 2. Gael Marquez: `gaels_branch` (`bgs_playground/`)

**What it does.** `new_pipeline.py`: ROI (from `--zone-json` or a UTCRS rail-segmentation model) ->
pybgs SuBSENSE -> connected-component blobs (3,800-10,000 px) -> DeepSORT -> per-track count of
consecutive frames with the box bottom-centre in the ROI -> ALARM while any count >= 30 frames.
`Stabilization.py` adds an LK + partial-affine ROI stabiliser with a dead-zone. pybgs cannot be built
on Windows, so the alarm logic was tested by replaying Gael's own committed SuBSENSE/ViBe mask
videos (frame counts match the clips) through the unmodified `new_pipeline.py`.

### Errors

- **High: anything that stays 30 frames in the zone alarms, moving or not**
  (`new_pipeline.py:347-368`). A block moving at a constant 300 px/s alarms 3.21-5.09 s. With the
  real SuBSENSE masks all 5 negative clips false-alarm (38 spans: freight cars, the far gate arm, a
  car waiting in front of the zone).
- **Critical: a stopped vehicle alarms briefly or not at all.** clip07: ALARM only 4.00-4.97 s of the
  3.5 s event, then CLEAR 0.8 s before impact, when the van and train merge into one blob above
  `max_area`. clip08: missed with SuBSENSE, ViBe, KNN and MOG2. clip01: the SUV's blob (36-49k px)
  is always dropped by `max_area=10000`. There is no latch, dwell resets on one missed frame, and
  SuBSENSE absorbs stopped objects (and cannot see one present at start-up: a block present from
  frame 0 gives 0 ALARM frames).
- **High: the bottom-centre foot point misses a van parked across a thin oblique zone** (clip08: foot
  inside in 6 of 84 samples while 41-51 % of the bottom-quarter footprint overlaps the zone).
- **High: thresholds are absolute pixels and frames** (`min_area=3800`, `max_area=10000`,
  `dwell_frames=30`), never scaled to resolution or fps; only `--dwell-frames` is on the CLI.
- **Medium: coasting DeepSORT tracks (Kalman predictions, no detection) keep counting dwell**; on
  clip04 every SuBSENSE ALARM frame is held up by a coasting track (e.g. a predicted box
  [-271,125,1377,691] in a 640x360 frame).
- **Medium: the ROI stabiliser accepts any affine.** One grey frame gives a 2-inlier affine that
  moves the zone 222-269 px; it snaps back on the next trackable frame, but stays wrong as long as
  later frames fail the 12-point gate, and even the one-frame jump resets every dwell counter.
- **Medium: cannot import on Python 3.9** (`train_rail_seg.py:32,102`, `infer_seg_lines.py:137`: `Path
  | None` without `from __future__ import annotations`), and `new_pipeline.py` imports these at the
  top level. `reqs.txt` pins `torch==2.6.0+cu124` (wheels only for linux_x86_64/win_amd64) and
  `numpy==2.4.6` (Python >= 3.11); pybgs is missing; `deep-sort-realtime` needs `pkg_resources`.
- **Medium: `new_pipeline.py` tracks nothing without `--annotate-video`**, calls `cv2.imshow` every
  frame (crashes headless), and never logs the status (scoring needed a `cv2.putText` hook).
- **Medium (rail segmentation):** the corridor extraction accepts a single hot pixel and hulls all
  bands together (one false line gives 62.9 % of the frame); its final dilate inflates even a
  perfect corridor 1.8-2.6x (IoU 0.39-0.42 with the hand-drawn zones; 0.79-0.92 without it).
  Training: CVAT training crashes in DataLoader workers on macOS/Windows (module global filled only
  in `main`); random per-image split leaks frames of a video into validation; horizontal flip does
  not swap left/right classes; frozen BatchNorm statistics still update; the model is rebuilt (and
  ImageNet weights requested) for every scanned frame.
- **Low:** `stablize()` writes an unreadable video (writer opened at w, frames are 2w wide);
  `blob_analysis.detect_blobs(mask)` raises TypeError with its defaults; `bgsub.py` loops forever on a
  missing hard-coded path.
- **Hygiene:** 44 annotated `.mp4` files (676 MB; one ViBe "clip01" is clip02, the SuBSENSE clip01 is
  truncated), all produced before the alarm logic existed; `plan.md` is gitignored but tracked; the
  branch deletes `main`'s vehicle and infrastructure weights, the PDFs, the April notebook and
  `scripts/pipeline_helpers.py`, so merging it into `main` as-is would remove them.

### Worth keeping

The ROI stabiliser's 4-DOF similarity model with a dead-zone is accurate and inert on static
footage (0.0 px on five static clips, at most 2.8 px under 12 px synthetic shake). Ported into our
zone stabiliser it cut our worst error from 21 px to 2 px. The rail-segmentation corridor is a good
seed for zone calibration once the post-processing is fixed and the model is retrained on wayside
views with a per-video split.

---

## 3. Dr. Chuprov: `occupancy_monitor/`

**What it does.** Raises an alarm for anything in the zone that differs from an empty-crossing
reference and stays still, whatever its class: frame registration to a fixed anchor, a per-16-px-cell
comparison with the reference (roughness/NCC/chroma, "uncertain" for blinded cells), a 1 s-lag
stability gate, per-cell timers with a 1 s grace, selective reference update, optional YOLO for
labels, train handling, occluders and a second track-based path, and a JSONL audit log. WARNING at
3 s, ALARM at 10 s.

**Reproduced.** 32 fast tests + 1 slow test pass on Python 3.9.25 and 3.12. E2 occupancy-only
reproduces exactly (clip01 at 26.05 s, clip07/08 missed, 12 false alarms). Selective update did not
absorb a car stopped for 10 minutes, even a low-contrast one. On four synthetic cases (small debris,
a stopped car under camera jitter, an object dropped after a lighting step, debris under traffic) it
beat our frozen-MOG2 debris path.

### Errors

- **High: reference ghosts never heal** (`pipeline.py:208-219`). Anything present for more than half
  of the 8 s bootstrap becomes a permanent ALARM once it leaves; nothing can re-learn a "changed"
  cell. 3 of the 12 E2 false alarms are such ghosts (a pickup, a minivan, a burned-in caption), and
  they re-alarm after every train.
- **High: a train detection cancels every active alarm zone-wide** (`pipeline.py:162-169`). One
  `train` box covering >= 15 % of the zone, with no debounce, ends all events and resets all timers,
  so the "ALARM > TRAIN" priority can never apply: a train 74 px away from a stopped car ends its
  ALARM, then CLEAR while the car is still there.
- **High: cells behind a detected occluder are paused indefinitely** with no DEGRADED (the threshold
  is 50 % of the zone), and the pause also blocks track corroboration: a stopped car hidden behind a
  waiting truck never alarms.
- **High: a black or blank first frame leaves the monitor INITIALIZING forever** (no reference, no
  alarm, no DEGRADED); this is the default path without `--reference`.
- **Medium:** border replication in the warp turns a persistent camera shift of >= 16 px into a
  permanent edge ALARM (a validity mask removes it). Track dwell counts from the first uncorroborated
  observation, so a parked or baked-in car becomes an instant ALARM when a pedestrian walks past.
  Region dwell is the median over merged cells, so a truck stopping beside a car delays the car's
  ALARM by up to 7.5 s. A 1.5 Hz hazard light keeps a stalled car "moving" and delays the
  occupancy ALARM about 2.5x. Events are matched by cell IoU, so one narrow tracked person produced
  three events.
- **Low:** config validation only rejects unknown keys (an `alarm_seconds` below `warning_seconds` is
  silently clamped); the prepared-reference cache is keyed by version only; a long all-uncertain
  period ends in one alpha = 1 blend (unreachable on the real clips).

### Methodology

- **"12 of 13 false alarms are lowered gate arms" comes from a time-overlap tag.** The "gates down"
  spans cover 96 % of the alarm-capable negative time. A frame review of the 12 occupancy-only
  false alarms finds 5 gate arms/mechanisms, 3 bootstrap ghosts, 3 vehicles waiting just outside the
  thin band and 1 with no visible object. A gates-down reference plus a sweep mask would remove
  fewer than half.
- **False alarms are counted per region event, not per alarm episode**: 4 operator-visible episodes
  vs 12 records; 483 "false-alarm seconds" for 388 s of footage. E2's 13 -> 33 false alarms at a 3 s
  alarm are the same 8 episodes split into more regions.
- **The E2 detector rows were produced by `578083f`, not the tip.** At the tip the 3 s variant also
  catches clip08 (TTA 2.31 s), and two extra false alarms come from the pedestrian now counted by
  the footprint.
- **Scoring credits alarms by time overlap only** (an alarm already running before the event counts
  with TTA 0); E4's references are medians of the whole recording (not available online); the
  ground-truth durations come from container frame counts (497.8 s vs 475.4 s decoded).

### Adopted in our package

Frame registration before the background comparison, selective background update, grace/hold
timers, seconds-based parameters, a health/DEGRADED signal, and the ground-truth JSON format with
labelled negatives and ignore spans. Not adopted: the bootstrap reference as a production default,
and the ghost deadlock.

---

## 4. Dr. Chuprov: experiments E1-E12 and `pipelines/`

**Reproduced.** `policy_results*.txt` (E6), `candidate_score_results.txt` (E5) and
`detector_comparison.txt` (E7) rebuild byte for byte from the committed features; the E2 benchmark
and the McNemar p-values reproduce. E3, E4 (V-JEPA), E8 and E9-E12 cannot be re-run from the
repository because their inputs are not committed.

### Findings

- **The recommended `pipelines.yolo_only` does not transfer to the team's low-angle gated clips**:
  recall 1/3 (the vans' longest stationary cluster is 1.8 s), 7 false alarms in 7.9 min with the
  merged policy (12 on-track), in ALARM 62 % of normal-operation time. The causes are cars waiting
  behind the far gate or parked near the thin 20-25 px zone bands, not gate arms. The F1 0.94
  headline comes from two elevated cameras and is scored per segment. (The occupancy monitor on the
  same clips also gets 1/3 with 12 false alarms, so YOLO-only is still the better of the two there.)
- **Label bias.** The rule's author labelled scanner-selected segments with the system's evidence on
  screen. For E6 the on-screen evidence was V-JEPA's (made before the geometry rule existed); for
  E8's 74 YOLO reviews the page shows the YOLO footprint box and "stationary N s". The E4 footprint
  fix and the v1 -> v2 redesign were measured on the same segments they were designed on.
- **E8 recall is pooled over the scanners' own finds.** 30 of the 169 "known real events" are
  YOLO-only alarms, guaranteed hits; events both scanners missed never enter the denominator. On the
  139 positives found without YOLO, YOLO-only and V-JEPA+YOLO both score 128/139, so the whole
  0.93-vs-0.84 gap ("V-JEPA loses 16 real events") is measured on the 30 positives YOLO itself
  found.
- **Precision excludes 21 % of alarms** (no scorable label). If those are false: precision 0.70 and
  16.7 false alarms/h for YOLO-only (worst case 0.89 and 4.0/h with the V-JEPA gate), so "V-JEPA
  removes about 4 false alarms/h" ranges from 4.0 to 12.7.
- **Matching is time-only with any overlap**, and `merge_intervals` joins stops of different
  vehicles (clip06: one merged alarm 3.0-109.9 s holds 21 stop events); the merged policy looks
  better on false alarms/h partly through this bridging. The 169 positives are 141 distinct events
  when adjacent fragments are merged.
- **The ID-free stationarity rule fires on dense flowing traffic**: the anchor never moves, so cars
  re-hitting it within 1 s extend the "stop" (up to 290 false alarms/h on synthetic streams with
  0.6-1.2 s headways; 0 at 2-3 s headways). The real railwayvids negatives show little of it.
- **The alarm decision is offline**: it uses the median of every box of a finished cluster (an
  online replay differs on 3 of 59 alarmed clusters), and all boxes stay in memory.
- **Smaller:** the headline F1 was measured on ByteTrack boxes while the pipeline uses raw predict
  boxes (pipeline-equivalent F1 0.92 / 0.79); frame times are `k/fps` (0.84 s off on the
  variable-frame-rate clip01); "+9 recall points from dropping IDs" is confounded with two other rule
  changes; the 5 % overlap threshold has no effect on the merged policy; the CLI reuses a stale
  `detections.json` for a different video without checking; `yolo_only` ignores persons; the VLM
  conclusions are specific to one 4-bit 8B model at reduced resolution.
- **A small buffer explains most of the merged-vs-on-track gap.** On the committed E6 features, a
  0.1-0.25 box-height buffer around the zone recovers 19 of the 24 on-track stops the strict rule
  misses, with 0/110 safe segments alarmed, matching the learned gradient-boosting model. On the
  low-angle team clips any buffer adds false-alarm time (134.6 s -> 177-264 s).
- **E1's absorption times are at 10 fps**: at 30 fps both MOG2 and KNN absorb a stopped car in 2 s.

### Adopted in our package

The bottom-quarter footprint test, a stationarity clock that survives id switches and box jitter,
the "too close" notion as a WARNING tier, the 3 s alarm operating point from the task definition,
and the ground-truth format.

---

## 5. What changed in our package

Our package was reviewed the same way (about 60 confirmed findings) and changed in response. The
details are in `docs/PROGRESS.md`; the design rules in `docs/ARCHITECTURE.md`. In short:

- **Zone:** similarity motion model with a dead-zone (from `gaels_branch`), a temporal gate that
  refuses single-frame jumps, a calibration reference (`zone.reference_size` / `reference_image`),
  and the cumulative frame-to-calibration transform that the debris path needs.
- **Decision rule** (the advisor's E4/E8 ideas, as amended by an independent critic): footprint
  overlap instead of the bottom-centre point, a stationarity clock that survives id switches and box
  jitter, a "near the tracks" WARNING tier, ALARM after 3 s as in the task definition. The old rule
  is one config away: `configs/legacy_rule.yaml` reproduces the previous events exactly.
- **Debris backstop:** box-free warm-up measured in seconds, selective background update, frame
  registration, lighting compensation, a hold while a train is in the zone.
- **Evaluation:** the team's ground-truth format and scoring rule, run-aware event logs, and the
  8-clip benchmark inside the repository.
- **Operations:** exit codes that say whether the crossing was watched, `RUN_START`/`RUN_END`
  events, a DEGRADED status, config validation, the team's zone JSON files accepted as-is, webhook
  retries.

Same 8 clips, zones and ground truth, COCO `yolo11n`, default settings:

| | Before (`ae66b8f`) | After |
|---|---|---|
| Positives with an ALARM | 0 / 3 | 3 / 3 (SUV 3.5 s; van, both cameras, 3.3 s and 2.5 s, before the train) |
| Positives with a WARNING or ALARM | 0 / 3 | 3 / 3 (WARNING after 1.0-2.0 s) |
| False alarms (461 s of negatives) | 7 | 5 |
| Time in false ALARM | 154 s | 256 s |
| Time in false WARNING or ALARM | 190 s | 360 s |

Every remaining false alarm is a car waiting at the lowered gate whose box sits inside the thin
hand-drawn zone (recommendation 2 below). The earlier ALARM (3 s instead of 8 s), the hold on an
alarmed car hidden by a train and the clocks that keep running while a stopped vehicle is briefly
undetected are why the false ALARM time went up; the last of these is also what makes both
trapped-van clips ALARM, and it was designed on these same clips, so it needs confirming on new
footage. With 3 positives and 8 clips these numbers show behaviour, not rates.

---

## 6. Recommendations for the whole team

1. **One shared benchmark.** Score every pipeline on the same clips, zones and ground truth, with
   episode-level false alarms per hour, false-ALARM seconds and time-to-alarm, per camera geometry.
   `benchmarks/team_clips/` and `tools/run_benchmark.py` in our package do this for the 8 team
   clips. Add longer unedited footage per camera before quoting rates.
2. **Draw zones over the rails only**, ending at the far rail, and check them on a frame with
   traffic waiting at the gates. On every low-angle clip, the dominant false-alarm source for every
   pipeline (ours, David's, Gael's, both of the advisor's) is cars waiting at the gate whose boxes
   sit inside a thin hand-drawn band.
3. **Label blind.** A second annotator without system overlays, exhaustive labelling of a few
   recordings end to end, and recording-level confidence intervals.
4. **Repository hygiene.** CI that installs the package and runs the tests on Python 3.8, 3.9 and
   3.12; pre-commit hooks for `check-yaml`, `check-merge-conflict`, large files and `nbstripout`;
   weights and output videos on Releases or Drive with sha256 sums, never in git; merge feature
   branches by directory, not wholesale (`gaels_branch` deletes `main`'s assets).
5. **Tie results to artefacts.** Record the weights hash, class names, config and code commit in
   every run's metadata.
