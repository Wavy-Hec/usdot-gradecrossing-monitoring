# Progress and findings log

**Written:** 2026-09-16. **Updated:** 2026-10-14.
**State today:** the October changes are committed on `main` of the fork (`Wavy-Hec/usdot-gradecrossing-monitoring`), on top of the September commits (`ae66b8f`); they are not in the team repository yet. Test suite: 629 passed, 4 skipped on each of Python 3.12, 3.9.25 and 3.8.20.

This file is the one-page handoff: what the project is, what we found wrong with the prototype, what we built, how it was checked, and what is still open. Detailed evidence: [AUDIT.md](AUDIT.md) (the September audit, with the status of each finding today); module contracts: [ARCHITECTURE.md](ARCHITECTURE.md); the review of the team's branches: [TEAM_REVIEW.md](TEAM_REVIEW.md); user-facing overview: the [README](../README.md).

---

## 1. Timeline

| Date | What happened |
|---|---|
| 2026-04-03 to 04-24 | Team commits on upstream `main`: README with dataset links, infrastructure tuning script and weights, the two vehicle checkpoints. Upstream `main` has not moved since `30de329` (2026-04-24). |
| 2026-06 to 2026-10 | Team work on upstream branches only: `dataset-pipeline` (David Sanchez), `gaels_branch` (Gael Marquez), `sergei-crossing-experiments` (Dr. Chuprov: `occupancy_monitor/`, experiments E1-E12, the 8 labelled team clips with zones and ground truth). |
| 2026-09-11 | Fork cloned; audit of the prototype (`docs/AUDIT.md`); `gradecrossing/` package written, reviewed and fixed; four commits pushed to the fork (`746f177`, `d75b495`, `92375b5`, `ae66b8f`). |
| 2026-09-16 | This log first written. |
| 2026-10-14 | Upstream remote added (fetch only). Review of every team branch and of our own package, run on the 8 labelled team clips (findings re-checked by independent verifiers). Fixes implemented in scratch clones by parallel agents, each independently reviewed, merged, reviewed again as a whole (three lenses: missed alarms, false alarms/operations, code quality), fixed, then committed and pushed to the fork's `main`. `docs/TEAM_REVIEW.md` and `docs/FIXES.md` written. |

---

## 2. The project in one paragraph

UTRGV / USDOT rail grade-crossing monitor (advisor Dr. Sergei Chuprov; contributors Ruby Gonzalez, Gael Marquez, David Sanchez). A fixed camera watches a zone of interest (ZOI) on the tracks; the software must (1) find and hold that zone under camera motion, (2) detect and track vehicles, (3) alert when a vehicle is stuck in the zone ("stops on the railway tracks ... for more than 3 s"), and (4) flag debris a detector cannot name. Target hardware is an NVIDIA Jetson at the crossing, pushing events to dispatch.

---

## 3. What we found in the April prototype (September audit)

Unchanged; see [AUDIT.md](AUDIT.md) for the evidence and today's status of each finding. In short: the notebook alarmed the instant any box touched the zone (S1), had no class filter (S2), used any-pixel overlap (S3), one global timer (S4) and no alert output (S5); its MOG2 learned stationary debris in about 1 s (C1); hyperparameter tuning never ran (T1); the headline infrastructure mAP was the validation split (T2); neither vehicle checkpoint is reproducible (T3).

---

## 4. What the package is now

### `gradecrossing/` (10,444 lines; 4,544 in September)

| Module | Lines | Role |
|---|---|---|
| `monitor.py` | 1,819 | `Monitor`: per-frame pipeline, DEGRADED status, run loop (file / live, reconnect with stream timeouts, SIGTERM), RUN_START/RUN_END, guarded outputs, CLI with exit codes 0/1/2 |
| `background.py` | 1,460 | Debris backstop: box-free warm-up in seconds, selective update, registration to the calibration frame, lighting compensation, train hold, per-pixel persistence, health |
| `tracking.py` | 1,355 | Footprint zone test, speed OR ID-free still cluster, near-the-tracks tier, re-keying, hidden hazards behind trains, label hold |
| `zone.py` | 1,244 | Stabiliser: similarity model, dead-zone, temporal gate, calibration reference (size and image), `to_reference`, health |
| `evaluate.py` | 1,173 | Team ground-truth JSON, batch mode, run-aware logs, any_overlap / one_to_one matching, FA-seconds |
| `alerting.py` | 1,163 | AlertEngine (WARNING 1.5 s / ALARM 3 s, latch, ALARM_UPDATE, episodes) and sinks (webhook retries, priority, event ids) |
| `config.py` | 838 | Dataclass config, strict validation, retired keys, old-default warning, team zone files |
| others | 1,392 | `detection.py`, `autozone.py`, `geometry.py`, `visualize.py`, `types.py`, `_entry.py`, `__init__.py` |

### Tools, configs, benchmark, tests

- `tools/`: `calibrate_zone.py` (writes only site keys, saves a reference image), `fetch_clips.py` + `run_benchmark.py` (the team-clip benchmark), `benchmark.py` (per-stage cost, steady state), `inspect_checkpoint.py` (allowlist unpickler, no code runs), `tune_hyperparameters.py` (finds ultralytics 8.4 runs), `train_vehicle.py`, `export_tensorrt.py`, `record_legacy_goldens.py`.
- `configs/`: `default.yaml` (every key, pinned to the dataclasses), `stresstest.yaml` (FOX13 clip), `legacy_rule.yaml` (rollback of the vehicle rule), `team_rule.yaml` (approximation of the advisor's rule).
- `benchmarks/team_clips/`: the advisor's 8 labelled clips: zones, ground truth (decoded durations), video registry with sha256 (the videos stay on the team's Drive).
- `tests/`: 41 files; 633 tests (629 run without ultralytics; 4 need ultralytics or POSIX). Every module parses as Python 3.8, and the suite is run on 3.8, 3.9 and 3.12.
- Packaging: base install without OpenCV or torch; `[detector]` (headless OpenCV + YOLO, the default monitor install), `[display]` (GUI OpenCV), `[headless]`, `[dev]`; `setup.py` shim so `pip install -e` works with JetPack's pip 20.0.2.

---

## 5. October 2026: the team clips, the reviews and the fixes

**First real-footage run.** On the advisor's 8 labelled clips (7.9 min of decoded video, 3 positives: an SUV trapped on the crossing, and one van trapped before a train seen by two cameras), the package as committed in September caught **0 of 3** positives and raised **7 false alarms** (154 s in false ALARM). The causes were all found and fixed:

- bottom-centre point test missed vehicles across thin zones; speed-only stationarity reset on box changes and id switches; ALARM at 8 s against a 3 s task definition;
- MOG2 warm-up ghosts, a camera bump and lighting drift raised debris alarms; the stabiliser accepted 45-64 px single-frame jumps as healthy;
- operations: the CLI exited 0 when the crossing stopped being watched, wrong-typed config values loaded silently, one failed webhook POST lost an ALARM, appended event logs were mis-scored.

**Results**, same clips, zones and ground truth, COCO `yolo11n`, default settings (`tools/run_benchmark.py`):

| | Before (`ae66b8f`) | After |
|---|---|---|
| Positives with an ALARM | 0 / 3 | 3 / 3 (SUV 3.5 s; van 3.3 s and 2.5 s, before the train) |
| Positives with a WARNING or ALARM | 0 / 3 | 3 / 3 (WARNING after 1.0-2.0 s) |
| False alarms | 7 (53.0 / h) | 5 (37.9 / h) |
| Time in false ALARM | 154.1 s | 256.4 s |
| Time in false WARNING or ALARM | 190.1 s | 359.8 s |
| Debris false alarms | 3 | 0 |

Every remaining false alarm is a car waiting at the lowered gate whose box sits inside the thin hand-drawn zone; the rule cannot tell such a car from one on the rails, so the next step is redrawing the zones over the rails only. With 3 positives, these numbers show behaviour, not rates.

**Decisions the team should confirm** (each has a measured cost and a switch):

1. **ALARM 3 s / WARNING 1.5 s** (was 8 s / 3 s): the task definition says "more than 3 s". `configs/legacy_rule.yaml` restores the old vehicle rule exactly.
2. **Near-the-tracks WARNING** (`stuck.close_policy: warning`): no ALARM, but about 90 s more WARNING-or-ALARM time on the clips; `off` disables it.
3. **Hidden-hazard hold** (`stuck.occlusion_memory_seconds: 3`): an alarmed vehicle hidden by a passing train stays an ALARM. It changes no positive on the clips and holds existing false alarms through train passages (198.2 s of false ALARM without it, 256.4 s with it).
4. **Clocks of a briefly undetected, stopped vehicle keep running** (`stuck.undetected_clock_while_live: true`) and a still cluster re-anchors where the vehicle stopped: this is what makes both trapped-van clips ALARM before the train. It was designed and measured on these same clips, so it needs confirming on new footage.
5. **Gate model off** (`infrastructure.enabled: false`): never validated on crossing video; it does not see the gates on clip01.

The review of the team's own branches (errors, methodology, what is worth keeping) is in [TEAM_REVIEW.md](TEAM_REVIEW.md).

---

## 6. How the code was checked

- **September:** one review pass over every module, eight confirmed defects fixed before the commit; a verifier drove the real detector, tracker, background model and alert loop on synthetic input.
- **October:** every finding of the package and team reviews was re-checked by an independent verifier (corrected where needed, none refuted). Each fix branch was implemented in its own clone, reviewed by an independent agent that re-ran the suites and the real-footage harness, and fixed. The merged result was reviewed again by three reviewers (missed alarms, false alarms and operations, code quality and compatibility; 26 findings, all fixed or answered), then re-measured on the 8 clips with the GPU harness.
- Test suite today: 629 passed, 4 skipped on each of Python 3.12, 3.9.25 and 3.8.20 (the skipped tests need ultralytics, or POSIX for a real SIGTERM).

---

## 7. Where each stage of the plan stands

| Stage | Now | Still missing |
|---|---|---|
| 1. Dynamic zone | Drawn once per camera with a reference image; stabilised every frame (similarity model, temporal gate, registration to calibration); health and DEGRADED | Automatic zone finding; zones drawn over the rails only for the team sites |
| 2. Vehicle tracking | YOLO11 + ByteTrack, ID-free still clusters, re-keying, hidden hazards behind trains | A detector trained on US crossing traffic; ground-plane speeds |
| 3. Stuck-vehicle alerting | Footprint test, speed OR still, near tier, WARNING 1.5 s / ALARM 3 s, latch, ALARM_UPDATE | Team sign-off on the policy; more positives |
| 4. Debris | Selectively updated, registered, lighting-compensated background with a train hold | Real debris footage; a stored verified-empty reference |
| Deployment | CLI with exit codes, RUN events, SIGTERM, stream timeouts, webhook delivery contract, packaging for JetPack's pip | A run on a Jetson; a soak test on a live stream |
| Evaluation | The team-clip benchmark in the repo, team scoring rules, batch reports | Hours of normal traffic, night and rain; blind labels |

---

## 8. Open items

- **Zones:** the advisor's thin hand-drawn zones include road where cars wait at the gates; every remaining false alarm comes from that.
- **Policy decisions** listed in section 5 need the team's confirmation.
- **Debris size bar** (`debris.min_object_frame_fraction`, 0.2 % of the frame): partly visible dark objects of a few hundred pixels are missed; lowering it brings a clip03 train fragment close to the trigger.
- **An undetected obstruction present at start-up** is learned during the warm-up (needs a stored verified-empty reference).
- **The train hold relies on the `train` label;** trains labelled bus or car are excluded from debris but do not hold it.
- **Jetson:** not run on hardware; ultralytics pulls polars, whose only cp38 aarch64 wheel needs a newer pip than JetPack's 20.0.2; Ultralytics' Jetson guide pins numpy 1.23.5 against our numpy >= 1.24.
- **Repository hygiene:** 27 MB of model binaries and the notebook's embedded outputs are still in git; weights belong on Releases. No license is chosen (TODO in `pyproject.toml`).
- **Vehicle weights remain unreproduced** until `tools/train_vehicle.py` is rerun with a committed `data.yaml`.

---

## 9. Suggested next steps

1. **Open a pull request** from the fork's `main` to the team repository, pointing reviewers at `docs/FIXES.md` and `docs/TEAM_REVIEW.md`.
2. **Redraw the zones over the rails only** for the team sites (`tools/calibrate_zone.py` with a reference image) and re-run `tools/run_benchmark.py`.
3. **Confirm the policy decisions** in section 5 with the advisor.
4. **Grow the evaluation set:** long unedited footage per camera (normal traffic, night, rain), labelled without system overlays.
5. **Jetson soak test** on a live stream: FPS (`tools/benchmark.py`), reconnects, HEALTH volume, SIGTERM.
6. Validate the gate model on crossing video before enabling it; rerun vehicle training reproducibly.

---

## 10. Links

- Fork (`main`): <https://github.com/Wavy-Hec/usdot-gradecrossing-monitoring>
- Team repository and Releases (datasets, test clips): <https://github.com/ruby-gonzalez/usdot-gradecrossing-monotoring>
- Railroad v3 dataset: <https://universe.roboflow.com/obstacle-detection/railroad-crossings-object-detection>
- Team write-up: `docs/MOG2_YOLO_StressTest_Findings.pdf`; code reference: `docs/Rail_Grade_Crossing_Code_Reference.pdf`
