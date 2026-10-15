# What we fixed (October 2026)

**In short:** the monitor now raises an ALARM for all 3 labelled "vehicle stuck on the tracks"
incidents in the 8 team clips; before these fixes it caught none. Dozens of reliability,
safety and usability problems were also fixed.

- **Before vs after:** "before" is the package as committed in September (commit `ae66b8f`);
  "after" is this work.
- **Where it is:** the `main` branch of the fork
  <https://github.com/Wavy-Hec/usdot-gradecrossing-monitoring>. It is not in the team repository
  yet; a pull request from the fork would bring it there.
- **What the team needs to do:** confirm five settings ([section 2](#2-decisions-the-team-needs-to-make))
  and redraw the zones over the rails only ([section 10](#10-known-limits-not-fixed-yet)).
- **How it was checked:** every change was reviewed by independent AI review agents that
  re-ran the tests and the 8 clips; the final numbers come from a fresh GPU run of the final
  code. 633 automated tests run on Python 3.8, 3.9 and 3.12.
- **New terms** (ALARM, WARNING, clock, still cluster, DEGRADED...) are explained in the
  [glossary](#12-glossary).

**How the monitor works, in one line:** YOLO11 finds vehicles and ByteTrack follows them; a
zone drawn over the tracks is kept in place as the camera shakes; a rule decides when a
vehicle is stuck in that zone; a background model catches objects YOLO has no name for.

**More detail:** [README](../README.md) (how it works), [ARCHITECTURE.md](ARCHITECTURE.md)
(how the code fits together), [TEAM_REVIEW.md](TEAM_REVIEW.md) (review of the team's
branches), [PROGRESS.md](PROGRESS.md) (project log).

---

## Contents

1. [Results at a glance](#1-results-at-a-glance)
2. [Decisions the team needs to make](#2-decisions-the-team-needs-to-make)
3. [The stuck-vehicle rule](#3-the-stuck-vehicle-rule)
4. [Keeping the zone in the right place](#4-keeping-the-zone-in-the-right-place)
5. [Objects the detector cannot name (debris)](#5-objects-the-detector-cannot-name-debris)
6. [Running reliably](#6-running-reliably)
7. [Settings and configuration](#7-settings-and-configuration)
8. [Measuring how well it works, tools and installation](#8-measuring-how-well-it-works-tools-and-installation)
9. [Bugs caught by the final review](#9-bugs-caught-by-the-final-review)
10. [Known limits (not fixed yet)](#10-known-limits-not-fixed-yet)
11. [How to check it yourself](#11-how-to-check-it-yourself)
12. [Glossary](#12-glossary)

---

## 1. Results at a glance

Both columns use the same 8 clips, zones, ground truth and detector (the COCO-trained
`yolo11n`); each version runs with its own default settings. The 3 incidents are an SUV
stopped on the crossing (clip01) and one van trapped before a train, filmed by two cameras
(clip07 and clip08).

| | Before | After |
|---|---|---|
| Incidents that got an **ALARM** | **0 of 3** | **3 of 3** |
| How fast the ALARM came (from the start of the incident) | never | SUV 3.5 s, van 3.3 s and 2.5 s |
| Incidents that got at least a **WARNING** | 0 of 3 | 3 of 3 (after 1.0-2.0 s) |
| False alarms | 7 | 5 |
| Time spent in a false ALARM | 154 s | 256 s |
| False alarms started by the debris detector | 3 | 0 |
| Automated tests | 172 | 633 |

**What this means:**

- For the van, the monitor now raises the ALARM about 0.2-0.4 s before the train hits it.
  Before, it raised nothing.
- There are fewer false alarms, but they last longer ([section 2](#2-decisions-the-team-needs-to-make)
  lists what each new default adds). All five remaining ones are vehicles (labelled car,
  truck or bus) **waiting at the gates**, two of them from before the gates came down,
  whose boxes sit inside the thin, hand-drawn zones. The monitor cannot tell those vehicles
  apart from one stopped on the rails, so the fix is to **redraw the zones over the rails
  only** ([section 10](#10-known-limits-not-fixed-yet)).
- 8 clips and 3 incidents show how the system *behaves*; they are far too few to quote
  reliable detection or false-alarm *rates*.

---

## 2. Decisions the team needs to make

These are policy choices, not bugs. Each one can be changed in the config file.

| Decision | Our default | Why | Cost on the 8 clips | How to change it |
|---|---|---|---|---|
| Alarm timing | ALARM 3 s, WARNING 1.5 s | The task says "more than 3 s" | Waiting cars become false alarms sooner | `stuck.alarm_seconds: 8` and `stuck.warning_seconds: 3` (the old values) |
| WARNING for vehicles stopped **next to** the tracks | on | Early heads-up; never an ALARM | About 90 s more WARNING time | `stuck.close_policy: off` |
| Keep an ALARM while a train hides the vehicle | on (until 3 s after the train) | A hidden vehicle is still a hazard | 58 s more false-ALARM time | `stuck.occlusion_memory_seconds: 0` |
| Keep a stopped vehicle's clock running while the detector briefly misses it | on | This is what makes both van clips ALARM | 13 s more false-ALARM time | `stuck.undetected_clock_while_live: false` |
| Gate (crossing arm) detector | off | Not validated on crossing video: on the team clips it finds no lowered gate in clip01, and turning it on (tested with the old rule) added false alarms | none | `infrastructure.enabled: true` |

**Caution:** the "keep a stopped vehicle's clock running" setting
(`stuck.undetected_clock_while_live`) was designed while looking at the van clips and
measured on the same clips, so it should be confirmed on new footage.

**Full rollback:** `configs/legacy_rule.yaml` brings back the whole old stuck-vehicle rule
(which catches 0 of 3), not just the timing.

---

## 3. The stuck-vehicle rule

This is the part that decides "is a vehicle stuck on the tracks?". Its problems caused the
0 of 3. Each tracked vehicle has **clocks**, such as how long it has been stopped in the zone;
WARNING and ALARM fire when that passes 1.5 s and 3 s.

| Problem before | What we changed |
|---|---|
| "In the zone" was decided by **one point** (the bottom-centre of the vehicle's box). A vehicle whose bottom-centre fell just outside a thin or slanted zone did not count (the SUV's was 12-64 px outside), so the SUV clearly stopped on the crossing was missed. | A vehicle is in the zone when at least 5 % of the **bottom quarter of its box** (the part touching the road) is inside the zone. It counts as having left only after 0.5 s outside. |
| "Stopped" was decided by **speed only**. Box changes (the gate arm, a passing train or a person in front of the car) looked like movement and restarted the clock: 9 times for the stopped SUV in clip01. So did an ID switch the tracker failed to match back. A van creeping 20 px counted as "moving". | A vehicle is stopped if it is slow **or** its box keeps re-appearing in the same place (a "still cluster"). A still cluster does not use the tracker's IDs, so it survives ID switches and box jitter. It starts at the vehicle's first box and is moved once, to where the vehicle actually stops. |
| ALARM only after **8 s**, although the task says "stops on the tracks for more than 3 s". WARNING after 3 s. | **ALARM after 3 s**, **WARNING after 1.5 s** (the team must confirm this, see [section 2](#2-decisions-the-team-needs-to-make)). |
| One missed detection could reset the clocks of a stopped car. Running the detector only every few frames could drop every track. | A track is removed only after the detector misses it **twice in a row and** for more than 1 s. While a stopped vehicle is briefly not detected (for example a train starts to hide it), its clock keeps running. |
| A car hidden by a passing train ended its own ALARM. | A vehicle that was part of an ALARM and is hidden by a train **stays an ALARM** while the train covers it and for 3 s after ("occluded by train"). |
| A second vehicle joining an active ALARM was not announced. | A new `ALARM_UPDATE` event says so immediately. |
| A WARNING flickered on and off for a car that never moved. | A WARNING stays on for 2 s after its cause ends, as long as the vehicle is still in or near the zone. |
| An ID switch could hand a stopped car's clock to a different car. | A new ID can take over clocks only from a stopped vehicle (if the new ID is a vehicle too) or from a person (if it is a person too). If the old ID comes back, it gets its clocks back. |
| Camera shake was read as the vehicle moving, which restarted its clock. | Speeds are measured after removing the camera's own motion. |
| A label flicker (car to "train" for a moment) dropped the vehicle from the rule. | A stopped vehicle keeps its category through label flickers shorter than 1 s, and a vehicle that is part of an ALARM keeps it as long as the ALARM lasts. |

**New:** a vehicle stopped **right next to** the tracks (not on them) gets a WARNING
("stopped near the tracks"), never an ALARM.

**Rollback:** `configs/legacy_rule.yaml` brings back the old stuck-vehicle rule exactly
(checked by tests against recorded results of the old code). Only this rule goes back; the
zone, debris and other fixes stay on.

---

## 4. Keeping the zone in the right place

The zone is the polygon drawn over the tracks. Cameras shake and drift, so the monitor moves
the zone with the picture.

| Problem before | What we changed | Result |
|---|---|---|
| On a still camera the zone **jittered** by up to about 6 px every few frames. | A smoother camera-motion model; zone shifts under 1 px are ignored. | Still cameras now give a perfectly still zone. |
| The zone could **jump 45-64 px for a single frame**, and the monitor called that healthy. | A jump has to be confirmed by 3 frames in a row (unless the whole picture clearly moved). | No false jumps on the clips; a real camera bump (clip02) is still followed at once. |
| With 12 px of camera shake, the zone was off by up to 13 px. | Same changes. | Off by at most about 3 px. |
| If the video's resolution differed from the one the zone was drawn on, the zone landed in the wrong place without any warning. | The config records the size of the frame the zone was drawn on (`zone.reference_size`) and a copy of that frame without overlays (`zone.reference_image`). | The zone is scaled to the video's size; a different aspect ratio stops the run with a clear error. |
| The built-in default zone belonged to one specific clip and covered only a sliver of other videos. | A check at start-up. | An error in the log (and a HEALTH event) when less than 90 % of the zone is inside the picture. |
| A black or blank first frame could stall the zone forever. | It waits and retries. | It starts on the first usable frame. |

---

## 5. Objects the detector cannot name (debris)

The detector only knows its classes (car, truck, person, train...). Anything else on the
crossing, like a fallen load, is left to a background model that looks for things that
appear and stay still. This part raised 3 of the 7 false alarms (in each of those spans a
waiting vehicle would have raised one anyway).

| Problem before | What we changed |
|---|---|
| The background was learned during a **300-frame (10 s) warm-up**, longer than some clips, and it learned whatever was there, including parked vehicles. When they left, the empty spot looked like an object (a "ghost" ALARM). | Warm-up is **2 s**, and pixels under a detected vehicle are never learned. Pixels never seen without a detection box over them are marked "unknown" until they are. |
| A **camera bump** made the whole background look new: debris ALARM. | Each frame is aligned to the calibration view before comparing. |
| **Slow lighting changes** became debris; on a busy crossing the background never got refreshed. | The background slowly learns only what is clearly background (never anything in the zone that could be an object), and overall brightness changes are cancelled out. |
| **Passing trains** could look like a still object. | Debris timing pauses while a train is in the zone and for 3 s after. |
| Small objects on a long zone were too small to count (the limit was 3 % of the zone). | The limit is now the smaller of 3 % of the zone and 0.2 % of the picture. |
| Half a pixel of camera jitter could hide a real object; a car driving over an object restarted its timer. | A more tolerant "is it still?" test, and an object's timer survives being covered for up to 5 s. |

**Results:** 0 debris false alarms on the 8 clips (3 before). The test objects we pasted into
real footage are still caught, except some partly hidden, low-contrast ones of a few hundred
pixels (one of which the old code caught; see [section 10](#10-known-limits-not-fixed-yet)).
On the large 1772x872 video the debris step now takes about half the time.

---

## 6. Running reliably

| Problem before | What we changed |
|---|---|
| The program **exited "OK" (code 0)** even when it had stopped watching the crossing (lost stream, missing model file, repeated errors). A service manager would think all was fine. | Clear exit codes: **0** = finished normally, **1** = the crossing stopped being watched (or never started), **2** = configuration mistake. A one-line error message (an unexpected crash still writes its traceback to the log). |
| A missing model file showed up as 5 frame errors. | Checked on the first frame: one clear error. |
| Logs from several runs in one file were mixed up when scoring. | Every run starts with a `RUN_START` event and ends with `RUN_END` (which says why it stopped). |
| Problems such as a lost camera stream were only in the log file. | `HEALTH` events go to every output (screen, log, webhook): stream lost, reconnect attempts, give-up. |
| No way to know the monitor was partly blind (zone lost, debris model still warming up). | A **DEGRADED** flag with the reasons, on the status line, on the video overlay and as HEALTH events. |
| A broken output (snapshot folder, video file, display window) crashed the monitor. | That output is switched off with a HEALTH event; monitoring continues. |
| Times came from frame counts, which drift on variable-frame-rate video (up to 0.84 s on clip01). | Times come from the timestamps stored in the video file. |
| `systemctl stop` / `docker stop` killed the program without closing anything. | The stop signal is handled: the run is closed properly and alerts still waiting are sent. |
| A dead camera blocked each reconnect attempt for about 37 s. | Each attempt times out after 10 s. |
| One failed webhook request **lost the ALARM**; a slow server held up everything behind it. | Each alert is tried up to 3 times. ALARMs are sent first. Every alert has an ID so the receiver can drop duplicates. At shutdown, alerts still waiting are sent (for up to 30 s). |
| A camera stream that changed resolution reset everything: an active ALARM was dropped without an `ALARM_END`, the zone was not rescaled, and the debris model re-learned the scene. | The new frames are resized to the original size; nothing is reset. |

---

## 7. Settings and configuration

| Problem before | What we changed |
|---|---|
| A wrong value type was accepted silently. Writing `vehicle_classes: car` instead of a list turned **every vehicle into "other" and disabled every stuck-vehicle alarm**. | Every setting is checked against its type when loaded; a mistake names the setting and stops with exit code 2. |
| Removed settings gave a confusing error. | They are listed together, with what to do instead. |
| Old site configs (written by the old calibration tool) held a copy of every default, so they silently kept old values such as the 8 s ALARM. | Such a file is now refused with one error (exit code 2) that names the removed settings and the old defaults it holds; a config that only holds old defaults loads with one warning that lists them. |
| The team's zone files could not be used directly. | `--zone clip07.json` (or the `zone.polygon_file` setting) reads the team's `{"vertices": ...}` format. |
| The calibration tool wrote every default (about 120 lines) into each site config. | It writes only what belongs to the site (video source, zone, calibration frame size and picture). |
| The calibration tool kept an old zone file setting, so a new calibration was silently ignored. | It removes the old setting and says so. |

---

## 8. Measuring how well it works, tools and installation

| Problem before | What we changed |
|---|---|
| Scoring only read a simple CSV, not the team's ground-truth files. | It reads the team's JSON format (incidents, labelled non-incidents, ignore periods) and scores like the team's own scorer (checked: same result in 224 of 224 cases). |
| A second ALARM inside the same incident counted as a false alarm. | It counts as a duplicate. |
| No shared benchmark. | `benchmarks/team_clips/` holds the 8 clips' zones and ground truth; `tools/fetch_clips.py` downloads and checks the videos; `tools/run_benchmark.py` runs and scores everything with one command. |
| The speed benchmark mostly timed the warm-up. | It times normal running and reports the warm-up separately. |
| The tuning tool could not find the results of current Ultralytics versions and always failed. | Fixed, and tested with a real Ultralytics run. |
| The checkpoint inspector could run hidden code from a crafted `.pt` file. | It only rebuilds a safe list of object types; anything else is refused. |
| Following the README installed **two OpenCV builds** at once, which breaks OpenCV. | Each install option brings exactly one OpenCV build: `.[detector]` (the monitor with YOLO, OpenCV without windows; the default), `.[display]` (the same, with OpenCV windows), `.[headless]` (no YOLO; for the tools and tests). |
| `pip install -e .` failed with the old pip on the Jetson (JetPack 5). | Works now (checked with JetPack's pip 20.0.2, but not on a Jetson itself). |

---

## 9. Bugs caught by the final review

After all the fixes were combined on this branch, three independent AI review agents checked
the whole result: one looking for missed alarms, one for false alarms and operations, one for
code quality. They found **26 issues; all were fixed** or answered. The two most serious:

1. **A car slowly covered by a train lost its ALARM.** As the train covered the car, the
   car's box shrank, its clocks restarted, and the "keep the ALARM while hidden" rule no
   longer applied. Now the rule covers every vehicle that was part of the ALARM, not just
   the ones alarming on the last frame.
2. **An "all clear" while the obstruction was still there.** An earlier version of this work
   sent an "all clear" when a camera stream changed resolution, and the debris model
   re-learned the obstruction as background. Now the frames are resized and nothing is
   reset.

Others included: a car labelled "train" for a few seconds losing its ALARM, a car stopped at
the edge of the picture being timed too late, the stop signal, alerts lost at shutdown, and
the calibration tool ignoring a new zone.

---

## 10. Known limits (not fixed yet)

- **The zones.** Every remaining false alarm comes from the hand-drawn zones covering road
  where vehicles wait at the gates. Redraw them over the rails only (`tools/calibrate_zone.py`).
- **Too little test video.** 8 short clips and 3 incidents. Hours of normal traffic, night,
  rain and more incidents are needed.
- **Debris blind spots:** an object already on the crossing when the monitor starts is
  learned as background; a partly hidden object of only a few hundred pixels can be too
  small; the train pause relies on the detector calling the train a "train".
- **Not yet run on a Jetson at all**, and never run for a long time on a live camera. On the
  Jetson, one dependency of Ultralytics (polars) only has a wheel that JetPack's old pip
  cannot install, so pip has to be upgraded first.
- **Housekeeping:** model files (27 MB) are stored in git, no license has been chosen, and
  the fine-tuned vehicle models cannot be reproduced yet.

---

## 11. How to check it yourself

```bash
# all automated tests (CPU only, no downloads): 629 pass, 4 skip without Ultralytics
python -m pytest -q

# the team-clip benchmark (downloads the 8 clips from the team's Drive)
python tools/fetch_clips.py
python tools/run_benchmark.py --out runs/bench --device 0   # 0 = first CUDA GPU; use --device cpu if you have none

# the same benchmark with the old stuck-vehicle rule, for comparison (the zone and debris
# fixes stay on, so the numbers differ somewhat from the Before column)
python tools/run_benchmark.py --out runs/bench_legacy --extra-yaml configs/legacy_rule.yaml

# score any run against the team's ground truth
python -m gradecrossing.evaluate --events runs/bench --truth benchmarks/team_clips/ground_truth
```

---

## 12. Glossary

| Term | Meaning |
|---|---|
| **Zone** | The polygon drawn over the tracks; the area that must stay clear. |
| **WARNING** | Early notice: a vehicle has been stopped in (or right next to) the zone for 1.5 s. |
| **ALARM** | A vehicle has been stopped in the zone for 3 s, or an unknown object has stayed in it for 3 s. |
| **False alarm** | An ALARM when nothing was actually stuck on the tracks. |
| **Clock** | A timer kept for each tracked vehicle, such as how long it has been stopped in the zone. |
| **Footprint** | The bottom quarter of a vehicle's box, the part that touches the road. |
| **Still cluster** | A record of a box that keeps re-appearing in the same place; it counts as stopped even if the tracker changes the vehicle's ID. |
| **Debris detector** | The background model that catches objects the AI detector has no name for. |
| **Warm-up** | The first 2 s, while the debris detector learns what the empty scene looks like. |
| **Calibration frame** | The video frame the zone was drawn on. |
| **Site config** | The config file for one camera site: its video source, zone, and calibration frame size and picture. |
| **Event** | One message the monitor sends to its outputs (screen, log, webhook), such as ALARM_START, ALARM_UPDATE, HEALTH, RUN_START or RUN_END. |
| **DEGRADED** | The monitor is running but partly blind (for example the zone is lost or the debris model is warming up). |
| **HEALTH event** | A message about the monitor itself (stream lost, warm-up done, degraded), not about the crossing. |
| **Exit code** | The number the program returns when it stops: 0 normal, 1 stopped watching, 2 configuration mistake. |
| **Webhook** | The web address alerts are sent to (for example a dispatch system). |
