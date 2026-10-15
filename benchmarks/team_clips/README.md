# Team benchmark clips

The 8 labelled clips that the advisor used to evaluate his occupancy monitor, with his
hand-drawn danger zones and ground truth. They are packaged here so that any checkout of
this repository can be run and scored the same way the team scores its own pipelines, from
files under version control rather than paths on one machine.

The videos are not in the repository (85 MB in total, and `*.mp4` is git-ignored).
`tools/fetch_clips.py` downloads them from the team's Google Drive and checks each file's
size and sha256 before keeping it.

## Provenance

| Path | Source | Changes |
| --- | --- | --- |
| `zones/clipNN.json` | `occupancy_monitor/zones/clipNN.json` on `upstream/sergei-crossing-experiments` at commit `2f90d19` (author: Sergei Chuprov; the files were added in `578083f`, "Add zone-occupancy crossing monitor") | None: byte-for-byte copies. |
| `ground_truth/clipNN.json` | `occupancy_monitor/ground_truth/clipNN.json`, same branch and commit | `duration` is the decoded duration; the team's value is kept as `container_duration`. Events, notes and ignore spans are unchanged. |
| `clips.yaml`: `id`, `drive_id` | `data/manifests/videos.yaml` on `upstream/dataset-pipeline` at commit `ca5ca4e` (author: David Sanchez) | The Drive file id is taken from each `source` link. |
| `clips.yaml`: size, sha256, fps, resolution, frame counts | Measured on the files those Drive links serve | `frames` is the number of frames that actually decode. |

The files were read with `git show <commit>:<path>` from the team's branches; nothing on
those branches was changed. Both OpenCV 4.14 and 5.0 decode the same number of frames, and
on 2026-10-14 `tools/fetch_clips.py` downloaded clip01, clip07 and clip08 from Drive with
the recorded sha256.

## The clips

| id | size | fps | decoded frames | container frames | duration (s) | content |
| --- | --- | --- | --- | --- | --- | --- |
| clip01 | 1772x872 | 31.447 | 1015 | 1027 | 32.28 | Positive: an SUV is on the crossing from 2.0 to 32.7 s, until the train arrives (stress-test clip, variable frame rate). |
| clip02 | 640x360 | 29.97 | 2189 | 2370 | 73.04 | Negative: gates down 8-66 s, freight train. |
| clip03 | 640x360 | 29.97 | 2638 | 2666 | 88.02 | Negative: gates down 8-88 s, freight train, then traffic resumes. |
| clip04 | 640x360 | 29.97 | 1798 | 1928 | 59.99 | Negative: rural crossing, gates down 6-52 s, passenger train. |
| clip05 | 640x360 | 29.97 | 2907 | 3065 | 97.00 | Negative: gates down 6-97 s, tank-car train, then traffic resumes. |
| clip06 | 640x360 | 29.97 | 3297 | 3412 | 110.01 | Negative: gates down 10-110 s; from 38 s a car waits at the gate in front of the zone. |
| clip07 | 640x360 | 29.97 | 211 | 237 | 7.04 | Positive: a van is trapped on the track 2.3-5.8 s before the train hits it; 5.8-7.9 s is ignored. |
| clip08 | 640x360 | 29.97 | 241 | 262 | 8.04 | Positive: the same van seen by a second camera, 3.3-6.2 s; 6.2-8.7 s is ignored. |

That is 7.9 minutes of video with 3 positives, two of them the same event.

## Why the decoded duration

Every container reports more frames than it decodes. For example, clip02 claims 2370 frames,
but OpenCV and ffmpeg decode 2189. The team's `duration` was `CAP_PROP_FRAME_COUNT / fps`
(79.1 s for clip02), which is 6 s longer than the video that can be played
(team-occupancy review OCC-17). That error matters in two places:

- False alarms per hour are divided by a duration that is too long.
- An alarm still open at the end of a clip is counted up to the end of the clip, which
  inflates its FA-seconds.

`duration` is therefore `frames / fps` with the decoded frame count. This also matches the
monitor's clock: it timestamps file frames with the container clock (`CAP_PROP_POS_MSEC`),
and on the constant-frame-rate clips 02-08 that clock ends at the decoded duration (clip02:
2189 frames, last frame at 73.04 s). clip01 is the exception, see the caveats.

## Usage

```
python tools/fetch_clips.py                                   # all clips into ~/.cache/gradecrossing/team_clips
python tools/run_benchmark.py --out runs/bench_defaults --device 0 --weights yolo11n.pt
python tools/run_benchmark.py --out runs/bench_alarm3 --set stuck.alarm_seconds=3 --set stuck.warning_seconds=1.5
python -m gradecrossing.evaluate --events runs/bench_defaults --truth benchmarks/team_clips/ground_truth
```

`run_benchmark.py` writes a minimal configuration for each clip. It contains the source, this
clip's zone, the weights, the device and stdout off, plus your `--set` and `--extra-yaml`
overrides. Everything else comes from the checkout's defaults, so a run measures those
defaults. The tool then scores the event logs with `gradecrossing.evaluate`'s batch mode. Set
`$GRADECROSSING_CLIPS` or pass `--cache-dir` / `--clips-dir` to keep the videos elsewhere.

## Scoring rules

Team JSON truth is scored with the team's rules (`match="any_overlap"`, tolerance 1 s),
which are reproduced exactly from `occupancy_monitor/crossing_monitor/evaluate.py`:

- Every ALARM episode that overlaps `[start - 1 s, end + 1 s]` of a positive is a hit. The
  earliest hit gives the time-to-alarm.
- An unmatched episode that overlaps an ignore span, and starts no earlier than 1 s before
  it, is not a false alarm.
- Each false alarm is tagged with the note of the first labelled negative it overlaps, for
  example "gates down".

The batch report adds FA-seconds: the time in ALARM outside the positive windows and the
ignore spans. It gives the same figure for WARNING-or-ALARM, and counts the WARNING episodes
that fall outside the positives. Note that the team's own `false_alarm_seconds` is a
different number: it adds up the lengths of the false-alarm spans.

## Caveats

- The set is small. These clips show how the monitor behaves, but they are too few to
  measure rates.
- The "gates down" negatives cover 88 % of the decoded length of clips 02-06. Every false
  alarm on them gets that tag, so the tag does not identify a cause.
- clip01 has a variable frame rate: `frame / fps` drifts by up to 0.84 s from the container
  timestamps (team-experiments F10). The monitor follows the container clock, so its event
  times match the video; a pipeline that counts `frame / fps` (as the monitor did before, or
  does with `video.fps_override` set) shifts clip01's time-to-alarm by up to that much. Frame
  indices and times are not proportional on clip01: the debris warm-up ends at 2.05 s on
  frame 89.
- The debris warm-up is 2 s (`debris.warmup_seconds`), so the debris detector is armed on
  every clip, including the 7-8 s clip07 and clip08 (from 2.04 s, frame 61).
- The zones are thin hand-drawn bands at low camera angles. Cars waiting at the far gate sit
  inside them, which explains most false alarms on clips 02-06 under any in-zone rule.
