"""Per-object dwell / speed / stuck bookkeeping for the grade-crossing monitor.

This module turns a stream of per-frame detections into *per-object* safety
measurements. The design rules it implements (see docs/ARCHITECTURE.md):

* **Per-object, not per-frame.** Every measurement hangs off a track id. A detector
  that provides ids (ultralytics + ByteTrack) is used as-is; detections without ids
  are matched to existing tracks by a small greedy IoU matcher so the module works
  with any detector.
* **A vehicle is where its ground footprint is.** With ``stuck.zone_test="footprint"``
  (the default) a vehicle is in the zone when the bottom quarter of its box (full width:
  the part of the box nearest the ground) overlaps the zone by at least 5 %, and "near"
  (too close) when it is not in the zone but within 0.75 box heights of it - the team's
  rule (EXPERIMENTS.md E4/E6). A single bottom-center point misses vehicles seen
  obliquely across thin zones (clip08's trapped van, clip01's SUV with the advisor's
  zone). A box whose *upper* part crosses the zone (a car on the adjacent road) is still
  not in it, and a vehicle overlapping between 0 and 5 % is near, never in. Persons,
  trains and unmapped classes keep the bottom-center point test (``zone_test=
  "bottom_center"`` restores it for vehicles too). A vehicle stays in the zone until its
  real detections have been out for ``zone_exit_grace_seconds``, so a box bottom
  jittering by one pixel on the zone edge does not restart its clocks.
* **Stuck = in zone AND stopped, continuously, for T seconds.** "Stopped" is the speed
  rule OR the ID-free rule (``stuck.stationarity``):

  - *speed*: a least-squares linear fit of x(t), y(t) over a short window of
    bottom-center samples, far more robust to box jitter than frame-to-frame differences.
    Until enough samples exist there is *no* speed estimate, and without one an object
    can never be stationary - so a brand-new track can never trigger an instant alarm.
  - *still*: the box keeps re-appearing with IoU >= 0.5 to the first box seen at that
    spot (:class:`StationaryClusters`, a port of the team's pipelines/common/geometry.py),
    whatever ids the tracker gives it. It survives id switches and the box changes that
    restart the speed clock (clip01's SUV: one cluster over 23 s, nine speed restarts).
    The anchor moves once, to the owner's box when its speed first drops below
    ``speed_threshold`` (the start is kept): a vehicle braking into the zone opens the
    cluster while it still rolls (clip07's van: anchored at 93 px/s, 45 px before its stop,
    IoU 0.58 with the stopped box), and a creep of 20 px after the stop then left the
    cluster and restarted the clock.
    Moving traffic can carry still time: at most (w/3)/v for a vehicle anchoring its own
    cluster (up to 2w/3 of travel if its speed estimate dips below ``speed_threshold`` in
    it, which moves the anchor), (2w/3)/v as a guest passing a stopped vehicle, more for
    long slow vehicles (a 400 px bus at 40 px/s: 3.3 s). A box cut by the image border
    (``still_border_margin``) with a train box beside it within the last 2 s restarts its
    cluster's still clock: the visible end of a train passing the edge of the picture,
    labelled bus or truck, keeps one box while the train moves. A vehicle stopped at the
    edge with no train beside it is still like any other.

  A cluster belongs to the track that created it. It changes owner only when the owner's
  id is absent from the frame, and only to a track born inside it (a re-identification
  on top of the stop), so a vehicle that drives into a spot never inherits the previous
  vehicle's stop. The cluster also carries its owner's in-zone (and on-or-near) streak,
  so an id switch the re-keying cannot match keeps both the stop and the in-zone clock.
* **Timestamp-based counters.** ``dwell_seconds``, ``stationary_seconds``,
  ``still_seconds`` and ``stuck_seconds`` are ``now - since`` differences, not frame
  counts, so thresholds in seconds mean the same thing at 10 fps on a Jetson and 30 fps on
  a workstation. A track is dropped only when ``lost_track_missed_runs`` *detector runs* in
  a row have missed it and its last detection is older than ``lost_track_grace_seconds``:
  frames on which the detector is skipped never age a track out, and with sparse detection
  one missed run does not either, whatever ``detect_every_n_frames`` is. Until then the
  dwell and stuck clocks of a live in-zone vehicle the detector missed keep running from
  their start (``undetected_clock_while_live``) instead of freezing until it is seen again.
* **An id swap in place does not restart the clocks.** ByteTrack gives a vehicle a new id
  after an occlusion. A fresh id means fresh counters and no speed estimate, so a vehicle
  that has not moved would drop back to CLEAR and need the whole threshold again. Before
  a brand-new detector id creates a track, it is therefore matched by IoU against the live
  tracks no detection claimed this frame, and the existing track is re-keyed to the new id
  with every counter intact (see ``_rekey_unseen_id``). A vehicle box can only adopt a
  stopped vehicle track, a person box only a person track; if the old id comes back while
  the take-over is recent (a truck passing in front of a stopped car was first detected on
  top of it) the clocks go back to the old id.
* **A vehicle hidden by a train is still a hazard.** The re-keying only bridges
  occlusions shorter than the grace period; a train passage takes far longer. An ALARMED
  vehicle whose track is lost in or near the zone while a train box overlaps it is kept as
  a hidden hazard (``ObjectState.occluded``; the alert engine holds the ALARM) while the
  train covers it and for ``occlusion_memory_seconds`` after; a re-detection at IoU >= 0.5
  restores its clocks. Only alarmed vehicles get this memory: remembering every vehicle
  would keep the clocks of cars waiting at the gates running and add false alarms.
  "Alarmed" means in the ALARM episode in progress, latched included
  (``AlertEngine.alarm_episode_track_ids``): a train covering a car gradually cuts its box
  short first, which restarts its clocks before it is lost. An alarmed vehicle in or near
  the zone also keeps its vehicle category whatever the detector calls it (a car box
  merging with the arriving train is labelled 'train' for seconds).
* **Camera motion is not vehicle motion.** When the zone provides
  ``ZoneState.to_reference`` (current-frame pixels -> reference-frame pixels), speed
  samples and still-cluster boxes are mapped into the reference frame, so a camera bump
  neither resets a stopped vehicle's speed clock nor breaks its cluster. The zone
  relation needs nothing: the footprint is tested against the warped zone of the frame.

Nothing here imports torch or ultralytics; the whole module is exercised with synthetic
detections in ``tests/test_tracking.py``, ``tests/test_footprint.py`` and ``tests/test_still.py``.
"""
from __future__ import annotations

import copy
import dataclasses
import logging
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from .config import StuckConfig
from .detection import CATEGORY_OTHER, CATEGORY_PERSON, CATEGORY_TRAIN, CATEGORY_VEHICLE  # classify() results
from .geometry import (
    bottom_center,
    box_iou,
    box_iou_matrix,
    footprint_relation,
    point_in_polygon,
    transform_box,
    transform_points,
)
from .types import Detection, ObjectState, ZoneState

LOGGER = logging.getLogger(__name__)

# Categories that count as "something is on the crossing" for zone_occupied. That flag, and
# zone_empty_since derived from it, feed two things: the monitor's scene_static, which gates
# the zone keyframe re-anchoring (a re-anchor while something stands in the zone would take
# its features for background), and the debris illumination guard, which may re-warm the
# background model of a whole-frame zone only while the tracker sees the zone empty.
# CATEGORY_OTHER counts too. An object of a class the config does not map (COCO
# "bench"/"boat"/"suitcase", or a class of a fine-tuned model missing from vehicle_classes/
# person_classes/train_classes) is invisible to the stuck logic but stays visible to the
# debris backstop (only vehicle, person and train boxes are cut out of its mask); calling
# the zone empty while it stands there would let the keyframe re-anchor on it and a re-warm
# forget it.
_OCCUPYING_CATEGORIES = (CATEGORY_VEHICLE, CATEGORY_PERSON, CATEGORY_TRAIN, CATEGORY_OTHER)

_ZONE_TESTS = ("footprint", "bottom_center")
_STATIONARITY = ("speed", "still", "speed_or_still")
_REKEY_POLICIES = ("strict", "any")

# Timestamps are compared against second thresholds with this slack, so k/fps arithmetic
# (6/5 - 1/5 = 1.0000000000000002) never decides whether a track survives or a hold ends.
_EPS = 1e-6
# For re-keying and label stickiness a track counts as stopped when the speed rule says
# so, or when its box has stayed in one still cluster this long: a 60 px car moving at
# >= 70 px/s never stays 0.3 s in one cluster ((w/3)/v), while a stopped vehicle whose
# speed estimate is still noisy (box jitter, a fresh speed window) already counts.
_STOPPED_MIN_STILL_SECONDS = 0.3
# IoU between a new box and a hidden hazard's last box that restores the hazard's track.
_HAZARD_REACQUIRE_IOU = 0.5
# A box cut by the image border restarts its still cluster's clock only when a train box was
# seen beside it (overlapping it grown by its own height on every side) this recently. On
# clip06 the 'bus' end of the passing train at x2 = 640 sat beside the train's own boxes in
# the same band for its whole 10 s; a vehicle stopped at the edge of a full-width zone has no
# train beside it and keeps its still clock (work/rev-missed-alarms: the clip01 SUV moved to
# the right edge alarmed at 12.35 s instead of 5.42 s when every border box restarted it).
_BORDER_TRAIN_SECONDS = 2.0


class PredictedDetection(Detection):
    """A ``Detection`` synthesised by :meth:`StuckVehicleTracker.predict_boxes`.

    On frames where the detector is skipped, the monitor feeds the tracker's own
    velocity-propagated boxes back into :meth:`StuckVehicleTracker.update` so the
    per-object counters keep advancing. Those boxes are *not evidence* that the object
    is still there, so the tracker must be able to tell them apart from real detector
    output: a predicted box never refreshes ``last_seen``, never adds a position sample
    to the speed window, never feeds a still cluster and never changes zone membership.
    Without this distinction a track that the detector has lost would keep re-observing
    its own predictions and never be dropped. The class adds no fields; it only marks
    provenance.
    """


@dataclass
class StillCluster:
    """One ID-free "something stands here" cluster (see :class:`StationaryClusters`)."""

    anchor: np.ndarray  # first box seen at this spot; moved once, where the owner stopped (``settled``)
    start: float  # timestamp of that first box
    last: float  # timestamp of the latest box that matched the anchor
    count: int = 1
    owner: Optional[int] = None  # track id whose stop this is (carries the age); others are guests
    zone_since: Optional[float] = None  # the owner's in-zone streak start, kept across owner changes
    near_since: Optional[float] = None  # the owner's on-or-near streak start, likewise
    live: bool = True  # still among the active clusters (False once expired or reset)
    # Last time it took a box cut by the image border with a train box beside it: its still
    # clock counts from there (see _BORDER_TRAIN_SECONDS).
    cut_t: Optional[float] = None
    # True once the tracker moved the anchor to the owner's box where its speed first dropped
    # below speed_threshold (StuckVehicleTracker._settle_anchor); it never moves again.
    settled: bool = False


class StationaryClusters:
    """Track-id-free stationarity, online (port of the team's pipelines/common/geometry.py).

    A box that keeps re-appearing where it was first seen - IoU >= ``min_iou`` with the
    cluster's *anchor* (its first box), with gaps no longer than ``max_gap_seconds`` - is one
    stationary thing, whatever ids a tracker gives it. ``last - start`` is its stationary
    time. On 249 labelled stops at 2 sites this beat tracker-id stop timers by 9 recall
    points (EXPERIMENTS.md E6) because it survives id switches, box jitter and short
    dropouts. Anchoring to the *first* box (not the latest) is what stops a slowly creeping
    vehicle from being "stationary" forever: once it has moved about a third of its width
    the IoU falls below 0.5 and a new cluster (age 0) starts. The tracker moves an anchor at
    most once (``StillCluster.settled``), to where the owner stopped, so that still holds.

    Only real detections may be fed in (a predicted box is not evidence), and only boxes of
    the vehicle category (persons and trains are handled elsewhere).

    Cost: one vectorised IoU of the frame's boxes against every live anchor per frame. A moving
    vehicle opens a new cluster every (w/3)/v seconds and each lives ``max_gap_seconds``, so the
    number of live clusters grows with traffic; the matrix keeps dense traffic cheap.
    """

    def __init__(self, min_iou: float = 0.5, max_gap_seconds: float = 1.0) -> None:
        self.min_iou = float(min_iou)
        self.max_gap = float(max_gap_seconds)
        self.active: List[StillCluster] = []

    def reset(self) -> None:
        for c in self.active:
            c.live = False
        self.active = []

    def alive(self, cluster: Optional[StillCluster], t: float) -> bool:
        """True while ``cluster`` has been matched within ``max_gap_seconds`` of ``t``."""
        return cluster is not None and cluster.live and (float(t) - cluster.last) <= self.max_gap + _EPS

    def revive(self, cluster: StillCluster, t: float) -> None:
        """Put a cluster that expired back in play at ``t`` (a hidden hazard was re-detected)."""
        cluster.last = float(t)
        if not cluster.live:
            cluster.live = True
            self.active.append(cluster)

    def update(self, timestamp: float, boxes: Sequence[np.ndarray]) -> List[StillCluster]:
        """Expire clusters unseen for more than ``max_gap_seconds``, then give every box the
        live cluster whose anchor it overlaps best (IoU >= ``min_iou``; on a tie the newest
        cluster) or a new one. Boxes are taken in order, so a cluster a box opens can take a
        later box of the same frame. Returns the cluster of each box, in input order."""
        t = float(timestamp)
        kept: List[StillCluster] = []
        for c in self.active:
            if t - c.last <= self.max_gap + _EPS:
                kept.append(c)
            else:
                c.live = False
        self.active = kept
        out: List[StillCluster] = []
        if len(boxes) == 0:
            return out
        bx = np.stack([np.asarray(b, dtype=np.float32).reshape(4) for b in boxes])
        n_old = len(self.active)
        old = box_iou_matrix(bx, np.stack([c.anchor for c in self.active])) if n_old else np.empty((len(bx), 0))
        # A cluster opened by an earlier box of this frame has that box as its anchor.
        pair = box_iou_matrix(bx, bx) if len(bx) > 1 else None
        opened: List[int] = []  # indices of this frame's boxes that opened a cluster, in order
        for j in range(len(bx)):
            v = old[j]
            if opened:
                v = np.concatenate([v, pair[j, opened]])  # type: ignore[index]
            best = None
            if v.size:
                k = v.size - 1 - int(np.argmax(v[::-1]))  # the last of the best (ties -> newest)
                if v[k] >= self.min_iou:
                    best = self.active[k]
            if best is None:
                best = StillCluster(anchor=bx[j].copy(), start=t, last=t)
                self.active.append(best)
                opened.append(j)
            else:
                best.last = t
                best.count += 1
            out.append(best)
        return out


@dataclass
class _Track:
    """Internal per-id record. ``state`` is the public ``ObjectState`` we hand out."""

    state: ObjectState
    category: str
    cls_id: int
    conf: float
    seen_count: int  # number of *real* detections observed for this id
    last_detected_t: float  # timestamp of the last real detection
    last_detected_box: np.ndarray  # box of the last real detection (prediction anchor)
    missed_runs: int = 0  # detector runs in a row that did not detect it (since the last real detection)
    history: Deque[Tuple[float, float, float]] = field(default_factory=deque)  # (t, x, y) speed samples
    velocity_px: Tuple[float, float] = (0.0, 0.0)  # least-squares velocity (px/s)
    has_speed: bool = False  # True once a speed estimate exists
    stationary: bool = False  # has_speed and speed < threshold, at the last real detection
    stationary_since: Optional[float] = None
    stuck_since: Optional[float] = None
    stopped_since: Optional[float] = None  # earliest "continuously stopped since" of the enabled rules
    last_ref_box: Optional[np.ndarray] = None  # last real box in reference-frame pixels
    cluster: Optional[StillCluster] = None  # still cluster of the last real detection (vehicles)
    cluster_joined: float = 0.0  # when this track's detections started matching ``cluster``
    birth_cluster: Optional[StillCluster] = None  # cluster of the track's first real detection
    zone_out_since: Optional[float] = None  # first out-of-zone real detection while held in the zone
    onnear: bool = False  # debounced "in the zone or too close" state
    onnear_out_since: Optional[float] = None
    near_since: Optional[float] = None  # start of the current on-or-near streak
    vehicle_since: Optional[float] = None  # since when the track is continuously a vehicle
    label_run_since: Optional[float] = None  # first detection of the current run of other labels


@dataclass
class _Hidden:
    """An alarmed vehicle track lost under a train box (see ``occlusion_memory_seconds``)."""

    track: _Track
    box_ref: np.ndarray  # its last real box, reference-frame pixels
    last_train_t: float  # last time a train box overlapped it


@dataclass
class _Rekeyed:
    """Clocks a new id took over, kept so they can go back if the old id reappears."""

    new_id: int
    t: float
    snapshot: _Track
    runs: int = 0  # detector runs since the take-over (the snapshot expires like a lost track)


class StuckVehicleTracker:
    """Maintain dwell / speed / stuck measurements for every tracked object.

    Parameters
    ----------
    cfg:
        ``StuckConfig`` with the zone test, the stationarity rule, the speed threshold and
        window, the track bookkeeping limits and the optional ground-plane homography.
        Unknown enum values and out-of-range numbers raise ``ValueError`` here.
    classify:
        Callable mapping a detector class name to ``"vehicle" | "person" | "train" |
        "other"`` (normally ``VehicleDetector.classify``). The tracker never looks at
        class names itself so that class lists live in exactly one place (the config).

    Typical use per frame::

        ran = detector_frame
        dets = detector.detect(frame) if ran else tracker.predict_boxes(t)
        objects = tracker.update(dets, zone_state, t, detector_ran=ran,
                                 alarmed_ids=engine.alarm_episode_track_ids)

    ``update`` returns the state of every *live* object (not dropped yet) plus every hidden
    hazard (``occluded`` True), not only the ones seen this frame, so a detector dropout
    does not make a stuck vehicle disappear from the alert logic.
    """

    def __init__(self, cfg: StuckConfig, classify: Callable[[str], str]):
        self.cfg = cfg
        self._classify = classify
        self._class_cache: Dict[str, str] = {}
        # Need at least two distinct timestamps for a slope; guard silly configs.
        self._min_history = max(2, int(cfg.min_history_for_speed))
        self._ground_H: Optional[np.ndarray] = None
        if cfg.ground_plane_homography is not None:
            H = np.asarray(cfg.ground_plane_homography, dtype=np.float64)
            if H.shape != (3, 3) or not np.all(np.isfinite(H)):
                raise ValueError("ground_plane_homography must be a finite 3x3 matrix")
            self._ground_H = H
        self._speed_units = "m/s" if self._ground_H is not None else "px/s"
        self._validate(cfg)
        self._clusters = StationaryClusters(cfg.still_min_iou, cfg.still_max_gap_seconds)
        # State initialised in reset() so that reset() is the single source of truth.
        self._tracks: Dict[int, _Track] = {}
        self._hidden: Dict[int, _Hidden] = {}
        self._rekeyed: Dict[int, _Rekeyed] = {}
        self._recent_trains: Deque[Tuple[float, np.ndarray]] = deque()  # (t, reference box) of real train boxes
        self._next_fallback_id = 1
        self._next_uid = 1
        self._zone_empty_since: Optional[float] = None
        self._last_predicted: Dict[int, np.ndarray] = {}
        self._last_predicted_t: Optional[float] = None
        self._ref_H: Optional[np.ndarray] = None
        self._last_ref_H: Optional[np.ndarray] = None
        self._last_run_t: Optional[float] = None
        self._gap_reported = False
        self._pending_gap: Optional[float] = None
        self.reset()

    @staticmethod
    def _validate(cfg: StuckConfig) -> None:
        def check(ok: bool, msg: str) -> None:
            if not ok:
                raise ValueError(msg)

        check(cfg.zone_test in _ZONE_TESTS, f"stuck.zone_test must be one of {_ZONE_TESTS}, got {cfg.zone_test!r}")
        check(cfg.stationarity in _STATIONARITY,
              f"stuck.stationarity must be one of {_STATIONARITY}, got {cfg.stationarity!r}")
        check(cfg.rekey_policy in _REKEY_POLICIES,
              f"stuck.rekey_policy must be one of {_REKEY_POLICIES}, got {cfg.rekey_policy!r}")
        check(0.0 < float(cfg.footprint_fraction) <= 1.0, "stuck.footprint_fraction must be in (0, 1]")
        check(0.0 <= float(cfg.footprint_min_overlap) <= 1.0, "stuck.footprint_min_overlap must be in [0, 1]")
        check(float(cfg.close_box_heights) >= 0.0, "stuck.close_box_heights must be >= 0")
        check(float(cfg.zone_exit_grace_seconds) >= 0.0, "stuck.zone_exit_grace_seconds must be >= 0")
        check(0.0 < float(cfg.still_min_iou) <= 1.0, "stuck.still_min_iou must be in (0, 1]")
        check(float(cfg.still_max_gap_seconds) > 0.0, "stuck.still_max_gap_seconds must be > 0")
        check(0.0 <= float(cfg.still_border_margin) < 0.5, "stuck.still_border_margin must be in [0, 0.5)")
        check(float(cfg.label_hold_seconds) >= 0.0, "stuck.label_hold_seconds must be >= 0")
        check(float(cfg.occlusion_memory_seconds) >= 0.0, "stuck.occlusion_memory_seconds must be >= 0")
        runs = cfg.lost_track_missed_runs
        check(not isinstance(runs, bool) and float(runs) == int(runs) and int(runs) >= 1,
              f"stuck.lost_track_missed_runs must be an integer >= 1, got {runs!r}")

    # ------------------------------------------------------------------ public API

    def reset(self) -> None:
        """Forget every track (new video / detector reset). Ids restart from 1."""
        self._tracks = {}
        self._hidden = {}
        self._rekeyed = {}
        self._recent_trains = deque()
        self._next_fallback_id = 1
        self._next_uid = 1
        self._zone_empty_since = None
        self._last_predicted = {}
        self._last_predicted_t = None
        self._ref_H = None
        self._last_ref_H = None
        self._last_run_t = None
        self._gap_reported = False
        self._pending_gap = None
        self._clusters.reset()

    def update(self, detections: List[Detection], zone: ZoneState, timestamp: float,
               detector_ran: bool = True, alarmed_ids: Iterable[int] = ()) -> List[ObjectState]:
        """Ingest one frame of detections at ``timestamp`` and return all live objects.

        ``detector_ran`` says whether ``detections`` are this frame's detector output (True)
        or the tracker's own :meth:`predict_boxes` on a skipped frame (False). Only detector
        runs can drop a track. ``alarmed_ids`` are the track ids of the alert engine's ALARM
        episode in progress, latched included (``AlertEngine.alarm_episode_track_ids``); only
        those can become hidden hazards, and those in or near the zone keep their vehicle
        category whatever the detector calls them (``label_hold_seconds`` > 0).

        Steps:

        1. An old id that a new id took over reappears while the take-over is still recent
           (as long as a lost track would live): the clocks go back to it. A new id (or an
           id-less box) on top of a hidden hazard restores the hazard's track.
        2. Every real vehicle box joins (or starts) a still cluster, before any id is
           processed, so an id switch cannot break it. A box cut by the image border
           (``still_border_margin``) with a train box beside it restarts its cluster's still
           clock.
        3. Detections that carry a ``track_id`` update (or create) the track with that id.
           A *new* id whose box still overlaps a live track that no detection claimed this
           frame re-keys that track instead of starting a fresh one (``rekey_policy``).
        4. Detections without an id are matched greedily by IoU against the remaining
           live tracks (``fallback_iou_threshold``); unmatched ones get fresh ids.
        5. On detector runs, a track is dropped once ``lost_track_missed_runs`` runs in a
           row have missed it and its last detection is older than
           ``lost_track_grace_seconds`` (or hidden instead, when alarmed, last seen in or near
           the zone and covered by a train box). Hidden hazards expire
           ``occlusion_memory_seconds`` after the last train box over them. A live in-zone
           vehicle this run missed keeps its dwell and stuck clocks running
           (``undetected_clock_while_live``).
        6. ``zone_empty_since`` is refreshed from the surviving objects.

        Timestamps must be seconds and should be non-decreasing; counters are clamped
        at zero if a timestamp ever steps backwards.

        Field semantics worth knowing: ``ObjectState.last_seen`` and ``history`` reflect
        *real* detections only, while ``last_box`` follows predicted boxes too (so an
        overlay keeps moving on skipped frames). Fallback ids are allocated above every
        detector id seen so far; a stream that mixes detector ids and id-less boxes can
        still collide if the detector later issues an id we already handed out.
        """
        t = float(timestamp)
        self._ref_H = self._reference_transform(zone)
        mask = getattr(zone, "mask", None)
        frame_hw = None if mask is None else (int(mask.shape[0]), int(mask.shape[1]))
        alarmed = {int(i) for i in alarmed_ids}
        updated: Set[int] = set()

        with_id: List[Detection] = []
        without_id: List[Detection] = []
        for det in detections:
            (with_id if det.track_id is not None else without_id).append(det)
        real = [d for d in detections if not self._is_predicted(d, t)]

        # Ids the detector uses this frame: a track one of them owns must never be stolen
        # by the re-keying below, even if its detection has not been processed yet.
        frame_ids: Set[int] = {int(d.track_id) for d in with_id}  # type: ignore[arg-type]
        if self._rekeyed:
            self._restore_rekeyed({int(d.track_id) for d in real if d.track_id is not None}, t)
        if self._hidden:
            self._reacquire_hidden(real, t)
        clusters = self._assign_clusters(real, t, frame_hw)

        for det in with_id:
            tid = int(det.track_id)  # type: ignore[arg-type]
            if tid in updated:
                continue  # a detector should never emit an id twice per frame; keep the first
            if tid not in self._tracks and not self._is_predicted(det, t):
                self._rekey_unseen_id(tid, det, frame_ids, t)
            self._observe(tid, det, zone, t, clusters.get(id(det)), tid in alarmed)
            updated.add(tid)

        if without_id:
            for det, tid in zip(without_id, self._match_fallback(without_id, updated, t)):
                self._observe(tid, det, zone, t, clusters.get(id(det)), tid in alarmed)
                updated.add(tid)

        # Drop tracks the detector has genuinely lost. Predicted boxes never refresh
        # last_detected_t, so a ghost cannot keep itself alive; and only detector runs
        # count as misses, so sparse detection (detect_every_n_frames) never ages a track
        # out between runs, and one missed run does not drop it either.
        if detector_ran:
            self._count_missed_run(t)
            self._drop_lost(t, alarmed)
            if self.cfg.undetected_clock_while_live:
                self._advance_undetected(t, updated)
            self._note_detector_run(t)
        if self._hidden:
            self._update_hidden(t)

        if self.zone_occupied:
            self._zone_empty_since = None
        elif self._zone_empty_since is None:
            self._zone_empty_since = t
        return self.objects()

    def predict_boxes(self, timestamp: float) -> List[Detection]:
        """Velocity-propagated boxes for every live object, for frames without a detector run.

        Each box is the last *detected* box of the object shifted by its least-squares
        velocity times the elapsed time since that detection (zero shift when no velocity
        estimate exists yet). Ids, class and confidence are preserved. The returned objects
        are :class:`PredictedDetection` so that feeding them back into :meth:`update`
        advances the clocks without counting as new evidence. Hidden hazards get no box:
        nothing is visible of them.
        """
        t = float(timestamp)
        out: List[Detection] = []
        self._last_predicted = {}
        self._last_predicted_t = t
        for tid, tr in self._tracks.items():
            box = self._predicted_box(tr, t)
            out.append(PredictedDetection(xyxy=box, conf=tr.conf, cls_id=tr.cls_id,
                                          cls_name=tr.state.cls_name, track_id=tid))
            self._last_predicted[tid] = box.copy()
        return out

    @property
    def zone_occupied(self) -> bool:
        """True when any live tracked object, of any class, or any hidden hazard is in the zone.

        The monitor's ``scene_static`` is its negation (the zone keyframe re-anchors only
        while the zone is empty), and :attr:`zone_empty_since` follows it (the debris
        illumination guard re-warms the background of a whole-frame zone only while the zone
        is empty). Unmapped ("other") classes count too: such an object never feeds the stuck
        logic but stays visible to the debris backstop, and a re-anchor or a re-warm while it
        stands in the zone would undermine both. "Near" (too close) vehicles do not count. The
        cost is that a permanent static false positive inside the zone also blocks the
        re-anchoring - visible as an object in the overlay - whatever category it is mapped
        to; listing its class in ``detector.ignore_classes`` drops it before tracking, which
        is the fix.
        """
        return (any(tr.state.in_zone and tr.category in _OCCUPYING_CATEGORIES for tr in self._tracks.values())
                or any(h.track.state.in_zone for h in self._hidden.values()))

    @property
    def zone_empty_since(self) -> Optional[float]:
        """Timestamp at which the zone last became empty, or None while occupied.

        None also before the first ``update`` call (nothing is known yet). The monitor
        derives ``zone_empty_seconds = t - zone_empty_since`` for the debris detector.
        """
        return self._zone_empty_since

    def objects(self) -> List[ObjectState]:
        """Live objects in the order their tracks were created, then the hidden hazards."""
        return [tr.state for tr in self._tracks.values()] + [h.track.state for h in self._hidden.values()]

    def pop_detection_gap(self) -> Optional[float]:
        """Seconds between two detector runs that exceeded ``still_max_gap_seconds``, once per run.

        ID-free stationarity needs a real detection at least every ``still_max_gap_seconds``;
        slower than that every cluster expires between detector runs and the still clock
        never grows (only the speed rule is left). The first time it happens (until
        :meth:`reset`) the gap is logged as a warning and returned here once, so the monitor
        can emit a HEALTH event; None otherwise.
        """
        gap, self._pending_gap = self._pending_gap, None
        return gap

    # ------------------------------------------------------------------ internals

    def _category(self, cls_name: str) -> str:
        """Cached lookup through the injected ``classify`` callable."""
        cat = self._class_cache.get(cls_name)
        if cat is None:
            cat = str(self._classify(cls_name))
            if cat not in (CATEGORY_VEHICLE, CATEGORY_PERSON, CATEGORY_TRAIN):
                cat = CATEGORY_OTHER
            self._class_cache[cls_name] = cat
        return cat

    def _is_predicted(self, det: Detection, t: float) -> bool:
        """Was this detection produced by our own ``predict_boxes`` for this timestamp?

        The explicit signal is the ``PredictedDetection`` type. As a belt-and-braces
        check we also recognise a plain ``Detection`` whose id and box are bit-identical
        to what we predicted for exactly this timestamp (an integrator may rebuild
        Detection objects). A genuine detection can never collide with that.
        """
        if isinstance(det, PredictedDetection):
            return True
        if det.track_id is None or self._last_predicted_t is None or t != self._last_predicted_t:
            return False
        ref = self._last_predicted.get(int(det.track_id))
        return ref is not None and np.array_equal(ref, np.asarray(det.xyxy, dtype=np.float32).reshape(4))

    # --- reference coordinates (camera motion) -----------------------------------------

    def _reference_transform(self, zone: ZoneState) -> Optional[np.ndarray]:
        """The current-frame -> reference-frame matrix to use for this frame, or None.

        Contract with the zone stabiliser: ``ZoneState.to_reference`` maps current-frame
        pixels to reference-frame pixels; None on a healthy frame means the two coincide.
        On an unhealthy (held) frame the stabiliser holds the last good polygon, so the last
        healthy transform is held too: switching to raw pixels there would read the whole
        camera offset as one jump of every vehicle.
        """
        if not self.cfg.camera_motion_compensation:
            return None
        if bool(getattr(zone, "healthy", True)):
            M = getattr(zone, "to_reference", None)
            if M is None:
                self._last_ref_H = None
            else:
                M = np.asarray(M, dtype=np.float64)
                if M.shape == (3, 3) and np.all(np.isfinite(M)):
                    self._last_ref_H = M
        return self._last_ref_H

    def _ref_point(self, x: float, y: float) -> Tuple[float, float]:
        if self._ref_H is None:
            return x, y
        p = transform_points(np.array([[x, y]], dtype=np.float32), self._ref_H)[0]
        return float(p[0]), float(p[1])

    def _ref_box(self, box: Sequence[float]) -> np.ndarray:
        if self._ref_H is None:
            return np.asarray(box, dtype=np.float32).reshape(4).copy()
        return transform_box(box, self._ref_H)

    # --- identity -------------------------------------------------------------------------

    def _new_track(self, tid: int, det: Detection, box: np.ndarray, t: float) -> _Track:
        cat = self._category(det.cls_name)
        state = ObjectState(
            track_id=tid,
            cls_name=det.cls_name,
            is_vehicle=cat == CATEGORY_VEHICLE,
            is_person=cat == CATEGORY_PERSON,
            is_train=cat == CATEGORY_TRAIN,
            last_box=box,
            last_seen=t,
            first_seen=t,
            in_zone=False,
            in_zone_since=None,
            dwell_seconds=0.0,
            speed=0.0,
            speed_units=self._speed_units,
            stationary_seconds=0.0,
            stuck_seconds=0.0,
            history=[],
            uid=self._next_uid,
        )
        self._next_uid += 1
        # Keep fallback ids clear of any id the detector has used so far.
        self._next_fallback_id = max(self._next_fallback_id, tid + 1)
        return _Track(state=state, category=cat, cls_id=int(det.cls_id), conf=float(det.conf),
                      seen_count=0, last_detected_t=t, last_detected_box=box,
                      vehicle_since=t if cat == CATEGORY_VEHICLE else None)

    def _is_stopped(self, tr: _Track) -> bool:
        """Stopped by the speed rule, or standing in one still cluster for a while."""
        return tr.stationary or (self.cfg.stationarity != "speed"
                                 and tr.state.still_seconds >= _STOPPED_MIN_STILL_SECONDS)

    @staticmethod
    def _snapshot(tr: _Track) -> _Track:
        """Copy of a track's clocks and history (still clusters stay shared)."""
        snap = copy.copy(tr)
        snap.state = dataclasses.replace(tr.state, last_box=np.array(tr.state.last_box, copy=True),
                                         history=list(tr.state.history))
        snap.history = deque(tr.history)
        snap.last_detected_box = tr.last_detected_box.copy()
        return snap

    def _rekey_unseen_id(self, tid: int, det: Detection, frame_ids: Set[int], t: float) -> bool:
        """Adopt a live track for a brand-new detector id that appeared on top of it.

        Safety rationale (the whole reason this exists): a detector's tracker hands out a
        *new* id for the same physical vehicle after an occlusion. Creating a fresh track
        then would zero ``stuck_seconds`` and ``dwell_seconds`` and throw away the speed
        estimate, so the status would fall back to CLEAR and the ALARM would be delayed by
        the whole threshold again, for a vehicle that never moved. Since the box is still
        where it was, the right answer is that this is the same object.

        The match is deliberately narrow: only *live* tracks that no detection claims this
        frame, compared at their velocity-predicted box, and only above
        ``fallback_iou_threshold``. A dropped track is not resurrected (a vehicle that
        vanished for longer than the grace period and came back is a new object by contract,
        docs/ARCHITECTURE.md tracking test 5; alarmed vehicles behind a train are the one
        exception, see ``occlusion_memory_seconds``). With ``rekey_policy="strict"`` a vehicle
        box can only adopt a *stopped* vehicle track - a truck passing in front of a stopped
        car must not take its clocks, and a moving vehicle's new id is a new object - and a
        person box only a person track (stopped or not: walking along the tracks counts as
        dwell). Trains and unmapped classes are never re-keyed. If a truck was still first
        detected on top of the hidden car, the adopted clocks are remembered as long as a
        lost track would live (``lost_track_missed_runs`` detector runs and
        ``lost_track_grace_seconds``) and go back to the old id when it reappears
        (``_restore_rekeyed``). There is no box-size test: real same-car id flips on clip06
        have area ratios of 0.5-2. The best IoU wins; ties go to the oldest track so the
        result does not depend on dict ordering.

        Returns True when a track was re-keyed to ``tid`` (counters, history, in-zone and
        stuck streaks and ``seen_count`` all preserved).
        """
        strict = self.cfg.rekey_policy == "strict"
        cat = self._category(det.cls_name)
        if strict and cat not in (CATEGORY_VEHICLE, CATEGORY_PERSON):
            return False
        thr = float(self.cfg.fallback_iou_threshold)
        best_id: Optional[int] = None
        best_iou = 0.0
        for other, tr in self._tracks.items():
            if other in frame_ids:
                continue  # some detection of this frame owns that track
            if strict and (tr.category != cat or (cat == CATEGORY_VEHICLE and not self._is_stopped(tr))):
                continue
            iou = box_iou(det.xyxy, self._predicted_box(tr, t))
            if iou >= thr and iou > best_iou:
                best_id, best_iou = other, iou
        if best_id is None:
            return False
        tr = self._tracks.pop(best_id)
        if strict:
            self._rekeyed[best_id] = _Rekeyed(new_id=tid, t=t, snapshot=self._snapshot(tr))
        tr.state.track_id = tid
        if tr.cluster is not None and tr.cluster.owner == best_id:
            tr.cluster.owner = tid
        self._tracks[tid] = tr
        # Keep fallback ids clear of the id the detector has just introduced.
        self._next_fallback_id = max(self._next_fallback_id, tid + 1)
        return True

    def _restore_rekeyed(self, real_ids: Set[int], t: float) -> None:
        """Give the clocks back to an old id that reappears after a new id took them over.

        A take-over is remembered as long as the old track would have lived had it simply
        been lost: until ``lost_track_missed_runs`` detector runs have passed since it and it
        is older than ``lost_track_grace_seconds``.
        """
        grace = float(self.cfg.lost_track_grace_seconds)
        need = int(self.cfg.lost_track_missed_runs)
        for old in list(self._rekeyed):
            rec = self._rekeyed[old]
            if rec.runs >= need and t - rec.t > grace + _EPS:
                del self._rekeyed[old]
                continue
            if old not in real_ids or old in self._tracks:
                continue
            del self._rekeyed[old]
            # The new id started from the old clocks; whatever it is, it starts over.
            self._tracks.pop(rec.new_id, None)
            snap = rec.snapshot
            if snap.cluster is not None and snap.cluster.owner == rec.new_id:
                snap.cluster.owner = old
            self._tracks[old] = snap
            LOGGER.debug("track %d got its clocks back from id %d", old, rec.new_id)

    # --- hidden hazards ---------------------------------------------------------------------

    def _train_boxes_ref(self) -> List[np.ndarray]:
        return [self._ref_box(tr.state.last_box) for tr in self._tracks.values() if tr.category == CATEGORY_TRAIN]

    @staticmethod
    def _overlaps(a: np.ndarray, boxes: Sequence[np.ndarray]) -> bool:
        ax1, ay1, ax2, ay2 = [float(v) for v in a]
        for b in boxes:
            bx1, by1, bx2, by2 = [float(v) for v in b]
            if min(ax2, bx2) > max(ax1, bx1) and min(ay2, by2) > max(ay1, by1):
                return True
        return False

    def _count_missed_run(self, t: float) -> None:
        """A detector run at ``t``: every track it did not detect has missed one more run."""
        for tr in self._tracks.values():
            if tr.last_detected_t != t:
                tr.missed_runs += 1
        for rec in self._rekeyed.values():
            if rec.t != t:
                rec.runs += 1

    def _drop_lost(self, t: float, alarmed: Set[int]) -> None:
        """Drop tracks the detector has lost; hide alarmed vehicles under trains instead.

        Lost = missed by ``lost_track_missed_runs`` detector runs in a row AND last detected
        more than ``lost_track_grace_seconds`` ago. With the detector on every frame the grace
        decides (30 runs at 30 fps); with sparse detection (5 fps, detect_every_n_frames 7:
        runs 1.4 s apart) the run count does, so one missed run does not restart a parked
        car's clocks.

        ``alarmed`` is the ALARM episode's ids, latched included, so it can name a vehicle
        that has since driven off; only one last seen in or near the zone is hidden.
        """
        grace = float(self.cfg.lost_track_grace_seconds)
        need = int(self.cfg.lost_track_missed_runs)
        lost = [k for k, tr in self._tracks.items()
                if tr.missed_runs >= need and t - tr.last_detected_t > grace + _EPS]
        if not lost:
            return
        memory = float(self.cfg.occlusion_memory_seconds)
        trains = self._train_boxes_ref() if memory > 0 and alarmed else []
        for tid in lost:
            tr = self._tracks.pop(tid)
            if (memory > 0 and tid in alarmed and tr.category == CATEGORY_VEHICLE
                    and (tr.state.in_zone or tr.onnear)
                    and tr.last_ref_box is not None and self._overlaps(tr.last_ref_box, trains)):
                tr.state.occluded = True
                tr.velocity_px = (0.0, 0.0)  # it stopped where it was last seen
                self._hidden[tid] = _Hidden(track=tr, box_ref=tr.last_ref_box.copy(), last_train_t=t)
                LOGGER.info("alarmed vehicle %d lost under a train box at t=%.2f s: kept as a hidden hazard", tid, t)

    def _advance_undetected(self, t: float, updated: Set[int]) -> None:
        """Run the dwell and stuck clocks of the live in-zone vehicles this detector run missed.

        Without this they froze at the last detection until the vehicle was seen again or
        dropped (clip07's van, no longer detected once the train reached it at 5.0 s, kept
        the stuck time of its last box until it was dropped at 6.0 s). They run from the same
        since-timestamps as at that detection, until the track is dropped; a hidden hazard
        runs in ``_update_hidden``. Skipped detector frames do not need this: the predicted
        boxes advance the clocks there. Off with ``undetected_clock_while_live: false``.
        """
        for tid, tr in self._tracks.items():
            if tid in updated or tr.category != CATEGORY_VEHICLE or not tr.state.in_zone:
                continue
            st = tr.state
            if st.in_zone_since is not None:
                st.dwell_seconds = max(0.0, t - st.in_zone_since)
            if tr.stuck_since is not None:
                st.stuck_seconds = max(0.0, t - tr.stuck_since)

    def _update_hidden(self, t: float) -> None:
        """Keep hidden hazards while a train box covers them; advance their clocks."""
        memory = float(self.cfg.occlusion_memory_seconds)
        trains = self._train_boxes_ref()
        for tid in list(self._hidden):
            h = self._hidden[tid]
            if self._overlaps(h.box_ref, trains):
                h.last_train_t = t
            elif t - h.last_train_t > memory + _EPS:
                del self._hidden[tid]
                LOGGER.info("hidden hazard %d expired at t=%.2f s (not re-detected)", tid, t)
                continue
            st, tr = h.track.state, h.track
            if st.in_zone_since is not None:
                st.dwell_seconds = max(0.0, t - st.in_zone_since)
            if tr.stuck_since is not None:
                st.stuck_seconds = max(0.0, t - tr.stuck_since)

    def _reacquire_hidden(self, real: Sequence[Detection], t: float) -> None:
        """A box on top of a hidden hazard (or carrying its id) brings the hazard's track back."""
        for det in real:
            if not self._hidden:
                return
            tid = None if det.track_id is None else int(det.track_id)
            if tid is not None and tid in self._tracks:
                continue  # an existing track's own detection
            if tid is not None and tid in self._hidden:
                key = tid
            else:
                if self._category(det.cls_name) != CATEGORY_VEHICLE:
                    continue
                box = self._ref_box(det.xyxy)
                key, best = None, _HAZARD_REACQUIRE_IOU
                for hid, h in self._hidden.items():
                    v = box_iou(h.box_ref, box)
                    if v >= best:
                        key, best = hid, v
                if key is None:
                    continue
            h = self._hidden.pop(key)
            tr = h.track
            tr.state.occluded = False
            new_id = key if tid is None else tid  # an id-less box is matched to it by the fallback matcher
            tr.state.track_id = new_id
            if tr.cluster is not None:
                tr.cluster.owner = new_id
                self._clusters.revive(tr.cluster, t)
            self._tracks[new_id] = tr
            self._next_fallback_id = max(self._next_fallback_id, new_id + 1)
            LOGGER.info("hidden hazard %d re-detected as %d at t=%.2f s: clocks restored", key, new_id, t)

    def _note_detector_run(self, t: float) -> None:
        """Remember the detector cadence; report once when it is slower than the still gap."""
        if self._last_run_t is not None and self.cfg.stationarity != "speed" and not self._gap_reported:
            gap = t - self._last_run_t
            if gap > float(self.cfg.still_max_gap_seconds) + _EPS:
                self._gap_reported = True
                self._pending_gap = gap
                LOGGER.warning("detector runs %.2f s apart but stuck.still_max_gap_seconds is %.2f s: ID-free "
                               "stationarity cannot accumulate; lower detect_every_n_frames or raise the gap",
                               gap, float(self.cfg.still_max_gap_seconds))
        self._last_run_t = t

    # --- still clusters ---------------------------------------------------------------------

    def _cut_by_border(self, box: Sequence[float], frame_hw: Optional[Tuple[int, int]]) -> bool:
        """True when ``box`` reaches within ``still_border_margin`` of the image border."""
        margin = float(self.cfg.still_border_margin)
        if margin <= 0.0 or frame_hw is None:
            return False
        h, w = frame_hw
        x1, y1, x2, y2 = [float(v) for v in box]
        mx, my = margin * w, margin * h
        return x1 <= mx or y1 <= my or x2 >= w - mx or y2 >= h - my

    def _train_beside(self, box: np.ndarray) -> bool:
        """True when one of the recent real train boxes (``_BORDER_TRAIN_SECONDS``) overlaps
        ``box`` grown by its own height on every side (reference-frame pixels)."""
        if not self._recent_trains:
            return False
        x1, y1, x2, y2 = [float(v) for v in box]
        m = max(0.0, y2 - y1)
        grown = np.array([x1 - m, y1 - m, x2 + m, y2 + m], dtype=np.float32)
        return self._overlaps(grown, [b for _, b in self._recent_trains])

    def _assign_clusters(self, real: Sequence[Detection], t: float,
                         frame_hw: Optional[Tuple[int, int]] = None) -> Dict[int, StillCluster]:
        """Feed this frame's real vehicle boxes to the still clusters and settle ownership.

        A box cut by the image border with a train box beside it (seen within
        ``_BORDER_TRAIN_SECONDS``) sets its cluster's ``cut_t``, from which the cluster's
        still clock restarts: the visible part of a train passing the edge of the picture
        keeps the same box while the train moves (clip06: the end of a passing train cut off
        by the right edge, labelled 'bus', stood "still" for 10 s beside the train's own
        boxes). While that lasts the speed rule alone decides, as before the port. Without a
        train beside it a border box is an ordinary box: a vehicle stopped at the edge of a
        zone that reaches the image border keeps its still clock (six of the eight team zones
        span the full frame width). The boxes always feed the clusters, so the anchors of
        everything else stay where they were.

        Ownership: the creator of a cluster owns it, and keeps it while its id is among this
        frame's real detections anywhere. Only when the owner's id is absent does the cluster
        go to an id-carrying detection in it whose track was born there or whose id is new
        this frame (a re-identification on top of the stop, including ids the re-keying will
        adopt). Every other member is a guest and only gets its own time at the spot, so a
        dense moving queue never inherits the age of the car ahead.
        """
        if self.cfg.stationarity == "speed":
            return {}
        vehicles = []
        border = float(self.cfg.still_border_margin) > 0.0 and frame_hw is not None
        for d in real:
            own = self._tracks.get(int(d.track_id)) if d.track_id is not None else None
            if self._category(d.cls_name) == CATEGORY_VEHICLE or (own is not None and own.category == CATEGORY_VEHICLE):
                vehicles.append(d)
            elif border and self._category(d.cls_name) == CATEGORY_TRAIN:
                self._recent_trains.append((t, self._ref_box(d.xyxy)))
        while self._recent_trains and t - self._recent_trains[0][0] > _BORDER_TRAIN_SECONDS + _EPS:
            self._recent_trains.popleft()
        if not vehicles:
            self._clusters.update(t, [])  # still expire old clusters
            return {}
        refs = [self._ref_box(d.xyxy) for d in vehicles]
        assigned = self._clusters.update(t, refs)
        clusters = {id(d): c for d, c in zip(vehicles, assigned)}
        if border:
            for d, ref, c in zip(vehicles, refs, assigned):
                if self._cut_by_border(d.xyxy, frame_hw) and self._train_beside(ref):
                    c.cut_t = t
        present: Dict[int, List[int]] = {}
        for d in vehicles:
            if d.track_id is not None:
                present.setdefault(id(clusters[id(d)]), []).append(int(d.track_id))
        frame_ids = {int(d.track_id) for d in real if d.track_id is not None}
        seen: Set[int] = set()
        for c in assigned:
            if id(c) in seen:
                continue
            seen.add(id(c))
            ids = present.get(id(c), [])
            if not ids or c.owner in ids:
                continue
            if c.owner is not None and c.owner in frame_ids:
                continue  # the owner drove off and is still visible: newcomers are guests
            for i in ids:
                other = self._tracks.get(i)
                if other is None or other.birth_cluster is c:
                    c.owner = i
                    break
        return clusters

    def _join_cluster(self, tr: _Track, tid: int, cluster: Optional[StillCluster], t: float) -> None:
        if cluster is not None:
            if tr.seen_count == 1:
                tr.birth_cluster = cluster
            if cluster.owner is None:
                cluster.owner = tid  # the creator, or an id-less (fallback) detection on first sight
            if cluster is not tr.cluster:
                tr.cluster, tr.cluster_joined = cluster, t
        elif tr.category != CATEGORY_VEHICLE:
            tr.cluster = None

    def _settle_anchor(self, tr: _Track, tid: int, cluster: Optional[StillCluster], box: np.ndarray) -> None:
        """Move the anchor of the owner's cluster, once, to its box where its speed first drops
        below ``speed_threshold``; the cluster keeps its start.

        A vehicle braking into the zone opens its last cluster while it still rolls, up to a
        third of its width before the stop. Anchored there, a creep of 20 px after the stop
        leaves the cluster and restarts the still clock (clip07's van: anchor taken at 93 px/s,
        45 px before the stop; IoU 0.58 with the stopped box, 0.73 with the anchor at the
        stop). Only the owner moves it, only from a real box of this cluster, and only once
        (``settled``), so a creeping vehicle still leaves its cluster after a third of its
        width. Clusters exist only when ``stationarity`` is not "speed".
        """
        c = tr.cluster
        if (c is None or cluster is not c or c.settled or c.owner != tid or not tr.has_speed
                or tr.state.speed >= float(self.cfg.speed_threshold)):
            return
        c.anchor = self._ref_box(box)
        c.settled = True

    # --- per-detection update -----------------------------------------------------------------

    def _apply_label(self, tr: _Track, det: Detection, t: float, alarmed: bool = False) -> None:
        """Follow the detector's label, except a short flicker on a stopped in-zone vehicle
        and any run of other labels on an alarmed vehicle in or near the zone.

        Class can flicker (car <-> truck, but also car -> train/boat for a few frames while
        the box merges with a person or a gate arm). A track that has been a vehicle for at
        least ``label_hold_seconds`` and stands in the zone keeps its vehicle category
        through a run of other labels shorter than that; a longer run changes it. Moving
        tracks always follow the label, so passing trains labelled bus/car are never turned
        into in-zone vehicles by a vote.

        A vehicle of the ALARM episode in progress (``alarmed``) that is in or near the zone
        keeps its vehicle category (and label) for as long as the episode lasts, whatever
        the run length, moving or not: a stuck car whose box merges with the arriving train
        is labelled 'train' for seconds (clip07's car 29 at 4.4 s), and as a train it would
        leave the engine, its still cluster and the hidden-hazard memory, ending the ALARM
        with the latch while the car is still under the train. When the episode ends, a run
        longer than ``label_hold_seconds`` changes the category at the next detection.
        ``label_hold_seconds`` = 0 turns this off too.
        """
        st = tr.state
        if det.cls_name == st.cls_name:
            tr.label_run_since = None
            return
        new_cat = self._category(det.cls_name)
        hold = float(self.cfg.label_hold_seconds)
        if hold > 0.0 and tr.category == CATEGORY_VEHICLE and new_cat != CATEGORY_VEHICLE:
            if alarmed and (st.in_zone or tr.onnear):
                if tr.label_run_since is None:
                    tr.label_run_since = t
                return
            if (st.in_zone and self._is_stopped(tr)
                    and tr.vehicle_since is not None and t - tr.vehicle_since >= hold - _EPS):
                if tr.label_run_since is None:
                    tr.label_run_since = t
                if t - tr.label_run_since < hold - _EPS:
                    return
        previous = tr.category
        st.cls_name = det.cls_name
        tr.category = new_cat
        st.is_vehicle = new_cat == CATEGORY_VEHICLE
        st.is_person = new_cat == CATEGORY_PERSON
        st.is_train = new_cat == CATEGORY_TRAIN
        tr.label_run_since = None
        if new_cat != CATEGORY_VEHICLE:
            tr.vehicle_since = None
        elif previous != CATEGORY_VEHICLE:
            tr.vehicle_since = t

    @staticmethod
    def _debounce(tr: _Track, attr: str, raw: bool, previous: bool, t: float, grace: float) -> bool:
        """``raw`` with an exit hold: True stays True until ``raw`` has been False > ``grace`` s."""
        if raw:
            setattr(tr, attr, None)
            return True
        if previous and grace > 0.0:
            since = getattr(tr, attr)
            if since is None:
                setattr(tr, attr, t)
                since = t
            if t - since <= grace + _EPS:
                return True
        setattr(tr, attr, None)
        return False

    def _update_membership(self, tr: _Track, box: np.ndarray, point: Tuple[float, float],
                           zone: ZoneState, t: float) -> None:
        """Debounced in-zone and on-or-near state from one real detection."""
        st = tr.state
        # Brand-new ids are ignored for zone logic: ByteTrack (and the IoU fallback)
        # can spawn a one-frame id on a flickering box, and we never want such an id
        # to start a dwell timer.
        if tr.seen_count < int(self.cfg.min_track_age_frames):
            st.in_zone = st.near_zone = tr.onnear = False
            tr.zone_out_since = tr.onnear_out_since = None
            return
        raw_in, raw_near = self._zone_membership(tr, box, point, zone)
        grace = float(self.cfg.zone_exit_grace_seconds) if tr.category == CATEGORY_VEHICLE else 0.0
        st.in_zone = self._debounce(tr, "zone_out_since", raw_in, st.in_zone, t, grace)
        tr.onnear = self._debounce(tr, "onnear_out_since", raw_in or raw_near, tr.onnear, t, grace)
        st.near_zone = tr.onnear and not st.in_zone

    def _zone_membership(self, tr: _Track, box: np.ndarray, point: Tuple[float, float],
                         zone: ZoneState) -> Tuple[bool, bool]:
        """``(in_zone, near_zone)`` for one real detection, against the zone of this frame.

        Vehicles under ``zone_test="footprint"``: in the zone when the bottom
        ``footprint_fraction`` of the box overlaps the zone mask by at least
        ``footprint_min_overlap`` (and by more than 0); "near" (too close) when not in it
        but within ``close_box_heights`` box heights of it. The point test is deliberately
        not OR-ed in: on the team clips it only added far-side cars touching a thin band
        (+37 s of false ALARM) and no recall. Everything else (persons, trains, unmapped
        classes, or ``zone_test="bottom_center"``) keeps the bottom-center point test and is
        never "near".
        """
        st = tr.state
        if tr.category == CATEGORY_VEHICLE and self.cfg.zone_test == "footprint":
            overlap, dist_bh = footprint_relation(box, zone.polygon, zone.mask, float(self.cfg.footprint_fraction))
            st.zone_overlap, st.zone_distance_bh = overlap, dist_bh
            inside = overlap > 0.0 and overlap >= float(self.cfg.footprint_min_overlap)
            close = float(self.cfg.close_box_heights)
            return inside, (not inside) and close > 0.0 and dist_bh <= close
        inside = point_in_polygon(point, zone.polygon)
        st.zone_overlap, st.zone_distance_bh = (1.0, 0.0) if inside else (0.0, math.inf)
        return inside, False

    def _streak_start(self, tr: _Track, active: bool, own_since: Optional[float],
                      owns_still: bool, predicted: bool, attr: str) -> Optional[float]:
        """Start of a membership streak; for a still cluster's owner, the cluster's streak.

        The streak belongs to the place, not the id: when the re-keying cannot match an id
        switch, the new owner of the cluster keeps the in-zone (or on-or-near) clock. The
        first frames of a brand-new id (``min_track_age_frames``) are not zone evidence, so
        they neither extend nor end the cluster's streak.
        """
        if not owns_still or tr.cluster is None:
            return own_since
        c = tr.cluster
        if not predicted and tr.seen_count >= int(self.cfg.min_track_age_frames):
            if active and own_since is not None:
                cur = getattr(c, attr)
                setattr(c, attr, own_since if cur is None else min(cur, own_since))
            else:
                setattr(c, attr, None)
        cur = getattr(c, attr)
        if active and cur is not None and own_since is not None:
            return min(own_since, cur)
        return own_since

    def _observe(self, tid: int, det: Detection, zone: ZoneState, t: float,
                 cluster: Optional[StillCluster] = None, alarmed: bool = False) -> None:
        """Apply one detection (real or predicted) to track ``tid`` at time ``t``.

        ``cluster`` is the still cluster this (real, vehicle) detection joined, if any;
        ``alarmed`` says whether ``tid`` is in the alert engine's ALARM episode.
        """
        predicted = self._is_predicted(det, t)
        box = np.asarray(det.xyxy, dtype=np.float32).reshape(4).copy()
        tr = self._tracks.get(tid)
        if tr is None:
            if predicted:
                return  # a prediction for an id we no longer track is not evidence of anything
            tr = self._new_track(tid, det, box, t)
            self._tracks[tid] = tr
        st = tr.state

        if not predicted:
            # --- real evidence: refresh identity, position history, speed, zone membership
            tr.seen_count += 1
            tr.missed_runs = 0
            tr.last_detected_t = t
            tr.last_detected_box = box
            tr.conf = float(det.conf)
            tr.cls_id = int(det.cls_id)
            self._apply_label(tr, det, t, alarmed)
            st.last_seen = t
            tr.last_ref_box = self._ref_box(box)

            x, y = bottom_center(box)
            tr.history.append((t,) + self._ref_point(x, y))
            self._prune_history(tr, t)
            self._estimate_speed(tr)
            self._join_cluster(tr, tid, cluster, t)
            self._settle_anchor(tr, tid, cluster, box)
            self._update_membership(tr, box, (x, y), zone, t)
            # Safety: "stationary" requires an actual speed estimate. Speed reads 0.0
            # before one exists, and 0.0 < threshold would otherwise start the stuck
            # clock on the very first frame of every new track.
            tr.stationary = tr.has_speed and st.speed < float(self.cfg.speed_threshold)

        # A predicted box only moves the drawn box and advances the clocks; zone
        # membership, speed and clusters are left exactly as the last real detection set them.
        st.last_box = box

        # --- timestamp-based counters (frame-rate independent)
        if st.in_zone:
            if st.in_zone_since is None:
                st.in_zone_since = t
            st.dwell_seconds = max(0.0, t - st.in_zone_since)
        else:
            st.in_zone_since = None
            st.dwell_seconds = 0.0

        if tr.stationary:
            if tr.stationary_since is None:
                tr.stationary_since = t
            st.stationary_seconds = max(0.0, t - tr.stationary_since)
        else:
            tr.stationary_since = None
            st.stationary_seconds = 0.0

        # ID-free clock: how long this vehicle's box has kept re-appearing where it stood.
        # A border box with a train beside it restarts it (cut_t; the speed rule decides then).
        still_since: Optional[float] = None
        owns_still = False
        c = tr.cluster
        if tr.category == CATEGORY_VEHICLE and c is not None and self._clusters.alive(c, t):
            owns_still = c.owner == st.track_id
            # The owner carries the stop's age; a guest only its own time at this spot.
            still_since = c.start if owns_still else max(c.start, tr.cluster_joined)
            if c.cut_t is not None:
                still_since = max(still_since, c.cut_t)
        st.still_seconds = 0.0 if still_since is None else max(0.0, t - still_since)

        # Stopped by either measure (cfg.stationarity): the earlier of the two "continuously
        # since" timestamps - both runs extend to now, so their union does too.
        sinces = []
        if self.cfg.stationarity != "still" and tr.stationary and tr.stationary_since is not None:
            sinces.append(tr.stationary_since)
        if self.cfg.stationarity != "speed" and still_since is not None:
            sinces.append(still_since)
        tr.stopped_since = min(sinces) if sinces else None

        zone_start = self._streak_start(tr, st.in_zone, st.in_zone_since, owns_still, predicted, "zone_since")
        if st.in_zone and tr.stopped_since is not None and zone_start is not None:
            tr.stuck_since = max(zone_start, tr.stopped_since)
            st.stuck_seconds = max(0.0, t - tr.stuck_since)
        else:
            tr.stuck_since = None
            st.stuck_seconds = 0.0

        # "On the tracks or too close" streak: the merged region of the team's rule.
        if tr.onnear:
            if tr.near_since is None:
                tr.near_since = t
        else:
            tr.near_since = None
        near_start = self._streak_start(tr, tr.onnear, tr.near_since, owns_still, predicted, "near_since")
        if tr.onnear and tr.stopped_since is not None and near_start is not None:
            st.near_stuck_seconds = max(0.0, t - max(near_start, tr.stopped_since))
        else:
            st.near_stuck_seconds = 0.0

        if not predicted:
            st.history = list(tr.history)

    def _prune_history(self, tr: _Track, t: float) -> None:
        """Keep samples covering at least ``speed_window_seconds`` (and at least the
        minimum sample count, so a low frame rate still yields an estimate)."""
        cutoff = t - float(self.cfg.speed_window_seconds)
        h = tr.history
        while len(h) > self._min_history and h[0][0] < cutoff:
            h.popleft()

    def _estimate_speed(self, tr: _Track) -> None:
        """Least-squares velocity over the history window; sets speed / velocity / has_speed.

        We fit x = a + vx*t and y = b + vy*t independently (closed-form slope
        cov(t, x) / var(t)) and report the magnitude of (vx, vy). The samples are in
        reference-frame pixels when the zone provides ``to_reference`` (camera motion
        removed), image pixels otherwise. The pixel velocity is always kept because
        ``predict_boxes`` needs it; when a ground-plane homography is configured the
        *reported* speed is the magnitude of the same fit done on the ground-plane points,
        in m/s.
        """
        st = tr.state
        h = tr.history
        if len(h) < self._min_history:
            tr.has_speed, tr.velocity_px, st.speed = False, (0.0, 0.0), 0.0
            return
        arr = np.asarray(h, dtype=np.float64)  # (n, 3): t, x, y
        ts = arr[:, 0]
        tm = ts - ts.mean()
        denom = float(np.dot(tm, tm))
        if denom <= 1e-12:
            # All samples share one timestamp (duplicate updates); no slope is defined.
            tr.has_speed, tr.velocity_px, st.speed = False, (0.0, 0.0), 0.0
            return
        vx = float(np.dot(tm, arr[:, 1] - arr[:, 1].mean()) / denom)
        vy = float(np.dot(tm, arr[:, 2] - arr[:, 2].mean()) / denom)
        tr.velocity_px = (vx, vy)

        if self._ground_H is None:
            st.speed = math.hypot(vx, vy)
            tr.has_speed = True
            return
        ground = transform_points(arr[:, 1:3], self._ground_H).astype(np.float64)
        if not np.all(np.isfinite(ground)):
            # Point at/above the horizon of the homography: no meaningful ground speed.
            # Failing "no estimate" is the safe direction (no stuck accumulation).
            tr.has_speed, st.speed = False, 0.0
            return
        gvx = float(np.dot(tm, ground[:, 0] - ground[:, 0].mean()) / denom)
        gvy = float(np.dot(tm, ground[:, 1] - ground[:, 1].mean()) / denom)
        st.speed = math.hypot(gvx, gvy)
        tr.has_speed = True

    def _predicted_box(self, tr: _Track, t: float) -> np.ndarray:
        """Last detected box shifted by velocity x elapsed time (never shifted backwards)."""
        dt = max(0.0, t - tr.last_detected_t)
        vx, vy = tr.velocity_px
        shift = np.array([vx * dt, vy * dt, vx * dt, vy * dt], dtype=np.float32)
        return (tr.last_detected_box + shift).astype(np.float32)

    def _match_fallback(self, dets: Sequence[Detection], claimed: Set[int], t: float) -> List[int]:
        """Greedy IoU assignment of id-less detections to live tracks; new ids otherwise.

        Candidates are the live tracks not already claimed by an id-carrying detection
        this frame, compared at their velocity-predicted position for time ``t``. Pairs
        are taken in descending IoU order, each detection and each track used at most
        once, and only when IoU >= ``fallback_iou_threshold``.
        """
        thr = float(self.cfg.fallback_iou_threshold)
        cands = [tid for tid in self._tracks if tid not in claimed]
        cand_boxes = [self._predicted_box(self._tracks[tid], t) for tid in cands]
        pairs: List[Tuple[float, int, int]] = []
        for i, det in enumerate(dets):
            for j, cb in enumerate(cand_boxes):
                iou = box_iou(det.xyxy, cb)
                if iou >= thr:
                    pairs.append((iou, i, j))
        pairs.sort(key=lambda p: (-p[0], p[1], p[2]))  # best IoU first; deterministic ties
        assigned: List[Optional[int]] = [None] * len(dets)
        used: Set[int] = set()
        for _, i, j in pairs:
            if assigned[i] is None and j not in used:
                assigned[i] = cands[j]
                used.add(j)
        out: List[int] = []
        for a in assigned:
            if a is None:
                a = self._next_fallback_id
                self._next_fallback_id += 1
            out.append(a)
        return out
