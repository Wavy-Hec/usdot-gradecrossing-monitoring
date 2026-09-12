"""Per-object dwell / speed / stuck bookkeeping for the grade-crossing monitor.

This module turns a stream of per-frame detections into *per-object* safety
measurements. The design rules it implements (see docs/ARCHITECTURE.md):

* **Per-object, not per-frame.** Every measurement hangs off a track id. A detector
  that provides ids (ultralytics + ByteTrack) is used as-is; detections without ids
  are matched to existing tracks by a small greedy IoU matcher so the module works
  with any detector.
* **Point-in-polygon on the bottom-center of the box.** The bottom-center of a box is
  the point where the object touches the ground, so it is the right thing to test
  against the zone polygon. Bounding-box pixel overlap is never used: a car on the
  adjacent road whose box corner crosses the zone in image space is not in the zone.
* **Stuck = in zone AND slow, continuously, for T seconds.** Speed is a least-squares
  linear fit of x(t), y(t) over a short window of bottom-center samples, which is far
  more robust to box jitter than frame-to-frame differences. Until enough samples exist
  there is *no* speed estimate, and without a speed estimate an object can never be
  "stationary" - so a brand-new track can never trigger an instant alarm.
* **Timestamp-based counters.** ``dwell_seconds``, ``stationary_seconds`` and
  ``stuck_seconds`` are ``now - since`` differences, not frame counts, so thresholds in
  seconds mean the same thing at 10 fps on a Jetson and 30 fps on a workstation, and a
  short detector dropout (shorter than ``lost_track_grace_seconds``) does not reset them.
* **An id swap in place does not restart the clocks.** ByteTrack gives a vehicle a new id
  after an occlusion (a passing train, a truck crossing in front of it). A fresh id means
  fresh counters and no speed estimate, so a vehicle that has not moved would drop back to
  CLEAR and need the whole threshold again - the exact opposite of what a stuck vehicle
  should do. Before a brand-new detector id creates a track, it is therefore matched by
  IoU against the live tracks no detection claimed this frame, and the existing track is
  re-keyed to the new id with every counter intact (see ``_rekey_unseen_id``).

Nothing here imports torch or ultralytics; the whole module is exercised with synthetic
detections in ``tests/test_tracking.py``.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from .config import StuckConfig
from .geometry import bottom_center, box_iou, point_in_polygon, transform_points
from .types import Detection, ObjectState, ZoneState

# Categories returned by the ``classify`` callable (see VehicleDetector.classify).
CATEGORY_VEHICLE = "vehicle"
CATEGORY_PERSON = "person"
CATEGORY_TRAIN = "train"
CATEGORY_OTHER = "other"

# Categories that count as "something is on the crossing" for zone_occupied.
# The zone-occupied flag gates background refresh and zone keyframe refresh, so it must
# be True whenever *anything* is on the tracks - including CATEGORY_OTHER. An object of
# a class the config does not map (COCO "bench"/"boat"/"suitcase", or any class of a
# fine-tuned model missing from vehicle_classes/person_classes/train_classes) is already
# invisible to the stuck logic and is still subtracted from the debris mask; if it also
# failed to block the empty-zone background refresh, a real obstruction sitting in the
# crossing would be quietly learned into the background and could never raise anything.
# Blocking refresh is the one protection left, so "other" counts here.
_OCCUPYING_CATEGORIES = (CATEGORY_VEHICLE, CATEGORY_PERSON, CATEGORY_TRAIN, CATEGORY_OTHER)


class PredictedDetection(Detection):
    """A ``Detection`` synthesised by :meth:`StuckVehicleTracker.predict_boxes`.

    On frames where the detector is skipped, the monitor feeds the tracker's own
    velocity-propagated boxes back into :meth:`StuckVehicleTracker.update` so the
    per-object counters keep advancing. Those boxes are *not evidence* that the object
    is still there, so the tracker must be able to tell them apart from real detector
    output: a predicted box never refreshes ``last_seen``, never adds a position sample
    to the speed window, and never changes zone membership. Without this distinction a
    track that the detector has lost would keep re-observing its own predictions and
    never be dropped. The class adds no fields; it only marks provenance.
    """


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
    history: Deque[Tuple[float, float, float]] = field(default_factory=deque)  # (t, x, y) pixels
    velocity_px: Tuple[float, float] = (0.0, 0.0)  # least-squares pixel velocity (px/s)
    has_speed: bool = False  # True once a speed estimate exists
    stationary: bool = False  # has_speed and speed < threshold, at the last real detection
    stationary_since: Optional[float] = None
    stuck_since: Optional[float] = None


class StuckVehicleTracker:
    """Maintain dwell / speed / stuck measurements for every tracked object.

    Parameters
    ----------
    cfg:
        ``StuckConfig`` with the speed threshold, speed window, track bookkeeping
        limits and the optional ground-plane homography.
    classify:
        Callable mapping a detector class name to ``"vehicle" | "person" | "train" |
        "other"`` (normally ``VehicleDetector.classify``). The tracker never looks at
        class names itself so that class lists live in exactly one place (the config).

    Typical use per frame::

        dets = detector.detect(frame) if detector_frame else tracker.predict_boxes(t)
        objects = tracker.update(dets, zone_state, t)

    ``update`` returns the state of every *live* object (seen within
    ``lost_track_grace_seconds``), not only the ones seen this frame, so a one-frame
    detector dropout does not make a stuck vehicle disappear from the alert logic.
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
        # State initialised in reset() so that reset() is the single source of truth.
        self._tracks: Dict[int, _Track] = {}
        self._next_fallback_id = 1
        self._zone_empty_since: Optional[float] = None
        self._last_predicted: Dict[int, np.ndarray] = {}
        self._last_predicted_t: Optional[float] = None
        self.reset()

    # ------------------------------------------------------------------ public API

    def reset(self) -> None:
        """Forget every track (new video / detector reset). Ids restart from 1."""
        self._tracks = {}
        self._next_fallback_id = 1
        self._zone_empty_since = None
        self._last_predicted = {}
        self._last_predicted_t = None

    def update(self, detections: List[Detection], zone: ZoneState, timestamp: float) -> List[ObjectState]:
        """Ingest one frame of detections at ``timestamp`` and return all live objects.

        Steps:

        1. Detections that carry a ``track_id`` update (or create) the track with that id.
           A *new* id whose box still overlaps a live track that no detection claimed this
           frame re-keys that track instead of starting a fresh one (id swap in place).
        2. Detections without an id are matched greedily by IoU against the remaining
           live tracks (``fallback_iou_threshold``); unmatched ones get fresh ids.
        3. Tracks not seen by a real detection for more than ``lost_track_grace_seconds``
           are dropped.
        4. ``zone_empty_since`` is refreshed from the surviving objects.

        Timestamps must be seconds and should be non-decreasing; counters are clamped
        at zero if a timestamp ever steps backwards.

        Field semantics worth knowing: ``ObjectState.last_seen`` and ``history`` reflect
        *real* detections only, while ``last_box`` follows predicted boxes too (so an
        overlay keeps moving on skipped frames). Fallback ids are allocated above every
        detector id seen so far; a stream that mixes detector ids and id-less boxes can
        still collide if the detector later issues an id we already handed out.
        """
        t = float(timestamp)
        updated: Set[int] = set()

        with_id: List[Detection] = []
        without_id: List[Detection] = []
        for det in detections:
            (with_id if det.track_id is not None else without_id).append(det)

        # Ids the detector uses this frame: a track one of them owns must never be stolen
        # by the re-keying below, even if its detection has not been processed yet.
        frame_ids: Set[int] = {int(d.track_id) for d in with_id}  # type: ignore[arg-type]
        for det in with_id:
            tid = int(det.track_id)  # type: ignore[arg-type]
            if tid in updated:
                continue  # a detector should never emit an id twice per frame; keep the first
            if tid not in self._tracks and not self._is_predicted(det, t):
                self._rekey_unseen_id(tid, det, frame_ids, t)
            self._observe(tid, det, zone, t)
            updated.add(tid)

        if without_id:
            for det, tid in zip(without_id, self._match_fallback(without_id, updated, t)):
                self._observe(tid, det, zone, t)
                updated.add(tid)

        # Drop tracks the detector has genuinely lost. Predicted boxes never refresh
        # last_detected_t, so a ghost cannot keep itself alive.
        grace = float(self.cfg.lost_track_grace_seconds)
        for tid in [k for k, tr in self._tracks.items() if t - tr.last_detected_t > grace]:
            del self._tracks[tid]

        if self.zone_occupied:
            self._zone_empty_since = None
        elif self._zone_empty_since is None:
            self._zone_empty_since = t
        return self.objects()

    def predict_boxes(self, timestamp: float) -> List[Detection]:
        """Velocity-propagated boxes for every live object, for frames without a detector run.

        Each box is the last *detected* box of the object shifted by its least-squares
        pixel velocity times the elapsed time since that detection (zero shift when no
        velocity estimate exists yet). Ids, class and confidence are preserved. The
        returned objects are :class:`PredictedDetection` so that feeding them back into
        :meth:`update` advances the clocks without counting as new evidence.
        """
        t = float(timestamp)
        grace = float(self.cfg.lost_track_grace_seconds)
        out: List[Detection] = []
        self._last_predicted = {}
        self._last_predicted_t = t
        for tid, tr in self._tracks.items():
            if t - tr.last_detected_t > grace:
                continue  # would be dropped at the next update anyway; do not resurrect it
            box = self._predicted_box(tr, t)
            out.append(PredictedDetection(xyxy=box, conf=tr.conf, cls_id=tr.cls_id,
                                          cls_name=tr.state.cls_name, track_id=tid))
            self._last_predicted[tid] = box.copy()
        return out

    @property
    def zone_occupied(self) -> bool:
        """True when any live tracked object, of any class, has ``in_zone`` True.

        Used by the monitor to decide whether the scene is static enough to refresh the
        background model and the zone keyframe. Unmapped ("other") classes count too: such
        an object never feeds the stuck logic and is still excluded from the debris mask,
        so refusing the refresh is the only thing left that stops a real obstruction from
        being absorbed into the background. The cost is that a permanent static false
        positive inside the zone also blocks refresh - visible as a stale background and
        an object in the overlay, and fixable by mapping the class in the config, which is
        much better than silently losing an obstruction.
        """
        return any(tr.state.in_zone and tr.category in _OCCUPYING_CATEGORIES
                   for tr in self._tracks.values())

    @property
    def zone_empty_since(self) -> Optional[float]:
        """Timestamp at which the zone last became empty, or None while occupied.

        None also before the first ``update`` call (nothing is known yet). The monitor
        derives ``zone_empty_seconds = t - zone_empty_since`` for the debris detector.
        """
        return self._zone_empty_since

    def objects(self) -> List[ObjectState]:
        """Current state of every live object, in the order the tracks were created."""
        return [tr.state for tr in self._tracks.values()]

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
        )
        # Keep fallback ids clear of any id the detector has used so far.
        self._next_fallback_id = max(self._next_fallback_id, tid + 1)
        return _Track(state=state, category=cat, cls_id=int(det.cls_id), conf=float(det.conf),
                      seen_count=0, last_detected_t=t, last_detected_box=box)

    def _rekey_unseen_id(self, tid: int, det: Detection, frame_ids: Set[int], t: float) -> bool:
        """Adopt a live track for a brand-new detector id that appeared on top of it.

        Safety rationale (the whole reason this exists): a detector's tracker hands out a
        *new* id for the same physical vehicle after an occlusion - the classic case at a
        crossing is a train passing in front of a car stopped on the tracks. Creating a
        fresh track then would zero ``stuck_seconds`` and ``dwell_seconds`` and throw away
        the speed estimate, so the status would fall back to CLEAR and the ALARM would be
        delayed by the whole threshold again, for a vehicle that never moved. Since the
        box is still where it was, the right answer is that this is the same object.

        The match is deliberately narrow: only *live* tracks (seen by a real detection
        within ``lost_track_grace_seconds``) that no detection claims this frame, compared
        at their velocity-predicted box, and only above ``fallback_iou_threshold``. A track
        that has already been dropped is not resurrected - a vehicle that vanished for
        longer than the grace period and came back is a new object by contract (see
        docs/ARCHITECTURE.md, tracking test 5). The best IoU wins; ties go to the oldest
        track so the result does not depend on dict ordering.

        Returns True when a track was re-keyed to ``tid`` (counters, history, in-zone and
        stuck streaks and ``seen_count`` all preserved).
        """
        thr = float(self.cfg.fallback_iou_threshold)
        best_id: Optional[int] = None
        best_iou = 0.0
        for other, tr in self._tracks.items():
            if other in frame_ids:
                continue  # some detection of this frame owns that track
            iou = box_iou(det.xyxy, self._predicted_box(tr, t))
            if iou >= thr and iou > best_iou:
                best_id, best_iou = other, iou
        if best_id is None:
            return False
        tr = self._tracks.pop(best_id)
        tr.state.track_id = tid
        self._tracks[tid] = tr
        # Keep fallback ids clear of the id the detector has just introduced.
        self._next_fallback_id = max(self._next_fallback_id, tid + 1)
        return True

    def _observe(self, tid: int, det: Detection, zone: ZoneState, t: float) -> None:
        """Apply one detection (real or predicted) to track ``tid`` at time ``t``."""
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
            tr.last_detected_t = t
            tr.last_detected_box = box
            tr.conf = float(det.conf)
            tr.cls_id = int(det.cls_id)
            if det.cls_name != st.cls_name:
                # Class can flicker (car <-> truck). Re-classify so the flags stay honest.
                st.cls_name = det.cls_name
                tr.category = self._category(det.cls_name)
                st.is_vehicle = tr.category == CATEGORY_VEHICLE
                st.is_person = tr.category == CATEGORY_PERSON
                st.is_train = tr.category == CATEGORY_TRAIN
            st.last_seen = t

            x, y = bottom_center(box)
            tr.history.append((t, x, y))
            self._prune_history(tr, t)
            self._estimate_speed(tr)

            # Brand-new ids are ignored for zone logic: ByteTrack (and the IoU fallback)
            # can spawn a one-frame id on a flickering box, and we never want such an id
            # to start a dwell timer.
            if tr.seen_count >= int(self.cfg.min_track_age_frames):
                st.in_zone = point_in_polygon((x, y), zone.polygon)
            else:
                st.in_zone = False
            # Safety: "stationary" requires an actual speed estimate. Speed reads 0.0
            # before one exists, and 0.0 < threshold would otherwise start the stuck
            # clock on the very first frame of every new track.
            tr.stationary = tr.has_speed and st.speed < float(self.cfg.speed_threshold)

        # A predicted box only moves the drawn box and advances the clocks; zone
        # membership and speed are left exactly as the last real detection set them.
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

        if st.in_zone and tr.stationary:
            if tr.stuck_since is None:
                tr.stuck_since = t
            st.stuck_seconds = max(0.0, t - tr.stuck_since)
        else:
            tr.stuck_since = None
            st.stuck_seconds = 0.0

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
        cov(t, x) / var(t)) and report the magnitude of (vx, vy). The pixel-space
        velocity is always kept because ``predict_boxes`` needs it; when a ground-plane
        homography is configured the *reported* speed is the magnitude of the same fit
        done on the ground-plane points, in m/s.
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
