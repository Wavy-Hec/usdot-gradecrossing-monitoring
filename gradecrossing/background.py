"""Debris / obstruction detection on a *frozen* MOG2 background model.

Why this module exists
----------------------
The object detector only finds the classes it was trained on. A fallen branch, a
shed load, a shopping trolley, or a stalled car the detector misses at night are all
"not a detection" - and they are exactly the obstructions a crossing monitor must
catch. ``DebrisDetector`` is the backstop: it reports anything that differs from the
learned background *inside the zone* and is not explained by a detection box.

Why the background is frozen while monitoring
---------------------------------------------
The first prototype (``scripts/pipeline_helpers.py``) called ``mog2.apply(frame)``
on every frame. That uses OpenCV's automatic learning rate (``1/history``), and MOG2
promotes a new stationary mode to *background* as soon as its weight exceeds
``1 - backgroundRatio`` (10 % with OpenCV defaults). That takes only about
``ln(0.9) / ln(1 - rate)`` frames - 3 frames at rate 1/30, 21 frames at rate 0.005,
about 50 frames at the prototype's default ``history=500`` - so a stalled car faded
out of the debris mask within two seconds, at the very moment an alarm should have
been raised. With ``learningRate=0`` MOG2 performs classification only: no weight,
mean or variance is updated and no new mode is created, so a stationary obstruction
can never be absorbed. ``tests/test_background.py`` proves both behaviours.

Refresh while the zone is empty
-------------------------------
A frozen model accumulates lighting drift (dawn, dusk, clouds) as false foreground.
The contract therefore lets the model adapt at ``empty_learning_rate`` while the
*tracker* reports the zone empty for ``empty_seconds_before_refresh``. Be aware of
the caveat above: MOG2 absorbs a static mode after roughly ``0.105 / rate`` frames,
which for the default 0.005 is ~21 frames (< 1 s at 30 fps), shorter than the default
``persist_seconds``. Because the tracker cannot see debris, "zone empty" is exactly
the state in which undetected debris is present, so the refresh is additionally
suppressed whenever anything is a *candidate*: the rate falls back to
``monitoring_learning_rate`` while the debris alarm is triggered **and** while an
over-threshold streak is merely in progress (``_over_since is not None``). Refresh
therefore only ever nibbles at change that is still below ``min_area_fraction``,
which is what lighting drift looks like until it is absorbed - it can no longer eat
an obstruction that appeared while the tracker happened to report the zone empty.
The caller should still pass ``zone_empty_seconds=None`` while
``DebrisResult.persist_seconds > 0`` (the Monitor does), which is the same rule one
frame earlier.

Debris must be immobile
-----------------------
Persistence alone is not evidence of an obstruction. The foreground mask is
everything that differs from a frozen background and is not covered by a detection
box, so *any* object the detector does not fire on keeps the timer running while it
crosses the zone - a freight train under weights that have no "train" class, a crowd,
a vehicle class missing from the config. Anything slower than ``persist_seconds``
across the crossing would raise "debris N% of zone" and the alarm latch would hold it
after the object left. The persistence timer therefore also requires the candidate to
be *static*, on two independent counts:

* its **content** must be still - at most ``static_motion_fraction`` of the candidate
  pixels may change by more than ``static_motion_diff`` grey levels between consecutive
  frames (texture streaming past inside a stalled-looking outline);
* its **footprint** must be still - the pixels that have been candidate debris on every
  frame of the streak must themselves still cover ``min_area_fraction`` of the zone (an
  outline sliding across the crossing, even a uniform block with no texture to difference).

Either test failing restarts the streak at the current frame, so an object has to stay
inside its own footprint, unchanged, for the whole of ``persist_seconds``. A dropped
object satisfies both from its second frame on; a crossing object satisfies neither for
long. Residual limitation: something very large and uniform that is slower than
``persist_seconds`` across its *own width* (a slow freight train filling the zone) can
still look static to both tests - give the detector weights that contain the train /
wagon classes for sites with rail traffic, or disable debris for such clips.

Sudden global illumination change
---------------------------------
Floodlights switching on, a day/night camera flipping its IR-cut filter at dusk, or an
auto-exposure jump turn most of the zone into foreground in a single frame. Such a step
is not an obstruction (nothing that size can sit on a crossing unseen) and it is static,
so without a guard it would start the persistence timer, block the refresh that would
have absorbed it, and latch an ALARM for the rest of the process's life. When the
foreground covers at least ``max_area_fraction`` of the zone, as much of the frame
*outside* the zone changed too, and the tracker reports the zone empty, the model is
therefore re-warmed from scratch (``reset()`` plus a new warm-up) and
``last_illumination_event`` is set so the caller can emit a HEALTH event. All three
conditions are required because re-warming a *real* obstruction away would be a silent
failure, while failing to recognise a lighting step is only a loud one.
Debris detection is blind for ``warmup_frames`` frames afterwards, which is the price of
not being permanently blinded by a stale background; set ``max_area_fraction = 0`` to
disable the guard and treat such a step as debris again.

Known limitation: MOG2's shadow detector (``detect_shadows=True``) labels any pixel
that is a darker copy of the background with the same chromaticity as shadow (127),
and shadows are dropped. A dark, neutral-coloured object on grey asphalt (a tyre, a
black bin bag) can therefore be missed; set ``detect_shadows=False`` if that matters
more than false debris from real shadows. ``tests/test_background.py`` documents it.
"""
from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .config import DebrisConfig
from .geometry import dilate_box
from .types import DebrisResult

LOGGER = logging.getLogger(__name__)

# OpenCV convention: a negative learning rate means "choose automatically"
# (1 / min(2 * frames_seen, history)). Used for the warm-up phase only.
AUTO_LEARNING_RATE = -1.0
# MOG2 mask values: 0 background, 127 shadow (when detectShadows), 255 foreground.
FOREGROUND_VALUE = 255
SHADOW_VALUE = 127
# OpenCV's default backgroundRatio ("TB"). A mode becomes background once the modes
# ranked above it weigh less than this, i.e. once the new mode weighs > 1 - TB.
MOG2_BACKGROUND_RATIO = 0.9


class DebrisDetector:
    """Frozen-background debris detector for one zone (see module docstring).

    Lifecycle: the first ``cfg.warmup_frames`` calls to :meth:`update` learn the
    scene with OpenCV's automatic rate and report nothing. Afterwards every call
    classifies the frame against the model, updates the model only at the rate
    :meth:`_choose_learning_rate` selects (normally 0 = frozen), masks the
    foreground to the zone minus detection boxes, keeps connected components of a
    minimum size, and runs a timestamp-based persistence timer that sets
    ``DebrisResult.triggered`` once the debris area has stayed above
    ``min_area_fraction``, *without moving*, for ``persist_seconds``. A foreground
    that covers more than ``max_area_fraction`` of an empty zone is treated as a
    lighting step and re-warms the model instead (``last_illumination_event``).

    All thresholds are fractions of the zone area and seconds, so a config transfers
    between cameras and resolutions.
    """

    def __init__(self, cfg: DebrisConfig) -> None:
        self.cfg = cfg
        k = int(cfg.morph_kernel)
        # Ellipse rather than a square so that thin diagonal noise streaks (rain,
        # compression artefacts) are removed by the opening without rounding the
        # corners of real objects too aggressively. kernel <= 1 disables morphology.
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)) if k > 1 else None
        self._mog2: Optional[cv2.BackgroundSubtractorMOG2] = None
        self._frames_seen = 0
        # Timestamp at which the current over-threshold streak began (None = not over).
        self._over_since: Optional[float] = None
        # Whether the previous result was triggered; consulted by the refresh guard.
        self._triggered = False
        # Previous frame in grey, for the immobility test on candidate debris.
        self._prev_gray: Optional[np.ndarray] = None
        # Pixels that have been candidate debris on every frame of the current streak.
        self._streak_mask: Optional[np.ndarray] = None
        # Set on the update where a global illumination step forced a re-warm; the
        # caller (Monitor) turns it into a HEALTH event. None on every other update.
        self.last_illumination_event: Optional[Dict[str, object]] = None
        self.reset()

    # ------------------------------------------------------------------ state

    def reset(self) -> None:
        """Forget the background model and all timers (new video / seek / loop).

        MOG2 has no public "clear" in the Python API, so a fresh subtractor is
        created. The next ``warmup_frames`` calls learn the scene again.
        """
        cfg = self.cfg
        self._mog2 = cv2.createBackgroundSubtractorMOG2(
            history=int(cfg.history),
            varThreshold=float(cfg.var_threshold),
            detectShadows=bool(cfg.detect_shadows),
        )
        self._frames_seen = 0
        self._over_since = None
        self._triggered = False
        self._prev_gray = None
        self._streak_mask = None
        self.last_illumination_event = None

    @property
    def warmed_up(self) -> bool:
        """True once ``warmup_frames`` frames have been learned.

        Note the off-by-one that the contract implies: the result returned by the
        last warm-up call still says ``warmed_up=False`` (it carries no debris
        information), while this property already reports True because the *next*
        call will monitor.
        """
        return self._frames_seen >= int(self.cfg.warmup_frames)

    @property
    def frames_seen(self) -> int:
        """Total number of frames fed to :meth:`update` since the last reset."""
        return self._frames_seen

    @property
    def triggered(self) -> bool:
        """Whether the most recent result was triggered (persistence threshold met)."""
        return self._triggered

    def background_image(self) -> Optional[np.ndarray]:
        """The model's current background estimate (BGR), for overlays and debugging.

        Returns None before the first frame. Useful in tests to prove that a frozen
        model is bit-identical after monitoring N frames.
        """
        if self._frames_seen == 0:
            return None
        return self._mog2.getBackgroundImage()

    @staticmethod
    def absorption_frames(learning_rate: float,
                          background_ratio: float = MOG2_BACKGROUND_RATIO) -> float:
        """Frames until MOG2 promotes a new *static* mode to background at ``learning_rate``.

        Derivation: the original background mode's weight decays as ``(1 - rate)^n``
        and the new mode is classified as background once the modes ranked above it
        weigh less than ``background_ratio``. Solving ``(1 - rate)^n < background_ratio``
        gives ``n = ln(background_ratio) / ln(1 - rate)``. Pruning (``-rate * 0.05``
        per frame) shortens this slightly, so treat the value as an upper bound for
        "how long an undetected obstruction survives while the model adapts".
        Returns ``inf`` for rate 0 (frozen) and ~1 for rate >= 1. A negative rate
        (OpenCV automatic) is not resolvable here because the effective rate depends
        on the frame count; pass ``1 / history`` for the steady-state answer.
        """
        if learning_rate <= 0.0:
            return math.inf
        if learning_rate >= 1.0:
            return 1.0
        return math.log(background_ratio) / math.log(1.0 - learning_rate)

    # ----------------------------------------------------------------- update

    def update(self, frame_bgr: np.ndarray, zone_mask: np.ndarray,
               exclusion_boxes: List[np.ndarray], zone_empty_seconds: Optional[float],
               timestamp: float) -> DebrisResult:
        """Classify one frame and advance the persistence timer.

        Args:
            frame_bgr: HxWx3 (or HxW grey) uint8 frame at processing resolution.
            zone_mask: HxW mask of the zone (non-zero inside), from ``ZoneState.mask``.
            exclusion_boxes: xyxy boxes of *every* detection (any class); each is
                dilated by ``exclusion_dilate_px`` and removed from the foreground so
                that a moving or tracked object never counts as debris.
            zone_empty_seconds: how long the tracker has seen the zone empty
                (None = unknown / occupied). Drives the slow background refresh and is
                the "nothing is really there" evidence the illumination guard needs.
            timestamp: seconds (monotonic within a run). The persistence timer is
                timestamp based so that frame skipping or variable fps does not change
                how long debris must persist before it triggers.

        Returns:
            :class:`DebrisResult`; during warm-up ``warmed_up=False`` and area 0. When a
            global illumination step is detected the model is re-warmed and a warm-up
            result is returned with :attr:`last_illumination_event` set for this call.
        """
        frame_bgr = np.asarray(frame_bgr)
        if frame_bgr.ndim not in (2, 3):
            raise ValueError(f"frame must be HxW or HxWxC, got shape {frame_bgr.shape}")
        h, w = frame_bgr.shape[:2]
        zone_mask = np.asarray(zone_mask)
        if zone_mask.ndim != 2 or zone_mask.shape != (h, w):
            raise ValueError(
                f"zone_mask shape {zone_mask.shape} does not match frame shape {(h, w)}")
        if zone_mask.dtype != np.uint8:
            # cv2 mask arguments must be 8-bit; any non-zero value counts as "inside".
            zone_mask = zone_mask.astype(np.uint8)

        cfg = self.cfg
        self.last_illumination_event = None
        gray = self._grey(frame_bgr)

        if not self.warmed_up:
            # Learn the scene with OpenCV's automatic rate. Nothing that appears
            # during warm-up can be reported, so the caller should start the monitor
            # on a clear crossing (or keep warmup_frames short and rely on refresh).
            self._mog2.apply(frame_bgr)
            self._frames_seen += 1
            self._over_since = None
            self._streak_mask = None
            self._triggered = False
            self._prev_gray = gray
            return self._empty_result((h, w), AUTO_LEARNING_RATE, warmed_up=False)

        lr = self._choose_learning_rate(zone_empty_seconds)
        # NOTE (OpenCV quirk): on the very first apply() of a fresh model the rate is
        # forced to automatic regardless of the argument. With warmup_frames >= 1 this
        # never affects monitoring; with warmup_frames == 0 the first frame is learned.
        raw = self._mog2.apply(frame_bgr, learningRate=lr)
        self._frames_seen += 1

        # Foreground = 255 only. Shadow pixels (127) are dropped: a real shadow that
        # sweeps across the tracks must not count as an obstruction.
        _, fg = cv2.threshold(raw, FOREGROUND_VALUE - 1, 255, cv2.THRESH_BINARY)

        # Opening removes speckle (sensor noise, rain, compression), closing fills
        # holes inside an object so it forms one component with one area.
        if self._kernel is not None:
            fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, self._kernel)
            fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, self._kernel)

        # Foreground over the whole frame, before the zone is applied: the illumination
        # guard uses how much of the scene *outside* the zone changed too (see
        # _is_illumination_step).
        outside_fraction = self._outside_zone_fraction(fg, zone_mask)

        # Restrict to the zone and remove every detection box (dilated). The dilation
        # covers box jitter and the halo MOG2 draws around a moving object.
        keep = zone_mask
        if exclusion_boxes:
            keep = zone_mask.copy()
            for box in exclusion_boxes:
                x1, y1, x2, y2 = dilate_box(box, float(cfg.exclusion_dilate_px), (h, w))
                # floor/ceil so the exclusion is never smaller than the dilated box.
                cv2.rectangle(keep, (int(math.floor(x1)), int(math.floor(y1))),
                              (int(math.ceil(x2)), int(math.ceil(y2))), 0, thickness=-1)
        fg = cv2.bitwise_and(fg, fg, mask=keep)

        zone_area = int(cv2.countNonZero(zone_mask))
        mask, blobs, area_px = self._filter_components(fg, zone_area)
        area_fraction = (area_px / float(zone_area)) if zone_area > 0 else 0.0

        if self._is_illumination_step(area_fraction, outside_fraction, zone_empty_seconds):
            # Not an obstruction: re-warm instead of latching an alarm forever.
            percent = 100.0 * area_fraction
            LOGGER.warning(
                "background invalid: %.0f%% of the zone turned to foreground at once while the "
                "zone was reported empty; re-warming the background model (%d frames)",
                percent, int(cfg.warmup_frames))
            self.reset()
            self._mog2.apply(frame_bgr)  # start the new warm-up on this frame
            self._frames_seen = 1
            self._prev_gray = gray
            self.last_illumination_event = {
                "reason": "illumination step",
                "area_fraction": float(area_fraction),
                "outside_zone_fraction": (None if outside_fraction is None
                                          else float(outside_fraction)),
                "max_area_fraction": float(cfg.max_area_fraction),
                "timestamp": float(timestamp),
                "warmup_frames": int(cfg.warmup_frames),
            }
            return self._empty_result((h, w), AUTO_LEARNING_RATE, warmed_up=False)

        moving_fraction = self._moving_fraction(gray, mask, area_px)
        self._prev_gray = gray
        persist = self._advance_timer(mask, area_px, area_fraction, zone_area, timestamp,
                                      moving_fraction)
        triggered = persist >= float(cfg.persist_seconds)
        self._triggered = triggered

        return DebrisResult(
            mask=mask,
            area_px=int(area_px),
            area_fraction=float(area_fraction),
            blobs=blobs,
            persist_seconds=float(persist),
            triggered=bool(triggered),
            learning_rate_used=float(lr),
            warmed_up=True,
        )

    # ---------------------------------------------------------------- helpers

    @staticmethod
    def _grey(frame: np.ndarray) -> np.ndarray:
        """Single-channel uint8 copy of the frame, used only by the immobility test."""
        if frame.ndim == 3:
            if frame.shape[2] == 3:
                return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if frame.shape[2] == 4:
                return cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
            frame = frame[:, :, 0]
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(frame)

    def _choose_learning_rate(self, zone_empty_seconds: Optional[float]) -> float:
        """Learning rate for this frame: frozen unless a slow refresh is allowed.

        Contract: use ``empty_learning_rate`` when refresh is enabled
        (``empty_seconds_before_refresh > 0``) and the tracker has reported the zone
        empty for at least that long; otherwise ``monitoring_learning_rate`` (0).

        Safety guard on top of the contract: **never refresh while anything is a
        candidate obstruction** - neither while the debris alarm is triggered
        (``_triggered``) nor while an over-threshold streak is merely in progress
        (``_over_since is not None``). The tracker cannot see debris, so "zone empty" is
        exactly the state in which undetected debris - or a stalled car the detector lost
        - is present. Adapting then would erase the evidence in about
        ``absorption_frames(empty_learning_rate)`` frames (~21 at the default 0.005, well
        under the default 3 s ``persist_seconds``), so the obstruction would vanish before
        it could ever trigger and a triggered alarm would silently fade. Alarms must
        latch, not fade, and a candidate must be allowed to become one.

        What this leaves the refresh path: change that is still *below*
        ``min_area_fraction``, which is exactly what gradual lighting drift looks like on
        its way up. It is nibbled away while it is small and therefore never accumulates.
        A change that arrives all at once and is large is handled by the illumination
        guard (see :meth:`_is_illumination_step`), not by the refresh.
        """
        cfg = self.cfg
        refresh = (
            float(cfg.empty_seconds_before_refresh) > 0.0
            and zone_empty_seconds is not None
            and float(zone_empty_seconds) >= float(cfg.empty_seconds_before_refresh)
        )
        if refresh and (self._triggered or self._over_since is not None):
            refresh = False
        return float(cfg.empty_learning_rate) if refresh else float(cfg.monitoring_learning_rate)

    @staticmethod
    def _outside_zone_fraction(fg_full: np.ndarray, zone_mask: np.ndarray) -> Optional[float]:
        """Share of the pixels *outside* the zone that are foreground this frame.

        Corroborating evidence for the illumination guard: a lighting change is a property
        of the camera or the scene as a whole, so it lights up the road, the ballast and
        the sky as well, while an obstruction - however big - is confined to the crossing.
        Returns None when the zone covers the whole frame (nothing to corroborate with), so
        the guard then rests on the zone fraction and the tracker alone.
        """
        outside_total = int(zone_mask.size) - int(cv2.countNonZero(zone_mask))
        if outside_total <= 0:
            return None
        outside_fg = int(cv2.countNonZero(cv2.bitwise_and(
            fg_full, fg_full, mask=cv2.bitwise_not(zone_mask))))
        return outside_fg / float(outside_total)

    def _is_illumination_step(self, area_fraction: float, outside_fraction: Optional[float],
                              zone_empty_seconds: Optional[float]) -> bool:
        """True when this frame's foreground is too big to be anything but a lighting change.

        Three conditions, all required:

        * the foreground covers at least ``max_area_fraction`` of the zone - no obstruction
          that big can be on a crossing without the detector seeing *something*;
        * at least as much of the frame *outside* the zone changed as well (when there is
          any outside) - a floodlight, an IR-cut filter switch or an auto-exposure jump
          changes the whole image, while even a lorry parked across a tightly drawn zone
          leaves the rest of the frame alone;
        * the tracker reports the zone empty (``zone_empty_seconds is not None``), i.e. the
          independent, object-based view of the scene agrees that nothing is there.

        The conditions are deliberately hard to satisfy because the two failure modes are
        not symmetric. Failing to recognise a lighting step costs a false alarm that cannot
        clear until the monitor restarts - bad, but loud and safe. Wrongly recognising a
        real obstruction as one would re-warm it into the background - silent, and exactly
        the failure this package exists to remove. So the guard only fires when the zone,
        the rest of the frame and the tracker all agree.

        Returning True makes :meth:`update` re-warm the model, which costs ``warmup_frames``
        frames of debris blindness and is reported in a HEALTH event.
        """
        cap = float(self.cfg.max_area_fraction)
        if cap <= 0.0 or zone_empty_seconds is None:
            return False
        corroborated = outside_fraction is None or float(outside_fraction) >= cap
        return float(area_fraction) >= cap and corroborated

    def _moving_fraction(self, gray: np.ndarray, mask: np.ndarray, area_px: int) -> float:
        """Fraction of the candidate debris pixels that changed since the previous frame.

        Debris is static by definition, so this is how a candidate proves it is not just
        an object the detector failed to name crossing the zone. Frame differencing inside
        the blob is used rather than blob-centroid tracking because it also catches an
        object whose outline barely moves while its texture streams past (a train filling
        the zone), and it costs one ``absdiff`` per frame.

        Returns 0.0 when there is nothing to test (no candidate, no previous frame, or the
        check disabled with ``static_motion_diff <= 0``) and 1.0 when the frame geometry
        changed, which restarts the streak rather than trusting a stale comparison.
        """
        thr = float(self.cfg.static_motion_diff)
        if area_px <= 0 or thr <= 0.0 or self._prev_gray is None:
            return 0.0
        if self._prev_gray.shape != gray.shape:
            return 1.0
        diff = cv2.absdiff(gray, self._prev_gray)
        _, moved = cv2.threshold(diff, thr, 255, cv2.THRESH_BINARY)
        moved_in_blob = int(cv2.countNonZero(cv2.bitwise_and(moved, moved, mask=mask)))
        return moved_in_blob / float(area_px)

    def _filter_components(self, fg: np.ndarray, zone_area: int
                           ) -> Tuple[np.ndarray, List[Tuple[int, int, int, int, int]], int]:
        """Keep connected components with ``area >= min_blob_area_fraction * zone_area``.

        Returns ``(mask, blobs, area_px)`` where blobs are ``(x, y, w, h, area)``.
        The minimum blob size is a fraction of the zone so a config is resolution
        independent; it removes leftover speckle without hiding a real object.
        """
        n, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8, ltype=cv2.CV_32S)
        if n <= 1:
            return fg, [], 0
        min_px = float(self.cfg.min_blob_area_fraction) * float(zone_area)
        areas = stats[1:, cv2.CC_STAT_AREA]
        kept = np.flatnonzero(areas >= min_px) + 1  # +1: label 0 is background
        if kept.size == 0:
            return np.zeros_like(fg), [], 0
        blobs: List[Tuple[int, int, int, int, int]] = [
            (int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP]),
             int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT]),
             int(stats[i, cv2.CC_STAT_AREA]))
            for i in kept
        ]
        area_px = int(stats[kept, cv2.CC_STAT_AREA].sum())
        if kept.size == n - 1:
            mask = fg  # nothing dropped: avoid the relabel pass
        else:
            lut = np.zeros(n, dtype=np.uint8)
            lut[kept] = 255
            mask = lut[labels]
        return mask, blobs, area_px

    def _advance_timer(self, mask: np.ndarray, area_px: int, area_fraction: float,
                       zone_area: int, timestamp: float, moving_fraction: float = 0.0) -> float:
        """Continuous seconds the debris area has been >= ``min_area_fraction`` *and static*.

        Timestamp based (not frame counts) so skipped frames, variable fps and slow
        hardware do not change the real-world persistence requirement. The streak
        restarts (``_over_since`` moves to the current frame) in four cases:

        1. nothing is over the area threshold - the streak ends outright;
        2. time went backwards (a looped file that was not reset);
        3. the candidate's *content* is changing - more than ``static_motion_fraction`` of
           its pixels differ from the previous frame (see :meth:`_moving_fraction`);
        4. the candidate's *footprint* has drifted - the pixels that have been foreground
           on **every** frame of the streak (``_streak_mask``, the running intersection)
           no longer cover ``min_area_fraction`` of the zone.

        (3) and (4) together are the immobility criterion. Debris is static by definition;
        without them any object the detector does not fire on raises an alarm merely by
        taking longer than ``persist_seconds`` to cross the zone. (3) catches texture
        streaming past inside a stalled-looking outline, (4) catches an outline sliding
        across the crossing even when it is a uniform block with no internal texture to
        difference. An object therefore has to stay inside its own footprint, unchanged,
        for the whole of ``persist_seconds`` before it can be called debris.

        Restarting at the current timestamp rather than clearing the streak keeps the
        timing of a dropped object unchanged: the frame on which it appears is a
        "moving" frame (it was not there before), and that frame is also the frame on
        which a streak would have started, so persistence still counts from first sight.
        """
        over = area_px > 0 and area_fraction >= float(self.cfg.min_area_fraction)
        if not over:
            self._over_since = None
            self._streak_mask = None
            return 0.0

        min_px = float(self.cfg.min_area_fraction) * float(zone_area)
        restart = (self._over_since is None or timestamp < self._over_since
                   or float(moving_fraction) > float(self.cfg.static_motion_fraction)
                   or self._streak_mask is None or self._streak_mask.shape != mask.shape)
        if not restart:
            core = cv2.bitwise_and(self._streak_mask, mask)
            if int(cv2.countNonZero(core)) < min_px:
                restart = True  # the continuously-present part is too small: it moved
            else:
                self._streak_mask = core
        if restart:
            self._over_since = float(timestamp)
            self._streak_mask = mask.copy()
        return float(timestamp) - self._over_since

    @staticmethod
    def _empty_result(shape_hw: Tuple[int, int], learning_rate: float, warmed_up: bool) -> DebrisResult:
        """A result with no debris (warm-up frames)."""
        return DebrisResult(
            mask=np.zeros(shape_hw, dtype=np.uint8),
            area_px=0,
            area_fraction=0.0,
            blobs=[],
            persist_seconds=0.0,
            triggered=False,
            learning_rate_used=float(learning_rate),
            warmed_up=warmed_up,
        )
