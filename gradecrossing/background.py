"""Debris / obstruction detection on a selectively updated MOG2 background model.

Why this module exists
----------------------
The object detector only finds the classes it was trained on. A fallen branch, a
shed load, a shopping trolley, a cow, or a stalled car the detector misses at night are
all "not a detection" - and they are exactly the obstructions a crossing monitor must
catch. ``DebrisDetector`` is the backstop: it reports anything that differs from the
learned background *inside the zone*, is not explained by a detection box, and stays
put for ``persist_seconds``.

The background never learns what it has to report
-------------------------------------------------
The first prototype (``scripts/pipeline_helpers.py``) called ``mog2.apply(frame)`` on
every frame. That uses OpenCV's automatic learning rate (``1/history``), and MOG2
promotes a new stationary mode to *background* as soon as its weight exceeds
``1 - backgroundRatio`` (10 % with OpenCV defaults). That takes only about
``ln(0.9) / ln(1 - rate)`` frames - 3 frames at rate 1/30, 21 frames at rate 0.005 - so a
stalled car faded out of the debris mask within seconds. Every frame is therefore
*classified* with ``learningRate=0`` (no weight, mean or variance changes, no new mode),
and the model is taught only through two controlled paths:

* **Warm-up** (``warmup_seconds``, 2 s by default). The scene is learned with the
  automatic rate, but never under a (dilated) detection box: those pixels are pasted
  from the previous warm-up frame before ``apply()``. A pixel that was box-free for less
  than half of the warm-up has no trustworthy background and is **unknown**. Unknown
  pixels are excluded from every debris decision (``DebrisResult.unknown_fraction``)
  until they have been seen box-free and classified background for ``learn_seconds``.
  A car waiting at the crossing when the monitor starts therefore leaves no ghost when it
  drives off (clip05's warm-up pickup used to latch a debris ALARM twice).
* **Selective update** (time constant ``update_time_constant_seconds``). Every
  ``update_interval_seconds`` one learning ``apply()`` runs on a composite frame: zone
  pixels classified background, outside the dilated detection boxes and outside every
  foreground / candidate pixel in or near the zone (plus a margin) show the current frame,
  as do the other box-free pixels outside the zone (they decide nothing and keep the gain
  reference fresh); every other pixel is pasted from the model's own background image,
  which reinforces what is already there. Slow lighting drift is followed even on a
  crossing that is never empty (clip03 and clip06 never had 5 empty seconds, so the old
  empty-zone refresh never ran), while no foreground pixel in the zone can ever be
  absorbed. Pasting the background exactly would feed a zero residual and shrink a covered
  pixel's variance towards MOG2's ``varMin`` (it would come back hypersensitive); each
  pasted pixel is therefore dithered by +-its own residual standard deviation (estimated
  over the warm-up, then on every learning apply where it showed background), so its
  variance stays where it was. No global variance floor is imposed: a floor measured on the
  whole modelled region, motion outside the zone included, made every zone pixel less
  sensitive (``test_busy_motion_outside_the_zone_does_not_blind_the_zone``).

Registration and the modelled region
------------------------------------
The background lives in the geometry of the frame it was learned in. When the caller
passes ``to_reference`` (current pixels -> reference pixels, e.g. the inverse of the
zone stabiliser's homography) every frame is warped into that geometry before
classification and before the immobility test, the exclusion boxes are transformed with
it, and the zone mask is warped back to the reference. Pixels that fall outside the
warped frame are unknown. A camera bump that the stabiliser tracks no longer turns every
edge of the scene into debris (clip02 at 34.4 s). While ``registration_ok`` is False the
timers are held: no new candidates, no reset, no learning.

Only a fixed region around the zone is modelled (its bounding box plus a margin of at
least 32 px, more for large zones): pixels far from the crossing decide nothing, and a
thin band zone then costs a quarter of a full frame. Frames wider than ``process_width``
are downscaled first; results are mapped back to the caller's pixels. The statistics
"outside the zone" (gain reference, illumination guard) are therefore taken in that region,
while the size bar below refers to the whole processing frame.

Illumination
------------
Before classification the per-channel gain *current / background* is estimated as the
median ratio over background pixels outside the zone and outside the boxes (an
obstruction confined to the crossing cannot bias it) and divided out, so exposure ramps,
the sun coming out and dusk do not become debris. Pixels whose background the gain
would clip (> 250 or < 5) are unknown for that frame. Local lighting the global gain
does not cover (a cloud edge over a pole's shadow, a sun patch) is caught per candidate:
a component whose grey values correlate with the background's (NCC >= 0.7) on a textured
background is the scene under different light - an object replaces the texture, light
only scales it - and is not a candidate. If a gain leaves
``[1/rewarm_gain, rewarm_gain]`` for a second, or foreground covers more than
``max_area_fraction`` of the zone *and* of the modelled region outside it (an IR-cut filter switch,
floodlights: change compensation cannot explain), the model is re-warmed and
``last_illumination_event`` is set for a HEALTH event - never while debris is triggered,
because re-warming a real obstruction away would be a silent failure.

Candidates, occlusion and persistence
-------------------------------------
Persistence is tracked per pixel. A zone pixel that is foreground, observable (learned,
in frame, not clipped, not under a box) and static accumulates *seen-static seconds*;
it is reset the moment it is seen as background again. A candidate pixel covered by a
box (or otherwise unobservable) is **unknown, not gone**: it keeps its age, without
counting the covered time, for at most ``occlusion_bridge_seconds``. Traffic driving over
an obstruction, or a detector that boxes it on one frame in forty, no longer restarts the
clock (the M2 / PORT-03 failures). Debris triggers once ``k`` candidate pixels have been
seen static for ``persist_seconds``, where ``k = min(min_area_fraction * zone area,
min_object_frame_fraction * frame area)``, both in processing pixels and the frame being
the whole processing frame (not the modelled region): the zone share alone missed a 30x24
px object on a large zone. ``DebrisResult.persist_seconds`` is the age of the k-th oldest
candidate pixel.

Debris must be immobile
-----------------------
Anything the detector does not fire on keeps differing from the background while it
crosses the zone. Two things keep it from timing:

* per-pixel ages - a pixel only ages while it stays foreground, so an outline sliding
  across the zone never keeps the same pixels for ``persist_seconds`` (a uniform block,
  the case frame differencing cannot see);
* content - a candidate component restarts when more than ``static_motion_fraction`` of
  its pixels changed by more than ``static_motion_diff`` grey levels against a frame
  ``static_motion_lag_seconds`` old, counted only where the pixel was already foreground
  then (an arrival is not motion) and beyond a +-1 px tolerance (the comparison frame is
  eroded / dilated by one pixel), so 0.3 px of camera jitter on a textured car does not
  read as motion. Motion has to accumulate over the lag, so a slow textured train still
  differs from itself while frame-to-frame jitter does not; pixels that changed inside an
  otherwise still component (hazard lights blinking on a stalled car) only reset
  themselves.

A uniform object much longer than the zone (a featureless train) cannot be told from a
parked one by any image test: the monitor sets ``hold`` while a train detection overlaps
the zone, and train / vehicle / person boxes are exclusions. While held the frame is still
classified and the motion reference is refreshed (so the train does not look like a fresh,
still arrival when the hold ends), but no candidate starts or ages, nothing is reset and
nothing is learned, until ``hold_linger_seconds`` after the last such frame.

The detector does not label a passing train on every frame (clip03: gaps of up to 3.5 s
between 'train' labels). In a gap longer than the linger the train is an ordinary candidate,
and a uniform freight car passes the immobility test; frozen by each hold rather than reset,
its age added up over the gaps of one passage (2.0 s of the 3.0 s on clip03, and a debris
ALARM with a smaller size bar). A hold that starts again less than ``persist_seconds`` after
the previous one ended therefore restarts the candidates that started in between: they are
most likely the same train. Candidates from before the previous hold (an obstruction that
was there before the train) keep their age, and nothing restarts while debris is triggered.

Ghosts
------
Anything the warm-up learned that is not a detection (a burned-in caption, an undetected
object) becomes a "ghost" when it leaves: the road it uncovers differs from the model. A
ghost has its outline only in the *model*; a real object has it in the *frame*. The
colour step across the outline of the changed region (every pixel that differs from the
model, shadow-labelled ones included, closed) - mean of a thin inner band minus a thin
outer band of pixels that match the model - is measured in both images; a candidate whose
frame step stays below a third of a clear model step for ``ghost_heal_seconds`` is made
unknown again and re-learned through the selective update. Image evidence alone is
ambiguous: a dark object lying across a bright rail on dark ballast differs from the model
only over the rail, so its changed region has the rail's outline - a step in the model and
none in the frame, exactly like a rail-shaped ghost (it healed real objects on clip04-06).
A heal therefore also needs *young* model content: at least three quarters of the
candidate's pixels must have been seen as background for less than
``ghost_heal_max_seen_seconds`` before they changed. Ghosts come from content the warm-up
(or an unknown pixel) learned shortly before it left - a caption, something moving off right
after start-up - while an obstruction arrives on road the model has watched for longer. A
heal never runs while debris is triggered (it would clear the ALARM silently). A shed load
left by a truck that drives off has its own outline in the frame and is never healed this
way (F04: no "re-learn where a box was" rule), and a low-contrast object never qualifies.
Content that stood for longer before it left (an undetected object parked at start-up) is
not healed: its ghost raises a debris ALARM, the fail-safe direction.

Known limitation: MOG2's shadow detector (``detect_shadows=True``) labels any pixel that is
a darker copy of the background with the same chromaticity as shadow (127), and shadows
are dropped. A dark, neutral-coloured object on grey asphalt (a tyre, a black bin bag) can
therefore be missed; set ``detect_shadows=False`` if that matters more than false debris
from real shadows. ``tests/test_background.py`` documents it.
"""
from __future__ import annotations

import logging
import math
from collections import deque
from typing import Deque, Dict, List, Optional, Sequence, Tuple

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
# A seconds-based warm-up still needs this many frames for a usable MOG2 model.
MIN_WARMUP_FRAMES = 5
# Timers advance by at most this much per frame: a stalled stream or a long gap between
# frames is not evidence that a candidate stayed put (supports sources down to 0.5 fps).
MAX_STEP_SECONDS = 2.0
# Gain estimation: sample grid step, minimum samples, background range used, dead band.
GAIN_SAMPLE_STEP = 6
GAIN_MIN_SAMPLES = 200
GAIN_BG_RANGE = (16, 240)
GAIN_DEADBAND = 0.02
# Under a gain g > 1 a background value >= CLIP_HIGH / g saturates (and <= CLIP_LOW / g
# crushes under g < 1): those pixels cannot be compared and are unknown for the frame.
CLIP_HIGH = 250.0
CLIP_LOW = 5.0
# A gain beyond rewarm_gain must persist this long before the model is re-warmed.
REWARM_GAIN_SECONDS = 1.0
# Foreground in or near the zone (within PROTECT_BAND_PX of it, so the part of an object that
# sticks out of the zone is not learned either) and every candidate are protected from the
# selective update with this margin (pixels at processing resolution), so the soft rim of an
# object is not learned.
PROTECT_MARGIN_PX = 3
PROTECT_BAND_PX = 12
# Ghost test: the colour step across the changed region's outline (inner band mean minus outer
# band mean) in the frame is below this share of the step in the model, and the model's step is
# at least GHOST_MIN_CONTRAST (Euclidean BGR); bands are GHOST_RING_PX wide. The outer band only
# counts pixels classified background, and must be at least GHOST_MIN_OUTER_SHARE of the ring
# (otherwise the outline is not known well enough to decide). At most GHOST_MAX_OLD_SHARE of the
# candidate's pixels may have model content seen for ghost_heal_max_seen_seconds or longer.
GHOST_CONTRAST_RATIO = 0.35
GHOST_MIN_CONTRAST = 20.0
GHOST_RING_PX = 2
GHOST_MIN_OUTER_SHARE = 0.5
GHOST_MAX_OLD_SHARE = 0.25
GHOST_CHECK_SECONDS = 0.25  # cadence of the ghost test
# Relit test: a component whose grey values correlate with the background's (NCC) on a
# textured background is the scene under different light, not an object.
RELIT_MIN_NCC = 0.7
RELIT_MIN_STD = 8.0
RELIT_MIN_PIXELS = 30
# trusted recovers only once the unknown share is back below this share of the limit.
TRUST_RECOVERY = 0.8
# The zone region of interest is padded and snapped to this grid so it stays stable while
# the warped zone mask jitters by a pixel.
ROI_PAD_PX = 8
ROI_SNAP_PX = 16
# Minimum margin (processing pixels) of the modelled region around the zone's bounding box.
REGION_MARGIN_PX = 32


def _u8(mask: np.ndarray) -> np.ndarray:
    """0/255 uint8 copy of any mask (bool, 0/1 or 0/255)."""
    if mask.dtype == np.bool_:
        return mask.astype(np.uint8) * 255
    if mask.dtype != np.uint8:
        return (np.asarray(mask) != 0).astype(np.uint8) * 255
    return cv2.compare(mask, 0, cv2.CMP_GT)


class DebrisDetector:
    """Background-model debris detector for one zone (see module docstring).

    Lifecycle: the first ``warmup_seconds`` (or ``warmup_frames``) of calls to
    :meth:`update` learn the scene, never under detection boxes, and report nothing.
    Afterwards every call registers the frame (when ``to_reference`` is given), divides
    out the global gain, classifies the frame against the frozen model, masks the
    foreground to the observable part of the zone, keeps connected components of a
    minimum size, advances per-pixel persistence for static candidates, and every
    ``update_interval_seconds`` teaches the model the pixels that are safe to learn.
    ``DebrisResult.triggered`` is set once enough candidate pixels have been seen static
    for ``persist_seconds``.

    Areas are fractions of the zone or frame and times are seconds, so those settings
    transfer between frame rates and resolutions. Some knobs are pixels and do not:
    ``exclusion_dilate_px`` (pixels of the frame passed in, i.e. after
    ``video.resize_width``), ``morph_kernel`` (processing pixels, after ``process_width``)
    and ``process_width`` itself here, and elsewhere the zone stabiliser's ``min_distance``,
    ``block_size``, ``lk_win_size``, ``ransac_reproj_threshold``, ``max_step_px``,
    ``deadzone_px`` and ``reanchor_max_drift_px`` and the tracker's ``stuck.speed_threshold``
    (px/s). Re-check them when the camera or the processing resolution changes.
    """

    def __init__(self, cfg: DebrisConfig) -> None:
        self.cfg = cfg
        k = int(cfg.morph_kernel)
        # Ellipse rather than a square so that thin diagonal noise streaks (rain,
        # compression artefacts) are removed by the opening without rounding the
        # corners of real objects too aggressively. kernel <= 1 disables morphology.
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)) if k > 1 else None
        self._tolerance_kernel = np.ones((3, 3), np.uint8)  # +-1 px for the motion test
        m = 2 * PROTECT_MARGIN_PX + 1
        self._protect_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (m, m))
        b = 2 * PROTECT_BAND_PX + 1
        self._band_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (b, b))
        r = 2 * GHOST_RING_PX + 1
        self._ring_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r, r))
        self._mog2: Optional[cv2.BackgroundSubtractorMOG2] = None
        # Set on the update where the model had to be re-warmed (illumination the gain
        # cannot explain); the caller (Monitor) turns it into a HEALTH event.
        self.last_illumination_event: Optional[Dict[str, object]] = None
        self.reset()

    # ------------------------------------------------------------------ state

    def reset(self) -> None:
        """Forget the background model and all timers (new video / seek / loop).

        MOG2 has no public "clear" in the Python API, so a fresh subtractor is
        created. The next warm-up learns the scene again.
        """
        cfg = self.cfg
        self._mog2 = cv2.createBackgroundSubtractorMOG2(
            history=int(cfg.history),
            varThreshold=float(cfg.var_threshold),
            detectShadows=bool(cfg.detect_shadows),
        )
        self._shape: Optional[Tuple[int, ...]] = None  # internal (h, w[, c]) the model was built for
        self._region: Optional[Tuple[int, int, int, int]] = None  # modelled region (y0, y1, x0, x1)
        self._full_hw: Optional[Tuple[int, int]] = None  # internal frame size the region lies in
        self._frames_seen = 0
        self._warm_frames = 0
        self._warm_elapsed = 0.0
        self._warmed = False
        self._t_prev: Optional[float] = None
        self._composite: Optional[np.ndarray] = None  # warm-up: last frame fed to MOG2
        self._free_count: Optional[np.ndarray] = None  # warm-up: frames each pixel was box-free
        self._acc1: Optional[np.ndarray] = None  # warm-up: sum of the frames fed to MOG2 (float32)
        self._acc2: Optional[np.ndarray] = None  # warm-up: sum of their squares
        self._learned: Optional[np.ndarray] = None  # 255 where the model knows the background
        self._pending: Optional[np.ndarray] = None  # seconds an unknown pixel has looked learned
        self._age: Optional[np.ndarray] = None  # seen-static seconds of each candidate pixel
        self._born: Optional[np.ndarray] = None  # time each candidate pixel's age (re)started
        self._cand: Optional[np.ndarray] = None  # bool, candidate pixels (visible or bridged)
        self._covered: Optional[np.ndarray] = None  # seconds a candidate has been unobservable
        self._ghost: Optional[np.ndarray] = None  # seconds a candidate pixel has looked like a ghost
        self._seen_bg: Optional[np.ndarray] = None  # seconds the model content was seen as background
        self._resvar: Optional[np.ndarray] = None  # residual variance per pixel (sum over channels)
        self._dither: Optional[np.ndarray] = None  # +-1 checkerboard for the pasted-pixel dither
        self._dither_phase = 1.0
        self._bg: Optional[np.ndarray] = None  # cached background image (changes only on learning)
        self._bg_gray: Optional[np.ndarray] = None  # its grey version (relit test)
        self._prev_raw: Optional[np.ndarray] = None  # last classification (gain reference)
        self._history: Deque[Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = deque()
        self._last_update_t: Optional[float] = None
        self._gain = np.ones(3, dtype=np.float64)
        self._gain_bad_since: Optional[float] = None
        self._hold_until = -math.inf
        self._was_held = False  # the previous monitoring frame was held (train)
        self._hold_ended_at: Optional[float] = None  # first unheld frame after the last hold
        self._triggered = False
        self._held_persist = 0.0  # persistence reported (frozen) while held
        self._trusted = True
        self._roi: Optional[Tuple[int, int, int, int]] = None
        self._zone_cache: Optional[Tuple[tuple, np.ndarray]] = None  # (signature, reference zone)
        self._gain_idx: Optional[np.ndarray] = None  # flat indices of the gain sample grid
        self._ghost_due = -math.inf  # next time the ghost test runs
        self._ghost_last_t: Optional[float] = None
        self._last_result: Optional[DebrisResult] = None
        self.last_illumination_event = None

    @property
    def warmed_up(self) -> bool:
        """True once the warm-up has been learned.

        Note the off-by-one that the contract implies: the result returned by the
        last warm-up call still says ``warmed_up=False`` (it carries no debris
        information), while this property already reports True because the *next*
        call will monitor.
        """
        if self._warmed:
            return True
        frames = self.cfg.warmup_frames
        return frames is not None and int(frames) <= 0

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

        It is in the processing geometry of the model (reference geometry when frames
        are registered, ``process_width`` when they are downscaled) and black outside the
        modelled region around the zone. Returns None before the first frame. Useful in
        tests to prove that the model is bit-identical after monitoring N frames with the
        selective update disabled.
        """
        if self._frames_seen == 0:
            return None
        bg = self._mog2.getBackgroundImage()
        if self._full_hw is None or self._region is None or bg.shape[:2] == self._full_hw:
            return bg
        y0, y1, x0, x1 = self._region
        out = np.zeros(tuple(self._full_hw) + bg.shape[2:], dtype=bg.dtype)
        out[y0:y1, x0:x1] = bg
        return out

    @staticmethod
    def absorption_frames(learning_rate: float,
                          background_ratio: float = MOG2_BACKGROUND_RATIO) -> float:
        """Frames until MOG2 promotes a new *static* mode to background at ``learning_rate``.

        Derivation: the original background mode's weight decays as ``(1 - rate)^n``
        and the new mode is classified as background once the modes ranked above it
        weigh less than ``background_ratio``. Solving ``(1 - rate)^n < background_ratio``
        gives ``n = ln(background_ratio) / ln(1 - rate)``. Pruning (``-rate * 0.05``
        per frame) shortens this slightly, so treat the value as an upper bound for
        "how long an unprotected static change survives while the model adapts".
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
               exclusion_boxes: Sequence[np.ndarray], zone_empty_seconds: Optional[float],
               timestamp: float, *, to_reference: Optional[np.ndarray] = None,
               registration_ok: bool = True, hold: bool = False,
               ignore_mask: Optional[np.ndarray] = None) -> DebrisResult:
        """Classify one frame and advance the persistence timers.

        Args:
            frame_bgr: HxWx3 (or HxW grey) uint8 frame at processing resolution.
            zone_mask: HxW mask of the zone in the coordinates of ``frame_bgr`` (non-zero
                inside), i.e. ``ZoneState.mask``.
            exclusion_boxes: xyxy boxes (frame coordinates) of the detections that explain
                what they cover (the monitor passes vehicles, persons and trains, not
                unmapped classes); each is dilated by ``exclusion_dilate_px``. Covered
                pixels are unknown for this frame: never learned, never a new candidate,
                and an existing candidate under a box keeps its persistence.
            zone_empty_seconds: how long the tracker has seen the zone empty (None =
                unknown / occupied). Only consulted by the illumination guard when the zone
                covers the whole frame (nothing outside it to corroborate a lighting change).
            timestamp: seconds (monotonic within a run). Timers are timestamp based so
                frame skipping or variable fps does not change how long debris must persist.
            to_reference: optional 3x3 matrix mapping pixels of ``frame_bgr`` to pixels of
                the frame the background was learned in (the stabiliser's cumulative
                current -> reference transform). The frame is warped with it before
                anything else; pixels outside the warped frame are unknown.
            registration_ok: False while the caller cannot register the frame (the zone
                stabiliser holds): timers are held, nothing is learned.
            hold: True while a train detection overlaps the zone: timers are held, no new
                candidates start and nothing is learned, until ``hold_linger_seconds`` after
                the last held frame. A hold that starts less than ``persist_seconds`` after
                the previous one ended restarts the candidates that started in between (the
                same train, seen while the detector missed it).
            ignore_mask: optional HxW mask (reference geometry, frame resolution) of pixels
                excluded from every debris decision (``debris.ignore_polygons``).

        Returns:
            :class:`DebrisResult`. During warm-up ``warmed_up=False`` and area 0. When the
            model has to be re-warmed a warm-up result is returned and
            :attr:`last_illumination_event` is set for this call.
        """
        frame = np.asarray(frame_bgr)
        if frame.ndim not in (2, 3):
            raise ValueError(f"frame must be HxW or HxWxC, got shape {frame.shape}")
        if frame.ndim == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        elif frame.ndim == 3 and frame.shape[2] == 1:
            frame = frame[:, :, 0]
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        H, W = frame.shape[:2]
        zone_mask = np.asarray(zone_mask)
        if zone_mask.ndim != 2 or zone_mask.shape != (H, W):
            raise ValueError(
                f"zone_mask shape {zone_mask.shape} does not match frame shape {(H, W)}")
        M = None
        if to_reference is not None:
            M = np.asarray(to_reference, dtype=np.float64)
            if M.shape != (3, 3):
                raise ValueError(f"to_reference must be a 3x3 matrix, got shape {M.shape}")
            if not np.all(np.isfinite(M)):
                M, registration_ok = None, False

        cfg = self.cfg
        self.last_illumination_event = None
        t = float(timestamp)
        geo = self._prepare(frame, _u8(zone_mask), exclusion_boxes, M, ignore_mask)
        if self._shape is not None and (self._shape != geo["img"].shape or self._region != geo["region"]
                                        or self._full_hw != geo["full_hw"]):
            LOGGER.info("debris: frame geometry or zone region changed (%s %s -> %s %s); re-warming",
                        self._full_hw, self._region, geo["full_hw"], geo["region"])
            self.reset()
        self._shape = geo["img"].shape
        self._region = geo["region"]
        self._full_hw = geo["full_hw"]

        if self._t_prev is None:
            dt = 0.0
        else:
            dt = t - self._t_prev
            if dt < 0.0:
                # A looped file that was not reset: every candidate restarts, and every
                # deadline kept in the old time base (hold, motion history, update cadence)
                # is re-anchored, or it would stall until the clock caught up again.
                self._restart_candidates()
                self._history.clear()
                self._hold_until = -math.inf
                self._was_held = False
                self._hold_ended_at = None
                self._gain_bad_since = None
                self._ghost_due = -math.inf
                self._ghost_last_t = None
                if self._last_update_t is not None:
                    self._last_update_t = t
                dt = 0.0
        dt = min(dt, MAX_STEP_SECONDS)
        self._t_prev = t

        if not self._warmed:
            if not registration_ok:
                return self._finish(self._warmup_result(geo, t, registration_ok))
            return self._finish(self._warmup_step(geo, dt, t, registration_ok))

        if hold:
            self._hold_until = max(self._hold_until, t + float(cfg.hold_linger_seconds))
        if not registration_ok:
            return self._finish(self._held_result(geo, "registration", registration_ok))
        held = hold or t < self._hold_until
        return self._finish(self._monitor_step(geo, dt, t, zone_empty_seconds, held))

    # ---------------------------------------------------------------- geometry

    def _prepare(self, frame: np.ndarray, zone: np.ndarray, boxes: Sequence[np.ndarray],
                 M: Optional[np.ndarray], ignore_mask: Optional[np.ndarray]) -> Dict[str, object]:
        """Scale to ``process_width``, register, crop to the modelled region, and rasterise
        the zone, boxes and validity.

        Everything downstream works in the *internal* geometry: the reference geometry of
        the model (when ``M`` is given) at the processing scale, cropped to a fixed region
        around the zone (``_region_for``). Pixels far from the zone decide nothing, so the
        model does not spend time on them; the margin keeps enough background around the
        zone for the gain estimate and the illumination guard.
        """
        cfg = self.cfg
        H, W = frame.shape[:2]
        pw = int(cfg.process_width or 0)
        if pw > 0 and W > pw:
            w, h = pw, max(1, int(round(H * pw / float(W))))
            sx, sy = w / float(W), h / float(H)
            full = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        else:
            w, h, sx, sy, full = W, H, 1.0, 1.0, frame
        Mi = None
        if M is not None:
            S = np.diag([sx, sy, 1.0])
            Mi = S @ M @ np.linalg.inv(S)
        zone_full = self._reference_zone(zone, Mi, (h, w), ignore_mask)
        region = self._region_for(zone_full)
        y0, y1, x0, x1 = region
        rh, rw = y1 - y0, x1 - x0
        valid = None
        if Mi is not None:
            # Warp straight into the region: reference pixel (x, y) is region pixel (x-x0, y-y0).
            A = np.array([[1.0, 0.0, -x0], [0.0, 1.0, -y0], [0.0, 0.0, 1.0]]) @ Mi
            img = cv2.warpPerspective(full, A, (rw, rh), flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            # Valid = pixels whose source lies inside the frame, one pixel in from the edge
            # (the bilinear fringe mixes in the black border).
            corners = np.float32([[1, 1], [w - 2, 1], [w - 2, h - 2], [1, h - 2]]).reshape(-1, 1, 2)
            quad = cv2.perspectiveTransform(corners, A).reshape(-1, 2)
            valid = np.zeros((rh, rw), np.uint8)
            if np.all(np.isfinite(quad)):
                cv2.fillPoly(valid, [np.round(quad).astype(np.int32).reshape(-1, 1, 2)], 255)
        elif (rh, rw) != (h, w):
            img = np.ascontiguousarray(full[y0:y1, x0:x1])
        else:
            img = full
        zone_r = zone_full if (rh, rw) == (h, w) else np.ascontiguousarray(zone_full[y0:y1, x0:x1])

        box_mask = np.zeros((rh, rw), np.uint8)
        pad = float(cfg.exclusion_dilate_px)
        for box in boxes or ():
            x1, y1, x2, y2 = [float(v) for v in dilate_box(np.asarray(box, dtype=np.float32).reshape(4), pad, (H, W))]
            if M is not None:
                pts = np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]).reshape(-1, 1, 2)
                pts = cv2.perspectiveTransform(pts, M).reshape(-1, 2)
                if not np.all(np.isfinite(pts)):
                    continue
                x1, y1 = float(pts[:, 0].min()), float(pts[:, 1].min())
                x2, y2 = float(pts[:, 0].max()), float(pts[:, 1].max())
            # floor/ceil so the exclusion is never smaller than the dilated box.
            cv2.rectangle(box_mask, (int(math.floor(x1 * sx)) - x0, int(math.floor(y1 * sy)) - y0),
                          (int(math.ceil(x2 * sx)) - x0, int(math.ceil(y2 * sy)) - y0), 255, thickness=-1)

        free = cv2.bitwise_not(box_mask) if valid is None else cv2.bitwise_and(valid, cv2.bitwise_not(box_mask))
        return {"img": img, "zone": zone_r, "valid": valid, "box": box_mask, "free": free,
                "Mi": Mi, "sx": sx, "sy": sy, "out_hw": (H, W), "full_hw": (h, w), "region": region,
                "registered": M is not None}

    def _region_for(self, zone_full: np.ndarray) -> Tuple[int, int, int, int]:
        """The modelled region (y0, y1, x0, x1): the zone's bounding box plus a margin.

        The margin is at least REGION_MARGIN_PX and grows with the zone (its height above and
        below, a quarter of its width left and right), so there is background around the zone
        for the gain estimate and a camera that is not registered can move a little. The
        region of a running model is kept while it still contains the zone; when the zone
        leaves it (a new calibration, or an unregistered camera that moved further than the
        margin) a new region is chosen and :meth:`update` rebuilds the model.
        """
        h, w = zone_full.shape[:2]
        x, y, bw, bh = cv2.boundingRect(zone_full)
        cur = self._region
        if cur is not None and self._full_hw == (h, w):
            y0, y1, x0, x1 = cur
            if bw == 0 or (x >= x0 and y >= y0 and x + bw <= x1 and y + bh <= y1):
                return cur
        if bw == 0 or bh == 0:
            return (0, h, 0, w)
        my = max(REGION_MARGIN_PX, bh)
        mx = max(REGION_MARGIN_PX, bw // 4)
        return (max(0, y - my), min(h, y + bh + my), max(0, x - mx), min(w, x + bw + mx))

    def _reference_zone(self, zone: np.ndarray, Mi: Optional[np.ndarray], hw: Tuple[int, int],
                        ignore_mask: Optional[np.ndarray]) -> np.ndarray:
        """The zone in the internal geometry, minus the ignored pixels.

        Registered frames: in the reference geometry the zone does not move, so the warp is
        cached and only redone when the caller's zone mask changes by more than jitter (area
        +-2 %, bounding box +-4 px), e.g. after a camera bump or a new calibration.
        Unregistered frames use the caller's mask as it is on every frame.
        """
        h, w = hw
        sig = None
        if Mi is not None:
            x, y, bw, bh = cv2.boundingRect(zone)
            sig = (int(cv2.countNonZero(zone)), x, y, bw, bh, (h, w),
                   None if ignore_mask is None else (id(ignore_mask), ignore_mask.shape))
            cached = self._zone_cache
            if cached is not None:
                old = cached[0]
                if (old[5:] == sig[5:] and abs(sig[0] - old[0]) <= 0.02 * max(1, old[0])
                        and max(abs(a - b) for a, b in zip(sig[1:5], old[1:5])) <= 4):
                    return cached[1]
        ref = zone
        if (h, w) != zone.shape[:2]:
            ref = cv2.resize(ref, (w, h), interpolation=cv2.INTER_NEAREST)
        if Mi is not None:
            ref = cv2.warpPerspective(ref, Mi, (w, h), flags=cv2.INTER_NEAREST,
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        if ignore_mask is not None:
            ign = _u8(np.asarray(ignore_mask))
            if ign.shape != (h, w):
                ign = cv2.resize(ign, (w, h), interpolation=cv2.INTER_NEAREST)
            ref = cv2.bitwise_and(ref, cv2.bitwise_not(ign))
        if sig is not None:
            self._zone_cache = (sig, ref)
        return ref

    def _zone_roi(self, zone: np.ndarray) -> Tuple[int, int, int, int]:
        """Padded, grid-snapped bounding box (y0, y1, x0, x1) of the zone.

        Per-pixel timers are only updated inside it; when it changes, timers outside the
        new box are cleared so no stale candidate survives a zone change.
        """
        h, w = zone.shape[:2]
        x, y, bw, bh = cv2.boundingRect(zone)
        if bw == 0 or bh == 0:
            roi = (0, 0, 0, 0)
        else:
            s = ROI_SNAP_PX
            y0 = max(0, (y - ROI_PAD_PX) // s * s)
            x0 = max(0, (x - ROI_PAD_PX) // s * s)
            y1 = min(h, -(-(y + bh + ROI_PAD_PX) // s) * s)
            x1 = min(w, -(-(x + bw + ROI_PAD_PX) // s) * s)
            roi = (int(y0), int(y1), int(x0), int(x1))
        if self._roi is not None and roi != self._roi and self._cand is not None:
            keep = np.zeros(self._cand.shape, dtype=bool)
            keep[roi[0]:roi[1], roi[2]:roi[3]] = True
            for arr in (self._age, self._covered, self._ghost):
                arr[~keep] = 0.0
            self._cand &= keep
        self._roi = roi
        return roi

    # ----------------------------------------------------------------- warm-up

    def _warmup_done(self) -> bool:
        frames = self.cfg.warmup_frames
        if frames is not None:
            return self._warm_frames >= max(1, int(frames))
        return (self._warm_elapsed >= float(self.cfg.warmup_seconds) - 1e-9
                and self._warm_frames >= MIN_WARMUP_FRAMES)

    def _warmup_step(self, geo: Dict[str, object], dt: float, t: float,
                     registration_ok: bool) -> DebrisResult:
        """Learn one warm-up frame, never under a detection box (see module docstring)."""
        img = geo["img"]
        free = geo["free"]
        if self._composite is None:
            composite = img.copy()  # very first frame: nothing better to paste under the boxes
            self._free_count = np.zeros(img.shape[:2], np.uint16)
        else:
            composite = img.copy()
            np.copyto(composite, self._composite, where=(free == 0) if img.ndim == 2 else (free == 0)[..., None])
        self._composite = composite
        self._mog2.apply(composite, learningRate=AUTO_LEARNING_RATE)
        # Running sums for each pixel's temporal variance: the starting point of the residual
        # variance that keeps pasted pixels at their own noise level (see _learn).
        if self._acc1 is None:
            self._acc1 = np.zeros(composite.shape, np.float32)
            self._acc2 = np.zeros(composite.shape, np.float32)
        cv2.accumulate(composite, self._acc1)
        cv2.accumulateSquare(composite, self._acc2)
        self._free_count[free > 0] += 1
        self._warm_frames += 1
        self._frames_seen += 1
        self._warm_elapsed += dt
        if self._warmup_done():
            self._finish_warmup(t)
        return self._warmup_result(geo, t, registration_ok)

    def _finish_warmup(self, t: float) -> None:
        h, w = self._shape[:2]
        n = max(1, self._warm_frames)
        self._learned = np.where(self._free_count.astype(np.int32) * 2 >= n, 255, 0).astype(np.uint8)
        self._pending = np.zeros((h, w), np.float32)
        self._age = np.zeros((h, w), np.float64)  # float64: summed frame steps must reach persist_seconds exactly
        self._born = np.zeros((h, w), np.float64)
        self._cand = np.zeros((h, w), dtype=bool)
        self._covered = np.zeros((h, w), np.float32)
        self._ghost = np.zeros((h, w), np.float32)
        self._seen_bg = np.zeros((h, w), np.float32)
        self._bg = self._mog2.getBackgroundImage()
        self._bg_gray = self._grey(self._bg)
        # Temporal variance of the warm-up frames, summed over channels like MOG2's own
        # per-mode variance, and clamped to the model's variance range.
        mean = self._acc1 / float(n)
        var = np.maximum(self._acc2 / float(n) - mean * mean, 0.0)
        var = var.sum(axis=2) if var.ndim == 3 else var
        self._resvar = np.clip(var, self._mog2.getVarMin(), self._mog2.getVarMax()).astype(np.float32)
        self._acc1 = self._acc2 = None
        self._composite = None
        self._free_count = None
        self._history.clear()
        # The model was just learned on these pixels: treat them as background for the gain
        # estimate of the first monitoring frame (a step right after the warm-up is compensated).
        self._prev_raw = np.zeros((h, w), np.uint8)
        self._last_update_t = t
        self._warmed = True
        self._triggered = False
        self._trusted = True
        unknown = int(np.count_nonzero(self._learned == 0))
        LOGGER.info("debris warm-up complete after %d frames (%.1f s); %.1f%% of the frame never seen box-free",
                    self._warm_frames, self._warm_elapsed, 100.0 * unknown / float(h * w))

    def _warmup_result(self, geo: Dict[str, object], t: float, registration_ok: bool) -> DebrisResult:
        frames = self.cfg.warmup_frames
        if frames is not None:
            state = f"warmup {self._warm_frames}/{int(frames)}f"
        else:
            state = f"warmup {self._warm_elapsed:.1f}/{float(self.cfg.warmup_seconds):.1f}s"
        res = self._empty_result(geo["out_hw"], AUTO_LEARNING_RATE, warmed_up=False)
        res.unknown_fraction = 1.0
        res.registered = bool(geo["registered"])
        res.trusted = False
        res.state = state if registration_ok else state + " (no registration)"
        return res

    # ---------------------------------------------------------------- monitoring

    def _held_result(self, geo: Dict[str, object], reason: str, registration_ok: bool) -> DebrisResult:
        """Registration lost: nothing is classified; report the last observation, timers frozen."""
        last = self._last_result
        if last is None or not last.warmed_up:
            res = self._empty_result(geo["out_hw"], 0.0, warmed_up=True)
        else:
            res = DebrisResult(mask=last.mask, area_px=last.area_px, area_fraction=last.area_fraction,
                               blobs=list(last.blobs), persist_seconds=last.persist_seconds,
                               triggered=last.triggered, learning_rate_used=0.0, warmed_up=True,
                               unknown_fraction=last.unknown_fraction, covered_fraction=last.covered_fraction,
                               gain=last.gain)
        res.learning_rate_used = 0.0
        res.registered = bool(geo["registered"])
        res.held = True
        res.trusted = bool(registration_ok) and self._trusted
        res.state = f"held ({reason})"
        self._triggered = bool(res.triggered)
        return res

    def _monitor_step(self, geo: Dict[str, object], dt: float, t: float,
                      zone_empty_seconds: Optional[float], held: bool) -> DebrisResult:
        """One monitoring frame. ``held`` (train over the zone): the frame is classified and
        the motion reference is refreshed, so the train does not look like a fresh arrival
        once the hold ends, but candidates, timers and the model are left untouched."""
        cfg = self.cfg
        img = geo["img"]
        zone = geo["zone"]
        valid = geo["valid"]
        box = geo["box"]
        h, w = img.shape[:2]

        # (1) global gain on background pixels outside the zone and the boxes
        outside = cv2.bitwise_not(zone) if valid is None else cv2.bitwise_and(valid, cv2.bitwise_not(zone))
        gain = self._estimate_gain(img, outside, geo["free"], zone)
        rg = float(cfg.rewarm_gain)
        if rg > 0.0 and (float(gain.max()) > rg or float(gain.min()) < 1.0 / rg):
            if self._gain_bad_since is None:
                self._gain_bad_since = t
            elif t - self._gain_bad_since >= REWARM_GAIN_SECONDS and not self._triggered and not held:
                return self._rewarm(geo, t, "gain", {"gain": [float(g) for g in gain]})
        else:
            self._gain_bad_since = None
        comp, clipped = self._compensate(img, gain)

        # (2) classification against the frozen model
        raw = self._mog2.apply(comp, learningRate=float(cfg.monitoring_learning_rate))
        self._frames_seen += 1
        fg = cv2.compare(raw, FOREGROUND_VALUE - 1, cv2.CMP_GT)
        # Opening removes speckle (sensor noise, rain, compression), closing fills holes
        # inside an object so it forms one component with one area.
        if self._kernel is not None:
            fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, self._kernel)
            fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, self._kernel)

        # (3) what can be judged: learned, in frame, not clipped, not under a box
        obs = cv2.bitwise_and(geo["free"], self._learned)
        if clipped is not None:
            obs = cv2.bitwise_and(obs, cv2.bitwise_not(clipped))
        zone_obs = cv2.bitwise_and(zone, obs)
        fg_zone = cv2.bitwise_and(fg, zone_obs)
        zone_area = int(cv2.countNonZero(zone))
        y0, y1, x0, x1 = self._zone_roi(zone)
        fg_vis, labels, stats, kept = self._filter_components(fg_zone, zone_area, (y0, y1, x0, x1))
        area_px = int(stats[kept, cv2.CC_STAT_AREA].sum()) if kept.size else 0
        area_fraction = (area_px / float(zone_area)) if zone_area > 0 else 0.0

        # (4) illumination the gain cannot explain: re-warm instead of latching an ALARM
        cap = float(cfg.max_area_fraction)
        zone_obs_area = int(cv2.countNonZero(zone_obs))
        judged = outside if clipped is None else cv2.bitwise_and(outside, cv2.bitwise_not(clipped))
        outside_total = int(cv2.countNonZero(judged))
        outside_fraction = (int(cv2.countNonZero(cv2.bitwise_and(fg, judged))) / float(outside_total)
                            if outside_total > 0 else None)
        if (cap > 0.0 and not self._triggered and not held and zone_obs_area >= 0.25 * max(1, zone_area)
                and area_px / float(max(1, zone_obs_area)) >= cap
                and (zone_empty_seconds is not None if outside_fraction is None else outside_fraction >= cap)):
            return self._rewarm(geo, t, "illumination step", {
                "area_fraction": float(area_px / float(max(1, zone_obs_area))),
                "outside_zone_fraction": None if outside_fraction is None else float(outside_fraction),
                "max_area_fraction": cap})

        # (5) lighting the global gain does not cover (a cloud over a pole's shadow, a sun
        # patch): the background's own texture under different light is not an object
        gray = self._grey(comp)
        fg_vis, kept = self._drop_relit_components(fg_vis, labels, kept, gray, (y0, y1, x0, x1))
        area_px = int(stats[kept, cv2.CC_STAT_AREA].sum()) if kept.size else 0
        area_fraction = (area_px / float(zone_area)) if zone_area > 0 else 0.0

        # (6) immobility, persistence and ghosts (inside the zone's region of interest)
        # The size bar refers to the whole processing frame, not the modelled region.
        fh, fw = geo["full_hw"]
        frame_area = int(fh) * int(fw)
        if held:
            if not self._was_held:
                self._restart_gap_candidates(t, zone, zone_area, (y0, y1, x0, x1), frame_area)
            self._lag_entry(t)  # keep the history window moving
            persist, triggered = self._held_persist, self._triggered
        else:
            if self._was_held:
                self._hold_ended_at = t
            self._advance(fg_vis, labels, stats, kept, zone, zone_obs, gray, dt, t, (y0, y1, x0, x1))
            persist, _n_cand = self._persistence(zone, zone_area, (y0, y1, x0, x1), frame_area)
            triggered = persist >= float(cfg.persist_seconds) - 1e-6
            self._triggered = bool(triggered)
            self._held_persist = float(persist)
            if float(cfg.ghost_heal_seconds) > 0.0 and kept.size and t >= self._ghost_due:
                # A few times a second is plenty for a heal that takes seconds.
                elapsed = dt if self._ghost_last_t is None else min(t - self._ghost_last_t,
                                                                    2.0 * GHOST_CHECK_SECONDS)
                self._ghost_last_t = t
                self._ghost_due = t + GHOST_CHECK_SECONDS
                self._heal_ghosts(raw, labels, stats, kept, comp, obs, elapsed, (y0, y1, x0, x1))
        self._was_held = held
        self._push_history(t, gray, fg_vis, zone_obs)

        # (7) selective update of the model (never while held)
        lr = float(cfg.monitoring_learning_rate)
        tau = float(cfg.update_time_constant_seconds)
        if held:
            lr = min(lr, 0.0)
            if self._last_update_t is not None:
                self._last_update_t = t  # the next learning apply waits a full interval
        elif tau > 0.0 and self._last_update_t is not None \
                and t - self._last_update_t >= float(cfg.update_interval_seconds) - 1e-9:
            lr = self._learn(comp, raw, fg, zone, geo, clipped, t, tau)
        self._prev_raw = raw

        # (8) health
        unknown = cv2.bitwise_not(self._learned)
        if valid is not None:
            unknown = cv2.bitwise_or(unknown, cv2.bitwise_not(valid))
        if clipped is not None:
            unknown = cv2.bitwise_or(unknown, clipped)
        unknown_fraction = (cv2.countNonZero(cv2.bitwise_and(unknown, zone)) / float(zone_area)) if zone_area else 1.0
        covered = cv2.bitwise_and(box, zone) if valid is None else cv2.bitwise_and(cv2.bitwise_and(box, zone), valid)
        covered_fraction = (cv2.countNonZero(covered) / float(zone_area)) if zone_area else 0.0
        limit = float(cfg.max_unknown_fraction)
        if self._trusted and unknown_fraction > limit:
            self._trusted = False
        elif not self._trusted and unknown_fraction <= TRUST_RECOVERY * limit:
            self._trusted = True

        last = self._last_result
        if held and last is not None and last.warmed_up:
            # Report the observation from before the hold (what is visible now is the train).
            mask, blobs, area_out, area_fraction = last.mask, list(last.blobs), last.area_px, last.area_fraction
        else:
            mask, blobs, area_out = self._to_caller(fg_vis, stats, kept, area_px, geo)
        state = "ok" if self._trusted else "untrusted"
        if unknown_fraction >= 0.05:
            state += f" unknown {100.0 * unknown_fraction:.0f}%"
        if held:
            state = "held (train)"
        return DebrisResult(
            mask=mask,
            area_px=int(area_out),
            area_fraction=float(area_fraction),
            blobs=blobs,
            persist_seconds=float(persist),
            triggered=bool(triggered),
            learning_rate_used=float(lr),
            warmed_up=True,
            unknown_fraction=float(unknown_fraction),
            covered_fraction=float(covered_fraction),
            registered=bool(geo["registered"]),
            trusted=bool(self._trusted),
            gain=float(np.mean(gain)),
            held=bool(held),
            state=state,
        )

    # ---------------------------------------------------------------- helpers

    @staticmethod
    def _grey(frame: np.ndarray) -> np.ndarray:
        """Single-channel uint8 copy of the frame, used by the immobility and ghost tests."""
        if frame.ndim == 3:
            if frame.shape[2] == 3:
                return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if frame.shape[2] == 4:
                return cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
            frame = frame[:, :, 0]
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(frame)

    def _estimate_gain(self, img: np.ndarray, outside: np.ndarray, free: np.ndarray,
                       zone: np.ndarray) -> np.ndarray:
        """Per-channel gain current / background on last frame's background pixels.

        Reference pixels: outside the zone (an obstruction confined to the crossing cannot
        bias the estimate), outside the dilated boxes, classified background on the
        previous frame, background value in a range that neither clips nor crushes. When
        the zone leaves too little outside, background pixels inside it are used (the median
        is robust while objects cover less than half of them).
        """
        nch = 1 if img.ndim == 2 else img.shape[2]
        if not self.cfg.gain_compensation or self._prev_raw is None or self._bg is None:
            self._gain = np.ones(nch, dtype=np.float64)
            return self._gain
        if self._gain_idx is None or self._gain_idx[-1] >= img.shape[0] * img.shape[1]:
            h, w = img.shape[:2]
            ys, xs = np.mgrid[0:h:GAIN_SAMPLE_STEP, 0:w:GAIN_SAMPLE_STEP]
            self._gain_idx = (ys * w + xs).reshape(-1)
        idx = self._gain_idx
        bg = self._bg.reshape(-1, nch)[idx].astype(np.float32)
        cur = img.reshape(-1, nch)[idx].astype(np.float32)
        lo, hi = GAIN_BG_RANGE
        base = ((free.reshape(-1)[idx] > 0) & (self._prev_raw.reshape(-1)[idx] == 0)
                & (self._learned.reshape(-1)[idx] > 0)
                & np.all((bg >= lo) & (bg <= hi) & (cur >= 3.0) & (cur <= 252.0), axis=1))
        sel = base & (outside.reshape(-1)[idx] > 0)
        if int(np.count_nonzero(sel)) < GAIN_MIN_SAMPLES:
            sel = base & ((zone.reshape(-1)[idx] > 0) | sel)
        if int(np.count_nonzero(sel)) < GAIN_MIN_SAMPLES:
            return self._gain if self._gain.size == nch else np.ones(nch, dtype=np.float64)
        ratio = cur[sel] / bg[sel]
        self._gain = np.median(ratio, axis=0).astype(np.float64).reshape(nch)
        return self._gain

    def _compensate(self, img: np.ndarray, gain: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Divide the gain out; return (compensated frame, mask of clipped pixels or None)."""
        if np.all(np.abs(gain - 1.0) < GAIN_DEADBAND):
            return img, None
        if img.ndim == 2:
            comp = cv2.convertScaleAbs(img, alpha=1.0 / float(gain[0]))
        else:
            comp = cv2.multiply(img, tuple(float(1.0 / g) for g in gain) + (0.0,) * (4 - len(gain)))
        # Clipped: a background value the gain pushes beyond CLIP_HIGH (or below CLIP_LOW) in
        # any channel. One inRange over the per-channel range that stays comparable.
        lower = [0.0 if g >= 1.0 else math.floor(CLIP_LOW / g) + 1.0 for g in gain]
        upper = [255.0 if g <= 1.0 else math.ceil(CLIP_HIGH / g) - 1.0 for g in gain]
        comparable = cv2.inRange(self._bg, np.array(lower), np.array(upper))
        if cv2.countNonZero(comparable) == comparable.size:
            return comp, None
        return comp, cv2.bitwise_not(comparable)

    def _filter_components(self, fg: np.ndarray, zone_area: int, roi: Tuple[int, int, int, int]
                           ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Keep connected components with ``area >= min_blob_area_fraction * zone_area``.

        Works on the zone's region of interest. Returns ``(mask, labels, stats, kept)``:
        the full-size mask of kept components, the ROI label image, the ROI stats (offsets
        are ROI coordinates) and the kept label indices.
        """
        y0, y1, x0, x1 = roi
        mask = np.zeros_like(fg)
        if y1 <= y0 or x1 <= x0:
            return mask, np.zeros((0, 0), np.int32), np.zeros((1, 5), np.int32), np.zeros(0, np.int64)
        sub = fg[y0:y1, x0:x1]
        n, labels, stats, _ = cv2.connectedComponentsWithStats(sub, connectivity=8, ltype=cv2.CV_32S)
        if n <= 1:
            return mask, labels, stats, np.zeros(0, np.int64)
        min_px = float(self.cfg.min_blob_area_fraction) * float(zone_area)
        kept = np.flatnonzero(stats[1:, cv2.CC_STAT_AREA] >= min_px) + 1  # +1: label 0 is background
        if kept.size == n - 1:
            mask[y0:y1, x0:x1] = sub  # nothing dropped: avoid the relabel pass
        elif kept.size:
            lut = np.zeros(n, dtype=np.uint8)
            lut[kept] = 255
            mask[y0:y1, x0:x1] = lut[labels]
        return mask, labels, stats, kept

    def _drop_relit_components(self, fg_vis: np.ndarray, labels: np.ndarray, kept: np.ndarray,
                               gray: np.ndarray, roi: Tuple[int, int, int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """Remove foreground components that are the background under different light.

        A shadow or a light patch scales the texture that is already there; an object
        replaces it. Over each kept component the grey values of the frame are correlated
        with the background image's: where the background is textured (std >= RELIT_MIN_STD)
        and the correlation is at least RELIT_MIN_NCC, the change is lighting (a cloud edge
        over a pole's shadow is darker than MOG2's shadow ratio and used to time as debris)
        and the component is not a candidate. On flat background nothing can be concluded,
        so the component stays a candidate. The pixels are re-tested on every frame, so an
        object arriving in a relit area is still seen.
        """
        if not kept.size:
            return fg_vis, kept
        y0, y1, x0, x1 = roi
        sl = (slice(y0, y1), slice(x0, x1))
        vis = fg_vis[sl] > 0
        lab = labels[vis]
        if lab.size == 0:
            return fg_vis, kept
        c = gray[sl][vis].astype(np.float64)
        b = self._bg_gray[sl][vis].astype(np.float64)
        n_lab = int(labels.max()) + 1
        n = np.bincount(lab, minlength=n_lab).astype(np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            mc = np.bincount(lab, weights=c, minlength=n_lab) / n
            mb = np.bincount(lab, weights=b, minlength=n_lab) / n
            vc = np.bincount(lab, weights=c * c, minlength=n_lab) / n - mc * mc
            vb = np.bincount(lab, weights=b * b, minlength=n_lab) / n - mb * mb
            cov = np.bincount(lab, weights=c * b, minlength=n_lab) / n - mc * mb
            ncc = cov / np.sqrt(np.maximum(vc, 1e-6) * np.maximum(vb, 1e-6))
        relit = (n >= RELIT_MIN_PIXELS) & (vb >= RELIT_MIN_STD ** 2) & (ncc >= RELIT_MIN_NCC)
        drop = kept[relit[kept]]
        if drop.size == 0:
            return fg_vis, kept
        lut = np.zeros(n_lab, dtype=bool)
        lut[drop] = True
        sub = fg_vis[sl]
        sub[lut[labels]] = 0
        return fg_vis, kept[~relit[kept]]

    def _restart_candidates(self) -> None:
        if self._cand is not None:
            self._cand[:] = False
            self._age[:] = 0.0
            self._covered[:] = 0.0
            self._ghost[:] = 0.0

    def _restart_gap_candidates(self, t: float, zone: np.ndarray, zone_area: int,
                                roi: Tuple[int, int, int, int], frame_area: int) -> None:
        """First held frame of a hold: restart the candidates of the gap since the last hold.

        Only when the previous hold ended less than ``persist_seconds`` ago (one passage; in
        a longer gap a static candidate could trigger on its own, so a restart would protect
        nothing and only delay a real obstruction) and debris is not triggered (a restart
        must never end an ALARM). Candidates whose age started before the previous hold
        ended keep it, and the persistence reported while held is recomputed without the
        restarted pixels (see the module docstring).
        """
        ended = self._hold_ended_at
        if (ended is None or self._cand is None or self._triggered
                or t - ended >= float(self.cfg.persist_seconds)):
            return
        gap = self._cand & (self._born >= ended)
        if not gap.any():
            return
        self._cand[gap] = False
        self._age[gap] = 0.0
        self._covered[gap] = 0.0
        self._ghost[gap] = 0.0
        self._held_persist, _n_cand = self._persistence(zone, zone_area, roi, frame_area)
        LOGGER.debug("debris: hold again %.2f s after the last one; %d candidate px of the gap restarted",
                     t - ended, int(np.count_nonzero(gap)))

    def _lag_entry(self, t: float):
        """The newest history frame at least ``static_motion_lag_seconds`` old (or None)."""
        lag = float(self.cfg.static_motion_lag_seconds)
        hist = self._history
        while len(hist) >= 2 and t - hist[1][0] >= lag:
            hist.popleft()
        if hist and t - hist[0][0] >= lag - 1e-9:
            return hist[0]
        return None

    def _push_history(self, t: float, gray: np.ndarray, fg_vis: np.ndarray, zone_obs: np.ndarray) -> None:
        lag = float(self.cfg.static_motion_lag_seconds)
        if self._history and t - self._history[-1][0] < lag / 4.0:
            return
        lo = cv2.erode(gray, self._tolerance_kernel)
        hi = cv2.dilate(gray, self._tolerance_kernel)
        # Visible background at that time: observable zone pixels that were not foreground.
        bg_vis = cv2.bitwise_and(zone_obs, cv2.bitwise_not(fg_vis))
        self._history.append((t, lo, hi, fg_vis.copy(), bg_vis))

    def _advance(self, fg_vis: np.ndarray, labels: np.ndarray, stats: np.ndarray, kept: np.ndarray,
                 zone: np.ndarray, zone_obs: np.ndarray, gray: np.ndarray, dt: float, t: float,
                 roi: Tuple[int, int, int, int]) -> None:
        """Per-pixel persistence: advance static candidates, restart moving ones, bridge covers."""
        cfg = self.cfg
        y0, y1, x0, x1 = roi
        if y1 <= y0 or x1 <= x0:
            return
        sl = (slice(y0, y1), slice(x0, x1))
        vis = fg_vis[sl] > 0
        age, cand, covered = self._age[sl], self._cand[sl], self._covered[sl]

        restart = np.zeros(vis.shape, dtype=bool)
        moving = np.zeros(vis.shape, dtype=bool)
        entry = self._lag_entry(t)
        thr = float(cfg.static_motion_diff)
        if entry is not None and kept.size and labels.size and thr > 0.0:
            _, lo, hi, fg_lag, _bg_lag = entry
            g = gray[sl]
            change = cv2.max(cv2.subtract(g, hi[sl]), cv2.subtract(lo[sl], g))
            moving = (change > thr) & (fg_lag[sl] > 0) & vis
            n = stats.shape[0]
            areas = np.maximum(stats[:, cv2.CC_STAT_AREA], 1).astype(np.float64)
            bad = np.bincount(labels[moving], minlength=n) > float(cfg.static_motion_fraction) * areas
            bad[0] = False
            if bad.any():
                restart = bad[labels] & vis

        cont = vis & cand & ~restart & ~moving
        age[cont] += dt
        fresh = vis & ~cont
        age[fresh] = 0.0
        self._born[sl][fresh] = t
        cand[vis] = True
        covered[vis] = 0.0
        # Visible background ends a candidate at once, and confirms the model content there
        # (the ghost test only heals content that was seen for a short time).
        seen_bg = (zone_obs[sl] > 0) & ~vis
        cand[seen_bg] = False
        age[seen_bg] = 0.0
        covered[seen_bg] = 0.0
        self._ghost[sl][seen_bg] = 0.0
        self._seen_bg[sl][seen_bg] += dt
        # Unobservable (covered, off-frame, unknown) candidates: keep the age, count the
        # cover, and give up after occlusion_bridge_seconds.
        hidden = cand & ~vis & ~seen_bg
        if hidden.any():
            covered[hidden] += dt
            lost = hidden & (covered > float(cfg.occlusion_bridge_seconds))
            if lost.any():
                cand[lost] = False
                age[lost] = 0.0
                covered[lost] = 0.0
        # Candidates must lie in the zone.
        out = cand & (zone[sl] == 0)
        if out.any():
            cand[out] = False
            age[out] = 0.0
            covered[out] = 0.0

    def _threshold_px(self, zone_area: int, frame_area: int) -> float:
        cfg = self.cfg
        zone_px = float(cfg.min_area_fraction) * float(zone_area)
        frame_px = float(cfg.min_object_frame_fraction) * float(frame_area)
        bar = min(zone_px, frame_px) if frame_px > 0.0 else zone_px
        return max(1.0, bar)

    def _persistence(self, zone: np.ndarray, zone_area: int, roi: Tuple[int, int, int, int],
                     frame_area: int) -> Tuple[float, int]:
        """Seen-static seconds of the k-th oldest candidate pixel (0 when fewer than k)."""
        y0, y1, x0, x1 = roi
        if y1 <= y0 or x1 <= x0 or zone_area <= 0:
            return 0.0, 0
        sl = (slice(y0, y1), slice(x0, x1))
        cand = self._cand[sl] & (zone[sl] > 0)
        ages = self._age[sl][cand]
        k = int(math.ceil(self._threshold_px(zone_area, frame_area)))
        if ages.size < k:
            return 0.0, int(ages.size)
        return float(np.partition(ages, ages.size - k)[ages.size - k]), int(ages.size)

    def _heal_ghosts(self, raw: np.ndarray, labels: np.ndarray, stats: np.ndarray, kept: np.ndarray,
                     comp_img: np.ndarray, obs: np.ndarray, dt: float, roi: Tuple[int, int, int, int]) -> None:
        """Make candidates whose outline exists only in the model unknown again.

        Never while debris is triggered: a heal would end the ALARM while the thing is still
        there. For each kept component:

        * *young content*: at most ``GHOST_MAX_OLD_SHARE`` of its pixels may have had their
          model content seen as background for ``ghost_heal_max_seen_seconds`` or longer (an
          obstruction arrives on road the model has watched; a ghost appears where the warm-up
          learned something shortly before it left);
        * *outline*: the changed region that contains it - every pixel that differs from the
          model, shadow-labelled ones included, closed with the morphology kernel - so a dark
          object that is 'shadow' over dark ballast and 'foreground' over a bright rail is
          measured on its own outline, not on the rail's. The outer band keeps only observable
          pixels classified background and must be at least ``GHOST_MIN_OUTER_SHARE`` of the
          ring, otherwise nothing is concluded;
        * *step*: the colour step across that outline - mean colour of a thin band just inside
          minus a band just outside - is measured in the current frame and in the background
          image. A real object has it in the frame (object against road) and hardly any in the
          model (road against road); a ghost has it only in the model.

        A component whose frame step stays below ``GHOST_CONTRAST_RATIO`` of a clear model step
        (``GHOST_MIN_CONTRAST``) for ``ghost_heal_seconds`` loses its candidacy, and its changed
        region loses its learned status and is re-learned through the selective update. Band
        means average out texture, so a textured road does not hide the step and a low-contrast
        object is never healed (its model step is too small to qualify).
        """
        y0, y1, x0, x1 = roi
        sl = (slice(y0, y1), slice(x0, x1))
        ghost = self._ghost[sl]
        if self._triggered:
            ghost[:] = 0.0
            return
        cfg = self.cfg
        cur_img = comp_img[sl]
        bg_img = self._bg[sl]
        raw_r = raw[sl]
        matches = (obs[sl] > 0) & (raw_r == 0)  # observable and classified background
        changed = cv2.compare(raw_r, 0, cv2.CMP_GT)
        if self._kernel is not None:
            changed = cv2.morphologyEx(changed, cv2.MORPH_CLOSE, self._kernel)
        _n, ch_labels, ch_stats, _ = cv2.connectedComponentsWithStats(changed, connectivity=8, ltype=cv2.CV_32S)
        seen = self._seen_bg[sl]
        max_seen = float(cfg.ghost_heal_max_seen_seconds)
        heal_after = float(cfg.ghost_heal_seconds)
        healed = np.zeros(ghost.shape, dtype=bool)
        p = GHOST_RING_PX + 1
        rh, rw = labels.shape[:2]
        for lab in kept:
            x, y, bw, bh = (int(v) for v in stats[lab, :4])
            comp = labels[y:y + bh, x:x + bw] == lab
            ids = np.unique(ch_labels[y:y + bh, x:x + bw][comp])
            ids = ids[ids > 0]
            if ids.size == 0:
                continue
            rx0 = int(ch_stats[ids, cv2.CC_STAT_LEFT].min())
            ry0 = int(ch_stats[ids, cv2.CC_STAT_TOP].min())
            rx1 = int((ch_stats[ids, cv2.CC_STAT_LEFT] + ch_stats[ids, cv2.CC_STAT_WIDTH]).max())
            ry1 = int((ch_stats[ids, cv2.CC_STAT_TOP] + ch_stats[ids, cv2.CC_STAT_HEIGHT]).max())
            ys, ye = max(0, min(y, ry0) - p), min(rh, max(y + bh, ry1) + p)
            xs, xe = max(0, min(x, rx0) - p), min(rw, max(x + bw, rx1) + p)
            inside = labels[ys:ye, xs:xe] == lab
            gsub = ghost[ys:ye, xs:xe]
            if max_seen > 0.0:
                old = int(np.count_nonzero(seen[ys:ye, xs:xe][inside] >= max_seen))
                if old > GHOST_MAX_OLD_SHARE * int(np.count_nonzero(inside)):
                    gsub[inside] = 0.0
                    continue
            region = np.isin(ch_labels[ys:ye, xs:xe], ids).astype(np.uint8)
            inner = (region > 0) & (cv2.erode(region, self._ring_kernel) == 0)
            ring = (cv2.dilate(region, self._ring_kernel) > 0) & (region == 0)
            outer = ring & matches[ys:ye, xs:xe]
            n_outer = int(np.count_nonzero(outer))
            if (int(np.count_nonzero(inner)) < 8 or n_outer < 8
                    or n_outer < GHOST_MIN_OUTER_SHARE * int(np.count_nonzero(ring))):
                gsub[inside] = 0.0
                continue
            step_cur = self._colour_step(cur_img[ys:ye, xs:xe], inner, outer)
            step_bg = self._colour_step(bg_img[ys:ye, xs:xe], inner, outer)
            if step_bg >= GHOST_MIN_CONTRAST and step_cur < GHOST_CONTRAST_RATIO * step_bg:
                gsub[inside] += dt
                done = inside & (gsub >= heal_after)
                if 2 * int(np.count_nonzero(done)) >= int(np.count_nonzero(inside)):
                    hsub = healed[ys:ye, xs:xe]
                    hsub |= (region > 0) | inside
            else:
                gsub[inside] = 0.0
        if healed.any():
            n = int(healed.sum())
            self._learned[sl][healed] = 0
            self._pending[sl][healed] = 0.0
            self._cand[sl][healed] = False
            self._age[sl][healed] = 0.0
            self._covered[sl][healed] = 0.0
            self._seen_bg[sl][healed] = 0.0
            ghost[healed] = 0.0
            LOGGER.info("debris: %d px of a ghost (outline only in the background model) set to re-learn", n)

    @staticmethod
    def _colour_step(img: np.ndarray, inner: np.ndarray, outer: np.ndarray) -> float:
        """Euclidean distance between the mean colours of two pixel sets of ``img``."""
        a = img[inner].reshape(-1, 1 if img.ndim == 2 else img.shape[2]).astype(np.float32).mean(axis=0)
        b = img[outer].reshape(-1, 1 if img.ndim == 2 else img.shape[2]).astype(np.float32).mean(axis=0)
        return float(np.linalg.norm(a - b))

    def _learn(self, comp: np.ndarray, raw: np.ndarray, fg: np.ndarray, zone: np.ndarray,
               geo: Dict[str, object], clipped: Optional[np.ndarray], t: float, tau: float) -> float:
        """One selective learning apply (see module docstring); returns the rate used."""
        cfg = self.cfg
        dt_u = min(t - float(self._last_update_t), 4.0 * float(cfg.update_interval_seconds))
        self._last_update_t = t
        rate = min(1.0, max(0.0, dt_u / tau))
        free = geo["free"]  # valid and not under a box
        if clipped is not None:
            free = cv2.bitwise_and(free, cv2.bitwise_not(clipped))
        # Foreground in the zone and in a band around it (the part of an object that sticks out
        # of the zone) and every candidate are protected, with a margin.
        protect = cv2.bitwise_and(fg, cv2.dilate(zone, self._band_kernel))
        cand = self._cand.astype(np.uint8) * 255
        protect = cv2.dilate(cv2.bitwise_or(protect, cand), self._protect_kernel)
        is_bg = cv2.compare(raw, 0, cv2.CMP_EQ)
        # In the zone: learned pixels classified background, outside the protected foreground
        # and candidates. Outside the zone (never a debris decision) every box-free pixel
        # that is not protected follows the scene, which keeps the gain reference fresh.
        # Unknown pixels anywhere learn what they show, so they can join the model.
        known = cv2.bitwise_and(is_bg, self._learned)
        known = cv2.bitwise_or(known, cv2.bitwise_not(zone))
        known = cv2.bitwise_and(known, cv2.bitwise_not(protect))
        unknown = cv2.bitwise_not(self._learned)
        learn = cv2.bitwise_and(free, cv2.bitwise_or(known, cv2.bitwise_and(unknown, cv2.bitwise_not(cand))))
        where = learn > 0
        bg_before = self._bg
        composite = self._dithered_background()
        np.copyto(composite, comp, where=where if comp.ndim == 2 else where[..., None])
        self._mog2.apply(composite, learningRate=rate)
        self._bg = self._mog2.getBackgroundImage()
        self._bg_gray = self._grey(self._bg)
        # Residual variance of the learned background pixels, at MOG2's own rate, so the
        # dither of a pixel that gets covered later matches the variance the model has there.
        seen = where & (is_bg > 0) & (self._learned > 0)
        if seen.any() and rate > 0.0:
            d = comp[seen].astype(np.float32) - bg_before[seen].astype(np.float32)
            dist2 = (d * d).sum(axis=1) if d.ndim == 2 else d * d
            rv = self._resvar[seen]
            self._resvar[seen] = np.clip(rv + rate * (dist2 - rv), self._mog2.getVarMin(), self._mog2.getVarMax())
        # Unknown pixels join the model once they have looked like background, box-free,
        # for learn_seconds; their content is new, so it has not been seen for long.
        pending = cv2.bitwise_and(free, unknown) > 0
        looks_bg = pending & (is_bg > 0)
        self._pending[looks_bg] += dt_u
        self._pending[pending & ~looks_bg] = 0.0
        done = looks_bg & (self._pending >= float(cfg.learn_seconds))
        if done.any():
            self._learned[done] = 255
            self._pending[done] = 0.0
            self._seen_bg[done] = 0.0
        return rate

    def _dithered_background(self) -> np.ndarray:
        """The model's background image dithered by +-each pixel's residual standard deviation.

        Pasting the background exactly would feed every covered pixel a zero residual, and
        MOG2 would shrink its variance towards ``varMin`` (time constant ``tau``): a pixel
        under a car waiting for a minute would come back hypersensitive. The dither feeds the
        residual the pixel normally has (per-channel amplitude ``sqrt(resvar / channels)``,
        sign alternating in a checkerboard and between applies, so the mean does not move)
        and the pixel's variance stays where it was.
        """
        bg = self._bg
        h, w = bg.shape[:2]
        if self._dither is None or self._dither.shape != (h, w):
            yy, xx = np.mgrid[0:h, 0:w]
            self._dither = (((yy + xx) & 1) * 2 - 1).astype(np.float32)
        self._dither_phase = -self._dither_phase
        nch = 1 if bg.ndim == 2 else bg.shape[2]
        amp = np.sqrt(self._resvar / float(nch)) * (self._dither * self._dither_phase)
        out = bg.astype(np.float32) + (amp if bg.ndim == 2 else amp[..., None])
        return np.clip(np.rint(out), 0, 255).astype(np.uint8)

    def _rewarm(self, geo: Dict[str, object], t: float, reason: str, extra: Dict[str, object]) -> DebrisResult:
        """Start a fresh warm-up on this frame and report why (``last_illumination_event``)."""
        LOGGER.warning("debris background invalid (%s %s); re-warming the background model", reason, extra)
        self.reset()
        self._shape = geo["img"].shape
        self._region = geo["region"]
        self._full_hw = geo["full_hw"]
        self._t_prev = t
        res = self._warmup_step(geo, 0.0, t, True)
        event: Dict[str, object] = {"reason": reason, "timestamp": float(t),
                                    "warmup_seconds": float(self.cfg.warmup_seconds)}
        if self.cfg.warmup_frames is not None:
            event["warmup_frames"] = int(self.cfg.warmup_frames)
        event.update(extra)
        self.last_illumination_event = event
        return res

    def _to_caller(self, fg_vis: np.ndarray, stats: np.ndarray, kept: np.ndarray, area_px: int,
                   geo: Dict[str, object]) -> Tuple[np.ndarray, List[Tuple[int, int, int, int, int]], int]:
        """Map the debris mask and blobs from the internal geometry back to the caller's frame."""
        H, W = geo["out_hw"]
        h, w = geo["full_hw"]
        sx, sy = float(geo["sx"]), float(geo["sy"])
        Mi = geo["Mi"]
        scale_area = sx * sy
        if area_px <= 0:
            return np.zeros((H, W), np.uint8), [], 0
        ry0, ry1, rx0, rx1 = geo["region"]
        y0, _, x0, _ = self._roi
        y0, x0 = y0 + ry0, x0 + rx0  # timer ROI -> internal frame coordinates
        if fg_vis.shape[:2] == (h, w):
            mask = fg_vis
        else:
            mask = np.zeros((h, w), np.uint8)
            mask[ry0:ry1, rx0:rx1] = fg_vis
        inv = None
        if Mi is not None:
            inv = np.linalg.inv(Mi)
            corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
            moved = float(np.abs(cv2.perspectiveTransform(corners, inv) - corners).max())
            if moved >= 0.5:  # sub-pixel: the internal mask is already in place
                mask = cv2.warpPerspective(mask, inv, (w, h), flags=cv2.INTER_NEAREST,
                                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            else:
                inv = None
        if (H, W) != mask.shape[:2]:
            mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
        blobs: List[Tuple[int, int, int, int, int]] = []
        for lab in kept:
            x, y, bw, bh, a = (int(v) for v in stats[lab])
            x, y = x + x0, y + y0
            if inv is None and scale_area == 1.0:
                blobs.append((x, y, bw, bh, a))
                continue
            pts = np.float32([[x, y], [x + bw, y], [x + bw, y + bh], [x, y + bh]]).reshape(-1, 1, 2)
            if inv is not None:
                pts = cv2.perspectiveTransform(pts, inv)
            pts = pts.reshape(-1, 2) / np.float32([sx, sy])
            bx, by = int(math.floor(pts[:, 0].min())), int(math.floor(pts[:, 1].min()))
            ex, ey = int(math.ceil(pts[:, 0].max())), int(math.ceil(pts[:, 1].max()))
            blobs.append((bx, by, max(1, ex - bx), max(1, ey - by), int(round(a / scale_area))))
        return mask, blobs, int(round(area_px / scale_area))

    def _finish(self, res: DebrisResult) -> DebrisResult:
        if not res.held:
            self._last_result = res
        return res

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
