"""Zone-of-interest stabilisation.

The operator draws the crossing polygon once, on a *calibration* frame. A real camera does
not stay still: wind, traffic vibration, a pan-tilt bump, or a thermal drift all move the
image by a few pixels (occasionally a lot more). If the polygon stayed fixed in image
coordinates, the "zone" would slowly stop covering the tracks and every downstream
decision (in-zone test, debris mask) would silently degrade.

``ZoneStabilizer`` estimates, every frame, the transform from the calibration frame to the
current frame and warps the polygon with it. The design is deliberately conservative,
because a *wrong* zone is worse than a *stale* one:

* Sparse Lucas-Kanade optical flow (fast, sub-pixel) is the primary method. A
  forward-backward round-trip check drops tracks that drifted onto the wrong texture.
* The motion model is a 4-DOF similarity by default (``cv2.estimateAffinePartial2D``:
  translation, rotation, uniform scale). For the small pan/tilt/roll of a pole-mounted
  camera the image motion is close to a similarity, and four parameters cannot be bent to
  fit whatever minority of points survives a passing train. The 8-DOF homography the
  package used before did exactly that: on the team clips it slid the band zone by up to
  its full height (13-21 px on clip05 under 12 px synthetic shake, depending on the OpenCV
  build) and produced single-frame jumps reported as healthy (5-14 px on clip03, 45-64 px
  on clip02). The price is that a real perspective change of the view is approximated (on
  clip02's ~1 degree view shift the far end of the 640 px band is ~3 px off).
  ``motion_model: affine`` and ``homography`` remain available.
* RANSAC plus per-model sanity bounds (scale, rotation, translation, area, perspective)
  reject estimates built from too few inliers or with implausible geometry.
* An LK estimate must keep a *fraction* of the reference features as inliers
  (``min_inlier_fraction``), not just an absolute count: a minority of points locked onto
  self-similar sleeper/ballast texture can agree on a consistent but wrong flow and still
  clear ``min_inliers``. ORB, the fallback for large jumps and exposure changes, likewise
  needs ``orb_min_inlier_fraction`` of its ratio-test matches to agree.
* When LK fails and the frame's grey levels no longer match the keyframe's (dusk, an
  auto-exposure step, a calibration image taken in other light), LK is retried on the
  frame remapped to the keyframe's histogram before falling back to ORB.
* Temporal consistency: a camera does not jump tens of pixels for one frame and come back.
  An estimate that moves a vertex more than the step limit (``max_step_px`` /
  ``max_step_fraction`` of the zone thickness) from the last accepted polygon is accepted
  only when the frame itself refutes the last accepted pose (that pose no longer fits the
  tracked points: the camera really moved), or when ``confirm_frames`` consecutive
  estimates agree on the new pose. Otherwise the polygon is held and ``healthy=False``.
* Burned-in overlays (timestamp, camera name, logo) do not move with the camera. A feature
  that stays put while accepted estimates say the picture moved is learned as overlay and
  masked out of every later fit and of the gate's vote, so an overlay cannot keep a stale
  pose alive while the camera shakes.
* A dead-zone (``deadzone_px``) keeps the *reported polygon* perfectly still on a static
  camera: sub-pixel estimation noise snaps to the previous polygon. The transform handed
  to the debris path (``ZoneState.to_reference``) is never dead-zoned.
* If no estimate can be trusted the stabilizer **holds the last healthy polygon** and
  reports ``healthy=False``; it never silently snaps back to the raw reference polygon,
  which would look plausible on the overlay while pointing at the wrong pixels.
* When the scene has been static (no tracked objects, healthy estimate) for a long time the
  keyframe LK tracks from is re-anchored to the current frame, so slow lighting change does
  not erode the feature set. The calibration keyframe is kept: each re-anchor is checked
  with a *direct* estimate against it and anchored on that estimate, so the pose does not
  random-walk across re-anchors (a re-anchor that disagrees with the calibration frame is
  refused; one that cannot be checked is counted in :meth:`ZoneStabilizer.health`).
* A featureless first frame (black, uniform grey, lens cap) is not used as the reference:
  the stabilizer stays ``"uninitialized"`` and retries on the next frame.
* Health changes are reported (``health_reports``, turned into HEALTH events by the
  monitor) only once they have lasted ``health_report_seconds``, so a flickering estimate
  in deep low light does not flood the alert sinks; ``ZoneState.healthy`` stays per frame.

Calibration geometry (``zone.reference_size`` / ``zone.reference_image``) is handled by
:func:`build_stabilizer`, which the monitor calls on its first frame: the polygon is scaled
from calibration pixels to processing pixels (refused when the aspect ratio differs) and a
polygon that mostly lies outside the frame is reported. A stored calibration image becomes
the calibration keyframe: live frames are *registered* against it, so a camera that moved
between calibration and start-up is corrected on the first live frame, and the first
registered live frame becomes the keyframe LK tracks from. When no live frame can be
registered within ``registration_timeout_seconds`` (night against a daylight image, a
changed scene) the stabilizer tracks from a live frame at the calibration pose, says so in
a HEALTH report, and registers later, at a keyframe refresh, when it can.

The legacy prototype (``scripts/pipeline_helpers.py::stabilize_zoi``) did LK + RANSAC with
no health check and fell back to the raw reference polygon; this module replaces it.
"""
from __future__ import annotations

import dataclasses
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .config import ZoneConfig
from .geometry import as_polygon, polygon_mask, transform_points
from .types import ZoneState

LOGGER = logging.getLogger(__name__)

# Forward-backward round-trip tolerance for LK tracks, in pixels. A track whose
# reference -> current -> reference round trip lands more than this far from where it
# started has drifted onto a different texture and must not vote in RANSAC.
FB_MAX_ERROR_PX = 1.0
MOTION_MODELS = ("similarity", "affine", "homography")
# A frame whose grey levels vary less than this (standard deviation) is treated as
# featureless whatever goodFeaturesToTrack says: its quality threshold is *relative* to
# the strongest corner, so a black frame with sensor noise still yields "corners".
MIN_TEXTURE_STD = 3.0
# A big step is taken without waiting for confirmation only when the frame refutes the
# last accepted pose: that pose fails the acceptance bar on this frame's correspondences,
# fits at most REFUTE_MAX_FRACTION of them, and the new estimate fits at least
# REFUTE_SUPPORT_RATIO times as many. Measured on the team clips: under 12 px synthetic
# shake the old pose keeps p90 < 1% (max 30%) of the correspondences, the real clip02
# camera shift 16%; the homography's single-frame errors on clip02/clip03 keep 74-94%.
REFUTE_MAX_FRACTION = 0.25
REFUTE_SUPPORT_RATIO = 2.0
# Start-up check: report a polygon with less than this share of its area inside the frame.
MIN_POLYGON_INSIDE_FRACTION = 0.9
# Calibration and processed frames must have the same aspect ratio within this tolerance.
ASPECT_TOLERANCE = 0.01
# Photometric matching: when LK fails and the frame's grey-level distribution (outside the
# zone) differs from the keyframe's by more than this many levels somewhere in its body
# (1st-99th percentile), LK is retried on the frame remapped to the keyframe's histogram.
PHOTOMETRIC_TOLERANCE = 4
# Static overlays: a keyframe feature that moved at most OVERLAY_STILL_PX while an accepted
# estimate says it should have moved at least OVERLAY_MIN_MOTION_PX, on OVERLAY_MIN_HITS
# frames, is part of a burned-in overlay and is masked out from then on. At most
# OVERLAY_MAX_FRACTION of a keyframe's features are ever masked: more than that is a run of
# wrong estimates, not an overlay.
OVERLAY_STILL_PX = 0.5
OVERLAY_MIN_MOTION_PX = 2.0
OVERLAY_MIN_HITS = 3
OVERLAY_MAX_FRACTION = 0.5
# The repository root: a relative zone.reference_image that is not found from the working
# directory is also looked up here (configs written by tools/calibrate_zone.py from the
# repository root keep working when the monitor is started elsewhere, e.g. by systemd).
_REPO_ROOT = Path(__file__).resolve().parents[1]


class ZoneGeometryError(ValueError):
    """The zone calibration cannot be mapped onto the processed frames.

    Raised by :func:`build_stabilizer` when the stream's aspect ratio differs from the
    calibration frame's, or when ``zone.reference_size`` is malformed. A polygon scaled
    onto a differently shaped picture would put the zone in the wrong place silently, so
    the only safe answer is to recalibrate (the monitor CLI exits with status 2).
    """


def _to_gray(frame: np.ndarray) -> np.ndarray:
    """Convert a BGR/BGRA/gray frame to a uint8 single-channel image.

    Optical flow and ORB both require 8-bit single-channel input; accepting an already
    grayscale frame keeps the unit tests and headless tools simple.
    """
    if frame is None or frame.ndim not in (2, 3):
        raise ValueError("frame must be an HxW or HxWxC image")
    if frame.ndim == 3:
        channels = frame.shape[2]
        if channels == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif channels == 4:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
        elif channels == 1:
            gray = frame[:, :, 0]
        else:
            raise ValueError(f"unsupported channel count {channels}")
    else:
        gray = frame
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(gray)


def _max_vertex_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Largest distance between corresponding vertices of two (N,2) polygons."""
    d = np.asarray(a, dtype=np.float64).reshape(-1, 2) - np.asarray(b, dtype=np.float64).reshape(-1, 2)
    return float(np.sqrt((d * d).sum(axis=1)).max())


def _polygon_area(polygon: np.ndarray) -> float:
    """Shoelace area of an (N,2) polygon (float64, no rasterisation)."""
    p = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _grey_levels(gray: np.ndarray, mask: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Grey-level probabilities (256,) of ``gray`` where ``mask`` != 0 (None if empty)."""
    hist = cv2.calcHist([gray], [0], mask, [256], [0, 256]).ravel().astype(np.float64)
    total = float(hist.sum())
    if total <= 0:
        return None
    return hist / total


def zone_thickness(polygon: np.ndarray) -> float:
    """Typical thickness of the zone: ``2 * area / perimeter``.

    For a band-shaped zone (the usual crossing: long, 20-30 px tall) this is the height
    of the band; for a compact shape it is about half the width. It scales the per-frame
    step limit of the temporal-consistency gate.
    """
    p = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
    perimeter = float(np.sqrt(((p - np.roll(p, -1, axis=0)) ** 2).sum(axis=1)).sum())
    return 2.0 * _polygon_area(p) / perimeter if perimeter > 0 else 0.0


def polygon_inside_fraction(polygon: np.ndarray, frame_size: Tuple[int, int]) -> float:
    """Share of the polygon's area that lies inside a ``frame_size = (w, h)`` frame.

    Exact (Sutherland-Hodgman clipping against the frame rectangle, valid for any simple
    polygon), so a thin band along the frame edge is not penalised by rasterisation.
    """
    w, h = float(frame_size[0]), float(frame_size[1])
    pts = [tuple(p) for p in np.asarray(polygon, dtype=np.float64).reshape(-1, 2)]
    total = _polygon_area(np.asarray(pts))
    if total <= 0:
        return 0.0
    # (axis, bound, keep_greater): x >= 0, x <= w, y >= 0, y <= h
    for axis, bound, keep_greater in ((0, 0.0, True), (0, w, False), (1, 0.0, True), (1, h, False)):
        if not pts:
            break
        out: List[Tuple[float, float]] = []
        for i, cur in enumerate(pts):
            prev = pts[i - 1]
            cur_in = cur[axis] >= bound if keep_greater else cur[axis] <= bound
            prev_in = prev[axis] >= bound if keep_greater else prev[axis] <= bound
            if cur_in != prev_in:
                t = (bound - prev[axis]) / (cur[axis] - prev[axis])
                out.append((prev[0] + t * (cur[0] - prev[0]), prev[1] + t * (cur[1] - prev[1])))
            if cur_in:
                out.append(cur)
        pts = out
    if len(pts) < 3:
        return 0.0
    return min(1.0, _polygon_area(np.asarray(pts)) / total)


def calibration_scale(reference_size: Sequence[float], processed_size: Sequence[float]) -> Tuple[float, float]:
    """``(sx, sy)`` mapping calibration pixels to processed pixels.

    Raises :class:`ZoneGeometryError` when the two frames do not have the same aspect
    ratio (within ``ASPECT_TOLERANCE``): stretching the polygon onto a picture with another
    shape (a 4:3 substream of a 16:9 camera, a cropped stream) cannot be right.
    """
    rw, rh = float(reference_size[0]), float(reference_size[1])
    pw, ph = float(processed_size[0]), float(processed_size[1])
    if rw <= 0 or rh <= 0 or pw <= 0 or ph <= 0:
        raise ZoneGeometryError(f"invalid frame sizes: calibration {reference_size}, processed {processed_size}")
    sx, sy = pw / rw, ph / rh
    if abs(sx / sy - 1.0) > ASPECT_TOLERANCE:
        raise ZoneGeometryError(
            f"the zone was calibrated on a {rw:.0f}x{rh:.0f} frame (aspect {rw / rh:.3f}) but the processed "
            f"frames are {pw:.0f}x{ph:.0f} (aspect {pw / ph:.3f}); the polygon cannot be mapped onto a "
            f"differently shaped picture - recalibrate with tools/calibrate_zone.py on this stream")
    return sx, sy


def resolve_reference_path(path: str) -> Path:
    """Where ``zone.reference_image`` is: as given (absolute, or relative to the working
    directory like every other path in the config), else relative to the repository root."""
    p = Path(str(path))
    if p.is_absolute() or p.exists():
        return p
    alt = _REPO_ROOT / p
    return alt if alt.exists() else p


def load_reference_image(path: str) -> np.ndarray:
    """Read the clean calibration frame ``zone.reference_image`` (BGR); raise if unreadable."""
    resolved = resolve_reference_path(path)
    img = cv2.imread(str(resolved), cv2.IMREAD_COLOR)
    if img is None:
        raise ZoneGeometryError(f"cannot read zone.reference_image {str(path)!r} (looked for {str(resolved)!r})")
    return img


def build_stabilizer(cfg: ZoneConfig, processed_size: Tuple[int, int], default_reference_size: Tuple[int, int],
                     timestamp: float = 0.0) -> Tuple["ZoneStabilizer", Tuple[float, float], List[str]]:
    """Create the stabilizer for frames of ``processed_size = (w, h)``.

    The calibration frame size is, in order: ``cfg.reference_size``, the size of
    ``cfg.reference_image``, else ``default_reference_size`` (the monitor passes the
    native size of the first frame it ever saw, so a live re-initialisation after the
    stream renegotiated its resolution still scales the polygon from calibration pixels).
    The polygon is scaled by ``processed / calibration`` on both axes. When
    ``cfg.reference_image`` is set (and ``cfg.stabilize``) it is resized to the processed
    size and becomes the calibration keyframe that live frames are registered against.

    Returns ``(stabilizer, (sx, sy), problems)``; ``problems`` are human-readable
    start-up issues that do not stop monitoring (polygon mostly outside the frame, a
    missing, unreadable or featureless calibration image). Raises
    :class:`ZoneGeometryError` for an aspect-ratio mismatch or a malformed
    ``reference_size``.
    """
    pw, ph = int(processed_size[0]), int(processed_size[1])
    problems: List[str] = []
    image: Optional[np.ndarray] = None
    if cfg.reference_image:
        try:
            image = load_reference_image(cfg.reference_image)
        except ZoneGeometryError as exc:
            # An optional input must not stop the crossing being watched: say so loudly and
            # anchor on the live frame, exactly as without a reference image.
            fallback = ("the stabilizer anchors on the first textured live frame instead (camera motion "
                        "since calibration is not corrected)" if cfg.stabilize else
                        "it is not needed while zone.stabilize is off")
            problems.append(
                f"{exc}; {fallback}"
                + ("" if cfg.reference_size is not None else
                   " and the polygon is read in the pixels of the first frame (zone.reference_size is not set)"))
    if cfg.reference_size is not None:
        try:
            size = [float(v) for v in cfg.reference_size]
        except (TypeError, ValueError):
            size = []
        if len(size) != 2 or not all(math.isfinite(v) and v > 0 for v in size):
            raise ZoneGeometryError(f"zone.reference_size must be [width, height] in pixels, got {cfg.reference_size!r}")
        ref_size: Tuple[float, float] = (size[0], size[1])
        origin = "zone.reference_size"
        if image is not None:
            calibration_scale(ref_size, (image.shape[1], image.shape[0]))  # same picture shape, or refuse
    elif image is not None:
        ref_size = (float(image.shape[1]), float(image.shape[0]))
        origin = "zone.reference_image"
    else:
        ref_size = (float(default_reference_size[0]), float(default_reference_size[1]))
        origin = "first frame; zone.reference_size not set"
    sx, sy = calibration_scale(ref_size, (pw, ph))
    polygon = [[float(x) * sx, float(y) * sy] for x, y in cfg.polygon]
    stab = ZoneStabilizer(dataclasses.replace(cfg, polygon=polygon))

    inside = polygon_inside_fraction(np.asarray(polygon, dtype=np.float64), (pw, ph))
    if inside < MIN_POLYGON_INSIDE_FRACTION:
        problems.append(
            f"zone polygon is only {100.0 * inside:.0f}% inside the {pw}x{ph} frame (polygon read as "
            f"{ref_size[0]:.0f}x{ref_size[1]:.0f} pixels: {origin}); the monitor is watching the wrong region - "
            f"recalibrate with tools/calibrate_zone.py (the built-in default polygon belongs to the FOX13 "
            f"stress-test clip)")
    if image is not None and cfg.stabilize:
        if image.shape[1] != pw or image.shape[0] != ph:
            interp = cv2.INTER_AREA if image.shape[1] > pw else cv2.INTER_LINEAR
            image = cv2.resize(image, (pw, ph), interpolation=interp)
        if not stab.initialize_reference(image, timestamp):
            problems.append(
                f"zone.reference_image {cfg.reference_image!r} has too few trackable features; the "
                f"stabilizer will anchor on the first textured live frame instead (camera motion since "
                f"calibration is then not corrected)")
    return stab, (sx, sy), problems


class _Keyframe:
    """A frame the stabilizer tracks from, with its pose relative to the calibration frame."""

    __slots__ = ("gray", "pts", "mask", "orb", "pose", "timestamp", "cdf", "static_hits", "overlay_marked")

    def __init__(self, gray: np.ndarray, pts: np.ndarray, mask: Optional[np.ndarray], pose: np.ndarray,
                 timestamp: float) -> None:
        self.gray = gray
        self.pts = pts  # (M,1,2) float32 Shi-Tomasi corners
        self.mask = mask  # where features may be detected (zone and overlays excluded), or None
        self.orb: Optional[Tuple[list, Optional[np.ndarray]]] = None  # ORB keypoints/descriptors, lazily
        self.pose = pose  # 3x3 float64: calibration pixels -> this keyframe's pixels
        self.timestamp = float(timestamp)
        self.cdf: Optional[np.ndarray] = None  # grey-level CDF outside the zone, lazily
        self.static_hits = np.zeros(int(pts.shape[0]), dtype=np.int32)  # static-overlay evidence per corner
        self.overlay_marked = 0  # corners of this keyframe learned as static overlay


class _Estimate:
    """One frame's candidate transform and the correspondences behind it."""

    __slots__ = ("method", "M", "pose", "inliers", "src", "dst", "idx", "kf")

    def __init__(self, method: str, M: np.ndarray, pose: np.ndarray, inliers: int, src: np.ndarray,
                 dst: np.ndarray, idx: Optional[np.ndarray] = None, kf: Optional[_Keyframe] = None) -> None:
        self.method = method  # "lk" | "orb"
        self.M = M  # 3x3 keyframe -> current
        self.pose = pose  # 3x3 calibration -> current
        self.inliers = int(inliers)
        self.src = src  # (K,2) keyframe points that took part in the fit
        self.dst = dst  # (K,2) their positions in the current frame
        self.idx = idx  # LK: index of each correspondence in kf.pts
        self.kf = kf  # the keyframe the estimate is relative to


class ZoneStabilizer:
    """Track the operator-drawn crossing polygon through camera motion.

    Usage::

        stab = ZoneStabilizer(cfg.zone)
        stab.initialize(first_frame, t0)       # optional: update() initialises on its first frame
        state = stab.update(frame, t, scene_static=not tracker.zone_occupied)
        for text in stab.health_reports:       # debounced health changes, registration notices
            log_health_event(text, stab.health())

    ``cfg.polygon`` is in the pixels of the frames passed in (the monitor scales it with
    :func:`build_stabilizer`). ``state.polygon`` is the polygon in the current frame;
    ``state.healthy`` says whether it was freshly estimated (``lk``/``orb``) or merely held
    from the last good estimate; ``state.to_reference`` maps current-frame pixels to
    calibration-frame pixels (for registering the debris background).
    """

    def __init__(self, cfg: ZoneConfig):
        self.cfg = cfg
        model = str(getattr(cfg, "motion_model", "similarity") or "similarity").strip().lower()
        if model not in MOTION_MODELS:
            raise ValueError(f"unknown zone.motion_model {cfg.motion_model!r} (valid: {list(MOTION_MODELS)})")
        self._model = model
        # Calibration polygon, in the pixels of the processed frames (constant).
        self._polygon0 = as_polygon(cfg.polygon).astype(np.float64)
        self._thickness = zone_thickness(self._polygon0)
        self._shape: Optional[Tuple[int, int]] = None  # (h, w) every frame must have
        self._reference: Optional[_Keyframe] = None  # calibration keyframe; never replaced
        self._keyframe: Optional[_Keyframe] = None  # keyframe LK tracks from (re-anchored on refresh)
        # The calibration keyframe is zone.reference_image: live frames are registered
        # against it, and _keyframe stays None until one is (or the timeout passes).
        self._reference_from_image = False
        # The tracked pose is tied to the calibration keyframe by an estimate (always true
        # when the calibration keyframe is a live frame).
        self._registered = True
        self._registration_started: Optional[float] = None
        self._overlay: Optional[np.ndarray] = None  # HxW uint8, 255 = learned static overlay
        self._overlay_features = 0
        self._orb = None
        self._bf = None
        self._reset_pose()
        # Last estimate bookkeeping (drives hold + health reporting). Healthy until shown
        # otherwise, so that a first frame that cannot be used raises a HEALTH flip.
        self._last_method: str = "uninitialized"
        self._last_inliers: int = 0
        self._last_healthy: bool = True
        self._last_reason: str = ""
        self._last_timestamp: float = 0.0
        self._consecutive_unhealthy: int = 0
        self._static_since: Optional[float] = None
        self._reanchors = 0
        self._unverified_reanchors = 0
        self._refused_reanchors = 0
        self._drift_px: Optional[float] = None
        # Debounced health reporting: the state last reported, and since when the current
        # per-frame state has held.
        self._reported_healthy = True
        self._raw_healthy = True
        self._raw_since: Optional[float] = None
        # True on the update where ``healthy`` flipped (per frame, not debounced).
        self.health_changed: bool = False
        # HEALTH texts for this update: a health change that has lasted health_report_seconds,
        # reference-image registration notices. Reset by every initialize()/update().
        self.health_reports: List[str] = []

    def _reset_pose(self) -> None:
        self._pose = np.eye(3)  # last accepted calibration -> current transform
        self._to_reference = np.eye(3)  # its inverse
        self._polygon = self._polygon0.copy()  # last accepted polygon (raw), current frame
        self._shown = self._polygon0.copy()  # reported polygon (dead-zone applied)
        self._pending: Optional[np.ndarray] = None  # unconfirmed jump candidate polygon
        self._pending_count = 0
        self._mask_key: Optional[bytes] = None
        self._mask: Optional[np.ndarray] = None
        self._mask_area = 0

    # ------------------------------------------------------------------ properties
    @property
    def initialized(self) -> bool:
        """True once a calibration keyframe (a textured live frame or reference image) is stored."""
        return self._reference is not None

    @property
    def registered(self) -> bool:
        """False while the pose is not tied to the calibration frame by an estimate: before
        the first live frame is registered against ``zone.reference_image``, and after the
        registration timed out (tracking then starts at the calibration pose)."""
        return self._registered

    @property
    def motion_model(self) -> str:
        return self._model

    @property
    def reference_polygon(self) -> np.ndarray:
        """The polygon in the coordinates of the *current reference keyframe*, (N,2) float32.

        It equals the calibration polygon until the keyframe is re-anchored (keyframe
        refresh, registration of a live frame against a reference image), after which it is
        the calibration polygon warped by the keyframe's pose.
        """
        if self._keyframe is None:
            return as_polygon(self.cfg.polygon)
        return transform_points(self._polygon0, self._keyframe.pose).astype(np.float32)

    @property
    def reference_features(self) -> np.ndarray:
        """Shi-Tomasi corners tracked from the keyframe, (M,2) float32. Exposed for tests/overlays."""
        if self._keyframe is None:
            return np.zeros((0, 2), dtype=np.float32)
        return self._keyframe.pts.reshape(-1, 2).copy()

    def max_step_px(self) -> float:
        """Per-frame vertex step above which an estimate needs confirmation (0 = gate off).

        ``max(max_step_px, max_step_fraction * thickness)``, where the thickness is that of
        the calibration polygon in processed pixels (:func:`zone_thickness`).
        """
        floor = float(getattr(self.cfg, "max_step_px", 0.0) or 0.0)
        frac = float(getattr(self.cfg, "max_step_fraction", 0.0) or 0.0)
        return max(floor, frac * self._thickness)

    # ------------------------------------------------------------------ public API
    def initialize(self, frame_bgr: np.ndarray, timestamp: float) -> ZoneState:
        """Store ``frame_bgr`` as the calibration keyframe; return the identity state.

        Called once at start-up with the first live frame (see :meth:`initialize_reference`
        for a stored calibration image). Features inside the polygon are discarded when
        ``cfg.exclude_features_in_zone`` because anything in the zone is likely to move
        (vehicles, trains) and would corrupt a background motion estimate. A featureless
        frame is *not* stored: the returned state has ``method="uninitialized"`` and the
        next :meth:`update` tries again on its own frame.
        """
        self.health_reports = []
        gray = _to_gray(frame_bgr)
        self._reference = None
        self._keyframe = None
        self._reference_from_image = False
        self._registered = True
        self._registration_started = None
        if self._shape is not None and gray.shape[:2] != self._shape:
            self._overlay = None  # learned overlays belong to the old geometry
            self._overlay_features = 0
        self._shape = gray.shape[:2]
        self._static_since = None
        # Re-anchor bookkeeping describes the previous calibration keyframe.
        self._reanchors = self._unverified_reanchors = self._refused_reanchors = 0
        self._drift_px = None
        return self._try_initialize(gray, float(timestamp))

    def initialize_reference(self, image_bgr: np.ndarray, timestamp: float = 0.0) -> bool:
        """Use a stored calibration image (``zone.reference_image``) as the calibration keyframe.

        Live frames are then registered against it by :meth:`update` (LK, histogram-matched
        LK, ORB, through the temporal gate); the first registered live frame becomes the
        keyframe LK tracks from, so the image's lighting matters once, not on every frame.
        Returns False, and leaves the stabilizer uninitialised (it anchors on the first
        textured live frame), when the image is featureless.
        """
        self.initialize(image_bgr, timestamp)
        if not self.initialized:
            return False
        self._keyframe = None
        self._reference_from_image = True
        self._registered = False
        return True

    def update(self, frame_bgr: np.ndarray, timestamp: float, scene_static: bool = False) -> ZoneState:
        """Estimate where the polygon is in ``frame_bgr`` and return the :class:`ZoneState`.

        ``scene_static`` should be True only when the caller knows nothing is moving in the
        zone (no tracked objects, no debris trigger); it gates keyframe refresh. Calling
        ``update`` before ``initialize`` initialises on this frame and returns the
        identity state (or the uninitialised state while frames are featureless), so a
        monitor loop can call ``update`` uniformly from frame 0. HEALTH texts for this
        frame are left in :attr:`health_reports`.

        Raises ``ValueError`` if the frame size differs from the first frame's: the
        polygon is in those pixels, so a silently resized frame would put the zone in the
        wrong place.
        """
        self.health_reports = []
        gray = _to_gray(frame_bgr)
        if self._shape is None:
            self._shape = gray.shape[:2]
        elif gray.shape[:2] != self._shape:
            raise ValueError(
                f"frame size {gray.shape[1]}x{gray.shape[0]} does not match the reference "
                f"keyframe {self._shape[1]}x{self._shape[0]}; resize before update()"
            )
        timestamp = float(timestamp)
        if not self.initialized:
            return self._try_initialize(gray, timestamp)
        self._last_timestamp = timestamp
        prev_healthy = self._last_healthy

        if not self.cfg.stabilize:
            # Stabilisation disabled: the zone is the calibration polygon, always "healthy".
            self._record(method="identity", inliers=0, healthy=True, reason="")
            return self._finish(None, 0, "identity", True, prev_healthy, timestamp)

        if self._keyframe is None:
            # Calibrated on zone.reference_image: register this live frame against it.
            return self._register_live(gray, timestamp, prev_healthy)

        # 1. LK (then histogram-matched LK, then ORB) against the keyframe, with inlier and
        #    sanity checks.
        est, lk_inliers, reason = self._estimate(self._keyframe, gray)
        if est is None:
            # A frame without any candidate breaks a run of agreeing jump candidates.
            self._pending = None
            self._pending_count = 0
        else:
            # 2. Temporal consistency against the last accepted pose.
            est, reason = self._temporal_gate(est, self._keyframe)
        if est is not None:
            self._commit(est.pose)
            self._learn_overlay(est)
            method, inliers, healthy, H = est.method, est.inliers, True, est.M
        else:
            # 3. Hold. We deliberately keep the last polygon that passed the checks rather
            #    than snapping to the raw reference polygon: the camera may still be
            #    displaced, so the reference polygon is not "safe".
            method, inliers, healthy, H = "held", lk_inliers, False, None
        self._record(method=method, inliers=int(inliers), healthy=healthy, reason=reason)

        # 4. Keyframe refresh when the scene has been static for long enough.
        if self._maybe_refresh_keyframe(gray, timestamp, scene_static, healthy):
            H = np.eye(3)  # this frame is the new keyframe
        return self._finish(H, int(inliers), method, healthy, prev_healthy, timestamp)

    def health(self) -> Dict[str, object]:
        """Summary of the last estimate for logging and HEALTH events.

        Keys: ``method`` (uninitialized/identity/lk/orb/held), ``inliers`` (RANSAC inlier
        count of the accepted or attempted LK estimate), ``healthy``, ``keyframe_age_s``
        (seconds since the keyframe was (re-)anchored), ``consecutive_unhealthy`` (updates
        in a row that had to hold the polygon), ``initialized``, ``motion_model``,
        ``reason`` (why the last update held, "" when healthy), ``offset_px`` (largest
        vertex distance between the reported polygon and the calibration polygon: how far
        the camera has moved since calibration), ``reanchors`` (keyframe re-anchors so
        far), ``unverified_reanchors`` (re-anchors since the last one that could be checked
        against the calibration frame - drift can accumulate only across these),
        ``refused_reanchors``, ``drift_px`` (discrepancy between the chained and the
        direct calibration estimate at the last check, None before the first),
        ``registered`` (see :attr:`registered`) and ``overlay_features`` (keyframe corners
        learned as static overlay and masked out).
        """
        kf_age = float(self._last_timestamp - self._keyframe.timestamp) if self._keyframe is not None else 0.0
        return {
            "method": self._last_method,
            "inliers": int(self._last_inliers),
            "healthy": bool(self._last_healthy),
            "keyframe_age_s": kf_age,
            "consecutive_unhealthy": int(self._consecutive_unhealthy),
            "initialized": bool(self.initialized),
            "motion_model": self._model,
            "reason": self._last_reason,
            "offset_px": round(_max_vertex_distance(self._shown, self._polygon0), 2),
            "reanchors": int(self._reanchors),
            "unverified_reanchors": int(self._unverified_reanchors),
            "refused_reanchors": int(self._refused_reanchors),
            "drift_px": None if self._drift_px is None else round(float(self._drift_px), 3),
            "registered": bool(self._registered),
            "overlay_features": int(self._overlay_features),
        }

    def min_lk_inliers(self) -> int:
        """Inliers an LK estimate needs: ``max(min_inliers, min_inlier_fraction * n_features)``.

        Why a fraction and not only an absolute count: LK on a rail crossing tracks a few
        hundred corners, many of them on self-similar texture (sleepers, ballast, lane
        markings). A subset of those can slide by the same amount onto the neighbouring
        sleeper, agree with each other, and give RANSAC a consistent *wrong* transform
        with, say, 45 inliers - comfortably above the absolute ``min_inliers`` of 30 - while
        the other 355 features disagree. Requiring a share of the reference features makes
        that impossible: a transform that most of the background does not support is
        refused, the stabilizer tries ORB and otherwise holds the last healthy polygon,
        which is the conservative failure the module is built around.

        The feature count is that of the current keyframe (corners learned as static
        overlay excluded), so the bar is stable between refreshes. ``min_inlier_fraction =
        0`` restores the old absolute-only rule.
        """
        n_ref = int(self._usable(self._keyframe).sum()) if self._keyframe is not None else 0
        return self._min_inliers_for(n_ref)

    # ------------------------------------------------------------------ internals: setup
    def _min_inliers_for(self, n_ref: int) -> int:
        frac = float(getattr(self.cfg, "min_inlier_fraction", 0.0) or 0.0)
        needed = int(math.ceil(frac * n_ref)) if frac > 0.0 and n_ref > 0 else 0
        return max(int(self.cfg.min_inliers), needed)

    def _min_features(self) -> int:
        """Corners a keyframe needs: LK can never reach ``min_inliers`` with fewer."""
        return max(4, int(self.cfg.min_inliers))

    def _texture_problem(self, gray: np.ndarray, kf: _Keyframe) -> str:
        """Why ``gray`` cannot be a keyframe ("" when it has enough texture)."""
        n = int(kf.pts.shape[0])
        std = float(cv2.meanStdDev(gray)[1][0][0])
        if std < MIN_TEXTURE_STD or n < self._min_features():
            return (f"frame too featureless to anchor on ({n} corners, grey-level std {std:.1f}; "
                    f"need {self._min_features()} corners and std >= {MIN_TEXTURE_STD:.0f})")
        return ""

    def _try_initialize(self, gray: np.ndarray, timestamp: float) -> ZoneState:
        """Anchor the calibration keyframe on ``gray`` unless it is featureless."""
        prev_healthy = self._last_healthy
        self._last_timestamp = timestamp
        self._reset_pose()
        kf = self._make_keyframe(gray, np.eye(3), timestamp)
        if self.cfg.stabilize:
            problem = self._texture_problem(gray, kf)
            if problem:
                # Black/uniform frame (stream start, fade-in, lens cap): nothing to register
                # against. Anchoring here would hold an unregistrable reference forever.
                self._record(method="uninitialized", inliers=0, healthy=False, reason=f"{problem}; retrying")
                return self._finish(None, 0, "uninitialized", False, prev_healthy, timestamp)
        self._reference = kf
        self._keyframe = kf
        self._static_since = None
        self._record(method="identity", inliers=0, healthy=True, reason="")
        return self._finish(None, 0, "identity", True, prev_healthy, timestamp)

    def _make_keyframe(self, gray: np.ndarray, pose: np.ndarray, timestamp: float) -> _Keyframe:
        """Detect the corners of a (candidate) keyframe whose pose is ``pose``."""
        if self.cfg.exclude_features_in_zone or self._overlay is not None:
            # goodFeaturesToTrack only searches where mask != 0: allow everywhere except the
            # zone (when excluded) and the learned static overlays.
            mask = np.full(gray.shape[:2], 255, dtype=np.uint8)
            if self.cfg.exclude_features_in_zone:
                poly = transform_points(self._polygon0, pose)
                cv2.fillPoly(mask, [np.round(poly).astype(np.int32).reshape(-1, 1, 2)], 0)
            if self._overlay is not None:
                mask[self._overlay > 0] = 0
        else:
            mask = None
        pts = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=int(self.cfg.max_corners),
            qualityLevel=float(self.cfg.quality_level),
            minDistance=int(self.cfg.min_distance),
            mask=mask,
            blockSize=int(self.cfg.block_size),
        )
        if pts is None:  # uniform image: nothing to track
            pts = np.zeros((0, 1, 2), dtype=np.float32)
        return _Keyframe(gray, np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2), mask,
                         np.asarray(pose, dtype=np.float64), timestamp)

    def _record(self, method: str, inliers: int, healthy: bool, reason: str) -> None:
        self._last_method = method
        self._last_inliers = int(inliers)
        self._last_healthy = bool(healthy)
        self._last_reason = reason
        if healthy:
            self._consecutive_unhealthy = 0
        else:
            self._consecutive_unhealthy += 1

    def _finish(self, H: Optional[np.ndarray], inliers: int, method: str, healthy: bool, prev_healthy: bool,
                timestamp: float) -> ZoneState:
        """Common tail of every update: flip flag, debounced health report, the state."""
        self.health_changed = prev_healthy != healthy
        self._report_health(timestamp)
        return self._make_state(H, int(inliers), method, healthy)

    # ------------------------------------------------------------------ internals: health reports
    def _notify(self, text: str) -> None:
        self.health_reports.append(text)

    def _report_health(self, timestamp: float) -> None:
        """Report a change of ``healthy`` once it has lasted ``health_report_seconds``.

        Single held frames (a train edge, a decoder glitch) and an estimate that flickers
        in deep low light then produce no HEALTH traffic, while a zone that stays held - or
        stays recovered - is reported. ``health_report_seconds = 0`` reports every flip.
        """
        healthy = self._last_healthy
        if healthy != self._raw_healthy or self._raw_since is None or timestamp < self._raw_since:
            self._raw_healthy = healthy
            self._raw_since = timestamp
        if healthy == self._reported_healthy:
            return
        lasted = timestamp - self._raw_since
        if lasted + 1e-9 < max(0.0, float(getattr(self.cfg, "health_report_seconds", 0.0) or 0.0)):
            return
        self._reported_healthy = healthy
        method, inliers = self._last_method, self._last_inliers
        if healthy:
            state = "healthy"
        elif method == "uninitialized":
            state = "uninitialised (no usable reference frame yet; retrying)"
        else:
            state = "unhealthy (holding last good polygon)"
        if lasted > 0:
            state += f" for {lasted:.2f} s"
        detail = f" - {self._last_reason}" if self._last_reason else ""
        self._notify(f"zone {state}: {method} {inliers} inliers{detail}")

    # ------------------------------------------------------------------ internals: registration
    def _register_live(self, gray: np.ndarray, timestamp: float, prev_healthy: bool) -> ZoneState:
        """Register a live frame against the calibration image (``zone.reference_image``).

        An estimate that passes every check and the temporal gate (from the calibration
        pose: a camera that moved since calibration refutes it at once) registers the
        stream; this frame then becomes the keyframe LK tracks from, so later frames do not
        depend on the image's lighting. Until then the zone is held at the calibration
        pose (``healthy=False``). After ``registration_timeout_seconds`` without success
        the stabilizer tracks from a textured live frame at the calibration pose
        (``registered=False``, reported once), as it would without a reference image, and
        registers at a later keyframe refresh if a direct estimate then succeeds.
        """
        if self._registration_started is None:
            self._registration_started = timestamp
        elapsed = max(0.0, timestamp - self._registration_started)
        est, lk_inliers, reason = self._estimate(self._reference, gray)
        if est is not None:
            est, reason = self._temporal_gate(est, self._reference)
        if est is not None:
            self._commit(est.pose)
            self._registered = True
            kf = self._make_keyframe(gray, est.pose, timestamp)
            H = est.M
            if not self._texture_problem(gray, kf):
                self._keyframe = kf  # track from this live frame from now on
                H = np.eye(3)
            self._record(method=est.method, inliers=est.inliers, healthy=True, reason="")
            return self._finish(H, est.inliers, est.method, True, prev_healthy, timestamp)
        timeout = max(0.0, float(getattr(self.cfg, "registration_timeout_seconds", 0.0) or 0.0))
        if elapsed >= timeout:
            # Give up on the image for now: track from this live frame at the last accepted
            # pose (the calibration pose unless an earlier frame registered).
            kf = self._make_keyframe(gray, self._pose, timestamp)
            if not self._texture_problem(gray, kf):
                self._keyframe = kf
                self._pending = None
                self._pending_count = 0
                if not self._registered:
                    message = (f"zone not registered against zone.reference_image after {elapsed:.1f} s "
                               f"({reason}); tracking from a live frame at the calibration pose - camera motion "
                               f"since calibration is NOT corrected; registration is retried at every keyframe "
                               f"refresh")
                    LOGGER.warning("%s", message)
                    self._notify(message)
                self._record(method="identity", inliers=0, healthy=True, reason="")
                return self._finish(None, 0, "identity", True, prev_healthy, timestamp)
        reason = f"registering against zone.reference_image ({elapsed:.1f} of {timeout:.1f} s): {reason}"
        self._record(method="held", inliers=lk_inliers, healthy=False, reason=reason)
        return self._finish(None, lk_inliers, "held", False, prev_healthy, timestamp)

    # ------------------------------------------------------------------ internals: estimation
    def _fit(self, src: np.ndarray, dst: np.ndarray) -> Tuple[Optional[np.ndarray], int, Optional[np.ndarray]]:
        """RANSAC fit of the motion model src->dst; returns (3x3 or None, inliers, inlier mask)."""
        if src.shape[0] < 4:
            return None, 0, None
        thr = float(self.cfg.ransac_reproj_threshold)
        if self._model == "homography":
            M, mask = cv2.findHomography(src, dst, cv2.RANSAC, thr)
            if M is None:
                return None, 0, None
            M = np.asarray(M, dtype=np.float64)
        else:
            estimate = cv2.estimateAffinePartial2D if self._model == "similarity" else cv2.estimateAffine2D
            A, mask = estimate(src, dst, method=cv2.RANSAC, ransacReprojThreshold=thr)
            if A is None:
                return None, 0, None
            M = np.vstack([np.asarray(A, dtype=np.float64), [0.0, 0.0, 1.0]])
        if mask is None:
            return M, 0, None
        mask = np.asarray(mask).reshape(-1).astype(bool)
        return M, int(mask.sum()), mask

    def _pose_problem(self, pose: np.ndarray) -> str:
        """Why the calibration->current transform ``pose`` is implausible ("" when sane).

        Bounds, all against the calibration frame: no mirroring/collapse, scale within
        ``1 +- max_scale_change`` (singular values), rotation within ``max_rotation_deg``,
        polygon area within ``1 +- max_area_change``, polygon centroid shift within
        ``max_translation_fraction`` of the frame's larger side and, for a homography, the
        perspective factor within ``1 +- max_scale_change`` at every frame corner (the
        'left end moved 53 px' class of single-frame ORB errors).
        """
        T = np.asarray(pose, dtype=np.float64)
        if not np.all(np.isfinite(T)) or abs(T[2, 2]) < 1e-12:
            return "degenerate transform"
        T = T / T[2, 2]
        A = T[:2, :2]
        if np.linalg.det(A) <= 0:
            return "mirrored or collapsed transform"
        msc = float(self.cfg.max_scale_change)
        s = np.linalg.svd(A, compute_uv=False)
        if s.min() < 1.0 - msc or s.max() > 1.0 + msc:
            return f"scale {s.min():.2f}-{s.max():.2f} outside 1+-{msc:.2f}"
        rot = math.degrees(math.atan2(A[1, 0] - A[0, 1], A[0, 0] + A[1, 1]))
        max_rot = float(getattr(self.cfg, "max_rotation_deg", 0.0) or 0.0)
        if max_rot > 0 and abs(rot) > max_rot:
            return f"rotation {rot:.1f} deg beyond {max_rot:.1f}"
        h, w = self._shape if self._shape is not None else (1, 1)
        if self._model == "homography":
            corners = np.array([[0.0, 0.0], [w, 0.0], [0.0, h], [w, h]])
            factor = corners @ T[2, :2] + 1.0
            if factor.min() < 1.0 - msc or factor.max() > 1.0 + msc:
                return f"perspective factor {factor.min():.2f}-{factor.max():.2f} outside 1+-{msc:.2f}"
        warped = transform_points(self._polygon0, T).astype(np.float64)
        if not np.all(np.isfinite(warped)):
            return "polygon maps to infinity"
        area0 = _polygon_area(self._polygon0)
        mac = float(self.cfg.max_area_change)
        if area0 > 0:
            ratio = _polygon_area(warped) / area0
            if not (1.0 - mac <= ratio <= 1.0 + mac):
                return f"zone area x{ratio:.2f} outside 1+-{mac:.2f}"
        max_shift = float(getattr(self.cfg, "max_translation_fraction", 0.0) or 0.0) * float(max(w, h))
        shift = float(np.hypot(*(warped.mean(axis=0) - self._polygon0.mean(axis=0))))
        if max_shift > 0 and shift > max_shift:
            return f"zone moved {shift:.0f} px (limit {max_shift:.0f})"
        return ""

    def _outside_overlay(self, points: np.ndarray) -> np.ndarray:
        """Boolean per (N,2) point: True unless it lies on a learned static overlay."""
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        if self._overlay is None or pts.shape[0] == 0:
            return np.ones(pts.shape[0], dtype=bool)
        h, w = self._overlay.shape[:2]
        x = np.clip(np.round(pts[:, 0]).astype(np.int64), 0, w - 1)
        y = np.clip(np.round(pts[:, 1]).astype(np.int64), 0, h - 1)
        return self._overlay[y, x] == 0

    def _usable(self, kf: _Keyframe) -> np.ndarray:
        """Boolean per keyframe corner: True unless it was learned as static overlay."""
        return self._outside_overlay(kf.pts)

    def _photometric_match(self, kf: _Keyframe, gray: np.ndarray) -> Optional[np.ndarray]:
        """``gray`` remapped to the keyframe's grey-level distribution, or None if they match.

        Histogram specification (both measured outside the zone) undoes any global
        monotonic change - gain, offset, gamma: dusk, an auto-exposure step, a calibration
        image taken in other light - which breaks LK's brightness constancy but not the
        geometry. None when the two distributions already agree within
        ``PHOTOMETRIC_TOLERANCE`` grey levels (nothing to correct).
        """
        if kf.cdf is None:
            levels = _grey_levels(kf.gray, kf.mask)
            kf.cdf = np.zeros(0) if levels is None else np.cumsum(levels)
        p = _grey_levels(gray, kf.mask)
        if kf.cdf.size == 0 or p is None:
            return None
        cdf = np.cumsum(p)
        # Each populated level goes to the keyframe level at the middle of its quantile
        # interval (identical distributions give the identity exactly).
        lut = np.clip(np.searchsorted(kf.cdf, cdf - 0.5 * p, side="left"), 0, 255)
        body = (p > 0) & (cdf > 0.01) & (cdf - p < 0.99)
        if not body.any() or int(np.abs(lut - np.arange(256))[body].max()) <= PHOTOMETRIC_TOLERANCE:
            return None
        return cv2.LUT(gray, lut.astype(np.uint8))

    def _estimate(self, kf: _Keyframe, gray: np.ndarray) -> Tuple[Optional[_Estimate], int, str]:
        """LK, then histogram-matched LK, then ORB; returns (estimate or None, LK inliers, reason)."""
        est, lk_inliers, reason = self._lk_estimate(kf, gray)
        if est is not None:
            return est, lk_inliers, ""
        # The frame's exposure may simply differ from the keyframe's: LK on the frame
        # remapped to the keyframe's histogram keeps LK precision through it.
        matched = self._photometric_match(kf, gray)
        if matched is not None:
            est, matched_inliers, matched_reason = self._lk_estimate(kf, matched)
            if est is not None:
                return est, matched_inliers, ""
            reason = f"{reason}; histogram-matched {matched_reason}"
        if self.cfg.orb_fallback:
            # Fallback: ORB descriptor matching (handles jumps beyond LK's capture range and
            # strong appearance change, at higher cost and lower precision). On the frame as
            # captured: binary descriptors are already exposure-robust, and a histogram
            # remapped with a train in view has flat segments that blur them (measured on
            # clip05 under shake: 3.0 px max error on the raw frame, 13.8 px on the remapped one).
            orb, orb_reason = self._orb_estimate(kf, gray)
            if orb is not None:
                return orb, lk_inliers, ""
            reason = f"{reason}; {orb_reason}"
        return None, lk_inliers, reason

    def _lk_estimate(self, kf: _Keyframe, gray: np.ndarray) -> Tuple[Optional[_Estimate], int, str]:
        """Track keyframe corners into ``gray`` with pyramidal LK and fit the model.

        Forward-backward check: every corner is tracked keyframe -> current -> keyframe
        and dropped if it does not come back within ``FB_MAX_ERROR_PX``. This is the
        cheapest reliable way to detect tracks that slid along an edge or jumped to a
        similar-looking patch (the classic LK failure on repetitive rail sleepers).
        Corners learned as static overlay are not tracked.
        """
        usable = self._usable(kf)
        p0 = kf.pts[usable]
        if p0.shape[0] < 4:
            return None, 0, "no reference features"
        win = (int(self.cfg.lk_win_size), int(self.cfg.lk_win_size))
        level = int(self.cfg.lk_max_level)
        p1, st_fwd, _err = cv2.calcOpticalFlowPyrLK(kf.gray, gray, p0, None, winSize=win, maxLevel=level)
        if p1 is None:
            return None, 0, "LK: tracking failed"
        p0_back, st_bwd, _err = cv2.calcOpticalFlowPyrLK(gray, kf.gray, p1, None, winSize=win, maxLevel=level)
        if p0_back is None:
            return None, 0, "LK: tracking failed"
        fb_err = np.linalg.norm((p0 - p0_back).reshape(-1, 2), axis=1)
        finite = np.isfinite(p1.reshape(-1, 2)).all(axis=1) & np.isfinite(fb_err)
        ok = (st_fwd.reshape(-1) == 1) & (st_bwd.reshape(-1) == 1) & finite & (fb_err <= FB_MAX_ERROR_PX)
        if int(ok.sum()) < 4:
            return None, int(ok.sum()), f"LK: {int(ok.sum())} tracks survived"
        src = p0[ok].reshape(-1, 2)
        dst = p1[ok].reshape(-1, 2)
        M, inliers, _mask = self._fit(src.reshape(-1, 1, 2), dst.reshape(-1, 1, 2))
        if M is None:
            return None, 0, "LK: model fit failed"
        need = self._min_inliers_for(int(p0.shape[0]))
        if inliers < need:
            return None, inliers, f"LK: {inliers} inliers < {need}"
        pose = M @ kf.pose
        problem = self._pose_problem(pose)
        if problem:
            return None, inliers, f"LK: {problem}"
        return _Estimate("lk", M, pose, inliers, src, dst, idx=np.flatnonzero(usable)[ok], kf=kf), inliers, ""

    def _orb_estimate(self, kf: _Keyframe, gray: np.ndarray) -> Tuple[Optional[_Estimate], str]:
        """ORB + ratio-test matching between keyframe and ``gray``, then the model fit.

        Only attempted when LK fails, because it is several times slower and ~1 px less
        precise, but it is invariant to translation size and robust to exposure change.
        Keyframe descriptors are cached and honour the zone exclusion mask for the same
        reason the Shi-Tomasi corners do; matches on learned static overlays are dropped.
        An ORB estimate needs ``orb_min_matches`` inliers *and* ``orb_min_inlier_fraction``
        of the ratio-test matches: a fit that most matches disagree with is a coincidence,
        not the camera.
        """
        if self._orb is None:
            self._orb = cv2.ORB_create(nfeatures=int(self.cfg.orb_features))
            self._bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        if kf.orb is None:
            kf.orb = self._orb.detectAndCompute(kf.gray, kf.mask)
        kp_ref, des_ref = kf.orb
        kp_cur, des_cur = self._orb.detectAndCompute(gray, None)
        if des_ref is None or des_cur is None or len(kp_ref) < 2 or len(kp_cur) < 2:
            return None, "ORB: no keypoints"
        pairs = self._bf.knnMatch(des_ref, des_cur, k=2)
        ratio = float(self.cfg.orb_ratio_test)
        good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < ratio * p[1].distance]
        min_matches = int(self.cfg.orb_min_matches)
        if len(good) < min_matches:
            return None, f"ORB: {len(good)} matches < {min_matches}"
        src = np.float32([kp_ref[m.queryIdx].pt for m in good]).reshape(-1, 2)
        dst = np.float32([kp_cur[m.trainIdx].pt for m in good]).reshape(-1, 2)
        if self._overlay is not None:
            keep = self._outside_overlay(src) & self._outside_overlay(dst)
            src, dst = src[keep], dst[keep]
            if src.shape[0] < min_matches:
                return None, f"ORB: {src.shape[0]} matches off static overlays < {min_matches}"
        M, inliers, _mask = self._fit(src.reshape(-1, 1, 2), dst.reshape(-1, 1, 2))
        if M is None:
            return None, "ORB: model fit failed"
        if inliers < min_matches:
            return None, f"ORB: {inliers} inliers < {min_matches}"
        min_frac = float(getattr(self.cfg, "orb_min_inlier_fraction", 0.0) or 0.0)
        if inliers < min_frac * src.shape[0]:
            return None, f"ORB: {inliers}/{src.shape[0]} matches agree (< {min_frac:.2f})"
        pose = M @ kf.pose
        problem = self._pose_problem(pose)
        if problem:
            return None, f"ORB: {problem}"
        return _Estimate("orb", M, pose, inliers, src, dst, kf=kf), ""

    # ------------------------------------------------------------------ internals: temporal
    def _support(self, M: np.ndarray, est: _Estimate, tol: float) -> int:
        """How many of ``est``'s correspondences the keyframe->current transform ``M`` fits."""
        pred = transform_points(est.src, M).astype(np.float64)
        res = np.sqrt(((pred - est.dst.astype(np.float64)) ** 2).sum(axis=1))
        return int((res <= tol).sum())

    def _acceptance_bar(self, est: _Estimate, kf: _Keyframe) -> int:
        """Inliers an estimate of ``est``'s kind needed on this frame (count and fraction)."""
        if est.method == "lk":
            return self._min_inliers_for(int(self._usable(kf).sum()))
        frac = float(getattr(self.cfg, "orb_min_inlier_fraction", 0.0) or 0.0)
        return max(int(self.cfg.orb_min_matches), int(math.ceil(frac * est.src.shape[0])))

    def _last_pose_refuted(self, est: _Estimate, kf: _Keyframe) -> bool:
        """True when this frame's correspondences rule out the last accepted pose.

        The last accepted pose (expressed relative to ``kf``, the keyframe the estimate is
        relative to) and the new estimate are scored on the correspondences the estimate
        came from, with a tolerance of half the step limit (at most the RANSAC threshold),
        so that a real move just above the limit leaves the old pose with no support at
        all. If the old pose would fail the very acceptance bar the estimate had to pass
        (``min_lk_inliers`` for LK; for ORB ``orb_min_matches`` and
        ``orb_min_inlier_fraction`` of the matches), fits at most ``REFUTE_MAX_FRACTION`` of
        the correspondences, and the new estimate fits ``REFUTE_SUPPORT_RATIO`` times as
        many, the image content really moved - a camera bump or vibration - and waiting for
        confirmation would only leave the zone behind. When the old pose still fits a
        sizeable part of the frame, a jump is the classic estimation error (a passing
        train, a minority of points on repetitive texture, an extrapolation far from the
        features): it must be confirmed first. Corners learned as static overlay do not
        vote (they would always "support" the pose the camera had when they were seen).
        """
        tol = min(float(self.cfg.ransac_reproj_threshold), 0.5 * self.max_step_px())
        old = self._pose @ np.linalg.inv(kf.pose)
        n_old = self._support(old, est, tol)
        n_new = self._support(est.M, est, tol)
        return (n_old < self._acceptance_bar(est, kf) and n_old <= REFUTE_MAX_FRACTION * est.src.shape[0]
                and n_new >= REFUTE_SUPPORT_RATIO * max(n_old, 1))

    def _temporal_gate(self, est: _Estimate, kf: _Keyframe) -> Tuple[Optional[_Estimate], str]:
        """Accept ``est`` or reject it as a jump inconsistent with the last accepted pose.

        A step up to :meth:`max_step_px` is ordinary motion or noise. A larger step is
        accepted when the frame refutes the last pose (see :meth:`_last_pose_refuted`) or
        once ``confirm_frames`` consecutive estimates agree (within the step limit of the
        first of them) on the new pose; until then the update holds.
        """
        limit = self.max_step_px()
        need = int(getattr(self.cfg, "confirm_frames", 1) or 1)
        candidate = transform_points(self._polygon0, est.pose).astype(np.float64)
        step = _max_vertex_distance(candidate, self._polygon)
        if limit <= 0 or need <= 1 or step <= limit or self._last_pose_refuted(est, kf):
            self._pending = None
            self._pending_count = 0
            return est, ""
        if self._pending is not None and _max_vertex_distance(candidate, self._pending) <= limit:
            self._pending_count += 1
        else:
            self._pending = candidate
            self._pending_count = 1
        if self._pending_count >= need:
            self._pending = None
            self._pending_count = 0
            return est, ""
        return None, (f"{est.method.upper()} step {step:.1f} px > {limit:.1f} px from the last accepted zone, "
                      f"unconfirmed ({self._pending_count}/{need})")

    def _commit(self, pose: np.ndarray) -> None:
        """Make ``pose`` the accepted calibration->current transform; apply the dead-zone."""
        pose = np.asarray(pose, dtype=np.float64)
        pose = pose / pose[2, 2]
        self._pose = pose
        self._to_reference = np.linalg.inv(pose)
        self._to_reference /= self._to_reference[2, 2]
        self._polygon = transform_points(self._polygon0, pose).astype(np.float64)
        dead = float(getattr(self.cfg, "deadzone_px", 0.0) or 0.0)
        if dead <= 0 or _max_vertex_distance(self._polygon, self._shown) >= dead:
            self._shown = self._polygon.copy()

    def _learn_overlay(self, est: _Estimate) -> None:
        """Collect static-overlay evidence from an accepted LK estimate against the keyframe.

        A corner that did not move (``OVERLAY_STILL_PX``) where the accepted transform says
        the picture moved (``OVERLAY_MIN_MOTION_PX``) is burned into the image, not part of
        the scene; after ``OVERLAY_MIN_HITS`` such frames it is masked (with a disc of half
        ``min_distance``) out of this and every later keyframe, of ORB matching and of the
        temporal gate's vote. A static camera produces no evidence at all.
        """
        kf = est.kf
        if est.method != "lk" or est.idx is None or kf is None or kf is not self._keyframe or est.src.shape[0] == 0:
            return
        src = est.src.astype(np.float64)
        still = np.hypot(*(est.dst.astype(np.float64) - src).T) <= OVERLAY_STILL_PX
        moved = np.hypot(*(transform_points(est.src, est.M).astype(np.float64) - src).T) >= OVERLAY_MIN_MOTION_PX
        hit = est.idx[still & moved]
        if hit.size == 0:
            return
        kf.static_hits[hit] += 1
        new = hit[kf.static_hits[hit] == OVERLAY_MIN_HITS]
        if new.size == 0 or kf.overlay_marked + new.size > OVERLAY_MAX_FRACTION * kf.pts.shape[0]:
            return
        if self._overlay is None:
            self._overlay = np.zeros(kf.gray.shape[:2], dtype=np.uint8)
        radius = max(3, int(self.cfg.min_distance) // 2)
        for x, y in kf.pts[new].reshape(-1, 2):
            cv2.circle(self._overlay, (int(round(float(x))), int(round(float(y)))), radius, 255, -1)
        kf.overlay_marked += int(new.size)
        self._overlay_features += int(new.size)
        LOGGER.info("zone stabilizer: %d corner(s) learned as static overlay (do not move with the camera)",
                    int(new.size))

    # ------------------------------------------------------------------ internals: refresh
    def _maybe_refresh_keyframe(self, gray: np.ndarray, timestamp: float, scene_static: bool,
                                healthy: bool) -> bool:
        """Re-anchor the keyframe after a long static, healthy period; True if it happened.

        Re-anchoring only happens while the zone is empty *and* the current estimate is
        fresh and healthy, so that the new keyframe's pose is a good estimate and its
        feature set is background, not a parked vehicle. The new pose is checked against a
        direct estimate from the calibration keyframe: when the two agree within
        ``reanchor_max_drift_px`` the keyframe is anchored on the direct estimate (no error
        accumulates across re-anchors), when they disagree the re-anchor is refused, and
        when no direct estimate is possible (the scene changed too much since calibration)
        the chained pose is used and counted in ``unverified_reanchors``. While the stream
        is not registered against ``zone.reference_image`` (registration timed out at
        start-up) a direct estimate *is* the registration: it is applied when it is a small
        step or the frame refutes the uncorrected pose, and reported.
        """
        period = float(self.cfg.keyframe_refresh_static_seconds)
        if period <= 0 or not scene_static:
            self._static_since = None
            return False
        if self._static_since is None:
            self._static_since = timestamp
        if not healthy or (timestamp - self._static_since) < period:
            return False
        self._static_since = timestamp  # start a fresh period whatever happens below
        pose = self._pose
        verified = self._keyframe is self._reference  # the estimate was direct already
        registering = False
        drift = 0.0
        if not verified:
            direct, _lk, _reason = self._estimate(self._reference, gray)
            if direct is not None:
                drift = _max_vertex_distance(transform_points(self._polygon0, direct.pose), self._polygon)
                self._drift_px = drift
                if not self._registered:
                    if drift <= self.max_step_px() or self._last_pose_refuted(direct, self._reference):
                        pose, verified, registering = direct.pose, True, True
                elif drift > float(self.cfg.reanchor_max_drift_px):
                    self._refused_reanchors += 1
                    LOGGER.warning("keyframe re-anchor refused: the chained zone estimate is %.2f px from a "
                                   "direct estimate against the calibration frame", drift)
                    return False
                else:
                    pose = direct.pose
                    verified = True
        candidate = self._make_keyframe(gray, pose, timestamp)
        if candidate.pts.shape[0] < self._min_features():
            return False  # too little texture to track from; keep the current keyframe
        self._keyframe = candidate
        self._reanchors += 1
        self._unverified_reanchors = 0 if verified else self._unverified_reanchors + 1
        if pose is not self._pose:
            self._commit(pose)
        if registering:
            self._registered = True
            message = (f"zone registered against zone.reference_image at a keyframe refresh: the zone moved "
                       f"{drift:.1f} px (camera motion since calibration is now corrected)")
            LOGGER.warning("%s", message)
            self._notify(message)
        return True

    # ------------------------------------------------------------------ internals: state
    def _make_state(self, H: Optional[np.ndarray], inliers: int, method: str, healthy: bool) -> ZoneState:
        poly_i = np.round(self._shown).astype(np.int32)
        key = poly_i.tobytes()
        if key != self._mask_key or self._mask is None:
            shape = self._shape if self._shape is not None else (1, 1)
            self._mask = polygon_mask(shape, self._shown.astype(np.float32))
            self._mask_area = int(cv2.countNonZero(self._mask))
            self._mask_key = key
        return ZoneState(
            polygon=poly_i,
            mask=self._mask.copy(),
            homography=None if H is None else np.asarray(H, dtype=np.float64),
            inliers=int(inliers),
            method=method,
            healthy=bool(healthy),
            area_px=self._mask_area,
            to_reference=self._to_reference.copy() if self.initialized else None,
        )
