"""Zone-of-interest stabilisation.

The operator draws the crossing polygon once, on a *reference* frame. A real camera does
not stay still: wind, traffic vibration, a pan-tilt bump, or a thermal drift all move the
image by a few pixels (occasionally a lot more). If the polygon stayed fixed in image
coordinates, the "zone" would slowly stop covering the tracks and every downstream
decision (in-zone test, debris mask) would silently degrade.

``ZoneStabilizer`` estimates a homography from the reference keyframe to the current
frame every update and warps the polygon with it. The design is deliberately
conservative, because a *wrong* zone is worse than a *stale* one:

* Sparse Lucas-Kanade optical flow (fast, sub-pixel) is the primary method. A
  forward-backward round-trip check drops tracks that drifted onto the wrong texture.
* RANSAC + :func:`gradecrossing.geometry.homography_is_sane` reject estimates built from
  too few inliers or with implausible scale/area change (mirrors, collapses, blow-ups).
* An LK estimate must additionally keep a *fraction* of the reference features
  (``min_inlier_fraction``), not just an absolute count: a minority of points locked onto
  self-similar sleeper/ballast texture can agree on a consistent but wrong flow and still
  clear ``min_inliers``. When most reference features disagree, the estimate is refused.
* ORB descriptor matching is the fallback when LK fails (large jump, exposure change).
* If both fail the stabilizer **holds the last healthy polygon** and reports
  ``healthy=False``; it never silently snaps back to the raw reference polygon, which
  would look plausible on the overlay while pointing at the wrong pixels.
* When the scene has been static (no tracked objects, healthy estimate) for a long time
  the reference keyframe is re-anchored to the current frame, so slow lighting change
  does not erode the feature set. Re-anchoring while objects are in the zone is refused
  because the polygon would then be estimated from a partially occluded scene.

The legacy prototype (``scripts/pipeline_helpers.py::stabilize_zoi``) did LK + RANSAC with
no health check and fell back to the raw reference polygon; this module replaces it.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from .config import ZoneConfig
from .geometry import as_polygon, homography_is_sane, polygon_mask, transform_points
from .types import ZoneState

# Forward-backward round-trip tolerance for LK tracks, in pixels. A track whose
# reference -> current -> reference round trip lands more than this far from where it
# started has drifted onto a different texture and must not vote in RANSAC.
FB_MAX_ERROR_PX = 1.0


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


class ZoneStabilizer:
    """Track the operator-drawn crossing polygon through camera motion.

    Usage::

        stab = ZoneStabilizer(cfg.zone)
        stab.initialize(first_frame, t0)
        state = stab.update(frame, t, scene_static=not tracker.zone_occupied)

    ``state.polygon`` is the polygon in the current frame, ``state.healthy`` says whether
    it was freshly estimated (``lk``/``orb``) or merely held from the last good estimate.
    Downstream code should treat a long run of unhealthy frames as a reason to raise a
    HEALTH event (see :meth:`health` and :attr:`health_changed`).
    """

    def __init__(self, cfg: ZoneConfig):
        self.cfg = cfg
        # Reference keyframe (what the polygon is drawn on)
        self._ref_gray: Optional[np.ndarray] = None
        self._ref_shape: Optional[Tuple[int, int]] = None
        self._ref_polygon: Optional[np.ndarray] = None  # (N,2) float32
        self._ref_pts: np.ndarray = np.zeros((0, 1, 2), dtype=np.float32)  # Shi-Tomasi corners
        self._feature_mask: Optional[np.ndarray] = None  # where features are allowed
        self._keyframe_time: float = 0.0
        # ORB descriptors of the keyframe, computed lazily on first fallback use
        self._ref_orb: Optional[Tuple[list, Optional[np.ndarray]]] = None
        self._orb = None
        self._bf = None
        # Last estimate bookkeeping (drives hold + health reporting)
        self._last_healthy_polygon: Optional[np.ndarray] = None  # (N,2) float32, current-frame coords
        self._last_method: str = "uninitialized"
        self._last_inliers: int = 0
        self._last_healthy: bool = False
        self._last_timestamp: float = 0.0
        self._consecutive_unhealthy: int = 0
        self._static_since: Optional[float] = None
        # True on the update where healthy flipped (for HEALTH events in the monitor)
        self.health_changed: bool = False

    # ------------------------------------------------------------------ properties
    @property
    def initialized(self) -> bool:
        """True once :meth:`initialize` (or the first :meth:`update`) has stored a keyframe."""
        return self._ref_gray is not None

    @property
    def reference_polygon(self) -> np.ndarray:
        """The polygon in the coordinates of the *current reference keyframe*, (N,2) float32.

        This changes only when the keyframe is re-anchored (see keyframe refresh).
        """
        if self._ref_polygon is None:
            return as_polygon(self.cfg.polygon)
        return self._ref_polygon.copy()

    @property
    def reference_features(self) -> np.ndarray:
        """Shi-Tomasi corners tracked from the keyframe, (M,2) float32. Exposed for tests/overlays."""
        return self._ref_pts.reshape(-1, 2).copy()

    # ------------------------------------------------------------------ public API
    def initialize(self, frame_bgr: np.ndarray, timestamp: float) -> ZoneState:
        """Store the reference keyframe, polygon and features; return the identity state.

        Called once at start-up (and internally on keyframe refresh). Features inside the
        polygon are discarded when ``cfg.exclude_features_in_zone`` because anything in the
        zone is likely to move (vehicles, trains) and would corrupt a background motion
        estimate.
        """
        gray = _to_gray(frame_bgr)
        polygon = as_polygon(self.cfg.polygon)
        self._anchor(gray, polygon, timestamp)
        self._last_healthy_polygon = polygon.copy()
        self._last_method = "identity"
        self._last_inliers = 0
        self._last_healthy = True
        self._last_timestamp = float(timestamp)
        self._consecutive_unhealthy = 0
        self._static_since = None
        self.health_changed = False
        return self._make_state(polygon, None, 0, "identity", True)

    def update(self, frame_bgr: np.ndarray, timestamp: float, scene_static: bool = False) -> ZoneState:
        """Estimate where the polygon is in ``frame_bgr`` and return the :class:`ZoneState`.

        ``scene_static`` should be True only when the caller knows nothing is moving in the
        zone (no tracked objects, no debris trigger); it gates keyframe refresh. Calling
        ``update`` before ``initialize`` initialises on this frame and returns the
        identity state, so a monitor loop can call ``update`` uniformly from frame 0.

        Raises ``ValueError`` if the frame size differs from the reference keyframe: the
        polygon is in reference pixels, so a silently resized frame would put the zone
        in the wrong place.
        """
        if not self.initialized:
            return self.initialize(frame_bgr, timestamp)
        gray = _to_gray(frame_bgr)
        if gray.shape[:2] != self._ref_shape:
            raise ValueError(
                f"frame size {gray.shape[1]}x{gray.shape[0]} does not match the reference "
                f"keyframe {self._ref_shape[1]}x{self._ref_shape[0]}; resize before update()"
            )
        timestamp = float(timestamp)
        self._last_timestamp = timestamp
        prev_healthy = self._last_healthy

        if not self.cfg.stabilize:
            # Stabilisation disabled: the zone is the reference polygon, always "healthy".
            polygon = self._ref_polygon.copy()
            self._last_healthy_polygon = polygon.copy()
            self._record(method="identity", inliers=0, healthy=True)
            self.health_changed = prev_healthy != self._last_healthy
            return self._make_state(polygon, None, 0, "identity", True)

        # 1. Primary: sparse LK with forward-backward check + RANSAC homography.
        H, inliers = self._lk_homography(gray)
        method = "lk"
        if not self._accept(H, inliers, self.min_lk_inliers()):
            lk_inliers = inliers
            H = None
            # 2. Fallback: ORB descriptor matching (handles jumps beyond LK's capture range
            #    and strong appearance change, at higher cost and lower precision).
            if self.cfg.orb_fallback:
                H_orb, orb_inliers, _n_matches = self._orb_homography(gray)
                if self._accept(H_orb, orb_inliers, self.cfg.orb_min_matches):
                    H, inliers, method = H_orb, orb_inliers, "orb"
            if H is None:
                # 3. Hold. We deliberately keep the last polygon that passed the sanity
                #    checks rather than snapping to the raw reference polygon: the camera
                #    may still be displaced, so the reference polygon is not "safe".
                inliers = lk_inliers
                method = "held"

        if H is not None:
            polygon = transform_points(self._ref_polygon, H).astype(np.float32)
            self._last_healthy_polygon = polygon.copy()
            healthy = True
        else:
            polygon = self._last_healthy_polygon.copy()
            healthy = False

        self._record(method=method, inliers=int(inliers), healthy=healthy)
        self.health_changed = prev_healthy != healthy

        # 4. Keyframe refresh when the scene has been static for long enough.
        self._maybe_refresh_keyframe(gray, polygon, timestamp, scene_static, healthy)

        return self._make_state(polygon, H, int(inliers), method, healthy)

    def health(self) -> Dict[str, object]:
        """Summary of the last estimate for logging and HEALTH events.

        Keys: ``method`` (identity/lk/orb/held), ``inliers`` (RANSAC inlier count of the
        accepted or attempted LK estimate), ``healthy``, ``keyframe_age_s`` (seconds since
        the reference keyframe was (re-)anchored) and ``consecutive_unhealthy`` (number of
        updates in a row that had to hold the polygon).
        """
        return {
            "method": self._last_method,
            "inliers": int(self._last_inliers),
            "healthy": bool(self._last_healthy),
            "keyframe_age_s": float(self._last_timestamp - self._keyframe_time) if self.initialized else 0.0,
            "consecutive_unhealthy": int(self._consecutive_unhealthy),
        }

    # ------------------------------------------------------------------ internals
    def _anchor(self, gray: np.ndarray, polygon: np.ndarray, timestamp: float) -> None:
        """Make ``gray`` the reference keyframe with ``polygon`` drawn on it; re-detect features."""
        self._ref_gray = gray
        self._ref_shape = gray.shape[:2]
        self._ref_polygon = np.asarray(polygon, dtype=np.float32).reshape(-1, 2).copy()
        self._keyframe_time = float(timestamp)
        self._ref_orb = None  # ORB descriptors belong to the old keyframe

        if self.cfg.exclude_features_in_zone:
            # goodFeaturesToTrack only searches where mask != 0: allow everywhere except the zone.
            mask = np.full(self._ref_shape, 255, dtype=np.uint8)
            cv2.fillPoly(mask, [np.round(self._ref_polygon).astype(np.int32).reshape(-1, 1, 2)], 0)
        else:
            mask = None
        self._feature_mask = mask

        pts = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=int(self.cfg.max_corners),
            qualityLevel=float(self.cfg.quality_level),
            minDistance=int(self.cfg.min_distance),
            mask=mask,
            blockSize=int(self.cfg.block_size),
        )
        if pts is None:  # uniform image: nothing to track, LK will be skipped
            pts = np.zeros((0, 1, 2), dtype=np.float32)
        self._ref_pts = np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)

    def _record(self, method: str, inliers: int, healthy: bool) -> None:
        self._last_method = method
        self._last_inliers = int(inliers)
        self._last_healthy = bool(healthy)
        if healthy:
            self._consecutive_unhealthy = 0
        else:
            self._consecutive_unhealthy += 1

    def min_lk_inliers(self) -> int:
        """Inliers an LK estimate needs: ``max(min_inliers, min_inlier_fraction * n_features)``.

        Why a fraction and not only an absolute count: LK on a rail crossing tracks a few
        hundred corners, many of them on self-similar texture (sleepers, ballast, lane
        markings). A subset of those can slide by the same amount onto the neighbouring
        sleeper, agree with each other, and give RANSAC a consistent *wrong* homography
        with, say, 45 inliers - comfortably above the absolute ``min_inliers`` of 30 - while
        the other 355 features disagree. Requiring a share of the reference features makes
        that impossible: a homography that most of the background does not support is
        refused, the stabilizer tries ORB and otherwise holds the last healthy polygon,
        which is the conservative failure the module is built around.

        The reference feature count shrinks only when the keyframe is re-anchored, so the
        bar is stable between refreshes. ``min_inlier_fraction = 0`` restores the old
        absolute-only rule.
        """
        n_ref = int(self._ref_pts.shape[0])
        frac = float(getattr(self.cfg, "min_inlier_fraction", 0.0) or 0.0)
        needed = int(math.ceil(frac * n_ref)) if frac > 0.0 and n_ref > 0 else 0
        return max(int(self.cfg.min_inliers), needed)

    def _accept(self, H: Optional[np.ndarray], inliers: int, min_inliers: int) -> bool:
        """Common acceptance rule: enough RANSAC inliers *and* a geometrically sane homography."""
        if H is None or inliers < int(min_inliers):
            return False
        return homography_is_sane(H, self._ref_polygon, self.cfg.max_scale_change, self.cfg.max_area_change)

    def _lk_homography(self, gray: np.ndarray) -> Tuple[Optional[np.ndarray], int]:
        """Track keyframe corners into ``gray`` with pyramidal LK; return (H, inlier_count).

        Forward-backward check: every corner is tracked reference -> current -> reference
        and dropped if it does not come back within ``FB_MAX_ERROR_PX``. This is the
        cheapest reliable way to detect tracks that slid along an edge or jumped to a
        similar-looking patch (the classic LK failure on repetitive rail sleepers).
        """
        p0 = self._ref_pts
        if p0.shape[0] < 4:
            return None, 0
        win = (int(self.cfg.lk_win_size), int(self.cfg.lk_win_size))
        level = int(self.cfg.lk_max_level)
        p1, st_fwd, _err = cv2.calcOpticalFlowPyrLK(self._ref_gray, gray, p0, None, winSize=win, maxLevel=level)
        if p1 is None:
            return None, 0
        p0_back, st_bwd, _err = cv2.calcOpticalFlowPyrLK(gray, self._ref_gray, p1, None, winSize=win, maxLevel=level)
        if p0_back is None:
            return None, 0
        fb_err = np.linalg.norm((p0 - p0_back).reshape(-1, 2), axis=1)
        finite = np.isfinite(p1.reshape(-1, 2)).all(axis=1) & np.isfinite(fb_err)
        ok = (st_fwd.reshape(-1) == 1) & (st_bwd.reshape(-1) == 1) & finite & (fb_err <= FB_MAX_ERROR_PX)
        if int(ok.sum()) < 4:
            return None, int(ok.sum())
        ref_good = p0[ok].reshape(-1, 1, 2)
        cur_good = p1[ok].reshape(-1, 1, 2)
        return self._ransac(ref_good, cur_good)

    def _ransac(self, src: np.ndarray, dst: np.ndarray) -> Tuple[Optional[np.ndarray], int]:
        """RANSAC homography src->dst; returns (H or None, inlier count)."""
        if src.shape[0] < 4:
            return None, 0
        H, inlier_mask = cv2.findHomography(src, dst, cv2.RANSAC, float(self.cfg.ransac_reproj_threshold))
        if H is None:
            return None, 0
        inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
        return np.asarray(H, dtype=np.float64), inliers

    def _orb_homography(self, gray: np.ndarray) -> Tuple[Optional[np.ndarray], int, int]:
        """ORB + ratio-test matching between keyframe and ``gray``; returns (H, inliers, n_good).

        Only attempted when LK fails, because it is several times slower and ~1 px less
        precise, but it is invariant to translation size and robust to exposure change.
        Reference descriptors are cached per keyframe and honour the zone exclusion mask
        for the same reason the Shi-Tomasi corners do.
        """
        if self._orb is None:
            self._orb = cv2.ORB_create(nfeatures=int(self.cfg.orb_features))
            self._bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        if self._ref_orb is None:
            self._ref_orb = self._orb.detectAndCompute(self._ref_gray, self._feature_mask)
        kp_ref, des_ref = self._ref_orb
        kp_cur, des_cur = self._orb.detectAndCompute(gray, None)
        if des_ref is None or des_cur is None or len(kp_ref) < 2 or len(kp_cur) < 2:
            return None, 0, 0
        pairs = self._bf.knnMatch(des_ref, des_cur, k=2)
        ratio = float(self.cfg.orb_ratio_test)
        good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < ratio * p[1].distance]
        if len(good) < int(self.cfg.orb_min_matches):
            return None, 0, len(good)
        src = np.float32([kp_ref[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp_cur[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, inliers = self._ransac(src, dst)
        return H, inliers, len(good)

    def _maybe_refresh_keyframe(self, gray: np.ndarray, polygon: np.ndarray, timestamp: float,
                                scene_static: bool, healthy: bool) -> None:
        """Re-anchor the keyframe after a long static, healthy period; reset the timer otherwise.

        Re-anchoring only happens while the zone is empty *and* the current estimate is
        healthy, so that the new reference polygon is a good estimate and the new feature
        set is background, not a parked vehicle. Errors accumulate a little on each
        re-anchor, which is why the period is long (minutes) rather than per-frame.
        """
        period = float(self.cfg.keyframe_refresh_static_seconds)
        if period <= 0 or not scene_static:
            self._static_since = None
            return
        if self._static_since is None:
            self._static_since = timestamp
        if healthy and (timestamp - self._static_since) >= period:
            self._anchor(gray, polygon, timestamp)
            self._last_healthy_polygon = np.asarray(polygon, dtype=np.float32).reshape(-1, 2).copy()
            self._static_since = timestamp  # start a fresh period rather than re-anchoring every frame

    def _make_state(self, polygon: np.ndarray, H: Optional[np.ndarray], inliers: int,
                    method: str, healthy: bool) -> ZoneState:
        poly_f = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
        poly_i = np.round(poly_f).astype(np.int32)
        mask = polygon_mask(self._ref_shape, poly_f)
        return ZoneState(
            polygon=poly_i,
            mask=mask,
            homography=None if H is None else np.asarray(H, dtype=np.float64),
            inliers=int(inliers),
            method=method,
            healthy=bool(healthy),
            area_px=int(cv2.countNonZero(mask)),
        )
