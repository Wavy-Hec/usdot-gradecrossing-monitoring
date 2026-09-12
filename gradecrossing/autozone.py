"""Experimental automatic zone proposal (stage-1 seed for ``tools/calibrate_zone.py``).

**This is not a production zone.** The heuristic below finds the two strongest, roughly
parallel families of straight edges in the image and assumes they are the rails. On a
clean camera view of a crossing that is often right; on a view with a road kerb, a
platform edge, a fence or strong shadows it will happily pick those instead. The result
is meant to be drawn on the calibration screen as a *starting polygon* that the operator
drags into place and confirms. Nothing in the monitor should consume it unreviewed,
because every safety decision (in-zone test, debris mask) depends on the polygon being
right and there is no way to verify it automatically.

Pipeline: Gaussian blur -> Canny -> probabilistic Hough line segments -> cluster segments
by orientation (period 180 degrees) -> take the dominant orientation and its near-parallel
neighbours -> split that family by perpendicular offset -> the two strongest offset
clusters are the rail edges -> bound the strip along the rail direction either by the
horizontal extent of gate/sign detections (when given) or by the extent of the segments.
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .geometry import polygon_area
from .types import Detection

# Tunables of the heuristic. They are module constants (not config) on purpose: this is
# a calibration-time helper, and exposing them in the monitor config would suggest a
# level of trust the method does not deserve.
ANGLE_CLUSTER_TOL_DEG = 4.0  # segments within this orientation distance share a cluster
PARALLEL_FAMILY_TOL_DEG = 20.0  # clusters within this of the dominant one count as "near-parallel"
OFFSET_CLUSTER_FRACTION = 0.025  # perpendicular-offset cluster width, fraction of the image diagonal
CANNY_LOW, CANNY_HIGH = 50, 150
HOUGH_THRESHOLD = 60
HOUGH_MIN_LENGTH_FRACTION = 0.08  # of the longer image side
HOUGH_MAX_GAP = 10
# A rail edge seen from a crossing camera is a *long* straight line. Each chosen rail
# cluster must contain at least one segment this long (fraction of the longer image
# side), otherwise the "rails" are probably texture/noise and we return None.
MIN_RAIL_SEGMENT_FRACTION = 0.15
# Cluster strength is the sum of squared segment lengths: a few long edges must beat
# many short ones (ballast, gravel and JPEG noise produce swarms of short segments).
STRENGTH_POWER = 2.0
# If more than this fraction of pixels are Canny edges the frame is noise/texture, not a
# scene with lines, and the Hough transform will hallucinate diagonals: give up instead.
# Real camera frames sit around 1-10%; pure sensor noise is above 30%.
MAX_EDGE_DENSITY = 0.25


def _circular_diff_deg(a: float, b: float) -> float:
    """Distance between two orientations with period 180 degrees."""
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def _weighted_orientation_deg(angles_deg: np.ndarray, weights: np.ndarray) -> float:
    """Length-weighted mean of orientations (period 180) via the doubled-angle trick."""
    a2 = np.radians(angles_deg) * 2.0
    c = float(np.sum(weights * np.cos(a2)))
    s = float(np.sum(weights * np.sin(a2)))
    return (math.degrees(math.atan2(s, c)) / 2.0) % 180.0


def _cluster_by_angle(angles_deg: np.ndarray, weights: np.ndarray, tol_deg: float) -> List[Tuple[float, float, List[int]]]:
    """Greedy orientation clustering; returns [(mean_angle, strength, indices)] strongest first.

    Segments are visited longest-first so that the cluster centres are seeded by the most
    reliable evidence, and each segment joins the first centre within ``tol_deg``.
    """
    clusters: List[dict] = []
    for i in np.argsort(-weights):
        a = float(angles_deg[i])
        for cl in clusters:
            if _circular_diff_deg(a, cl["angle"]) <= tol_deg:
                cl["idx"].append(int(i))
                idx = np.asarray(cl["idx"])
                cl["angle"] = _weighted_orientation_deg(angles_deg[idx], weights[idx])
                cl["strength"] += float(weights[i])
                break
        else:
            clusters.append({"angle": a, "strength": float(weights[i]), "idx": [int(i)]})
    out = [(cl["angle"], cl["strength"], cl["idx"]) for cl in clusters]
    out.sort(key=lambda c: -c[1])
    return out


def _cluster_1d(values: np.ndarray, weights: np.ndarray, tol: float) -> List[Tuple[float, float, List[int]]]:
    """Greedy 1-D clustering of perpendicular offsets; returns [(mean, strength, indices)] strongest first."""
    clusters: List[dict] = []
    for i in np.argsort(-weights):
        v = float(values[i])
        for cl in clusters:
            if abs(v - cl["mean"]) <= tol:
                cl["idx"].append(int(i))
                idx = np.asarray(cl["idx"])
                cl["mean"] = float(np.average(values[idx], weights=weights[idx]))
                cl["strength"] += float(weights[i])
                break
        else:
            clusters.append({"mean": v, "strength": float(weights[i]), "idx": [int(i)]})
    out = [(cl["mean"], cl["strength"], cl["idx"]) for cl in clusters]
    out.sort(key=lambda c: -c[1])
    return out


def _fit_line(points: np.ndarray, direction_hint: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Least-squares line through (M,2) points; returns (point_on_line, unit_direction).

    The direction is flipped if needed so it agrees with ``direction_hint``; both rails
    must be parametrised the same way for the corner ordering below to form a simple quad.
    """
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    vx, vy, x0, y0 = [float(v) for v in cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).reshape(-1)]
    d = np.array([vx, vy], dtype=np.float64)
    d /= max(np.linalg.norm(d), 1e-9)
    if float(np.dot(d, direction_hint)) < 0:
        d = -d
    return np.array([x0, y0], dtype=np.float64), d


def _hough_segments(gray: np.ndarray) -> Optional[np.ndarray]:
    """Canny + probabilistic Hough; returns (K,4) float64 segments or None."""
    h, w = gray.shape[:2]
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, CANNY_LOW, CANNY_HIGH)
    if cv2.countNonZero(edges) > MAX_EDGE_DENSITY * edges.size:
        return None
    min_len = max(20, int(HOUGH_MIN_LENGTH_FRACTION * max(h, w)))
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180.0, HOUGH_THRESHOLD, minLineLength=min_len, maxLineGap=HOUGH_MAX_GAP)
    if lines is None or len(lines) == 0:
        return None
    return lines.reshape(-1, 4).astype(np.float64)


def _gate_x_extent(infra_detections: Optional[List[Detection]], gate_classes: Sequence[str]) -> Optional[Tuple[float, float]]:
    """Horizontal [x_min, x_max] spanned by gate/sign boxes of the requested classes, or None."""
    if not infra_detections:
        return None
    wanted = {c.lower() for c in gate_classes}
    xs: List[float] = []
    for det in infra_detections:
        if det.cls_name.lower() in wanted:
            xs.extend([float(det.xyxy[0]), float(det.xyxy[2])])
    if len(xs) < 2:
        return None
    lo, hi = min(xs), max(xs)
    return (lo, hi) if hi > lo else None


def propose_zone(frame_bgr: np.ndarray, infra_detections: Optional[List[Detection]] = None,
                 gate_classes: Sequence[str] = ("railroad-crossing-down", "railroad-crossings")) -> Optional[np.ndarray]:
    """Propose a 4-point crossing polygon from rail-like edges in ``frame_bgr``.

    Returns a (4,2) float32 polygon ordered around its boundary, or ``None`` when the
    image does not contain two strong near-parallel edge families. When gate/sign
    detections of ``gate_classes`` are supplied, the polygon is cut to their horizontal
    extent (gates stand at either side of the roadway, so the crossing lies between
    them); otherwise it spans the detected segments.

    This is a **seed for manual calibration**, see the module docstring. Never feed it to
    the monitor without an operator confirming it.
    """
    if frame_bgr is None or frame_bgr.ndim not in (2, 3):
        return None
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY) if frame_bgr.ndim == 3 else frame_bgr
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    h, w = gray.shape[:2]

    segs = _hough_segments(gray)
    if segs is None:
        return None
    dx = segs[:, 2] - segs[:, 0]
    dy = segs[:, 3] - segs[:, 1]
    lengths = np.hypot(dx, dy)
    keep = lengths > 1e-6
    segs, dx, dy, lengths = segs[keep], dx[keep], dy[keep], lengths[keep]
    if len(segs) < 2:
        return None
    angles = np.degrees(np.arctan2(dy, dx)) % 180.0
    weights = lengths ** STRENGTH_POWER
    min_rail_len = MIN_RAIL_SEGMENT_FRACTION * max(h, w)

    # 1. Dominant orientation and its near-parallel neighbours (the rail "family").
    angle_clusters = _cluster_by_angle(angles, weights, ANGLE_CLUSTER_TOL_DEG)
    dominant_angle = angle_clusters[0][0]
    family_idx: List[int] = []
    for ang, _strength, idx in angle_clusters:
        if _circular_diff_deg(ang, dominant_angle) <= PARALLEL_FAMILY_TOL_DEG:
            family_idx.extend(idx)
    family = np.asarray(sorted(set(family_idx)), dtype=int)
    theta = math.radians(dominant_angle)
    d_family = np.array([math.cos(theta), math.sin(theta)], dtype=np.float64)
    n_family = np.array([-d_family[1], d_family[0]], dtype=np.float64)

    # 2. Split the family by perpendicular offset: each rail edge is one offset cluster.
    mids = np.stack([(segs[family, 0] + segs[family, 2]) / 2.0, (segs[family, 1] + segs[family, 3]) / 2.0], axis=1)
    rho = mids @ n_family
    tol = OFFSET_CLUSTER_FRACTION * math.hypot(w, h)
    offset_clusters = _cluster_1d(rho, weights[family], tol)
    if len(offset_clusters) < 2:
        return None
    rails = []
    for _mean, _strength, local_idx in offset_clusters[:2]:
        seg_idx = family[np.asarray(local_idx, dtype=int)]
        if float(lengths[seg_idx].max()) < min_rail_len:
            return None  # strongest evidence is only short segments: not a rail
        endpoints = np.concatenate([segs[seg_idx, 0:2], segs[seg_idx, 2:4]], axis=0)
        rails.append((_fit_line(endpoints, d_family), endpoints))
    # Order the two rails by offset so the quad winds consistently (A then B).
    rails.sort(key=lambda r: float(np.dot(r[0][0], n_family)))
    (pa, da), ends_a = rails[0]
    (pb, db), ends_b = rails[1]

    # 3. Extent along the rail direction: gate boxes if available, else the segments.
    corners: Optional[np.ndarray] = None
    gate_x = _gate_x_extent(infra_detections, gate_classes)
    if gate_x is not None and abs(da[0]) > 0.3 and abs(db[0]) > 0.3:
        # Cut each rail at the vertical lines x = gate_x[0] and x = gate_x[1]: the zone is
        # the trapezoid between the two rails and the two gate positions.
        def _at_x(p: np.ndarray, d: np.ndarray, x: float) -> np.ndarray:
            return p + ((x - p[0]) / d[0]) * d

        corners = np.array([
            _at_x(pa, da, gate_x[0]), _at_x(pa, da, gate_x[1]),
            _at_x(pb, db, gate_x[1]), _at_x(pb, db, gate_x[0]),
        ], dtype=np.float64)
    if corners is None:
        all_ends = np.concatenate([ends_a, ends_b], axis=0)
        proj = (all_ends - pa) @ d_family
        t_range = (float(proj.min()), float(proj.max()))
        if t_range[1] - t_range[0] < 1.0:
            return None

        def _at(p: np.ndarray, d: np.ndarray, t: float) -> np.ndarray:
            # Each rail is re-parametrised from the common origin pa so the same t means
            # the same position along the strip on both rails (a parallelogram).
            base_t = float(np.dot(pa - p, d))
            return p + (base_t + t) * d

        corners = np.array([
            _at(pa, da, t_range[0]), _at(pa, da, t_range[1]),
            _at(pb, db, t_range[1]), _at(pb, db, t_range[0]),
        ], dtype=np.float64)
    corners[:, 0] = np.clip(corners[:, 0], 0, w - 1)
    corners[:, 1] = np.clip(corners[:, 1], 0, h - 1)
    poly = corners.astype(np.float32)
    if polygon_area(poly) <= 0:
        return None
    return poly
