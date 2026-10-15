"""Small geometry helpers shared by the zone, tracking and debris modules."""
from __future__ import annotations

import math
from typing import Iterable, Sequence, Tuple

import cv2
import numpy as np


def as_polygon(points: Iterable[Sequence[float]]) -> np.ndarray:
    """Return an (N,2) float32 array from any iterable of (x, y) pairs."""
    poly = np.asarray(list(points), dtype=np.float32).reshape(-1, 2)
    if poly.shape[0] < 3:
        raise ValueError("polygon needs at least 3 points")
    return poly


def polygon_mask(shape_hw: Tuple[int, int], polygon: np.ndarray) -> np.ndarray:
    """Filled uint8 mask (255 inside) of ``polygon`` for an image of shape (H, W)."""
    mask = np.zeros(shape_hw[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.round(polygon).astype(np.int32).reshape(-1, 1, 2)], 255)
    return mask


def polygon_area(polygon: np.ndarray) -> float:
    return float(abs(cv2.contourArea(np.asarray(polygon, dtype=np.float32).reshape(-1, 1, 2))))


def point_in_polygon(point: Tuple[float, float], polygon: np.ndarray) -> bool:
    """True if ``point`` is inside or on the edge of ``polygon`` (cv2.pointPolygonTest)."""
    contour = np.asarray(polygon, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), False) >= 0


def bottom_center(xyxy: Sequence[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    return ((x1 + x2) / 2.0, y2)


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def box_iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU of every box of ``a`` (M,4) with every box of ``b`` (N,4): an (M,N) float64 array.

    The arithmetic is :func:`box_iou`'s (float64, same operations in the same order), so a
    vectorised caller gets bit-identical values and thresholds decide exactly as before.
    """
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    iw = np.maximum(0.0, np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]))
    ih = np.maximum(0.0, np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]))
    inter = iw * ih
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    ua = area_a[:, None] + area_b[None, :] - inter
    ok = (inter > 0) & (ua > 0)
    return np.where(ok, inter / np.where(ok, ua, 1.0), 0.0)


def box_mask_overlap_fraction(xyxy: Sequence[float], mask: np.ndarray) -> float:
    """Fraction of the box pixels that fall on non-zero pixels of ``mask``."""
    h, w = mask.shape[:2]
    x1, y1, x2, y2 = [int(round(float(v))) for v in xyxy]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    roi = mask[y1:y2, x1:x2]
    return float(np.count_nonzero(roi)) / float(roi.size)


def footprint_box(xyxy: Sequence[float], fraction: float = 0.25) -> np.ndarray:
    """Ground-contact strip of a box: its bottom ``fraction`` of the height, full width.

    A single bottom-center point misses a vehicle seen obliquely across a thin zone: the
    box bottom lies outside the band while the strip of the box nearest the ground still
    overlaps it (team EXPERIMENTS.md E4: recall 7/15 -> 11/15 at one site).
    """
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    h = max(0.0, y2 - y1)
    return np.array([x1, y2 - float(fraction) * h, x2, y2], dtype=np.float32)


def _segments_intersect(p1, p2, q1, q2) -> bool:
    """True when segment p1-p2 and segment q1-q2 share a point (touching counts)."""
    def orient(a, b, c):
        v = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        return 0 if abs(v) < 1e-9 else (1 if v > 0 else -1)

    def on_seg(a, b, c):
        return (min(a[0], b[0]) - 1e-9 <= c[0] <= max(a[0], b[0]) + 1e-9
                and min(a[1], b[1]) - 1e-9 <= c[1] <= max(a[1], b[1]) + 1e-9)

    o1, o2, o3, o4 = orient(p1, p2, q1), orient(p1, p2, q2), orient(q1, q2, p1), orient(q1, q2, p2)
    if o1 != o2 and o3 != o4:
        return True
    return ((o1 == 0 and on_seg(p1, p2, q1)) or (o2 == 0 and on_seg(p1, p2, q2))
            or (o3 == 0 and on_seg(q1, q2, p1)) or (o4 == 0 and on_seg(q1, q2, p2)))


def rect_polygon_distance(rect: Sequence[float], polygon: np.ndarray) -> float:
    """Euclidean distance in pixels between an axis-aligned rectangle and a polygon.

    0 when they touch or overlap. Exact for any simple polygon (convex or not): for two
    disjoint sets the minimum is reached between a vertex of one and the boundary of the
    other, so it is the smaller of (rectangle corners -> polygon edges, via
    ``cv2.pointPolygonTest``) and (polygon vertices -> rectangle). An edge that crosses the
    rectangle with no vertex of either inside the other (a thin band passing straight
    through a footprint strip) is caught by a segment intersection test. Works on the
    polygon of the current frame; no mask or distance transform is needed.
    """
    contour = np.asarray(polygon, dtype=np.float32).reshape(-1, 1, 2)
    x1, y1, x2, y2 = [float(v) for v in rect]
    corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    best = math.inf
    for c in corners:
        d = cv2.pointPolygonTest(contour, c, True)  # > 0 inside, 0 on the edge
        if d >= 0:
            return 0.0
        best = min(best, -d)
    pts = contour.reshape(-1, 2).astype(np.float64)
    dx = np.maximum(np.maximum(x1 - pts[:, 0], 0.0), pts[:, 0] - x2)
    dy = np.maximum(np.maximum(y1 - pts[:, 1], 0.0), pts[:, 1] - y2)
    dv = np.hypot(dx, dy)
    if dv.size and float(dv.min()) <= 0.0:
        return 0.0  # a polygon vertex lies in the rectangle
    best = min(best, float(dv.min()) if dv.size else math.inf)
    edges = [(corners[i], corners[(i + 1) % 4]) for i in range(4)]
    n = len(pts)
    for i in range(n):
        a, b = tuple(pts[i]), tuple(pts[(i + 1) % n])
        if any(_segments_intersect(a, b, e0, e1) for e0, e1 in edges):
            return 0.0
    return best


def footprint_relation(xyxy: Sequence[float], polygon: np.ndarray, mask: np.ndarray,
                       fraction: float = 0.25) -> Tuple[float, float]:
    """``(overlap, distance_bh)`` of a box's ground footprint against the zone.

    ``overlap`` is the share of footprint pixels on the zone ``mask`` (0..1), the advisor's
    ``Zone.relation`` definition; ``distance_bh`` is the footprint-to-polygon distance
    divided by the *box* height (0 when they overlap): a resolution- and range-free "how
    close", tuned to 0.75 box heights on two sites (E6). Both are measured against the zone
    of the current frame, so a stabilised (warped) polygon needs nothing extra.

    The pixel block is the team's exactly (``int()`` truncation, at least one row and one
    column, clipped to the image), so the 5 % and 0.75 thresholds tuned on their data mean
    the same here: with rounding instead, the on/off decision differed for 2 % of real
    vehicle boxes, almost all 12-48 px tall, where one pixel row is 6-25 % of the footprint.
    """
    h_img, w_img = mask.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    bh = max(1.0, y2 - y1)
    fy1 = int(np.clip(y2 - float(fraction) * bh, 0, h_img - 1))
    fy2 = int(np.clip(y2, fy1 + 1, h_img))
    fx1 = int(np.clip(x1, 0, w_img - 1))
    fx2 = int(np.clip(x2, fx1 + 1, w_img))
    foot = mask[fy1:fy2, fx1:fx2]
    overlap = float(np.count_nonzero(foot)) / float(foot.size) if foot.size else 0.0
    if overlap > 0.0:
        return overlap, 0.0
    # distance from the footprint's pixel block to the polygon
    return 0.0, rect_polygon_distance((fx1, fy1, fx2 - 1, fy2 - 1), polygon) / bh


def dilate_box(xyxy: Sequence[float], pad: float, shape_hw: Tuple[int, int]) -> np.ndarray:
    """Grow a box by ``pad`` pixels on every side, clipped to the image."""
    h, w = shape_hw[:2]
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    return np.array(
        [max(0.0, x1 - pad), max(0.0, y1 - pad), min(float(w), x2 + pad), min(float(h), y2 + pad)],
        dtype=np.float32,
    )


def transform_points(points: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Apply a 3x3 homography to (N,2) points; returns (N,2) float32."""
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, np.asarray(H, dtype=np.float64)).reshape(-1, 2)


def transform_box(xyxy: Sequence[float], H: np.ndarray) -> np.ndarray:
    """Axis-aligned float32 box enclosing the four corners of ``xyxy`` mapped by ``H``.

    Used to compare boxes seen at different times in one coordinate frame (the zone's
    reference keyframe) when the camera moves; for the small near-rigid motions the
    stabiliser accepts, the enclosing box differs from the true quadrilateral by a
    fraction of a pixel.
    """
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    pts = transform_points(np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32), H)
    return np.array([pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()], dtype=np.float32)


def homography_is_sane(H: np.ndarray, polygon_ref: np.ndarray, max_scale_change: float = 0.35,
                       max_area_change: float = 0.5) -> bool:
    """Reject degenerate, mirrored or wildly scaled homographies.

    Checks: finite entries, positive determinant of the affine part (no mirroring), the
    singular values of the 2x2 affine part within [1-max_scale_change, 1+max_scale_change],
    and the polygon area ratio within [1-max_area_change, 1+max_area_change].
    """
    if H is None or not np.all(np.isfinite(H)):
        return False
    A = np.asarray(H, dtype=np.float64)[:2, :2]
    if np.linalg.det(A) <= 0:
        return False
    s = np.linalg.svd(A, compute_uv=False)
    if s.min() < (1 - max_scale_change) or s.max() > (1 + max_scale_change):
        return False
    ref_area = polygon_area(polygon_ref)
    if ref_area <= 0:
        return False
    warped = transform_points(polygon_ref, H)
    if not np.all(np.isfinite(warped)):
        return False
    ratio = polygon_area(warped) / ref_area
    return (1 - max_area_change) <= ratio <= (1 + max_area_change)
