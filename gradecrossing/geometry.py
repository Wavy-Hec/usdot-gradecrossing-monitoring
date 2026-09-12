"""Small geometry helpers shared by the zone, tracking and debris modules."""
from __future__ import annotations

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


def box_center(xyxy: Sequence[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


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
