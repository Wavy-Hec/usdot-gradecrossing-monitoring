"""Tests for the experimental gradecrossing.autozone.propose_zone seed heuristic.

Synthetic 640x480 images: dark background with seeded Gaussian noise and two bright,
parallel "rails" drawn at about 10 degrees. The heuristic is only a calibration seed, so
the tests check orientation and extent, not pixel-exact corners.
"""
from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

from gradecrossing.autozone import propose_zone
from gradecrossing.geometry import point_in_polygon, polygon_area
from gradecrossing.types import Detection

W, H = 640, 480
RAIL_A = ((40, 200), (600, 300))  # first rail, ~10.1 degrees below horizontal
RAIL_SEP = 60  # second rail is this many px lower
LINE_ANGLE_DEG = float(np.degrees(np.arctan2(300 - 200, 600 - 40)))


def rails_image(seed: int = 0, noise_sigma: float = 8.0, with_distractors: bool = True) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.clip(40.0 + rng.normal(0.0, noise_sigma, (H, W)), 0, 255).astype(np.uint8)
    (x0, y0), (x1, y1) = RAIL_A
    for off in (0, RAIL_SEP):
        cv2.line(img, (x0, y0 + off), (x1, y1 + off), 230, 4)
    if with_distractors:
        # a shorter horizontal kerb and a short steep pole must not win over the rails
        cv2.line(img, (100, 60), (350, 60), 200, 3)
        cv2.line(img, (560, 80), (575, 200), 200, 3)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def edge_lengths_and_angles(poly: np.ndarray) -> List[Tuple[float, float]]:
    """(length, orientation in [0,180)) for every edge of the polygon."""
    out = []
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        length = float(np.hypot(b[0] - a[0], b[1] - a[1]))
        angle = float(np.degrees(np.arctan2(b[1] - a[1], b[0] - a[0])) % 180.0)
        out.append((length, angle))
    return out


def angle_diff(a: float, b: float) -> float:
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def test_two_parallel_lines_give_polygon_aligned_with_lines():
    poly = propose_zone(rails_image())
    assert poly is not None
    assert poly.shape == (4, 2) and poly.dtype == np.float32
    assert polygon_area(poly) > 0
    edges = sorted(edge_lengths_and_angles(poly), reverse=True)
    long_edges = edges[:2]
    for length, angle in long_edges:
        assert length > 300  # spans most of the drawn rails
        assert angle_diff(angle, LINE_ANGLE_DEG) <= 5.0
    # the strip lies between the rails: its centroid is midway between them
    cx, cy = poly.mean(axis=0)
    (x0, y0), (x1, y1) = RAIL_A
    y_rail_a = y0 + (y1 - y0) * (cx - x0) / (x1 - x0)
    assert y_rail_a < cy < y_rail_a + RAIL_SEP


def test_robust_to_noise_level_and_seed():
    for seed in (1, 2, 3):
        for sigma in (8.0, 25.0):
            poly = propose_zone(rails_image(seed=seed, noise_sigma=sigma))
            assert poly is not None, (seed, sigma)
            edges = sorted(edge_lengths_and_angles(poly), reverse=True)
            for _length, angle in edges[:2]:
                assert angle_diff(angle, LINE_ANGLE_DEG) <= 5.0, (seed, sigma)


def test_gate_boxes_bound_horizontal_extent():
    gates = [
        Detection(np.float32([200, 100, 240, 400]), 0.9, 0, "railroad-crossing-down"),
        Detection(np.float32([400, 100, 440, 400]), 0.8, 1, "railroad-crossings"),
        Detection(np.float32([10, 10, 60, 60]), 0.9, 2, "railroad-sign-light-on"),  # ignored class
    ]
    poly = propose_zone(rails_image(), gates)
    assert poly is not None
    assert abs(poly[:, 0].min() - 200.0) < 1.0
    assert abs(poly[:, 0].max() - 440.0) < 1.0
    edges = sorted(edge_lengths_and_angles(poly), reverse=True)
    for _length, angle in edges[:2]:
        assert angle_diff(angle, LINE_ANGLE_DEG) <= 5.0
    # the trapezoid still sits between the rails
    assert point_in_polygon((320.0, 300.0), poly)


def test_gate_boxes_of_other_classes_are_ignored():
    only_other = [Detection(np.float32([200, 100, 240, 400]), 0.9, 0, "railroad-sign-light-on")]
    poly_free = propose_zone(rails_image())
    poly_other = propose_zone(rails_image(), only_other)
    assert poly_free is not None and poly_other is not None
    assert np.allclose(poly_free, poly_other)


def test_blank_image_returns_none():
    assert propose_zone(np.full((H, W, 3), 60, dtype=np.uint8)) is None


def test_pure_noise_returns_none():
    """White noise makes the Hough transform hallucinate diagonals; the density guard refuses it."""
    rng = np.random.default_rng(0)
    noise = rng.integers(0, 256, size=(H, W), dtype=np.uint8)
    assert propose_zone(cv2.cvtColor(noise, cv2.COLOR_GRAY2BGR)) is None


def test_single_line_returns_none():
    img = np.full((H, W), 40, dtype=np.uint8)
    cv2.line(img, RAIL_A[0], RAIL_A[1], 230, 4)
    assert propose_zone(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)) is None


def test_accepts_grayscale_input():
    gray = cv2.cvtColor(rails_image(), cv2.COLOR_BGR2GRAY)
    poly = propose_zone(gray)
    assert poly is not None and poly.shape == (4, 2)
