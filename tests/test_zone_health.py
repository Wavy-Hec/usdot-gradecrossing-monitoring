"""Zone health reporting (debounced HEALTH reports) and static-overlay learning.

* Every healthy <-> held flip used to become a HEALTH event for every sink, webhook
  included; in deep low light the estimate flickers and the review counted ~460 events a
  minute. A change is now reported once it has lasted ``zone.health_report_seconds``
  (``ZoneState.healthy`` itself stays per frame).
* A burned-in overlay (timestamp, camera name, logo) does not move with the camera, so it
  "supported" a stale pose in the temporal gate: under strong shake the zone was held for
  runs of frames while the picture moved (19 px worst case in the review's probe). Corners
  that stay put while accepted estimates say the picture moved are learned as overlay and
  masked out.
"""
from __future__ import annotations

import math
from typing import List

import cv2
import numpy as np
import pytest

from gradecrossing.config import ZoneConfig
from gradecrossing.geometry import transform_points
from gradecrossing.zone import ZoneStabilizer

W, H = 640, 480
POLY = [[150, 300], [450, 280], [520, 400], [120, 420]]
BAND = [[0, 330], [640, 322], [640, 350], [0, 358]]  # a thin band zone, like the team clips


def textured_image(seed: int = 0) -> np.ndarray:
    """640x480 BGR image with rich, corner-heavy texture (same recipe as test_zone.py)."""
    rng = np.random.default_rng(seed)
    noise = rng.integers(0, 256, size=(H, W), dtype=np.uint8)
    gray = cv2.GaussianBlur(noise, (3, 3), 0)
    for _ in range(12):
        x, y = int(rng.integers(20, W - 80)), int(rng.integers(20, H - 80))
        w, h = int(rng.integers(20, 70)), int(rng.integers(20, 70))
        cv2.rectangle(gray, (x, y), (x + w, y + h), int(rng.choice([0, 255])), -1)
    for _ in range(8):
        x, y = int(rng.integers(30, W - 30)), int(rng.integers(30, H - 30))
        cv2.circle(gray, (x, y), int(rng.integers(8, 25)), int(rng.choice([0, 255])), -1)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def make(polygon=POLY, **overrides) -> ZoneStabilizer:
    return ZoneStabilizer(ZoneConfig(polygon=[list(map(float, p)) for p in polygon], **overrides))


def max_err(a, b) -> float:
    return float(np.linalg.norm(np.asarray(a, np.float64) - np.asarray(b, np.float64), axis=1).max())


@pytest.fixture(autouse=True)
def _seed_opencv():
    cv2.setRNGSeed(0)


@pytest.fixture(scope="module")
def image() -> np.ndarray:
    return textured_image(0)


BLANK = np.full((H, W, 3), 128, dtype=np.uint8)  # no estimate possible: held


# ----------------------------------------------------------------------------- debounce
def test_short_holds_are_not_reported_and_a_lasting_hold_is(image):
    stab = make()
    stab.initialize(image, 0.0)
    reports: List[str] = []
    flips = 0
    for i in range(1, 61):  # 2 s with every third frame held
        state = stab.update(BLANK if i % 3 == 0 else image, i / 30.0)
        flips += int(stab.health_changed)
        reports += stab.health_reports
        assert state.healthy == (i % 3 != 0)  # the per-frame signal is untouched
    assert flips == 39 and reports == []  # 20 held frames, 19 recoveries
    # a hold that lasts: reported once, when it has lasted health_report_seconds (0.5 s)
    at = []
    for i in range(61, 91):
        stab.update(BLANK, i / 30.0)
        at += [(i, r) for r in stab.health_reports]
    assert len(at) == 1 and at[0][0] == 60 + 15  # frame 60 (the last of the loop above) was held already
    assert at[0][1].startswith("zone unhealthy (holding last good polygon) for 0.50 s: held 0 inliers - LK:")
    # the recovery likewise, once it has lasted 0.5 s
    at = []
    for i in range(91, 121):
        stab.update(image, i / 30.0)
        at += [(i, r) for r in stab.health_reports]
    assert len(at) == 1 and at[0][0] == 91 + 15 and at[0][1].startswith("zone healthy for 0.50 s: lk ")


def test_flapping_health_is_reported_once_it_settles(image):
    stab = make()
    stab.initialize(image, 0.0)
    reports: List[str] = []
    for i in range(1, 121):  # 4 s of 2 held / 2 fresh frames: flips every 67 ms
        stab.update(BLANK if (i // 2) % 2 else image, i / 30.0)
        reports += stab.health_reports
    assert reports == []
    for i in range(121, 151):  # settles on held
        stab.update(BLANK, i / 30.0)
        reports += stab.health_reports
    assert len(reports) == 1 and reports[0].startswith("zone unhealthy")


def test_health_report_seconds_zero_reports_every_change(image):
    stab = make(health_report_seconds=0.0)
    stab.initialize(image, 0.0)
    reports: List[str] = []
    for i in range(1, 13):
        stab.update(BLANK if i % 3 == 0 else image, i / 30.0)
        reports += stab.health_reports
    assert len(reports) == 7  # 4 held frames (3, 6, 9, 12) and 3 recoveries: every change
    assert reports[0].startswith("zone unhealthy (holding last good polygon): held")
    assert reports[1].startswith("zone healthy: lk")


# ----------------------------------------------------------------------------- static overlays
def _shaken_with_overlay(image: np.ndarray, banner: np.ndarray, n: int, amp: float, seed: int = 0):
    """Frames of ``image`` shaken by a sway + per-frame jitter (GAEL-20 recipe; frame 0, the
    calibration frame, is not shaken), with the top rows replaced by a banner that does NOT
    move (a burned-in OSD); yields (frame, true 3x3 transform)."""
    rng = np.random.default_rng(seed)
    for i in range(n):
        t = i / 30.0
        dx = amp * math.sin(2 * math.pi * t / 1.3) + rng.normal(0, 0.3 * amp)
        dy = 0.6 * amp * (math.sin(2 * math.pi * t / 1.7 + 1.0) - math.sin(1.0)) + rng.normal(0, 0.3 * amp)
        M = cv2.getRotationMatrix2D((W / 2, H / 2), 0.2 * math.sin(2 * math.pi * t / 2.0), 1.0)
        if i > 0:
            M[0, 2] += dx
            M[1, 2] += dy
        frame = cv2.warpAffine(image, M, (W, H), borderMode=cv2.BORDER_REPLICATE)
        frame[: banner.shape[0]] = banner
        yield frame, np.vstack([M, [0.0, 0.0, 1.0]])


def _run_shaken(image, learn: bool, n: int = 150, amp: float = 6.0):
    banner = textured_image(7)[:150]
    stab = make(polygon=BAND)
    if not learn:
        stab._learn_overlay = lambda est: None  # type: ignore[assignment]
    held, errs = 0, []
    for i, (frame, T) in enumerate(_shaken_with_overlay(image, banner, n, amp)):
        state = stab.initialize(frame, 0.0) if i == 0 else stab.update(frame, i / 30.0)
        truth = transform_points(np.float32(BAND), T)
        if i >= 15:  # after 0.5 s: the overlay has had time to show itself
            held += int(not state.healthy)
            errs.append(max_err(state.polygon, truth))
    return stab, held, max(errs)


def test_static_overlay_is_learned_and_no_longer_holds_the_zone_under_shake(image):
    """Continuous 6 px shake, a banner over the top third that does not move, a 28 px band
    zone. Without overlay learning the banner keeps the old pose 'supported' and the zone is
    held for runs of frames while the picture moves; with it the zone follows every frame."""
    stab, held, worst = _run_shaken(image, learn=True)
    assert stab.health()["overlay_features"] >= 10
    assert held == 0 and worst <= 2.0
    _stab, held_unlearned, worst_unlearned = _run_shaken(image, learn=False)
    assert held_unlearned >= 10 and worst_unlearned > 2 * worst


def test_static_camera_learns_no_overlay(image):
    """No camera motion, no evidence: a still picture with a banner and sensor noise never
    masks anything (the overlay test needs the accepted estimate to say the picture moved)."""
    banner = textured_image(7)[:150]
    rng = np.random.default_rng(3)
    stab = make(polygon=BAND)
    for i in range(60):
        frame = image.copy()
        frame[:150] = banner
        frame = np.clip(frame + rng.normal(0, 3.0, frame.shape), 0, 255).astype(np.uint8)
        state = stab.initialize(frame, 0.0) if i == 0 else stab.update(frame, i / 30.0)
        assert state.healthy
    assert stab.health()["overlay_features"] == 0 and stab._overlay is None


def test_learned_overlay_is_masked_out_of_new_keyframes(image):
    """A new keyframe (re-anchor) does not pick the overlay corners up again: the mask
    lives in image coordinates, where the overlay is."""
    banner = textured_image(7)[:150]
    stab = make(polygon=BAND)
    frames = list(_shaken_with_overlay(image, banner, 60, 6.0))
    for i, (frame, _T) in enumerate(frames):
        if i == 0:
            stab.initialize(frame, 0.0)
        else:
            stab.update(frame, i / 30.0)
    assert stab.health()["overlay_features"] >= 10
    learned = stab._overlay.copy()
    gray = cv2.cvtColor(frames[-1][0], cv2.COLOR_BGR2GRAY)
    kf = stab._make_keyframe(gray, stab._pose, 2.0)
    pts = kf.pts.reshape(-1, 2).round().astype(int)
    assert len(pts) > 100 and not (learned[pts[:, 1], pts[:, 0]] > 0).any()
    # ... while without the learned mask the same frame yields corners on the banner
    stab._overlay = None
    pts = stab._make_keyframe(gray, stab._pose, 2.0).pts.reshape(-1, 2).round().astype(int)
    assert (learned[pts[:, 1], pts[:, 0]] > 0).sum() >= 10
